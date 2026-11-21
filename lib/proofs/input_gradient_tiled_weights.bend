# Phase views write directly into their consecutive packed weight intervals.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../tensor_tiled.bend as Tiled
import ../tensor_view.bend as Views
import ./tensor_copy_values.bend as Values
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ../storage_copy.bend as StorageCopy
import ./storage_copy_refinement.bend as StorageCopyLaws
import ./write_program.bend as Writes
import ./traversal_array_refinement.bend as Arrays
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding
import ./nat_order.bend as Order

def phase_count(context: Kernel.Phase) -> U32:
  Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = context
  (Tiled.groups(c) * inner : U32)

def components(-first: Storage.Buffer<Tile.Tile8>,-second: Storage.Buffer<Tile.Tile8>,+source: Buffer.Model<Tile.Tile8>,+packed: Buffer.Model<Tile.Tile8>,
  kept: {first == Buffer.pack(~Tile.Tile8,source) : Storage.Buffer<Tile.Tile8>},observed: {second == Buffer.pack(~Tile.Tile8,packed) : Storage.Buffer<Tile.Tile8>}) ->
  {(first,second) == (Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,packed)) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  %Equal.sym(Storage.Buffer<Tile.Tile8>,first,Buffer.pack(~Tile.Tile8,source),kept) : {(_,second) == (Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,packed)) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,second,Buffer.pack(~Tile.Tile8,packed),observed) : {(Buffer.pack(~Tile.Tile8,source),_) == (Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,packed)) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}
  {==}

def pair_parts(pair: Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) ->
  {pair == (Pair.fst(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,pair),Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,pair)) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}:
  (first,second) = pair
  {==}

type Packed is Data:
  Packed{source: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,weight_base: U32}

def represent(state: Packed) -> Kernel.Packed:
  Packed{source,weights,weight_base} = state
  Kernel.Packed{Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,weights),weight_base}

# A phase with a tile has taps.
def count_active(empty: Bool,+context: Kernel.Phase,+slot: Nat,compared: {U32.is_eq(Kernel.phase_inner(context),0) == empty : Bool},
  inside: {Nat.is_lt(slot,U32.to_nat(phase_count(context))) == True{} : Bool}) -> {empty == False{} : Bool}:
  match empty:
    case False{}: {==}
    case True{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = context
      +zero = Arithmetic.equal_of_is_eq(inner,0,compared)
      +empty_count = {
        %Equal.sym(U32,inner,0,zero) : {((c + 7) / 8 * _ : U32) == 0 : U32}
        Arithmetic.multiply_zero_right(((c + 7) / 8 : U32)) : {((c + 7) / 8 * inner : U32) == 0 : U32}}
      +below = Equal.trans(Bool,Nat.is_lt(slot,U32.to_nat(0)),Nat.is_lt(slot,U32.to_nat(((c + 7) / 8 * inner : U32))),True{},
        Equal.cong(U32,Bool,count => Nat.is_lt(slot,U32.to_nat(count)),0,((c + 7) / 8 * inner : U32),Equal.sym(U32,((c + 7) / 8 * inner : U32),0,empty_count)),inside)
      Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(slot,0n),True{},
        Equal.sym(Bool,Nat.is_lt(slot,0n),False{},Order.not_below(0n,slot,Order.zero_le(slot))),below)))

# A phase copies its source view into packed weights from weight_base.
def copy_pair(+context: Kernel.Phase,source: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,+weight_base: U32) ->
  Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  +view = Kernel.weight_view(context)
  StorageCopy.view(~Tile.Tile8,Views.is_contiguous(view),phase_count(context),view,weight_base,Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,weights))

def copied(+context: Kernel.Phase,source: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,+weight_base: U32) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,copy_pair(context,source,weights,weight_base)))

def packed_phase(+context: Kernel.Phase,+source: Buffer.Model<Tile.Tile8>,weights: Buffer.Model<Tile.Tile8>,+weight_base: U32) -> Packed:
  Packed{source,copied(context,source,weights,weight_base),(weight_base + phase_count(context) : U32)}

def packed_phase_matches(+context: Kernel.Phase,+source: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+weight_base: U32) ->
  {Kernel.packed_phase(context,weight_base,Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,weights)) ==
    represent(packed_phase(context,source,weights,weight_base)) : Kernel.Packed}:
  Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = context
  +count = ((c + 7) / 8 * inner : U32)
  +represented = {
    Equal.trans(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,copy_pair(context,source,weights,weight_base),
      (Pair.fst(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,copy_pair(context,source,weights,weight_base)),Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,copy_pair(context,source,weights,weight_base))),
      (Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,copied(context,source,weights,weight_base))),pair_parts(copy_pair(context,source,weights,weight_base)),
      components(Pair.fst(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,copy_pair(context,source,weights,weight_base)),Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,copy_pair(context,source,weights,weight_base)),source,copied(context,source,weights,weight_base),
        StorageCopyLaws.source(~Tile.Tile8,Views.is_contiguous(Kernel.weight_view(context)),count,Kernel.weight_view(context),weight_base,Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,weights)),
        Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,copied(context,source,weights,weight_base)),Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,copy_pair(context,source,weights,weight_base)),
          Buffer.remodeled(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,copy_pair(context,source,weights,weight_base)))))) :
      {copy_pair(context,source,weights,weight_base) == (Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,copied(context,source,weights,weight_base))) : Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>}}
  Equal.cong(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>,Kernel.Packed,Kernel.packed_copied(weight_base,count),copy_pair(context,source,weights,weight_base),
    (Buffer.pack(~Tile.Tile8,source),Buffer.pack(~Tile.Tile8,copied(context,source,weights,weight_base))),represented)

def pack_selected(empty: Bool,+context: Kernel.Phase,state: Packed) -> Packed:
  match empty:
    case True{}: state
    case False{}:
      Packed{+source,weights,weight_base} = state
      packed_phase(context,source,weights,weight_base)

def selected_matches(empty: Bool,+context: Kernel.Phase,+state: Packed) ->
  {Kernel.pack_selected(empty,context,represent(state)) == represent(pack_selected(empty,context,state)) : Kernel.Packed}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
      Packed{+source,+weights,+weight_base} = state
      packed_phase_matches(context,source,weights,weight_base)

def pack_phase(index: U32,geometry: Kernel.Geometry,state: Packed) -> Packed:
  +phase = Kernel.phase_of(geometry,index)
  pack_selected(U32.is_eq(Kernel.phase_inner(phase),0),phase,state)

def phase_step_matches(+index: U32,+geometry: Kernel.Geometry,+state: Packed) ->
  {Kernel.pack_phase(index,geometry,represent(state)) == represent(pack_phase(index,geometry,state)) : Kernel.Packed}:
  +phase = Kernel.phase_of(geometry,index)
  selected_matches(U32.is_eq(Kernel.phase_inner(phase),0),phase,state)

def packing_matches(+count: Nat,+first: U32,+geometry: Kernel.Geometry,+state: Packed) ->
  {Traversal.indexed_fold(~Kernel.Packed,~Kernel.Geometry,~Kernel.pack_phase,count,first,geometry,represent(state)) ==
    represent(Traversal.indexed_fold(~Packed,~Kernel.Geometry,~pack_phase,count,first,geometry,state)) : Kernel.Packed}:
  Arrays.indexed_fold_refines_model(~Kernel.Packed,~Packed,~Kernel.Geometry,~Kernel.pack_phase,~pack_phase,~represent,~phase_step_matches,count,first,geometry,state)

def source(state: Packed) -> Buffer.Model<Tile.Tile8>:
  Packed{source,weights,weight_base} = state
  source

def weights(state: Packed) -> Buffer.Model<Tile.Tile8>:
  Packed{source,weights,weight_base} = state
  weights

def weight_base(state: Packed) -> U32:
  Packed{source,weights,weight_base} = state
  weight_base

def selected_keeps_source(empty: Bool,+context: Kernel.Phase,+state: Packed) ->
  {source(pack_selected(empty,context,state)) == source(state) : Buffer.Model<Tile.Tile8>}:
  match empty:
    case True{}: {==}
    case False{}:
      Packed{source,weights,weight_base} = state
      {==}

def phase_keeps_source(+index: U32,+geometry: Kernel.Geometry,+state: Packed) ->
  {source(pack_phase(index,geometry,state)) == source(state) : Buffer.Model<Tile.Tile8>}:
  +selected = Kernel.phase_of(geometry,index)
  selected_keeps_source(U32.is_eq(Kernel.phase_inner(selected),0),selected,state)

def kept_source(index: U32,geometry: Kernel.Geometry,kept: Buffer.Model<Tile.Tile8>) -> Buffer.Model<Tile.Tile8>:
  kept

def kept_source_fixed(+index: U32,+geometry: Kernel.Geometry,+kept: Buffer.Model<Tile.Tile8>) -> {kept_source(index,geometry,kept) == kept : Buffer.Model<Tile.Tile8>}:
  {==}

# Packing the phases gives the source back.
def phases_keep_source(+count: Nat,+first: U32,+geometry: Kernel.Geometry,+state: Packed) ->
  {source(Traversal.indexed_fold(~Packed,~Kernel.Geometry,~pack_phase,count,first,geometry,state)) == source(state) : Buffer.Model<Tile.Tile8>}:
  Equal.trans(Buffer.Model<Tile.Tile8>,source(Traversal.indexed_fold(~Packed,~Kernel.Geometry,~pack_phase,count,first,geometry,state)),
    Traversal.indexed_fold(~Buffer.Model<Tile.Tile8>,~Kernel.Geometry,~kept_source,count,first,geometry,source(state)),source(state),
    Arrays.indexed_fold_observes(~Packed,~Buffer.Model<Tile.Tile8>,~Kernel.Geometry,~pack_phase,~source,~kept_source,~phase_keeps_source,count,first,geometry,state),
    Arrays.indexed_fold_fixed(~Buffer.Model<Tile.Tile8>,~Kernel.Geometry,~kept_source,~kept_source_fixed,count,first,geometry,source(state)))

# A selected phase moves the weight base by its tiles; an empty phase has none.
def selected_base(empty: Bool,+context: Kernel.Phase,+state: Packed,compared: {U32.is_eq(Kernel.phase_inner(context),0) == empty : Bool}) ->
  {weight_base(pack_selected(empty,context,state)) == (weight_base(state) + phase_count(context) : U32) : U32}:
  match empty:
    case True{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = context
      Packed{source,weights,+base} = state
      +zero = Arithmetic.equal_of_is_eq(inner,0,compared)
      %Equal.sym(U32,inner,0,zero) : {base == (base + (c + 7) / 8 * _ : U32) : U32}
      %Equal.sym(U32,(((c + 7) / 8) * 0 : U32),0,Arithmetic.multiply_zero_right(((c + 7) / 8 : U32))) : {base == (base + _ : U32) : U32}
      %Equal.sym(U32,(base + 0 : U32),base,Arithmetic.add_zero(base)) : {base == _ : U32}
      {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
      Packed{source,weights,base} = state
      {==}

def weight_step(+index: U32,+geometry: Kernel.Geometry,+state: Packed) ->
  {weight_base(pack_phase(index,geometry,state)) == Kernel.weight_total(index,geometry,weight_base(state)) : U32}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Packed{source,weights,base} = state
  +selected = Kernel.phase_of(geometry,index)
  selected_base(U32.is_eq(Kernel.phase_inner(selected),0),selected,state,{==})

# What a copy stores: the copy program over the packed weights.
def copy_content(~Element: Data,owners: StorageCopyLaws.Owners<Element>) -> Buffer.Content<Element>:
  StorageCopyLaws.Owners{source,source_length,output,output_length} = owners
  Buffer.Content{output,output_length}

def second_content(~Element: Data,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,pair)) == copy_content(~Element,StorageCopyLaws.observe(~Element,pair)) : Buffer.Content<Element>}:
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate}) = pair
  {==}

def copy_writes(+context: Kernel.Phase,source: Buffer.Model<Tile.Tile8>,+weight_base: U32) -> List<&2,Writes.Assignment<Tile.Tile8>>:
  +view = Kernel.weight_view(context)
  Values.copy_writes(~Tile.Tile8,Views.is_contiguous(view),U32.to_nat(phase_count(context)),view,weight_base,Buffer.tree(~Tile.Tile8,source))

def copied_content(+context: Kernel.Phase,+source: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+weight_base: U32) ->
  {Buffer.content(~Tile.Tile8,copied(context,source,weights,weight_base)) ==
    Buffer.Content{Writes.pure(Tile.Tile8,copy_writes(context,source,weight_base),Buffer.tree(~Tile.Tile8,weights)),Buffer.length(~Tile.Tile8,weights)} : Buffer.Content<Tile.Tile8>}:
  Buffer.Model{+source_tree,+source_length,+source_certificate} = source
  Buffer.Model{+tree,+length,+certificate} = weights
  Equal.trans(Buffer.Content<Tile.Tile8>,
    Buffer.content(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
      copy_pair(context,Buffer.Model{source_tree,source_length,source_certificate},Buffer.Model{tree,length,certificate},weight_base)))),
    Buffer.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
      copy_pair(context,Buffer.Model{source_tree,source_length,source_certificate},Buffer.Model{tree,length,certificate},weight_base))),
    Buffer.Content{Writes.pure(Tile.Tile8,copy_writes(context,source,weight_base),tree),length},
    Buffer.remodel_content(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
      copy_pair(context,Buffer.Model{source_tree,source_length,source_certificate},Buffer.Model{tree,length,certificate},weight_base))),
    Equal.trans(Buffer.Content<Tile.Tile8>,
      Buffer.contents(~Tile.Tile8,Pair.snd(Storage.Buffer<Tile.Tile8>,Storage.Buffer<Tile.Tile8>,
        copy_pair(context,Buffer.Model{source_tree,source_length,source_certificate},Buffer.Model{tree,length,certificate},weight_base))),
      copy_content(~Tile.Tile8,StorageCopyLaws.observe(~Tile.Tile8,copy_pair(context,Buffer.Model{source_tree,source_length,source_certificate},Buffer.Model{tree,length,certificate},weight_base))),
      Buffer.Content{Writes.pure(Tile.Tile8,copy_writes(context,source,weight_base),tree),length},
      second_content(~Tile.Tile8,copy_pair(context,Buffer.Model{source_tree,source_length,source_certificate},Buffer.Model{tree,length,certificate},weight_base)),
      Equal.cong(StorageCopyLaws.Owners<Tile.Tile8>,Buffer.Content<Tile.Tile8>,owners => copy_content(~Tile.Tile8,owners),
        StorageCopyLaws.observe(~Tile.Tile8,copy_pair(context,Buffer.Model{source_tree,source_length,source_certificate},Buffer.Model{tree,length,certificate},weight_base)),
        StorageCopyLaws.Owners{source_tree,source_length,Writes.pure(Tile.Tile8,copy_writes(context,source,weight_base),tree),length},
        StorageCopyLaws.view_packed(~Tile.Tile8,Views.is_contiguous(Kernel.weight_view(context)),phase_count(context),Kernel.weight_view(context),weight_base,source_tree,source_length,tree,length,source_certificate,certificate))))

def copied_tree(+context: Kernel.Phase,+source: Buffer.Model<Tile.Tile8>,+weights: Buffer.Model<Tile.Tile8>,+weight_base: U32) ->
  {Buffer.tree(~Tile.Tile8,copied(context,source,weights,weight_base)) == Writes.pure(Tile.Tile8,copy_writes(context,source,weight_base),Buffer.tree(~Tile.Tile8,weights)) : Tree.Tree<Tile.Tile8>}:
  +model = copied(context,source,weights,weight_base)
  Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,model),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,model)),
    Writes.pure(Tile.Tile8,copy_writes(context,source,weight_base),Buffer.tree(~Tile.Tile8,weights)),
    Equal.sym(Tree.Tree<Tile.Tile8>,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,model)),Buffer.tree(~Tile.Tile8,model),Buffer.content_tree_of(~Tile.Tile8,model)),
    Equal.cong(Buffer.Content<Tile.Tile8>,Tree.Tree<Tile.Tile8>,content => Buffer.content_tree(~Tile.Tile8,content),Buffer.content(~Tile.Tile8,model),
      Buffer.Content{Writes.pure(Tile.Tile8,copy_writes(context,source,weight_base),Buffer.tree(~Tile.Tile8,weights)),Buffer.length(~Tile.Tile8,weights)},copied_content(context,source,weights,weight_base)))

