# The mapped phase weights and their concatenation in the actual packed owner.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_gemm_buffer.bend as Gemm
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ./storage_tree_model.bend as Tree
import ./storage_buffer_model.bend as Buffer
import ./input_gradient_tiled_refinement.bend as Copies
import ./kernel_weights_buffer_refinement.bend as Weights
import ./traversal_array_refinement.bend as Arrays
import ../kernel_weights.bend as Layouts
import ./kernel_weights_refinement.bend as Mapped
import ./traversal_indexed_values.bend as Indexed
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./loop_projection.bend as Projection
import ./storage_shape.bend as Shape
import ./storage_readback.bend as StorageReadback

def channels(context: Kernel.Phase) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
  c
def spatial(context: Kernel.Phase) -> U32:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw},y,x,inner} = context
  (k * k : U32)

def phase_weights(+context: Kernel.Phase,weights: Buffer.Model<F32>) -> Buffer.Model<Tile.Tile8>:
  Buffer.remodel(~Tile.Tile8,Pair.snd(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,
    Gemm.reorder_mapped(~Kernel.WeightWindow,~Kernel.weight_window_offset,channels(context),Kernel.phase_inner(context),spatial(context),Kernel.weight_window(context),Buffer.pack(~F32,weights))))

def phase_observed(+context: Kernel.Phase,+source: Buffer.Model<F32>) ->
  {Buffer.pack(~Tile.Tile8,phase_weights(context,source)) == Pair.snd(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,
    Gemm.reorder_mapped(~Kernel.WeightWindow,~Kernel.weight_window_offset,channels(context),Kernel.phase_inner(context),spatial(context),Kernel.weight_window(context),Buffer.pack(~F32,source))) : Storage.Buffer<Tile.Tile8>}:
  Buffer.remodeled(~Tile.Tile8,Pair.snd(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,
    Gemm.reorder_mapped(~Kernel.WeightWindow,~Kernel.weight_window_offset,channels(context),Kernel.phase_inner(context),spatial(context),Kernel.weight_window(context),Buffer.pack(~F32,source))))

def phase_layout(+context: Kernel.Phase) -> Layouts.MappedLayout<Kernel.WeightWindow>:
  Layouts.MappedLayout{channels(context),Kernel.phase_inner(context),spatial(context),Kernel.weight_window(context)}

def mapped_tree(+layout: Layouts.MappedLayout<Kernel.WeightWindow>,input: Tree.Tree<F32>) -> Tree.Tree<Tile.Tile8>:
  Mapped.output(Mapped.run_mapped(~Kernel.WeightWindow,~Kernel.weight_window_offset,Mapped.mapped_rows(~Kernel.WeightWindow,layout),
    Mapped.mapped_inner(~Kernel.WeightWindow,layout),Mapped.mapped_stride(~Kernel.WeightWindow,layout),Mapped.mapped_context(~Kernel.WeightWindow,layout),input))

def phase_tree(+context: Kernel.Phase,+source: Buffer.Model<F32>) ->
  {Buffer.tree(~Tile.Tile8,phase_weights(context,source)) == mapped_tree(phase_layout(context),Buffer.tree(~F32,source)) : Tree.Tree<Tile.Tile8>}:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
  Buffer.Model{tree,length,certificate} = source
  Weights.packed_mapped_tree(~Kernel.WeightWindow,~Kernel.weight_window_offset,c,inner,(k * k : U32),Kernel.weight_window(context),source)

def phase_readback(+context: Kernel.Phase,+source: Buffer.Model<F32>,+query: Mapped.Query<Kernel.WeightWindow>,
  layout_equal: {Mapped.query_layout(~Kernel.WeightWindow,query) == phase_layout(context) : Layouts.MappedLayout<Kernel.WeightWindow>},
  source_equal: {Mapped.query_input(~Kernel.WeightWindow,query) == Buffer.tree(~F32,source) : Tree.Tree<F32>}) ->
  {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,phase_weights(context,source)),Mapped.query_address(~Kernel.WeightWindow,query)) ==
    Mapped.query_wanted(~Kernel.WeightWindow,~Kernel.weight_window_offset,query) : Tile.Tile8}:
  +layout = Mapped.query_layout(~Kernel.WeightWindow,query)
  +input = Mapped.query_input(~Kernel.WeightWindow,query)
  +address = Mapped.query_address(~Kernel.WeightWindow,query)
  +same = Equal.trans(Tree.Tree<Tile.Tile8>,Buffer.tree(~Tile.Tile8,phase_weights(context,source)),mapped_tree(phase_layout(context),Buffer.tree(~F32,source)),mapped_tree(layout,input),
    phase_tree(context,source),Equal.trans(Tree.Tree<Tile.Tile8>,mapped_tree(phase_layout(context),Buffer.tree(~F32,source)),mapped_tree(layout,Buffer.tree(~F32,source)),mapped_tree(layout,input),
      Equal.cong(Layouts.MappedLayout<Kernel.WeightWindow>,Tree.Tree<Tile.Tile8>,layout => mapped_tree(layout,Buffer.tree(~F32,source)),phase_layout(context),layout,
        Equal.sym(Layouts.MappedLayout<Kernel.WeightWindow>,layout,phase_layout(context),layout_equal)),
      Equal.cong(Tree.Tree<F32>,Tree.Tree<Tile.Tile8>,input => mapped_tree(layout,input),Buffer.tree(~F32,source),input,Equal.sym(Tree.Tree<F32>,input,Buffer.tree(~F32,source),source_equal))))
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,phase_weights(context,source)),address),Tree.read(Tile.Tile8,mapped_tree(layout,input),address),
    Mapped.query_wanted(~Kernel.WeightWindow,~Kernel.weight_window_offset,query),
    Equal.cong(Tree.Tree<Tile.Tile8>,Tile.Tile8,tree => Tree.read(Tile.Tile8,tree,address),Buffer.tree(~Tile.Tile8,phase_weights(context,source)),mapped_tree(layout,input),same),
    Mapped.mapped_lookup(~Kernel.WeightWindow,~Kernel.weight_window_offset,query))

def components(-first: Storage.Buffer<F32>,-second: Storage.Buffer<Tile.Tile8>,+source: Buffer.Model<F32>,+packed: Buffer.Model<Tile.Tile8>,
  kept: {first == Buffer.pack(~F32,source) : Storage.Buffer<F32>},observed: {second == Buffer.pack(~Tile.Tile8,packed) : Storage.Buffer<Tile.Tile8>}) ->
  {(first,second) == (Buffer.pack(~F32,source),Buffer.pack(~Tile.Tile8,packed)) : Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>}:
  %Equal.sym(Storage.Buffer<F32>,first,Buffer.pack(~F32,source),kept) : {(_,second) == (Buffer.pack(~F32,source),Buffer.pack(~Tile.Tile8,packed)) : Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,second,Buffer.pack(~Tile.Tile8,packed),observed) : {(Buffer.pack(~F32,source),_) == (Buffer.pack(~F32,source),Buffer.pack(~Tile.Tile8,packed)) : Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>}
  {==}

def pair_parts(pair: Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>) ->
  {pair == (Pair.fst(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,pair),Pair.snd(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,pair)) : Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>}:
  (first,second) = pair
  {==}

def phase_matches(+context: Kernel.Phase,+source: Buffer.Model<F32>) ->
  {Gemm.reorder_mapped(~Kernel.WeightWindow,~Kernel.weight_window_offset,channels(context),Kernel.phase_inner(context),spatial(context),Kernel.weight_window(context),Buffer.pack(~F32,source)) ==
    (Buffer.pack(~F32,source),Buffer.pack(~Tile.Tile8,phase_weights(context,source))) : Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>}:
  pair = Gemm.reorder_mapped(~Kernel.WeightWindow,~Kernel.weight_window_offset,channels(context),Kernel.phase_inner(context),spatial(context),Kernel.weight_window(context),Buffer.pack(~F32,source))
  Equal.trans(Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>,pair,(Pair.fst(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,pair),Pair.snd(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,pair)),
    (Buffer.pack(~F32,source),Buffer.pack(~Tile.Tile8,phase_weights(context,source))),pair_parts(pair),
    components(Pair.fst(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,pair),Pair.snd(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,pair),source,phase_weights(context,source),
      Weights.source_mapped(~Kernel.WeightWindow,~Kernel.weight_window_offset,channels(context),Kernel.phase_inner(context),spatial(context),Kernel.weight_window(context),source),
      Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,phase_weights(context,source)),Pair.snd(Storage.Buffer<F32>,Storage.Buffer<Tile.Tile8>,pair),phase_observed(context,source))))

type Packed is Data:
  Packed{source: Buffer.Model<F32>,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,weight_base: U32,offset_base: U32}

def represent(state: Packed) -> Kernel.Packed:
  Packed{source,weights,offsets,weight_base,offset_base} = state
  Kernel.Packed{Buffer.pack(~F32,source),Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),weight_base,offset_base}

def phase_table(context: Kernel.Phase,layout: Kernel.Tiling,base: U32) -> Kernel.Table:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,+xs},inner} = context
  Kernel.Tiling{groups,+shifts,+outputs,+column_low,kernel,height,width,+chunk_groups,chunk_rows} = layout
  Kernel.Table{yr,xr,(chunk_groups * (outputs * shifts + 1) : U32),shifts,(xs + k - column_low : U32),base}

def copied(+context: Kernel.Phase,+layout: Kernel.Tiling,source: Buffer.Model<F32>,offsets: Tree.Tree<U32>,+weight_base: U32,+offset_base: U32,+count: U32,weights: Tree.Tree<Tile.Tile8>) -> Packed:
  Packed{source,weights,Traversal.indexed_fold(~Tree.Tree<U32>,~Kernel.Table,~Copies.offset,U32.to_nat(Kernel.phase_inner(context)),0,phase_table(context,layout,offset_base),offsets),
    (weight_base + count : U32),(offset_base + Kernel.phase_inner(context) : U32)}

def copied_matches(+context: Kernel.Phase,+layout: Kernel.Tiling,+source: Buffer.Model<F32>,+offsets: Tree.Tree<U32>,+weight_base: U32,+offset_base: U32,+count: U32,+weights: Tree.Tree<Tile.Tile8>) ->
  {Kernel.packed_copied(context,layout,Buffer.pack(~F32,source),Tree.pack(U32,offsets),weight_base,offset_base,count,Tree.pack(Tile.Tile8,weights)) ==
    represent(copied(context,layout,source,offsets,weight_base,offset_base,count,weights)) : Kernel.Packed}:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,xs},+inner} = context
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  Equal.cong(Array<U32>,Kernel.Packed,
    offsets => Kernel.Packed{Buffer.pack(~F32,source),Tree.pack(Tile.Tile8,weights),offsets,(weight_base + count : U32),(offset_base + inner : U32)},
    Traversal.indexed_fold(~Array<U32>,~Kernel.Table,~Kernel.offset,U32.to_nat(inner),0,phase_table(context,layout,offset_base),Tree.pack(U32,offsets)),
    Tree.pack(U32,Traversal.indexed_fold(~Tree.Tree<U32>,~Kernel.Table,~Copies.offset,U32.to_nat(inner),0,phase_table(context,layout,offset_base),offsets)),
    Copies.offsets_match(U32.to_nat(inner),0,phase_table(context,layout,offset_base),offsets))

def packed_phase(+context: Kernel.Phase,+layout: Kernel.Tiling,source: Buffer.Model<F32>,weights: Tree.Tree<Tile.Tile8>,offsets: Tree.Tree<U32>,+weight_base: U32,+offset_base: U32,+packed: Buffer.Model<Tile.Tile8>) -> Packed:
  +count = ((channels(context) + 7) / 8 * Kernel.phase_inner(context) : U32)
  copied(context,layout,source,offsets,weight_base,offset_base,count,
    Copies.tile_target(Loop.run(~Copies.TileCopy,~Copies.tile_step,U32.to_nat(count),Copies.TileCopy{Buffer.tree(~Tile.Tile8,packed),weights,0,weight_base})))

def packed_phase_matches(+context: Kernel.Phase,+layout: Kernel.Tiling,+source: Buffer.Model<F32>,+weights: Tree.Tree<Tile.Tile8>,+offsets: Tree.Tree<U32>,+weight_base: U32,+offset_base: U32,+packed: Buffer.Model<Tile.Tile8>) ->
  {Kernel.packed_phase(context,layout,Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),weight_base,offset_base,(Buffer.pack(~F32,source),Buffer.pack(~Tile.Tile8,packed))) ==
    represent(packed_phase(context,layout,source,weights,offsets,weight_base,offset_base,packed)) : Kernel.Packed}:
  Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = context
  Buffer.Model{+tiles,length,certificate} = packed
  +count = ((c + 7) / 8 * inner : U32)
  +initial = {Copies.TileCopy{tiles,weights,0,weight_base} : Copies.TileCopy}
  %Equal.sym(Kernel.TileCopy,Loop.run(~Kernel.TileCopy,~Kernel.tile_step,U32.to_nat(count),Copies.pack_copy(initial)),
    Copies.pack_copy(Loop.run(~Copies.TileCopy,~Copies.tile_step,U32.to_nat(count),initial)),Copies.tile_copy_matches(U32.to_nat(count),initial)) :
    {Kernel.packed_copied(context,layout,Buffer.pack(~F32,source),Tree.pack(U32,offsets),weight_base,offset_base,count,Kernel.tile_target(_)) ==
      represent(packed_phase(context,layout,source,weights,offsets,weight_base,offset_base,packed)) : Kernel.Packed}
  %Equal.sym(Array<Tile.Tile8>,Kernel.tile_target(Copies.pack_copy(Loop.run(~Copies.TileCopy,~Copies.tile_step,U32.to_nat(count),initial))),
    Tree.pack(Tile.Tile8,Copies.tile_target(Loop.run(~Copies.TileCopy,~Copies.tile_step,U32.to_nat(count),initial))),Copies.tile_target_matches(Loop.run(~Copies.TileCopy,~Copies.tile_step,U32.to_nat(count),initial))) :
    {Kernel.packed_copied(context,layout,Buffer.pack(~F32,source),Tree.pack(U32,offsets),weight_base,offset_base,count,_) ==
      represent(packed_phase(context,layout,source,weights,offsets,weight_base,offset_base,packed)) : Kernel.Packed}
  copied_matches(context,layout,source,offsets,weight_base,offset_base,count,Copies.tile_target(Loop.run(~Copies.TileCopy,~Copies.tile_step,U32.to_nat(count),initial)))

def pack_selected(empty: Bool,+context: Kernel.Phase,+layout: Kernel.Tiling,state: Packed) -> Packed:
  match empty:
    case True{}: state
    case False{}:
      Packed{+source,weights,offsets,weight_base,offset_base} = state
      packed_phase(context,layout,source,weights,offsets,weight_base,offset_base,phase_weights(context,source))

def selected_matches(empty: Bool,+context: Kernel.Phase,+layout: Kernel.Tiling,+state: Packed) ->
  {Kernel.pack_selected(empty,context,layout,represent(state)) == represent(pack_selected(empty,context,layout,state)) : Kernel.Packed}:
  match empty:
    case True{}: {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
      Packed{+source,+weights,+offsets,+weight_base,+offset_base} = state
      %Equal.sym(Storage.Buffer<F32> & Storage.Buffer<Tile.Tile8>,Gemm.reorder_mapped(~Kernel.WeightWindow,~Kernel.weight_window_offset,c,inner,(k * k : U32),Kernel.weight_window(context),Buffer.pack(~F32,source)),
        (Buffer.pack(~F32,source),Buffer.pack(~Tile.Tile8,phase_weights(context,source))),phase_matches(context,source)) :
        {Kernel.packed_phase(context,layout,Tree.pack(Tile.Tile8,weights),Tree.pack(U32,offsets),weight_base,offset_base,_) == represent(pack_selected(False{},context,layout,state)) : Kernel.Packed}
      packed_phase_matches(context,layout,source,weights,offsets,weight_base,offset_base,phase_weights(context,source))

def pack_phase(index: U32,context: Kernel.PackContext,state: Packed) -> Packed:
  Kernel.PackContext{+geometry,+layout} = context
  +phase = Kernel.phase_of(geometry,index)
  pack_selected(U32.is_eq(Kernel.phase_inner(phase),0),phase,layout,state)

def phase_step_matches(+index: U32,+context: Kernel.PackContext,+state: Packed) ->
  {Kernel.pack_phase(index,context,represent(state)) == represent(pack_phase(index,context,state)) : Kernel.Packed}:
  Kernel.PackContext{+geometry,+layout} = context
  +phase = Kernel.phase_of(geometry,index)
  selected_matches(U32.is_eq(Kernel.phase_inner(phase),0),phase,layout,state)

def packing_matches(+count: Nat,+first: U32,+context: Kernel.PackContext,+state: Packed) ->
  {Traversal.indexed_fold(~Kernel.Packed,~Kernel.PackContext,~Kernel.pack_phase,count,first,context,represent(state)) ==
    represent(Traversal.indexed_fold(~Packed,~Kernel.PackContext,~pack_phase,count,first,context,state)) : Kernel.Packed}:
  Arrays.indexed_fold_refines_model(~Kernel.Packed,~Packed,~Kernel.PackContext,~Kernel.pack_phase,~pack_phase,~represent,~phase_step_matches,count,first,context,state)

def source(state: Packed) -> Buffer.Model<F32>:
  Packed{source,weights,offsets,weight_base,offset_base} = state
  source

def selected_keeps_source(empty: Bool,+context: Kernel.Phase,+layout: Kernel.Tiling,+state: Packed) ->
  {source(pack_selected(empty,context,layout,state)) == source(state) : Buffer.Model<F32>}:
  match empty:
    case True{}: {==}
    case False{}:
      Packed{source,weights,offsets,weight_base,offset_base} = state
      {==}

def phase_keeps_source(+index: U32,+context: Kernel.PackContext,+state: Packed) ->
  {source(pack_phase(index,context,state)) == source(state) : Buffer.Model<F32>}:
  Kernel.PackContext{+geometry,+layout} = context
  +selected = Kernel.phase_of(geometry,index)
  selected_keeps_source(U32.is_eq(Kernel.phase_inner(selected),0),selected,layout,state)

# The packed copy also uses the affine readback at depth 31.
type Copy is Data:
  Copy{source: Tree.Tree<Tile.Tile8>,from: U32,to: U32}

def copy_sample(context: Copy,position: U32) -> Tile.Tile8:
  Copy{source,from,to} = context
  Tree.read(Tile.Tile8,source,(from + position : U32))

def copy_update(position: U32,context: Copy,target: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Copy{source,from,+to} = context
  +position = position
  Tree.write(Tile.Tile8,target,(to + position : U32),copy_sample(context,position))

def copy_represent(context: Copy,position: U32,target: Tree.Tree<Tile.Tile8>) -> Copies.TileCopy:
  Copy{source,from,to} = context
  +position = position
  Copies.TileCopy{source,target,(from + position : U32),(to + position : U32)}

def copy_step_matches(+context: Copy,+position: U32,+target: Tree.Tree<Tile.Tile8>) ->
  {Copies.tile_step(copy_represent(context,position,target)) == copy_represent(context,(position + 1 : U32),copy_update(position,context,target)) : Copies.TileCopy}:
  Copy{source,+from,+to} = context
  %Equal.sym(U32,((from + position : U32) + 1 : U32),(from + (position + 1 : U32) : U32),Arithmetic.add_associative(from,position,1)) :
    {Copies.TileCopy{source,copy_update(position,context,target),_,((to + position : U32) + 1 : U32)} == copy_represent(context,(position + 1 : U32),copy_update(position,context,target)) : Copies.TileCopy}
  %Equal.sym(U32,((to + position : U32) + 1 : U32),(to + (position + 1 : U32) : U32),Arithmetic.add_associative(to,position,1)) :
    {Copies.TileCopy{source,copy_update(position,context,target),(from + (position + 1 : U32) : U32),_} == copy_represent(context,(position + 1 : U32),copy_update(position,context,target)) : Copies.TileCopy}
  {==}

def copy_target_matches(+context: Copy,+position: U32,+target: Tree.Tree<Tile.Tile8>) ->
  {Copies.tile_target(copy_represent(context,position,target)) == target : Tree.Tree<Tile.Tile8>}:
  Copy{source,from,to} = context
  {==}

def copy_folded(+count: Nat,+context: Copy,+target: Tree.Tree<Tile.Tile8>) ->
  {Copies.tile_target(Loop.run(~Copies.TileCopy,~Copies.tile_step,count,copy_represent(context,0,target))) ==
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Copy,~copy_update,count,0,context,target) : Tree.Tree<Tile.Tile8>}:
  +written = Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Copy,~copy_update,count,0,context,target)
  %Equal.sym(Copies.TileCopy,Loop.run(~Copies.TileCopy,~Copies.tile_step,count,copy_represent(context,0,target)),
    copy_represent(context,(0 + U32.from_nat(count) : U32),written),
    Arrays.indexed_loop(~Copies.TileCopy,~Tree.Tree<Tile.Tile8>,~Copy,~Copies.tile_step,~copy_update,~copy_represent,~copy_step_matches,count,0,context,target)) :
    {Copies.tile_target(_) == written : Tree.Tree<Tile.Tile8>}
  copy_target_matches(context,(0 + U32.from_nat(count) : U32),written)

type CopyQuery is Data:
  CopyQuery{context: Copy,count: Nat,index: Nat,depth: Nat,capacity: U32}

def copy_context(query: CopyQuery) -> Copy:
  CopyQuery{context,count,index,depth,capacity} = query
  context

def copy_affine(query: CopyQuery) -> Indexed.AffineQuery<Copy>:
  CopyQuery{context,count,index,depth,capacity} = query
  Copy{source,from,+to} = context
  Indexed.AffineQuery{Copy{source,from,to},to,count,index,depth,capacity}

def copy_queried(position: U32,query: CopyQuery,target: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  copy_update(position,copy_context(query),target)

def copy_same_step(+position: U32,+query: CopyQuery,+target: Tree.Tree<Tile.Tile8>) ->
  {copy_queried(position,query,target) == copy_update(position,copy_context(query),target) : Tree.Tree<Tile.Tile8>}:
  {==}

def copy_affine_matches(+position: U32,+query: CopyQuery,+target: Tree.Tree<Tile.Tile8>) ->
  {copy_queried(position,query,target) == Indexed.affine_update(~Tile.Tile8,~Copy,~copy_sample,position,copy_affine(query),target) : Tree.Tree<Tile.Tile8>}:
  CopyQuery{context,count,index,depth,capacity} = query
  Copy{source,from,to} = context
  {==}

def copy_readback(+query: CopyQuery,+target: Tree.Tree<Tile.Tile8>,evidence: Indexed.AffineFrame<Tile.Tile8,Copy,copy_affine(query),target>) ->
  {Tree.read(Tile.Tile8,Copies.tile_target(Loop.run(~Copies.TileCopy,~Copies.tile_step,Indexed.affine_count(~Copy,copy_affine(query)),copy_represent(copy_context(query),0,target))),
    Indexed.affine_address(~Copy,copy_affine(query))) == Indexed.affine_wanted(~Tile.Tile8,~Copy,~copy_sample,copy_affine(query)) : Tile.Tile8}:
  CopyQuery{context,count,index,depth,capacity} = query
  Copy{source,from,to} = context
  %Equal.sym(Tree.Tree<Tile.Tile8>,Copies.tile_target(Loop.run(~Copies.TileCopy,~Copies.tile_step,count,copy_represent(context,0,target))),
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Copy,~copy_update,count,0,context,target),copy_folded(count,context,target)) :
    {Tree.read(Tile.Tile8,_,Indexed.affine_address(~Copy,copy_affine(query))) == Indexed.affine_wanted(~Tile.Tile8,~Copy,~copy_sample,copy_affine(query)) : Tile.Tile8}
  %Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~CopyQuery,~Copy,~copy_context,~copy_queried,~copy_update,~copy_same_step,count,0,query,target) :
    {Tree.read(Tile.Tile8,_,Indexed.affine_address(~Copy,copy_affine(query))) == Indexed.affine_wanted(~Tile.Tile8,~Copy,~copy_sample,copy_affine(query)) : Tile.Tile8}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~CopyQuery,~copy_queried,count,0,query,target),
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Indexed.AffineQuery<Copy>,~Indexed.affine_update(~Tile.Tile8,~Copy,~copy_sample),count,0,copy_affine(query),target),
    Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~CopyQuery,~Indexed.AffineQuery<Copy>,~copy_affine,~copy_queried,~Indexed.affine_update(~Tile.Tile8,~Copy,~copy_sample),~copy_affine_matches,count,0,query,target)) :
    {Tree.read(Tile.Tile8,_,Indexed.affine_address(~Copy,copy_affine(query))) == Indexed.affine_wanted(~Tile.Tile8,~Copy,~copy_sample,copy_affine(query)) : Tile.Tile8}
  Indexed.affine_readback(~Tile.Tile8,~Copy,~copy_sample,copy_affine(query),target,evidence)

def copy_read_before(+query: CopyQuery,+address: U32,+target: Tree.Tree<Tile.Tile8>,
  frame: Indexed.AffineFrame<Tile.Tile8,Copy,copy_affine(query),target>,
  before: {Nat.is_lt(U32.to_nat(address),U32.to_nat(Indexed.affine_base(~Copy,copy_affine(query)))) == True{} : Bool},
  inside: {U32.is_lt(address,Indexed.affine_capacity(~Copy,copy_affine(query))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,Copies.tile_target(Loop.run(~Copies.TileCopy,~Copies.tile_step,Indexed.affine_count(~Copy,copy_affine(query)),copy_represent(copy_context(query),0,target))),address) ==
    Tree.read(Tile.Tile8,target,address) : Tile.Tile8}:
  CopyQuery{context,count,index,depth,capacity} = query
  Copy{source,from,to} = context
  %Equal.sym(Tree.Tree<Tile.Tile8>,Copies.tile_target(Loop.run(~Copies.TileCopy,~Copies.tile_step,count,copy_represent(context,0,target))),
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Copy,~copy_update,count,0,context,target),copy_folded(count,context,target)) :
    {Tree.read(Tile.Tile8,_,address) == Tree.read(Tile.Tile8,target,address) : Tile.Tile8}
  %Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~CopyQuery,~Copy,~copy_context,~copy_queried,~copy_update,~copy_same_step,count,0,query,target) :
    {Tree.read(Tile.Tile8,_,address) == Tree.read(Tile.Tile8,target,address) : Tile.Tile8}
  %Equal.sym(Tree.Tree<Tile.Tile8>,Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~CopyQuery,~copy_queried,count,0,query,target),
    Traversal.indexed_fold(~Tree.Tree<Tile.Tile8>,~Indexed.AffineQuery<Copy>,~Indexed.affine_update(~Tile.Tile8,~Copy,~copy_sample),count,0,copy_affine(query),target),
    Arrays.indexed_fold_change_context(~Tree.Tree<Tile.Tile8>,~CopyQuery,~Indexed.AffineQuery<Copy>,~copy_affine,~copy_queried,~Indexed.affine_update(~Tile.Tile8,~Copy,~copy_sample),~copy_affine_matches,count,0,query,target)) :
    {Tree.read(Tile.Tile8,_,address) == Tree.read(Tile.Tile8,target,address) : Tile.Tile8}
  Indexed.affine_read_before(~Tile.Tile8,~Copy,~copy_sample,copy_affine(query),address,target,frame,before,inside)

type Counters is Data:
  Counters{weights: U32,offsets: U32}

def counters(state: Packed) -> Counters:
  Packed{source,weights,offsets,weight_base,offset_base} = state
  Counters{weight_base,offset_base}

def phase_count(context: Kernel.Phase) -> U32:
  Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = context
  ((c + 7) / 8 * inner : U32)

def advanced(+context: Kernel.Phase,state: Packed) -> Counters:
  Packed{source,weights,offsets,weight_base,offset_base} = state
  Counters{(weight_base + phase_count(context) : U32),(offset_base + Kernel.phase_inner(context) : U32)}

def selected_counters(empty: Bool,+context: Kernel.Phase,+layout: Kernel.Tiling,+state: Packed,
  compared: {U32.is_eq(Kernel.phase_inner(context),0) == empty : Bool}) ->
  {counters(pack_selected(empty,context,layout,state)) == advanced(context,state) : Counters}:
  match empty:
    case True{}:
      Kernel.Phase{Kernel.Geometry{+c,o,h,w,k,s,p,gh,gw},y,x,+inner} = context
      Packed{source,weights,offsets,+weight_base,+offset_base} = state
      +zero = Arithmetic.equal_of_is_eq(inner,0,compared)
      %Equal.sym(U32,inner,0,zero) : {Counters{weight_base,offset_base} == Counters{(weight_base + (c + 7) / 8 * _ : U32),(offset_base + _ : U32)} : Counters}
      %Equal.sym(U32,(((c + 7) / 8) * 0 : U32),0,Arithmetic.multiply_zero_right(((c + 7) / 8 : U32))) :
        {Counters{weight_base,offset_base} == Counters{(weight_base + _ : U32),(offset_base + 0 : U32)} : Counters}
      %Equal.sym(U32,(weight_base + 0 : U32),weight_base,Arithmetic.add_zero(weight_base)) :
        {Counters{weight_base,offset_base} == Counters{_,(offset_base + 0 : U32)} : Counters}
      %Equal.sym(U32,(offset_base + 0 : U32),offset_base,Arithmetic.add_zero(offset_base)) :
        {Counters{weight_base,offset_base} == Counters{weight_base,_} : Counters}
      {==}
    case False{}:
      Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},y,x,inner} = context
      Packed{source,weights,offsets,weight_base,offset_base} = state
      {==}

def counter_update(index: U32,context: Kernel.PackContext,state: Counters) -> Counters:
  Kernel.PackContext{+geometry,layout} = context
  Counters{weights,offsets} = state
  +index = index
  Counters{Kernel.weight_total(index,geometry,weights),Kernel.inner_total(index,geometry,offsets)}

def phase_counters(+index: U32,+context: Kernel.PackContext,+state: Packed) ->
  {counters(pack_phase(index,context,state)) == counter_update(index,context,counters(state)) : Counters}:
  Kernel.PackContext{+geometry,+layout} = context
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Packed{source,weights,offsets,weight_base,offset_base} = state
  +selected = Kernel.phase_of(geometry,index)
  selected_counters(U32.is_eq(Kernel.phase_inner(selected),0),selected,layout,state,{==})

def pack_geometry(context: Kernel.PackContext) -> Kernel.Geometry:
  Kernel.PackContext{geometry,layout} = context
  geometry

def weight_projection(state: Counters) -> U32:
  Counters{weights,offsets} = state
  weights
def offset_projection(state: Counters) -> U32:
  Counters{weights,offsets} = state
  offsets
def weight_base(state: Packed) -> U32:
  weight_projection(counters(state))
def offset_base(state: Packed) -> U32:
  offset_projection(counters(state))

def weight_progress(index: U32,context: Kernel.PackContext,value: U32) -> U32:
  Kernel.weight_total(index,pack_geometry(context),value)
def offset_progress(index: U32,context: Kernel.PackContext,value: U32) -> U32:
  Kernel.inner_total(index,pack_geometry(context),value)

def weight_step(+index: U32,+context: Kernel.PackContext,+state: Packed) ->
  {weight_base(pack_phase(index,context,state)) == weight_progress(index,context,weight_base(state)) : U32}:
  Kernel.PackContext{geometry,layout} = context
  Packed{source,weights,offsets,weight_start,offset_start} = state
  Equal.cong(Counters,U32,weight_projection,counters(pack_phase(index,context,state)),counter_update(index,context,counters(state)),phase_counters(index,context,state))

def offset_step(+index: U32,+context: Kernel.PackContext,+state: Packed) ->
  {offset_base(pack_phase(index,context,state)) == offset_progress(index,context,offset_base(state)) : U32}:
  Kernel.PackContext{geometry,layout} = context
  Packed{source,weights,offsets,weight_start,offset_start} = state
  Equal.cong(Counters,U32,offset_projection,counters(pack_phase(index,context,state)),counter_update(index,context,counters(state)),phase_counters(index,context,state))

def copy_source(context: Copy) -> Tree.Tree<Tile.Tile8>:
  Copy{source,from,to} = context
  source
def copy_from(context: Copy) -> U32:
  Copy{source,from,to} = context
  from
def copy_to(context: Copy) -> U32:
  Copy{source,from,to} = context
  to

def copy_initial(+context: Copy,+target: Tree.Tree<Tile.Tile8>) ->
  {copy_represent(context,0,target) == Copies.TileCopy{copy_source(context),target,copy_from(context),copy_to(context)} : Copies.TileCopy}:
  Copy{source,+from,+to} = context
  %Equal.sym(U32,(from + 0 : U32),from,Arithmetic.add_zero(from)) : {Copies.TileCopy{source,target,_,(to + 0 : U32)} == Copies.TileCopy{copy_source(context),target,copy_from(context),copy_to(context)} : Copies.TileCopy}
  %Equal.sym(U32,(to + 0 : U32),to,Arithmetic.add_zero(to)) : {Copies.TileCopy{source,target,from,_} == Copies.TileCopy{copy_source(context),target,copy_from(context),copy_to(context)} : Copies.TileCopy}
  {==}
def copied_capacity(state: Copies.TileCopy) -> U32:
  Tree.capacity(Tile.Tile8,Copies.tile_target(state))
def copy_keeps_capacity(+state: Copies.TileCopy) -> {copied_capacity(Copies.tile_step(state)) == copied_capacity(state) : U32}:
  Copies.TileCopy{+source,+target,+from,+to} = state
  StorageReadback.write_capacity(~Tile.Tile8,target,to,Tree.read(Tile.Tile8,source,from))
def copy_capacity(+count: Nat,+state: Copies.TileCopy) ->
  {Tree.capacity(Tile.Tile8,Copies.tile_target(Loop.run(~Copies.TileCopy,~Copies.tile_step,count,state))) == Tree.capacity(Tile.Tile8,Copies.tile_target(state)) : U32}:
  Projection.preserves_observation(~Copies.TileCopy,~U32,~Copies.tile_step,~copied_capacity,~copy_keeps_capacity,count,state)

def copied_balanced(depth: Nat,state: Copies.TileCopy) -> Data:
  Shape.Balanced(Tile.Tile8,depth,Copies.tile_target(state))
def copy_keeps_balance(+depth: Nat,+state: Copies.TileCopy,evidence: Shape.Balanced(Tile.Tile8,depth,Copies.tile_target(state))) ->
  Shape.Balanced(Tile.Tile8,depth,Copies.tile_target(Copies.tile_step(state))):
  Copies.TileCopy{+source,+target,+from,+to} = state
  Shape.write(Tile.Tile8,depth,target,to,Tree.read(Tile.Tile8,source,from),evidence)
def copy_balanced(+count: Nat,+depth: Nat,+state: Copies.TileCopy,evidence: Shape.Balanced(Tile.Tile8,depth,Copies.tile_target(state))) ->
  Shape.Balanced(Tile.Tile8,depth,Copies.tile_target(Loop.run(~Copies.TileCopy,~Copies.tile_step,count,state))):
  Projection.preserves_invariant(~Copies.TileCopy,~Nat,~copied_balanced,~Copies.tile_step,~copy_keeps_balance,count,depth,state,evidence)
