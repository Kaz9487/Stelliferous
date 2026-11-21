# Tensors.placed over the parts of pieces cut between tasks: in the order of
# their cells, each part is copied into the slot that holds it, and a slot
# holds its piece once its last part is in.
import Base
import ../storage_buffer.bend as Storage
import ../storage_copy.bend as Copy
import ../tensor.bend as Tensors
import ../tensor_pieces.bend as Pieces
import ./traversal_partition_refinement.bend as PartitionLaws
import ../traversal_partition.bend as Partition
import ../kernel_gemm.bend as KernelGemm
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./storage_copy_refinement.bend as CopyLaws
import ./contiguous_copy.bend as Contiguous
import ./contiguous_copy_readback.bend as CopyRead
import ./write_program.bend as Writes
import ./write_program_readback.bend as Readback
import ./tensor_pieces_content.bend as Content
import ./machine_limit_bounds.bend as Limits
import ../machine_limits.bend as Machine
import ./tensor_task_runs.bend as Runs
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_move_readback.bend as Held
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_allocation_depth.bend as Depth
import ./tensor_placed.bend as Placed
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./word_power_of_two.bend as Power
import ./nat_interval.bend as Interval
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./boolean_logic.bend as Boolean
import ./u32_comparison.bend as Comparison

# ---- A part copied into a slot ----

def owner_content(~Element: Data,owners: CopyLaws.Owners<Element>) -> Buffer.Content<Element>:
  CopyLaws.Owners{source,source_length,output,output_length} = owners
  Buffer.Content{output,output_length}

def second_seen(~Element: Data,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,owners)) == owner_content(~Element,CopyLaws.observe(~Element,owners)) : Buffer.Content<Element>}:
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate}) = owners
  {==}

# The target of a copy of count cells of a part to destination holds the
# copy writes over its tree and keeps its length.
def copied_from_content(~Element: Data,+count: U32,+source: U32,+destination: U32,+part: Buffer.Model<Element>,+target: Buffer.Model<Element>) ->
  {Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Copy.range(~Element,count,source,destination,Buffer.pack(~Element,part),Buffer.pack(~Element,target))))) ==
    Buffer.Content{Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),Buffer.tree(~Element,part),0,source,destination),Buffer.tree(~Element,target)),Buffer.length(~Element,target)} : Buffer.Content<Element>}:
  Buffer.Model{+part_tree,+part_length,+part_certificate} = part
  Buffer.Model{+tree,+length,+certificate} = target
  Equal.trans(Buffer.Content<Element>,
    Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
      Copy.range(~Element,count,source,destination,Storage.Buffer{Tree.pack(Element,part_tree),part_length,part_certificate},Storage.Buffer{Tree.pack(Element,tree),length,certificate})))),
    Buffer.contents(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
      Copy.range(~Element,count,source,destination,Storage.Buffer{Tree.pack(Element,part_tree),part_length,part_certificate},Storage.Buffer{Tree.pack(Element,tree),length,certificate}))),
    Buffer.Content{Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,source,destination),tree),length},
    Buffer.remodel_content(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
      Copy.range(~Element,count,source,destination,Storage.Buffer{Tree.pack(Element,part_tree),part_length,part_certificate},Storage.Buffer{Tree.pack(Element,tree),length,certificate}))),
    Equal.trans(Buffer.Content<Element>,
      Buffer.contents(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
        Copy.range(~Element,count,source,destination,Storage.Buffer{Tree.pack(Element,part_tree),part_length,part_certificate},Storage.Buffer{Tree.pack(Element,tree),length,certificate}))),
      owner_content(~Element,CopyLaws.observe(~Element,
        Copy.range(~Element,count,source,destination,Storage.Buffer{Tree.pack(Element,part_tree),part_length,part_certificate},Storage.Buffer{Tree.pack(Element,tree),length,certificate}))),
      Buffer.Content{Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,source,destination),tree),length},
      second_seen(~Element,
        Copy.range(~Element,count,source,destination,Storage.Buffer{Tree.pack(Element,part_tree),part_length,part_certificate},Storage.Buffer{Tree.pack(Element,tree),length,certificate})),
      Equal.cong(CopyLaws.Owners<Element>,Buffer.Content<Element>,owners => owner_content(~Element,owners),
        CopyLaws.observe(~Element,
          Copy.range(~Element,count,source,destination,Storage.Buffer{Tree.pack(Element,part_tree),part_length,part_certificate},Storage.Buffer{Tree.pack(Element,tree),length,certificate})),
        CopyLaws.model(~Element,count,source,destination,CopyLaws.Owners{part_tree,part_length,tree,length}),
        CopyLaws.packed(~Element,CopyLaws.Call{count,source,destination,part_length,length},[],part_tree,tree,part_certificate,certificate))))

def copied_content(~Element: Data,+count: U32,+destination: U32,+part: Buffer.Model<Element>,+target: Buffer.Model<Element>) ->
  {Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Copy.range(~Element,count,0,destination,Buffer.pack(~Element,part),Buffer.pack(~Element,target))))) ==
    Buffer.Content{Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),Buffer.tree(~Element,part),0,0,destination),Buffer.tree(~Element,target)),Buffer.length(~Element,target)} : Buffer.Content<Element>}:
  copied_from_content(~Element,count,0,destination,part,target)

# A ready tree has a capacity a word can hold.
def ready_capacity(~Element: Data,+content: Buffer.Content<Element>,+ready: Buffer.ContentReady<Element,content>) ->
  {U32.is_le(Tree.capacity(Element,Buffer.content_tree(~Element,content)),2147483648) == True{} : Bool}:
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  %Equal.sym(U32,Tree.capacity(Element,Buffer.content_tree(~Element,content)),U32.shln(1,depth),Shape.capacity(Element,depth,Buffer.content_tree(~Element,content),balanced)) :
    {U32.is_le(_,2147483648) == True{} : Bool}
  Power.native_power_bound(depth,Order.successor(depth,30n,depth_ok))

# ---- The cells a slot holds ----

# Cells [0, covered) of a slot buffer hold the values of its cells from slot.
def Covered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+slot: U32,+covered: Nat,+tree: Tree.Tree<Element>,+length: Nat) -> Data:
  Members.All(Nat,{index => Content.At(~Element,~Logical,~original,logical,slot,tree,length,index) : Nat -> Data},Ranges.range(covered,0n))

# What a copy into a slot reads from: the slot before it, covered up to
# where the part starts, the part, and the room of the copy.
type Copying<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  Copying{logical: Logical,slot: U32,tree: Tree.Tree<Element>,length: Nat,depth: Nat,balanced: Shape.Balanced(Element,depth,tree),depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
    capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},
    room: {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
    covered: Nat,before: Covered(~Element,~Logical,~original,logical,slot,covered,tree,length),
    first: U32,count: U32,part_tree: Tree.Tree<Element>,part_length: Nat,
    part: Content.Leaf(~Element,~Logical,~original,logical,first,count,part_tree,part_length),
    destination: U32,destination_exact: {U32.to_nat(destination) == covered : Nat},
    first_exact: {U32.to_nat(first) == Nat.add(U32.to_nat(slot),covered) : Nat},
    inside: {Nat.is_le(Nat.add(covered,U32.to_nat(count)),length) == True{} : Bool}}

def copying_end(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,copying: Copying<Element,Logical,original>) -> Nat:
  Copying{logical,slot,tree,length,depth,balanced,depth_ok,capacity_ok,room,covered,before,first,count,part_tree,part_length,part,destination,destination_exact,first_exact,inside} = copying
  Nat.add(covered,U32.to_nat(count))

def CopiedAt(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+copying: Copying<Element,Logical,original>,+index: Nat) -> Data:
  Copying{+logical,+slot,+tree,+length,+depth,+balanced,+depth_ok,+capacity_ok,+room,+covered,+before,+first,+count,+part_tree,+part_length,+part,+destination,+destination_exact,
    +first_exact,+inside} = copying
  Content.At(~Element,~Logical,~original,logical,slot,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),tree),length,index)

# ---- A cell of a copied slot ----

# A cell of a buffer holds the value of its piece.
def at_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+first: U32,+tree: Tree.Tree<Element>,+length: Nat,+index: Nat,
  +at: Content.At(~Element,~Logical,~original,logical,first,tree,length,index)) ->
  {Tree.read(Element,tree,U32.from_nat(index)) == original(logical,U32.from_nat(Nat.add(U32.to_nat(first),index))) : Element}:
  KernelGemm.Both{+below,+value} = at
  value

# After the copy, a cell before the part keeps its value and a cell of the
# part holds the part's value.
def copied_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,less: Bool,+logical: Logical,+slot: U32,+tree: Tree.Tree<Element>,+length: Nat,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},+room: {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
  +covered: Nat,+before: Covered(~Element,~Logical,~original,logical,slot,covered,tree,length),+first: U32,+count: U32,+part_tree: Tree.Tree<Element>,+part_length: Nat,
  +part: Content.Leaf(~Element,~Logical,~original,logical,first,count,part_tree,part_length),+destination: U32,+destination_exact: {U32.to_nat(destination) == covered : Nat},
  +first_exact: {U32.to_nat(first) == Nat.add(U32.to_nat(slot),covered) : Nat},+inside: {Nat.is_le(Nat.add(covered,U32.to_nat(count)),length) == True{} : Bool},
  +index: Nat,+index_inside: {Nat.is_lt(index,Nat.add(covered,U32.to_nat(count))) == True{} : Bool},+less_exact: {Nat.is_lt(index,covered) == less : Bool}) ->
  {Tree.read(Element,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),tree),U32.from_nat(index)) == original(logical,U32.from_nat(Nat.add(U32.to_nat(slot),index))) : Element}:
  match less:
    case True{}:
      Equal.trans(Element,Tree.read(Element,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),tree),U32.from_nat(index)),Readback.selected(~Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),U32.from_nat(index),Tree.read(Element,tree,U32.from_nat(index))),original(logical,U32.from_nat(Nat.add(U32.to_nat(slot),index))),Readback.program(~Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),depth,tree,U32.from_nat(index),balanced,depth_ok,{%Equal.sym(Bool,U32.is_lt(U32.from_nat(index),Tree.capacity(Element,tree)),Nat.is_lt(U32.to_nat(U32.from_nat(index)),U32.to_nat(Tree.capacity(Element,tree))),Comparison.lt(U32.from_nat(index),Tree.capacity(Element,tree))) : {_ == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,Tree.capacity(Element,tree),capacity_ok,Bounds.strict_le(index,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,length,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,Nat.add(covered,U32.to_nat(count)),length,index_inside,inside),room)))) : {Nat.is_lt(_,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
      Bounds.nat_trans(index,length,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,Nat.add(covered,U32.to_nat(count)),length,index_inside,inside),room) : {U32.is_lt(U32.from_nat(index),Tree.capacity(Element,tree)) == True{} : Bool}},CopyRead.writes_inside(~Element,U32.to_nat(count),0n,part_tree,0,destination,Tree.capacity(Element,tree),capacity_ok,{%Equal.sym(Nat,Nat.add(U32.to_nat(destination),0n),U32.to_nat(destination),Addition.zero_right(U32.to_nat(destination))) :
        {Nat.is_le(Nat.add(_,U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(destination),covered,destination_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
      Order.transitive(Nat.add(covered,U32.to_nat(count)),length,U32.to_nat(Tree.capacity(Element,tree)),inside,room) :
      {Nat.is_le(Nat.add(Nat.add(U32.to_nat(destination),0n),U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}})),
        Equal.trans(Element,Readback.selected(~Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),U32.from_nat(index),Tree.read(Element,tree,U32.from_nat(index))),Tree.read(Element,tree,U32.from_nat(index)),original(logical,U32.from_nat(Nat.add(U32.to_nat(slot),index))),
          CopyRead.selected_outside(~Element,U32.to_nat(count),0n,U32.from_nat(index),part_tree,0,destination,Tree.capacity(Element,tree),capacity_ok,{%Equal.sym(Nat,Nat.add(U32.to_nat(destination),0n),U32.to_nat(destination),Addition.zero_right(U32.to_nat(destination))) :
        {Nat.is_le(Nat.add(_,U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(destination),covered,destination_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
      Order.transitive(Nat.add(covered,U32.to_nat(count)),length,U32.to_nat(Tree.capacity(Element,tree)),inside,room) :
      {Nat.is_le(Nat.add(Nat.add(U32.to_nat(destination),0n),U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}},
            Interval.Before{{
              %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,Tree.capacity(Element,tree),capacity_ok,Bounds.strict_le(index,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,length,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,Nat.add(covered,U32.to_nat(count)),length,index_inside,inside),room)))) : {Nat.is_lt(_,Nat.add(U32.to_nat(destination),0n)) == True{} : Bool}
              %Equal.sym(Nat,Nat.add(U32.to_nat(destination),0n),U32.to_nat(destination),Addition.zero_right(U32.to_nat(destination))) : {Nat.is_lt(index,_) == True{} : Bool}
              %Equal.sym(Nat,U32.to_nat(destination),covered,destination_exact) : {Nat.is_lt(index,_) == True{} : Bool}
              less_exact : {Nat.is_lt(U32.to_nat(U32.from_nat(index)),Nat.add(U32.to_nat(destination),0n)) == True{} : Bool}}},
            Tree.read(Element,tree,U32.from_nat(index))),
          at_read(~Element,~Logical,~original,logical,slot,tree,length,index,
            Members.all_member(Nat,{index => Content.At(~Element,~Logical,~original,logical,slot,tree,length,index) : Nat -> Data},index,Ranges.range(covered,0n),before,Members.range_member(covered,0n,index,less_exact)))))
    case False{}:
      +ge = Order.not_less(index,covered,less_exact)
      +restore = Order.restore_shift(covered,index,ge)
      +part_inside = Order.sub_lt(index,covered,U32.to_nat(count),ge,index_inside)
      +offset_room = {%Equal.sym(Nat,U32.to_nat(destination),covered,destination_exact) : {Nat.is_le(Nat.add(_,Nat.sub(index,covered)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(covered,Nat.sub(index,covered)),index,restore) : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
        Bounds.strict_le(index,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,length,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,Nat.add(covered,U32.to_nat(count)),length,index_inside,inside),room)) : {Nat.is_le(Nat.add(U32.to_nat(destination),Nat.sub(index,covered)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}}
      +offset = Equal.trans(Nat,U32.to_nat((destination + U32.from_nat(Nat.sub(index,covered)) : U32)),Nat.add(U32.to_nat(destination),Nat.sub(index,covered)),index,
        CopyRead.offset_exact(destination,Nat.sub(index,covered),Tree.capacity(Element,tree),capacity_ok,offset_room),
        Equal.trans(Nat,Nat.add(U32.to_nat(destination),Nat.sub(index,covered)),Nat.add(covered,Nat.sub(index,covered)),index,
          Equal.cong(Nat,Nat,number => Nat.add(number,Nat.sub(index,covered)),U32.to_nat(destination),covered,destination_exact),restore))
      +word = Arithmetic.to_nat_injective(U32.from_nat(index),(destination + U32.from_nat(Nat.sub(index,covered)) : U32),
        Equal.trans(Nat,U32.to_nat(U32.from_nat(index)),index,U32.to_nat((destination + U32.from_nat(Nat.sub(index,covered)) : U32)),Bounds.roundtrip(index,Tree.capacity(Element,tree),capacity_ok,Bounds.strict_le(index,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,length,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,Nat.add(covered,U32.to_nat(count)),length,index_inside,inside),room))),Equal.sym(Nat,U32.to_nat((destination + U32.from_nat(Nat.sub(index,covered)) : U32)),index,offset)))
      +position = Equal.trans(Nat,Nat.add(U32.to_nat(first),Nat.sub(index,covered)),Nat.add(Nat.add(U32.to_nat(slot),covered),Nat.sub(index,covered)),Nat.add(U32.to_nat(slot),index),
        Equal.cong(Nat,Nat,number => Nat.add(number,Nat.sub(index,covered)),U32.to_nat(first),Nat.add(U32.to_nat(slot),covered),first_exact),
        Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(slot),covered),Nat.sub(index,covered)),Nat.add(U32.to_nat(slot),Nat.add(covered,Nat.sub(index,covered))),Nat.add(U32.to_nat(slot),index),
          Addition.associative(U32.to_nat(slot),covered,Nat.sub(index,covered)),Equal.cong(Nat,Nat,number => Nat.add(U32.to_nat(slot),number),Nat.add(covered,Nat.sub(index,covered)),index,restore)))
      Equal.trans(Element,Tree.read(Element,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),tree),U32.from_nat(index)),Readback.selected(~Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),U32.from_nat(index),Tree.read(Element,tree,U32.from_nat(index))),original(logical,U32.from_nat(Nat.add(U32.to_nat(slot),index))),Readback.program(~Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),depth,tree,U32.from_nat(index),balanced,depth_ok,{%Equal.sym(Bool,U32.is_lt(U32.from_nat(index),Tree.capacity(Element,tree)),Nat.is_lt(U32.to_nat(U32.from_nat(index)),U32.to_nat(Tree.capacity(Element,tree))),Comparison.lt(U32.from_nat(index),Tree.capacity(Element,tree))) : {_ == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Bounds.roundtrip(index,Tree.capacity(Element,tree),capacity_ok,Bounds.strict_le(index,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,length,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,Nat.add(covered,U32.to_nat(count)),length,index_inside,inside),room)))) : {Nat.is_lt(_,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
      Bounds.nat_trans(index,length,U32.to_nat(Tree.capacity(Element,tree)),Bounds.nat_trans(index,Nat.add(covered,U32.to_nat(count)),length,index_inside,inside),room) : {U32.is_lt(U32.from_nat(index),Tree.capacity(Element,tree)) == True{} : Bool}},CopyRead.writes_inside(~Element,U32.to_nat(count),0n,part_tree,0,destination,Tree.capacity(Element,tree),capacity_ok,{%Equal.sym(Nat,Nat.add(U32.to_nat(destination),0n),U32.to_nat(destination),Addition.zero_right(U32.to_nat(destination))) :
        {Nat.is_le(Nat.add(_,U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(destination),covered,destination_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
      Order.transitive(Nat.add(covered,U32.to_nat(count)),length,U32.to_nat(Tree.capacity(Element,tree)),inside,room) :
      {Nat.is_le(Nat.add(Nat.add(U32.to_nat(destination),0n),U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}})),
        Equal.trans(Element,Readback.selected(~Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),U32.from_nat(index),Tree.read(Element,tree,U32.from_nat(index))),Tree.read(Element,part_tree,(0 + U32.from_nat(Nat.sub(index,covered)) : U32)),original(logical,U32.from_nat(Nat.add(U32.to_nat(slot),index))),
          {%Equal.sym(U32,U32.from_nat(index),(destination + U32.from_nat(Nat.sub(index,covered)) : U32),word) :
              {Readback.selected(~Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),_,Tree.read(Element,tree,U32.from_nat(index))) == Tree.read(Element,part_tree,(0 + U32.from_nat(Nat.sub(index,covered)) : U32)) : Element}
            CopyRead.selected_inside(~Element,U32.to_nat(count),0n,Nat.sub(index,covered),part_tree,0,destination,Tree.capacity(Element,tree),capacity_ok,{%Equal.sym(Nat,Nat.add(U32.to_nat(destination),0n),U32.to_nat(destination),Addition.zero_right(U32.to_nat(destination))) :
        {Nat.is_le(Nat.add(_,U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(destination),covered,destination_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}
      Order.transitive(Nat.add(covered,U32.to_nat(count)),length,U32.to_nat(Tree.capacity(Element,tree)),inside,room) :
      {Nat.is_le(Nat.add(Nat.add(U32.to_nat(destination),0n),U32.to_nat(count)),U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool}},part_inside,Tree.read(Element,tree,U32.from_nat(index))) :
            {Readback.selected(~Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),U32.from_nat(index),Tree.read(Element,tree,U32.from_nat(index))) == Tree.read(Element,part_tree,(0 + U32.from_nat(Nat.sub(index,covered)) : U32)) : Element}},
          Equal.trans(Element,Tree.read(Element,part_tree,(0 + U32.from_nat(Nat.sub(index,covered)) : U32)),Tree.read(Element,part_tree,U32.from_nat(Nat.sub(index,covered))),original(logical,U32.from_nat(Nat.add(U32.to_nat(slot),index))),
            Equal.cong(U32,Element,word_index => Tree.read(Element,part_tree,word_index),(0 + U32.from_nat(Nat.sub(index,covered)) : U32),U32.from_nat(Nat.sub(index,covered)),Arithmetic.add_zero_left(U32.from_nat(Nat.sub(index,covered)))),
            Equal.trans(Element,Tree.read(Element,part_tree,U32.from_nat(Nat.sub(index,covered))),original(logical,U32.from_nat(Nat.add(U32.to_nat(first),Nat.sub(index,covered)))),original(logical,U32.from_nat(Nat.add(U32.to_nat(slot),index))),
              at_read(~Element,~Logical,~original,logical,first,part_tree,part_length,Nat.sub(index,covered),
                Content.leaf_cell(~Element,~Logical,~original,logical,first,count,part_tree,part_length,Nat.sub(index,covered),part,part_inside)),
              Equal.cong(Nat,Element,number => original(logical,U32.from_nat(number)),Nat.add(U32.to_nat(first),Nat.sub(index,covered)),Nat.add(U32.to_nat(slot),index),position)))))

def copied_at(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+copying: Copying<Element,Logical,original>,+index: Nat,
  +index_inside: {Nat.is_lt(index,copying_end(~Element,~Logical,~original,copying)) == True{} : Bool}) -> CopiedAt(~Element,~Logical,~original,copying,index):
  Copying{+logical,+slot,+tree,+length,+depth,+balanced,+depth_ok,+capacity_ok,+room,+covered,+before,+first,+count,+part_tree,+part_length,+part,+destination,+destination_exact,+first_exact,+inside} = copying
  KernelGemm.Both{Bounds.nat_trans(index,Nat.add(covered,U32.to_nat(count)),length,index_inside,inside),copied_read(~Element,~Logical,~original,Nat.is_lt(index,covered),logical,slot,tree,length,depth,balanced,depth_ok,capacity_ok,room,covered,before,first,count,part_tree,part_length,part,destination,destination_exact,first_exact,inside,index,index_inside,{==})}

# After the copy of a part that starts where the slot is covered, the slot is
# covered to the end of the part.
def copied_covered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+slot: U32,+tree: Tree.Tree<Element>,+length: Nat,+depth: Nat,+balanced: Shape.Balanced(Element,depth,tree),+depth_ok: {Nat.is_le(depth,30n) == True{} : Bool},
  +capacity_ok: {U32.is_le(Tree.capacity(Element,tree),2147483648) == True{} : Bool},+room: {Nat.is_le(length,U32.to_nat(Tree.capacity(Element,tree))) == True{} : Bool},
  +covered: Nat,+before: Covered(~Element,~Logical,~original,logical,slot,covered,tree,length),+first: U32,+count: U32,+part_tree: Tree.Tree<Element>,+part_length: Nat,
  +part: Content.Leaf(~Element,~Logical,~original,logical,first,count,part_tree,part_length),+destination: U32,+destination_exact: {U32.to_nat(destination) == covered : Nat},
  +first_exact: {U32.to_nat(first) == Nat.add(U32.to_nat(slot),covered) : Nat},+inside: {Nat.is_le(Nat.add(covered,U32.to_nat(count)),length) == True{} : Bool}) ->
  Covered(~Element,~Logical,~original,logical,slot,Nat.add(covered,U32.to_nat(count)),Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,destination),tree),length):
  Members.range_all(~Copying<Element,Logical,original>,~CopiedAt(~Element,~Logical,~original),~copying_end(~Element,~Logical,~original),~copied_at(~Element,~Logical,~original),
    Nat.add(covered,U32.to_nat(count)),0n,Copying{logical,slot,tree,length,depth,balanced,depth_ok,capacity_ok,room,covered,before,first,count,part_tree,part_length,part,destination,destination_exact,first_exact,inside},Order.reflexive(Nat.add(covered,U32.to_nat(count))))

# ---- The state of a slot ----

# Cells before position are placed: a slot that ends by then holds its
# piece, a slot that starts before it is sized, ready and covered up to it,
# and a slot after it is still empty.
def SlotWhen(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,done: Bool,started: Bool,+logical: Logical,+position: Nat,+slot: U32,+span: U32,
  +content: Buffer.Content<Element>) -> Data:
  match done:
    case True{}: Placed.PieceHolds(~Element,~Logical,~original,logical,slot,span,content)
    case False{}:
      match started:
        case True{}:
          KernelGemm.Both<{Buffer.content_length(~Element,content) == U32.to_nat(span) : Nat},KernelGemm.Both<Buffer.ContentReady<Element,content>,
            Covered(~Element,~Logical,~original,logical,slot,Nat.sub(position,U32.to_nat(slot)),Buffer.content_tree(~Element,content),Buffer.content_length(~Element,content))>>
        case False{}: {Buffer.content_length(~Element,content) == 0n : Nat}

def SlotState(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+position: Nat,+slot: U32,+span: U32,+content: Buffer.Content<Element>) -> Data:
  SlotWhen(~Element,~Logical,~original,Nat.is_le(Nat.add(U32.to_nat(slot),U32.to_nat(span)),position),Nat.is_lt(U32.to_nat(slot),position),logical,position,slot,span,content)

# ---- Cells of a slot ----

def shifted(+position: Nat,+start: Nat,+count: Nat,+ordered: {Nat.is_le(start,position) == True{} : Bool}) ->
  {Nat.add(position,count) == Nat.add(start,Nat.add(Nat.sub(position,start),count)) : Nat}:
  Equal.trans(Nat,Nat.add(position,count),Nat.add(Nat.add(start,Nat.sub(position,start)),count),Nat.add(start,Nat.add(Nat.sub(position,start),count)),
    Equal.cong(Nat,Nat,number => Nat.add(number,count),position,Nat.add(start,Nat.sub(position,start)),
      Equal.sym(Nat,Nat.add(start,Nat.sub(position,start)),position,Order.restore_shift(start,position,ordered))),
    Addition.associative(start,Nat.sub(position,start),count))

def covered_after(+position: Nat,+start: Nat,+count: Nat,+ordered: {Nat.is_le(start,position) == True{} : Bool}) ->
  {Nat.sub(Nat.add(position,count),start) == Nat.add(Nat.sub(position,start),count) : Nat}:
  Equal.trans(Nat,Nat.sub(Nat.add(position,count),start),Nat.sub(Nat.add(start,Nat.add(Nat.sub(position,start),count)),start),Nat.add(Nat.sub(position,start),count),
    Equal.cong(Nat,Nat,number => Nat.sub(number,start),Nat.add(position,count),Nat.add(start,Nat.add(Nat.sub(position,start),count)),shifted(position,start,count,ordered)),
    Order.subtract_shift(start,Nat.add(Nat.sub(position,start),count)))

def covered_room(+position: Nat,+start: Nat,+count: Nat,+span: Nat,+ordered: {Nat.is_le(start,position) == True{} : Bool},
  +finish: {Nat.is_le(Nat.add(position,count),Nat.add(start,span)) == True{} : Bool}) -> {Nat.is_le(Nat.add(Nat.sub(position,start),count),span) == True{} : Bool}:
  Order.cancel_offset_le(start,Nat.add(Nat.sub(position,start),count),span,
    {%shifted(position,start,count,ordered) :
        {Nat.is_le(_,Nat.add(start,span)) == True{} : Bool}
      finish : {Nat.is_le(Nat.add(start,Nat.add(Nat.sub(position,start),count)),Nat.add(start,span)) == True{} : Bool}})

def covered_full(+position: Nat,+start: Nat,+count: Nat,+span: Nat,+ordered: {Nat.is_le(start,position) == True{} : Bool},
  +equal: {Nat.add(position,count) == Nat.add(start,span) : Nat}) -> {Nat.add(Nat.sub(position,start),count) == span : Nat}:
  Order.cancel_sum(start,Nat.add(Nat.sub(position,start),count),span,
    Equal.trans(Nat,Nat.add(start,Nat.add(Nat.sub(position,start),count)),Nat.add(position,count),Nat.add(start,span),
      Equal.sym(Nat,Nat.add(position,count),Nat.add(start,Nat.add(Nat.sub(position,start),count)),shifted(position,start,count,ordered)),equal))

# ---- A part copied into a sized slot ----

# The state of the slot after the copy, by whether the part ends it.
def part_state(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,done: Bool,+logical: Logical,+position: Nat,+slot: U32,+span: U32,+count: U32,
  +tree: Tree.Tree<Element>,+length: Nat,+ready: Buffer.ContentReady<Element,Buffer.Content{tree,length}>,+sized: {length == U32.to_nat(span) : Nat},
  +start: {Nat.is_le(U32.to_nat(slot),position) == True{} : Bool},+finish: {Nat.is_le(Nat.add(position,U32.to_nat(count)),Nat.add(U32.to_nat(slot),U32.to_nat(span))) == True{} : Bool},
  +covered: Covered(~Element,~Logical,~original,logical,slot,Nat.add(Nat.sub(position,U32.to_nat(slot)),U32.to_nat(count)),tree,length),+done_exact: {Nat.is_le(Nat.add(U32.to_nat(slot),U32.to_nat(span)),Nat.add(position,U32.to_nat(count))) == done : Bool}) ->
  SlotWhen(~Element,~Logical,~original,done,True{},logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.Content{tree,length}):
  match done:
    case True{}:
      +full = covered_full(position,U32.to_nat(slot),U32.to_nat(count),U32.to_nat(span),start,Order.antisymmetric(Nat.add(position,U32.to_nat(count)),Nat.add(U32.to_nat(slot),U32.to_nat(span)),finish,done_exact))
      KernelGemm.Both{{%full : Covered(~Element,~Logical,~original,logical,slot,_,tree,length)
          covered : Covered(~Element,~Logical,~original,logical,slot,U32.to_nat(span),tree,length)},
        KernelGemm.Both{sized,ready}}
    case False{}:
      KernelGemm.Both{sized,KernelGemm.Both{ready,
        {%Equal.sym(Nat,Nat.sub(Nat.add(position,U32.to_nat(count)),U32.to_nat(slot)),Nat.add(Nat.sub(position,U32.to_nat(slot)),U32.to_nat(count)),covered_after(position,U32.to_nat(slot),U32.to_nat(count),start)) : Covered(~Element,~Logical,~original,logical,slot,_,tree,length)
          covered : Covered(~Element,~Logical,~original,logical,slot,Nat.sub(Nat.add(position,U32.to_nat(count)),U32.to_nat(slot)),tree,length)}}}

# A part copied into a sized slot covered up to the part's first cell.
def step_part(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+position: Nat,+slot: U32,+span: U32,+first: U32,+count: U32,
  +part: Buffer.Model<Element>,+target: Buffer.Model<Element>,+first_exact: {U32.to_nat(first) == position : Nat},+start: {Nat.is_le(U32.to_nat(slot),position) == True{} : Bool},
  +finish: {Nat.is_le(Nat.add(position,U32.to_nat(count)),Nat.add(U32.to_nat(slot),U32.to_nat(span))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +ready: Buffer.ContentReady<Element,Buffer.content(~Element,target)>,+sized: {Buffer.length(~Element,target) == U32.to_nat(span) : Nat},
  +before: Covered(~Element,~Logical,~original,logical,slot,Nat.sub(position,U32.to_nat(slot)),Buffer.tree(~Element,target),Buffer.length(~Element,target)),
  +holds: Placed.PieceHolds(~Element,~Logical,~original,logical,first,count,Buffer.content(~Element,part))) ->
  SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,
    Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Copy.range(~Element,count,0,(first - slot : U32),Buffer.pack(~Element,part),Buffer.pack(~Element,target)))))):
  Buffer.Model{+part_tree,+part_length,+part_certificate} = part
  Buffer.Model{+tree,+length,+certificate} = target
  Buffer.ContentReady{+depth,+balanced,+depth_ok,+room} = ready
  KernelGemm.Both{+values,+rest} = holds
  +capacity_ok = ready_capacity(~Element,Buffer.Content{tree,length},ready)
  +room_part = covered_room(position,U32.to_nat(slot),U32.to_nat(count),U32.to_nat(span),start,finish)
  +inside = {%Equal.sym(Nat,length,U32.to_nat(span),sized) : {Nat.is_le(Nat.add(Nat.sub(position,U32.to_nat(slot)),U32.to_nat(count)),_) == True{} : Bool} room_part : {Nat.is_le(Nat.add(Nat.sub(position,U32.to_nat(slot)),U32.to_nat(count)),length) == True{} : Bool}}
  +covered_cap = Order.transitive(Nat.sub(position,U32.to_nat(slot)),length,U32.to_nat(Tree.capacity(Element,tree)),Order.transitive(Nat.sub(position,U32.to_nat(slot)),Nat.add(Nat.sub(position,U32.to_nat(slot)),U32.to_nat(count)),length,Order.le_add(Nat.sub(position,U32.to_nat(slot)),U32.to_nat(count)),inside),room)
  +ordered = {%Equal.sym(Nat,U32.to_nat(first),position,first_exact) : {Nat.is_le(U32.to_nat(slot),_) == True{} : Bool} start : {Nat.is_le(U32.to_nat(slot),U32.to_nat(first)) == True{} : Bool}}
  +destination_exact = Equal.trans(Nat,U32.to_nat((first - slot : U32)),U32.to_nat(U32.from_nat(Nat.sub(U32.to_nat(first),U32.to_nat(slot)))),Nat.sub(position,U32.to_nat(slot)),
    Equal.cong(U32,Nat,word => U32.to_nat(word),(first - slot : U32),U32.from_nat(Nat.sub(U32.to_nat(first),U32.to_nat(slot))),Arithmetic.native_subtract(first,slot,ordered)),
    {%Equal.sym(Nat,U32.to_nat(first),position,first_exact) : {U32.to_nat(U32.from_nat(Nat.sub(_,U32.to_nat(slot)))) == Nat.sub(position,U32.to_nat(slot)) : Nat}
      Bounds.roundtrip(Nat.sub(position,U32.to_nat(slot)),Tree.capacity(Element,tree),capacity_ok,covered_cap) : {U32.to_nat(U32.from_nat(Nat.sub(U32.to_nat(first),U32.to_nat(slot)))) == Nat.sub(position,U32.to_nat(slot)) : Nat}})
  +first_cover = Equal.trans(Nat,U32.to_nat(first),position,Nat.add(U32.to_nat(slot),Nat.sub(position,U32.to_nat(slot))),first_exact,Equal.sym(Nat,Nat.add(U32.to_nat(slot),Nat.sub(position,U32.to_nat(slot))),position,Order.restore_shift(U32.to_nat(slot),position,start)))
  +covered = copied_covered(~Element,~Logical,~original,logical,slot,tree,length,depth,balanced,depth_ok,capacity_ok,room,Nat.sub(position,U32.to_nat(slot)),before,first,count,part_tree,part_length,values,(first - slot : U32),destination_exact,first_cover,inside)
  +started = Order.le_lt_trans(U32.to_nat(slot),position,Nat.add(position,U32.to_nat(count)),start,
    {%Addition.zero_right(position) : {Nat.is_lt(_,Nat.add(position,U32.to_nat(count))) == True{} : Bool}
      Order.offset_lt(position,0n,U32.to_nat(count),positive) : {Nat.is_lt(position,Nat.add(position,U32.to_nat(count))) == True{} : Bool}})
  %Equal.sym(Buffer.Content<Element>,
      Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
        Copy.range(~Element,count,0,(first - slot : U32),Buffer.pack(~Element,Buffer.Model{part_tree,part_length,part_certificate}),Buffer.pack(~Element,Buffer.Model{tree,length,certificate}))))),
      Buffer.Content{Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,(first - slot : U32)),tree),length},copied_content(~Element,count,(first - slot : U32),Buffer.Model{part_tree,part_length,part_certificate},Buffer.Model{tree,length,certificate})) :
    SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,_)
  %Equal.sym(Bool,Nat.is_lt(U32.to_nat(slot),Nat.add(position,U32.to_nat(count))),True{},started) : SlotWhen(~Element,~Logical,~original,Nat.is_le(Nat.add(U32.to_nat(slot),U32.to_nat(span)),Nat.add(position,U32.to_nat(count))),_,logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.Content{Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,(first - slot : U32)),tree),length})
  part_state(~Element,~Logical,~original,Nat.is_le(Nat.add(U32.to_nat(slot),U32.to_nat(span)),Nat.add(position,U32.to_nat(count))),logical,position,slot,span,count,Writes.pure(Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,(first - slot : U32)),tree),length,
    Writes.content_ready(~Element,Contiguous.writes(~Element,U32.to_nat(count),part_tree,0,0,(first - slot : U32)),tree,length,ready),sized,start,finish,covered,{==})

# ---- A part deposited in its slot ----

# A part that is its whole slot is the slot.
def step_whole(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+position: Nat,+slot: U32,+span: U32,+first: U32,+count: U32,+fill: Element,+data: Buffer.Model<Element>,+part: Buffer.Model<Element>,
  +first_exact: {U32.to_nat(first) == position : Nat},+start: {Nat.is_le(U32.to_nat(slot),position) == True{} : Bool},+finish: {Nat.is_le(Nat.add(position,U32.to_nat(count)),Nat.add(U32.to_nat(slot),U32.to_nat(span))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+span_room: {U32.is_le(span,1073741824) == True{} : Bool},+holds: Placed.PieceHolds(~Element,~Logical,~original,logical,first,count,Buffer.content(~Element,part)),
  +whole_exact: {U32.is_eq(slot,first) && U32.is_eq(span,count) == True{} : Bool}) -> SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.content(~Element,Placed.slot_model(~Element,True{},data,slot,span,first,count,fill,part))):
  +slot_eq = Arithmetic.equal_of_is_eq(slot,first,Boolean.and_true_left(U32.is_eq(slot,first),U32.is_eq(span,count),whole_exact))
  +span_eq = Arithmetic.equal_of_is_eq(span,count,Boolean.and_true_right(U32.is_eq(slot,first),U32.is_eq(span,count),whole_exact))
  +at = Equal.trans(Nat,U32.to_nat(slot),U32.to_nat(first),position,Equal.cong(U32,Nat,word => U32.to_nat(word),slot,first,slot_eq),first_exact)
  +ends = Equal.trans(Nat,Nat.add(U32.to_nat(slot),U32.to_nat(span)),Nat.add(position,U32.to_nat(span)),Nat.add(position,U32.to_nat(count)),Equal.cong(Nat,Nat,number => Nat.add(number,U32.to_nat(span)),U32.to_nat(slot),position,at),
    Equal.cong(U32,Nat,word => Nat.add(position,U32.to_nat(word)),span,count,span_eq))
  +done = {%Equal.sym(Nat,Nat.add(U32.to_nat(slot),U32.to_nat(span)),Nat.add(position,U32.to_nat(count)),ends) : {Nat.is_le(_,Nat.add(position,U32.to_nat(count))) == True{} : Bool} Order.reflexive(Nat.add(position,U32.to_nat(count))) : {Nat.is_le(Nat.add(U32.to_nat(slot),U32.to_nat(span)),Nat.add(position,U32.to_nat(count))) == True{} : Bool}}
  {%Equal.sym(Bool,Nat.is_le(Nat.add(U32.to_nat(slot),U32.to_nat(span)),Nat.add(position,U32.to_nat(count))),True{},done) : SlotWhen(~Element,~Logical,~original,_,Nat.is_lt(U32.to_nat(slot),Nat.add(position,U32.to_nat(count))),logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.content(~Element,part))
    {%Equal.sym(U32,slot,first,slot_eq) : Placed.PieceHolds(~Element,~Logical,~original,logical,_,span,Buffer.content(~Element,part))
      {%Equal.sym(U32,span,count,span_eq) : Placed.PieceHolds(~Element,~Logical,~original,logical,first,_,Buffer.content(~Element,part))
        holds : Placed.PieceHolds(~Element,~Logical,~original,logical,first,span,Buffer.content(~Element,part))} : Placed.PieceHolds(~Element,~Logical,~original,logical,slot,span,Buffer.content(~Element,part))} :
    SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.content(~Element,Placed.slot_model(~Element,True{},data,slot,span,first,count,fill,part)))}

# A later part goes into its sized slot, the first part of a slot into a
# fresh buffer of the slot's cells.
def step_old(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,done: Bool,started: Bool,+logical: Logical,+position: Nat,+slot: U32,+span: U32,+first: U32,+count: U32,+fill: Element,+data: Buffer.Model<Element>,+part: Buffer.Model<Element>,
  +first_exact: {U32.to_nat(first) == position : Nat},+start: {Nat.is_le(U32.to_nat(slot),position) == True{} : Bool},+finish: {Nat.is_le(Nat.add(position,U32.to_nat(count)),Nat.add(U32.to_nat(slot),U32.to_nat(span))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+span_room: {U32.is_le(span,1073741824) == True{} : Bool},+holds: Placed.PieceHolds(~Element,~Logical,~original,logical,first,count,Buffer.content(~Element,part)),
  +state: SlotWhen(~Element,~Logical,~original,done,started,logical,position,slot,span,Buffer.content(~Element,data)),
  +done_exact: {Nat.is_le(Nat.add(U32.to_nat(slot),U32.to_nat(span)),position) == done : Bool},+started_exact: {Nat.is_lt(U32.to_nat(slot),position) == started : Bool}) -> SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.content(~Element,Placed.slot_model(~Element,False{},data,slot,span,first,count,fill,part))):
  match done:
    case True{}:
      Empty.absurd(SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.content(~Element,Placed.slot_model(~Element,False{},data,slot,span,first,count,fill,part))),Held.lt_le_absurd(position,Nat.add(U32.to_nat(slot),U32.to_nat(span)),
        Order.lt_le(position,Nat.add(position,U32.to_nat(count)),Nat.add(U32.to_nat(slot),U32.to_nat(span)),
          {%Addition.zero_right(position) : {Nat.is_lt(_,Nat.add(position,U32.to_nat(count))) == True{} : Bool} Order.offset_lt(position,0n,U32.to_nat(count),positive) : {Nat.is_lt(position,Nat.add(position,U32.to_nat(count))) == True{} : Bool}},finish),
        done_exact))
    case False{}:
      match started:
        case True{}:
          Buffer.Model{+tree,+length,+certificate} = data
          KernelGemm.Both{+sized,+facts} = state
          KernelGemm.Both{+ready,+before} = facts
          +kept = {%Equal.sym(Nat,length,U32.to_nat(span),sized) : {Nat.is_lt(_,U32.to_nat(span)) == False{} : Bool} Order.not_below(U32.to_nat(span),U32.to_nat(span),Order.reflexive(U32.to_nat(span))) : {Nat.is_lt(length,U32.to_nat(span)) == False{} : Bool}}
          {%Equal.sym(Bool,Nat.is_lt(length,U32.to_nat(span)),False{},kept) :
              SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
                Copy.range(~Element,count,0,(first - slot : U32),Buffer.pack(~Element,part),Tensors.sized_when(~Element,_,Storage.Buffer{Tree.pack(Element,tree),length,certificate},span,fill))))))
            step_part(~Element,~Logical,~original,logical,position,slot,span,first,count,part,Buffer.Model{tree,length,certificate},first_exact,start,finish,positive,ready,sized,before,holds) :
            SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.content(~Element,Placed.slot_model(~Element,False{},Buffer.Model{tree,length,certificate},slot,span,first,count,fill,part)))}
        case False{}:
          Buffer.Model{+tree,+length,+certificate} = data
          +at = Order.antisymmetric(U32.to_nat(slot),position,start,Order.not_less(U32.to_nat(slot),position,started_exact))
          +span_positive = Order.cancel_offset_lt(U32.to_nat(slot),0n,U32.to_nat(span),
            {%Equal.sym(Nat,Nat.add(U32.to_nat(slot),0n),U32.to_nat(slot),Addition.zero_right(U32.to_nat(slot))) : {Nat.is_lt(_,Nat.add(U32.to_nat(slot),U32.to_nat(span))) == True{} : Bool}
              %Equal.sym(Nat,U32.to_nat(slot),position,at) : {Nat.is_lt(_,Nat.add(U32.to_nat(slot),U32.to_nat(span))) == True{} : Bool}
              Order.lt_le(position,Nat.add(position,U32.to_nat(count)),Nat.add(U32.to_nat(slot),U32.to_nat(span)),
                {%Addition.zero_right(position) : {Nat.is_lt(_,Nat.add(position,U32.to_nat(count))) == True{} : Bool} Order.offset_lt(position,0n,U32.to_nat(count),positive) : {Nat.is_lt(position,Nat.add(position,U32.to_nat(count))) == True{} : Bool}},finish) :
              {Nat.is_lt(Nat.add(U32.to_nat(slot),0n),Nat.add(U32.to_nat(slot),U32.to_nat(span))) == True{} : Bool}})
          +short = {%Equal.sym(Nat,length,0n,state) : {Nat.is_lt(_,U32.to_nat(span)) == True{} : Bool} span_positive : {Nat.is_lt(length,U32.to_nat(span)) == True{} : Bool}}
          +sized_span = Depth.bounded(span,span_room)
          +fresh_content = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span))),Buffer.contents(~Element,Tensors.buffer(~Element,fill,span)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(span),fill),U32.to_nat(span)},
            Buffer.remodel_content(~Element,Tensors.buffer(~Element,fill,span)),AllocationLaws.contents(~Element,Tensors.depth_for(span),fill,U32.to_nat(span)))
          +fresh_ready = {%Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span))),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(span),fill),U32.to_nat(span)},fresh_content) : Buffer.ContentReady<Element,_>
            Buffer.ContentReady{Tensors.depth_for(span),Shape.fresh(Element,Tensors.depth_for(span),fill),Depth.allocation_depth(span,sized_span),
              {%Equal.sym(U32,Tree.capacity(Element,Tree.fresh(Element,Tensors.depth_for(span),fill)),U32.shln(1,Tensors.depth_for(span)),Shape.capacity(Element,Tensors.depth_for(span),Tree.fresh(Element,Tensors.depth_for(span),fill),Shape.fresh(Element,Tensors.depth_for(span),fill))) :
                  {Nat.is_le(U32.to_nat(span),U32.to_nat(_)) == True{} : Bool}
                Depth.allocation_room(span,sized_span) : {Nat.is_le(U32.to_nat(span),U32.to_nat(Tree.capacity(Element,Tree.fresh(Element,Tensors.depth_for(span),fill)))) == True{} : Bool}}} :
            Buffer.ContentReady<Element,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span)))>}
          +fresh_length = Equal.trans(Nat,Buffer.length(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span))),Buffer.content_length(~Element,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span)))),U32.to_nat(span),
            Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span)))),Buffer.length(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span))),Buffer.content_length_of(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span)))),
            Equal.cong(Buffer.Content<Element>,Nat,content => Buffer.content_length(~Element,content),Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span))),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(span),fill),U32.to_nat(span)},fresh_content))
          +none = Equal.trans(Nat,Nat.sub(position,U32.to_nat(slot)),Nat.sub(Nat.add(U32.to_nat(slot),0n),U32.to_nat(slot)),0n,
            Equal.cong(Nat,Nat,number => Nat.sub(number,U32.to_nat(slot)),position,Nat.add(U32.to_nat(slot),0n),
              Equal.trans(Nat,position,U32.to_nat(slot),Nat.add(U32.to_nat(slot),0n),Equal.sym(Nat,U32.to_nat(slot),position,at),Equal.sym(Nat,Nat.add(U32.to_nat(slot),0n),U32.to_nat(slot),Addition.zero_right(U32.to_nat(slot))))),
            Order.subtract_shift(U32.to_nat(slot),0n))
          +before = {%Equal.sym(Nat,Nat.sub(position,U32.to_nat(slot)),0n,none) : Covered(~Element,~Logical,~original,logical,slot,_,Buffer.tree(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span))),Buffer.length(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span))))
            Unit{} : Covered(~Element,~Logical,~original,logical,slot,Nat.sub(position,U32.to_nat(slot)),Buffer.tree(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span))),Buffer.length(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span))))}
          {%Equal.sym(Bool,Nat.is_lt(length,U32.to_nat(span)),True{},short) :
              SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
                Copy.range(~Element,count,0,(first - slot : U32),Buffer.pack(~Element,part),Tensors.sized_when(~Element,_,Storage.Buffer{Tree.pack(Element,tree),length,certificate},span,fill))))))
            %Buffer.remodeled(~Element,Tensors.buffer(~Element,fill,span)) :
              SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.content(~Element,Buffer.remodel(~Element,Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,
                Copy.range(~Element,count,0,(first - slot : U32),Buffer.pack(~Element,part),_)))))
            step_part(~Element,~Logical,~original,logical,position,slot,span,first,count,part,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,span)),first_exact,start,finish,positive,fresh_ready,fresh_length,before,holds) :
            SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.content(~Element,Placed.slot_model(~Element,False{},Buffer.Model{tree,length,certificate},slot,span,first,count,fill,part)))}

# A part deposited in the slot that holds it moves the slot to its state
# after the part.
def slot_step(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,whole: Bool,+logical: Logical,+position: Nat,+slot: U32,+span: U32,+first: U32,+count: U32,+fill: Element,+data: Buffer.Model<Element>,+part: Buffer.Model<Element>,
  +first_exact: {U32.to_nat(first) == position : Nat},+start: {Nat.is_le(U32.to_nat(slot),position) == True{} : Bool},+finish: {Nat.is_le(Nat.add(position,U32.to_nat(count)),Nat.add(U32.to_nat(slot),U32.to_nat(span))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},+span_room: {U32.is_le(span,1073741824) == True{} : Bool},+holds: Placed.PieceHolds(~Element,~Logical,~original,logical,first,count,Buffer.content(~Element,part)),
  +state: SlotState(~Element,~Logical,~original,logical,position,slot,span,Buffer.content(~Element,data)),+whole_exact: {U32.is_eq(slot,first) && U32.is_eq(span,count) == whole : Bool}) ->
  SlotState(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),slot,span,Buffer.content(~Element,Placed.slot_model(~Element,whole,data,slot,span,first,count,fill,part))):
  match whole:
    case True{}: step_whole(~Element,~Logical,~original,logical,position,slot,span,first,count,fill,data,part,first_exact,start,finish,positive,span_room,holds,whole_exact)
    case False{}:
      step_old(~Element,~Logical,~original,Nat.is_le(Nat.add(U32.to_nat(slot),U32.to_nat(span)),position),Nat.is_lt(U32.to_nat(slot),position),logical,position,slot,span,first,count,fill,data,part,first_exact,start,finish,positive,span_room,holds,state,{==},{==})

# ---- The slots of a layout ----

# The cells of a part lie in one leaf of a layout.
def Inside(layout: Pieces.Ranges,+position: Nat,+count: Nat) -> Data:
  match layout:
    case Pieces.Range{+first,+span}:
      KernelGemm.Both<{Nat.is_le(U32.to_nat(first),position) == True{} : Bool},{Nat.is_le(Nat.add(position,count),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}>
    case Pieces.Split{+left,+right}:
      Placed.Side<KernelGemm.Both<{Nat.is_lt(position,Runs.range_high(left)) == True{} : Bool},Inside(left,position,count)>,
        KernelGemm.Both<{Nat.is_le(Runs.range_low(right),position) == True{} : Bool},Inside(right,position,count)>>

# Every slot is in its state at position.
def Good(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+position: Nat,model: PiecesModel.Model<Element>) -> Data:
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}: SlotState(~Element,~Logical,~original,logical,position,first,rows,Buffer.content(~Element,buffer))
    case PiecesModel.Fork{+left,+right}: KernelGemm.Both<Good(~Element,~Logical,~original,logical,position,left),Good(~Element,~Logical,~original,logical,position,right)>

def run_order(+ranges: Pieces.Ranges,+unit: Nat,+bound: Nat,+run: Runs.Run(ranges,unit,bound)) -> {Nat.is_le(Runs.range_low(ranges),Runs.range_high(ranges)) == True{} : Bool}:
  Bounds.strict_le(Runs.range_low(ranges),Runs.range_high(ranges),
    Runs.fits_less(unit,Runs.range_low(ranges),Runs.range_high(ranges),bound,Runs.run_fits(ranges,unit,bound,run)))

def inside_end(layout: Pieces.Ranges,+position: Nat,+count: Nat,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound),+inside: Inside(layout,position,count)) ->
  {Nat.is_le(Nat.add(position,count),Runs.range_high(layout)) == True{} : Bool}:
  match layout:
    case Pieces.Range{+first,+span}:
      KernelGemm.Both{+start,+finish} = inside
      finish
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      match inside:
        case Placed.OnLeft{+facts}:
          KernelGemm.Both{+below,+left_inside} = facts
          Order.transitive(Nat.add(position,count),Runs.range_high(left),Runs.range_high(right),inside_end(left,position,count,unit,bound,left_run,left_inside),
            {%Equal.sym(Nat,Runs.range_high(left),Runs.range_low(right),glue) : {Nat.is_le(_,Runs.range_high(right)) == True{} : Bool}
              run_order(right,unit,bound,right_run) : {Nat.is_le(Runs.range_high(left),Runs.range_high(right)) == True{} : Bool}})
        case Placed.OnRight{+facts}:
          KernelGemm.Both{+above,+right_inside} = facts
          inside_end(right,position,count,unit,bound,right_run,right_inside)

# Slots that end by position stay placed.
def done_kept(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,model: PiecesModel.Model<Element>,+position: Nat,+later: Nat,
  +unit: Nat,+bound: Nat,+run: Runs.Run(PiecesModel.layout(~Element,model),unit,bound),+ended: {Nat.is_le(Runs.range_high(PiecesModel.layout(~Element,model)),position) == True{} : Bool},
  +ordered: {Nat.is_le(position,later) == True{} : Bool},+good: Good(~Element,~Logical,~original,logical,position,model)) -> Good(~Element,~Logical,~original,logical,later,model):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      {%Equal.sym(Bool,Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),later),True{},Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),position,later,ended,ordered)) :
          SlotWhen(~Element,~Logical,~original,_,Nat.is_lt(U32.to_nat(first),later),logical,later,first,rows,Buffer.content(~Element,buffer))
        {%ended : SlotWhen(~Element,~Logical,~original,_,Nat.is_lt(U32.to_nat(first),position),logical,position,first,rows,Buffer.content(~Element,buffer))
          good : SlotWhen(~Element,~Logical,~original,True{},Nat.is_lt(U32.to_nat(first),position),logical,position,first,rows,Buffer.content(~Element,buffer))} :
        SlotState(~Element,~Logical,~original,logical,later,first,rows,Buffer.content(~Element,buffer))}
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      KernelGemm.Both{+left_good,+right_good} = good
      KernelGemm.Both{
        done_kept(~Element,~Logical,~original,logical,left,position,later,unit,bound,left_run,
          Order.transitive(Runs.range_high(PiecesModel.layout(~Element,left)),Runs.range_high(PiecesModel.layout(~Element,right)),position,
            {%Equal.sym(Nat,Runs.range_high(PiecesModel.layout(~Element,left)),Runs.range_low(PiecesModel.layout(~Element,right)),glue) : {Nat.is_le(_,Runs.range_high(PiecesModel.layout(~Element,right))) == True{} : Bool}
              run_order(PiecesModel.layout(~Element,right),unit,bound,right_run) : {Nat.is_le(Runs.range_high(PiecesModel.layout(~Element,left)),Runs.range_high(PiecesModel.layout(~Element,right))) == True{} : Bool}},ended),ordered,left_good),
        done_kept(~Element,~Logical,~original,logical,right,position,later,unit,bound,right_run,ended,ordered,right_good)}

# Slots that start at or after later stay empty.
def untouched_kept(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,model: PiecesModel.Model<Element>,+position: Nat,+later: Nat,
  +unit: Nat,+bound: Nat,+run: Runs.Run(PiecesModel.layout(~Element,model),unit,bound),+starts: {Nat.is_le(later,Runs.range_low(PiecesModel.layout(~Element,model))) == True{} : Bool},
  +ordered: {Nat.is_le(position,later) == True{} : Bool},+good: Good(~Element,~Logical,~original,logical,position,model)) -> Good(~Element,~Logical,~original,logical,later,model):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      +less = Runs.fits_less(unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),bound,run)
      +before = Order.transitive(position,later,U32.to_nat(first),ordered,starts)
      {%Equal.sym(Bool,Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),later),False{},Held.lt_not_le(later,Nat.add(U32.to_nat(first),U32.to_nat(rows)),Order.le_lt_trans(later,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),starts,less))) :
          SlotWhen(~Element,~Logical,~original,_,Nat.is_lt(U32.to_nat(first),later),logical,later,first,rows,Buffer.content(~Element,buffer))
        %Equal.sym(Bool,Nat.is_lt(U32.to_nat(first),later),False{},Order.not_below(later,U32.to_nat(first),starts)) :
          SlotWhen(~Element,~Logical,~original,False{},_,logical,later,first,rows,Buffer.content(~Element,buffer))
        {%Held.lt_not_le(position,Nat.add(U32.to_nat(first),U32.to_nat(rows)),Order.le_lt_trans(position,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(rows)),before,less)) :
            SlotWhen(~Element,~Logical,~original,_,False{},logical,position,first,rows,Buffer.content(~Element,buffer))
          %Order.not_below(position,U32.to_nat(first),before) : SlotWhen(~Element,~Logical,~original,Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),position),_,logical,position,first,rows,Buffer.content(~Element,buffer))
          good : SlotWhen(~Element,~Logical,~original,False{},False{},logical,position,first,rows,Buffer.content(~Element,buffer))} :
        SlotState(~Element,~Logical,~original,logical,later,first,rows,Buffer.content(~Element,buffer))}
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      KernelGemm.Both{+left_good,+right_good} = good
      KernelGemm.Both{untouched_kept(~Element,~Logical,~original,logical,left,position,later,unit,bound,left_run,starts,ordered,left_good),
        untouched_kept(~Element,~Logical,~original,logical,right,position,later,unit,bound,right_run,
          Order.transitive(later,Runs.range_low(PiecesModel.layout(~Element,left)),Runs.range_low(PiecesModel.layout(~Element,right)),starts,
            {%glue : {Nat.is_le(Runs.range_low(PiecesModel.layout(~Element,left)),_) == True{} : Bool}
              run_order(PiecesModel.layout(~Element,left),unit,bound,left_run) : {Nat.is_le(Runs.range_low(PiecesModel.layout(~Element,left)),Runs.range_low(PiecesModel.layout(~Element,right))) == True{} : Bool}}),ordered,right_good)}

# A part deposited in its slot of a run of slots: the slot moves to its state
# after the part and every other slot keeps its state.
def deposit_good(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,model: PiecesModel.Model<Element>,+position: Nat,+first: U32,+count: U32,
  +fill: Element,+part: Buffer.Model<Element>,+unit: Nat,+bound: Nat,+limits: Machine.Limits,+run: Runs.Run(PiecesModel.layout(~Element,model),unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+first_exact: {U32.to_nat(first) == position : Nat},+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +inside: Inside(PiecesModel.layout(~Element,model),position,U32.to_nat(count)),+holds: Placed.PieceHolds(~Element,~Logical,~original,logical,first,count,Buffer.content(~Element,part)),+good: Good(~Element,~Logical,~original,logical,position,model)) ->
  Good(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),Placed.deposit_at(~Element,model,PiecesModel.layout(~Element,model),first,count,fill,part)):
  match model:
    case PiecesModel.Leaf{+data,+slot,+span}:
      KernelGemm.Both{+start,+finish} = inside
      +span_room = Order.word_le(span,Machine.capacity(limits),1073741824,
        {%Equal.sym(Bool,U32.is_le(span,Machine.capacity(limits)),Nat.is_le(U32.to_nat(span),U32.to_nat(Machine.capacity(limits))),Comparison.le(span,Machine.capacity(limits))) : {_ == True{} : Bool}
          Order.transitive(U32.to_nat(span),Nat.add(U32.to_nat(slot),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits)),Order.right_le(U32.to_nat(slot),U32.to_nat(span)),
            Order.transitive(Nat.add(U32.to_nat(slot),U32.to_nat(span)),bound,U32.to_nat(Machine.capacity(limits)),Runs.fits_inside(unit,U32.to_nat(slot),Nat.add(U32.to_nat(slot),U32.to_nat(span)),bound,run),room)) : {U32.is_le(span,Machine.capacity(limits)) == True{} : Bool}},
        Limits.capacity_allocatable(limits))
      slot_step(~Element,~Logical,~original,U32.is_eq(slot,first) && U32.is_eq(span,count),logical,position,slot,span,first,count,fill,data,part,first_exact,start,finish,positive,span_room,holds,good,{==})
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      match inside:
        case Placed.OnLeft{+facts}:
          KernelGemm.Both{+below,+left_inside} = facts
          KernelGemm.Both{+left_good,+right_good} = good
          +side = {%Equal.sym(Bool,U32.is_lt(first,Pieces.first(PiecesModel.layout(~Element,right))),Nat.is_lt(U32.to_nat(first),U32.to_nat(Pieces.first(PiecesModel.layout(~Element,right)))),Comparison.lt(first,Pieces.first(PiecesModel.layout(~Element,right)))) : {_ == True{} : Bool}
            %Equal.sym(Nat,U32.to_nat(first),position,first_exact) : {Nat.is_lt(_,U32.to_nat(Pieces.first(PiecesModel.layout(~Element,right)))) == True{} : Bool}
            %Equal.sym(Nat,U32.to_nat(Pieces.first(PiecesModel.layout(~Element,right))),Runs.range_low(PiecesModel.layout(~Element,right)),Runs.first_low(PiecesModel.layout(~Element,right))) : {Nat.is_lt(position,_) == True{} : Bool}
            %glue : {Nat.is_lt(position,_) == True{} : Bool}
            below : {U32.is_lt(first,Pieces.first(PiecesModel.layout(~Element,right))) == True{} : Bool}}
          {%Equal.sym(Bool,U32.is_lt(first,Pieces.first(PiecesModel.layout(~Element,right))),True{},side) :
              Good(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),Placed.deposit(~Element,PiecesModel.Fork{left,right},Pieces.Split{PiecesModel.layout(~Element,left),PiecesModel.layout(~Element,right)},_,first,count,fill,part))
            KernelGemm.Both{deposit_good(~Element,~Logical,~original,logical,left,position,first,count,fill,part,unit,bound,limits,left_run,room,first_exact,positive,left_inside,holds,left_good),
              untouched_kept(~Element,~Logical,~original,logical,right,position,Nat.add(position,U32.to_nat(count)),unit,bound,right_run,
                {%glue : {Nat.is_le(Nat.add(position,U32.to_nat(count)),_) == True{} : Bool}
                  inside_end(PiecesModel.layout(~Element,left),position,U32.to_nat(count),unit,bound,left_run,left_inside) : {Nat.is_le(Nat.add(position,U32.to_nat(count)),Runs.range_low(PiecesModel.layout(~Element,right))) == True{} : Bool}},
                Order.le_add(position,U32.to_nat(count)),right_good)} :
            Good(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),Placed.deposit_at(~Element,PiecesModel.Fork{left,right},Pieces.Split{PiecesModel.layout(~Element,left),PiecesModel.layout(~Element,right)},first,count,fill,part))}
        case Placed.OnRight{+facts}:
          KernelGemm.Both{+above,+right_inside} = facts
          KernelGemm.Both{+left_good,+right_good} = good
          +side = {%Equal.sym(Bool,U32.is_lt(first,Pieces.first(PiecesModel.layout(~Element,right))),Nat.is_lt(U32.to_nat(first),U32.to_nat(Pieces.first(PiecesModel.layout(~Element,right)))),Comparison.lt(first,Pieces.first(PiecesModel.layout(~Element,right)))) : {_ == False{} : Bool}
            %Equal.sym(Nat,U32.to_nat(first),position,first_exact) : {Nat.is_lt(_,U32.to_nat(Pieces.first(PiecesModel.layout(~Element,right)))) == False{} : Bool}
            %Equal.sym(Nat,U32.to_nat(Pieces.first(PiecesModel.layout(~Element,right))),Runs.range_low(PiecesModel.layout(~Element,right)),Runs.first_low(PiecesModel.layout(~Element,right))) : {Nat.is_lt(position,_) == False{} : Bool}
            Order.not_below(Runs.range_low(PiecesModel.layout(~Element,right)),position,above) : {U32.is_lt(first,Pieces.first(PiecesModel.layout(~Element,right))) == False{} : Bool}}
          {%Equal.sym(Bool,U32.is_lt(first,Pieces.first(PiecesModel.layout(~Element,right))),False{},side) :
              Good(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),Placed.deposit(~Element,PiecesModel.Fork{left,right},Pieces.Split{PiecesModel.layout(~Element,left),PiecesModel.layout(~Element,right)},_,first,count,fill,part))
            KernelGemm.Both{
              done_kept(~Element,~Logical,~original,logical,left,position,Nat.add(position,U32.to_nat(count)),unit,bound,left_run,
                {%Equal.sym(Nat,Runs.range_high(PiecesModel.layout(~Element,left)),Runs.range_low(PiecesModel.layout(~Element,right)),glue) : {Nat.is_le(_,position) == True{} : Bool}
                  above : {Nat.is_le(Runs.range_high(PiecesModel.layout(~Element,left)),position) == True{} : Bool}},Order.le_add(position,U32.to_nat(count)),left_good),
              deposit_good(~Element,~Logical,~original,logical,right,position,first,count,fill,part,unit,bound,limits,right_run,room,first_exact,positive,right_inside,holds,right_good)} :
            Good(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(count)),Placed.deposit_at(~Element,PiecesModel.Fork{left,right},Pieces.Split{PiecesModel.layout(~Element,left),PiecesModel.layout(~Element,right)},first,count,fill,part))}

# ---- The parts of the tasks ----

# The cell after the pieces of a task, and after the tasks, from position.
def pieces_end(~Element: Data,pieces: PiecesModel.Model<Element>,+position: Nat) -> Nat:
  match pieces:
    case PiecesModel.Leaf{buffer,first,+rows}: Nat.add(position,U32.to_nat(rows))
    case PiecesModel.Fork{left,right}: pieces_end(~Element,right,pieces_end(~Element,left,position))

def tasks_end(~Element: Data,results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+position: Nat) -> Nat:
  match results:
    case Partition.Leaf{pieces,first,count,details}: pieces_end(~Element,pieces,position)
    case Partition.Fork{left,right}: tasks_end(~Element,right,tasks_end(~Element,left,position))

# The pieces of the tasks in order: each starts where the one before ends,
# lies in one slot, is not empty and holds its values.
def PartsAt(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+layout: Pieces.Ranges,+position: Nat,pieces: PiecesModel.Model<Element>) -> Data:
  match pieces:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      KernelGemm.Both<{U32.to_nat(first) == position : Nat},KernelGemm.Both<{Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
        KernelGemm.Both<Inside(layout,position,U32.to_nat(rows)),Placed.PieceHolds(~Element,~Logical,~original,logical)(first,rows,Buffer.content(~Element,buffer))>>>
    case PiecesModel.Fork{+left,+right}:
      KernelGemm.Both<PartsAt(~Element,~Logical,~original,logical,layout,position,left),PartsAt(~Element,~Logical,~original,logical,layout,pieces_end(~Element,left,position),right)>

def TasksParts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+layout: Pieces.Ranges,+position: Nat,results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>) -> Data:
  match results:
    case Partition.Leaf{+pieces,+first,+count,+details}: PartsAt(~Element,~Logical,~original,logical,layout,position,pieces)
    case Partition.Fork{+left,+right}:
      KernelGemm.Both<TasksParts(~Element,~Logical,~original,logical,layout,position,left),TasksParts(~Element,~Logical,~original,logical,layout,tasks_end(~Element,left,position),right)>

# Placing the pieces of the tasks in order moves the position to their end.
def pieces_good(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+layout: Pieces.Ranges,+fill: Element,pieces: PiecesModel.Model<Element>,
  +slots: PiecesModel.Model<Element>,+position: Nat,+unit: Nat,+bound: Nat,+limits: Machine.Limits,+run: Runs.Run(layout,unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+same: {PiecesModel.layout(~Element,slots) == layout : Pieces.Ranges},
  +facts: PartsAt(~Element,~Logical,~original,logical,layout,position,pieces),+good: Good(~Element,~Logical,~original,logical,position,slots)) ->
  Good(~Element,~Logical,~original,logical,pieces_end(~Element,pieces,position),Placed.placed_pieces(~Element,layout,fill,pieces,slots)):
  match pieces:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      KernelGemm.Both{+at,+rest} = facts
      KernelGemm.Both{+positive,+parts} = rest
      KernelGemm.Both{+inside,+holds} = parts
      {%Equal.sym(Bool,U32.is_zero(rows),False{},Arithmetic.word_positive_nonzero(rows,positive)) :
          Good(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(rows)),Placed.placed_when_model(~Element,_,layout,fill,first,rows,buffer,slots))
        %same : Good(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(rows)),Placed.deposit_at(~Element,slots,_,first,rows,fill,buffer))
        deposit_good(~Element,~Logical,~original,logical,slots,position,first,rows,fill,buffer,unit,bound,limits,
          {%Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,slots),layout,same) : Runs.Run(_,unit,bound) run : Runs.Run(PiecesModel.layout(~Element,slots),unit,bound)},
          room,at,positive,
          {%Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,slots),layout,same) : Inside(_,position,U32.to_nat(rows)) inside : Inside(PiecesModel.layout(~Element,slots),position,U32.to_nat(rows))},
          holds,good) :
        Good(~Element,~Logical,~original,logical,Nat.add(position,U32.to_nat(rows)),Placed.placed_pieces(~Element,layout,fill,PiecesModel.Leaf{buffer,first,rows},slots))}
    case PiecesModel.Fork{+left,+right}:
      KernelGemm.Both{+left_facts,+right_facts} = facts
      +middle = Placed.placed_pieces(~Element,layout,fill,left,slots)
      pieces_good(~Element,~Logical,~original,logical,layout,fill,right,middle,pieces_end(~Element,left,position),unit,bound,limits,run,room,
        Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,middle),PiecesModel.layout(~Element,slots),layout,Placed.placed_pieces_layout(~Element,layout,fill,left,slots),same),
        right_facts,pieces_good(~Element,~Logical,~original,logical,layout,fill,left,slots,position,unit,bound,limits,run,room,same,left_facts,good))

def tasks_good(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+layout: Pieces.Ranges,+fill: Element,results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,
  +slots: PiecesModel.Model<Element>,+position: Nat,+unit: Nat,+bound: Nat,+limits: Machine.Limits,+run: Runs.Run(layout,unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+same: {PiecesModel.layout(~Element,slots) == layout : Pieces.Ranges},
  +facts: TasksParts(~Element,~Logical,~original,logical,layout,position,results),+good: Good(~Element,~Logical,~original,logical,position,slots)) ->
  Good(~Element,~Logical,~original,logical,tasks_end(~Element,results,position),Placed.placed_tasks(~Element,layout,fill,results,slots)):
  match results:
    case Partition.Leaf{+pieces,+first,+count,+details}:
      pieces_good(~Element,~Logical,~original,logical,layout,fill,pieces,slots,position,unit,bound,limits,run,room,same,facts,good)
    case Partition.Fork{+left,+right}:
      KernelGemm.Both{+left_facts,+right_facts} = facts
      +middle = Placed.placed_tasks(~Element,layout,fill,left,slots)
      tasks_good(~Element,~Logical,~original,logical,layout,fill,right,middle,tasks_end(~Element,left,position),unit,bound,limits,run,room,
        Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,middle),PiecesModel.layout(~Element,slots),layout,Placed.placed_tasks_layout(~Element,layout,fill,left,slots),same),
        right_facts,tasks_good(~Element,~Logical,~original,logical,layout,fill,left,slots,position,unit,bound,limits,run,room,same,left_facts,good))

# Before any part, every slot of a run from position on is empty.
def empty_good(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,layout: Pieces.Ranges,+fill: Element,+position: Nat,+unit: Nat,+bound: Nat,
  +run: Runs.Run(layout,unit,bound),+starts: {Nat.is_le(position,Runs.range_low(layout)) == True{} : Bool}) -> Good(~Element,~Logical,~original,logical,position,Placed.empty_slots(~Element,layout,fill)):
  match layout:
    case Pieces.Range{+first,+count}:
      +less = Runs.fits_less(unit,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),bound,run)
      +empty = Equal.cong(Buffer.Content<Element>,Nat,content => Buffer.content_length(~Element,content),Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,0))),
        Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),U32.to_nat(0)},
        Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,0))),Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),
          Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),U32.to_nat(0)},Buffer.remodel_content(~Element,Tensors.buffer(~Element,fill,0)),
          AllocationLaws.contents(~Element,Tensors.depth_for(0),fill,U32.to_nat(0))))
      {%Equal.sym(Bool,Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),position),False{},Held.lt_not_le(position,Nat.add(U32.to_nat(first),U32.to_nat(count)),Order.le_lt_trans(position,U32.to_nat(first),Nat.add(U32.to_nat(first),U32.to_nat(count)),starts,less))) :
          SlotWhen(~Element,~Logical,~original,_,Nat.is_lt(U32.to_nat(first),position),logical,position,first,count,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,0))))
        %Equal.sym(Bool,Nat.is_lt(U32.to_nat(first),position),False{},Order.not_below(position,U32.to_nat(first),starts)) :
          SlotWhen(~Element,~Logical,~original,False{},_,logical,position,first,count,Buffer.content(~Element,Buffer.remodel(~Element,Tensors.buffer(~Element,fill,0))))
        empty : Good(~Element,~Logical,~original,logical,position,PiecesModel.Leaf{Buffer.remodel(~Element,Tensors.buffer(~Element,fill,0)),first,count})}
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      KernelGemm.Both{empty_good(~Element,~Logical,~original,logical,left,fill,position,unit,bound,left_run,starts),
        empty_good(~Element,~Logical,~original,logical,right,fill,position,unit,bound,right_run,
          Order.transitive(position,Runs.range_low(left),Runs.range_low(right),starts,
            {%glue : {Nat.is_le(Runs.range_low(left),_) == True{} : Bool} run_order(left,unit,bound,left_run) : {Nat.is_le(Runs.range_low(left),Runs.range_low(right)) == True{} : Bool}}))}

# Once position passes every slot, every slot holds its piece.
def all_done(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,model: PiecesModel.Model<Element>,+position: Nat,+unit: Nat,+bound: Nat,
  +run: Runs.Run(PiecesModel.layout(~Element,model),unit,bound),+ended: {Nat.is_le(Runs.range_high(PiecesModel.layout(~Element,model)),position) == True{} : Bool},
  +good: Good(~Element,~Logical,~original,logical,position,model)) -> PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),PiecesModel.content(~Element,model)):
  match model:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      {%ended : SlotWhen(~Element,~Logical,~original,_,Nat.is_lt(U32.to_nat(first),position),logical,position,first,rows,Buffer.content(~Element,buffer))
        good : SlotWhen(~Element,~Logical,~original,True{},Nat.is_lt(U32.to_nat(first),position),logical,position,first,rows,Buffer.content(~Element,buffer))}
    case PiecesModel.Fork{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      KernelGemm.Both{+left_good,+right_good} = good
      KernelGemm.Both{
        all_done(~Element,~Logical,~original,logical,left,position,unit,bound,left_run,
          Order.transitive(Runs.range_high(PiecesModel.layout(~Element,left)),Runs.range_high(PiecesModel.layout(~Element,right)),position,
            {%Equal.sym(Nat,Runs.range_high(PiecesModel.layout(~Element,left)),Runs.range_low(PiecesModel.layout(~Element,right)),glue) :
                {Nat.is_le(_,Runs.range_high(PiecesModel.layout(~Element,right))) == True{} : Bool}
              run_order(PiecesModel.layout(~Element,right),unit,bound,right_run) :
              {Nat.is_le(Runs.range_high(PiecesModel.layout(~Element,left)),Runs.range_high(PiecesModel.layout(~Element,right))) == True{} : Bool}},ended),left_good),
        all_done(~Element,~Logical,~original,logical,right,position,unit,bound,right_run,ended,right_good)}

# ---- The placed pieces ----

# Tasks whose pieces in order are parts of the slots of a run, from its first
# cell to its last, each part holding the values of a reader, place pieces laid
# as the run, each holding the values of the reader.
def placed_parts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+layout: Pieces.Ranges,+fill: Element,+results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+unit: Nat,+bound: Nat,+limits: Machine.Limits,+run: Runs.Run(layout,unit,bound),+room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +facts: TasksParts(~Element,~Logical,~original,logical,layout,Runs.range_low(layout),results),+ended: {tasks_end(~Element,results,Runs.range_low(layout)) == Runs.range_high(layout) : Nat}) ->
  Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~Logical,~original,logical),layout,Tensors.placed(~Element,layout,fill,PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),results))>:
  +slots = Placed.empty_slots(~Element,layout,fill)
  +laid = Equal.trans(Pieces.Ranges,PiecesModel.layout(~Element,Placed.placed_tasks(~Element,layout,fill,results,Placed.empty_slots(~Element,layout,fill))),PiecesModel.layout(~Element,slots),layout,
    Placed.placed_tasks_layout(~Element,layout,fill,results,slots),Placed.slots_layout(~Element,layout,fill))
  +good = tasks_good(~Element,~Logical,~original,logical,layout,fill,results,slots,Runs.range_low(layout),unit,bound,limits,run,room,Placed.slots_layout(~Element,layout,fill),facts,
    empty_good(~Element,~Logical,~original,logical,layout,fill,Runs.range_low(layout),unit,bound,run,Order.reflexive(Runs.range_low(layout))))
  Placed.PlacedModel{Placed.placed_tasks(~Element,layout,fill,results,Placed.empty_slots(~Element,layout,fill)),Placed.placed_exact(~Element,layout,fill,results),laid,
    all_done(~Element,~Logical,~original,logical,Placed.placed_tasks(~Element,layout,fill,results,Placed.empty_slots(~Element,layout,fill)),Runs.range_high(layout),unit,bound,
      {%Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,Placed.placed_tasks(~Element,layout,fill,results,Placed.empty_slots(~Element,layout,fill))),layout,laid) : Runs.Run(_,unit,bound) run : Runs.Run(PiecesModel.layout(~Element,Placed.placed_tasks(~Element,layout,fill,results,Placed.empty_slots(~Element,layout,fill))),unit,bound)},
      {%Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,Placed.placed_tasks(~Element,layout,fill,results,Placed.empty_slots(~Element,layout,fill))),layout,laid) : {Nat.is_le(Runs.range_high(_),Runs.range_high(layout)) == True{} : Bool}
        Order.reflexive(Runs.range_high(layout)) : {Nat.is_le(Runs.range_high(PiecesModel.layout(~Element,Placed.placed_tasks(~Element,layout,fill,results,Placed.empty_slots(~Element,layout,fill)))),Runs.range_high(layout)) == True{} : Bool}},
      {%ended : Good(~Element,~Logical,~original,logical,_,Placed.placed_tasks(~Element,layout,fill,results,Placed.empty_slots(~Element,layout,fill)))
        good : Good(~Element,~Logical,~original,logical,Runs.range_high(layout),Placed.placed_tasks(~Element,layout,fill,results,Placed.empty_slots(~Element,layout,fill)))})}

# ---- Pieces laid as a layout ----

# Where the ranges of a layout end when they start at position.
def ranges_end(layout: Pieces.Ranges,+position: Nat) -> Nat:
  match layout:
    case Pieces.Range{first,+count}: Nat.add(position,U32.to_nat(count))
    case Pieces.Split{left,right}: ranges_end(right,ranges_end(left,position))

def ranges_end_after(+layout: Pieces.Ranges,+position: Nat) -> {Nat.is_le(position,ranges_end(layout,position)) == True{} : Bool}:
  match layout:
    case Pieces.Range{+first,+count}: Order.le_add(position,U32.to_nat(count))
    case Pieces.Split{+left,+right}:
      Order.transitive(position,ranges_end(left,position),ranges_end(right,ranges_end(left,position)),ranges_end_after(left,position),
        ranges_end_after(right,ranges_end(left,position)))

# The ranges of a layout in order from position: each starts where the one
# before ends, is not empty and lies in one slot of target.
def LaidParts(+target: Pieces.Ranges,layout: Pieces.Ranges,+position: Nat) -> Data:
  match layout:
    case Pieces.Range{+first,+count}:
      KernelGemm.Both<{U32.to_nat(first) == position : Nat},KernelGemm.Both<{Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},Inside(target,position,U32.to_nat(count))>>
    case Pieces.Split{+left,+right}:
      KernelGemm.Both<LaidParts(target,left,position),LaidParts(target,right,ranges_end(left,position))>

def laid_end(~Element: Data,pieces: PiecesModel.Model<Element>,+position: Nat) -> {pieces_end(~Element,pieces,position) == ranges_end(PiecesModel.layout(~Element,pieces),position) : Nat}:
  match pieces:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      {==}
    case PiecesModel.Fork{+left,+right}:
      {%Equal.sym(Nat,pieces_end(~Element,left,position),ranges_end(PiecesModel.layout(~Element,left),position),laid_end(~Element,left,position)) :
          {pieces_end(~Element,right,_) == ranges_end(PiecesModel.layout(~Element,right),ranges_end(PiecesModel.layout(~Element,left),position)) : Nat}
        laid_end(~Element,right,ranges_end(PiecesModel.layout(~Element,left),position)) :
          {pieces_end(~Element,right,pieces_end(~Element,left,position)) == ranges_end(PiecesModel.layout(~Element,right),ranges_end(PiecesModel.layout(~Element,left),position)) : Nat}}

# Pieces laid as ranges in order that hold their values are parts in order.
def laid_parts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+target: Pieces.Ranges,pieces: PiecesModel.Model<Element>,+position: Nat,
  +laid: LaidParts(target,PiecesModel.layout(~Element,pieces),position),
  +holds: PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),PiecesModel.content(~Element,pieces))) ->
  PartsAt(~Element,~Logical,~original,logical,target,position,pieces):
  match pieces:
    case PiecesModel.Leaf{+buffer,+first,+rows}:
      KernelGemm.Both{+at,+rest} = laid
      KernelGemm.Both{+positive,+inside} = rest
      KernelGemm.Both{at,KernelGemm.Both{positive,KernelGemm.Both{inside,holds}}}
    case PiecesModel.Fork{+left,+right}:
      KernelGemm.Both{+left_laid,+right_laid} = laid
      KernelGemm.Both{+left_holds,+right_holds} = holds
      KernelGemm.Both{laid_parts(~Element,~Logical,~original,logical,target,left,position,left_laid,left_holds),
        laid_parts(~Element,~Logical,~original,logical,target,right,pieces_end(~Element,left,position),
          {%Equal.sym(Nat,pieces_end(~Element,left,position),ranges_end(PiecesModel.layout(~Element,left),position),laid_end(~Element,left,position)) :
              LaidParts(target,PiecesModel.layout(~Element,right),_)
            right_laid : LaidParts(target,PiecesModel.layout(~Element,right),pieces_end(~Element,left,position))},right_holds)}

# ---- The slot of a range ----

# Some span of spans holds the cells [position, position + count).
def Covers(spans: Held.Spans,+position: Nat,+count: Nat) -> Data:
  match spans:
    case Held.Ended{}: Empty
    case Held.Spanned{+first,+span,rest}:
      Placed.Side<KernelGemm.Both<{Nat.is_le(U32.to_nat(first),position) == True{} : Bool},{Nat.is_le(Nat.add(position,count),Nat.add(U32.to_nat(first),U32.to_nat(span))) == True{} : Bool}>,Covers(rest,position,count)>

def inside_low(layout: Pieces.Ranges,+position: Nat,+count: Nat,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound),+inside: Inside(layout,position,count)) ->
  {Nat.is_le(Runs.range_low(layout),position) == True{} : Bool}:
  match layout:
    case Pieces.Range{+first,+span}:
      KernelGemm.Both{+start,+finish} = inside
      start
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+rest} = run
      Held.Both{+right_run,+glue} = rest
      match inside:
        case Placed.OnLeft{+facts}:
          KernelGemm.Both{+below,+left_inside} = facts
          inside_low(left,position,count,unit,bound,left_run,left_inside)
        case Placed.OnRight{+facts}:
          KernelGemm.Both{+above,+right_inside} = facts
          Order.transitive(Runs.range_low(left),Runs.range_low(right),position,
            {%glue : {Nat.is_le(Runs.range_low(left),_) == True{} : Bool}
              run_order(left,unit,bound,left_run) : {Nat.is_le(Runs.range_low(left),Runs.range_low(right)) == True{} : Bool}},above)

def below_end(+position: Nat,+count: Nat,+high: Nat,+positive: {Nat.is_lt(0n,count) == True{} : Bool},+ended: {Nat.is_le(Nat.add(position,count),high) == True{} : Bool}) ->
  {Nat.is_lt(position,high) == True{} : Bool}:
  Order.lt_le(position,Nat.add(position,count),high,
    {%Addition.zero_right(position) : {Nat.is_lt(_,Nat.add(position,count)) == True{} : Bool}
      Order.offset_lt(position,0n,count,positive) : {Nat.is_lt(position,Nat.add(position,count)) == True{} : Bool}},ended)

def right_side(+left: Pieces.Ranges,+right: Pieces.Ranges,+rest: Held.Spans,+position: Nat,+count: Nat,+unit: Nat,+bound: Nat,+right_run: Runs.Run(right,unit,bound),
  +inner: Placed.Side<Inside(right,position,count),Covers(rest,position,count)>) ->
  Placed.Side<Placed.Side<KernelGemm.Both<{Nat.is_lt(position,Runs.range_high(left)) == True{} : Bool},Inside(left,position,count)>,KernelGemm.Both<{Nat.is_le(Runs.range_low(right),position) == True{} : Bool},Inside(right,position,count)>>,Covers(rest,position,count)>:
  match inner:
    case Placed.OnLeft{+right_inside}:
      Placed.OnLeft{Placed.OnRight{KernelGemm.Both{inside_low(right,position,count,unit,bound,right_run,right_inside),right_inside}}}
    case Placed.OnRight{+after}:
      Placed.OnRight{after}

def split_side(+left: Pieces.Ranges,+right: Pieces.Ranges,+rest: Held.Spans,+position: Nat,+count: Nat,+unit: Nat,+bound: Nat,+left_run: Runs.Run(left,unit,bound),
  +right_run: Runs.Run(right,unit,bound),+positive: {Nat.is_lt(0n,count) == True{} : Bool},
  first: Placed.Side<Inside(left,position,count),Covers(Held.layout_spans(right,rest),position,count)>,
  later: Covers(Held.layout_spans(right,rest),position,count) -> Placed.Side<Inside(right,position,count),Covers(rest,position,count)>) ->
  Placed.Side<Inside(Pieces.Split{left,right},position,count),Covers(rest,position,count)>:
  match first:
    case Placed.OnLeft{+left_inside}:
      Placed.OnLeft{Placed.OnLeft{KernelGemm.Both{below_end(position,count,Runs.range_high(left),positive,inside_end(left,position,count,unit,bound,left_run,left_inside)),left_inside}}}
    case Placed.OnRight{+others}:
      right_side(left,right,rest,position,count,unit,bound,right_run,later(others))

# A nonempty range that a span of a run's layout holds is inside that layout,
# else a later span holds it.
def covers_inside(layout: Pieces.Ranges,+rest: Held.Spans,+position: Nat,+count: Nat,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound),
  +positive: {Nat.is_lt(0n,count) == True{} : Bool},+covers: Covers(Held.layout_spans(layout,rest),position,count)) ->
  Placed.Side<Inside(layout,position,count),Covers(rest,position,count)>:
  match layout:
    case Pieces.Range{+first,+span}:
      covers
    case Pieces.Split{+left,+right}:
      Held.Both{+left_run,+others} = run
      Held.Both{+right_run,+glue} = others
      split_side(left,right,rest,position,count,unit,bound,left_run,right_run,positive,
        covers_inside(left,Held.layout_spans(right,rest),position,count,unit,bound,left_run,positive,covers),
        later => covers_inside(right,rest,position,count,unit,bound,right_run,positive,later))

def only_inside(+layout: Pieces.Ranges,+position: Nat,+count: Nat,+side: Placed.Side<Inside(layout,position,count),Covers(Held.Ended{},position,count)>) -> Inside(layout,position,count):
  match side:
    case Placed.OnLeft{+inside}: inside
    case Placed.OnRight{+none}: Empty.absurd(Inside(layout,position,count),none)

# A nonempty range that a span of a run's layout holds is inside the layout.
def spans_inside(+layout: Pieces.Ranges,+position: Nat,+count: Nat,+unit: Nat,+bound: Nat,+run: Runs.Run(layout,unit,bound),
  +positive: {Nat.is_lt(0n,count) == True{} : Bool},+covers: Covers(Held.layout_spans(layout,Held.Ended{}),position,count)) -> Inside(layout,position,count):
  only_inside(layout,position,count,covers_inside(layout,Held.Ended{},position,count,unit,bound,run,positive,covers))

# A tree laid as a layout whose ranges lie in order in their slots up to high,
# and whose pieces hold their values, gives its parts in order up to high.
def laid_task(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+target: Pieces.Ranges,+layout: Pieces.Ranges,+tree: PiecesModel.Model<Element>,
  +position: Nat,+high: Nat,+laid: {PiecesModel.layout(~Element,tree) == layout : Pieces.Ranges},
  +holds: PiecesModel.ContentAll(~Element,~Placed.PieceHolds(~Element,~Logical,~original,logical),PiecesModel.content(~Element,tree)),
  +facts: Held.Both<LaidParts(target,layout,position),{ranges_end(layout,position) == high : Nat}>) ->
  Held.Both<PartsAt(~Element,~Logical,~original,logical,target,position,tree),{pieces_end(~Element,tree,position) == high : Nat}>:
  Held.Both{+parts,+reach} = facts
  +moved = Equal.sym(Pieces.Ranges,PiecesModel.layout(~Element,tree),layout,laid)
  Held.Both{laid_parts(~Element,~Logical,~original,logical,target,tree,position,
      {%moved : LaidParts(target,_,position) parts : LaidParts(target,PiecesModel.layout(~Element,tree),position)},holds),
    Equal.trans(Nat,pieces_end(~Element,tree,position),ranges_end(PiecesModel.layout(~Element,tree),position),high,laid_end(~Element,tree,position),
      {%moved : {ranges_end(_,position) == high : Nat} reach : {ranges_end(PiecesModel.layout(~Element,tree),position) == high : Nat}})}

# The parts of two runs of tasks in order, the second starting where the
# first ends.
def joined_parts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+target: Pieces.Ranges,+left: Pieces.Ranges,+right: Pieces.Ranges,
  +left_results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+right_results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,
  +glue: {Runs.range_high(left) == Runs.range_low(right) : Nat},
  +left_parts: Held.Both<TasksParts(~Element,~Logical,~original,logical,target,Runs.range_low(left),left_results),{tasks_end(~Element,left_results,Runs.range_low(left)) == Runs.range_high(left) : Nat}>,
  +right_parts: Held.Both<TasksParts(~Element,~Logical,~original,logical,target,Runs.range_low(right),right_results),{tasks_end(~Element,right_results,Runs.range_low(right)) == Runs.range_high(right) : Nat}>) ->
  Held.Both<TasksParts(~Element,~Logical,~original,logical,target,Runs.range_low(left),Partition.Fork{left_results,right_results}),
    {tasks_end(~Element,Partition.Fork{left_results,right_results},Runs.range_low(left)) == Runs.range_high(right) : Nat}>:
  Held.Both{+left_laid,+left_end} = left_parts
  Held.Both{+right_laid,+right_end} = right_parts
  +moved = Equal.trans(Nat,Runs.range_low(right),Runs.range_high(left),tasks_end(~Element,left_results,Runs.range_low(left)),
    Equal.sym(Nat,Runs.range_high(left),Runs.range_low(right),glue),Equal.sym(Nat,tasks_end(~Element,left_results,Runs.range_low(left)),Runs.range_high(left),left_end))
  Held.Both{KernelGemm.Both{left_laid,
      {%moved : TasksParts(~Element,~Logical,~original,logical,target,_,right_results)
        right_laid : TasksParts(~Element,~Logical,~original,logical,target,tasks_end(~Element,left_results,Runs.range_low(left)),right_results)}},
    {%moved : {tasks_end(~Element,right_results,_) == Runs.range_high(right) : Nat}
      right_end : {tasks_end(~Element,right_results,tasks_end(~Element,left_results,Runs.range_low(left))) == Runs.range_high(right) : Nat}}}

# Tasks whose parts run in order from the first cell of a run's layout to its
# end place every piece of the layout.
def placed_run(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+layout: Pieces.Ranges,+fill: Element,+tasks: Pieces.Ranges,
  +results: Partition.Tree<&2,PiecesModel.Model<Element>,Unit>,+unit: Nat,+bound: Nat,+limits: Machine.Limits,+run: Runs.Run(layout,unit,bound),
  +room: {Nat.is_le(bound,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +low_same: {Runs.range_low(tasks) == Runs.range_low(layout) : Nat},+high_same: {Runs.range_high(tasks) == Runs.range_high(layout) : Nat},
  +parts: Held.Both<TasksParts(~Element,~Logical,~original,logical,layout,Runs.range_low(tasks),results),{tasks_end(~Element,results,Runs.range_low(tasks)) == Runs.range_high(tasks) : Nat}>) ->
  Placed.PlacedModel<Element,Placed.PieceHolds(~Element,~Logical,~original,logical),layout,Tensors.placed(~Element,layout,fill,PartitionLaws.carry(~&2,~&1,~PiecesModel.Model<Element>,~Pieces.Pieces(~Element),~Unit,~PiecesModel.pack(~Element),results))>:
  Held.Both{+laid,+reach} = parts
  placed_parts(~Element,~Logical,~original,logical,layout,fill,results,unit,bound,limits,run,room,
    {%low_same : TasksParts(~Element,~Logical,~original,logical,layout,_,results) laid : TasksParts(~Element,~Logical,~original,logical,layout,Runs.range_low(layout),results)},
    {%low_same : {tasks_end(~Element,results,_) == Runs.range_high(layout) : Nat}
      {%high_same : {tasks_end(~Element,results,Runs.range_low(tasks)) == _ : Nat} reach : {tasks_end(~Element,results,Runs.range_low(tasks)) == Runs.range_high(layout) : Nat}} :
      {tasks_end(~Element,results,Runs.range_low(layout)) == Runs.range_high(layout) : Nat}})
