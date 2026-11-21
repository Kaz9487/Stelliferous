# The weight kernel over the band buffers of an image: the cell sums, the
# group sums as adjacent trees of cells and their numeric values, and the cells
# of the result in output block and tap order.
import Base
import ../kernel_weight_gradient.bend as Kernel
import ../kernel_parameter_gradient.bend as Parameter
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_array.bend as Traversal
import ../traversal_grouped.bend as Grouped
import ./kernel_weight_gradient_group.bend as Groups
import ./kernel_tile_refinement.bend as TileLaws
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as Traversals
import ./grouped_reduction_refinement.bend as GroupLaws
import ./traversal_batch_refinement.bend as Batch
import ./traversal_map_readback.bend as MapReadback
import ./storage_shape.bend as Shape
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./storage_readback.bend as StorageReadback
import ./sequence_membership.bend as Members
import ./kernel_weight_gradient_packing.bend as Packing
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./ordered_ranges.bend as Ranges
import ./list_storage_readback.bend as ListRead
import ./tensor_allocation_depth.bend as AllocationDepth
import ../traversal_loop.bend as Loop
import ../tensor.bend as Tensors
import ./loop_projection.bend as Projection
import ./tensor_list_values.bend as Listed
import ../kernel_gemm.bend as KernelGemm

type SumOwners is Data:
  SumOwners{left: Tree.Tree<Tile.Tile8>,right: Tree.Tree<Tile.Tile8>}

def represent_sum(state: SumOwners) -> Array<Tile.Tile8> & Array<Tile.Tile8>:
  SumOwners{left,right} = state
  (Tree.pack(Tile.Tile8,left),Tree.pack(Tile.Tile8,right))

# The cell of a sum at a position: the left cell plus the right one.
def sum_cell(right: Tree.Tree<Tile.Tile8>,position: U32,left: Tile.Tile8) -> Tile.Tile8:
  Tile.plus(left,Tree.read(Tile.Tile8,right,position))

def sum_step(position: U32,context: Unit,state: SumOwners) -> SumOwners:
  SumOwners{+left,+right} = state
  +position = position
  SumOwners{Tree.write(Tile.Tile8,left,position,sum_cell(right,position,Tree.read(Tile.Tile8,left,position))),right}

def sum_step_matches(+index: U32,+context: Unit,+state: SumOwners) ->
  {Kernel.sum_step(index,context,represent_sum(state)) == represent_sum(sum_step(index,context,state)) : Array<Tile.Tile8> & Array<Tile.Tile8>}:
  SumOwners{+left,+right} = state
  +a = Tree.read(Tile.Tile8,left,index)
  +b = Tree.read(Tile.Tile8,right,index)
  +tile = Tile.plus(a,b)
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,left),index),(Tree.pack(Tile.Tile8,left),a),StorageLaws.read(Tile.Tile8,left,index)) :
    {Kernel.sum_result(index,_,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,right),index)) == represent_sum(sum_step(index,context,state)) : Array<Tile.Tile8> & Array<Tile.Tile8>}
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,right),index),(Tree.pack(Tile.Tile8,right),b),StorageLaws.read(Tile.Tile8,right,index)) :
    {Kernel.sum_result(index,(Tree.pack(Tile.Tile8,left),a),_) == (Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,left,index,tile)),Tree.pack(Tile.Tile8,right)) : Array<Tile.Tile8> & Array<Tile.Tile8>}
  Equal.cong(Array<Tile.Tile8>,Array<Tile.Tile8> & Array<Tile.Tile8>,array => (array,Tree.pack(Tile.Tile8,right)),Array.set(Tile.Tile8,Tree.pack(Tile.Tile8,left),index,tile),
    Tree.pack(Tile.Tile8,Tree.write(Tile.Tile8,left,index,tile)),StorageLaws.write(Tile.Tile8,left,index,tile))

def sum_output(state: SumOwners) -> Tree.Tree<Tile.Tile8>:
  SumOwners{left,right} = state
  left

def sum_output_matches(+state: SumOwners) ->
  {Pair.fst(Array<Tile.Tile8>,Array<Tile.Tile8>,represent_sum(state)) == Tree.pack(Tile.Tile8,sum_output(state)) : Array<Tile.Tile8>}:
  SumOwners{left,right} = state
  {==}

def sum(+count: U32,left: Tree.Tree<Tile.Tile8>,right: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  sum_output(Traversal.indexed_fold(~SumOwners,~Unit,~sum_step,U32.to_nat(count),0,Unit{},SumOwners{left,right}))

def sum_matches(+count: U32,+left: Tree.Tree<Tile.Tile8>,+right: Tree.Tree<Tile.Tile8>) ->
  {Kernel.sum(count,Tree.pack(Tile.Tile8,left),Tree.pack(Tile.Tile8,right)) == Tree.pack(Tile.Tile8,sum(count,left,right)) : Array<Tile.Tile8>}:
  +initial = {SumOwners{left,right} : SumOwners}
  +rounds = U32.to_nat(count)
  +state = Traversal.indexed_fold(~SumOwners,~Unit,~sum_step,rounds,0,Unit{},initial)
  %Equal.sym(Array<Tile.Tile8> & Array<Tile.Tile8>,Traversal.indexed_fold(~(Array<Tile.Tile8> & Array<Tile.Tile8>),~Unit,~Kernel.sum_step,rounds,0,Unit{},represent_sum(initial)),represent_sum(state),
    Traversals.indexed_fold_refines_model(~(Array<Tile.Tile8> & Array<Tile.Tile8>),~SumOwners,~Unit,~Kernel.sum_step,~sum_step,~represent_sum,~sum_step_matches,rounds,0,Unit{},initial)) :
    {Pair.fst(Array<Tile.Tile8>,Array<Tile.Tile8>,_) == Tree.pack(Tile.Tile8,sum(count,left,right)) : Array<Tile.Tile8>}
  sum_output_matches(state)

def sum_right(state: SumOwners) -> Tree.Tree<Tile.Tile8>:
  SumOwners{left,right} = state
  right

def sum_frame(right: Tree.Tree<Tile.Tile8>,state: SumOwners) -> Data:
  {sum_right(state) == right : Tree.Tree<Tile.Tile8>}

def sum_fold_step(index: U32,context: Tree.Tree<Tile.Tile8>,state: SumOwners) -> SumOwners:
  sum_step(index,Unit{},state)

def sum_frame_step(+context: Tree.Tree<Tile.Tile8>,+index: U32,+state: SumOwners,frame: sum_frame(context,state)) ->
  sum_frame(context,sum_fold_step(index,context,state)):
  SumOwners{left,right} = state
  frame

# One round writes the cell at its index.
def sum_batch_matches(+context: Tree.Tree<Tile.Tile8>,+index: Nat,+state: SumOwners,+depth: Nat,+capacity: U32,
  frame: sum_frame(context,state),capacity_equal: {Tree.capacity(Tile.Tile8,sum_output(state)) == capacity : U32},
  balanced: Shape.Balanced(Tile.Tile8,depth,sum_output(state)),limit: {Nat.is_le(depth,31n) == True{} : Bool},
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(Nat.add(Nat.mul(index,1n),1n),U32.to_nat(capacity)) == True{} : Bool}) ->
  {sum_output(sum_fold_step(U32.from_nat(index),context,state)) ==
    MapReadback.run_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,1n,U32.from_nat(Nat.mul(index,1n)),context,sum_output(state)) : Tree.Tree<Tile.Tile8>}:
  SumOwners{+left,+right} = state
  +position = U32.from_nat(index)
  %Equal.sym(Nat,Nat.mul(index,1n),index,Order.times_one(index)) :
    {Tree.write(Tile.Tile8,left,position,sum_cell(right,position,Tree.read(Tile.Tile8,left,position))) ==
      MapReadback.run_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,1n,U32.from_nat(_),context,left) : Tree.Tree<Tile.Tile8>}
  %frame :
    {Tree.write(Tile.Tile8,left,position,sum_cell(right,position,Tree.read(Tile.Tile8,left,position))) ==
      MapReadback.run_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,1n,position,_,left) : Tree.Tree<Tile.Tile8>}
  {==}

def sum_fold_inner(+index: U32,+context: Tree.Tree<Tile.Tile8>,+state: SumOwners) ->
  {sum_fold_step(index,context,state) == sum_step(index,Unit{},state) : SumOwners}:
  {==}

def sum_mapped(+count: U32,+left: Tree.Tree<Tile.Tile8>,+right: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Tile.Tile8,left) == capacity : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,left),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool}) ->
  {sum(count,left,right) == MapReadback.run_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,U32.to_nat(count),0,right,left) : Tree.Tree<Tile.Tile8>}:
  +rounds = U32.to_nat(count)
  +initial = {SumOwners{left,right} : SumOwners}
  %Order.times_one(rounds) :
    {sum(count,left,right) == MapReadback.run_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,_,0,right,left) : Tree.Tree<Tile.Tile8>}
  %Traversals.indexed_fold_change_context(~SumOwners,~Tree.Tree<Tile.Tile8>,~Unit,~(right => Unit{}),~sum_fold_step,~sum_step,~sum_fold_inner,rounds,0,right,initial) :
    {sum_output(_) == MapReadback.run_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,Nat.mul(rounds,1n),0,right,left) : Tree.Tree<Tile.Tile8>}
  Shape.second(sum_frame(right,Traversal.indexed_fold(~SumOwners,~Tree.Tree<Tile.Tile8>,~sum_fold_step,rounds,0,right,initial)),
    {sum_output(Traversal.indexed_fold(~SumOwners,~Tree.Tree<Tile.Tile8>,~sum_fold_step,rounds,0,right,initial)) == MapReadback.run_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,Nat.mul(rounds,1n),0,right,left) : Tree.Tree<Tile.Tile8>},
    Batch.fold_mapped(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~SumOwners,~Tree.Tree<Tile.Tile8>,~1n,~sum_output,~{right => right : Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8>},~sum_cell,~sum_frame,~sum_fold_step,~sum_frame_step,~sum_batch_matches,
      rounds,right,initial,depth,capacity,{==},capacity_equal,balanced,limit,capacity_ok,
      %Equal.sym(Nat,Nat.mul(rounds,1n),rounds,Order.times_one(rounds)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
      room))

def sum_numeric(+count: U32,+left: Tree.Tree<Tile.Tile8>,+right: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,+index: Nat,
  +capacity_equal: {Tree.capacity(Tile.Tile8,left) == capacity : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,left),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool},
  selected: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,sum(count,left,right),U32.from_nat(index)) == Tile.plus(Tree.read(Tile.Tile8,left,U32.from_nat(index)),Tree.read(Tile.Tile8,right,U32.from_nat(index))) : Tile.Tile8}:
  +extent = U32.to_nat(count)
  %Equal.sym(Tree.Tree<Tile.Tile8>,sum(count,left,right),MapReadback.run_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,extent,0,right,left),
    sum_mapped(count,left,right,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room)) :
    {Tree.read(Tile.Tile8,_,U32.from_nat(index)) == Tile.plus(Tree.read(Tile.Tile8,left,U32.from_nat(index)),Tree.read(Tile.Tile8,right,U32.from_nat(index))) : Tile.Tile8}
  MapReadback.inside_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,extent,0n,index,right,left,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room,selected)

def sum_storage(+count: U32,+left: Tree.Tree<Tile.Tile8>,+right: Tree.Tree<Tile.Tile8>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(Tile.Tile8,left) == capacity : U32},+balanced: Shape.Balanced(Tile.Tile8,depth,left),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool}) ->
  Groups.StorageFrame(depth,capacity,sum(count,left,right)):
  +extent = U32.to_nat(count)
  %Equal.sym(Tree.Tree<Tile.Tile8>,sum(count,left,right),MapReadback.run_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,extent,0,right,left),
    sum_mapped(count,left,right,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room)) : Groups.StorageFrame(depth,capacity,_)
  KernelGemm.Both{MapReadback.balanced_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,extent,0,right,left,depth,balanced),
    Equal.trans(U32,Tree.capacity(Tile.Tile8,MapReadback.run_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,extent,0,right,left)),Tree.capacity(Tile.Tile8,left),capacity,
      MapReadback.capacity_indexed(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~sum_cell,extent,0,right,left),capacity_equal)}

def zero(count: U32) -> Tree.Tree<Tile.Tile8>:
  Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0))

def zero_matches(+count: U32) ->
  {Kernel.array(~Tile.Tile8,count,Tile.filled(0.0)) == Tree.pack(Tile.Tile8,zero(count)) : Array<Tile.Tile8>}:
  StorageLaws.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0))

# One lane of one cell of the sums.
type SumQuery is Data:
  SumQuery{count: U32,index: Nat,lane: Nat,depth: Nat,capacity: U32,
    zero_depth: {Storage.depth_for(count) == depth : Nat},
    zero_capacity: {U32.shln(1,depth) == capacity : U32},
    depth_ok: {Nat.is_le(depth,31n) == True{} : Bool},capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    room: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool},
    selected: {Nat.is_lt(index,U32.to_nat(count)) == True{} : Bool}}

def plan_sum_query(+plan: Groups.GroupPlan,+index: Nat,+lane: Nat,selected: {Nat.is_lt(index,U32.to_nat(Groups.plan_count(plan))) == True{} : Bool}) -> SumQuery:
  SumQuery{Groups.plan_count(plan),index,lane,Groups.plan_depth(plan),Groups.plan_capacity(plan),{==},{==},
    Order.successor(Groups.plan_depth(plan),30n,Groups.plan_depth_limit(plan)),Groups.plan_capacity_bound(plan),
    AllocationDepth.allocation_room(Groups.plan_count(plan),Groups.plan_allocation(plan)),selected}

def sum_query_count(query: SumQuery) -> U32:
  SumQuery{count,index,lane,depth,capacity,zero_depth,zero_capacity,depth_ok,capacity_ok,room,selected} = query
  count

def sum_query_read(query: SumQuery,tree: Tree.Tree<Tile.Tile8>) -> F32:
  SumQuery{count,index,lane,depth,capacity,zero_depth,zero_capacity,depth_ok,capacity_ok,room,selected} = query
  TileLaws.select_lane(lane,Tree.read(Tile.Tile8,tree,U32.from_nat(index)))

def sum_query_frame(query: SumQuery,tree: Tree.Tree<Tile.Tile8>) -> Data:
  SumQuery{count,index,lane,depth,capacity,zero_depth,zero_capacity,depth_ok,capacity_ok,room,selected} = query
  Groups.StorageFrame(depth,capacity,tree)

def sum_zero_frame(+query: SumQuery) -> sum_query_frame(query,zero(sum_query_count(query))):
  SumQuery{+count,index,lane,+depth,+capacity,+zero_depth,+zero_capacity,depth_ok,capacity_ok,room,selected} = query
  %Equal.sym(Nat,Storage.depth_for(count),depth,zero_depth) : Groups.StorageFrame(depth,capacity,Tree.fresh(Tile.Tile8,_,Tile.filled(0.0)))
  KernelGemm.Both{Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)),Equal.trans(U32,Tree.capacity(Tile.Tile8,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0))),U32.shln(1,depth),capacity,
    Shape.capacity(Tile.Tile8,depth,Tree.fresh(Tile.Tile8,depth,Tile.filled(0.0)),Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0))),zero_capacity)}

def sum_zero_numeric(+query: SumQuery) -> {sum_query_read(query,zero(sum_query_count(query))) == 0.0 : F32}:
  SumQuery{+count,+index,+lane,depth,capacity,zero_depth,zero_capacity,depth_ok,capacity_ok,room,selected} = query
  +fresh = Tree.fresh(Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0))
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,fresh,U32.from_nat(index)),Tile.filled(0.0),StorageReadback.read_fresh(~Tile.Tile8,Storage.depth_for(count),Tile.filled(0.0),U32.from_nat(index))) :
    {TileLaws.select_lane(lane,_) == 0.0 : F32}
  TileLaws.filled_lane(lane,0.0)

def sum_combine_frame(+query: SumQuery,+left: Tree.Tree<Tile.Tile8>,+right: Tree.Tree<Tile.Tile8>,left_frame: sum_query_frame(query,left),right_frame: sum_query_frame(query,right)) ->
  sum_query_frame(query,sum(sum_query_count(query),left,right)):
  SumQuery{+count,index,lane,+depth,+capacity,zero_depth,zero_capacity,+depth_ok,+capacity_ok,+room,selected} = query
  KernelGemm.Both{balanced,capacity_equal} = left_frame
  sum_storage(count,left,right,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room)

def sum_combine_numeric(+query: SumQuery,+left: Tree.Tree<Tile.Tile8>,+right: Tree.Tree<Tile.Tile8>,left_frame: sum_query_frame(query,left),right_frame: sum_query_frame(query,right)) ->
  {sum_query_read(query,sum(sum_query_count(query),left,right)) == (sum_query_read(query,left) + sum_query_read(query,right) : F32) : F32}:
  SumQuery{+count,+index,+lane,+depth,+capacity,zero_depth,zero_capacity,+depth_ok,+capacity_ok,+room,+selected} = query
  KernelGemm.Both{balanced,capacity_equal} = left_frame
  +position = U32.from_nat(index)
  +a = Tree.read(Tile.Tile8,left,position)
  +b = Tree.read(Tile.Tile8,right,position)
  %Equal.sym(Tile.Tile8,Tree.read(Tile.Tile8,sum(count,left,right),position),Tile.plus(a,b),
      sum_numeric(count,left,right,depth,capacity,index,capacity_equal,balanced,depth_ok,capacity_ok,room,selected)) :
    {TileLaws.select_lane(lane,_) == (TileLaws.select_lane(lane,a) + TileLaws.select_lane(lane,b) : F32) : F32}
  TileLaws.added_lane(a,b,lane)

type GroupList is Data:
  GroupList{input: Buffer.Model<Tile.Tile8>,gradient: Buffer.Model<Tile.Tile8>,values: List<&2,Tree.Tree<Tile.Tile8>>}

def represent_list(result: GroupList) -> (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>:
  GroupList{input,gradient,values} = result
  ((Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,gradient)),GroupLaws.pack_values(~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~(tree => Tree.pack(Tile.Tile8,tree)),values))

def source_reader(source: Groups.Sources) -> Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>:
  Groups.Sources{input,gradient} = source
  (Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,gradient))

def group_split(result: Groups.Group) -> Groups.Sources & Tree.Tree<Tile.Tile8>:
  Groups.Group{input,gradient,values} = result
  (Groups.Sources{input,gradient},values)

def group_split_matches(+result: Groups.Group) ->
  {Groups.represent(result) == Traversals.collect_list_read(~&1,~(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>),~Groups.Sources,~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~source_reader,~(tree => Tree.pack(Tile.Tile8,tree)),group_split(result)) : (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>}:
  Groups.Group{input,gradient,values} = result
  {==}

def collected_groups(result: Groups.Sources & List<&2,Tree.Tree<Tile.Tile8>>) -> GroupList:
  (Groups.Sources{input,gradient},values) = result
  GroupList{input,gradient,values}

def collected_groups_matches(result: Groups.Sources & List<&2,Tree.Tree<Tile.Tile8>>) ->
  {Traversals.collect_result(~&1,~(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>),~Groups.Sources,~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~source_reader,~(tree => Tree.pack(Tile.Tile8,tree)),result) == represent_list(collected_groups(result)) : (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>}:
  (Groups.Sources{input,gradient},values) = result
  {==}

def group_read(source: Groups.Sources,context: Kernel.GroupContext,range: Grouped.Range) -> Groups.Sources & Tree.Tree<Tile.Tile8>:
  Groups.Sources{input,gradient} = source
  Kernel.GroupContext{geometry,x_base,d_base} = context
  group_split(Groups.group(geometry,x_base,d_base,range,input,gradient))

def group_read_matches(+source: Groups.Sources,+context: Kernel.GroupContext,+range: Grouped.Range) ->
  {Kernel.group_read(source_reader(source),context,range) == Traversals.collect_list_read(~&1,~(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>),~Groups.Sources,~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~source_reader,~(tree => Tree.pack(Tile.Tile8,tree)),group_read(source,context,range)) : (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>}:
  Groups.Sources{+input,+gradient} = source
  Kernel.GroupContext{+geometry,+x_base,+d_base} = context
  Equal.trans((Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & Array<Tile.Tile8>,Kernel.group(geometry,x_base,d_base,range,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,gradient)),Groups.represent(Groups.group(geometry,x_base,d_base,range,input,gradient)),
    Traversals.collect_list_read(~&1,~(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>),~Groups.Sources,~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~source_reader,~(tree => Tree.pack(Tile.Tile8,tree)),group_read(source,context,range)),Groups.group_matches(geometry,x_base,d_base,range,input,gradient),group_split_matches(Groups.group(geometry,x_base,d_base,range,input,gradient)))

def groups(ranges: List<&2,Grouped.Range>,geometry: Kernel.Geometry,x_base: U32,d_base: U32,input: Buffer.Model<Tile.Tile8>,gradient: Buffer.Model<Tile.Tile8>) -> GroupList:
  collected_groups(Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,x_base,d_base}))

def groups_matches(+ranges: List<&2,Grouped.Range>,+geometry: Kernel.Geometry,+x_base: U32,+d_base: U32,+input: Buffer.Model<Tile.Tile8>,+gradient: Buffer.Model<Tile.Tile8>) ->
  {Kernel.groups(ranges,geometry,x_base,d_base,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,gradient)) == represent_list(groups(ranges,geometry,x_base,d_base,input,gradient)) :
    (Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>}:
  Equal.trans((Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) & List<&1,Array<Tile.Tile8>>,Kernel.groups(ranges,geometry,x_base,d_base,Buffer.pack(~Tile.Tile8,input),Buffer.pack(~Tile.Tile8,gradient)),
    Traversals.collect_result(~&1,~(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>),~Groups.Sources,~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~source_reader,~(tree => Tree.pack(Tile.Tile8,tree)),Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,x_base,d_base})),represent_list(groups(ranges,geometry,x_base,d_base,input,gradient)),
    Traversals.collect_list_loop_refines_model(~&1,~(Storage.Buffer<Tile.Tile8> & Storage.Buffer<Tile.Tile8>),~Groups.Sources,~Array<Tile.Tile8>,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~Kernel.GroupContext,~source_reader,~(tree => Tree.pack(Tile.Tile8,tree)),~Kernel.group_read,~group_read,~group_read_matches,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,x_base,d_base}),
    collected_groups_matches(Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,x_base,d_base})))

# The cells of a tree from a first one on, in order.
def tile_at(values: Tree.Tree<Tile.Tile8>,index: Nat) -> Tile.Tile8:
  Tree.read(Tile.Tile8,values,U32.from_nat(index))

def tiles(count: Nat,+first: Nat,+values: Tree.Tree<Tile.Tile8>) -> List<&2,Tile.Tile8>:
  ListRead.enumerate(~Tile.Tile8,~Tree.Tree<Tile.Tile8>,~tile_at,count,first,values)

# ---- The kernel gives back the input and the gradient it read ----

def group_split_sources(+result: Groups.Group) -> {Pair.fst(Groups.Sources,Tree.Tree<Tile.Tile8>,group_split(result)) == Groups.group_sources(result) : Groups.Sources}:
  Groups.Group{input,gradient,values} = result
  {==}

def group_read_kept(+source: Groups.Sources,+context: Kernel.GroupContext,+range: Grouped.Range) ->
  {Pair.fst(Groups.Sources,Tree.Tree<Tile.Tile8>,group_read(source,context,range)) == source : Groups.Sources}:
  Groups.Sources{+input,+gradient} = source
  Kernel.GroupContext{+geometry,+x_base,+d_base} = context
  Equal.trans(Groups.Sources,Pair.fst(Groups.Sources,Tree.Tree<Tile.Tile8>,group_split(Groups.group(geometry,x_base,d_base,range,input,gradient))),
    Groups.group_sources(Groups.group(geometry,x_base,d_base,range,input,gradient)),Groups.Sources{input,gradient},
    group_split_sources(Groups.group(geometry,x_base,d_base,range,input,gradient)),Groups.group_sources_kept(geometry,x_base,d_base,range,input,gradient))

# A collection that kept its sources holds them.
def collected_kept(+sources: Groups.Sources,+values: List<&2,Tree.Tree<Tile.Tile8>>,+input: Buffer.Model<Tile.Tile8>,+gradient: Buffer.Model<Tile.Tile8>,
  same: {sources == Groups.Sources{input,gradient} : Groups.Sources}) -> {collected_groups((sources,values)) == GroupList{input,gradient,values} : GroupList}:
  %Equal.sym(Groups.Sources,sources,Groups.Sources{input,gradient},same) : {collected_groups((_,values)) == GroupList{input,gradient,values} : GroupList}
  {==}

def collected_result(result: Groups.Sources & List<&2,Tree.Tree<Tile.Tile8>>,+input: Buffer.Model<Tile.Tile8>,+gradient: Buffer.Model<Tile.Tile8>,
  same: {Pair.fst(Groups.Sources,List<&2,Tree.Tree<Tile.Tile8>>,result) == Groups.Sources{input,gradient} : Groups.Sources}) ->
  {collected_groups(result) == GroupList{input,gradient,Pair.snd(Groups.Sources,List<&2,Tree.Tree<Tile.Tile8>>,result)} : GroupList}:
  (+sources,+values) = result
  collected_kept(sources,values,input,gradient,same)

def groups_kept(+ranges: List<&2,Grouped.Range>,+geometry: Kernel.Geometry,+x_base: U32,+d_base: U32,+input: Buffer.Model<Tile.Tile8>,+gradient: Buffer.Model<Tile.Tile8>) ->
  {groups(ranges,geometry,x_base,d_base,input,gradient) ==
    GroupList{input,gradient,Pair.snd(Groups.Sources,List<&2,Tree.Tree<Tile.Tile8>>,
      Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,x_base,d_base}))} : GroupList}:
  collected_result(Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,x_base,d_base}),
    input,gradient,
    Traversals.collect_list_preserves_source(~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~Kernel.GroupContext,~group_read,~group_read_kept,ranges,Groups.Sources{input,gradient},
      Kernel.GroupContext{geometry,x_base,d_base}))

# The collection only calls the already certified group leaf. Numeric
# observations and storage/source frames use the common collection theorem.
type GroupQuery is Data:
  GroupQuery{geometry: Kernel.Geometry,domain: Groups.GeometryReady<geometry>,size: U32,x_base: U32,d_base: U32,
    input: Buffer.Model<Tile.Tile8>,gradient: Buffer.Model<Tile.Tile8>,tap: U32,output: U32,
    tap_inside: {U32.is_lt(tap,Packing.geometry_taps(geometry)) == True{} : Bool},
    output_inside: {U32.is_lt(output,Groups.geometry_outputs(geometry)) == True{} : Bool}}

def group_query_context(query: GroupQuery) -> Kernel.GroupContext:
  GroupQuery{geometry,domain,size,x_base,d_base,input,gradient,tap,output,tap_inside,output_inside} = query
  Kernel.GroupContext{geometry,x_base,d_base}

def group_query_source(query: GroupQuery) -> Groups.Sources:
  GroupQuery{geometry,domain,size,x_base,d_base,input,gradient,tap,output,tap_inside,output_inside} = query
  Groups.Sources{input,gradient}

def group_query_sum(+query: GroupQuery) -> SumQuery:
  GroupQuery{+geometry,+domain,size,x_base,d_base,input,gradient,+tap,+output,+tap_inside,+output_inside} = query
  Groups.GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  +plan = Groups.geometry_plan(geometry,domain)
  plan_sum_query(plan,Nat.add(Nat.mul(Nat.div(U32.to_nat(output),8n),U32.to_nat(Groups.plan_taps(plan))),U32.to_nat(tap)),Nat.mod(U32.to_nat(output),8n),
    Groups.plan_query_inside(plan,tap,output,tap_inside,output_inside))

def group_query_frame(query: GroupQuery,source: Groups.Sources) -> Data:
  {source == group_query_source(query) : Groups.Sources}

def group_query_value(query: GroupQuery,tree: Tree.Tree<Tile.Tile8>) -> Data:
  sum_query_frame(group_query_sum(query),tree)

def group_query_observe(query: GroupQuery,tree: Tree.Tree<Tile.Tile8>) -> F32:
  sum_query_read(group_query_sum(query),tree)

def group_query_key(query: GroupQuery,range: Grouped.Range) -> Data:
  GroupQuery{+geometry,domain,size,x_base,d_base,input,gradient,tap,output,tap_inside,output_inside} = query
  GroupLaws.RangeBounded(U32.to_nat(Groups.geometry_rows(geometry)),U32.to_nat(Groups.geometry_columns(geometry)),U32.to_nat(size),range)

def group_query_sample(query: GroupQuery,range: Grouped.Range) -> F32:
  GroupQuery{geometry,domain,size,x_base,d_base,input,gradient,tap,output,tap_inside,output_inside} = query
  Grouped.Range{first,count} = range
  Groups.numeric_dot(Groups.NumericSource{input,gradient,geometry,x_base,d_base},tap,output,count,first,0.0)

def group_query_group(query: GroupQuery,range: Grouped.Range) -> Groups.Group:
  GroupQuery{geometry,domain,size,x_base,d_base,input,gradient,tap,output,tap_inside,output_inside} = query
  Groups.group(geometry,x_base,d_base,range,input,gradient)

def group_query_numeric(+query: GroupQuery,+range: Grouped.Range,+evidence: group_query_key(query,range)) ->
  {group_query_observe(query,Groups.group_output(group_query_group(query,range))) == group_query_sample(query,range) : F32}:
  GroupQuery{+geometry,+domain,+size,+x_base,+d_base,+input,+gradient,+tap,+output,+tap_inside,+output_inside} = query
  Groups.GeometryReady{+limits,+columns_positive,+kernel_positive,+stride_positive,+square,+taps_positive,+outputs_positive,+count,+aligned,+taps_exact,+pixels,+gradients} = domain
  Grouped.Range{+first,+remaining} = range
  KernelGemm.Both{+nonempty,KernelGemm.Both{+bounded,+room}} = evidence
  Groups.range_group_numeric(geometry,domain,x_base,d_base,first,remaining,input,gradient,nonempty,room,tap,output,tap_inside,output_inside)

def group_query_storage(+query: GroupQuery,+range: Grouped.Range) ->
  KernelGemm.Both<group_query_value(query,Groups.group_output(group_query_group(query,range))),
    {Groups.group_sources(group_query_group(query,range)) == group_query_source(query) : Groups.Sources}>:
  GroupQuery{+geometry,+domain,size,+x_base,+d_base,+input,+gradient,tap,output,tap_inside,output_inside} = query
  Groups.GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned,taps_exact,pixels,gradients} = domain
  Groups.group_frame(geometry,domain,x_base,d_base,range,input,gradient)

def group_query_split_frame(+query: GroupQuery,+range: Grouped.Range,result: Groups.Group,
  frame: KernelGemm.Both<group_query_value(query,Groups.group_output(result)),{Groups.group_sources(result) == group_query_source(query) : Groups.Sources}>,
  numeric: {group_query_observe(query,Groups.group_output(result)) == group_query_sample(query,range) : F32}) ->
  Traversals.collected_leaf_frame(~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~GroupQuery,~F32,
    ~group_query_frame,~group_query_value,~group_query_observe,~group_query_sample,query,range,group_split(result)):
  Groups.Group{input,gradient,values} = result
  KernelGemm.Both{values_frame,sources_equal} = frame
  KernelGemm.Both{sources_equal,KernelGemm.Both{values_frame,numeric}}

def group_query_leaf(+source: Groups.Sources,+query: GroupQuery,+range: Grouped.Range,
  source_frame: group_query_frame(query,source),key_frame: group_query_key(query,range)) ->
  Traversals.collected_leaf_frame(~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~GroupQuery,~F32,
    ~group_query_frame,~group_query_value,~group_query_observe,~group_query_sample,query,range,group_read(source,group_query_context(query),range)):
  GroupQuery{+geometry,+domain,+size,+x_base,+d_base,+input,+gradient,+tap,+output,+tap_inside,+output_inside} = query
  %Equal.sym(Groups.Sources,source,group_query_source(query),source_frame) :
    Traversals.collected_leaf_frame(~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~GroupQuery,~F32,
      ~group_query_frame,~group_query_value,~group_query_observe,~group_query_sample,query,range,group_read(_,group_query_context(query),range))
  group_query_split_frame(query,range,group_query_group(query,range),group_query_storage(query,range),group_query_numeric(query,range,key_frame))

def group_query_collect(+query: GroupQuery,+ranges: List<&2,Grouped.Range>,
  evidence: Members.All(Grouped.Range,{range => group_query_key(query,range) : Grouped.Range -> Data},ranges)) ->
  Traversals.collected_list_frame(~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~GroupQuery,~F32,
    ~group_query_frame,~group_query_value,~group_query_observe,~group_query_sample,query,ranges,
    Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,group_query_source(query),group_query_context(query))):
  Traversals.collect_list_certified(~Groups.Sources,~Tree.Tree<Tile.Tile8>,~Grouped.Range,~GroupQuery,~F32,~Kernel.GroupContext,~group_query_context,
    ~group_query_frame,~group_query_value,~group_query_key,~group_query_observe,~group_query_sample,~group_read,~group_query_leaf,ranges,group_query_source(query),query,{==},evidence)

# The adjacent tree of cell trees of the sums keeps their frames, and its
# lane is the adjacent tree of their lanes.
def sum_tree(+fuel: Nat,+query: SumQuery,+values: List<&2,Tree.Tree<Tile.Tile8>>,
  evidence: Members.All(Tree.Tree<Tile.Tile8>,{tree => sum_query_frame(query,tree) : Tree.Tree<Tile.Tile8> -> Data},values)) ->
  KernelGemm.Both<sum_query_frame(query,Grouped.levels(~&2,~Tree.Tree<Tile.Tile8>,~U32,~zero,~{count => left => right => sum(count,left,right) : U32 -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8>},fuel,sum_query_count(query),values)),
    {sum_query_read(query,Grouped.levels(~&2,~Tree.Tree<Tile.Tile8>,~U32,~zero,~{count => left => right => sum(count,left,right) : U32 -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8>},fuel,sum_query_count(query),values)) ==
      Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~SumQuery,~sum_query_read,values,query)) : F32}>:
  GroupLaws.levels_projected(~Tree.Tree<Tile.Tile8>,~F32,~SumQuery,~U32,~{query => sum_query_count(query) : SumQuery -> U32},~Unit,~(query => Unit{}),
    ~zero,~{context => 0.0 : Unit -> F32},~{count => left => right => sum(count,left,right) : U32 -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8> -> Tree.Tree<Tile.Tile8>},~{context => F32.add : Unit -> F32 -> F32 -> F32},
    ~sum_query_read,~sum_query_frame,
    ~{query => sum_zero_frame(query) : @+query: SumQuery -> sum_query_frame(query,zero(sum_query_count(query)))},
    ~{query => sum_zero_numeric(query) : @+query: SumQuery -> {sum_query_read(query,zero(sum_query_count(query))) == 0.0 : F32}},
    ~{query => left => right => left_frame => right_frame => sum_combine_frame(query,left,right,left_frame,right_frame) :
      @+query: SumQuery -> @+left: Tree.Tree<Tile.Tile8> -> @+right: Tree.Tree<Tile.Tile8> -> sum_query_frame(query,left) -> sum_query_frame(query,right) -> sum_query_frame(query,sum(sum_query_count(query),left,right))},
    ~{query => left => right => left_frame => right_frame => sum_combine_numeric(query,left,right,left_frame,right_frame) :
      @+query: SumQuery -> @+left: Tree.Tree<Tile.Tile8> -> @+right: Tree.Tree<Tile.Tile8> -> sum_query_frame(query,left) -> sum_query_frame(query,right) ->
        {sum_query_read(query,sum(sum_query_count(query),left,right)) == (sum_query_read(query,left) + sum_query_read(query,right) : F32) : F32}},fuel,query,values,evidence)

# A query observes the lane of its sums.
def observed_query(+query: GroupQuery,values: List<&2,Tree.Tree<Tile.Tile8>>) ->
  {GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~GroupQuery,~group_query_observe,values,query) ==
    GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~SumQuery,~sum_query_read,values,group_query_sum(query)) : List<&2,F32>}:
  match values:
    case Nil{}: {==}
    case +head <> +rest:
      Equal.cong(List<&2,F32>,List<&2,F32>,tail => sum_query_read(group_query_sum(query),head) <> tail,
        GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~GroupQuery,~group_query_observe,rest,query),
        GroupLaws.observed_values(~Tree.Tree<Tile.Tile8>,~F32,~SumQuery,~sum_query_read,rest,group_query_sum(query)),observed_query(query,rest))

# ---- The cells of the result ----

# Cells of values from index on, copied into the result from first + index.
type CopyModel is Data:
  CopyModel{values: Tree.Tree<Tile.Tile8>,result: Buffer.Model<Tile.Tile8>,index: Nat,first: U32}

def represent_copy(state: CopyModel) -> Kernel.TileCopy:
  CopyModel{values,result,index,first} = state
  Kernel.TileCopy{Tree.pack(Tile.Tile8,values),Buffer.pack(~Tile.Tile8,result),U32.from_nat(index),first}

def tile_written(+values: Tree.Tree<Tile.Tile8>,+index: Nat,+first: U32,result: Buffer.Model<Tile.Tile8>) -> Storage.Buffer<Tile.Tile8>:
  Storage.write(~Tile.Tile8,(first + U32.from_nat(index) : U32),tile_at(values,index),Buffer.pack(~Tile.Tile8,result))

def tile_copy(state: CopyModel) -> CopyModel:
  CopyModel{+values,result,+index,+first} = state
  CopyModel{values,Buffer.remodel(~Tile.Tile8,tile_written(values,index,first,result)),1n+index,first}

def tile_copy_matches(+state: CopyModel) -> {Kernel.tile_copy(represent_copy(state)) == represent_copy(tile_copy(state)) : Kernel.TileCopy}:
  CopyModel{+values,+result,+index,+first} = state
  +position = U32.from_nat(index)
  %Equal.sym(Array<Tile.Tile8> & Tile.Tile8,Array.get(Tile.Tile8,Tree.pack(Tile.Tile8,values),position),(Tree.pack(Tile.Tile8,values),Tree.read(Tile.Tile8,values,position)),StorageLaws.read(Tile.Tile8,values,position)) :
    {Kernel.tile_copied(position,first,Buffer.pack(~Tile.Tile8,result),_) == represent_copy(tile_copy(state)) : Kernel.TileCopy}
  %Equal.sym(U32,(position + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)) :
    {Kernel.TileCopy{Tree.pack(Tile.Tile8,values),tile_written(values,index,first,result),_,first} == Kernel.TileCopy{Tree.pack(Tile.Tile8,values),Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,tile_written(values,index,first,result))),U32.from_nat(1n+index),first} : Kernel.TileCopy}
  Equal.cong(Storage.Buffer<Tile.Tile8>,Kernel.TileCopy,buffer => Kernel.TileCopy{Tree.pack(Tile.Tile8,values),buffer,U32.from_nat(1n+index),first},
    tile_written(values,index,first,result),Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,tile_written(values,index,first,result))),
    Equal.sym(Storage.Buffer<Tile.Tile8>,Buffer.pack(~Tile.Tile8,Buffer.remodel(~Tile.Tile8,tile_written(values,index,first,result))),tile_written(values,index,first,result),Buffer.remodeled(~Tile.Tile8,tile_written(values,index,first,result))))

def copy_result(state: CopyModel) -> Buffer.Model<Tile.Tile8>:
  CopyModel{values,result,index,first} = state
  result

def copy_result_matches(+state: CopyModel) -> {Kernel.tiles_result(represent_copy(state)) == Buffer.pack(~Tile.Tile8,copy_result(state)) : Storage.Buffer<Tile.Tile8>}:
  CopyModel{values,result,index,first} = state
  {==}

# The place after first + index is first + (index + 1).
def next_position(+first: U32,+index: Nat) -> {(first + U32.from_nat(1n+index) : U32) == ((first + U32.from_nat(index) : U32) + 1 : U32) : U32}:
  Equal.trans(U32,(first + U32.from_nat(1n+index) : U32),(first + (U32.from_nat(index) + 1 : U32) : U32),((first + U32.from_nat(index) : U32) + 1 : U32),
    Equal.cong(U32,U32,value => (first + value : U32),U32.from_nat(1n+index),(U32.from_nat(index) + 1 : U32),
      Equal.sym(U32,(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index))),
    Equal.sym(U32,((first + U32.from_nat(index) : U32) + 1 : U32),(first + (U32.from_nat(index) + 1 : U32) : U32),Arithmetic.add_associative(first,U32.from_nat(index),1)))

# Count cells from index on, written in order from first + index.
def copy_content(count: Nat,+values: Tree.Tree<Tile.Tile8>,+result: Buffer.Model<Tile.Tile8>,+index: Nat,+first: U32) ->
  {Buffer.content(~Tile.Tile8,copy_result(Loop.run(~CopyModel,~tile_copy,count,CopyModel{values,result,index,first}))) ==
    Buffer.Content{Tree.write_sequence(~Tile.Tile8,tiles(count,index,values),(first + U32.from_nat(index) : U32),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))),
      Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))} : Buffer.Content<Tile.Tile8>}:
  match count:
    case 0n: Buffer.content_eta(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))
    case 1n+ +rest:
      +value = tile_at(values,index)
      +position = (first + U32.from_nat(index) : U32)
      +later = tiles(rest,1n+index,values)
      +before = Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))
      +length = Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))
      +written = Buffer.remodel(~Tile.Tile8,tile_written(values,index,first,result))
      +written_content = Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,written),Buffer.contents(~Tile.Tile8,tile_written(values,index,first,result)),
        Buffer.Content{Tree.write(Tile.Tile8,before,position,value),length},
        Buffer.remodel_content(~Tile.Tile8,tile_written(values,index,first,result)),Buffer.write_value_packed(~Tile.Tile8,position,value,result))
      Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,copy_result(Loop.run(~CopyModel,~tile_copy,rest,CopyModel{values,written,1n+index,first}))),
        Buffer.Content{Tree.write_sequence(~Tile.Tile8,later,(first + U32.from_nat(1n+index) : U32),Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,written))),
          Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,written))},
        Buffer.Content{Tree.write_sequence(~Tile.Tile8,later,(position + 1 : U32),Tree.write(Tile.Tile8,before,position,value)),length},
        copy_content(rest,values,written,1n+index,first),
        %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,written),Buffer.Content{Tree.write(Tile.Tile8,before,position,value),length},written_content) :
          {Buffer.Content{Tree.write_sequence(~Tile.Tile8,later,(first + U32.from_nat(1n+index) : U32),Buffer.content_tree(~Tile.Tile8,_)),Buffer.content_length(~Tile.Tile8,_)} ==
            Buffer.Content{Tree.write_sequence(~Tile.Tile8,later,(position + 1 : U32),Tree.write(Tile.Tile8,before,position,value)),length} : Buffer.Content<Tile.Tile8>}
        Equal.cong(U32,Buffer.Content<Tile.Tile8>,start => Buffer.Content{Tree.write_sequence(~Tile.Tile8,later,start,Tree.write(Tile.Tile8,before,position,value)),length},
          (first + U32.from_nat(1n+index) : U32),(position + 1 : U32),next_position(first,index)))

# Count cells of values written into a result from first: the cells in order
# from first.
def tiles_written(+count: Nat,+first: U32,+values: Tree.Tree<Tile.Tile8>,+result: Buffer.Model<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,Kernel.tiles_written(count,first,Tree.pack(Tile.Tile8,values),Buffer.pack(~Tile.Tile8,result))) ==
    Buffer.Content{Tree.write_sequence(~Tile.Tile8,tiles(count,0n,values),first,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))),
      Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))} : Buffer.Content<Tile.Tile8>}:
  +state = {CopyModel{values,result,0n,first} : CopyModel}
  +final = Loop.run(~CopyModel,~tile_copy,count,state)
  +list = tiles(count,0n,values)
  +before = Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))
  +length = Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,result))
  %Equal.sym(Kernel.TileCopy,Loop.run(~Kernel.TileCopy,~Kernel.tile_copy,count,represent_copy(state)),represent_copy(final),
    Projection.iteration_refines_model(~Kernel.TileCopy,~CopyModel,~Kernel.tile_copy,~tile_copy,~represent_copy,~tile_copy_matches,count,state)) :
    {Buffer.contents(~Tile.Tile8,Kernel.tiles_result(_)) == Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,first,before),length} : Buffer.Content<Tile.Tile8>}
  %Equal.sym(Storage.Buffer<Tile.Tile8>,Kernel.tiles_result(represent_copy(final)),Buffer.pack(~Tile.Tile8,copy_result(final)),copy_result_matches(final)) :
    {Buffer.contents(~Tile.Tile8,_) == Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,first,before),length} : Buffer.Content<Tile.Tile8>}
  %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Buffer.pack(~Tile.Tile8,copy_result(final))),Buffer.content(~Tile.Tile8,copy_result(final)),Buffer.packed_contents(~Tile.Tile8,copy_result(final))) :
    {_ == Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,first,before),length} : Buffer.Content<Tile.Tile8>}
  %Arithmetic.add_zero(first) :
    {Buffer.content(~Tile.Tile8,copy_result(final)) == Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,_,before),length} : Buffer.Content<Tile.Tile8>}
  copy_content(count,values,result,0n,first)

# Every cell written into a fresh result: the cells in order from 0.
def tiles_fresh(+count: U32,+values: Tree.Tree<Tile.Tile8>) ->
  {Buffer.contents(~Tile.Tile8,Kernel.tiles_written(U32.to_nat(count),0,Tree.pack(Tile.Tile8,values),Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))) ==
    Buffer.Content{Tree.write_sequence(~Tile.Tile8,tiles(U32.to_nat(count),0n,values),0,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}:
  +fresh = Buffer.remodel(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))
  +list = tiles(U32.to_nat(count),0n,values)
  +start = {Buffer.Content{Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}
  %Buffer.remodeled(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)) :
    {Buffer.contents(~Tile.Tile8,Kernel.tiles_written(U32.to_nat(count),0,Tree.pack(Tile.Tile8,values),_)) ==
      Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}
  Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.contents(~Tile.Tile8,Kernel.tiles_written(U32.to_nat(count),0,Tree.pack(Tile.Tile8,values),Buffer.pack(~Tile.Tile8,fresh))),
    Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Buffer.content_tree(~Tile.Tile8,Buffer.content(~Tile.Tile8,fresh))),Buffer.content_length(~Tile.Tile8,Buffer.content(~Tile.Tile8,fresh))},
    Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))),U32.to_nat(count)},
    tiles_written(U32.to_nat(count),0,values,fresh),
    %Equal.sym(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,fresh),start,
        Equal.trans(Buffer.Content<Tile.Tile8>,Buffer.content(~Tile.Tile8,fresh),Buffer.contents(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)),start,
          Buffer.remodel_content(~Tile.Tile8,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count)),Listed.allocated(~Tile.Tile8,Tile.filled(0.0),count))) :
      {Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Buffer.content_tree(~Tile.Tile8,_)),Buffer.content_length(~Tile.Tile8,_)} ==
        Buffer.Content{Tree.write_sequence(~Tile.Tile8,list,0,Tree.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0))),U32.to_nat(count)} : Buffer.Content<Tile.Tile8>}
    {==})
