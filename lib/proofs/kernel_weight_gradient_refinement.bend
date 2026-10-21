# Whole-array adjacent trees and transposed readback of the packed groups.
import Base
import ../kernel_weight_gradient.bend as Kernel
import ../kernel_parameter_gradient.bend as Parameter
import ../kernel_tile.bend as Tile
import ../storage_buffer.bend as Storage
import ../traversal_array.bend as Traversal
import ../traversal_grouped.bend as Grouped
import ../traversal_lanes.bend as Lanes
import ../tensor_lanes.bend as TensorLanes
import ./kernel_weight_gradient_group.bend as Groups
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_refinement.bend as StorageLaws
import ./traversal_array_refinement.bend as Traversals
import ./traversal_lanes_refinement.bend as LaneLaws
import ./tensor_lanes_refinement.bend as TensorLaneLaws
import ./grouped_reduction_refinement.bend as GroupLaws
import ./traversal_batch_refinement.bend as Batch
import ./traversal_map_readback.bend as MapReadback
import ./storage_shape.bend as Shape
import ./encoded_coordinates.bend as Encoded
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./storage_readback.bend as StorageReadback
import ./sequence_membership.bend as Members
import ./kernel_weight_gradient_packing.bend as Packing
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_full_range.bend as FullRange
import ./ordered_ranges.bend as Ranges
import ./list_storage_readback.bend as ListRead
import ../traversal_loop.bend as Loop
import ../tensor.bend as Tensors
import ./loop_projection.bend as Projection
import ./tensor_list_values.bend as Listed

type SumOwners is Data:
  SumOwners{left: Tree.Tree<F32>,right: Tree.Tree<F32>}

def represent_sum(state: SumOwners) -> Array<F32> & Array<F32>:
  SumOwners{left,right} = state
  (Tree.pack(F32,left),Tree.pack(F32,right))

def sum_step(position: U32,context: Unit,state: SumOwners) -> SumOwners:
  SumOwners{+left,+right} = state
  +position = (position * 8 : U32)
  SumOwners{TensorLaneLaws.written(left,position,TensorLaneLaws.pair_cell(~F32.add,left,right,position)),right}

def sum_step_matches(+index: U32,+context: Unit,+state: SumOwners) ->
  {Kernel.sum_step(index,context,represent_sum(state)) == represent_sum(sum_step(index,context,state)) : Array<F32> & Array<F32>}:
  SumOwners{+left,+right} = state
  +position = (index * 8 : U32)
  +a = LaneLaws.cell(~Unit,~TensorLanes.identity,left,position,Unit{})
  +b = LaneLaws.cell(~Unit,~TensorLanes.identity,right,position,Unit{})
  +tile = TensorLanes.combine_tile(~F32.add,a,b)
  %Equal.sym(Array<F32> & Tile.Tile8,Lanes.gather(~Unit,~TensorLanes.identity,Tree.pack(F32,left),position,Unit{}),(Tree.pack(F32,left),a),
    LaneLaws.gathered(~Unit,~TensorLanes.identity,left,position,Unit{})) :
    {Kernel.sum_result(position,_,Lanes.gather(~Unit,~TensorLanes.identity,Tree.pack(F32,right),position,Unit{})) == represent_sum(sum_step(index,context,state)) : Array<F32> & Array<F32>}
  %Equal.sym(Array<F32> & Tile.Tile8,Lanes.gather(~Unit,~TensorLanes.identity,Tree.pack(F32,right),position,Unit{}),(Tree.pack(F32,right),b),
    LaneLaws.gathered(~Unit,~TensorLanes.identity,right,position,Unit{})) :
    {Kernel.sum_result(position,(Tree.pack(F32,left),a),_) == (Tree.pack(F32,TensorLaneLaws.written(left,position,tile)),Tree.pack(F32,right)) : Array<F32> & Array<F32>}
  Equal.cong(Array<F32>,Array<F32> & Array<F32>,left => (left,Tree.pack(F32,right)),Lanes.scatter(Tree.pack(F32,left),position,tile),
    Tree.pack(F32,TensorLaneLaws.written(left,position,tile)),TensorLaneLaws.scattered(left,position,tile))

def sum_output(state: SumOwners) -> Tree.Tree<F32>:
  SumOwners{left,right} = state
  left

def sum_output_matches(+state: SumOwners) ->
  {Pair.fst(Array<F32>,Array<F32>,represent_sum(state)) == Tree.pack(F32,sum_output(state)) : Array<F32>}:
  SumOwners{left,right} = state
  {==}

def sum(+count: U32,left: Tree.Tree<F32>,right: Tree.Tree<F32>) -> Tree.Tree<F32>:
  sum_output(Traversal.indexed_fold(~SumOwners,~Unit,~sum_step,U32.to_nat(((count + 7) / 8 : U32)),0,Unit{},SumOwners{left,right}))

def sum_matches(+count: U32,+left: Tree.Tree<F32>,+right: Tree.Tree<F32>) ->
  {Kernel.sum(count,Tree.pack(F32,left),Tree.pack(F32,right)) == Tree.pack(F32,sum(count,left,right)) : Array<F32>}:
  +initial = {SumOwners{left,right} : SumOwners}
  +rounds = U32.to_nat(((count + 7) / 8 : U32))
  +state = Traversal.indexed_fold(~SumOwners,~Unit,~sum_step,rounds,0,Unit{},initial)
  %Equal.sym(Array<F32> & Array<F32>,Traversal.indexed_fold(~(Array<F32> & Array<F32>),~Unit,~Kernel.sum_step,rounds,0,Unit{},represent_sum(initial)),represent_sum(state),
    Traversals.indexed_fold_refines_model(~(Array<F32> & Array<F32>),~SumOwners,~Unit,~Kernel.sum_step,~sum_step,~represent_sum,~sum_step_matches,rounds,0,Unit{},initial)) :
    {Pair.fst(Array<F32>,Array<F32>,_) == Tree.pack(F32,sum(count,left,right)) : Array<F32>}
  sum_output_matches(state)

def sum_right(state: SumOwners) -> Tree.Tree<F32>:
  SumOwners{left,right} = state
  right

def sum_frame(right: Tree.Tree<F32>,state: SumOwners) -> Data:
  {sum_right(state) == right : Tree.Tree<F32>}

def sum_fold_step(index: U32,context: Tree.Tree<F32>,state: SumOwners) -> SumOwners:
  sum_step(index,Unit{},state)

def sum_frame_step(+context: Tree.Tree<F32>,+index: U32,+state: SumOwners,frame: sum_frame(context,state)) ->
  sum_frame(context,sum_fold_step(index,context,state)):
  SumOwners{left,right} = state
  frame

def sum_batch_matches(+context: Tree.Tree<F32>,+index: Nat,+state: SumOwners,+depth: Nat,+capacity: U32,
  frame: sum_frame(context,state),capacity_equal: {Tree.capacity(F32,sum_output(state)) == capacity : U32},
  balanced: Shape.Balanced(F32,depth,sum_output(state)),limit: {Nat.is_le(depth,31n) == True{} : Bool},
  capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(Nat.add(Nat.mul(index,8n),8n),U32.to_nat(capacity)) == True{} : Bool}) ->
  {sum_output(sum_fold_step(U32.from_nat(index),context,state)) ==
    MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),8n,U32.from_nat(Nat.mul(index,8n)),context,sum_output(state)) : Tree.Tree<F32>}:
  SumOwners{+left,+right} = state
  %Equal.sym(U32,(U32.from_nat(index) * 8 : U32),U32.from_nat(Nat.mul(index,8n)),Encoded.scale_encoded(index,8)) :
    {TensorLaneLaws.written(left,_,TensorLaneLaws.pair_cell(~F32.add,left,right,_)) ==
      MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),8n,U32.from_nat(Nat.mul(index,8n)),context,left) : Tree.Tree<F32>}
  %frame :
    {TensorLaneLaws.written(left,U32.from_nat(Nat.mul(index,8n)),TensorLaneLaws.pair_cell(~F32.add,left,right,U32.from_nat(Nat.mul(index,8n)))) ==
      MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),8n,U32.from_nat(Nat.mul(index,8n)),_,left) : Tree.Tree<F32>}
  TensorLaneLaws.pair_batch_mapped(~F32.add,left,right,Nat.mul(index,8n),depth,capacity,capacity_equal,balanced,limit,capacity_ok,room)

def sum_fold_inner(+index: U32,+context: Tree.Tree<F32>,+state: SumOwners) ->
  {sum_fold_step(index,context,state) == sum_step(index,Unit{},state) : SumOwners}:
  {==}

def sum_mapped(+count: U32,+left: Tree.Tree<F32>,+right: Tree.Tree<F32>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(F32,left) == capacity : U32},+balanced: Shape.Balanced(F32,depth,left),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(((count + 7) / 8 : U32)),8n),U32.to_nat(capacity)) == True{} : Bool}) ->
  {sum(count,left,right) == MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),Nat.mul(U32.to_nat(((count + 7) / 8 : U32)),8n),0,right,left) : Tree.Tree<F32>}:
  +rounds = U32.to_nat(((count + 7) / 8 : U32))
  +initial = {SumOwners{left,right} : SumOwners}
  %Traversals.indexed_fold_change_context(~SumOwners,~Tree.Tree<F32>,~Unit,~(right => Unit{}),~sum_fold_step,~sum_step,~sum_fold_inner,rounds,0,right,initial) :
    {sum_output(_) == MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),Nat.mul(rounds,8n),0,right,left) : Tree.Tree<F32>}
  Shape.second(sum_frame(right,Traversal.indexed_fold(~SumOwners,~Tree.Tree<F32>,~sum_fold_step,rounds,0,right,initial)),
    {sum_output(Traversal.indexed_fold(~SumOwners,~Tree.Tree<F32>,~sum_fold_step,rounds,0,right,initial)) == MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),Nat.mul(rounds,8n),0,right,left) : Tree.Tree<F32>},
    Batch.fold_mapped(~F32,~Tree.Tree<F32>,~SumOwners,~Tree.Tree<F32>,~8n,~sum_output,~{right => right : Tree.Tree<F32> -> Tree.Tree<F32>},~TensorLaneLaws.pair_value(~F32.add),~sum_frame,~sum_fold_step,~sum_frame_step,~sum_batch_matches,
      rounds,right,initial,depth,capacity,{==},capacity_equal,balanced,limit,capacity_ok,room))

def sum_numeric(+count: U32,+left: Tree.Tree<F32>,+right: Tree.Tree<F32>,+depth: Nat,+capacity: U32,+index: Nat,
  +capacity_equal: {Tree.capacity(F32,left) == capacity : U32},+balanced: Shape.Balanced(F32,depth,left),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(((count + 7) / 8 : U32)),8n),U32.to_nat(capacity)) == True{} : Bool},
  selected: {Nat.is_lt(index,Nat.mul(U32.to_nat(((count + 7) / 8 : U32)),8n)) == True{} : Bool}) ->
  {Tree.read(F32,sum(count,left,right),U32.from_nat(index)) == (Tree.read(F32,left,U32.from_nat(index)) + Tree.read(F32,right,U32.from_nat(index)) : F32) : F32}:
  +extent = Nat.mul(U32.to_nat(((count + 7) / 8 : U32)),8n)
  %Equal.sym(Tree.Tree<F32>,sum(count,left,right),MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),extent,0,right,left),
    sum_mapped(count,left,right,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room)) :
    {Tree.read(F32,_,U32.from_nat(index)) == (Tree.read(F32,left,U32.from_nat(index)) + Tree.read(F32,right,U32.from_nat(index)) : F32) : F32}
  MapReadback.inside_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),extent,0n,index,right,left,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room,selected)

def sum_storage(+count: U32,+left: Tree.Tree<F32>,+right: Tree.Tree<F32>,+depth: Nat,+capacity: U32,
  +capacity_equal: {Tree.capacity(F32,left) == capacity : U32},+balanced: Shape.Balanced(F32,depth,left),
  +limit: {Nat.is_le(depth,31n) == True{} : Bool},+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(((count + 7) / 8 : U32)),8n),U32.to_nat(capacity)) == True{} : Bool}) ->
  Groups.StorageFrame(depth,capacity,sum(count,left,right)):
  +extent = Nat.mul(U32.to_nat(((count + 7) / 8 : U32)),8n)
  %Equal.sym(Tree.Tree<F32>,sum(count,left,right),MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),extent,0,right,left),
    sum_mapped(count,left,right,depth,capacity,capacity_equal,balanced,limit,capacity_ok,room)) : Groups.StorageFrame(depth,capacity,_)
  Shape.Both{MapReadback.balanced_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),extent,0,right,left,depth,balanced),
    Equal.trans(U32,Tree.capacity(F32,MapReadback.run_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),extent,0,right,left)),Tree.capacity(F32,left),capacity,
      MapReadback.capacity_indexed(~F32,~Tree.Tree<F32>,~TensorLaneLaws.pair_value(~F32.add),extent,0,right,left),capacity_equal)}

def zero(count: U32) -> Tree.Tree<F32>:
  Tree.fresh(F32,Storage.depth_for((((count + 7) / 8) * 8 : U32)),0.0)

def zero_matches(+count: U32) ->
  {Kernel.array(~F32,(((count + 7) / 8) * 8 : U32),0.0) == Tree.pack(F32,zero(count)) : Array<F32>}:
  StorageLaws.fresh(F32,Storage.depth_for((((count + 7) / 8) * 8 : U32)),0.0)

type SumQuery is Data:
  SumQuery{count: U32,index: Nat,depth: Nat,capacity: U32,
    zero_depth: {Storage.depth_for((((count + 7) / 8) * 8 : U32)) == depth : Nat},
    zero_capacity: {U32.shln(1,depth) == capacity : U32},
    depth_ok: {Nat.is_le(depth,31n) == True{} : Bool},capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
    room: {Nat.is_le(Nat.mul(U32.to_nat(((count + 7) / 8 : U32)),8n),U32.to_nat(capacity)) == True{} : Bool},
    selected: {Nat.is_lt(index,Nat.mul(U32.to_nat(((count + 7) / 8 : U32)),8n)) == True{} : Bool}}

def plan_sum_query(+plan: Groups.GroupPlan,+index: Nat,selected: {Nat.is_lt(index,U32.to_nat(Groups.plan_count(plan))) == True{} : Bool}) -> SumQuery:
  SumQuery{Groups.plan_count(plan),index,Groups.plan_depth(plan),Groups.plan_capacity(plan),{==},{==},
    Order.successor(Groups.plan_depth(plan),30n,Groups.plan_depth_limit(plan)),Groups.plan_capacity_bound(plan),
    Groups.plan_round_room(plan),Groups.plan_round_selected(plan,index,selected)}

def sum_query_count(query: SumQuery) -> U32:
  SumQuery{count,index,depth,capacity,zero_depth,zero_capacity,depth_ok,capacity_ok,room,selected} = query
  count

def sum_query_read(query: SumQuery,tree: Tree.Tree<F32>) -> F32:
  SumQuery{count,index,depth,capacity,zero_depth,zero_capacity,depth_ok,capacity_ok,room,selected} = query
  Tree.read(F32,tree,U32.from_nat(index))

def sum_query_frame(query: SumQuery,tree: Tree.Tree<F32>) -> Data:
  SumQuery{count,index,depth,capacity,zero_depth,zero_capacity,depth_ok,capacity_ok,room,selected} = query
  Groups.StorageFrame(depth,capacity,tree)

def sum_zero_frame(+query: SumQuery) -> sum_query_frame(query,zero(sum_query_count(query))):
  SumQuery{+count,index,+depth,+capacity,+zero_depth,+zero_capacity,depth_ok,capacity_ok,room,selected} = query
  %Equal.sym(Nat,Storage.depth_for((((count + 7) / 8) * 8 : U32)),depth,zero_depth) : Groups.StorageFrame(depth,capacity,Tree.fresh(F32,_,0.0))
  Shape.Both{Shape.fresh(F32,depth,0.0),Equal.trans(U32,Tree.capacity(F32,Tree.fresh(F32,depth,0.0)),U32.shln(1,depth),capacity,
    Shape.capacity(F32,depth,Tree.fresh(F32,depth,0.0),Shape.fresh(F32,depth,0.0)),zero_capacity)}

def sum_zero_numeric(+query: SumQuery) -> {sum_query_read(query,zero(sum_query_count(query))) == 0.0 : F32}:
  SumQuery{+count,index,depth,capacity,zero_depth,zero_capacity,depth_ok,capacity_ok,room,selected} = query
  StorageReadback.read_fresh(~F32,Storage.depth_for((((count + 7) / 8) * 8 : U32)),0.0,U32.from_nat(index))

def sum_combine_frame(+query: SumQuery,+left: Tree.Tree<F32>,+right: Tree.Tree<F32>,left_frame: sum_query_frame(query,left),right_frame: sum_query_frame(query,right)) ->
  sum_query_frame(query,sum(sum_query_count(query),left,right)):
  SumQuery{+count,index,+depth,+capacity,zero_depth,zero_capacity,+depth_ok,+capacity_ok,+room,selected} = query
  Shape.Both{balanced,capacity_equal} = left_frame
  sum_storage(count,left,right,depth,capacity,capacity_equal,balanced,depth_ok,capacity_ok,room)

def sum_combine_numeric(+query: SumQuery,+left: Tree.Tree<F32>,+right: Tree.Tree<F32>,left_frame: sum_query_frame(query,left),right_frame: sum_query_frame(query,right)) ->
  {sum_query_read(query,sum(sum_query_count(query),left,right)) == (sum_query_read(query,left) + sum_query_read(query,right) : F32) : F32}:
  SumQuery{+count,+index,+depth,+capacity,zero_depth,zero_capacity,+depth_ok,+capacity_ok,+room,+selected} = query
  Shape.Both{balanced,capacity_equal} = left_frame
  sum_numeric(count,left,right,depth,capacity,index,capacity_equal,balanced,depth_ok,capacity_ok,room,selected)

type GroupList is Data:
  GroupList{input: Buffer.Model<F32>,gradient: Buffer.Model<F32>,values: List<&2,Tree.Tree<F32>>}

def represent_list(result: GroupList) -> (Storage.Buffer<F32> & Storage.Buffer<F32>) & List<&1,Array<F32>>:
  GroupList{input,gradient,values} = result
  ((Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),GroupLaws.pack_values(~Array<F32>,~Tree.Tree<F32>,~(tree => Tree.pack(F32,tree)),values))

def reduced(fuel: Nat,count: U32,result: GroupList) -> Groups.Group:
  GroupList{input,gradient,values} = result
  Groups.Group{input,gradient,Grouped.levels(~&2,~Tree.Tree<F32>,~U32,~zero,~(count => left => right => sum(count,left,right)),fuel,count,values)}

def reduced_matches(+fuel: Nat,+count: U32,+result: GroupList) ->
  {Kernel.reduced(fuel,count,represent_list(result)) == Groups.represent(reduced(fuel,count,result)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}:
  GroupList{+input,+gradient,+values} = result
  Equal.cong(Array<F32>,(Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>,values => ((Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),values),
    Grouped.levels(~&1,~Array<F32>,~U32,~(count => Kernel.array(~F32,(((count + 7) / 8) * 8 : U32),0.0)),~(count => left => right => Kernel.sum(count,left,right)),fuel,count,
      GroupLaws.pack_values(~Array<F32>,~Tree.Tree<F32>,~(tree => Tree.pack(F32,tree)),values)),
    Tree.pack(F32,Grouped.levels(~&2,~Tree.Tree<F32>,~U32,~zero,~(count => left => right => sum(count,left,right)),fuel,count,values)),
    GroupLaws.levels_refines_model(~Array<F32>,~Tree.Tree<F32>,~U32,~(count => Kernel.array(~F32,(((count + 7) / 8) * 8 : U32),0.0)),~zero,
      ~(count => left => right => Kernel.sum(count,left,right)),~(count => left => right => sum(count,left,right)),~(tree => Tree.pack(F32,tree)),~zero_matches,~sum_matches,fuel,count,values))

def source_reader(source: Groups.Sources) -> Storage.Buffer<F32> & Storage.Buffer<F32>:
  Groups.Sources{input,gradient} = source
  (Buffer.pack(~F32,input),Buffer.pack(~F32,gradient))

def group_split(result: Groups.Group) -> Groups.Sources & Tree.Tree<F32>:
  Groups.Group{input,gradient,values} = result
  (Groups.Sources{input,gradient},values)

def group_split_matches(+result: Groups.Group) ->
  {Groups.represent(result) == Traversals.collect_list_read(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Groups.Sources,~Array<F32>,~Tree.Tree<F32>,~source_reader,~(tree => Tree.pack(F32,tree)),group_split(result)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}:
  Groups.Group{input,gradient,values} = result
  {==}

def collected_groups(result: Groups.Sources & List<&2,Tree.Tree<F32>>) -> GroupList:
  (Groups.Sources{input,gradient},values) = result
  GroupList{input,gradient,values}

def collected_groups_matches(result: Groups.Sources & List<&2,Tree.Tree<F32>>) ->
  {Traversals.collect_result(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Groups.Sources,~Array<F32>,~Tree.Tree<F32>,~source_reader,~(tree => Tree.pack(F32,tree)),result) == represent_list(collected_groups(result)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & List<&1,Array<F32>>}:
  (Groups.Sources{input,gradient},values) = result
  {==}

def group_read(source: Groups.Sources,context: Kernel.GroupContext,range: Grouped.Range) -> Groups.Sources & Tree.Tree<F32>:
  Groups.Sources{input,gradient} = source
  Kernel.GroupContext{geometry,image} = context
  group_split(Groups.group(geometry,image,range,input,gradient))

def group_read_matches(+source: Groups.Sources,+context: Kernel.GroupContext,+range: Grouped.Range) ->
  {Kernel.group_read(source_reader(source),context,range) == Traversals.collect_list_read(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Groups.Sources,~Array<F32>,~Tree.Tree<F32>,~source_reader,~(tree => Tree.pack(F32,tree)),group_read(source,context,range)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}:
  Groups.Sources{+input,+gradient} = source
  Kernel.GroupContext{+geometry,+image} = context
  Equal.trans((Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>,Kernel.group(geometry,image,range,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),Groups.represent(Groups.group(geometry,image,range,input,gradient)),
    Traversals.collect_list_read(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Groups.Sources,~Array<F32>,~Tree.Tree<F32>,~source_reader,~(tree => Tree.pack(F32,tree)),group_read(source,context,range)),Groups.group_matches(geometry,image,range,input,gradient),group_split_matches(Groups.group(geometry,image,range,input,gradient)))

def groups(ranges: List<&2,Grouped.Range>,geometry: Kernel.Geometry,image: U32,input: Buffer.Model<F32>,gradient: Buffer.Model<F32>) -> GroupList:
  collected_groups(Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,image}))

def groups_matches(+ranges: List<&2,Grouped.Range>,+geometry: Kernel.Geometry,+image: U32,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {Kernel.groups(ranges,geometry,image,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)) == represent_list(groups(ranges,geometry,image,input,gradient)) :
    (Storage.Buffer<F32> & Storage.Buffer<F32>) & List<&1,Array<F32>>}:
  Equal.trans((Storage.Buffer<F32> & Storage.Buffer<F32>) & List<&1,Array<F32>>,Kernel.groups(ranges,geometry,image,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),
    Traversals.collect_result(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Groups.Sources,~Array<F32>,~Tree.Tree<F32>,~source_reader,~(tree => Tree.pack(F32,tree)),Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,image})),represent_list(groups(ranges,geometry,image,input,gradient)),
    Traversals.collect_list_loop_refines_model(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Groups.Sources,~Array<F32>,~Tree.Tree<F32>,~Grouped.Range,~Kernel.GroupContext,~source_reader,~(tree => Tree.pack(F32,tree)),~Kernel.group_read,~group_read,~group_read_matches,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,image}),
    collected_groups_matches(Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,image})))

def image(+geometry: Kernel.Geometry,index: U32,+ranges: List<&2,Grouped.Range>,input: Buffer.Model<F32>,gradient: Buffer.Model<F32>) -> Groups.Group:
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  reduced(List.length(&2,Grouped.Range,ranges),(o * taps : U32),groups(ranges,geometry,index,input,gradient))

def image_matches(+geometry: Kernel.Geometry,+index: U32,+ranges: List<&2,Grouped.Range>,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {Kernel.image(geometry,index,ranges,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)) == Groups.represent(image(geometry,index,ranges,input,gradient)) :
    (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}:
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  +fuel = List.length(&2,Grouped.Range,ranges)
  +count = (o * taps : U32)
  +result = groups(ranges,geometry,index,input,gradient)
  %Equal.sym((Storage.Buffer<F32> & Storage.Buffer<F32>) & List<&1,Array<F32>>,Kernel.groups(ranges,geometry,index,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),represent_list(result),groups_matches(ranges,geometry,index,input,gradient)) :
    {Kernel.reduced(fuel,count,_) == Groups.represent(image(geometry,index,ranges,input,gradient)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}
  reduced_matches(fuel,count,result)

def image_collected(context: Kernel.ImageContext,result: Groups.Group) -> Groups.Sources & Traversal.Collected<&2,Kernel.ImageContext,Tree.Tree<F32>>:
  Groups.Group{input,gradient,values} = result
  (Groups.Sources{input,gradient},Traversal.Collected{context,values})

def image_collected_matches(+context: Kernel.ImageContext,+result: Groups.Group) ->
  {Kernel.image_collected(context,Groups.represent(result)) == Traversals.collected_result(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Groups.Sources,~Array<F32>,~Tree.Tree<F32>,~Kernel.ImageContext,~source_reader,~(tree => Tree.pack(F32,tree)),image_collected(context,result)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Traversal.Collected<&1,Kernel.ImageContext,Array<F32>>}:
  Groups.Group{input,gradient,values} = result
  {==}

def image_read(source: Groups.Sources,context: Kernel.ImageContext) -> Groups.Sources & Traversal.Collected<&2,Kernel.ImageContext,Tree.Tree<F32>>:
  Groups.Sources{input,gradient} = source
  Kernel.ImageContext{+geometry,+index,+rows,+columns,+group} = context
  image_collected(Kernel.ImageContext{geometry,(index + 1 : U32),rows,columns,group},image(geometry,index,Grouped.ranges(rows,columns,group),input,gradient))

def image_read_matches(+source: Groups.Sources,+context: Kernel.ImageContext) ->
  {Kernel.image_read(source_reader(source),context) == Traversals.collected_result(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Groups.Sources,~Array<F32>,~Tree.Tree<F32>,~Kernel.ImageContext,~source_reader,~(tree => Tree.pack(F32,tree)),image_read(source,context)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Traversal.Collected<&1,Kernel.ImageContext,Array<F32>>}:
  Groups.Sources{+input,+gradient} = source
  Kernel.ImageContext{+geometry,+index,+rows,+columns,+group} = context
  +ranges = Grouped.ranges(rows,columns,group)
  %Equal.sym((Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>,Kernel.image(geometry,index,ranges,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),Groups.represent(image(geometry,index,ranges,input,gradient)),image_matches(geometry,index,ranges,input,gradient)) :
    {Kernel.image_collected(Kernel.ImageContext{geometry,(index + 1 : U32),rows,columns,group},_) == Traversals.collected_result(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Groups.Sources,~Array<F32>,~Tree.Tree<F32>,~Kernel.ImageContext,~source_reader,~(tree => Tree.pack(F32,tree)),image_read(source,context)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Traversal.Collected<&1,Kernel.ImageContext,Array<F32>>}
  image_collected_matches(Kernel.ImageContext{geometry,(index + 1 : U32),rows,columns,group},image(geometry,index,ranges,input,gradient))

def images(remaining: Nat,index: U32,geometry: Kernel.Geometry,rows: Nat,columns: Nat,group: Nat,input: Buffer.Model<F32>,gradient: Buffer.Model<F32>) -> GroupList:
  collected_groups(Traversal.collect(~&2,~Groups.Sources,~Tree.Tree<F32>,~Kernel.ImageContext,~image_read,remaining,Groups.Sources{input,gradient},Kernel.ImageContext{geometry,index,rows,columns,group}))

def images_matches(+remaining: Nat,+index: U32,+geometry: Kernel.Geometry,+rows: Nat,+columns: Nat,+group: Nat,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {Kernel.images(remaining,index,geometry,rows,columns,group,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)) == represent_list(images(remaining,index,geometry,rows,columns,group,input,gradient)) :
    (Storage.Buffer<F32> & Storage.Buffer<F32>) & List<&1,Array<F32>>}:
  +context = {Kernel.ImageContext{geometry,index,rows,columns,group} : Kernel.ImageContext}
  Equal.trans((Storage.Buffer<F32> & Storage.Buffer<F32>) & List<&1,Array<F32>>,Kernel.images(remaining,index,geometry,rows,columns,group,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),
    Traversals.collect_result(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Groups.Sources,~Array<F32>,~Tree.Tree<F32>,~source_reader,~(tree => Tree.pack(F32,tree)),Traversal.collect(~&2,~Groups.Sources,~Tree.Tree<F32>,~Kernel.ImageContext,~image_read,remaining,Groups.Sources{input,gradient},context)),represent_list(images(remaining,index,geometry,rows,columns,group,input,gradient)),
    Traversals.collect_loop_refines_model(~&1,~(Storage.Buffer<F32> & Storage.Buffer<F32>),~Groups.Sources,~Array<F32>,~Tree.Tree<F32>,~Kernel.ImageContext,~source_reader,~(tree => Tree.pack(F32,tree)),~Kernel.image_read,~image_read,~image_read_matches,remaining,Groups.Sources{input,gradient},context),
    collected_groups_matches(Traversal.collect(~&2,~Groups.Sources,~Tree.Tree<F32>,~Kernel.ImageContext,~image_read,remaining,Groups.Sources{input,gradient},context)))

# Output row of the values: taps values read at stride outputs.
type RowContext is Data:
  RowContext{address: U32,stride: U32}

def row_read(input: Tree.Tree<F32>,context: RowContext) -> Tree.Tree<F32> & Traversal.Collected<&2,RowContext,F32>:
  RowContext{+address,+stride} = context
  +source = input
  (source,Traversal.Collected{RowContext{(address + stride : U32),stride},Tree.read(F32,source,address)})

def row_result(remaining: Nat,address: U32,stride: U32,input: Tree.Tree<F32>) -> Tree.Tree<F32> & List<&2,F32>:
  Traversal.collect(~&2,~Tree.Tree<F32>,~F32,~RowContext,~row_read,remaining,input,RowContext{address,stride})

def row_values(remaining: Nat,address: U32,stride: U32,input: Tree.Tree<F32>) -> List<&2,F32>:
  Pair.snd(Tree.Tree<F32>,List<&2,F32>,row_result(remaining,address,stride,input))

# The model reads the values, [tap][output], as rows of outputs.
type TransposeContext is Data:
  TransposeContext{output: U32,outputs: U32,taps: Nat}

def transpose_read_row(input: Tree.Tree<F32>,context: TransposeContext) -> Tree.Tree<F32> & Traversal.Collected<&2,TransposeContext,List<&2,F32>>:
  TransposeContext{+output,+outputs,+taps} = context
  +source = input
  (source,Traversal.Collected{TransposeContext{(output + 1 : U32),outputs,taps},row_values(taps,output,outputs,source)})

def transpose_read_row_kept(+input: Tree.Tree<F32>,+context: TransposeContext) ->
  {Pair.fst(Tree.Tree<F32>,Traversal.Collected<&2,TransposeContext,List<&2,F32>>,transpose_read_row(input,context)) == input : Tree.Tree<F32>}:
  TransposeContext{output,outputs,taps} = context
  {==}

def transpose_rows(remaining: Nat,output: U32,outputs: U32,taps: Nat,input: Tree.Tree<F32>) -> Tree.Tree<F32> & List<&2,List<&2,F32>>:
  Traversal.collect(~&2,~Tree.Tree<F32>,~List<&2,F32>,~TransposeContext,~transpose_read_row,remaining,input,TransposeContext{output,outputs,taps})

def transpose(remaining: Nat,output: U32,outputs: U32,taps: Nat,input: Tree.Tree<F32>) -> List<&2,F32>:
  List.concat(&2,F32,Pair.snd(Tree.Tree<F32>,List<&2,List<&2,F32>>,transpose_rows(remaining,output,outputs,taps,input)))

type WeightValues is Data:
  WeightValues{input: Buffer.Model<F32>,gradient: Buffer.Model<F32>,values: List<&2,F32>}

def finished(geometry: Kernel.Geometry,result: Groups.Group) -> WeightValues:
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,taps} = geometry
  Groups.Group{input,gradient,values} = result
  WeightValues{input,gradient,transpose(U32.to_nat(o),0,o,U32.to_nat(taps),values)}

# The values of every tap and output, [tap][output].
def values_selected(+n: Nat,rows: Nat,columns: Nat,g: Nat,+geometry: Kernel.Geometry,input: Buffer.Model<F32>,gradient: Buffer.Model<F32>) -> Groups.Group:
  Kernel.Geometry{gc,+go,gh,gw,gk,gs,gp,gr,gcols,+taps} = geometry
  reduced(n,(go * taps : U32),images(n,0,geometry,rows,columns,g,input,gradient))

def values_selected_matches(+n: Nat,+rows: Nat,+columns: Nat,+g: Nat,+geometry: Kernel.Geometry,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {Kernel.values_selected(n,rows,columns,g,geometry,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)) == Groups.represent(values_selected(n,rows,columns,g,geometry,input,gradient)) :
    (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}:
  Kernel.Geometry{gc,+go,gh,gw,gk,gs,gp,gr,gcols,+taps} = geometry
  +all = images(n,0,geometry,rows,columns,g,input,gradient)
  %Equal.sym((Storage.Buffer<F32> & Storage.Buffer<F32>) & List<&1,Array<F32>>,Kernel.images(n,0,geometry,rows,columns,g,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),represent_list(all),
    images_matches(n,0,geometry,rows,columns,g,input,gradient)) :
    {Kernel.reduced(n,(go * taps : U32),_) == Groups.represent(values_selected(n,rows,columns,g,geometry,input,gradient)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}
  reduced_matches(n,(go * taps : U32),all)

def values(+work: Parameter.Work,input: Buffer.Model<F32>,gradient: Buffer.Model<F32>) -> Groups.Group:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  values_selected(n,rows,columns,g,Kernel.geometry(work),input,gradient)

def values_matches(+work: Parameter.Work,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {Kernel.weight_values(work,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)) == Groups.represent(values(work,input,gradient)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>}:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  values_selected_matches(n,rows,columns,g,Kernel.geometry(work),input,gradient)

# The weight gradient as a list, [output][tap].
def weight(+work: Parameter.Work,input: Buffer.Model<F32>,gradient: Buffer.Model<F32>) -> WeightValues:
  finished(Kernel.geometry(work),values(work,input,gradient))

# ---- The kernel gives back the input and the gradient it read ----

def group_split_sources(+result: Groups.Group) -> {Pair.fst(Groups.Sources,Tree.Tree<F32>,group_split(result)) == Groups.group_sources(result) : Groups.Sources}:
  Groups.Group{input,gradient,values} = result
  {==}

def group_read_kept(+source: Groups.Sources,+context: Kernel.GroupContext,+range: Grouped.Range) ->
  {Pair.fst(Groups.Sources,Tree.Tree<F32>,group_read(source,context,range)) == source : Groups.Sources}:
  Groups.Sources{+input,+gradient} = source
  Kernel.GroupContext{+geometry,+image} = context
  Equal.trans(Groups.Sources,Pair.fst(Groups.Sources,Tree.Tree<F32>,group_split(Groups.group(geometry,image,range,input,gradient))),
    Groups.group_sources(Groups.group(geometry,image,range,input,gradient)),Groups.Sources{input,gradient},
    group_split_sources(Groups.group(geometry,image,range,input,gradient)),Groups.group_sources_kept(geometry,image,range,input,gradient))

# A collection that kept its sources holds them.
def collected_kept(+sources: Groups.Sources,+values: List<&2,Tree.Tree<F32>>,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>,
  same: {sources == Groups.Sources{input,gradient} : Groups.Sources}) -> {collected_groups((sources,values)) == GroupList{input,gradient,values} : GroupList}:
  %Equal.sym(Groups.Sources,sources,Groups.Sources{input,gradient},same) : {collected_groups((_,values)) == GroupList{input,gradient,values} : GroupList}
  {==}

def collected_result(result: Groups.Sources & List<&2,Tree.Tree<F32>>,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>,
  same: {Pair.fst(Groups.Sources,List<&2,Tree.Tree<F32>>,result) == Groups.Sources{input,gradient} : Groups.Sources}) ->
  {collected_groups(result) == GroupList{input,gradient,Pair.snd(Groups.Sources,List<&2,Tree.Tree<F32>>,result)} : GroupList}:
  (+sources,+values) = result
  collected_kept(sources,values,input,gradient,same)

def groups_kept(+ranges: List<&2,Grouped.Range>,+geometry: Kernel.Geometry,+image: U32,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {groups(ranges,geometry,image,input,gradient) ==
    GroupList{input,gradient,Pair.snd(Groups.Sources,List<&2,Tree.Tree<F32>>,
      Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,image}))} : GroupList}:
  collected_result(Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,image}),
    input,gradient,
    Traversals.collect_list_preserves_source(~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~Kernel.GroupContext,~group_read,~group_read_kept,ranges,Groups.Sources{input,gradient},
      Kernel.GroupContext{geometry,image}))

def image_kept(+geometry: Kernel.Geometry,+index: U32,+ranges: List<&2,Grouped.Range>,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {image(geometry,index,ranges,input,gradient) == Groups.Group{input,gradient,Groups.group_output(image(geometry,index,ranges,input,gradient))} : Groups.Group}:
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  +fuel = List.length(&2,Grouped.Range,ranges)
  +count = (o * taps : U32)
  %Equal.sym(GroupList,groups(ranges,geometry,index,input,gradient),
    GroupList{input,gradient,Pair.snd(Groups.Sources,List<&2,Tree.Tree<F32>>,
      Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,Groups.Sources{input,gradient},Kernel.GroupContext{geometry,index}))},
    groups_kept(ranges,geometry,index,input,gradient)) :
    {reduced(fuel,count,_) == Groups.Group{input,gradient,Groups.group_output(reduced(fuel,count,_))} : Groups.Group}
  {==}

def image_read_kept(+source: Groups.Sources,+context: Kernel.ImageContext) ->
  {Pair.fst(Groups.Sources,Traversal.Collected<&2,Kernel.ImageContext,Tree.Tree<F32>>,image_read(source,context)) == source : Groups.Sources}:
  Groups.Sources{+input,+gradient} = source
  Kernel.ImageContext{+geometry,+index,+rows,+columns,+group} = context
  +ranges = Grouped.ranges(rows,columns,group)
  %Equal.sym(Groups.Group,image(geometry,index,ranges,input,gradient),Groups.Group{input,gradient,Groups.group_output(image(geometry,index,ranges,input,gradient))},
    image_kept(geometry,index,ranges,input,gradient)) :
    {Pair.fst(Groups.Sources,Traversal.Collected<&2,Kernel.ImageContext,Tree.Tree<F32>>,image_collected(Kernel.ImageContext{geometry,(index + 1 : U32),rows,columns,group},_)) ==
      Groups.Sources{input,gradient} : Groups.Sources}
  {==}

def images_kept(+remaining: Nat,+index: U32,+geometry: Kernel.Geometry,+rows: Nat,+columns: Nat,+group: Nat,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {images(remaining,index,geometry,rows,columns,group,input,gradient) ==
    GroupList{input,gradient,Pair.snd(Groups.Sources,List<&2,Tree.Tree<F32>>,
      Traversal.collect(~&2,~Groups.Sources,~Tree.Tree<F32>,~Kernel.ImageContext,~image_read,remaining,Groups.Sources{input,gradient},Kernel.ImageContext{geometry,index,rows,columns,group}))} : GroupList}:
  collected_result(Traversal.collect(~&2,~Groups.Sources,~Tree.Tree<F32>,~Kernel.ImageContext,~image_read,remaining,Groups.Sources{input,gradient},Kernel.ImageContext{geometry,index,rows,columns,group}),
    input,gradient,
    Traversals.collect_preserves_source(~Groups.Sources,~Tree.Tree<F32>,~Kernel.ImageContext,~image_read,~image_read_kept,remaining,Groups.Sources{input,gradient},
      Kernel.ImageContext{geometry,index,rows,columns,group}))

# The collection only calls the already certified group leaf. Numeric
# observations and storage/source frames use the common collection theorem.
type GroupQuery is Data:
  GroupQuery{geometry: Kernel.Geometry,domain: Groups.GeometryReady<geometry>,size: U32,image: U32,
    input: Buffer.Model<F32>,gradient: Buffer.Model<F32>,tap: U32,output: U32,
    tap_inside: {U32.is_lt(tap,Packing.geometry_taps(geometry)) == True{} : Bool},
    output_inside: {U32.is_lt(output,Groups.geometry_outputs(geometry)) == True{} : Bool}}

def group_query_context(query: GroupQuery) -> Kernel.GroupContext:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  Kernel.GroupContext{geometry,image}

def group_query_source(query: GroupQuery) -> Groups.Sources:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  Groups.Sources{input,gradient}

def group_query_sum(+query: GroupQuery) -> SumQuery:
  GroupQuery{+geometry,+domain,size,image,input,gradient,+tap,+output,+tap_inside,+output_inside} = query
  Groups.GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned} = domain
  +plan = Groups.geometry_plan(geometry,domain)
  plan_sum_query(plan,Nat.add(Nat.mul(U32.to_nat(tap),U32.to_nat(Groups.geometry_outputs(geometry))),U32.to_nat(output)),Groups.plan_query_inside(plan,tap,output,tap_inside,output_inside))

def group_query_frame(query: GroupQuery,source: Groups.Sources) -> Data:
  {source == group_query_source(query) : Groups.Sources}

def group_query_value(query: GroupQuery,tree: Tree.Tree<F32>) -> Data:
  sum_query_frame(group_query_sum(query),tree)

def group_query_observe(query: GroupQuery,tree: Tree.Tree<F32>) -> F32:
  sum_query_read(group_query_sum(query),tree)

def group_query_key(query: GroupQuery,range: Grouped.Range) -> Data:
  GroupQuery{+geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  GroupLaws.RangeBounded(U32.to_nat(Groups.geometry_rows(geometry)),U32.to_nat(Groups.geometry_columns(geometry)),U32.to_nat(size),range)

def group_query_sample(query: GroupQuery,range: Grouped.Range) -> F32:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  Grouped.Range{first,count} = range
  Groups.numeric_dot(Groups.NumericSource{input,gradient,geometry,image},tap,output,count,first,0.0)

def group_query_group(query: GroupQuery,range: Grouped.Range) -> Groups.Group:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  Groups.group(geometry,image,range,input,gradient)

def group_query_numeric(+query: GroupQuery,+range: Grouped.Range,+evidence: group_query_key(query,range)) ->
  {group_query_observe(query,Groups.group_output(group_query_group(query,range))) == group_query_sample(query,range) : F32}:
  GroupQuery{+geometry,+domain,+size,+image,+input,+gradient,+tap,+output,+tap_inside,+output_inside} = query
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  Groups.GeometryReady{+limits,+columns_positive,+kernel_positive,+stride_positive,+square,+taps_positive,+outputs_positive,+count,+aligned} = domain
  Grouped.Range{+first,+remaining} = range
  Shape.Both{+nonempty,Shape.Both{+bounded,+room}} = evidence
  %Encoded.encoded_madd(U32.to_nat(tap),U32.to_nat(o),U32.to_nat(output)) :
    {Tree.read(F32,Groups.group_output(Groups.group(geometry,image,range,input,gradient)),_) == group_query_sample(query,range) : F32}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(tap)),tap,Arithmetic.native_roundtrip(tap)) :
    {Tree.read(F32,Groups.group_output(Groups.group(geometry,image,range,input,gradient)),(_ * U32.from_nat(U32.to_nat(o)) + U32.from_nat(U32.to_nat(output)) : U32)) == group_query_sample(query,range) : F32}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(o)),o,Arithmetic.native_roundtrip(o)) :
    {Tree.read(F32,Groups.group_output(Groups.group(geometry,image,range,input,gradient)),(tap * _ + U32.from_nat(U32.to_nat(output)) : U32)) == group_query_sample(query,range) : F32}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(output)),output,Arithmetic.native_roundtrip(output)) :
    {Tree.read(F32,Groups.group_output(Groups.group(geometry,image,range,input,gradient)),(tap * o + _ : U32)) == group_query_sample(query,range) : F32}
  %FullRange.roundtrip(remaining,size,bounded) :
    {Tree.read(F32,Groups.group_output(Groups.group(geometry,image,range,input,gradient)),(tap * o + output : U32)) ==
      Groups.numeric_dot(Groups.NumericSource{input,gradient,geometry,image},tap,output,_,first,0.0) : F32}
  Groups.range_group_numeric(geometry,domain,image,range,size,input,gradient,evidence,tap,output,tap_inside,output_inside)

def group_query_storage(+query: GroupQuery,+range: Grouped.Range) ->
  Shape.Both<group_query_value(query,Groups.group_output(group_query_group(query,range))),
    {Groups.group_sources(group_query_group(query,range)) == group_query_source(query) : Groups.Sources}>:
  GroupQuery{+geometry,+domain,size,+image,+input,+gradient,tap,output,tap_inside,output_inside} = query
  Groups.GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned} = domain
  Groups.group_frame(geometry,domain,image,range,input,gradient)

def group_query_split_frame(+query: GroupQuery,+range: Grouped.Range,result: Groups.Group,
  frame: Shape.Both<group_query_value(query,Groups.group_output(result)),{Groups.group_sources(result) == group_query_source(query) : Groups.Sources}>,
  numeric: {group_query_observe(query,Groups.group_output(result)) == group_query_sample(query,range) : F32}) ->
  Traversals.collected_leaf_frame(~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~GroupQuery,~F32,
    ~group_query_frame,~group_query_value,~group_query_observe,~group_query_sample,query,range,group_split(result)):
  Groups.Group{input,gradient,values} = result
  Shape.Both{values_frame,sources_equal} = frame
  Shape.Both{sources_equal,Shape.Both{values_frame,numeric}}

def group_query_leaf(+source: Groups.Sources,+query: GroupQuery,+range: Grouped.Range,
  source_frame: group_query_frame(query,source),key_frame: group_query_key(query,range)) ->
  Traversals.collected_leaf_frame(~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~GroupQuery,~F32,
    ~group_query_frame,~group_query_value,~group_query_observe,~group_query_sample,query,range,group_read(source,group_query_context(query),range)):
  GroupQuery{+geometry,+domain,+size,+image,+input,+gradient,+tap,+output,+tap_inside,+output_inside} = query
  %Equal.sym(Groups.Sources,source,group_query_source(query),source_frame) :
    Traversals.collected_leaf_frame(~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~GroupQuery,~F32,
      ~group_query_frame,~group_query_value,~group_query_observe,~group_query_sample,query,range,group_read(_,group_query_context(query),range))
  group_query_split_frame(query,range,group_query_group(query,range),group_query_storage(query,range),group_query_numeric(query,range,key_frame))

def group_query_collect(+query: GroupQuery,+ranges: List<&2,Grouped.Range>,
  evidence: Members.All(Grouped.Range,{range => group_query_key(query,range) : Grouped.Range -> Data},ranges)) ->
  Traversals.collected_list_frame(~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~GroupQuery,~F32,
    ~group_query_frame,~group_query_value,~group_query_observe,~group_query_sample,query,ranges,
    Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,group_query_source(query),group_query_context(query))):
  Traversals.collect_list_certified(~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~GroupQuery,~F32,~Kernel.GroupContext,~group_query_context,
    ~group_query_frame,~group_query_value,~group_query_key,~group_query_observe,~group_query_sample,~group_read,~group_query_leaf,ranges,group_query_source(query),query,{==},evidence)

def group_query_tree(+fuel: Nat,+query: GroupQuery,+values: List<&2,Tree.Tree<F32>>,
  evidence: Members.All(Tree.Tree<F32>,{tree => group_query_value(query,tree) : Tree.Tree<F32> -> Data},values)) ->
  Shape.Both<group_query_value(query,Grouped.levels(~&2,~Tree.Tree<F32>,~U32,~zero,~{count => left => right => sum(count,left,right) : U32 -> Tree.Tree<F32> -> Tree.Tree<F32> -> Tree.Tree<F32>},fuel,sum_query_count(group_query_sum(query)),values)),
    {group_query_observe(query,Grouped.levels(~&2,~Tree.Tree<F32>,~U32,~zero,~{count => left => right => sum(count,left,right) : U32 -> Tree.Tree<F32> -> Tree.Tree<F32> -> Tree.Tree<F32>},fuel,sum_query_count(group_query_sum(query)),values)) ==
      Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},GroupLaws.observed_values(~Tree.Tree<F32>,~F32,~GroupQuery,~group_query_observe,values,query)) : F32}>:
  GroupLaws.levels_projected(~Tree.Tree<F32>,~F32,~GroupQuery,~U32,~{query => sum_query_count(group_query_sum(query)) : GroupQuery -> U32},~Unit,~(query => Unit{}),
    ~zero,~{context => 0.0 : Unit -> F32},~{count => left => right => sum(count,left,right) : U32 -> Tree.Tree<F32> -> Tree.Tree<F32> -> Tree.Tree<F32>},~{context => F32.add : Unit -> F32 -> F32 -> F32},
    ~group_query_observe,~group_query_value,
    ~{query => sum_zero_frame(group_query_sum(query)) : @+query: GroupQuery -> group_query_value(query,zero(sum_query_count(group_query_sum(query))))},
    ~{query => sum_zero_numeric(group_query_sum(query)) : @+query: GroupQuery -> {group_query_observe(query,zero(sum_query_count(group_query_sum(query)))) == 0.0 : F32}},
    ~{query => left => right => left_frame => right_frame => sum_combine_frame(group_query_sum(query),left,right,left_frame,right_frame) :
      @+query: GroupQuery -> @+left: Tree.Tree<F32> -> @+right: Tree.Tree<F32> -> group_query_value(query,left) -> group_query_value(query,right) -> group_query_value(query,sum(sum_query_count(group_query_sum(query)),left,right))},
    ~{query => left => right => left_frame => right_frame => sum_combine_numeric(group_query_sum(query),left,right,left_frame,right_frame) :
      @+query: GroupQuery -> @+left: Tree.Tree<F32> -> @+right: Tree.Tree<F32> -> group_query_value(query,left) -> group_query_value(query,right) ->
        {group_query_observe(query,sum(sum_query_count(group_query_sum(query)),left,right)) == (group_query_observe(query,left) + group_query_observe(query,right) : F32) : F32}},fuel,query,values,evidence)

def group_query_reduced(+query: GroupQuery,+ranges: List<&2,Grouped.Range>,result: Groups.Sources & List<&2,Tree.Tree<F32>>,
  evidence: Traversals.collected_list_frame(~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~GroupQuery,~F32,
    ~group_query_frame,~group_query_value,~group_query_observe,~group_query_sample,query,ranges,result)) ->
  Shape.Both<group_query_frame(query,Groups.group_sources(reduced(List.length(&2,Grouped.Range,ranges),sum_query_count(group_query_sum(query)),collected_groups(result)))),
    Shape.Both<group_query_value(query,Groups.group_output(reduced(List.length(&2,Grouped.Range,ranges),sum_query_count(group_query_sum(query)),collected_groups(result)))),
      {group_query_observe(query,Groups.group_output(reduced(List.length(&2,Grouped.Range,ranges),sum_query_count(group_query_sum(query)),collected_groups(result)))) ==
        Grouped.tree(Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~GroupQuery,~group_query_sample,ranges,query)) : F32}>>:
  (Groups.Sources{input,gradient},+values) = result
  Shape.Both{source_frame,Shape.Both{+values_frame,+numeric}} = evidence
  +count = sum_query_count(group_query_sum(query))
  +fuel = List.length(&2,Grouped.Range,ranges)
  +value = Grouped.levels(~&2,~Tree.Tree<F32>,~U32,~zero,~{count => left => right => sum(count,left,right) : U32 -> Tree.Tree<F32> -> Tree.Tree<F32> -> Tree.Tree<F32>},fuel,count,values)
  +model = Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~GroupQuery,~group_query_sample,ranges,query)
  +tree_proof = group_query_tree(fuel,query,values,values_frame)
  value_frame = Shape.first(group_query_value(query,value),
    {group_query_observe(query,value) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},GroupLaws.observed_values(~Tree.Tree<F32>,~F32,~GroupQuery,~group_query_observe,values,query)) : F32},tree_proof)
  +value_equal = Shape.second(group_query_value(query,value),
    {group_query_observe(query,value) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},GroupLaws.observed_values(~Tree.Tree<F32>,~F32,~GroupQuery,~group_query_observe,values,query)) : F32},tree_proof)
  Shape.Both{source_frame,Shape.Both{value_frame,
    %Equal.sym(Nat,List.length(&2,F32,model),fuel,Traversals.collect_values_context_length(~&2,~F32,~Grouped.Range,~GroupQuery,~group_query_sample,ranges,query)) :
      {group_query_observe(query,value) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),_,Unit{},model) : F32}
    %numeric : {group_query_observe(query,value) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},_) : F32}
    value_equal}}

def query_image(query: GroupQuery,ranges: List<&2,Grouped.Range>) -> Groups.Group:
  GroupQuery{geometry,domain,size,index,input,gradient,tap,output,tap_inside,output_inside} = query
  image(geometry,index,ranges,input,gradient)

def group_query_image(+query: GroupQuery,+ranges: List<&2,Grouped.Range>,
  evidence: Members.All(Grouped.Range,{range => group_query_key(query,range) : Grouped.Range -> Data},ranges)) ->
  Shape.Both<group_query_frame(query,Groups.group_sources(query_image(query,ranges))),
    Shape.Both<group_query_value(query,Groups.group_output(query_image(query,ranges))),
      {group_query_observe(query,Groups.group_output(query_image(query,ranges))) ==
        Grouped.tree(Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~GroupQuery,~group_query_sample,ranges,query)) : F32}>>:
  GroupQuery{+geometry,+domain,size,+index,+input,+gradient,tap,output,tap_inside,output_inside} = query
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  Groups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,count,aligned} = domain
  result = Traversal.collect_list(~&2,~Groups.Sources,~Tree.Tree<F32>,~Grouped.Range,~Kernel.GroupContext,~group_read,ranges,group_query_source(query),group_query_context(query))
  %Equal.sym(U32,(o * taps : U32),(taps * o : U32),Arithmetic.multiply_commutative(o,taps)) :
    Shape.Both<group_query_frame(query,Groups.group_sources(reduced(List.length(&2,Grouped.Range,ranges),_,collected_groups(result)))),
      Shape.Both<group_query_value(query,Groups.group_output(reduced(List.length(&2,Grouped.Range,ranges),_,collected_groups(result)))),
        {group_query_observe(query,Groups.group_output(reduced(List.length(&2,Grouped.Range,ranges),_,collected_groups(result)))) ==
          Grouped.tree(Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~GroupQuery,~group_query_sample,ranges,query)) : F32}>>
  group_query_reduced(query,ranges,result,group_query_collect(query,ranges,evidence))

def group_query_at(query: GroupQuery,index: U32) -> GroupQuery:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  GroupQuery{geometry,domain,size,index,input,gradient,tap,output,tap_inside,output_inside}

type BatchQuery is Data:
  BatchQuery{query: GroupQuery,rows: Nat,columns: Nat,size: Nat,images: U32,
    ranges_frame: Members.All(Grouped.Range,{range => group_query_key(query,range) : Grouped.Range -> Data},Grouped.ranges(rows,columns,size))}

def batch_query_group(query: BatchQuery) -> GroupQuery:
  BatchQuery{group,rows,columns,size,images,ranges_frame} = query
  group

def batch_query_context(query: BatchQuery,index: Nat) -> Kernel.ImageContext:
  BatchQuery{GroupQuery{geometry,domain,group_size,image,input,gradient,tap,output,tap_inside,output_inside},rows,columns,size,images,ranges_frame} = query
  Kernel.ImageContext{geometry,U32.from_nat(index),rows,columns,size}

def batch_query_frame(query: BatchQuery,source: Groups.Sources) -> Data:
  group_query_frame(batch_query_group(query),source)

def batch_query_value(query: BatchQuery,value: Tree.Tree<F32>) -> Data:
  group_query_value(batch_query_group(query),value)

def batch_query_observe(query: BatchQuery,value: Tree.Tree<F32>) -> F32:
  group_query_observe(batch_query_group(query),value)

def batch_query_sample(query: BatchQuery,index: Nat) -> F32:
  BatchQuery{group,rows,columns,size,images,ranges_frame} = query
  Grouped.tree(Traversals.collect_values_context(~&2,~F32,~Grouped.Range,~GroupQuery,~group_query_sample,Grouped.ranges(rows,columns,size),group_query_at(group,U32.from_nat(index))))

def batch_query_size(query: BatchQuery) -> Nat:
  BatchQuery{group,rows,columns,size,images,ranges_frame} = query
  U32.to_nat(images)

def batch_query_key(query: BatchQuery,index: Nat) -> Data:
  {Nat.is_lt(index,batch_query_size(query)) == True{} : Bool}

def batch_query_next_context(query: BatchQuery,index: Nat) -> Kernel.ImageContext:
  BatchQuery{GroupQuery{geometry,domain,group_size,image,input,gradient,tap,output,tap_inside,output_inside},rows,columns,size,images,ranges_frame} = query
  Kernel.ImageContext{geometry,(U32.from_nat(index) + 1 : U32),rows,columns,size}

def batch_query_leaf_result(+query: BatchQuery,+index: Nat,result: Groups.Group,
  evidence: Shape.Both<batch_query_frame(query,Groups.group_sources(result)),
    Shape.Both<batch_query_value(query,Groups.group_output(result)),
      {batch_query_observe(query,Groups.group_output(result)) == batch_query_sample(query,index) : F32}>>) ->
  Traversals.collected_indexed_leaf_frame(~Groups.Sources,~Tree.Tree<F32>,~BatchQuery,~F32,~Kernel.ImageContext,~batch_query_context,
    ~batch_query_frame,~batch_query_value,~batch_query_observe,~batch_query_sample,query,index,
    image_collected(batch_query_next_context(query,index),result)):
  BatchQuery{GroupQuery{+geometry,domain,group_size,image,input,gradient,tap,output,tap_inside,output_inside},+rows,+columns,+size,images,ranges_frame} = query
  Groups.Group{result_input,result_gradient,result_values} = result
  Shape.Both{source_frame,Shape.Both{value_frame,numeric}} = evidence
  Shape.Both{Equal.cong(U32,Kernel.ImageContext,value => Kernel.ImageContext{geometry,value,rows,columns,size},(U32.from_nat(index) + 1 : U32),U32.from_nat(1n+index),Bounds.increment_encoded(index)),
    Shape.Both{source_frame,Shape.Both{value_frame,numeric}}}

def batch_query_leaf(+source: Groups.Sources,+query: BatchQuery,+index: Nat,
  source_frame: batch_query_frame(query,source),key_frame: batch_query_key(query,index)) ->
  Traversals.collected_indexed_leaf_frame(~Groups.Sources,~Tree.Tree<F32>,~BatchQuery,~F32,~Kernel.ImageContext,~batch_query_context,
    ~batch_query_frame,~batch_query_value,~batch_query_observe,~batch_query_sample,query,index,image_read(source,batch_query_context(query,index))):
  BatchQuery{+group,+range_rows,+range_columns,+range_size,images,+ranges_frame} = query
  GroupQuery{+geometry,+domain,+size,image,+input,+gradient,+tap,+output,+tap_inside,+output_inside} = group
  Groups.GeometryReady{limits,columns,kernel,stride,square,taps,outputs,count,aligned} = domain
  +ranges = Grouped.ranges(range_rows,range_columns,range_size)
  %Equal.sym(Groups.Sources,source,group_query_source(group),source_frame) :
    Traversals.collected_indexed_leaf_frame(~Groups.Sources,~Tree.Tree<F32>,~BatchQuery,~F32,~Kernel.ImageContext,~batch_query_context,
      ~batch_query_frame,~batch_query_value,~batch_query_observe,~batch_query_sample,query,index,image_read(_,batch_query_context(query,index)))
  batch_query_leaf_result(query,index,query_image(group_query_at(group,U32.from_nat(index)),ranges),group_query_image(group_query_at(group,U32.from_nat(index)),ranges,ranges_frame))

def batch_query_collect(+query: BatchQuery) ->
  Traversals.collected_list_frame(~Groups.Sources,~Tree.Tree<F32>,~Nat,~BatchQuery,~F32,
    ~batch_query_frame,~batch_query_value,~batch_query_observe,~batch_query_sample,query,Ranges.range(batch_query_size(query),0n),
    Traversal.collect(~&2,~Groups.Sources,~Tree.Tree<F32>,~Kernel.ImageContext,~image_read,batch_query_size(query),group_query_source(batch_query_group(query)),batch_query_context(query,0n))):
  Traversals.collect_certified(~Groups.Sources,~Tree.Tree<F32>,~BatchQuery,~F32,~Kernel.ImageContext,~batch_query_context,
    ~batch_query_frame,~batch_query_value,~batch_query_key,~batch_query_observe,~batch_query_sample,~image_read,~batch_query_leaf,
    batch_query_size(query),0n,group_query_source(batch_query_group(query)),query,{==},
    Members.range_all(~BatchQuery,~{query => index => batch_query_key(query,index) : @+query: BatchQuery -> @+index: Nat -> Data},~batch_query_size,
      ~{query => index => inside => inside : @+query: BatchQuery -> @+index: Nat -> @+inside: {Nat.is_lt(index,batch_query_size(query)) == True{} : Bool} -> batch_query_key(query,index)},
      batch_query_size(query),0n,query,Order.reflexive(batch_query_size(query))))

def batch_query_tree(+fuel: Nat,+query: BatchQuery,+values: List<&2,Tree.Tree<F32>>,
  evidence: Members.All(Tree.Tree<F32>,{tree => batch_query_value(query,tree) : Tree.Tree<F32> -> Data},values)) ->
  Shape.Both<batch_query_value(query,Grouped.levels(~&2,~Tree.Tree<F32>,~U32,~zero,~{count => left => right => sum(count,left,right) : U32 -> Tree.Tree<F32> -> Tree.Tree<F32> -> Tree.Tree<F32>},fuel,sum_query_count(group_query_sum(batch_query_group(query))),values)),
    {batch_query_observe(query,Grouped.levels(~&2,~Tree.Tree<F32>,~U32,~zero,~{count => left => right => sum(count,left,right) : U32 -> Tree.Tree<F32> -> Tree.Tree<F32> -> Tree.Tree<F32>},fuel,sum_query_count(group_query_sum(batch_query_group(query))),values)) ==
      Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},GroupLaws.observed_values(~Tree.Tree<F32>,~F32,~BatchQuery,~batch_query_observe,values,query)) : F32}>:
  GroupLaws.levels_projected(~Tree.Tree<F32>,~F32,~BatchQuery,~U32,~{query => sum_query_count(group_query_sum(batch_query_group(query))) : BatchQuery -> U32},~Unit,~(query => Unit{}),
    ~zero,~{context => 0.0 : Unit -> F32},~{count => left => right => sum(count,left,right) : U32 -> Tree.Tree<F32> -> Tree.Tree<F32> -> Tree.Tree<F32>},~{context => F32.add : Unit -> F32 -> F32 -> F32},
    ~batch_query_observe,~batch_query_value,
    ~{query => sum_zero_frame(group_query_sum(batch_query_group(query))) : @+query: BatchQuery -> batch_query_value(query,zero(sum_query_count(group_query_sum(batch_query_group(query)))))},
    ~{query => sum_zero_numeric(group_query_sum(batch_query_group(query))) : @+query: BatchQuery -> {batch_query_observe(query,zero(sum_query_count(group_query_sum(batch_query_group(query))))) == 0.0 : F32}},
    ~{query => left => right => left_frame => right_frame => sum_combine_frame(group_query_sum(batch_query_group(query)),left,right,left_frame,right_frame) :
      @+query: BatchQuery -> @+left: Tree.Tree<F32> -> @+right: Tree.Tree<F32> -> batch_query_value(query,left) -> batch_query_value(query,right) -> batch_query_value(query,sum(sum_query_count(group_query_sum(batch_query_group(query))),left,right))},
    ~{query => left => right => left_frame => right_frame => sum_combine_numeric(group_query_sum(batch_query_group(query)),left,right,left_frame,right_frame) :
      @+query: BatchQuery -> @+left: Tree.Tree<F32> -> @+right: Tree.Tree<F32> -> batch_query_value(query,left) -> batch_query_value(query,right) ->
        {batch_query_observe(query,sum(sum_query_count(group_query_sum(batch_query_group(query))),left,right)) == (batch_query_observe(query,left) + batch_query_observe(query,right) : F32) : F32}},fuel,query,values,evidence)

def batch_query_reduced(+query: BatchQuery,result: Groups.Sources & List<&2,Tree.Tree<F32>>,
  evidence: Traversals.collected_list_frame(~Groups.Sources,~Tree.Tree<F32>,~Nat,~BatchQuery,~F32,
    ~batch_query_frame,~batch_query_value,~batch_query_observe,~batch_query_sample,query,Ranges.range(batch_query_size(query),0n),result)) ->
  Shape.Both<batch_query_frame(query,Groups.group_sources(reduced(batch_query_size(query),sum_query_count(group_query_sum(batch_query_group(query))),collected_groups(result)))),
    Shape.Both<batch_query_value(query,Groups.group_output(reduced(batch_query_size(query),sum_query_count(group_query_sum(batch_query_group(query))),collected_groups(result)))),
      {batch_query_observe(query,Groups.group_output(reduced(batch_query_size(query),sum_query_count(group_query_sum(batch_query_group(query))),collected_groups(result)))) ==
        Grouped.tree(ListRead.enumerate(~F32,~BatchQuery,~batch_query_sample,batch_query_size(query),0n,query)) : F32}>>:
  (Groups.Sources{input,gradient},+values) = result
  Shape.Both{source_frame,Shape.Both{+values_frame,+numeric}} = evidence
  +count = sum_query_count(group_query_sum(batch_query_group(query)))
  +fuel = batch_query_size(query)
  +value = Grouped.levels(~&2,~Tree.Tree<F32>,~U32,~zero,~{count => left => right => sum(count,left,right) : U32 -> Tree.Tree<F32> -> Tree.Tree<F32> -> Tree.Tree<F32>},fuel,count,values)
  +model = ListRead.enumerate(~F32,~BatchQuery,~batch_query_sample,fuel,0n,query)
  +tree_proof = batch_query_tree(fuel,query,values,values_frame)
  value_frame = Shape.first(batch_query_value(query,value),
    {batch_query_observe(query,value) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},GroupLaws.observed_values(~Tree.Tree<F32>,~F32,~BatchQuery,~batch_query_observe,values,query)) : F32},tree_proof)
  +value_equal = Shape.second(batch_query_value(query,value),
    {batch_query_observe(query,value) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},GroupLaws.observed_values(~Tree.Tree<F32>,~F32,~BatchQuery,~batch_query_observe,values,query)) : F32},tree_proof)
  Shape.Both{source_frame,Shape.Both{value_frame,
    %Equal.sym(Nat,List.length(&2,F32,model),fuel,ListRead.enumerate_length(~F32,~BatchQuery,~batch_query_sample,fuel,0n,query)) :
      {batch_query_observe(query,value) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),_,Unit{},model) : F32}
    %Traversals.collect_values_context_range(~F32,~BatchQuery,~batch_query_sample,fuel,0n,query) :
      {batch_query_observe(query,value) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},_) : F32}
    %numeric : {batch_query_observe(query,value) == Grouped.levels(~&2,~F32,~Unit,~(context => 0.0),~(context => F32.add),fuel,Unit{},_) : F32}
    value_equal}}

def query_batch(query: BatchQuery) -> Groups.Group:
  BatchQuery{GroupQuery{+geometry,domain,group_size,image,input,gradient,tap,output,tap_inside,output_inside},range_rows,range_columns,range_size,+count,ranges_frame} = query
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  reduced(U32.to_nat(count),(o * taps : U32),images(U32.to_nat(count),0,geometry,range_rows,range_columns,range_size,input,gradient))

def batch_query_numeric(+query: BatchQuery) ->
  Shape.Both<batch_query_frame(query,Groups.group_sources(query_batch(query))),
    Shape.Both<batch_query_value(query,Groups.group_output(query_batch(query))),
      {batch_query_observe(query,Groups.group_output(query_batch(query))) ==
        Grouped.tree(ListRead.enumerate(~F32,~BatchQuery,~batch_query_sample,batch_query_size(query),0n,query)) : F32}>>:
  BatchQuery{+group,range_rows,range_columns,range_size,+count,ranges_frame} = query
  GroupQuery{+geometry,+domain,size,image,+input,+gradient,tap,output,tap_inside,output_inside} = group
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  Groups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,bound,aligned} = domain
  result = Traversal.collect(~&2,~Groups.Sources,~Tree.Tree<F32>,~Kernel.ImageContext,~image_read,U32.to_nat(count),group_query_source(group),batch_query_context(query,0n))
  %Equal.sym(U32,(o * taps : U32),(taps * o : U32),Arithmetic.multiply_commutative(o,taps)) :
    Shape.Both<batch_query_frame(query,Groups.group_sources(reduced(U32.to_nat(count),_,collected_groups(result)))),
      Shape.Both<batch_query_value(query,Groups.group_output(reduced(U32.to_nat(count),_,collected_groups(result)))),
        {batch_query_observe(query,Groups.group_output(reduced(U32.to_nat(count),_,collected_groups(result)))) ==
          Grouped.tree(ListRead.enumerate(~F32,~BatchQuery,~batch_query_sample,U32.to_nat(count),0n,query)) : F32}>>
  batch_query_reduced(query,result,batch_query_collect(query))

def group_query_size(query: GroupQuery) -> U32:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  size

def native_batch_query(+query: GroupQuery,count: U32,+size_positive: {Nat.is_lt(0n,U32.to_nat(group_query_size(query))) == True{} : Bool}) -> BatchQuery:
  GroupQuery{+geometry,+domain,+size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  Kernel.Geometry{c,o,h,w,k,s,p,+rows,+columns,taps} = geometry
  Groups.GeometryReady{limits,+columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,bound,aligned} = domain
  BatchQuery{query,U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(size),count,
    GroupLaws.ranges_bounded(U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(size),columns_positive,size_positive)}

type RowQuery is Data:
  RowQuery{input: Tree.Tree<F32>,count: Nat,address: U32,stride: U32}

def row_query_tree(query: RowQuery) -> Tree.Tree<F32>:
  RowQuery{input,count,address,stride} = query
  input

def row_query_size(query: RowQuery) -> Nat:
  RowQuery{input,count,address,stride} = query
  count

def row_query_address(query: RowQuery,index: Nat) -> U32:
  RowQuery{input,count,address,stride} = query
  U32.from_nat(Nat.add(U32.to_nat(address),Nat.mul(index,U32.to_nat(stride))))

def row_query_context(query: RowQuery,index: Nat) -> RowContext:
  RowQuery{input,count,+address,+stride} = query
  RowContext{U32.from_nat(Nat.add(U32.to_nat(address),Nat.mul(index,U32.to_nat(stride)))),stride}

def row_query_frame(query: RowQuery,source: Tree.Tree<F32>) -> Data:
  {source == row_query_tree(query) : Tree.Tree<F32>}

def row_query_key(query: RowQuery,index: Nat) -> Data:
  {Nat.is_lt(index,row_query_size(query)) == True{} : Bool}

def row_query_sample(query: RowQuery,index: Nat) -> F32:
  RowQuery{input,count,address,stride} = query
  Tree.read(F32,input,U32.from_nat(Nat.add(U32.to_nat(address),Nat.mul(index,U32.to_nat(stride)))))

def row_query_leaf(+source: Tree.Tree<F32>,+query: RowQuery,+index: Nat,
  source_frame: row_query_frame(query,source),key_frame: row_query_key(query,index)) ->
  Traversals.collected_indexed_leaf_frame(~Tree.Tree<F32>,~F32,~RowQuery,~F32,~RowContext,~row_query_context,
    ~row_query_frame,~{query => value => Unit : RowQuery -> F32 -> Data},~Traversals.collect_identity(~F32,~RowQuery),~row_query_sample,
    query,index,row_read(source,row_query_context(query,index))):
  RowQuery{+input,count,+address,+stride} = query
  +selected = row_query_address(query,index)
  %Equal.sym(Tree.Tree<F32>,source,input,source_frame) :
    Traversals.collected_indexed_leaf_frame(~Tree.Tree<F32>,~F32,~RowQuery,~F32,~RowContext,~row_query_context,
      ~row_query_frame,~{query => value => Unit : RowQuery -> F32 -> Data},~Traversals.collect_identity(~F32,~RowQuery),~row_query_sample,
      query,index,row_read(_,row_query_context(query,index)))
  Shape.Both{Equal.cong(U32,RowContext,value => RowContext{value,stride},(selected + stride : U32),row_query_address(query,1n+index),Encoded.stride_next(address,stride,index)),
    Shape.Both{{==},Shape.Both{Unit{},{==}}}}

def row_query_collect(+query: RowQuery) ->
  Traversals.collected_list_frame(~Tree.Tree<F32>,~F32,~Nat,~RowQuery,~F32,
    ~row_query_frame,~{query => value => Unit : RowQuery -> F32 -> Data},~Traversals.collect_identity(~F32,~RowQuery),~row_query_sample,
    query,Ranges.range(row_query_size(query),0n),Traversal.collect(~&2,~Tree.Tree<F32>,~F32,~RowContext,~row_read,row_query_size(query),row_query_tree(query),row_query_context(query,0n))):
  Traversals.collect_certified(~Tree.Tree<F32>,~F32,~RowQuery,~F32,~RowContext,~row_query_context,
    ~row_query_frame,~{query => value => Unit : RowQuery -> F32 -> Data},~row_query_key,~Traversals.collect_identity(~F32,~RowQuery),~row_query_sample,~row_read,~row_query_leaf,
    row_query_size(query),0n,row_query_tree(query),query,{==},
    Members.range_all(~RowQuery,~{query => index => row_query_key(query,index) : @+query: RowQuery -> @+index: Nat -> Data},~row_query_size,
      ~{query => index => inside => inside : @+query: RowQuery -> @+index: Nat -> @+inside: {Nat.is_lt(index,row_query_size(query)) == True{} : Bool} -> row_query_key(query,index)},
      row_query_size(query),0n,query,Order.reflexive(row_query_size(query))))

def row_query_projection(+query: RowQuery,result: Tree.Tree<F32> & List<&2,F32>,
  evidence: Traversals.collected_list_frame(~Tree.Tree<F32>,~F32,~Nat,~RowQuery,~F32,
    ~row_query_frame,~{query => value => Unit : RowQuery -> F32 -> Data},~Traversals.collect_identity(~F32,~RowQuery),~row_query_sample,query,Ranges.range(row_query_size(query),0n),result)) ->
  {Pair.snd(Tree.Tree<F32>,List<&2,F32>,result) == ListRead.enumerate(~F32,~RowQuery,~row_query_sample,row_query_size(query),0n,query) : List<&2,F32>}:
  (source,+values) = result
  Shape.Both{source_frame,Shape.Both{values_frame,+numeric}} = evidence
  Equal.trans(List<&2,F32>,values,Traversals.collect_values_context(~&2,~F32,~F32,~RowQuery,~Traversals.collect_identity(~F32,~RowQuery),values,query),ListRead.enumerate(~F32,~RowQuery,~row_query_sample,row_query_size(query),0n,query),
    Equal.sym(List<&2,F32>,Traversals.collect_values_context(~&2,~F32,~F32,~RowQuery,~Traversals.collect_identity(~F32,~RowQuery),values,query),values,Traversals.collect_values_context_identity(~F32,~RowQuery,values,query)),
    Equal.trans(List<&2,F32>,Traversals.collect_values_context(~&2,~F32,~F32,~RowQuery,~Traversals.collect_identity(~F32,~RowQuery),values,query),
      Traversals.collect_values_context(~&2,~F32,~Nat,~RowQuery,~row_query_sample,Ranges.range(row_query_size(query),0n),query),ListRead.enumerate(~F32,~RowQuery,~row_query_sample,row_query_size(query),0n,query),numeric,
      Traversals.collect_values_context_range(~F32,~RowQuery,~row_query_sample,row_query_size(query),0n,query)))

def row_query_initial(+input: Tree.Tree<F32>,+count: Nat,+address: U32,+stride: U32) ->
  {row_query_context(RowQuery{input,count,address,stride},0n) == RowContext{address,stride} : RowContext}:
  Equal.cong(U32,RowContext,value => RowContext{value,stride},U32.from_nat(Nat.add(U32.to_nat(address),0n)),address,
    Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(address),0n)),U32.from_nat(U32.to_nat(address)),address,
      Equal.cong(Nat,U32,U32.from_nat,Nat.add(U32.to_nat(address),0n),U32.to_nat(address),Addition.zero_right(U32.to_nat(address))),Arithmetic.native_roundtrip(address)))

def row_values_enumerated(+count: Nat,+address: U32,+stride: U32,+input: Tree.Tree<F32>) ->
  {row_values(count,address,stride,input) == ListRead.enumerate(~F32,~RowQuery,~row_query_sample,count,0n,RowQuery{input,count,address,stride}) : List<&2,F32>}:
  +query = {RowQuery{input,count,address,stride} : RowQuery}
  %row_query_initial(input,count,address,stride) :
    {Pair.snd(Tree.Tree<F32>,List<&2,F32>,Traversal.collect(~&2,~Tree.Tree<F32>,~F32,~RowContext,~row_read,count,input,_)) == ListRead.enumerate(~F32,~RowQuery,~row_query_sample,count,0n,query) : List<&2,F32>}
  row_query_projection(query,Traversal.collect(~&2,~Tree.Tree<F32>,~F32,~RowContext,~row_read,count,input,row_query_context(query,0n)),row_query_collect(query))

def row_values_length(+count: Nat,+address: U32,+stride: U32,+input: Tree.Tree<F32>) -> {List.length(&2,F32,row_values(count,address,stride,input)) == count : Nat}:
  Traversals.collect_length(~Tree.Tree<F32>,~F32,~RowContext,~row_read,count,input,RowContext{address,stride})

def row_values_numeric(+count: Nat,+address: U32,+stride: U32,+input: Tree.Tree<F32>,+index: Nat,inside: {Nat.is_lt(index,count) == True{} : Bool}) ->
  {ListRead.at(~F32,0.0,index,row_values(count,address,stride,input)) == Tree.read(F32,input,U32.from_nat(Nat.add(U32.to_nat(address),Nat.mul(index,U32.to_nat(stride))))) : F32}:
  +query = {RowQuery{input,count,address,stride} : RowQuery}
  %Equal.sym(List<&2,F32>,row_values(count,address,stride,input),ListRead.enumerate(~F32,~RowQuery,~row_query_sample,count,0n,query),row_values_enumerated(count,address,stride,input)) :
    {ListRead.at(~F32,0.0,index,_) == row_query_sample(query,index) : F32}
  %Equal.sym(Nat,Nat.add(0n,index),index,{==}) : {ListRead.at(~F32,0.0,index,ListRead.enumerate(~F32,~RowQuery,~row_query_sample,count,0n,query)) == row_query_sample(query,_) : F32}
  ListRead.enumerate_at(~F32,~RowQuery,~row_query_sample,0.0,count,index,0n,query,inside)

type TransposeQuery is Data:
  TransposeQuery{input: Tree.Tree<F32>,count: Nat,output: U32,outputs: U32,taps: Nat}

def transpose_query_tree(query: TransposeQuery) -> Tree.Tree<F32>:
  TransposeQuery{input,count,output,outputs,taps} = query
  input

def transpose_query_size(query: TransposeQuery) -> Nat:
  TransposeQuery{input,count,output,outputs,taps} = query
  count

def transpose_query_width(query: TransposeQuery) -> Nat:
  TransposeQuery{input,count,output,outputs,taps} = query
  taps

def transpose_query_address(query: TransposeQuery,index: Nat) -> U32:
  TransposeQuery{input,count,output,outputs,taps} = query
  U32.from_nat(Nat.add(U32.to_nat(output),Nat.mul(index,1n)))

def transpose_query_context(query: TransposeQuery,index: Nat) -> TransposeContext:
  TransposeQuery{input,count,output,outputs,taps} = query
  TransposeContext{U32.from_nat(Nat.add(U32.to_nat(output),Nat.mul(index,1n))),outputs,taps}

def transpose_query_frame(query: TransposeQuery,source: Tree.Tree<F32>) -> Data:
  {source == transpose_query_tree(query) : Tree.Tree<F32>}

def transpose_query_key(query: TransposeQuery,index: Nat) -> Data:
  {Nat.is_lt(index,transpose_query_size(query)) == True{} : Bool}

def transpose_query_sample(query: TransposeQuery,index: Nat) -> List<&2,F32>:
  TransposeQuery{input,count,output,outputs,taps} = query
  row_values(taps,U32.from_nat(Nat.add(U32.to_nat(output),Nat.mul(index,1n))),outputs,input)

def transpose_query_leaf(+source: Tree.Tree<F32>,+query: TransposeQuery,+index: Nat,
  source_frame: transpose_query_frame(query,source),key_frame: transpose_query_key(query,index)) ->
  Traversals.collected_indexed_leaf_frame(~Tree.Tree<F32>,~List<&2,F32>,~TransposeQuery,~List<&2,F32>,~TransposeContext,~transpose_query_context,
    ~transpose_query_frame,~{query => value => Unit : TransposeQuery -> List<&2,F32> -> Data},~Traversals.collect_identity(~List<&2,F32>,~TransposeQuery),~transpose_query_sample,
    query,index,transpose_read_row(source,transpose_query_context(query,index))):
  TransposeQuery{+input,count,+output,+outputs,+taps} = query
  +selected = transpose_query_address(query,index)
  %Equal.sym(Tree.Tree<F32>,source,input,source_frame) :
    Traversals.collected_indexed_leaf_frame(~Tree.Tree<F32>,~List<&2,F32>,~TransposeQuery,~List<&2,F32>,~TransposeContext,~transpose_query_context,
      ~transpose_query_frame,~{query => value => Unit : TransposeQuery -> List<&2,F32> -> Data},~Traversals.collect_identity(~List<&2,F32>,~TransposeQuery),~transpose_query_sample,
      query,index,transpose_read_row(_,transpose_query_context(query,index)))
  Shape.Both{Equal.cong(U32,TransposeContext,value => TransposeContext{value,outputs,taps},(selected + 1 : U32),transpose_query_address(query,1n+index),Encoded.stride_next(output,1,index)),
    Shape.Both{{==},Shape.Both{Unit{},{==}}}}

def transpose_query_collect(+query: TransposeQuery) ->
  Traversals.collected_list_frame(~Tree.Tree<F32>,~List<&2,F32>,~Nat,~TransposeQuery,~List<&2,F32>,
    ~transpose_query_frame,~{query => value => Unit : TransposeQuery -> List<&2,F32> -> Data},~Traversals.collect_identity(~List<&2,F32>,~TransposeQuery),~transpose_query_sample,
    query,Ranges.range(transpose_query_size(query),0n),Traversal.collect(~&2,~Tree.Tree<F32>,~List<&2,F32>,~TransposeContext,~transpose_read_row,transpose_query_size(query),transpose_query_tree(query),transpose_query_context(query,0n))):
  Traversals.collect_certified(~Tree.Tree<F32>,~List<&2,F32>,~TransposeQuery,~List<&2,F32>,~TransposeContext,~transpose_query_context,
    ~transpose_query_frame,~{query => value => Unit : TransposeQuery -> List<&2,F32> -> Data},~transpose_query_key,~Traversals.collect_identity(~List<&2,F32>,~TransposeQuery),~transpose_query_sample,~transpose_read_row,~transpose_query_leaf,
    transpose_query_size(query),0n,transpose_query_tree(query),query,{==},
    Members.range_all(~TransposeQuery,~{query => index => transpose_query_key(query,index) : @+query: TransposeQuery -> @+index: Nat -> Data},~transpose_query_size,
      ~{query => index => inside => inside : @+query: TransposeQuery -> @+index: Nat -> @+inside: {Nat.is_lt(index,transpose_query_size(query)) == True{} : Bool} -> transpose_query_key(query,index)},
      transpose_query_size(query),0n,query,Order.reflexive(transpose_query_size(query))))

def transpose_query_projection(+query: TransposeQuery,result: Tree.Tree<F32> & List<&2,List<&2,F32>>,
  evidence: Traversals.collected_list_frame(~Tree.Tree<F32>,~List<&2,F32>,~Nat,~TransposeQuery,~List<&2,F32>,
    ~transpose_query_frame,~{query => value => Unit : TransposeQuery -> List<&2,F32> -> Data},~Traversals.collect_identity(~List<&2,F32>,~TransposeQuery),~transpose_query_sample,query,Ranges.range(transpose_query_size(query),0n),result)) ->
  {Pair.snd(Tree.Tree<F32>,List<&2,List<&2,F32>>,result) == ListRead.enumerate(~List<&2,F32>,~TransposeQuery,~transpose_query_sample,transpose_query_size(query),0n,query) : List<&2,List<&2,F32>>}:
  (source,+values) = result
  Shape.Both{source_frame,Shape.Both{values_frame,+numeric}} = evidence
  Equal.trans(List<&2,List<&2,F32>>,values,Traversals.collect_values_context(~&2,~List<&2,F32>,~List<&2,F32>,~TransposeQuery,~Traversals.collect_identity(~List<&2,F32>,~TransposeQuery),values,query),ListRead.enumerate(~List<&2,F32>,~TransposeQuery,~transpose_query_sample,transpose_query_size(query),0n,query),
    Equal.sym(List<&2,List<&2,F32>>,Traversals.collect_values_context(~&2,~List<&2,F32>,~List<&2,F32>,~TransposeQuery,~Traversals.collect_identity(~List<&2,F32>,~TransposeQuery),values,query),values,Traversals.collect_values_context_identity(~List<&2,F32>,~TransposeQuery,values,query)),
    Equal.trans(List<&2,List<&2,F32>>,Traversals.collect_values_context(~&2,~List<&2,F32>,~List<&2,F32>,~TransposeQuery,~Traversals.collect_identity(~List<&2,F32>,~TransposeQuery),values,query),
      Traversals.collect_values_context(~&2,~List<&2,F32>,~Nat,~TransposeQuery,~transpose_query_sample,Ranges.range(transpose_query_size(query),0n),query),ListRead.enumerate(~List<&2,F32>,~TransposeQuery,~transpose_query_sample,transpose_query_size(query),0n,query),numeric,
      Traversals.collect_values_context_range(~List<&2,F32>,~TransposeQuery,~transpose_query_sample,transpose_query_size(query),0n,query)))

def transpose_query_initial(+input: Tree.Tree<F32>,+count: Nat,+output: U32,+outputs: U32,+taps: Nat) ->
  {transpose_query_context(TransposeQuery{input,count,output,outputs,taps},0n) == TransposeContext{output,outputs,taps} : TransposeContext}:
  Equal.cong(U32,TransposeContext,value => TransposeContext{value,outputs,taps},U32.from_nat(Nat.add(U32.to_nat(output),0n)),output,
    Equal.trans(U32,U32.from_nat(Nat.add(U32.to_nat(output),0n)),U32.from_nat(U32.to_nat(output)),output,
      Equal.cong(Nat,U32,U32.from_nat,Nat.add(U32.to_nat(output),0n),U32.to_nat(output),Addition.zero_right(U32.to_nat(output))),Arithmetic.native_roundtrip(output)))

def transpose_rows_enumerated(+count: Nat,+output: U32,+outputs: U32,+taps: Nat,+input: Tree.Tree<F32>) ->
  {Pair.snd(Tree.Tree<F32>,List<&2,List<&2,F32>>,transpose_rows(count,output,outputs,taps,input)) ==
    ListRead.enumerate(~List<&2,F32>,~TransposeQuery,~transpose_query_sample,count,0n,TransposeQuery{input,count,output,outputs,taps}) : List<&2,List<&2,F32>>}:
  +query = {TransposeQuery{input,count,output,outputs,taps} : TransposeQuery}
  %transpose_query_initial(input,count,output,outputs,taps) :
    {Pair.snd(Tree.Tree<F32>,List<&2,List<&2,F32>>,Traversal.collect(~&2,~Tree.Tree<F32>,~List<&2,F32>,~TransposeContext,~transpose_read_row,count,input,_)) ==
      ListRead.enumerate(~List<&2,F32>,~TransposeQuery,~transpose_query_sample,count,0n,query) : List<&2,List<&2,F32>>}
  transpose_query_projection(query,Traversal.collect(~&2,~Tree.Tree<F32>,~List<&2,F32>,~TransposeContext,~transpose_read_row,count,input,transpose_query_context(query,0n)),transpose_query_collect(query))

def transpose_query_row_length(+query: TransposeQuery,+index: Nat) ->
  {List.length(&2,F32,transpose_query_sample(query,index)) == transpose_query_width(query) : Nat}:
  TransposeQuery{+input,count,output,+outputs,+taps} = query
  row_values_length(taps,transpose_query_address(query,index),outputs,input)

def transpose_length(+count: Nat,+output: U32,+outputs: U32,+taps: Nat,+input: Tree.Tree<F32>) ->
  {List.length(&2,F32,transpose(count,output,outputs,taps,input)) == Nat.mul(count,taps) : Nat}:
  %Equal.sym(List<&2,List<&2,F32>>,Pair.snd(Tree.Tree<F32>,List<&2,List<&2,F32>>,transpose_rows(count,output,outputs,taps,input)),
      ListRead.enumerate(~List<&2,F32>,~TransposeQuery,~transpose_query_sample,count,0n,TransposeQuery{input,count,output,outputs,taps}),transpose_rows_enumerated(count,output,outputs,taps,input)) :
    {List.length(&2,F32,List.concat(&2,F32,_)) == Nat.mul(count,taps) : Nat}
  ListRead.concat_enumerate_length(~F32,~TransposeQuery,~transpose_query_sample,~transpose_query_width,~transpose_query_row_length,count,0n,TransposeQuery{input,count,output,outputs,taps})

def transpose_numeric(+count: Nat,+output: U32,+outputs: U32,+taps: Nat,+input: Tree.Tree<F32>,+row_index: Nat,+tap_index: Nat,
  row_inside: {Nat.is_lt(row_index,count) == True{} : Bool},+tap_inside: {Nat.is_lt(tap_index,taps) == True{} : Bool}) ->
  {ListRead.at(~F32,0.0,Nat.add(Nat.mul(row_index,taps),tap_index),transpose(count,output,outputs,taps,input)) ==
    Tree.read(F32,input,U32.from_nat(Nat.add(U32.to_nat(transpose_query_address(TransposeQuery{input,count,output,outputs,taps},row_index)),Nat.mul(tap_index,U32.to_nat(outputs))))) : F32}:
  +query = {TransposeQuery{input,count,output,outputs,taps} : TransposeQuery}
  +address = transpose_query_address(query,row_index)
  %Equal.sym(List<&2,List<&2,F32>>,Pair.snd(Tree.Tree<F32>,List<&2,List<&2,F32>>,transpose_rows(count,output,outputs,taps,input)),ListRead.enumerate(~List<&2,F32>,~TransposeQuery,~transpose_query_sample,count,0n,query),transpose_rows_enumerated(count,output,outputs,taps,input)) :
    {ListRead.at(~F32,0.0,Nat.add(Nat.mul(row_index,taps),tap_index),List.concat(&2,F32,_)) == Tree.read(F32,input,U32.from_nat(Nat.add(U32.to_nat(address),Nat.mul(tap_index,U32.to_nat(outputs))))) : F32}
  Equal.trans(F32,ListRead.at(~F32,0.0,Nat.add(Nat.mul(row_index,taps),tap_index),List.concat(&2,F32,ListRead.enumerate(~List<&2,F32>,~TransposeQuery,~transpose_query_sample,count,0n,query))),
    ListRead.at(~F32,0.0,tap_index,transpose_query_sample(query,row_index)),Tree.read(F32,input,U32.from_nat(Nat.add(U32.to_nat(address),Nat.mul(tap_index,U32.to_nat(outputs))))),
    ListRead.concat_enumerate_at(~F32,~TransposeQuery,~transpose_query_sample,~transpose_query_width,~transpose_query_row_length,0.0,count,0n,row_index,tap_index,query,row_inside,tap_inside),
    row_values_numeric(taps,address,outputs,input,tap_index,tap_inside))

def transpose_matrix_numeric(+outputs: U32,+taps: U32,+input: Tree.Tree<F32>,+output: U32,+tap: U32,
  +output_inside: {U32.is_lt(output,outputs) == True{} : Bool},+tap_inside: {U32.is_lt(tap,taps) == True{} : Bool}) ->
  {ListRead.at(~F32,0.0,Nat.add(Nat.mul(U32.to_nat(output),U32.to_nat(taps)),U32.to_nat(tap)),transpose(U32.to_nat(outputs),0,outputs,U32.to_nat(taps),input)) ==
    Tree.read(F32,input,(tap * outputs + output : U32)) : F32}:
  +query = {TransposeQuery{input,U32.to_nat(outputs),0,outputs,U32.to_nat(taps)} : TransposeQuery}
  +address = transpose_query_address(query,U32.to_nat(output))
  +address_equal = Equal.trans(U32,address,U32.from_nat(U32.to_nat(output)),output,
    Equal.cong(Nat,U32,U32.from_nat,Nat.mul(U32.to_nat(output),1n),U32.to_nat(output),Order.times_one(U32.to_nat(output))),Arithmetic.native_roundtrip(output))
  %Equal.sym(U32,(tap * outputs + output : U32),U32.from_nat(Nat.add(Nat.mul(U32.to_nat(tap),U32.to_nat(outputs)),U32.to_nat(output))),Encoded.native_madd(tap,outputs,output)) :
    {ListRead.at(~F32,0.0,Nat.add(Nat.mul(U32.to_nat(output),U32.to_nat(taps)),U32.to_nat(tap)),transpose(U32.to_nat(outputs),0,outputs,U32.to_nat(taps),input)) == Tree.read(F32,input,_) : F32}
  %Equal.sym(Nat,Nat.add(Nat.mul(U32.to_nat(tap),U32.to_nat(outputs)),U32.to_nat(output)),Nat.add(U32.to_nat(output),Nat.mul(U32.to_nat(tap),U32.to_nat(outputs))),Order.commutative(Nat.mul(U32.to_nat(tap),U32.to_nat(outputs)),U32.to_nat(output))) :
    {ListRead.at(~F32,0.0,Nat.add(Nat.mul(U32.to_nat(output),U32.to_nat(taps)),U32.to_nat(tap)),transpose(U32.to_nat(outputs),0,outputs,U32.to_nat(taps),input)) == Tree.read(F32,input,U32.from_nat(_)) : F32}
  %address_equal :
    {ListRead.at(~F32,0.0,Nat.add(Nat.mul(U32.to_nat(output),U32.to_nat(taps)),U32.to_nat(tap)),transpose(U32.to_nat(outputs),0,outputs,U32.to_nat(taps),input)) ==
      Tree.read(F32,input,U32.from_nat(Nat.add(U32.to_nat(_),Nat.mul(U32.to_nat(tap),U32.to_nat(outputs))))) : F32}
  transpose_numeric(U32.to_nat(outputs),0,outputs,U32.to_nat(taps),input,U32.to_nat(output),U32.to_nat(tap),Bounds.as_nat_lt(output,outputs,output_inside),Bounds.as_nat_lt(tap,taps,tap_inside))

def weight_values(result: WeightValues) -> List<&2,F32>:
  WeightValues{input,gradient,values} = result
  values

def values_selected_kept(+n: Nat,+rows: Nat,+columns: Nat,+g: Nat,+geometry: Kernel.Geometry,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {values_selected(n,rows,columns,g,geometry,input,gradient) ==
    Groups.Group{input,gradient,Groups.group_output(values_selected(n,rows,columns,g,geometry,input,gradient))} : Groups.Group}:
  Kernel.Geometry{gc,+go,gh,gw,gk,gs,gp,gr,gcols,+taps} = geometry
  +count = (go * taps : U32)
  %Equal.sym(GroupList,images(n,0,geometry,rows,columns,g,input,gradient),
    GroupList{input,gradient,Pair.snd(Groups.Sources,List<&2,Tree.Tree<F32>>,
      Traversal.collect(~&2,~Groups.Sources,~Tree.Tree<F32>,~Kernel.ImageContext,~image_read,n,Groups.Sources{input,gradient},Kernel.ImageContext{geometry,0,rows,columns,g}))},
    images_kept(n,0,geometry,rows,columns,g,input,gradient)) :
    {reduced(n,count,_) == Groups.Group{input,gradient,Groups.group_output(reduced(n,count,_))} : Groups.Group}
  {==}

# The weight kernel gives back the input and the gradient it read.
def values_kept(+work: Parameter.Work,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {values(work,input,gradient) == Groups.Group{input,gradient,Groups.group_output(values(work,input,gradient))} : Groups.Group}:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  values_selected_kept(n,rows,columns,g,Kernel.geometry(work),input,gradient)

# The list of the weight gradient is the transposed values.
def finished_list(+work: Parameter.Work,+result: Groups.Group) ->
  {weight_values(finished(Kernel.geometry(work),result)) ==
    transpose(U32.to_nat(Kernel.outputs(work)),0,Kernel.outputs(work),U32.to_nat(Kernel.taps(work)),Groups.group_output(result)) : List<&2,F32>}:
  Parameter.Work{n,c,o,h,w,k,s,p,rows,columns,g} = work
  Groups.Group{input,gradient,values} = result
  {==}

def weight_list(+work: Parameter.Work,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {weight_values(weight(work,input,gradient)) ==
    transpose(U32.to_nat(Kernel.outputs(work)),0,Kernel.outputs(work),U32.to_nat(Kernel.taps(work)),Groups.group_output(values(work,input,gradient))) : List<&2,F32>}:
  finished_list(work,values(work,input,gradient))

def group_query_element(query: GroupQuery) -> Nat:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Nat.add(Nat.mul(U32.to_nat(output),U32.to_nat(taps)),U32.to_nat(tap))

def query_weight(+query: BatchQuery) -> WeightValues:
  BatchQuery{GroupQuery{geometry,domain,group_size,image,input,gradient,tap,output,tap_inside,output_inside},range_rows,range_columns,range_size,count,ranges_frame} = query
  finished(geometry,query_batch(query))

def finished_numeric(+geometry: Kernel.Geometry,+result: Groups.Group,+tap: U32,+output: U32,
  +tap_inside: {U32.is_lt(tap,Packing.geometry_taps(geometry)) == True{} : Bool},+output_inside: {U32.is_lt(output,Groups.geometry_outputs(geometry)) == True{} : Bool}) ->
  {ListRead.at(~F32,0.0,Nat.add(Nat.mul(U32.to_nat(output),U32.to_nat(Packing.geometry_taps(geometry))),U32.to_nat(tap)),weight_values(finished(geometry,result))) ==
    Tree.read(F32,Groups.group_output(result),(tap * Groups.geometry_outputs(geometry) + output : U32)) : F32}:
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,+taps} = geometry
  Groups.Group{input,gradient,+values} = result
  transpose_matrix_numeric(o,taps,values,output,tap,output_inside,tap_inside)

def query_weight_numeric(+query: BatchQuery) ->
  {ListRead.at(~F32,0.0,group_query_element(batch_query_group(query)),weight_values(query_weight(query))) ==
    Grouped.tree(ListRead.enumerate(~F32,~BatchQuery,~batch_query_sample,batch_query_size(query),0n,query)) : F32}:
  BatchQuery{+group,range_rows,range_columns,range_size,count,ranges_frame} = query
  GroupQuery{+geometry,+domain,size,image,input,gradient,+tap,+output,+tap_inside,+output_inside} = group
  Kernel.Geometry{c,+o,h,w,k,s,p,rows,columns,taps} = geometry
  Groups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,bound,aligned} = domain
  +result = query_batch(query)
  +index = Nat.add(Nat.mul(U32.to_nat(tap),U32.to_nat(o)),U32.to_nat(output))
  +numeric = Shape.second(batch_query_frame(query,Groups.group_sources(result)),
    Shape.Both<batch_query_value(query,Groups.group_output(result)),
      {batch_query_observe(query,Groups.group_output(result)) == Grouped.tree(ListRead.enumerate(~F32,~BatchQuery,~batch_query_sample,batch_query_size(query),0n,query)) : F32}>,batch_query_numeric(query))
  Equal.trans(F32,ListRead.at(~F32,0.0,group_query_element(group),weight_values(finished(geometry,result))),Tree.read(F32,Groups.group_output(result),(tap * o + output : U32)),
    Grouped.tree(ListRead.enumerate(~F32,~BatchQuery,~batch_query_sample,batch_query_size(query),0n,query)),finished_numeric(geometry,result,tap,output,tap_inside,output_inside),
    %Equal.sym(U32,(tap * o + output : U32),U32.from_nat(index),Encoded.native_madd(tap,o,output)) :
      {Tree.read(F32,Groups.group_output(result),_) == Grouped.tree(ListRead.enumerate(~F32,~BatchQuery,~batch_query_sample,batch_query_size(query),0n,query)) : F32}
    Shape.second(batch_query_value(query,Groups.group_output(result)),
      {batch_query_observe(query,Groups.group_output(result)) == Grouped.tree(ListRead.enumerate(~F32,~BatchQuery,~batch_query_sample,batch_query_size(query),0n,query)) : F32},numeric))

def group_query_numeric_source(query: GroupQuery) -> Groups.NumericSource:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  Groups.NumericSource{input,gradient,geometry,image}

def group_query_tap(query: GroupQuery) -> U32:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  tap

def native_work(geometry: Kernel.Geometry,count: U32,size: U32) -> Parameter.Work:
  Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,taps} = geometry
  Parameter.Work{U32.to_nat(count),U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p),U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(size)}

def GeometryEncoded(geometry: Kernel.Geometry) -> Data:
  Kernel.Geometry{c,o,h,w,+k,s,p,rows,columns,taps} = geometry
  {U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))) == taps : U32}

def native_work_geometry(+geometry: Kernel.Geometry,+count: U32,+size: U32,encoding: GeometryEncoded(geometry)) ->
  {Kernel.geometry(native_work(geometry,count,size)) == geometry : Kernel.Geometry}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+rows,+columns,+taps} = geometry
  %Equal.sym(U32,U32.from_nat(U32.to_nat(c)),c,Arithmetic.native_roundtrip(c)) :
    {Kernel.Geometry{_,U32.from_nat(U32.to_nat(o)),U32.from_nat(U32.to_nat(h)),U32.from_nat(U32.to_nat(w)),U32.from_nat(U32.to_nat(k)),U32.from_nat(U32.to_nat(s)),U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))} == geometry : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(o)),o,Arithmetic.native_roundtrip(o)) :
    {Kernel.Geometry{c,_,U32.from_nat(U32.to_nat(h)),U32.from_nat(U32.to_nat(w)),U32.from_nat(U32.to_nat(k)),U32.from_nat(U32.to_nat(s)),U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))} == geometry : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(h)),h,Arithmetic.native_roundtrip(h)) :
    {Kernel.Geometry{c,o,_,U32.from_nat(U32.to_nat(w)),U32.from_nat(U32.to_nat(k)),U32.from_nat(U32.to_nat(s)),U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))} == geometry : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(w)),w,Arithmetic.native_roundtrip(w)) :
    {Kernel.Geometry{c,o,h,_,U32.from_nat(U32.to_nat(k)),U32.from_nat(U32.to_nat(s)),U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))} == geometry : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(k)),k,Arithmetic.native_roundtrip(k)) :
    {Kernel.Geometry{c,o,h,w,_,U32.from_nat(U32.to_nat(s)),U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))} == geometry : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(s)),s,Arithmetic.native_roundtrip(s)) :
    {Kernel.Geometry{c,o,h,w,k,_,U32.from_nat(U32.to_nat(p)),U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))} == geometry : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(p)),p,Arithmetic.native_roundtrip(p)) :
    {Kernel.Geometry{c,o,h,w,k,s,_,U32.from_nat(U32.to_nat(rows)),U32.from_nat(U32.to_nat(columns)),U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))} == geometry : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(rows)),rows,Arithmetic.native_roundtrip(rows)) :
    {Kernel.Geometry{c,o,h,w,k,s,p,_,U32.from_nat(U32.to_nat(columns)),U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))} == geometry : Kernel.Geometry}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(columns)),columns,Arithmetic.native_roundtrip(columns)) :
    {Kernel.Geometry{c,o,h,w,k,s,p,rows,_,U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))} == geometry : Kernel.Geometry}
  Equal.cong(U32,Kernel.Geometry,value => Kernel.Geometry{c,o,h,w,k,s,p,rows,columns,value},U32.from_nat(Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k)))),taps,encoding)

def packed_query_geometry(query: GroupQuery) -> Kernel.Geometry:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  geometry

def packed_query_input(query: GroupQuery) -> Buffer.Model<F32>:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  input

def packed_query_gradient(query: GroupQuery) -> Buffer.Model<F32>:
  GroupQuery{geometry,domain,size,image,input,gradient,tap,output,tap_inside,output_inside} = query
  gradient

def native_query_weight(+query: GroupQuery,+count: U32,+size_positive: {Nat.is_lt(0n,U32.to_nat(group_query_size(query))) == True{} : Bool},
  +encoding: GeometryEncoded(packed_query_geometry(query))) ->
  {query_weight(native_batch_query(query,count,size_positive)) == weight(native_work(packed_query_geometry(query),count,group_query_size(query)),packed_query_input(query),packed_query_gradient(query)) : WeightValues}:
  GroupQuery{+geometry,+domain,+size,image,+input,+gradient,tap,output,tap_inside,output_inside} = query
  Kernel.Geometry{c,o,h,w,k,s,p,+rows,+columns,taps} = geometry
  Groups.GeometryReady{limits,columns_positive,kernel_positive,stride_positive,square,taps_positive,outputs_positive,bound,aligned} = domain
  %Equal.sym(Kernel.Geometry,Kernel.geometry(native_work(geometry,count,size)),geometry,native_work_geometry(geometry,count,size,encoding)) :
    {query_weight(native_batch_query(query,count,size_positive)) ==
      finished(_,values_selected(U32.to_nat(count),U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(size),_,input,gradient)) : WeightValues}
  {==}

# ---- The rows of the result ----

type CopyModel is Data:
  CopyModel{values: Tree.Tree<F32>,result: Buffer.Model<F32>,address: U32,stride: U32,destination: U32}

def represent_copy(state: CopyModel) -> Kernel.RowCopy:
  CopyModel{values,result,address,stride,destination} = state
  Kernel.RowCopy{Tree.pack(F32,values),Buffer.pack(~F32,result),address,stride,destination}

def tap_written(+values: Tree.Tree<F32>,+address: U32,+destination: U32,result: Buffer.Model<F32>) -> Storage.Buffer<F32>:
  Storage.write(~F32,destination,Tree.read(F32,values,address),Buffer.pack(~F32,result))

def tap_copy(state: CopyModel) -> CopyModel:
  CopyModel{+values,result,+address,+stride,+destination} = state
  CopyModel{values,Buffer.remodel(~F32,tap_written(values,address,destination,result)),(address + stride : U32),stride,(destination + 1 : U32)}

def tap_copy_matches(+state: CopyModel) -> {Kernel.tap_copy(represent_copy(state)) == represent_copy(tap_copy(state)) : Kernel.RowCopy}:
  CopyModel{+values,+result,+address,+stride,+destination} = state
  %Equal.sym(Array<F32> & F32,Array.get(F32,Tree.pack(F32,values),address),(Tree.pack(F32,values),Tree.read(F32,values,address)),StorageLaws.read(F32,values,address)) :
    {Kernel.tap_copied(address,stride,destination,Buffer.pack(~F32,result),_) == represent_copy(tap_copy(CopyModel{values,result,address,stride,destination})) : Kernel.RowCopy}
  Equal.cong(Storage.Buffer<F32>,Kernel.RowCopy,buffer => Kernel.RowCopy{Tree.pack(F32,values),buffer,(address + stride : U32),stride,(destination + 1 : U32)},
    tap_written(values,address,destination,result),Buffer.pack(~F32,Buffer.remodel(~F32,tap_written(values,address,destination,result))),
    Equal.sym(Storage.Buffer<F32>,Buffer.pack(~F32,Buffer.remodel(~F32,tap_written(values,address,destination,result))),tap_written(values,address,destination,result),
      Buffer.remodeled(~F32,tap_written(values,address,destination,result))))

def copy_values(state: CopyModel) -> Tree.Tree<F32>:
  CopyModel{values,result,address,stride,destination} = state
  values

def copy_result(state: CopyModel) -> Buffer.Model<F32>:
  CopyModel{values,result,address,stride,destination} = state
  result

def copy_kept(count: Nat,+state: CopyModel) -> {copy_values(Loop.run(~CopyModel,~tap_copy,count,state)) == copy_values(state) : Tree.Tree<F32>}:
  match count:
    case 0n: {==}
    case 1n+rest:
      CopyModel{+values,+result,+address,+stride,+destination} = state
      copy_kept(rest,tap_copy(CopyModel{values,result,address,stride,destination}))

type RowWriteModel is Data:
  RowWriteModel{values: Tree.Tree<F32>,result: Buffer.Model<F32>,row: U32,rows: U32,taps: U32,first: U32}

def represent_rows(state: RowWriteModel) -> Kernel.RowWrite:
  RowWriteModel{values,result,row,rows,taps,first} = state
  Kernel.RowWrite{Tree.pack(F32,values),Buffer.pack(~F32,result),row,rows,taps,first}

def row_copy(+row: U32,+rows: U32,+taps: U32,+first: U32,+values: Tree.Tree<F32>,result: Buffer.Model<F32>) -> CopyModel:
  Loop.run(~CopyModel,~tap_copy,U32.to_nat(taps),CopyModel{values,result,row,rows,((first + row) * taps : U32)})

def row_step(state: RowWriteModel) -> RowWriteModel:
  RowWriteModel{+values,result,+row,+rows,+taps,+first} = state
  RowWriteModel{values,copy_result(row_copy(row,rows,taps,first,values,result)),(row + 1 : U32),rows,taps,first}

def row_copied_matches(+row: U32,+rows: U32,+taps: U32,+first: U32,+values: Tree.Tree<F32>,+copy: CopyModel,
  kept: {copy_values(copy) == values : Tree.Tree<F32>}) ->
  {Kernel.row_copied(row,rows,taps,first,represent_copy(copy)) ==
    Kernel.RowWrite{Tree.pack(F32,values),Buffer.pack(~F32,copy_result(copy)),(row + 1 : U32),rows,taps,first} : Kernel.RowWrite}:
  CopyModel{+copied,+result,address,stride,destination} = copy
  Equal.cong(Tree.Tree<F32>,Kernel.RowWrite,tree => Kernel.RowWrite{Tree.pack(F32,tree),Buffer.pack(~F32,result),(row + 1 : U32),rows,taps,first},copied,values,kept)

def row_step_matches(+state: RowWriteModel) -> {Kernel.row_step(represent_rows(state)) == represent_rows(row_step(state)) : Kernel.RowWrite}:
  RowWriteModel{+values,+result,+row,+rows,+taps,+first} = state
  +start = {CopyModel{values,result,row,rows,((first + row) * taps : U32)} : CopyModel}
  %Equal.sym(Kernel.RowCopy,Loop.run(~Kernel.RowCopy,~Kernel.tap_copy,U32.to_nat(taps),represent_copy(start)),represent_copy(row_copy(row,rows,taps,first,values,result)),
    Projection.iteration_refines_model(~Kernel.RowCopy,~CopyModel,~Kernel.tap_copy,~tap_copy,~represent_copy,~tap_copy_matches,U32.to_nat(taps),start)) :
    {Kernel.row_copied(row,rows,taps,first,_) == represent_rows(row_step(RowWriteModel{values,result,row,rows,taps,first})) : Kernel.RowWrite}
  row_copied_matches(row,rows,taps,first,values,row_copy(row,rows,taps,first,values,result),copy_kept(U32.to_nat(taps),start))

def rows_result(state: RowWriteModel) -> Buffer.Model<F32>:
  RowWriteModel{values,result,row,rows,taps,first} = state
  result

def rows_result_matches(+state: RowWriteModel) -> {Kernel.rows_result(represent_rows(state)) == Buffer.pack(~F32,rows_result(state)) : Storage.Buffer<F32>}:
  RowWriteModel{values,result,row,rows,taps,first} = state
  {==}

# The rows from row on write the transposed values of those rows once.
def rows_content(count: Nat,state: RowWriteModel) -> Buffer.Content<F32>:
  RowWriteModel{+values,result,+row,+rows,+taps,+first} = state
  +content = Buffer.content(~F32,result)
  Buffer.Content{Tree.write_sequence(~F32,transpose(count,row,rows,U32.to_nat(taps),values),((first + row) * taps : U32),Buffer.content_tree(~F32,content)),Buffer.content_length(~F32,content)}

def content_eta(+content: Buffer.Content<F32>) -> {content == Buffer.Content{Buffer.content_tree(~F32,content),Buffer.content_length(~F32,content)} : Buffer.Content<F32>}:
  Buffer.Content{tree,length} = content
  {==}

def row_prepended(+value: F32,result: Tree.Tree<F32> & List<&2,F32>) ->
  {Pair.snd(Tree.Tree<F32>,List<&2,F32>,Traversal.collect_prepend(~&2,~Tree.Tree<F32>,~F32,value,result)) ==
    value <> Pair.snd(Tree.Tree<F32>,List<&2,F32>,result) : List<&2,F32>}:
  (+source,+values) = result
  {==}

def row_values_next(+count: Nat,+address: U32,+stride: U32,+input: Tree.Tree<F32>) ->
  {row_values(1n+count,address,stride,input) == Tree.read(F32,input,address) <> row_values(count,(address + stride : U32),stride,input) : List<&2,F32>}:
  row_prepended(Tree.read(F32,input,address),row_result(count,(address + stride : U32),stride,input))

# Count taps of a row: the values read at stride, written in order from destination.
def copy_content(count: Nat,+values: Tree.Tree<F32>,+result: Buffer.Model<F32>,+address: U32,+stride: U32,+destination: U32) ->
  {Buffer.content(~F32,copy_result(Loop.run(~CopyModel,~tap_copy,count,CopyModel{values,result,address,stride,destination}))) ==
    Buffer.Content{Tree.write_sequence(~F32,row_values(count,address,stride,values),destination,Buffer.content_tree(~F32,Buffer.content(~F32,result))),
      Buffer.content_length(~F32,Buffer.content(~F32,result))} : Buffer.Content<F32>}:
  match count:
    case 0n: content_eta(Buffer.content(~F32,result))
    case 1n+ +rest:
      +value = Tree.read(F32,values,address)
      +later = row_values(rest,(address + stride : U32),stride,values)
      +before = Buffer.content_tree(~F32,Buffer.content(~F32,result))
      +length = Buffer.content_length(~F32,Buffer.content(~F32,result))
      +written = Buffer.remodel(~F32,tap_written(values,address,destination,result))
      +written_content = Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,written),Buffer.contents(~F32,tap_written(values,address,destination,result)),
        Buffer.Content{Tree.write(F32,before,destination,value),length},
        Buffer.remodel_content(~F32,tap_written(values,address,destination,result)),Buffer.write_value_packed(~F32,destination,value,result))
      %Equal.sym(List<&2,F32>,row_values(1n+rest,address,stride,values),value <> later,row_values_next(rest,address,stride,values)) :
        {Buffer.content(~F32,copy_result(Loop.run(~CopyModel,~tap_copy,1n+rest,CopyModel{values,result,address,stride,destination}))) ==
          Buffer.Content{Tree.write_sequence(~F32,_,destination,before),length} : Buffer.Content<F32>}
      Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,copy_result(Loop.run(~CopyModel,~tap_copy,rest,CopyModel{values,written,(address + stride : U32),stride,(destination + 1 : U32)}))),
        Buffer.Content{Tree.write_sequence(~F32,later,(destination + 1 : U32),Buffer.content_tree(~F32,Buffer.content(~F32,written))),Buffer.content_length(~F32,Buffer.content(~F32,written))},
        Buffer.Content{Tree.write_sequence(~F32,later,(destination + 1 : U32),Tree.write(F32,before,destination,value)),length},
        copy_content(rest,values,written,(address + stride : U32),stride,(destination + 1 : U32)),
        %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,written),Buffer.Content{Tree.write(F32,before,destination,value),length},written_content) :
          {Buffer.Content{Tree.write_sequence(~F32,later,(destination + 1 : U32),Buffer.content_tree(~F32,_)),Buffer.content_length(~F32,_)} ==
            Buffer.Content{Tree.write_sequence(~F32,later,(destination + 1 : U32),Tree.write(F32,before,destination,value)),length} : Buffer.Content<F32>}
        {==})

def rows_prepended(+value: List<&2,F32>,result: Tree.Tree<F32> & List<&2,List<&2,F32>>) ->
  {List.concat(&2,F32,Pair.snd(Tree.Tree<F32>,List<&2,List<&2,F32>>,Traversal.collect_prepend(~&2,~Tree.Tree<F32>,~List<&2,F32>,value,result))) ==
    List.append(&2,F32,value,List.concat(&2,F32,Pair.snd(Tree.Tree<F32>,List<&2,List<&2,F32>>,result))) : List<&2,F32>}:
  (+source,+rows) = result
  {==}

# The row after a row of taps values starts taps further on.
def row_offset_next(+first: U32,+row: U32,+rows: U32,+taps: U32,+values: Tree.Tree<F32>) ->
  {((first + (row + 1 : U32) : U32) * taps : U32) == ListRead.end_position(~F32,row_values(U32.to_nat(taps),row,rows,values),((first + row) * taps : U32)) : U32}:
  +offset = ((first + row) * taps : U32)
  +row_list = row_values(U32.to_nat(taps),row,rows,values)
  Equal.trans(U32,((first + (row + 1 : U32) : U32) * taps : U32),(((first + row : U32) + 1 : U32) * taps : U32),ListRead.end_position(~F32,row_list,offset),
    Equal.cong(U32,U32,value => (value * taps : U32),(first + (row + 1 : U32) : U32),((first + row : U32) + 1 : U32),
      Equal.sym(U32,((first + row : U32) + 1 : U32),(first + (row + 1 : U32) : U32),Arithmetic.add_associative(first,row,1))),
    Equal.trans(U32,(((first + row : U32) + 1 : U32) * taps : U32),(offset + taps : U32),ListRead.end_position(~F32,row_list,offset),
      Arithmetic.multiply_successor((first + row : U32),taps),
      Equal.trans(U32,(offset + taps : U32),(offset + U32.from_nat(List.length(&2,F32,row_list)) : U32),ListRead.end_position(~F32,row_list,offset),
        %Equal.sym(Nat,List.length(&2,F32,row_list),U32.to_nat(taps),row_values_length(U32.to_nat(taps),row,rows,values)) : {(offset + taps : U32) == (offset + U32.from_nat(_) : U32) : U32}
        Equal.sym(U32,(offset + U32.from_nat(U32.to_nat(taps)) : U32),(offset + taps : U32),
          Equal.cong(U32,U32,value => (offset + value : U32),U32.from_nat(U32.to_nat(taps)),taps,Arithmetic.native_roundtrip(taps))),
        Equal.sym(U32,ListRead.end_position(~F32,row_list,offset),(offset + U32.from_nat(List.length(&2,F32,row_list)) : U32),ListRead.end_position_length(~F32,row_list,offset)))))

def rows_loop_content(count: Nat,+state: RowWriteModel) ->
  {Buffer.content(~F32,rows_result(Loop.run(~RowWriteModel,~row_step,count,state))) == rows_content(count,state) : Buffer.Content<F32>}:
  match count:
    case 0n:
      RowWriteModel{values,+result,row,rows,taps,first} = state
      content_eta(Buffer.content(~F32,result))
    case 1n+ +rest:
      RowWriteModel{+values,+result,+row,+rows,+taps,+first} = state
      +offset = ((first + row) * taps : U32)
      +row_list = row_values(U32.to_nat(taps),row,rows,values)
      +later = transpose(rest,(row + 1 : U32),rows,U32.to_nat(taps),values)
      +before = Buffer.content_tree(~F32,Buffer.content(~F32,result))
      +length = Buffer.content_length(~F32,Buffer.content(~F32,result))
      +written = copy_result(row_copy(row,rows,taps,first,values,result))
      +next = {RowWriteModel{values,written,(row + 1 : U32),rows,taps,first} : RowWriteModel}
      +written_content = copy_content(U32.to_nat(taps),values,result,row,rows,offset)
      Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,rows_result(Loop.run(~RowWriteModel,~row_step,rest,next))),rows_content(rest,next),
        Buffer.Content{Tree.write_sequence(~F32,transpose(1n+rest,row,rows,U32.to_nat(taps),values),offset,before),length},
        rows_loop_content(rest,next),
        %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,written),Buffer.Content{Tree.write_sequence(~F32,row_list,offset,before),length},written_content) :
          {Buffer.Content{Tree.write_sequence(~F32,later,((first + (row + 1 : U32) : U32) * taps : U32),Buffer.content_tree(~F32,_)),Buffer.content_length(~F32,_)} ==
            Buffer.Content{Tree.write_sequence(~F32,transpose(1n+rest,row,rows,U32.to_nat(taps),values),offset,before),length} : Buffer.Content<F32>}
        %Equal.sym(List<&2,F32>,
            List.concat(&2,F32,Pair.snd(Tree.Tree<F32>,List<&2,List<&2,F32>>,Traversal.collect_prepend(~&2,~Tree.Tree<F32>,~List<&2,F32>,row_list,transpose_rows(rest,(row + 1 : U32),rows,U32.to_nat(taps),values)))),
            List.append(&2,F32,row_list,later),rows_prepended(row_list,transpose_rows(rest,(row + 1 : U32),rows,U32.to_nat(taps),values))) :
          {Buffer.Content{Tree.write_sequence(~F32,later,((first + (row + 1 : U32) : U32) * taps : U32),Tree.write_sequence(~F32,row_list,offset,before)),length} ==
            Buffer.Content{Tree.write_sequence(~F32,_,offset,before),length} : Buffer.Content<F32>}
        %Equal.sym(Tree.Tree<F32>,Tree.write_sequence(~F32,List.append(&2,F32,row_list,later),offset,before),
            Tree.write_sequence(~F32,later,ListRead.end_position(~F32,row_list,offset),Tree.write_sequence(~F32,row_list,offset,before)),ListRead.concatenated(~F32,row_list,later,offset,before)) :
          {Buffer.Content{Tree.write_sequence(~F32,later,((first + (row + 1 : U32) : U32) * taps : U32),Tree.write_sequence(~F32,row_list,offset,before)),length} == Buffer.Content{_,length} : Buffer.Content<F32>}
        Equal.cong(U32,Buffer.Content<F32>,position => Buffer.Content{Tree.write_sequence(~F32,later,position,Tree.write_sequence(~F32,row_list,offset,before)),length},
          ((first + (row + 1 : U32) : U32) * taps : U32),ListRead.end_position(~F32,row_list,offset),row_offset_next(first,row,rows,taps,values)))

# Rows [first, first + rows) of a result, from the values of rows outputs:
# the transposed values written once at the first row.
def rows_written(+rows: U32,+taps: U32,+first: U32,+values: Tree.Tree<F32>,+result: Buffer.Model<F32>) ->
  {Buffer.contents(~F32,Kernel.rows_written(rows,taps,first,Tree.pack(F32,values),Buffer.pack(~F32,result))) ==
    rows_content(U32.to_nat(rows),RowWriteModel{values,result,0,rows,taps,first}) : Buffer.Content<F32>}:
  +state = {RowWriteModel{values,result,0,rows,taps,first} : RowWriteModel}
  +final = Loop.run(~RowWriteModel,~row_step,U32.to_nat(rows),state)
  %Equal.sym(Kernel.RowWrite,Loop.run(~Kernel.RowWrite,~Kernel.row_step,U32.to_nat(rows),represent_rows(state)),represent_rows(final),
    Projection.iteration_refines_model(~Kernel.RowWrite,~RowWriteModel,~Kernel.row_step,~row_step,~represent_rows,~row_step_matches,U32.to_nat(rows),state)) :
    {Buffer.contents(~F32,Kernel.rows_result(_)) == rows_content(U32.to_nat(rows),state) : Buffer.Content<F32>}
  %Equal.sym(Storage.Buffer<F32>,Kernel.rows_result(represent_rows(final)),Buffer.pack(~F32,rows_result(final)),rows_result_matches(final)) :
    {Buffer.contents(~F32,_) == rows_content(U32.to_nat(rows),state) : Buffer.Content<F32>}
  %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,rows_result(final))),Buffer.content(~F32,rows_result(final)),Buffer.packed_contents(~F32,rows_result(final))) :
    {_ == rows_content(U32.to_nat(rows),state) : Buffer.Content<F32>}
  rows_loop_content(U32.to_nat(rows),state)

# Every output written into a fresh result: the transposed values written once.
def rows_fresh(+outputs: U32,+taps: U32,+values: Tree.Tree<F32>) ->
  {Buffer.contents(~F32,Kernel.rows_written(outputs,taps,0,Tree.pack(F32,values),Tensors.buffer(~F32,0.0,(outputs * taps : U32)))) ==
    Buffer.Content{Tree.write_sequence(~F32,transpose(U32.to_nat(outputs),0,outputs,U32.to_nat(taps),values),0,Tree.fresh(F32,Tensors.depth_for((outputs * taps : U32)),0.0)),
      U32.to_nat((outputs * taps : U32))} : Buffer.Content<F32>}:
  +count = (outputs * taps : U32)
  +fresh = Buffer.remodel(~F32,Tensors.buffer(~F32,0.0,count))
  +list = transpose(U32.to_nat(outputs),0,outputs,U32.to_nat(taps),values)
  %Buffer.remodeled(~F32,Tensors.buffer(~F32,0.0,count)) :
    {Buffer.contents(~F32,Kernel.rows_written(outputs,taps,0,Tree.pack(F32,values),_)) ==
      Buffer.Content{Tree.write_sequence(~F32,list,0,Tree.fresh(F32,Tensors.depth_for(count),0.0)),U32.to_nat(count)} : Buffer.Content<F32>}
  Equal.trans(Buffer.Content<F32>,Buffer.contents(~F32,Kernel.rows_written(outputs,taps,0,Tree.pack(F32,values),Buffer.pack(~F32,fresh))),
    rows_content(U32.to_nat(outputs),RowWriteModel{values,fresh,0,outputs,taps,0}),
    Buffer.Content{Tree.write_sequence(~F32,list,0,Tree.fresh(F32,Tensors.depth_for(count),0.0)),U32.to_nat(count)},
    rows_written(outputs,taps,0,values,fresh),
    %Equal.sym(Buffer.Content<F32>,Buffer.content(~F32,fresh),Buffer.Content{Tree.fresh(F32,Tensors.depth_for(count),0.0),U32.to_nat(count)},
        Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,fresh),Buffer.contents(~F32,Tensors.buffer(~F32,0.0,count)),
          Buffer.Content{Tree.fresh(F32,Tensors.depth_for(count),0.0),U32.to_nat(count)},
          Buffer.remodel_content(~F32,Tensors.buffer(~F32,0.0,count)),Listed.allocated(~F32,0.0,count))) :
      {Buffer.Content{Tree.write_sequence(~F32,list,((0 + 0 : U32) * taps : U32),Buffer.content_tree(~F32,_)),Buffer.content_length(~F32,_)} ==
        Buffer.Content{Tree.write_sequence(~F32,list,0,Tree.fresh(F32,Tensors.depth_for(count),0.0)),U32.to_nat(count)} : Buffer.Content<F32>}
    Equal.cong(U32,Buffer.Content<F32>,position => Buffer.Content{Tree.write_sequence(~F32,list,position,Tree.fresh(F32,Tensors.depth_for(count),0.0)),U32.to_nat(count)},
      ((0 + 0 : U32) * taps : U32),0,
      Equal.trans(U32,((0 + 0 : U32) * taps : U32),(0 * taps : U32),0,
        Equal.cong(U32,U32,value => (value * taps : U32),(0 + 0 : U32),0,Arithmetic.add_zero(0)),Arithmetic.multiply_zero_left(taps))))

# ---- The weight kernel ----

# The sources back, and every output written into a fresh result.
def written(+work: Parameter.Work,result: Groups.Group) -> (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>:
  Groups.Group{input,gradient,values} = result
  +outputs = Kernel.outputs(work)
  +taps = Kernel.taps(work)
  ((Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),Kernel.rows_written(outputs,taps,0,Tree.pack(F32,values),Tensors.buffer(~F32,0.0,(outputs * taps : U32))))

def written_matches(+work: Parameter.Work,+result: Groups.Group) ->
  {Kernel.written(Kernel.outputs(work),Kernel.taps(work),Groups.represent(result)) == written(work,result) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>}:
  Groups.Group{input,gradient,values} = result
  {==}

def weight_matches(+work: Parameter.Work,+input: Buffer.Model<F32>,+gradient: Buffer.Model<F32>) ->
  {Kernel.weight(work,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)) == written(work,values(work,input,gradient)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>}:
  %Equal.sym((Storage.Buffer<F32> & Storage.Buffer<F32>) & Array<F32>,Kernel.weight_values(work,Buffer.pack(~F32,input),Buffer.pack(~F32,gradient)),Groups.represent(values(work,input,gradient)),
    values_matches(work,input,gradient)) :
    {Kernel.written(Kernel.outputs(work),Kernel.taps(work),_) == written(work,values(work,input,gradient)) : (Storage.Buffer<F32> & Storage.Buffer<F32>) & Storage.Buffer<F32>}
  written_matches(work,values(work,input,gradient))
