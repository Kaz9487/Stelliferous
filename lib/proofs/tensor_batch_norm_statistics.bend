# The statistics of a batch normalization read the cells [N,H,W,G] of their
# input once, in rows that are the H x W planes of an image and a group of
# channels, and merge the rows of the images. A lane of a sum is the sum of the
# planes of its channel in that order: the sums, the centered squares and the
# sums of dy and dy * xhat.
import Base
import ../kernel_tile.bend as Tile
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_view.bend as Views
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_reduce.bend as TiledReduce
import ../tensor_partials.bend as Partials
import ../tensor_line_chain.bend as LineChain
import ../math_sum.bend as Sum
import ../machine_limits.bend as Machine
import ../kernel_gemm.bend as KernelGemm
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./tensor_reduce_values.bend as Reduce
import ./storage_buffer_model.bend as Buffer
import ./tensor_partials_values.bend as PartialsValues
import ./tensor_tiled_reduce.bend as ReduceLaws
import ./tensor_tiled_sums.bend as TiledSums
import ./tensor_line_chain_values.bend as LineChainValues
import ./list_storage_readback.bend as ListReadback
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./tensor_move_readback.bend as Held
import ./word_encoding.bend as WordEncoding
import ./u32_full_range.bend as FullRange
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./tensor_tile_coordinates.bend as Coordinates
import ../tensor_tiled_cells.bend as Cells
import ./storage_tree_model.bend as Tree
import ../tensor_batch_norm.bend as BN
import ../tensor_batch_norm_cells.bend as BNCells
import ../tensor_operations.bend as Operations
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_storage_count.bend as StorageCount
import ../default_machine_limits.bend as Default
import ./tensor_contract.bend as Contract
import ./tensor_global_shape_count.bend as Global
import ../tensor_layout.bend as Layout
import ./nat_to_u32_bounds.bend as Bounds
import ../tensor_shape.bend as Shapes
import ./tensor_count_completion.bend as CountCompletion
import ./machine_limit_bounds.bend as Limits
import ./tensor_box_cells.bend as BoxCells
import ../kernel_batch_norm.bend as BNKernel
import ./kernel_tile_refinement.bend as TileLaws
import ./tensor_pieces_each.bend as Each
import ./tensor_pieces_content.bend as Content
import ./tensor_list_values.bend as Listed
import ./tensor_pieces_plan.bend as PiecesPlan
import ./tensor_tiled_exit.bend as TiledExit
import ./nat_division_order.bend as DivisionOrder
import ./word_multiplication.bend as WordMultiplication
import ./tensor_tile_slots_values.bend as SlotsValues
import ./tensor_tile_slots_lanes.bend as SlotsLanes
import ./tensor_target_cells.bend as Targets
import ./tensor_box_digits.bend as Digits
import ../traversal_array.bend as Traversal
import ./traversal_witness.bend as TraversalWitness
import ./traversal_array_refinement.bend as ArrayLaws
import ../storage_buffer.bend as Storage
import ./kernel_batch_norm_certificate.bend as KernelCertificate
import ./storage_refinement.bend as StorageLaws
import ./tensor_batch_norm_twins.bend as TwinsLaws
import ./tensor_pieces_combine_leaf.bend as CombineLeaf
# ---- Cells with the groups innermost and outermost ----

# A cell of values reads the same at its index among the cells with the
# groups innermost as at its index among the cells with the groups outermost.
def cell_moved(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+axis: Nat,+outer: Nat,+group: Nat,+cap: U32,
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+outer_inside: {Nat.is_lt(outer,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},
  +inner_room: {Nat.is_le(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(cap)) == True{} : Bool},+outer_room: {Nat.is_le(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer),U32.to_nat(cap)) == True{} : Bool}) ->
  {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Innermost{})},U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Outermost{})},U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8}:
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,Grid.Innermost{}),Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}},Public.geometry_exact(values,axis,Grid.Innermost{},inside)) :
    {Representation.lifted(~Logical,~original,Representation.Lifted{logical,_},U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,axis,Grid.Outermost{})},U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8}
  %Equal.sym(Cells.Cells,Public.geometry_of(values,axis,Grid.Outermost{}),Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}},Public.geometry_exact(values,axis,Grid.Outermost{},inside)) :
    {Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}}},U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == Representation.lifted(~Logical,~original,Representation.Lifted{logical,_},U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8}
  Equal.trans(Tile.Tile8,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}}},U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))),Representation.cell(~Logical,~original,logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}},outer,group),
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}}},U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))),
    Representation.decoded(~Logical,~original,logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Innermost{}},outer,group,U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group)),FullRange.roundtrip(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),cap,inner_room),outer_inside,group_inside),
    Equal.sym(Tile.Tile8,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}}},U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))),Representation.cell(~Logical,~original,logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}},outer,group),
      Representation.decoded(~Logical,~original,logical,Cells.Cells{(Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32),Tiled.product(Public.lane_after(values,axis)),Grid.lanes_at(values,axis),Tiled.groups(Grid.lanes_at(values,axis)),Grid.Outermost{}},outer,group,U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer)),FullRange.roundtrip(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer),cap,outer_room),outer_inside,group_inside)))

# The input of the statistics read as cells: its source at a geometry of the
# cells reads a cell index.
def lifted_at(~X: Data,x: X,geometry: Cells.Cells) -> Representation.Lifted<X>:
  Representation.Lifted{x,geometry}

# A cell of the source with the groups innermost reads as the cell at its index
# among the cells with the groups outermost.
def SourceMoved(~X: Data,~Source: Data,~source_value: Source -> U32 -> Tile.Tile8,~source_at: X -> Cells.Cells -> Source) -> Type:
  @+x: X -> @+values: List<&2,U32> -> @+axis: Nat -> @+outer: Nat -> @+group: Nat -> @+cap: U32 ->
  @+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool} -> @+outer_inside: {Nat.is_lt(outer,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))) == True{} : Bool} ->
  @+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))) == True{} : Bool} ->
  @+inner_room: {Nat.is_le(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(cap)) == True{} : Bool} -> @+outer_room: {Nat.is_le(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer),U32.to_nat(cap)) == True{} : Bool} ->
    {source_value(source_at(x,Public.geometry_of(values,axis,Grid.Innermost{})),U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == source_value(source_at(x,Public.geometry_of(values,axis,Grid.Outermost{})),U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8}

# The contexts of the source with the groups innermost and outermost.
def source_inner(~X: Data,~Source: Data,~source_at: X -> Cells.Cells -> Source,+x: X,+left_context: Source,+right_context: Source,+values: List<&2,U32>,+axis: Nat,+group: Nat,+cap: U32) -> Data:
  KernelGemm.Both<{left_context == source_at(x,Public.geometry_of(values,axis,Grid.Innermost{})) : Source},{right_context == source_at(x,Public.geometry_of(values,axis,Grid.Outermost{})) : Source}>

# A cell of the source read with the groups innermost reads the same at its
# index among the cells with the groups outermost.
def source_carried(~X: Data,~Source: Data,~source_value: Source -> U32 -> Tile.Tile8,~source_at: X -> Cells.Cells -> Source,~source_moved: SourceMoved(~X,~Source,~source_value,~source_at),
  +x: X,+left_context: Source,+right_context: Source,+values: List<&2,U32>,+axis: Nat,+group: Nat,
  +cap: U32,+related: source_inner(~X,~Source,~source_at,x,left_context,right_context,values,axis,group,cap),+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+outer: Nat,
  +outer_inside: {Nat.is_lt(outer,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},
  +inner_room: {Nat.is_le(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(cap)) == True{} : Bool},+outer_room: {Nat.is_le(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer),U32.to_nat(cap)) == True{} : Bool}) ->
  {source_value(left_context,U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == source_value(right_context,U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8}:
  KernelGemm.Both{+left_exact,+right_exact} = related
  {%Equal.sym(Source,left_context,source_at(x,Public.geometry_of(values,axis,Grid.Innermost{})),left_exact) :
    {source_value(_,U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == source_value(right_context,U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8}
    {%Equal.sym(Source,right_context,source_at(x,Public.geometry_of(values,axis,Grid.Outermost{})),right_exact) :
      {source_value(source_at(x,Public.geometry_of(values,axis,Grid.Innermost{})),U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == source_value(_,U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8}
      source_moved(x,values,axis,outer,group,cap,inside,outer_inside,group_inside,inner_room,outer_room) :
      {source_value(source_at(x,Public.geometry_of(values,axis,Grid.Innermost{})),U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == source_value(right_context,U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8}} :
    {source_value(left_context,U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == source_value(right_context,U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8}}

# The last two of three terms change places.
def swap_last(+first: Nat,+second: Nat,+third: Nat) -> {Nat.add(Nat.add(third,first),second) == Nat.add(Nat.add(first,second),third) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(third,first),second),Nat.add(Nat.add(first,third),second),Nat.add(Nat.add(first,second),third),
    Equal.cong(Nat,Nat,value => Nat.add(value,second),Nat.add(third,first),Nat.add(first,third),Order.commutative(third,first)),
    Equal.trans(Nat,Nat.add(Nat.add(first,third),second),Nat.add(first,Nat.add(third,second)),Nat.add(Nat.add(first,second),third),Addition.associative(first,third,second),
      Equal.trans(Nat,Nat.add(first,Nat.add(third,second)),Nat.add(first,Nat.add(second,third)),Nat.add(Nat.add(first,second),third),
        Equal.cong(Nat,Nat,value => Nat.add(first,value),Nat.add(third,second),Nat.add(second,third),Order.commutative(third,second)),
        Equal.sym(Nat,Nat.add(Nat.add(first,second),third),Nat.add(first,Nat.add(second,third)),Addition.associative(first,second,third)))))

# The row of a group along an axis of the cells with the groups innermost,
# the groups a stride apart, agrees with the row of the cells with the groups
# outermost, from done cells of the row on.
def row_agrees(~Left: Data,~left_value: Left -> U32 -> Tile.Tile8,~Right: Data,~right_value: Right -> U32 -> Tile.Tile8,~Moved: @+left_context: Left -> @+right_context: Right -> @+values: List<&2,U32> -> @+axis: Nat -> @+group: Nat -> @+cap: U32 -> Data,~moved: @+left_context: Left -> @+right_context: Right -> @+values: List<&2,U32> -> @+axis: Nat -> @+group: Nat -> @+cap: U32 -> @+related: Moved(left_context,right_context,values,axis,group,cap) -> @+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool} -> @+outer: Nat -> @+outer_inside: {Nat.is_lt(outer,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))) == True{} : Bool} -> @+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))) == True{} : Bool} -> @+inner_room: {Nat.is_le(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(cap)) == True{} : Bool} -> @+outer_room: {Nat.is_le(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer),U32.to_nat(cap)) == True{} : Bool} -> {left_value(left_context,U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == right_value(right_context,U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8},+left_context: Left,+right_context: Right,+values: List<&2,U32>,+axis: Nat,+cap: U32,+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
  +blocks: Nat,+width: Nat,+outer_exact: {U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)) == Nat.mul(blocks,width) : Nat},+cells_room: {Nat.is_le(Nat.mul(U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),U32.to_nat(cap)) == True{} : Bool},
  +block: Nat,+block_inside: {Nat.is_lt(block,blocks) == True{} : Bool},+group: Nat,+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},+related: Moved(left_context,right_context,values,axis,group,cap),
  remaining: Nat,+done: Nat,+fits: {Nat.add(done,remaining) == width : Nat}) ->
  Reduce.StridedAgrees(~Tile.Tile8,~Left,~left_value,~Right,~right_value,left_context,right_context,remaining,Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),done),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.add(Nat.mul(block,width),done))):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +done_inside = {%fits : {Nat.is_lt(done,_) == True{} : Bool} Order.below_sum(done,rest) : {Nat.is_lt(done,width) == True{} : Bool}}
      +outer_inside = {%Equal.sym(Nat,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)),Nat.mul(blocks,width),outer_exact) : {Nat.is_lt(Nat.add(Nat.mul(block,width),done),_) == True{} : Bool}
        RowMajor.radix(block,blocks,done,width,block_inside,done_inside) : {Nat.is_lt(Nat.add(Nat.mul(block,width),done),U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))) == True{} : Bool}}
      +inner_room = Order.transitive(Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),done),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),Nat.mul(U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),U32.to_nat(cap),
        Bounds.strict_le(Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),done),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),Nat.mul(U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),RowMajor.radix(Nat.add(Nat.mul(block,width),done),U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)),group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),outer_inside,group_inside)),cells_room)
      +outer_room = Order.transitive(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.add(Nat.mul(block,width),done)),Nat.mul(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),U32.to_nat(cap),
        Bounds.strict_le(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.add(Nat.mul(block,width),done)),Nat.mul(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),RowMajor.radix(group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),Nat.add(Nat.mul(block,width),done),U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)),group_inside,outer_inside)),
        Held.nat_le_moved(Nat.mul(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.mul(U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),U32.to_nat(cap),RowMajor.multiply_commutative(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),cells_room))
      +step_at = Addition.successor_right(Nat.mul(block,width),done)
      +left_next = Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),1n+done),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),Nat.add(Nat.mul(1n+Nat.add(Nat.mul(block,width),done),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),Nat.add(Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),done),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),Nat.add(Nat.mul(block,width),1n+done),1n+Nat.add(Nat.mul(block,width),done),step_at),
        swap_last(Nat.mul(Nat.add(Nat.mul(block,width),done),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))))
      +right_next = Equal.trans(Nat,Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.add(Nat.mul(block,width),1n+done)),Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),1n+Nat.add(Nat.mul(block,width),done)),1n+Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.add(Nat.mul(block,width),done)),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),value),Nat.add(Nat.mul(block,width),1n+done),1n+Nat.add(Nat.mul(block,width),done),step_at),
        Addition.successor_right(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.add(Nat.mul(block,width),done)))
      +later = row_agrees(~Left,~left_value,~Right,~right_value,~Moved,~moved,left_context,right_context,values,axis,cap,inside,blocks,width,outer_exact,cells_room,block,block_inside,group,group_inside,related,rest,1n+done,
        Equal.trans(Nat,Nat.add(1n+done,rest),Nat.add(done,1n+rest),width,Equal.sym(Nat,Nat.add(done,1n+rest),1n+Nat.add(done,rest),Addition.successor_right(done,rest)),fits))
      +shifted = {%left_next : Reduce.StridedAgrees(~Tile.Tile8,~Left,~left_value,~Right,~right_value,left_context,right_context,rest,_,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.add(Nat.mul(block,width),done)))
        {%right_next : Reduce.StridedAgrees(~Tile.Tile8,~Left,~left_value,~Right,~right_value,left_context,right_context,rest,Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),1n+done),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),_)
          later : Reduce.StridedAgrees(~Tile.Tile8,~Left,~left_value,~Right,~right_value,left_context,right_context,rest,Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),1n+done),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.add(Nat.mul(block,width),done)))} :
        Reduce.StridedAgrees(~Tile.Tile8,~Left,~left_value,~Right,~right_value,left_context,right_context,rest,Nat.add(Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),done),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.add(Nat.mul(block,width),done)))}
      KernelGemm.Both{moved(left_context,right_context,values,axis,group,cap,related,inside,Nat.add(Nat.mul(block,width),done),outer_inside,group_inside,inner_room,outer_room),shifted}

# Line group of a block of the cells with the groups innermost folds as the
# row of the group and the block among the cells with the groups outermost.
def row_folds(~Row: Data,~step: Row -> Tile.Tile8 -> Row,~Left: Data,~left_value: Left -> U32 -> Tile.Tile8,~Right: Data,~right_value: Right -> U32 -> Tile.Tile8,~Moved: @+left_context: Left -> @+right_context: Right -> @+values: List<&2,U32> -> @+axis: Nat -> @+group: Nat -> @+cap: U32 -> Data,~moved: @+left_context: Left -> @+right_context: Right -> @+values: List<&2,U32> -> @+axis: Nat -> @+group: Nat -> @+cap: U32 -> @+related: Moved(left_context,right_context,values,axis,group,cap) -> @+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool} -> @+outer: Nat -> @+outer_inside: {Nat.is_lt(outer,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))) == True{} : Bool} -> @+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))) == True{} : Bool} -> @+inner_room: {Nat.is_le(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(cap)) == True{} : Bool} -> @+outer_room: {Nat.is_le(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer),U32.to_nat(cap)) == True{} : Bool} -> {left_value(left_context,U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == right_value(right_context,U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8},+left_context: Left,+right_context: Right,+values: List<&2,U32>,+axis: Nat,+cap: U32,
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
  +blocks: Nat,+width: Nat,+outer_exact: {U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)) == Nat.mul(blocks,width) : Nat},+cells_room: {Nat.is_le(Nat.mul(U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32)),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),U32.to_nat(cap)) == True{} : Bool},
  +block: Nat,+block_inside: {Nat.is_lt(block,blocks) == True{} : Bool},+group: Nat,+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},+related: Moved(left_context,right_context,values,axis,group,cap),
  +size: Nat,+size_exact: {size == Nat.mul(width,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))) : Nat},+inner: Nat,+inner_exact: {inner == U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))) : Nat},+state: Row) ->
  {LineChainValues.taken_list(~Tile.Tile8,~Row,~step,Reduce.values_from(~Tile.Tile8,~Left,~left_value,left_context,size,Nat.mul(block,size)),group,inner,state) == Reduce.line_fold(~Tile.Tile8,~Row,~step,~Right,~right_value,right_context,width,Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.mul(block,width)),state) : Row}:
  +split = Order.split_at_element(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),group,group_inside)
  +inner_period = Equal.trans(Nat,inner,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group)),inner_exact,Equal.sym(Nat,Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group)),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),split))
  +size_period = Equal.trans(Nat,size,Nat.mul(width,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),Nat.mul(width,Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group))),size_exact,Equal.cong(Nat,Nat,count => Nat.mul(width,count),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group)),Equal.sym(Nat,Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group)),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),split)))
  +positive = Held.nat_lt_moved(0n,inner,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),inner_exact,Order.positive_bound(group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),group_inside))
  +counted = Equal.trans(Nat,Reduce.taken_count(size,group,inner),Reduce.taken_count(Nat.mul(width,Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group))),group,inner),width,
    Equal.cong(Nat,Nat,count => Reduce.taken_count(count,group,inner),size,Nat.mul(width,Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group))),size_period),
    Equal.trans(Nat,Reduce.taken_count(Nat.mul(width,Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group))),group,inner),Reduce.taken_count(Nat.mul(width,Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group))),group,Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group))),width,
      Equal.cong(Nat,Nat,period => Reduce.taken_count(Nat.mul(width,Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group))),group,period),inner,Nat.add(group,1n+Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group)),inner_period),
      Reduce.taken_whole(width,group,Nat.sub(U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),1n+group))))
  +start_exact = Addition.zero_right(Nat.mul(block,width))
  +left_exact = Equal.cong(Nat,Nat,first => Nat.add(first,group),Nat.mul(Nat.add(Nat.mul(block,width),0n),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),Nat.mul(block,size),
    Equal.trans(Nat,Nat.mul(Nat.add(Nat.mul(block,width),0n),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),Nat.mul(Nat.mul(block,width),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),Nat.mul(block,size),
      Equal.cong(Nat,Nat,first => Nat.mul(first,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),Nat.add(Nat.mul(block,width),0n),Nat.mul(block,width),start_exact),
      Equal.trans(Nat,Nat.mul(Nat.mul(block,width),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),Nat.mul(block,Nat.mul(width,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))))),Nat.mul(block,size),
        RowMajor.assoc(block,width,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),
        Equal.cong(Nat,Nat,count => Nat.mul(block,count),Nat.mul(width,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),size,Equal.sym(Nat,size,Nat.mul(width,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),size_exact)))))
  +right_exact = Equal.cong(Nat,Nat,first => Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),first),Nat.add(Nat.mul(block,width),0n),Nat.mul(block,width),start_exact)
  +agrees_start = row_agrees(~Left,~left_value,~Right,~right_value,~Moved,~moved,left_context,right_context,values,axis,cap,inside,blocks,width,outer_exact,cells_room,block,block_inside,group,group_inside,related,width,0n,{==})
  +agrees = {%left_exact : Reduce.StridedAgrees(~Tile.Tile8,~Left,~left_value,~Right,~right_value,left_context,right_context,width,_,inner,Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.mul(block,width)))
    {%Equal.sym(Nat,inner,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),inner_exact) : Reduce.StridedAgrees(~Tile.Tile8,~Left,~left_value,~Right,~right_value,left_context,right_context,width,Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),0n),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),_,Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.mul(block,width)))
      {%right_exact : Reduce.StridedAgrees(~Tile.Tile8,~Left,~left_value,~Right,~right_value,left_context,right_context,width,Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),0n),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),_)
        agrees_start : Reduce.StridedAgrees(~Tile.Tile8,~Left,~left_value,~Right,~right_value,left_context,right_context,width,Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),0n),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis))),Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.mul(block,width)))} :
      Reduce.StridedAgrees(~Tile.Tile8,~Left,~left_value,~Right,~right_value,left_context,right_context,width,Nat.add(Nat.mul(Nat.add(Nat.mul(block,width),0n),U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),inner,Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.mul(block,width)))} :
    Reduce.StridedAgrees(~Tile.Tile8,~Left,~left_value,~Right,~right_value,left_context,right_context,width,Nat.add(Nat.mul(block,size),group),inner,Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.mul(block,width)))}
  Equal.trans(Row,LineChainValues.taken_list(~Tile.Tile8,~Row,~step,Reduce.values_from(~Tile.Tile8,~Left,~left_value,left_context,size,Nat.mul(block,size)),group,inner,state),Reduce.strided_fold(~Tile.Tile8,~Row,~step,~Left,~left_value,left_context,Reduce.taken_count(size,group,inner),Nat.add(Nat.mul(block,size),group),inner,state),Reduce.line_fold(~Tile.Tile8,~Row,~step,~Right,~right_value,right_context,width,Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.mul(block,width)),state),
    Reduce.taken_strided(~Tile.Tile8,~Row,~step,~Left,~left_value,left_context,size,Nat.mul(block,size),group,inner,state,positive),
    Equal.trans(Row,Reduce.strided_fold(~Tile.Tile8,~Row,~step,~Left,~left_value,left_context,Reduce.taken_count(size,group,inner),Nat.add(Nat.mul(block,size),group),inner,state),Reduce.strided_fold(~Tile.Tile8,~Row,~step,~Left,~left_value,left_context,width,Nat.add(Nat.mul(block,size),group),inner,state),Reduce.line_fold(~Tile.Tile8,~Row,~step,~Right,~right_value,right_context,width,Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.mul(block,width)),state),
      Equal.cong(Nat,Row,count => Reduce.strided_fold(~Tile.Tile8,~Row,~step,~Left,~left_value,left_context,count,Nat.add(Nat.mul(block,size),group),inner,state),Reduce.taken_count(size,group,inner),width,counted),
      Reduce.strided_line(~Tile.Tile8,~Row,~step,~Left,~left_value,~Right,~right_value,left_context,right_context,width,Nat.add(Nat.mul(block,size),group),inner,Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),Nat.mul(block,width)),state,agrees)))


# ---- The specification ----

# The box of the sums of the statistics of values [N,C,H,W]: a row is the
# H x W plane of an image and a channel, the rows of the images merged.
def plane_box(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32) -> PartialsValues.Box<Sum.Parts>:
  PartialsValues.box_at(~F32,~Sum.Lanes,~Sum.Parts,~Sum.step,~Sum.parts,~Logical,~original,True{},logical,Sum.zero(),TiledReduce.parts_zero(),[images,channels,height,width],[0n,2n,3n],2n)

# The tiled values of the sums of the statistics: at a channel, the rounded
# merge of the partials of its planes.
def PlaneSummed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32,tiled: Tiled.Tiled) -> Data:
  Public.Contract(~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),plane_box(~Logical,~original,logical,images,channels,height,width),[channels],tiled)

# ---- The rows of the statistics ----

# The partials of the planes of the cells [N,H,W,G] of values [N,C,H,W], a
# line of a group of an image folded into slots from the start of the group.
def plane_folds(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,~X: Data,~Source: Data,~source_value: Source -> U32 -> Tile.Tile8,~source_at: X -> Cells.Cells -> Source,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,+starts: List<&2,SlotsValues.SlotsView<Extra>>) ->
  List<&2,Sum.TileParts>:
  Reduce.block_folds(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~Sum.TileParts,~SlotsValues.cell_view(~Extra,~point),~SlotsValues.turn_view(~Extra),ShapeLaws.elements([images]),
    Grid.product([height,width]),Grid.product([Tiled.groups(channels)]),
    Reduce.values_from(~Tile.Tile8,~Source,~source_value,source_at(x,Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{})),
      ShapeLaws.elements([images,height,width,Tiled.groups(channels)]),0n),starts)

# The partials of the planes of the values.
def plane_lines(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32) -> List<&2,Sum.Parts>:
  LineChainValues.turned_values(~Sum.Lanes,~Sum.Parts,~Sum.parts,Reduce.line_folds(~F32,~Sum.Lanes,~Sum.step,~Logical,~original,logical,ShapeLaws.elements([images,channels]),ShapeLaws.elements([height,width]),0n,Sum.zero()))

# The values of a plane of the cells, a slot of the cells per value, are no
# more than the values of the cells.
def plane_size(+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,
  +room: {Nat.is_le(ShapeLaws.elements([images,height,width,Tiled.groups(channels)]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements([images,height,width,Tiled.groups(channels)])) == True{} : Bool}) ->
  {U32.to_nat((Grid.product([height,width]) * Grid.product([Tiled.groups(channels)]) : U32)) == Nat.mul(ShapeLaws.elements([height,width]),U32.to_nat(Tiled.groups(channels))) : Nat}:
  +shape = {[images,height,width,Tiled.groups(channels)] : List<&2,U32>}
  +plane = PartialsValues.tail_word(limits,[images,height,width],1n,
    Order.transitive(ShapeLaws.elements([images,height,width]),ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits)),TiledSums.removed_fewer(shape,3n,CountCompletion.positive_not_zero(shape,positive)),room),
    CountCompletion.positive_elements([images,height,width],
      TiledSums.removed_nonzero(shape,3n,CountCompletion.positive_not_zero(shape,positive))))
  +groups = Equal.cong(U32,Nat,word => U32.to_nat(word),Grid.product([Tiled.groups(channels)]),Tiled.groups(channels),Arithmetic.multiply_one(Tiled.groups(channels)))
  +whole = Equal.trans(Nat,Nat.mul(ShapeLaws.elements([height,width]),U32.to_nat(Tiled.groups(channels))),Nat.mul(ShapeLaws.elements([height,width]),ShapeLaws.elements([Tiled.groups(channels)])),ShapeLaws.elements([height,width,Tiled.groups(channels)]),
    Equal.cong(Nat,Nat,count => Nat.mul(ShapeLaws.elements([height,width]),count),U32.to_nat(Tiled.groups(channels)),ShapeLaws.elements([Tiled.groups(channels)]),Equal.sym(Nat,Nat.mul(U32.to_nat(Tiled.groups(channels)),1n),U32.to_nat(Tiled.groups(channels)),Order.times_one(U32.to_nat(Tiled.groups(channels))))),
    Equal.sym(Nat,ShapeLaws.elements([height,width,Tiled.groups(channels)]),Nat.mul(ShapeLaws.elements([height,width]),ShapeLaws.elements([Tiled.groups(channels)])),CountCompletion.elements_split([height,width,Tiled.groups(channels)],2n)))
  WordMultiplication.multiply_exact(Grid.product([height,width]),Grid.product([Tiled.groups(channels)]),ShapeLaws.elements([height,width]),U32.to_nat(Tiled.groups(channels)),Machine.capacity(limits),Limits.capacity_fits(limits),
    plane,groups,
    {%Equal.sym(Nat,Nat.mul(ShapeLaws.elements([height,width]),U32.to_nat(Tiled.groups(channels))),ShapeLaws.elements([height,width,Tiled.groups(channels)]),whole) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      Order.transitive(ShapeLaws.elements([height,width,Tiled.groups(channels)]),ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits)),PartialsValues.tail_room(shape,1n,positive),room) :
      {Nat.is_le(Nat.mul(ShapeLaws.elements([height,width]),U32.to_nat(Tiled.groups(channels))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})

# The values of a plane of a group of the cells are the values of the plane
# times the groups.
def plane_whole(+channels: U32,+height: U32,+width: U32) ->
  {Nat.mul(ShapeLaws.elements([height,width]),U32.to_nat(Tiled.groups(channels))) == ShapeLaws.elements([height,width,Tiled.groups(channels)]) : Nat}:
  Equal.trans(Nat,Nat.mul(ShapeLaws.elements([height,width]),U32.to_nat(Tiled.groups(channels))),Nat.mul(ShapeLaws.elements([height,width]),ShapeLaws.elements([Tiled.groups(channels)])),ShapeLaws.elements([height,width,Tiled.groups(channels)]),
    Equal.cong(Nat,Nat,count => Nat.mul(ShapeLaws.elements([height,width]),count),U32.to_nat(Tiled.groups(channels)),ShapeLaws.elements([Tiled.groups(channels)]),Equal.sym(Nat,Nat.mul(U32.to_nat(Tiled.groups(channels)),1n),U32.to_nat(Tiled.groups(channels)),Order.times_one(U32.to_nat(Tiled.groups(channels))))),
    Equal.sym(Nat,ShapeLaws.elements([height,width,Tiled.groups(channels)]),Nat.mul(ShapeLaws.elements([height,width]),ShapeLaws.elements([Tiled.groups(channels)])),CountCompletion.elements_split([height,width,Tiled.groups(channels)],2n)))

# The cells of values [N,C,H,W] outside a lane are the images times their
# planes.
def plane_outer(+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,
  +room: {Nat.is_le(ShapeLaws.elements([images,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32)) == Nat.mul(ShapeLaws.elements([images]),ShapeLaws.elements([height,width])) : Nat}:
  Equal.trans(Nat,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32)),ShapeLaws.elements([images,height,width]),
    Nat.mul(ShapeLaws.elements([images]),ShapeLaws.elements([height,width])),
    Coordinates.outer_word(limits,[images,channels,height,width],1n,room),CountCompletion.elements_split([images,height,width],1n))

# Lane digit of the slots of the plane of an image and a group of the cells is
# the sum of the plane of the image and the channel of the lane, when the
# lanes of the cells agree with the values.
def plane_row(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,~X: Data,~Source: Data,~source_value: Source -> U32 -> Tile.Tile8,~source_at: X -> Cells.Cells -> Source,~source_moved: SourceMoved(~X,~Source,~source_value,~source_at),~Logical: Data,~original: Logical -> U32 -> F32,+x: X,+logical: Logical,
  +limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+starts: List<&2,SlotsValues.SlotsView<Extra>>,+extra: Extra,+image: Nat,+group: Nat,+digit: Nat,
  +room: {Nat.is_le(ShapeLaws.elements([images,height,width,Tiled.groups(channels)]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements([images,height,width,Tiled.groups(channels)])) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,ShapeLaws.elements([images])) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool},
  +row_inside: {Nat.is_lt(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),ShapeLaws.elements([images,channels])) == True{} : Bool},
  +started: {ListReadback.at(~SlotsValues.SlotsView<Extra>,SlotsValues.begin_view(~Extra,extra),group,starts) == SlotsValues.begin_view(~Extra,extra) : SlotsValues.SlotsView<Extra>},
  +enough: {Nat.is_le(U32.to_nat(Grid.product([Tiled.groups(channels)])),List.length(&2,SlotsValues.SlotsView<Extra>,starts)) == True{} : Bool},
  +agrees: ReduceLaws.Agrees(~SlotsLanes.LanePoint<Extra,Source>,~SlotsLanes.lane_point(~Extra,~point,~Source,~source_value),~Logical,~original,SlotsLanes.LanePoint{digit,extra,source_at(x,Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{}))},logical,
    ShapeLaws.elements([height,width]),Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32))),Nat.mul(image,ShapeLaws.elements([height,width]))),Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),ShapeLaws.elements([height,width])))) ->
  {SlotsLanes.parts_lane(ListReadback.at(~Sum.TileParts,Sum.tile_parts_zero(),Nat.add(Nat.mul(image,U32.to_nat(Grid.product([Tiled.groups(channels)]))),group),
      plane_folds(~Extra,~point,~X,~Source,~source_value,~source_at,x,images,channels,height,width,starts)),digit) ==
    ListReadback.at(~Sum.Parts,TiledReduce.parts_zero(),Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),plane_lines(~Logical,~original,logical,images,channels,height,width)) : Sum.Parts}:
  +values = {[images,channels,height,width] : List<&2,U32>}
  +cells = {[images,height,width,Tiled.groups(channels)] : List<&2,U32>}
  +inner = {source_at(x,Public.geometry_of(values,1n,Grid.Innermost{})) : Source}
  +outer = {source_at(x,Public.geometry_of(values,1n,Grid.Outermost{})) : Source}
  +lines = U32.to_nat(Grid.product([Tiled.groups(channels)]))
  +size = U32.to_nat((Grid.product([height,width]) * Grid.product([Tiled.groups(channels)]) : U32))
  +plane = ShapeLaws.elements([height,width])
  +groups = U32.to_nat(Tiled.groups(channels))
  +outer_count = U32.to_nat((Public.lane_prefix(values,1n,1) * Tiled.product(Public.lane_after(values,1n)) : U32))
  +row = Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n)))
  +start = SlotsValues.begin_view(~Extra,extra)
  +cells_c = Reduce.values_from(~Tile.Tile8,~Source,~source_value,inner,ShapeLaws.elements(cells),0n)
  +nonzero = CountCompletion.positive_not_zero(cells,positive)
  +outer_room = Order.transitive(ShapeLaws.elements([images,height,width]),ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits)),TiledSums.removed_fewer(cells,3n,nonzero),room)
  +outer_exact = plane_outer(limits,images,channels,height,width,outer_room)
  +size_exact = plane_size(limits,images,channels,height,width,room,positive)
  +lines_exact = Equal.cong(U32,Nat,word => U32.to_nat(word),Grid.product([Tiled.groups(channels)]),Tiled.groups(channels),Arithmetic.multiply_one(Tiled.groups(channels)))
  +line_inside = {%Equal.sym(Nat,lines,groups,lines_exact) : {Nat.is_lt(group,_) == True{} : Bool} group_inside : {Nat.is_lt(group,lines) == True{} : Bool}}
  +count_exact = Equal.trans(Nat,ShapeLaws.elements(cells),Nat.mul(ShapeLaws.elements([images]),ShapeLaws.elements([height,width,Tiled.groups(channels)])),Nat.mul(ShapeLaws.elements([images]),size),
    CountCompletion.elements_split(cells,1n),
    Equal.cong(Nat,Nat,count => Nat.mul(ShapeLaws.elements([images]),count),ShapeLaws.elements([height,width,Tiled.groups(channels)]),size,
      Equal.sym(Nat,size,ShapeLaws.elements([height,width,Tiled.groups(channels)]),Equal.trans(Nat,size,Nat.mul(plane,groups),ShapeLaws.elements([height,width,Tiled.groups(channels)]),size_exact,plane_whole(channels,height,width)))))
  +cells_room = {%Equal.sym(Nat,outer_count,Nat.mul(ShapeLaws.elements([images]),plane),outer_exact) : {Nat.is_le(Nat.mul(_,groups),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    {%Equal.sym(Nat,Nat.mul(Nat.mul(ShapeLaws.elements([images]),plane),groups),Nat.mul(ShapeLaws.elements([images]),Nat.mul(plane,groups)),RowMajor.assoc(ShapeLaws.elements([images]),plane,groups)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      {%Equal.sym(Nat,Nat.mul(plane,groups),ShapeLaws.elements([height,width,Tiled.groups(channels)]),plane_whole(channels,height,width)) : {Nat.is_le(Nat.mul(ShapeLaws.elements([images]),_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
        {%CountCompletion.elements_split(cells,1n) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
          room : {Nat.is_le(Nat.mul(ShapeLaws.elements([images]),ShapeLaws.elements([height,width,Tiled.groups(channels)])),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}} :
        {Nat.is_le(Nat.mul(ShapeLaws.elements([images]),Nat.mul(plane,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}} :
      {Nat.is_le(Nat.mul(Nat.mul(ShapeLaws.elements([images]),plane),groups),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}} :
    {Nat.is_le(Nat.mul(outer_count,groups),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +position_c = Nat.add(Nat.mul(group,outer_count),Nat.mul(image,plane))
  +position_v = Nat.mul(row,plane)
  +lane_at = {SlotsLanes.LanePoint{digit,extra,outer} : SlotsLanes.LanePoint<Extra,Source>}
  +read_c = Reduce.line_fold(~F32,~Sum.Lanes,~Sum.step,~SlotsLanes.LanePoint<Extra,Source>,~SlotsLanes.lane_point(~Extra,~point,~Source,~source_value),lane_at,plane,position_c,Sum.zero())
  +read_v = Reduce.line_fold(~F32,~Sum.Lanes,~Sum.step,~Logical,~original,logical,plane,position_v,Sum.zero())
  +walk = LineChainValues.taken_list(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~SlotsValues.cell_view(~Extra,~point),Reduce.values_from(~Tile.Tile8,~Source,~source_value,inner,size,Nat.mul(image,size)),group,lines,start)
  +line_c = Reduce.line_fold(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~SlotsValues.cell_view(~Extra,~point),~Source,~source_value,outer,plane,position_c,start)
  %Equal.sym(Nat,ShapeLaws.elements(cells),Nat.mul(ShapeLaws.elements([images]),size),count_exact) : {SlotsLanes.parts_lane(ListReadback.at(~Sum.TileParts,Sum.tile_parts_zero(),Nat.add(Nat.mul(image,lines),group),
      Reduce.block_folds(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~Sum.TileParts,~SlotsValues.cell_view(~Extra,~point),~SlotsValues.turn_view(~Extra),ShapeLaws.elements([images]),
        Grid.product([height,width]),Grid.product([Tiled.groups(channels)]),Reduce.values_from(~Tile.Tile8,~Source,~source_value,inner,_,0n),starts)),digit) ==
    ListReadback.at(~Sum.Parts,TiledReduce.parts_zero(),row,plane_lines(~Logical,~original,logical,images,channels,height,width)) : Sum.Parts}
  Equal.trans(Sum.Parts,SlotsLanes.parts_lane(ListReadback.at(~Sum.TileParts,Sum.tile_parts_zero(),Nat.add(Nat.mul(image,lines),group),
      Reduce.block_folds(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~Sum.TileParts,~SlotsValues.cell_view(~Extra,~point),~SlotsValues.turn_view(~Extra),ShapeLaws.elements([images]),
        Grid.product([height,width]),Grid.product([Tiled.groups(channels)]),Reduce.values_from(~Tile.Tile8,~Source,~source_value,inner,Nat.mul(ShapeLaws.elements([images]),size),0n),starts)),digit),
    SlotsLanes.parts_lane(SlotsValues.turn_view(~Extra,line_c),digit),
    ListReadback.at(~Sum.Parts,TiledReduce.parts_zero(),row,plane_lines(~Logical,~original,logical,images,channels,height,width)),
    Equal.cong(Sum.TileParts,Sum.Parts,parts => SlotsLanes.parts_lane(parts,digit),
      ListReadback.at(~Sum.TileParts,Sum.tile_parts_zero(),Nat.add(Nat.mul(image,lines),group),
        Reduce.block_folds(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~Sum.TileParts,~SlotsValues.cell_view(~Extra,~point),~SlotsValues.turn_view(~Extra),ShapeLaws.elements([images]),
          Grid.product([height,width]),Grid.product([Tiled.groups(channels)]),Reduce.values_from(~Tile.Tile8,~Source,~source_value,inner,Nat.mul(ShapeLaws.elements([images]),size),0n),starts)),
      SlotsValues.turn_view(~Extra,line_c),
      Equal.trans(Sum.TileParts,ListReadback.at(~Sum.TileParts,Sum.tile_parts_zero(),Nat.add(Nat.mul(image,lines),group),
          Reduce.block_folds(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~Sum.TileParts,~SlotsValues.cell_view(~Extra,~point),~SlotsValues.turn_view(~Extra),ShapeLaws.elements([images]),
            Grid.product([height,width]),Grid.product([Tiled.groups(channels)]),Reduce.values_from(~Tile.Tile8,~Source,~source_value,inner,Nat.mul(ShapeLaws.elements([images]),size),0n),starts)),
        SlotsValues.turn_view(~Extra,walk),SlotsValues.turn_view(~Extra,line_c),
        %started :
          {ListReadback.at(~Sum.TileParts,Sum.tile_parts_zero(),Nat.add(Nat.mul(image,lines),group),
              Reduce.block_folds(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~Sum.TileParts,~SlotsValues.cell_view(~Extra,~point),~SlotsValues.turn_view(~Extra),ShapeLaws.elements([images]),
                Grid.product([height,width]),Grid.product([Tiled.groups(channels)]),Reduce.values_from(~Tile.Tile8,~Source,~source_value,inner,Nat.mul(ShapeLaws.elements([images]),size),0n),starts)) ==
            SlotsValues.turn_view(~Extra,LineChainValues.taken_list(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~SlotsValues.cell_view(~Extra,~point),Reduce.values_from(~Tile.Tile8,~Source,~source_value,inner,size,Nat.mul(image,size)),group,lines,_)) : Sum.TileParts}
        PartialsValues.block_read(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~Sum.TileParts,~SlotsValues.cell_view(~Extra,~point),~SlotsValues.turn_view(~Extra),~Source,~source_value,inner,
          ShapeLaws.elements([images]),Grid.product([height,width]),Grid.product([Tiled.groups(channels)]),0n,starts,Sum.tile_parts_zero(),start,image,group,image_inside,line_inside,enough),
        Equal.cong(SlotsValues.SlotsView<Extra>,Sum.TileParts,view => SlotsValues.turn_view(~Extra,view),walk,line_c,
          row_folds(~SlotsValues.SlotsView<Extra>,~SlotsValues.cell_view(~Extra,~point),~Source,~source_value,~Source,~source_value,
            ~source_inner(~X,~Source,~source_at,x),~source_carried(~X,~Source,~source_value,~source_at,~source_moved,x),inner,outer,values,1n,Machine.capacity(limits),{==},ShapeLaws.elements([images]),plane,outer_exact,cells_room,
            image,image_inside,group,group_inside,KernelGemm.Both{{==},{==}},size,size_exact,lines,lines_exact,start)))),
    Equal.trans(Sum.Parts,SlotsLanes.parts_lane(SlotsValues.turn_view(~Extra,line_c),digit),Sum.parts(read_c),
      ListReadback.at(~Sum.Parts,TiledReduce.parts_zero(),row,plane_lines(~Logical,~original,logical,images,channels,height,width)),
      SlotsLanes.line_lane(~Extra,~point,~Source,~source_value,outer,plane,position_c,extra,digit),
      Equal.trans(Sum.Parts,Sum.parts(read_c),Sum.parts(read_v),ListReadback.at(~Sum.Parts,TiledReduce.parts_zero(),row,plane_lines(~Logical,~original,logical,images,channels,height,width)),
        Equal.cong(Sum.Lanes,Sum.Parts,lanes => Sum.parts(lanes),read_c,read_v,
          ReduceLaws.folds_agree(~Sum.Lanes,~Sum.step,~SlotsLanes.LanePoint<Extra,Source>,~SlotsLanes.lane_point(~Extra,~point,~Source,~source_value),~Logical,~original,
            lane_at,logical,plane,position_c,position_v,Sum.zero(),agrees)),
        Equal.sym(Sum.Parts,ListReadback.at(~Sum.Parts,TiledReduce.parts_zero(),row,plane_lines(~Logical,~original,logical,images,channels,height,width)),Sum.parts(read_v),
          PartialsValues.closed_at(~F32,~Sum.Lanes,~Sum.Parts,~Sum.step,~Sum.parts,~Logical,~original,logical,ShapeLaws.elements([images,channels]),plane,0n,Sum.zero(),row,TiledReduce.parts_zero(),row_inside)))))

# ---- The lanes of the statistics ----

# The partial of a row and a kept index of a box [rows, kept] lies at the row
# times the kept extent plus the index.
def pair_address(+rows: U32,+kept: U32,+index: Nat,+row: Nat) ->
  {ShapeLaws.digits_value([rows,kept],ReduceLaws.merged([True{},False{}],Targets.cell_coordinates([kept],index),[row])) == Nat.add(Nat.mul(row,U32.to_nat(kept)),index) : Nat}:
  +quotient = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(index,1n),(index,0n),BoxCells.by_one(index))
  +last = Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(index,1n),1n),0n),Nat.mul(Nat.div(index,1n),1n),index,Addition.zero_right(Nat.mul(Nat.div(index,1n),1n)),
    Equal.trans(Nat,Nat.mul(Nat.div(index,1n),1n),Nat.div(index,1n),index,Order.times_one(Nat.div(index,1n)),quotient))
  Equal.trans(Nat,Nat.add(Nat.mul(row,Nat.mul(U32.to_nat(kept),1n)),Nat.add(Nat.mul(Nat.div(index,1n),1n),0n)),Nat.add(Nat.mul(row,U32.to_nat(kept)),Nat.add(Nat.mul(Nat.div(index,1n),1n),0n)),Nat.add(Nat.mul(row,U32.to_nat(kept)),index),
    Equal.cong(Nat,Nat,extent => Nat.add(Nat.mul(row,extent),Nat.add(Nat.mul(Nat.div(index,1n),1n),0n)),Nat.mul(U32.to_nat(kept),1n),U32.to_nat(kept),Order.times_one(U32.to_nat(kept))),
    Equal.cong(Nat,Nat,later => Nat.add(Nat.mul(row,U32.to_nat(kept)),later),Nat.add(Nat.mul(Nat.div(index,1n),1n),0n),index,last))

# No digits lie below no extents.
def none_below(others: List<&2,Nat>,+below: Coordinates.DigitsBelow([],others)) -> {others == [] : List<&2,Nat>}:
  match others:
    case Nil{}: {==}
    case Con{+digit,+more}: Empty.absurd({digit <> more == [] : List<&2,Nat>},below)

# What the statistics of a batch normalization read: the values of the
# specification and the extents [N,C,H,W].
type Plane<-Logical: Data> is Data:
  Plane{logical: Logical,images: U32,channels: U32,height: U32,width: U32}

# A plane specification fits a reduction of axes [0, 2, 3] of its extents,
# the groups of the channels outermost, when the instance is ready for it.
def plane_fits(~Logical: Data,~Ready: Logical -> U32 -> U32 -> U32 -> U32 -> Data,spec: Plane<Logical>,values: List<&2,U32>,axes: List<&2,Nat>,lane: Nat,position: Grid.Position) -> Data:
  Plane{+logical,+images,+channels,+height,+width} = spec
  KernelGemm.Both<{values == [images,channels,height,width] : List<&2,U32>},KernelGemm.Both<{axes == [0n,2n,3n] : List<&2,Nat>},KernelGemm.Both<{lane == 1n : Nat},
    KernelGemm.Both<{position == Grid.Outermost{} : Grid.Position},Ready(logical,images,channels,height,width)>>>>

# The cells the statistics read: the slots of the planes of the input from the
# starts of the groups, the images merged.
def plane_cells(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,~X: Data,~Source: Data,~source_value: Source -> U32 -> Tile.Tile8,~source_at: X -> Cells.Cells -> Source,~Logical: Data,~input_of: Logical -> X,~extras_of: Logical -> U32 -> List<&2,Extra>,
  +spec: Plane<Logical>,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position) -> PartialsValues.Box<Sum.TileParts>:
  Plane{+logical,+images,+channels,+height,+width} = spec
  PartialsValues.Box{PartialsValues.written(~Sum.TileParts,plane_folds(~Extra,~point,~X,~Source,~source_value,~source_at,input_of(logical),images,channels,height,width,
      Reduce.block_starts(~Extra,~SlotsValues.SlotsView<Extra>,~SlotsValues.begin_view(~Extra),extras_of(logical,channels),Grid.product([Tiled.groups(channels)]))),Sum.tile_parts_zero()),
    PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],False{}),PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],True{}),0,Sum.tile_parts_zero()}

# The sums the statistics specify.
def plane_values(~Logical: Data,~original: Logical -> U32 -> F32,+spec: Plane<Logical>,+values: List<&2,U32>,+axes: List<&2,Nat>) -> PartialsValues.Box<Sum.Parts>:
  Plane{+logical,+images,+channels,+height,+width} = spec
  plane_box(~Logical,~original,logical,images,channels,height,width)

# The lane digit of a group of a kept cell of a fitting reduction is the
# channel of the group and the digit.
def plane_channel(+images: U32,+channels: U32,+height: U32,+width: U32,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+others: List<&2,Nat>,+digit: Nat,+group: Nat,
  +values_eq: {values == [images,channels,height,width] : List<&2,U32>},+axes_eq: {axes == [0n,2n,3n] : List<&2,Nat>},+lane_eq: {lane == 1n : Nat},+none: {others == [] : List<&2,Nat>}) ->
  {ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))) ==
    Nat.add(digit,Nat.mul(group,8n)) : Nat}:
  +shape = {[images,channels,height,width] : List<&2,U32>}
  +channel = Nat.add(digit,Nat.mul(group,8n))
  %Equal.sym(List<&2,U32>,values,shape,values_eq) :
    {ShapeLaws.digits_value(Layout.extents_at(_,Layout.axes_where(List.length(&2,U32,_),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,_),0n,axes,False{}),lane,0n),others,channel)) == channel : Nat}
  %Equal.sym(List<&2,Nat>,axes,[0n,2n,3n],axes_eq) :
    {ShapeLaws.digits_value(Layout.extents_at(shape,Layout.axes_where(4n,0n,_,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(4n,0n,_,False{}),lane,0n),others,channel)) == channel : Nat}
  %Equal.sym(Nat,lane,1n,lane_eq) :
    {ShapeLaws.digits_value(Layout.extents_at(shape,Layout.axes_where(4n,0n,[0n,2n,3n],False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(4n,0n,[0n,2n,3n],False{}),_,0n),others,channel)) == channel : Nat}
  %Equal.sym(List<&2,Nat>,others,[],none) :
    {ShapeLaws.digits_value([channels],Coordinates.inserted_digit(0n,_,channel)) == channel : Nat}
  Equal.trans(Nat,Nat.add(Nat.mul(channel,1n),0n),Nat.mul(channel,1n),channel,Addition.zero_right(Nat.mul(channel,1n)),Order.times_one(channel))

# The kept cells of a fitting reduction have no other kept axes.
def plane_none(+images: U32,+channels: U32,+height: U32,+width: U32,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+others: List<&2,Nat>,
  +values_eq: {values == [images,channels,height,width] : List<&2,U32>},+axes_eq: {axes == [0n,2n,3n] : List<&2,Nat>},+lane_eq: {lane == 1n : Nat},
  +others_below: Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),others)) -> {others == [] : List<&2,Nat>}:
  +shape = {[images,channels,height,width] : List<&2,U32>}
  +below = {%values_eq : Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(_,Layout.axes_where(List.length(&2,U32,_),0n,[0n,2n,3n],False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,_),0n,[0n,2n,3n],False{}),1n,0n)),others)
    {%axes_eq : Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,_,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,_,False{}),1n,0n)),others)
    {%lane_eq : Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),_,0n)),others)
    others_below : Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),1n,0n)),others)} : Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,[0n,2n,3n],False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,[0n,2n,3n],False{}),1n,0n)),others)} : Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,[0n,2n,3n],False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,shape),0n,[0n,2n,3n],False{}),1n,0n)),others)}
  none_below(others,below)

# The output word of a kept cell of a fitting reduction is its group.
def plane_word(+images: U32,+channels: U32,+height: U32,+width: U32,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,+word: U32,+others: List<&2,Nat>,+group: Nat,
  +values_eq: {values == [images,channels,height,width] : List<&2,U32>},+axes_eq: {axes == [0n,2n,3n] : List<&2,Nat>},+lane_eq: {lane == 1n : Nat},+position_eq: {position == Grid.Outermost{} : Grid.Position},+none: {others == [] : List<&2,Nat>},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group)) : Nat}) -> {U32.to_nat(word) == group : Nat}:
  +shape = {[images,channels,height,width] : List<&2,U32>}
  +word_exact = {%values_eq : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(_,Layout.axes_where(List.length(&2,U32,_),0n,[0n,2n,3n],False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,_),0n,[0n,2n,3n],False{}),1n,0n),Grid.Outermost{}),Coordinates.cell_digits(Grid.Outermost{},[],group)) : Nat}
    {%axes_eq : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,_,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,_,False{}),1n,0n),Grid.Outermost{}),Coordinates.cell_digits(Grid.Outermost{},[],group)) : Nat}
    {%lane_eq : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),_,0n),Grid.Outermost{}),Coordinates.cell_digits(Grid.Outermost{},[],group)) : Nat}
    {%position_eq : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),_),Coordinates.cell_digits(_,[],group)) : Nat}
    {%none : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,_,group)) : Nat}
    exact : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,[],group)) : Nat}} : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),Grid.Outermost{}),Coordinates.cell_digits(Grid.Outermost{},[],group)) : Nat}} : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),1n,0n),Grid.Outermost{}),Coordinates.cell_digits(Grid.Outermost{},[],group)) : Nat}} : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,[0n,2n,3n],False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,[0n,2n,3n],False{}),1n,0n),Grid.Outermost{}),Coordinates.cell_digits(Grid.Outermost{},[],group)) : Nat}} : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(shape,Layout.axes_where(List.length(&2,U32,shape),0n,[0n,2n,3n],False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,shape),0n,[0n,2n,3n],False{}),1n,0n),Grid.Outermost{}),Coordinates.cell_digits(Grid.Outermost{},[],group)) : Nat}}
  Equal.trans(Nat,U32.to_nat(word),Nat.add(Nat.mul(group,1n),0n),group,word_exact,Equal.trans(Nat,Nat.add(Nat.mul(group,1n),0n),Nat.mul(group,1n),group,Addition.zero_right(Nat.mul(group,1n)),Order.times_one(group)))

# The values of two extents are the values of the first times the second.
def pair_elements(+first: U32,+second: U32) -> {ShapeLaws.elements([first,second]) == Nat.mul(ShapeLaws.elements([first]),U32.to_nat(second)) : Nat}:
  Equal.trans(Nat,Nat.mul(U32.to_nat(first),Nat.mul(U32.to_nat(second),1n)),Nat.mul(U32.to_nat(first),U32.to_nat(second)),Nat.mul(Nat.mul(U32.to_nat(first),1n),U32.to_nat(second)),
    Equal.cong(Nat,Nat,count => Nat.mul(U32.to_nat(first),count),Nat.mul(U32.to_nat(second),1n),U32.to_nat(second),Order.times_one(U32.to_nat(second))),
    Equal.cong(Nat,Nat,count => Nat.mul(count,U32.to_nat(second)),U32.to_nat(first),Nat.mul(U32.to_nat(first),1n),Equal.sym(Nat,Nat.mul(U32.to_nat(first),1n),U32.to_nat(first),Order.times_one(U32.to_nat(first)))))

# A group starts its plane from the begin of its extra.
def PlaneStarted(~Extra: Data,~Logical: Data,~extras_of: Logical -> U32 -> List<&2,Extra>,~extra_of: Logical -> U32 -> Nat -> Extra) -> Type:
  @+logical: Logical -> @+channels: U32 -> @+group: Nat -> @+inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool} ->
    {ListReadback.at(~SlotsValues.SlotsView<Extra>,SlotsValues.begin_view(~Extra,extra_of(logical,channels,group)),group,
        Reduce.block_starts(~Extra,~SlotsValues.SlotsView<Extra>,~SlotsValues.begin_view(~Extra),extras_of(logical,channels),Grid.product([Tiled.groups(channels)]))) ==
      SlotsValues.begin_view(~Extra,extra_of(logical,channels,group)) : SlotsValues.SlotsView<Extra>}

# Every group has a start.
def PlaneEnough(~Extra: Data,~Logical: Data,~extras_of: Logical -> U32 -> List<&2,Extra>) -> Type:
  @+logical: Logical -> @+channels: U32 ->
    {Nat.is_le(U32.to_nat(Grid.product([Tiled.groups(channels)])),List.length(&2,SlotsValues.SlotsView<Extra>,
      Reduce.block_starts(~Extra,~SlotsValues.SlotsView<Extra>,~SlotsValues.begin_view(~Extra),extras_of(logical,channels),Grid.product([Tiled.groups(channels)])))) == True{} : Bool}

# A lane of the plane of an image and a group of the cells, as the point takes
# it, agrees with the plane of the image and the channel of the lane.
def PlaneAgrees(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,~X: Data,~Source: Data,~source_value: Source -> U32 -> Tile.Tile8,~source_at: X -> Cells.Cells -> Source,~Logical: Data,~original: Logical -> U32 -> F32,~input_of: Logical -> X,~extra_of: Logical -> U32 -> Nat -> Extra,~Ready: Logical -> U32 -> U32 -> U32 -> U32 -> Data) -> Type:
  @+logical: Logical -> @+limits: Machine.Limits -> @+images: U32 -> @+channels: U32 -> @+height: U32 -> @+width: U32 -> @+image: Nat -> @+group: Nat -> @+digit: Nat ->
  @+room: {Nat.is_le(ShapeLaws.elements([images,channels,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} -> @+cells_room: {Nat.is_le(ShapeLaws.elements([Tiled.groups(channels),images,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} ->
  @+ready: Ready(logical,images,channels,height,width) -> @+image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool} -> @+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool} ->
  @+small: {Nat.is_le(digit,7n) == True{} : Bool} -> @+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool} ->
    ReduceLaws.Agrees(~SlotsLanes.LanePoint<Extra,Source>,~SlotsLanes.lane_point(~Extra,~point,~Source,~source_value),~Logical,~original,
      SlotsLanes.LanePoint{digit,extra_of(logical,channels,group),source_at(input_of(logical),Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{}))},logical,ShapeLaws.elements([height,width]),
      Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32))),Nat.mul(image,ShapeLaws.elements([height,width]))),
      Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),ShapeLaws.elements([height,width])))


# The partial of a kept group of the cells at an image, at a lane, is the
# partial of the channel of the lane at the image.
def plane_row_at(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,~X: Data,~Source: Data,~source_value: Source -> U32 -> Tile.Tile8,~source_at: X -> Cells.Cells -> Source,~source_moved: SourceMoved(~X,~Source,~source_value,~source_at),~Logical: Data,~original: Logical -> U32 -> F32,
  ~input_of: Logical -> X,~extras_of: Logical -> U32 -> List<&2,Extra>,~extra_of: Logical -> U32 -> Nat -> Extra,~Ready: Logical -> U32 -> U32 -> U32 -> U32 -> Data,
  ~started: PlaneStarted(~Extra,~Logical,~extras_of,~extra_of),~enough: PlaneEnough(~Extra,~Logical,~extras_of),~agrees: PlaneAgrees(~Extra,~point,~X,~Source,~source_value,~source_at,~Logical,~original,~input_of,~extra_of,~Ready),
  +logical: Logical,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+image: Nat,+group: Nat,+digit: Nat,
  +room: {Nat.is_le(ShapeLaws.elements([images,channels,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements([images,channels,height,width])) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool},+small: {Nat.is_le(digit,7n) == True{} : Bool},
  +valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool},+ready: Ready(logical,images,channels,height,width)) ->
  {SlotsLanes.parts_lane(PartialsValues.model_read(~Sum.TileParts,PartialsValues.written(~Sum.TileParts,plane_folds(~Extra,~point,~X,~Source,~source_value,~source_at,input_of(logical),images,channels,height,width,
        Reduce.block_starts(~Extra,~SlotsValues.SlotsView<Extra>,~SlotsValues.begin_view(~Extra),extras_of(logical,channels),Grid.product([Tiled.groups(channels)]))),Sum.tile_parts_zero()),
      PartialsValues.digits_address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],False{}),U32.from_nat(group),0),[image])),digit) ==
    PartialsValues.model_read(~Sum.Parts,PartialsValues.written(~Sum.Parts,plane_lines(~Logical,~original,logical,images,channels,height,width),TiledReduce.parts_zero()),
      PartialsValues.digits_address(PartialsValues.axes_with([images,channels],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,channels],[True{},False{}],False{}),U32.from_nat(Nat.add(digit,Nat.mul(group,8n))),0),[image])) : Sum.Parts}:
  +values = {[images,channels,height,width] : List<&2,U32>}
  +cells = {[images,height,width,Tiled.groups(channels)] : List<&2,U32>}
  +capacity = U32.to_nat(Machine.capacity(limits))
  +starts = Reduce.block_starts(~Extra,~SlotsValues.SlotsView<Extra>,~SlotsValues.begin_view(~Extra),extras_of(logical,channels),Grid.product([Tiled.groups(channels)]))
  +folds = plane_folds(~Extra,~point,~X,~Source,~source_value,~source_at,input_of(logical),images,channels,height,width,starts)
  +lines_v = plane_lines(~Logical,~original,logical,images,channels,height,width)
  +lines = U32.to_nat(Grid.product([Tiled.groups(channels)]))
  +groups = U32.to_nat(Tiled.groups(channels))
  +channel = Nat.add(digit,Nat.mul(group,8n))
  +row = Nat.add(Nat.mul(image,U32.to_nat(channels)),channel)
  +below = {KernelGemm.Both{image_inside,Unit{}} : Coordinates.DigitsBelow([images],[image])}
  +rooms_in = ReduceLaws.kept_rooms(limits,values,[0n,2n,3n],1n,Grid.Innermost{},{==},{==},room,positive)
  +cells_room = Order.transitive(ShapeLaws.elements(cells),ShapeLaws.elements(values),capacity,ReduceLaws.rooms_cells_fewer(limits,values,[0n,2n,3n],1n,Grid.Innermost{},rooms_in),room)
  +cells_positive = ReduceLaws.rooms_cells_positive(limits,values,[0n,2n,3n],1n,Grid.Innermost{},rooms_in)
  +rooms_out = ReduceLaws.kept_rooms(limits,values,[0n,2n,3n],1n,Grid.Outermost{},{==},{==},room,positive)
  +outer_room = Order.transitive(ShapeLaws.elements(Tiled.cell_shape(values,1n,Grid.Outermost{})),ShapeLaws.elements(values),capacity,ReduceLaws.rooms_cells_fewer(limits,values,[0n,2n,3n],1n,Grid.Outermost{},rooms_out),room)
  +cells_nonzero = CountCompletion.positive_not_zero(cells,cells_positive)
  +nwg_nonzero = TiledSums.removed_nonzero(cells,1n,cells_nonzero)
  +ng_nonzero = TiledSums.removed_nonzero([images,width,Tiled.groups(channels)],1n,nwg_nonzero)
  +ng_room = Order.transitive(ShapeLaws.elements([images,Tiled.groups(channels)]),ShapeLaws.elements([images,width,Tiled.groups(channels)]),capacity,
    TiledSums.removed_fewer([images,width,Tiled.groups(channels)],1n,nwg_nonzero),
    Order.transitive(ShapeLaws.elements([images,width,Tiled.groups(channels)]),ShapeLaws.elements(cells),capacity,TiledSums.removed_fewer(cells,1n,cells_nonzero),cells_room))
  +half = Bounds.as_nat_le(Machine.capacity(limits),Machine.half_word(limits),Limits.capacity_within_half(limits))
  +ng_half = Order.transitive(ShapeLaws.elements([images,Tiled.groups(channels)]),capacity,U32.to_nat(Machine.half_word(limits)),ng_room,half)
  +g_room = Order.transitive(ShapeLaws.elements([Tiled.groups(channels)]),ShapeLaws.elements([images,Tiled.groups(channels)]),capacity,TiledSums.removed_fewer([images,Tiled.groups(channels)],0n,ng_nonzero),ng_room)
  +g_inside = {%Equal.sym(Nat,Nat.mul(groups,1n),groups,Order.times_one(groups)) : {Nat.is_lt(group,_) == True{} : Bool} group_inside : {Nat.is_lt(group,ShapeLaws.elements([Tiled.groups(channels)])) == True{} : Bool}}
  +lines_exact = Equal.cong(U32,Nat,word => U32.to_nat(word),Grid.product([Tiled.groups(channels)]),Tiled.groups(channels),Arithmetic.multiply_one(Tiled.groups(channels)))
  +address_c = Equal.trans(U32,PartialsValues.digits_address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],False{}),U32.from_nat(group),0),[image]),
    U32.from_nat(ShapeLaws.digits_value([images,Tiled.groups(channels)],ReduceLaws.merged([True{},False{}],Targets.cell_coordinates([Tiled.groups(channels)],group),[image]))),U32.from_nat(Nat.add(Nat.mul(image,lines),group)),
    TiledSums.row_address([images,Tiled.groups(channels)],[True{},False{}],group,[image],Machine.half_word(limits),Limits.half_word_fits(limits),ng_nonzero,ng_half,
      Limits.double_bound(limits,ShapeLaws.elements([Tiled.groups(channels)]),g_room),g_inside,below),
    Equal.cong(Nat,U32,index => U32.from_nat(index),ShapeLaws.digits_value([images,Tiled.groups(channels)],ReduceLaws.merged([True{},False{}],Targets.cell_coordinates([Tiled.groups(channels)],group),[image])),Nat.add(Nat.mul(image,lines),group),
      Equal.trans(Nat,ShapeLaws.digits_value([images,Tiled.groups(channels)],ReduceLaws.merged([True{},False{}],Targets.cell_coordinates([Tiled.groups(channels)],group),[image])),Nat.add(Nat.mul(image,groups),group),Nat.add(Nat.mul(image,lines),group),
        pair_address(images,Tiled.groups(channels),group,image),
        Equal.cong(Nat,Nat,count => Nat.add(Nat.mul(image,count),group),groups,lines,Equal.sym(Nat,lines,groups,lines_exact)))))
  +folds_count = PartialsValues.block_count(~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~Sum.TileParts,~SlotsValues.cell_view(~Extra,~point),~SlotsValues.turn_view(~Extra),ShapeLaws.elements([images]),
    Grid.product([height,width]),Grid.product([Tiled.groups(channels)]),
    Reduce.values_from(~Tile.Tile8,~Source,~source_value,source_at(input_of(logical),Public.geometry_of(values,1n,Grid.Innermost{})),ShapeLaws.elements(cells),0n),starts,enough(logical,channels))
  +image_inside_e = {%Equal.sym(Nat,Nat.mul(U32.to_nat(images),1n),U32.to_nat(images),Order.times_one(U32.to_nat(images))) : {Nat.is_lt(image,_) == True{} : Bool} image_inside : {Nat.is_lt(image,ShapeLaws.elements([images])) == True{} : Bool}}
  +line_inside = {%Equal.sym(Nat,lines,groups,lines_exact) : {Nat.is_lt(group,_) == True{} : Bool} group_inside : {Nat.is_lt(group,lines) == True{} : Bool}}
  +folds_inside = {%Equal.sym(Nat,List.length(&2,Sum.TileParts,folds),Nat.mul(ShapeLaws.elements([images]),lines),folds_count) : {Nat.is_lt(Nat.add(Nat.mul(image,lines),group),_) == True{} : Bool}
    RowMajor.radix(image,ShapeLaws.elements([images]),group,lines,image_inside_e,line_inside) : {Nat.is_lt(Nat.add(Nat.mul(image,lines),group),List.length(&2,Sum.TileParts,folds)) == True{} : Bool}}
  +folds_exact = Equal.trans(Nat,List.length(&2,Sum.TileParts,folds),Nat.mul(ShapeLaws.elements([images]),lines),ShapeLaws.elements([images,Tiled.groups(channels)]),folds_count,
    Equal.trans(Nat,Nat.mul(ShapeLaws.elements([images]),lines),Nat.mul(ShapeLaws.elements([images]),groups),ShapeLaws.elements([images,Tiled.groups(channels)]),
      Equal.cong(Nat,Nat,count => Nat.mul(ShapeLaws.elements([images]),count),lines,groups,lines_exact),
      Equal.sym(Nat,ShapeLaws.elements([images,Tiled.groups(channels)]),Nat.mul(ShapeLaws.elements([images]),groups),pair_elements(images,Tiled.groups(channels)))))
  +folds_room = {%Equal.sym(Nat,List.length(&2,Sum.TileParts,folds),ShapeLaws.elements([images,Tiled.groups(channels)]),folds_exact) : {Nat.is_le(_,capacity) == True{} : Bool} ng_room :
    {Nat.is_le(List.length(&2,Sum.TileParts,folds),capacity) == True{} : Bool}}
  +read_c = PartialsValues.written_at(~Sum.TileParts,folds,Sum.tile_parts_zero(),Nat.add(Nat.mul(image,lines),group),Machine.capacity(limits),Limits.capacity_allocatable(limits),folds_room,folds_inside)
  +values_nonzero = CountCompletion.positive_not_zero(values,positive)
  +ncw_nonzero = TiledSums.removed_nonzero(values,2n,values_nonzero)
  +nc_nonzero = TiledSums.removed_nonzero([images,channels,width],2n,ncw_nonzero)
  +nc_room = Order.transitive(ShapeLaws.elements([images,channels]),ShapeLaws.elements([images,channels,width]),capacity,TiledSums.removed_fewer([images,channels,width],2n,ncw_nonzero),
    Order.transitive(ShapeLaws.elements([images,channels,width]),ShapeLaws.elements(values),capacity,TiledSums.removed_fewer(values,2n,values_nonzero),room))
  +nc_half = Order.transitive(ShapeLaws.elements([images,channels]),capacity,U32.to_nat(Machine.half_word(limits)),nc_room,half)
  +c_room = Order.transitive(ShapeLaws.elements([channels]),ShapeLaws.elements([images,channels]),capacity,TiledSums.removed_fewer([images,channels],0n,nc_nonzero),nc_room)
  +c_inside = {%Equal.sym(Nat,Nat.mul(U32.to_nat(channels),1n),U32.to_nat(channels),Order.times_one(U32.to_nat(channels))) : {Nat.is_lt(channel,_) == True{} : Bool} valid : {Nat.is_lt(channel,ShapeLaws.elements([channels])) == True{} : Bool}}
  +address_v = Equal.trans(U32,PartialsValues.digits_address(PartialsValues.axes_with([images,channels],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,channels],[True{},False{}],False{}),U32.from_nat(channel),0),[image]),
    U32.from_nat(ShapeLaws.digits_value([images,channels],ReduceLaws.merged([True{},False{}],Targets.cell_coordinates([channels],channel),[image]))),U32.from_nat(row),
    TiledSums.row_address([images,channels],[True{},False{}],channel,[image],Machine.half_word(limits),Limits.half_word_fits(limits),nc_nonzero,nc_half,
      Limits.double_bound(limits,ShapeLaws.elements([channels]),c_room),c_inside,below),
    Equal.cong(Nat,U32,index => U32.from_nat(index),ShapeLaws.digits_value([images,channels],ReduceLaws.merged([True{},False{}],Targets.cell_coordinates([channels],channel),[image])),row,
      pair_address(images,channels,channel,image)))
  +lines_count = Equal.trans(Nat,List.length(&2,Sum.Parts,lines_v),List.length(&2,Sum.Lanes,Reduce.line_folds(~F32,~Sum.Lanes,~Sum.step,~Logical,~original,logical,ShapeLaws.elements([images,channels]),ShapeLaws.elements([height,width]),0n,Sum.zero())),
    ShapeLaws.elements([images,channels]),
    Reduce.turned_length(~Sum.Parts,~Sum.Lanes,~Sum.parts,Reduce.line_folds(~F32,~Sum.Lanes,~Sum.step,~Logical,~original,logical,ShapeLaws.elements([images,channels]),ShapeLaws.elements([height,width]),0n,Sum.zero())),
    Reduce.line_folds_length(~F32,~Sum.Lanes,~Sum.step,~Logical,~original,logical,ShapeLaws.elements([images,channels]),ShapeLaws.elements([height,width]),0n,Sum.zero()))
  +nc_exact = Equal.cong(Nat,Nat,count => Nat.mul(U32.to_nat(images),count),Nat.mul(U32.to_nat(channels),1n),U32.to_nat(channels),Order.times_one(U32.to_nat(channels)))
  +row_inside = {%Equal.sym(Nat,ShapeLaws.elements([images,channels]),Nat.mul(U32.to_nat(images),U32.to_nat(channels)),nc_exact) : {Nat.is_lt(row,_) == True{} : Bool}
    RowMajor.radix(image,U32.to_nat(images),channel,U32.to_nat(channels),image_inside,valid) : {Nat.is_lt(row,ShapeLaws.elements([images,channels])) == True{} : Bool}}
  +lines_inside = {%Equal.sym(Nat,List.length(&2,Sum.Parts,lines_v),ShapeLaws.elements([images,channels]),lines_count) : {Nat.is_lt(row,_) == True{} : Bool} row_inside : {Nat.is_lt(row,List.length(&2,Sum.Parts,lines_v)) == True{} : Bool}}
  +lines_room = {%Equal.sym(Nat,List.length(&2,Sum.Parts,lines_v),ShapeLaws.elements([images,channels]),lines_count) : {Nat.is_le(_,capacity) == True{} : Bool} nc_room : {Nat.is_le(List.length(&2,Sum.Parts,lines_v),capacity) == True{} : Bool}}
  +read_v = PartialsValues.written_at(~Sum.Parts,lines_v,TiledReduce.parts_zero(),row,Machine.capacity(limits),Limits.capacity_allocatable(limits),lines_room,lines_inside)
  +row_c = plane_row(~Extra,~point,~X,~Source,~source_value,~source_at,~source_moved,~Logical,~original,input_of(logical),logical,limits,images,channels,height,width,starts,extra_of(logical,channels,group),image,group,digit,
    cells_room,cells_positive,image_inside_e,group_inside,row_inside,started(logical,channels,group,group_inside),enough(logical,channels),
    agrees(logical,limits,images,channels,height,width,image,group,digit,room,outer_room,ready,image_inside,group_inside,small,valid))
  +model_c = PartialsValues.written(~Sum.TileParts,folds,Sum.tile_parts_zero())
  +model_v = PartialsValues.written(~Sum.Parts,lines_v,TiledReduce.parts_zero())
  +spot_c = PartialsValues.digits_address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],False{}),U32.from_nat(group),0),[image])
  +spot_v = PartialsValues.digits_address(PartialsValues.axes_with([images,channels],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,channels],[True{},False{}],False{}),U32.from_nat(channel),0),[image])
  Equal.trans(Sum.Parts,SlotsLanes.parts_lane(PartialsValues.model_read(~Sum.TileParts,model_c,spot_c),digit),SlotsLanes.parts_lane(PartialsValues.model_read(~Sum.TileParts,model_c,U32.from_nat(Nat.add(Nat.mul(image,lines),group))),digit),
    PartialsValues.model_read(~Sum.Parts,model_v,spot_v),
    Equal.cong(U32,Sum.Parts,word => SlotsLanes.parts_lane(PartialsValues.model_read(~Sum.TileParts,model_c,word),digit),spot_c,U32.from_nat(Nat.add(Nat.mul(image,lines),group)),address_c),
    Equal.trans(Sum.Parts,SlotsLanes.parts_lane(PartialsValues.model_read(~Sum.TileParts,model_c,U32.from_nat(Nat.add(Nat.mul(image,lines),group))),digit),
      SlotsLanes.parts_lane(ListReadback.at(~Sum.TileParts,Sum.tile_parts_zero(),Nat.add(Nat.mul(image,lines),group),folds),digit),PartialsValues.model_read(~Sum.Parts,model_v,spot_v),
      Equal.cong(Sum.TileParts,Sum.Parts,parts => SlotsLanes.parts_lane(parts,digit),PartialsValues.model_read(~Sum.TileParts,model_c,U32.from_nat(Nat.add(Nat.mul(image,lines),group))),
        ListReadback.at(~Sum.TileParts,Sum.tile_parts_zero(),Nat.add(Nat.mul(image,lines),group),folds),read_c),
      Equal.trans(Sum.Parts,SlotsLanes.parts_lane(ListReadback.at(~Sum.TileParts,Sum.tile_parts_zero(),Nat.add(Nat.mul(image,lines),group),folds),digit),
        ListReadback.at(~Sum.Parts,TiledReduce.parts_zero(),row,lines_v),PartialsValues.model_read(~Sum.Parts,model_v,spot_v),row_c,
        Equal.trans(Sum.Parts,ListReadback.at(~Sum.Parts,TiledReduce.parts_zero(),row,lines_v),PartialsValues.model_read(~Sum.Parts,model_v,U32.from_nat(row)),PartialsValues.model_read(~Sum.Parts,model_v,spot_v),
          Equal.sym(Sum.Parts,PartialsValues.model_read(~Sum.Parts,model_v,U32.from_nat(row)),ListReadback.at(~Sum.Parts,TiledReduce.parts_zero(),row,lines_v),read_v),
          Equal.cong(U32,Sum.Parts,word => PartialsValues.model_read(~Sum.Parts,model_v,word),U32.from_nat(row),spot_v,Equal.sym(U32,spot_v,U32.from_nat(row),address_v))))))

# The partial of a kept group of the cells at an image row, at a lane, is the
# partial of the channel of the lane at the row.
def plane_digits(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,~X: Data,~Source: Data,~source_value: Source -> U32 -> Tile.Tile8,~source_at: X -> Cells.Cells -> Source,~source_moved: SourceMoved(~X,~Source,~source_value,~source_at),~Logical: Data,~original: Logical -> U32 -> F32,
  ~input_of: Logical -> X,~extras_of: Logical -> U32 -> List<&2,Extra>,~extra_of: Logical -> U32 -> Nat -> Extra,~Ready: Logical -> U32 -> U32 -> U32 -> U32 -> Data,
  ~started: PlaneStarted(~Extra,~Logical,~extras_of,~extra_of),~enough: PlaneEnough(~Extra,~Logical,~extras_of),~agrees: PlaneAgrees(~Extra,~point,~X,~Source,~source_value,~source_at,~Logical,~original,~input_of,~extra_of,~Ready),
  +logical: Logical,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+group: Nat,+digit: Nat,
  +room: {Nat.is_le(ShapeLaws.elements([images,channels,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements([images,channels,height,width])) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool},+small: {Nat.is_le(digit,7n) == True{} : Bool},
  +valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool},+ready: Ready(logical,images,channels,height,width),
  digits: List<&2,Nat>,+below: Coordinates.DigitsBelow([images],digits)) ->
  {SlotsLanes.parts_lane(PartialsValues.model_read(~Sum.TileParts,PartialsValues.written(~Sum.TileParts,plane_folds(~Extra,~point,~X,~Source,~source_value,~source_at,input_of(logical),images,channels,height,width,
        Reduce.block_starts(~Extra,~SlotsValues.SlotsView<Extra>,~SlotsValues.begin_view(~Extra),extras_of(logical,channels),Grid.product([Tiled.groups(channels)]))),Sum.tile_parts_zero()),
      PartialsValues.digits_address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],False{}),U32.from_nat(group),0),digits)),digit) ==
    PartialsValues.model_read(~Sum.Parts,PartialsValues.written(~Sum.Parts,plane_lines(~Logical,~original,logical,images,channels,height,width),TiledReduce.parts_zero()),
      PartialsValues.digits_address(PartialsValues.axes_with([images,channels],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,channels],[True{},False{}],False{}),U32.from_nat(Nat.add(digit,Nat.mul(group,8n))),0),digits)) : Sum.Parts}:
  match digits:
    case Nil{}:
      Empty.absurd({SlotsLanes.parts_lane(PartialsValues.model_read(~Sum.TileParts,PartialsValues.written(~Sum.TileParts,plane_folds(~Extra,~point,~X,~Source,~source_value,~source_at,input_of(logical),images,channels,height,width,
            Reduce.block_starts(~Extra,~SlotsValues.SlotsView<Extra>,~SlotsValues.begin_view(~Extra),extras_of(logical,channels),Grid.product([Tiled.groups(channels)]))),Sum.tile_parts_zero()),
          PartialsValues.digits_address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],False{}),U32.from_nat(group),0),[])),digit) ==
        PartialsValues.model_read(~Sum.Parts,PartialsValues.written(~Sum.Parts,plane_lines(~Logical,~original,logical,images,channels,height,width),TiledReduce.parts_zero()),
          PartialsValues.digits_address(PartialsValues.axes_with([images,channels],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,channels],[True{},False{}],False{}),U32.from_nat(Nat.add(digit,Nat.mul(group,8n))),0),[])) : Sum.Parts},below)
    case Con{+image,rest}:
      match rest:
        case Con{+next,+more}:
          KernelGemm.Both{+first,+empty} = below
          Empty.absurd({SlotsLanes.parts_lane(PartialsValues.model_read(~Sum.TileParts,PartialsValues.written(~Sum.TileParts,plane_folds(~Extra,~point,~X,~Source,~source_value,~source_at,input_of(logical),images,channels,height,width,
                Reduce.block_starts(~Extra,~SlotsValues.SlotsView<Extra>,~SlotsValues.begin_view(~Extra),extras_of(logical,channels),Grid.product([Tiled.groups(channels)]))),Sum.tile_parts_zero()),
              PartialsValues.digits_address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],False{}),U32.from_nat(group),0),image <> next <> more)),digit) ==
            PartialsValues.model_read(~Sum.Parts,PartialsValues.written(~Sum.Parts,plane_lines(~Logical,~original,logical,images,channels,height,width),TiledReduce.parts_zero()),
              PartialsValues.digits_address(PartialsValues.axes_with([images,channels],[True{},False{}],True{}),PartialsValues.address(PartialsValues.axes_with([images,channels],[True{},False{}],False{}),U32.from_nat(Nat.add(digit,Nat.mul(group,8n))),0),image <> next <> more)) : Sum.Parts},empty)
        case Nil{}:
          KernelGemm.Both{+image_inside,+rest_below} = below
          plane_row_at(~Extra,~point,~X,~Source,~source_value,~source_at,~source_moved,~Logical,~original,~input_of,~extras_of,~extra_of,~Ready,~started,~enough,~agrees,logical,limits,images,channels,height,width,image,group,digit,room,positive,image_inside,group_inside,small,valid,ready)

# A lane of a kept cell of the statistics holds the sum of the planes of its
# channel: the lanes of the merged rows are the merged rows of the lanes.
def plane_lane(~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,~X: Data,~Source: Data,~source_value: Source -> U32 -> Tile.Tile8,~source_at: X -> Cells.Cells -> Source,~source_moved: SourceMoved(~X,~Source,~source_value,~source_at),~Logical: Data,~original: Logical -> U32 -> F32,
  ~input_of: Logical -> X,~extras_of: Logical -> U32 -> List<&2,Extra>,~extra_of: Logical -> U32 -> Nat -> Extra,~Ready: Logical -> U32 -> U32 -> U32 -> U32 -> Data,
  ~started: PlaneStarted(~Extra,~Logical,~extras_of,~extra_of),~enough: PlaneEnough(~Extra,~Logical,~extras_of),~agrees: PlaneAgrees(~Extra,~point,~X,~Source,~source_value,~source_at,~Logical,~original,~input_of,~extra_of,~Ready),
  +spec: Plane<Logical>,+limits: Machine.Limits,+values: List<&2,U32>,+axes: List<&2,Nat>,+lane: Nat,+position: Grid.Position,
  +within: {Nat.is_lt(lane,List.length(&2,U32,values)) == True{} : Bool},+absent: {Views.contains_index(axes,lane) == False{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+rooms: ReduceLaws.KeptRooms<limits,values,axes,lane,position>,
  +arranged: plane_fits(~Logical,~Ready,spec,values,axes,lane,position),+word: U32,+others: List<&2,Nat>,+group: Nat,
  +others_below: Coordinates.DigitsBelow(Grid.removed(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n)),others),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n))))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),position),Coordinates.cell_digits(position,others,group)) : Nat},
  +digit: Nat,+small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,lane))) == True{} : Bool}) ->
  {TileLaws.select_lane(digit,PartialsValues.box_value(~Tile.Tile8,~Sum.TileParts,~Sum.tile_merge,~Sum.tile_rounded_total,plane_cells(~Extra,~point,~X,~Source,~source_value,~source_at,~Logical,~input_of,~extras_of,spec,values,axes,lane,position),word)) ==
    PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded,plane_values(~Logical,~original,spec,values,axes),
      U32.from_nat(ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,Nat.add(digit,Nat.mul(group,8n)))))) : F32}:
  Plane{+logical,+images,+channels,+height,+width} = spec
  ReduceLaws.KeptRooms{+values_room,+values_positive,+removed_room,+cells_room,+cells_positive,+kept_room,+kept_positive,+kept_removed_room,+reduced_positive,+cells_fewer} = rooms
  KernelGemm.Both{+values_eq,KernelGemm.Both{+axes_eq,KernelGemm.Both{+lane_eq,KernelGemm.Both{+position_eq,+ready}}}} = arranged
  +shape = {[images,channels,height,width] : List<&2,U32>}
  +capacity = U32.to_nat(Machine.capacity(limits))
  +channel = Nat.add(digit,Nat.mul(group,8n))
  +room = {%values_eq : {Nat.is_le(ShapeLaws.elements(_),capacity) == True{} : Bool} values_room : {Nat.is_le(ShapeLaws.elements(shape),capacity) == True{} : Bool}}
  +positive = {%values_eq : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} values_positive : {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool}}
  +valid_c = {%values_eq : {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(_,1n))) == True{} : Bool}
    {%lane_eq : {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,_))) == True{} : Bool} valid : {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,1n))) == True{} : Bool}} :
    {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool}}
  +group_c = {%values_eq : {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(_,Layout.axes_where(List.length(&2,U32,_),0n,[0n,2n,3n],False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,_),0n,[0n,2n,3n],False{}),1n,0n))))) == True{} : Bool}
    {%axes_eq : {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,_,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,_,False{}),1n,0n))))) == True{} : Bool}
      {%lane_eq : {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),_,0n))))) == True{} : Bool}
        group_inside : {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),1n,0n))))) == True{} : Bool}} :
      {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,[0n,2n,3n],False{})),TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,[0n,2n,3n],False{}),1n,0n))))) == True{} : Bool}} :
    {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool}}
  +none = plane_none(images,channels,height,width,values,axes,lane,others,values_eq,axes_eq,lane_eq,others_below)
  +word_eq = plane_word(images,channels,height,width,values,axes,lane,position,word,others,group,values_eq,axes_eq,lane_eq,position_eq,none,exact)
  +index_eq = plane_channel(images,channels,height,width,values,axes,lane,others,digit,group,values_eq,axes_eq,lane_eq,none)
  +starts = Reduce.block_starts(~Extra,~SlotsValues.SlotsView<Extra>,~SlotsValues.begin_view(~Extra),extras_of(logical,channels),Grid.product([Tiled.groups(channels)]))
  +model_c = PartialsValues.written(~Sum.TileParts,plane_folds(~Extra,~point,~X,~Source,~source_value,~source_at,input_of(logical),images,channels,height,width,starts),Sum.tile_parts_zero())
  +model_v = PartialsValues.written(~Sum.Parts,plane_lines(~Logical,~original,logical,images,channels,height,width),TiledReduce.parts_zero())
  +rows_c = PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],True{})
  +rows_v = PartialsValues.axes_with([images,channels],[True{},False{}],True{})
  +base_c = PartialsValues.address(PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],False{}),U32.from_nat(group),0)
  +base_v = PartialsValues.address(PartialsValues.axes_with([images,channels],[True{},False{}],False{}),U32.from_nat(channel),0)
  +depth = PartialsValues.depth_of(rows_c)
  +single = U32.is_le(PartialsValues.first_extent(rows_c),1)
  +first = PartialsValues.first_extent(rows_c)
  +rooms_in = ReduceLaws.kept_rooms(limits,shape,[0n,2n,3n],1n,Grid.Innermost{},{==},{==},room,positive)
  +cells_nonzero = CountCompletion.positive_not_zero([images,height,width,Tiled.groups(channels)],ReduceLaws.rooms_cells_positive(limits,shape,[0n,2n,3n],1n,Grid.Innermost{},rooms_in))
  +images_nonzero = TiledSums.removed_nonzero([images,Tiled.groups(channels)],1n,TiledSums.removed_nonzero([images,width,Tiled.groups(channels)],1n,TiledSums.removed_nonzero([images,height,width,Tiled.groups(channels)],1n,cells_nonzero)))
  +images_room = Order.transitive(ShapeLaws.elements([images]),ShapeLaws.elements(shape),capacity,
    Order.transitive(ShapeLaws.elements([images]),ShapeLaws.elements([images,channels]),ShapeLaws.elements(shape),TiledSums.removed_fewer([images,channels],1n,
        TiledSums.removed_nonzero([images,channels,width],2n,TiledSums.removed_nonzero(shape,2n,CountCompletion.positive_not_zero(shape,positive)))),
      Order.transitive(ShapeLaws.elements([images,channels]),ShapeLaws.elements([images,channels,width]),ShapeLaws.elements(shape),
        TiledSums.removed_fewer([images,channels,width],2n,TiledSums.removed_nonzero(shape,2n,CountCompletion.positive_not_zero(shape,positive))),
        TiledSums.removed_fewer(shape,2n,CountCompletion.positive_not_zero(shape,positive)))),room)
  +within_c = {PartialsValues.within_of([images],Machine.capacity(limits),images_nonzero,images_room) : PartialsValues.Within(Views.axis_extents(rows_c),Machine.capacity(limits))}
  +lanes_c = PartialsValues.tree_of(~Sum.Parts,~Sum.merge,~Unit,~(unit => index => SlotsLanes.parts_lane(PartialsValues.model_read(~Sum.TileParts,model_c,index),digit)),Unit{},depth,single,rows_c,0,first,base_c,TiledReduce.parts_zero())
  +read_v = PartialsValues.tree_of(~Sum.Parts,~Sum.merge,~Unit,~(unit => index => PartialsValues.model_read(~Sum.Parts,model_v,index)),Unit{},depth,single,rows_v,0,first,base_v,TiledReduce.parts_zero())
  +tree_c = PartialsValues.tree_of(~Sum.TileParts,~Sum.tile_merge,~Tree.Tree<Sum.TileParts>,~PartialsValues.model_read(~Sum.TileParts),model_c,depth,single,rows_c,0,first,base_c,Sum.tile_parts_zero())
  +tree_v = PartialsValues.tree_of(~Sum.Parts,~Sum.merge,~Tree.Tree<Sum.Parts>,~PartialsValues.model_read(~Sum.Parts),model_v,depth,single,rows_v,0,first,base_v,TiledReduce.parts_zero())
  +mapped = {%SlotsLanes.zero_lane(digit) :
      {SlotsLanes.parts_lane(tree_c,digit) == PartialsValues.tree_of(~Sum.Parts,~Sum.merge,~Unit,~(unit => index => SlotsLanes.parts_lane(PartialsValues.model_read(~Sum.TileParts,model_c,index),digit)),Unit{},depth,single,rows_c,0,first,base_c,_) : Sum.Parts}
    PartialsValues.tree_mapped(~Sum.TileParts,~Sum.tile_merge,~Sum.Parts,~Sum.merge,~(parts => SlotsLanes.parts_lane(parts,digit)),~(left => right => SlotsLanes.merge_lane(left,right,digit)),
      ~Tree.Tree<Sum.TileParts>,~PartialsValues.model_read(~Sum.TileParts),depth,single,rows_c,model_c,0,first,base_c,Sum.tile_parts_zero()) :
    {SlotsLanes.parts_lane(tree_c,digit) == lanes_c : Sum.Parts}}
  +agreed = PartialsValues.roots_agree(~Sum.Parts,~Sum.merge,~(unit => index => SlotsLanes.parts_lane(PartialsValues.model_read(~Sum.TileParts,model_c,index),digit)),~(unit => index => PartialsValues.model_read(~Sum.Parts,model_v,index)),
    ~rows_c,~rows_v,~base_c,~base_v,
    ~(digits => below => plane_digits(~Extra,~point,~X,~Source,~source_value,~source_at,~source_moved,~Logical,~original,~input_of,~extras_of,~extra_of,~Ready,~started,~enough,~agrees,logical,limits,images,channels,height,width,group,digit,room,positive,group_c,small,valid_c,ready,digits,below)),
    TiledReduce.parts_zero(),Machine.capacity(limits),Limits.capacity_allocatable(limits),within_c,{==})
  +identity = PartialsValues.tree_mapped(~Sum.Parts,~Sum.merge,~Sum.Parts,~Sum.merge,~(partial => partial),~(left => right => {==}),~Tree.Tree<Sum.Parts>,~PartialsValues.model_read(~Sum.Parts),
    depth,single,rows_v,model_v,0,first,base_v,TiledReduce.parts_zero())
  +trees = Equal.trans(Sum.Parts,SlotsLanes.parts_lane(tree_c,digit),lanes_c,tree_v,mapped,Equal.trans(Sum.Parts,lanes_c,read_v,tree_v,agreed,Equal.sym(Sum.Parts,tree_v,read_v,identity)))
  %Equal.sym(Nat,ShapeLaws.digits_value(Layout.extents_at(values,Layout.axes_where(List.length(&2,U32,values),0n,axes,False{})),Coordinates.inserted_digit(TiledReduce.kept_place(Layout.axes_where(List.length(&2,U32,values),0n,axes,False{}),lane,0n),others,channel)),channel,index_eq) :
    {TileLaws.select_lane(digit,Sum.tile_rounded_total(PartialsValues.box_output(~Sum.TileParts,~Sum.tile_merge,plane_cells(~Extra,~point,~X,~Source,~source_value,~source_at,~Logical,~input_of,~extras_of,Plane{logical,images,channels,height,width},values,axes,lane,position),U32.to_nat(word)))) ==
      PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded,plane_box(~Logical,~original,logical,images,channels,height,width),U32.from_nat(_)) : F32}
  %Equal.sym(Nat,U32.to_nat(word),group,word_eq) :
    {TileLaws.select_lane(digit,Sum.tile_rounded_total(PartialsValues.box_output(~Sum.TileParts,~Sum.tile_merge,plane_cells(~Extra,~point,~X,~Source,~source_value,~source_at,~Logical,~input_of,~extras_of,Plane{logical,images,channels,height,width},values,axes,lane,position),_))) ==
      PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded,plane_box(~Logical,~original,logical,images,channels,height,width),U32.from_nat(channel)) : F32}
  Equal.trans(F32,TileLaws.select_lane(digit,Sum.tile_rounded_total(tree_c)),Sum.rounded(SlotsLanes.parts_lane(tree_c,digit)),
    PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded,plane_box(~Logical,~original,logical,images,channels,height,width),U32.from_nat(channel)),
    SlotsLanes.rounded_lane(tree_c,digit),
    Equal.trans(F32,Sum.rounded(SlotsLanes.parts_lane(tree_c,digit)),Sum.rounded(tree_v),
      PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded,plane_box(~Logical,~original,logical,images,channels,height,width),U32.from_nat(channel)),
      Equal.cong(Sum.Parts,F32,partial => Sum.rounded(partial),SlotsLanes.parts_lane(tree_c,digit),tree_v,trees),
      Equal.sym(F32,PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded,plane_box(~Logical,~original,logical,images,channels,height,width),U32.from_nat(channel)),Sum.rounded(tree_v),
        PartialsValues.box_word(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded,plane_box(~Logical,~original,logical,images,channels,height,width),channel))))

# Lines drawn from one context all begin from it.
def single_drawn(~Context: Data,~Value: Data,~begin_value: Context -> Value,remaining: Nat,+context: Context,index: Nat) ->
  {ListReadback.at(~Value,begin_value(context),index,LineChainValues.drawn(~Context,~Value,~begin_value,remaining,[context],[context])) == begin_value(context) : Value}:
  match remaining index:
    case 0n 0n: {==}
    case 0n 1n+previous: {==}
    case 1n+ +rest 0n: {==}
    case 1n+ +rest 1n+ +previous: single_drawn(~Context,~Value,~begin_value,rest,context,previous)

# ---- The sums of the statistics ----

# The logical index of a channel at a point of the plane of an image, with the
# groups outermost, is the point of the plane of the image and the channel.
def plane_index(+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+image: Nat,+channel: Nat,+at: Nat,
  +plane_room: {Nat.is_le(ShapeLaws.elements([height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside: {Nat.is_lt(at,ShapeLaws.elements([height,width])) == True{} : Bool}) ->
  {Representation.logical_index(Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{}),Nat.add(Nat.mul(image,ShapeLaws.elements([height,width])),at),channel) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),ShapeLaws.elements([height,width])),at) : Nat}:
  +plane = ShapeLaws.elements([height,width])
  +point = Nat.add(Nat.mul(image,plane),at)
  +lanes = U32.to_nat(channels)
  +target = Nat.add(Nat.mul(Nat.add(Nat.mul(image,lanes),channel),plane),at)
  +word = Equal.trans(Nat,U32.to_nat(Tiled.product([height,width])),U32.to_nat(U32.from_nat(plane)),plane,
    Equal.cong(U32,Nat,value => U32.to_nat(value),Tiled.product([height,width]),U32.from_nat(plane),Equal.sym(U32,U32.from_nat(plane),Tiled.product([height,width]),Digits.product_from([height,width]))),
    FullRange.roundtrip(plane,Machine.capacity(limits),plane_room))
  +least = Representation.least_positive(plane,Order.positive_bound(at,plane,inside))
  +split = Representation.split_at(image,at,plane,inside)
  +quotient = Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(point,Representation.least_one(plane)),(image,at),split)
  +remainder = Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(point,Representation.least_one(plane)),(image,at),split)
  %Equal.sym(Nat,U32.to_nat(Tiled.product([height,width])),plane,word) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(point,Representation.least_one(_)),lanes),channel),Representation.least_one(_)),Nat.mod(point,Representation.least_one(_))) == target : Nat}
  %Equal.sym(Nat,Representation.least_one(plane),plane,least) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(point,Representation.least_one(plane)),lanes),channel),_),Nat.mod(point,Representation.least_one(plane))) == target : Nat}
  %Equal.sym(Nat,Nat.div(point,Representation.least_one(plane)),image,quotient) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(_,lanes),channel),plane),Nat.mod(point,Representation.least_one(plane))) == target : Nat}
  %Equal.sym(Nat,Nat.mod(point,Representation.least_one(plane)),at,remainder) :
    {Nat.add(Nat.mul(Nat.add(Nat.mul(image,lanes),channel),plane),_) == target : Nat}
  {==}

# A lane of the cells of values [N,C,H,W] with the groups outermost, at a
# point of the plane of an image, is the value of the channel of the lane there.
def plane_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+image: Nat,+group: Nat,+digit: Nat,
  +cells_room: {Nat.is_le(ShapeLaws.elements([Tiled.groups(channels),images,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool},
  +small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool},
  +at: Nat,+inside: {Nat.is_lt(at,ShapeLaws.elements([height,width])) == True{} : Bool}) ->
  {SlotsLanes.lane_point(~Unit,~SlotsValues.held(~Unit),~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      SlotsLanes.LanePoint{digit,Unit{},Representation.Lifted{logical,Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})}},
      U32.from_nat(Nat.add(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32))),Nat.mul(image,ShapeLaws.elements([height,width]))),at))) ==
    original(logical,U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),ShapeLaws.elements([height,width])),at))) : F32}:
  +shape = {[images,channels,height,width] : List<&2,U32>}
  +geometry = Public.geometry_of(shape,1n,Grid.Outermost{})
  +capacity = U32.to_nat(Machine.capacity(limits))
  +plane = ShapeLaws.elements([height,width])
  +counted = ShapeLaws.elements([images,height,width])
  +groups = U32.to_nat(Tiled.groups(channels))
  +outer = U32.to_nat((Public.lane_prefix(shape,1n,1) * Tiled.product(Public.lane_after(shape,1n)) : U32))
  +point = Nat.add(Nat.mul(image,plane),at)
  +channel = Nat.add(digit,Nat.mul(group,8n))
  +index = Nat.add(Nat.mul(group,outer),point)
  +number = Nat.add(Nat.add(Nat.mul(group,outer),Nat.mul(image,plane)),at)
  +word = U32.from_nat(number)
  +located = Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),plane),at)
  +groups_positive = Order.positive_bound(group,groups,group_inside)
  +images_positive = Order.positive_bound(image,U32.to_nat(images),image_inside)
  +counted_room = Order.transitive(counted,Nat.mul(groups,counted),capacity,
    {%RowMajor.multiply_commutative(counted,groups) : {Nat.is_le(counted,_) == True{} : Bool} RowMajor.product_covers(counted,groups,groups_positive) : {Nat.is_le(counted,Nat.mul(groups,counted)) == True{} : Bool}},cells_room)
  +plane_room = Order.transitive(plane,counted,capacity,
    {%RowMajor.multiply_commutative(plane,U32.to_nat(images)) : {Nat.is_le(plane,_) == True{} : Bool} RowMajor.product_covers(plane,U32.to_nat(images),images_positive) : {Nat.is_le(plane,counted) == True{} : Bool}},counted_room)
  +outer_counted = Coordinates.outer_word(limits,shape,1n,counted_room)
  +point_inside = {%Equal.sym(Nat,outer,counted,outer_counted) : {Nat.is_lt(point,_) == True{} : Bool} RowMajor.radix(image,U32.to_nat(images),at,plane,image_inside,inside) : {Nat.is_lt(point,outer) == True{} : Bool}}
  +index_room = Order.transitive(index,Nat.mul(groups,outer),capacity,
    Bounds.strict_le(index,Nat.mul(groups,outer),RowMajor.radix(group,groups,point,outer,group_inside,point_inside)),
    {%Equal.sym(Nat,outer,counted,outer_counted) : {Nat.is_le(Nat.mul(groups,_),capacity) == True{} : Bool} cells_room : {Nat.is_le(Nat.mul(groups,outer),capacity) == True{} : Bool}})
  +joined = Addition.associative(Nat.mul(group,outer),Nat.mul(image,plane),at)
  +number_room = {%Equal.sym(Nat,number,index,joined) : {Nat.is_le(_,capacity) == True{} : Bool} index_room : {Nat.is_le(number,capacity) == True{} : Bool}}
  +exact = Equal.trans(Nat,U32.to_nat(word),number,index,FullRange.roundtrip(number,Machine.capacity(limits),number_room),joined)
  +lifted = Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},word)
  +cell = Representation.cell(~Logical,~original,logical,geometry,point,group)
  +value = original(logical,U32.from_nat(Representation.logical_index(geometry,point,channel)))
  +picked = {%Equal.sym(Bool,Nat.is_lt(channel,U32.to_nat(channels)),True{},valid) : {Bool.pick(F32,_,value,0.0) == value : F32}
    {==} : {Representation.lane(~Logical,~original,logical,geometry,point,group,digit) == value : F32}}
  Equal.trans(F32,TileLaws.select_lane(digit,lifted),TileLaws.select_lane(digit,cell),original(logical,U32.from_nat(located)),
    Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(digit,tile),lifted,cell,Representation.decoded(~Logical,~original,logical,geometry,point,group,word,exact,point_inside,group_inside)),
    Equal.trans(F32,TileLaws.select_lane(digit,cell),Representation.lane(~Logical,~original,logical,geometry,point,group,digit),original(logical,U32.from_nat(located)),
      Representation.cell_lane(~Logical,~original,logical,geometry,point,group,digit,small),
      Equal.trans(F32,Representation.lane(~Logical,~original,logical,geometry,point,group,digit),value,original(logical,U32.from_nat(located)),picked,
        Equal.cong(Nat,F32,count => original(logical,U32.from_nat(count)),Representation.logical_index(geometry,point,channel),located,
          plane_index(limits,images,channels,height,width,image,channel,at,plane_room,inside)))))

# The sums read the input itself.
def plane_input(~Logical: Data,logical: Logical) -> Logical:
  logical

# The sums take nothing besides a cell.
def plane_unit(~Logical: Data,logical: Logical,channels: U32,group: Nat) -> Unit:
  Unit{}

# The sums need nothing of their input.
def sums_ready(~Logical: Data,logical: Logical,images: U32,channels: U32,height: U32,width: U32) -> Data:
  Unit

# A lane of a plane of the cells of the input agrees with the plane of the
# channel of the lane.
def sums_agrees(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+image: Nat,+group: Nat,+digit: Nat,
  +room: {Nat.is_le(ShapeLaws.elements([images,channels,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_room: {Nat.is_le(ShapeLaws.elements([Tiled.groups(channels),images,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +ready: sums_ready(~Logical,logical,images,channels,height,width),
  +image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool},
  +small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool}) ->
  ReduceLaws.Agrees(~SlotsLanes.LanePoint<Unit,Representation.Lifted<Logical>>,~SlotsLanes.lane_point(~Unit,~SlotsValues.held(~Unit),~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Logical,~original,
    SlotsLanes.LanePoint{digit,plane_unit(~Logical,logical,channels,group),Representation.Lifted{plane_input(~Logical,logical),Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})}},logical,ShapeLaws.elements([height,width]),
    Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32))),Nat.mul(image,ShapeLaws.elements([height,width]))),
    Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),ShapeLaws.elements([height,width]))):
  +plane = ShapeLaws.elements([height,width])
  +left = Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32))),Nat.mul(image,plane))
  +right = Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),plane)
  +at = {SlotsLanes.LanePoint{digit,Unit{},Representation.Lifted{logical,Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})}} : SlotsLanes.LanePoint<Unit,Representation.Lifted<Logical>>}
  %Addition.zero_right(left) :
    ReduceLaws.Agrees(~SlotsLanes.LanePoint<Unit,Representation.Lifted<Logical>>,~SlotsLanes.lane_point(~Unit,~SlotsValues.held(~Unit),~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Logical,~original,at,logical,plane,_,right)
  %Addition.zero_right(right) :
    ReduceLaws.Agrees(~SlotsLanes.LanePoint<Unit,Representation.Lifted<Logical>>,~SlotsLanes.lane_point(~Unit,~SlotsValues.held(~Unit),~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Logical,~original,at,logical,plane,Nat.add(left,0n),_)
  ReduceLaws.agrees_all(~SlotsLanes.LanePoint<Unit,Representation.Lifted<Logical>>,~SlotsLanes.lane_point(~Unit,~SlotsValues.held(~Unit),~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original)),~Logical,~original,
    ~at,~logical,~plane,~left,~right,
    ~(offset => below => plane_value(~Logical,~original,logical,limits,images,channels,height,width,image,group,digit,cells_room,image_inside,group_inside,small,valid,offset,below)),
    plane,0n,{==})

# The finished sums of the runtime are the rounded totals of their partials.
def finished_rounded(partials: List<&2,Sum.TileParts>) ->
  {Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,partials) == Partials.finished(~Tile.Tile8,~Sum.TileParts,~Sum.tile_rounded_total,partials) : List<&2,Tile.Tile8>}:
  match partials:
    case []: {==}
    case +partial <> +rest:
      Equal.cong(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,tail => Sum.tile_rounded_total(partial) <> tail,
        Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,rest),Partials.finished(~Tile.Tile8,~Sum.TileParts,~Sum.tile_rounded_total,rest),finished_rounded(rest))

# The first extent of extents within a capacity is positive and within it.
def within_positive(+extent: U32,+rest: List<&2,U32>,+cap: U32,+within: PartialsValues.Within(extent <> rest,cap)) -> {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}:
  KernelGemm.Both{+positive,+others} = within
  positive

def within_room(+extent: U32,+rest: List<&2,U32>,+cap: U32,+within: PartialsValues.Within(extent <> rest,cap)) -> {Nat.is_le(U32.to_nat(extent),U32.to_nat(cap)) == True{} : Bool}:
  KernelGemm.Both{+positive,KernelGemm.Both{+room,+others}} = within
  room

# The read of the planes of the cells [N,H,W,G] of a source of cells, the
# groups begun from the extras, finishes the box of the planes and hands back
# the pieces.
def plane_shaped(~Block: Type,~Model: Data,~represent: Model -> Block,~cell: Model -> U32 -> Tile.Tile8,~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,
  ~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Operations.TileSlots<Extra> -> Block -> Block & Operations.TileSlots<Extra>,
  ~vacant: Tile.Tile8 -> Block,~vacant_model: Tile.Tile8 -> Model,~vacant_same: @+fill: Tile.Tile8 -> {vacant(fill) == represent(vacant_model(fill)) : Block},
  ~chain_same: LineChainValues.ChainSame(~Block,~Model,~represent,~Extra,~Operations.TileSlots<Extra>,~SlotsValues.SlotsView<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),
    ~SlotsValues.begin_view(~Extra),~fold,~LineChainValues.strided_model(~Model,~Tile.Tile8,~SlotsValues.SlotsView<Extra>,~cell,~SlotsValues.cell_view(~Extra,~point)),
    ~Operations.slots_closed(~Extra),~SlotsValues.turn_view(~Extra),~Tile.Tile8,~vacant,~vacant_model),
  ~X: Data,~Source: Data,~source_value: Source -> U32 -> Tile.Tile8,~source_at: X -> Cells.Cells -> Source,~Models: Data,~pieces_of: Models -> Partition.Tree<&1,Block,Unit>,
  +x: X,+extras: List<&2,Extra>,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+bytes: F32,+operations: F32,+workers: U32,+models: Models,
  +source: Reduce.Streamed<Block,Model,Tile.Tile8,represent,cell,pieces_of(models),Pieces.ranges_of_grid(Tensors.grid_of([images,height,width,Tiled.groups(channels)])),
    Reduce.values_from(~Tile.Tile8,~Source,~source_value,source_at(x,Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{})),ShapeLaws.elements([images,height,width,Tiled.groups(channels)]),0n)>,
  +turns: Reduce.Turns<Extra,extras,Grid.product([Tiled.groups(channels)])>,
  +room: {Nat.is_le(ShapeLaws.elements([images,height,width,Tiled.groups(channels)]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements([images,height,width,Tiled.groups(channels)])) == True{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements([images,height,width,Tiled.groups(channels)])),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  KernelGemm.Both<Global.Shaped(~Tile.Tile8,~PartialsValues.Box<Sum.TileParts>,~PartialsValues.box_value(~Tile.Tile8,~Sum.TileParts,~Sum.tile_merge,~Sum.tile_rounded_total),
      PartialsValues.Box{PartialsValues.written(~Sum.TileParts,plane_folds(~Extra,~point,~X,~Source,~source_value,~source_at,x,images,channels,height,width,
          Reduce.block_starts(~Extra,~SlotsValues.SlotsView<Extra>,~SlotsValues.begin_view(~Extra),extras,Grid.product([Tiled.groups(channels)]))),Sum.tile_parts_zero()),
        PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],False{}),PartialsValues.axes_with([images,Tiled.groups(channels)],[True{},False{}],True{}),0,Sum.tile_parts_zero()},
      [Tiled.groups(channels)],
      Tensors.with_workers(~Tile.Tile8,Tensors.from_list(~Tile.Tile8,[Tiled.groups(channels)],Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,
        Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Sum.TileParts>,Partials.read(~Block,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),~fold,~Operations.slots_closed(~Extra),~Sum.tile_merge,~vacant,extras,bytes,operations,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces_of(models)))),Tiled.zero_tile()),workers)),
    {Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Sum.TileParts>,Partials.read(~Block,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),~fold,~Operations.slots_closed(~Extra),~Sum.tile_merge,~vacant,extras,bytes,operations,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces_of(models))) == pieces_of(models) : Partition.Tree<&1,Block,Unit>}>:
  +shape = {[images,height,width,Tiled.groups(channels)] : List<&2,U32>}
  +capacity = Machine.capacity(limits)
  +groups = Tiled.groups(channels)
  +count = U32.to_nat(groups)
  +inner = Grid.product([Tiled.groups(channels)])
  +plane = ShapeLaws.elements([height,width])
  +context = source_at(x,Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{}))
  +box = {PartialsValues.Box{PartialsValues.written(~Sum.TileParts,plane_folds(~Extra,~point,~X,~Source,~source_value,~source_at,x,images,channels,height,width,
        Reduce.block_starts(~Extra,~SlotsValues.SlotsView<Extra>,~SlotsValues.begin_view(~Extra),extras,inner)),Sum.tile_parts_zero()),
      PartialsValues.axes_with([images,groups],[True{},False{}],False{}),PartialsValues.axes_with([images,groups],[True{},False{}],True{}),0,Sum.tile_parts_zero()} : PartialsValues.Box<Sum.TileParts>}
  +outputs = PartialsValues.kept_count(PartialsValues.axes_with([images,groups],[True{},False{}],False{}))
  +parts = ListReadback.enumerate(~Sum.TileParts,~PartialsValues.Box<Sum.TileParts>,~PartialsValues.box_output(~Sum.TileParts,~Sum.tile_merge),outputs,0n,box)
  +finals = ListReadback.enumerate(~Tile.Tile8,~PartialsValues.Box<Sum.TileParts>,~PartialsValues.box_finished(~Tile.Tile8,~Sum.TileParts,~Sum.tile_merge,~Sum.tile_rounded_total),outputs,0n,box)
  +values = Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Sum.TileParts>,Partials.read(~Block,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),~fold,~Operations.slots_closed(~Extra),~Sum.tile_merge,~vacant,extras,bytes,operations,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces_of(models))))
  +within_cells = PartialsValues.within_of(shape,capacity,CountCompletion.positive_not_zero(shape,positive),room)
  +within_box = PartialsValues.within_removed([images,width,groups],1n,capacity,PartialsValues.within_removed(shape,1n,capacity,within_cells))
  +within_groups = PartialsValues.within_removed([images,groups],0n,capacity,within_box)
  +within_plane = PartialsValues.within_removed([height,width,groups],2n,capacity,PartialsValues.within_removed(shape,0n,capacity,within_cells))
  +groups_positive = within_positive(groups,[],capacity,within_groups)
  +groups_room = within_room(groups,[],capacity,within_groups)
  +plane_positive = Order.product_positive(U32.to_nat(height),Nat.mul(U32.to_nat(width),1n),within_positive(height,[width],capacity,within_plane),
    Order.product_positive(U32.to_nat(width),1n,within_positive(width,[],capacity,PartialsValues.within_removed([height,width],0n,capacity,within_plane)),{==}))
  +inner_count = Equal.cong(U32,Nat,word => U32.to_nat(word),inner,groups,Arithmetic.multiply_one(groups))
  +inner_positive = {%Equal.sym(Nat,U32.to_nat(inner),count,inner_count) : {Nat.is_lt(0n,_) == True{} : Bool} groups_positive : {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool}}
  +size = plane_size(limits,images,channels,height,width,room,positive)
  +length_exact = Equal.trans(Nat,U32.to_nat((Grid.product([height,width]) * inner : U32)),Nat.mul(plane,count),ShapeLaws.elements([height,width,groups]),size,plane_whole(channels,height,width))
  +inner_le = {%Equal.sym(Nat,U32.to_nat((Grid.product([height,width]) * inner : U32)),Nat.mul(plane,count),size) : {Nat.is_le(U32.to_nat(inner),_) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(inner),count,inner_count) : {Nat.is_le(_,Nat.mul(plane,count)) == True{} : Bool}
    {%RowMajor.multiply_commutative(count,plane) : {Nat.is_le(count,_) == True{} : Bool} RowMajor.product_covers(count,plane,plane_positive) : {Nat.is_le(count,Nat.mul(plane,count)) == True{} : Bool}} :
    {Nat.is_le(U32.to_nat(inner),U32.to_nat((Grid.product([height,width]) * inner : U32))) == True{} : Bool}}
  +box_exact = Equal.trans(Nat,ShapeLaws.elements([images,groups]),Nat.mul(ShapeLaws.elements([images]),count),Nat.mul(ShapeLaws.elements([images]),U32.to_nat(inner)),pair_elements(images,groups),
    Equal.cong(Nat,Nat,number => Nat.mul(ShapeLaws.elements([images]),number),count,U32.to_nat(inner),Equal.sym(Nat,U32.to_nat(inner),count,inner_count)))
  +read_eq = PartialsValues.read_values(~Block,~Model,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~SlotsValues.SlotsView<Extra>,~Sum.TileParts,
    ~represent,~cell,~SlotsValues.cell_view(~Extra,~point),~Operations.slots_begun(~Extra),~SlotsValues.begin_view(~Extra),~fold,
    ~Operations.slots_closed(~Extra),~SlotsValues.turn_view(~Extra),~Sum.tile_merge,~Tile.Tile8,~vacant,~vacant_model,~vacant_same,~chain_same,~Source,~source_value,
    limits,shape,1n,Grid.product([height,width]),inner,extras,bytes,operations,Tile.filled(0.0),workers,context,pieces_of(models),[images,groups],[True{},False{}],Sum.tile_parts_zero(),BNCells.summed_flags(),1n,2n,True{},
    {==},{==},{==},{==},room,positive,length_exact,turns,inner_positive,inner_le,cap,cap_ok,count_room,source,within_box,box_exact,{==})
  +listed = Equal.trans(List<&2,Tile.Tile8>,values,Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,parts),finals,
    Equal.cong(Partition.Tree<&1,Block,Unit> & List<&2,Sum.TileParts>,List<&2,Tile.Tile8>,result => Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Sum.TileParts>,result)),
      Partials.read(~Block,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),~fold,~Operations.slots_closed(~Extra),~Sum.tile_merge,~vacant,extras,bytes,operations,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces_of(models)),(pieces_of(models),parts),read_eq),
    Equal.trans(List<&2,Tile.Tile8>,Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,parts),Partials.finished(~Tile.Tile8,~Sum.TileParts,~Sum.tile_rounded_total,parts),finals,
      finished_rounded(parts),PartialsValues.finished_enumerate(~Tile.Tile8,~Sum.TileParts,~Sum.tile_merge,~Sum.tile_rounded_total,outputs,0n,box)))
  +groups_one = Equal.sym(Nat,Nat.mul(count,1n),count,Order.times_one(count))
  +length = Equal.trans(Nat,List.length(&2,Tile.Tile8,values),List.length(&2,Tile.Tile8,finals),ShapeLaws.elements([groups]),
    Equal.cong(List<&2,Tile.Tile8>,Nat,list => List.length(&2,Tile.Tile8,list),values,finals,listed),
    Equal.trans(Nat,List.length(&2,Tile.Tile8,finals),outputs,ShapeLaws.elements([groups]),
      ListReadback.enumerate_length(~Tile.Tile8,~PartialsValues.Box<Sum.TileParts>,~PartialsValues.box_finished(~Tile.Tile8,~Sum.TileParts,~Sum.tile_merge,~Sum.tile_rounded_total),outputs,0n,box),
      Equal.trans(Nat,outputs,count,ShapeLaws.elements([groups]),inner_count,groups_one)))
  +groups_counted = StorageCount.counted_within(limits,[groups],{%groups_one : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} groups_room : {Nat.is_le(ShapeLaws.elements([groups]),U32.to_nat(capacity)) == True{} : Bool}},default_capacity)
  +groups_nonempty = {%groups_one : {Nat.is_lt(0n,_) == True{} : Bool} groups_positive : {Nat.is_lt(0n,ShapeLaws.elements([groups])) == True{} : Bool}}
  +ready = Listed.listed_ready(~Tile.Tile8,[groups],values,Tiled.zero_tile(),workers,groups_counted,length)
  KernelGemm.Both{PartialsValues.listed_shaped(~Tile.Tile8,~Sum.TileParts,~Sum.tile_merge,~Sum.tile_rounded_total,box,[groups],values,Tiled.zero_tile(),workers,outputs,listed,groups_nonempty,ready),
    Equal.cong(Partition.Tree<&1,Block,Unit> & List<&2,Sum.TileParts>,Partition.Tree<&1,Block,Unit>,result => Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Sum.TileParts>,result),Partials.read(~Block,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),~fold,~Operations.slots_closed(~Extra),~Sum.tile_merge,~vacant,extras,bytes,operations,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces_of(models)),(pieces_of(models),parts),read_eq)}

# The sums the statistics read from the planes of their source, as kept
# cells, hold the sums of the planes of every channel.
def plane_contract(~Block: Type,~Extra: Data,~point: Extra -> Tile.Tile8 -> Tile.Tile8,
  ~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: Operations.TileSlots<Extra> -> Block -> Block & Operations.TileSlots<Extra>,~vacant: Tile.Tile8 -> Block,
  ~X: Data,~Source: Data,~source_value: Source -> U32 -> Tile.Tile8,~source_at: X -> Cells.Cells -> Source,~source_moved: SourceMoved(~X,~Source,~source_value,~source_at),
  ~Models: Data,~pieces_of: Models -> Partition.Tree<&1,Block,Unit>,~Logical: Data,~original: Logical -> U32 -> F32,
  ~input_of: Logical -> X,~extras_of: Logical -> U32 -> List<&2,Extra>,~extra_of: Logical -> U32 -> Nat -> Extra,~Ready: Logical -> U32 -> U32 -> U32 -> U32 -> Data,
  ~started: PlaneStarted(~Extra,~Logical,~extras_of,~extra_of),~enough: PlaneEnough(~Extra,~Logical,~extras_of),
  ~agrees: PlaneAgrees(~Extra,~point,~X,~Source,~source_value,~source_at,~Logical,~original,~input_of,~extra_of,~Ready),
  +logical: Logical,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+bytes: F32,+operations: F32,+workers: U32,+models: Models,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(4n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rooms: ReduceLaws.KeptRooms<limits,[images,channels,height,width],[0n,2n,3n],1n,Grid.Outermost{}>,+ready: Ready(logical,images,channels,height,width),
  +read_facts: KernelGemm.Both<Global.Shaped(~Tile.Tile8,~PartialsValues.Box<Sum.TileParts>,~PartialsValues.box_value(~Tile.Tile8,~Sum.TileParts,~Sum.tile_merge,~Sum.tile_rounded_total),
      plane_cells(~Extra,~point,~X,~Source,~source_value,~source_at,~Logical,~input_of,~extras_of,Plane{logical,images,channels,height,width},[images,channels,height,width],[0n,2n,3n],1n,Grid.Outermost{}),
      [Tiled.groups(channels)],
      Tensors.with_workers(~Tile.Tile8,Tensors.from_list(~Tile.Tile8,[Tiled.groups(channels)],Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,
        Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Sum.TileParts>,Partials.read(~Block,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),~fold,~Operations.slots_closed(~Extra),~Sum.tile_merge,~vacant,extras_of(logical,channels),bytes,operations,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces_of(models)))),Tiled.zero_tile()),workers)),
    {Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Sum.TileParts>,Partials.read(~Block,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),~fold,~Operations.slots_closed(~Extra),~Sum.tile_merge,~vacant,extras_of(logical,channels),bytes,operations,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces_of(models))) == pieces_of(models) : Partition.Tree<&1,Block,Unit>}>) ->
  KernelGemm.Both<PlaneSummed(~Logical,~original,logical,images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,
      Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Sum.TileParts>,Partials.read(~Block,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),~fold,~Operations.slots_closed(~Extra),~Sum.tile_merge,~vacant,extras_of(logical,channels),bytes,operations,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces_of(models)))))),
    {Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Sum.TileParts>,Partials.read(~Block,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),~fold,~Operations.slots_closed(~Extra),~Sum.tile_merge,~vacant,extras_of(logical,channels),bytes,operations,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces_of(models))) == pieces_of(models) : Partition.Tree<&1,Block,Unit>}>:
  KernelGemm.Both{+shaped,+kept} = read_facts
  KernelGemm.Both{ReduceLaws.kept_contract(~Plane<Logical>,~plane_fits(~Logical,~Ready),~PartialsValues.Box<Sum.TileParts>,~PartialsValues.box_value(~Tile.Tile8,~Sum.TileParts,~Sum.tile_merge,~Sum.tile_rounded_total),
      ~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),~plane_cells(~Extra,~point,~X,~Source,~source_value,~source_at,~Logical,~input_of,~extras_of),~plane_values(~Logical,~original),
      ~ReduceLaws.kept_agree(~Plane<Logical>,~plane_fits(~Logical,~Ready),~PartialsValues.Box<Sum.TileParts>,~PartialsValues.box_value(~Tile.Tile8,~Sum.TileParts,~Sum.tile_merge,~Sum.tile_rounded_total),
        ~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),~plane_cells(~Extra,~point,~X,~Source,~source_value,~source_at,~Logical,~input_of,~extras_of),~plane_values(~Logical,~original),
        ~plane_lane(~Extra,~point,~X,~Source,~source_value,~source_at,~source_moved,~Logical,~original,~input_of,~extras_of,~extra_of,~Ready,~started,~enough,~agrees)),
      Plane{logical,images,channels,height,width},limits,[images,channels,height,width],[0n,2n,3n],1n,Grid.Outermost{},{==},{==},rank_room,rooms,
      KernelGemm.Both{{==},KernelGemm.Both{{==},KernelGemm.Both{{==},KernelGemm.Both{{==},ready}}}},channels,{==},default_capacity,
      Tensors.with_workers(~Tile.Tile8,Tensors.from_list(~Tile.Tile8,[Tiled.groups(channels)],Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,
        Pair.snd(Partition.Tree<&1,Block,Unit>,List<&2,Sum.TileParts>,Partials.read(~Block,~Tile.Tile8,~Extra,~Operations.TileSlots<Extra>,~Sum.TileParts,~Operations.slots_begun(~Extra),~fold,~Operations.slots_closed(~Extra),~Sum.tile_merge,~vacant,extras_of(logical,channels),bytes,operations,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,pieces_of(models)))),Tiled.zero_tile()),workers),shaped),kept}

# The sums begin every group from no extra.
def sums_extras(~Logical: Data,logical: Logical,channels: U32) -> List<&2,Unit>:
  [Unit{}]

def sums_started(~Logical: Data,+logical: Logical,+channels: U32,+group: Nat,+inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool}) ->
  {ListReadback.at(~SlotsValues.SlotsView<Unit>,SlotsValues.begin_view(~Unit,plane_unit(~Logical,logical,channels,group)),group,
      Reduce.block_starts(~Unit,~SlotsValues.SlotsView<Unit>,~SlotsValues.begin_view(~Unit),sums_extras(~Logical,logical,channels),Grid.product([Tiled.groups(channels)]))) ==
    SlotsValues.begin_view(~Unit,plane_unit(~Logical,logical,channels,group)) : SlotsValues.SlotsView<Unit>}:
  single_drawn(~Unit,~SlotsValues.SlotsView<Unit>,~SlotsValues.begin_view(~Unit),U32.to_nat(Grid.product([Tiled.groups(channels)])),Unit{},group)

def sums_enough(~Logical: Data,+logical: Logical,+channels: U32) ->
  {Nat.is_le(U32.to_nat(Grid.product([Tiled.groups(channels)])),List.length(&2,SlotsValues.SlotsView<Unit>,
    Reduce.block_starts(~Unit,~SlotsValues.SlotsView<Unit>,~SlotsValues.begin_view(~Unit),sums_extras(~Logical,logical,channels),Grid.product([Tiled.groups(channels)])))) == True{} : Bool}:
  +count = U32.to_nat(Grid.product([Tiled.groups(channels)]))
  %Equal.sym(Nat,List.length(&2,SlotsValues.SlotsView<Unit>,LineChainValues.drawn(~Unit,~SlotsValues.SlotsView<Unit>,~SlotsValues.begin_view(~Unit),count,[Unit{}],[Unit{}])),count,
    PartialsValues.drawn_length(~Unit,~SlotsValues.SlotsView<Unit>,~SlotsValues.begin_view(~Unit),count,[Unit{}],[Unit{}],{==},{==})) : {Nat.is_le(count,_) == True{} : Bool}
  Order.reflexive(count)

# The sums of the runtime are the finished partials of their read, and its
# pieces.
def sums_parts(result: Pieces.Pieces(~Tile.Tile8) & List<&2,Sum.TileParts>) ->
  {Pair.snd(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.sums_of(result)) == Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,Pair.snd(Pieces.Pieces(~Tile.Tile8),List<&2,Sum.TileParts>,result)) : List<&2,Tile.Tile8>}:
  match result:
    case (pieces,partials): {==}

def sums_pieces(result: Pieces.Pieces(~Tile.Tile8) & List<&2,Sum.TileParts>) ->
  {Pair.fst(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.sums_of(result)) == Pair.fst(Pieces.Pieces(~Tile.Tile8),List<&2,Sum.TileParts>,result) : Pieces.Pieces(~Tile.Tile8)}:
  match result:
    case (pieces,partials): {==}

# The sums of the statistics of a batch normalization of tiled values [N,C,H,W]
# are the sums of the planes of every channel.
def channel_sums_summed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+workers: U32,
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+cell_workers: U32,
  +contract: Public.Contract(~Logical,~original,logical,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),[images,height,width,Tiled.groups(channels)],fill,Tensors.Usable{},cell_workers},1n,channels,Grid.Innermost{}}),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(4n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements([images,channels,height,width])) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements([images,height,width,Tiled.groups(channels)])),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  KernelGemm.Both<PlaneSummed(~Logical,~original,logical,images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Pair.snd(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.sums([images,height,width,Tiled.groups(channels)],workers,PiecesModel.pack(~Tile.Tile8,model))))),
    {Pair.fst(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.sums([images,height,width,Tiled.groups(channels)],workers,PiecesModel.pack(~Tile.Tile8,model))) == PiecesModel.pack(~Tile.Tile8,model) : Pieces.Pieces(~Tile.Tile8)}>:
  KernelGemm.Both{+global,+laid} = contract
  Public.Laid{+inside,+shaped_cells,+restored,+counted,+pinned} = laid
  +values = {[images,channels,height,width] : List<&2,U32>}
  +shape = {[images,height,width,Tiled.groups(channels)] : List<&2,U32>}
  +capacity = U32.to_nat(Machine.capacity(limits))
  +inner = Grid.product([Tiled.groups(channels)])
  +values_room = StorageCount.counted_room(limits,values,counted,default_capacity)
  +rooms_out = ReduceLaws.kept_rooms(limits,values,[0n,2n,3n],1n,Grid.Outermost{},{==},{==},values_room,nonempty)
  +rooms_in = ReduceLaws.kept_rooms(limits,values,[0n,2n,3n],1n,Grid.Innermost{},{==},{==},values_room,nonempty)
  +room_in = Order.transitive(ShapeLaws.elements(shape),ShapeLaws.elements(values),capacity,ReduceLaws.rooms_cells_fewer(limits,values,[0n,2n,3n],1n,Grid.Innermost{},rooms_in),values_room)
  +positive_in = ReduceLaws.rooms_cells_positive(limits,values,[0n,2n,3n],1n,Grid.Innermost{},rooms_in)
  +turns = {Reduce.Turns{U32.to_nat(inner),{==},Equal.sym(Nat,Nat.mul(U32.to_nat(inner),1n),U32.to_nat(inner),Order.times_one(U32.to_nat(inner)))} : Reduce.Turns<Unit,[Unit{}],inner>}
  +summed = plane_contract(~BNCells.Cells(),~Unit,~SlotsValues.held(~Unit),~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),~Pieces.vacant_buffer(~Tile.Tile8),
    ~Logical,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~lifted_at(~Logical),~cell_moved(~Logical,~original),
    ~PiecesModel.Model<Tile.Tile8>,~PiecesModel.pack(~Tile.Tile8),~Logical,~original,
    ~plane_input(~Logical),~sums_extras(~Logical),~plane_unit(~Logical),~sums_ready(~Logical),~sums_started(~Logical),~sums_enough(~Logical),~sums_agrees(~Logical,~original),
    logical,limits,images,channels,height,width,7.0,2.0,workers,model,default_capacity,rank_room,rooms_out,Unit{},
    plane_shaped(~BNCells.Cells(),~Buffer.Model<Tile.Tile8>,~Buffer.pack(~Tile.Tile8),~Reduce.cell(~Tile.Tile8),~Unit,~SlotsValues.held(~Unit),~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),
      ~Pieces.vacant_buffer(~Tile.Tile8),~Reduce.vacant_model(~Tile.Tile8),~Reduce.vacant_same(~Tile.Tile8),
      ~SlotsValues.slots_chain(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit),~SlotsValues.held(~Unit),~SlotsValues.spaced_matches(~Unit)),
      ~Logical,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~lifted_at(~Logical),~PiecesModel.Model<Tile.Tile8>,~PiecesModel.pack(~Tile.Tile8),
      logical,[Unit{}],limits,images,channels,height,width,7.0,2.0,workers,model,
      PartialsValues.contract_streamed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
        Representation.Lifted{logical,Public.geometry_of(values,1n,Grid.Innermost{})},model,shape,fill,cell_workers,global),
      turns,room_in,positive_in,default_capacity,cap,cap_ok,count_room))
  %Equal.sym(List<&2,Tile.Tile8>,Pair.snd(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.sums_of(Partials.read(~BNCells.Cells(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),~Operations.slots_closed(~Unit),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),[Unit{}],7.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))),Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,Pair.snd(Pieces.Pieces(~Tile.Tile8),List<&2,Sum.TileParts>,Partials.read(~BNCells.Cells(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),~Operations.slots_closed(~Unit),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),[Unit{}],7.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))),sums_parts(Partials.read(~BNCells.Cells(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),~Operations.slots_closed(~Unit),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),[Unit{}],7.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))) :
    KernelGemm.Both<PlaneSummed(~Logical,~original,logical,images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},_)),
      {Pair.fst(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.sums_of(Partials.read(~BNCells.Cells(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),~Operations.slots_closed(~Unit),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),[Unit{}],7.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))) == PiecesModel.pack(~Tile.Tile8,model) : Pieces.Pieces(~Tile.Tile8)}>
  %Equal.sym(Pieces.Pieces(~Tile.Tile8),Pair.fst(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.sums_of(Partials.read(~BNCells.Cells(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),~Operations.slots_closed(~Unit),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),[Unit{}],7.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))),Pair.fst(Pieces.Pieces(~Tile.Tile8),List<&2,Sum.TileParts>,Partials.read(~BNCells.Cells(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),~Operations.slots_closed(~Unit),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),[Unit{}],7.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model))),sums_pieces(Partials.read(~BNCells.Cells(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),~Operations.slots_closed(~Unit),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),[Unit{}],7.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))) :
    KernelGemm.Both<PlaneSummed(~Logical,~original,logical,images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,Pair.snd(Pieces.Pieces(~Tile.Tile8),List<&2,Sum.TileParts>,Partials.read(~BNCells.Cells(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~Operations.cells_slots(~Unit,~Traversal.spaced(~Tile.Tile8,~Unit),~TraversalWitness.spaced_read(~Tile.Tile8,~Unit)),~Operations.slots_closed(~Unit),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),[Unit{}],7.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))))),
      {_ == PiecesModel.pack(~Tile.Tile8,model) : Pieces.Pieces(~Tile.Tile8)}>
  summed

# ---- The centered squares of the statistics ----

# Lines drawn from contexts that do not run out begin from the context at
# their place.
def drawn_at(~Context: Data,~Value: Data,~begin_value: Context -> Value,remaining: Nat,+all: List<&2,Context>,upcoming: List<&2,Context>,index: Nat,+fallback: Context,
  +inside: {Nat.is_lt(index,remaining) == True{} : Bool},+enough: {Nat.is_le(remaining,List.length(&2,Context,upcoming)) == True{} : Bool}) ->
  {ListReadback.at(~Value,begin_value(fallback),index,LineChainValues.drawn(~Context,~Value,~begin_value,remaining,all,upcoming)) == begin_value(ListReadback.at(~Context,fallback,index,upcoming)) : Value}:
  match remaining upcoming index:
    case 0n upcoming +index:
      Empty.absurd({ListReadback.at(~Value,begin_value(fallback),index,LineChainValues.drawn(~Context,~Value,~begin_value,0n,all,upcoming)) == begin_value(ListReadback.at(~Context,fallback,index,upcoming)) : Value},
        Order.below_zero(index,inside))
    case 1n+ +rest Nil{} index:
      Empty.absurd({ListReadback.at(~Value,begin_value(fallback),index,LineChainValues.drawn(~Context,~Value,~begin_value,1n+rest,all,[])) == begin_value(ListReadback.at(~Context,fallback,index,[])) : Value},
        WordEncoding.false_true(enough))
    case 1n+ +rest Con{+context,later} 0n: {==}
    case 1n+ +rest Con{+context,Nil{}} 1n+ +previous:
      Empty.absurd({ListReadback.at(~Value,begin_value(fallback),1n+previous,LineChainValues.drawn(~Context,~Value,~begin_value,1n+rest,all,[context])) == begin_value(ListReadback.at(~Context,fallback,1n+previous,[context])) : Value},
        Order.below_zero(previous,{%Order.zero_below(rest,enough) : {Nat.is_lt(previous,_) == True{} : Bool} inside : {Nat.is_lt(previous,0n) == True{} : Bool}}))
    case 1n+ +rest Con{+context,Con{+next,+tail}} 1n+ +previous: drawn_at(~Context,~Value,~begin_value,rest,all,next <> tail,previous,fallback,inside,enough)

# Lines drawn from contexts that do not run out are all there.
def drawn_full(~Context: Data,~Value: Data,~begin_value: Context -> Value,remaining: Nat,+all: List<&2,Context>,upcoming: List<&2,Context>,
  +enough: {Nat.is_le(remaining,List.length(&2,Context,upcoming)) == True{} : Bool}) ->
  {List.length(&2,Value,LineChainValues.drawn(~Context,~Value,~begin_value,remaining,all,upcoming)) == remaining : Nat}:
  match remaining upcoming:
    case 0n upcoming: {==}
    case 1n+ +rest Nil{}:
      Empty.absurd({List.length(&2,Value,LineChainValues.drawn(~Context,~Value,~begin_value,1n+rest,all,[])) == 1n+rest : Nat},WordEncoding.false_true(enough))
    case 1n+ +rest Con{+context,Nil{}}:
      %Equal.sym(Nat,rest,0n,Order.zero_below(rest,enough)) :
        {List.length(&2,Value,LineChainValues.drawn(~Context,~Value,~begin_value,1n+_,all,[context])) == 1n+_ : Nat}
      {==}
    case 1n+ +rest Con{+context,Con{+next,+tail}}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Value,LineChainValues.drawn(~Context,~Value,~begin_value,rest,all,next <> tail)),rest,
        drawn_full(~Context,~Value,~begin_value,rest,all,next <> tail,enough))

# The means of the channels of values in a geometry of the values.
type ChannelMeans<-Means: Data> is Data:
  ChannelMeans{means: Means,geometry: Cells.Cells}

# The mean of the channel of a value.
def channel_mean(~Means: Data,~mean: Means -> U32 -> F32,context: ChannelMeans<Means>,index: U32) -> F32:
  ChannelMeans{means,geometry} = context
  mean(means,U32.from_nat(Representation.value_channel(geometry,U32.to_nat(index))))

def centered_pair(left: F32,right: F32) -> F32:
  +low = left
  +high = right
  ((low - high : F32) * (low - high : F32) : F32)

# The cells of the means of the channels, one a group.
def mean_cell(~Means: Data,~mean: Means -> U32 -> F32,means: Representation.Lifted<Means>,group: Nat) -> Tile.Tile8:
  Representation.Lifted{+held,+geometry} = means
  Representation.cell(~Means,~mean,held,geometry,0n,group)

# The logical index of a channel of the means is the channel.
def mean_index(+lanes: U32,+channel: Nat) -> {Representation.logical_index(Public.geometry_of([1,lanes,1,1],1n,Grid.Outermost{}),0n,channel) == channel : Nat}:
  Equal.trans(Nat,Representation.logical_index(Public.geometry_of([1,lanes,1,1],1n,Grid.Outermost{}),0n,channel),Nat.mul(channel,1n),channel,
    Addition.zero_right(Nat.mul(channel,1n)),Order.times_one(channel))

# A centered square of a cell on the mean of its group.
def centered_point(mean: Tile.Tile8,cell: Tile.Tile8) -> Tile.Tile8:
  BNKernel.squared(Tile.minus(cell,mean))

# The centered read of the runtime samples the centered square of its cell.
def centered_matches(+reader: Tree.Tree<Tile.Tile8>,+context: Traversal.Spaced<Tile.Tile8>,+position: U32) ->
  {BNKernel.read_centered(ArrayLaws.pack_tree(~Tile.Tile8,reader),context,position) ==
    (ArrayLaws.pack_tree(~Tile.Tile8,reader),SlotsValues.spaced_sample(~Tile.Tile8,~centered_point,reader,context,position)) : Array<Tile.Tile8> & Tile.Tile8}:
  Traversal.Spaced{+base,+stride,+mean} = context
  Equal.cong(Array<Tile.Tile8> & Tile.Tile8,Array<Tile.Tile8> & Tile.Tile8,sample => BNKernel.centered_cell(mean,sample),
    Array.get(Tile.Tile8,ArrayLaws.pack_tree(~Tile.Tile8,reader),(base + position * stride : U32)),
    (ArrayLaws.pack_tree(~Tile.Tile8,reader),Tree.read(Tile.Tile8,reader,(base + position * stride : U32))),
    StorageLaws.read(Tile.Tile8,reader,(base + position * stride : U32)))

# The input of the centered squares.
def centered_input(~X: Data,~Means: Data,logical: CombineLeaf.Context<X,ChannelMeans<Means>>) -> X:
  CombineLeaf.Context{+left,+right} = logical
  left

# The cells of the means of the groups the centered squares begin from.
def centered_extras(~X: Data,~Means: Data,~mean: Means -> U32 -> F32,logical: CombineLeaf.Context<X,ChannelMeans<Means>>,channels: U32) -> List<&2,Tile.Tile8>:
  CombineLeaf.Context{+left,+right} = logical
  ChannelMeans{+means,+geometry} = right
  +lanes = channels
  ListReadback.enumerate(~Tile.Tile8,~Representation.Lifted<Means>,~mean_cell(~Means,~mean),U32.to_nat(Tiled.groups(lanes)),0n,
    Representation.Lifted{means,Public.geometry_of([1,lanes,1,1],1n,Grid.Outermost{})})

def centered_extra(~X: Data,~Means: Data,~mean: Means -> U32 -> F32,logical: CombineLeaf.Context<X,ChannelMeans<Means>>,channels: U32,group: Nat) -> Tile.Tile8:
  CombineLeaf.Context{+left,+right} = logical
  ChannelMeans{+means,+geometry} = right
  mean_cell(~Means,~mean,Representation.Lifted{means,Public.geometry_of([1,channels,1,1],1n,Grid.Outermost{})},group)

# The centered squares find the channel of a value in the geometry of the
# values.
def centered_ready(~X: Data,~Means: Data,logical: CombineLeaf.Context<X,ChannelMeans<Means>>,images: U32,channels: U32,height: U32,width: U32) -> Data:
  CombineLeaf.Context{+left,+right} = logical
  ChannelMeans{+means,+geometry} = right
  {geometry == Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{}) : Cells.Cells}

# A group begins its plane from the cell of its means.
def centered_started(~X: Data,~Means: Data,~mean: Means -> U32 -> F32,+logical: CombineLeaf.Context<X,ChannelMeans<Means>>,+channels: U32,+group: Nat,
  +inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool}) ->
  {ListReadback.at(~SlotsValues.SlotsView<Tile.Tile8>,SlotsValues.begin_view(~Tile.Tile8,centered_extra(~X,~Means,~mean,logical,channels,group)),group,
      Reduce.block_starts(~Tile.Tile8,~SlotsValues.SlotsView<Tile.Tile8>,~SlotsValues.begin_view(~Tile.Tile8),centered_extras(~X,~Means,~mean,logical,channels),Grid.product([Tiled.groups(channels)]))) ==
    SlotsValues.begin_view(~Tile.Tile8,centered_extra(~X,~Means,~mean,logical,channels,group)) : SlotsValues.SlotsView<Tile.Tile8>}:
  CombineLeaf.Context{+left,+right} = logical
  ChannelMeans{+means,+geometry} = right
  +count = U32.to_nat(Tiled.groups(channels))
  +inner = Grid.product([Tiled.groups(channels)])
  +lifted = {Representation.Lifted{means,Public.geometry_of([1,channels,1,1],1n,Grid.Outermost{})} : Representation.Lifted<Means>}
  +cells = ListReadback.enumerate(~Tile.Tile8,~Representation.Lifted<Means>,~mean_cell(~Means,~mean),count,0n,lifted)
  +first = mean_cell(~Means,~mean,lifted,group)
  +inner_count = Equal.cong(U32,Nat,word => U32.to_nat(word),inner,Tiled.groups(channels),Arithmetic.multiply_one(Tiled.groups(channels)))
  +length = ListReadback.enumerate_length(~Tile.Tile8,~Representation.Lifted<Means>,~mean_cell(~Means,~mean),count,0n,lifted)
  +enough = {%Equal.sym(Nat,List.length(&2,Tile.Tile8,cells),count,length) : {Nat.is_le(U32.to_nat(inner),_) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(inner),count,inner_count) : {Nat.is_le(_,count) == True{} : Bool}
    Order.reflexive(count) : {Nat.is_le(U32.to_nat(inner),List.length(&2,Tile.Tile8,cells)) == True{} : Bool}}
  +below = {%Equal.sym(Nat,U32.to_nat(inner),count,inner_count) : {Nat.is_lt(group,_) == True{} : Bool} inside : {Nat.is_lt(group,U32.to_nat(inner)) == True{} : Bool}}
  Equal.trans(SlotsValues.SlotsView<Tile.Tile8>,
    ListReadback.at(~SlotsValues.SlotsView<Tile.Tile8>,SlotsValues.begin_view(~Tile.Tile8,first),group,LineChainValues.drawn(~Tile.Tile8,~SlotsValues.SlotsView<Tile.Tile8>,~SlotsValues.begin_view(~Tile.Tile8),U32.to_nat(inner),cells,cells)),
    SlotsValues.begin_view(~Tile.Tile8,ListReadback.at(~Tile.Tile8,first,group,cells)),SlotsValues.begin_view(~Tile.Tile8,first),
    drawn_at(~Tile.Tile8,~SlotsValues.SlotsView<Tile.Tile8>,~SlotsValues.begin_view(~Tile.Tile8),U32.to_nat(inner),cells,cells,group,first,below,enough),
    Equal.cong(Tile.Tile8,SlotsValues.SlotsView<Tile.Tile8>,tile => SlotsValues.begin_view(~Tile.Tile8,tile),ListReadback.at(~Tile.Tile8,first,group,cells),first,
      ListReadback.enumerate_at(~Tile.Tile8,~Representation.Lifted<Means>,~mean_cell(~Means,~mean),first,count,group,0n,lifted,inside)))

# Every group has the cell of its means.
def centered_enough(~X: Data,~Means: Data,~mean: Means -> U32 -> F32,+logical: CombineLeaf.Context<X,ChannelMeans<Means>>,+channels: U32) ->
  {Nat.is_le(U32.to_nat(Grid.product([Tiled.groups(channels)])),List.length(&2,SlotsValues.SlotsView<Tile.Tile8>,
    Reduce.block_starts(~Tile.Tile8,~SlotsValues.SlotsView<Tile.Tile8>,~SlotsValues.begin_view(~Tile.Tile8),centered_extras(~X,~Means,~mean,logical,channels),Grid.product([Tiled.groups(channels)])))) == True{} : Bool}:
  CombineLeaf.Context{+left,+right} = logical
  ChannelMeans{+means,+geometry} = right
  +count = U32.to_nat(Tiled.groups(channels))
  +inner = Grid.product([Tiled.groups(channels)])
  +lifted = {Representation.Lifted{means,Public.geometry_of([1,channels,1,1],1n,Grid.Outermost{})} : Representation.Lifted<Means>}
  +cells = ListReadback.enumerate(~Tile.Tile8,~Representation.Lifted<Means>,~mean_cell(~Means,~mean),count,0n,lifted)
  +inner_count = Equal.cong(U32,Nat,word => U32.to_nat(word),inner,Tiled.groups(channels),Arithmetic.multiply_one(Tiled.groups(channels)))
  +length = ListReadback.enumerate_length(~Tile.Tile8,~Representation.Lifted<Means>,~mean_cell(~Means,~mean),count,0n,lifted)
  +enough = {%Equal.sym(Nat,List.length(&2,Tile.Tile8,cells),count,length) : {Nat.is_le(U32.to_nat(inner),_) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(inner),count,inner_count) : {Nat.is_le(_,count) == True{} : Bool}
    Order.reflexive(count) : {Nat.is_le(U32.to_nat(inner),List.length(&2,Tile.Tile8,cells)) == True{} : Bool}}
  %Equal.sym(Nat,List.length(&2,SlotsValues.SlotsView<Tile.Tile8>,LineChainValues.drawn(~Tile.Tile8,~SlotsValues.SlotsView<Tile.Tile8>,~SlotsValues.begin_view(~Tile.Tile8),U32.to_nat(inner),cells,cells)),U32.to_nat(inner),
    drawn_full(~Tile.Tile8,~SlotsValues.SlotsView<Tile.Tile8>,~SlotsValues.begin_view(~Tile.Tile8),U32.to_nat(inner),cells,cells,enough)) : {Nat.is_le(U32.to_nat(inner),_) == True{} : Bool}
  Order.reflexive(U32.to_nat(inner))

# A lane of the centered square of a cell of x with the groups outermost, at a
# point of the plane of an image, is the centered square of the value of the
# channel of the lane there on the mean of the channel.
def centered_value(~X: Data,~xv: X -> U32 -> F32,~Means: Data,~mean: Means -> U32 -> F32,+x: X,+means: Means,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+image: Nat,+group: Nat,+digit: Nat,
  +room: {Nat.is_le(ShapeLaws.elements([images,channels,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements([Tiled.groups(channels),images,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool},
  +small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool},
  +at: Nat,+inside: {Nat.is_lt(at,ShapeLaws.elements([height,width])) == True{} : Bool}) ->
  {SlotsLanes.lane_point(~Tile.Tile8,~centered_point,~Representation.Lifted<X>,~Representation.lifted(~X,~xv),
      SlotsLanes.LanePoint{digit,mean_cell(~Means,~mean,Representation.Lifted{means,Public.geometry_of([1,channels,1,1],1n,Grid.Outermost{})},group),
        Representation.Lifted{x,Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})}},
      U32.from_nat(Nat.add(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32))),Nat.mul(image,ShapeLaws.elements([height,width]))),at))) ==
    CombineLeaf.value(~F32,~X,~ChannelMeans<Means>,~centered_pair,~xv,~channel_mean(~Means,~mean),CombineLeaf.Context{x,ChannelMeans{means,Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})}},
      U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),ShapeLaws.elements([height,width])),at))) : F32}:
  +shape = {[images,channels,height,width] : List<&2,U32>}
  +geometry = Public.geometry_of(shape,1n,Grid.Outermost{})
  +lanes = Public.geometry_of([1,channels,1,1],1n,Grid.Outermost{})
  +capacity = U32.to_nat(Machine.capacity(limits))
  +plane = ShapeLaws.elements([height,width])
  +count = U32.to_nat(channels)
  +point = Nat.add(Nat.mul(image,plane),at)
  +channel = Nat.add(digit,Nat.mul(group,8n))
  +row = Nat.add(Nat.mul(image,count),channel)
  +located = Nat.add(Nat.mul(row,plane),at)
  +word = U32.from_nat(Nat.add(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(shape,1n,1) * Tiled.product(Public.lane_after(shape,1n)) : U32))),Nat.mul(image,plane)),at))
  +cell = Representation.lifted(~X,~xv,Representation.Lifted{x,geometry},word)
  +means_cell = mean_cell(~Means,~mean,Representation.Lifted{means,lanes},group)
  +value = xv(x,U32.from_nat(located))
  +average = mean(means,U32.from_nat(channel))
  +centered = channel_mean(~Means,~mean,ChannelMeans{means,geometry},U32.from_nat(located))
  +images_positive = Order.positive_bound(image,U32.to_nat(images),image_inside)
  +channels_positive = Order.positive_bound(channel,count,valid)
  +rows_inside = RowMajor.radix(image,U32.to_nat(images),channel,count,image_inside,valid)
  +located_inside = {%RowMajor.assoc(U32.to_nat(images),count,plane) : {Nat.is_lt(located,_) == True{} : Bool}
    RowMajor.radix(row,Nat.mul(U32.to_nat(images),count),at,plane,rows_inside,inside) : {Nat.is_lt(located,ShapeLaws.elements(shape)) == True{} : Bool}}
  +located_room = Order.transitive(located,ShapeLaws.elements(shape),capacity,Bounds.strict_le(located,ShapeLaws.elements(shape),located_inside),room)
  +channel_plane = {%RowMajor.multiply_commutative(plane,count) : {Nat.is_le(plane,_) == True{} : Bool} RowMajor.product_covers(plane,count,channels_positive) : {Nat.is_le(plane,Nat.mul(count,plane)) == True{} : Bool}}
  +plane_values = {%RowMajor.multiply_commutative(Nat.mul(count,plane),U32.to_nat(images)) : {Nat.is_le(Nat.mul(count,plane),_) == True{} : Bool}
    RowMajor.product_covers(Nat.mul(count,plane),U32.to_nat(images),images_positive) : {Nat.is_le(Nat.mul(count,plane),ShapeLaws.elements(shape)) == True{} : Bool}}
  +plane_room = Order.transitive(plane,ShapeLaws.elements(shape),capacity,Order.transitive(plane,Nat.mul(count,plane),ShapeLaws.elements(shape),channel_plane,plane_values),room)
  +located_index = plane_index(limits,images,channels,height,width,image,channel,at,plane_room,inside)
  +channel_of = Equal.trans(Nat,Representation.value_channel(geometry,U32.to_nat(U32.from_nat(located))),Representation.value_channel(geometry,located),channel,
    Equal.cong(Nat,Nat,number => Representation.value_channel(geometry,number),U32.to_nat(U32.from_nat(located)),located,FullRange.roundtrip(located,Machine.capacity(limits),located_room)),
    {%located_index : {Representation.value_channel(geometry,_) == channel : Nat} Representation.lane_channel(geometry,point,channel,valid) : {Representation.value_channel(geometry,located) == channel : Nat}})
  +mean_lane = {%Equal.sym(Bool,Nat.is_lt(channel,count),True{},valid) :
      {Bool.pick(F32,_,mean(means,U32.from_nat(Representation.logical_index(lanes,0n,channel))),0.0) == average : F32}
    Equal.cong(Nat,F32,number => mean(means,U32.from_nat(number)),Representation.logical_index(lanes,0n,channel),channel,mean_index(channels,channel)) :
    {Representation.lane(~Means,~mean,means,lanes,0n,group,digit) == average : F32}}
  +mean_eq = Equal.trans(F32,TileLaws.select_lane(digit,means_cell),Representation.lane(~Means,~mean,means,lanes,0n,group,digit),centered,
    Representation.cell_lane(~Means,~mean,means,lanes,0n,group,digit,small),
    Equal.trans(F32,Representation.lane(~Means,~mean,means,lanes,0n,group,digit),average,centered,mean_lane,
      Equal.cong(Nat,F32,number => mean(means,U32.from_nat(number)),channel,Representation.value_channel(geometry,U32.to_nat(U32.from_nat(located))),
        Equal.sym(Nat,Representation.value_channel(geometry,U32.to_nat(U32.from_nat(located))),channel,channel_of))))
  +value_eq = plane_value(~X,~xv,x,limits,images,channels,height,width,image,group,digit,cells_room,image_inside,group_inside,small,valid,at,inside)
  Equal.trans(F32,TileLaws.select_lane(digit,centered_point(means_cell,cell)),centered_pair(TileLaws.select_lane(digit,cell),TileLaws.select_lane(digit,means_cell)),centered_pair(value,centered),
    TileLaws.square_lane(cell,means_cell,digit),
    Equal.trans(F32,centered_pair(TileLaws.select_lane(digit,cell),TileLaws.select_lane(digit,means_cell)),centered_pair(value,TileLaws.select_lane(digit,means_cell)),centered_pair(value,centered),
      Equal.cong(F32,F32,low => centered_pair(low,TileLaws.select_lane(digit,means_cell)),TileLaws.select_lane(digit,cell),value,value_eq),
      Equal.cong(F32,F32,high => centered_pair(value,high),TileLaws.select_lane(digit,means_cell),centered,mean_eq)))

# A lane of a plane of the centered squares of the cells agrees with the plane
# of the centered squares of the channel of the lane.
def centered_agrees(~X: Data,~xv: X -> U32 -> F32,~Means: Data,~mean: Means -> U32 -> F32,+logical: CombineLeaf.Context<X,ChannelMeans<Means>>,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,
  +image: Nat,+group: Nat,+digit: Nat,
  +room: {Nat.is_le(ShapeLaws.elements([images,channels,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements([Tiled.groups(channels),images,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +ready: centered_ready(~X,~Means,logical,images,channels,height,width),
  +image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool},
  +small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool}) ->
  ReduceLaws.Agrees(~SlotsLanes.LanePoint<Tile.Tile8,Representation.Lifted<X>>,~SlotsLanes.lane_point(~Tile.Tile8,~centered_point,~Representation.Lifted<X>,~Representation.lifted(~X,~xv)),
    ~CombineLeaf.Context<X,ChannelMeans<Means>>,~CombineLeaf.value(~F32,~X,~ChannelMeans<Means>,~centered_pair,~xv,~channel_mean(~Means,~mean)),
    SlotsLanes.LanePoint{digit,centered_extra(~X,~Means,~mean,logical,channels,group),Representation.Lifted{centered_input(~X,~Means,logical),Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})}},logical,ShapeLaws.elements([height,width]),
    Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32))),Nat.mul(image,ShapeLaws.elements([height,width]))),
    Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),ShapeLaws.elements([height,width]))):
  CombineLeaf.Context{+x,+right} = logical
  ChannelMeans{+means,+geometry} = right
  +plane = ShapeLaws.elements([height,width])
  +shape = {[images,channels,height,width] : List<&2,U32>}
  +left = Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(shape,1n,1) * Tiled.product(Public.lane_after(shape,1n)) : U32))),Nat.mul(image,plane))
  +rows = Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),plane)
  +at = {SlotsLanes.LanePoint{digit,mean_cell(~Means,~mean,Representation.Lifted{means,Public.geometry_of([1,channels,1,1],1n,Grid.Outermost{})},group),
      Representation.Lifted{x,Public.geometry_of(shape,1n,Grid.Outermost{})}} : SlotsLanes.LanePoint<Tile.Tile8,Representation.Lifted<X>>}
  %Equal.sym(Cells.Cells,geometry,Public.geometry_of(shape,1n,Grid.Outermost{}),ready) :
    ReduceLaws.Agrees(~SlotsLanes.LanePoint<Tile.Tile8,Representation.Lifted<X>>,~SlotsLanes.lane_point(~Tile.Tile8,~centered_point,~Representation.Lifted<X>,~Representation.lifted(~X,~xv)),
      ~CombineLeaf.Context<X,ChannelMeans<Means>>,~CombineLeaf.value(~F32,~X,~ChannelMeans<Means>,~centered_pair,~xv,~channel_mean(~Means,~mean)),at,CombineLeaf.Context{x,ChannelMeans{means,_}},plane,left,rows)
  %Addition.zero_right(left) :
    ReduceLaws.Agrees(~SlotsLanes.LanePoint<Tile.Tile8,Representation.Lifted<X>>,~SlotsLanes.lane_point(~Tile.Tile8,~centered_point,~Representation.Lifted<X>,~Representation.lifted(~X,~xv)),
      ~CombineLeaf.Context<X,ChannelMeans<Means>>,~CombineLeaf.value(~F32,~X,~ChannelMeans<Means>,~centered_pair,~xv,~channel_mean(~Means,~mean)),at,CombineLeaf.Context{x,ChannelMeans{means,Public.geometry_of(shape,1n,Grid.Outermost{})}},plane,_,rows)
  %Addition.zero_right(rows) :
    ReduceLaws.Agrees(~SlotsLanes.LanePoint<Tile.Tile8,Representation.Lifted<X>>,~SlotsLanes.lane_point(~Tile.Tile8,~centered_point,~Representation.Lifted<X>,~Representation.lifted(~X,~xv)),
      ~CombineLeaf.Context<X,ChannelMeans<Means>>,~CombineLeaf.value(~F32,~X,~ChannelMeans<Means>,~centered_pair,~xv,~channel_mean(~Means,~mean)),at,CombineLeaf.Context{x,ChannelMeans{means,Public.geometry_of(shape,1n,Grid.Outermost{})}},plane,Nat.add(left,0n),_)
  ReduceLaws.agrees_all(~SlotsLanes.LanePoint<Tile.Tile8,Representation.Lifted<X>>,~SlotsLanes.lane_point(~Tile.Tile8,~centered_point,~Representation.Lifted<X>,~Representation.lifted(~X,~xv)),
    ~CombineLeaf.Context<X,ChannelMeans<Means>>,~CombineLeaf.value(~F32,~X,~ChannelMeans<Means>,~centered_pair,~xv,~channel_mean(~Means,~mean)),
    ~at,~CombineLeaf.Context{x,ChannelMeans{means,Public.geometry_of(shape,1n,Grid.Outermost{})}},~plane,~left,~rows,
    ~(offset => below => centered_value(~X,~xv,~Means,~mean,x,means,limits,images,channels,height,width,image,group,digit,room,cells_room,image_inside,group_inside,small,valid,offset,below)),
    plane,0n,{==})

# The centered squares of the statistics of a batch normalization of tiled
# values [N,C,H,W] on the means of the channels are the sums of the planes of
# the centered squares of every channel.
def centered_sums_summed(~Values: Data,~value: Values -> U32 -> F32,~Means: Data,~mean: Means -> U32 -> F32,+held: Values,+means: Means,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+workers: U32,
  +model: PiecesModel.Model<Tile.Tile8>,+fill: Tile.Tile8,+cell_workers: U32,
  +contract: Public.Contract(~Values,~value,held,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),[images,height,width,Tiled.groups(channels)],fill,Tensors.Usable{},cell_workers},1n,channels,Grid.Innermost{}}),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(4n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements([images,channels,height,width])) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements([images,height,width,Tiled.groups(channels)])),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  KernelGemm.Both<PlaneSummed(~CombineLeaf.Context<Values,ChannelMeans<Means>>,~CombineLeaf.value(~F32,~Values,~ChannelMeans<Means>,~centered_pair,~value,~channel_mean(~Means,~mean)),
      CombineLeaf.Context{held,ChannelMeans{means,Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})}},images,channels,height,width,
      BN.channel_sums(BN.Layer{images,channels,height,width,workers},Pair.snd(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.centered_sums(
        ListReadback.enumerate(~Tile.Tile8,~Representation.Lifted<Means>,~mean_cell(~Means,~mean),U32.to_nat(Tiled.groups(channels)),0n,Representation.Lifted{means,Public.geometry_of([1,channels,1,1],1n,Grid.Outermost{})}),
        [images,height,width,Tiled.groups(channels)],workers,PiecesModel.pack(~Tile.Tile8,model))))),
    {Pair.fst(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.centered_sums(
        ListReadback.enumerate(~Tile.Tile8,~Representation.Lifted<Means>,~mean_cell(~Means,~mean),U32.to_nat(Tiled.groups(channels)),0n,Representation.Lifted{means,Public.geometry_of([1,channels,1,1],1n,Grid.Outermost{})}),
        [images,height,width,Tiled.groups(channels)],workers,PiecesModel.pack(~Tile.Tile8,model))) == PiecesModel.pack(~Tile.Tile8,model) : Pieces.Pieces(~Tile.Tile8)}>:
  KernelGemm.Both{+global,+laid} = contract
  Public.Laid{+inside,+shaped_cells,+restored,+counted,+pinned} = laid
  +values = {[images,channels,height,width] : List<&2,U32>}
  +shape = {[images,height,width,Tiled.groups(channels)] : List<&2,U32>}
  +capacity = Machine.capacity(limits)
  +groups = Tiled.groups(channels)
  +count = U32.to_nat(groups)
  +inner = Grid.product([Tiled.groups(channels)])
  +lifted = {Representation.Lifted{means,Public.geometry_of([1,channels,1,1],1n,Grid.Outermost{})} : Representation.Lifted<Means>}
  +cells = ListReadback.enumerate(~Tile.Tile8,~Representation.Lifted<Means>,~mean_cell(~Means,~mean),count,0n,lifted)
  +logical = {CombineLeaf.Context{held,ChannelMeans{means,Public.geometry_of(values,1n,Grid.Outermost{})}} : CombineLeaf.Context<Values,ChannelMeans<Means>>}
  +values_room = StorageCount.counted_room(limits,values,counted,default_capacity)
  +rooms_out = ReduceLaws.kept_rooms(limits,values,[0n,2n,3n],1n,Grid.Outermost{},{==},{==},values_room,nonempty)
  +rooms_in = ReduceLaws.kept_rooms(limits,values,[0n,2n,3n],1n,Grid.Innermost{},{==},{==},values_room,nonempty)
  +room_in = Order.transitive(ShapeLaws.elements(shape),ShapeLaws.elements(values),U32.to_nat(capacity),ReduceLaws.rooms_cells_fewer(limits,values,[0n,2n,3n],1n,Grid.Innermost{},rooms_in),values_room)
  +positive_in = ReduceLaws.rooms_cells_positive(limits,values,[0n,2n,3n],1n,Grid.Innermost{},rooms_in)
  +within_box = PartialsValues.within_removed([images,width,groups],1n,capacity,PartialsValues.within_removed(shape,1n,capacity,
    PartialsValues.within_of(shape,capacity,CountCompletion.positive_not_zero(shape,positive_in),room_in)))
  +groups_positive = within_positive(groups,[],capacity,PartialsValues.within_removed([images,groups],0n,capacity,within_box))
  +inner_count = Equal.cong(U32,Nat,word => U32.to_nat(word),inner,groups,Arithmetic.multiply_one(groups))
  +length = ListReadback.enumerate_length(~Tile.Tile8,~Representation.Lifted<Means>,~mean_cell(~Means,~mean),count,0n,lifted)
  +turns = {Reduce.Turns{1n,{%Equal.sym(Nat,List.length(&2,Tile.Tile8,cells),count,length) : {Nat.is_lt(0n,_) == True{} : Bool} groups_positive : {Nat.is_lt(0n,List.length(&2,Tile.Tile8,cells)) == True{} : Bool}},
      Equal.trans(Nat,U32.to_nat(inner),count,Nat.add(List.length(&2,Tile.Tile8,cells),0n),inner_count,
        Equal.trans(Nat,count,List.length(&2,Tile.Tile8,cells),Nat.add(List.length(&2,Tile.Tile8,cells),0n),Equal.sym(Nat,List.length(&2,Tile.Tile8,cells),count,length),
          Equal.sym(Nat,Nat.add(List.length(&2,Tile.Tile8,cells),0n),List.length(&2,Tile.Tile8,cells),Addition.zero_right(List.length(&2,Tile.Tile8,cells)))))} :
    Reduce.Turns<Tile.Tile8,cells,inner>}
  +summed = plane_contract(~BNCells.Cells(),~Tile.Tile8,~centered_point,~Operations.cells_slots(~Tile.Tile8,~BNKernel.read_centered,~KernelCertificate.read_centered),~Pieces.vacant_buffer(~Tile.Tile8),
    ~Values,~Representation.Lifted<Values>,~Representation.lifted(~Values,~value),~lifted_at(~Values),~cell_moved(~Values,~value),
    ~PiecesModel.Model<Tile.Tile8>,~PiecesModel.pack(~Tile.Tile8),~CombineLeaf.Context<Values,ChannelMeans<Means>>,
    ~CombineLeaf.value(~F32,~Values,~ChannelMeans<Means>,~centered_pair,~value,~channel_mean(~Means,~mean)),~centered_input(~Values,~Means),~centered_extras(~Values,~Means,~mean),~centered_extra(~Values,~Means,~mean),
    ~centered_ready(~Values,~Means),~centered_started(~Values,~Means,~mean),~centered_enough(~Values,~Means,~mean),~centered_agrees(~Values,~value,~Means,~mean),
    logical,limits,images,channels,height,width,9.0,2.0,workers,model,default_capacity,rank_room,rooms_out,{==},
    plane_shaped(~BNCells.Cells(),~Buffer.Model<Tile.Tile8>,~Buffer.pack(~Tile.Tile8),~Reduce.cell(~Tile.Tile8),~Tile.Tile8,~centered_point,~Operations.cells_slots(~Tile.Tile8,~BNKernel.read_centered,~KernelCertificate.read_centered),
      ~Pieces.vacant_buffer(~Tile.Tile8),~Reduce.vacant_model(~Tile.Tile8),~Reduce.vacant_same(~Tile.Tile8),
      ~SlotsValues.slots_chain(~Tile.Tile8,~BNKernel.read_centered,~KernelCertificate.read_centered,~centered_point,~centered_matches),
      ~Values,~Representation.Lifted<Values>,~Representation.lifted(~Values,~value),~lifted_at(~Values),~PiecesModel.Model<Tile.Tile8>,~PiecesModel.pack(~Tile.Tile8),
      held,cells,limits,images,channels,height,width,9.0,2.0,workers,model,
      PartialsValues.contract_streamed(~Tile.Tile8,~Representation.Lifted<Values>,~Representation.lifted(~Values,~value),
        Representation.Lifted{held,Public.geometry_of(values,1n,Grid.Innermost{})},model,shape,fill,cell_workers,global),
      turns,room_in,positive_in,default_capacity,cap,cap_ok,count_room))
  %Equal.sym(List<&2,Tile.Tile8>,Pair.snd(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.sums_of(Partials.read(~BNCells.Cells(),~Tile.Tile8,~Tile.Tile8,~Operations.TileSlots<Tile.Tile8>,~Sum.TileParts,~Operations.slots_begun(~Tile.Tile8),~Operations.cells_slots(~Tile.Tile8,~BNKernel.read_centered,~KernelCertificate.read_centered),~Operations.slots_closed(~Tile.Tile8),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),cells,9.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))),Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,Pair.snd(Pieces.Pieces(~Tile.Tile8),List<&2,Sum.TileParts>,Partials.read(~BNCells.Cells(),~Tile.Tile8,~Tile.Tile8,~Operations.TileSlots<Tile.Tile8>,~Sum.TileParts,~Operations.slots_begun(~Tile.Tile8),~Operations.cells_slots(~Tile.Tile8,~BNKernel.read_centered,~KernelCertificate.read_centered),~Operations.slots_closed(~Tile.Tile8),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),cells,9.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))),sums_parts(Partials.read(~BNCells.Cells(),~Tile.Tile8,~Tile.Tile8,~Operations.TileSlots<Tile.Tile8>,~Sum.TileParts,~Operations.slots_begun(~Tile.Tile8),~Operations.cells_slots(~Tile.Tile8,~BNKernel.read_centered,~KernelCertificate.read_centered),~Operations.slots_closed(~Tile.Tile8),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),cells,9.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))) :
    KernelGemm.Both<PlaneSummed(~CombineLeaf.Context<Values,ChannelMeans<Means>>,~CombineLeaf.value(~F32,~Values,~ChannelMeans<Means>,~centered_pair,~value,~channel_mean(~Means,~mean)),logical,images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},_)),
      {Pair.fst(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.sums_of(Partials.read(~BNCells.Cells(),~Tile.Tile8,~Tile.Tile8,~Operations.TileSlots<Tile.Tile8>,~Sum.TileParts,~Operations.slots_begun(~Tile.Tile8),~Operations.cells_slots(~Tile.Tile8,~BNKernel.read_centered,~KernelCertificate.read_centered),~Operations.slots_closed(~Tile.Tile8),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),cells,9.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))) == PiecesModel.pack(~Tile.Tile8,model) : Pieces.Pieces(~Tile.Tile8)}>
  %Equal.sym(Pieces.Pieces(~Tile.Tile8),Pair.fst(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,BNCells.sums_of(Partials.read(~BNCells.Cells(),~Tile.Tile8,~Tile.Tile8,~Operations.TileSlots<Tile.Tile8>,~Sum.TileParts,~Operations.slots_begun(~Tile.Tile8),~Operations.cells_slots(~Tile.Tile8,~BNKernel.read_centered,~KernelCertificate.read_centered),~Operations.slots_closed(~Tile.Tile8),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),cells,9.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))),Pair.fst(Pieces.Pieces(~Tile.Tile8),List<&2,Sum.TileParts>,Partials.read(~BNCells.Cells(),~Tile.Tile8,~Tile.Tile8,~Operations.TileSlots<Tile.Tile8>,~Sum.TileParts,~Operations.slots_begun(~Tile.Tile8),~Operations.cells_slots(~Tile.Tile8,~BNKernel.read_centered,~KernelCertificate.read_centered),~Operations.slots_closed(~Tile.Tile8),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),cells,9.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model))),sums_pieces(Partials.read(~BNCells.Cells(),~Tile.Tile8,~Tile.Tile8,~Operations.TileSlots<Tile.Tile8>,~Sum.TileParts,~Operations.slots_begun(~Tile.Tile8),~Operations.cells_slots(~Tile.Tile8,~BNKernel.read_centered,~KernelCertificate.read_centered),~Operations.slots_closed(~Tile.Tile8),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),cells,9.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))) :
    KernelGemm.Both<PlaneSummed(~CombineLeaf.Context<Values,ChannelMeans<Means>>,~CombineLeaf.value(~F32,~Values,~ChannelMeans<Means>,~centered_pair,~value,~channel_mean(~Means,~mean)),logical,images,channels,height,width,
        BN.channel_sums(BN.Layer{images,channels,height,width,workers},Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,Pair.snd(Pieces.Pieces(~Tile.Tile8),List<&2,Sum.TileParts>,Partials.read(~BNCells.Cells(),~Tile.Tile8,~Tile.Tile8,~Operations.TileSlots<Tile.Tile8>,~Sum.TileParts,~Operations.slots_begun(~Tile.Tile8),~Operations.cells_slots(~Tile.Tile8,~BNKernel.read_centered,~KernelCertificate.read_centered),~Operations.slots_closed(~Tile.Tile8),~Sum.tile_merge,~Pieces.vacant_buffer(~Tile.Tile8),cells,9.0,2.0,shape,BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,PiecesModel.pack(~Tile.Tile8,model)))))),
      {_ == PiecesModel.pack(~Tile.Tile8,model) : Pieces.Pieces(~Tile.Tile8)}>
  summed

# ---- The products of the statistics ----

# The product of two values.
def product_pair(left: F32,right: F32) -> F32:
  (left * right : F32)

# The cells of two readers at a geometry of their cells.
def pair_at(~DL: Data,~XL: Data,logical: CombineLeaf.Context<DL,XL>,geometry: Cells.Cells) -> CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>:
  CombineLeaf.Context{+left,+right} = logical
  +at = geometry
  CombineLeaf.Context{Representation.Lifted{left,at},Representation.Lifted{right,at}}

# The products of the cells of two readers with the groups innermost read as
# at the index of the cell with the groups outermost.
def pair_moved(~DL: Data,~dv: DL -> U32 -> F32,~XL: Data,~xv: XL -> U32 -> F32,+x: CombineLeaf.Context<DL,XL>,+values: List<&2,U32>,+axis: Nat,+outer: Nat,+group: Nat,+cap: U32,
  +inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+outer_inside: {Nat.is_lt(outer,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))) == True{} : Bool},
  +group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},
  +inner_room: {Nat.is_le(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group),U32.to_nat(cap)) == True{} : Bool},+outer_room: {Nat.is_le(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer),U32.to_nat(cap)) == True{} : Bool}) ->
  {CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.Lifted<XL>,~Tile.times,~Representation.lifted(~DL,~dv),~Representation.lifted(~XL,~xv),pair_at(~DL,~XL,x,Public.geometry_of(values,axis,Grid.Innermost{})),U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))) == CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.Lifted<XL>,~Tile.times,~Representation.lifted(~DL,~dv),~Representation.lifted(~XL,~xv),pair_at(~DL,~XL,x,Public.geometry_of(values,axis,Grid.Outermost{})),U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))) : Tile.Tile8}:
  CombineLeaf.Context{+left,+right} = x
  +inner_word = U32.from_nat(Nat.add(Nat.mul(outer,U32.to_nat(Tiled.groups(Grid.lanes_at(values,axis)))),group))
  +outer_word = U32.from_nat(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix(values,axis,1) * Tiled.product(Public.lane_after(values,axis)) : U32))),outer))
  +left_in = Representation.lifted(~DL,~dv,Representation.Lifted{left,Public.geometry_of(values,axis,Grid.Innermost{})},inner_word)
  +left_out = Representation.lifted(~DL,~dv,Representation.Lifted{left,Public.geometry_of(values,axis,Grid.Outermost{})},outer_word)
  +right_in = Representation.lifted(~XL,~xv,Representation.Lifted{right,Public.geometry_of(values,axis,Grid.Innermost{})},inner_word)
  +right_out = Representation.lifted(~XL,~xv,Representation.Lifted{right,Public.geometry_of(values,axis,Grid.Outermost{})},outer_word)
  Equal.trans(Tile.Tile8,Tile.times(left_in,right_in),Tile.times(left_out,right_in),Tile.times(left_out,right_out),
    Equal.cong(Tile.Tile8,Tile.Tile8,cell => Tile.times(cell,right_in),left_in,left_out,
      cell_moved(~DL,~dv,left,values,axis,outer,group,cap,inside,outer_inside,group_inside,inner_room,outer_room)),
    Equal.cong(Tile.Tile8,Tile.Tile8,cell => Tile.times(left_out,cell),right_in,right_out,
      cell_moved(~XL,~xv,right,values,axis,outer,group,cap,inside,outer_inside,group_inside,inner_room,outer_room)))

# A lane of the product of the cells of two readers with the groups outermost
# is the product of their values at the channel of the lane.
def product_value(~DL: Data,~dv: DL -> U32 -> F32,~XL: Data,~xv: XL -> U32 -> F32,+dl: DL,+xl: XL,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+image: Nat,+group: Nat,+digit: Nat,
  +cells_room: {Nat.is_le(ShapeLaws.elements([Tiled.groups(channels),images,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool},
  +small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool},
  +at: Nat,+inside: {Nat.is_lt(at,ShapeLaws.elements([height,width])) == True{} : Bool}) ->
  {SlotsLanes.lane_point(~Unit,~SlotsValues.held(~Unit),~CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>,~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.Lifted<XL>,~Tile.times,~Representation.lifted(~DL,~dv),~Representation.lifted(~XL,~xv)),SlotsLanes.LanePoint{digit,Unit{},pair_at(~DL,~XL,CombineLeaf.Context{dl,xl},Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{}))},U32.from_nat(Nat.add(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32))),Nat.mul(image,ShapeLaws.elements([height,width]))),at))) ==
    CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv,CombineLeaf.Context{dl,xl},U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),ShapeLaws.elements([height,width])),at))) : F32}:
  +word = U32.from_nat(Nat.add(Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32))),Nat.mul(image,ShapeLaws.elements([height,width]))),at))
  +index = U32.from_nat(Nat.add(Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),ShapeLaws.elements([height,width])),at))
  +left_cell = Representation.lifted(~DL,~dv,Representation.Lifted{dl,Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})},word)
  +right_cell = Representation.lifted(~XL,~xv,Representation.Lifted{xl,Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})},word)
  +left_lane = TileLaws.select_lane(digit,left_cell)
  +right_lane = TileLaws.select_lane(digit,right_cell)
  Equal.trans(F32,TileLaws.select_lane(digit,Tile.times(left_cell,right_cell)),(left_lane * right_lane : F32),(dv(dl,index) * xv(xl,index) : F32),
    TileLaws.times_lane(left_cell,right_cell,digit),
    Equal.trans(F32,(left_lane * right_lane : F32),(dv(dl,index) * right_lane : F32),(dv(dl,index) * xv(xl,index) : F32),
      Equal.cong(F32,F32,value => (value * right_lane : F32),left_lane,dv(dl,index),
        plane_value(~DL,~dv,dl,limits,images,channels,height,width,image,group,digit,cells_room,image_inside,group_inside,small,valid,at,inside)),
      Equal.cong(F32,F32,value => (dv(dl,index) * value : F32),right_lane,xv(xl,index),
        plane_value(~XL,~xv,xl,limits,images,channels,height,width,image,group,digit,cells_room,image_inside,group_inside,small,valid,at,inside))))

# A lane of the products of the plane of an image and a group agrees with the
# products of the plane of the image and the channel of the lane.
def product_agrees(~DL: Data,~dv: DL -> U32 -> F32,~XL: Data,~xv: XL -> U32 -> F32,+logical: CombineLeaf.Context<DL,XL>,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+image: Nat,+group: Nat,+digit: Nat,
  +room: {Nat.is_le(ShapeLaws.elements([images,channels,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_room: {Nat.is_le(ShapeLaws.elements([Tiled.groups(channels),images,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +ready: sums_ready(~CombineLeaf.Context<DL,XL>,logical,images,channels,height,width),
  +image_inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(channels))) == True{} : Bool},
  +small: {Nat.is_le(digit,7n) == True{} : Bool},+valid: {Nat.is_lt(Nat.add(digit,Nat.mul(group,8n)),U32.to_nat(channels)) == True{} : Bool}) ->
  ReduceLaws.Agrees(~SlotsLanes.LanePoint<Unit,CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>>,~SlotsLanes.lane_point(~Unit,~SlotsValues.held(~Unit),~CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>,~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.Lifted<XL>,~Tile.times,~Representation.lifted(~DL,~dv),~Representation.lifted(~XL,~xv))),~CombineLeaf.Context<DL,XL>,~CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv),
    SlotsLanes.LanePoint{digit,plane_unit(~CombineLeaf.Context<DL,XL>,logical,channels,group),pair_at(~DL,~XL,plane_input(~CombineLeaf.Context<DL,XL>,logical),Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{}))},logical,ShapeLaws.elements([height,width]),Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32))),Nat.mul(image,ShapeLaws.elements([height,width]))),Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),ShapeLaws.elements([height,width]))):
  CombineLeaf.Context{+dl,+xl} = logical
  +plane = ShapeLaws.elements([height,width])
  +left = Nat.add(Nat.mul(group,U32.to_nat((Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32))),Nat.mul(image,ShapeLaws.elements([height,width])))
  +right = Nat.mul(Nat.add(Nat.mul(image,U32.to_nat(channels)),Nat.add(digit,Nat.mul(group,8n))),ShapeLaws.elements([height,width]))
  +at = {SlotsLanes.LanePoint{digit,Unit{},pair_at(~DL,~XL,CombineLeaf.Context{dl,xl},Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{}))} : SlotsLanes.LanePoint<Unit,CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>>}
  %Addition.zero_right(left) :
    ReduceLaws.Agrees(~SlotsLanes.LanePoint<Unit,CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>>,~SlotsLanes.lane_point(~Unit,~SlotsValues.held(~Unit),~CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>,~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.Lifted<XL>,~Tile.times,~Representation.lifted(~DL,~dv),~Representation.lifted(~XL,~xv))),~CombineLeaf.Context<DL,XL>,~CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv),at,CombineLeaf.Context{dl,xl},plane,_,right)
  %Addition.zero_right(right) :
    ReduceLaws.Agrees(~SlotsLanes.LanePoint<Unit,CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>>,~SlotsLanes.lane_point(~Unit,~SlotsValues.held(~Unit),~CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>,~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.Lifted<XL>,~Tile.times,~Representation.lifted(~DL,~dv),~Representation.lifted(~XL,~xv))),~CombineLeaf.Context<DL,XL>,~CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv),at,CombineLeaf.Context{dl,xl},plane,Nat.add(left,0n),_)
  ReduceLaws.agrees_all(~SlotsLanes.LanePoint<Unit,CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>>,~SlotsLanes.lane_point(~Unit,~SlotsValues.held(~Unit),~CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>,~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.Lifted<XL>,~Tile.times,~Representation.lifted(~DL,~dv),~Representation.lifted(~XL,~xv))),~CombineLeaf.Context<DL,XL>,~CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv),
    ~at,~CombineLeaf.Context{dl,xl},~plane,~left,~right,
    ~(offset => below => product_value(~DL,~dv,~XL,~xv,dl,xl,limits,images,channels,height,width,image,group,digit,cells_room,image_inside,group_inside,small,valid,offset,below)),
    plane,0n,{==})

# The pieces of two tensors read together.
def paired_models(models: CombineLeaf.Context<PiecesModel.Model<Tile.Tile8>,PiecesModel.Model<Tile.Tile8>>) -> BNCells.Pairs():
  CombineLeaf.Context{left,right} = models
  Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,left),PiecesModel.pack(~Tile.Tile8,right))

# The zipped pieces of two tensors of one layout stream the products of their
# cells.
def contracts_streamed(~DL: Data,~dv: DL -> U32 -> F32,~XL: Data,~xv: XL -> U32 -> F32,+dl: DL,+xl: XL,+images: U32,+channels: U32,+height: U32,+width: U32,
  +dmodel: PiecesModel.Model<Tile.Tile8>,+xmodel: PiecesModel.Model<Tile.Tile8>,+dfill: Tile.Tile8,+xfill: Tile.Tile8,+dworkers: U32,+xworkers: U32,
  +dglobal: Global.Contract(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~dv),Representation.Lifted{dl,Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{})},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dmodel),[images,height,width,Tiled.groups(channels)],dfill,Tensors.Usable{},dworkers}),
  +xglobal: Global.Contract(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~xv),Representation.Lifted{xl,Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{})},Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xmodel),[images,height,width,Tiled.groups(channels)],xfill,Tensors.Usable{},xworkers})) ->
  Reduce.Streamed<BNCells.CellPair(),Each.PiecePair<Tile.Tile8>,Tile.Tile8,Each.pair_pack(~Tile.Tile8),TwinsLaws.product_cell,paired_models(CombineLeaf.Context{dmodel,xmodel}),Pieces.ranges_of_grid(Tensors.grid_of([images,height,width,Tiled.groups(channels)])),
    Reduce.values_from(~Tile.Tile8,~CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>,~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.Lifted<XL>,~Tile.times,~Representation.lifted(~DL,~dv),~Representation.lifted(~XL,~xv)),pair_at(~DL,~XL,CombineLeaf.Context{dl,xl},Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{})),ShapeLaws.elements([images,height,width,Tiled.groups(channels)]),0n)>:
  KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+dfields,+dready,+dcoverage},KernelGemm.Both{+dcounted,+dcanonical}},+dvalues} = dglobal
  KernelGemm.Both{KernelGemm.Both{Content.WellFormed{+xfields,+xready,+xcoverage},KernelGemm.Both{+xcounted,+xcanonical}},+xvalues} = xglobal
  +dpacked = PiecesModel.packed_contents(~Tile.Tile8,dmodel)
  +xpacked = PiecesModel.packed_contents(~Tile.Tile8,xmodel)
  TwinsLaws.product_streamed(~Representation.Lifted<DL>,~Representation.lifted(~DL,~dv),~Representation.Lifted<XL>,~Representation.lifted(~XL,~xv),
    Representation.Lifted{dl,Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{})},Representation.Lifted{xl,Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{})},dmodel,xmodel,Pieces.ranges_of_grid(Tensors.grid_of([images,height,width,Tiled.groups(channels)])),ShapeLaws.elements([images,height,width,Tiled.groups(channels)]),
    {%dpacked : {PiecesModel.content_layout(~Tile.Tile8,_) == Pieces.ranges_of_grid(Tensors.grid_of([images,height,width,Tiled.groups(channels)])) : Pieces.Ranges} dcanonical : {PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,dmodel)) == Pieces.ranges_of_grid(Tensors.grid_of([images,height,width,Tiled.groups(channels)])) : Pieces.Ranges}},
    {%xpacked : {PiecesModel.content_layout(~Tile.Tile8,_) == Pieces.ranges_of_grid(Tensors.grid_of([images,height,width,Tiled.groups(channels)])) : Pieces.Ranges} xcanonical : {PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,xmodel)) == Pieces.ranges_of_grid(Tensors.grid_of([images,height,width,Tiled.groups(channels)])) : Pieces.Ranges}},
    {%dpacked : PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,_),0n,ShapeLaws.elements([images,height,width,Tiled.groups(channels)])) dcoverage : PiecesPlan.Coverage(PiecesModel.content_layout(~Tile.Tile8,PiecesModel.content(~Tile.Tile8,dmodel)),0n,ShapeLaws.elements([images,height,width,Tiled.groups(channels)]))},
    {%dpacked : Content.Values(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~dv),Representation.Lifted{dl,Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{})},_) dvalues :
      Content.Values(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.lifted(~DL,~dv),Representation.Lifted{dl,Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{})},PiecesModel.content(~Tile.Tile8,dmodel))},
    {%xpacked : Content.Values(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~xv),Representation.Lifted{xl,Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{})},_) xvalues :
      Content.Values(~Tile.Tile8,~Representation.Lifted<XL>,~Representation.lifted(~XL,~xv),Representation.Lifted{xl,Public.geometry_of([images,channels,height,width],1n,Grid.Innermost{})},PiecesModel.content(~Tile.Tile8,xmodel))})

# The products the statistics read from the planes of two tensors, as kept
# cells, hold the sums of the products of the planes of every channel.
def product_sums_summed(~DL: Data,~dv: DL -> U32 -> F32,~XL: Data,~xv: XL -> U32 -> F32,+dl: DL,+xl: XL,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+workers: U32,
  +dmodel: PiecesModel.Model<Tile.Tile8>,+xmodel: PiecesModel.Model<Tile.Tile8>,+dfill: Tile.Tile8,+xfill: Tile.Tile8,+dworkers: U32,+xworkers: U32,
  +dcontract: Public.Contract(~DL,~dv,dl,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dmodel),[images,height,width,Tiled.groups(channels)],dfill,Tensors.Usable{},dworkers},1n,channels,Grid.Innermost{}}),
  +xcontract: Public.Contract(~XL,~xv,xl,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xmodel),[images,height,width,Tiled.groups(channels)],xfill,Tensors.Usable{},xworkers},1n,channels,Grid.Innermost{}}),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(4n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements([images,channels,height,width])) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements([images,height,width,Tiled.groups(channels)])),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  KernelGemm.Both<PlaneSummed(~CombineLeaf.Context<DL,XL>,~CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv),CombineLeaf.Context{dl,xl},images,channels,height,width,
      BN.channel_sums(BN.Layer{images,channels,height,width,workers},Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,Pair.snd(BNCells.Pairs(),List<&2,Sum.TileParts>,Partials.read(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),~Sum.tile_merge,~BNCells.vacant_pair,[Unit{}],9.0,2.0,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel))))))),
    {Pair.fst(BNCells.Pairs(),List<&2,Sum.TileParts>,Partials.read(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),~Sum.tile_merge,~BNCells.vacant_pair,[Unit{}],9.0,2.0,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel)))) == Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel)) : BNCells.Pairs()}>:
  KernelGemm.Both{+dglobal,+dlaid} = dcontract
  Public.Laid{+inside,+shaped_cells,+restored,+counted,+pinned} = dlaid
  KernelGemm.Both{+xglobal,+xlaid} = xcontract
  +values = {[images,channels,height,width] : List<&2,U32>}
  +shape = {[images,height,width,Tiled.groups(channels)] : List<&2,U32>}
  +capacity = U32.to_nat(Machine.capacity(limits))
  +inner = Grid.product([Tiled.groups(channels)])
  +values_room = StorageCount.counted_room(limits,values,counted,default_capacity)
  +rooms_out = ReduceLaws.kept_rooms(limits,values,[0n,2n,3n],1n,Grid.Outermost{},{==},{==},values_room,nonempty)
  +rooms_in = ReduceLaws.kept_rooms(limits,values,[0n,2n,3n],1n,Grid.Innermost{},{==},{==},values_room,nonempty)
  +room_in = Order.transitive(ShapeLaws.elements(shape),ShapeLaws.elements(values),capacity,ReduceLaws.rooms_cells_fewer(limits,values,[0n,2n,3n],1n,Grid.Innermost{},rooms_in),values_room)
  +positive_in = ReduceLaws.rooms_cells_positive(limits,values,[0n,2n,3n],1n,Grid.Innermost{},rooms_in)
  +turns = {Reduce.Turns{U32.to_nat(inner),{==},Equal.sym(Nat,Nat.mul(U32.to_nat(inner),1n),U32.to_nat(inner),Order.times_one(U32.to_nat(inner)))} : Reduce.Turns<Unit,[Unit{}],inner>}
  plane_contract(~BNCells.CellPair(),~Unit,~SlotsValues.held(~Unit),~BNCells.product_slots,~BNCells.vacant_pair,
    ~CombineLeaf.Context<DL,XL>,~CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>,~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.Lifted<XL>,~Tile.times,~Representation.lifted(~DL,~dv),~Representation.lifted(~XL,~xv)),~pair_at(~DL,~XL),~pair_moved(~DL,~dv,~XL,~xv),~CombineLeaf.Context<PiecesModel.Model<Tile.Tile8>,PiecesModel.Model<Tile.Tile8>>,~paired_models,~CombineLeaf.Context<DL,XL>,~CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv),
    ~plane_input(~CombineLeaf.Context<DL,XL>),~sums_extras(~CombineLeaf.Context<DL,XL>),~plane_unit(~CombineLeaf.Context<DL,XL>),~sums_ready(~CombineLeaf.Context<DL,XL>),~sums_started(~CombineLeaf.Context<DL,XL>),~sums_enough(~CombineLeaf.Context<DL,XL>),~product_agrees(~DL,~dv,~XL,~xv),
    CombineLeaf.Context{dl,xl},limits,images,channels,height,width,9.0,2.0,workers,CombineLeaf.Context{dmodel,xmodel},default_capacity,rank_room,rooms_out,Unit{},
    plane_shaped(~BNCells.CellPair(),~Each.PiecePair<Tile.Tile8>,~Each.pair_pack(~Tile.Tile8),~TwinsLaws.product_cell,~Unit,~SlotsValues.held(~Unit),~BNCells.product_slots,
      ~BNCells.vacant_pair,~TwinsLaws.vacant_twin,~TwinsLaws.vacant_same,~TwinsLaws.pair_chain,~CombineLeaf.Context<DL,XL>,~CombineLeaf.Context<Representation.Lifted<DL>,Representation.Lifted<XL>>,~CombineLeaf.value(~Tile.Tile8,~Representation.Lifted<DL>,~Representation.Lifted<XL>,~Tile.times,~Representation.lifted(~DL,~dv),~Representation.lifted(~XL,~xv)),~pair_at(~DL,~XL),~CombineLeaf.Context<PiecesModel.Model<Tile.Tile8>,PiecesModel.Model<Tile.Tile8>>,~paired_models,
      CombineLeaf.Context{dl,xl},[Unit{}],limits,images,channels,height,width,9.0,2.0,workers,CombineLeaf.Context{dmodel,xmodel},
      contracts_streamed(~DL,~dv,~XL,~xv,dl,xl,images,channels,height,width,dmodel,xmodel,dfill,xfill,dworkers,xworkers,dglobal,xglobal),
      turns,room_in,positive_in,default_capacity,cap,cap_ok,count_room))

# The scales of a read of products: its pairs, the shifts and its finished
# partials.
def scales_parts(shifts: List<&2,Tile.Tile8>,result: BNCells.Pairs() & List<&2,Sum.TileParts>) ->
  {BNCells.scales_of(shifts,result) == (Pair.fst(BNCells.Pairs(),List<&2,Sum.TileParts>,result),(shifts,Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,Pair.snd(BNCells.Pairs(),List<&2,Sum.TileParts>,result)))) :
    BNCells.Pairs() & (List<&2,Tile.Tile8> & List<&2,Tile.Tile8>)}:
  match result:
    case (pairs,partials): {==}

# The sums of dy that hand dy back and the sums of the products of the pieces
# handed back with xhat are the sums of the gradient.
def gradient_joined(~DL: Data,~dv: DL -> U32 -> F32,~XL: Data,~xv: XL -> U32 -> F32,+dl: DL,+xl: XL,+images: U32,+channels: U32,+height: U32,+width: U32,+workers: U32,
  +dmodel: PiecesModel.Model<Tile.Tile8>,+xmodel: PiecesModel.Model<Tile.Tile8>,result: Pieces.Pieces(~Tile.Tile8) & List<&2,Tile.Tile8>,
  sums_facts: KernelGemm.Both<PlaneSummed(~DL,~dv,dl,images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Pair.snd(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,result))),
    {Pair.fst(Pieces.Pieces(~Tile.Tile8),List<&2,Tile.Tile8>,result) == PiecesModel.pack(~Tile.Tile8,dmodel) : Pieces.Pieces(~Tile.Tile8)}>,
  product_facts: KernelGemm.Both<PlaneSummed(~CombineLeaf.Context<DL,XL>,~CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv),CombineLeaf.Context{dl,xl},images,channels,height,width,
      BN.channel_sums(BN.Layer{images,channels,height,width,workers},Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,Pair.snd(BNCells.Pairs(),List<&2,Sum.TileParts>,Partials.read(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),~Sum.tile_merge,~BNCells.vacant_pair,[Unit{}],9.0,2.0,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel))))))),
    {Pair.fst(BNCells.Pairs(),List<&2,Sum.TileParts>,Partials.read(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),~Sum.tile_merge,~BNCells.vacant_pair,[Unit{}],9.0,2.0,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel)))) == Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel)) : BNCells.Pairs()}>) ->
  KernelGemm.Both<KernelGemm.Both<PlaneSummed(~DL,~dv,dl,images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Pair.fst(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,Pair.snd(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,BNCells.scales_after([images,height,width,Tiled.groups(channels)],workers,PiecesModel.pack(~Tile.Tile8,xmodel),result))))),
      PlaneSummed(~CombineLeaf.Context<DL,XL>,~CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv),CombineLeaf.Context{dl,xl},images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Pair.snd(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,Pair.snd(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,BNCells.scales_after([images,height,width,Tiled.groups(channels)],workers,PiecesModel.pack(~Tile.Tile8,xmodel),result)))))>,
    {Pair.fst(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,BNCells.scales_after([images,height,width,Tiled.groups(channels)],workers,PiecesModel.pack(~Tile.Tile8,xmodel),result)) == Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel)) : BNCells.Pairs()}>:
  match result:
    case (dy,shifts):
      KernelGemm.Both{+summed,+kept} = sums_facts
      KernelGemm.Both{+product_summed,+product_kept} = product_facts
      %Equal.sym(Pieces.Pieces(~Tile.Tile8),dy,PiecesModel.pack(~Tile.Tile8,dmodel),kept) :
        KernelGemm.Both<KernelGemm.Both<PlaneSummed(~DL,~dv,dl,images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Pair.fst(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,Pair.snd(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,BNCells.scales_of(shifts,Partials.read(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),~Sum.tile_merge,~BNCells.vacant_pair,[Unit{}],9.0,2.0,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),_,PiecesModel.pack(~Tile.Tile8,xmodel)))))))),
      PlaneSummed(~CombineLeaf.Context<DL,XL>,~CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv),CombineLeaf.Context{dl,xl},images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Pair.snd(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,Pair.snd(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,BNCells.scales_of(shifts,Partials.read(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),~Sum.tile_merge,~BNCells.vacant_pair,[Unit{}],9.0,2.0,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),_,PiecesModel.pack(~Tile.Tile8,xmodel))))))))>,
    {Pair.fst(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,BNCells.scales_of(shifts,Partials.read(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),~Sum.tile_merge,~BNCells.vacant_pair,[Unit{}],9.0,2.0,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),_,PiecesModel.pack(~Tile.Tile8,xmodel))))) == Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel)) : BNCells.Pairs()}>
      %Equal.sym(BNCells.Pairs() & (List<&2,Tile.Tile8> & List<&2,Tile.Tile8>),BNCells.scales_of(shifts,Partials.read(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),~Sum.tile_merge,~BNCells.vacant_pair,[Unit{}],9.0,2.0,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel)))),
          (Pair.fst(BNCells.Pairs(),List<&2,Sum.TileParts>,Partials.read(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),~Sum.tile_merge,~BNCells.vacant_pair,[Unit{}],9.0,2.0,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel)))),(shifts,Partials.finished(~Tile.Tile8,~Sum.TileParts,~BNCells.sum_finished,Pair.snd(BNCells.Pairs(),List<&2,Sum.TileParts>,Partials.read(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),~Sum.tile_merge,~BNCells.vacant_pair,[Unit{}],9.0,2.0,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel))))))),
          scales_parts(shifts,Partials.read(~BNCells.CellPair(),~Tile.Tile8,~Unit,~Operations.TileSlots<Unit>,~Sum.TileParts,~Operations.slots_begun(~Unit),~BNCells.product_slots,~Operations.slots_closed(~Unit),~Sum.tile_merge,~BNCells.vacant_pair,[Unit{}],9.0,2.0,[images,height,width,Tiled.groups(channels)],BNCells.summed_flags(),1n,2n,Tile.filled(0.0),workers,Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel))))) :
        KernelGemm.Both<KernelGemm.Both<PlaneSummed(~DL,~dv,dl,images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Pair.fst(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,Pair.snd(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,_)))),
      PlaneSummed(~CombineLeaf.Context<DL,XL>,~CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv),CombineLeaf.Context{dl,xl},images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Pair.snd(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,Pair.snd(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,_))))>,
    {Pair.fst(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,_) == Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel)) : BNCells.Pairs()}>
      KernelGemm.Both{KernelGemm.Both{summed,product_summed},product_kept}

# The sums of the gradient of a batch normalization of tiled dy and xhat
# [N,C,H,W]: the sums of the planes of dy and of dy * xhat of every channel.
def gradient_sums_summed(~DL: Data,~dv: DL -> U32 -> F32,~XL: Data,~xv: XL -> U32 -> F32,+dl: DL,+xl: XL,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+workers: U32,
  +dmodel: PiecesModel.Model<Tile.Tile8>,+xmodel: PiecesModel.Model<Tile.Tile8>,+dfill: Tile.Tile8,+xfill: Tile.Tile8,+dworkers: U32,+xworkers: U32,
  +dcontract: Public.Contract(~DL,~dv,dl,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dmodel),[images,height,width,Tiled.groups(channels)],dfill,Tensors.Usable{},dworkers},1n,channels,Grid.Innermost{}}),
  +xcontract: Public.Contract(~XL,~xv,xl,[images,channels,height,width],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xmodel),[images,height,width,Tiled.groups(channels)],xfill,Tensors.Usable{},xworkers},1n,channels,Grid.Innermost{}}),
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rank_room: {Nat.is_le(4n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements([images,channels,height,width])) == True{} : Bool},+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(Nat.double(ShapeLaws.elements([images,height,width,Tiled.groups(channels)])),4n),U32.to_nat(cap)) == True{} : Bool}) ->
  KernelGemm.Both<KernelGemm.Both<PlaneSummed(~DL,~dv,dl,images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Pair.fst(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,Pair.snd(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,BNCells.gradient_sums([images,height,width,Tiled.groups(channels)],workers,PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel)))))),
      PlaneSummed(~CombineLeaf.Context<DL,XL>,~CombineLeaf.value(~F32,~DL,~XL,~product_pair,~dv,~xv),CombineLeaf.Context{dl,xl},images,channels,height,width,BN.channel_sums(BN.Layer{images,channels,height,width,workers},Pair.snd(List<&2,Tile.Tile8>,List<&2,Tile.Tile8>,Pair.snd(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,BNCells.gradient_sums([images,height,width,Tiled.groups(channels)],workers,PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel))))))>,
    {Pair.fst(BNCells.Pairs(),List<&2,Tile.Tile8> & List<&2,Tile.Tile8>,BNCells.gradient_sums([images,height,width,Tiled.groups(channels)],workers,PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel))) == Pieces.zip(~BNCells.Cells(),~BNCells.Cells(),PiecesModel.pack(~Tile.Tile8,dmodel),PiecesModel.pack(~Tile.Tile8,xmodel)) : BNCells.Pairs()}>:
  gradient_joined(~DL,~dv,~XL,~xv,dl,xl,images,channels,height,width,workers,dmodel,xmodel,BNCells.sums([images,height,width,Tiled.groups(channels)],workers,PiecesModel.pack(~Tile.Tile8,dmodel)),
    channel_sums_summed(~DL,~dv,dl,limits,images,channels,height,width,workers,dmodel,dfill,dworkers,dcontract,default_capacity,rank_room,nonempty,cap,cap_ok,count_room),
    product_sums_summed(~DL,~dv,~XL,~xv,dl,xl,limits,images,channels,height,width,workers,dmodel,xmodel,dfill,xfill,dworkers,xworkers,dcontract,xcontract,
      default_capacity,rank_room,nonempty,cap,cap_ok,count_room))

# The logical indices of the values lie within the capacity.
def centered_room(+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,
  +values_room: {Nat.is_le(ShapeLaws.elements([images,channels,height,width]),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+split: Public.Split<limits,[images,channels,height,width],1n,channels>) -> TiledExit.Room<Cells.Cells{(Public.lane_prefix([images,channels,height,width],1n,1) * Tiled.product(Public.lane_after([images,channels,height,width],1n)) : U32),Tiled.product(Public.lane_after([images,channels,height,width],1n)),Grid.lanes_at([images,channels,height,width],1n),Tiled.groups(Grid.lanes_at([images,channels,height,width],1n)),Grid.Outermost{}},Machine.capacity(limits)>:
  Public.Split{+blocks,+outer_exact,+elements_exact,+lanes_within,+after_positive} = split
  +total = {%elements_exact : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} values_room :
    {Nat.is_le(Nat.mul(Nat.mul(blocks,U32.to_nat(channels)),Representation.least_one(U32.to_nat(Tiled.product(Public.lane_after([images,channels,height,width],1n))))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  TiledExit.Room{blocks,outer_exact,total,Limits.capacity_fits(limits)}
