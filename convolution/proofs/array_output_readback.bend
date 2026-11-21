# The cells of the root layout, gathered into fresh output cells and finished,
# hold the matrix values.
import Base
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/proofs/tensor_tiled_values.bend as TiledValues
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ./supported_coordinate_bounds.bend as SupportedCoordinates
import ./cell_gather_coordinates.bend as Cells
import ./cell_gather_values.bend as CellValues
import ./microkernel_storage_execution.bend as Execution
import ./array_cell_execution.bend as CellExecution
import ./input_storage.bend as Inputs
import ../../lib/traversal_partition.bend as Partition
import ../../lib/storage_buffer.bend as Storage
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../inference.bend as NativeConv
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/u32_comparison.bend as Comparison
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./array_leaf_cells.bend as ArrayLeaf
import ../../lib/proofs/kernel_partition_bounds.bend as TilePartitionBounds
import ./tile_partition_coverage.bend as TilePartitionCoverage
import ./tile_layout.bend as TileLayout
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packed_output_bounds.bend as Output
import ./output_values.bend as Values
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./gather_tree_readback.bend as GatherTreeReadback
import ./gather_copy_coordinates.bend as GatherCopyCoordinates
import ../../lib/proofs/word_multiplication.bend as WordMultiplication

def selected_depth(+p: ConvSpec.Problem,depth: U32) -> Nat:
  U32.to_nat(U32.min(depth,(U32.from_nat(Storage.bit_length(32n,(Padded.padded(p) / 8 : U32))) - 1 : U32)))

# Every leaf of the root layout lies aligned within the padded grid.
def geometry(+limits: Machine.Limits,+problem: ConvSpec.Problem,+depth: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  ArrayLeaf.AllGeometry(problem,TileLayout.divide(selected_depth(problem,depth),0,Padded.padded(problem))):
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  +span = Padded.padded(problem)
  +span_exact = Padded.padded_exact(limits,problem,facts,fast)
  +span_bound = {%Equal.sym(Nat,U32.to_nat(span),Support.padded(problem),span_exact) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    Order.reflexive(Support.padded(problem)) : {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}}
  ArrayLeaf.divide_geometry(limits,problem,selected_depth(problem,depth),0,span,valid,supported,
    TilePartitionBounds.BlockAligned{0n,{==}},TilePartitionBounds.BlockAligned{Nat.div(Nat.add(GatherCopyCoordinates.spatial(problem),7n),8n),span_exact},
    Padded.root_room(limits,problem,facts,fast),span_bound,span_bound)

def position_member(+limits: Machine.Limits,+problem: ConvSpec.Problem,+depth: U32,+output_row_index: Nat,+output_column_index: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  row_inside: {Nat.is_lt(output_row_index,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(output_column_index,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),TilePartitionCoverage.positions(TileLayout.divide(selected_depth(problem,depth),0,Padded.padded(problem)))):
  +facts = Guards.valid(problem,valid)
  +fast = Support.fast(limits,problem,supported)
  %Equal.sym(List<&2,Nat>,TilePartitionCoverage.positions(TileLayout.divide(selected_depth(problem,depth),0,Padded.padded(problem))),Ranges.range(U32.to_nat(Padded.padded(problem)),0n),
    TilePartitionCoverage.split_tiles_cover_range(selected_depth(problem,depth),0,Padded.padded(problem),Machine.capacity(limits),Limits.capacity_fits(limits),Padded.root_room(limits,problem,facts,fast))) :
    Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),_)
  %Equal.sym(Nat,U32.to_nat(Padded.padded(problem)),Support.padded(problem),Padded.padded_exact(limits,problem,facts,fast)) :
    Members.Member(Nat,Coordinates.position(problem,output_row_index,output_column_index),Ranges.range(_,0n))
  Members.range_member(Support.padded(problem),0n,Coordinates.position(problem,output_row_index,output_column_index),Bounds.nat_trans(Coordinates.position(problem,output_row_index,output_column_index),GatherCopyCoordinates.spatial(problem),Support.padded(problem),
    RowMajor.radix(output_row_index,ConvSpec.out_height(problem),output_column_index,ConvSpec.out_width(problem),row_inside,column_inside),Padded.spatial_within_padded(limits,problem,facts,fast)))

def count(+problem: ConvSpec.Problem) -> U32:
  Pipeline.output_tiles(MicroLeaf.rows(problem),Padded.spatial(problem))

def initial(+problem: ConvSpec.Problem) -> Tree.Tree<Tile.Tile8>:
  Execution.allocated_tiles(MicroLeaf.rows(problem),Padded.spatial(problem))

def count_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) ->
  {U32.to_nat(count(problem)) == Nat.mul(U32.to_nat(Cells.groups(problem)),U32.to_nat(Padded.spatial(problem))) : Nat}:
  WordMultiplication.bounded_multiply(Cells.groups(problem),Padded.spatial(problem),Machine.capacity(limits),Limits.capacity_fits(limits),Cells.slots_bound(limits,problem,valid,supported))

def allocated(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> Depth.Sized<count(problem)>:
  Depth.bounded(count(problem),Arithmetic.count_within(count(problem),Nat.mul(U32.to_nat(Cells.groups(problem)),U32.to_nat(Padded.spatial(problem))),Machine.capacity(limits),1073741824,
    count_exact(limits,problem,valid,supported),Cells.slots_bound(limits,problem,valid,supported),Limits.capacity_allocatable(limits)))

def allocated_gather_readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+layout: TileLayout.Layout(),+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificates: ArrayLeaf.AllGeometry(problem,layout),
  +member: Members.Member(Nat,Coordinates.position(problem,row,column),TilePartitionCoverage.positions(layout)),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {CellValues.lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Execution.gather(GatherTreeReadback.result(problem,storage_contract,layout),initial(problem),Padded.spatial(problem)),U32.from_nat(Cells.query_index(problem,channel,row,column)))) == Values.matrix_value(problem,channel,row,column) : F32}:
  +ready = allocated(limits,problem,valid,supported)
  +size = Nat.mul(U32.to_nat(Cells.groups(problem)),U32.to_nat(Padded.spatial(problem)))
  +room = Execution.allocated_room(~Tile.Tile8,count(problem),Tile.filled(0.0),size,count_exact(limits,problem,valid,supported),ready)
  +inside = {%RowMajor.multiply_commutative(U32.to_nat(Padded.spatial(problem)),U32.to_nat(Cells.groups(problem))) : {Nat.is_lt(Cells.query_index(problem,channel,row,column),_) == True{} : Bool}
    Cells.query_inside(limits,problem,channel,row,column,valid,supported,channel_inside,row_inside,column_inside) : {Nat.is_lt(Cells.query_index(problem,channel,row,column),size) == True{} : Bool}}
  GatherTreeReadback.readback(limits,problem,storage_contract,layout,initial(problem),Tensors.depth_for(count(problem)),channel,row,column,valid,supported,certificates,member,channel_inside,row_inside,column_inside,
    Shape.fresh(Tile.Tile8,Tensors.depth_for(count(problem)),Tile.filled(0.0)),Depth.allocation_depth(count(problem),ready),
    Bounds.native_bound(Cells.query_index(problem,channel,row,column),Tree.capacity(Tile.Tile8,initial(problem)),Cells.query_exact(limits,problem,channel,row,column,valid,supported,channel_inside,row_inside,column_inside),
      Bounds.nat_trans(Cells.query_index(problem,channel,row,column),size,U32.to_nat(Tree.capacity(Tile.Tile8,initial(problem))),inside,room)),room)

def finish_leaf_readback(+identity: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+offset: U32,+span: U32,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificate: ArrayLeaf.Geometry<problem,offset,span>,
  +member: Members.Member(Nat,Coordinates.position(problem,row,column),Ranges.range(U32.to_nat(span),U32.to_nat(offset))),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
  +guard: {U32.is_zero(offset) && U32.is_eq(span,Padded.spatial(problem)) == identity : Bool}) ->
  {CellValues.lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Execution.finish_leaf(identity,ArrayLeaf.leaf_cells(problem,storage_contract,offset,span),MicroLeaf.rows(problem),offset,span,Padded.spatial(problem)),U32.from_nat(Cells.query_index(problem,channel,row,column)))) == Values.matrix_value(problem,channel,row,column) : F32}:
  match identity:
    case False{}:
      allocated_gather_readback(limits,problem,storage_contract,Partition.Leaf{Unit{},offset,span,Unit{}},channel,row,column,valid,supported,certificate,member,channel_inside,row_inside,column_inside)
    case True{}:
      +offset_zero = Order.equal_of_is_eq(U32.to_nat(offset),0n,
        %Comparison.eq(offset,0) : {_ == True{} : Bool}
        Boolean.left(U32.is_zero(offset),U32.is_eq(span,Padded.spatial(problem)),guard))
      +span_exact = Order.equal_of_is_eq(U32.to_nat(span),U32.to_nat(Padded.spatial(problem)),
        %Comparison.eq(span,Padded.spatial(problem)) : {_ == True{} : Bool}
        Boolean.right(U32.is_zero(offset),U32.is_eq(span,Padded.spatial(problem)),guard))
      ArrayLeaf.leaf_cells_every_position(limits,problem,storage_contract,offset,span,channel,Coordinates.position(problem,row,column),row,column,valid,supported,certificate,channel_inside,
        %Equal.sym(Nat,U32.to_nat(span),U32.to_nat(Padded.spatial(problem)),span_exact) : {Nat.is_lt(Coordinates.position(problem,row,column),_) == True{} : Bool}
        Cells.position_inside(limits,problem,row,column,valid,supported,row_inside,column_inside),
        %Equal.sym(Nat,U32.to_nat(offset),0n,offset_zero) : {Nat.add(_,Coordinates.position(problem,row,column)) == Coordinates.position(problem,row,column) : Nat} {==},row_inside,column_inside)

def finish_readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+layout: TileLayout.Layout(),+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+certificates: ArrayLeaf.AllGeometry(problem,layout),
  +member: Members.Member(Nat,Coordinates.position(problem,row,column),TilePartitionCoverage.positions(layout)),
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {CellValues.lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Execution.finish(GatherTreeReadback.result(problem,storage_contract,layout),MicroLeaf.rows(problem),Padded.spatial(problem)),U32.from_nat(Cells.query_index(problem,channel,row,column)))) == Values.matrix_value(problem,channel,row,column) : F32}:
  match layout:
    case Partition.Leaf{Unit{},+offset,+span,Unit{}}:
      finish_leaf_readback(U32.is_zero(offset) && U32.is_eq(span,Padded.spatial(problem)),limits,problem,storage_contract,offset,span,channel,row,column,valid,supported,certificates,member,channel_inside,row_inside,column_inside,{==})
    case Partition.Fork{+left,+right}:
      allocated_gather_readback(limits,problem,storage_contract,Partition.Fork{left,right},channel,row,column,valid,supported,certificates,member,channel_inside,row_inside,column_inside)

def output(+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,depth: U32) -> Tree.Tree<F32>:
  CellExecution.output(problem,storage_contract,depth)

def every_cell_matches_matrix(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+depth: U32,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {CellValues.lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,CellExecution.cells_output(problem,storage_contract,depth),U32.from_nat(Cells.query_index(problem,channel,row,column)))) == Values.matrix_value(problem,channel,row,column) : F32}:
  finish_readback(limits,problem,storage_contract,TileLayout.divide(selected_depth(problem,depth),0,Padded.padded(problem)),channel,row,column,valid,supported,geometry(limits,problem,depth,valid,supported),
    position_member(limits,problem,depth,row,column,valid,supported,row_inside,column_inside),channel_inside,row_inside,column_inside)

def every_output_matches_matrix(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage_contract: Inputs.Contract<problem>,+depth: U32,+channel: Nat,+row: Nat,+column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {Tree.read(F32,output(problem,storage_contract,depth),U32.from_nat(RowMajor.index(channel,ConvSpec.out_height(problem),ConvSpec.out_width(problem),row,column))) == Values.matrix_value(problem,channel,row,column) : F32}:
  +fast = Support.fast(limits,problem,supported)
  +rows = MicroLeaf.rows(problem)
  +space = Padded.spatial(problem)
  +channels = Guards.output_channel_count(problem)
  +rows_exact = SupportedCoordinates.co_exact(limits,problem,Padded.scalar(limits,problem,fast))
  +spatial_exact = Padded.spatial_exact(limits,problem,Guards.valid(problem,valid),fast)
  +span_bound = Cells.spatial_bound(limits,problem,valid,supported)
  +count = (rows * space : U32)
  +count_exact = Output.count_exact(limits,problem,space,fast,span_bound)
  +exact = Equal.trans(Nat,Nat.mul(U32.to_nat(rows),U32.to_nat(space)),Nat.mul(channels,U32.to_nat(space)),U32.to_nat(count),
    Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(space)),U32.to_nat(rows),channels,rows_exact),Equal.sym(Nat,U32.to_nat(count),Nat.mul(channels,U32.to_nat(space)),count_exact))
  +bounded = {%Equal.sym(Bool,U32.is_le(count,Machine.capacity(limits)),Nat.is_le(U32.to_nat(count),U32.to_nat(Machine.capacity(limits))),Comparison.le(count,Machine.capacity(limits))) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(count),Nat.mul(channels,U32.to_nat(space)),count_exact) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    Output.elements_bound(limits,problem,space,fast,span_bound) : {U32.is_le(count,Machine.capacity(limits)) == True{} : Bool}}
  +point = Coordinates.position(problem,row,column)
  +cells = CellExecution.cells_output(problem,storage_contract,depth)
  +selected = CellValues.lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,cells,U32.from_nat(Cells.query_index(problem,channel,row,column))))
  %spatial_exact : {Tree.read(F32,output(problem,storage_contract,depth),U32.from_nat(Nat.add(Nat.mul(channel,_),point))) == Values.matrix_value(problem,channel,row,column) : F32}
  Equal.trans(F32,Tree.read(F32,output(problem,storage_contract,depth),U32.from_nat(Nat.add(Nat.mul(channel,U32.to_nat(space)),point))),selected,Values.matrix_value(problem,channel,row,column),
    TiledValues.untiled_read(limits,count,space,space,rows,KernelGemm.blocks(rows),cells,channel,point,exact,bounded,
      %Equal.sym(Nat,U32.to_nat(rows),channels,rows_exact) : {Nat.is_lt(channel,_) == True{} : Bool} channel_inside,
      Cells.position_inside(limits,problem,row,column,valid,supported,row_inside,column_inside)),
    every_cell_matches_matrix(limits,problem,storage_contract,depth,channel,row,column,valid,supported,channel_inside,row_inside,column_inside))
