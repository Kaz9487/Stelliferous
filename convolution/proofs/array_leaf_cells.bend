# The cells of an Array leaf: its span packed into fresh tiles as the problem
# values and multiplied into fresh output cells. A layout of leaves carries the
# geometry each leaf needs.
import Base
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/machine_limits.bend as Machine
import ../../lib/traversal_partition.bend as Partition
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/tensor_allocation_depth.bend as Depth
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/word_multiplication.bend as WordMultiplication
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/kernel_gemm_readback.bend as GemmReadback
import ../../lib/proofs/kernel_partition_bounds.bend as TilePartitionBounds
import ../../lib/proofs/traversal_partition_refinement.bend as PartitionLaws
import ../../lib/kernel_pipeline.bend as Pipeline
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ./input_storage.bend as Inputs
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./reduction_coordinate_bounds.bend as Coordinates
import ./packing_tile_bounds.bend as PackingBounds
import ./packed_output_bounds.bend as Output
import ./workspace_packing_samples.bend as Prepared
import ./convolution_leaf_contract.bend as LeafContract
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./microkernel_storage_execution.bend as Execution
import ./tile_layout.bend as TileLayout
import ./output_values.bend as Values

# ---- The cells of a leaf ----

def fresh_tiles(+problem: ConvSpec.Problem,+span: U32) -> Tree.Tree<Tile.Tile8>:
  Tree.fresh(Tile.Tile8,Tensors.depth_for(PackingBounds.count(span,problem)),Tile.filled(0.0))

def packed_tiles(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+offset: U32,+span: U32) -> Tree.Tree<Tile.Tile8>:
  Prepared.workspace_filled(problem,storage,offset,span,fresh_tiles(problem,span))

def leaf_cells(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+offset: U32,+span: U32) -> Tree.Tree<Tile.Tile8>:
  MicroLeaf.leaf_tree(problem,storage,packed_tiles(problem,storage,offset,span),span,0,span,Execution.allocated_tiles(MicroLeaf.rows(problem),span))

type Stored is Data:
  Stored{problem: ConvSpec.Problem,storage: Inputs.Contract<problem>}

def part_rows(+context: Stored,+details: Unit) -> U32:
  Stored{+problem,storage} = context
  MicroLeaf.rows(problem)

def part_cells(+context: Stored,+offset: U32,+span: U32,+details: Unit,block: Unit) -> Tree.Tree<Tile.Tile8>:
  Stored{+problem,+storage} = context
  leaf_cells(problem,storage,offset,span)

def layout_cells(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,layout: TileLayout.Layout()) -> Partition.Tree<&2,Tree.Tree<Tile.Tile8>,U32>:
  PartitionLaws.map(~&2,~&2,~Unit,~Tree.Tree<Tile.Tile8>,~Unit,~U32,~Stored,~part_cells,~part_rows,Stored{problem,storage},layout)

# ---- The geometry of a layout ----

type Geometry<-problem: ConvSpec.Problem,-offset: U32,-span: U32> is Data:
  Geometry{span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},alignment: TilePartitionBounds.BlockAligned<offset>,
    span_alignment: TilePartitionBounds.BlockAligned<span>,range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool}}

def PartGeometry(+problem: ConvSpec.Problem,+offset: U32,+span: U32,+details: Unit,block: Unit) -> Data:
  Geometry<problem,offset,span>

def AllGeometry(+problem: ConvSpec.Problem,layout: TileLayout.Layout()) -> Data:
  PartitionLaws.All(~&2,~Unit,~Unit,~ConvSpec.Problem,~PartGeometry,problem,layout)

def span_bound(+problem: ConvSpec.Problem,+offset: U32,+span: U32,+geometry: Geometry<problem,offset,span>) ->
  {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}:
  Geometry{+span_bound,alignment,span_alignment,range_bound} = geometry
  span_bound

def offset_bound(+problem: ConvSpec.Problem,+offset: U32,+span: U32,+geometry: Geometry<problem,offset,span>) ->
  {Nat.is_le(U32.to_nat(offset),Support.padded(problem)) == True{} : Bool}:
  Geometry{span_bound,alignment,span_alignment,+range_bound} = geometry
  Order.transitive(U32.to_nat(offset),Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem),Order.le_add(U32.to_nat(offset),U32.to_nat(span)),range_bound)

# Halving an aligned range inside the padded grid keeps both halves aligned and
# inside it.
def divide_geometry(+limits: Machine.Limits,+problem: ConvSpec.Problem,+depth: Nat,+offset: U32,+span: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +alignment: TilePartitionBounds.BlockAligned<offset>,+span_alignment: TilePartitionBounds.BlockAligned<span>,
  +room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +range_bound: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(span)),Support.padded(problem)) == True{} : Bool}) -> AllGeometry(problem,TileLayout.divide(depth,offset,span)):
  match depth:
    case 0n:
      Geometry{span_bound,alignment,span_alignment,range_bound}
    case 1n+ +rest:
      +half = TilePartitionBounds.half(span)
      +capacity = Machine.capacity(limits)
      +capacity_ok = Limits.capacity_fits(limits)
      +bounded = Order.transitive(U32.to_nat(span),Support.padded(problem),U32.to_nat(capacity),span_bound,
        Padded.padded_bound(limits,problem,Guards.valid(problem,valid),Support.fast(limits,problem,supported)))
      +left_bound = Order.transitive(U32.to_nat(half),U32.to_nat(span),Support.padded(problem),TilePartitionBounds.half_le(span,capacity,capacity_ok,bounded),span_bound)
      +right_bound = Order.transitive(U32.to_nat((span - half : U32)),U32.to_nat(span),Support.padded(problem),TilePartitionBounds.remainder_le(span,capacity,capacity_ok,bounded),span_bound)
      Gemm.Both{
        divide_geometry(limits,problem,rest,offset,half,valid,supported,alignment,TilePartitionBounds.half_alignment(span,capacity,capacity_ok,bounded),
          TilePartitionBounds.left_room(offset,span,capacity,capacity_ok,room),left_bound,
          TilePartitionBounds.left_interval_bound(offset,span,capacity,Support.padded(problem),capacity_ok,room,range_bound)),
        divide_geometry(limits,problem,rest,(offset + half : U32),(span - half : U32),valid,supported,
          TilePartitionBounds.right_alignment(offset,span,capacity,capacity_ok,room,alignment),TilePartitionBounds.remainder_alignment(span,capacity,capacity_ok,bounded,span_alignment),
          TilePartitionBounds.right_room(offset,span,capacity,capacity_ok,room),right_bound,
          TilePartitionBounds.right_interval_bound(offset,span,capacity,Support.padded(problem),capacity_ok,room,range_bound))}

# ---- The values of a leaf ----

# The fresh tiles of a leaf hold its packing and are filled as the problem
# values.
def evidence(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+offset: U32,+span: U32,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}) ->
  LeafContract.WorkspaceEvidence<problem,offset,span,packed_tiles(problem,storage,offset,span)>:
  +count = PackingBounds.count(span,problem)
  +within = PackingBounds.count_bound(limits,problem,span,Guards.valid(problem,valid),Support.fast(limits,problem,supported),span_bound)
  +ready = Depth.bounded(count,Arithmetic.count_within(count,U32.to_nat(count),Machine.capacity(limits),1073741824,{==},within,Limits.capacity_allocatable(limits)))
  LeafContract.WorkspaceEvidence{storage,fresh_tiles(problem,span),Tensors.depth_for(count),Shape.fresh(Tile.Tile8,Tensors.depth_for(count),Tile.filled(0.0)),
    Depth.allocation_depth(count,ready),Execution.allocated_room(~Tile.Tile8,count,Tile.filled(0.0),U32.to_nat(count),{==},ready),span_bound,{==}}

# Every channel of every position of a leaf holds the matrix value.
def leaf_cells_every_position(+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +offset: U32,+span: U32,+channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +geometry: Geometry<problem,offset,span>,
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat(offset),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_in_width: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,leaf_cells(problem,storage,offset,span),
      U32.from_nat(Nat.add(Nat.mul(column,U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),Nat.div(channel,8n))))) ==
    Values.matrix_value(problem,channel,output_row,output_column) : F32}:
  Geometry{+span_bound,alignment,span_alignment,range_bound} = geometry
  TilePartitionBounds.BlockAligned{+blocks,+span_exact} = span_alignment
  +fast = Support.fast(limits,problem,supported)
  +capacity = Machine.capacity(limits)
  +rows_word = MicroLeaf.rows(problem)
  +rows_exact = SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,fast))
  +elements = Output.elements_bound(limits,problem,span,fast,span_bound)
  +units = U32.to_nat(Gemm.blocks(rows_word))
  +tiles_count = Pipeline.output_tiles(rows_word,span)
  +product = MicroLeaf.leaf_product(limits,problem,storage,packed_tiles(problem,storage,offset,span),span,blocks,0,span,Guards.valid(problem,valid),fast,span_exact,
    Order.le_add(U32.to_nat(span),7n),elements)
  +tile_elements = Order.transitive(Nat.mul(units,U32.to_nat(span)),Nat.mul(U32.to_nat(rows_word),U32.to_nat(span)),U32.to_nat(capacity),
    RowMajor.mul_le(units,U32.to_nat(rows_word),U32.to_nat(span),GemmReadback.units_within(product)),
    %Equal.sym(Nat,U32.to_nat(rows_word),Guards.output_channel_count(problem),rows_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(span)),U32.to_nat(capacity)) == True{} : Bool} elements)
  +tiles_exact = WordMultiplication.bounded_multiply(Gemm.blocks(rows_word),span,capacity,Limits.capacity_fits(limits),tile_elements)
  +tiles_ready = Depth.bounded(tiles_count,Arithmetic.count_within(tiles_count,Nat.mul(units,U32.to_nat(span)),capacity,1073741824,tiles_exact,tile_elements,Limits.capacity_allocatable(limits)))
  +depth = Tensors.depth_for(tiles_count)
  MicroLeaf.relocated_readback(~LeafContract.workspace_evidence,~LeafContract.workspace_samples,limits,problem,storage,valid,supported,packed_tiles(problem,storage,offset,span),
    offset,span,blocks,channel,column,output_row,output_column,evidence(limits,problem,storage,offset,span,valid,supported,span_bound),
    span_exact,span_bound,channel_inside,column_inside,position,row_inside,column_in_width,
    0,span,Execution.allocated_tiles(rows_word,span),depth,Shape.fresh(Tile.Tile8,depth,Tile.filled(0.0)),Depth.allocation_depth(tiles_count,tiles_ready),
    %RowMajor.multiply_commutative(units,U32.to_nat(span)) : {Nat.is_le(_,U32.to_nat(Tree.capacity(Tile.Tile8,Execution.allocated_tiles(rows_word,span)))) == True{} : Bool}
    Execution.allocated_room(~Tile.Tile8,tiles_count,Tile.filled(0.0),Nat.mul(units,U32.to_nat(span)),tiles_exact,tiles_ready),
    Order.le_add(U32.to_nat(span),7n),elements,column_inside)
