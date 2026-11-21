# The blocks of a leaf on the values of a convolution problem store its matrix
# values: packed from the pack origin on, written from the output offset on.
import Base
import ../../lib/kernel_window.bend as Window
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/kernel_pipeline_bounds.bend as BlockBounds
import ../../lib/proofs/kernel_partition_bounds.bend as Partition
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_extrema.bend as Extrema
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/nat_division.bend as Division
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/kernel_gemm_readback.bend as GemmReadback
import ../../lib/proofs/kernel_gemm_jobs.bend as GemmJobs
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/write_program.bend as Writes
import ./supported_coordinate_bounds.bend as SupportedCoordinates
import ./supported_padded_coordinates.bend as Padded
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/proofs/kernel_pipeline_model.bend as Model
import ./input_storage.bend as Inputs
import ./workspace_packing_samples.bend as Prepared
import ../../lib/proofs/kernel_spatial_packing_samples.bend as SpatialSamples
import ./packing_tile_bounds.bend as PackingBounds
import ./convolution_leaf_contract.bend as LeafContract
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./reduction_coordinate_bounds.bend as Coordinates
import ./output_values.bend as Values

# The window work of a leaf: the matrix of the problem's output channels by its
# reduction, written up to the position end.
def work(+problem: ConvSpec.Problem,end: U32) -> Pipeline.WindowWork:
  Pipeline.WindowWork{MicroLeaf.rows(problem),end,Prepared.window(problem)}

def local_position(problem: ConvSpec.Problem,origin: U32,row: Nat,column: Nat) -> Nat:
  Nat.sub(Coordinates.position(problem,row,column),U32.to_nat(origin))

def state(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,workspace: Tree.Tree<Tile.Tile8>,output: Tree.Tree<Tile.Tile8>) -> Model.State:
  Model.State{Inputs.input_tree(problem,storage),workspace,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}

def tiles(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,workspace: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  Prepared.workspace_filled(problem,storage,start,span,workspace)

def output(+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,workspace: Tree.Tree<Tile.Tile8>,previous: Tree.Tree<Tile.Tile8>) -> Tree.Tree<Tile.Tile8>:
  MicroLeaf.leaf_tree(problem,storage,tiles(problem,storage,(origin + start : U32),span,workspace),span,(offset + start : U32),(offset + positions : U32),previous)

# A block packs from origin + start and writes from offset + start.
def step_state(+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>) ->
  {Model.step_window(work(problem,(offset + positions : U32)),(origin + start : U32),(offset + start : U32),span,state(problem,storage,workspace,previous)) ==
    state(problem,storage,tiles(problem,storage,(origin + start : U32),span,workspace),output(problem,origin,offset,positions,storage,start,span,workspace,previous)) : Model.State}:
  {==}

def tiles_balanced(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+workspace: Tree.Tree<Tile.Tile8>,+depth: Nat,
  balanced: Shape.Balanced(Tile.Tile8,depth,workspace)) -> Shape.Balanced(Tile.Tile8,depth,tiles(problem,storage,start,span,workspace)):
  SpatialSamples.fill_balanced(U32.to_nat(PackingBounds.count(span,problem)),Inputs.input_tree(problem,storage),Prepared.window(problem),start,depth,workspace,balanced)

def output_balanced(+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+depth: Nat,
  balanced: Shape.Balanced(Tile.Tile8,depth,previous)) -> Shape.Balanced(Tile.Tile8,depth,output(problem,origin,offset,positions,storage,start,span,workspace,previous)):
  +input = tiles(problem,storage,(origin + start : U32),span,workspace)
  +weights = Inputs.weight_cells(problem,storage)
  +bias = Inputs.bias_cells(problem,storage)
  +matrix = MicroLeaf.leaf_matrix(problem,span,(offset + start : U32),(offset + positions : U32))
  +writes = GemmJobs.cell_all_writes(U32.to_nat(Gemm.tile_jobs(matrix)),0,matrix,
    GemmJobs.CellReaders{input,weights,bias,Tree.fresh(F32,0n,0.0)})
  %Equal.sym(Tree.Tree<Tile.Tile8>,output(problem,origin,offset,positions,storage,start,span,workspace,previous),Writes.pure(Tile.Tile8,writes,previous),
    GemmReadback.cell_tree_writes(matrix,input,weights,bias,previous)) : Shape.Balanced(Tile.Tile8,depth,_)
  Writes.writes_preserve_balance(Tile.Tile8,writes,previous,depth,balanced)

# An offset word that adds exactly moves a sum by the offset.
def shifted(+offset: U32,+start: U32,+left: Nat,+exact: {U32.to_nat((offset + start : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(start)) : Nat}) ->
  {Nat.add(U32.to_nat((offset + start : U32)),left) == Nat.add(U32.to_nat(offset),Nat.add(U32.to_nat(start),left)) : Nat}:
  Equal.trans(Nat,Nat.add(U32.to_nat((offset + start : U32)),left),Nat.add(Nat.add(U32.to_nat(offset),U32.to_nat(start)),left),Nat.add(U32.to_nat(offset),Nat.add(U32.to_nat(start),left)),
    Equal.cong(Nat,Nat,value => Nat.add(value,left),U32.to_nat((offset + start : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(start)),exact),
    Addition.associative(U32.to_nat(offset),U32.to_nat(start),left))

def shifted_le(+offset: U32,+start: U32,+left: Nat,+end: U32,+right: Nat,
  +start_exact: {U32.to_nat((offset + start : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(start)) : Nat},
  +end_exact: {U32.to_nat((offset + end : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(end)) : Nat},
  +ordered: {Nat.is_le(Nat.add(U32.to_nat(start),left),Nat.add(U32.to_nat(end),right)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat((offset + start : U32)),left),Nat.add(U32.to_nat((offset + end : U32)),right)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(U32.to_nat((offset + start : U32)),left),Nat.add(U32.to_nat(offset),Nat.add(U32.to_nat(start),left)),shifted(offset,start,left,start_exact)) :
    {Nat.is_le(_,Nat.add(U32.to_nat((offset + end : U32)),right)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(U32.to_nat((offset + end : U32)),right),Nat.add(U32.to_nat(offset),Nat.add(U32.to_nat(end),right)),shifted(offset,end,right,end_exact)) :
    {Nat.is_le(Nat.add(U32.to_nat(offset),Nat.add(U32.to_nat(start),left)),_) == True{} : Bool}
  Order.offset_le(U32.to_nat(offset),Nat.add(U32.to_nat(start),left),Nat.add(U32.to_nat(end),right),ordered)

def shifted_lt(+offset: U32,+start: U32,+left: Nat,+end: U32,
  +start_exact: {U32.to_nat((offset + start : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(start)) : Nat},
  +end_exact: {U32.to_nat((offset + end : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(end)) : Nat},
  +ordered: {Nat.is_lt(Nat.add(U32.to_nat(start),left),U32.to_nat(end)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat((offset + start : U32)),left),U32.to_nat((offset + end : U32))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(U32.to_nat((offset + start : U32)),left),Nat.add(U32.to_nat(offset),Nat.add(U32.to_nat(start),left)),shifted(offset,start,left,start_exact)) :
    {Nat.is_lt(_,U32.to_nat((offset + end : U32))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((offset + end : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(end)),end_exact) :
    {Nat.is_lt(Nat.add(U32.to_nat(offset),Nat.add(U32.to_nat(start),left)),_) == True{} : Bool}
  Order.offset_lt(U32.to_nat(offset),Nat.add(U32.to_nat(start),left),U32.to_nat(end),ordered)

# The tile of a channel's group at a point of the leaf, written from the offset.
# The cell of a channel at output position at.
def cell_address(+problem: ConvSpec.Problem,+channel: Nat,at: Nat) -> U32:
  U32.from_nat(Nat.add(Nat.mul(at,U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),Nat.div(channel,8n)))

def address(+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+channel: Nat,+row: Nat,+column: Nat) -> U32:
  cell_address(problem,channel,Nat.add(U32.to_nat(offset),local_position(problem,origin,row,column)))

# Only the prefix packed for this block is sampled by the unchanged dot chain.
def block_readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+blocks: Nat,
  +workspace: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+previous: Tree.Tree<Tile.Tile8>,+output_depth: Nat,
  +channel: Nat,+column: Nat,+output_row: Nat,+output_column: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tile_balanced: Shape.Balanced(Tile.Tile8,tile_depth,workspace),+tile_depth_ok: {Nat.is_le(tile_depth,30n) == True{} : Bool},
  +tile_room: {Nat.is_le(U32.to_nat(PackingBounds.count(span,problem)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool},
  +output_balanced: Shape.Balanced(Tile.Tile8,output_depth,previous),+output_depth_ok: {Nat.is_le(output_depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat((offset + positions : U32)),U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),U32.to_nat(Tree.capacity(Tile.Tile8,previous))) == True{} : Bool},
  +outputs_room: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),U32.to_nat((offset + positions : U32))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +start_exact: {U32.to_nat((offset + start : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(start)) : Nat},
  +end_exact: {U32.to_nat((offset + positions : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(positions)) : Nat},
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},+span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+column_inside: {Nat.is_lt(column,U32.to_nat(span)) == True{} : Bool},
  +position: {Nat.add(U32.to_nat((origin + start : U32)),column) == Coordinates.position(problem,output_row,output_column) : Nat},
  +row_inside: {Nat.is_lt(output_row,ConvSpec.out_height(problem)) == True{} : Bool},+column_in_width: {Nat.is_lt(output_column,ConvSpec.out_width(problem)) == True{} : Bool},
  +in_output: {Nat.is_lt(Nat.add(U32.to_nat(start),column),U32.to_nat(positions)) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,output(problem,origin,offset,positions,storage,start,span,workspace,previous),
    U32.from_nat(Nat.add(Nat.mul(Nat.add(U32.to_nat(offset),Nat.add(U32.to_nat(start),column)),U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),Nat.div(channel,8n))))) ==
    Values.matrix_value(problem,channel,output_row,output_column) : F32}:
  %shifted(offset,start,column,start_exact) :
    {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,output(problem,origin,offset,positions,storage,start,span,workspace,previous),
      U32.from_nat(Nat.add(Nat.mul(_,U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),Nat.div(channel,8n))))) == Values.matrix_value(problem,channel,output_row,output_column) : F32}
  MicroLeaf.relocated_readback(~LeafContract.workspace_evidence,~LeafContract.workspace_samples,limits,problem,storage,valid,supported,tiles(problem,storage,(origin + start : U32),span,workspace),(origin + start : U32),span,blocks,channel,column,output_row,output_column,
    LeafContract.WorkspaceEvidence{storage,workspace,tile_depth,tile_balanced,tile_depth_ok,tile_room,span_bound,{==}},
    span_exact,span_bound,channel_inside,column_inside,position,row_inside,column_in_width,(offset + start : U32),(offset + positions : U32),previous,output_depth,output_balanced,output_depth_ok,output_room,
    shifted_le(offset,start,U32.to_nat(span),positions,7n,start_exact,end_exact,end_room),outputs_room,shifted_lt(offset,start,column,positions,start_exact,end_exact,in_output))

# The frame property requires no samples from the block being written.
def block_preserves_before(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+blocks: Nat,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+output_depth: Nat,+channel: Nat,+at: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +output_balanced: Shape.Balanced(Tile.Tile8,output_depth,previous),+output_depth_ok: {Nat.is_le(output_depth,30n) == True{} : Bool},
  +output_room: {Nat.is_le(Nat.mul(U32.to_nat((offset + positions : U32)),U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),U32.to_nat(Tree.capacity(Tile.Tile8,previous))) == True{} : Bool},
  +outputs_room: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),U32.to_nat((offset + positions : U32))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +start_exact: {U32.to_nat((offset + start : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(start)) : Nat},
  +end_exact: {U32.to_nat((offset + positions : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(positions)) : Nat},
  +span_exact: {U32.to_nat(span) == Nat.mul(blocks,8n) : Nat},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(span)),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
  +at_inside: {Nat.is_lt(at,U32.to_nat((offset + positions : U32))) == True{} : Bool},+before: {Nat.is_lt(at,U32.to_nat((offset + start : U32))) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,output(problem,origin,offset,positions,storage,start,span,workspace,previous),cell_address(problem,channel,at)) ==
    Tree.read(Tile.Tile8,previous,cell_address(problem,channel,at)) : Tile.Tile8}:
  +fast = Support.fast(limits,problem,supported)
  +rows_exact = SupportedCoordinates.co_exact(limits,problem,Padded.scalar(limits,problem,fast))
  +product = MicroLeaf.leaf_product(limits,problem,storage,tiles(problem,storage,(origin + start : U32),span,workspace),span,blocks,(offset + start : U32),(offset + positions : U32),
    Guards.valid(problem,valid),fast,span_exact,shifted_le(offset,start,U32.to_nat(span),positions,7n,start_exact,end_exact,end_room),outputs_room)
  +channel_rows = {%Equal.sym(Nat,U32.to_nat(MicroLeaf.rows(problem)),Guards.output_channel_count(problem),rows_exact) : {Nat.is_lt(channel,_) == True{} : Bool}
    channel_inside : {Nat.is_lt(channel,U32.to_nat(MicroLeaf.rows(problem))) == True{} : Bool}}
  GemmReadback.cell_keeps(
    GemmReadback.TileKeep{product,Nat.div(channel,8n),at,GemmReadback.channel_unit(product,channel,channel_rows),at_inside,before},Inputs.bias_cells(problem,storage),output_depth,previous,
    output_balanced,output_depth_ok,output_room)

# The loop carries fixed geometry separately from the two changing trees.
type Geometry<-limits: Machine.Limits,-problem: ConvSpec.Problem,-origin: U32,-offset: U32,-positions: U32,-padded: U32,-span: U32,-cap: U32> is Data:
  Geometry{pack_room: {Nat.is_le(Nat.add(U32.to_nat(origin),U32.to_nat(padded)),U32.to_nat(cap)) == True{} : Bool},valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
    cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
    padded_bound: {Nat.is_le(U32.to_nat(padded),Support.padded(problem)) == True{} : Bool},
    padded_end: {Nat.is_le(U32.to_nat(padded),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},
    outputs_room: {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),U32.to_nat((offset + positions : U32))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    end_exact: {U32.to_nat((offset + positions : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(positions)) : Nat},
    offset_room: {Nat.is_le(Nat.add(U32.to_nat(offset),U32.to_nat(padded)),U32.to_nat(cap)) == True{} : Bool},
    sum_room: {Nat.is_le(Nat.add(U32.to_nat(padded),U32.to_nat(span)),U32.to_nat(cap)) == True{} : Bool},
    span_positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool},
    padded_aligned: Partition.BlockAligned<padded>,span_aligned: Partition.BlockAligned<span>,padded_value: {padded == Window.padded(positions) : U32}}

type StepGeometry<-problem: ConvSpec.Problem,-origin: U32,-offset: U32,-positions: U32,-start: U32,-padded: U32,-span: U32> is Data:
  StepGeometry{pack_exact: {U32.to_nat((origin + start : U32)) == Nat.add(U32.to_nat(origin),U32.to_nat(start)) : Nat},
    output_exact: {U32.to_nat((offset + start : U32)) == Nat.add(U32.to_nat(offset),U32.to_nat(start)) : Nat},
    advance: {U32.to_nat((start + span : U32)) == Nat.add(U32.to_nat(start),U32.to_nat(span)) : Nat},
    clipped: {U32.to_nat(BlockBounds.current(start,padded,span)) == Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))) : Nat},
    positive: {Nat.is_lt(0n,U32.to_nat(BlockBounds.current(start,padded,span))) == True{} : Bool},
    span_bound: {Nat.is_le(U32.to_nat(BlockBounds.current(start,padded,span)),Support.padded(problem)) == True{} : Bool},
    end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(BlockBounds.current(start,padded,span))),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool},
    aligned: Partition.BlockAligned<BlockBounds.current(start,padded,span)>,next_aligned: Partition.BlockAligned<(start + span : U32)>}

def step_geometry(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +geometry: Geometry<limits,problem,origin,offset,positions,padded,span,cap>,
  +inside: {Nat.is_lt(U32.to_nat(start),U32.to_nat(padded)) == True{} : Bool},+aligned: Partition.BlockAligned<start>) -> StepGeometry<problem,origin,offset,positions,start,padded,span>:
  Geometry{+pack_room,+valid,+supported,+cap_ok,+padded_bound,+padded_end,+outputs_room,+end_exact,+offset_room,+sum_room,+span_positive,+padded_aligned,+span_aligned,+padded_value} = geometry
  +ordered = Bounds.strict_le(U32.to_nat(start),U32.to_nat(padded),inside)
  +padded_room = Order.transitive(U32.to_nat(padded),Nat.add(U32.to_nat(padded),U32.to_nat(span)),U32.to_nat(cap),Order.le_add(U32.to_nat(padded),U32.to_nat(span)),sum_room)
  +difference = Arithmetic.subtract(padded,start,cap,cap_ok,ordered,Order.transitive(Nat.sub(U32.to_nat(padded),U32.to_nat(start)),U32.to_nat(padded),U32.to_nat(cap),Order.sub_le(U32.to_nat(padded),U32.to_nat(start)),padded_room))
  +advance = Arithmetic.add(start,span,cap,cap_ok,Order.transitive(Nat.add(U32.to_nat(start),U32.to_nat(span)),Nat.add(U32.to_nat(padded),U32.to_nat(span)),U32.to_nat(cap),Order.add_le(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),ordered),sum_room))
  +clipped = BlockBounds.clipped_exact(start,padded,span,difference)
  +pack_exact = Arithmetic.add(origin,start,cap,cap_ok,
    Order.transitive(Nat.add(U32.to_nat(origin),U32.to_nat(start)),Nat.add(U32.to_nat(origin),U32.to_nat(padded)),U32.to_nat(cap),
      Order.offset_le(U32.to_nat(origin),U32.to_nat(start),U32.to_nat(padded),ordered),pack_room))
  +output_exact = Arithmetic.add(offset,start,cap,cap_ok,
    Order.transitive(Nat.add(U32.to_nat(offset),U32.to_nat(start)),Nat.add(U32.to_nat(offset),U32.to_nat(padded)),U32.to_nat(cap),
      Order.offset_le(U32.to_nat(offset),U32.to_nat(start),U32.to_nat(padded),ordered),offset_room))
  +end_room = {
    %Equal.sym(Nat,U32.to_nat(BlockBounds.current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_le(Nat.add(U32.to_nat(start),_),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool}
    Order.transitive(Nat.add(U32.to_nat(start),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start)))),U32.to_nat(padded),Nat.add(U32.to_nat(positions),7n),BlockBounds.clipped_end(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),ordered),padded_end) :
      {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(BlockBounds.current(start,padded,span))),Nat.add(U32.to_nat(positions),7n)) == True{} : Bool}}
  StepGeometry{pack_exact,output_exact,advance,clipped,
    %Equal.sym(Nat,U32.to_nat(BlockBounds.current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_lt(0n,_) == True{} : Bool}
    Extrema.minimum_strict_lower(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start)),0n,span_positive,Order.positive_difference(U32.to_nat(padded),U32.to_nat(start),inside)),
    %Equal.sym(Nat,U32.to_nat(BlockBounds.current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_le(_,Support.padded(problem)) == True{} : Bool}
    Order.transitive(Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),U32.to_nat(padded),Support.padded(problem),BlockBounds.clipped_bound(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span)),padded_bound),
    end_room,BlockBounds.minimum_aligned(span,(padded - start : U32),span_aligned,BlockBounds.difference_aligned(padded,start,difference,padded_aligned,aligned)),
    BlockBounds.sum_aligned(start,span,advance,aligned,span_aligned)}

type Memory<-problem: ConvSpec.Problem,-origin: U32,-offset: U32,-positions: U32,-span: U32,-workspace: Tree.Tree<Tile.Tile8>,-previous: Tree.Tree<Tile.Tile8>,-tile_depth: Nat,-output_depth: Nat> is Data:
  Memory{tile_balanced: Shape.Balanced(Tile.Tile8,tile_depth,workspace),tile_depth_ok: {Nat.is_le(tile_depth,30n) == True{} : Bool},
    tile_room: {Nat.is_le(Nat.mul(PackingBounds.groups(U32.min(span,Window.padded(positions))),Coordinates.inner(problem)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool},
    output_balanced: Shape.Balanced(Tile.Tile8,output_depth,previous),output_depth_ok: {Nat.is_le(output_depth,30n) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(U32.to_nat((offset + positions : U32)),U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),U32.to_nat(Tree.capacity(Tile.Tile8,previous))) == True{} : Bool}}

def block_tile_room(+limits: Machine.Limits,+problem: ConvSpec.Problem,+positions: U32,+span: U32,+clipped: U32,+workspace: Tree.Tree<Tile.Tile8>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(clipped),Support.padded(problem)) == True{} : Bool},
  +clipped_bound: {Nat.is_le(U32.to_nat(clipped),U32.to_nat(U32.min(span,Window.padded(positions)))) == True{} : Bool},
  +tile_room: {Nat.is_le(Nat.mul(PackingBounds.groups(U32.min(span,Window.padded(positions))),Coordinates.inner(problem)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(PackingBounds.count(clipped,problem)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(PackingBounds.count(clipped,problem)),Nat.mul(PackingBounds.groups(clipped),Coordinates.inner(problem)),PackingBounds.count_exact(limits,problem,clipped,Guards.valid(problem,valid),Support.fast(limits,problem,supported),span_bound)) :
    {Nat.is_le(_,U32.to_nat(Tree.capacity(Tile.Tile8,workspace))) == True{} : Bool}
  Order.transitive(Nat.mul(PackingBounds.groups(clipped),Coordinates.inner(problem)),Nat.mul(PackingBounds.groups(U32.min(span,Window.padded(positions))),Coordinates.inner(problem)),U32.to_nat(Tree.capacity(Tile.Tile8,workspace)),
    RowMajor.mul_le(PackingBounds.groups(clipped),PackingBounds.groups(U32.min(span,Window.padded(positions))),Coordinates.inner(problem),DivisionOrder.quotient_le(U32.to_nat(clipped),U32.to_nat(U32.min(span,Window.padded(positions))),8n,{==},clipped_bound)),tile_room)

def memory_advanced(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+span: U32,+clipped: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+output_depth: Nat,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +span_bound: {Nat.is_le(U32.to_nat(clipped),Support.padded(problem)) == True{} : Bool},
  memory: Memory<problem,origin,offset,positions,span,workspace,previous,tile_depth,output_depth>) ->
  Memory<problem,origin,offset,positions,span,tiles(problem,storage,(origin + start : U32),clipped,workspace),output(problem,origin,offset,positions,storage,start,clipped,workspace,previous),tile_depth,output_depth>:
  Memory{+tile_balanced,+tile_depth_ok,+tile_room,+previous_balanced,+output_depth_ok,+output_room} = memory
  +next_tiles = tiles(problem,storage,(origin + start : U32),clipped,workspace)
  +next_output = output(problem,origin,offset,positions,storage,start,clipped,workspace,previous)
  +next_tiles_balanced = tiles_balanced(problem,storage,(origin + start : U32),clipped,workspace,tile_depth,tile_balanced)
  +next_output_balanced = output_balanced(problem,origin,offset,positions,storage,start,clipped,workspace,previous,output_depth,previous_balanced)
  Memory{next_tiles_balanced,tile_depth_ok,Shape.transfer_room(Tile.Tile8,tile_depth,workspace,next_tiles,Nat.mul(PackingBounds.groups(U32.min(span,Window.padded(positions))),Coordinates.inner(problem)),tile_balanced,next_tiles_balanced,tile_room),
    next_output_balanced,output_depth_ok,Shape.transfer_room(Tile.Tile8,output_depth,previous,next_output,Nat.mul(U32.to_nat((offset + positions : U32)),U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),previous_balanced,next_output_balanced,output_room)}

def Earlier(+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+channel: Nat,+row: Nat,+column: Nat,+start: U32,+previous: Tree.Tree<Tile.Tile8>) -> Data:
  Boolean.When(Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat(start)),{TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,previous,address(problem,origin,offset,channel,row,column))) == Values.matrix_value(problem,channel,row,column) : F32})

type Point<-problem: ConvSpec.Problem,-origin: U32,-positions: U32,-channel: Nat,-row: Nat,-column: Nat,-padded: U32> is Data:
  Point{restored: {Nat.add(U32.to_nat(origin),local_position(problem,origin,row,column)) == Coordinates.position(problem,row,column) : Nat},channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},
    row_inside: {Nat.is_lt(row,ConvSpec.out_height(problem)) == True{} : Bool},column_inside: {Nat.is_lt(column,ConvSpec.out_width(problem)) == True{} : Bool},
    spatial_inside: {Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat(positions)) == True{} : Bool},
    padded_inside: {Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat(padded)) == True{} : Bool}}

def point_written(+before: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +geometry: Geometry<limits,problem,origin,offset,positions,padded,span,cap>,+step: StepGeometry<problem,origin,offset,positions,start,padded,span>,
  +memory: Memory<problem,origin,offset,positions,span,workspace,previous,tile_depth,output_depth>,+point: Point<problem,origin,positions,channel,row,column,padded>,
  +inside: {Nat.is_lt(U32.to_nat(start),U32.to_nat(padded)) == True{} : Bool},
  +before_exact: {Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat(start)) == before : Bool},
  +before_next: {Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat((start + span : U32))) == True{} : Bool},
  earlier: Boolean.When(before,{TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,previous,address(problem,origin,offset,channel,row,column))) == Values.matrix_value(problem,channel,row,column) : F32})) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,output(problem,origin,offset,positions,storage,start,BlockBounds.current(start,padded,span),workspace,previous),address(problem,origin,offset,channel,row,column))) == Values.matrix_value(problem,channel,row,column) : F32}:
  match before:
    case True{}:
      Geometry{pack_room,+valid,+supported,cap_ok,padded_bound,padded_end,+outputs_room,+end_exact,offset_room,sum_room,span_positive,padded_aligned,span_aligned,+padded_value} = geometry
      StepGeometry{pack_exact,+output_exact,advance,clipped,positive,span_bound,+end_room,+aligned,next_aligned} = step
      Partition.BlockAligned{+blocks,+span_exact} = aligned
      Memory{tile_balanced,tile_depth_ok,tile_room,+output_balanced,+output_depth_ok,+output_room} = memory
      Point{+restored_point,+channel_inside,row_inside,column_inside,+spatial_inside,padded_inside} = point
      Equal.trans(F32,TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,output(problem,origin,offset,positions,storage,start,BlockBounds.current(start,padded,span),workspace,previous),address(problem,origin,offset,channel,row,column))),TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,previous,address(problem,origin,offset,channel,row,column))),Values.matrix_value(problem,channel,row,column),
        Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(Nat.mod(channel,8n),tile),
          Tree.read(Tile.Tile8,output(problem,origin,offset,positions,storage,start,BlockBounds.current(start,padded,span),workspace,previous),address(problem,origin,offset,channel,row,column)),
          Tree.read(Tile.Tile8,previous,address(problem,origin,offset,channel,row,column)),
          block_preserves_before(limits,problem,origin,offset,positions,storage,start,BlockBounds.current(start,padded,span),blocks,workspace,previous,output_depth,channel,
            Nat.add(U32.to_nat(offset),local_position(problem,origin,row,column)),valid,supported,output_balanced,output_depth_ok,output_room,
            outputs_room,output_exact,end_exact,span_exact,end_room,channel_inside,
            %Equal.sym(Nat,U32.to_nat((offset + positions : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(positions)),end_exact) :
              {Nat.is_lt(Nat.add(U32.to_nat(offset),local_position(problem,origin,row,column)),_) == True{} : Bool}
            Order.offset_lt(U32.to_nat(offset),local_position(problem,origin,row,column),U32.to_nat(positions),spatial_inside),
            %Equal.sym(Nat,U32.to_nat((offset + start : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(start)),output_exact) :
              {Nat.is_lt(Nat.add(U32.to_nat(offset),local_position(problem,origin,row,column)),_) == True{} : Bool}
            Order.offset_lt(U32.to_nat(offset),local_position(problem,origin,row,column),U32.to_nat(start),before_exact))),earlier)
    case False{}:
      Geometry{+pack_room,+valid,+supported,cap_ok,padded_bound,padded_end,+outputs_room,+end_exact,offset_room,sum_room,span_positive,padded_aligned,span_aligned,+padded_value} = geometry
      StepGeometry{+pack_exact,+output_exact,+advance,+clipped,positive,+span_bound,+end_room,+aligned,next_aligned} = step
      Partition.BlockAligned{+blocks,+span_exact} = aligned
      Memory{+tile_balanced,+tile_depth_ok,+tile_room,+output_balanced,+output_depth_ok,+output_room} = memory
      Point{+restored_point,+channel_inside,+row_inside,+column_inside,+spatial_inside,+padded_inside} = point
      +position = local_position(problem,origin,row,column)
      +lower = Order.ge_true(position,U32.to_nat(start),Order.comparison_not_less(Nat.cmp(position,U32.to_nat(start)),before_exact))
      +local = Nat.sub(position,U32.to_nat(start))
      +restored = Order.restore_shift(U32.to_nat(start),position,lower)
      %restored : {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,output(problem,origin,offset,positions,storage,start,BlockBounds.current(start,padded,span),workspace,previous),
          U32.from_nat(Nat.add(Nat.mul(Nat.add(U32.to_nat(offset),_),U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),Nat.div(channel,8n))))) == Values.matrix_value(problem,channel,row,column) : F32}
      block_readback(limits,problem,origin,offset,positions,storage,start,BlockBounds.current(start,padded,span),blocks,workspace,tile_depth,previous,output_depth,channel,local,row,column,valid,supported,tile_balanced,tile_depth_ok,
        block_tile_room(limits,problem,positions,span,BlockBounds.current(start,padded,span),workspace,valid,supported,span_bound,
          BlockBounds.current_prefix_bound(positions,start,padded,span,padded_value,clipped),tile_room),
        output_balanced,output_depth_ok,output_room,outputs_room,output_exact,end_exact,span_exact,span_bound,end_room,channel_inside,
        %Equal.sym(Nat,U32.to_nat(BlockBounds.current(start,padded,span)),Nat.min(U32.to_nat(span),Nat.sub(U32.to_nat(padded),U32.to_nat(start))),clipped) : {Nat.is_lt(local,_) == True{} : Bool}
        BlockBounds.clipped_contains(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),position,Bounds.strict_le(U32.to_nat(start),U32.to_nat(padded),inside),lower,padded_inside,
          %advance : {Nat.is_lt(position,_) == True{} : Bool} before_next),
        %Equal.sym(Nat,U32.to_nat((origin + start : U32)),Nat.add(U32.to_nat(origin),U32.to_nat(start)),pack_exact) : {Nat.add(_,local) == Coordinates.position(problem,row,column) : Nat}
        Equal.trans(Nat,Nat.add(Nat.add(U32.to_nat(origin),U32.to_nat(start)),local),Nat.add(U32.to_nat(origin),Nat.add(U32.to_nat(start),local)),Coordinates.position(problem,row,column),
          Addition.associative(U32.to_nat(origin),U32.to_nat(start),local),
          %Equal.sym(Nat,Nat.add(U32.to_nat(start),local),position,restored) : {Nat.add(U32.to_nat(origin),_) == Coordinates.position(problem,row,column) : Nat} restored_point),row_inside,column_inside,
        %Equal.sym(Nat,Nat.add(U32.to_nat(start),local),position,restored) : {Nat.is_lt(_,U32.to_nat(positions)) == True{} : Bool} spatial_inside)

def earlier_selected(+after: Bool,+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +geometry: Geometry<limits,problem,origin,offset,positions,padded,span,cap>,+step: StepGeometry<problem,origin,offset,positions,start,padded,span>,
  +memory: Memory<problem,origin,offset,positions,span,workspace,previous,tile_depth,output_depth>,+point: Point<problem,origin,positions,channel,row,column,padded>,
  +inside: {Nat.is_lt(U32.to_nat(start),U32.to_nat(padded)) == True{} : Bool},
  +after_exact: {Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat((start + span : U32))) == after : Bool},
  earlier: Earlier(problem,origin,offset,channel,row,column,start,previous)) ->
  Boolean.When(after,{TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,output(problem,origin,offset,positions,storage,start,BlockBounds.current(start,padded,span),workspace,previous),address(problem,origin,offset,channel,row,column))) == Values.matrix_value(problem,channel,row,column) : F32}):
  match after:
    case False{}: Unit{}
    case True{}:
      point_written(Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat(start)),limits,problem,origin,offset,positions,storage,start,padded,span,cap,workspace,previous,tile_depth,output_depth,channel,row,column,geometry,step,memory,point,inside,{==},after_exact,earlier)

def step_memory(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+output_depth: Nat,
  geometry: Geometry<limits,problem,origin,offset,positions,padded,span,cap>,step: StepGeometry<problem,origin,offset,positions,start,padded,span>,memory: Memory<problem,origin,offset,positions,span,workspace,previous,tile_depth,output_depth>) ->
  Memory<problem,origin,offset,positions,span,tiles(problem,storage,(origin + start : U32),BlockBounds.current(start,padded,span),workspace),output(problem,origin,offset,positions,storage,start,BlockBounds.current(start,padded,span),workspace,previous),tile_depth,output_depth>:
  Geometry{+pack_room,+valid,+supported,cap_ok,padded_bound,padded_end,outputs_room,end_exact,offset_room,sum_room,span_positive,padded_aligned,span_aligned,+padded_value} = geometry
  StepGeometry{pack_exact,output_exact,advance,clipped,positive,+span_bound,end_room,aligned,next_aligned} = step
  memory_advanced(limits,problem,origin,offset,positions,storage,start,span,BlockBounds.current(start,padded,span),workspace,previous,tile_depth,output_depth,valid,supported,span_bound,memory)

def step_schedule(+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+rest: Nat,+start: U32,+padded: U32,+span: U32,step: StepGeometry<problem,origin,offset,positions,start,padded,span>,schedule: BlockBounds.Schedule<1n+rest,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span)>) ->
  BlockBounds.Schedule<rest,U32.to_nat((start + span : U32)),U32.to_nat(padded),U32.to_nat(span)>:
  StepGeometry{+pack_exact,+output_exact,+advance,clipped,positive,span_bound,end_room,aligned,next_aligned} = step
  %Equal.sym(Nat,U32.to_nat((start + span : U32)),Nat.add(U32.to_nat(start),U32.to_nat(span)),advance) : BlockBounds.Schedule<rest,_,U32.to_nat(padded),U32.to_nat(span)>
  BlockBounds.schedule_advanced(rest,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),schedule)

def step_alignment(+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+start: U32,+padded: U32,+span: U32,step: StepGeometry<problem,origin,offset,positions,start,padded,span>) -> Partition.BlockAligned<(start + span : U32)>:
  StepGeometry{pack_exact,output_exact,advance,clipped,positive,span_bound,end_room,aligned,next_aligned} = step
  next_aligned

# The clip of a block written from an offset is the clip of its position from zero.
def clip_same(+offset: U32,+start: U32,+padded: U32,+span: U32) ->
  {U32.min(span,((offset + padded : U32) - (offset + start : U32) : U32)) == BlockBounds.current(start,padded,span) : U32}:
  Equal.cong(U32,U32,difference => U32.min(span,difference),((offset + padded : U32) - (offset + start : U32) : U32),(padded - start : U32),
    Equal.trans(U32,((offset + padded : U32) - (offset + start : U32) : U32),(((offset + padded : U32) - offset : U32) - start : U32),(padded - start : U32),
      Equal.sym(U32,(((offset + padded : U32) - offset : U32) - start : U32),((offset + padded : U32) - (offset + start : U32) : U32),Arithmetic.subtract_sum((offset + padded : U32),offset,start)),
      Equal.cong(U32,U32,value => (value - start : U32),((offset + padded : U32) - offset : U32),padded,Arithmetic.subtract_first(offset,padded))))

type LoopEvidence<-problem: ConvSpec.Problem,-origin: U32,-offset: U32,-channel: Nat,-row: Nat,-column: Nat,-output_depth: Nat,-output: Tree.Tree<Tile.Tile8>> is Data:
  LoopEvidence{balanced: Shape.Balanced(Tile.Tile8,output_depth,output),value: {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,output,address(problem,origin,offset,channel,row,column))) == Values.matrix_value(problem,channel,row,column) : F32}}

# Once the schedule is exhausted every real point is in the completed prefix.
def loop_readback(+remaining: Nat,+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +geometry: Geometry<limits,problem,origin,offset,positions,padded,span,cap>,+memory: Memory<problem,origin,offset,positions,span,workspace,previous,tile_depth,output_depth>,
  +point: Point<problem,origin,positions,channel,row,column,padded>,+schedule: BlockBounds.Schedule<remaining,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span)>,
  +aligned: Partition.BlockAligned<start>,earlier: Earlier(problem,origin,offset,channel,row,column,start,previous)) ->
  LoopEvidence<problem,origin,offset,channel,row,column,output_depth,Model.result(Model.blocks_window(remaining,work(problem,(offset + positions : U32)),(origin + start : U32),(offset + start : U32),(offset + padded : U32),span,state(problem,storage,workspace,previous)))>:
  match remaining:
    case 0n:
      Memory{tile_balanced,tile_depth_ok,tile_room,+output_balanced,output_depth_ok,output_room} = memory
      Point{restored_point,channel_inside,row_inside,column_inside,spatial_inside,+padded_inside} = point
      LoopEvidence{output_balanced,
      %Bounds.nat_trans(local_position(problem,origin,row,column),U32.to_nat(padded),U32.to_nat(start),padded_inside,BlockBounds.schedule_finished(U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),schedule)) :
        Boolean.When(_,{TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,previous,address(problem,origin,offset,channel,row,column))) == Values.matrix_value(problem,channel,row,column) : F32})
      earlier}
    case 1n+ +rest:
      +inside = BlockBounds.schedule_inside(rest,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),schedule)
      +step = step_geometry(limits,problem,origin,offset,positions,start,padded,span,cap,geometry,inside,aligned)
      +clip = BlockBounds.current(start,padded,span)
      +task = work(problem,(offset + positions : U32))
      +next_tiles = tiles(problem,storage,(origin + start : U32),clip,workspace)
      +next_output = output(problem,origin,offset,positions,storage,start,clip,workspace,previous)
      %Equal.sym(U32,U32.min(span,((offset + padded : U32) - (offset + start : U32) : U32)),clip,clip_same(offset,start,padded,span)) :
        LoopEvidence<problem,origin,offset,channel,row,column,output_depth,Model.result(Model.blocks_window(rest,task,((origin + start : U32) + span : U32),((offset + start : U32) + span : U32),(offset + padded : U32),span,
          Model.step_window(task,(origin + start : U32),(offset + start : U32),_,state(problem,storage,workspace,previous))))>
      %Equal.sym(Model.State,Model.step_window(task,(origin + start : U32),(offset + start : U32),clip,state(problem,storage,workspace,previous)),state(problem,storage,next_tiles,next_output),
        step_state(problem,origin,offset,positions,storage,start,clip,workspace,previous)) :
        LoopEvidence<problem,origin,offset,channel,row,column,output_depth,Model.result(Model.blocks_window(rest,task,((origin + start : U32) + span : U32),((offset + start : U32) + span : U32),(offset + padded : U32),span,_))>
      %Equal.sym(U32,((origin + start : U32) + span : U32),(origin + (start + span : U32) : U32),Arithmetic.add_associative(origin,start,span)) :
        LoopEvidence<problem,origin,offset,channel,row,column,output_depth,Model.result(Model.blocks_window(rest,task,_,((offset + start : U32) + span : U32),(offset + padded : U32),span,state(problem,storage,next_tiles,next_output)))>
      %Equal.sym(U32,((offset + start : U32) + span : U32),(offset + (start + span : U32) : U32),Arithmetic.add_associative(offset,start,span)) :
        LoopEvidence<problem,origin,offset,channel,row,column,output_depth,Model.result(Model.blocks_window(rest,task,(origin + (start + span : U32) : U32),_,(offset + padded : U32),span,state(problem,storage,next_tiles,next_output)))>
      loop_readback(rest,limits,problem,origin,offset,positions,storage,(start + span : U32),padded,span,cap,next_tiles,next_output,tile_depth,output_depth,channel,row,column,geometry,
        step_memory(limits,problem,origin,offset,positions,storage,start,padded,span,cap,workspace,previous,tile_depth,output_depth,geometry,step,memory),point,
        step_schedule(problem,origin,offset,positions,rest,start,padded,span,step,schedule),step_alignment(problem,origin,offset,positions,start,padded,span,step),
        earlier_selected(Nat.is_lt(local_position(problem,origin,row,column),U32.to_nat((start + span : U32))),limits,problem,origin,offset,positions,storage,start,padded,span,cap,workspace,previous,tile_depth,output_depth,channel,row,column,geometry,step,memory,point,inside,{==},earlier))

# The cell of a channel at an output position below the offset of a call.
def Keeps(+problem: ConvSpec.Problem,+channel: Nat,+at: Nat,+previous: Tree.Tree<Tile.Tile8>,output: Tree.Tree<Tile.Tile8>) -> Data:
  {Tree.read(Tile.Tile8,output,cell_address(problem,channel,at)) == Tree.read(Tile.Tile8,previous,cell_address(problem,channel,at)) : Tile.Tile8}

def step_keeps(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+at: Nat,
  +geometry: Geometry<limits,problem,origin,offset,positions,padded,span,cap>,+step: StepGeometry<problem,origin,offset,positions,start,padded,span>,
  +memory: Memory<problem,origin,offset,positions,span,workspace,previous,tile_depth,output_depth>,
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+below: {Nat.is_lt(at,U32.to_nat(offset)) == True{} : Bool}) ->
  Keeps(problem,channel,at,previous,output(problem,origin,offset,positions,storage,start,BlockBounds.current(start,padded,span),workspace,previous)):
  Geometry{pack_room,+valid,+supported,cap_ok,padded_bound,padded_end,+outputs_room,+end_exact,offset_room,sum_room,span_positive,padded_aligned,span_aligned,padded_value} = geometry
  StepGeometry{pack_exact,+output_exact,advance,clipped,positive,span_bound,+end_room,+aligned,next_aligned} = step
  Partition.BlockAligned{+blocks,+span_exact} = aligned
  Memory{tile_balanced,tile_depth_ok,tile_room,+output_balanced,+output_depth_ok,+output_room} = memory
  +at_inside = {%Equal.sym(Nat,U32.to_nat((offset + positions : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(positions)),end_exact) : {Nat.is_lt(at,_) == True{} : Bool}
    Bounds.nat_trans(at,U32.to_nat(offset),Nat.add(U32.to_nat(offset),U32.to_nat(positions)),below,Order.le_add(U32.to_nat(offset),U32.to_nat(positions))) :
    {Nat.is_lt(at,U32.to_nat((offset + positions : U32))) == True{} : Bool}}
  +before = {%Equal.sym(Nat,U32.to_nat((offset + start : U32)),Nat.add(U32.to_nat(offset),U32.to_nat(start)),output_exact) : {Nat.is_lt(at,_) == True{} : Bool}
    Bounds.nat_trans(at,U32.to_nat(offset),Nat.add(U32.to_nat(offset),U32.to_nat(start)),below,Order.le_add(U32.to_nat(offset),U32.to_nat(start))) :
    {Nat.is_lt(at,U32.to_nat((offset + start : U32))) == True{} : Bool}}
  block_preserves_before(limits,problem,origin,offset,positions,storage,start,BlockBounds.current(start,padded,span),blocks,workspace,previous,output_depth,channel,at,valid,supported,
    output_balanced,output_depth_ok,output_room,outputs_room,output_exact,end_exact,span_exact,end_room,channel_inside,at_inside,before)

# The blocks of a call keep the cells of the positions below its offset.
def loop_keeps(+remaining: Nat,+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+at: Nat,
  +geometry: Geometry<limits,problem,origin,offset,positions,padded,span,cap>,+memory: Memory<problem,origin,offset,positions,span,workspace,previous,tile_depth,output_depth>,
  +schedule: BlockBounds.Schedule<remaining,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span)>,+aligned: Partition.BlockAligned<start>,
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+below: {Nat.is_lt(at,U32.to_nat(offset)) == True{} : Bool}) ->
  Keeps(problem,channel,at,previous,Model.result(Model.blocks_window(remaining,work(problem,(offset + positions : U32)),(origin + start : U32),(offset + start : U32),(offset + padded : U32),span,state(problem,storage,workspace,previous)))):
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +inside = BlockBounds.schedule_inside(rest,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),schedule)
      +step = step_geometry(limits,problem,origin,offset,positions,start,padded,span,cap,geometry,inside,aligned)
      +clip = BlockBounds.current(start,padded,span)
      +task = work(problem,(offset + positions : U32))
      +next_tiles = tiles(problem,storage,(origin + start : U32),clip,workspace)
      +next_output = output(problem,origin,offset,positions,storage,start,clip,workspace,previous)
      %Equal.sym(U32,U32.min(span,((offset + padded : U32) - (offset + start : U32) : U32)),clip,clip_same(offset,start,padded,span)) :
        Keeps(problem,channel,at,previous,Model.result(Model.blocks_window(rest,task,((origin + start : U32) + span : U32),((offset + start : U32) + span : U32),(offset + padded : U32),span,
          Model.step_window(task,(origin + start : U32),(offset + start : U32),_,state(problem,storage,workspace,previous)))))
      %Equal.sym(Model.State,Model.step_window(task,(origin + start : U32),(offset + start : U32),clip,state(problem,storage,workspace,previous)),state(problem,storage,next_tiles,next_output),
        step_state(problem,origin,offset,positions,storage,start,clip,workspace,previous)) :
        Keeps(problem,channel,at,previous,Model.result(Model.blocks_window(rest,task,((origin + start : U32) + span : U32),((offset + start : U32) + span : U32),(offset + padded : U32),span,_)))
      %Equal.sym(U32,((origin + start : U32) + span : U32),(origin + (start + span : U32) : U32),Arithmetic.add_associative(origin,start,span)) :
        Keeps(problem,channel,at,previous,Model.result(Model.blocks_window(rest,task,_,((offset + start : U32) + span : U32),(offset + padded : U32),span,state(problem,storage,next_tiles,next_output))))
      %Equal.sym(U32,((offset + start : U32) + span : U32),(offset + (start + span : U32) : U32),Arithmetic.add_associative(offset,start,span)) :
        Keeps(problem,channel,at,previous,Model.result(Model.blocks_window(rest,task,(origin + (start + span : U32) : U32),_,(offset + padded : U32),span,state(problem,storage,next_tiles,next_output))))
      +final = Model.result(Model.blocks_window(rest,task,(origin + (start + span : U32) : U32),(offset + (start + span : U32) : U32),(offset + padded : U32),span,state(problem,storage,next_tiles,next_output)))
      Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,final,cell_address(problem,channel,at)),Tree.read(Tile.Tile8,next_output,cell_address(problem,channel,at)),
        Tree.read(Tile.Tile8,previous,cell_address(problem,channel,at)),
        loop_keeps(rest,limits,problem,origin,offset,positions,storage,(start + span : U32),padded,span,cap,next_tiles,next_output,tile_depth,output_depth,channel,at,geometry,
          step_memory(limits,problem,origin,offset,positions,storage,start,padded,span,cap,workspace,previous,tile_depth,output_depth,geometry,step,memory),
          step_schedule(problem,origin,offset,positions,rest,start,padded,span,step,schedule),step_alignment(problem,origin,offset,positions,start,padded,span,step),channel_inside,below),
        step_keeps(limits,problem,origin,offset,positions,storage,start,padded,span,cap,workspace,previous,tile_depth,output_depth,channel,at,geometry,step,memory,channel_inside,below))

# The workspace and the output a loop leaves keep their shapes.
type Balances<-tile_depth: Nat,-output_depth: Nat,-tiles: Tree.Tree<Tile.Tile8>,-output: Tree.Tree<Tile.Tile8>> is Data:
  Balances{tiles: Shape.Balanced(Tile.Tile8,tile_depth,tiles),output: Shape.Balanced(Tile.Tile8,output_depth,output)}

def loop_balances(+remaining: Nat,+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+start: U32,+padded: U32,+span: U32,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+output_depth: Nat,
  +geometry: Geometry<limits,problem,origin,offset,positions,padded,span,cap>,+memory: Memory<problem,origin,offset,positions,span,workspace,previous,tile_depth,output_depth>,
  +schedule: BlockBounds.Schedule<remaining,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span)>,+aligned: Partition.BlockAligned<start>) ->
  Balances<tile_depth,output_depth,Model.workspace(Model.blocks_window(remaining,work(problem,(offset + positions : U32)),(origin + start : U32),(offset + start : U32),(offset + padded : U32),span,state(problem,storage,workspace,previous))),Model.result(Model.blocks_window(remaining,work(problem,(offset + positions : U32)),(origin + start : U32),(offset + start : U32),(offset + padded : U32),span,state(problem,storage,workspace,previous)))>:
  match remaining:
    case 0n:
      Memory{+tile_balanced,tile_depth_ok,tile_room,+output_balanced,output_depth_ok,output_room} = memory
      Balances{tile_balanced,output_balanced}
    case 1n+ +rest:
      +inside = BlockBounds.schedule_inside(rest,U32.to_nat(start),U32.to_nat(padded),U32.to_nat(span),schedule)
      +step = step_geometry(limits,problem,origin,offset,positions,start,padded,span,cap,geometry,inside,aligned)
      +clip = BlockBounds.current(start,padded,span)
      +task = work(problem,(offset + positions : U32))
      +next_tiles = tiles(problem,storage,(origin + start : U32),clip,workspace)
      +next_output = output(problem,origin,offset,positions,storage,start,clip,workspace,previous)
      %Equal.sym(U32,U32.min(span,((offset + padded : U32) - (offset + start : U32) : U32)),clip,clip_same(offset,start,padded,span)) :
        Balances<tile_depth,output_depth,Model.workspace(Model.blocks_window(rest,task,((origin + start : U32) + span : U32),((offset + start : U32) + span : U32),(offset + padded : U32),span,Model.step_window(task,(origin + start : U32),(offset + start : U32),_,state(problem,storage,workspace,previous)))),Model.result(Model.blocks_window(rest,task,((origin + start : U32) + span : U32),((offset + start : U32) + span : U32),(offset + padded : U32),span,Model.step_window(task,(origin + start : U32),(offset + start : U32),_,state(problem,storage,workspace,previous))))>
      %Equal.sym(Model.State,Model.step_window(task,(origin + start : U32),(offset + start : U32),clip,state(problem,storage,workspace,previous)),state(problem,storage,next_tiles,next_output),
        step_state(problem,origin,offset,positions,storage,start,clip,workspace,previous)) :
        Balances<tile_depth,output_depth,Model.workspace(Model.blocks_window(rest,task,((origin + start : U32) + span : U32),((offset + start : U32) + span : U32),(offset + padded : U32),span,_)),Model.result(Model.blocks_window(rest,task,((origin + start : U32) + span : U32),((offset + start : U32) + span : U32),(offset + padded : U32),span,_))>
      %Equal.sym(U32,((origin + start : U32) + span : U32),(origin + (start + span : U32) : U32),Arithmetic.add_associative(origin,start,span)) :
        Balances<tile_depth,output_depth,Model.workspace(Model.blocks_window(rest,task,_,((offset + start : U32) + span : U32),(offset + padded : U32),span,state(problem,storage,next_tiles,next_output))),Model.result(Model.blocks_window(rest,task,_,((offset + start : U32) + span : U32),(offset + padded : U32),span,state(problem,storage,next_tiles,next_output)))>
      %Equal.sym(U32,((offset + start : U32) + span : U32),(offset + (start + span : U32) : U32),Arithmetic.add_associative(offset,start,span)) :
        Balances<tile_depth,output_depth,Model.workspace(Model.blocks_window(rest,task,(origin + (start + span : U32) : U32),_,(offset + padded : U32),span,state(problem,storage,next_tiles,next_output))),Model.result(Model.blocks_window(rest,task,(origin + (start + span : U32) : U32),_,(offset + padded : U32),span,state(problem,storage,next_tiles,next_output)))>
      loop_balances(rest,limits,problem,origin,offset,positions,storage,(start + span : U32),padded,span,cap,next_tiles,next_output,tile_depth,output_depth,geometry,
        step_memory(limits,problem,origin,offset,positions,storage,start,padded,span,cap,workspace,previous,tile_depth,output_depth,geometry,step,memory),
        step_schedule(problem,origin,offset,positions,rest,start,padded,span,step,schedule),step_alignment(problem,origin,offset,positions,start,padded,span,step))

# The blocks of a leaf on values, written from the offset on, store the matrix
# value of every point of the leaf.
def into_readback(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +geometry: Geometry<limits,problem,origin,offset,positions,Window.padded(positions),Pipeline.block_positions(Coordinates.native_inner(problem)),cap>,
  +memory: Memory<problem,origin,offset,positions,Pipeline.block_positions(Coordinates.native_inner(problem)),workspace,previous,tile_depth,output_depth>,
  +point: Point<problem,origin,positions,channel,row,column,Window.padded(positions)>,
  +schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(positions) + Pipeline.block_positions(Coordinates.native_inner(problem)) - 1) / Pipeline.block_positions(Coordinates.native_inner(problem)) : U32)),0n,U32.to_nat(Window.padded(positions)),U32.to_nat(Pipeline.block_positions(Coordinates.native_inner(problem)))>) ->
  LoopEvidence<problem,origin,offset,channel,row,column,output_depth,Model.result(Model.values_into(work(problem,(offset + positions : U32)),origin,offset,state(problem,storage,workspace,previous)))>:
  +span = Pipeline.block_positions(Coordinates.native_inner(problem))
  +padded = Window.padded(positions)
  +task = work(problem,(offset + positions : U32))
  +initial = state(problem,storage,workspace,previous)
  %Equal.sym(U32,((offset + positions : U32) - offset : U32),positions,Arithmetic.subtract_first(offset,positions)) :
    LoopEvidence<problem,origin,offset,channel,row,column,output_depth,Model.result(Model.blocks_window(U32.to_nat((((offset + Window.padded(_) : U32) - offset + span - 1) / span : U32)),task,origin,offset,(offset + Window.padded(_) : U32),span,initial))>
  %Equal.sym(U32,((offset + padded : U32) - offset : U32),padded,Arithmetic.subtract_first(offset,padded)) :
    LoopEvidence<problem,origin,offset,channel,row,column,output_depth,Model.result(Model.blocks_window(U32.to_nat(((_ + span - 1) / span : U32)),task,origin,offset,(offset + padded : U32),span,initial))>
  %Arithmetic.add_zero(origin) : LoopEvidence<problem,origin,offset,channel,row,column,output_depth,Model.result(Model.blocks_window(U32.to_nat(((padded + span - 1) / span : U32)),task,_,offset,(offset + padded : U32),span,initial))>
  %Arithmetic.add_zero(offset) : LoopEvidence<problem,origin,offset,channel,row,column,output_depth,Model.result(Model.blocks_window(U32.to_nat(((padded + span - 1) / span : U32)),task,(origin + 0 : U32),_,(offset + padded : U32),span,initial))>
  loop_readback(U32.to_nat(((Window.padded(positions) + Pipeline.block_positions(Coordinates.native_inner(problem)) - 1) / Pipeline.block_positions(Coordinates.native_inner(problem)) : U32)),limits,problem,origin,offset,positions,storage,0,padded,span,cap,workspace,previous,tile_depth,output_depth,channel,row,column,geometry,memory,point,schedule,
    Partition.BlockAligned{0n,{==}},
    %Equal.sym(Bool,Nat.is_lt(local_position(problem,origin,row,column),0n),False{},Division.lt_zero(local_position(problem,origin,row,column))) :
      Boolean.When(_,{TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,previous,address(problem,origin,offset,channel,row,column))) == Values.matrix_value(problem,channel,row,column) : F32})
    Unit{})

# A call keeps the cells of the positions below its offset.
def into_keeps(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+output_depth: Nat,+channel: Nat,+at: Nat,
  +geometry: Geometry<limits,problem,origin,offset,positions,Window.padded(positions),Pipeline.block_positions(Coordinates.native_inner(problem)),cap>,
  +memory: Memory<problem,origin,offset,positions,Pipeline.block_positions(Coordinates.native_inner(problem)),workspace,previous,tile_depth,output_depth>,
  +schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(positions) + Pipeline.block_positions(Coordinates.native_inner(problem)) - 1) / Pipeline.block_positions(Coordinates.native_inner(problem)) : U32)),0n,U32.to_nat(Window.padded(positions)),U32.to_nat(Pipeline.block_positions(Coordinates.native_inner(problem)))>,
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool},+below: {Nat.is_lt(at,U32.to_nat(offset)) == True{} : Bool}) ->
  Keeps(problem,channel,at,previous,Model.result(Model.values_into(work(problem,(offset + positions : U32)),origin,offset,state(problem,storage,workspace,previous)))):
  +span = Pipeline.block_positions(Coordinates.native_inner(problem))
  +padded = Window.padded(positions)
  +task = work(problem,(offset + positions : U32))
  +initial = state(problem,storage,workspace,previous)
  %Equal.sym(U32,((offset + positions : U32) - offset : U32),positions,Arithmetic.subtract_first(offset,positions)) :
    Keeps(problem,channel,at,previous,Model.result(Model.blocks_window(U32.to_nat((((offset + Window.padded(_) : U32) - offset + span - 1) / span : U32)),task,origin,offset,(offset + Window.padded(_) : U32),span,initial)))
  %Equal.sym(U32,((offset + padded : U32) - offset : U32),padded,Arithmetic.subtract_first(offset,padded)) :
    Keeps(problem,channel,at,previous,Model.result(Model.blocks_window(U32.to_nat(((_ + span - 1) / span : U32)),task,origin,offset,(offset + padded : U32),span,initial)))
  %Arithmetic.add_zero(origin) : Keeps(problem,channel,at,previous,Model.result(Model.blocks_window(U32.to_nat(((padded + span - 1) / span : U32)),task,_,offset,(offset + padded : U32),span,initial)))
  %Arithmetic.add_zero(offset) : Keeps(problem,channel,at,previous,Model.result(Model.blocks_window(U32.to_nat(((padded + span - 1) / span : U32)),task,(origin + 0 : U32),_,(offset + padded : U32),span,initial)))
  loop_keeps(U32.to_nat(((padded + span - 1) / span : U32)),limits,problem,origin,offset,positions,storage,0,padded,span,cap,workspace,previous,tile_depth,output_depth,channel,at,
    geometry,memory,schedule,Partition.BlockAligned{0n,{==}},channel_inside,below)

# The workspace and the output of a call keep their shapes.
def into_balances(+limits: Machine.Limits,+problem: ConvSpec.Problem,+origin: U32,+offset: U32,+positions: U32,+storage: Inputs.Contract<problem>,+cap: U32,
  +workspace: Tree.Tree<Tile.Tile8>,+previous: Tree.Tree<Tile.Tile8>,+tile_depth: Nat,+output_depth: Nat,
  +geometry: Geometry<limits,problem,origin,offset,positions,Window.padded(positions),Pipeline.block_positions(Coordinates.native_inner(problem)),cap>,
  +memory: Memory<problem,origin,offset,positions,Pipeline.block_positions(Coordinates.native_inner(problem)),workspace,previous,tile_depth,output_depth>,
  +schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(positions) + Pipeline.block_positions(Coordinates.native_inner(problem)) - 1) / Pipeline.block_positions(Coordinates.native_inner(problem)) : U32)),0n,U32.to_nat(Window.padded(positions)),U32.to_nat(Pipeline.block_positions(Coordinates.native_inner(problem)))>) ->
  Balances<tile_depth,output_depth,Model.workspace(Model.values_into(work(problem,(offset + positions : U32)),origin,offset,state(problem,storage,workspace,previous))),Model.result(Model.values_into(work(problem,(offset + positions : U32)),origin,offset,state(problem,storage,workspace,previous)))>:
  +span = Pipeline.block_positions(Coordinates.native_inner(problem))
  +padded = Window.padded(positions)
  +task = work(problem,(offset + positions : U32))
  +initial = state(problem,storage,workspace,previous)
  %Equal.sym(U32,((offset + positions : U32) - offset : U32),positions,Arithmetic.subtract_first(offset,positions)) :
    Balances<tile_depth,output_depth,Model.workspace(Model.blocks_window(U32.to_nat((((offset + Window.padded(_) : U32) - offset + span - 1) / span : U32)),task,origin,offset,(offset + Window.padded(_) : U32),span,initial)),Model.result(Model.blocks_window(U32.to_nat((((offset + Window.padded(_) : U32) - offset + span - 1) / span : U32)),task,origin,offset,(offset + Window.padded(_) : U32),span,initial))>
  %Equal.sym(U32,((offset + padded : U32) - offset : U32),padded,Arithmetic.subtract_first(offset,padded)) :
    Balances<tile_depth,output_depth,Model.workspace(Model.blocks_window(U32.to_nat(((_ + span - 1) / span : U32)),task,origin,offset,(offset + padded : U32),span,initial)),Model.result(Model.blocks_window(U32.to_nat(((_ + span - 1) / span : U32)),task,origin,offset,(offset + padded : U32),span,initial))>
  %Arithmetic.add_zero(origin) : Balances<tile_depth,output_depth,Model.workspace(Model.blocks_window(U32.to_nat(((padded + span - 1) / span : U32)),task,_,offset,(offset + padded : U32),span,initial)),Model.result(Model.blocks_window(U32.to_nat(((padded + span - 1) / span : U32)),task,_,offset,(offset + padded : U32),span,initial))>
  %Arithmetic.add_zero(offset) : Balances<tile_depth,output_depth,Model.workspace(Model.blocks_window(U32.to_nat(((padded + span - 1) / span : U32)),task,(origin + 0 : U32),_,(offset + padded : U32),span,initial)),Model.result(Model.blocks_window(U32.to_nat(((padded + span - 1) / span : U32)),task,(origin + 0 : U32),_,(offset + padded : U32),span,initial))>
  loop_balances(U32.to_nat(((padded + span - 1) / span : U32)),limits,problem,origin,offset,positions,storage,0,padded,span,cap,workspace,previous,tile_depth,output_depth,
    geometry,memory,schedule,Partition.BlockAligned{0n,{==}})
