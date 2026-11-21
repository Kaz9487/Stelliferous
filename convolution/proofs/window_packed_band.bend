# The band of a call of the packed window kernel: the rows of an image it
# holds are the input of their window problem, and the call writes the
# matrix values of that problem.
import Base
import ../specification.bend as ConvSpec
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/kernel_pipeline_model.bend as Model
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ./convolution_guard_facts.bend as Guards
import ./reduction_coordinate_bounds.bend as Coordinates
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./window_samples.bend as Samples
import ./blocked_convolution_readback.bend as Blocked
import ../../lib/kernel_tile.bend as Tile
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/tensor_move.bend as Move
import ../../lib/proofs/tensor_move_readback.bend as Held
import ./input_storage.bend as Inputs
import ./window_cell_contents.bend as Cells
import ../../lib/proofs/kernel_pipeline_buffer_shape.bend as BufferShape
import ../../lib/kernel_window_pipeline.bend as WindowPipeline
import ../../lib/math_activation.bend as Activations
import ../../lib/storage_buffer.bend as Storage
import ../../lib/kernel_window.bend as Window
import ../../lib/proofs/kernel_pipeline_bounds.bend as BlockBounds
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../inference.bend as NativeConv
import ./window_cell_values.bend as CellValues
import ./window_problem.bend as WindowProblem
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinates
import ./window_skeleton.bend as Skeleton
import ../../lib/proofs/kernel_spatial_packing_pixels.bend as Pixels
import ../../lib/proofs/kernel_spatial_packing_samples.bend as SpatialSamples
import ./window_call_ready.bend as Ready

# The work of a call on a band, read from zero, is the work of the window
# problem of its rows.
def band_work(+problem: ConvSpec.Problem,+base: U32,+first: Nat,+rows: U32,+end: U32) ->
  {Pipeline.WindowWork{MicroLeaf.rows(problem),end,SpatialSamples.plain(Ready.band_window(problem,base,rows))} ==
    Blocked.work(Samples.window(problem,first,U32.to_nat(rows)),end) : Pipeline.WindowWork}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input,weights,bias} : ConvSpec.Problem}
  +k = U32.from_nat(kernel)
  +p = U32.from_nat(padding)
  +inner = Coordinates.native_inner(problem)
  %Equal.sym(U32,U32.from_nat(U32.to_nat(rows)),rows,Arithmetic.native_roundtrip(rows)) :
    {Pipeline.WindowWork{MicroLeaf.rows(problem),end,SpatialSamples.plain(Ready.band_window(problem,base,rows))} ==
      Pipeline.WindowWork{U32.from_nat(co),end,Packing.Stencil{0,_,U32.from_nat(width),k,k,U32.from_nat(stride),(0 - p : U32),(0 - p : U32),
        U32.from_nat(ConvSpec.axis_size(width,kernel,stride,padding)),inner,Gemm.blocks((inner / (k * k) : U32))}} : Pipeline.WindowWork}
  {==}

# A cell of a band read from the cells of an image starting at start is the
# cell of the image it holds.
def band_at(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+from: U32,+start: Nat,+row_first: Nat,+index: Nat,
  +from_exact: {U32.to_nat(from) == Nat.add(start,Cells.count(problem,row_first)) : Nat},
  cell: {Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band),U32.from_nat(Nat.add(U32.to_nat(base),index))) == value(holder,U32.from_nat(Nat.add(U32.to_nat(from),index))) : Tile.Tile8},
  +image: {value(holder,U32.from_nat(Nat.add(start,Cells.source(problem,row_first,index)))) ==
    Tree.read(Tile.Tile8,Cells.input_cells(problem,storage),U32.from_nat(Cells.source(problem,row_first,index))) : Tile.Tile8}) ->
  Cells.At(problem,Cells.input_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),U32.to_nat(base),row_first,index):
  +source = Cells.source(problem,row_first,index)
  +moved = Equal.cong(Nat,U32,position => U32.from_nat(position),Nat.add(U32.to_nat(from),index),Nat.add(start,source),
    Equal.trans(Nat,Nat.add(U32.to_nat(from),index),Nat.add(Nat.add(start,Cells.count(problem,row_first)),index),Nat.add(start,source),
      Equal.cong(Nat,Nat,position => Nat.add(position,index),U32.to_nat(from),Nat.add(start,Cells.count(problem,row_first)),from_exact),
      Addition.associative(start,Cells.count(problem,row_first),index)))
  Equal.trans(Tile.Tile8,Tree.read(Tile.Tile8,Buffer.tree(~Tile.Tile8,band),U32.from_nat(Nat.add(U32.to_nat(base),index))),
    value(holder,U32.from_nat(Nat.add(U32.to_nat(from),index))),Tree.read(Tile.Tile8,Cells.input_cells(problem,storage),U32.from_nat(source)),
    cell,
    Equal.trans(Tile.Tile8,value(holder,U32.from_nat(Nat.add(U32.to_nat(from),index))),value(holder,U32.from_nat(Nat.add(start,source))),
      Tree.read(Tile.Tile8,Cells.input_cells(problem,storage),U32.from_nat(source)),
      Equal.cong(U32,Tile.Tile8,word => value(holder,word),U32.from_nat(Nat.add(U32.to_nat(from),index)),U32.from_nat(Nat.add(start,source)),moved),
      image))

# The image cells a holder keeps from start on.
def HeldImage(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+start: Nat) -> Data:
  Members.All(Nat,{cell => {value(holder,U32.from_nat(Nat.add(start,cell))) == Tree.read(Tile.Tile8,Cells.input_cells(problem,storage),U32.from_nat(cell)) : Tile.Tile8} : Nat -> Data},
    Ranges.range(Cells.count(problem,Guards.height(problem)),0n))

def image_at(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+start: Nat,+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  +cell: Nat,+inside: {Nat.is_lt(cell,Cells.count(problem,Guards.height(problem))) == True{} : Bool}) ->
  {value(holder,U32.from_nat(Nat.add(start,cell))) == Tree.read(Tile.Tile8,Cells.input_cells(problem,storage),U32.from_nat(cell)) : Tile.Tile8}:
  Members.all_member(Nat,{position => {value(holder,U32.from_nat(Nat.add(start,position))) == Tree.read(Tile.Tile8,Cells.input_cells(problem,storage),U32.from_nat(position)) : Tile.Tile8} : Nat -> Data},
    cell,Ranges.range(Cells.count(problem,Guards.height(problem)),0n),image,Members.range_member(Cells.count(problem,Guards.height(problem)),0n,cell,inside))

# The cells of consecutive rows add up.
def count_sum(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat) ->
  {Cells.count(problem,Nat.add(first,rows)) == Nat.add(Cells.count(problem,first),Cells.count(problem,rows)) : Nat}:
  Equal.trans(Nat,Cells.count(problem,Nat.add(first,rows)),
    Nat.mul(Nat.add(Nat.mul(first,Guards.width(problem)),Nat.mul(rows,Guards.width(problem))),Cells.group_count(problem)),
    Nat.add(Cells.count(problem,first),Cells.count(problem,rows)),
    Equal.cong(Nat,Nat,area => Nat.mul(area,Cells.group_count(problem)),Nat.mul(Nat.add(first,rows),Guards.width(problem)),
      Nat.add(Nat.mul(first,Guards.width(problem)),Nat.mul(rows,Guards.width(problem))),RowMajor.distrib(first,rows,Guards.width(problem))),
    RowMajor.distrib(Nat.mul(first,Guards.width(problem)),Nat.mul(rows,Guards.width(problem)),Cells.group_count(problem)))

# A cell of rows rows from row first lies in the image.
def source_inside(+problem: ConvSpec.Problem,+first: Nat,+rows: Nat,+rows_fit: {Nat.is_le(Nat.add(first,rows),Guards.height(problem)) == True{} : Bool},
  +index: Nat,+less: {Nat.is_lt(index,Cells.count(problem,rows)) == True{} : Bool}) ->
  {Nat.is_lt(Cells.source(problem,first,index),Cells.count(problem,Guards.height(problem))) == True{} : Bool}:
  +area = RowMajor.mul_le(Nat.add(first,rows),Guards.height(problem),Guards.width(problem),rows_fit)
  +fit = RowMajor.mul_le(Nat.mul(Nat.add(first,rows),Guards.width(problem)),Nat.mul(Guards.height(problem),Guards.width(problem)),Cells.group_count(problem),area)
  +sum_fit = {%count_sum(problem,first,rows) : {Nat.is_le(_,Cells.count(problem,Guards.height(problem))) == True{} : Bool}
    fit : {Nat.is_le(Nat.add(Cells.count(problem,first),Cells.count(problem,rows)),Cells.count(problem,Guards.height(problem))) == True{} : Bool}}
  Bounds.nat_trans(Cells.source(problem,first,index),Nat.add(Cells.count(problem,first),Cells.count(problem,rows)),Cells.count(problem,Guards.height(problem)),
    Order.offset_lt(Cells.count(problem,first),index,Cells.count(problem,rows),less),sum_fit)

# A cell of the band from first holds its image cell.
def band_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: Nat,+rows: Nat,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,row_first)) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(row_first,rows),Guards.height(problem)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,rows),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  +index: Nat,+less: {Nat.is_lt(index,Cells.count(problem,rows)) == True{} : Bool}) ->
  Cells.At(problem,Cells.input_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),U32.to_nat(base),row_first,index):
  band_at(~Holder,~value,holder,problem,storage,band,base,first,start,row_first,index,from_exact,
    Members.all_member(Nat,{cell => Skeleton.BandRead(~Holder,~value,holder,band,base,first,cell) : Nat -> Data},index,Ranges.range(U32.to_nat(count),0n),reads,
      Members.range_member(U32.to_nat(count),0n,index,Bounds.nat_trans(index,Cells.count(problem,rows),U32.to_nat(count),less,cells_fit))),
    image_at(~Holder,~value,holder,problem,storage,start,image,Cells.source(problem,row_first,index),source_inside(problem,row_first,rows,rows_fit,index,less)))

def band_cells(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: Nat,+rows: Nat,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,row_first)) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(row_first,rows),Guards.height(problem)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,rows),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  remaining: Nat,+position: Nat,+room: {Nat.is_le(Nat.add(position,remaining),Cells.count(problem,rows)) == True{} : Bool}) ->
  Members.All(Nat,{index => Cells.At(problem,Cells.input_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),U32.to_nat(base),row_first,index) : Nat -> Data},
    Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      KernelGemm.Both{band_cell(~Holder,~value,holder,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,cells_fit,image,position,Order.room_index(position,rest,Cells.count(problem,rows),room)),
        band_cells(~Holder,~value,holder,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,cells_fit,image,rest,1n+position,Order.advance_room(position,rest,Cells.count(problem,rows),room))}

# The band from first holds rows rows of the image from row row_first.
def band_contents(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: Nat,+rows: Nat,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,row_first)) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(row_first,rows),Guards.height(problem)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,rows),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start)) ->
  Cells.Contents(problem,Cells.input_cells(problem,storage),Buffer.tree(~Tile.Tile8,band),U32.to_nat(base),row_first,rows):
  band_cells(~Holder,~value,holder,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,cells_fit,image,Cells.count(problem,rows),0n,Order.reflexive(Cells.count(problem,rows)))

# The rows of the image a band holds, read as the values of a window.
def band_context(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: U32,+rows: U32,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,U32.to_nat(row_first))) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(row_first),U32.to_nat(rows)),Guards.height(problem)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(rows)),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> CellValues.Context:
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +width = U32.from_nat(Guards.width(problem))
  +kernel = U32.from_nat(Guards.kernel(problem))
  +channels = (Coordinates.native_inner(problem) / (kernel * kernel) : U32)
  CellValues.Context{limits,problem,storage,Buffer.tree(~Tile.Tile8,band),base,row_first,rows,channels,width,valid,supported,
    Coordinates.inner_channels(limits,problem,Guards.valid(problem,valid),support),SupportedCoordinates.width_exact(limits,problem,support),rows_fit,rows_positive,
    band_contents(~Holder,~value,holder,problem,storage,band,base,first,count,reads,start,U32.to_nat(row_first),U32.to_nat(rows),from_exact,rows_fit,cells_fit,image)}

# The rows a band holds are the input of their window problem.
def band_input(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: U32,+rows: U32,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,U32.to_nat(row_first))) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(row_first),U32.to_nat(rows)),Guards.height(problem)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(rows)),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> Inputs.Contract<Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows))>:
  +context = band_context(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported)
  WindowProblem.contract(problem,storage,U32.to_nat(row_first),U32.to_nat(rows),CellValues.values(context),CellValues.listed(context),CellValues.room(context))
# The state of the window problem of the rows a band holds is the state on
# the values of the cells of the band.
def band_state(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: U32,+rows: U32,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,U32.to_nat(row_first))) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(row_first),U32.to_nat(rows)),Guards.height(problem)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(rows)),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>) ->
  {Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output) ==
    Model.State{Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output} : Model.State}:
  ConvSpec.Problem{shape,+input,+weights,+bias} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Inputs.Contract{+input_evidence,+weight_evidence,+bias_evidence,+parameters} = storage
  {==}

# A call on a band runs the blocked loop of the window problem of its rows on
# the values the band holds.
def band_values(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: U32,+rows: U32,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,U32.to_nat(row_first))) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(row_first),U32.to_nat(rows)),Guards.height(problem)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(rows)),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +pack_start: U32,+offset: U32,+positions: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,
  +tile_depth: Nat,+pair_depth: Nat,+units: Nat,+cap: U32,+ready: Model.CallReady<Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,pairs,tiles,tile_depth,pair_depth,units,cap>) ->
  {Model.values_of(Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))) == Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output)) : Model.State}:
  %band_work(problem,base,U32.to_nat(row_first),rows,(offset + positions : U32)) :
    {Model.values_of(Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))) == Model.values_into(_,pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output)) : Model.State}
  %Equal.sym(Model.State,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output),Model.State{Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output},
    band_state(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported,tiles,output)) :
    {Model.values_of(Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))) == Model.values_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),SpatialSamples.plain(Ready.band_window(problem,base,rows))},pack_start,offset,_) : Model.State}
  Model.into_ready(MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows),pack_start,offset,Buffer.tree(~Tile.Tile8,band),pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output,
    tile_depth,pair_depth,units,cap,ready)

# What a call on a band writes is what the blocked loop of the window problem
# of its rows writes.
def band_output(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: U32,+rows: U32,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,U32.to_nat(row_first))) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(row_first),U32.to_nat(rows)),Guards.height(problem)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(rows)),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +pack_start: U32,+offset: U32,+positions: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,
  +tile_depth: Nat,+pair_depth: Nat,+units: Nat,+cap: U32,+ready: Model.CallReady<Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,pairs,tiles,tile_depth,pair_depth,units,cap>) ->
  {Model.cells_result(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))) == Model.result(Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output))) : Tree.Tree<Tile.Tile8>}:
  Equal.trans(Tree.Tree<Tile.Tile8>,Model.cells_result(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))),Model.result(Model.values_of(Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output})))),Model.result(Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output))),
    Equal.sym(Tree.Tree<Tile.Tile8>,Model.result(Model.values_of(Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output})))),Model.cells_result(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))),Model.values_result(Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output})))),
    Equal.cong(Model.State,Tree.Tree<Tile.Tile8>,state => Model.result(state),Model.values_of(Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))),Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output)),band_values(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported,pack_start,offset,positions,pairs,tiles,output,tile_depth,pair_depth,units,cap,ready)))

# The workspace a call on a band leaves is the one the blocked loop leaves.
def band_workspace(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: U32,+rows: U32,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,U32.to_nat(row_first))) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(row_first),U32.to_nat(rows)),Guards.height(problem)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(rows)),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +pack_start: U32,+offset: U32,+positions: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,
  +tile_depth: Nat,+pair_depth: Nat,+units: Nat,+cap: U32,+ready: Model.CallReady<Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,pairs,tiles,tile_depth,pair_depth,units,cap>) ->
  {Model.cells_tiles(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))) == Model.workspace(Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output))) : Tree.Tree<Tile.Tile8>}:
  Equal.trans(Tree.Tree<Tile.Tile8>,Model.cells_tiles(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))),Model.workspace(Model.values_of(Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output})))),Model.workspace(Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output))),
    Equal.sym(Tree.Tree<Tile.Tile8>,Model.workspace(Model.values_of(Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output})))),Model.cells_tiles(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))),Model.values_workspace(Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output})))),
    Equal.cong(Model.State,Tree.Tree<Tile.Tile8>,state => Model.workspace(state),Model.values_of(Pixels.window_values(Ready.band_window(problem,base,rows),Buffer.tree(~Tile.Tile8,band)),Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))),Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output)),band_values(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported,pack_start,offset,positions,pairs,tiles,output,tile_depth,pair_depth,units,cap,ready)))

# Every point a call on a band writes holds the matrix value of the window
# problem of the rows of the band.
def band_readback(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: U32,+rows: U32,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,U32.to_nat(row_first))) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(row_first),U32.to_nat(rows)),Guards.height(problem)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(rows)),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +pack_start: U32,+offset: U32,+positions: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,
  +tile_depth: Nat,+pair_depth: Nat,+units: Nat,+cap: U32,+ready: Model.CallReady<Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,pairs,tiles,tile_depth,pair_depth,units,cap>,
  +output_depth: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +geometry: Blocked.Geometry<limits,Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,offset,positions,Window.padded(positions),Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))),cap>,
  +memory: Blocked.Memory<Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,offset,positions,Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))),tiles,output,tile_depth,output_depth>,
  +schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(positions) + Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))) - 1) / Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))) : U32)),0n,U32.to_nat(Window.padded(positions)),U32.to_nat(Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))))>,
  +point: Blocked.Point<Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,positions,channel,row,column,Window.padded(positions)>) ->
  Blocked.LoopEvidence<Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,offset,channel,row,column,output_depth,Model.cells_result(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output})))>:
  %Equal.sym(Tree.Tree<Tile.Tile8>,Model.cells_result(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))),Model.result(Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output))),band_output(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported,pack_start,offset,positions,pairs,tiles,output,tile_depth,pair_depth,units,cap,ready)) : Blocked.LoopEvidence<Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,offset,channel,row,column,output_depth,_>
  Blocked.into_readback(limits,Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,offset,positions,band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),cap,tiles,output,tile_depth,output_depth,channel,row,column,geometry,memory,point,schedule)

# A call on a band keeps the cells of the positions below its offset.
def band_keeps(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: U32,+rows: U32,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,U32.to_nat(row_first))) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(row_first),U32.to_nat(rows)),Guards.height(problem)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(rows)),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +pack_start: U32,+offset: U32,+positions: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,
  +tile_depth: Nat,+pair_depth: Nat,+units: Nat,+cap: U32,+ready: Model.CallReady<Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,pairs,tiles,tile_depth,pair_depth,units,cap>,
  +output_depth: Nat,+channel: Nat,+at: Nat,
  +geometry: Blocked.Geometry<limits,Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,offset,positions,Window.padded(positions),Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))),cap>,
  +memory: Blocked.Memory<Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,offset,positions,Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))),tiles,output,tile_depth,output_depth>,
  +schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(positions) + Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))) - 1) / Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))) : U32)),0n,U32.to_nat(Window.padded(positions)),U32.to_nat(Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))))>,
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))) == True{} : Bool},+below: {Nat.is_lt(at,U32.to_nat(offset)) == True{} : Bool}) ->
  Blocked.Keeps(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),channel,at,output,Model.cells_result(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output})))):
  %Equal.sym(Tree.Tree<Tile.Tile8>,Model.cells_result(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))),Model.result(Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output))),band_output(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported,pack_start,offset,positions,pairs,tiles,output,tile_depth,pair_depth,units,cap,ready)) : Blocked.Keeps(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),channel,at,output,_)
  Blocked.into_keeps(limits,Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,offset,positions,band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),cap,tiles,output,tile_depth,output_depth,channel,at,geometry,memory,schedule,channel_inside,below)

# A call on a band leaves its workspace and its output balanced.
def band_balances(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +band: Buffer.Model<Tile.Tile8>,+base: U32,+first: U32,+count: U32,+reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,first,count),
  +start: Nat,+row_first: U32,+rows: U32,+from_exact: {U32.to_nat(first) == Nat.add(start,Cells.count(problem,U32.to_nat(row_first))) : Nat},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(row_first),U32.to_nat(rows)),Guards.height(problem)) == True{} : Bool},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(rows)),U32.to_nat(count)) == True{} : Bool},+image: HeldImage(~Holder,~value,holder,problem,storage,start),
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +pack_start: U32,+offset: U32,+positions: U32,+pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,
  +tile_depth: Nat,+pair_depth: Nat,+units: Nat,+cap: U32,+ready: Model.CallReady<Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,pairs,tiles,tile_depth,pair_depth,units,cap>,
  +output_depth: Nat,
  +geometry: Blocked.Geometry<limits,Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,offset,positions,Window.padded(positions),Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))),cap>,
  +memory: Blocked.Memory<Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,offset,positions,Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))),tiles,output,tile_depth,output_depth>,
  +schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(positions) + Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))) - 1) / Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))) : U32)),0n,U32.to_nat(Window.padded(positions)),U32.to_nat(Pipeline.block_positions(Coordinates.native_inner(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)))))>) ->
  Blocked.Balances<tile_depth,output_depth,Model.cells_tiles(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))),Model.cells_result(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output})))>:
  %Equal.sym(Tree.Tree<Tile.Tile8>,Model.cells_tiles(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))),Model.workspace(Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output))),band_workspace(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported,pack_start,offset,positions,pairs,tiles,output,tile_depth,pair_depth,units,cap,ready)) : Blocked.Balances<tile_depth,output_depth,_,Model.cells_result(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output})))>
  %Equal.sym(Tree.Tree<Tile.Tile8>,Model.cells_result(Model.call_cells(Model.cells_into(Pipeline.WindowWork{MicroLeaf.rows(problem),(offset + positions : U32),Ready.band_window(problem,base,rows)},pack_start,offset,Buffer.tree(~Tile.Tile8,band),Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))),Model.result(Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output))),band_output(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported,pack_start,offset,positions,pairs,tiles,output,tile_depth,pair_depth,units,cap,ready)) : Blocked.Balances<tile_depth,output_depth,Model.workspace(Model.values_into(Blocked.work(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),(offset + positions : U32)),pack_start,offset,Blocked.state(Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),tiles,output))),_>
  Blocked.into_balances(limits,Samples.window(problem,U32.to_nat(row_first),U32.to_nat(rows)),pack_start,offset,positions,band_input(~Holder,~value,holder,limits,problem,storage,band,base,first,count,reads,start,row_first,rows,from_exact,rows_fit,rows_positive,cells_fit,image,valid,supported),cap,tiles,output,tile_depth,output_depth,geometry,memory,schedule)
