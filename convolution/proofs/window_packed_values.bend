# The output of a call of the packed window kernel on the problem of an
# image: the cells it writes hold the matrix values of their positions and
# the cells before them are kept.
import Base
import ../specification.bend as ConvSpec
import ../inference.bend as NativeConv
import ../shape.bend as ConvShape
import ../../lib/machine_limits.bend as Machine
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_window.bend as Window
import ../../lib/kernel_window_pipeline.bend as WindowPipeline
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_move_readback.bend as Held
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_tree_model.bend as Tree
import ./image_problems.bend as ImageProblems
import ./input_storage.bend as Inputs
import ./list_array_roundtrip.bend as ListArrayRoundtrip
import ./list_array_allocation.bend as ListArrayAllocation
import ./window_band_cover.bend as BandCover
import ./window_packed_band.bend as Band
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/binary_radix.bend as Radix
import ../../lib/kernel_gemm.bend as Gemm
import ../../lib/proofs/storage_shape.bend as Shape
import ./window_cell_values.bend as CellValues
import ./concrete_support_guard.bend as Support
import ./supported_padded_coordinates.bend as Padded
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./window_contents.bend as Contents
import ../../lib/proofs/nat_ceiling.bend as Ceiling
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_extrema.bend as NatExtrema
import ../../lib/proofs/u32_extrema.bend as Extrema
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/nat_division_order.bend as DivisionOrder
import ../../lib/proofs/u32_division.bend as U32Division
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/kernel_pipeline_buffer_shape.bend as BufferShape
import ../../lib/tensor_move.bend as Move
import ../../lib/tensor_grid.bend as Grid
import ./microkernel_leaf_readback.bend as MicroLeaf
import ./convolution_guard_facts.bend as Guards
import ./reduction_coordinate_bounds.bend as Coordinates
import ../../lib/kernel_pipeline.bend as Pipeline
import ../../lib/kernel_spatial_packing.bend as Packing
import ../../lib/storage_buffer.bend as Storage
import ../../lib/storage_copy.bend as Copy
import ../../lib/tensor.bend as Tensors
import ../../lib/proofs/storage_copy_refinement.bend as CopyLaws
import ../../lib/proofs/write_program.bend as Writes
import ../../lib/proofs/contiguous_copy.bend as Contiguous
import ../../lib/proofs/kernel_pipeline_bounds.bend as BlockBounds
import ../../lib/proofs/kernel_pipeline_model.bend as Model
import ../../lib/proofs/kernel_tile_refinement.bend as TileLaws
import ./output_values.bend as Values
import ./blocked_convolution_readback.bend as Blocked
import ./window_cell_contents.bend as Cells
import ./window_positions.bend as Positions
import ./window_call_evidence.bend as Evidence
import ../../lib/proofs/tensor_tiled_padded_cells.bend as PaddedCells
import ../../lib/proofs/sequence_membership.bend as Members
import ../../lib/proofs/ordered_ranges.bend as Ranges
import ./window_skeleton.bend as Skeleton
import ./window_packed_call.bend as Call
import ./window_call_ready.bend as Ready

# The calls of a task of the windowed convolution, one for the positions of
# each image the task covers.

# The images of a batch share their shape, their inputs have the same length
# and so have their weights, so one image's problem is supported when
# another's is.
def image_supported(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,
  +kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,+other: Nat,
  +supported: {NativeConv.supports_packed_with_limits(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,other)) == True{} : Bool}) ->
  {NativeConv.supports_packed_with_limits(limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool}:
  +area = Nat.mul(channels,Nat.mul(height,width))
  +shape = {ConvSpec.Shape{channels,outputs,height,width,kernel,stride,padding} : ConvSpec.Shape}
  %Equal.sym(Nat,ConvSpec.length(ImageProblems.image_values(~Logical,~original,area,Nat.mul(image,area),logical)),area,
    ImageProblems.image_length(~Logical,~original,area,Nat.mul(image,area),logical)) :
    {ConvShape.supports_packed(limits,shape,_,ConvSpec.length(weights)) == True{} : Bool}
  %ImageProblems.image_length(~Logical,~original,area,Nat.mul(other,area),logical) :
    {ConvShape.supports_packed(limits,shape,_,ConvSpec.length(weights)) == True{} : Bool}
  supported

# The validity conditions of a problem with an input and weights of the given
# lengths.
def valid_length(+ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+pad: Nat,+input_length: Nat,+weight_length: Nat,+bias: List<&2,F32>) -> Bool:
  Nat.is_lt(0n,ci) && Nat.is_lt(0n,co) && Nat.is_lt(0n,height) && Nat.is_lt(0n,width) && Nat.is_lt(0n,kernel) && Nat.is_lt(0n,stride) &&
  Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,pad))) && Nat.is_le(kernel,Nat.add(width,Nat.mul(2n,pad))) &&
  Nat.is_eq(input_length,Nat.mul(ci,Nat.mul(height,width))) &&
  Nat.is_eq(weight_length,Nat.mul(co,Nat.mul(ci,Nat.mul(kernel,kernel)))) &&
  Nat.is_eq(ConvSpec.length(bias),co)

def image_valid(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,
  +kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,+other: Nat,
  +valid: {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,other)) == True{} : Bool}) ->
  {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)) == True{} : Bool}:
  +area = Nat.mul(channels,Nat.mul(height,width))
  %Equal.sym(Nat,ConvSpec.length(ImageProblems.image_values(~Logical,~original,area,Nat.mul(image,area),logical)),area,
    ImageProblems.image_length(~Logical,~original,area,Nat.mul(image,area),logical)) :
    {valid_length(channels,outputs,height,width,kernel,stride,padding,_,ConvSpec.length(weights),bias) == True{} : Bool}
  %ImageProblems.image_length(~Logical,~original,area,Nat.mul(other,area),logical) :
    {valid_length(channels,outputs,height,width,kernel,stride,padding,_,ConvSpec.length(weights),bias) == True{} : Bool}
  valid

# The storage of one image: its values in a tree of their own, with the
# weights, bias and parameter cells every image shares.
def image_ready(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,
  +kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},+positive: {Nat.is_lt(0n,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(channels,Nat.mul(height,width)),U32.to_nat(upper)) == True{} : Bool}) -> ListArrayAllocation.Ready(ImageProblems.image_values(~Logical,~original,Nat.mul(channels,Nat.mul(height,width)),Nat.mul(image,Nat.mul(channels,Nat.mul(height,width))),logical)):
  +length_exact = Equal.sym(Nat,ConvSpec.length(ImageProblems.image_values(~Logical,~original,Nat.mul(channels,Nat.mul(height,width)),Nat.mul(image,Nat.mul(channels,Nat.mul(height,width))),logical)),Nat.mul(channels,Nat.mul(height,width)),ImageProblems.image_length(~Logical,~original,Nat.mul(channels,Nat.mul(height,width)),Nat.mul(image,Nat.mul(channels,Nat.mul(height,width))),logical))
  ListArrayAllocation.ready(ImageProblems.image_values(~Logical,~original,Nat.mul(channels,Nat.mul(height,width)),Nat.mul(image,Nat.mul(channels,Nat.mul(height,width))),logical),upper,upper_ok,upper_bound,
    %length_exact : {Nat.is_lt(0n,_) == True{} : Bool} positive,
    %length_exact : {Nat.is_le(_,U32.to_nat(upper)) == True{} : Bool} room)

# The storage of one image: its values in a tree of their own, with the
# weights and bias every image shares.
def image_contract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,
  +kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},+positive: {Nat.is_lt(0n,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(channels,Nat.mul(height,width)),U32.to_nat(upper)) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +parameters: Inputs.Parameters(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,0n))) -> Inputs.Contract<ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image)>:
  Inputs.Contract{ListArrayRoundtrip.sequence_evidence(ImageProblems.image_values(~Logical,~original,Nat.mul(channels,Nat.mul(height,width)),Nat.mul(image,Nat.mul(channels,Nat.mul(height,width))),logical),image_ready(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,positive,room)),
    weight_evidence,bias_evidence,parameters}

# The weight cells a task holds: those every image reads.
def weight_tree(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,
  +kernel: Nat,+stride: Nat,+padding: Nat,+weights: List<&2,F32>,+bias: List<&2,F32>,+image: Nat,+upper: U32,+upper_ok: {U32.is_le(upper,2147483648) == True{} : Bool},
  +upper_bound: {U32.is_le(upper,1073741824) == True{} : Bool},+positive: {Nat.is_lt(0n,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(channels,Nat.mul(height,width)),U32.to_nat(upper)) == True{} : Bool},
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +parameters: Inputs.Parameters(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,0n))) -> Tree.Tree<Tile.Tile8>:
  Inputs.weight_cells(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image),image_contract(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weights,bias,image,upper,upper_ok,upper_bound,positive,room,weight_evidence,bias_evidence,parameters))

# The record of the convolution of a problem, as the windowed pipeline reads
# it: the shape words, the output channels, the inner extent of the product
# and its groups.
def groups_in(+problem: ConvSpec.Problem) -> U32:
  Grid.groups((Coordinates.native_inner(problem) / (BandCover.kernel_word(problem) * BandCover.kernel_word(problem)) : U32))

# The cells of a row of the image: its width times the groups of its channels.
def groups_word_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool}) -> {U32.to_nat(groups_in(problem)) == Cells.group_count(problem) : Nat}:
  CellValues.group_count_exact(limits,problem,(Coordinates.native_inner(problem) / (U32.from_nat(Guards.kernel(problem)) * U32.from_nat(Guards.kernel(problem))) : U32),
    supported,Coordinates.inner_channels(limits,problem,Guards.valid(problem,valid),Padded.scalar(limits,problem,Support.fast(limits,problem,supported))))

def record(+problem: ConvSpec.Problem,+groups_out: U32,+activation: Activations.Activation) -> WindowPipeline.Convolution:
  WindowPipeline.Convolution{BandCover.height_word(problem),U32.from_nat(Guards.width(problem)),BandCover.kernel_word(problem),BandCover.stride_word(problem),
    BandCover.padding_word(problem),U32.from_nat(ConvSpec.out_height(problem)),BandCover.width_word(problem),MicroLeaf.rows(problem),Coordinates.native_inner(problem),
    groups_in(problem),groups_out,activation}

# The band of input cells a call on positions [local, local + count) of image
# image reads.
def band_first(+problem: ConvSpec.Problem,+image: U32,+local: U32) -> U32:
  ((image * BandCover.height_word(problem) + BandCover.first_word(problem,local)) * U32.from_nat(Guards.width(problem)) * groups_in(problem) : U32)

def band_cells(+problem: ConvSpec.Problem,+local: U32,+count: U32) -> U32:
  (BandCover.rows_word(problem,local,count) * U32.from_nat(Guards.width(problem)) * groups_in(problem) : U32)

def band_result(+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+band: Buffer.Model<Tile.Tile8>,+base: U32,+local: U32,+count: U32,+offset: U32,
  +pairs: Tree.Tree<Tile.Tile16>,+tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>) -> Model.Cells:
  Model.call_cells(Model.cells_into(Ready.band_work(problem,base,local,count,offset),Evidence.pack_start(problem,local),offset,Buffer.tree(~Tile.Tile8,band),
    Model.Cells{pairs,tiles,Inputs.weight_cells(problem,storage),Inputs.bias_cells(problem,storage),output}))

def loop_value(-problem: ConvSpec.Problem,-origin: U32,-offset: U32,-channel: Nat,-row: Nat,-column: Nat,-depth: Nat,-output: Tree.Tree<Tile.Tile8>,
  evidence: Blocked.LoopEvidence<problem,origin,offset,channel,row,column,depth,output>) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,output,Blocked.address(problem,origin,offset,channel,row,column))) == Values.matrix_value(problem,channel,row,column) : F32}:
  Blocked.LoopEvidence{balanced,value} = evidence
  value

# Point at of a call holds the matrix value of its position in the image.
def written_located(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +band: Buffer.Model<Tile.Tile8>,+base: U32,
  +image: U32,+local: U32,+count: U32,+offset: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},
  +start: Nat,+held: Band.HeldImage(~Holder,~value,holder,problem,storage,start),
  +from_exact: {U32.to_nat(band_first(problem,image,local)) == Nat.add(start,Cells.count(problem,U32.to_nat(BandCover.first_word(problem,local)))) : Nat},
  +reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,band_first(problem,image,local),band_cells(problem,local,count)),
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(BandCover.rows_word(problem,local,count))),U32.to_nat(band_cells(problem,local,count))) == True{} : Bool},
  +cover: Positions.Cover<problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count)>,+cap: U32,+tile_depth: Nat,+output_depth: Nat,
  +pairs: Tree.Tree<Tile.Tile16>,+pair_depth: Nat,+tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,+geometry: Blocked.Geometry<limits,Evidence.window(problem,local,count),Evidence.pack_start(problem,local),offset,count,Window.padded(count),Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))),cap>,+schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(count) + Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))) - 1) / Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))) : U32)),0n,U32.to_nat(Window.padded(count)),U32.to_nat(Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))))>,+memory: Blocked.Memory<Evidence.window(problem,local,count),Evidence.pack_start(problem,local),offset,count,Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))),tiles,output,tile_depth,output_depth>,+ready: Model.CallReady<Ready.band_work(problem,base,local,count,offset),Evidence.pack_start(problem,local),offset,pairs,tiles,tile_depth,pair_depth,Nat.div(Nat.add(U32.to_nat(count),7n),8n),cap>,
  +result: Tree.Tree<Tile.Tile8>,+result_exact: {result == Model.cells_result(band_result(problem,storage,band,base,local,count,offset,pairs,tiles,output)) : Tree.Tree<Tile.Tile8>},+at: Nat,+channel: Nat,+located: Evidence.Located<problem,local,count,channel,at>) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,result,Blocked.cell_address(problem,channel,Nat.add(U32.to_nat(offset),at)))) ==
    Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(local),at),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(local),at),ConvSpec.out_width(problem))) : F32}:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Evidence.Located{+row,+column,+point,+local_exact,+value_exact} = located
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  +read = loop_value(Evidence.window(problem,local,count),Evidence.pack_start(problem,local),offset,channel,row,column,output_depth,Model.cells_result(band_result(problem,storage,band,base,local,count,offset,pairs,tiles,output)),
    Band.band_readback(~Holder,~value,holder,limits,problem,storage,band,base,band_first(problem,image,local),band_cells(problem,local,count),reads,start,BandCover.first_word(problem,local),BandCover.rows_word(problem,local,count),from_exact,Evidence.cover_fits(problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count),cover),Evidence.cover_positive(problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count),cover),cells_fit,held,valid,supported,Evidence.pack_start(problem,local),offset,count,pairs,tiles,output,tile_depth,pair_depth,Nat.div(Nat.add(U32.to_nat(count),7n),8n),cap,ready,output_depth,channel,row,column,geometry,memory,schedule,point))
  %Equal.sym(Tree.Tree<Tile.Tile8>,result,Model.cells_result(band_result(problem,storage,band,base,local,count,offset,pairs,tiles,output)),result_exact) :
    {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,_,Blocked.cell_address(problem,channel,Nat.add(U32.to_nat(offset),at)))) ==
      Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(local),at),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(local),at),ConvSpec.out_width(problem))) : F32}
  %value_exact :
    {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Model.cells_result(band_result(problem,storage,band,base,local,count,offset,pairs,tiles,output)),Blocked.cell_address(problem,channel,Nat.add(U32.to_nat(offset),at)))) == _ : F32}
  %local_exact :
    {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,Model.cells_result(band_result(problem,storage,band,base,local,count,offset,pairs,tiles,output)),Blocked.cell_address(problem,channel,Nat.add(U32.to_nat(offset),_)))) ==
      Values.matrix_value(Evidence.window(problem,local,count),channel,row,column) : F32}
  read

def written_at(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +band: Buffer.Model<Tile.Tile8>,+base: U32,
  +image: U32,+local: U32,+count: U32,+offset: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},
  +start: Nat,+held: Band.HeldImage(~Holder,~value,holder,problem,storage,start),
  +from_exact: {U32.to_nat(band_first(problem,image,local)) == Nat.add(start,Cells.count(problem,U32.to_nat(BandCover.first_word(problem,local)))) : Nat},
  +reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,band_first(problem,image,local),band_cells(problem,local,count)),
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(BandCover.rows_word(problem,local,count))),U32.to_nat(band_cells(problem,local,count))) == True{} : Bool},
  +cover: Positions.Cover<problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count)>,+cap: U32,+tile_depth: Nat,+output_depth: Nat,
  +pairs: Tree.Tree<Tile.Tile16>,+pair_depth: Nat,+tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,+geometry: Blocked.Geometry<limits,Evidence.window(problem,local,count),Evidence.pack_start(problem,local),offset,count,Window.padded(count),Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))),cap>,+schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(count) + Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))) - 1) / Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))) : U32)),0n,U32.to_nat(Window.padded(count)),U32.to_nat(Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))))>,+memory: Blocked.Memory<Evidence.window(problem,local,count),Evidence.pack_start(problem,local),offset,count,Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))),tiles,output,tile_depth,output_depth>,+ready: Model.CallReady<Ready.band_work(problem,base,local,count,offset),Evidence.pack_start(problem,local),offset,pairs,tiles,tile_depth,pair_depth,Nat.div(Nat.add(U32.to_nat(count),7n),8n),cap>,
  +result: Tree.Tree<Tile.Tile8>,+result_exact: {result == Model.cells_result(band_result(problem,storage,band,base,local,count,offset,pairs,tiles,output)) : Tree.Tree<Tile.Tile8>},+at: Nat,+channel: Nat,+at_inside: {Nat.is_lt(at,U32.to_nat(count)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,result,Blocked.cell_address(problem,channel,Nat.add(U32.to_nat(offset),at)))) ==
    Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(local),at),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(local),at),ConvSpec.out_width(problem))) : F32}:
  written_located(~Holder,~value,holder,limits,problem,storage,valid,supported,band,base,image,local,count,offset,positive,grid,start,held,from_exact,reads,cells_fit,cover,cap,tile_depth,output_depth,pairs,pair_depth,tiles,output,geometry,schedule,memory,ready,result,result_exact,at,channel,
    Evidence.covered_located(limits,problem,valid,supported,local,count,positive,grid,cover,channel,at,channel_inside,at_inside))

# A call keeps the output cells of the positions before its offset.
def kept_at(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,
  +valid: {ConvSpec.valid(problem) == True{} : Bool},+supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},
  +band: Buffer.Model<Tile.Tile8>,+base: U32,
  +image: U32,+local: U32,+count: U32,+offset: U32,+positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  +grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},
  +start: Nat,+held: Band.HeldImage(~Holder,~value,holder,problem,storage,start),
  +from_exact: {U32.to_nat(band_first(problem,image,local)) == Nat.add(start,Cells.count(problem,U32.to_nat(BandCover.first_word(problem,local)))) : Nat},
  +reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,band_first(problem,image,local),band_cells(problem,local,count)),
  +cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(BandCover.rows_word(problem,local,count))),U32.to_nat(band_cells(problem,local,count))) == True{} : Bool},
  +cover: Positions.Cover<problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count)>,+cap: U32,+tile_depth: Nat,+output_depth: Nat,
  +pairs: Tree.Tree<Tile.Tile16>,+pair_depth: Nat,+tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,+geometry: Blocked.Geometry<limits,Evidence.window(problem,local,count),Evidence.pack_start(problem,local),offset,count,Window.padded(count),Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))),cap>,+schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(count) + Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))) - 1) / Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))) : U32)),0n,U32.to_nat(Window.padded(count)),U32.to_nat(Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))))>,+memory: Blocked.Memory<Evidence.window(problem,local,count),Evidence.pack_start(problem,local),offset,count,Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))),tiles,output,tile_depth,output_depth>,+ready: Model.CallReady<Ready.band_work(problem,base,local,count,offset),Evidence.pack_start(problem,local),offset,pairs,tiles,tile_depth,pair_depth,Nat.div(Nat.add(U32.to_nat(count),7n),8n),cap>,
  +result: Tree.Tree<Tile.Tile8>,+result_exact: {result == Model.cells_result(band_result(problem,storage,band,base,local,count,offset,pairs,tiles,output)) : Tree.Tree<Tile.Tile8>},+at: Nat,+channel: Nat,+below: {Nat.is_lt(at,U32.to_nat(offset)) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,result,Blocked.cell_address(problem,channel,at)) == Tree.read(Tile.Tile8,output,Blocked.cell_address(problem,channel,at)) : Tile.Tile8}:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  %Equal.sym(Tree.Tree<Tile.Tile8>,result,Model.cells_result(band_result(problem,storage,band,base,local,count,offset,pairs,tiles,output)),result_exact) :
    {Tree.read(Tile.Tile8,_,Blocked.cell_address(problem,channel,at)) == Tree.read(Tile.Tile8,output,Blocked.cell_address(problem,channel,at)) : Tile.Tile8}
  Band.band_keeps(~Holder,~value,holder,limits,problem,storage,band,base,band_first(problem,image,local),band_cells(problem,local,count),reads,start,BandCover.first_word(problem,local),BandCover.rows_word(problem,local,count),from_exact,Evidence.cover_fits(problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count),cover),Evidence.cover_positive(problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count),cover),cells_fit,held,valid,supported,Evidence.pack_start(problem,local),offset,count,pairs,tiles,output,tile_depth,pair_depth,Nat.div(Nat.add(U32.to_nat(count),7n),8n),cap,ready,output_depth,channel,at,geometry,memory,schedule,channel_inside,below)

# What reads back the output cells of a call on positions [local, local +
# count) of a problem at output position offset: its premises and the output
# before and after it.
type CallReadback<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-holder: Holder,-limits: Machine.Limits,-problem: ConvSpec.Problem,-storage: Inputs.Contract<problem>,-local: U32,-count: U32,-offset: U32,-output: Tree.Tree<Tile.Tile8>,-result: Tree.Tree<Tile.Tile8>> is Data:
  CallReadback{valid: {ConvSpec.valid(problem) == True{} : Bool},supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},band: Buffer.Model<Tile.Tile8>,base: U32,image: U32,positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),Nat.mul(ConvSpec.out_height(problem),ConvSpec.out_width(problem))) == True{} : Bool},start: Nat,held: Band.HeldImage(~Holder,~value,holder,problem,storage,start),from_exact: {U32.to_nat(band_first(problem,image,local)) == Nat.add(start,Cells.count(problem,U32.to_nat(BandCover.first_word(problem,local)))) : Nat},reads: Skeleton.BandHolds(~Holder,~value,holder,band,base,band_first(problem,image,local),band_cells(problem,local,count)),cells_fit: {Nat.is_le(Cells.count(problem,U32.to_nat(BandCover.rows_word(problem,local,count))),U32.to_nat(band_cells(problem,local,count))) == True{} : Bool},cover: Positions.Cover<problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count)>,cap: U32,tile_depth: Nat,output_depth: Nat,pairs: Tree.Tree<Tile.Tile16>,pair_depth: Nat,tiles: Tree.Tree<Tile.Tile8>,geometry: Blocked.Geometry<limits,Evidence.window(problem,local,count),Evidence.pack_start(problem,local),offset,count,Window.padded(count),Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))),cap>,schedule: BlockBounds.Schedule<U32.to_nat(((Window.padded(count) + Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))) - 1) / Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))) : U32)),0n,U32.to_nat(Window.padded(count)),U32.to_nat(Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))))>,memory: Blocked.Memory<Evidence.window(problem,local,count),Evidence.pack_start(problem,local),offset,count,Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))),tiles,output,tile_depth,output_depth>,ready: Model.CallReady<Ready.band_work(problem,base,local,count,offset),Evidence.pack_start(problem,local),offset,pairs,tiles,tile_depth,pair_depth,Nat.div(Nat.add(U32.to_nat(count),7n),8n),cap>,result_exact: {result == Model.cells_result(band_result(problem,storage,band,base,local,count,offset,pairs,tiles,output)) : Tree.Tree<Tile.Tile8>}}

# Point at of a call holds the matrix value of its position in the image.
def readback_written(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+local: U32,+count: U32,+offset: U32,+output: Tree.Tree<Tile.Tile8>,+result: Tree.Tree<Tile.Tile8>,+readback: CallReadback<Holder,value,holder,limits,problem,storage,local,count,offset,output,result>,+at: Nat,+channel: Nat,+at_inside: {Nat.is_lt(at,U32.to_nat(count)) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,result,Blocked.cell_address(problem,channel,Nat.add(U32.to_nat(offset),at)))) ==
    Values.matrix_value(problem,channel,Nat.div(Nat.add(U32.to_nat(local),at),ConvSpec.out_width(problem)),Nat.mod(Nat.add(U32.to_nat(local),at),ConvSpec.out_width(problem))) : F32}:
  CallReadback{+valid,+supported,+band,+base,+image,+positive,+grid,+start,+held,+from_exact,+reads,+cells_fit,+cover,+cap,+tile_depth,+output_depth,+pairs,+pair_depth,+tiles,+geometry,+schedule,+memory,+ready,+result_exact} = readback
  written_at(~Holder,~value,holder,limits,problem,storage,valid,supported,band,base,image,local,count,offset,positive,grid,start,held,from_exact,reads,cells_fit,cover,cap,tile_depth,output_depth,pairs,pair_depth,tiles,output,geometry,schedule,memory,ready,result,result_exact,at,channel,at_inside,channel_inside)

# A call keeps the output cells of the positions before its offset.
def readback_kept(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,+holder: Holder,+limits: Machine.Limits,+problem: ConvSpec.Problem,+storage: Inputs.Contract<problem>,+local: U32,+count: U32,+offset: U32,+output: Tree.Tree<Tile.Tile8>,+result: Tree.Tree<Tile.Tile8>,+readback: CallReadback<Holder,value,holder,limits,problem,storage,local,count,offset,output,result>,+at: Nat,+channel: Nat,+below: {Nat.is_lt(at,U32.to_nat(offset)) == True{} : Bool},+channel_inside: {Nat.is_lt(channel,Guards.output_channel_count(problem)) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,result,Blocked.cell_address(problem,channel,at)) == Tree.read(Tile.Tile8,output,Blocked.cell_address(problem,channel,at)) : Tile.Tile8}:
  CallReadback{+valid,+supported,+band,+base,+image,+positive,+grid,+start,+held,+from_exact,+reads,+cells_fit,+cover,+cap,+tile_depth,+output_depth,+pairs,+pair_depth,+tiles,+geometry,+schedule,+memory,+ready,+result_exact} = readback
  kept_at(~Holder,~value,holder,limits,problem,storage,valid,supported,band,base,image,local,count,offset,positive,grid,start,held,from_exact,reads,cells_fit,cover,cap,tile_depth,output_depth,pairs,pair_depth,tiles,output,geometry,schedule,memory,ready,result,result_exact,at,channel,below,channel_inside)

# The cells of a supported image come in groups of eight channels, at least
# one.
def group_positive(+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool}) -> {Nat.is_lt(0n,Cells.group_count(problem)) == True{} : Bool}:
  +channels = Guards.input_channel_count(problem)
  +groups = Cells.group_count(problem)
  RowMajor.left_factor_positive(groups,8n,
    Bounds.nat_trans(0n,channels,Nat.mul(groups,8n),Contents.valid_condition(problem,valid,10n),Ceiling.covers(channels,8n,{==})))

# The first cell and the cells of the band of a call, as numbers.
type BandWords<-problem: ConvSpec.Problem,-image: U32,-local: U32,-count: U32> is Data:
  BandWords{first_exact: {U32.to_nat(band_first(problem,image,local)) == Nat.add(Nat.mul(U32.to_nat(image),Cells.count(problem,Guards.height(problem))),Cells.count(problem,U32.to_nat(BandCover.first_word(problem,local)))) : Nat},
    cells_exact: {U32.to_nat(band_cells(problem,local,count)) == Cells.count(problem,U32.to_nat(BandCover.rows_word(problem,local,count))) : Nat}}

def band_words(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+image: U32,+local: U32,+count: U32,
  +cover: Positions.Cover<problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count)>,
  +room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(image),Cells.count(problem,Guards.height(problem))),Cells.count(problem,Guards.height(problem))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> BandWords<problem,image,local,count>:
  +cap = Machine.capacity(limits)
  +fits = Limits.capacity_fits(limits)
  +support = Padded.scalar(limits,problem,Support.fast(limits,problem,supported))
  +i = U32.to_nat(image)
  +height = Guards.height(problem)
  +width = Guards.width(problem)
  +groups = Cells.group_count(problem)
  +f = U32.to_nat(BandCover.first_word(problem,local))
  +r = U32.to_nat(BandCover.rows_word(problem,local,count))
  +whole = Cells.count(problem,Guards.height(problem))
  +above = Nat.mul(i,whole)
  +rows_fit = Evidence.cover_fits(problem,f,r,U32.to_nat(local),U32.to_nat(count),cover)
  +f_le = Order.transitive(f,Nat.add(f,r),height,Order.le_add(f,r),rows_fit)
  +r_le = Order.transitive(r,Nat.add(f,r),height,Order.right_le(f,r),rows_fit)
  +width_positive = Contents.valid_condition(problem,valid,7n)
  +groups_positive = group_positive(problem,valid)
  +height_exact = SupportedCoordinateBounds.height_exact(limits,problem,support)
  +width_exact = SupportedCoordinateBounds.width_exact(limits,problem,support)
  +groups_exact = CellValues.group_count_exact(limits,problem,(Coordinates.native_inner(problem) / (BandCover.kernel_word(problem) * BandCover.kernel_word(problem)) : U32),
    supported,Coordinates.inner_channels(limits,problem,Guards.valid(problem,valid),support))
  +total = Order.transitive(above,Nat.add(above,whole),U32.to_nat(cap),Order.le_add(above,whole),room)
  +whole_room = Order.transitive(whole,Nat.add(above,whole),U32.to_nat(cap),Order.right_le(above,whole),room)
  +v1 = Nat.mul(i,height)
  +v2 = Nat.add(v1,f)
  +v3 = Nat.mul(v2,width)
  +v4 = Nat.mul(v3,groups)
  +shape = Equal.trans(Nat,above,Nat.mul(i,Nat.mul(height,Nat.mul(width,groups))),Nat.mul(Nat.mul(v1,width),groups),
    Equal.cong(Nat,Nat,value => Nat.mul(i,value),whole,Nat.mul(height,Nat.mul(width,groups)),RowMajor.assoc(height,width,groups)),
    Equal.sym(Nat,Nat.mul(Nat.mul(v1,width),groups),Nat.mul(i,Nat.mul(height,Nat.mul(width,groups))),
      Equal.trans(Nat,Nat.mul(Nat.mul(v1,width),groups),Nat.mul(v1,Nat.mul(width,groups)),Nat.mul(i,Nat.mul(height,Nat.mul(width,groups))),
        RowMajor.assoc(v1,width,groups),RowMajor.assoc(i,height,Nat.mul(width,groups)))))
  +value_exact = Equal.trans(Nat,v4,Nat.mul(Nat.add(Nat.mul(v1,width),Nat.mul(f,width)),groups),Nat.add(above,Cells.count(problem,f)),
    Equal.cong(Nat,Nat,value => Nat.mul(value,groups),v3,Nat.add(Nat.mul(v1,width),Nat.mul(f,width)),RowMajor.distrib(v1,f,width)),
    %Equal.sym(Nat,above,Nat.mul(Nat.mul(v1,width),groups),shape) : {Nat.mul(Nat.add(Nat.mul(v1,width),Nat.mul(f,width)),groups) == Nat.add(_,Cells.count(problem,f)) : Nat}
    RowMajor.distrib(Nat.mul(v1,width),Nat.mul(f,width),groups))
  +part_le = RowMajor.mul_le(Nat.mul(f,width),Nat.mul(height,width),groups,RowMajor.mul_le(f,height,width,f_le))
  +v4_room = {%Equal.sym(Nat,v4,Nat.add(above,Cells.count(problem,f)),value_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(Nat.add(above,Cells.count(problem,f)),Nat.add(above,whole),U32.to_nat(cap),Order.offset_le(above,Cells.count(problem,f),whole,part_le),room) :
    {Nat.is_le(v4,U32.to_nat(cap)) == True{} : Bool}}
  +v3_room = Order.transitive(v3,v4,U32.to_nat(cap),RowMajor.product_covers(v3,groups,groups_positive),v4_room)
  +v2_room = Order.transitive(v2,v3,U32.to_nat(cap),RowMajor.product_covers(v2,width,width_positive),v3_room)
  +v1_room = Order.transitive(v1,v2,U32.to_nat(cap),Order.le_add(v1,f),v2_room)
  +h_word = BandCover.height_word(problem)
  +w_word = U32.from_nat(width)
  +g_word = groups_in(problem)
  +e1 = Arithmetic.product_exact(cap,fits,image,h_word,i,height,{==},height_exact,v1_room)
  +e2 = Arithmetic.sum_exact(cap,fits,(image * h_word : U32),BandCover.first_word(problem,local),v1,f,e1,{==},v2_room)
  +e3 = Arithmetic.product_exact(cap,fits,(image * h_word + BandCover.first_word(problem,local) : U32),w_word,v2,width,e2,width_exact,v3_room)
  +e4 = Arithmetic.product_exact(cap,fits,((image * h_word + BandCover.first_word(problem,local) : U32) * w_word : U32),g_word,v3,groups,e3,groups_exact,v4_room)
  +c1 = Nat.mul(r,width)
  +c2 = Nat.mul(c1,groups)
  +c2_room = Order.transitive(c2,whole,U32.to_nat(cap),RowMajor.mul_le(c1,Nat.mul(height,width),groups,RowMajor.mul_le(r,height,width,r_le)),whole_room)
  +c1_room = Order.transitive(c1,c2,U32.to_nat(cap),RowMajor.product_covers(c1,groups,groups_positive),c2_room)
  +d1 = Arithmetic.product_exact(cap,fits,BandCover.rows_word(problem,local,count),w_word,r,width,{==},width_exact,c1_room)
  +d2 = Arithmetic.product_exact(cap,fits,(BandCover.rows_word(problem,local,count) * w_word : U32),g_word,c1,groups,d1,groups_exact,c2_room)
  BandWords{Equal.trans(Nat,U32.to_nat(band_first(problem,image,local)),v4,Nat.add(above,Cells.count(problem,f)),e4,value_exact),d2}

# The workspace and the output of a task hold what one call writes.
def call_memory(+limits: Machine.Limits,+problem: ConvSpec.Problem,+valid: {ConvSpec.valid(problem) == True{} : Bool},
  +supported: {NativeConv.supports_packed_with_limits(limits,problem) == True{} : Bool},+local: U32,+count: U32,+offset: U32,
  +cover: Positions.Cover<problem,U32.to_nat(BandCover.first_word(problem,local)),U32.to_nat(BandCover.rows_word(problem,local,count)),U32.to_nat(local),U32.to_nat(count)>,+plane: U32,+ordered: {Nat.is_le(U32.to_nat(count),U32.to_nat(plane)) == True{} : Bool},
  +plane_room: {Nat.is_le(U32.to_nat(plane),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+tile_depth: Nat,+output_depth: Nat,+tiles: Tree.Tree<Tile.Tile8>,+output: Tree.Tree<Tile.Tile8>,
  +balances: Blocked.Balances<tile_depth,output_depth,tiles,output>,+tile_depth_ok: {Nat.is_le(tile_depth,30n) == True{} : Bool},
  +output_depth_ok: {Nat.is_le(output_depth,30n) == True{} : Bool},
  +tile_need: {Nat.is_le(U32.to_nat(Pipeline.workspace_tiles(8192,Coordinates.native_inner(problem),plane)),U32.to_nat(U32.shln(1,tile_depth))) == True{} : Bool},
  +output_need: {Nat.is_le(Nat.mul(U32.to_nat((offset + count : U32)),U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),U32.to_nat(U32.shln(1,output_depth))) == True{} : Bool}) ->
  Blocked.Memory<Evidence.window(problem,local,count),Evidence.pack_start(problem,local),offset,count,Pipeline.block_positions(Coordinates.native_inner(Evidence.window(problem,local,count))),tiles,output,tile_depth,output_depth>:
  ConvSpec.Problem{shape,+input_values,+weight_values,+bias_values} = problem
  ConvSpec.Shape{+ci,+co,+height,+width,+kernel,+stride,+padding} = shape
  Blocked.Balances{+tile_balanced,+output_balanced} = balances
  +problem = {ConvSpec.Problem{ConvSpec.Shape{ci,co,height,width,kernel,stride,padding},input_values,weight_values,bias_values} : ConvSpec.Problem}
  +first = U32.to_nat(BandCover.first_word(problem,local))
  +rows = U32.to_nat(BandCover.rows_word(problem,local,count))
  +window_valid = Evidence.cover_valid(problem,first,rows,U32.to_nat(local),U32.to_nat(count),valid,cover)
  +window_supported = Evidence.cover_supported(limits,problem,first,rows,U32.to_nat(local),U32.to_nat(count),valid,supported,cover)
  Blocked.Memory{tile_balanced,tile_depth_ok,
    Evidence.tile_room(limits,Evidence.window(problem,local,count),window_valid,window_supported,count,plane,ordered,plane_room,8192,1073741824,2147483648,BlockBounds.configuration(limits),
      Tree.capacity(Tile.Tile8,tiles),
      %Equal.sym(U32,Tree.capacity(Tile.Tile8,tiles),U32.shln(1,tile_depth),Shape.capacity(Tile.Tile8,tile_depth,tiles,tile_balanced)) :
        {Nat.is_le(U32.to_nat(Pipeline.workspace_tiles(8192,Coordinates.native_inner(problem),plane)),U32.to_nat(_)) == True{} : Bool}
      tile_need),
    output_balanced,output_depth_ok,
    %Equal.sym(U32,Tree.capacity(Tile.Tile8,output),U32.shln(1,output_depth),Shape.capacity(Tile.Tile8,output_depth,output,output_balanced)) :
      {Nat.is_le(Nat.mul(U32.to_nat((offset + count : U32)),U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),U32.to_nat(_)) == True{} : Bool}
    output_need}

# The value of output position at of a task from output position start on:
# the matrix value of its image at its place in the image.
def written_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+start: U32,+at: Nat,+channel: Nat) -> F32:
  Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,Nat.div(Nat.add(U32.to_nat(start),at),ImageProblems.plane(height,width,kernel,stride,padding))),channel,Nat.div(Nat.mod(Nat.add(U32.to_nat(start),at),ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(Nat.mod(Nat.add(U32.to_nat(start),at),ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding)))

# The output cells of a task hold eight channels of a position each.
def cell_groups(+outputs: Nat) -> Nat:
  U32.to_nat(Gemm.blocks(U32.from_nat(outputs)))

# The tile of channel group group at output position at of a task from output
# position start on, +0 past the last channel.
def written_tile(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+start: U32,+at: Nat,+group: Nat) -> Tile.Tile8:
  PaddedCells.kept_tile(~(lane => written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,at,lane)),outputs,group)

# Output cell cell of a task holds the tile of its position and group.
def WrittenCell(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+start: U32,+tree: Tree.Tree<Tile.Tile8>,+cell: Nat) -> Data:
  PaddedCells.Valid<outputs,Nat.mod(cell,cell_groups(outputs)),Tree.read(Tile.Tile8,tree,U32.from_nat(cell)),written_tile(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,Nat.div(cell,cell_groups(outputs)),Nat.mod(cell,cell_groups(outputs)))>

# Lane lane of group group is channel lane + 8 group.
def lane_group(+lane: Nat,+group: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) -> {Nat.div(Nat.add(lane,Nat.mul(group,8n)),8n) == group : Nat}:
  %Order.commutative(Nat.mul(group,8n),lane) : {Nat.div(_,8n) == group : Nat}
  Radix.quotient(group,lane,8n,{==},small)

def lane_place(+lane: Nat,+group: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) -> {Nat.mod(Nat.add(lane,Nat.mul(group,8n)),8n) == lane : Nat}:
  %Order.commutative(Nat.mul(group,8n),lane) : {Nat.mod(_,8n) == lane : Nat}
  Radix.remainder(group,lane,8n,{==},small)

# A lane of the group of a cell at the position of the cell is read there.
def lane_address(+problem: ConvSpec.Problem,+cell: Nat,+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))) == True{} : Bool}) ->
  {Blocked.cell_address(problem,Nat.add(lane,Nat.mul(Nat.mod(cell,U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))),8n)),Nat.div(cell,U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem))))) ==
    U32.from_nat(cell) : U32}:
  +groups = U32.to_nat(Gemm.blocks(MicroLeaf.rows(problem)))
  +group = Nat.mod(cell,groups)
  %Equal.sym(Nat,Nat.div(Nat.add(lane,Nat.mul(group,8n)),8n),group,lane_group(lane,group,small)) :
    {U32.from_nat(Nat.add(Nat.mul(Nat.div(cell,groups),groups),_)) == U32.from_nat(cell) : U32}
  %DivisionOrder.quotient_exact(cell,groups,positive) : {U32.from_nat(_) == U32.from_nat(cell) : U32}
  {==}

# A cell from position offset on lies at a position from offset on.
def position_from(+cell: Nat,+offset: Nat,+groups: Nat,+positive: {Nat.is_lt(0n,groups) == True{} : Bool},
  +late: {Nat.is_lt(cell,Nat.mul(offset,groups)) == False{} : Bool}) -> {Nat.is_le(offset,Nat.div(cell,groups)) == True{} : Bool}:
  +exact = {%Addition.zero_right(Nat.mul(offset,groups)) : {Nat.div(_,groups) == offset : Nat}
    Radix.quotient(offset,0n,groups,positive,positive) : {Nat.div(Nat.mul(offset,groups),groups) == offset : Nat}}
  %exact : {Nat.is_le(_,Nat.div(cell,groups)) == True{} : Bool}
  DivisionOrder.quotient_le(Nat.mul(offset,groups),cell,groups,positive,Order.not_less(cell,Nat.mul(offset,groups),late))

# A position the call writes lies inside its band of positions.
def call_step_inside(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+groups_out: U32,+activation: Activations.Activation,+start: U32,+position: U32,+end: U32,+image: U32,+local: U32,+count: U32,+offset: U32,+after: {Nat.is_le(U32.to_nat(start),U32.to_nat(position)) == True{} : Bool},+offset_exact: {U32.to_nat(offset) == Nat.sub(U32.to_nat(position),U32.to_nat(start)) : Nat},+next_exact: {U32.to_nat((position + count : U32)) == Nat.add(U32.to_nat(position),U32.to_nat(count)) : Nat},
  +at: Nat,+ordered: {Nat.is_le(U32.to_nat(offset),at) == True{} : Bool},+below: {Nat.is_lt(at,Nat.sub(U32.to_nat((position + count : U32)),U32.to_nat(start))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(at,U32.to_nat(offset)),U32.to_nat(count)) == True{} : Bool}:
  +offset = U32.to_nat(offset)
  +count = U32.to_nat(count)
  +at_position = U32.to_nat(position)
  +first = U32.to_nat(start)
  +below_sum = {%Equal.sym(Nat,offset,Nat.sub(at_position,first),offset_exact) : {Nat.is_lt(at,Nat.add(_,count)) == True{} : Bool}
    %Positions.sub_after(at_position,first,count,after) : {Nat.is_lt(at,_) == True{} : Bool}
    %next_exact : {Nat.is_lt(at,Nat.sub(_,first)) == True{} : Bool}
    below : {Nat.is_lt(at,Nat.add(offset,count)) == True{} : Bool}}
  Order.sub_lt(at,offset,count,ordered,below_sum)

# The matrix value a call writes at a position of a task is the value of the
# position.
def moved_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+groups_out: U32,+activation: Activations.Activation,+start: U32,+position: U32,+end: U32,+image: U32,+local: U32,+count: U32,+offset: U32,+after: {Nat.is_le(U32.to_nat(start),U32.to_nat(position)) == True{} : Bool},+word_positive: {Nat.is_lt(0n,U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+plane_positive: {Nat.is_lt(0n,ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool},+plane_exact: {U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},+local_exact: {U32.to_nat(local) == Nat.mod(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+image_exact: {U32.to_nat(image) == Nat.div(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+offset_exact: {U32.to_nat(offset) == Nat.sub(U32.to_nat(position),U32.to_nat(start)) : Nat},+next_exact: {U32.to_nat((position + count : U32)) == Nat.add(U32.to_nat(position),U32.to_nat(count)) : Nat},+at: Nat,+channel: Nat,
  +ordered: {Nat.is_le(U32.to_nat(offset),at) == True{} : Bool},+step_inside: {Nat.is_lt(Nat.sub(at,U32.to_nat(offset)),U32.to_nat(count)) == True{} : Bool}) ->
  {Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),channel,Nat.div(Nat.add(U32.to_nat(local),Nat.sub(at,U32.to_nat(offset))),ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(Nat.add(U32.to_nat(local),Nat.sub(at,U32.to_nat(offset))),ConvSpec.axis_size(width,kernel,stride,padding))) == written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,at,channel) : F32}:
  +offset = U32.to_nat(offset)
  +count = U32.to_nat(count)
  +at_position = U32.to_nat(position)
  +first = U32.to_nat(start)
  +image = U32.to_nat(image)
  +local = U32.to_nat(local)
  +size = U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))
  +step = Nat.sub(at,offset)
  +restored = Order.restore_shift(offset,at,ordered)
  +inner = Nat.add(local,step)
  +plane_grid = {%plane_exact : {Nat.is_le(Nat.add(local,count),_) == True{} : Bool} grid : {Nat.is_le(Nat.add(local,count),ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool}}
  +inner_small = Bounds.nat_trans(inner,Nat.add(local,count),ImageProblems.plane(height,width,kernel,stride,padding),
    %Equal.sym(Bool,Nat.is_lt(inner,Nat.add(local,count)),Nat.is_lt(step,count),Order.lt_offset(local,step,count)) : {_ == True{} : Bool} step_inside,plane_grid)
  +quotient_form = Equal.trans(Nat,at_position,Nat.add(Nat.mul(Nat.div(at_position,size),size),Nat.mod(at_position,size)),Nat.add(Nat.mul(image,ImageProblems.plane(height,width,kernel,stride,padding)),local),
    DivisionOrder.quotient_exact(at_position,size,word_positive),
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(at_position,size),size),Nat.mod(at_position,size)),Nat.add(Nat.mul(image,size),local),Nat.add(Nat.mul(image,ImageProblems.plane(height,width,kernel,stride,padding)),local),
      Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(at_position,size),size),Nat.mod(at_position,size)),Nat.add(Nat.mul(image,size),Nat.mod(at_position,size)),Nat.add(Nat.mul(image,size),local),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,size),Nat.mod(at_position,size)),Nat.div(at_position,size),image,Equal.sym(Nat,image,Nat.div(at_position,size),image_exact)),
        Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(image,size),value),Nat.mod(at_position,size),local,Equal.sym(Nat,local,Nat.mod(at_position,size),local_exact))),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(image,value),local),size,ImageProblems.plane(height,width,kernel,stride,padding),plane_exact)))
  +start_offset = Equal.trans(Nat,Nat.add(first,offset),Nat.add(first,Nat.sub(at_position,first)),at_position,
    Equal.cong(Nat,Nat,value => Nat.add(first,value),offset,Nat.sub(at_position,first),offset_exact),Order.restore_shift(first,at_position,after))
  +position_exact = Equal.trans(Nat,Nat.add(first,at),Nat.add(Nat.add(first,offset),step),Nat.add(Nat.mul(image,ImageProblems.plane(height,width,kernel,stride,padding)),inner),
    Equal.trans(Nat,Nat.add(first,at),Nat.add(first,Nat.add(offset,step)),Nat.add(Nat.add(first,offset),step),
      Equal.cong(Nat,Nat,value => Nat.add(first,value),at,Nat.add(offset,step),Equal.sym(Nat,Nat.add(offset,step),at,restored)),
      Equal.sym(Nat,Nat.add(Nat.add(first,offset),step),Nat.add(first,Nat.add(offset,step)),Addition.associative(first,offset,step))),
    Equal.trans(Nat,Nat.add(Nat.add(first,offset),step),Nat.add(Nat.add(Nat.mul(image,ImageProblems.plane(height,width,kernel,stride,padding)),local),step),Nat.add(Nat.mul(image,ImageProblems.plane(height,width,kernel,stride,padding)),inner),
      Equal.cong(Nat,Nat,value => Nat.add(value,step),Nat.add(first,offset),Nat.add(Nat.mul(image,ImageProblems.plane(height,width,kernel,stride,padding)),local),
        Equal.trans(Nat,Nat.add(first,offset),at_position,Nat.add(Nat.mul(image,ImageProblems.plane(height,width,kernel,stride,padding)),local),start_offset,quotient_form)),
      Addition.associative(Nat.mul(image,ImageProblems.plane(height,width,kernel,stride,padding)),local,step)))
  +global = Nat.add(Nat.mul(image,ImageProblems.plane(height,width,kernel,stride,padding)),inner)
  +value_exact = Equal.trans(F32,Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,Nat.div(Nat.add(first,at),ImageProblems.plane(height,width,kernel,stride,padding))),channel,Nat.div(Nat.mod(Nat.add(first,at),ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(Nat.mod(Nat.add(first,at),ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding))),Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,Nat.div(global,ImageProblems.plane(height,width,kernel,stride,padding))),channel,Nat.div(Nat.mod(global,ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(Nat.mod(global,ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding))),
    Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,image),channel,Nat.div(inner,ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(inner,ConvSpec.axis_size(width,kernel,stride,padding))),
    Equal.cong(Nat,F32,value => Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,Nat.div(value,ImageProblems.plane(height,width,kernel,stride,padding))),channel,Nat.div(Nat.mod(value,ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(Nat.mod(value,ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding))),Nat.add(first,at),global,position_exact),
    Equal.trans(F32,Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,Nat.div(global,ImageProblems.plane(height,width,kernel,stride,padding))),channel,Nat.div(Nat.mod(global,ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(Nat.mod(global,ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding))),
      Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,image),channel,Nat.div(Nat.mod(global,ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(Nat.mod(global,ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding))),
      Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,image),channel,Nat.div(inner,ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(inner,ConvSpec.axis_size(width,kernel,stride,padding))),
      Equal.cong(Nat,F32,value => Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,value),channel,Nat.div(Nat.mod(global,ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(Nat.mod(global,ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding))),
        Nat.div(global,ImageProblems.plane(height,width,kernel,stride,padding)),image,Radix.quotient(image,inner,ImageProblems.plane(height,width,kernel,stride,padding),plane_positive,inner_small)),
      Equal.cong(Nat,F32,value => Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,image),channel,Nat.div(value,ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(value,ConvSpec.axis_size(width,kernel,stride,padding))),
        Nat.mod(global,ImageProblems.plane(height,width,kernel,stride,padding)),inner,Radix.remainder(image,inner,ImageProblems.plane(height,width,kernel,stride,padding),plane_positive,inner_small))))
  Equal.sym(F32,Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,Nat.div(Nat.add(first,at),ImageProblems.plane(height,width,kernel,stride,padding))),channel,Nat.div(Nat.mod(Nat.add(first,at),ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(Nat.mod(Nat.add(first,at),ImageProblems.plane(height,width,kernel,stride,padding)),ConvSpec.axis_size(width,kernel,stride,padding))),Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,image),channel,Nat.div(inner,ConvSpec.axis_size(width,kernel,stride,padding)),Nat.mod(inner,ConvSpec.axis_size(width,kernel,stride,padding))),value_exact)

# A lane of a cell before the call keeps its value.
def old_lane_tile(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> F32,+holder: Holder,+limits: Machine.Limits,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+groups_out: U32,+activation: Activations.Activation,+start: U32,+position: U32,+end: U32,+image: U32,+local: U32,+count: U32,+offset: U32,+previous: Tree.Tree<Tile.Tile8>,+next: Tree.Tree<Tile.Tile8>,+after: {Nat.is_le(U32.to_nat(start),U32.to_nat(position)) == True{} : Bool},+word_positive: {Nat.is_lt(0n,U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+plane_positive: {Nat.is_lt(0n,ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool},+plane_exact: {U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},+local_exact: {U32.to_nat(local) == Nat.mod(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+image_exact: {U32.to_nat(image) == Nat.div(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+offset_exact: {U32.to_nat(offset) == Nat.sub(U32.to_nat(position),U32.to_nat(start)) : Nat},+next_exact: {U32.to_nat((position + count : U32)) == Nat.add(U32.to_nat(position),U32.to_nat(count)) : Nat},+old: Members.All(Nat,{index => WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,previous,index) : Nat -> Data},Ranges.range(Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs)),0n)),+storage: Inputs.Contract<ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image))>,+readback: CallReadback<Holder,value,holder,limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),storage,local,count,offset,previous,next>,+cell: Nat,+positive: {Nat.is_lt(0n,cell_groups(outputs)) == True{} : Bool},+early: {Nat.is_lt(cell,Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs))) == True{} : Bool},+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n)),outputs) == True{} : Bool}) ->
  {Tree.read(Tile.Tile8,next,U32.from_nat(cell)) == Tree.read(Tile.Tile8,previous,U32.from_nat(cell)) : Tile.Tile8}:
  +at_below = {%Equal.sym(Nat,U32.to_nat(offset),Nat.sub(U32.to_nat(position),U32.to_nat(start)),offset_exact) : {Nat.is_lt(Nat.div(cell,cell_groups(outputs)),_) == True{} : Bool}
    DivisionOrder.quotient_below(cell,Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs),positive,early) : {Nat.is_lt(Nat.div(cell,cell_groups(outputs)),U32.to_nat(offset)) == True{} : Bool}}
  %lane_address(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),cell,lane,small,positive) : {Tree.read(Tile.Tile8,next,_) == Tree.read(Tile.Tile8,previous,_) : Tile.Tile8}
  readback_kept(~Holder,~value,holder,limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),storage,local,count,offset,previous,next,readback,Nat.div(cell,cell_groups(outputs)),Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n)),at_below,inside)

def old_lane(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> F32,+holder: Holder,+limits: Machine.Limits,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+groups_out: U32,+activation: Activations.Activation,+start: U32,+position: U32,+end: U32,+image: U32,+local: U32,+count: U32,+offset: U32,+previous: Tree.Tree<Tile.Tile8>,+next: Tree.Tree<Tile.Tile8>,+after: {Nat.is_le(U32.to_nat(start),U32.to_nat(position)) == True{} : Bool},+word_positive: {Nat.is_lt(0n,U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+plane_positive: {Nat.is_lt(0n,ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool},+plane_exact: {U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},+local_exact: {U32.to_nat(local) == Nat.mod(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+image_exact: {U32.to_nat(image) == Nat.div(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+offset_exact: {U32.to_nat(offset) == Nat.sub(U32.to_nat(position),U32.to_nat(start)) : Nat},+next_exact: {U32.to_nat((position + count : U32)) == Nat.add(U32.to_nat(position),U32.to_nat(count)) : Nat},+old: Members.All(Nat,{index => WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,previous,index) : Nat -> Data},Ranges.range(Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs)),0n)),+storage: Inputs.Contract<ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image))>,+readback: CallReadback<Holder,value,holder,limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),storage,local,count,offset,previous,next>,+cell: Nat,+positive: {Nat.is_lt(0n,cell_groups(outputs)) == True{} : Bool},+early: {Nat.is_lt(cell,Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs))) == True{} : Bool},+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +before: {PaddedCells.kept(outputs,Nat.mod(cell,cell_groups(outputs)),lane,TileLaws.select_lane(lane,Tree.read(Tile.Tile8,previous,U32.from_nat(cell)))) == PaddedCells.kept(outputs,Nat.mod(cell,cell_groups(outputs)),lane,written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,Nat.div(cell,cell_groups(outputs)),Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n)))) : F32}) -> {PaddedCells.kept(outputs,Nat.mod(cell,cell_groups(outputs)),lane,TileLaws.select_lane(lane,Tree.read(Tile.Tile8,next,U32.from_nat(cell)))) == PaddedCells.kept(outputs,Nat.mod(cell,cell_groups(outputs)),lane,written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,Nat.div(cell,cell_groups(outputs)),Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n)))) : F32}:
  +left = TileLaws.select_lane(lane,Tree.read(Tile.Tile8,next,U32.from_nat(cell)))
  +middle = TileLaws.select_lane(lane,Tree.read(Tile.Tile8,previous,U32.from_nat(cell)))
  Equal.trans(F32,PaddedCells.kept(outputs,Nat.mod(cell,cell_groups(outputs)),lane,left),PaddedCells.kept(outputs,Nat.mod(cell,cell_groups(outputs)),lane,middle),PaddedCells.kept(outputs,Nat.mod(cell,cell_groups(outputs)),lane,written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,Nat.div(cell,cell_groups(outputs)),Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n)))),
    PaddedCells.kept_agree(outputs,Nat.mod(cell,cell_groups(outputs)),lane,left,middle,Nat.is_lt(Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n)),outputs),{==},
      inside => Equal.cong(Tile.Tile8,F32,tile => TileLaws.select_lane(lane,tile),Tree.read(Tile.Tile8,next,U32.from_nat(cell)),Tree.read(Tile.Tile8,previous,U32.from_nat(cell)),
        old_lane_tile(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early,lane,small,inside))),
    before)

# A lane of a cell the call wrote holds the value of its position.
def new_lane_value(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> F32,+holder: Holder,+limits: Machine.Limits,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+groups_out: U32,+activation: Activations.Activation,+start: U32,+position: U32,+end: U32,+image: U32,+local: U32,+count: U32,+offset: U32,+previous: Tree.Tree<Tile.Tile8>,+next: Tree.Tree<Tile.Tile8>,+after: {Nat.is_le(U32.to_nat(start),U32.to_nat(position)) == True{} : Bool},+word_positive: {Nat.is_lt(0n,U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+plane_positive: {Nat.is_lt(0n,ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool},+plane_exact: {U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},+local_exact: {U32.to_nat(local) == Nat.mod(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+image_exact: {U32.to_nat(image) == Nat.div(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+offset_exact: {U32.to_nat(offset) == Nat.sub(U32.to_nat(position),U32.to_nat(start)) : Nat},+next_exact: {U32.to_nat((position + count : U32)) == Nat.add(U32.to_nat(position),U32.to_nat(count)) : Nat},+old: Members.All(Nat,{index => WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,previous,index) : Nat -> Data},Ranges.range(Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs)),0n)),+storage: Inputs.Contract<ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image))>,+readback: CallReadback<Holder,value,holder,limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),storage,local,count,offset,previous,next>,+cell: Nat,+positive: {Nat.is_lt(0n,cell_groups(outputs)) == True{} : Bool},+late: {Nat.is_lt(cell,Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs))) == False{} : Bool},
  +inside_cell: {Nat.is_lt(cell,Nat.mul(Nat.sub(U32.to_nat((position + count : U32)),U32.to_nat(start)),cell_groups(outputs))) == True{} : Bool},+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n)),outputs) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tree.read(Tile.Tile8,next,U32.from_nat(cell))) == written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,Nat.div(cell,cell_groups(outputs)),Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n))) : F32}:
  +at = Nat.div(cell,cell_groups(outputs))
  +channel = Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n))
  +ordered = {%Equal.sym(Nat,U32.to_nat(offset),Nat.sub(U32.to_nat(position),U32.to_nat(start)),offset_exact) : {Nat.is_le(_,at) == True{} : Bool}
    position_from(cell,Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs),positive,late) : {Nat.is_le(U32.to_nat(offset),at) == True{} : Bool}}
  +below = DivisionOrder.quotient_below(cell,Nat.sub(U32.to_nat((position + count : U32)),U32.to_nat(start)),cell_groups(outputs),positive,inside_cell)
  +step_inside = call_step_inside(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,after,offset_exact,next_exact,at,ordered,below)
  +step = Nat.sub(at,U32.to_nat(offset))
  %lane_place(lane,Nat.mod(cell,cell_groups(outputs)),small) : {TileLaws.select_lane(_,Tree.read(Tile.Tile8,next,U32.from_nat(cell))) == written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,at,channel) : F32}
  %lane_address(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),cell,lane,small,positive) : {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,next,_)) == written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,at,channel) : F32}
  %Order.restore_shift(U32.to_nat(offset),at,ordered) :
    {TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,next,Blocked.cell_address(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),channel,_))) == written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,at,channel) : F32}
  Equal.trans(F32,TileLaws.select_lane(Nat.mod(channel,8n),Tree.read(Tile.Tile8,next,Blocked.cell_address(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),channel,Nat.add(U32.to_nat(offset),step)))),
    Values.matrix_value(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),channel,Nat.div(Nat.add(U32.to_nat(local),step),ConvSpec.out_width(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)))),Nat.mod(Nat.add(U32.to_nat(local),step),ConvSpec.out_width(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image))))),
    written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,at,channel),
    readback_written(~Holder,~value,holder,limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),storage,local,count,offset,previous,next,readback,step,channel,step_inside,inside),
    moved_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,at,channel,ordered,step_inside))

def new_lane(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> F32,+holder: Holder,+limits: Machine.Limits,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+groups_out: U32,+activation: Activations.Activation,+start: U32,+position: U32,+end: U32,+image: U32,+local: U32,+count: U32,+offset: U32,+previous: Tree.Tree<Tile.Tile8>,+next: Tree.Tree<Tile.Tile8>,+after: {Nat.is_le(U32.to_nat(start),U32.to_nat(position)) == True{} : Bool},+word_positive: {Nat.is_lt(0n,U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+plane_positive: {Nat.is_lt(0n,ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool},+plane_exact: {U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},+local_exact: {U32.to_nat(local) == Nat.mod(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+image_exact: {U32.to_nat(image) == Nat.div(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+offset_exact: {U32.to_nat(offset) == Nat.sub(U32.to_nat(position),U32.to_nat(start)) : Nat},+next_exact: {U32.to_nat((position + count : U32)) == Nat.add(U32.to_nat(position),U32.to_nat(count)) : Nat},+old: Members.All(Nat,{index => WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,previous,index) : Nat -> Data},Ranges.range(Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs)),0n)),+storage: Inputs.Contract<ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image))>,+readback: CallReadback<Holder,value,holder,limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),storage,local,count,offset,previous,next>,+cell: Nat,+positive: {Nat.is_lt(0n,cell_groups(outputs)) == True{} : Bool},+late: {Nat.is_lt(cell,Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs))) == False{} : Bool},
  +inside_cell: {Nat.is_lt(cell,Nat.mul(Nat.sub(U32.to_nat((position + count : U32)),U32.to_nat(start)),cell_groups(outputs))) == True{} : Bool},+lane: Nat,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) -> {PaddedCells.kept(outputs,Nat.mod(cell,cell_groups(outputs)),lane,TileLaws.select_lane(lane,Tree.read(Tile.Tile8,next,U32.from_nat(cell)))) == PaddedCells.kept(outputs,Nat.mod(cell,cell_groups(outputs)),lane,written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,Nat.div(cell,cell_groups(outputs)),Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n)))) : F32}:
  PaddedCells.kept_agree(outputs,Nat.mod(cell,cell_groups(outputs)),lane,TileLaws.select_lane(lane,Tree.read(Tile.Tile8,next,U32.from_nat(cell))),written_value(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,Nat.div(cell,cell_groups(outputs)),Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n))),
    Nat.is_lt(Nat.add(lane,Nat.mul(Nat.mod(cell,cell_groups(outputs)),8n)),outputs),{==},
    inside => new_lane_value(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,late,inside_cell,lane,small,inside))

# A cell before the call keeps the tile of its position and group.
def old_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> F32,+holder: Holder,+limits: Machine.Limits,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+groups_out: U32,+activation: Activations.Activation,+start: U32,+position: U32,+end: U32,+image: U32,+local: U32,+count: U32,+offset: U32,+previous: Tree.Tree<Tile.Tile8>,+next: Tree.Tree<Tile.Tile8>,+after: {Nat.is_le(U32.to_nat(start),U32.to_nat(position)) == True{} : Bool},+word_positive: {Nat.is_lt(0n,U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+plane_positive: {Nat.is_lt(0n,ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool},+plane_exact: {U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},+local_exact: {U32.to_nat(local) == Nat.mod(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+image_exact: {U32.to_nat(image) == Nat.div(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+offset_exact: {U32.to_nat(offset) == Nat.sub(U32.to_nat(position),U32.to_nat(start)) : Nat},+next_exact: {U32.to_nat((position + count : U32)) == Nat.add(U32.to_nat(position),U32.to_nat(count)) : Nat},+old: Members.All(Nat,{index => WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,previous,index) : Nat -> Data},Ranges.range(Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs)),0n)),+storage: Inputs.Contract<ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image))>,+readback: CallReadback<Holder,value,holder,limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),storage,local,count,offset,previous,next>,+cell: Nat,+positive: {Nat.is_lt(0n,cell_groups(outputs)) == True{} : Bool},+early: {Nat.is_lt(cell,Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs))) == True{} : Bool},
  +before: WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,previous,cell)) -> WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,next,cell):
  PaddedCells.Valid{+lane0,+lane1,+lane2,+lane3,+lane4,+lane5,+lane6,+lane7} = before
  PaddedCells.Valid{old_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early,0n,{==},lane0),old_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early,1n,{==},lane1),old_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early,2n,{==},lane2),old_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early,3n,{==},lane3),old_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early,4n,{==},lane4),old_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early,5n,{==},lane5),old_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early,6n,{==},lane6),old_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early,7n,{==},lane7)}

# A cell of the task after the call holds the tile of its position and group.
def cell_cases(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> F32,+holder: Holder,+limits: Machine.Limits,+logical: Logical,+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+weight_list: List<&2,F32>,+bias_list: List<&2,F32>,+groups_out: U32,+activation: Activations.Activation,+start: U32,+position: U32,+end: U32,+image: U32,+local: U32,+count: U32,+offset: U32,+previous: Tree.Tree<Tile.Tile8>,+next: Tree.Tree<Tile.Tile8>,+after: {Nat.is_le(U32.to_nat(start),U32.to_nat(position)) == True{} : Bool},+word_positive: {Nat.is_lt(0n,U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+plane_positive: {Nat.is_lt(0n,ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool},+plane_exact: {U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},+local_exact: {U32.to_nat(local) == Nat.mod(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+image_exact: {U32.to_nat(image) == Nat.div(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},+grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},+offset_exact: {U32.to_nat(offset) == Nat.sub(U32.to_nat(position),U32.to_nat(start)) : Nat},+next_exact: {U32.to_nat((position + count : U32)) == Nat.add(U32.to_nat(position),U32.to_nat(count)) : Nat},+old: Members.All(Nat,{index => WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,previous,index) : Nat -> Data},Ranges.range(Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs)),0n)),+storage: Inputs.Contract<ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image))>,+readback: CallReadback<Holder,value,holder,limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),storage,local,count,offset,previous,next>,+cell: Nat,+inside: {Nat.is_lt(cell,Nat.mul(Nat.sub(U32.to_nat((position + count : U32)),U32.to_nat(start)),cell_groups(outputs))) == True{} : Bool},
  early: Bool,+early_exact: {Nat.is_lt(cell,Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs))) == early : Bool}) -> WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,next,cell):
  match early:
    case True{}:
      old_cell(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,DivisionOrder.right_factor_positive(Nat.sub(U32.to_nat((position + count : U32)),U32.to_nat(start)),cell_groups(outputs),Order.le_lt_trans(0n,cell,Nat.mul(Nat.sub(U32.to_nat((position + count : U32)),U32.to_nat(start)),cell_groups(outputs)),Order.zero_le(cell),inside)),early_exact,
        Members.all_member(Nat,{index => WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,previous,index) : Nat -> Data},cell,Ranges.range(Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs)),0n),old,
          Members.range_member(Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs)),0n,cell,early_exact)))
    case False{}:
      +positive = DivisionOrder.right_factor_positive(Nat.sub(U32.to_nat((position + count : U32)),U32.to_nat(start)),cell_groups(outputs),Order.le_lt_trans(0n,cell,Nat.mul(Nat.sub(U32.to_nat((position + count : U32)),U32.to_nat(start)),cell_groups(outputs)),Order.zero_le(cell),inside))
      PaddedCells.Valid{new_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early_exact,inside,0n,{==}),new_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early_exact,inside,1n,{==}),new_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early_exact,inside,2n,{==}),new_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early_exact,inside,3n,{==}),new_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early_exact,inside,4n,{==}),new_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early_exact,inside,5n,{==}),new_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early_exact,inside,6n,{==}),new_lane(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,positive,early_exact,inside,7n,{==})}

# The cells of a task after a call: those before it, and what the call wrote
# and read back.
type NextCells<-Holder: Data,-value: Holder -> U32 -> Tile.Tile8,-Logical: Data,-original: Logical -> U32 -> F32> is Data:
  NextCells{holder: Holder,limits: Machine.Limits,logical: Logical,channels: Nat,outputs: Nat,height: Nat,width: Nat,kernel: Nat,stride: Nat,padding: Nat,weight_list: List<&2,F32>,bias_list: List<&2,F32>,groups_out: U32,activation: Activations.Activation,start: U32,position: U32,end: U32,image: U32,local: U32,count: U32,offset: U32,previous: Tree.Tree<Tile.Tile8>,next: Tree.Tree<Tile.Tile8>,after: {Nat.is_le(U32.to_nat(start),U32.to_nat(position)) == True{} : Bool},word_positive: {Nat.is_lt(0n,U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},plane_positive: {Nat.is_lt(0n,ImageProblems.plane(height,width,kernel,stride,padding)) == True{} : Bool},plane_exact: {U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1)) == ImageProblems.plane(height,width,kernel,stride,padding) : Nat},local_exact: {U32.to_nat(local) == Nat.mod(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},image_exact: {U32.to_nat(image) == Nat.div(U32.to_nat(position),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) : Nat},grid: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(count)),U32.to_nat(U32.max(WindowPipeline.plane_of(record(ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,0n),groups_out,activation)),1))) == True{} : Bool},offset_exact: {U32.to_nat(offset) == Nat.sub(U32.to_nat(position),U32.to_nat(start)) : Nat},next_exact: {U32.to_nat((position + count : U32)) == Nat.add(U32.to_nat(position),U32.to_nat(count)) : Nat},old: Members.All(Nat,{index => WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,previous,index) : Nat -> Data},Ranges.range(Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs)),0n)),storage: Inputs.Contract<ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image))>,readback: CallReadback<Holder,value,holder,limits,ImageProblems.problem(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,U32.to_nat(image)),storage,local,count,offset,previous,next>}

def next_size(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> F32,context: NextCells<Holder,value,Logical,original>) -> Nat:
  NextCells{+holder,+limits,+logical,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weight_list,+bias_list,+groups_out,+activation,+start,+position,+end,+image,+local,+count,+offset,+previous,+next,+after,+word_positive,+plane_positive,+plane_exact,+local_exact,+image_exact,+grid,+offset_exact,+next_exact,+old,+storage,+readback} = context
  Nat.mul(Nat.sub(U32.to_nat((position + count : U32)),U32.to_nat(start)),cell_groups(outputs))

def NextCell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> F32,+context: NextCells<Holder,value,Logical,original>,+cell: Nat) -> Data:
  NextCells{+holder,+limits,+logical,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weight_list,+bias_list,+groups_out,+activation,+start,+position,+end,+image,+local,+count,+offset,+previous,+next,+after,+word_positive,+plane_positive,+plane_exact,+local_exact,+image_exact,+grid,+offset_exact,+next_exact,+old,+storage,+readback} = context
  WrittenCell(~Logical,~original,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,start,next,cell)

def next_cell(~Holder: Data,~value: Holder -> U32 -> Tile.Tile8,~Logical: Data,~original: Logical -> U32 -> F32,+context: NextCells<Holder,value,Logical,original>,+cell: Nat,
  +inside: {Nat.is_lt(cell,next_size(~Holder,~value,~Logical,~original,context)) == True{} : Bool}) -> NextCell(~Holder,~value,~Logical,~original,context,cell):
  NextCells{+holder,+limits,+logical,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weight_list,+bias_list,+groups_out,+activation,+start,+position,+end,+image,+local,+count,+offset,+previous,+next,+after,+word_positive,+plane_positive,+plane_exact,+local_exact,+image_exact,+grid,+offset_exact,+next_exact,+old,+storage,+readback} = context
  cell_cases(~Holder,~value,~Logical,~original,holder,limits,logical,channels,outputs,height,width,kernel,stride,padding,weight_list,bias_list,groups_out,activation,start,position,end,image,local,count,offset,previous,next,after,word_positive,plane_positive,plane_exact,local_exact,image_exact,grid,offset_exact,next_exact,old,storage,readback,cell,inside,Nat.is_lt(cell,Nat.mul(Nat.sub(U32.to_nat(position),U32.to_nat(start)),cell_groups(outputs))),{==})

def band_first_exact(-problem: ConvSpec.Problem,-image: U32,-local: U32,-count: U32,words: BandWords<problem,image,local,count>) ->
  {U32.to_nat(band_first(problem,image,local)) == Nat.add(Nat.mul(U32.to_nat(image),Cells.count(problem,Guards.height(problem))),Cells.count(problem,U32.to_nat(BandCover.first_word(problem,local)))) : Nat}:
  BandWords{first_exact,cells_exact} = words
  first_exact

def band_cells_exact(-problem: ConvSpec.Problem,-image: U32,-local: U32,-count: U32,words: BandWords<problem,image,local,count>) ->
  {U32.to_nat(band_cells(problem,local,count)) == Cells.count(problem,U32.to_nat(BandCover.rows_word(problem,local,count))) : Nat}:
  BandWords{first_exact,cells_exact} = words
  cells_exact
