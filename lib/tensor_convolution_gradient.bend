# Gradients of conv2d in a fixed FP32 order. Every function gives the
# tensors it only reads back to its caller.
import Base
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_convolution.bend as TensorConvolution
import ./kernel_input_gradient.bend as InputKernel
import ./kernel_shape.bend as KernelShape
import ./storage_buffer.bend as Storage
import ./default_machine_limits.bend as Default
import ./machine_limits.bend as Machine
import ./tensor_parameter_gradient.bend as ParameterGradient
import ./tensor_tiled.bend as Tiled
import ./kernel_tile.bend as Tile

def Cells() -> Type:
  Tensors.Tensor<Tile.Tile8>

def Tensor() -> Type:
  Tiled.Tiled

type Arguments is Data:
  Arguments{geometry: InputKernel.Geometry,batch: U32,batched: Bool}

def input_dimensions(arguments: Arguments) -> KernelShape.Dimensions:
  Arguments{InputKernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,batched} = arguments
  KernelShape.Dimensions{U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)}

def input_shape(arguments: Arguments) -> List<&2,U32>:
  Arguments{InputKernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,batched} = arguments
  TensorConvolution.output_shape(batched,batch,c,h,w)

# The cells of the input gradient: the positions of every image, the groups
# of a position side by side.
def input_cell_shape(arguments: Arguments) -> List<&2,U32>:
  Arguments{InputKernel.Geometry{+c,o,h,w,k,s,p,gh,gw},batch,batched} = arguments
  TensorConvolution.image_cells(batched,batch,h,w,Tiled.groups(c))

# The input gradient as cells with the channels in their lanes.
def input_cells(+arguments: Arguments,+workers: U32,output: Pieces.Pieces(~Tile.Tile8)) -> Tensor():
  Arguments{InputKernel.Geometry{+c,o,h,w,k,s,p,gh,gw},batch,+batched} = arguments
  Tiled.Tiled{Tensors.Tensor{output,input_cell_shape(arguments),Tiled.zero_tile(),Tensors.Usable{},workers},TensorConvolution.lane_of(batched),c,Grid.Innermost{}}

# The limits are an argument so that a proof can keep them symbolic.
def input_valid_with(+arguments: Arguments,+d_cells: U32,+weight_cells: U32,+limits: Machine.Limits) -> Bool:
  Arguments{+geometry,+batch,batched} = arguments
  InputKernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  Machine.Limits{dimension,+capacity,coordinate,half_word,facts} = limits
  +dimensions = input_dimensions(arguments)
  +input_length = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w)))
  +logical_weights = Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))))
  KernelShape.valid(dimensions,input_length,logical_weights,U32.to_nat(o)) &&
    KernelShape.direct_bounds(dimension,capacity,coordinate,dimensions,input_length,logical_weights) &&
    Nat.is_eq(U32.to_nat(gh),KernelShape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))) &&
    Nat.is_eq(U32.to_nat(gw),KernelShape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))) &&
    Nat.is_eq(U32.to_nat(d_cells),Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(Tiled.groups(o)),Nat.mul(U32.to_nat(gh),U32.to_nat(gw))))) &&
    Nat.is_eq(U32.to_nat(weight_cells),Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(Tiled.groups(c)),Nat.mul(U32.to_nat(k),U32.to_nat(k))))) &&
    Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(gh),U32.to_nat(gw)))),U32.to_nat(capacity)) &&
    Nat.is_le(Nat.mul(U32.to_nat(batch),input_length),U32.to_nat(capacity)) &&
    InputKernel.fits(geometry,capacity)

# The weights come back as cells of the groups outermost.
def weights_back(+shape: List<&2,U32>,+lanes: U32,+fill: Tile.Tile8,status: Tensors.Status,+workers: U32,weights: Storage.Buffer<Tile.Tile8>) -> Tensor():
  Tiled.Tiled{Tensors.whole(~Tile.Tile8,weights,shape,fill,status,workers),1n,lanes,Grid.Outermost{}}

# What the call keeps of the weights besides their cells.
type Kept is Data:
  Kept{shape: List<&2,U32>,lanes: U32,fill: Tile.Tile8,status: Tensors.Status,workers: U32}

def kept_shape(kept: Kept) -> List<&2,U32>:
  Kept{shape,lanes,fill,status,workers} = kept
  shape

def kept_lanes(kept: Kept) -> U32:
  Kept{shape,lanes,fill,status,workers} = kept
  lanes

def kept_workers(kept: Kept) -> U32:
  Kept{shape,lanes,fill,status,workers} = kept
  workers

def kept_back(kept: Kept,weights: Storage.Buffer<Tile.Tile8>) -> Tensor():
  Kept{+shape,+lanes,+fill,status,+workers} = kept
  weights_back(shape,lanes,fill,status,workers,weights)

def input_result(+arguments: Arguments,+workers: U32,+kept: Kept,result: InputKernel.Outcome) -> Tensor() & Tensor():
  InputKernel.Outcome{weights,output} = result
  (input_cells(arguments,workers,output),kept_back(kept,weights))

# A task holds whole images of d and of the input gradient when the pieces
# of both do.
def input_whole(+arguments: Arguments,+d_grid: Grid.Grid) -> U32:
  Arguments{InputKernel.Geometry{+c,+o,+h,+w,k,s,p,+gh,+gw},batch,+batched} = arguments
  TensorConvolution.whole_cells(Bool.pick(U32,batched,1,0),d_grid,Tensors.grid_of(input_cell_shape(arguments)),(gh * gw * Tiled.groups(o) : U32),(h * w * Tiled.groups(c) : U32))

def input_checked(accepted: Bool,+arguments: Arguments,+d_shape: List<&2,U32>,gradient: Pieces.Pieces(~Tile.Tile8),+workers: U32,weights: Storage.Buffer<Tile.Tile8>,+kept: Kept) ->
  Tensor() & Tensor():
  match accepted:
    case False{}:
      (Tiled.failed("conv2d_input_gradient",Tensors.InvalidArgument{input_shape(arguments)},input_shape(arguments),workers),kept_back(kept,weights))
    case True{}:
      Arguments{+geometry,batch,batched} = arguments
      +d_grid = Tensors.grid_of(d_shape)
      input_result(arguments,workers,kept,
        InputKernel.run(geometry,input_cell_shape(arguments),d_grid,input_whole(arguments,d_grid),U32.max(workers,U32.max(kept_workers(kept),1)),gradient,weights))

def input_geometry(+stride: U32,+padding: U32,+height: U32,+width: U32,+limits: Machine.Limits,shape: Maybe<&2,TensorConvolution.InputShape>,weight_logical: List<&2,U32>,+d_shape: List<&2,U32>,
  gradient: Pieces.Pieces(~Tile.Tile8),+workers: U32,weights: Storage.Buffer<Tile.Tile8>,+kept: Kept) -> Tensor() & Tensor():
  match shape weight_logical:
    case Some{TensorConvolution.InputShape{+batch,+batched,+outputs,+gh,+gw}} Con{wo,Con{+channels,Con{+kernel,Con{kw,Nil{}}}}}:
      +arguments = {Arguments{InputKernel.Geometry{channels,outputs,height,width,kernel,stride,padding,gh,gw},batch,batched} : Arguments}
      input_checked(U32.is_eq(outputs,wo) && U32.is_eq(kernel,kw) && input_valid_with(arguments,Tensors.count_or_zero(d_shape),Tensors.count_or_zero(kept_shape(kept)),limits),arguments,d_shape,gradient,
        workers,weights,kept)
    case shape weight_logical:
      +weight_shape = kept_shape(kept)
      (Tiled.failed("conv2d_input_gradient",Tensors.InvalidShape{weight_shape},weight_shape,workers),kept_back(kept,weights))

def input_status(status: Tensors.Status,+stride: U32,+padding: U32,+height: U32,+width: U32,gradient: Pieces.Pieces(~Tile.Tile8),+d_cells: List<&2,U32>,+d_lanes: U32,
  +d_fill: Tile.Tile8,+workers: U32,weights: Storage.Buffer<Tile.Tile8>,+kept: Kept) -> Tensor() & Tensor():
  match status:
    case Tensors.Failed{problem}:
      (Tiled.Tiled{Tensors.Tensor{gradient,d_cells,d_fill,Tensors.Failed{problem},workers},Tiled.lane_axis(List.length(&2,U32,d_cells)),d_lanes,Grid.Innermost{}},kept_back(kept,weights))
    case Tensors.Usable{}:
      input_geometry(stride,padding,height,width,Default.limits(),
        TensorConvolution.input_shape(Grid.logical_shape(d_cells,Grid.Layout{Tiled.lane_axis(List.length(&2,U32,d_cells)),Grid.Innermost{}},d_lanes)),
        Grid.logical_shape(kept_shape(kept),Grid.Layout{1n,Grid.Outermost{}},kept_lanes(kept)),d_cells,gradient,workers,weights,kept)

# d with the channels in its lanes, the groups innermost, and the weights
# with the input channels in their lanes, the groups outermost.
def input_values(d: Cells(),+d_lanes: U32,weights: Cells(),+weight_lanes: U32,+stride: U32,+padding: U32,+height: U32,+width: U32) -> Tensor() & Tensor():
  Tensors.Tensor{gradient,+d_cells,+d_fill,d_status,+workers} = d
  Tensors.with_whole(~Tile.Tile8,~(Tensor() & Tensor()),weights,weights => weight_shape => weight_fill => weight_status => weight_workers =>
    +weight_status = weight_status
    input_status(Tensors.first_failure(d_status,weight_status),stride,padding,height,width,gradient,d_cells,d_lanes,d_fill,workers,weights,
      Kept{weight_shape,weight_lanes,weight_fill,weight_status,weight_workers}))

def input_aligned(d: Tensor(),weights: Tensor(),stride: U32,padding: U32,height: U32,width: U32) -> Tensor() & Tensor():
  Tiled.Tiled{d_cells,d_axis,+d_lanes,d_position} = d
  Tiled.Tiled{weight_cells,weight_axis,+weight_lanes,weight_position} = weights
  input_values(d_cells,d_lanes,weight_cells,weight_lanes,stride,padding,height,width)

# The cells of x and d with their lanes on the channels, the axis before the
# rows, the groups innermost.
def channel_lanes(pair: Tensor() & List<&2,U32>) -> Tensor():
  (tiled,shape) = pair
  Tiled.arranged(tiled,Tiled.lane_axis(List.length(&2,U32,shape)),Grid.Innermost{})

# d [N,O,Ho,Wo] or [O,Ho,Wo] and weights [O,C,k,k] of a convolution of an
# input of the given height and width: (the gradient of that input, the
# weights with the input channels in their lanes). Several input sizes give
# the same output size when the stride is above 1, so height and width are
# given.
def input(d: Tensor(),weights: Tensor(),stride: U32,padding: U32,height: U32,width: U32) -> Tensor() & Tensor():
  input_aligned(channel_lanes(Tiled.shape(d)),Tiled.arranged(weights,1n,Grid.Outermost{}),stride,padding,height,width)

# The group length of the weight and bias gradient contract.
def group_length() -> U32:
  256

def weight_result(+x_axis: Nat,+x_lanes: U32,+d_axis: Nat,+d_lanes: U32,result: Tensor() & (Cells() & Cells())) -> Tensor() & (Tensor() & Tensor()):
  (gradient,cells) = result
  (x,d) = cells
  (gradient,(Tiled.Tiled{x,x_axis,x_lanes,Grid.Innermost{}},Tiled.Tiled{d,d_axis,d_lanes,Grid.Innermost{}}))

def weight_cells(x: Tensor(),d: Tensor(),+kernel: U32,+stride: U32,+padding: U32,+group: U32) -> Tensor() & (Tensor() & Tensor()):
  Tiled.Tiled{x_cells,+x_axis,+x_lanes,x_position} = x
  Tiled.Tiled{d_cells,+d_axis,+d_lanes,d_position} = d
  weight_result(x_axis,x_lanes,d_axis,d_lanes,ParameterGradient.weight(x_cells,x_lanes,d_cells,d_lanes,kernel,stride,padding,group))

# x [N,C,H,W], d [N,O,Ho,Wo], or both without their batch axes: (weight
# gradient [O,C,k,k] with the outputs in its lanes, the groups outermost,
# (x,d)), read from the cells of the channels, its sums in groups of group
# positions.
def weight_grouped(x: Tensor(),d: Tensor(),kernel: U32,stride: U32,padding: U32,group: U32) -> Tensor() & (Tensor() & Tensor()):
  weight_cells(channel_lanes(Tiled.shape(x)),channel_lanes(Tiled.shape(d)),kernel,stride,padding,group)

def weight(x: Tensor(),d: Tensor(),kernel: U32,stride: U32,padding: U32) -> Tensor() & (Tensor() & Tensor()):
  weight_grouped(x,d,kernel,stride,padding,group_length())

def bias_result(+axis: Nat,+lanes: U32,result: Tensor() & Cells()) -> Tensor() & Tensor():
  (gradient,d) = result
  (gradient,Tiled.Tiled{d,axis,lanes,Grid.Innermost{}})

def bias_cells(d: Tensor(),+group: U32) -> Tensor() & Tensor():
  Tiled.Tiled{cells,+axis,+lanes,position} = d
  bias_result(axis,lanes,ParameterGradient.bias(cells,lanes,group))

# d [N,O,Ho,Wo] or [O,Ho,Wo]: (bias gradient [O] with the outputs in its
# lanes, d), read from the cells of the channels in the order of the
# contract, its sums in groups of group positions.
def bias(d: Tensor()) -> Tensor() & Tensor():
  bias_cells(channel_lanes(Tiled.shape(d)),group_length())
