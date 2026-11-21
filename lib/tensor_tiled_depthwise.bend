# Depthwise convolution of tiled tensors. An input of shape [N,]C,H,W takes
# its lanes on the channels with the groups innermost, the image cells of the
# window, and the weights of shape C,1,K,K and the bias of shape C take
# theirs on the channel axis with the groups outermost, so lane l of a cell of
# group g convolves channel 8g + l with its own window, weights and bias. The
# result keeps the lanes on the channels with the groups innermost, +0 past
# the true channel count.
import Base
import ./kernel_tile.bend as Tile
import ./storage_buffer.bend as Storage
import ./tensor_access.bend as Access
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_window.bend as Window
import ./tensor_tiled.bend as Tiled
import ./tensor_tiled_map.bend as TiledMap
import ./default_machine_limits.bend as Default

def channels(shape: List<&2,U32>) -> Maybe<&2,U32>:
  match shape:
    case [channels,height,width]: Some{channels}
    case [batch,channels,height,width]: Some{channels}
    case shape: None{}

def kernel_when(valid: Bool,+kernel: U32) -> Maybe<&2,U32>:
  match valid:
    case True{}: Some{kernel}
    case False{}: None{}

# The weights and the bias of every channel group fit one buffer of the
# machine.
def parameters_fit(+channels: U32,+kernel: U32) -> Bool:
  +groups = Tiled.groups(channels)
  U32.is_le((groups * kernel * kernel + groups : U32),Default.capacity())

# The kernel extent of weights of shape C,1,K,K and a bias of shape C for an
# input of C channels whose parameters fit the machine.
def kernel_of(channels: Maybe<&2,U32>,weights: List<&2,U32>,bias: List<&2,U32>) -> Maybe<&2,U32>:
  match channels weights bias:
    case Some{+channels} Con{+weight_channels,Con{+one,Con{+kernel,Con{+kernel_width,Nil{}}}}} Con{+bias_channels,Nil{}}:
      kernel_when(U32.is_eq(channels,weight_channels) && U32.is_eq(channels,bias_channels) && U32.is_eq(one,1) && U32.is_eq(kernel,kernel_width) && parameters_fit(channels,kernel),kernel)
    case channels weights bias: None{}

def failed(+name: String,+shape: List<&2,U32>,status: Tensors.Status,+workers: U32) -> Tiled.Tiled:
  match status:
    case Tensors.Usable{}: Tiled.failed(name,Tensors.InvalidArgument{shape},shape,workers)
    case Tensors.Failed{problem}: Tiled.failed_as(problem,shape,workers)

def parameters_of(weights: Result<&2,&2,Tensors.Problem,List<&2,Tile.Tile8>>,bias: Result<&2,&2,Tensors.Problem,List<&2,Tile.Tile8>>) ->
  Result<&2,&2,Tensors.Problem,List<&2,Tile.Tile8>>:
  match weights bias:
    case Done{weights} Done{bias}: Done{List.append(&2,Tile.Tile8,weights,bias)}
    case Fail{problem} bias: Fail{problem}
    case Done{weights} Fail{problem}: Fail{problem}

# The output cells of the input cells. Rows and steps of an output cell:
# every tap loads a sample tile and a weight tile and runs one lane multiply,
# one lane add and thirteen steps of addressing and bounds, every kernel row
# twelve steps, and the cell loads its bias tile and stores one tile with
# twenty steps around them.
def convolved(+taps: Window.Taps,+lanes: U32,+out_height: U32,+out_width: U32,+outputs: List<&2,U32>,+workers: U32,parameters: List<&2,Tile.Tile8>,
  cells: Tensors.Tensor<Tile.Tile8>) -> Tensors.Tensor<Tile.Tile8>:
  Window.Taps{+kernel,stride,padding,height,width,+groups} = taps
  Tensors.Tensor{pieces,+shape,fill,status,cell_workers} = cells
  +count = (groups * kernel * kernel + groups : U32)
  +rows = U32.to_f32(kernel)
  Tensors.Tensor{Window.windowed(~TiledMap.pairs(~F32.add),Window.Windowing{Window.Weighting{},taps,lanes,out_height,out_width,Grid.grid(outputs,Grid.piece_target())},
      Tensors.grid_of(shape),(rows * rows * 4.0 + 2.0 : F32),(rows * rows * 13.0 + rows * 12.0 + 20.0 : F32),workers,count,pieces,
      Storage.write_at(~Tile.Tile8,parameters,0,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),count))),
    outputs,Tiled.zero_tile(),status,workers}

def run(+taps: Window.Taps,+axis: Nat,+lanes: U32,+out_height: U32,+out_width: U32,+outputs: List<&2,U32>,+workers: U32,
  parameters: Result<&2,&2,Tensors.Problem,List<&2,Tile.Tile8>>,cells: Tensors.Tensor<Tile.Tile8>) -> Tiled.Tiled:
  match parameters:
    case Done{parameters}:
      Tiled.Tiled{convolved(taps,lanes,out_height,out_width,Tiled.cell_shape(outputs,axis,Grid.Innermost{}),workers,parameters,cells),axis,lanes,Grid.Innermost{}}
    case Fail{problem}: Tiled.failed_as(problem,outputs,workers)

def planned(+pool: Access.Pool,+shape: List<&2,U32>,+workers: U32,input: Tiled.Tiled,weights: Tiled.Tiled,bias: Tiled.Tiled) -> Tiled.Tiled:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +axis = Tiled.lane_axis(List.length(&2,U32,shape))
  +lanes = Grid.lanes_at(shape,axis)
  run(Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},axis,lanes,out_height,out_width,Window.output_shape(shape,pool),workers,
    parameters_of(Tiled.listed(weights),Tiled.listed(bias)),Tiled.cells_of(Tiled.arranged(input,axis,Grid.Innermost{})))

def checked(valid: Bool,+pool: Access.Pool,+shape: List<&2,U32>,+workers: U32,input: Tiled.Tiled,weights: Tiled.Tiled,bias: Tiled.Tiled) -> Tiled.Tiled:
  match valid:
    case True{}: planned(pool,shape,workers,input,weights,bias)
    case False{}: failed("depthwise_conv2d",shape,Tensors.Usable{},workers)

def shaped(status: Tensors.Status,kernel: Maybe<&2,U32>,+shape: List<&2,U32>,+stride: U32,+padding: U32,+workers: U32,
  input: Tiled.Tiled,weights: Tiled.Tiled,bias: Tiled.Tiled) -> Tiled.Tiled:
  match status kernel:
    case Tensors.Usable{} Some{+kernel}:
      +pool = Window.geometry(shape,kernel,stride,padding)
      checked(Window.valid(shape,pool),pool,shape,workers,input,weights,bias)
    case Tensors.Usable{} None{}: failed("depthwise_conv2d",shape,Tensors.Usable{},workers)
    case Tensors.Failed{problem} kernel: failed("depthwise_conv2d",shape,Tensors.Failed{problem},workers)

# Every channel of an input of shape [N,]C,H,W convolves with its own weights
# of shape 1,K,K and adds its own bias, as conv2d of one channel does; the
# result keeps the channels of the input.
def depthwise_conv2d(input: Tiled.Tiled,weights: Tiled.Tiled,bias: Tiled.Tiled,+stride: U32,+padding: U32) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,input,input => shape => status => workers =>
    Tiled.with_metadata(~Tiled.Tiled,weights,weights => weight_shape => weight_status => weight_workers =>
      Tiled.with_metadata(~Tiled.Tiled,bias,bias => bias_shape => bias_status => bias_workers =>
        +shape = shape
        shaped(Tensors.first_failure(status,Tensors.first_failure(weight_status,bias_status)),kernel_of(channels(shape),weight_shape,bias_shape),shape,stride,padding,
          U32.max(workers,U32.max(weight_workers,bias_workers)),input,weights,bias))))
