# Pooling over the last two axes of a tiled tensor. A usable tensor of rank
# three or more with a valid pool takes its lanes on the third axis from the
# end with the groups innermost, the image cells of the window; the cells
# pool with the operation at every lane from a tile of pad, and the lanes
# past the true extent are +0. A tensor of rank two pools as one of rank
# three with a leading axis of extent 1. Any other input fails with an
# invalid argument, or keeps its problem.
import Base
import ./kernel_tile.bend as Tile
import ./tensor_access.bend as Access
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_window.bend as Window
import ./tensor_tiled.bend as Tiled
import ./tensor_tiled_map.bend as TiledMap
import ./tensor_tiled_layout.bend as TiledLayout

# The output cells of the input cells. Rows and steps of an output cell:
# every tap loads a sample tile and runs one lane operation and thirteen
# steps of addressing and bounds, every kernel row one lane operation and
# twelve steps, and the cell stores one tile with twenty steps around it.
def windowed(~operation: F32 -> F32 -> F32,+pad: F32,+taps: Window.Taps,+lanes: U32,+out_height: U32,+out_width: U32,+outputs: List<&2,U32>,+workers: U32,
  cells: Tensors.Tensor<Tile.Tile8>) -> Tensors.Tensor<Tile.Tile8>:
  Window.Taps{+kernel,stride,padding,height,width,groups} = taps
  Tensors.Tensor{pieces,+shape,fill,status,cell_workers} = cells
  +rows = U32.to_f32(kernel)
  Tensors.Tensor{Window.windowed(~TiledMap.pairs(~operation),Window.Windowing{Window.Pooling{Tile.filled(pad)},taps,lanes,out_height,out_width,Grid.grid(outputs,Grid.piece_target())},
      Tensors.grid_of(shape),(rows * rows * 2.0 + rows + 1.0 : F32),(rows * rows * 13.0 + rows * 12.0 + 20.0 : F32),workers,0,pieces,Tensors.buffer(~Tile.Tile8,Tile.filled(0.0),0)),
    outputs,Tiled.zero_tile(),status,workers}

def pooled(~operation: F32 -> F32 -> F32,+pad: F32,+pool: Access.Pool,+shape: List<&2,U32>,+workers: U32,tiled: Tiled.Tiled) -> Tiled.Tiled:
  Access.Pool{+height,+width,+kernel,+stride,+padding,+out_height,+out_width} = pool
  +axis = Tiled.lane_axis(List.length(&2,U32,shape))
  +lanes = Grid.lanes_at(shape,axis)
  Tiled.Tiled{windowed(~operation,pad,Window.Taps{kernel,stride,padding,height,width,Tiled.groups(lanes)},lanes,out_height,out_width,
      Tiled.cell_shape(Window.output_shape(shape,pool),axis,Grid.Innermost{}),workers,Tiled.cells_of(Tiled.arranged(tiled,axis,Grid.Innermost{}))),
    axis,lanes,Grid.Innermost{}}

def routed(~operation: F32 -> F32 -> F32,name: String,route: Bool,+pad: F32,+pool: Access.Pool,+shape: List<&2,U32>,+workers: U32,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match route:
    case True{}: pooled(~operation,pad,pool,shape,workers,tiled)
    case False{}: Tiled.rejected(name,tiled)

def ranked(~operation: F32 -> F32 -> F32,name: String,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,tiled: Tiled.Tiled,+shape: List<&2,U32>,
  status: Tensors.Status,+workers: U32) -> Tiled.Tiled:
  +pool = Window.geometry(shape,kernel,stride,padding)
  routed(~operation,name,Tensors.usable(status) && Window.valid(shape,pool) && Nat.is_le(3n,List.length(&2,U32,shape)),pad,pool,shape,workers,tiled)

# The pool on the cells of a tensor of rank three or more.
def cells_pool(~operation: F32 -> F32 -> F32,name: String,+pad: F32,tiled: Tiled.Tiled,+kernel: U32,+stride: U32,+padding: U32) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers => ranked(~operation,name,pad,kernel,stride,padding,tiled,shape,status,workers))

# A valid pool of rank two pools the tensor with a leading axis of extent 1.
def planar(~operation: F32 -> F32 -> F32,name: String,+shape: List<&2,U32>,valid: Bool,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,tiled: Tiled.Tiled) -> Tiled.Tiled:
  match shape valid:
    case Con{+height,Con{+width,Nil{}}} True{}:
      TiledLayout.reshape(cells_pool(~operation,name,pad,TiledLayout.reshape(tiled,[1,height,width]),kernel,stride,padding),
        Window.output_shape(shape,Window.geometry(shape,kernel,stride,padding)))
    case shape valid: Tiled.rejected(name,tiled)

def dispatched(~operation: F32 -> F32 -> F32,name: String,three: Bool,+pad: F32,+kernel: U32,+stride: U32,+padding: U32,tiled: Tiled.Tiled,+shape: List<&2,U32>,
  status: Tensors.Status,+workers: U32) -> Tiled.Tiled:
  match three:
    case True{}: ranked(~operation,name,pad,kernel,stride,padding,tiled,shape,status,workers)
    case False{}: planar(~operation,name,shape,Tensors.usable(status) && Window.valid(shape,Window.geometry(shape,kernel,stride,padding)),pad,kernel,stride,padding,tiled)

def pool2d(~operation: F32 -> F32 -> F32,name: String,+pad: F32,tiled: Tiled.Tiled,+kernel: U32,+stride: U32,+padding: U32) -> Tiled.Tiled:
  Tiled.with_metadata(~Tiled.Tiled,tiled,tiled => shape => status => workers =>
    +shape = shape
    dispatched(~operation,name,Nat.is_le(3n,List.length(&2,U32,shape)),pad,kernel,stride,padding,tiled,shape,status,workers))
