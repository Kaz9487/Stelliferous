# The products of left [count,rows,inner] with a shared right [outputs,inner] or a
# right of every image [count,inner,outputs], followed by bias [outputs].
# The left takes its lanes on the inner axis. The right takes its lanes on
# the outputs, with the groups outermost for shared weights and innermost for
# image panels. The result [count,rows,outputs] has the output lanes, with its
# groups outermost for the shared-group schedule and innermost otherwise.
import Base
import ./kernel_tile.bend as Tile
import ./kernel_gemm_buffer.bend as GemmBuffer
import ./machine_limits.bend as Machine
import ./storage_buffer.bend as Storage
import ./tensor.bend as Tensors
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_tiled.bend as Tiled
import ./tensor_tiled_map.bend as TiledMap
import ./tensor_convolution.bend as Convolution
import ./tensor_point_products.bend as PointProducts

# The lanes of the last group of rows and of outputs, the inner tiles and the
# result fit the machine.
def supported(+count: U32,+rows: U32,+inner: U32,+outputs: U32,+limits: Machine.Limits) -> Bool:
  +capacity = U32.to_nat(Machine.capacity(limits))
  U32.is_lt(0,count) && U32.is_lt(0,rows) && U32.is_lt(0,inner) && U32.is_lt(0,outputs) &&
    Nat.is_le(Nat.add(U32.to_nat(rows),7n),capacity) && Nat.is_le(Nat.add(U32.to_nat(outputs),7n),capacity) && Nat.is_le(U32.to_nat(inner),capacity) &&
    Nat.is_le(Nat.mul(U32.to_nat(count),Nat.mul(U32.to_nat(outputs),U32.to_nat(rows))),capacity)

def innermost(tiled: Tiled.Tiled,+axis: Nat) -> Tensors.Tensor<Tile.Tile8>:
  Tiled.cells_of(Tiled.arranged(tiled,axis,Grid.Innermost{}))

# Shared weights take their lanes along outputs, with complete groups adjacent.
def weight_cells(right: Tiled.Tiled) -> Tensors.Tensor<Tile.Tile8>:
  Tiled.cells_of(Tiled.arranged(right,0n,Grid.Outermost{}))

def result(+count: U32,+rows: U32,+outputs: U32,status: Tensors.Status,cells: Tensors.Tensor<Tile.Tile8>) -> Tiled.Tiled:
  Tiled.Tiled{Tensors.with_status(~Tile.Tile8,status,Tensors.relabeled(~Tile.Tile8,cells,[count,rows,Tiled.groups(outputs)])),2n,outputs,Grid.Innermost{}}

type Schedule is Data:
  Window{}
  SharedGroups{}
  ImagePanels{}

def shared_schedule(+count: U32,+rows: U32,+inner: U32,+outputs: U32) -> Schedule:
  Bool.pick(Schedule,U32.is_eq(count,1) && U32.is_zero((inner % 16 : U32)) && U32.is_le(Tiled.groups(rows),Tiled.groups(outputs)),SharedGroups{},Window{})

def schedule(panels: Bool,+count: U32,+rows: U32,+inner: U32,+outputs: U32) -> Schedule:
  match panels:
    case True{}: ImagePanels{}
    case False{}: shared_schedule(count,rows,inner,outputs)

def panel_products(whole: Bool,+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,left: Pieces.Pieces(~Tile.Tile8),right: Pieces.Pieces(~Tile.Tile8),
  bias: Storage.Buffer<Tile.Tile8>) -> Pieces.Pieces(~Tile.Tile8):
  match whole:
    case True{}: PointProducts.panels(~GemmBuffer.point_keeping,count,rows,inner,outputs,workers,left,right,bias)
    case False{}: PointProducts.panels(~GemmBuffer.point_keeping_tail,count,rows,inner,outputs,workers,left,right,bias)

# The window reads the weights as one buffer, the other schedules read their
# pieces where they lie.
def apply_schedule(route: Schedule,+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,left: Pieces.Pieces(~Tile.Tile8),weights: Tensors.Tensor<Tile.Tile8>,
  bias: Storage.Buffer<Tile.Tile8>) -> Tiled.Tiled:
  match route:
    case Window{}:
      Tensors.with_whole(~Tile.Tile8,~Tiled.Tiled,weights,weights => weight_shape => weight_fill => weight_status => weight_workers =>
        result(count,rows,outputs,Tensors.Usable{},Convolution.products(count,rows,inner,outputs,left,weights,bias,workers)))
    case SharedGroups{}:
      Tensors.Tensor{right,weight_shape,weight_fill,weight_status,weight_workers} = weights
      TiledMap.restore_padding((Tiled.Tiled{Tensors.Tensor{PointProducts.shared(rows,inner,outputs,workers,left,right,bias),[Tiled.groups(outputs),1,rows],
        Tiled.zero_tile(),Tensors.Usable{},workers},2n,outputs,Grid.Outermost{}},TiledMap.Padding{outputs,Tiled.groups(outputs),rows,Grid.Outermost{}}))
    case ImagePanels{}:
      Tensors.Tensor{right,weight_shape,weight_fill,weight_status,weight_workers} = weights
      TiledMap.restore_padding((Tiled.Tiled{Tensors.Tensor{panel_products(U32.is_zero((inner % 16 : U32)),count,rows,inner,outputs,workers,left,right,bias),
        [count,rows,Tiled.groups(outputs)],Tiled.zero_tile(),Tensors.Usable{},workers},2n,outputs,Grid.Innermost{}},
        TiledMap.Padding{outputs,Tiled.groups(outputs),(count * rows : U32),Grid.Innermost{}}))

def scheduled_status(status: Tensors.Status,+route: Schedule,+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,left: Pieces.Pieces(~Tile.Tile8),
  weights: Tensors.Tensor<Tile.Tile8>,bias: Storage.Buffer<Tile.Tile8>) -> Tiled.Tiled:
  match status:
    case Tensors.Usable{}: apply_schedule(route,count,rows,inner,outputs,workers,left,weights,bias)
    case Tensors.Failed{problem}:
      result(count,rows,outputs,Tensors.Failed{problem},Tensors.Tensor{Tensors.filled(~Tile.Tile8,[count,rows,1,Tiled.groups(outputs)],Tiled.zero_tile()),
        [count,rows,1,Tiled.groups(outputs)],Tiled.zero_tile(),Tensors.Usable{},workers})

def scheduled_run(+route: Schedule,+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,left: Tensors.Tensor<Tile.Tile8>,weights: Tensors.Tensor<Tile.Tile8>,
  bias: Tensors.Tensor<Tile.Tile8>) -> Tiled.Tiled:
  Tensors.Tensor{pieces,left_shape,left_fill,left_status,left_workers} = left
  Tensors.with_metadata(~Tile.Tile8,~Tiled.Tiled,weights,weights => weight_shape => weight_fill => weight_status => weight_workers =>
    Tensors.with_whole(~Tile.Tile8,~Tiled.Tiled,bias,bias => bias_shape => bias_fill => bias_status => bias_workers =>
      scheduled_status(Tensors.first_failure(left_status,Tensors.first_failure(weight_status,bias_status)),route,count,rows,inner,outputs,workers,pieces,weights,bias)))

# The rows of left as [count,rows,1,groups].
def left_rows(+count: U32,+rows: U32,+inner: U32,left: Tensors.Tensor<Tile.Tile8>) -> Tensors.Tensor<Tile.Tile8>:
  Tensors.relabeled(~Tile.Tile8,left,[count,rows,1,Tiled.groups(inner)])

def scheduled_lines(route: Schedule,+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,left: Tensors.Tensor<Tile.Tile8>,right: Tiled.Tiled,bias: Tensors.Tensor<Tile.Tile8>) ->
  Tiled.Tiled:
  match route:
    case SharedGroups{}: scheduled_run(SharedGroups{},count,rows,inner,outputs,workers,left_rows(count,rows,inner,left),weight_cells(right),bias)
    case Window{}: scheduled_run(Window{},count,rows,inner,outputs,workers,left_rows(count,rows,inner,left),weight_cells(right),bias)
    case ImagePanels{}: scheduled_run(ImagePanels{},count,rows,inner,outputs,workers,left,innermost(right,2n),bias)

# left [count,rows,inner], right with its outputs at right_axis and bias [outputs]:
# the result [count,rows,outputs] with its lanes on the outputs, or the first
# failure. Every output adds its products in increasing inner index from +0.0,
# then its bias.
def lines(+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,+right_axis: Nat,left: Tiled.Tiled,right: Tiled.Tiled,bias: Tiled.Tiled) -> Tiled.Tiled:
  scheduled_lines(schedule(Nat.is_eq(right_axis,2n),count,rows,inner,outputs),count,rows,inner,outputs,workers,innermost(left,2n),right,innermost(bias,0n))
