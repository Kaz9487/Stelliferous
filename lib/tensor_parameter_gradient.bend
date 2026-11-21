# Shape validation and affine tensor ownership for the shared reduction kernel.
import Base
import ./tensor.bend as Tensors
import ./kernel_tile.bend as Tile
import ./tensor_tiled.bend as Tiled
import ./tensor_grid.bend as Grid
import ./tensor_tiled_map.bend as TiledMap
import ./storage_buffer.bend as Storage
import ./kernel_parameter_gradient.bend as Kernel
import ./tensor_move.bend as Move
import ./tensor_pieces.bend as Pieces
import ./traversal_partition.bend as Partition
import ./kernel_weight_gradient.bend as WeightKernel
import ./kernel_shape.bend as Shape
import ./machine_limits.bend as Machine
import ./default_machine_limits.bend as Default
import ./tensor_view.bend as TensorView

type ImageGeometry is Data:
  ImageGeometry{images: U32,channels: U32,height: U32,width: U32}

def positive(image: ImageGeometry) -> Bool:
  ImageGeometry{n,c,h,w} = image
  U32.is_lt(0,n) && U32.is_lt(0,c) && U32.is_lt(0,h) && U32.is_lt(0,w)

def work(x: ImageGeometry,d: ImageGeometry,kernel: U32,stride: U32,padding: U32,group: U32) -> Kernel.Work:
  ImageGeometry{n,c,h,w} = x
  ImageGeometry{dn,o,rows,columns} = d
  Kernel.Work{U32.to_nat(n),U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding),U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(group)}

# The packed pixel planes and gradient tiles of a call fit the capacity.
def packed_fits(+x: ImageGeometry,+d: ImageGeometry,+kernel: U32,+stride: U32,+capacity: U32) -> Bool:
  ImageGeometry{n,+c,h,w} = x
  ImageGeometry{dn,+o,+rows,+columns} = d
  WeightKernel.fits(U32.to_nat(c),U32.to_nat(o),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(rows),U32.to_nat(columns),capacity)

# The limits are an argument so that a proof can keep them symbolic.
def valid_with(+x: ImageGeometry,+d: ImageGeometry,+kernel: U32,+stride: U32,+padding: U32,group: U32,+limits: Machine.Limits) -> Bool:
  ImageGeometry{+n,+c,+h,+w} = x
  ImageGeometry{dn,+o,rows,columns} = d
  positive(x) && positive(d) && U32.is_eq(n,dn) && U32.is_lt(0,kernel) && U32.is_lt(0,stride) && U32.is_lt(0,group) &&
    Nat.is_le(U32.to_nat(kernel),Nat.add(U32.to_nat(h),Nat.mul(2n,U32.to_nat(padding)))) &&
    Nat.is_le(U32.to_nat(kernel),Nat.add(U32.to_nat(w),Nat.mul(2n,U32.to_nat(padding)))) &&
    Nat.is_eq(U32.to_nat(rows),Shape.axis_size(U32.to_nat(h),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding))) &&
    Nat.is_eq(U32.to_nat(columns),Shape.axis_size(U32.to_nat(w),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding))) && TensorView.has_count(Tensors.count_of([o,c,kernel,kernel])) &&
    packed_fits(x,d,kernel,stride,Machine.capacity(limits))

def valid(+x: ImageGeometry,+d: ImageGeometry,+kernel: U32,+stride: U32,+padding: U32,group: U32) -> Bool:
  valid_with(x,d,kernel,stride,padding,group,Default.limits())

# The image of cells [n, h, w, groups] or [h, w, groups] whose lanes hold
# lanes channels.
def grouped(same: Bool,image: ImageGeometry) -> Maybe<&2,ImageGeometry>:
  match same:
    case True{}: Some{image}
    case False{}: None{}

def cell_image(shape: List<&2,U32>,+lanes: U32) -> Maybe<&2,ImageGeometry>:
  match shape:
    case [n,h,w,g]: grouped(U32.is_eq(g,((lanes + 7) / 8 : U32)),ImageGeometry{n,lanes,h,w})
    case [h,w,g]: grouped(U32.is_eq(g,((lanes + 7) / 8 : U32)),ImageGeometry{1,lanes,h,w})
    case shape: None{}

# A tensor whose pieces a kernel holds while it reads them, given back.
type Read is Type:
  Read{shape: List<&2,U32>,fill: Tile.Tile8,status: Tensors.Status,workers: U32}

# The held pieces back in the places of their grid.
def back(read: Read,holding: Move.Holding<Tile.Tile8>) -> Tensors.Tensor<Tile.Tile8>:
  Read{+shape,+fill,status,workers} = read
  Tensors.Tensor{Tensors.placed(~Tile.Tile8,Pieces.ranges_of_grid(Tensors.grid_of(shape)),fill,
    Partition.Leaf{Move.released(~Tile.Tile8,holding,fill),0,Tensors.count_or_zero(shape),Unit{}}),shape,fill,status,workers}

def read_of(tensor: Tensors.Tensor<Tile.Tile8>) -> Read & Move.Holding<Tile.Tile8>:
  Tensors.Tensor{pieces,shape,fill,status,workers} = tensor
  (Read{shape,fill,status,workers},Move.held(~Tile.Tile8,pieces))

# The weight gradient [o,c,k,k] as cells with the outputs in their lanes, the
# groups outermost, +0 past the outputs.
def weight_finished(xr: Read,dr: Read,+workers: U32,+o: U32,+taps: U32,+cells: List<&2,U32>,result: WeightKernel.Held() & Storage.Buffer<Tile.Tile8>) ->
  Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>):
  ((x,d),values) = result
  +count = Grid.product(cells)
  (Tiled.Tiled{Tensors.whole(~Tile.Tile8,TiledMap.padded(TiledMap.Padding{o,Tiled.groups(o),taps,Grid.Outermost{}},0,count,values),cells,Tiled.zero_tile(),Tensors.Usable{},workers),
    0n,o,Grid.Outermost{}},(back(xr,x),back(dr,d)))

def weight_selected(accepted: Bool,+xi: ImageGeometry,+di: ImageGeometry,+k: U32,s: U32,p: U32,g: U32,xr: Read,x: Move.Holding<Tile.Tile8>,dr: Read,d: Move.Holding<Tile.Tile8>,
  +workers: U32,+xs: List<&2,U32>) -> Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>):
  match accepted:
    case False{}: (Tiled.failed("conv2d_weight_gradient",Tensors.InvalidArgument{xs},[],workers),(back(xr,x),back(dr,d)))
    case True{}:
      ImageGeometry{n,+c,h,w} = xi
      ImageGeometry{dn,+o,rows,columns} = di
      weight_finished(xr,dr,workers,o,(c * k * k : U32),[Tiled.groups(o),c,k,k],WeightKernel.weight_parts(work(xi,di,k,s,p,g),workers,x,d))

def usable_pair(left: Tensors.Status,right: Tensors.Status) -> Bool:
  Tensors.usable(left) && Tensors.usable(right)

def weight_shaped(xi: Maybe<&2,ImageGeometry>,di: Maybe<&2,ImageGeometry>,usable: Bool,+k: U32,+s: U32,+p: U32,+g: U32,xr: Read,x: Move.Holding<Tile.Tile8>,dr: Read,
  d: Move.Holding<Tile.Tile8>,+workers: U32,+xs: List<&2,U32>) -> Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>):
  match xi di usable:
    case Some{+xi} Some{+di} True{}: weight_selected(valid(xi,di,k,s,p,g),xi,di,k,s,p,g,xr,x,dr,d,workers,xs)
    case xi di usable: weight_selected(False{},ImageGeometry{0,0,0,0},ImageGeometry{0,0,0,0},k,s,p,g,xr,x,dr,d,workers,xs)

def read_shape(read: Read) -> Read & (List<&2,U32> & (Tensors.Status & U32)):
  Read{+shape,fill,+status,+workers} = read
  (Read{shape,fill,status,workers},(shape,(status,workers)))

def weight_read(+x_lanes: U32,+d_lanes: U32,+k: U32,+s: U32,+p: U32,+g: U32,xs: Read & (List<&2,U32> & (Tensors.Status & U32)),x: Move.Holding<Tile.Tile8>,
  ds: Read & (List<&2,U32> & (Tensors.Status & U32)),d: Move.Holding<Tile.Tile8>) -> Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>):
  (xr,(+x_shape,(x_status,x_workers))) = xs
  (dr,(+d_shape,(d_status,d_workers))) = ds
  weight_shaped(cell_image(x_shape,x_lanes),cell_image(d_shape,d_lanes),usable_pair(x_status,d_status),k,s,p,g,xr,x,dr,d,U32.max(x_workers,d_workers),x_shape)

def weight_buffers(+x_lanes: U32,+d_lanes: U32,+k: U32,+s: U32,+p: U32,+g: U32,x: Read & Move.Holding<Tile.Tile8>,d: Read & Move.Holding<Tile.Tile8>) ->
  Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>):
  (xr,xb) = x
  (dr,db) = d
  weight_read(x_lanes,d_lanes,k,s,p,g,read_shape(xr),xb,read_shape(dr),db)

# The weight gradient of cells x and d whose lanes hold x_lanes and d_lanes
# channels: (the gradient, (x, d)). The kernel reads each from its pieces.
def weight(x: Tensors.Tensor<Tile.Tile8>,+x_lanes: U32,d: Tensors.Tensor<Tile.Tile8>,+d_lanes: U32,+k: U32,+s: U32,+p: U32,+g: U32) ->
  Tiled.Tiled & (Tensors.Tensor<Tile.Tile8> & Tensors.Tensor<Tile.Tile8>):
  weight_buffers(x_lanes,d_lanes,k,s,p,g,read_of(x),read_of(d))

def bias_work(d: ImageGeometry,group: U32) -> Kernel.Work:
  ImageGeometry{n,o,rows,columns} = d
  Kernel.Work{U32.to_nat(n),0n,U32.to_nat(o),0n,0n,1n,1n,0n,U32.to_nat(rows),U32.to_nat(columns),U32.to_nat(group)}

# The bias gradient [o] as cells with the outputs in their lanes.
def bias_finished(dr: Read,+workers: U32,+outputs: U32,result: Move.Holding<Tile.Tile8> & Storage.Buffer<Tile.Tile8>) -> Tiled.Tiled & Tensors.Tensor<Tile.Tile8>:
  (d,values) = result
  (Tiled.Tiled{Tensors.whole(~Tile.Tile8,values,[Tiled.groups(outputs)],Tiled.zero_tile(),Tensors.Usable{},workers),0n,outputs,Grid.Innermost{}},back(dr,d))

def bias_selected(accepted: Bool,+di: ImageGeometry,+g: U32,dr: Read,d: Move.Holding<Tile.Tile8>,+workers: U32,+shape: List<&2,U32>) -> Tiled.Tiled & Tensors.Tensor<Tile.Tile8>:
  match accepted:
    case False{}: (Tiled.failed("conv2d_bias_gradient",Tensors.InvalidArgument{shape},[],workers),back(dr,d))
    case True{}:
      ImageGeometry{n,+o,rows,columns} = di
      bias_finished(dr,workers,o,Kernel.bias(bias_work(ImageGeometry{n,o,rows,columns},g),d,Tensors.buffer(~Tile.Tile8,Tiled.zero_tile(),Tiled.groups(o))))

def bias_shaped(di: Maybe<&2,ImageGeometry>,usable: Bool,+g: U32,dr: Read,d: Move.Holding<Tile.Tile8>,+workers: U32,+shape: List<&2,U32>) -> Tiled.Tiled & Tensors.Tensor<Tile.Tile8>:
  match di usable:
    case Some{+di} True{}: bias_selected(positive(di) && U32.is_lt(0,g),di,g,dr,d,workers,shape)
    case di usable: bias_selected(False{},ImageGeometry{0,0,0,0},g,dr,d,workers,shape)

def bias_read(+lanes: U32,+g: U32,read: Read & (List<&2,U32> & (Tensors.Status & U32)),d: Move.Holding<Tile.Tile8>) -> Tiled.Tiled & Tensors.Tensor<Tile.Tile8>:
  (dr,(+shape,(status,workers))) = read
  bias_shaped(cell_image(shape,lanes),Tensors.usable(status),g,dr,d,workers,shape)

def bias_buffer(+lanes: U32,+g: U32,d: Read & Move.Holding<Tile.Tile8>) -> Tiled.Tiled & Tensors.Tensor<Tile.Tile8>:
  (dr,db) = d
  bias_read(lanes,g,read_shape(dr),db)

# The bias gradient of cells d whose lanes hold lanes channels: (the
# gradient, d).
def bias(d: Tensors.Tensor<Tile.Tile8>,+lanes: U32,+g: U32) -> Tiled.Tiled & Tensors.Tensor<Tile.Tile8>:
  bias_buffer(lanes,g,read_of(d))
