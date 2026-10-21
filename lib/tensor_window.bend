# Two-dimensional pooling over the last two axes, as two ordered sweeps. Along
# the columns, each row of the window folds its k samples in increasing kx
# from pad: line = op(...op(op(pad, x0), x1)..., x(k-1)). Along the rows, the
# output folds the k lines in increasing ky from pad. A sample outside the
# input reads pad, so pad should be the identity of the operation.
import Base
import ./tensor_shape.bend as Shapes
import ./tensor_access.bend as Access
import ./tensor_operations.bend as Operations
import ./tensor.bend as Tensors
import ./tensor_layout.bend as Layout
import ./tensor_band.bend as Bands
import ./tensor_band_window.bend as BandWindow
import ./tensor_band_pool.bend as BandPool
import ./tensor_band_schedule.bend as BandSchedule
import ./tensor_view.bend as Views

def out_extent(+extent: U32,+kernel: U32,+stride: U32,+padding: U32) -> U32:
  ((extent + 2 * padding - kernel : U32) / U32.max(stride,1) + 1 : U32)

def geometry(+shape: List<&2,U32>,+kernel: U32,+stride: U32,+padding: U32) -> Access.Pool:
  +rank = List.length(&2,U32,shape)
  +height = Layout.extent_at(shape,Nat.sub(rank,2n))
  +width = Layout.extent_at(shape,Nat.sub(rank,1n))
  Access.Pool{height,width,kernel,stride,padding,out_extent(height,kernel,stride,padding),out_extent(width,kernel,stride,padding)}

# The column-swept planes keep the rows of the input and have the columns of
# the output.
def middle_shape(+shape: List<&2,U32>,pool: Access.Pool) -> List<&2,U32>:
  Access.Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[height,out_width])

def output_shape(+shape: List<&2,U32>,pool: Access.Pool) -> List<&2,U32>:
  Access.Pool{height,width,kernel,stride,padding,out_height,out_width} = pool
  List.append(&2,U32,List.take(&2,U32,shape,Nat.sub(List.length(&2,U32,shape),2n)),[out_height,out_width])

# pad is both the value outside the input and the initial accumulator.
# The pooled bands are the result when every band's pool accepted and the
# bands cover the output geometry.
def band_execute(~Element: Data,~operation: Element -> Element -> Element,name: String,+pad: Element,pool: Access.Pool,+geometry: Bands.Geometry,bands: Bands.Bands(~Element),
  shape: List<&2,U32>,out_shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match pool geometry:
    case Access.Pool{height,width,kernel,stride,padding,+oh,+ow} Bands.Geometry{+batch,+channels,ih,iw}:
      Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},Bands.Geometry{batch,channels,oh,ow},out_shape,fill,workers,
        BandPool.execute(~Element,~operation,BandPool.Context{BandWindow.Geometry{geometry,kernel,stride,padding,oh,ow},pad,fill,workers},bands))

# The input arrives in the layout of its planes, and the band pool runs there.
def planned(~Element: Data,~operation: Element -> Element -> Element,name: String,+pad: Element,+pool: Access.Pool,tensor: Tensors.Tensor<Element>,
  +shape: List<&2,U32>,+fill: Element,+workers: U32,target: BandSchedule.Target) -> Tensors.Tensor<Element>:
  BandSchedule.Target{+geometry,layout,split} = target
  band_execute(~Element,~operation,name,pad,pool,geometry,Tensors.arrive(~Element,geometry,layout,tensor),shape,output_shape(shape,pool),fill,workers)

def offered(~Element: Data,~operation: Element -> Element -> Element,name: String,+pad: Element,+pool: Access.Pool,+planes: U32,
  +shape: List<&2,U32>,+fill: Element,+workers: U32,owners: Tensors.Tensor<Element> & BandSchedule.Target) -> Tensors.Tensor<Element>:
  Access.Pool{+height,+width,kernel,stride,padding,out_height,out_width} = pool
  (tensor,target) = owners
  planned(~Element,~operation,name,pad,pool,tensor,shape,fill,workers,BandSchedule.plane_target(shape,planes,height,width,target))

# An empty output has no element to pool.
def nonempty(~Element: Data,~operation: Element -> Element -> Element,name: String,empty: Bool,+pad: Element,+pool: Access.Pool,+planes: U32,
  tensor: Tensors.Tensor<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match empty:
    case True{}: Tensors.whole(~Element,Tensors.buffer(~Element,fill,0),output_shape(shape,pool),fill,Tensors.Usable{},workers)
    case False{}: offered(~Element,~operation,name,pad,pool,planes,shape,fill,workers,Tensors.offer(~Element,tensor))

def checked(~Element: Data,~operation: Element -> Element -> Element,name: String,valid: Bool,planes: Maybe<&2,U32>,+pad: Element,+pool: Access.Pool,
  tensor: Tensors.Tensor<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match valid planes:
    case True{} Some{+planes}:
      nonempty(~Element,~operation,name,Shapes.has_zero(output_shape(shape,pool)),pad,pool,planes,tensor,shape,fill,workers)
    case valid planes: Tensors.failed(~Element,name,Tensors.InvalidArgument{shape},shape,fill,workers)

# A pool needs planes before the last two axes, positive extents and windows
# that fit them, and counts of the column-swept planes and of the output.
def valid(+shape: List<&2,U32>,+pool: Access.Pool) -> Bool:
  Access.Pool{+height,+width,kernel,stride,padding,out_height,out_width} = pool
  Operations.pool_geometry(pool) && U32.is_lt(0,height) && U32.is_lt(0,width) &&
    Views.has_count(Tensors.count_of(middle_shape(shape,pool))) && Views.has_count(Tensors.count_of(output_shape(shape,pool)))

def pool_status(~Element: Data,~operation: Element -> Element -> Element,name: String,status: Tensors.Status,+pad: Element,+kernel: U32,+stride: U32,+padding: U32,
  bands: Bands.Bands(~Element),layout: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensors.Tensor<Element>:
  match status:
    case Tensors.Failed{problem}: Tensors.Tensor{bands,layout,shape,fill,Tensors.Failed{problem},workers}
    case Tensors.Usable{}:
      +pool = geometry(shape,kernel,stride,padding)
      checked(~Element,~operation,name,valid(shape,pool),Layout.planes(shape),pad,pool,Tensors.Tensor{bands,layout,shape,fill,Tensors.Usable{},workers},shape,fill,workers)

def pool2d(~Element: Data,~operation: Element -> Element -> Element,name: String,+pad: Element,tensor: Tensors.Tensor<Element>,
  +kernel: U32,+stride: U32,+padding: U32) -> Tensors.Tensor<Element>:
  Tensors.Tensor{bands,layout,+shape,+fill,status,+workers} = tensor
  pool_status(~Element,~operation,name,status,pad,kernel,stride,padding,bands,layout,shape,fill,workers)
