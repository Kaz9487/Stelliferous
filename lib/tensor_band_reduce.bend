# A reduction runs on the lines of a view of its input: the view puts the
# reduced axes last, and line k holds their values at index k of the kept
# axes. The view is classified over the input's bands as a gather is: when
# the bands keep their images and rows and the reduced axes lie after the
# rows, every band folds the lines of its own rows through its local view;
# otherwise the input arrives as one band, which folds every line.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_view.bend as Views
import ./tensor_operations.bend as Operations
import ./tensor_band.bend as Bands
import ./tensor_band_view.bend as BandViews
import ./traversal_partition.bend as Partition
import ./tensor.bend as Tensors
import ./tensor_band_schedule.bend as BandSchedule

# What every band of a reduction shares: the banded view of the input
# geometry, the number of reduced axes, the initial accumulator and the fill
# of fresh storage.
type Context<-Accumulator: Data,-Element: Data> is Data:
  Context{banded: BandViews.Banded,reduced: Nat,geometry: Bands.Geometry,initial: Accumulator,fill: Element}

# A band's folds as a checked band: its outputs beside whether the call
# accepted.
def checked(~Element: Data,+first: U32,+rows: U32,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Bands.Transformed(~Element):
  match result:
    case Operations.Accepted{(input,output)}: Partition.Leaf{Bands.Checked{output,True{}},first,rows,Unit{}}
    case Operations.Rejected{(input,output)}: Partition.Leaf{Bands.Checked{output,False{}},first,rows,Unit{}}

# The axes of a view before its last reduced axes.
def kept_axes(+axes: List<&2,Views.Axis>,+reduced: Nat) -> List<&2,Views.Axis>:
  List.take(&2,Views.Axis,axes,Nat.sub(List.length(&2,Views.Axis,axes),reduced))

def folded_view(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +reduced: Nat,+initial: Accumulator,+fill: Element,+first: U32,+rows: U32,local: Views.View,input: Storage.Buffer<Element>) -> Bands.Transformed(~Element):
  Views.View{+axes,+offset} = local
  outputs = {Views.View{kept_axes(axes,reduced),0} : Views.View}
  checked(~Element,first,rows,
    Operations.reduce_lines(~Element,~Accumulator,~operation,~finish,input,axes,reduced,offset,initial,
      Tensors.buffer(~Element,fill,BandViews.known(Views.numel(outputs)))))

# The band of rows [first, first + rows) folds the lines of its rows; its
# output band holds their folds. A band of no rows has no line.
def folded_rows(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  empty: Bool,+banded: BandViews.Banded,+reduced: Nat,+geometry: Bands.Geometry,+initial: Accumulator,+fill: Element,+first: U32,+rows: U32,
  input: Storage.Buffer<Element>) -> Bands.Transformed(~Element):
  match empty:
    case True{}:
      +scale = BandViews.scale(banded)
      Partition.Leaf{Bands.Checked{Tensors.buffer(~Element,fill,0),True{}},(first * scale : U32),(rows * scale : U32),Unit{}}
    case False{}:
      +scale = BandViews.scale(banded)
      folded_view(~Element,~Accumulator,~operation,~finish,reduced,initial,fill,(first * scale : U32),(rows * scale : U32),
        BandViews.local(banded,geometry,rows),input)

def leaf(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  context: Context<Accumulator,Element>,first: U32,rows: U32,input: Storage.Buffer<Element>) -> Bands.Transformed(~Element):
  Context{+banded,+reduced,+geometry,+initial,+fill} = context
  +first = first
  +rows = rows
  folded_rows(~Element,~Accumulator,~operation,~finish,U32.is_zero(rows),banded,reduced,geometry,initial,fill,first,rows,input)

# The geometry of the outputs: the same images, the axes before the rows as
# channels, each row repeated, the kept axes after the rows as columns.
def output(+banded: BandViews.Banded,+geometry: Bands.Geometry,+reduced: Nat) -> Bands.Geometry:
  BandViews.Banded{before,repeats,after,offset} = banded
  Bands.Geometry{batch,channels,height,width} = geometry
  Bands.Geometry{batch,BandViews.product_or_zero(Views.axis_extents(before)),(height * BandViews.product_or_zero(repeats) : U32),
    BandViews.product_or_zero(Views.axis_extents(kept_axes(after,reduced)))}

# The banded view when the reduced axes lie among the axes after the rows.
def in_band(found: Maybe<&2,BandViews.Banded>,+reduced: Nat) -> Maybe<&2,BandViews.Banded>:
  match found:
    case Some{BandViews.Banded{+before,+repeats,+after,+offset}}:
      BandViews.kept_when(Nat.is_le(reduced,List.length(&2,Views.Axis,after)),BandViews.Banded{before,repeats,after,offset})
    case None{}: None{}

# Every band folds its own lines. operations prices the combination of one
# value, beside the read of it.
def banded_lines(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+name: String,
  +operations: F32,+initial: Accumulator,+reduced: Nat,+banded: BandViews.Banded,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  bands: Bands.Bands(~Element)) -> Tensors.Tensor<Element>:
  Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},output(banded,geometry,reduced),shape,fill,workers,
    BandSchedule.transform(~Element,~Context<Accumulator,Element>,~leaf(~Element,~Accumulator,~operation,~finish),Context{banded,reduced,geometry,initial,fill},
      geometry,workers,4.0,operations,bands))

def whole_lines(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+name: String,
  +operations: F32,+initial: Accumulator,+reduced: Nat,found: Maybe<&2,BandViews.Banded>,+whole: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  tensor: Tensors.Tensor<Element>) -> Tensors.Tensor<Element>:
  match found:
    case Some{+banded}:
      banded_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,banded,whole,shape,fill,workers,
        Tensors.arrive(~Element,whole,Bands.unsplit(whole),tensor))
    case None{}: Tensors.failed(~Element,name,Tensors.InvalidArgument{shape},shape,fill,workers)

def found_lines(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+name: String,
  +operations: F32,+initial: Accumulator,+reduced: Nat,found: Maybe<&2,BandViews.Banded>,+axes: List<&2,Views.Axis>,+offset: U32,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,+source_shape: List<&2,U32>,+source_fill: Element,+source_workers: U32,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  Tensors.Tensor<Element>:
  match found:
    case Some{+banded}: banded_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,banded,geometry,shape,fill,workers,bands)
    case None{}:
      +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(source_shape)} : Bands.Geometry}
      whole_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,in_band(BandViews.classify(axes,offset,whole),reduced),whole,shape,fill,workers,
        Tensors.Tensor{bands,geometry,source_shape,source_fill,Tensors.Usable{},source_workers})

# No output, or lines of no value, leave every output at finish(initial).
def lines_when(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+name: String,
  empty: Bool,+operations: F32,+count: U32,+initial: Accumulator,+reduced: Nat,+axes: List<&2,Views.Axis>,+offset: U32,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,+source_shape: List<&2,U32>,+source_fill: Element,+source_workers: U32,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  Tensors.Tensor<Element>:
  match empty:
    case True{}: Tensors.whole(~Element,Tensors.buffer(~Element,finish(initial),count),shape,fill,Tensors.Usable{},workers)
    case False{}:
      found_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,in_band(BandViews.classify(axes,offset,geometry),reduced),
        axes,offset,bands,geometry,source_shape,source_fill,source_workers,shape,fill,workers)

# count outputs of shape: output k folds the length values of line k of the
# view, the values of its last reduced axes at index k of the axes before
# them, in row-major order from initial, and holds finish of the fold. A
# failed tensor keeps its problem.
def lines(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+name: String,
  +operations: F32,+count: U32,+length: U32,+initial: Accumulator,+shape: List<&2,U32>,+fill: Element,+workers: U32,tensor: Tensors.Tensor<Element>,
  view: Maybe<&2,Views.View>,+reduced: Nat) -> Tensors.Tensor<Element>:
  Tensors.Tensor{bands,+geometry,+source_shape,+source_fill,status,+source_workers} = tensor
  match status view:
    case Tensors.Usable{} Some{Views.View{+axes,+offset}}:
      lines_when(~Element,~Accumulator,~operation,~finish,name,U32.is_zero(count) || U32.is_zero(length),operations,count,initial,reduced,axes,offset,bands,
        geometry,source_shape,source_fill,source_workers,shape,fill,workers)
    case Tensors.Usable{} None{}: Tensors.failed(~Element,name,Tensors.InvalidArgument{source_shape},shape,fill,workers)
    case Tensors.Failed{problem} view: Tensors.whole(~Element,Tensors.buffer(~Element,fill,0),shape,fill,Tensors.Failed{problem},workers)
