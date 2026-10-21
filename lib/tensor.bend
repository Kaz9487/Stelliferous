# A public tensor owns bands of row-major storage and the geometry they split,
# its extents, the value used to initialize allocations, a sticky status, and
# the number of workers its operations may use. A whole tensor is one band that
# holds every element. A failed operation leaves values untouched and records
# why; later operations skip a failed tensor, and check is the only way to
# observe or clear the failure.
import Base
import ./storage_buffer.bend as Storage
import ./storage_allocation.bend as Allocation
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./tensor_operations.bend as Operations
import ./machine_limits.bend as Machine
import ./default_machine_limits.bend as Default
import ./tensor_band.bend as Bands
import ./tensor_band_schedule.bend as BandSchedule
import ./tensor_band_view.bend as BandViews
import ./traversal_partition.bend as Partition

type Reason is Data:
  ShapeMismatch{left: List<&2,U32>,right: List<&2,U32>}
  InvalidShape{shape: List<&2,U32>}
  InvalidArgument{shape: List<&2,U32>}
  FileFault{path: String,detail: String}
  MissingName{name: String}

type Problem is Data:
  Problem{operation: String,reason: Reason}

type Status is Data:
  Usable{}
  Failed{problem: Problem}

# Like a PyTorch tensor's device, workers is execution context: the size of
# the runtime's worker pool, read once with IO.thread_count and set with
# with_workers. Pure constructors cannot read it and use one worker. Results
# keep the largest worker count of their operands. It never changes a value.
type Tensor<-Element: Data> is Type:
  Tensor{bands: Bands.Bands(~Element),geometry: Bands.Geometry,shape: List<&2,U32>,fill: Element,status: Status,workers: U32}

# A nonempty sequence of tensors. Base List is not used for tensors: the
# compiler shares every constructor of a list type that is duplicated anywhere,
# and that sharing would reach the certificate inside each tensor's storage.
type Parts<-Element: Data> is Type:
  Single{tensor: Tensor<Element>}
  Cons{tensor: Tensor<Element>,rest: Parts<Element>}

# Element count of a shape that the default machine can allocate.
def within(fits: Bool,count: U32) -> Maybe<&2,U32>:
  match fits:
    case True{}: Some{count}
    case False{}: None{}

def allocatable(count: Maybe<&2,U32>) -> Maybe<&2,U32>:
  match count:
    case None{}: None{}
    case Some{+count}: within(U32.is_le(count,Machine.capacity(Default.limits())),count)

def count_of(+shape: List<&2,U32>) -> Maybe<&2,U32>:
  allocatable(Shapes.numel(Shapes.Shape{shape}))

def known_count(count: Maybe<&2,U32>) -> U32:
  match count:
    case Some{count}: count
    case None{}: 0

# The element count as a number; a shape too large to allocate counts 0.
def count_or_zero(+shape: List<&2,U32>) -> U32:
  known_count(count_of(shape))

def view_of(+shape: List<&2,U32>) -> Maybe<&2,Views.View>:
  Views.contiguous(Shapes.Shape{shape},0)

# Bits of value, counted by tail recursion. Storage.bit_length is not tail
# recursive, and a non-tail recursive callee turns the calling chain into
# worklist dispatch: measured allocation of 2^20 floats took twice as long.
def significant_bits(remaining: Nat,+value: U32,+bits: Nat,zero: Bool) -> Nat:
  Storage.significant_bits(remaining,value,bits,zero)

# The smallest depth whose capacity holds count elements (at least one).
def depth_for(+count: U32) -> Nat:
  Storage.depth_for(count)

def buffer(~Element: Data,+fill: Element,+count: U32) -> Storage.Buffer<Element>:
  Allocation.allocate(~Element,depth_for(count),fill,U32.to_nat(count))

# Row-major storage as a tensor: one band of one row whose width is every
# element of the shape.
def whole(~Element: Data,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,fill: Element,status: Status,workers: U32) -> Tensor<Element>:
  Tensor{Partition.Leaf{storage,0,1,Unit{}},Bands.Geometry{1,1,1,count_or_zero(shape)},shape,fill,status,workers}

# The whole storage of bands. A leaf that holds the whole image already is it;
# other bands are gathered into one allocation.
def leaf_whole_storage(~Element: Data,whole: Bool,+geometry: Bands.Geometry,+fill: Element,data: Storage.Buffer<Element>,first: U32,rows: U32) -> Storage.Buffer<Element>:
  match whole:
    case True{}: data
    case False{}: Bands.gather(~Element,geometry,Partition.Leaf{data,first,rows,Unit{}},buffer(~Element,fill,Bands.elements(geometry)))

def whole_storage(~Element: Data,+geometry: Bands.Geometry,+fill: Element,bands: Bands.Bands(~Element)) -> Storage.Buffer<Element>:
  match bands:
    case Partition.Leaf{data,+first,+rows,details}: leaf_whole_storage(~Element,Bands.whole_image(geometry,first,rows),geometry,fill,data,first,rows)
    case Partition.Fork{left,right}: Bands.gather(~Element,geometry,Partition.Fork{left,right},buffer(~Element,fill,Bands.elements(geometry)))

def arrive_when(~Element: Data,same: Bool,geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,other: Bands.Geometry,bands: Bands.Bands(~Element)) -> Bands.Bands(~Element):
  match same:
    case True{}: Bands.align(~Element,geometry,fill,layout,bands)
    case False{}: Bands.from_whole(~Element,geometry,fill,layout,whole_storage(~Element,other,fill,bands))

# The bands of a tensor in the layout an operation runs in. Bands of the same
# geometry are aligned to the plan. Bands of another geometry are read in
# row-major order and cut by the plan, which relabels a whole image as is.
def arrive(~Element: Data,+geometry: Bands.Geometry,layout: Bands.Plan,tensor: Tensor<Element>) -> Bands.Bands(~Element):
  Tensor{bands,+other,shape,fill,status,workers} = tensor
  arrive_when(~Element,Bands.same_geometry(geometry,other),geometry,fill,layout,other,bands)

# The storage of a tensor arrived in the geometry of whole: the one leaf of
# its unsplit layout.
def with_whole(~Element: Data,~Result: Type,tensor: Tensor<Element>,
  body: Storage.Buffer<Element> -> List<&2,U32> -> Element -> Status -> U32 -> Result) -> Result:
  Tensor{bands,geometry,+shape,+fill,+status,+workers} = tensor
  +whole = {Bands.Geometry{1,1,1,count_or_zero(shape)} : Bands.Geometry}
  body(whole_storage(~Element,whole,fill,arrive(~Element,whole,Bands.unsplit(whole),Tensor{bands,geometry,shape,fill,status,workers})),shape,fill,status,workers)

# Metadata inspection retains the original layout and affine storage.
def with_metadata(~Element: Data,~Result: Type,tensor: Tensor<Element>,
  body: Tensor<Element> -> List<&2,U32> -> Element -> Status -> U32 -> Result) -> Result:
  Tensor{bands,geometry,+shape,+fill,+status,+workers} = tensor
  body(Tensor{bands,geometry,shape,fill,status,workers},shape,fill,status,workers)

def usable(status: Status) -> Bool:
  match status:
    case Usable{}: True{}
    case Failed{problem}: False{}

# The band a checked call made, as whole storage. It counts as usable only when
# the call was usable.
def checked_band(~Element: Data,+first: U32,+rows: U32,result: Tensor<Element>) -> Bands.Transformed(~Element):
  Tensor{bands,+geometry,shape,+fill,status,workers} = result
  Partition.Leaf{Bands.Checked{whole_storage(~Element,geometry,fill,bands),usable(status)},first,rows,Unit{}}

def failed(~Element: Data,operation: String,reason: Reason,shape: List<&2,U32>,+fill: Element,workers: U32) -> Tensor<Element>:
  whole(~Element,buffer(~Element,fill,0),shape,fill,Failed{Problem{operation,reason}},workers)

def transformed_accepted(~Element: Data,accepted: Bool,operation: String,reason: Reason,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,
  fill: Element,workers: U32) -> Tensor<Element>:
  match accepted:
    case True{}: Tensor{bands,geometry,shape,fill,Usable{},workers}
    case False{}: failed(~Element,operation,reason,shape,fill,workers)

def transformed_inspected(~Element: Data,usable: Bool,operation: String,reason: Reason,+geometry: Bands.Geometry,+shape: List<&2,U32>,fill: Element,workers: U32,
  inspected: Bands.Bands(~Element) & Bands.Inspection) -> Tensor<Element>:
  (bands,inspection) = inspected
  transformed_accepted(~Element,usable && Bands.covers(geometry,inspection),operation,reason,bands,geometry,shape,fill,workers)

# Bands made by checked calls are the result when every call was usable and
# the bands cover the geometry; otherwise the operation fails for reason.
def transformed(~Element: Data,operation: String,reason: Reason,+geometry: Bands.Geometry,+shape: List<&2,U32>,fill: Element,workers: U32,
  result: Bands.Bands(~Element) & Bool) -> Tensor<Element>:
  (bands,usable) = result
  transformed_inspected(~Element,usable,operation,reason,geometry,shape,fill,workers,Bands.inspect(~Element,geometry,bands))

# A usable tensor fails with an invalid argument of the operation; a failed one keeps its problem.
def rejected(~Element: Data,name: String,tensor: Tensor<Element>) -> Tensor<Element>:
  match tensor:
    case Tensor{bands,geometry,+shape,fill,Usable{},workers}: Tensor{bands,geometry,shape,fill,Failed{Problem{name,InvalidArgument{shape}}},workers}
    case Tensor{bands,geometry,shape,fill,Failed{problem},workers}: Tensor{bands,geometry,shape,fill,Failed{problem},workers}

def mismatch(same: Bool,left: List<&2,U32>,right: List<&2,U32>) -> Reason:
  match same:
    case True{}: InvalidShape{left}
    case False{}: ShapeMismatch{left,right}

def same_or_invalid(+left: List<&2,U32>,+right: List<&2,U32>) -> Reason:
  mismatch(Shapes.same_extents(left,right),left,right)

# Construction

def full_count(~Element: Data,+shape: List<&2,U32>,count: Maybe<&2,U32>,+value: Element) -> Tensor<Element>:
  match count:
    case Some{count}: whole(~Element,buffer(~Element,value,count),shape,value,Usable{},1)
    case None{}: failed(~Element,"full",InvalidShape{shape},shape,value,1)

def full(~Element: Data,+shape: List<&2,U32>,+value: Element) -> Tensor<Element>:
  full_count(~Element,shape,count_of(shape),value)

def listed_count(~Element: Data,matches: Bool,+shape: List<&2,U32>,+count: U32,values: List<&2,Element>,+fill: Element) -> Tensor<Element>:
  match matches:
    case True{}:
      whole(~Element,Storage.from_list(~Element,values,depth_for(count),fill,U32.to_nat(count)),shape,fill,Usable{},1)
    case False{}: failed(~Element,"from_list",InvalidArgument{shape},shape,fill,1)

def listed(~Element: Data,+shape: List<&2,U32>,count: Maybe<&2,U32>,+values: List<&2,Element>,+fill: Element) -> Tensor<Element>:
  match count:
    case Some{+count}: listed_count(~Element,Nat.is_eq(List.length(&2,Element,values),U32.to_nat(count)),shape,count,values,fill)
    case None{}: failed(~Element,"from_list",InvalidShape{shape},shape,fill,1)

# fill initializes storage this tensor or its results allocate; it is never
# observable in a successful result.
def from_list(~Element: Data,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element) -> Tensor<Element>:
  listed(~Element,shape,count_of(shape),values,fill)

def stored(~Element: Data,matches: Bool,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,fill: Element) -> Tensor<Element>:
  match matches:
    case True{}: whole(~Element,storage,shape,fill,Usable{},1)
    case False{}: whole(~Element,storage,shape,fill,Failed{Problem{"from_storage",InvalidShape{shape}}},1)

def matches_count(count: Maybe<&2,U32>,length: Nat) -> Bool:
  match count:
    case Some{count}: Nat.is_eq(length,U32.to_nat(count))
    case None{}: False{}

def storage_of(~Element: Data,+shape: List<&2,U32>,count: Maybe<&2,U32>,storage: Storage.Buffer<Element>,fill: Element) -> Tensor<Element>:
  Storage.Buffer{storage,+length,certificate} = storage
  stored(~Element,matches_count(count,length),Storage.Buffer{storage,length,certificate},shape,fill)

# Storage handed over by foreign I/O keeps its documented trust boundary.
def from_storage(~Element: Data,+shape: List<&2,U32>,storage: Storage.Buffer<Element>,+fill: Element) -> Tensor<Element>:
  storage_of(~Element,shape,count_of(shape),storage,fill)

# Execution context

# Sets the worker count, at least one; values and status are unchanged.
def with_workers(~Element: Data,tensor: Tensor<Element>,+count: U32) -> Tensor<Element>:
  Tensor{bands,geometry,shape,fill,status,workers} = tensor
  Tensor{bands,geometry,shape,fill,status,U32.max(1,count)}

def workers(~Element: Data,tensor: Tensor<Element>) -> Tensor<Element> & U32:
  Tensor{bands,geometry,shape,fill,status,+workers} = tensor
  (Tensor{bands,geometry,shape,fill,status,workers},workers)

# Observation

def shape(~Element: Data,tensor: Tensor<Element>) -> Tensor<Element> & List<&2,U32>:
  Tensor{bands,geometry,+shape,fill,status,workers} = tensor
  (Tensor{bands,geometry,shape,fill,status,workers},shape)

def checked(~Element: Data,status: Status,bands: Bands.Bands(~Element),geometry: Bands.Geometry,shape: List<&2,U32>,fill: Element,workers: U32) ->
  Result<&1,&1,Tensor<Element> & Problem,Tensor<Element>>:
  match status:
    case Usable{}: Done{Tensor{bands,geometry,shape,fill,Usable{},workers}}
    case Failed{problem}: Fail{(Tensor{bands,geometry,shape,fill,Usable{},workers},problem)}

def check(~Element: Data,tensor: Tensor<Element>) -> Result<&1,&1,Tensor<Element> & Problem,Tensor<Element>>:
  Tensor{bands,geometry,shape,fill,status,workers} = tensor
  checked(~Element,status,bands,geometry,shape,fill,workers)

def released(~Element: Data,status: Status,storage: Storage.Buffer<Element>) -> Result<&2,&1,Problem,Storage.Buffer<Element>>:
  match status:
    case Usable{}: Done{storage}
    case Failed{problem}: Fail{problem}

def into_storage(~Element: Data,tensor: Tensor<Element>) -> Result<&2,&1,Problem,Storage.Buffer<Element>>:
  with_whole(~Element,~Result<&2,&1,Problem,Storage.Buffer<Element>>,tensor,storage => shape => fill => status => workers => released(~Element,status,storage))

def listed_values(~Element: Data,status: Status,storage: Array<Element>,length: Nat) -> Result<&2,&2,Problem,List<&2,Element>>:
  match status:
    case Usable{}: Done{Storage.read_list(~Element,length,0,storage)}
    case Failed{problem}: Fail{problem}

def listed_buffer(~Element: Data,status: Status,buffer: Storage.Buffer<Element>) -> Result<&2,&2,Problem,List<&2,Element>>:
  Storage.Buffer{storage,length,certificate} = buffer
  listed_values(~Element,status,storage,length)

def to_list(~Element: Data,tensor: Tensor<Element>) -> Result<&2,&2,Problem,List<&2,Element>>:
  with_whole(~Element,~Result<&2,&2,Problem,List<&2,Element>>,tensor,buffer => shape => fill => status => workers =>
    listed_buffer(~Element,status,buffer))

def only_value(~Element: Data,+shape: List<&2,U32>,values: List<&2,Element>) -> Result<&2,&2,Problem,Element>:
  match values:
    case Con{value,Nil{}}: Done{value}
    case values: Fail{Problem{"item",InvalidShape{shape}}}

def single(~Element: Data,+shape: List<&2,U32>,values: Result<&2,&2,Problem,List<&2,Element>>) -> Result<&2,&2,Problem,Element>:
  match values:
    case Done{values}: only_value(~Element,shape,values)
    case Fail{problem}: Fail{problem}

def item(~Element: Data,tensor: Tensor<Element>) -> Result<&2,&2,Problem,Element>:
  with_whole(~Element,~Result<&2,&2,Problem,Element>,tensor,storage => shape => fill => status => workers =>
    +shape = shape
    single(~Element,shape,listed_buffer(~Element,status,storage)))

# The value read, or fallback when the tensor carries a failure.
def value_or(~Value: Data,result: Result<&2,&2,Problem,Value>,fallback: Value) -> Value:
  match result:
    case Done{value}: value
    case Fail{problem}: fallback

# Operations consume their tensor arguments and return one result; callers
# keep a tensor by copying it first. Binary elementwise operands follow NumPy
# broadcasting; the result reuses the left storage when it has the left
# extents. The result carries the first failure of its operands.

def first_failure(left: Status,right: Status) -> Status:
  match left:
    case Usable{}: right
    case Failed{problem}: Failed{problem}

# A view of an operand read with the extents target; see Views.stretch.
def stretched(view: Maybe<&2,Views.View>,+target: List<&2,U32>) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.stretch(view,target)
    case None{}: None{}

def with_status(~Element: Data,status: Status,tensor: Tensor<Element>) -> Tensor<Element>:
  Tensor{bands,geometry,shape,fill,previous,workers} = tensor
  Tensor{bands,geometry,shape,fill,status,workers}

def offered(~Element: Data,+geometry: Bands.Geometry,shape: List<&2,U32>,fill: Element,status: Status,workers: U32,
  owners: Bands.Bands(~Element) & Bands.Plan) -> Tensor<Element> & BandSchedule.Target:
  (bands,+layout) = owners
  (Tensor{bands,geometry,shape,fill,status,workers},BandSchedule.Target{geometry,layout,BandSchedule.splits(layout)})

# The layout a tensor offers to an operation: its own geometry and plan.
def offer(~Element: Data,tensor: Tensor<Element>) -> Tensor<Element> & BandSchedule.Target:
  Tensor{bands,+geometry,shape,fill,status,workers} = tensor
  offered(~Element,geometry,shape,fill,status,workers,Bands.plan(~Element,bands))

# Read-only operations return their tensor in a pair; Bend destructures only
# parameters, so a continuation receives both halves.
def then(~First: Type,~Second: Type,~Result: Type,pair: First & Second,body: First -> Second -> Result) -> Result:
  (first,second) = pair
  body(first,second)

# The two operands arrive in the layout they met in and one band kernel
# combines them.
def combine_met_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,operations: F32,
  target: BandSchedule.Target,shape: List<&2,U32>,fill: Element,+workers: U32,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  BandSchedule.Target{+geometry,+layout,split} = target
  Tensor{Bands.combine_with(~Element,~kernel,geometry,BandSchedule.parallel(geometry,workers,12.0,operations,layout),arrive(~Element,geometry,layout,left),arrive(~Element,geometry,layout,right)),
    geometry,shape,fill,Usable{},workers}

def combine_offered_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,operations: F32,
  +workers: U32,+shape: List<&2,U32>,+fill: Element,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  then(~Tensor<Element>,~BandSchedule.Target,~Tensor<Element>,offer(~Element,left),left => left_target =>
    then(~Tensor<Element>,~BandSchedule.Target,~Tensor<Element>,offer(~Element,right),right => right_target =>
      combine_met_with(~Element,~kernel,operations,BandSchedule.meet(left_target,right_target),shape,fill,workers,left,right)))

# Usable operands of equal extents meet in one layout. A failed operand fails
# the call, which keeps the left bands, geometry, extents and fill.
def combine_equal_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,operations: F32,
  status: Status,+workers: U32,+shape: List<&2,U32>,left: Bands.Bands(~Element),+geometry: Bands.Geometry,+left_fill: Element,
  right: Bands.Bands(~Element),+right_geometry: Bands.Geometry,right_fill: Element) -> Tensor<Element>:
  match status:
    case Usable{}:
      combine_offered_with(~Element,~kernel,operations,workers,shape,left_fill,Tensor{left,geometry,shape,left_fill,Usable{},workers},Tensor{right,right_geometry,shape,right_fill,Usable{},workers})
    case Failed{problem}: Tensor{left,geometry,shape,left_fill,Failed{problem},workers}

# A checked copy of the elements a view selects into a new whole tensor:
# (result, source).
def copied(~Element: Data,name: String,+source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Tensor<Element> & Tensor<Element>:
  match result:
    case Operations.Accepted{(input,output)}:
      (whole(~Element,output,target_shape,fill,Usable{},workers),whole(~Element,input,source_shape,fill,Usable{},workers))
    case Operations.Rejected{(input,output)}:
      (whole(~Element,output,target_shape,fill,Failed{Problem{name,InvalidArgument{source_shape}}},workers),
        whole(~Element,input,source_shape,fill,Usable{},workers))

def gather_views(~Element: Data,name: String,source_view: Maybe<&2,Views.View>,+reversed: List<&2,Bool>,target_view: Maybe<&2,Views.View>,count: Maybe<&2,U32>,
  input: Storage.Buffer<Element>,+source_shape: List<&2,U32>,+target_shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  Tensor<Element> & Tensor<Element>:
  match source_view target_view count:
    case Some{source_view} Some{target_view} Some{count}:
      copied(~Element,name,source_shape,target_shape,fill,workers,Operations.copy(~Element,input,source_view,reversed,buffer(~Element,fill,count),target_view))
    case source_view target_view count:
      (failed(~Element,name,InvalidArgument{source_shape},target_shape,fill,workers),whole(~Element,input,source_shape,fill,Usable{},workers))

# Copy the elements selected by source_view, in its row-major order with the
# flagged axes in reverse, into a new tensor whose storage is laid out as
# view_shape and labelled target_shape.
def gather_status(~Element: Data,name: String,status: Status,source_view: Maybe<&2,Views.View>,+reversed: List<&2,Bool>,+view_shape: List<&2,U32>,
  +target_shape: List<&2,U32>,input: Storage.Buffer<Element>,+source_shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  Tensor<Element> & Tensor<Element>:
  match status:
    case Usable{}:
      gather_views(~Element,name,source_view,reversed,view_of(view_shape),count_of(view_shape),input,source_shape,target_shape,fill,workers)
    case Failed{+problem}:
      (whole(~Element,buffer(~Element,fill,0),target_shape,fill,Failed{problem},workers),
        whole(~Element,input,source_shape,fill,Failed{problem},workers))

def result_of(~Element: Data,pair: Tensor<Element> & Tensor<Element>) -> Tensor<Element>:
  (result,source) = pair
  result

# What every band of a gather shares: the operation, the banded view of the
# input geometry, the flags of the local view and the fill of fresh storage.
type Gathered<-Element: Data> is Data:
  Gathered{name: String,banded: BandViews.Banded,geometry: Bands.Geometry,reversed: List<&2,Bool>,fill: Element}

# A band gathers the cells its local view selects into the output band of the
# same rows, each row repeated as the view repeats it. A band of no rows has
# no cells to read.
def gathered_rows(~Element: Data,empty: Bool,name: String,+banded: BandViews.Banded,+batch: U32,+channels: U32,+height: U32,+width: U32,+reversed: List<&2,Bool>,+fill: Element,
  +first: U32,+rows: U32,input: Storage.Buffer<Element>) -> Bands.Transformed(~Element):
  match empty:
    case True{}:
      +scale = BandViews.scale(banded)
      checked_band(~Element,(first * scale : U32),(rows * scale : U32),whole(~Element,buffer(~Element,fill,0),BandViews.extents(banded,rows),fill,Usable{},1))
    case False{}:
      +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
      +scale = BandViews.scale(banded)
      +extents = BandViews.extents(banded,rows)
      checked_band(~Element,(first * scale : U32),(rows * scale : U32),
        result_of(~Element,gather_status(~Element,name,Usable{},Some{BandViews.local(banded,geometry,rows)},reversed,extents,extents,input,[channels,rows,width],fill,1)))

def gathered_band(~Element: Data,context: Gathered<Element>,first: U32,rows: U32,input: Storage.Buffer<Element>) -> Bands.Transformed(~Element):
  Gathered{name,+banded,Bands.Geometry{+batch,+channels,+height,+width},+reversed,+fill} = context
  +first = first
  +rows = rows
  gathered_rows(~Element,U32.is_zero(rows),name,banded,batch,channels,height,width,reversed,fill,first,rows,input)

def gather_banded(~Element: Data,+name: String,+banded: BandViews.Banded,+geometry: Bands.Geometry,+reversed: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,bands: Bands.Bands(~Element)) -> Tensor<Element>:
  transformed(~Element,name,InvalidArgument{shape},BandViews.output(banded,geometry),target,fill,workers,
    BandSchedule.transform(~Element,~Gathered<Element>,~gathered_band(~Element),Gathered{name,banded,geometry,reversed,fill},geometry,workers,12.0,0.0,bands))

def gather_whole_local(~Element: Data,+name: String,+banded: BandViews.Banded,local: Maybe<&2,List<&2,Bool>>,+whole: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,tensor: Tensor<Element>) -> Tensor<Element>:
  match local:
    case Some{+flags}: gather_banded(~Element,name,banded,whole,flags,shape,target,fill,workers,arrive(~Element,whole,Bands.unsplit(whole),tensor))
    case None{}: failed(~Element,name,InvalidArgument{shape},target,fill,workers)

def gather_whole(~Element: Data,+name: String,found: Maybe<&2,BandViews.Banded>,+reversed: List<&2,Bool>,+whole: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,tensor: Tensor<Element>) -> Tensor<Element>:
  match found:
    case Some{+banded}: gather_whole_local(~Element,name,banded,BandViews.local_flags(banded,whole,reversed),whole,shape,target,fill,workers,tensor)
    case None{}: failed(~Element,name,InvalidArgument{shape},target,fill,workers)

# Other bands first arrive as the one band of one row of the whole geometry.
def gather_arrived(~Element: Data,+name: String,+reversed: List<&2,Bool>,+axes: List<&2,Views.Axis>,+offset: U32,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  +whole = {Bands.Geometry{1,1,1,count_or_zero(shape)} : Bands.Geometry}
  gather_whole(~Element,name,BandViews.classify(axes,offset,whole),reversed,whole,shape,target,fill,workers,Tensor{bands,geometry,shape,fill,Usable{},workers})

def gather_local(~Element: Data,+name: String,+banded: BandViews.Banded,local: Maybe<&2,List<&2,Bool>>,+reversed: List<&2,Bool>,+axes: List<&2,Views.Axis>,+offset: U32,
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match local:
    case Some{+flags}: gather_banded(~Element,name,banded,geometry,flags,shape,target,fill,workers,bands)
    case None{}: gather_arrived(~Element,name,reversed,axes,offset,bands,geometry,shape,target,fill,workers)

# Bands whose images and rows the view keeps in place and reads forward
# gather their own cells; other bands first arrive as one band.
def gather_found(~Element: Data,+name: String,found: Maybe<&2,BandViews.Banded>,+reversed: List<&2,Bool>,+axes: List<&2,Views.Axis>,+offset: U32,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,
  +shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match found:
    case Some{+banded}: gather_local(~Element,name,banded,BandViews.local_flags(banded,geometry,reversed),reversed,axes,offset,bands,geometry,shape,target,fill,workers)
    case None{}: gather_arrived(~Element,name,reversed,axes,offset,bands,geometry,shape,target,fill,workers)

def gather_counted(~Element: Data,+name: String,empty: Bool,+axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match empty:
    case True{}: whole(~Element,buffer(~Element,fill,0),target,fill,Usable{},workers)
    case False{}: gather_found(~Element,name,BandViews.classify(axes,offset,geometry),reversed,axes,offset,bands,geometry,shape,target,fill,workers)

# The result must have a count. The view comes taken apart, so that no View
# value is used twice: that would make every View constructor shared.
def gather_axes(~Element: Data,+name: String,count: Maybe<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match count:
    case Some{+count}: gather_counted(~Element,name,U32.is_zero(count),axes,offset,reversed,bands,geometry,shape,target,fill,workers)
    case None{}: failed(~Element,name,InvalidArgument{shape},target,fill,workers)

# The elements a view of the tensor's row-major elements selects, in the
# view's row-major order with the flagged axes in reverse, as a tensor of
# extents target. An axis without a flag is read forward.
def gather(~Element: Data,name: String,status: Status,source: Maybe<&2,Views.View>,+reversed: List<&2,Bool>,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  +shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match status source:
    case Usable{} Some{Views.View{+axes,+offset}}: gather_axes(~Element,name,count_of(target),axes,offset,reversed,bands,geometry,shape,target,fill,workers)
    case Usable{} None{}: failed(~Element,name,InvalidArgument{shape},target,fill,workers)
    case Failed{+problem} source: whole(~Element,buffer(~Element,fill,0),target,fill,Failed{problem},workers)

# The same gather labelled with the extents of the view.
def gather_extents(~Element: Data,name: String,status: Status,source: Maybe<&2,Views.View>,+reversed: List<&2,Bool>,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  +shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match source:
    case Some{Views.View{+axes,offset}}: gather(~Element,name,status,Some{Views.View{axes,offset}},reversed,bands,geometry,shape,Views.axis_extents(axes),fill,workers)
    case None{}: gather(~Element,name,status,None{},reversed,bands,geometry,shape,[],fill,workers)

# The tensor repeated to the extents target under NumPy broadcasting: one
# gather through the stretched view.
def broadcast_view(~Element: Data,status: Status,view: Maybe<&2,Views.View>,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,
  +target: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match status view:
    case Usable{} None{}: failed(~Element,"broadcast_to",ShapeMismatch{shape,target},target,fill,workers)
    case status view: gather(~Element,"broadcast_to",status,view,[],bands,geometry,shape,target,fill,workers)

def broadcast_to(~Element: Data,tensor: Tensor<Element>,+target: List<&2,U32>) -> Tensor<Element>:
  Tensor{bands,+geometry,+shape,+fill,status,+workers} = tensor
  broadcast_view(~Element,status,stretched(view_of(shape),target),bands,geometry,shape,target,fill,workers)

# Each band of the left takes the part of the right source it reads through
# the stretched view. A layout whose bands have no view of their cells reads
# the right repeated to the common extents.
def combine_spread_when(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,operations: F32,
  boxed: Bool,+shape: List<&2,U32>,+workers: U32,+geometry: Bands.Geometry,+layout: Bands.Plan,left: Bands.Bands(~Element),+fill: Element,
  source: Storage.Buffer<Element>,+right_shape: List<&2,U32>,right_fill: Element) -> Tensor<Element>:
  match boxed:
    case True{}:
      Tensor{Bands.combine_spread_with(~Element,~kernel,Bands.Spread{geometry,shape,right_shape},BandSchedule.parallel(geometry,workers,12.0,operations,layout),left,
        Bands.spread(~Element,Bands.Spread{geometry,shape,right_shape},right_fill,layout,source)),geometry,shape,fill,Usable{},workers}
    case False{}:
      combine_offered_with(~Element,~kernel,operations,workers,shape,fill,Tensor{left,geometry,shape,fill,Usable{},workers},
        broadcast_to(~Element,whole(~Element,source,right_shape,right_fill,Usable{},workers),shape))

def combine_viewed_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,operations: F32,
  +shape: List<&2,U32>,+workers: U32,+geometry: Bands.Geometry,+fill: Element,planned: Bands.Bands(~Element) & Bands.Plan,
  source: Storage.Buffer<Element>,+right_shape: List<&2,U32>,right_fill: Element) -> Tensor<Element>:
  (left,+layout) = planned
  combine_spread_when(~Element,~kernel,operations,Bands.spread_boxed(Bands.Spread{geometry,shape,right_shape},layout) && Bool.not(Shapes.has_zero(shape)),
    shape,workers,geometry,layout,left,fill,source,right_shape,right_fill)

# A left operand of the common extents keeps its layout; the right operand is
# read through its view stretched to those extents.
def combine_stretched_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,operations: F32,
  +shape: List<&2,U32>,+workers: U32,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  Tensor{bands,+geometry,left_shape,+fill,status,left_workers} = left
  with_whole(~Element,~Tensor<Element>,right,source => right_shape => right_fill => right_status => right_workers =>
    +right_shape = right_shape
    combine_viewed_with(~Element,~kernel,operations,shape,workers,geometry,fill,Bands.plan(~Element,bands),source,right_shape,right_fill))

def combine_repeated_when(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,operations: F32,
  same: Bool,+shape: List<&2,U32>,+workers: U32,+fill: Element,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  match same:
    case True{}: combine_offered_with(~Element,~kernel,operations,workers,shape,fill,left,right)
    case False{}: combine_stretched_with(~Element,~kernel,operations,shape,workers,left,right)

# A repeated left operand meets a right operand of the common extents as an
# operand of equal extents.
def combine_repeated_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,operations: F32,
  +shape: List<&2,U32>,+workers: U32,+fill: Element,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  Tensor{right_bands,right_geometry,+right_shape,right_fill,right_status,right_workers} = right
  combine_repeated_when(~Element,~kernel,operations,Shapes.same_extents(right_shape,shape),shape,workers,fill,left,
    Tensor{right_bands,right_geometry,right_shape,right_fill,right_status,right_workers})

# A left operand of other extents is first repeated to the common extents.
def combine_common_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,operations: F32,
  same: Bool,count: Maybe<&2,U32>,+shape: List<&2,U32>,+workers: U32,+fill: Element,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  match same count:
    case True{} count: combine_stretched_with(~Element,~kernel,operations,shape,workers,left,right)
    case False{} Some{count}: combine_repeated_with(~Element,~kernel,operations,shape,workers,fill,broadcast_to(~Element,left,shape),right)
    case False{} None{}: failed(~Element,name,InvalidShape{shape},shape,fill,workers)

# Operands of other extents follow NumPy broadcasting (Shapes.broadcast) and
# fail only when they are incompatible.
def combine_broadcast_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,operations: F32,
  status: Status,common: Maybe<&2,List<&2,U32>>,+workers: U32,left: Bands.Bands(~Element),+geometry: Bands.Geometry,+left_shape: List<&2,U32>,+left_fill: Element,
  right: Bands.Bands(~Element),+right_geometry: Bands.Geometry,+right_shape: List<&2,U32>,right_fill: Element) -> Tensor<Element>:
  match status common:
    case Usable{} Some{+shape}:
      combine_common_with(~Element,~kernel,name,operations,Shapes.same_extents(left_shape,shape),count_of(shape),shape,workers,left_fill,
        Tensor{left,geometry,left_shape,left_fill,Usable{},workers},Tensor{right,right_geometry,right_shape,right_fill,Usable{},workers})
    case Usable{} None{}: Tensor{left,geometry,left_shape,left_fill,Failed{Problem{name,ShapeMismatch{left_shape,right_shape}}},workers}
    case Failed{problem} common: Tensor{left,geometry,left_shape,left_fill,Failed{problem},workers}

# Operands of equal extents run in the layout they meet in; operands of other
# extents broadcast.
def combine_shapes(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,operations: F32,same: Bool,
  left: Bands.Bands(~Element),+geometry: Bands.Geometry,+left_shape: List<&2,U32>,+left_fill: Element,left_status: Status,left_workers: U32,
  right: Bands.Bands(~Element),+right_geometry: Bands.Geometry,+right_shape: List<&2,U32>,right_fill: Element,right_status: Status,right_workers: U32) -> Tensor<Element>:
  match same:
    case True{}:
      combine_equal_with(~Element,~kernel,operations,first_failure(left_status,right_status),U32.max(left_workers,right_workers),left_shape,left,geometry,left_fill,right,right_geometry,right_fill)
    case False{}:
      combine_broadcast_with(~Element,~kernel,name,operations,first_failure(left_status,right_status),Shapes.broadcast(left_shape,right_shape),U32.max(left_workers,right_workers),
        left,geometry,left_shape,left_fill,right,right_geometry,right_shape,right_fill)

# operation(left element, right element), retaining the band layout of the
# operands. Alignment is done once, before any parallel leaf work.
# operations counts the operation's scalar work for the band schedule.
def combine_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,name: String,operations: F32,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  match left right:
    case Tensor{left,geometry,+left_shape,left_fill,left_status,left_workers} Tensor{right,right_geometry,+right_shape,right_fill,right_status,right_workers}:
      combine_shapes(~Element,~kernel,name,operations,Shapes.same_extents(left_shape,right_shape),
        left,geometry,left_shape,left_fill,left_status,left_workers,right,right_geometry,right_shape,right_fill,right_status,right_workers)

# The elementwise form: the kernel applies operation to each pair in order.
def combine(~Element: Data,~operation: Element -> Element -> Element,name: String,operations: F32,left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  combine_with(~Element,~Operations.combine_kernel(~Element,~operation),name,operations,left,right)

def apply_bands_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,operations: F32,status: Status,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  shape: List<&2,U32>,fill: Element,+workers: U32) -> Tensor<Element>:
  match status:
    case Usable{}: Tensor{BandSchedule.map_context(~Element,~Context,~mapper,context,geometry,workers,operations,bands),geometry,shape,fill,Usable{},workers}
    case Failed{problem}: Tensor{bands,geometry,shape,fill,Failed{problem},workers}

def apply_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,operations: F32,tensor: Tensor<Element>) -> Tensor<Element>:
  Tensor{bands,geometry,shape,fill,status,workers} = tensor
  apply_bands_context(~Element,~Context,~mapper,context,operations,status,bands,geometry,shape,fill,workers)

def apply(~Element: Data,~mapper: @+count: Nat -> @+position: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element>,operations: F32,tensor: Tensor<Element>) -> Tensor<Element>:
  apply_context(~Element,~Unit,~Operations.without_context(~Element,~mapper),Unit{},operations,tensor)

def indexed_width_positive(geometry: Bands.Geometry) -> Bool:
  Bands.Geometry{batch,channels,height,width} = geometry
  Nat.is_lt(0n,U32.to_nat(width))

# An empty width holds no element, so the bands stay as they are. The choice
# is made once, before the band scheduler or element loop.
def apply_indexed_positive(~Element: Data,~operation: U32 -> Element -> Element,positive: Bool,operations: F32,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  +shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match positive:
    case True{}: Tensor{BandSchedule.indexed(~Element,~operation,geometry,workers,operations,fill,bands),geometry,shape,fill,Usable{},workers}
    case False{}: Tensor{bands,geometry,shape,fill,Usable{},workers}

def apply_indexed_bands(~Element: Data,~operation: U32 -> Element -> Element,operations: F32,status: Status,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match status:
    case Usable{}: apply_indexed_positive(~Element,~operation,indexed_width_positive(geometry),operations,bands,geometry,shape,fill,workers)
    case Failed{problem}: Tensor{bands,geometry,shape,fill,Failed{problem},workers}

# The function also receives each element's row-major index.
def apply_indexed(~Element: Data,~operation: U32 -> Element -> Element,operations: F32,tensor: Tensor<Element>) -> Tensor<Element>:
  Tensor{bands,geometry,shape,fill,status,workers} = tensor
  apply_indexed_bands(~Element,~operation,operations,status,bands,geometry,shape,fill,workers)

def same(~Element: Data,value: Element) -> Element:
  value

# Updates every element x to operation(x, value).
def scalar_bands(~Element: Data,~operation: Element -> Element -> Element,operations: F32,status: Status,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  shape: List<&2,U32>,fill: Element,+workers: U32,value: Element) -> Tensor<Element>:
  match status:
    case Usable{}: Tensor{BandSchedule.scalar(~Element,~operation,geometry,workers,operations,value,bands),geometry,shape,fill,Usable{},workers}
    case Failed{problem}: Tensor{bands,geometry,shape,fill,Failed{problem},workers}

def combine_scalar(~Element: Data,~operation: Element -> Element -> Element,operations: F32,tensor: Tensor<Element>,value: Element) -> Tensor<Element>:
  Tensor{bands,geometry,shape,fill,status,workers} = tensor
  scalar_bands(~Element,~operation,operations,status,bands,geometry,shape,fill,workers,value)

# Messages

def show_extents(extents: List<&2,U32>) -> String:
  match extents:
    case []: ""
    case [extent]: U32.show(extent)
    case extent <> rest: String.append(U32.show(extent),String.append(", ",show_extents(rest)))

def show_shape(extents: List<&2,U32>) -> String:
  String.append("[",String.append(show_extents(extents),"]"))

def explain(reason: Reason) -> String:
  match reason:
    case ShapeMismatch{left,right}:
      String.append("shapes ",String.append(show_shape(left),String.append(" and ",String.append(show_shape(right)," do not match"))))
    case InvalidShape{shape}:
      String.append("shape ",String.append(show_shape(shape)," is invalid or exceeds the machine capacity"))
    case InvalidArgument{shape}:
      String.append("invalid argument for shape ",show_shape(shape))
    case FileFault{path,detail}:
      String.append("file ",String.append(path,String.append(" ",detail)))
    case MissingName{name}:
      String.append("no tensor named ",name)

def describe(problem: Problem) -> String:
  Problem{operation,reason} = problem
  String.append(operation,String.append(": ",explain(reason)))

def expected(~Element: Data,result: Result<&1,&1,Tensor<Element> & Problem,Tensor<Element>>) -> IO(Tensor<Element>):
  match result:
    case Done{tensor}: IO.pure(Tensor<Element>,tensor)
    case Fail{(tensor,problem)}: IO.die(Tensor<Element>,1,describe(problem))

# Stops the program with the recorded problem, for scripts and examples.
def expect(~Element: Data,tensor: Tensor<Element>) -> IO(Tensor<Element>):
  expected(~Element,check(~Element,tensor))

def expected_storage(~Element: Data,result: Result<&2,&1,Problem,Storage.Buffer<Element>>) -> IO(Storage.Buffer<Element>):
  match result:
    case Done{storage}: IO.pure(Storage.Buffer<Element>,storage)
    case Fail{problem}: IO.die(Storage.Buffer<Element>,1,describe(problem))

def expect_storage(~Element: Data,tensor: Tensor<Element>) -> IO(Storage.Buffer<Element>):
  expected_storage(~Element,into_storage(~Element,tensor))
