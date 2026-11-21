# A public tensor owns pieces of its stored elements, the shape of those
# elements, the value used to initialize allocations, a sticky status, and
# the number of workers its operations may use. The pieces are the grid of
# the shape (lib/tensor_grid.bend): two tensors of one shape have the same
# pieces. A failed operation leaves values untouched and records why. Later
# operations skip a failed tensor, and check is the only way to observe or
# clear the failure.
import Base
import ./storage_buffer.bend as Storage
import ./storage_allocation.bend as Allocation
import ./storage_copy.bend as Copy
import ./storage_clone.bend as Clone
import ./tensor_shape.bend as Shapes
import ./tensor_view.bend as Views
import ./tensor_operations.bend as Operations
import ./machine_limits.bend as Machine
import ./default_machine_limits.bend as Default
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_tasks.bend as Tasks
import ./tensor_move.bend as Move
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
  Tensor{pieces: Pieces.Pieces(~Element),shape: List<&2,U32>,fill: Element,status: Status,workers: U32}

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

def grid_of(+shape: List<&2,U32>) -> Grid.Grid:
  Grid.grid(shape,Grid.piece_target())

# ---- Pieces from one buffer and back ----

def cut_copied(~Element: Data,+first: U32,+count: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Storage.Buffer<Element> & Pieces.Pieces(~Element):
  (source,piece) = owners
  (source,Partition.Leaf{piece,first,count,Unit{}})

def cut_piece(~Element: Data,fill: Element,first: U32,count: U32,source: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Pieces.Pieces(~Element):
  +fill = fill
  +first = first
  +count = count
  cut_copied(~Element,first,count,Copy.range(~Element,count,first,0,source,buffer(~Element,fill,count)))

def cut_when(~Element: Data,single: Bool,+grid: Grid.Grid,+fill: Element,source: Storage.Buffer<Element>) -> Pieces.Pieces(~Element):
  match single:
    case True{}: Partition.Leaf{source,0,Grid.count_of(grid),Unit{}}
    case False{}:
      Pair.snd(Storage.Buffer<Element>,Pieces.Pieces(~Element),
        Pieces.build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Element,~cut_piece(~Element),Pieces.ranges_of_grid(grid),fill,source))

# One buffer of the elements of a shape in the pieces of its grid. A grid of
# one piece keeps the buffer, otherwise every piece copies its range.
def cut(~Element: Data,+shape: List<&2,U32>,+fill: Element,source: Storage.Buffer<Element>) -> Pieces.Pieces(~Element):
  +grid = grid_of(shape)
  cut_when(~Element,U32.is_le(Grid.pieces_of(grid),1),grid,fill,source)

# The pieces back in one buffer, in order: a copy of their cells. A single
# piece is the buffer.
def joined(~Element: Data,+count: U32,+fill: Element,pieces: Pieces.Pieces(~Element)) -> Storage.Buffer<Element>:
  match pieces:
    case Partition.Leaf{data,first,span,details}: data
    case Partition.Fork{left,right}:
      Pair.snd(Move.Holding<Element>,Storage.Buffer<Element>,Move.copied(~Element,Move.held(~Element,Partition.Fork{left,right}),0,count,fill))

def filled_piece(~Element: Data,fill: Element,first: U32,count: U32,state: Unit) -> Unit & Pieces.Pieces(~Element):
  +fill = fill
  +first = first
  +count = count
  (Unit{},Partition.Leaf{buffer(~Element,fill,count),first,count,Unit{}})

# Pieces of a grid, every element value.
def filled_grid(~Element: Data,+grid: Grid.Grid,+value: Element) -> Pieces.Pieces(~Element):
  Pair.snd(Unit,Pieces.Pieces(~Element),Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~filled_piece(~Element),Pieces.ranges_of_grid(grid),value,Unit{}))

# Pieces of a shape, every element value.
def filled(~Element: Data,+shape: List<&2,U32>,+value: Element) -> Pieces.Pieces(~Element):
  filled_grid(~Element,grid_of(shape),value)

def whole_status(~Element: Data,status: Status,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,workers: U32) -> Tensor<Element>:
  match status:
    case Usable{}: Tensor{cut(~Element,shape,fill,storage),shape,fill,Usable{},workers}
    case Failed{problem}: Tensor{Partition.Leaf{storage,0,0,Unit{}},shape,fill,Failed{problem},workers}

# Row-major storage of a usable tensor as its pieces. A failed tensor keeps
# the storage as one piece.
def whole(~Element: Data,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,status: Status,workers: U32) -> Tensor<Element>:
  whole_status(~Element,status,storage,shape,fill,workers)

# Values at the boundary of the library: row-major storage of a shape kept as
# one piece. Only the boundary reads them, the tiled form takes them whole.
def values(~Element: Data,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,status: Status,workers: U32) -> Tensor<Element>:
  Tensor{Partition.Leaf{storage,0,count_or_zero(shape),Unit{}},shape,fill,status,workers}

# The storage of a tensor in row-major order.
def with_whole(~Element: Data,~Result: Type,tensor: Tensor<Element>,
  body: Storage.Buffer<Element> -> List<&2,U32> -> Element -> Status -> U32 -> Result) -> Result:
  Tensor{pieces,+shape,+fill,+status,+workers} = tensor
  body(joined(~Element,count_or_zero(shape),fill,pieces),shape,fill,status,workers)

# Metadata inspection retains the pieces and their owners.
def with_metadata(~Element: Data,~Result: Type,tensor: Tensor<Element>,
  body: Tensor<Element> -> List<&2,U32> -> Element -> Status -> U32 -> Result) -> Result:
  Tensor{pieces,+shape,+fill,+status,+workers} = tensor
  body(Tensor{pieces,shape,fill,status,workers},shape,fill,status,workers)

def usable(status: Status) -> Bool:
  match status:
    case Usable{}: True{}
    case Failed{problem}: False{}

# A failed tensor holds one empty piece.
def failed(~Element: Data,operation: String,reason: Reason,shape: List<&2,U32>,+fill: Element,workers: U32) -> Tensor<Element>:
  Tensor{Partition.Leaf{buffer(~Element,fill,0),0,0,Unit{}},shape,fill,Failed{Problem{operation,reason}},workers}

# Construction

def full_count(~Element: Data,+shape: List<&2,U32>,count: Maybe<&2,U32>,+value: Element) -> Tensor<Element>:
  match count:
    case Some{count}: Tensor{filled(~Element,shape,value),shape,value,Usable{},1}
    case None{}: failed(~Element,"full",InvalidShape{shape},shape,value,1)

def full(~Element: Data,+shape: List<&2,U32>,+value: Element) -> Tensor<Element>:
  full_count(~Element,shape,count_of(shape),value)

def listed_count(~Element: Data,matches: Bool,+shape: List<&2,U32>,storage: Storage.Buffer<Element>,+fill: Element) -> Tensor<Element>:
  match matches:
    case True{}: whole(~Element,storage,shape,fill,Usable{},1)
    case False{}: failed(~Element,"from_list",InvalidArgument{shape},shape,fill,1)

def listed_written(~Element: Data,+shape: List<&2,U32>,written: Storage.Buffer<Element> & Bool,+fill: Element) -> Tensor<Element>:
  (storage,matches) = written
  listed_count(~Element,matches,shape,storage,fill)

# The list is written while it is counted, so it is read once.
def listed(~Element: Data,+shape: List<&2,U32>,count: Maybe<&2,U32>,values: List<&2,Element>,+fill: Element) -> Tensor<Element>:
  match count:
    case Some{+count}:
      listed_written(~Element,shape,Storage.write_exactly(~Element,values,0,U32.to_nat(count),Storage.allocate(~Element,depth_for(count),fill,U32.to_nat(count))),fill)
    case None{}: failed(~Element,"from_list",InvalidShape{shape},shape,fill,1)

def values_count(~Element: Data,matches: Bool,+shape: List<&2,U32>,+count: U32,items: List<&2,Element>,+fill: Element) -> Tensor<Element>:
  match matches:
    case True{}: values(~Element,Storage.from_list(~Element,items,depth_for(count),fill,U32.to_nat(count)),shape,fill,Usable{},1)
    case False{}: failed(~Element,"from_list",InvalidArgument{shape},shape,fill,1)

# The list of an entry is counted before its buffer is allocated.
def values_listed(~Element: Data,+shape: List<&2,U32>,count: Maybe<&2,U32>,+items: List<&2,Element>,+fill: Element) -> Tensor<Element>:
  match count:
    case Some{+count}: values_count(~Element,Nat.is_eq(List.length(&2,Element,items),U32.to_nat(count)),shape,count,items,fill)
    case None{}: failed(~Element,"from_list",InvalidShape{shape},shape,fill,1)

# Values of a shape from a list, one piece.
def values_of(~Element: Data,+shape: List<&2,U32>,+items: List<&2,Element>,+fill: Element) -> Tensor<Element>:
  values_listed(~Element,shape,count_of(shape),items,fill)

# fill initializes storage this tensor or its results allocate; it is never
# observable in a successful result.
def from_list(~Element: Data,+shape: List<&2,U32>,+values: List<&2,Element>,+fill: Element) -> Tensor<Element>:
  listed(~Element,shape,count_of(shape),values,fill)

def stored(~Element: Data,matches: Bool,storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element) -> Tensor<Element>:
  match matches:
    case True{}: values(~Element,storage,shape,fill,Usable{},1)
    case False{}: Tensor{Partition.Leaf{storage,0,0,Unit{}},shape,fill,Failed{Problem{"from_storage",InvalidShape{shape}}},1}

def matches_count(count: Maybe<&2,U32>,length: Nat) -> Bool:
  match count:
    case Some{count}: Nat.is_eq(length,U32.to_nat(count))
    case None{}: False{}

def storage_of(~Element: Data,+shape: List<&2,U32>,count: Maybe<&2,U32>,storage: Storage.Buffer<Element>,+fill: Element) -> Tensor<Element>:
  Storage.Buffer{storage,+length,certificate} = storage
  stored(~Element,matches_count(count,length),Storage.Buffer{storage,length,certificate},shape,fill)

# Storage handed over by foreign I/O keeps its documented trust boundary. The
# values stay one piece.
def from_storage(~Element: Data,+shape: List<&2,U32>,storage: Storage.Buffer<Element>,+fill: Element) -> Tensor<Element>:
  storage_of(~Element,shape,count_of(shape),storage,fill)

# Execution context

# Sets the worker count, at least one. Values and status are unchanged.
def with_workers(~Element: Data,tensor: Tensor<Element>,+count: U32) -> Tensor<Element>:
  Tensor{pieces,shape,fill,status,workers} = tensor
  Tensor{pieces,shape,fill,status,U32.max(1,count)}

def workers(~Element: Data,tensor: Tensor<Element>) -> Tensor<Element> & U32:
  Tensor{pieces,shape,fill,status,+workers} = tensor
  (Tensor{pieces,shape,fill,status,workers},workers)

# Observation

def shape(~Element: Data,tensor: Tensor<Element>) -> Tensor<Element> & List<&2,U32>:
  Tensor{pieces,+shape,fill,status,workers} = tensor
  (Tensor{pieces,shape,fill,status,workers},shape)

def checked(~Element: Data,status: Status,pieces: Pieces.Pieces(~Element),shape: List<&2,U32>,fill: Element,workers: U32) ->
  Result<&1,&1,Tensor<Element> & Problem,Tensor<Element>>:
  match status:
    case Usable{}: Done{Tensor{pieces,shape,fill,Usable{},workers}}
    case Failed{problem}: Fail{(Tensor{pieces,shape,fill,Usable{},workers},problem)}

def check(~Element: Data,tensor: Tensor<Element>) -> Result<&1,&1,Tensor<Element> & Problem,Tensor<Element>>:
  Tensor{pieces,shape,fill,status,workers} = tensor
  checked(~Element,status,pieces,shape,fill,workers)

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

# Operations consume their tensor arguments and return one result. Callers
# keep a tensor by copying it first. Binary elementwise operands follow NumPy
# broadcasting, and the result reuses the left storage when it has the left
# extents. The result carries the first failure of its operands.

def first_failure(left: Status,right: Status) -> Status:
  match left:
    case Usable{}: right
    case Failed{problem}: Failed{problem}

# A view of an operand read with the extents target, see Views.stretch.
def stretched(view: Maybe<&2,Views.View>,+target: List<&2,U32>) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.stretch(view,target)
    case None{}: None{}

def with_status(~Element: Data,status: Status,tensor: Tensor<Element>) -> Tensor<Element>:
  Tensor{pieces,shape,fill,previous,workers} = tensor
  Tensor{pieces,shape,fill,status,workers}

# ---- Work over pieces ----

# A map of one piece: the cells of first cell first and count cells in place.
def PieceMap(~Element: Data,~Context: Data) -> Type:
  @+context: Context -> @+first: U32 -> @+count: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element>

# The kernel of an elementwise combination: count left elements in place
# with the right elements a view reads.
def Kernel(~Element: Data) -> Type:
  @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>

def mapped_piece(~Element: Data,~Context: Data,~leaf: PieceMap(~Element,~Context),context: Context,first: U32,count: U32,buffer: Storage.Buffer<Element>,resource: Unit) -> Pieces.Pieces(~Element):
  +first = first
  +count = count
  Partition.Leaf{leaf(context,first,count,buffer),first,count,Unit{}}

def mapped_decided(~Element: Data,~Context: Data,~leaf: PieceMap(~Element,~Context),context: Context,decision: Tasks.Decision,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,
  fill: Element,+workers: U32) -> Tensor<Element>:
  Tasks.Decision{+parallel,+tasks} = decision
  Tensor{Pieces.grouped(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Context,~mapped_piece(~Element,~Context,~leaf),tasks,parallel,context,pieces),shape,fill,Usable{},workers}

# What one cell of a pass over pieces counts: a load of each operand, the rows
# of its operation and a store, and two steps of its loop.
def pass_rows(+operands: F32,+operations: F32) -> F32:
  (operands + operations + 1.0 : F32)

def pass_steps() -> F32:
  2.0

def mapped_status(~Element: Data,~Context: Data,~leaf: PieceMap(~Element,~Context),context: Context,operations: F32,status: Status,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,
  fill: Element,+workers: U32) -> Tensor<Element>:
  match status:
    case Usable{}: mapped_decided(~Element,~Context,~leaf,context,Tasks.decide(grid_of(shape),workers,pass_rows(1.0,operations),pass_steps(),Tasks.no_extra()),pieces,shape,fill,workers)
    case Failed{problem}: Tensor{pieces,shape,fill,Failed{problem},workers}

# The leaf on every piece in place, tasks deciding parallelism. operations
# counts the vector rows of the leaf's work on one cell.
def map_pieces(~Element: Data,~Context: Data,~leaf: PieceMap(~Element,~Context),context: Context,operations: F32,tensor: Tensor<Element>) -> Tensor<Element>:
  Tensor{pieces,shape,fill,status,workers} = tensor
  mapped_status(~Element,~Context,~leaf,context,operations,status,pieces,shape,fill,workers)

def same(~Element: Data,value: Element) -> Element:
  value

def combined(~Element: Data,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Storage.Buffer<Element>:
  match result:
    case Operations.Accepted{(left,right)}: left
    case Operations.Rejected{(left,right)}: left

# Two pieces of one range combined in place in the left, then the step after.
def combine_buffers(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),
  later: Later,first: U32,count: U32,left: Storage.Buffer<Element>,right: Storage.Buffer<Element>,resource: Unit) -> Pieces.Pieces(~Element):
  +first = first
  +count = count
  Partition.Leaf{after(later,first,count,combined(~Element,Operations.combine_with(~Element,~kernel,left,Views.View{[Views.Axis{count,1}],0},right,Views.View{[Views.Axis{count,1}],0}))),
    first,count,Unit{}}

def zipped_decided(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,
  decision: Tasks.Decision,+shape: List<&2,U32>,fill: Element,+workers: U32,pairs: Partition.Tree<&1,Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,Unit>) -> Tensor<Element>:
  Tasks.Decision{+parallel,+tasks} = decision
  Tensor{Pieces.grouped(~Pieces.Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Later,~Pieces.paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Later,~combine_buffers(~Element,~Later,~kernel,~after)),tasks,parallel,later,pairs),
    shape,fill,Usable{},workers}

# Operands of one shape have the same pieces: the trees zip leaf by leaf.
def combine_equal(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,operations: F32,
  status: Status,+workers: U32,+shape: List<&2,U32>,left: Pieces.Pieces(~Element),+fill: Element,right: Pieces.Pieces(~Element)) -> Tensor<Element>:
  match status:
    case Usable{}:
      zipped_decided(~Element,~Later,~kernel,~after,later,Tasks.decide(grid_of(shape),workers,pass_rows(2.0,operations),pass_steps(),Tasks.no_extra()),shape,fill,workers,Pieces.zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,left,right))
    case Failed{problem}: Tensor{left,shape,fill,Failed{problem},workers}

# What every piece of a broadcast reads: the shape of the result, the view of
# the right operand with its extents and the context of the step after it.
type Spread<-Later: Data> is Data:
  Spread{shape: List<&2,U32>,axes: List<&2,Views.Axis>,offset: U32,later: Later}

def spread_view(view: Maybe<&2,Views.View>) -> Views.View:
  match view:
    case Some{view}: view
    case None{}: Views.View{[],0}

def spread_outcome(~Element: Data,~Later: Data,~after: PieceMap(~Element,~Later),+later: Later,+first: U32,+count: U32,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) ->
  Pieces.Pieces(~Element) & Storage.Buffer<Element>:
  match result:
    case Operations.Accepted{(left,right)}: (Partition.Leaf{after(later,first,count,left),first,count,Unit{}},right)
    case Operations.Rejected{(left,right)}: (Partition.Leaf{after(later,first,count,left),first,count,Unit{}},right)

def spread_piece(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),
  context: Spread<Later>,first: U32,count: U32,left: Storage.Buffer<Element>,right: Storage.Buffer<Element>) -> Pieces.Pieces(~Element) & Storage.Buffer<Element>:
  Spread{+shape,+axes,+offset,+later} = context
  +first = first
  +count = count
  +grid = grid_of(shape)
  spread_outcome(~Element,~Later,~after,later,first,count,Operations.combine_with(~Element,~kernel,left,Grid.box_view(grid,shape,count),right,
    Grid.piece_view(grid,shape,first,count,Views.View{axes,offset})))

def shared_buffer(~Element: Data,buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Clone.clone(~Element,buffer)

def spread_decided(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,
  decision: Tasks.Decision,+workers: U32,+shape: List<&2,U32>,left: Pieces.Pieces(~Element),fill: Element,right: Storage.Buffer<Element>,+axes: List<&2,Views.Axis>,+offset: U32) ->
  Tensor<Element>:
  Tasks.Decision{+parallel,+tasks} = decision
  Tensor{Pieces.grouped_with(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Spread<Later>,~shared_buffer(~Element),~spread_piece(~Element,~Later,~kernel,~after),
    tasks,parallel,Spread{shape,axes,offset,later},left,right),shape,fill,Usable{},workers}

# A right operand is read whole by every task, one owner a task, through a
# view of it with the left extents: a task beyond the first clones it.
def combine_spread(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,operations: F32,
  +workers: U32,+shape: List<&2,U32>,left: Pieces.Pieces(~Element),+fill: Element,right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32) -> Tensor<Element>:
  spread_decided(~Element,~Later,~kernel,~after,later,
    Tasks.decide(grid_of(shape),workers,pass_rows(2.0,operations),pass_steps(),Tasks.Extra{(U32.to_f32(count_or_zero(right_shape)) * 4.0 : F32),0.0,0.0}),workers,shape,left,fill,right,axes,offset)

# A right operand of other extents read through its view stretched to the
# left extents.
def combine_stretched(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,operations: F32,
  +workers: U32,+shape: List<&2,U32>,left: Pieces.Pieces(~Element),+fill: Element,right: Storage.Buffer<Element>,+right_shape: List<&2,U32>,view: Views.View) ->
  Tensor<Element>:
  Views.View{+axes,+offset} = view
  combine_spread(~Element,~Later,~kernel,~after,later,operations,workers,shape,left,fill,right,right_shape,axes,offset)

# ---- Copies ----

def copy_leaf(~Element: Data,context: Unit,first: U32,count: U32,buffer: Storage.Buffer<Element>,resource: Unit) ->
  Partition.Tree<&1,(Storage.Buffer<Element> & Storage.Buffer<Element>),Unit>:
  Partition.Leaf{Clone.clone(~Element,buffer),first,count,Unit{}}

def copy_pair(~Element: Data,+first: U32,+count: U32,pair: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Pieces.Pieces(~Element) & Pieces.Pieces(~Element):
  (left,right) = pair
  (Partition.Leaf{left,first,count,Unit{}},Partition.Leaf{right,first,count,Unit{}})

def copy_halves(~Element: Data,left: Pieces.Pieces(~Element) & Pieces.Pieces(~Element),right: Pieces.Pieces(~Element) & Pieces.Pieces(~Element)) ->
  Pieces.Pieces(~Element) & Pieces.Pieces(~Element):
  (left_original,left_copy) = left
  (right_original,right_copy) = right
  (Partition.Fork{left_original,right_original},Partition.Fork{left_copy,right_copy})

# The tree of pairs as the two trees.
def unzipped(~Element: Data,tree: Partition.Tree<&1,(Storage.Buffer<Element> & Storage.Buffer<Element>),Unit>) -> Pieces.Pieces(~Element) & Pieces.Pieces(~Element):
  match tree:
    case Partition.Leaf{pair,+first,+count,details}: copy_pair(~Element,first,count,pair)
    case Partition.Fork{left,right}: copy_halves(~Element,unzipped(~Element,left),unzipped(~Element,right))

def copy_decided(~Element: Data,decision: Tasks.Decision,pieces: Pieces.Pieces(~Element)) -> Pieces.Pieces(~Element) & Pieces.Pieces(~Element):
  Tasks.Decision{+parallel,+tasks} = decision
  unzipped(~Element,Pieces.grouped(~Storage.Buffer<Element>,~(Storage.Buffer<Element> & Storage.Buffer<Element>),~Unit,~copy_leaf(~Element),tasks,parallel,Unit{},pieces))

def copies_of(~Element: Data,+shape: List<&2,U32>,+fill: Element,+workers: U32,pair: Pieces.Pieces(~Element) & Pieces.Pieces(~Element)) -> Tensor<Element> & Tensor<Element>:
  (left,right) = pair
  (Tensor{left,shape,fill,Usable{},workers},Tensor{right,shape,fill,Usable{},workers})

def copy_status(~Element: Data,+bytes_per_element: F32,status: Status,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element> & Tensor<Element>:
  match status:
    case Usable{}:
      copies_of(~Element,shape,fill,workers,copy_decided(~Element,Tasks.decide_copy(grid_of(shape),workers,bytes_per_element),pieces))
    case Failed{+problem}:
      (Tensor{pieces,shape,fill,Failed{problem},workers},Tensor{Partition.Leaf{buffer(~Element,fill,0),0,0,Unit{}},shape,fill,Failed{problem},workers})

# The only operation that keeps its argument: (original, copy). Every task
# copies its own pieces. bytes_per_element counts the bytes of one element.
def copied(~Element: Data,+bytes_per_element: F32,tensor: Tensor<Element>) -> Tensor<Element> & Tensor<Element>:
  Tensor{pieces,+shape,+fill,status,+workers} = tensor
  copy_status(~Element,bytes_per_element,status,pieces,shape,fill,workers)

# ---- Relabelled and moved cells ----

def flat_leaf(~Element: Data,+first: U32,+count: U32,pair: Move.Holding<Element> & Storage.Buffer<Element>) -> Move.Holding<Element> & Pieces.Pieces(~Element):
  (holding,output) = pair
  (holding,Partition.Leaf{output,first,count,Unit{}})

def flat_piece(~Element: Data,fill: Element,first: U32,count: U32,holding: Move.Holding<Element>) -> Move.Holding<Element> & Pieces.Pieces(~Element):
  +fill = fill
  +first = first
  +count = count
  flat_leaf(~Element,first,count,Move.copied_into(~Element,holding,first,count,buffer(~Element,fill,count)))

# The same cells in the pieces of another grid, in one serial pass.
def flat_moved(~Element: Data,+grid: Grid.Grid,+fill: Element,pieces: Pieces.Pieces(~Element)) -> Pieces.Pieces(~Element):
  Pair.snd(Move.Holding<Element>,Pieces.Pieces(~Element),
    Pieces.build(~Storage.Buffer<Element>,~Move.Holding<Element>,~Element,~flat_piece(~Element),Pieces.ranges_of_grid(grid),fill,Move.held(~Element,pieces)))

# The same grid keeps the tree. A grid that cuts the cells at the same places
# keeps every piece in a tree of its ranges. Another grid moves the cells.
def relabel_when(~Element: Data,same: Bool,regrouped: Bool,pieces: Pieces.Pieces(~Element),+target: List<&2,U32>,+fill: Element,workers: U32) -> Tensor<Element>:
  match same regrouped:
    case True{} regrouped: Tensor{pieces,target,fill,Usable{},workers}
    case False{} True{}:
      Tensor{Pieces.blocks_regrouped(~Storage.Buffer<Element>,~Element,~Pieces.vacant_buffer(~Element),Pieces.ranges_of_grid(grid_of(target)),fill,pieces),target,fill,Usable{},workers}
    case False{} False{}: Tensor{flat_moved(~Element,grid_of(target),fill,pieces),target,fill,Usable{},workers}

def relabel_status(~Element: Data,status: Status,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,workers: U32) -> Tensor<Element>:
  match status:
    case Usable{}:
      relabel_when(~Element,Grid.same_grid(grid_of(shape),grid_of(target)),Grid.same_pieces(grid_of(shape),grid_of(target)),pieces,target,fill,workers)
    case Failed{problem}: Tensor{pieces,target,fill,Failed{problem},workers}

# The same cells in the same order under another shape of the same count. A
# shape of the same grid keeps the pieces, another grid moves the cells into
# its pieces.
def relabeled(~Element: Data,tensor: Tensor<Element>,+target: List<&2,U32>) -> Tensor<Element>:
  Tensor{pieces,+shape,+fill,status,workers} = tensor
  relabel_status(~Element,status,pieces,shape,target,fill,workers)

# One source of a moving operation: the shape, grid and ranges of its cells
# and the cell whose coordinates fix the source axes no target axis reads.
type Origin is Data:
  Origin{shape: List<&2,U32>,grid: Grid.Grid,ranges: Pieces.Ranges,offset: U32}

def origin(+shape: List<&2,U32>,+offset: U32) -> Origin:
  +grid = grid_of(shape)
  Origin{shape,grid,Pieces.ranges_of_grid(grid),offset}

# The pieces of every source of a moving operation, in the order of its
# origins.
type Sources<-Element: Data> is Type:
  Ended{}
  Source{pieces: Pieces.Pieces(~Element),rest: Sources<Element>}

# What every task of a moving operation reads: the components, the sources,
# the shape and grid of the target, the fill of new storage and the context
# of the gather.
type Moving<-Element: Data,-Context: Data> is Data:
  Moving{components: Tasks.Components,origins: List<&2,Origin>,target: List<&2,U32>,target_grid: Grid.Grid,fill: Element,context: Context}

# Whether the piece that starts at cell lies in the right half of a split.
def side_of(layout: Pieces.Ranges,+cell: U32) -> Bool:
  match layout:
    case Pieces.Range{first,count}: False{}
    case Pieces.Split{left,right}: U32.is_lt(cell,Pieces.first(right))

def exchanged_left(~Element: Data,right: Pieces.Pieces(~Element),pair: Pieces.Pieces(~Element) & Storage.Buffer<Element>) -> Pieces.Pieces(~Element) & Storage.Buffer<Element>:
  (left,found) = pair
  (Partition.Fork{left,right},found)

def exchanged_right(~Element: Data,left: Pieces.Pieces(~Element),pair: Pieces.Pieces(~Element) & Storage.Buffer<Element>) -> Pieces.Pieces(~Element) & Storage.Buffer<Element>:
  (right,found) = pair
  (Partition.Fork{left,right},found)

# The piece that starts at cell in a tree whose ranges are layout, and the
# tree with replacement in its place.
def exchanged(~Element: Data,tree: Pieces.Pieces(~Element),layout: Pieces.Ranges,left_side: Bool,+cell: U32,replacement: Storage.Buffer<Element>) -> Pieces.Pieces(~Element) & Storage.Buffer<Element>:
  match tree layout left_side:
    case Partition.Leaf{data,first,count,details} layout left_side: (Partition.Leaf{replacement,first,count,Unit{}},data)
    case Partition.Fork{left,right} Pieces.Split{+left_layout,right_layout} True{}:
      exchanged_left(~Element,right,exchanged(~Element,left,left_layout,side_of(left_layout,cell),cell,replacement))
    case Partition.Fork{left,right} Pieces.Split{left_layout,+right_layout} False{}:
      exchanged_right(~Element,left,exchanged(~Element,right,right_layout,side_of(right_layout,cell),cell,replacement))
    case Partition.Fork{left,right} layout left_side: (Partition.Fork{left,right},replacement)

def exchange(~Element: Data,tree: Pieces.Pieces(~Element),+layout: Pieces.Ranges,+cell: U32,replacement: Storage.Buffer<Element>) -> Pieces.Pieces(~Element) & Storage.Buffer<Element>:
  exchanged(~Element,tree,layout,side_of(layout,cell),cell,replacement)

def taken_joined(~Element: Data,data: Storage.Buffer<Element>,+first: U32,+count: U32,pair: Pieces.Pieces(~Element) & Pieces.Pieces(~Element)) ->
  Pieces.Pieces(~Element) & Pieces.Pieces(~Element):
  (tree,inputs) = pair
  (tree,Partition.Fork{Partition.Leaf{data,first,count,Unit{}},inputs})

# The piece exchanged out of the tree, of first cell first and count cells,
# and the pieces listed after it taken out in order.
def taken_on(~Element: Data,pieces: List<&2,U32>,exchanged: Pieces.Pieces(~Element) & Storage.Buffer<Element>,+grid: Grid.Grid,+layout: Pieces.Ranges,+fill: Element,
  +first: U32,+count: U32) -> Pieces.Pieces(~Element) & Pieces.Pieces(~Element):
  match pieces:
    case Nil{}:
      (tree,data) = exchanged
      (tree,Partition.Leaf{data,first,count,Unit{}})
    case Con{+piece,rest}:
      (tree,data) = exchanged
      +next = Grid.start(grid,piece)
      taken_joined(~Element,data,first,count,taken_on(~Element,rest,exchange(~Element,tree,layout,next,buffer(~Element,fill,0)),grid,layout,fill,next,Grid.span(grid,piece)))

# The listed source pieces taken out of the source tree in order, as the
# pieces of one task.
def taken_inputs(~Element: Data,pieces: List<&2,U32>,+grid: Grid.Grid,+layout: Pieces.Ranges,+fill: Element,tree: Pieces.Pieces(~Element)) -> Pieces.Pieces(~Element) & Pieces.Pieces(~Element):
  match pieces:
    case Nil{}: (tree,Partition.Leaf{buffer(~Element,fill,0),0,0,Unit{}})
    case Con{+piece,rest}:
      +first = Grid.start(grid,piece)
      taken_on(~Element,rest,exchange(~Element,tree,layout,first,buffer(~Element,fill,0)),grid,layout,fill,first,Grid.span(grid,piece))

def sources_joined(~Element: Data,own: Pieces.Pieces(~Element) & Pieces.Pieces(~Element),later: Sources<Element> & Sources<Element>) -> Sources<Element> & Sources<Element>:
  (pieces,inputs) = own
  (rest,taken) = later
  (Source{pieces,rest},Source{inputs,taken})

# The source pieces of component index of every source, taken out in order.
def taken_sources(~Element: Data,origins: List<&2,Origin>,+components: Tasks.Components,+source: U32,+index: U32,+fill: Element,sources: Sources<Element>) -> Sources<Element> & Sources<Element>:
  match origins sources:
    case Con{Origin{+shape,+grid,+ranges,+offset},origins} Source{pieces,rest}:
      sources_joined(~Element,taken_inputs(~Element,Tasks.box_pieces(grid,shape,Tasks.source_box(components,source,grid,shape,offset,index)),grid,ranges,fill,pieces),
        taken_sources(~Element,origins,components,(source + 1 : U32),index,fill,rest))
    case origins sources: (sources,Ended{})

def inputs_leaf(~Element: Data,+first: U32,+count: U32,pair: Sources<Element> & Sources<Element>) -> Sources<Element> & Partition.Tree<&1,Sources<Element>,Unit>:
  (sources,inputs) = pair
  (sources,Partition.Leaf{inputs,first,count,Unit{}})

# Prepare: the source pieces of component first, routed in source order.
def component_inputs(~Element: Data,~Context: Data,context: Moving<Element,Context>,first: U32,count: U32,sources: Sources<Element>) ->
  Sources<Element> & Partition.Tree<&1,Sources<Element>,Unit>:
  Moving{+components,+origins,target,target_grid,+fill,gather} = context
  +first = first
  +count = count
  inputs_leaf(~Element,first,count,taken_sources(~Element,origins,components,0,first,fill,sources))

# The pieces one task holds of every source.
type Holdings<-Element: Data> is Type:
  Unheld{}
  Holds{holding: Move.Holding<Element>,rest: Holdings<Element>}

def holdings(~Element: Data,sources: Sources<Element>) -> Holdings<Element>:
  match sources:
    case Ended{}: Unheld{}
    case Source{pieces,rest}: Holds{Move.held(~Element,pieces),holdings(~Element,rest)}

# A gather writes the cells of the target piece of first cell first and count
# cells from the pieces its task holds.
def Gather(~Element: Data,~Context: Data) -> Type:
  @+context: Context -> @+components: Tasks.Components -> @+first: U32 -> @+count: U32 -> @+fill: Element ->
    Holdings<Element> -> Holdings<Element> & Storage.Buffer<Element>

def gathered_walked(~Element: Data,lefts: Pieces.Lefts<Storage.Buffer<Element>>,+first: U32,+count: U32,result: Holdings<Element> & Storage.Buffer<Element>) ->
  Pieces.Walked<Storage.Buffer<Element>,Holdings<Element>>:
  (input,output) = result
  Pieces.Walked{Partition.Leaf{output,first,count,Unit{}},lefts,input}

# The target pieces of a task, each gathered in target order. Each piece
# waits on the left for the pieces after it, so the task calls no closure.
def gathered_pieces(~Element: Data,~Context: Data,~gather: Gather(~Element,~Context),pieces: List<&2,U32>,+context: Context,+components: Tasks.Components,
  +target_grid: Grid.Grid,+fill: Element,ready: Pieces.Walking<Storage.Buffer<Element>,Holdings<Element>>) -> Pieces.Walked<Storage.Buffer<Element>,Holdings<Element>>:
  match pieces:
    case Nil{}:
      Pieces.Walking{lefts,input} = ready
      Pieces.Walked{Partition.Leaf{buffer(~Element,fill,0),0,0,Unit{}},lefts,input}
    case Con{+piece,Nil{}}:
      Pieces.Walking{lefts,input} = ready
      +first = Grid.start(target_grid,piece)
      +count = Grid.span(target_grid,piece)
      gathered_walked(~Element,lefts,first,count,gather(context,components,first,count,fill,input))
    case Con{+piece,rest}:
      Pieces.Walking{lefts,input} = ready
      +first = Grid.start(target_grid,piece)
      +count = Grid.span(target_grid,piece)
      Pieces.joined(~Storage.Buffer<Element>,~Holdings<Element>,gathered_pieces(~Element,~Context,~gather,rest,context,components,target_grid,fill,
        Pieces.pushed(~Storage.Buffer<Element>,~Holdings<Element>,gathered_walked(~Element,lefts,first,count,gather(context,components,first,count,fill,input)))))

def walked_pieces(~Element: Data,walked: Pieces.Walked<Storage.Buffer<Element>,Holdings<Element>>) -> Pieces.Pieces(~Element):
  Pieces.Walked{tree,lefts,input} = walked
  tree

# One task: the pieces it holds of every source, then its target pieces.
def component_run(~Element: Data,~Context: Data,~gather: Gather(~Element,~Context),context: Moving<Element,Context>,first: U32,count: U32,inputs: Sources<Element>,resource: Unit) ->
  Partition.Tree<&1,Pieces.Pieces(~Element),Unit>:
  Moving{+components,origins,+target,+target_grid,+fill,+own} = context
  +first = first
  +count = count
  Partition.Leaf{walked_pieces(~Element,gathered_pieces(~Element,~Context,~gather,Tasks.target_pieces(components,target_grid,target,first),own,components,target_grid,fill,
    Pieces.Walking{Pieces.NoLefts{},holdings(~Element,inputs)})),first,count,Unit{}}

def slot_piece(~Element: Data,fill: Element,first: U32,count: U32,state: Unit) -> Unit & Pieces.Pieces(~Element):
  +fill = fill
  (Unit{},Partition.Leaf{buffer(~Element,fill,0),first,count,Unit{}})

# Where the pieces of a finish go: the ranges of the target pieces, and the
# fill of the storage of a target piece written in parts.
type Depositing<-Element: Data> is Data:
  Depositing{layout: Pieces.Ranges,fill: Element}

# A buffer of at least span cells: storage, as the slots of a finish start
# empty.
def sized_when(~Element: Data,short: Bool,data: Storage.Buffer<Element>,+span: U32,+fill: Element) -> Storage.Buffer<Element>:
  match short:
    case True{}: buffer(~Element,fill,span)
    case False{}: data

def sized(~Element: Data,data: Storage.Buffer<Element>,+span: U32,+fill: Element) -> Storage.Buffer<Element>:
  Storage.Buffer{storage,+length,certificate} = data
  sized_when(~Element,Nat.is_lt(length,U32.to_nat(span)),Storage.Buffer{storage,length,certificate},span,fill)

# The slot of span cells from slot after a part of count cells from first:
# the part itself when it is the whole slot, else the slot with the part
# copied in.
def slot_written(~Element: Data,whole: Bool,data: Storage.Buffer<Element>,+slot: U32,+span: U32,+first: U32,+count: U32,+fill: Element,part: Storage.Buffer<Element>) ->
  Storage.Buffer<Element>:
  match whole:
    case True{}: part
    case False{}: Pair.snd(Storage.Buffer<Element>,Storage.Buffer<Element>,Copy.range(~Element,count,0,(first - slot : U32),part,sized(~Element,data,span,fill)))

# The tree whose ranges are layout with the part of count cells from first
# written in the piece that holds its first cell.
def deposited(~Element: Data,tree: Pieces.Pieces(~Element),layout: Pieces.Ranges,left_side: Bool,+first: U32,+count: U32,+fill: Element,part: Storage.Buffer<Element>) ->
  Pieces.Pieces(~Element):
  match tree layout left_side:
    case Partition.Leaf{data,+slot,+span,details} layout left_side:
      Partition.Leaf{slot_written(~Element,U32.is_eq(slot,first) && U32.is_eq(span,count),data,slot,span,first,count,fill,part),slot,span,Unit{}}
    case Partition.Fork{left,right} Pieces.Split{+left_layout,right_layout} True{}:
      Partition.Fork{deposited(~Element,left,left_layout,side_of(left_layout,first),first,count,fill,part),right}
    case Partition.Fork{left,right} Pieces.Split{left_layout,+right_layout} False{}:
      Partition.Fork{left,deposited(~Element,right,right_layout,side_of(right_layout,first),first,count,fill,part)}
    case Partition.Fork{left,right} layout left_side: Partition.Fork{left,right}

def placed_when(~Element: Data,empty: Bool,+depositing: Depositing<Element>,+first: U32,+count: U32,input: Storage.Buffer<Element>,slots: Pieces.Pieces(~Element)) -> Pieces.Pieces(~Element):
  match empty:
    case True{}: slots
    case False{}:
      Depositing{+layout,+fill} = depositing
      deposited(~Element,slots,layout,side_of(layout,first),first,count,fill,input)

def placed_piece(~Element: Data,depositing: Depositing<Element>,first: U32,count: U32,input: Storage.Buffer<Element>,slots: Pieces.Pieces(~Element)) -> Pieces.Pieces(~Element):
  +depositing = depositing
  +first = first
  +count = count
  placed_when(~Element,U32.is_zero(count),depositing,first,count,input,slots)

def placed_component(~Element: Data,depositing: Depositing<Element>,first: U32,count: U32,pieces: Pieces.Pieces(~Element),slots: Pieces.Pieces(~Element)) -> Pieces.Pieces(~Element):
  Pieces.fold(~Storage.Buffer<Element>,~Pieces.Pieces(~Element),~Depositing<Element>,~placed_piece(~Element),depositing,pieces,slots)

# Finish: every target piece put in its place of the target ranges; the parts
# of a piece computed by several tasks are copied into it.
def placed(~Element: Data,+layout: Pieces.Ranges,+fill: Element,outputs: Partition.Tree<&1,Pieces.Pieces(~Element),Unit>) -> Pieces.Pieces(~Element):
  Pieces.fold(~Pieces.Pieces(~Element),~Pieces.Pieces(~Element),~Depositing<Element>,~placed_component(~Element),Depositing{layout,fill},outputs,
    Pair.snd(Unit,Pieces.Pieces(~Element),Pieces.build(~Storage.Buffer<Element>,~Unit,~Element,~slot_piece(~Element),layout,fill,Unit{})))

def moved_decided(~Element: Data,~Context: Data,~gather: Gather(~Element,~Context),+context: Moving<Element,Context>,decision: Tasks.Decision,+count: U32,+layout: Pieces.Ranges,+fill: Element,
  sources: Sources<Element>) -> Pieces.Pieces(~Element):
  Tasks.Decision{+parallel,+tasks} = decision
  placed(~Element,layout,fill,Pieces.grouped(~Sources<Element>,~Pieces.Pieces(~Element),~Moving<Element,Context>,~component_run(~Element,~Context,~gather),tasks,parallel,context,
    Pair.snd(Sources<Element>,Partition.Tree<&1,Sources<Element>,Unit>,
      Pieces.build(~Sources<Element>,~Sources<Element>,~Moving<Element,Context>,~component_inputs(~Element,~Context),Tasks.component_ranges(count),context,sources))))

# A view or a placing adds no rows to the move of a target cell.
def move_rows() -> F32:
  0.0

# The pieces of target cells, each component a task that reads only its own
# pieces of every source and gathers its target pieces. rows and steps count
# what one target cell adds to its move.
def moved_with(~Element: Data,~Context: Data,~gather: Gather(~Element,~Context),+components: Tasks.Components,+context: Context,+rows: F32,+steps: F32,
  sources: Sources<Element>,+origins: List<&2,Origin>,+target: List<&2,U32>,+fill: Element,+workers: U32) -> Pieces.Pieces(~Element):
  +target_grid = grid_of(target)
  +count = Tasks.components_count(components)
  moved_decided(~Element,~Context,~gather,Moving{components,origins,target,target_grid,fill,context},
    Tasks.decide_components(count_or_zero(target),components,workers,rows,steps),count,Pieces.ranges_of_grid(target_grid),fill,sources)

# A view of the source cells taken apart, read by every target piece.
type Viewed is Data:
  Viewed{source: List<&2,U32>,target_grid: Grid.Grid,target: List<&2,U32>,axes: List<&2,Views.Axis>,offset: U32,reversed: List<&2,Bool>}

# Every axis of a view restricted to a target piece, with its direction.
def view_axes(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+strides: List<&2,U32>) -> List<&2,Move.Axis>:
  match axes strides:
    case Con{Views.Axis{+extent,+stride},axes} Con{+target,strides}:
      +back = Views.flag(flags)
      Move.Axis{extent,Move.directed(back,stride),back,target} <> view_axes(axes,Views.later_flags(flags),strides)
    case axes strides: []

def view_part(+reversed: List<&2,Bool>,+extents: List<&2,U32>,view: Views.View) -> Move.Part:
  Views.View{+axes,+offset} = view
  Move.Part{view_axes(axes,reversed,Tasks.strides_of(extents)),offset,0}

def view_written(~Element: Data,rest: Holdings<Element>,pair: Move.Holding<Element> & Storage.Buffer<Element>) -> Holdings<Element> & Storage.Buffer<Element>:
  (holding,output) = pair
  (Holds{holding,rest},output)

def view_held(~Element: Data,+reversed: List<&2,Bool>,+extents: List<&2,U32>,+view: Views.View,+count: U32,+fill: Element,input: Holdings<Element>) ->
  Holdings<Element> & Storage.Buffer<Element>:
  match input:
    case Holds{holding,rest}: view_written(~Element,rest,Move.part(~Element,view_part(reversed,extents,view),holding,buffer(~Element,fill,count)))
    case Unheld{}: (Unheld{},buffer(~Element,fill,count))

# The target piece of first cell first and count cells read through the view.
def view_gather(~Element: Data,+context: Viewed,+components: Tasks.Components,+first: U32,+count: U32,+fill: Element,input: Holdings<Element>) ->
  Holdings<Element> & Storage.Buffer<Element>:
  Viewed{source,+target_grid,+target,+axes,+offset,+reversed} = context
  view_held(~Element,reversed,Grid.box_of(target_grid,target,count),
    Grid.piece_view(target_grid,target,first,count,Views.mirrored(Views.View{axes,offset},reversed)),count,fill,input)

# A gather, then a map of every target piece it fills.
type Then<-Context: Data,-Later: Data> is Data:
  Then{context: Context,later: Later}

def then_output(~Element: Data,~Later: Data,~after: PieceMap(~Element,~Later),+later: Later,+first: U32,+count: U32,result: Holdings<Element> & Storage.Buffer<Element>) ->
  Holdings<Element> & Storage.Buffer<Element>:
  (input,output) = result
  (input,after(later,first,count,output))

def then_gather(~Element: Data,~Context: Data,~Later: Data,~gather: Gather(~Element,~Context),~after: PieceMap(~Element,~Later),+context: Then<Context,Later>,
  +components: Tasks.Components,+first: U32,+count: U32,+fill: Element,input: Holdings<Element>) -> Holdings<Element> & Storage.Buffer<Element>:
  Then{+own,+later} = context
  then_output(~Element,~Later,~after,later,first,count,gather(own,components,first,count,fill,input))

# The piece as it is.
def unchanged(~Element: Data,+context: Unit,+first: U32,+count: U32,buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  buffer

def moved_counted(~Element: Data,~Later: Data,~after: PieceMap(~Element,~Later),+later: Later,empty: Bool,pieces: Pieces.Pieces(~Element),+source: List<&2,U32>,+target: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match empty:
    case True{}: Tensor{filled(~Element,target,fill),target,fill,Usable{},workers}
    case False{}:
      Tensor{moved_with(~Element,~Then<Viewed,Later>,~then_gather(~Element,~Viewed,~Later,~view_gather(~Element),~after),
        Tasks.components(source,grid_of(source),grid_of(target),Views.View{axes,offset},reversed),Then{Viewed{source,grid_of(target),target,axes,offset,reversed},later},move_rows(),0.0,
        Source{pieces,Ended{}},[origin(source,offset)],target,fill,workers),target,fill,Usable{},workers}

# A target filled from sources placed in it: the cell extents of every
# source, and per source and axis the segments it fills.
type Placing is Data:
  Placing{target_grid: Grid.Grid,target: List<&2,U32>,sources: List<&2,List<&2,U32>>,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>}

# The holdings of the sources already placed, each waiting for the holdings
# after it, and the target piece.
type Waiting<-Element: Data> is Type:
  NoneWaiting{}
  Waits{holding: Move.Holding<Element>,rest: Waiting<Element>}

type Placed<-Element: Data> is Type:
  Placed{holdings: Holdings<Element>,waiting: Waiting<Element>,output: Storage.Buffer<Element>}

def placed_after(~Element: Data,waiting: Waiting<Element>,pair: Move.Holding<Element> & Storage.Buffer<Element>) -> Placed<Element>:
  (holding,output) = pair
  Placed{Unheld{},Waits{holding,waiting},output}

# The holdings of the later sources join the holding waiting for them.
def placed_joined(~Element: Data,placed: Placed<Element>) -> Placed<Element>:
  match placed:
    case Placed{holdings,Waits{holding,rest},output}: Placed{Holds{holding,holdings},rest,output}
    case Placed{holdings,NoneWaiting{},output}: Placed{holdings,NoneWaiting{},output}

# Every source writes the parts it places; the task calls no closure.
def placed_sources(~Element: Data,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,+grid: Grid.Grid,+cells: List<&2,U32>,
  +first: U32,+count: U32,input: Holdings<Element>,placed: Placed<Element>) -> Placed<Element>:
  match segments sources input:
    case Con{+own,segments} Con{+source,sources} Holds{holding,rest}:
      Placed{done,waiting,output} = placed
      placed_joined(~Element,placed_sources(~Element,segments,sources,grid,cells,first,count,rest,
        placed_after(~Element,waiting,Move.parts(~Element,Tasks.placements(own,source,grid,cells,first,count),holding,output))))
    case segments sources input:
      Placed{done,waiting,output} = placed
      Placed{input,waiting,output}

def placed_result(~Element: Data,placed: Placed<Element>) -> Holdings<Element> & Storage.Buffer<Element>:
  Placed{holdings,waiting,output} = placed
  (holdings,output)

# A target piece starts as fill and takes from every source the parts of the
# piece that source places.
def placing_gather(~Element: Data,+context: Placing,+components: Tasks.Components,+first: U32,+count: U32,+fill: Element,input: Holdings<Element>) ->
  Holdings<Element> & Storage.Buffer<Element>:
  Placing{+target_grid,+target,+sources,+segments} = context
  placed_result(~Element,placed_sources(~Element,segments,sources,target_grid,target,first,count,input,Placed{Unheld{},NoneWaiting{},buffer(~Element,fill,count)}))

def origin_shapes(origins: List<&2,Origin>) -> List<&2,List<&2,U32>>:
  match origins:
    case Nil{}: []
    case Con{Origin{shape,grid,ranges,offset},rest}: shape <> origin_shapes(rest)

# The pieces of a target of shape target filled with fill where no source
# reaches, every source read as readings and placed as segments.
def placed_then(~Element: Data,~Later: Data,~after: PieceMap(~Element,~Later),+later: Later,+readings: List<&2,List<&2,Tasks.Reading>>,
  +segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: Sources<Element>,+origins: List<&2,Origin>,+target: List<&2,U32>,+fill: Element,+workers: U32) -> Pieces.Pieces(~Element):
  moved_with(~Element,~Then<Placing,Later>,~then_gather(~Element,~Placing,~Later,~placing_gather(~Element),~after),Tasks.components_of(readings),
    Then{Placing{grid_of(target),target,origin_shapes(origins),segments},later},move_rows(),0.0,sources,origins,target,fill,workers)

def placed_from(~Element: Data,+readings: List<&2,List<&2,Tasks.Reading>>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: Sources<Element>,+origins: List<&2,Origin>,
  +target: List<&2,U32>,+fill: Element,+workers: U32) -> Pieces.Pieces(~Element):
  placed_then(~Element,~Unit,~unchanged(~Element),Unit{},readings,segments,sources,origins,target,fill,workers)

def moved_status(~Element: Data,~Later: Data,~after: PieceMap(~Element,~Later),+later: Later,name: String,status: Status,valid: Bool,count: Maybe<&2,U32>,pieces: Pieces.Pieces(~Element),
  +source: List<&2,U32>,+target: List<&2,U32>,
  +axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match status valid count:
    case Usable{} True{} Some{+count}: moved_counted(~Element,~Later,~after,later,U32.is_zero(count),pieces,source,target,axes,offset,reversed,fill,workers)
    case Usable{} valid count: failed(~Element,name,InvalidArgument{source},target,fill,workers)
    case Failed{problem} valid count: Tensor{Partition.Leaf{buffer(~Element,fill,0),0,0,Unit{}},target,fill,Failed{problem},workers}

# The cells a view of a tensor's cells selects, in the view's row-major order
# with the flagged axes in reverse, as a tensor of the view's extents. Every
# component of the view is a task that reads only its own source pieces.
def moved_then(~Element: Data,~Later: Data,~after: PieceMap(~Element,~Later),later: Later,name: String,tensor: Tensor<Element>,+axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>) -> Tensor<Element>:
  Tensor{pieces,+shape,+fill,status,+workers} = tensor
  +target = Views.axis_extents(axes)
  moved_status(~Element,~Later,~after,later,name,status,Views.fits(Views.View{axes,offset},count_or_zero(shape)),count_of(target),pieces,shape,target,axes,offset,reversed,fill,workers)

def moved(~Element: Data,name: String,tensor: Tensor<Element>,+axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>) -> Tensor<Element>:
  moved_then(~Element,~Unit,~unchanged(~Element),Unit{},name,tensor,axes,offset,reversed)

def broadcast_status(~Element: Data,status: Status,view: Maybe<&2,Views.View>,pieces: Pieces.Pieces(~Element),+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) -> Tensor<Element>:
  match status view:
    case Usable{} None{}: failed(~Element,"broadcast_to",ShapeMismatch{shape,target},target,fill,workers)
    case Failed{problem} None{}: Tensor{Partition.Leaf{buffer(~Element,fill,0),0,0,Unit{}},target,fill,Failed{problem},workers}
    case status Some{Views.View{+axes,+offset}}: moved(~Element,"broadcast_to",Tensor{pieces,shape,fill,status,workers},axes,offset,[])

# The tensor repeated to the extents target under NumPy broadcasting: one
# moving operation through the stretched view.
def broadcast_to(~Element: Data,tensor: Tensor<Element>,+target: List<&2,U32>) -> Tensor<Element>:
  Tensor{pieces,+shape,+fill,status,+workers} = tensor
  broadcast_status(~Element,status,stretched(view_of(shape),target),pieces,shape,target,fill,workers)

def combine_repeated_status(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,operations: F32,
  status: Status,same: Bool,+shape: List<&2,U32>,+workers: U32,repeated: Pieces.Pieces(~Element),+fill: Element,right: Pieces.Pieces(~Element),+right_shape: List<&2,U32>,right_fill: Element) -> Tensor<Element>:
  match status same:
    case Usable{} True{}: combine_equal(~Element,~Later,~kernel,~after,later,operations,Usable{},workers,shape,repeated,fill,right)
    case Usable{} False{}:
      combine_stretched(~Element,~Later,~kernel,~after,later,operations,workers,shape,repeated,fill,joined(~Element,count_or_zero(right_shape),right_fill,right),right_shape,
        spread_view(stretched(view_of(right_shape),shape)))
    case Failed{problem} same: Tensor{repeated,shape,fill,Failed{problem},workers}

def combine_repeated(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,operations: F32,
  same: Bool,+shape: List<&2,U32>,+workers: U32,left: Tensor<Element>,right: Pieces.Pieces(~Element),+right_shape: List<&2,U32>,right_fill: Element) -> Tensor<Element>:
  Tensor{repeated,repeated_shape,+fill,status,repeated_workers} = left
  combine_repeated_status(~Element,~Later,~kernel,~after,later,operations,status,same,shape,workers,repeated,fill,right,right_shape,right_fill)

# A left operand of the common extents keeps its pieces and every task reads
# the right through its stretched view. A left operand of other extents is
# first repeated to the common extents.
def combine_common(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,name: String,operations: F32,
  same: Bool,count: Maybe<&2,U32>,+shape: List<&2,U32>,+workers: U32,left: Pieces.Pieces(~Element),+left_shape: List<&2,U32>,+fill: Element,
  right: Pieces.Pieces(~Element),+right_shape: List<&2,U32>,right_fill: Element) -> Tensor<Element>:
  match same count:
    case True{} count:
      combine_stretched(~Element,~Later,~kernel,~after,later,operations,workers,shape,left,fill,joined(~Element,count_or_zero(right_shape),right_fill,right),right_shape,
        spread_view(stretched(view_of(right_shape),shape)))
    case False{} Some{count}:
      combine_repeated(~Element,~Later,~kernel,~after,later,operations,Shapes.same_extents(right_shape,shape),shape,workers,
        broadcast_to(~Element,Tensor{left,left_shape,fill,Usable{},workers},shape),right,right_shape,right_fill)
    case False{} None{}: failed(~Element,name,InvalidShape{shape},shape,fill,workers)

# Operands of other extents follow NumPy broadcasting (Shapes.broadcast) and
# fail only when they are incompatible.
def combine_broadcast(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,name: String,operations: F32,
  status: Status,common: Maybe<&2,List<&2,U32>>,+workers: U32,left: Pieces.Pieces(~Element),+left_shape: List<&2,U32>,+left_fill: Element,
  right: Pieces.Pieces(~Element),+right_shape: List<&2,U32>,right_fill: Element) -> Tensor<Element>:
  match status common:
    case Usable{} Some{+shape}:
      combine_common(~Element,~Later,~kernel,~after,later,name,operations,Shapes.same_extents(left_shape,shape),count_of(shape),shape,workers,left,left_shape,left_fill,
        right,right_shape,right_fill)
    case Usable{} None{}: Tensor{left,left_shape,left_fill,Failed{Problem{name,ShapeMismatch{left_shape,right_shape}}},workers}
    case Failed{problem} common: Tensor{left,left_shape,left_fill,Failed{problem},workers}

def combine_shapes(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,name: String,operations: F32,
  same: Bool,left: Pieces.Pieces(~Element),+left_shape: List<&2,U32>,+left_fill: Element,left_status: Status,left_workers: U32,
  right: Pieces.Pieces(~Element),+right_shape: List<&2,U32>,right_fill: Element,right_status: Status,right_workers: U32) -> Tensor<Element>:
  match same:
    case True{}:
      combine_equal(~Element,~Later,~kernel,~after,later,operations,first_failure(left_status,right_status),U32.max(left_workers,right_workers),left_shape,left,left_fill,right)
    case False{}:
      combine_broadcast(~Element,~Later,~kernel,~after,later,name,operations,first_failure(left_status,right_status),Shapes.broadcast(left_shape,right_shape),
        U32.max(left_workers,right_workers),left,left_shape,left_fill,right,right_shape,right_fill)

# operation(left element, right element), then after on every result piece in
# the same task. Operands of equal extents zip piece by piece; operands of
# other extents follow NumPy broadcasting (Shapes.broadcast) and fail only
# when they are incompatible. operations counts the vector rows of the
# operation on one cell for the parallel decision.
def combine_with(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,name: String,operations: F32,
  left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  match left right:
    case Tensor{left,+left_shape,left_fill,left_status,left_workers} Tensor{right,+right_shape,right_fill,right_status,right_workers}:
      combine_shapes(~Element,~Later,~kernel,~after,later,name,operations,Shapes.same_extents(left_shape,right_shape),
        left,left_shape,left_fill,left_status,left_workers,right,right_shape,right_fill,right_status,right_workers)

# operation(left element, right element) with the right read through a view
# of it with the extents of the left, then after on every result piece in the
# same task.
def combine_viewed_status(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,operations: F32,status: Status,
  +workers: U32,left: Pieces.Pieces(~Element),+left_shape: List<&2,U32>,+left_fill: Element,right: Pieces.Pieces(~Element),+right_shape: List<&2,U32>,right_fill: Element,
  +axes: List<&2,Views.Axis>,+offset: U32) -> Tensor<Element>:
  match status:
    case Usable{}:
      combine_spread(~Element,~Later,~kernel,~after,later,operations,workers,left_shape,left,left_fill,joined(~Element,count_or_zero(right_shape),right_fill,right),right_shape,
        axes,offset)
    case Failed{problem}: Tensor{left,left_shape,left_fill,Failed{problem},workers}

def combine_viewed_with(~Element: Data,~Later: Data,~kernel: Kernel(~Element),~after: PieceMap(~Element,~Later),later: Later,operations: F32,
  left: Tensor<Element>,right: Tensor<Element>,+axes: List<&2,Views.Axis>,+offset: U32) -> Tensor<Element>:
  match left right:
    case Tensor{left,+left_shape,+left_fill,left_status,left_workers} Tensor{right,+right_shape,right_fill,right_status,right_workers}:
      combine_viewed_status(~Element,~Later,~kernel,~after,later,operations,first_failure(left_status,right_status),U32.max(left_workers,right_workers),
        left,left_shape,left_fill,right,right_shape,right_fill,axes,offset)

# The elementwise form: the kernel applies operation to each pair in order.
def combine_viewed(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after: PieceMap(~Element,~Later),later: Later,operations: F32,
  left: Tensor<Element>,right: Tensor<Element>,+axes: List<&2,Views.Axis>,+offset: U32) -> Tensor<Element>:
  combine_viewed_with(~Element,~Later,~Operations.combine_kernel(~Element,~operation),~after,later,operations,left,right,axes,offset)

def combine(~Element: Data,~Later: Data,~operation: Element -> Element -> Element,~after: PieceMap(~Element,~Later),later: Later,name: String,operations: F32,
  left: Tensor<Element>,right: Tensor<Element>) -> Tensor<Element>:
  combine_with(~Element,~Later,~Operations.combine_kernel(~Element,~operation),~after,later,name,operations,left,right)

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

def expected_storage(~Element: Data,result: Result<&2,&1,Problem,Storage.Buffer<Element>>) -> IO(Storage.Buffer<Element>):
  match result:
    case Done{storage}: IO.pure(Storage.Buffer<Element>,storage)
    case Fail{problem}: IO.die(Storage.Buffer<Element>,1,describe(problem))

def expect_storage(~Element: Data,tensor: Tensor<Element>) -> IO(Storage.Buffer<Element>):
  expected_storage(~Element,into_storage(~Element,tensor))
