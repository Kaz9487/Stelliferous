# Internal H-axis ownership. Offsets and spans are logical image rows, not
# element positions: offset / height identifies the image. No leaf crosses an
# image boundary. Its buffer stores [channels][span][width]. The common
# partition tree owns all buffers; shape and element type remain parameters.
import Base
import ./storage_buffer.bend as Storage
import ./storage_copy.bend as Copy
import ./storage_clone.bend as Clone
import ./traversal_partition.bend as Partition
import ./traversal_array.bend as Traversal
import ./proofs/traversal_certificate.bend as TraversalCertificate
import ./tensor_operations.bend as Operations
import ./storage_allocation.bend as Allocation
import ./tensor_view.bend as Views
import ./tensor_shape.bend as Shapes
import ./tensor_indexed.bend as IndexedAccess
import ./proofs/tensor_indexed_certificate.bend as IndexedCertificate
import ./proofs/storage_certificate.bend as Certificate

type Geometry is Data:
  Geometry{batch: U32,channels: U32,height: U32,width: U32}

def Bands(~Element: Data) -> Type:
  Partition.Tree<&1,Storage.Buffer<Element>,Unit>

# An existing tree with a state threaded through its leaves in order. The leaf
# takes the state and gives it back beside what it leaves in the tree.
def walk(~Input: Type,~Output: Type,~State: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Input -> State -> Partition.Tree<&1,Output,Unit> & State,
  +context: Context,tree: Partition.Tree<&1,Input,Unit>,state: State) -> Partition.Tree<&1,Output,Unit> & State:
  match tree:
    case Partition.Leaf{input,first,rows,details}: leaf(context,first,rows,input,state)
    case Partition.Fork{left,right}:
      Storage.bind_pair(~Partition.Tree<&1,Output,Unit>,~State,~(Partition.Tree<&1,Output,Unit> & State),
        walk(~Input,~Output,~State,~Context,~leaf,context,left,state),left => state =>
          Storage.bind_pair(~Partition.Tree<&1,Output,Unit>,~State,~(Partition.Tree<&1,Output,Unit> & State),
            walk(~Input,~Output,~State,~Context,~leaf,context,right,state),right => state => (Partition.Fork{left,right},state)))

# Every leaf of a tree consumed into a state, in order.
def fold(~Input: Type,~State: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Input -> State -> State,
  +context: Context,tree: Partition.Tree<&1,Input,Unit>,state: State) -> State:
  match tree:
    case Partition.Leaf{input,first,rows,details}: leaf(context,first,rows,input,state)
    case Partition.Fork{left,right}: fold(~Input,~State,~Context,~leaf,context,right,fold(~Input,~State,~Context,~leaf,context,left,state))

def reduced(~Output: Type,~Value: Type,~join: Value -> Value -> Value,left: Partition.Tree<&1,Output,Unit> & Value,right: Partition.Tree<&1,Output,Unit> & Value) -> Partition.Tree<&1,Output,Unit> & Value:
  (left,left_value) = left
  (right,right_value) = right
  (Partition.Fork{left,right},join(left_value,right_value))

# A value from every leaf, joined where the tree forks. The leaf gives its
# value beside what it leaves in the tree.
def reduce(~Input: Type,~Output: Type,~Value: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Input -> Partition.Tree<&1,Output,Unit> & Value,~join: Value -> Value -> Value,
  +context: Context,tree: Partition.Tree<&1,Input,Unit>) -> Partition.Tree<&1,Output,Unit> & Value:
  match tree:
    case Partition.Leaf{input,first,rows,details}: leaf(context,first,rows,input)
    case Partition.Fork{left,right}:
      reduced(~Output,~Value,~join,reduce(~Input,~Output,~Value,~Context,~leaf,~join,context,left),reduce(~Input,~Output,~Value,~Context,~leaf,~join,context,right))

def elements(geometry: Geometry) -> U32:
  Geometry{batch,channels,height,width} = geometry
  (batch * channels * height * width : U32)

def local_count(geometry: Geometry,rows: U32) -> U32:
  Geometry{batch,channels,height,width} = geometry
  (channels * rows * width : U32)

# Channel-major local indices map to the same global NCHW index for every
# partition. A leaf's first row is global across images; rows are image-local.
def global_index(geometry: Geometry,+first: U32,+rows: U32,+index: U32) -> U32:
  Geometry{batch,+channels,+height,+width} = geometry
  +plane = (rows * width : U32)
  (((first / height * channels + index / plane) * height + first % height) * width + index % plane : U32)

def planes(~Element: Data,remaining: Nat,+channel: U32,+count: U32,+source_stride: U32,+destination_stride: U32,
  +source_start: U32,+destination_start: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  match remaining:
    case 0n: owners
    case 1n+rest:
      (source,destination) = owners
      planes(~Element,rest,(channel + 1 : U32),count,source_stride,destination_stride,source_start,destination_start,
        Copy.range(~Element,count,(source_start + channel * source_stride : U32),
          (destination_start + channel * destination_stride : U32),source,destination))

def destination(~Element: Data,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  (source,destination) = owners
  destination

def gather_leaf(~Element: Data,geometry: Geometry,+first: U32,+rows: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Geometry{batch,+channels,+height,+width} = geometry
  destination(~Element,planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(rows * width : U32),(height * width : U32),0,
    ((first / height * channels * height + first % height) * width : U32),(input,output)))

# One allocation is provided by the caller. Each value is copied once, using
# one contiguous range per channel per band. No intermediate whole images.
def gather_band(~Element: Data,geometry: Geometry,first: U32,rows: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  +first = first
  +rows = rows
  gather_leaf(~Element,geometry,first,rows,input,output)

def gather(~Element: Data,+geometry: Geometry,bands: Bands(~Element),output: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  fold(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Geometry,~gather_band(~Element),geometry,bands,output)

# A leaf that holds every row of the only image is that image in row-major
# order.
def whole_image(+geometry: Geometry,first: U32,rows: U32) -> Bool:
  Geometry{batch,channels,+height,width} = geometry
  U32.is_eq(batch,1) && U32.is_zero(first) && U32.is_eq(rows,height)

def copied_leaf(~Element: Data,+first: U32,+rows: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Bands(~Element) & Bands(~Element):
  (left,right) = owners
  (Partition.Leaf{left,first,rows,Unit{}},Partition.Leaf{right,first,rows,Unit{}})

def copy_band(~Element: Data,context: Unit,first: U32,rows: U32,input: Storage.Buffer<Element>) -> Bands(~Element) & Bands(~Element):
  +first = first
  +rows = rows
  copied_leaf(~Element,first,rows,Clone.clone(~Element,input))

def fork(~Element: Data,left: Bands(~Element),right: Bands(~Element)) -> Bands(~Element):
  Partition.Fork{left,right}

def copy(~Element: Data,bands: Bands(~Element)) -> Bands(~Element) & Bands(~Element):
  reduce(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Bands(~Element),~Unit,~copy_band(~Element),~fork(~Element),Unit{},bands)

# A pure ownership plan contains no Buffer. It is a concrete Data type:
# duplicating Tree<&2,...> through a computed type makes all constructors
# shared in the official compiler, including the affine Buffer tree.
type Plan is Data:
  Rows{first: U32,count: U32}
  Split{left: Plan,right: Plan}

def plan_band(~Element: Data,context: Unit,first: U32,rows: U32,input: Storage.Buffer<Element>) -> Bands(~Element) & Plan:
  +first = first
  +rows = rows
  (Partition.Leaf{input,first,rows,Unit{}},Rows{first,rows})

def split_plan(left: Plan,right: Plan) -> Plan:
  Split{left,right}

def plan(~Element: Data,bands: Bands(~Element)) -> Bands(~Element) & Plan:
  reduce(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Plan,~Unit,~plan_band(~Element),~split_plan,Unit{},bands)

# A value from every leaf of a plan, joined where the plan splits. The join
# also sees the two halves of the plan.
def plan_fold(~Value: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Value,~join: Context -> Plan -> Plan -> Value -> Value -> Value,
  layout: Plan,+context: Context) -> Value:
  match layout:
    case Rows{first,rows}: leaf(context,first,rows)
    case Split{+left,+right}:
      join(context,left,right,plan_fold(~Value,~Context,~leaf,~join,left,context),plan_fold(~Value,~Context,~leaf,~join,right,context))

# A join that needs neither the context nor the halves of the plan.
def plain(~Value: Type,~Context: Data,~join: Value -> Value -> Value,context: Context,left: Plan,right: Plan,first: Value,second: Value) -> Value:
  join(first,second)

def same_plan(left: Plan,right: Plan) -> Bool:
  match left right:
    case Rows{first,rows} Rows{other_first,other_rows}:
      U32.is_eq(first,other_first) && U32.is_eq(rows,other_rows)
    case Split{left,right} Split{other_left,other_right}:
      same_plan(left,other_left) && same_plan(right,other_right)
    case left right: False{}

def same_geometry(left: Geometry,right: Geometry) -> Bool:
  Geometry{batch,channels,height,width} = left
  Geometry{other_batch,other_channels,other_height,other_width} = right
  U32.is_eq(batch,other_batch) && U32.is_eq(channels,other_channels) && U32.is_eq(height,other_height) && U32.is_eq(width,other_width)

# The geometry splits the image rows of an image shape, [channels, height,
# width] or [batch, channels, height, width]. Operations that read neighbouring
# rows or whole channels need this layout.
def image_rows(shape: List<&2,U32>,geometry: Geometry) -> Bool:
  match shape geometry:
    case [c,h,w] Geometry{n,ci,hi,wi}: U32.is_eq(n,1) && U32.is_eq(c,ci) && U32.is_eq(h,hi) && U32.is_eq(w,wi)
    case [n,c,h,w] Geometry{ni,ci,hi,wi}: U32.is_eq(n,ni) && U32.is_eq(c,ci) && U32.is_eq(h,hi) && U32.is_eq(w,wi)
    case shape geometry: False{}

def first(layout: Plan) -> U32:
  match layout:
    case Rows{first,count}: first
    case Split{left,right}: first(left)

def last(layout: Plan) -> U32:
  match layout:
    case Rows{first,count}: (first + count : U32)
    case Split{left,right}: last(right)

# A tree made from a plan, with a state threaded through the leaves in order.
# The leaf takes the state and gives it back beside the band it made.
def build(~Output: Type,~State: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> State -> State & Partition.Tree<&1,Output,Unit>,
  layout: Plan,+context: Context,state: State) -> State & Partition.Tree<&1,Output,Unit>:
  match layout:
    case Rows{first,rows}: leaf(context,first,rows,state)
    case Split{left,right}:
      Storage.bind_pair(~State,~Partition.Tree<&1,Output,Unit>,~(State & Partition.Tree<&1,Output,Unit>),
        build(~Output,~State,~Context,~leaf,left,context,state),state => left =>
          Storage.bind_pair(~State,~Partition.Tree<&1,Output,Unit>,~(State & Partition.Tree<&1,Output,Unit>),
            build(~Output,~State,~Context,~leaf,right,context,state),state => right => (state,Partition.Fork{left,right})))

# What a band made from scratch needs: its geometry and the value it starts with.
type Filling<-Element: Data> is Data:
  Filling{geometry: Geometry,fill: Element}

# Two trees of one shape, joined leaf by leaf. A left leaf without a partner
# keeps its own row range, so the left tree can be returned unchanged.
type Paired<-Left: Type,-Right: Type> is Type:
  Both{left: Left,right: Right}
  Only{left: Left,first: U32,rows: U32}

def only(~Left: Type,~Right: Type,tree: Partition.Tree<&1,Left,Unit>) -> Partition.Tree<&1,Paired<Left,Right>,Unit>:
  match tree:
    case Partition.Leaf{left,+first,+rows,details}: Partition.Leaf{Only{left,first,rows},first,rows,Unit{}}
    case Partition.Fork{left,right}: Partition.Fork{only(~Left,~Right,left),only(~Left,~Right,right)}

def zip(~Left: Type,~Right: Type,left: Partition.Tree<&1,Left,Unit>,right: Partition.Tree<&1,Right,Unit>) -> Partition.Tree<&1,Paired<Left,Right>,Unit>:
  match left right:
    case Partition.Leaf{left,first,rows,details} Partition.Leaf{right,other_first,other_rows,other_details}:
      Partition.Leaf{Both{left,right},first,rows,Unit{}}
    case Partition.Fork{left,right} Partition.Fork{other_left,other_right}:
      Partition.Fork{zip(~Left,~Right,left,other_left),zip(~Left,~Right,right,other_right)}
    case left right: only(~Left,~Right,left)

def left_leaf(~Left: Type,~Right: Type,pair: Paired<Left,Right>,first: U32,rows: U32) -> Partition.Tree<&1,Left,Unit>:
  match pair:
    case Both{left,right}: Partition.Leaf{left,first,rows,Unit{}}
    case Only{left,own_first,own_rows}: Partition.Leaf{left,own_first,own_rows,Unit{}}

def left_of(~Left: Type,~Right: Type,tree: Partition.Tree<&1,Paired<Left,Right>,Unit>) -> Partition.Tree<&1,Left,Unit>:
  match tree:
    case Partition.Leaf{pair,first,rows,details}: left_leaf(~Left,~Right,pair,first,rows)
    case Partition.Fork{left,right}: Partition.Fork{left_of(~Left,~Right,left),left_of(~Left,~Right,right)}

def paired(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Partition.Tree<&1,Left,Unit>,
  context: Context,first: U32,rows: U32,pair: Paired<Left,Right>,resource: Resource) -> Partition.Tree<&1,Left,Unit>:
  match pair:
    case Both{left,right}: leaf(context,first,rows,left,right,resource)
    case Only{left,own_first,own_rows}: Partition.Leaf{left,own_first,own_rows,Unit{}}

def unshared(resource: Unit) -> Unit & Unit:
  (Unit{},Unit{})

def same(~Block: Type,tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Block,Unit>:
  tree

# What a split gives its halves: a state for each, or the whole state when it
# cannot be divided.
type Division<-State: Type> is Type:
  Divided{left: State,right: State}
  Whole{state: State}

def divided(~State: Type,~Result: Type,division: Division<State>,whole: State -> Result,halves: State -> State -> Result) -> Result:
  match division:
    case Divided{left,right}: halves(left,right)
    case Whole{state}: whole(state)

# Every split may run its halves in parallel.
def always(schedule: Unit,layout: Plan) -> Bool:
  True{}

# The one plan recursion. The plan gives every part its rows and decides the
# structure. divide gives each half of a split its own state; a state it keeps
# whole becomes one part through keep. A split whose schedule allows it runs
# its halves in parallel. The last split calls its two leaves directly: a leaf
# that neither forks nor calls a closure is then its own task, which the
# runtime keeps for the work turn; a leaf reached through the forking
# recursion would run in the grow turn on one worker per row.
def each_part(~State: Type,~Output: Type,~Details: Data,~Context: Data,~Schedule: Data,
  ~parallel: Schedule -> Plan -> Bool,
  ~divide: Context -> Plan -> Plan -> State -> Division<State>,
  ~leaf: Context -> U32 -> U32 -> State -> Partition.Tree<&1,Output,Details>,
  ~keep: Context -> Plan -> State -> Partition.Tree<&1,Output,Details>,
  layout: Plan,mode: Bool,+schedule: Schedule,+context: Context,state: State) -> Partition.Tree<&1,Output,Details>:
  match layout mode:
    case Rows{first,rows} mode: leaf(context,first,rows,state)
    case Split{+left_plan,+right_plan} False{}:
      divided(~State,~Partition.Tree<&1,Output,Details>,divide(context,left_plan,right_plan,state),whole => keep(context,Split{left_plan,right_plan},whole),
        left_state => right_state =>
          Partition.Fork{each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left_plan,parallel(schedule,left_plan),schedule,context,left_state),
            each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right_plan,parallel(schedule,right_plan),schedule,context,right_state)})
    case Split{Rows{left_first,left_rows},Rows{right_first,right_rows}} True{}:
      divided(~State,~Partition.Tree<&1,Output,Details>,divide(context,Rows{left_first,left_rows},Rows{right_first,right_rows},state),
        whole => keep(context,Split{Rows{left_first,left_rows},Rows{right_first,right_rows}},whole),
        left_state => right_state =>
          left right = leaf(context,left_first,left_rows,left_state) leaf(context,right_first,right_rows,right_state)
          Partition.Fork{left,right})
    case Split{Rows{left_first,left_rows},+right_plan} True{}:
      divided(~State,~Partition.Tree<&1,Output,Details>,divide(context,Rows{left_first,left_rows},right_plan,state),
        whole => keep(context,Split{Rows{left_first,left_rows},right_plan},whole),
        left_state => right_state =>
          left right = leaf(context,left_first,left_rows,left_state) each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right_plan,parallel(schedule,right_plan),schedule,context,right_state)
          Partition.Fork{left,right})
    case Split{+left_plan,Rows{right_first,right_rows}} True{}:
      divided(~State,~Partition.Tree<&1,Output,Details>,divide(context,left_plan,Rows{right_first,right_rows},state),
        whole => keep(context,Split{left_plan,Rows{right_first,right_rows}},whole),
        left_state => right_state =>
          left right = each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left_plan,parallel(schedule,left_plan),schedule,context,left_state) leaf(context,right_first,right_rows,right_state)
          Partition.Fork{left,right})
    case Split{+left_plan,+right_plan} True{}:
      divided(~State,~Partition.Tree<&1,Output,Details>,divide(context,left_plan,right_plan,state),whole => keep(context,Split{left_plan,right_plan},whole),
        left_state => right_state =>
          left right = each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left_plan,parallel(schedule,left_plan),schedule,context,left_state) each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right_plan,parallel(schedule,right_plan),schedule,context,right_state)
          Partition.Fork{left,right})

# A band walk carries the tree beside the resource. A split divides the tree
# where it forks; a tree of another shape is kept.
def Banded(~Block: Type,~Resource: Type) -> Type:
  Partition.Tree<&1,Block,Unit> & Resource

# The planner decides once for an operation: every split of an enabled walk
# may run its halves in parallel, inside an image or across images.
def in_parallel(enabled: Bool,layout: Plan) -> Bool:
  enabled

def shared_halves(~Block: Type,~Resource: Type,left: Partition.Tree<&1,Block,Unit>,right: Partition.Tree<&1,Block,Unit>,resources: Resource & Resource) -> Division<Banded(~Block,~Resource)>:
  (left_resource,right_resource) = resources
  Divided{(left,left_resource),(right,right_resource)}

def band_divide(~Block: Type,~Resource: Type,~Context: Data,~share: Resource -> Resource & Resource,
  context: Context,left: Plan,right: Plan,state: Banded(~Block,~Resource)) -> Division<Banded(~Block,~Resource)>:
  (tree,resource) = state
  match tree:
    case Partition.Leaf{block,first,rows,details}: Whole{(Partition.Leaf{block,first,rows,details},resource)}
    case Partition.Fork{left_tree,right_tree}: shared_halves(~Block,~Resource,left_tree,right_tree,share(resource))

def band_leaf(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Partition.Tree<&1,Output,Unit>,
  ~keep: Partition.Tree<&1,Block,Unit> -> Partition.Tree<&1,Output,Unit>,
  context: Context,first: U32,rows: U32,state: Banded(~Block,~Resource)) -> Partition.Tree<&1,Output,Unit>:
  (tree,resource) = state
  match tree:
    case Partition.Leaf{block,offset,span,details}: leaf(context,first,rows,block,resource)
    case Partition.Fork{left,right}: keep(Partition.Fork{left,right})

def band_kept(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~keep: Partition.Tree<&1,Block,Unit> -> Partition.Tree<&1,Output,Unit>,
  context: Context,layout: Plan,state: Banded(~Block,~Resource)) -> Partition.Tree<&1,Output,Unit>:
  (tree,resource) = state
  keep(tree)

# The band recursion. share gives each half its own owner of a read-only
# resource.
def each(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Partition.Tree<&1,Output,Unit>,
  ~keep: Partition.Tree<&1,Block,Unit> -> Partition.Tree<&1,Output,Unit>,
  layout: Plan,+enabled: Bool,+context: Context,
  tree: Partition.Tree<&1,Block,Unit>,resource: Resource) -> Partition.Tree<&1,Output,Unit>:
  each_part(~Banded(~Block,~Resource),~Output,~Unit,~Context,~Bool,~in_parallel,~band_divide(~Block,~Resource,~Context,~share),
    ~band_leaf(~Block,~Output,~Resource,~Context,~leaf,~keep),~band_kept(~Block,~Output,~Resource,~Context,~keep),
    layout,enabled,enabled,context,(tree,resource))

type Indexed is Data:
  Indexed{geometry: Geometry,first: U32,rows: U32}

def map_leaf(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Storage.Buffer{storage,+length,certificate} = buffer
  Storage.Buffer{Traversal.map_in_place(~Element,~Context,~operation,length,0,context,storage),length,
    TraversalCertificate.map_in_place(~Element,~Context,~operation,length,0,context,storage,certificate)}

# A mapper over the whole of one band.
def map_whole_context(~Element: Data,~Context: Data,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,+context: Context,buffer: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Storage.Buffer{storage,+length,certificate} = buffer
  mapper(length,0,context,Storage.Buffer{storage,length,certificate})

def map_band(~Element: Data,~Context: Data,~leaf: @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,context: Context,first: U32,rows: U32,
  buffer: Storage.Buffer<Element>,resource: Unit) -> Bands(~Element):
  Partition.Leaf{leaf(context,buffer),first,rows,Unit{}}

# Parallelism is a planner decision. Sequential execution keeps the bands and
# their owners: a small operation need not pay either a fork or a gather.
def map_with(~Element: Data,~Context: Data,~leaf: @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,+layout: Plan,+parallel: Bool,
  context: Context,bands: Bands(~Element)) -> Bands(~Element):
  each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Context,~unshared,~map_band(~Element,~Context,~leaf),~same(~Storage.Buffer<Element>),
    layout,parallel,context,bands,Unit{})

def scalar_value(~Element: Data,~operation: Element -> Element -> Element,value: Element,element: Element) -> Element:
  operation(element,value)

def scalar(~Element: Data,~operation: Element -> Element -> Element,layout: Plan,parallel: Bool,value: Element,bands: Bands(~Element)) -> Bands(~Element):
  map_with(~Element,~Element,~map_leaf(~Element,~Element,~scalar_value(~Element,~operation)),layout,parallel,value,bands)

def result(~Element: Data,owners: Storage.Buffer<Element> & Bands(~Element)) -> Bands(~Element):
  (source,bands) = owners
  bands

def index_of(context: Indexed,index: U32) -> U32:
  Indexed{geometry,first,rows} = context
  global_index(geometry,first,rows,index)

# Channel c of a leaf holds the positions from c * rows * width; their global
# indices are consecutive from this base.
def channel_base(context: Indexed,channel: U32) -> U32:
  Indexed{Geometry{batch,channels,+height,width},+first,rows} = context
  (((first / height * channels + channel) * height + first % height) * width : U32)

def channels_of(context: Indexed) -> U32:
  Indexed{Geometry{batch,channels,height,width},first,rows} = context
  channels

def plane_of(context: Indexed) -> U32:
  Indexed{Geometry{batch,channels,height,width},first,rows} = context
  (rows * width : U32)

def indexed_buffers(~Element: Data,~operation: U32 -> Element -> Element,+count: U32,+context: Indexed,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  Storage.Buffer{input,input_length,input_certificate} = input
  Storage.Buffer{output,output_length,output_certificate} = output
  destination(~Element,Operations.pair_buffers(~Element,input_length,output_length,
    IndexedAccess.run_tiled(~Element,~Indexed,~index_of,~channel_base,~channels_of,~plane_of,~operation,count,context,input,output),
    IndexedCertificate.tiled_certificates(~Element,~Indexed,~index_of,~channel_base,~channels_of,~plane_of,~operation,count,context,input,output,input_certificate,output_certificate)))

def indexed_leaf(~Element: Data,~operation: U32 -> Element -> Element,
  +geometry: Geometry,+first: U32,+rows: U32,fill: Element,input: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  +count = local_count(geometry,rows)
  indexed_buffers(~Element,~operation,count,Indexed{geometry,first,rows},input,
    Allocation.allocate(~Element,Storage.allocation_depth(U32.max(count,1)),fill,U32.to_nat(count)))

type Filled<-Element: Data> is Data:
  Filled{geometry: Geometry,fill: Element}

def indexed_band(~Element: Data,~operation: U32 -> Element -> Element,context: Filled<Element>,first: U32,rows: U32,
  input: Storage.Buffer<Element>,resource: Unit) -> Bands(~Element):
  Filled{geometry,fill} = context
  +first = first
  +rows = rows
  Partition.Leaf{indexed_leaf(~Element,~operation,geometry,first,rows,fill,input),first,rows,Unit{}}

def apply_indexed(~Element: Data,~operation: U32 -> Element -> Element,
  +layout: Plan,+parallel: Bool,+geometry: Geometry,fill: Element,bands: Bands(~Element)) -> Bands(~Element):
  each(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Filled<Element>,~unshared,~indexed_band(~Element,~operation),~same(~Storage.Buffer<Element>),
    layout,parallel,Filled{geometry,fill},bands,Unit{})

def range_result(~Element: Data,+first: U32,+rows: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Bands(~Element) & Storage.Buffer<Element>:
  (source,output) = owners
  (Partition.Leaf{source,first,rows,Unit{}},output)

# An empty intersection copies nothing, so it skips the per-channel loop.
def range_planes(~Element: Data,empty: Bool,+channels: U32,+count: U32,+source_stride: U32,+destination_stride: U32,
  +source_start: U32,+destination_start: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  Storage.Buffer<Element> & Storage.Buffer<Element>:
  match empty:
    case True{}: owners
    case False{}: planes(~Element,U32.to_nat(channels),0,count,source_stride,destination_stride,source_start,destination_start,owners)

# Repartition and halo extraction share this owned range reader. Empty
# intersections copy zero elements; each source owner is returned unchanged.
def range_leaf(~Element: Data,geometry: Geometry,+target_first: U32,+target_rows: U32,target_offset: U32,+first: U32,+rows: U32,
  source: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Bands(~Element) & Storage.Buffer<Element>:
  Geometry{batch,channels,height,+width} = geometry
  +begin = U32.max(first,target_first)
  +end = U32.max(begin,U32.min((first + rows : U32),(target_first + target_rows : U32)))
  range_result(~Element,first,rows,range_planes(~Element,U32.is_eq(begin,end),channels,((end - begin) * width : U32),
    (rows * width : U32),(target_rows * width : U32),((begin - first) * width : U32),(target_offset + (begin - target_first) * width : U32),(source,output)))

# The rows a range reads and where it writes them.
type Range is Data:
  Range{geometry: Geometry,first: U32,rows: U32,offset: U32}

def range_band(~Element: Data,context: Range,first: U32,rows: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Bands(~Element) & Storage.Buffer<Element>:
  Range{geometry,+target_first,+target_rows,offset} = context
  +first = first
  +rows = rows
  range_leaf(~Element,geometry,target_first,target_rows,offset,first,rows,input,output)

def range_into(~Element: Data,+geometry: Geometry,+first: U32,+rows: U32,+target_offset: U32,bands: Bands(~Element),output: Storage.Buffer<Element>) -> Bands(~Element) & Storage.Buffer<Element>:
  walk(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Range,~range_band(~Element),Range{geometry,first,rows,target_offset},bands,output)

def range(~Element: Data,geometry: Geometry,first: U32,rows: U32,bands: Bands(~Element),output: Storage.Buffer<Element>) -> Bands(~Element) & Storage.Buffer<Element>:
  range_into(~Element,geometry,first,rows,0,bands,output)

def allocate(~Element: Data,fill: Element,+count: U32) -> Storage.Buffer<Element>:
  Allocation.allocate(~Element,Storage.depth_for(count),fill,U32.to_nat(count))

def repartition_band(~Element: Data,context: Filling<Element>,first: U32,rows: U32,bands: Bands(~Element)) -> Bands(~Element) & Bands(~Element):
  Filling{+geometry,fill} = context
  +first = first
  +rows = rows
  Storage.bind_pair(~Bands(~Element),~Storage.Buffer<Element>,~(Bands(~Element) & Bands(~Element)),
    range(~Element,geometry,first,rows,bands,allocate(~Element,fill,local_count(geometry,rows))),source => output =>
      (source,Partition.Leaf{output,first,rows,Unit{}}))

def repartition(~Element: Data,geometry: Geometry,fill: Element,layout: Plan,bands: Bands(~Element)) -> Bands(~Element) & Bands(~Element):
  build(~Storage.Buffer<Element>,~Bands(~Element),~Filling<Element>,~repartition_band(~Element),layout,Filling{geometry,fill},bands)

def repartitioned(~Element: Data,owners: Bands(~Element) & Bands(~Element)) -> Bands(~Element):
  (source,result) = owners
  result

def align_when(~Element: Data,same: Bool,geometry: Geometry,fill: Element,layout: Plan,bands: Bands(~Element)) -> Bands(~Element):
  match same:
    case True{}: bands
    case False{}: repartitioned(~Element,repartition(~Element,geometry,fill,layout,bands))

def align_planned(~Element: Data,geometry: Geometry,fill: Element,+layout: Plan,planned: Bands(~Element) & Plan) -> Bands(~Element):
  (bands,other) = planned
  align_when(~Element,same_plan(layout,other),geometry,fill,layout,bands)

def align(~Element: Data,geometry: Geometry,fill: Element,layout: Plan,bands: Bands(~Element)) -> Bands(~Element):
  align_planned(~Element,geometry,fill,layout,plan(~Element,bands))

def whole_result(~Element: Data,+first: U32,+rows: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Storage.Buffer<Element> & Bands(~Element):
  (source,band) = owners
  (source,Partition.Leaf{band,first,rows,Unit{}})

# Rows [first, first + rows) of every plane of a whole buffer, in a buffer of
# their own beside the source.
def whole_rows(~Element: Data,geometry: Geometry,fill: Element,+first: U32,+rows: U32,input: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Storage.Buffer<Element>:
  Geometry{batch,+channels,+height,+width} = geometry
  planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),
    ((first / height * channels * height + first % height) * width : U32),0,(input,allocate(~Element,fill,(channels * rows * width : U32))))

def whole_leaf(~Element: Data,geometry: Geometry,fill: Element,+first: U32,+rows: U32,input: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Bands(~Element):
  whole_result(~Element,first,rows,whole_rows(~Element,geometry,fill,first,rows,input))

def whole_band(~Element: Data,context: Filling<Element>,first: U32,rows: U32,input: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Bands(~Element):
  Filling{geometry,fill} = context
  whole_leaf(~Element,geometry,fill,first,rows,input)

def whole_ranges(~Element: Data,geometry: Geometry,fill: Element,layout: Plan,input: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Bands(~Element):
  build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Filling<Element>,~whole_band(~Element),layout,Filling{geometry,fill},input)

# A leaf that holds every row of the only image is that image in row-major
# order: the whole buffer is the band, relabelled. Other rows are copied out.
def from_rows(~Element: Data,whole: Bool,geometry: Geometry,fill: Element,+first: U32,+rows: U32,input: Storage.Buffer<Element>) -> Bands(~Element):
  match whole:
    case True{}: Partition.Leaf{input,first,rows,Unit{}}
    case False{}: result(~Element,whole_ranges(~Element,geometry,fill,Rows{first,rows},input))

# The bands of a row-major buffer cut by a plan.
def from_whole(~Element: Data,+geometry: Geometry,fill: Element,layout: Plan,input: Storage.Buffer<Element>) -> Bands(~Element):
  match layout:
    case Rows{+first,+rows}: from_rows(~Element,whole_image(geometry,first,rows),geometry,fill,first,rows,input)
    case Split{left,right}: result(~Element,whole_ranges(~Element,geometry,fill,Split{left,right},input))

def row_plan(depth: Nat,split: Bool,+first: U32,+rows: U32) -> Plan:
  match depth split:
    case 0n split: Rows{first,rows}
    case 1n+rest False{}: Rows{first,rows}
    case 1n+ +rest True{}:
      +half = (rows / 2 : U32)
      Split{row_plan(rest,U32.is_lt(1,half),first,half),
        row_plan(rest,U32.is_lt(1,(rows - half : U32)),(first + half : U32),(rows - half : U32))}

def image_plan(remaining: Nat,+image: U32,+depth: Nat,+height: U32) -> Plan:
  match remaining:
    case 0n: Rows{0,0}
    case 1n: row_plan(depth,U32.is_lt(1,height),(image * height : U32),height)
    case 2n+rest:
      Split{row_plan(depth,U32.is_lt(1,height),(image * height : U32),height),image_plan(1n+rest,(image + 1 : U32),depth,height)}

# Every image of the geometry as one band, its rows unsplit.
def unsplit(geometry: Geometry) -> Plan:
  Geometry{batch,channels,+height,width} = geometry
  image_plan(U32.to_nat(batch),0,0n,height)

# Rows [first, first + count) halved depth times. The first half of every
# split is a whole number of units.
def unit_plan(depth: Nat,+unit: U32,+first: U32,+count: U32) -> Plan:
  match depth:
    case 0n: Rows{first,count}
    case 1n+ +rest:
      +half = (count / (unit * 2) * unit : U32)
      Split{unit_plan(rest,unit,first,half),unit_plan(rest,unit,(first + half : U32),(count - half : U32))}

def combined(~Element: Data,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Storage.Buffer<Element>:
  match result:
    case Operations.Accepted{(left,right)}: left
    case Operations.Rejected{(left,right)}: left

def combine_leaf_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+count: U32,left: Storage.Buffer<Element>,right: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  combined(~Element,Operations.combine_with(~Element,~kernel,left,Views.View{[Views.Axis{count,1}],0},right,Views.View{[Views.Axis{count,1}],0}))

def combine_band_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,geometry: Geometry,first: U32,rows: U32,
  left: Storage.Buffer<Element>,right: Storage.Buffer<Element>,resource: Unit) -> Bands(~Element):
  +rows = rows
  Partition.Leaf{combine_leaf_with(~Element,~kernel,local_count(geometry,rows),left,right),first,rows,Unit{}}

def combine_planned_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+geometry: Geometry,+parallel: Bool,planned: Bands(~Element) & Plan,right: Bands(~Element)) -> Bands(~Element):
  (left,+layout) = planned
  each(~Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Unit,~Geometry,~unshared,
    ~paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Geometry,~combine_band_with(~Element,~kernel)),
    ~left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>),
    layout,parallel,geometry,zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,left,right),Unit{})

# The caller supplies the same plan on both sides, after shape and status
# checks; the plan is read from the left. Trees of another shape keep the left owner.
def combine_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,geometry: Geometry,parallel: Bool,left: Bands(~Element),right: Bands(~Element)) -> Bands(~Element):
  combine_planned_with(~Element,~kernel,geometry,parallel,plan(~Element,left),right)

# A source of the extents source read band by band as a tensor of the
# extents shape under NumPy broadcasting, the bands those of the geometry.
# Views are built again at every use: a View value used twice would make the
# compiler share every View constructor.
type Spread is Data:
  Spread{geometry: Geometry,shape: List<&2,U32>,source: List<&2,U32>}

def spread_geometry(context: Spread) -> Geometry:
  Spread{geometry,shape,source} = context
  geometry

def stretched_view(view: Maybe<&2,Views.View>,+shape: List<&2,U32>) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.stretch(view,shape)
    case None{}: None{}

def stretched_source(+source: List<&2,U32>,+shape: List<&2,U32>) -> Maybe<&2,Views.View>:
  stretched_view(Views.contiguous(Shapes.Shape{source},0),shape)

def sliced_rows(view: Maybe<&2,Views.View>,+start: U32,rows: U32) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.slice(view,2n,start,rows,1)
    case None{}: None{}

def sliced_image(view: Maybe<&2,Views.View>,+image: U32) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.slice(view,0n,image,1,1)
    case None{}: None{}

def sliced_plane_rows(view: Maybe<&2,Views.View>,+start: U32,rows: U32) -> Maybe<&2,Views.View>:
  match view:
    case Some{view}: Views.slice(view,1n,start,rows,1)
    case None{}: None{}

# The cells of a band of image rows: its image and its rows.
def rows_view(+shape: List<&2,U32>,+height: U32,+first: U32,rows: U32,view: Maybe<&2,Views.View>) -> Maybe<&2,Views.View>:
  match shape:
    case [c,h,w]: sliced_plane_rows(view,(first % height : U32),rows)
    case [n,c,h,w]: sliced_rows(sliced_image(view,(first / height : U32)),(first % height : U32),rows)
    case shape: None{}

def rows_extents(+shape: List<&2,U32>,rows: U32) -> Maybe<&2,List<&2,U32>>:
  match shape:
    case [c,h,w]: Some{[c,rows,w]}
    case [n,c,h,w]: Some{[1,c,rows,w]}
    case shape: None{}

def band_view_when(whole: Bool,rows_shaped: Bool,+shape: List<&2,U32>,+height: U32,+first: U32,rows: U32,view: Maybe<&2,Views.View>) -> Maybe<&2,Views.View>:
  match whole rows_shaped:
    case True{} rows_shaped: view
    case False{} True{}: rows_view(shape,height,first,rows,view)
    case False{} False{}: None{}

def band_extents_when(whole: Bool,rows_shaped: Bool,+shape: List<&2,U32>,rows: U32) -> Maybe<&2,List<&2,U32>>:
  match whole rows_shaped:
    case True{} rows_shaped: Some{shape}
    case False{} True{}: rows_extents(shape,rows)
    case False{} False{}: None{}

# The view of the source a band reads. A band of a whole image reads the whole
# stretched view, and a band of image rows of an image shape reads its image
# and its rows. Other bands have none.
def band_view(+context: Spread,+first: U32,+rows: U32) -> Maybe<&2,Views.View>:
  Spread{+geometry,+shape,+source} = context
  Geometry{batch,channels,+height,width} = geometry
  band_view_when(whole_image(geometry,first,rows),image_rows(shape,geometry),shape,height,first,rows,stretched_source(source,shape))

# The extents of the cells of that band, in the order the band stores them.
def band_extents(+context: Spread,+first: U32,+rows: U32) -> Maybe<&2,List<&2,U32>>:
  Spread{+geometry,+shape,source} = context
  band_extents_when(whole_image(geometry,first,rows),image_rows(shape,geometry),shape,rows)

def has_view(view: Maybe<&2,Views.View>) -> Bool:
  match view:
    case Some{view}: True{}
    case None{}: False{}

# Every band of the plan has a view.
def spread_boxed(+context: Spread,layout: Plan) -> Bool:
  match layout:
    case Rows{first,rows}: has_view(band_view(context,first,rows))
    case Split{left,right}: spread_boxed(context,left) && spread_boxed(context,right)

def compact_axes(source: List<&2,U32>,axes: List<&2,Views.Axis>) -> List<&2,Views.Axis>:
  match source axes:
    case Con{+single,rest} Con{Views.Axis{extent,stride},others}: Views.Axis{Bool.pick(U32,U32.is_eq(single,1),1,extent),stride} <> compact_axes(rest,others)
    case source axes: []

# The elements of a band view that the source holds once: an axis the source
# repeats keeps one element.
def compact_of(+source: List<&2,U32>,view: Maybe<&2,Views.View>) -> Maybe<&2,Views.View>:
  match view:
    case Some{Views.View{+axes,offset}}: Some{Views.View{compact_axes(Shapes.with_rank(source,List.length(&2,Views.Axis,axes)),axes),offset}}
    case None{}: None{}

def compact_extents(source: List<&2,U32>,extents: List<&2,U32>) -> List<&2,U32>:
  match source extents:
    case Con{+single,rest} Con{extent,others}: Bool.pick(U32,U32.is_eq(single,1),1,extent) <> compact_extents(rest,others)
    case source extents: []

def compact_with(+source: List<&2,U32>,extents: Maybe<&2,List<&2,U32>>) -> Maybe<&2,List<&2,U32>>:
  match extents:
    case Some{+extents}: Some{compact_extents(Shapes.with_rank(source,List.length(&2,U32,extents)),extents)}
    case None{}: None{}

# The extents of the part of the source a band reads: those of its cells, an
# axis the source repeats kept once.
def band_compact(+context: Spread,+first: U32,+rows: U32) -> Maybe<&2,List<&2,U32>>:
  Spread{geometry,shape,+source} = context
  compact_with(source,band_extents(context,first,rows))

def contiguous_of(extents: Maybe<&2,List<&2,U32>>) -> Maybe<&2,Views.View>:
  match extents:
    case Some{+extents}: Views.contiguous(Shapes.Shape{extents},0)
    case None{}: None{}

def count_of_extents(extents: Maybe<&2,List<&2,U32>>) -> Maybe<&2,U32>:
  match extents:
    case Some{extents}: Shapes.numel(Shapes.Shape{extents})
    case None{}: None{}

type Pieces<-Element: Data> is Data:
  Pieces{spread: Spread,fill: Element}

def piece_copied(~Element: Data,+first: U32,+rows: U32,result: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>) -> Storage.Buffer<Element> & Bands(~Element):
  match result:
    case Operations.Accepted{(source,piece)}: (source,Partition.Leaf{piece,first,rows,Unit{}})
    case Operations.Rejected{(source,piece)}: (source,Partition.Leaf{piece,first,rows,Unit{}})

# A band view lies inside the source it was stretched from, so the band
# copies through it without a check.
def piece_views(~Element: Data,source_view: Maybe<&2,Views.View>,piece_view: Maybe<&2,Views.View>,count: Maybe<&2,U32>,+fill: Element,+first: U32,+rows: U32,
  source: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Bands(~Element):
  match source_view piece_view count:
    case Some{source_view} Some{piece_view} Some{+count}:
      piece_copied(~Element,first,rows,Operations.copy_when(~Element,True{},Some{count},source_view,piece_view,source,allocate(~Element,fill,count)))
    case source_view piece_view count: (source,Partition.Leaf{allocate(~Element,fill,0),first,rows,Unit{}})

def source_of(context: Spread) -> List<&2,U32>:
  Spread{geometry,shape,source} = context
  source

def piece_band(~Element: Data,context: Pieces<Element>,first: U32,rows: U32,source: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Bands(~Element):
  Pieces{+spread,+fill} = context
  +first = first
  +rows = rows
  piece_views(~Element,compact_of(source_of(spread),band_view(spread,first,rows)),contiguous_of(band_compact(spread,first,rows)),
    count_of_extents(band_compact(spread,first,rows)),fill,first,rows,source)

def spread_leaf(~Element: Data,whole: Bool,+context: Spread,fill: Element,+first: U32,+rows: U32,source: Storage.Buffer<Element>) -> Bands(~Element):
  match whole:
    case True{}: Partition.Leaf{source,first,rows,Unit{}}
    case False{}: result(~Element,piece_band(~Element,Pieces{context,fill},first,rows,source))

# The part of a source each band of the plan reads, copied in plan order: the
# elements of its band view, an axis that repeats one element kept once. One
# band of a whole image reads all of the source, which it takes as is.
def spread(~Element: Data,+context: Spread,fill: Element,layout: Plan,source: Storage.Buffer<Element>) -> Bands(~Element):
  match layout:
    case Rows{+first,+rows}: spread_leaf(~Element,whole_image(spread_geometry(context),first,rows),context,fill,first,rows,source)
    case Split{left,right}:
      result(~Element,build(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Pieces<Element>,~piece_band(~Element),Split{left,right},Pieces{context,fill},source))

def stretched_to(view: Maybe<&2,Views.View>,target: Maybe<&2,List<&2,U32>>) -> Maybe<&2,Views.View>:
  match view target:
    case Some{view} Some{+target}: Views.stretch(view,target)
    case view target: None{}

def kernel_left(~Element: Data,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  (left,right) = owners
  left

# The kernel on a band and its part: the views were checked when the
# operation chose its layout, so the band runs the kernel itself.
def spread_views(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,
  +count: U32,right_view: Maybe<&2,Views.View>,left: Storage.Buffer<Element>,right: Storage.Buffer<Element>) -> Storage.Buffer<Element>:
  match right_view:
    case Some{right_view}: kernel_left(~Element,kernel(U32.to_nat(count),right_view,0,left,right))
    case None{}: left

# A band combined with its part of the source: the band read with the
# extents of its cells, the part stretched to them.
def combine_spread_band_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,
  context: Spread,first: U32,rows: U32,left: Storage.Buffer<Element>,right: Storage.Buffer<Element>,resource: Unit) -> Bands(~Element):
  +context = context
  +first = first
  +rows = rows
  Partition.Leaf{spread_views(~Element,~kernel,local_count(spread_geometry(context),rows),
    stretched_to(contiguous_of(band_compact(context,first,rows)),band_extents(context,first,rows)),left,right),first,rows,Unit{}}

def combine_spread_planned_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,
  +context: Spread,+parallel: Bool,planned: Bands(~Element) & Plan,right: Bands(~Element)) -> Bands(~Element):
  (left,+layout) = planned
  each(~Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Storage.Buffer<Element>,~Unit,~Spread,~unshared,
    ~paired(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Unit,~Spread,~combine_spread_band_with(~Element,~kernel)),
    ~left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>),
    layout,parallel,context,zip(~Storage.Buffer<Element>,~Storage.Buffer<Element>,left,right),Unit{})

# The bands combined with the parts spread copies for them from a source.
def combine_spread_with(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,
  +context: Spread,parallel: Bool,left: Bands(~Element),right: Bands(~Element)) -> Bands(~Element):
  combine_spread_planned_with(~Element,~kernel,context,parallel,plan(~Element,left),right)

# A band made by a checked call, beside whether the call was usable.
type Checked<-Element: Data> is Type:
  Checked{output: Storage.Buffer<Element>,usable: Bool}

def Transformed(~Element: Data) -> Type:
  Partition.Tree<&1,Checked<Element>,Unit>

def transform_band(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Transformed(~Element),
  context: Context,first: U32,rows: U32,input: Storage.Buffer<Element>,resource: Unit) -> Transformed(~Element):
  leaf(context,first,rows,input)

# A tree the plan does not describe is not transformed.
def untransformed(~Element: Data,bands: Bands(~Element)) -> Transformed(~Element):
  match bands:
    case Partition.Leaf{input,first,rows,details}: Partition.Leaf{Checked{input,False{}},first,rows,details}
    case Partition.Fork{left,right}: Partition.Fork{untransformed(~Element,left),untransformed(~Element,right)}

def checked_band(~Element: Data,context: Unit,first: U32,rows: U32,checked: Checked<Element>) -> Bands(~Element) & Bool:
  Checked{output,usable} = checked
  (Partition.Leaf{output,first,rows,Unit{}},usable)

# Every band through its own checked call. The bands come back beside whether
# every call was usable.
def transform(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Transformed(~Element),
  +layout: Plan,+parallel: Bool,context: Context,bands: Bands(~Element)) -> Bands(~Element) & Bool:
  reduce(~Checked<Element>,~Storage.Buffer<Element>,~Bool,~Unit,~checked_band(~Element),~Bool.and,Unit{},
    each(~Storage.Buffer<Element>,~Checked<Element>,~Unit,~Context,~unshared,~transform_band(~Element,~Context,~leaf),~untransformed(~Element),
      layout,parallel,context,bands,Unit{}))

def transform_pair(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Transformed(~Element),
  context: Context,first: U32,rows: U32,pair: Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,resource: Unit) -> Transformed(~Element):
  match pair:
    case Both{left,right}: leaf(context,first,rows,left,right)
    case Only{left,own_first,own_rows}: Partition.Leaf{Checked{left,False{}},own_first,own_rows,Unit{}}

def untransformed_pairs(~Element: Data,pairs: Partition.Tree<&1,Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,Unit>) -> Transformed(~Element):
  untransformed(~Element,left_of(~Storage.Buffer<Element>,~Storage.Buffer<Element>,pairs))

# Every band of the left through its own checked call with the band of the
# right of the same rows; a left band without a partner is not usable.
def transform_paired(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Transformed(~Element),
  +layout: Plan,+parallel: Bool,context: Context,pairs: Partition.Tree<&1,Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,Unit>) -> Bands(~Element) & Bool:
  reduce(~Checked<Element>,~Storage.Buffer<Element>,~Bool,~Unit,~checked_band(~Element),~Bool.and,Unit{},
    each(~Paired<Storage.Buffer<Element>,Storage.Buffer<Element>>,~Checked<Element>,~Unit,~Context,~unshared,~transform_pair(~Element,~Context,~leaf),~untransformed_pairs(~Element),
      layout,parallel,context,pairs,Unit{}))

type Inspection is Data:
  Inspection{valid: Bool,first: U32,end: U32,leaves: U32}

def join_inspection(left: Inspection,right: Inspection) -> Inspection:
  Inspection{lv,lf,+le,ln} = left
  Inspection{rv,+rf,re,rn} = right
  Inspection{lv && rv && U32.is_eq(le,rf),lf,re,(ln + rn : U32)}

def leaf_inspection(geometry: Geometry,+length: Nat,depth: Nat,+first: U32,+rows: U32) -> Inspection:
  Geometry{batch,+channels,+height,+width} = geometry
  Inspection{U32.is_le(rows,(height - first % height : U32)) &&
      Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) &&
      Nat.is_eq(length,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))) &&
      Nat.is_le(length,U32.to_nat(U32.shln(1,depth))),first,(first + rows : U32),1}

def inspect_leaf(~Element: Data,geometry: Geometry,buffer: Storage.Buffer<Element>,+first: U32,+rows: U32) -> Bands(~Element) & Inspection:
  Storage.Buffer{storage,+length,Certificate.Certificate{+depth,equation}} = buffer
  (Partition.Leaf{Storage.Buffer{storage,length,Certificate.Certificate{depth,equation}},first,rows,Unit{}},
    leaf_inspection(geometry,length,depth,first,rows))

def inspect_band(~Element: Data,geometry: Geometry,first: U32,rows: U32,buffer: Storage.Buffer<Element>) -> Bands(~Element) & Inspection:
  +first = first
  +rows = rows
  inspect_leaf(~Element,geometry,buffer,first,rows)

def inspect(~Element: Data,+geometry: Geometry,bands: Bands(~Element)) -> Bands(~Element) & Inspection:
  reduce(~Storage.Buffer<Element>,~Storage.Buffer<Element>,~Inspection,~Geometry,~inspect_band(~Element),~join_inspection,geometry,bands)

# Leaves that fit the geometry and cover all of its rows in order.
def covers(geometry: Geometry,inspection: Inspection) -> Bool:
  Geometry{batch,channels,height,width} = geometry
  Inspection{valid,first,end,leaves} = inspection
  valid && U32.is_zero(first) && Nat.is_eq(U32.to_nat(end),Nat.mul(U32.to_nat(batch),U32.to_nat(height)))

