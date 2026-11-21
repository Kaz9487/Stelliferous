# Pieces own the cells of a tensor: a tree of buffers whose leaves hold
# contiguous ranges of the canonical order of lib/tensor_grid.bend, offset and
# span counted in cells. Ranges describe the tree without its buffers. The
# skeleton below walks, folds, builds and runs work over the tree. The work of
# one task may own several adjacent pieces.
import Base
import ./storage_buffer.bend as Storage
import ./storage_clone.bend as Clone
import ./storage_allocation.bend as Allocation
import ./traversal_partition.bend as Partition
import ./tensor_grid.bend as Grid

def Pieces(~Element: Data) -> Type:
  Partition.Tree<&1,Storage.Buffer<Element>,Unit>

# The output trees left of the walk, each waiting for the tree on its right.
type Lefts<-Output: Type> is Type:
  NoLefts{}
  Left{tree: Partition.Tree<&1,Output,Unit>,rest: Lefts<Output>}

# What a walk carries into a subtree, and what it gives back after it.
type Walking<-Output: Type,-State: Type> is Type:
  Walking{lefts: Lefts<Output>,state: State}

type Walked<-Output: Type,-State: Type> is Type:
  Walked{tree: Partition.Tree<&1,Output,Unit>,lefts: Lefts<Output>,state: State}

def leaf_walked(~Output: Type,~State: Type,lefts: Lefts<Output>,result: Partition.Tree<&1,Output,Unit> & State) -> Walked<Output,State>:
  (tree,state) = result
  Walked{tree,lefts,state}

# One output piece of first cell first and count cells.
def piece_walked(~Element: Data,~State: Type,lefts: Lefts<Storage.Buffer<Element>>,+first: U32,+count: U32,result: Storage.Buffer<Element> & State) ->
  Walked<Storage.Buffer<Element>,State>:
  (output,state) = result
  Walked{Partition.Leaf{output,first,count,Unit{}},lefts,state}

def pushed(~Output: Type,~State: Type,walked: Walked<Output,State>) -> Walking<Output,State>:
  Walked{tree,lefts,state} = walked
  Walking{Left{tree,lefts},state}

# The tree just walked joins the tree waiting on its left.
def joined(~Output: Type,~State: Type,walked: Walked<Output,State>) -> Walked<Output,State>:
  match walked:
    case Walked{right,Left{left,rest},state}: Walked{Partition.Fork{left,right},rest,state}
    case Walked{right,NoLefts{},state}: Walked{right,NoLefts{},state}

def walk_result(~Output: Type,~State: Type,walked: Walked<Output,State>) -> Partition.Tree<&1,Output,Unit> & State:
  Walked{tree,lefts,state} = walked
  (tree,state)

# The walk calls no closure, so a task whose leaf walks its pieces stays
# fork-free and the runtime keeps it for the work turn.
def walked(~Input: Type,~Output: Type,~State: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Input -> State -> Partition.Tree<&1,Output,Unit> & State,
  +context: Context,tree: Partition.Tree<&1,Input,Unit>,ready: Walking<Output,State>) -> Walked<Output,State>:
  match tree:
    case Partition.Leaf{input,first,count,details}:
      Walking{lefts,state} = ready
      leaf_walked(~Output,~State,lefts,leaf(context,first,count,input,state))
    case Partition.Fork{left,right}:
      joined(~Output,~State,walked(~Input,~Output,~State,~Context,~leaf,context,right,
        pushed(~Output,~State,walked(~Input,~Output,~State,~Context,~leaf,context,left,ready))))

# An existing tree with a state threaded through its leaves in order. The leaf
# takes the state and gives it back beside what it leaves in the tree.
def walk(~Input: Type,~Output: Type,~State: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Input -> State -> Partition.Tree<&1,Output,Unit> & State,
  +context: Context,tree: Partition.Tree<&1,Input,Unit>,state: State) -> Partition.Tree<&1,Output,Unit> & State:
  walk_result(~Output,~State,walked(~Input,~Output,~State,~Context,~leaf,context,tree,Walking{NoLefts{},state}))

# Every leaf of a tree consumed into a state, in order.
def fold(~Input: Type,~State: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Input -> State -> State,
  +context: Context,tree: Partition.Tree<&1,Input,Unit>,state: State) -> State:
  match tree:
    case Partition.Leaf{input,first,count,details}: leaf(context,first,count,input,state)
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
    case Partition.Leaf{input,first,count,details}: leaf(context,first,count,input)
    case Partition.Fork{left,right}:
      reduced(~Output,~Value,~join,reduce(~Input,~Output,~Value,~Context,~leaf,~join,context,left),reduce(~Input,~Output,~Value,~Context,~leaf,~join,context,right))

# A description of the pieces holds no Buffer. It is a concrete Data type:
# duplicating Tree<&2,...> through a computed type makes all constructors
# shared in the official compiler, including the affine Buffer tree.
type Ranges is Data:
  Range{first: U32,count: U32}
  Split{left: Ranges,right: Ranges}

def ranges_fold(~Value: Type,~Context: Data,~leaf: Context -> U32 -> U32 -> Value,~join: Context -> Ranges -> Ranges -> Value -> Value -> Value,
  layout: Ranges,+context: Context) -> Value:
  match layout:
    case Range{first,count}: leaf(context,first,count)
    case Split{+left,+right}:
      join(context,left,right,ranges_fold(~Value,~Context,~leaf,~join,left,context),ranges_fold(~Value,~Context,~leaf,~join,right,context))

# A join that needs neither the context nor the halves of the split.
def plain(~Value: Type,~Context: Data,~join: Value -> Value -> Value,context: Context,left: Ranges,right: Ranges,first: Value,second: Value) -> Value:
  join(first,second)

def first(layout: Ranges) -> U32:
  match layout:
    case Range{first,count}: first
    case Split{left,right}: first(left)

def last(layout: Ranges) -> U32:
  match layout:
    case Range{first,count}: (first + count : U32)
    case Split{left,right}: last(right)

# A tree made from ranges, with a state threaded through the leaves in order.
# The leaf takes the state and gives it back beside the piece it made.
def build(~Output: Type,~State: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> State -> State & Partition.Tree<&1,Output,Unit>,
  layout: Ranges,+context: Context,state: State) -> State & Partition.Tree<&1,Output,Unit>:
  match layout:
    case Range{first,count}: leaf(context,first,count,state)
    case Split{left,right}:
      Storage.bind_pair(~State,~Partition.Tree<&1,Output,Unit>,~(State & Partition.Tree<&1,Output,Unit>),
        build(~Output,~State,~Context,~leaf,left,context,state),state => left =>
          Storage.bind_pair(~State,~Partition.Tree<&1,Output,Unit>,~(State & Partition.Tree<&1,Output,Unit>),
            build(~Output,~State,~Context,~leaf,right,context,state),state => right => (state,Partition.Fork{left,right})))

# Two trees of one shape, joined leaf by leaf. A left leaf without a partner
# keeps its own range, so the left tree can be returned unchanged.
type Paired<-Left: Type,-Right: Type> is Type:
  Both{left: Left,right: Right}
  Only{left: Left,first: U32,count: U32}

def only(~Left: Type,~Right: Type,tree: Partition.Tree<&1,Left,Unit>) -> Partition.Tree<&1,Paired<Left,Right>,Unit>:
  match tree:
    case Partition.Leaf{left,+first,+count,details}: Partition.Leaf{Only{left,first,count},first,count,Unit{}}
    case Partition.Fork{left,right}: Partition.Fork{only(~Left,~Right,left),only(~Left,~Right,right)}

def zip(~Left: Type,~Right: Type,left: Partition.Tree<&1,Left,Unit>,right: Partition.Tree<&1,Right,Unit>) -> Partition.Tree<&1,Paired<Left,Right>,Unit>:
  match left right:
    case Partition.Leaf{left,first,count,details} Partition.Leaf{right,other_first,other_count,other_details}:
      Partition.Leaf{Both{left,right},first,count,Unit{}}
    case Partition.Fork{left,right} Partition.Fork{other_left,other_right}:
      Partition.Fork{zip(~Left,~Right,left,other_left),zip(~Left,~Right,right,other_right)}
    case left right: only(~Left,~Right,left)

def left_leaf(~Left: Type,~Right: Type,pair: Paired<Left,Right>,first: U32,count: U32) -> Partition.Tree<&1,Left,Unit>:
  match pair:
    case Both{left,right}: Partition.Leaf{left,first,count,Unit{}}
    case Only{left,own_first,own_count}: Partition.Leaf{left,own_first,own_count,Unit{}}

def left_of(~Left: Type,~Right: Type,tree: Partition.Tree<&1,Paired<Left,Right>,Unit>) -> Partition.Tree<&1,Left,Unit>:
  match tree:
    case Partition.Leaf{pair,first,count,details}: left_leaf(~Left,~Right,pair,first,count)
    case Partition.Fork{left,right}: Partition.Fork{left_of(~Left,~Right,left),left_of(~Left,~Right,right)}

def paired(~Left: Type,~Right: Type,~Resource: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Left -> Right -> Resource -> Partition.Tree<&1,Left,Unit>,
  context: Context,first: U32,count: U32,pair: Paired<Left,Right>,resource: Resource) -> Partition.Tree<&1,Left,Unit>:
  match pair:
    case Both{left,right}: leaf(context,first,count,left,right,resource)
    case Only{left,own_first,own_count}: Partition.Leaf{left,own_first,own_count,Unit{}}

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
def always(schedule: Unit,layout: Ranges) -> Bool:
  True{}

# The one recursion over ranges. The ranges give every part its range and the
# structure. divide gives each half of a split its own state, a state it keeps
# whole becomes one part through keep. A split whose schedule allows it runs
# its halves in parallel. The last split calls its two leaves directly: a leaf
# that neither forks nor calls a closure is then its own task, which the
# runtime keeps for the work turn. A leaf reached through the forking
# recursion would run in the grow turn on one worker per ring row.
def each_part(~State: Type,~Output: Type,~Details: Data,~Context: Data,~Schedule: Data,
  ~parallel: Schedule -> Ranges -> Bool,
  ~divide: Context -> Ranges -> Ranges -> State -> Division<State>,
  ~leaf: Context -> U32 -> U32 -> State -> Partition.Tree<&1,Output,Details>,
  ~keep: Context -> Ranges -> State -> Partition.Tree<&1,Output,Details>,
  layout: Ranges,mode: Bool,+schedule: Schedule,+context: Context,state: State) -> Partition.Tree<&1,Output,Details>:
  match layout mode:
    case Range{first,count} mode: leaf(context,first,count,state)
    case Split{+left_ranges,+right_ranges} False{}:
      divided(~State,~Partition.Tree<&1,Output,Details>,divide(context,left_ranges,right_ranges,state),whole => keep(context,Split{left_ranges,right_ranges},whole),
        left_state => right_state =>
          Partition.Fork{each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left_ranges,parallel(schedule,left_ranges),schedule,context,left_state),
            each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right_ranges,parallel(schedule,right_ranges),schedule,context,right_state)})
    case Split{Range{left_first,left_count},Range{right_first,right_count}} True{}:
      divided(~State,~Partition.Tree<&1,Output,Details>,divide(context,Range{left_first,left_count},Range{right_first,right_count},state),
        whole => keep(context,Split{Range{left_first,left_count},Range{right_first,right_count}},whole),
        left_state => right_state =>
          left right = leaf(context,left_first,left_count,left_state) leaf(context,right_first,right_count,right_state)
          Partition.Fork{left,right})
    case Split{Range{left_first,left_count},+right_ranges} True{}:
      divided(~State,~Partition.Tree<&1,Output,Details>,divide(context,Range{left_first,left_count},right_ranges,state),
        whole => keep(context,Split{Range{left_first,left_count},right_ranges},whole),
        left_state => right_state =>
          left right = leaf(context,left_first,left_count,left_state) each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right_ranges,parallel(schedule,right_ranges),schedule,context,right_state)
          Partition.Fork{left,right})
    case Split{+left_ranges,Range{right_first,right_count}} True{}:
      divided(~State,~Partition.Tree<&1,Output,Details>,divide(context,left_ranges,Range{right_first,right_count},state),
        whole => keep(context,Split{left_ranges,Range{right_first,right_count}},whole),
        left_state => right_state =>
          left right = each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left_ranges,parallel(schedule,left_ranges),schedule,context,left_state) leaf(context,right_first,right_count,right_state)
          Partition.Fork{left,right})
    case Split{+left_ranges,+right_ranges} True{}:
      divided(~State,~Partition.Tree<&1,Output,Details>,divide(context,left_ranges,right_ranges,state),whole => keep(context,Split{left_ranges,right_ranges},whole),
        left_state => right_state =>
          left right = each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,left_ranges,parallel(schedule,left_ranges),schedule,context,left_state) each_part(~State,~Output,~Details,~Context,~Schedule,~parallel,~divide,~leaf,~keep,right_ranges,parallel(schedule,right_ranges),schedule,context,right_state)
          Partition.Fork{left,right})

# A piece walk carries the tree beside the resource. A split divides the tree
# where it forks. A tree of another shape is kept.
def Pieced(~Block: Type,~Resource: Type) -> Type:
  Partition.Tree<&1,Block,Unit> & Resource

# The planner decides once for an operation: every split of an enabled walk
# may run its halves in parallel, whatever cells they hold.
def in_parallel(enabled: Bool,layout: Ranges) -> Bool:
  enabled

def shared_halves(~Block: Type,~Resource: Type,left: Partition.Tree<&1,Block,Unit>,right: Partition.Tree<&1,Block,Unit>,resources: Resource & Resource) -> Division<Pieced(~Block,~Resource)>:
  (left_resource,right_resource) = resources
  Divided{(left,left_resource),(right,right_resource)}

def piece_divide(~Block: Type,~Resource: Type,~Context: Data,~share: Resource -> Resource & Resource,
  context: Context,left: Ranges,right: Ranges,state: Pieced(~Block,~Resource)) -> Division<Pieced(~Block,~Resource)>:
  (tree,resource) = state
  match tree:
    case Partition.Leaf{block,first,count,details}: Whole{(Partition.Leaf{block,first,count,details},resource)}
    case Partition.Fork{left_tree,right_tree}: shared_halves(~Block,~Resource,left_tree,right_tree,share(resource))

def piece_leaf(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Partition.Tree<&1,Output,Unit>,
  ~keep: Partition.Tree<&1,Block,Unit> -> Partition.Tree<&1,Output,Unit>,
  context: Context,first: U32,count: U32,state: Pieced(~Block,~Resource)) -> Partition.Tree<&1,Output,Unit>:
  (tree,resource) = state
  match tree:
    case Partition.Leaf{block,offset,span,details}: leaf(context,first,count,block,resource)
    case Partition.Fork{left,right}: keep(Partition.Fork{left,right})

def piece_kept(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~keep: Partition.Tree<&1,Block,Unit> -> Partition.Tree<&1,Output,Unit>,
  context: Context,layout: Ranges,state: Pieced(~Block,~Resource)) -> Partition.Tree<&1,Output,Unit>:
  (tree,resource) = state
  keep(tree)

# The piece recursion. share gives each half its own owner of a read-only
# resource.
def each(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Partition.Tree<&1,Output,Unit>,
  ~keep: Partition.Tree<&1,Block,Unit> -> Partition.Tree<&1,Output,Unit>,
  layout: Ranges,+enabled: Bool,+context: Context,
  tree: Partition.Tree<&1,Block,Unit>,resource: Resource) -> Partition.Tree<&1,Output,Unit>:
  each_part(~Pieced(~Block,~Resource),~Output,~Unit,~Context,~Bool,~in_parallel,~piece_divide(~Block,~Resource,~Context,~share),
    ~piece_leaf(~Block,~Output,~Resource,~Context,~leaf,~keep),~piece_kept(~Block,~Output,~Resource,~Context,~keep),
    layout,enabled,enabled,context,(tree,resource))

# A task may own several existing pieces. Its serial walk keeps their metadata
# and owners. The outer ranges are the tasks, the pieces stay as they are.
def group_piece(~Block: Type,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Unit -> Partition.Tree<&1,Output,Unit>,
  context: Context,first: U32,count: U32,block: Block) -> Partition.Tree<&1,Output,Unit> & Unit:
  (leaf(context,first,count,block,Unit{}),Unit{})

def group_unit(left: Unit,right: Unit) -> Unit:
  Unit{}

def group_run(~Block: Type,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Unit -> Partition.Tree<&1,Output,Unit>,
  context: Context,tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Output,Unit>:
  Pair.fst(Partition.Tree<&1,Output,Unit>,Unit,
    reduce(~Block,~Output,~Unit,~Context,~group_piece(~Block,~Output,~Context,~leaf),~group_unit,context,tree))

def group_divide(~Block: Type,~Context: Data,context: Context,left: Ranges,right: Ranges,tree: Partition.Tree<&1,Block,Unit>) -> Division<Partition.Tree<&1,Block,Unit>>:
  match tree:
    case Partition.Leaf{block,first,count,details}: Whole{Partition.Leaf{block,first,count,details}}
    case Partition.Fork{left,right}: Divided{left,right}

def group_task(~Block: Type,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Unit -> Partition.Tree<&1,Output,Unit>,
  context: Context,first: U32,count: U32,tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Output,Unit>:
  group_run(~Block,~Output,~Context,~leaf,context,tree)

# Positions [first, first + count) halved depth times. The first half of
# every split is a whole number of units.
def unit_ranges(depth: Nat,+unit: U32,+first: U32,+count: U32) -> Ranges:
  match depth:
    case 0n: Range{first,count}
    case 1n+ +rest:
      +half = (count / (unit * 2) * unit : U32)
      Split{unit_ranges(rest,unit,first,half),unit_ranges(rest,unit,(first + half : U32),(count - half : U32))}

# The first cell of ranges and how many cells they hold.
def bounds(+layout: Ranges) -> U32 & U32:
  +first = first(layout)
  (first,(last(layout) - first : U32))

def subtree_bounded(~Block: Type,~Output: Type,~Context: Data,
  ~task: Context -> U32 -> U32 -> Partition.Tree<&1,Block,Unit> -> Partition.Tree<&1,Output,Unit>,
  context: Context,bounded: U32 & U32,tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Output,Unit>:
  (first,count) = bounded
  task(context,first,count,tree)

def subtree_kept(~Block: Type,~Output: Type,~Context: Data,
  ~task: Context -> U32 -> U32 -> Partition.Tree<&1,Block,Unit> -> Partition.Tree<&1,Output,Unit>,
  context: Context,layout: Ranges,tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Output,Unit>:
  subtree_bounded(~Block,~Output,~Context,~task,context,bounds(layout),tree)

# Every task given the subtree of its pieces, its first cell and its cells.
# A split whose tree does not fork stays one task.
def subtrees(~Block: Type,~Output: Type,~Context: Data,
  ~task: Context -> U32 -> U32 -> Partition.Tree<&1,Block,Unit> -> Partition.Tree<&1,Output,Unit>,
  layout: Ranges,+enabled: Bool,context: Context,tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Output,Unit>:
  each_part(~Partition.Tree<&1,Block,Unit>,~Output,~Unit,~Context,~Bool,~in_parallel,~group_divide(~Block,~Context),
    ~task,~subtree_kept(~Block,~Output,~Context,~task),layout,enabled,enabled,context,tree)

def grouped(~Block: Type,~Output: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Unit -> Partition.Tree<&1,Output,Unit>,
  layout: Ranges,+enabled: Bool,context: Context,tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Output,Unit>:
  subtrees(~Block,~Output,~Context,~group_task(~Block,~Output,~Context,~leaf),layout,enabled,context,tree)

# The ranges of the pieces of a grid: a left-complete tree over the outer
# indices whose leaves are left-complete trees over the pieces of one outer
# index. A left half holds the largest power of two below the count, so every
# aligned block of a power of two is a complete subtree.
def cells_of(+grid: Grid.Grid,+first: U32,+count: U32) -> U32:
  Grid.Grid{+cells,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  (U32.min(Grid.start(grid,(first + count : U32)),cells) - U32.min(Grid.start(grid,first),cells) : U32)

def below(+count: U32) -> U32:
  U32.shln(1,Nat.sub(Storage.depth_for(count),1n))

def inner_ranges(depth: Nat,single: Bool,+grid: Grid.Grid,+first: U32,+count: U32) -> Ranges:
  match depth single:
    case 0n single: Range{Grid.start(grid,first),cells_of(grid,first,count)}
    case 1n+rest True{}: Range{Grid.start(grid,first),cells_of(grid,first,count)}
    case 1n+ +rest False{}:
      +left = below(count)
      +right = (count - left : U32)
      Split{inner_ranges(rest,U32.is_le(left,1),grid,first,left),inner_ranges(rest,U32.is_le(right,1),grid,(first + left : U32),right)}

def outer_ranges(depth: Nat,single: Bool,+grid: Grid.Grid,+per: U32,+first: U32,+count: U32) -> Ranges:
  match depth single:
    case 0n single: Range{Grid.start(grid,(first * per : U32)),cells_of(grid,(first * per : U32),(count * per : U32))}
    case 1n+rest True{}: inner_ranges(32n,U32.is_le(per,1),grid,(first * per : U32),per)
    case 1n+ +rest False{}:
      +left = below(count)
      +right = (count - left : U32)
      Split{outer_ranges(rest,U32.is_le(left,1),grid,per,first,left),outer_ranges(rest,U32.is_le(right,1),grid,per,(first + left : U32),right)}

# The ranges of groups of outer indices, group of them each: a left-complete
# tree over the groups whose leaves are left-complete trees over the pieces of
# one group.
def ranges_in(+grid: Grid.Grid,+group: U32) -> Ranges:
  Grid.Grid{cells,prefix,+outer,extent,unit_cells,piece_units,+per_outer,pieces} = grid
  +size = U32.max(group,1)
  +groups = U32.max((outer / size : U32),1)
  outer_ranges(32n,U32.is_le(groups,1),grid,(U32.max(per_outer,1) * size : U32),0,groups)

def ranges_of_grid(+grid: Grid.Grid) -> Ranges:
  ranges_in(grid,1)

# The blocks of a tree in order, and a tree of them in the shape of ranges
# whose every range is one block. Only the tree changes: no block moves. A
# range past the last block gets a vacant block.
type Queue<-Block: Type> is Type:
  Empty{}
  Queued{block: Block,rest: Queue<Block>}

def queued(~Block: Type,tree: Partition.Tree<&1,Block,Unit>,rest: Queue<Block>) -> Queue<Block>:
  match tree:
    case Partition.Leaf{block,first,count,details}: Queued{block,rest}
    case Partition.Fork{left,right}: queued(~Block,left,queued(~Block,right,rest))

def dequeued(~Block: Type,~Fill: Data,~vacant: Fill -> Block,fill: Fill,first: U32,count: U32,queue: Queue<Block>) -> Queue<Block> & Partition.Tree<&1,Block,Unit>:
  match queue:
    case Queued{block,rest}: (rest,Partition.Leaf{block,first,count,Unit{}})
    case Empty{}: (Empty{},Partition.Leaf{vacant(fill),first,count,Unit{}})

def blocks_regrouped(~Block: Type,~Fill: Data,~vacant: Fill -> Block,+layout: Ranges,+fill: Fill,tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Block,Unit>:
  Pair.snd(Queue<Block>,Partition.Tree<&1,Block,Unit>,
    build(~Block,~Queue<Block>,~Fill,~dequeued(~Block,~Fill,~vacant),layout,fill,queued(~Block,tree,Empty{})))

def vacant_buffer(~Element: Data,fill: Element) -> Storage.Buffer<Element>:
  Allocation.allocate(~Element,0n,fill,0n)

# Tasks over adjacent pieces with one owner of a read-only resource per task:
# share gives each half of a split its own owner, and a task walks its pieces
# in order with the one owner it holds.
def task_walk(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Partition.Tree<&1,Output,Unit> & Resource,
  context: Context,first: U32,count: U32,state: Pieced(~Block,~Resource)) -> Partition.Tree<&1,Output,Unit>:
  (tree,resource) = state
  Pair.fst(Partition.Tree<&1,Output,Unit>,Resource,walk(~Block,~Output,~Resource,~Context,~leaf,context,tree,resource))

def task_kept(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Partition.Tree<&1,Output,Unit> & Resource,
  context: Context,layout: Ranges,state: Pieced(~Block,~Resource)) -> Partition.Tree<&1,Output,Unit>:
  (tree,resource) = state
  Pair.fst(Partition.Tree<&1,Output,Unit>,Resource,walk(~Block,~Output,~Resource,~Context,~leaf,context,tree,resource))

def grouped_with(~Block: Type,~Output: Type,~Resource: Type,~Context: Data,
  ~share: Resource -> Resource & Resource,
  ~leaf: Context -> U32 -> U32 -> Block -> Resource -> Partition.Tree<&1,Output,Unit> & Resource,
  layout: Ranges,+enabled: Bool,+context: Context,tree: Partition.Tree<&1,Block,Unit>,resource: Resource) -> Partition.Tree<&1,Output,Unit>:
  each_part(~Pieced(~Block,~Resource),~Output,~Unit,~Context,~Bool,~in_parallel,~piece_divide(~Block,~Resource,~Context,~share),
    ~task_walk(~Block,~Output,~Resource,~Context,~leaf),~task_kept(~Block,~Output,~Resource,~Context,~leaf),
    layout,enabled,enabled,context,(tree,resource))
