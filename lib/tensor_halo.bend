# The input pieces a task of a window operation holds. The tasks own the input
# pieces in order: a piece belongs to the first task whose reach holds its
# first cell and that the tasks before it do not reach. A task also holds a
# copy of the cells of its reach before its own pieces, its halo. Two serial
# walks run before the tasks: the first copies every halo from the input
# pieces, the second hands the pieces to their tasks. A piece no task reaches
# is dropped.
import Base
import ./storage_buffer.bend as Storage
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_tasks.bend as Tasks
import ./tensor_move.bend as Move
import ./traversal_partition.bend as Partition

# The range of input cells a task reads from the range of output cells it
# writes.
def Reach(~Context: Data) -> Type:
  @+context: Context -> @+first: U32 -> @+count: U32 -> Tasks.Span

type Reaching<-Element: Data,-Context: Data> is Data:
  Reaching{context: Context,grid: Grid.Grid,fill: Element}

# The first piece of a grid that starts at or after cell index.
def piece_from(+grid: Grid.Grid,+index: U32) -> U32:
  +piece = Grid.piece_of(grid,index)
  U32.min(Bool.pick(U32,U32.is_lt(Grid.start(grid,piece),index),(piece + 1 : U32),piece),Grid.pieces_of(grid))

# A task's halo, cells [first, first + count), and its pieces [low, high).
type Halo<-Element: Data> is Type:
  Halo{buffer: Storage.Buffer<Element>,first: U32,count: U32,low: U32,high: U32}

# The first walk reads the input pieces and knows where the tasks before
# reach to.
type Reading<-Element: Data> is Type:
  Reading{holding: Move.Holding<Element>,reached: U32}

def halo_leaf(~Element: Data,+task_first: U32,+task_count: U32,+first: U32,+count: U32,+low: U32,+high: U32,+reached: U32,
  copied: Move.Holding<Element> & Storage.Buffer<Element>) -> Reading<Element> & Partition.Tree<&1,Halo<Element>,Unit>:
  (holding,buffer) = copied
  (Reading{holding,reached},Partition.Leaf{Halo{buffer,first,count,low,high},task_first,task_count,Unit{}})

def halo_spanned(~Element: Data,+grid: Grid.Grid,+fill: Element,+task_first: U32,+task_count: U32,span: Tasks.Span,reading: Reading<Element>) ->
  Reading<Element> & Partition.Tree<&1,Halo<Element>,Unit>:
  Tasks.Span{+start,+length} = span
  Reading{holding,+reached} = reading
  +end = (start + length : U32)
  +low = piece_from(grid,U32.max(start,reached))
  +high = U32.max(low,piece_from(grid,end))
  +halo = (Bool.pick(U32,U32.is_lt(low,high),U32.min(Grid.start(grid,low),end),end) - start : U32)
  halo_leaf(~Element,task_first,task_count,start,halo,low,high,U32.max(reached,end),Move.copied(~Element,holding,start,halo,fill))

def halo_task(~Element: Data,~Context: Data,~reach: Reach(~Context),context: Reaching<Element,Context>,first: U32,count: U32,reading: Reading<Element>) ->
  Reading<Element> & Partition.Tree<&1,Halo<Element>,Unit>:
  Reaching{+own,+grid,+fill} = context
  +first = first
  +count = count
  halo_spanned(~Element,grid,fill,first,count,reach(own,first,count),reading)

# The pieces of a task: after the first ones of the stack are dropped, the
# next count taken, in the other order.
type Handed<-Element: Data> is Type:
  Handed{own: Move.Stack<Element>,rest: Move.Stack<Element>}

def skipped(~Element: Data,count: Nat,stack: Move.Stack<Element>) -> Move.Stack<Element>:
  match count stack:
    case 1n+rest Move.Held{buffer,first,pieces,below}: skipped(~Element,rest,below)
    case count stack: stack

def handed(~Element: Data,count: Nat,stack: Move.Stack<Element>,own: Move.Stack<Element>) -> Handed<Element>:
  match count stack:
    case 1n+rest Move.Held{buffer,+first,+pieces,below}: handed(~Element,rest,below,Move.Held{buffer,first,pieces,own})
    case count stack: Handed{own,stack}

# The second walk: the stack of input pieces not yet handed and the index of
# its first.
type Handing<-Element: Data> is Type:
  Handing{stack: Move.Stack<Element>,next: U32}

def window_of(~Element: Data,empty: Bool,halo: Storage.Buffer<Element>,+first: U32,+count: U32,own: Move.Stack<Element>) -> Move.Holding<Element>:
  match empty:
    case True{}: Move.Holding{Move.Bottom{},own}
    case False{}: Move.Holding{Move.Bottom{},Move.Held{halo,first,count,own}}

def window_handed(~Element: Data,+task_first: U32,+task_count: U32,halo: Storage.Buffer<Element>,+first: U32,+count: U32,+next: U32,handed: Handed<Element>) ->
  Partition.Tree<&1,Move.Holding<Element>,Unit> & Handing<Element>:
  Handed{own,rest} = handed
  (Partition.Leaf{window_of(~Element,U32.is_zero(count),halo,first,count,Move.onto_ahead(~Element,own,Move.Bottom{})),task_first,task_count,Unit{}},Handing{rest,next})

def window_leaf(~Element: Data,context: Unit,task_first: U32,task_count: U32,halo: Halo<Element>,handing: Handing<Element>) ->
  Partition.Tree<&1,Move.Holding<Element>,Unit> & Handing<Element>:
  Halo{buffer,+first,+count,+low,+high} = halo
  Handing{stack,+next} = handing
  +task_first = task_first
  +task_count = task_count
  +start = U32.max(low,next)
  +finish = U32.max(high,start)
  window_handed(~Element,task_first,task_count,buffer,first,count,finish,
    handed(~Element,U32.to_nat((finish - start : U32)),skipped(~Element,U32.to_nat((start - next : U32)),stack),Move.Bottom{}))

def halos_of(~Element: Data,result: Reading<Element> & Partition.Tree<&1,Halo<Element>,Unit>,+fill: Element) -> Partition.Tree<&1,Halo<Element>,Unit> & Move.Stack<Element>:
  (reading,halos) = result
  Reading{holding,reached} = reading
  (halos,Move.stacked(~Element,Move.released(~Element,holding,fill),Move.Bottom{}))

def windows_handed(~Element: Data,pair: Partition.Tree<&1,Halo<Element>,Unit> & Move.Stack<Element>) -> Partition.Tree<&1,Move.Holding<Element>,Unit>:
  (halos,stack) = pair
  Pair.fst(Partition.Tree<&1,Move.Holding<Element>,Unit>,Handing<Element>,
    Pieces.walk(~Halo<Element>,~Move.Holding<Element>,~Handing<Element>,~Unit,~window_leaf(~Element),Unit{},halos,Handing{stack,0}))

# Prepare: the pieces every task of tasks holds, its halo first. A task leaf
# holds them with the output range of the task.
def windows(~Element: Data,~Context: Data,~reach: Reach(~Context),+context: Context,+grid: Grid.Grid,+fill: Element,tasks: Pieces.Ranges,pieces: Pieces.Pieces(~Element)) ->
  Partition.Tree<&1,Move.Holding<Element>,Unit>:
  windows_handed(~Element,halos_of(~Element,
    Pieces.build(~Halo<Element>,~Reading<Element>,~Reaching<Element,Context>,~halo_task(~Element,~Context,~reach),tasks,Reaching{context,grid,fill},
      Reading{Move.held(~Element,pieces),0}),fill))
