# Folds of every line of the cells. A line holds length cells of a block,
# inner cells apart, and the inner lines of a block interleave in its length *
# inner cells. Line i is line i % inner of block i / inner. The state of line
# i begins from context i modulo the count of contexts, folds the cells of
# the line in order and turns once into its final. A block longer than a
# piece lies in the pieces of whole outer indices and carries the states of
# its lines from piece to piece in order: a task holds whole blocks and walks
# its pieces once.
import Base
import ./storage_buffer.bend as Storage
import ./traversal_partition.bend as Partition
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./tensor_tasks.bend as Tasks

# How the lines lie: length cells a line, inner cells apart.
type Line is Data:
  Line{length: U32,inner: U32}

def cells_of(line: Line) -> U32:
  Line{+length,+inner} = line
  (length * inner : U32)

def inner_of(line: Line) -> U32:
  Line{length,inner} = line
  inner

# The contexts of the lines and how they lie.
type Lines<-Context: Data> is Data:
  Lines{contexts: List<&2,Context>,line: Line}

def line_of(~Context: Data,lines: Lines<Context>) -> Line:
  Lines{contexts,line} = lines
  line

def contexts_of(~Context: Data,lines: Lines<Context>) -> List<&2,Context>:
  Lines{contexts,line} = lines
  contexts

# The states of lines in order. A state may own storage, so it never lies in a
# List, which every program shares.
type States<-State: Type> is Type:
  NoStates{}
  Carried{state: State,rest: States<State>}

# A pass that folds: column cells of the block in progress folded into
# starts, the contexts of the lines from the next block on, and the finals of
# the lines done, last first.
type Folding<-Context: Data,-State: Type,-Final: Data> is Type:
  Folding{column: U32,upcoming: List<&2,Context>,starts: States<State>,finals: List<&2,Final>}

def onto(~State: Type,folded: States<State>,states: States<State>) -> States<State>:
  match folded:
    case NoStates{}: states
    case Carried{state,rest}: onto(~State,rest,Carried{state,states})

def unchanged(~Value: Data,value: Value) -> Value:
  value

# The contexts after one taken: all again once they run out.
def refilled(~Context: Data,+all: List<&2,Context>,later: List<&2,Context>) -> List<&2,Context>:
  match later:
    case Nil{}: all
    case later: later

# A folded line turned into its final when its block ends here, otherwise
# carried, last first.
def placed(~Block: Type,~State: Type,~Final: Data,~turn: State -> Final,ends: Bool,folded: States<State>,finals: List<&2,Final>,pair: Block & State) ->
  Block & (States<State> & List<&2,Final>):
  match ends:
    case True{}:
      (block,state) = pair
      (block,(folded,turn(state) <> finals))
    case False{}:
      (block,state) = pair
      (block,(Carried{state,folded},finals))

def lines_folded(~Block: Type,~Context: Data,~State: Type,~Final: Data,+column: U32,upcoming: List<&2,Context>,states: States<State>,progress: Block & (States<State> & List<&2,Final>)) ->
  Block & Folding<Context,State,Final>:
  (block,lists) = progress
  (folded,finals) = lists
  (block,Folding{column,upcoming,onto(~State,folded,states),finals})

# The lines of the block in progress from line on, each folded over its cells
# among count cells from offset whose first lies column cells into the block.
# A line begins from the next context when the block begins here, otherwise
# from the head of states. next is the column after the segment.
def fold_lines(~Block: Type,~Context: Data,~State: Type,~Final: Data,~begin: Context -> State,
  ~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,remaining: Nat,begins: Bool,+ends: Bool,
  +line: U32,+inner: U32,+column: U32,+offset: U32,+count: U32,+next: U32,+all: List<&2,Context>,upcoming: List<&2,Context>,states: States<State>,
  progress: Block & (States<State> & List<&2,Final>)) -> Block & Folding<Context,State,Final>:
  match remaining begins upcoming states:
    case 1n+rest True{} Con{+context,later} states:
      (block,lists) = progress
      (folded,finals) = lists
      +start = ((line + inner - column % inner) % inner : U32)
      +cells = Bool.pick(U32,U32.is_lt(start,count),((count - start + inner - 1) / inner : U32),0)
      fold_lines(~Block,~Context,~State,~Final,~begin,~fold,~turn,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,refilled(~Context,all,later),states,
        placed(~Block,~State,~Final,~turn,ends,folded,finals,fold(U32.to_nat(cells),(offset + start : U32),inner,begin(context),block)))
    case 1n+rest False{} upcoming Carried{state,tail}:
      (block,lists) = progress
      (folded,finals) = lists
      +start = ((line + inner - column % inner) % inner : U32)
      +cells = Bool.pick(U32,U32.is_lt(start,count),((count - start + inner - 1) / inner : U32),0)
      fold_lines(~Block,~Context,~State,~Final,~begin,~fold,~turn,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,tail,
        placed(~Block,~State,~Final,~turn,ends,folded,finals,fold(U32.to_nat(cells),(offset + start : U32),inner,state,block)))
    case remaining begins upcoming states: lines_folded(~Block,~Context,~State,~Final,next,upcoming,states,progress)

# The remaining cells of a block from offset on, one segment of a block at a
# time. The lines of a block turn in the segment where it ends.
def fold_segments(~Block: Type,~Context: Data,~State: Type,~Final: Data,~begin: Context -> State,
  ~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,fuel: Nat,done: Bool,+lines: Lines<Context>,+offset: U32,
  +remaining: U32,progress: Block & Folding<Context,State,Final>) -> Block & Folding<Context,State,Final>:
  match fuel done:
    case 1n+rest False{}:
      Lines{+all,+line} = lines
      (block,folding) = progress
      Folding{+column,upcoming,starts,finals} = folding
      +count = U32.min(remaining,(cells_of(line) - column : U32))
      +ends = U32.is_eq((column + count : U32),cells_of(line))
      fold_segments(~Block,~Context,~State,~Final,~begin,~fold,~turn,rest,U32.is_eq(count,remaining),lines,(offset + count : U32),(remaining - count : U32),
        fold_lines(~Block,~Context,~State,~Final,~begin,~fold,~turn,U32.to_nat(inner_of(line)),U32.is_zero(column),ends,0,inner_of(line),column,offset,count,
          Bool.pick(U32,ends,0,(column + count : U32)),all,upcoming,starts,(block,(NoStates{},finals))))
    case fuel done: progress

# Segments of a block of count cells: a block ends in it at most once a block
# it holds, and one block goes on past it.
def segments(+count: U32,+cells: U32) -> Nat:
  U32.to_nat((count / U32.max(cells,1) + 2 : U32))

def fold_folded(~Block: Type,~Context: Data,~State: Type,~Final: Data,+first: U32,+count: U32,result: Block & Folding<Context,State,Final>) ->
  Partition.Tree<&1,Block,Unit> & Folding<Context,State,Final>:
  (block,folding) = result
  (Partition.Leaf{block,first,count,Unit{}},folding)

# A line of no cell folds nothing.
def fold_leaf(~Block: Type,~Context: Data,~State: Type,~Final: Data,~begin: Context -> State,
  ~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,lines: Lines<Context>,first: U32,count: U32,block: Block,
  folding: Folding<Context,State,Final>) -> Partition.Tree<&1,Block,Unit> & Folding<Context,State,Final>:
  +lines = lines
  +count = count
  +line = line_of(~Context,lines)
  fold_folded(~Block,~Context,~State,~Final,first,count,
    fold_segments(~Block,~Context,~State,~Final,~begin,~fold,~turn,segments(count,cells_of(line)),U32.is_zero(count) || U32.is_zero(cells_of(line)),lines,0,count,(block,folding)))

def finals_of(~Context: Data,~State: Type,~Final: Data,folding: Folding<Context,State,Final>) -> List<&2,Final>:
  Folding{column,upcoming,starts,finals} = folding
  List.reverse(&2,Final,finals)

# The contexts from the line of cell first on.
def upcoming_of(~Context: Data,+contexts: List<&2,Context>,+line: Line,+first: U32) -> List<&2,Context>:
  +kinds = U32.max(U32.from_nat(List.length(&2,Context,contexts)),1)
  List.drop(&2,Context,contexts,U32.to_nat((first / U32.max(cells_of(line),1) * inner_of(line) % kinds : U32)))

def folds_task(~Block: Type,~Context: Data,~State: Type,~Final: Data,~begin: Context -> State,
  ~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,lines: Lines<Context>,first: U32,count: U32,
  tree: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & Folding<Context,State,Final>,Unit>:
  +lines = lines
  +first = first
  +count = count
  Partition.Leaf{Pieces.walk(~Block,~Block,~Folding<Context,State,Final>,~Lines<Context>,~fold_leaf(~Block,~Context,~State,~Final,~begin,~fold,~turn),lines,tree,
    Folding{0,upcoming_of(~Context,contexts_of(~Context,lines),line_of(~Context,lines),first),NoStates{},[]}),first,count,Unit{}}

def task_finals(~Block: Type,~Context: Data,~State: Type,~Final: Data,result: Partition.Tree<&1,Block,Unit> & Folding<Context,State,Final>,done: List<&2,Final>) ->
  Partition.Tree<&1,Block,Unit> & List<&2,Final>:
  (tree,folding) = result
  (tree,List.append(&2,Final,finals_of(~Context,~State,~Final,folding),done))

def tasks_forked(~Block: Type,~Final: Data,right: Partition.Tree<&1,Block,Unit>,left: Partition.Tree<&1,Block,Unit> & List<&2,Final>) -> Partition.Tree<&1,Block,Unit> & List<&2,Final>:
  (left,done) = left
  (Partition.Fork{left,right},done)

# The blocks of every task and the finals of their lines, in order.
def tasks_unzipped(~Block: Type,~Context: Data,~State: Type,~Final: Data,tree: Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & Folding<Context,State,Final>,Unit>,
  done: List<&2,Final>) -> Partition.Tree<&1,Block,Unit> & List<&2,Final>:
  match tree:
    case Partition.Leaf{result,first,count,details}: task_finals(~Block,~Context,~State,~Final,result,done)
    case Partition.Fork{left,right}:
      Storage.bind_pair(~Partition.Tree<&1,Block,Unit>,~List<&2,Final>,~(Partition.Tree<&1,Block,Unit> & List<&2,Final>),tasks_unzipped(~Block,~Context,~State,~Final,right,done),
        right => done => tasks_forked(~Block,~Final,right,tasks_unzipped(~Block,~Context,~State,~Final,left,done)))

def folds_decided(~Block: Type,~Context: Data,~State: Type,~Final: Data,~begin: Context -> State,
  ~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,decision: Tasks.Decision,lines: Lines<Context>,
  pieces: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Block,Unit> & List<&2,Final>:
  Tasks.Decision{+parallel,tasks} = decision
  tasks_unzipped(~Block,~Context,~State,~Final,Pieces.subtrees(~Block,~(Partition.Tree<&1,Block,Unit> & Folding<Context,State,Final>),~Lines<Context>,
    ~folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),tasks,parallel,lines,pieces),[])

# The final of every line, in order: the state of a line begins from its
# context, fold folds its cells and turn ends it in its final. A block lies in
# the pieces of group outer indices of the grid. The pieces come back in the
# tree of those groups. rows and steps count one cell.
def folds(~Block: Type,~Fill: Data,~Context: Data,~State: Type,~Final: Data,~begin: Context -> State,
  ~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final,~vacant: Fill -> Block,+contexts: List<&2,Context>,
  +line: Line,+group: U32,+rows: F32,+steps: F32,+grid: Grid.Grid,+fill: Fill,+workers: U32,pieces: Partition.Tree<&1,Block,Unit>) -> Partition.Tree<&1,Block,Unit> & List<&2,Final>:
  +layout = Pieces.ranges_in(grid,group)
  folds_decided(~Block,~Context,~State,~Final,~begin,~fold,~turn,Tasks.decide_over(layout,Grid.count_of(grid),workers,cells_of(line),rows,steps),Lines{contexts,line},
    Pieces.blocks_regrouped(~Block,~Fill,~vacant,layout,fill,pieces))
