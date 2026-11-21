# The lines of a line chain, by the models of their blocks and the values of
# their states. A block a fold reads represents a model and a state shows a
# value: the fold of a block gives the represented model back beside the
# shown fold of the value, a line begins from the shown value of its context
# and turns into the final of its value. Every pass over represented blocks is
# then the same pass over their models, and the lines of whole blocks fold
# their cells in order.
import Base
import ../storage_buffer.bend as Storage
import ../tensor_line_chain.bend as LineChain
import ../tensor_pieces.bend as Pieces
import ../traversal_partition.bend as Partition
import ./traversal_partition_refinement.bend as PartitionLaws
import ./tensor_pieces_regrouped.bend as Regrouped
import ./row_major_index.bend as RowMajor
import ./nat_extrema.bend as Extrema
import ./u32_extrema.bend as NativeExtrema
import ./list_storage_readback.bend as ListReadback
import ./u32_comparison.bend as Comparison
import ./u32_division.bend as U32Division
import ./word_encoding.bend as WordEncoding
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_to_u32_bounds.bend as NatWords
import ./nat_division.bend as Quotients
import ./nat_division_order.bend as DivisionOrder
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./tensor_pieces_walk.bend as PiecesWalk
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid

# A fold of a represented block keeps it and shows the fold of the value.
def FoldSame(~Block: Type,~Model: Data,~State: Type,~Value: Data,~represent: Model -> Block,~shown: Value -> State,
  ~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value) -> Type:
  @+count: Nat -> @+position: U32 -> @+step: U32 -> @+initial: Value -> @+model: Model ->
    {fold(count,position,step,shown(initial),represent(model)) == (represent(model),shown(fold_value(count,position,step,initial,model))) : Block & State}

# A line begins from the shown value of its context.
def BeginSame(~Context: Data,~State: Type,~Value: Data,~shown: Value -> State,~begin: Context -> State,~begin_value: Context -> Value) -> Type:
  @+context: Context -> {begin(context) == shown(begin_value(context)) : State}

# A shown value turns into the final of the value.
def TurnSame(~State: Type,~Value: Data,~Final: Data,~shown: Value -> State,~turn: State -> Final,~turn_value: Value -> Final) -> Type:
  @+value: Value -> {turn(shown(value)) == turn_value(value) : Final}

# ---- The passes of one block ----

# A pass that folds, by the values of its states.
type Folded<-Context: Data,-Value: Data,-Final: Data> is Data:
  Folded{column: U32,upcoming: List<&2,Context>,starts: List<&2,Value>,finals: List<&2,Final>}

def shown_states(~State: Type,~Value: Data,~shown: Value -> State,values: List<&2,Value>) -> LineChain.States<State>:
  match values:
    case Nil{}: LineChain.NoStates{}
    case Con{value,rest}: LineChain.Carried{shown(value),shown_states(~State,~Value,~shown,rest)}

def shown_folding(~Context: Data,~State: Type,~Value: Data,~Final: Data,~shown: Value -> State,folded: Folded<Context,Value,Final>) -> LineChain.Folding<Context,State,Final>:
  Folded{column,upcoming,starts,finals} = folded
  LineChain.Folding{column,upcoming,shown_states(~State,~Value,~shown,starts),finals}

# Elements pushed onto elements, the last first.
def onto_values(~Element: Data,folded: List<&2,Element>,+elements: List<&2,Element>) -> List<&2,Element>:
  match folded:
    case Nil{}: elements
    case Con{+element,rest}: onto_values(~Element,rest,element <> elements)

def onto_shown(~State: Type,~Value: Data,~shown: Value -> State,+folded: List<&2,Value>,+states: List<&2,Value>) ->
  {LineChain.onto(~State,shown_states(~State,~Value,~shown,folded),shown_states(~State,~Value,~shown,states)) == shown_states(~State,~Value,~shown,onto_values(~Value,folded,states)) : LineChain.States<State>}:
  match folded:
    case Nil{}: {==}
    case Con{+value,+rest}: onto_shown(~State,~Value,~shown,rest,value <> states)

# A folded line: carried while its block goes on, turned when it ends.
def placed_folded(~Value: Data,ends: Bool,+folded: List<&2,Value>,+value: Value) -> List<&2,Value>:
  match ends:
    case True{}: folded
    case False{}: value <> folded

def placed_finals(~Value: Data,~Final: Data,~turn_value: Value -> Final,ends: Bool,+finals: List<&2,Final>,+value: Value) -> List<&2,Final>:
  match ends:
    case True{}: turn_value(value) <> finals
    case False{}: finals

def placed_same(~Block: Type,~Model: Data,~State: Type,~Value: Data,~Final: Data,~represent: Model -> Block,~shown: Value -> State,~turn: State -> Final,~turn_value: Value -> Final,
  ~turn_same: TurnSame(~State,~Value,~Final,~shown,~turn,~turn_value),ends: Bool,+folded: List<&2,Value>,+finals: List<&2,Final>,+value: Value,+model: Model) ->
  {LineChain.placed(~Block,~State,~Final,~turn,ends,shown_states(~State,~Value,~shown,folded),finals,(represent(model),shown(value))) ==
    (represent(model),(shown_states(~State,~Value,~shown,placed_folded(~Value,ends,folded,value)),placed_finals(~Value,~Final,~turn_value,ends,finals,value))) : Block & (LineChain.States<State> & List<&2,Final>)}:
  match ends:
    case True{}:
      %turn_same(value) :
        {LineChain.placed(~Block,~State,~Final,~turn,True{},shown_states(~State,~Value,~shown,folded),finals,(represent(model),shown(value))) == (represent(model),(shown_states(~State,~Value,~shown,folded),_ <> finals)) :
          Block & (LineChain.States<State> & List<&2,Final>)}
      {==}
    case False{}: {==}

# The lines of the block in progress from line on, as fold_lines folds them.
def lines_model(~Model: Data,~Context: Data,~Value: Data,~Final: Data,~begin_value: Context -> Value,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn_value: Value -> Final,remaining: Nat,begins: Bool,+ends: Bool,+line: U32,+inner: U32,+column: U32,+offset: U32,+count: U32,+next: U32,+all: List<&2,Context>,
  +upcoming: List<&2,Context>,+starts: List<&2,Value>,+folded: List<&2,Value>,+finals: List<&2,Final>,+model: Model) -> Folded<Context,Value,Final>:
  match remaining begins upcoming starts:
    case 1n+rest True{} Con{+context,+later} starts:
      +value = fold_value(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,begin_value(context),model)
      lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,LineChain.refilled(~Context,all,later),starts,
        placed_folded(~Value,ends,folded,value),placed_finals(~Value,~Final,~turn_value,ends,finals,value),model)
    case 1n+rest False{} upcoming Con{+state,+tail}:
      +value = fold_value(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,state,model)
      lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,tail,
        placed_folded(~Value,ends,folded,value),placed_finals(~Value,~Final,~turn_value,ends,finals,value),model)
    case remaining begins upcoming starts: Folded{next,upcoming,onto_values(~Value,folded,starts),finals}

def lines_same(~Block: Type,~Model: Data,~represent: Model -> Block,~Context: Data,~State: Type,~Value: Data,~shown: Value -> State,~Final: Data,~begin: Context -> State,~begin_value: Context -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn: State -> Final,~turn_value: Value -> Final,~fold_same: FoldSame(~Block,~Model,~State,~Value,~represent,~shown,~fold,~fold_value),~begin_same: BeginSame(~Context,~State,~Value,~shown,~begin,~begin_value),~turn_same: TurnSame(~State,~Value,~Final,~shown,~turn,~turn_value),remaining: Nat,begins: Bool,+ends: Bool,+line: U32,+inner: U32,+column: U32,+offset: U32,+count: U32,+next: U32,+all: List<&2,Context>,
  +upcoming: List<&2,Context>,+starts: List<&2,Value>,+folded: List<&2,Value>,+finals: List<&2,Final>,+model: Model) ->
  {LineChain.fold_lines(~Block,~Context,~State,~Final,~begin,~fold,~turn,remaining,begins,ends,line,inner,column,offset,count,next,all,upcoming,shown_states(~State,~Value,~shown,starts),(represent(model),(shown_states(~State,~Value,~shown,folded),finals))) ==
    (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,remaining,begins,ends,line,inner,column,offset,count,next,all,upcoming,starts,folded,finals,model))) : Block & LineChain.Folding<Context,State,Final>}:
  match remaining begins upcoming starts:
    case 1n+rest True{} Con{+context,+later} starts:
      +value = fold_value(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,begin_value(context),model)
      +refilled = LineChain.refilled(~Context,all,later)
      +placed = placed_folded(~Value,ends,folded,value)
      +turned = placed_finals(~Value,~Final,~turn_value,ends,finals,value)
      %Equal.sym(State,begin(context),shown(begin_value(context)),begin_same(context)) :
        {LineChain.fold_lines(~Block,~Context,~State,~Final,~begin,~fold,~turn,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,refilled,shown_states(~State,~Value,~shown,starts),
            LineChain.placed(~Block,~State,~Final,~turn,ends,shown_states(~State,~Value,~shown,folded),finals,fold(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,_,represent(model)))) ==
          (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,refilled,starts,placed,turned,model))) : Block & LineChain.Folding<Context,State,Final>}
      %Equal.sym(Block & State,fold(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,shown(begin_value(context)),represent(model)),(represent(model),shown(value)),
        fold_same(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,begin_value(context),model)) :
        {LineChain.fold_lines(~Block,~Context,~State,~Final,~begin,~fold,~turn,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,refilled,shown_states(~State,~Value,~shown,starts),
            LineChain.placed(~Block,~State,~Final,~turn,ends,shown_states(~State,~Value,~shown,folded),finals,_)) ==
          (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,refilled,starts,placed,turned,model))) : Block & LineChain.Folding<Context,State,Final>}
      %Equal.sym(Block & (LineChain.States<State> & List<&2,Final>),LineChain.placed(~Block,~State,~Final,~turn,ends,shown_states(~State,~Value,~shown,folded),finals,(represent(model),shown(value))),
        (represent(model),(shown_states(~State,~Value,~shown,placed),turned)),placed_same(~Block,~Model,~State,~Value,~Final,~represent,~shown,~turn,~turn_value,~turn_same,ends,folded,finals,value,model)) :
        {LineChain.fold_lines(~Block,~Context,~State,~Final,~begin,~fold,~turn,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,refilled,shown_states(~State,~Value,~shown,starts),_) ==
          (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,refilled,starts,placed,turned,model))) : Block & LineChain.Folding<Context,State,Final>}
      lines_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,refilled,starts,placed,turned,model)
    case 1n+rest False{} upcoming Con{+state,+tail}:
      +value = fold_value(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,state,model)
      +placed = placed_folded(~Value,ends,folded,value)
      +turned = placed_finals(~Value,~Final,~turn_value,ends,finals,value)
      %Equal.sym(Block & State,fold(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,shown(state),represent(model)),(represent(model),shown(value)),
        fold_same(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,state,model)) :
        {LineChain.fold_lines(~Block,~Context,~State,~Final,~begin,~fold,~turn,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,shown_states(~State,~Value,~shown,tail),
            LineChain.placed(~Block,~State,~Final,~turn,ends,shown_states(~State,~Value,~shown,folded),finals,_)) ==
          (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,tail,placed,turned,model))) : Block & LineChain.Folding<Context,State,Final>}
      %Equal.sym(Block & (LineChain.States<State> & List<&2,Final>),LineChain.placed(~Block,~State,~Final,~turn,ends,shown_states(~State,~Value,~shown,folded),finals,(represent(model),shown(value))),
        (represent(model),(shown_states(~State,~Value,~shown,placed),turned)),placed_same(~Block,~Model,~State,~Value,~Final,~represent,~shown,~turn,~turn_value,~turn_same,ends,folded,finals,value,model)) :
        {LineChain.fold_lines(~Block,~Context,~State,~Final,~begin,~fold,~turn,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,shown_states(~State,~Value,~shown,tail),_) ==
          (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,tail,placed,turned,model))) : Block & LineChain.Folding<Context,State,Final>}
      lines_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,tail,placed,turned,model)
    case 0n begins upcoming starts:
      %Equal.sym(LineChain.States<State>,LineChain.onto(~State,shown_states(~State,~Value,~shown,folded),shown_states(~State,~Value,~shown,starts)),shown_states(~State,~Value,~shown,onto_values(~Value,folded,starts)),onto_shown(~State,~Value,~shown,folded,starts)) :
        {(represent(model),LineChain.Folding{next,upcoming,_,finals}) == (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,Folded{next,upcoming,onto_values(~Value,folded,starts),finals})) : Block & LineChain.Folding<Context,State,Final>}
      {==}
    case 1n+rest True{} Nil{} starts:
      %Equal.sym(LineChain.States<State>,LineChain.onto(~State,shown_states(~State,~Value,~shown,folded),shown_states(~State,~Value,~shown,starts)),shown_states(~State,~Value,~shown,onto_values(~Value,folded,starts)),onto_shown(~State,~Value,~shown,folded,starts)) :
        {(represent(model),LineChain.Folding{next,[],_,finals}) == (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,Folded{next,[],onto_values(~Value,folded,starts),finals})) : Block & LineChain.Folding<Context,State,Final>}
      {==}
    case 1n+rest False{} upcoming Nil{}:
      %Equal.sym(LineChain.States<State>,LineChain.onto(~State,shown_states(~State,~Value,~shown,folded),LineChain.NoStates{}),shown_states(~State,~Value,~shown,onto_values(~Value,folded,[])),onto_shown(~State,~Value,~shown,folded,[])) :
        {(represent(model),LineChain.Folding{next,upcoming,_,finals}) == (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,Folded{next,upcoming,onto_values(~Value,folded,[]),finals})) : Block & LineChain.Folding<Context,State,Final>}
      {==}

# The segments of a block, as fold_segments folds them.
def segments_model(~Model: Data,~Context: Data,~Value: Data,~Final: Data,~begin_value: Context -> Value,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn_value: Value -> Final,fuel: Nat,done: Bool,+lines: LineChain.Lines<Context>,+offset: U32,+remaining: U32,+folded: Folded<Context,Value,Final>,+model: Model) -> Folded<Context,Value,Final>:
  match fuel done:
    case 1n+rest False{}:
      LineChain.Lines{+all,+line} = lines
      Folded{+column,+upcoming,+starts,+finals} = folded
      +count = U32.min(remaining,(LineChain.cells_of(line) - column : U32))
      +ends = U32.is_eq((column + count : U32),LineChain.cells_of(line))
      segments_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,U32.is_eq(count,remaining),lines,(offset + count : U32),(remaining - count : U32),
        lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,U32.to_nat(LineChain.inner_of(line)),U32.is_zero(column),ends,0,LineChain.inner_of(line),column,offset,count,
          Bool.pick(U32,ends,0,(column + count : U32)),all,upcoming,starts,[],finals,model),model)
    case fuel done: folded

def segments_same(~Block: Type,~Model: Data,~represent: Model -> Block,~Context: Data,~State: Type,~Value: Data,~shown: Value -> State,~Final: Data,~begin: Context -> State,~begin_value: Context -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn: State -> Final,~turn_value: Value -> Final,~fold_same: FoldSame(~Block,~Model,~State,~Value,~represent,~shown,~fold,~fold_value),~begin_same: BeginSame(~Context,~State,~Value,~shown,~begin,~begin_value),~turn_same: TurnSame(~State,~Value,~Final,~shown,~turn,~turn_value),fuel: Nat,done: Bool,+lines: LineChain.Lines<Context>,+offset: U32,+remaining: U32,+folded: Folded<Context,Value,Final>,+model: Model) ->
  {LineChain.fold_segments(~Block,~Context,~State,~Final,~begin,~fold,~turn,fuel,done,lines,offset,remaining,(represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,folded))) ==
    (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,segments_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,fuel,done,lines,offset,remaining,folded,model))) : Block & LineChain.Folding<Context,State,Final>}:
  match fuel done:
    case 1n+rest False{}:
      LineChain.Lines{+all,+line} = lines
      Folded{+column,+upcoming,+starts,+finals} = folded
      +count = U32.min(remaining,(LineChain.cells_of(line) - column : U32))
      +ends = U32.is_eq((column + count : U32),LineChain.cells_of(line))
      +next = Bool.pick(U32,ends,0,(column + count : U32))
      +lined = lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,U32.to_nat(LineChain.inner_of(line)),U32.is_zero(column),ends,0,LineChain.inner_of(line),column,offset,count,next,all,upcoming,starts,[],finals,model)
      %Equal.sym(Block & LineChain.Folding<Context,State,Final>,LineChain.fold_lines(~Block,~Context,~State,~Final,~begin,~fold,~turn,U32.to_nat(LineChain.inner_of(line)),U32.is_zero(column),ends,0,LineChain.inner_of(line),column,offset,count,next,all,upcoming,
          shown_states(~State,~Value,~shown,starts),(represent(model),(LineChain.NoStates{},finals))),(represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,lined)),
        lines_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,U32.to_nat(LineChain.inner_of(line)),U32.is_zero(column),ends,0,LineChain.inner_of(line),column,offset,count,next,all,upcoming,starts,[],finals,model)) :
        {LineChain.fold_segments(~Block,~Context,~State,~Final,~begin,~fold,~turn,rest,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),_) ==
          (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,segments_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),lined,model))) : Block & LineChain.Folding<Context,State,Final>}
      segments_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,rest,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),lined,model)
    case 0n done: {==}
    case 1n+rest True{}: {==}

# One block of a walk, as fold_leaf folds it.
def leaf_model(~Model: Data,~Context: Data,~Value: Data,~Final: Data,~begin_value: Context -> Value,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn_value: Value -> Final,+lines: LineChain.Lines<Context>,+count: U32,+model: Model,+folded: Folded<Context,Value,Final>) -> Folded<Context,Value,Final>:
  +cells = LineChain.cells_of(LineChain.line_of(~Context,lines))
  segments_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,LineChain.segments(count,cells),U32.is_zero(count) || U32.is_zero(cells),lines,0,count,folded,model)

def leaf_same(~Block: Type,~Model: Data,~represent: Model -> Block,~Context: Data,~State: Type,~Value: Data,~shown: Value -> State,~Final: Data,~begin: Context -> State,~begin_value: Context -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn: State -> Final,~turn_value: Value -> Final,~fold_same: FoldSame(~Block,~Model,~State,~Value,~represent,~shown,~fold,~fold_value),~begin_same: BeginSame(~Context,~State,~Value,~shown,~begin,~begin_value),~turn_same: TurnSame(~State,~Value,~Final,~shown,~turn,~turn_value),+lines: LineChain.Lines<Context>,+first: U32,+count: U32,+model: Model,+folded: Folded<Context,Value,Final>) ->
  {LineChain.fold_leaf(~Block,~Context,~State,~Final,~begin,~fold,~turn,lines,first,count,represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,folded)) ==
    (Partition.Leaf{represent(model),first,count,Unit{}},shown_folding(~Context,~State,~Value,~Final,~shown,leaf_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,count,model,folded))) : Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>}:
  +cells = LineChain.cells_of(LineChain.line_of(~Context,lines))
  %Equal.sym(Block & LineChain.Folding<Context,State,Final>,LineChain.fold_segments(~Block,~Context,~State,~Final,~begin,~fold,~turn,LineChain.segments(count,cells),U32.is_zero(count) || U32.is_zero(cells),lines,0,count,(represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,folded))),
    (represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,segments_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,LineChain.segments(count,cells),U32.is_zero(count) || U32.is_zero(cells),lines,0,count,folded,model))),
    segments_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,LineChain.segments(count,cells),U32.is_zero(count) || U32.is_zero(cells),lines,0,count,folded,model)) :
    {LineChain.fold_folded(~Block,~Context,~State,~Final,first,count,_) == (Partition.Leaf{represent(model),first,count,Unit{}},shown_folding(~Context,~State,~Value,~Final,~shown,leaf_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,count,model,folded))) : Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>}
  {==}

# A walk over blocks in order, as Pieces.walked walks them.
def walk_model(~Model: Data,~Context: Data,~Value: Data,~Final: Data,~begin_value: Context -> Value,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn_value: Value -> Final,+lines: LineChain.Lines<Context>,tree: Partition.Tree<&2,Model,Unit>,+folded: Folded<Context,Value,Final>) -> Folded<Context,Value,Final>:
  match tree:
    case Partition.Leaf{model,first,+count,details}: leaf_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,count,model,folded)
    case Partition.Fork{left,right}: walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,right,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,left,folded))

def walked_same(~Block: Type,~Model: Data,~represent: Model -> Block,~Context: Data,~State: Type,~Value: Data,~shown: Value -> State,~Final: Data,~begin: Context -> State,~begin_value: Context -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn: State -> Final,~turn_value: Value -> Final,~fold_same: FoldSame(~Block,~Model,~State,~Value,~represent,~shown,~fold,~fold_value),~begin_same: BeginSame(~Context,~State,~Value,~shown,~begin,~begin_value),~turn_same: TurnSame(~State,~Value,~Final,~shown,~turn,~turn_value),+lines: LineChain.Lines<Context>,+tree: Partition.Tree<&2,Model,Unit>,-lefts: Pieces.Lefts<Block>,+folded: Folded<Context,Value,Final>) ->
  {Pieces.walked(~Block,~Block,~LineChain.Folding<Context,State,Final>,~LineChain.Lines<Context>,~LineChain.fold_leaf(~Block,~Context,~State,~Final,~begin,~fold,~turn),lines,Regrouped.tree_of(~Block,~Model,~represent,tree),Pieces.Walking{lefts,shown_folding(~Context,~State,~Value,~Final,~shown,folded)}) ==
    Pieces.Walked{Regrouped.tree_of(~Block,~Model,~represent,tree),lefts,shown_folding(~Context,~State,~Value,~Final,~shown,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,tree,folded))} : Pieces.Walked<Block,LineChain.Folding<Context,State,Final>>}:
  match tree:
    case Partition.Leaf{+model,+first,+count,Unit{}}:
      %Equal.sym(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,LineChain.fold_leaf(~Block,~Context,~State,~Final,~begin,~fold,~turn,lines,first,count,represent(model),shown_folding(~Context,~State,~Value,~Final,~shown,folded)),
        (Partition.Leaf{represent(model),first,count,Unit{}},shown_folding(~Context,~State,~Value,~Final,~shown,leaf_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,count,model,folded))),leaf_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,lines,first,count,model,folded)) :
        {Pieces.leaf_walked(~Block,~LineChain.Folding<Context,State,Final>,lefts,_) == Pieces.Walked{Partition.Leaf{represent(model),first,count,Unit{}},lefts,shown_folding(~Context,~State,~Value,~Final,~shown,leaf_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,count,model,folded))} :
          Pieces.Walked<Block,LineChain.Folding<Context,State,Final>>}
      {==}
    case Partition.Fork{+left,+right}:
      +after = walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,left,folded)
      %Equal.sym(Pieces.Walked<Block,LineChain.Folding<Context,State,Final>>,Pieces.walked(~Block,~Block,~LineChain.Folding<Context,State,Final>,~LineChain.Lines<Context>,~LineChain.fold_leaf(~Block,~Context,~State,~Final,~begin,~fold,~turn),lines,Regrouped.tree_of(~Block,~Model,~represent,left),Pieces.Walking{lefts,shown_folding(~Context,~State,~Value,~Final,~shown,folded)}),
        Pieces.Walked{Regrouped.tree_of(~Block,~Model,~represent,left),lefts,shown_folding(~Context,~State,~Value,~Final,~shown,after)},walked_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,lines,left,lefts,folded)) :
        {Pieces.joined(~Block,~LineChain.Folding<Context,State,Final>,Pieces.walked(~Block,~Block,~LineChain.Folding<Context,State,Final>,~LineChain.Lines<Context>,~LineChain.fold_leaf(~Block,~Context,~State,~Final,~begin,~fold,~turn),lines,Regrouped.tree_of(~Block,~Model,~represent,right),Pieces.pushed(~Block,~LineChain.Folding<Context,State,Final>,_))) ==
          Pieces.Walked{Regrouped.tree_of(~Block,~Model,~represent,Partition.Fork{left,right}),lefts,shown_folding(~Context,~State,~Value,~Final,~shown,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,right,after))} : Pieces.Walked<Block,LineChain.Folding<Context,State,Final>>}
      %Equal.sym(Pieces.Walked<Block,LineChain.Folding<Context,State,Final>>,
        Pieces.walked(~Block,~Block,~LineChain.Folding<Context,State,Final>,~LineChain.Lines<Context>,~LineChain.fold_leaf(~Block,~Context,~State,~Final,~begin,~fold,~turn),lines,Regrouped.tree_of(~Block,~Model,~represent,right),Pieces.Walking{Pieces.Left{Regrouped.tree_of(~Block,~Model,~represent,left),lefts},shown_folding(~Context,~State,~Value,~Final,~shown,after)}),
        Pieces.Walked{Regrouped.tree_of(~Block,~Model,~represent,right),Pieces.Left{Regrouped.tree_of(~Block,~Model,~represent,left),lefts},shown_folding(~Context,~State,~Value,~Final,~shown,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,right,after))},walked_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,lines,right,Pieces.Left{Regrouped.tree_of(~Block,~Model,~represent,left),lefts},after)) :
        {Pieces.joined(~Block,~LineChain.Folding<Context,State,Final>,_) == Pieces.Walked{Regrouped.tree_of(~Block,~Model,~represent,Partition.Fork{left,right}),lefts,shown_folding(~Context,~State,~Value,~Final,~shown,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,right,after))} : Pieces.Walked<Block,LineChain.Folding<Context,State,Final>>}
      {==}

# ---- The tasks ----

type Walked<-Model: Data,-Context: Data,-Value: Data,-Final: Data> is Data:
  Walked{tree: Partition.Tree<&2,Model,Unit>,folded: Folded<Context,Value,Final>}

def walked_of(~Block: Type,~Model: Data,~represent: Model -> Block,~Context: Data,~State: Type,~Value: Data,~shown: Value -> State,~Final: Data,walked: Walked<Model,Context,Value,Final>) -> Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>:
  Walked{tree,folded} = walked
  (Regrouped.tree_of(~Block,~Model,~represent,tree),shown_folding(~Context,~State,~Value,~Final,~shown,folded))

# The fold a task begins: the contexts from the line of its first cell on.
def task_start(~Context: Data,~Value: Data,~Final: Data,+lines: LineChain.Lines<Context>,+first: U32) -> Folded<Context,Value,Final>:
  Folded{0,LineChain.upcoming_of(~Context,LineChain.contexts_of(~Context,lines),LineChain.line_of(~Context,lines),first),[],[]}

# A task over whole blocks from first.
def task_model(~Model: Data,~Context: Data,~Value: Data,~Final: Data,~begin_value: Context -> Value,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn_value: Value -> Final,+lines: LineChain.Lines<Context>,+first: U32,+count: U32,+tree: Partition.Tree<&2,Model,Unit>) -> Partition.Tree<&2,Walked<Model,Context,Value,Final>,Unit>:
  Partition.Leaf{Walked{tree,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,tree,task_start(~Context,~Value,~Final,lines,first))},first,count,Unit{}}

def task_same(~Block: Type,~Model: Data,~represent: Model -> Block,~Context: Data,~State: Type,~Value: Data,~shown: Value -> State,~Final: Data,~begin: Context -> State,~begin_value: Context -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn: State -> Final,~turn_value: Value -> Final,~fold_same: FoldSame(~Block,~Model,~State,~Value,~represent,~shown,~fold,~fold_value),~begin_same: BeginSame(~Context,~State,~Value,~shown,~begin,~begin_value),~turn_same: TurnSame(~State,~Value,~Final,~shown,~turn,~turn_value),+lines: LineChain.Lines<Context>,+first: U32,+count: U32,+tree: Partition.Tree<&2,Model,Unit>) ->
  {LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn,lines,first,count,Regrouped.tree_of(~Block,~Model,~represent,tree)) == PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),task_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,first,count,tree)) : Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>}:
  +start = task_start(~Context,~Value,~Final,lines,first)
  %Equal.sym(Pieces.Walked<Block,LineChain.Folding<Context,State,Final>>,Pieces.walked(~Block,~Block,~LineChain.Folding<Context,State,Final>,~LineChain.Lines<Context>,~LineChain.fold_leaf(~Block,~Context,~State,~Final,~begin,~fold,~turn),lines,Regrouped.tree_of(~Block,~Model,~represent,tree),Pieces.Walking{Pieces.NoLefts{},shown_folding(~Context,~State,~Value,~Final,~shown,start)}),
    Pieces.Walked{Regrouped.tree_of(~Block,~Model,~represent,tree),Pieces.NoLefts{},shown_folding(~Context,~State,~Value,~Final,~shown,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,tree,start))},walked_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,lines,tree,Pieces.NoLefts{},start)) :
    {Partition.Leaf{Pieces.walk_result(~Block,~LineChain.Folding<Context,State,Final>,_),first,count,Unit{}} == PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),task_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,first,count,tree)) : Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>}
  {==}

# The tasks of ranges over a tree of blocks: a split whose tree does not fork
# stays one task over the bounds of the split.
def tasks_model(~Model: Data,~Context: Data,~Value: Data,~Final: Data,~begin_value: Context -> Value,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn_value: Value -> Final,+layout: Pieces.Ranges,+lines: LineChain.Lines<Context>,+tree: Partition.Tree<&2,Model,Unit>) -> Partition.Tree<&2,Walked<Model,Context,Value,Final>,Unit>:
  match layout:
    case Pieces.Range{+first,+count}: task_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,first,count,tree)
    case Pieces.Split{+left,+right}:
      match tree:
        case Partition.Leaf{+model,+first,+count,+details}:
          task_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,Pieces.first(Pieces.Split{left,right}),(Pieces.last(Pieces.Split{left,right}) - Pieces.first(Pieces.Split{left,right}) : U32),
            Partition.Leaf{model,first,count,details})
        case Partition.Fork{+low,+high}: Partition.Fork{tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,left,lines,low),tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,right,lines,high)}

def tasks_same(~Block: Type,~Model: Data,~represent: Model -> Block,~Context: Data,~State: Type,~Value: Data,~shown: Value -> State,~Final: Data,~begin: Context -> State,~begin_value: Context -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn: State -> Final,~turn_value: Value -> Final,~fold_same: FoldSame(~Block,~Model,~State,~Value,~represent,~shown,~fold,~fold_value),~begin_same: BeginSame(~Context,~State,~Value,~shown,~begin,~begin_value),~turn_same: TurnSame(~State,~Value,~Final,~shown,~turn,~turn_value),+layout: Pieces.Ranges,+mode: Bool,+enabled: Bool,+lines: LineChain.Lines<Context>,+tree: Partition.Tree<&2,Model,Unit>) ->
  {Pieces.each_part(~Partition.Tree<&1,Block,Unit>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~LineChain.Lines<Context>,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~LineChain.Lines<Context>),~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),~Pieces.subtree_kept(~Block,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~LineChain.Lines<Context>,~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn)),layout,mode,enabled,lines,Regrouped.tree_of(~Block,~Model,~represent,tree)) == PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,layout,lines,tree)) : Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>}:
  match layout:
    case Pieces.Range{+first,+count}: task_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,lines,first,count,tree)
    case Pieces.Split{+left,+right}:
      match tree:
        case Partition.Leaf{+model,+first,+count,Unit{}}:
          %Equal.sym(Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>,Pieces.each_part(~Partition.Tree<&1,Block,Unit>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~LineChain.Lines<Context>,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~LineChain.Lines<Context>),~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),~Pieces.subtree_kept(~Block,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~LineChain.Lines<Context>,~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn)),Pieces.Split{left,right},mode,enabled,lines,Regrouped.tree_of(~Block,~Model,~represent,Partition.Leaf{model,first,count,Unit{}})),
            PiecesWalk.sequential(~Partition.Tree<&1,Block,Unit>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~LineChain.Lines<Context>,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~LineChain.Lines<Context>),~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),~Pieces.subtree_kept(~Block,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~LineChain.Lines<Context>,~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn)),left,right,enabled,lines,Pieces.group_divide(~Block,~LineChain.Lines<Context>,lines,left,right,Regrouped.tree_of(~Block,~Model,~represent,Partition.Leaf{model,first,count,Unit{}}))),
            PiecesWalk.fork(~Partition.Tree<&1,Block,Unit>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~LineChain.Lines<Context>,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~LineChain.Lines<Context>),~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),~Pieces.subtree_kept(~Block,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~LineChain.Lines<Context>,~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn)),left,right,mode,enabled,lines,Regrouped.tree_of(~Block,~Model,~represent,Partition.Leaf{model,first,count,Unit{}}))) :
            {_ == PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,Pieces.Split{left,right},lines,Partition.Leaf{model,first,count,Unit{}})) : Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>}
          task_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,lines,Pieces.first(Pieces.Split{left,right}),(Pieces.last(Pieces.Split{left,right}) - Pieces.first(Pieces.Split{left,right}) : U32),
            Partition.Leaf{model,first,count,Unit{}})
        case Partition.Fork{+low,+high}:
          %Equal.sym(Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>,Pieces.each_part(~Partition.Tree<&1,Block,Unit>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~LineChain.Lines<Context>,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~LineChain.Lines<Context>),~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),~Pieces.subtree_kept(~Block,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~LineChain.Lines<Context>,~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn)),Pieces.Split{left,right},mode,enabled,lines,Regrouped.tree_of(~Block,~Model,~represent,Partition.Fork{low,high})),
            PiecesWalk.sequential(~Partition.Tree<&1,Block,Unit>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~LineChain.Lines<Context>,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~LineChain.Lines<Context>),~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),~Pieces.subtree_kept(~Block,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~LineChain.Lines<Context>,~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn)),left,right,enabled,lines,Pieces.group_divide(~Block,~LineChain.Lines<Context>,lines,left,right,Regrouped.tree_of(~Block,~Model,~represent,Partition.Fork{low,high}))),
            PiecesWalk.fork(~Partition.Tree<&1,Block,Unit>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~LineChain.Lines<Context>,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~LineChain.Lines<Context>),~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),~Pieces.subtree_kept(~Block,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~LineChain.Lines<Context>,~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn)),left,right,mode,enabled,lines,Regrouped.tree_of(~Block,~Model,~represent,Partition.Fork{low,high}))) :
            {_ == PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,Pieces.Split{left,right},lines,Partition.Fork{low,high})) : Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>}
          %Equal.sym(Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>,Pieces.each_part(~Partition.Tree<&1,Block,Unit>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~LineChain.Lines<Context>,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~LineChain.Lines<Context>),~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),~Pieces.subtree_kept(~Block,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~LineChain.Lines<Context>,~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn)),left,enabled,enabled,lines,Regrouped.tree_of(~Block,~Model,~represent,low)),PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,left,lines,low)),tasks_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,left,enabled,enabled,lines,low)) :
            {Partition.Fork{_,Pieces.each_part(~Partition.Tree<&1,Block,Unit>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~LineChain.Lines<Context>,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~LineChain.Lines<Context>),~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),~Pieces.subtree_kept(~Block,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~LineChain.Lines<Context>,~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn)),right,enabled,enabled,lines,Regrouped.tree_of(~Block,~Model,~represent,high))} == PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,Pieces.Split{left,right},lines,Partition.Fork{low,high})) : Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>}
          %Equal.sym(Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>,Pieces.each_part(~Partition.Tree<&1,Block,Unit>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~LineChain.Lines<Context>,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~LineChain.Lines<Context>),~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),~Pieces.subtree_kept(~Block,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~LineChain.Lines<Context>,~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn)),right,enabled,enabled,lines,Regrouped.tree_of(~Block,~Model,~represent,high)),PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,right,lines,high)),tasks_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,right,enabled,enabled,lines,high)) :
            {Partition.Fork{PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,left,lines,low)),_} == PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,Pieces.Split{left,right},lines,Partition.Fork{low,high})) :
              Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>}
          {==}

# ---- The finals of every task ----

# The blocks of the tasks back in one tree, and the finals of their lines in
# order before done.
type Unzipped<-Model: Data,-Final: Data> is Data:
  Unzipped{tree: Partition.Tree<&2,Model,Unit>,finals: List<&2,Final>}

def unzipped_tree(~Model: Data,~Final: Data,unzipped: Unzipped<Model,Final>) -> Partition.Tree<&2,Model,Unit>:
  Unzipped{tree,finals} = unzipped
  tree

def unzipped_finals(~Model: Data,~Final: Data,unzipped: Unzipped<Model,Final>) -> List<&2,Final>:
  Unzipped{tree,finals} = unzipped
  finals

# The finals of a fold in order.
def folded_finals(~Context: Data,~Value: Data,~Final: Data,folded: Folded<Context,Value,Final>) -> List<&2,Final>:
  Folded{column,upcoming,starts,finals} = folded
  List.reverse(&2,Final,finals)

def unzipped_model(~Model: Data,~Context: Data,~Value: Data,~Final: Data,tree: Partition.Tree<&2,Walked<Model,Context,Value,Final>,Unit>,+done: List<&2,Final>) -> Unzipped<Model,Final>:
  match tree:
    case Partition.Leaf{walked,first,count,details}:
      match walked:
        case Walked{blocks,folded}: Unzipped{blocks,List.append(&2,Final,folded_finals(~Context,~Value,~Final,folded),done)}
    case Partition.Fork{left,right}:
      +high = unzipped_model(~Model,~Context,~Value,~Final,right,done)
      +low = unzipped_model(~Model,~Context,~Value,~Final,left,unzipped_finals(~Model,~Final,high))
      Unzipped{Partition.Fork{unzipped_tree(~Model,~Final,low),unzipped_tree(~Model,~Final,high)},unzipped_finals(~Model,~Final,low)}

def unzipped_pair(~Block: Type,~Model: Data,~represent: Model -> Block,~Final: Data,+unzipped: Unzipped<Model,Final>) -> Partition.Tree<&1,Block,Unit> & List<&2,Final>:
  (Regrouped.tree_of(~Block,~Model,~represent,unzipped_tree(~Model,~Final,unzipped)),unzipped_finals(~Model,~Final,unzipped))

def unzipped_same(~Block: Type,~Model: Data,~represent: Model -> Block,~Context: Data,~State: Type,~Value: Data,~shown: Value -> State,~Final: Data,
  +tree: Partition.Tree<&2,Walked<Model,Context,Value,Final>,Unit>,+done: List<&2,Final>) ->
  {LineChain.tasks_unzipped(~Block,~Context,~State,~Final,PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),tree),done) ==
    unzipped_pair(~Block,~Model,~represent,~Final,unzipped_model(~Model,~Context,~Value,~Final,tree,done)) : Partition.Tree<&1,Block,Unit> & List<&2,Final>}:
  match tree:
    case Partition.Leaf{+walked,+first,+count,+details}:
      match walked:
        case Walked{+blocks,+folded}:
          match folded:
            case Folded{+column,+upcoming,+starts,+finals}: {==}
    case Partition.Fork{+left,+right}:
      +high = unzipped_model(~Model,~Context,~Value,~Final,right,done)
      +low = unzipped_model(~Model,~Context,~Value,~Final,left,unzipped_finals(~Model,~Final,high))
      %Equal.sym(Partition.Tree<&1,Block,Unit> & List<&2,Final>,LineChain.tasks_unzipped(~Block,~Context,~State,~Final,PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),right),done),
        unzipped_pair(~Block,~Model,~represent,~Final,high),unzipped_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,right,done)) :
        {Storage.bind_pair(~Partition.Tree<&1,Block,Unit>,~List<&2,Final>,~(Partition.Tree<&1,Block,Unit> & List<&2,Final>),_,
            right => done => LineChain.tasks_forked(~Block,~Final,right,LineChain.tasks_unzipped(~Block,~Context,~State,~Final,PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),left),done))) ==
          unzipped_pair(~Block,~Model,~represent,~Final,unzipped_model(~Model,~Context,~Value,~Final,Partition.Fork{left,right},done)) : Partition.Tree<&1,Block,Unit> & List<&2,Final>}
      %Equal.sym(Partition.Tree<&1,Block,Unit> & List<&2,Final>,LineChain.tasks_unzipped(~Block,~Context,~State,~Final,PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),left),unzipped_finals(~Model,~Final,high)),
        unzipped_pair(~Block,~Model,~represent,~Final,low),unzipped_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,left,unzipped_finals(~Model,~Final,high))) :
        {LineChain.tasks_forked(~Block,~Final,Regrouped.tree_of(~Block,~Model,~represent,unzipped_tree(~Model,~Final,high)),_) ==
          unzipped_pair(~Block,~Model,~represent,~Final,unzipped_model(~Model,~Context,~Value,~Final,Partition.Fork{left,right},done)) : Partition.Tree<&1,Block,Unit> & List<&2,Final>}
      {==}

# ---- Folds over a grid ----

def decided_parallel(decision: Tasks.Decision) -> Bool:
  Tasks.Decision{parallel,tasks} = decision
  parallel

def decided_tasks(decision: Tasks.Decision) -> Pieces.Ranges:
  Tasks.Decision{parallel,tasks} = decision
  tasks

def decision_eta(+decision: Tasks.Decision) -> {decision == Tasks.Decision{decided_parallel(decision),decided_tasks(decision)} : Tasks.Decision}:
  Tasks.Decision{parallel,tasks} = decision
  {==}

# The lines of the blocks of a grid, every task over its own blocks, as
# LineChain.folds folds them.
def folds_model(~Model: Data,~Context: Data,~Value: Data,~Final: Data,~begin_value: Context -> Value,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn_value: Value -> Final,~Fill: Data,~vacant_model: Fill -> Model,+contexts: List<&2,Context>,+line: LineChain.Line,+group: U32,+bytes: F32,+operations: F32,
  +grid: Grid.Grid,+fill: Fill,+workers: U32,+tree: Partition.Tree<&2,Model,Unit>) -> Unzipped<Model,Final>:
  +layout = Pieces.ranges_in(grid,group)
  +decision = Tasks.decide_over(layout,Grid.count_of(grid),workers,LineChain.cells_of(line),bytes,operations)
  unzipped_model(~Model,~Context,~Value,~Final,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,decided_tasks(decision),LineChain.Lines{contexts,line},Regrouped.regrouped(~Model,~Fill,~vacant_model,layout,fill,tree)),[])

def folds_same(~Block: Type,~Model: Data,~represent: Model -> Block,~Context: Data,~State: Type,~Value: Data,~shown: Value -> State,~Final: Data,~begin: Context -> State,~begin_value: Context -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn: State -> Final,~turn_value: Value -> Final,~fold_same: FoldSame(~Block,~Model,~State,~Value,~represent,~shown,~fold,~fold_value),~begin_same: BeginSame(~Context,~State,~Value,~shown,~begin,~begin_value),~turn_same: TurnSame(~State,~Value,~Final,~shown,~turn,~turn_value),~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~vacant_same: @+fill: Fill -> {vacant(fill) == represent(vacant_model(fill)) : Block},
  +contexts: List<&2,Context>,+line: LineChain.Line,+group: U32,+bytes: F32,+operations: F32,+grid: Grid.Grid,+fill: Fill,+workers: U32,+tree: Partition.Tree<&2,Model,Unit>) ->
  {LineChain.folds(~Block,~Fill,~Context,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,Regrouped.tree_of(~Block,~Model,~represent,tree)) ==
    unzipped_pair(~Block,~Model,~represent,~Final,folds_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,~Fill,~vacant_model,contexts,line,group,bytes,operations,grid,fill,workers,tree)) :
    Partition.Tree<&1,Block,Unit> & List<&2,Final>}:
  +layout = Pieces.ranges_in(grid,group)
  +decision = Tasks.decide_over(layout,Grid.count_of(grid),workers,LineChain.cells_of(line),bytes,operations)
  +lines = {LineChain.Lines{contexts,line} : LineChain.Lines<Context>}
  +regrouped = Regrouped.regrouped(~Model,~Fill,~vacant_model,layout,fill,tree)
  +work = tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,decided_tasks(decision),lines,regrouped)
  %Equal.sym(Partition.Tree<&1,Block,Unit>,Pieces.blocks_regrouped(~Block,~Fill,~vacant,layout,fill,Regrouped.tree_of(~Block,~Model,~represent,tree)),Regrouped.tree_of(~Block,~Model,~represent,regrouped),
    Regrouped.regrouped_same(~Block,~Model,~Fill,~vacant,~represent,~vacant_model,~vacant_same,layout,fill,tree)) :
    {LineChain.folds_decided(~Block,~Context,~State,~Final,~begin,~fold,~turn,decision,lines,_) == unzipped_pair(~Block,~Model,~represent,~Final,unzipped_model(~Model,~Context,~Value,~Final,work,[])) :
      Partition.Tree<&1,Block,Unit> & List<&2,Final>}
  %Equal.sym(Tasks.Decision,decision,Tasks.Decision{decided_parallel(decision),decided_tasks(decision)},decision_eta(decision)) :
    {LineChain.folds_decided(~Block,~Context,~State,~Final,~begin,~fold,~turn,_,lines,Regrouped.tree_of(~Block,~Model,~represent,regrouped)) == unzipped_pair(~Block,~Model,~represent,~Final,unzipped_model(~Model,~Context,~Value,~Final,work,[])) :
      Partition.Tree<&1,Block,Unit> & List<&2,Final>}
  +enabled = decided_parallel(decision)
  %Equal.sym(Partition.Tree<&1,Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>,Unit>,Pieces.each_part(~Partition.Tree<&1,Block,Unit>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~LineChain.Lines<Context>,~Bool,~Pieces.in_parallel,~Pieces.group_divide(~Block,~LineChain.Lines<Context>),~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn),~Pieces.subtree_kept(~Block,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~LineChain.Lines<Context>,~LineChain.folds_task(~Block,~Context,~State,~Final,~begin,~fold,~turn)),decided_tasks(decision),enabled,enabled,lines,Regrouped.tree_of(~Block,~Model,~represent,regrouped)),PartitionLaws.carry(~&2,~&1,~Walked<Model,Context,Value,Final>,~(Partition.Tree<&1,Block,Unit> & LineChain.Folding<Context,State,Final>),~Unit,~walked_of(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final),work),
    tasks_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~fold_same,~begin_same,~turn_same,decided_tasks(decision),enabled,enabled,lines,regrouped)) :
    {LineChain.tasks_unzipped(~Block,~Context,~State,~Final,_,[]) == unzipped_pair(~Block,~Model,~represent,~Final,unzipped_model(~Model,~Context,~Value,~Final,work,[])) :
      Partition.Tree<&1,Block,Unit> & List<&2,Final>}
  unzipped_same(~Block,~Model,~represent,~Context,~State,~Value,~shown,~Final,work,[])

# The tasks of a walk keep the tree of its blocks.
def tasks_kept(~Model: Data,~Context: Data,~Value: Data,~Final: Data,~begin_value: Context -> Value,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn_value: Value -> Final,+lines: LineChain.Lines<Context>,+layout: Pieces.Ranges,+tree: Partition.Tree<&2,Model,Unit>,+done: List<&2,Final>) ->
  {unzipped_tree(~Model,~Final,unzipped_model(~Model,~Context,~Value,~Final,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,layout,lines,tree),done)) == tree : Partition.Tree<&2,Model,Unit>}:
  match layout:
    case Pieces.Range{+first,+count}: {==}
    case Pieces.Split{+left,+right}:
      match tree:
        case Partition.Leaf{+model,+first,+count,+details}: {==}
        case Partition.Fork{+low,+high}:
          +high_unzipped = unzipped_model(~Model,~Context,~Value,~Final,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,right,lines,high),done)
          %Equal.sym(Partition.Tree<&2,Model,Unit>,
            unzipped_tree(~Model,~Final,unzipped_model(~Model,~Context,~Value,~Final,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,left,lines,low),unzipped_finals(~Model,~Final,high_unzipped))),low,
            tasks_kept(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,left,low,unzipped_finals(~Model,~Final,high_unzipped))) :
            {Partition.Fork{_,unzipped_tree(~Model,~Final,high_unzipped)} == Partition.Fork{low,high} : Partition.Tree<&2,Model,Unit>}
          %Equal.sym(Partition.Tree<&2,Model,Unit>,unzipped_tree(~Model,~Final,high_unzipped),high,tasks_kept(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,right,high,done)) :
            {Partition.Fork{low,_} == Partition.Fork{low,high} : Partition.Tree<&2,Model,Unit>}
          {==}

# A fold of lines of a grid whose blocks and finals are those of the model of
# its values.
def ChainSame(~Block: Type,~Model: Data,~represent: Model -> Block,~Context: Data,~State: Type,~Value: Data,~Final: Data,~begin: Context -> State,~begin_value: Context -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model) -> Type:
  @+contexts: List<&2,Context> -> @+line: LineChain.Line -> @+group: U32 -> @+bytes: F32 -> @+operations: F32 -> @+grid: Grid.Grid -> @+fill: Fill -> @+workers: U32 -> @+tree: Partition.Tree<&2,Model,Unit> ->
    {LineChain.folds(~Block,~Fill,~Context,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,Regrouped.tree_of(~Block,~Model,~represent,tree)) == unzipped_pair(~Block,~Model,~represent,~Final,folds_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,~Fill,~vacant_model,contexts,line,group,bytes,operations,grid,fill,workers,tree)) : Partition.Tree<&1,Block,Unit> & List<&2,Final>}

# A fold of lines hands back its blocks regrouped into the ranges of its
# tasks.
def folds_kept(~Block: Type,~Model: Data,~represent: Model -> Block,~Context: Data,~State: Type,~Value: Data,~Final: Data,~begin: Context -> State,~begin_value: Context -> Value,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn: State -> Final,~turn_value: Value -> Final,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,~chain_same: ChainSame(~Block,~Model,~represent,~Context,~State,~Value,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~Fill,~vacant,~vacant_model),
  +contexts: List<&2,Context>,+line: LineChain.Line,+group: U32,+bytes: F32,+operations: F32,+grid: Grid.Grid,+fill: Fill,+workers: U32,+tree: Partition.Tree<&2,Model,Unit>) ->
  {Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Final>,LineChain.folds(~Block,~Fill,~Context,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,
      Regrouped.tree_of(~Block,~Model,~represent,tree))) ==
    Regrouped.tree_of(~Block,~Model,~represent,Regrouped.regrouped(~Model,~Fill,~vacant_model,Pieces.ranges_in(grid,group),fill,tree)) : Partition.Tree<&1,Block,Unit>}:
  +layout = Pieces.ranges_in(grid,group)
  +decision = Tasks.decide_over(layout,Grid.count_of(grid),workers,LineChain.cells_of(line),bytes,operations)
  +regrouped = Regrouped.regrouped(~Model,~Fill,~vacant_model,layout,fill,tree)
  %Equal.sym(Partition.Tree<&1,Block,Unit> & List<&2,Final>,
    LineChain.folds(~Block,~Fill,~Context,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,Regrouped.tree_of(~Block,~Model,~represent,tree)),
    unzipped_pair(~Block,~Model,~represent,~Final,folds_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,~Fill,~vacant_model,contexts,line,group,bytes,operations,grid,fill,workers,tree)),
    chain_same(contexts,line,group,bytes,operations,grid,fill,workers,tree)) :
    {Pair.fst(Partition.Tree<&1,Block,Unit>,List<&2,Final>,_) == Regrouped.tree_of(~Block,~Model,~represent,regrouped) : Partition.Tree<&1,Block,Unit>}
  %Equal.sym(Partition.Tree<&2,Model,Unit>,
    unzipped_tree(~Model,~Final,unzipped_model(~Model,~Context,~Value,~Final,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,decided_tasks(decision),LineChain.Lines{contexts,line},regrouped),[])),regrouped,
    tasks_kept(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,LineChain.Lines{contexts,line},decided_tasks(decision),regrouped,[])) :
    {Regrouped.tree_of(~Block,~Model,~represent,_) == Regrouped.tree_of(~Block,~Model,~represent,regrouped) : Partition.Tree<&1,Block,Unit>}
  {==}

# ---- Folds by a view of their values ----

# A view of values that the begins, the folds and the turns of two models of
# the values agree on.
def BeginView(~Context: Data,~Value: Data,~View: Data,~view: Value -> View,~begin_value: Context -> Value,~begin_view: Context -> View) -> Type:
  @+context: Context -> {view(begin_value(context)) == begin_view(context) : View}

def FoldView(~Model: Data,~Value: Data,~View: Data,~view: Value -> View,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~fold_view: Nat -> U32 -> U32 -> View -> Model -> View) -> Type:
  @+count: Nat -> @+position: U32 -> @+step: U32 -> @+value: Value -> @+model: Model -> {view(fold_value(count,position,step,value,model)) == fold_view(count,position,step,view(value),model) : View}

def TurnView(~Value: Data,~View: Data,~Final: Data,~view: Value -> View,~turn_value: Value -> Final,~turn_view: View -> Final) -> Type:
  @+value: Value -> {turn_value(value) == turn_view(view(value)) : Final}

# The values a view takes, in order.
def viewed(~Value: Data,~View: Data,~view: Value -> View,values: List<&2,Value>) -> List<&2,View>:
  match values:
    case Nil{}: []
    case Con{value,rest}: view(value) <> viewed(~Value,~View,~view,rest)

def viewed_folded(~Context: Data,~Value: Data,~View: Data,~Final: Data,~view: Value -> View,folded: Folded<Context,Value,Final>) -> Folded<Context,View,Final>:
  Folded{column,upcoming,starts,finals} = folded
  Folded{column,upcoming,viewed(~Value,~View,~view,starts),finals}

def viewed_walked(~Model: Data,~Context: Data,~Value: Data,~View: Data,~Final: Data,~view: Value -> View,walked: Walked<Model,Context,Value,Final>) -> Walked<Model,Context,View,Final>:
  Walked{tree,folded} = walked
  Walked{tree,viewed_folded(~Context,~Value,~View,~Final,~view,folded)}

def viewed_tasks(~Model: Data,~Context: Data,~Value: Data,~View: Data,~Final: Data,~view: Value -> View,tree: Partition.Tree<&2,Walked<Model,Context,Value,Final>,Unit>) -> Partition.Tree<&2,Walked<Model,Context,View,Final>,Unit>:
  match tree:
    case Partition.Leaf{walked,first,count,details}: Partition.Leaf{viewed_walked(~Model,~Context,~Value,~View,~Final,~view,walked),first,count,details}
    case Partition.Fork{left,right}: Partition.Fork{viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,left),viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,right)}

def onto_view(~Value: Data,~View: Data,~view: Value -> View,folded: List<&2,Value>,+states: List<&2,Value>) ->
  {viewed(~Value,~View,~view,onto_values(~Value,folded,states)) == onto_values(~View,viewed(~Value,~View,~view,folded),viewed(~Value,~View,~view,states)) : List<&2,View>}:
  match folded:
    case Nil{}: {==}
    case Con{+value,+rest}: onto_view(~Value,~View,~view,rest,value <> states)

def placed_folded_view(~Value: Data,~View: Data,~view: Value -> View,ends: Bool,+folded: List<&2,Value>,+value: Value) ->
  {viewed(~Value,~View,~view,placed_folded(~Value,ends,folded,value)) == placed_folded(~View,ends,viewed(~Value,~View,~view,folded),view(value)) : List<&2,View>}:
  match ends:
    case True{}: {==}
    case False{}: {==}

def placed_finals_view(~Value: Data,~View: Data,~Final: Data,~view: Value -> View,~turn_value: Value -> Final,~turn_view: View -> Final,~turn_law: TurnView(~Value,~View,~Final,~view,~turn_value,~turn_view),
  ends: Bool,+finals: List<&2,Final>,+value: Value) ->
  {placed_finals(~Value,~Final,~turn_value,ends,finals,value) == placed_finals(~View,~Final,~turn_view,ends,finals,view(value)) : List<&2,Final>}:
  match ends:
    case True{}: Equal.cong(Final,List<&2,Final>,final => final <> finals,turn_value(value),turn_view(view(value)),turn_law(value))
    case False{}: {==}

# The lines of a block, by a view of their values.
def lines_view(~Model: Data,~Context: Data,~Value: Data,~View: Data,~Final: Data,~view: Value -> View,~begin_value: Context -> Value,~begin_view: Context -> View,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~fold_view: Nat -> U32 -> U32 -> View -> Model -> View,~turn_value: Value -> Final,~turn_view: View -> Final,~begin_law: BeginView(~Context,~Value,~View,~view,~begin_value,~begin_view),~fold_law: FoldView(~Model,~Value,~View,~view,~fold_value,~fold_view),~turn_law: TurnView(~Value,~View,~Final,~view,~turn_value,~turn_view),remaining: Nat,begins: Bool,+ends: Bool,+line: U32,+inner: U32,+column: U32,+offset: U32,+count: U32,+next: U32,+all: List<&2,Context>,
  +upcoming: List<&2,Context>,+starts: List<&2,Value>,+folded: List<&2,Value>,+finals: List<&2,Final>,+model: Model) ->
  {viewed_folded(~Context,~Value,~View,~Final,~view,lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,remaining,begins,ends,line,inner,column,offset,count,next,all,upcoming,starts,folded,finals,model)) ==
    lines_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,remaining,begins,ends,line,inner,column,offset,count,next,all,upcoming,viewed(~Value,~View,~view,starts),viewed(~Value,~View,~view,folded),finals,model) : Folded<Context,View,Final>}:
  match remaining begins upcoming starts:
    case 1n+rest True{} Con{+context,+later} starts:
      +value = fold_value(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,begin_value(context),model)
      +seen = fold_view(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,begin_view(context),model)
      +same = Equal.trans(View,view(value),fold_view(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,view(begin_value(context)),model),seen,
        fold_law(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,begin_value(context),model),Equal.cong(View,View,start => fold_view(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,start,model),view(begin_value(context)),begin_view(context),begin_law(context)))
      +folded_same = Equal.trans(List<&2,View>,viewed(~Value,~View,~view,placed_folded(~Value,ends,folded,value)),placed_folded(~View,ends,viewed(~Value,~View,~view,folded),view(value)),placed_folded(~View,ends,viewed(~Value,~View,~view,folded),seen),
        placed_folded_view(~Value,~View,~view,ends,folded,value),Equal.cong(View,List<&2,View>,state => placed_folded(~View,ends,viewed(~Value,~View,~view,folded),state),view(value),seen,same))
      +finals_same = Equal.trans(List<&2,Final>,placed_finals(~Value,~Final,~turn_value,ends,finals,value),placed_finals(~View,~Final,~turn_view,ends,finals,view(value)),placed_finals(~View,~Final,~turn_view,ends,finals,seen),
        placed_finals_view(~Value,~View,~Final,~view,~turn_value,~turn_view,~turn_law,ends,finals,value),Equal.cong(View,List<&2,Final>,state => placed_finals(~View,~Final,~turn_view,ends,finals,state),view(value),seen,same))
      %folded_same : {viewed_folded(~Context,~Value,~View,~Final,~view,lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,LineChain.refilled(~Context,all,later),starts,placed_folded(~Value,ends,folded,value),placed_finals(~Value,~Final,~turn_value,ends,finals,value),model)) ==
          lines_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,LineChain.refilled(~Context,all,later),viewed(~Value,~View,~view,starts),_,placed_finals(~View,~Final,~turn_view,ends,finals,seen),model) : Folded<Context,View,Final>}
      %finals_same : {viewed_folded(~Context,~Value,~View,~Final,~view,lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,LineChain.refilled(~Context,all,later),starts,placed_folded(~Value,ends,folded,value),placed_finals(~Value,~Final,~turn_value,ends,finals,value),model)) ==
          lines_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,LineChain.refilled(~Context,all,later),viewed(~Value,~View,~view,starts),viewed(~Value,~View,~view,placed_folded(~Value,ends,folded,value)),_,model) : Folded<Context,View,Final>}
      lines_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,LineChain.refilled(~Context,all,later),starts,placed_folded(~Value,ends,folded,value),placed_finals(~Value,~Final,~turn_value,ends,finals,value),model)
    case 1n+rest False{} upcoming Con{+state,+tail}:
      +value = fold_value(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,state,model)
      +seen = fold_view(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,view(state),model)
      +same = fold_law(U32.to_nat(Bool.pick(U32,U32.is_lt(((line + inner - column % inner) % inner : U32),count),((count - ((line + inner - column % inner) % inner : U32) + inner - 1) / inner : U32),0)),(offset + ((line + inner - column % inner) % inner : U32) : U32),inner,state,model)
      +folded_same = Equal.trans(List<&2,View>,viewed(~Value,~View,~view,placed_folded(~Value,ends,folded,value)),placed_folded(~View,ends,viewed(~Value,~View,~view,folded),view(value)),placed_folded(~View,ends,viewed(~Value,~View,~view,folded),seen),
        placed_folded_view(~Value,~View,~view,ends,folded,value),Equal.cong(View,List<&2,View>,next_state => placed_folded(~View,ends,viewed(~Value,~View,~view,folded),next_state),view(value),seen,same))
      +finals_same = Equal.trans(List<&2,Final>,placed_finals(~Value,~Final,~turn_value,ends,finals,value),placed_finals(~View,~Final,~turn_view,ends,finals,view(value)),placed_finals(~View,~Final,~turn_view,ends,finals,seen),
        placed_finals_view(~Value,~View,~Final,~view,~turn_value,~turn_view,~turn_law,ends,finals,value),Equal.cong(View,List<&2,Final>,next_state => placed_finals(~View,~Final,~turn_view,ends,finals,next_state),view(value),seen,same))
      %folded_same : {viewed_folded(~Context,~Value,~View,~Final,~view,lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,tail,placed_folded(~Value,ends,folded,value),placed_finals(~Value,~Final,~turn_value,ends,finals,value),model)) ==
          lines_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,viewed(~Value,~View,~view,tail),_,placed_finals(~View,~Final,~turn_view,ends,finals,seen),model) : Folded<Context,View,Final>}
      %finals_same : {viewed_folded(~Context,~Value,~View,~Final,~view,lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,tail,placed_folded(~Value,ends,folded,value),placed_finals(~Value,~Final,~turn_value,ends,finals,value),model)) ==
          lines_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,viewed(~Value,~View,~view,tail),viewed(~Value,~View,~view,placed_folded(~Value,ends,folded,value)),_,model) : Folded<Context,View,Final>}
      lines_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,tail,placed_folded(~Value,ends,folded,value),placed_finals(~Value,~Final,~turn_value,ends,finals,value),model)
    case 0n begins upcoming starts:
      Equal.cong(List<&2,View>,Folded<Context,View,Final>,states => Folded{next,upcoming,states,finals},viewed(~Value,~View,~view,onto_values(~Value,folded,starts)),onto_values(~View,viewed(~Value,~View,~view,folded),viewed(~Value,~View,~view,starts)),onto_view(~Value,~View,~view,folded,starts))
    case 1n+rest True{} Nil{} starts:
      Equal.cong(List<&2,View>,Folded<Context,View,Final>,states => Folded{next,[],states,finals},viewed(~Value,~View,~view,onto_values(~Value,folded,starts)),onto_values(~View,viewed(~Value,~View,~view,folded),viewed(~Value,~View,~view,starts)),onto_view(~Value,~View,~view,folded,starts))
    case 1n+rest False{} upcoming Nil{}:
      Equal.cong(List<&2,View>,Folded<Context,View,Final>,states => Folded{next,upcoming,states,finals},viewed(~Value,~View,~view,onto_values(~Value,folded,[])),onto_values(~View,viewed(~Value,~View,~view,folded),[]),onto_view(~Value,~View,~view,folded,[]))

# The segments of a block, by a view of their values.
def segments_view(~Model: Data,~Context: Data,~Value: Data,~View: Data,~Final: Data,~view: Value -> View,~begin_value: Context -> Value,~begin_view: Context -> View,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~fold_view: Nat -> U32 -> U32 -> View -> Model -> View,~turn_value: Value -> Final,~turn_view: View -> Final,~begin_law: BeginView(~Context,~Value,~View,~view,~begin_value,~begin_view),~fold_law: FoldView(~Model,~Value,~View,~view,~fold_value,~fold_view),~turn_law: TurnView(~Value,~View,~Final,~view,~turn_value,~turn_view),fuel: Nat,done: Bool,+lines: LineChain.Lines<Context>,+offset: U32,+remaining: U32,+folded: Folded<Context,Value,Final>,+model: Model) ->
  {viewed_folded(~Context,~Value,~View,~Final,~view,segments_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,fuel,done,lines,offset,remaining,folded,model)) == segments_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,fuel,done,lines,offset,remaining,viewed_folded(~Context,~Value,~View,~Final,~view,folded),model) : Folded<Context,View,Final>}:
  match fuel done:
    case 1n+rest False{}:
      LineChain.Lines{+all,+line} = lines
      Folded{+column,+upcoming,+starts,+finals} = folded
      +count = U32.min(remaining,(LineChain.cells_of(line) - column : U32))
      +ends = U32.is_eq((column + count : U32),LineChain.cells_of(line))
      +block = lines_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,U32.to_nat(LineChain.inner_of(line)),U32.is_zero(column),ends,0,LineChain.inner_of(line),column,offset,count,Bool.pick(U32,ends,0,(column + count : U32)),all,upcoming,starts,[],finals,model)
      Equal.trans(Folded<Context,View,Final>,viewed_folded(~Context,~Value,~View,~Final,~view,segments_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,rest,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),block,model)),
        segments_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,rest,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),viewed_folded(~Context,~Value,~View,~Final,~view,block),model),
        segments_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,rest,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),
          lines_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,U32.to_nat(LineChain.inner_of(line)),U32.is_zero(column),ends,0,LineChain.inner_of(line),column,offset,count,Bool.pick(U32,ends,0,(column + count : U32)),all,upcoming,viewed(~Value,~View,~view,starts),[],finals,model),model),
        segments_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,rest,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),block,model),
        Equal.cong(Folded<Context,View,Final>,Folded<Context,View,Final>,changed => segments_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,rest,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),changed,model),
          viewed_folded(~Context,~Value,~View,~Final,~view,block),lines_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,U32.to_nat(LineChain.inner_of(line)),U32.is_zero(column),ends,0,LineChain.inner_of(line),column,offset,count,Bool.pick(U32,ends,0,(column + count : U32)),all,upcoming,viewed(~Value,~View,~view,starts),[],finals,model),
          lines_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,U32.to_nat(LineChain.inner_of(line)),U32.is_zero(column),ends,0,LineChain.inner_of(line),column,offset,count,Bool.pick(U32,ends,0,(column + count : U32)),all,upcoming,starts,[],finals,model)))
    case 0n done: {==}
    case 1n+rest True{}: {==}

# A walk over blocks, by a view of its values.
def walk_view(~Model: Data,~Context: Data,~Value: Data,~View: Data,~Final: Data,~view: Value -> View,~begin_value: Context -> Value,~begin_view: Context -> View,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~fold_view: Nat -> U32 -> U32 -> View -> Model -> View,~turn_value: Value -> Final,~turn_view: View -> Final,~begin_law: BeginView(~Context,~Value,~View,~view,~begin_value,~begin_view),~fold_law: FoldView(~Model,~Value,~View,~view,~fold_value,~fold_view),~turn_law: TurnView(~Value,~View,~Final,~view,~turn_value,~turn_view),+lines: LineChain.Lines<Context>,tree: Partition.Tree<&2,Model,Unit>,+folded: Folded<Context,Value,Final>) ->
  {viewed_folded(~Context,~Value,~View,~Final,~view,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,tree,folded)) == walk_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,lines,tree,viewed_folded(~Context,~Value,~View,~Final,~view,folded)) : Folded<Context,View,Final>}:
  match tree:
    case Partition.Leaf{+model,+first,+count,+details}:
      segments_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,LineChain.segments(count,LineChain.cells_of(LineChain.line_of(~Context,lines))),U32.is_zero(count) || U32.is_zero(LineChain.cells_of(LineChain.line_of(~Context,lines))),lines,0,count,folded,model)
    case Partition.Fork{+left,+right}:
      Equal.trans(Folded<Context,View,Final>,viewed_folded(~Context,~Value,~View,~Final,~view,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,right,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,left,folded))),walk_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,lines,right,viewed_folded(~Context,~Value,~View,~Final,~view,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,left,folded))),
        walk_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,lines,right,walk_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,lines,left,viewed_folded(~Context,~Value,~View,~Final,~view,folded))),
        walk_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,lines,right,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,left,folded)),
        Equal.cong(Folded<Context,View,Final>,Folded<Context,View,Final>,changed => walk_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,lines,right,changed),viewed_folded(~Context,~Value,~View,~Final,~view,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,left,folded)),walk_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,lines,left,viewed_folded(~Context,~Value,~View,~Final,~view,folded)),
          walk_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,lines,left,folded)))

# The tasks of a walk, by a view of their values.
def task_view(~Model: Data,~Context: Data,~Value: Data,~View: Data,~Final: Data,~view: Value -> View,~begin_value: Context -> Value,~begin_view: Context -> View,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~fold_view: Nat -> U32 -> U32 -> View -> Model -> View,~turn_value: Value -> Final,~turn_view: View -> Final,~begin_law: BeginView(~Context,~Value,~View,~view,~begin_value,~begin_view),~fold_law: FoldView(~Model,~Value,~View,~view,~fold_value,~fold_view),~turn_law: TurnView(~Value,~View,~Final,~view,~turn_value,~turn_view),+lines: LineChain.Lines<Context>,+first: U32,+count: U32,+tree: Partition.Tree<&2,Model,Unit>) ->
  {viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,task_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,first,count,tree)) == task_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,lines,first,count,tree) : Partition.Tree<&2,Walked<Model,Context,View,Final>,Unit>}:
  Equal.cong(Folded<Context,View,Final>,Partition.Tree<&2,Walked<Model,Context,View,Final>,Unit>,changed => Partition.Leaf{Walked{tree,changed},first,count,Unit{}},viewed_folded(~Context,~Value,~View,~Final,~view,walk_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,lines,tree,task_start(~Context,~Value,~Final,lines,first))),
    walk_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,lines,tree,task_start(~Context,~View,~Final,lines,first)),walk_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,lines,tree,task_start(~Context,~Value,~Final,lines,first)))

def tasks_view(~Model: Data,~Context: Data,~Value: Data,~View: Data,~Final: Data,~view: Value -> View,~begin_value: Context -> Value,~begin_view: Context -> View,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~fold_view: Nat -> U32 -> U32 -> View -> Model -> View,~turn_value: Value -> Final,~turn_view: View -> Final,~begin_law: BeginView(~Context,~Value,~View,~view,~begin_value,~begin_view),~fold_law: FoldView(~Model,~Value,~View,~view,~fold_value,~fold_view),~turn_law: TurnView(~Value,~View,~Final,~view,~turn_value,~turn_view),layout: Pieces.Ranges,+lines: LineChain.Lines<Context>,tree: Partition.Tree<&2,Model,Unit>) ->
  {viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,layout,lines,tree)) == tasks_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,layout,lines,tree) : Partition.Tree<&2,Walked<Model,Context,View,Final>,Unit>}:
  match layout tree:
    case Pieces.Range{+first,+count} tree: task_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,lines,first,count,tree)
    case Pieces.Split{+left,+right} Partition.Leaf{+model,+first,+count,+details}:
      task_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,lines,Pieces.first(Pieces.Split{left,right}),(Pieces.last(Pieces.Split{left,right}) - Pieces.first(Pieces.Split{left,right}) : U32),Partition.Leaf{model,first,count,details})
    case Pieces.Split{+left,+right} Partition.Fork{+low,+high}:
      Equal.trans(Partition.Tree<&2,Walked<Model,Context,View,Final>,Unit>,Partition.Fork{viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,left,lines,low)),viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,right,lines,high))},
        Partition.Fork{tasks_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,left,lines,low),viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,right,lines,high))},Partition.Fork{tasks_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,left,lines,low),tasks_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,right,lines,high)},
        Equal.cong(Partition.Tree<&2,Walked<Model,Context,View,Final>,Unit>,Partition.Tree<&2,Walked<Model,Context,View,Final>,Unit>,changed => Partition.Fork{changed,viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,right,lines,high))},viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,left,lines,low)),tasks_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,left,lines,low),
          tasks_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,left,lines,low)),
        Equal.cong(Partition.Tree<&2,Walked<Model,Context,View,Final>,Unit>,Partition.Tree<&2,Walked<Model,Context,View,Final>,Unit>,changed => Partition.Fork{tasks_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,left,lines,low),changed},viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,right,lines,high)),tasks_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,right,lines,high),
          tasks_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,right,lines,high)))

# The blocks and finals of tasks do not depend on how their values are seen.
def unzipped_view(~Model: Data,~Context: Data,~Value: Data,~View: Data,~Final: Data,~view: Value -> View,tree: Partition.Tree<&2,Walked<Model,Context,Value,Final>,Unit>,+done: List<&2,Final>) ->
  {unzipped_model(~Model,~Context,~View,~Final,viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,tree),done) == unzipped_model(~Model,~Context,~Value,~Final,tree,done) : Unzipped<Model,Final>}:
  match tree:
    case Partition.Leaf{+walked,+first,+count,+details}:
      match walked:
        case Walked{+blocks,+folded}:
          match folded:
            case Folded{+column,+upcoming,+starts,+finals}: {==}
    case Partition.Fork{+left,+right}:
      +high = unzipped_model(~Model,~Context,~Value,~Final,right,done)
      +seen_high = unzipped_model(~Model,~Context,~View,~Final,viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,right),done)
      +low = unzipped_model(~Model,~Context,~Value,~Final,left,unzipped_finals(~Model,~Final,high))
      +seen_low = unzipped_model(~Model,~Context,~View,~Final,viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,left),unzipped_finals(~Model,~Final,seen_high))
      +high_same = unzipped_view(~Model,~Context,~Value,~View,~Final,~view,right,done)
      +low_same = Equal.trans(Unzipped<Model,Final>,seen_low,unzipped_model(~Model,~Context,~View,~Final,viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,left),unzipped_finals(~Model,~Final,high)),low,
        Equal.cong(Unzipped<Model,Final>,Unzipped<Model,Final>,changed => unzipped_model(~Model,~Context,~View,~Final,viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,left),unzipped_finals(~Model,~Final,changed)),seen_high,high,high_same),
        unzipped_view(~Model,~Context,~Value,~View,~Final,~view,left,unzipped_finals(~Model,~Final,high)))
      %Equal.sym(Unzipped<Model,Final>,seen_low,low,low_same) :
        {Unzipped{Partition.Fork{unzipped_tree(~Model,~Final,_),unzipped_tree(~Model,~Final,seen_high)},unzipped_finals(~Model,~Final,_)} ==
          Unzipped{Partition.Fork{unzipped_tree(~Model,~Final,low),unzipped_tree(~Model,~Final,high)},unzipped_finals(~Model,~Final,low)} : Unzipped<Model,Final>}
      %Equal.sym(Unzipped<Model,Final>,seen_high,high,high_same) :
        {Unzipped{Partition.Fork{unzipped_tree(~Model,~Final,low),unzipped_tree(~Model,~Final,_)},unzipped_finals(~Model,~Final,low)} ==
          Unzipped{Partition.Fork{unzipped_tree(~Model,~Final,low),unzipped_tree(~Model,~Final,high)},unzipped_finals(~Model,~Final,low)} : Unzipped<Model,Final>}
      {==}

# The folds of the lines of a grid, by a view of their values.
def folds_view(~Model: Data,~Context: Data,~Value: Data,~View: Data,~Final: Data,~view: Value -> View,~begin_value: Context -> Value,~begin_view: Context -> View,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~fold_view: Nat -> U32 -> U32 -> View -> Model -> View,~turn_value: Value -> Final,~turn_view: View -> Final,~begin_law: BeginView(~Context,~Value,~View,~view,~begin_value,~begin_view),~fold_law: FoldView(~Model,~Value,~View,~view,~fold_value,~fold_view),~turn_law: TurnView(~Value,~View,~Final,~view,~turn_value,~turn_view),~Fill: Data,~vacant_model: Fill -> Model,+contexts: List<&2,Context>,+line: LineChain.Line,+group: U32,+bytes: F32,+operations: F32,+grid: Grid.Grid,+fill: Fill,+workers: U32,+tree: Partition.Tree<&2,Model,Unit>) ->
  {folds_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,~Fill,~vacant_model,contexts,line,group,bytes,operations,grid,fill,workers,tree) == folds_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,~Fill,~vacant_model,contexts,line,group,bytes,operations,grid,fill,workers,tree) : Unzipped<Model,Final>}:
  +layout = Pieces.ranges_in(grid,group)
  +decision = Tasks.decide_over(layout,Grid.count_of(grid),workers,LineChain.cells_of(line),bytes,operations)
  +lines = {LineChain.Lines{contexts,line} : LineChain.Lines<Context>}
  +regrouped = Regrouped.regrouped(~Model,~Fill,~vacant_model,layout,fill,tree)
  +work = tasks_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,decided_tasks(decision),lines,regrouped)
  Equal.trans(Unzipped<Model,Final>,unzipped_model(~Model,~Context,~Value,~Final,work,[]),unzipped_model(~Model,~Context,~View,~Final,viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,work),[]),
    unzipped_model(~Model,~Context,~View,~Final,tasks_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,decided_tasks(decision),lines,regrouped),[]),
    Equal.sym(Unzipped<Model,Final>,unzipped_model(~Model,~Context,~View,~Final,viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,work),[]),unzipped_model(~Model,~Context,~Value,~Final,work,[]),unzipped_view(~Model,~Context,~Value,~View,~Final,~view,work,[])),
    Equal.cong(Partition.Tree<&2,Walked<Model,Context,View,Final>,Unit>,Unzipped<Model,Final>,changed => unzipped_model(~Model,~Context,~View,~Final,changed,[]),viewed_tasks(~Model,~Context,~Value,~View,~Final,~view,work),tasks_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,decided_tasks(decision),lines,regrouped),
      tasks_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,decided_tasks(decision),lines,regrouped)))

# A fold of lines whose values a view sees is the fold of the model of the
# view.
def chain_viewed(~Model: Data,~Context: Data,~Value: Data,~View: Data,~Final: Data,~view: Value -> View,~begin_value: Context -> Value,~begin_view: Context -> View,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~fold_view: Nat -> U32 -> U32 -> View -> Model -> View,~turn_value: Value -> Final,~turn_view: View -> Final,~begin_law: BeginView(~Context,~Value,~View,~view,~begin_value,~begin_view),~fold_law: FoldView(~Model,~Value,~View,~view,~fold_value,~fold_view),~turn_law: TurnView(~Value,~View,~Final,~view,~turn_value,~turn_view),~Block: Type,~represent: Model -> Block,~State: Type,~begin: Context -> State,~fold: @+count: Nat -> @+position: U32 -> @+step: U32 -> @initial: State -> Block -> Block & State,~turn: State -> Final
  ,~Fill: Data,~vacant: Fill -> Block,~vacant_model: Fill -> Model,
  ~chain_same: ChainSame(~Block,~Model,~represent,~Context,~State,~Value,~Final,~begin,~begin_value,~fold,~fold_value,~turn,~turn_value,~Fill,~vacant,~vacant_model),+contexts: List<&2,Context>,+line: LineChain.Line,+group: U32,+bytes: F32,+operations: F32,+grid: Grid.Grid,+fill: Fill,+workers: U32,+tree: Partition.Tree<&2,Model,Unit>) ->
  {LineChain.folds(~Block,~Fill,~Context,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,Regrouped.tree_of(~Block,~Model,~represent,tree)) == unzipped_pair(~Block,~Model,~represent,~Final,folds_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,~Fill,~vacant_model,contexts,line,group,bytes,operations,grid,fill,workers,tree)) : Partition.Tree<&1,Block,Unit> & List<&2,Final>}:
  Equal.trans(Partition.Tree<&1,Block,Unit> & List<&2,Final>,LineChain.folds(~Block,~Fill,~Context,~State,~Final,~begin,~fold,~turn,~vacant,contexts,line,group,bytes,operations,grid,fill,workers,Regrouped.tree_of(~Block,~Model,~represent,tree)),unzipped_pair(~Block,~Model,~represent,~Final,folds_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,~Fill,~vacant_model,contexts,line,group,bytes,operations,grid,fill,workers,tree)),
    unzipped_pair(~Block,~Model,~represent,~Final,folds_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,~Fill,~vacant_model,contexts,line,group,bytes,operations,grid,fill,workers,tree)),chain_same(contexts,line,group,bytes,operations,grid,fill,workers,tree),
    Equal.cong(Unzipped<Model,Final>,Partition.Tree<&1,Block,Unit> & List<&2,Final>,changed => unzipped_pair(~Block,~Model,~represent,~Final,changed),folds_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,~Fill,~vacant_model,contexts,line,group,bytes,operations,grid,fill,workers,tree),folds_model(~Model,~Context,~View,~Final,~begin_view,~fold_view,~turn_view,~Fill,~vacant_model,contexts,line,group,bytes,operations,grid,fill,workers,tree),
      folds_view(~Model,~Context,~Value,~View,~Final,~view,~begin_value,~begin_view,~fold_value,~fold_view,~turn_value,~turn_view,~begin_law,~fold_law,~turn_law,~Fill,~vacant_model,contexts,line,group,bytes,operations,grid,fill,workers,tree)))

# ---- The cells of a line ----

# The fold of count cells from position on, stride apart, in order.
def strided(~Model: Data,~Cell: Data,~State: Data,~cell: Model -> U32 -> Cell,~step: State -> Cell -> State,count: Nat,+position: U32,+stride: U32,+initial: State,+model: Model) -> State:
  match count:
    case 0n: initial
    case 1n+rest: strided(~Model,~Cell,~State,~cell,~step,rest,(position + stride : U32),stride,step(initial,cell(model,position)),model)

# count cells from offset on folded in order, gap cells skipped before the
# first taken and inner - 1 between two taken.
def taken(~Model: Data,~Cell: Data,~State: Data,~cell: Model -> U32 -> Cell,~step: State -> Cell -> State,count: Nat,+gap: Nat,+inner: Nat,+offset: U32,+initial: State,+model: Model) -> State:
  match count:
    case 0n: initial
    case 1n+rest:
      match gap:
        case 0n: taken(~Model,~Cell,~State,~cell,~step,rest,Nat.sub(inner,1n),inner,(offset + 1 : U32),step(initial,cell(model,offset)),model)
        case 1n+ +before: taken(~Model,~Cell,~State,~cell,~step,rest,before,inner,(offset + 1 : U32),initial,model)

# How many of count cells are taken from gap on, inner apart.
def cells_from(+gap: Nat,+inner: Nat,+count: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_lt(gap,count),Nat.div(Nat.sub(Nat.add(count,inner),1n+gap),inner),0n)

def next_position(+offset: U32,+before: Nat) -> {(offset + U32.from_nat(1n+before) : U32) == ((offset + 1 : U32) + U32.from_nat(before) : U32) : U32}:
  %NatWords.increment_encoded(before) : {(offset + _ : U32) == ((offset + 1 : U32) + U32.from_nat(before) : U32) : U32}
  Equal.trans(U32,(offset + (U32.from_nat(before) + 1 : U32) : U32),((offset + U32.from_nat(before) : U32) + 1 : U32),((offset + 1 : U32) + U32.from_nat(before) : U32),
    Equal.sym(U32,((offset + U32.from_nat(before) : U32) + 1 : U32),(offset + (U32.from_nat(before) + 1 : U32) : U32),Arithmetic.add_associative(offset,U32.from_nat(before),1)),
    Equal.sym(U32,((offset + 1 : U32) + U32.from_nat(before) : U32),((offset + U32.from_nat(before) : U32) + 1 : U32),Arithmetic.add_swap(offset,1,U32.from_nat(before))))

def first_position(+offset: U32,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool}) ->
  {(offset + U32.from_nat(inner) : U32) == ((offset + 1 : U32) + U32.from_nat(Nat.sub(inner,1n)) : U32) : U32}:
  %Order.positive_has_last(inner,positive) : {(offset + U32.from_nat(_) : U32) == ((offset + 1 : U32) + U32.from_nat(Nat.sub(inner,1n)) : U32) : U32}
  next_position(offset,Nat.sub(inner,1n))

def first_cells(+rest: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool}) -> {cells_from(0n,inner,1n+rest) == 1n+Nat.div(rest,inner) : Nat}:
  %Equal.sym(Bool,Nat.is_lt(0n,1n+rest),True{},Order.lt_succ(0n,rest,Order.zero_le(rest))) :
    {Bool.pick(Nat,_,Nat.div(Nat.sub(Nat.add(1n+rest,inner),1n),inner),0n) == 1n+Nat.div(rest,inner) : Nat}
  %Equal.sym(Nat,Nat.sub(Nat.add(rest,inner),0n),Nat.add(rest,inner),Order.subtract_zero(Nat.add(rest,inner))) : {Nat.div(_,inner) == 1n+Nat.div(rest,inner) : Nat}
  %Order.commutative(inner,rest) : {Nat.div(_,inner) == 1n+Nat.div(rest,inner) : Nat}
  %Addition.zero_right(inner) : {Nat.div(Nat.add(_,rest),inner) == 1n+Nat.div(rest,inner) : Nat}
  DivisionOrder.quotient_shift(rest,inner,1n,positive)

def last_below(+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool}) -> {Nat.is_lt(Nat.sub(inner,1n),inner) == True{} : Bool}:
  %Order.positive_has_last(inner,positive) : {Nat.is_lt(Nat.sub(inner,1n),_) == True{} : Bool}
  Order.lt_succ(Nat.sub(inner,1n),Nat.sub(inner,1n),Order.reflexive(Nat.sub(inner,1n)))

def last_cells_by(+rest: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},which: Bool,+classified: {Nat.is_lt(Nat.sub(inner,1n),rest) == which : Bool}) ->
  {cells_from(Nat.sub(inner,1n),inner,rest) == Nat.div(rest,inner) : Nat}:
  match which:
    case True{}:
      %Equal.sym(Bool,Nat.is_lt(Nat.sub(inner,1n),rest),True{},classified) : {Bool.pick(Nat,_,Nat.div(Nat.sub(Nat.add(rest,inner),1n+Nat.sub(inner,1n)),inner),0n) == Nat.div(rest,inner) : Nat}
      %Equal.sym(Nat,1n+Nat.sub(inner,1n),inner,Order.positive_has_last(inner,positive)) : {Nat.div(Nat.sub(Nat.add(rest,inner),_),inner) == Nat.div(rest,inner) : Nat}
      %Equal.sym(Nat,Nat.sub(Nat.add(rest,inner),inner),rest,DivisionOrder.subtract_sum(rest,inner)) : {Nat.div(_,inner) == Nat.div(rest,inner) : Nat}
      {==}
    case False{}:
      %Equal.sym(Bool,Nat.is_lt(Nat.sub(inner,1n),rest),False{},classified) : {Bool.pick(Nat,_,Nat.div(Nat.sub(Nat.add(rest,inner),1n+Nat.sub(inner,1n)),inner),0n) == Nat.div(rest,inner) : Nat}
      +small = Order.le_lt_trans(rest,Nat.sub(inner,1n),inner,Order.not_less(Nat.sub(inner,1n),rest,classified),last_below(inner,positive))
      Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),(0n,rest),Nat.divmod(rest,inner),Equal.sym(Nat & Nat,Nat.divmod(rest,inner),(0n,rest),Quotients.small(rest,inner,small)))

def last_cells(+rest: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool}) -> {cells_from(Nat.sub(inner,1n),inner,rest) == Nat.div(rest,inner) : Nat}:
  last_cells_by(rest,inner,positive,Nat.is_lt(Nat.sub(inner,1n),rest),{==})

def gap_below(+before: Nat,+inner: Nat,+below: {Nat.is_lt(1n+before,inner) == True{} : Bool}) -> {Nat.is_lt(before,inner) == True{} : Bool}:
  Order.le_lt_trans(before,1n+before,inner,Order.successor(before,before,Order.reflexive(before)),below)

def gapped_first(~Model: Data,~Cell: Data,~State: Data,~cell: Model -> U32 -> Cell,~step: State -> Cell -> State,+rest: Nat,+inner: Nat,+offset: U32,+initial: State,+model: Model,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},
  +after: {strided(~Model,~Cell,~State,~cell,~step,cells_from(Nat.sub(inner,1n),inner,rest),((offset + 1 : U32) + U32.from_nat(Nat.sub(inner,1n)) : U32),U32.from_nat(inner),step(initial,cell(model,offset)),model) ==
    taken(~Model,~Cell,~State,~cell,~step,rest,Nat.sub(inner,1n),inner,(offset + 1 : U32),step(initial,cell(model,offset)),model) : State}) ->
  {strided(~Model,~Cell,~State,~cell,~step,cells_from(0n,inner,1n+rest),(offset + U32.from_nat(0n) : U32),U32.from_nat(inner),initial,model) == taken(~Model,~Cell,~State,~cell,~step,1n+rest,0n,inner,offset,initial,model) : State}:
  %Equal.sym(Nat,cells_from(0n,inner,1n+rest),1n+Nat.div(rest,inner),first_cells(rest,inner,positive)) :
    {strided(~Model,~Cell,~State,~cell,~step,_,(offset + U32.from_nat(0n) : U32),U32.from_nat(inner),initial,model) == taken(~Model,~Cell,~State,~cell,~step,1n+rest,0n,inner,offset,initial,model) : State}
  %Equal.sym(U32,(offset + 0 : U32),offset,Arithmetic.add_zero(offset)) :
    {strided(~Model,~Cell,~State,~cell,~step,Nat.div(rest,inner),(_ + U32.from_nat(inner) : U32),U32.from_nat(inner),step(initial,cell(model,_)),model) ==
      taken(~Model,~Cell,~State,~cell,~step,rest,Nat.sub(inner,1n),inner,(offset + 1 : U32),step(initial,cell(model,offset)),model) : State}
  %Equal.sym(U32,(offset + U32.from_nat(inner) : U32),((offset + 1 : U32) + U32.from_nat(Nat.sub(inner,1n)) : U32),first_position(offset,inner,positive)) :
    {strided(~Model,~Cell,~State,~cell,~step,Nat.div(rest,inner),_,U32.from_nat(inner),step(initial,cell(model,offset)),model) ==
      taken(~Model,~Cell,~State,~cell,~step,rest,Nat.sub(inner,1n),inner,(offset + 1 : U32),step(initial,cell(model,offset)),model) : State}
  %last_cells(rest,inner,positive) :
    {strided(~Model,~Cell,~State,~cell,~step,_,((offset + 1 : U32) + U32.from_nat(Nat.sub(inner,1n)) : U32),U32.from_nat(inner),step(initial,cell(model,offset)),model) ==
      taken(~Model,~Cell,~State,~cell,~step,rest,Nat.sub(inner,1n),inner,(offset + 1 : U32),step(initial,cell(model,offset)),model) : State}
  after

# count cells from offset on, every inner-th taken from gap on, are the
# taken cells read stride apart from the first taken.
def gapped(~Model: Data,~Cell: Data,~State: Data,~cell: Model -> U32 -> Cell,~step: State -> Cell -> State,count: Nat,+gap: Nat,+inner: Nat,+offset: U32,+initial: State,+model: Model,+below: {Nat.is_lt(gap,inner) == True{} : Bool}) ->
  {strided(~Model,~Cell,~State,~cell,~step,cells_from(gap,inner,count),(offset + U32.from_nat(gap) : U32),U32.from_nat(inner),initial,model) == taken(~Model,~Cell,~State,~cell,~step,count,gap,inner,offset,initial,model) : State}:
  match count:
    case 0n:
      %Equal.sym(Bool,Nat.is_lt(gap,0n),False{},Order.not_below(0n,gap,Order.zero_le(gap))) :
        {strided(~Model,~Cell,~State,~cell,~step,Bool.pick(Nat,_,Nat.div(Nat.sub(Nat.add(0n,inner),1n+gap),inner),0n),(offset + U32.from_nat(gap) : U32),U32.from_nat(inner),initial,model) == initial : State}
      {==}
    case 1n+ +rest:
      match gap:
        case 0n:
          gapped_first(~Model,~Cell,~State,~cell,~step,rest,inner,offset,initial,model,below,
            gapped(~Model,~Cell,~State,~cell,~step,rest,Nat.sub(inner,1n),inner,(offset + 1 : U32),step(initial,cell(model,offset)),model,last_below(inner,below)))
        case 1n+ +before:
          %Equal.sym(U32,(offset + U32.from_nat(1n+before) : U32),((offset + 1 : U32) + U32.from_nat(before) : U32),next_position(offset,before)) :
            {strided(~Model,~Cell,~State,~cell,~step,cells_from(before,inner,rest),_,U32.from_nat(inner),initial,model) == taken(~Model,~Cell,~State,~cell,~step,rest,before,inner,(offset + 1 : U32),initial,model) : State}
          gapped(~Model,~Cell,~State,~cell,~step,rest,before,inner,(offset + 1 : U32),initial,model,gap_below(before,inner,below))

# The gap before the next cell of a line, one column on.
def gap_step(+gap: Nat,+inner: Nat) -> Nat:
  match gap:
    case 0n: Nat.sub(inner,1n)
    case 1n+ +before: before

def gap_after(count: Nat,+gap: Nat,+inner: Nat) -> Nat:
  match count:
    case 0n: gap
    case 1n+rest: gap_after(rest,gap_step(gap,inner),inner)

def gap_at(+line: Nat,+inner: Nat,+residue: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_le(residue,line),Nat.sub(line,residue),Nat.sub(Nat.add(line,inner),residue))

def next_residue(+residue: Nat,+inner: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_lt(1n+residue,inner),1n+residue,0n)

def residue_after(count: Nat,+residue: Nat,+inner: Nat) -> Nat:
  match count:
    case 0n: residue
    case 1n+rest: residue_after(rest,next_residue(residue,inner),inner)

def le_false_lt(+left: Nat,+right: Nat,order: {Nat.is_le(left,right) == False{} : Bool}) -> {Nat.is_lt(right,left) == True{} : Bool}:
  match left right:
    case 0n 0n: Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},order)))
    case 0n 1n+right: Empty.absurd({Nat.is_lt(1n+right,0n) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},order)))
    case 1n+left 0n: {==}
    case 1n+left 1n+right: le_false_lt(left,right,order)

def lt_le_false(+left: Nat,+right: Nat,+below: {Nat.is_lt(right,left) == True{} : Bool}) -> {Nat.is_le(left,right) == False{} : Bool}:
  match left right:
    case 0n +right: Empty.absurd({Nat.is_le(0n,right) == False{} : Bool},Order.below_zero(right,below))
    case 1n+ +rest 0n: {==}
    case 1n+ +rest 1n+ +more: lt_le_false(rest,more,below)

def lt_le(+left: Nat,+right: Nat,+below: {Nat.is_lt(left,right) == True{} : Bool}) -> {Nat.is_le(left,right) == True{} : Bool}:
  Order.transitive(left,1n+left,right,Order.successor(left,left,Order.reflexive(left)),Order.strict_follow(left,right,below))

def less_difference(+larger: Nat,+smaller: Nat,+below: {Nat.is_lt(smaller,larger) == True{} : Bool}) -> {Nat.sub(larger,smaller) == 1n+Nat.sub(larger,1n+smaller) : Nat}:
  match larger smaller:
    case 0n +smaller: Empty.absurd({Nat.sub(0n,smaller) == 1n+Nat.sub(0n,1n+smaller) : Nat},Order.below_zero(smaller,below))
    case 1n+ +rest 0n:
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {1n+rest == 1n+_ : Nat}
      {==}
    case 1n+ +rest 1n+ +before: less_difference(rest,before,below)

def self_gap(+line: Nat,+inner: Nat) -> {gap_at(line,inner,line) == 0n : Nat}:
  %Equal.sym(Bool,Nat.is_le(line,line),True{},Order.reflexive(line)) : {Bool.pick(Nat,_,Nat.sub(line,line),Nat.sub(Nat.add(line,inner),line)) == 0n : Nat}
  Order.subtract_below(line,line,Order.reflexive(line))

def same_gap(+line: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+line_below: {Nat.is_lt(line,inner) == True{} : Bool},inside: Bool,
  +next: {Nat.is_lt(1n+line,inner) == inside : Bool}) -> {gap_at(line,inner,next_residue(line,inner)) == gap_step(gap_at(line,inner,line),inner) : Nat}:
  match inside:
    case True{}:
      %Equal.sym(Nat,gap_at(line,inner,line),0n,self_gap(line,inner)) : {gap_at(line,inner,next_residue(line,inner)) == gap_step(_,inner) : Nat}
      %Equal.sym(Bool,Nat.is_lt(1n+line,inner),True{},next) : {gap_at(line,inner,Bool.pick(Nat,_,1n+line,0n)) == Nat.sub(inner,1n) : Nat}
      %Equal.sym(Bool,Nat.is_le(1n+line,line),False{},lt_le_false(1n+line,line,Order.lt_succ(line,line,Order.reflexive(line)))) :
        {Bool.pick(Nat,_,Nat.sub(line,1n+line),Nat.sub(Nat.add(line,inner),1n+line)) == Nat.sub(inner,1n) : Nat}
      %Order.positive_has_last(inner,positive) : {Nat.sub(Nat.add(line,_),1n+line) == Nat.sub(_,1n) : Nat}
      %Equal.sym(Nat,Nat.add(line,1n+Nat.sub(inner,1n)),1n+Nat.add(line,Nat.sub(inner,1n)),Addition.successor_right(line,Nat.sub(inner,1n))) : {Nat.sub(_,1n+line) == Nat.sub(1n+Nat.sub(inner,1n),1n) : Nat}
      Equal.trans(Nat,Nat.sub(Nat.add(line,Nat.sub(inner,1n)),line),Nat.sub(inner,1n),Nat.sub(Nat.sub(inner,1n),0n),
        Order.subtract_shift(line,Nat.sub(inner,1n)),Equal.sym(Nat,Nat.sub(Nat.sub(inner,1n),0n),Nat.sub(inner,1n),Order.subtract_zero(Nat.sub(inner,1n))))
    case False{}:
      %Equal.sym(Nat,gap_at(line,inner,line),0n,self_gap(line,inner)) : {gap_at(line,inner,next_residue(line,inner)) == gap_step(_,inner) : Nat}
      +full = DivisionOrder.not_below(line,inner,line_below,next)
      %Equal.sym(Bool,Nat.is_lt(1n+line,inner),False{},next) : {gap_at(line,inner,Bool.pick(Nat,_,1n+line,0n)) == Nat.sub(inner,1n) : Nat}
      %Equal.sym(Bool,Nat.is_le(0n,line),True{},Order.zero_le(line)) : {Bool.pick(Nat,_,Nat.sub(line,0n),Nat.sub(Nat.add(line,inner),0n)) == Nat.sub(inner,1n) : Nat}
      %full : {Nat.sub(line,0n) == Nat.sub(_,1n) : Nat}
      {==}

# One column on from a column before the line.
def before_gap(+line: Nat,+inner: Nat,+residue: Nat,+line_below: {Nat.is_lt(line,inner) == True{} : Bool},+order: {Nat.is_le(residue,line) == True{} : Bool},
  +less: {Nat.is_lt(residue,line) == True{} : Bool}) -> {gap_at(line,inner,next_residue(residue,inner)) == gap_step(gap_at(line,inner,residue),inner) : Nat}:
  +follow = Order.strict_follow(residue,line,less)
  %Equal.sym(Bool,Nat.is_lt(1n+residue,inner),True{},Order.le_lt_trans(1n+residue,line,inner,follow,line_below)) :
    {gap_at(line,inner,Bool.pick(Nat,_,1n+residue,0n)) == gap_step(gap_at(line,inner,residue),inner) : Nat}
  %Equal.sym(Bool,Nat.is_le(1n+residue,line),True{},follow) :
    {Bool.pick(Nat,_,Nat.sub(line,1n+residue),Nat.sub(Nat.add(line,inner),1n+residue)) == gap_step(gap_at(line,inner,residue),inner) : Nat}
  %Equal.sym(Bool,Nat.is_le(residue,line),True{},order) :
    {Nat.sub(line,1n+residue) == gap_step(Bool.pick(Nat,_,Nat.sub(line,residue),Nat.sub(Nat.add(line,inner),residue)),inner) : Nat}
  %Equal.sym(Nat,Nat.sub(line,residue),1n+Nat.sub(line,1n+residue),less_difference(line,residue,less)) : {Nat.sub(line,1n+residue) == gap_step(_,inner) : Nat}
  {==}

def succ_le_lt(+value: Nat,+limit: Nat,order: {Nat.is_le(1n+value,limit) == True{} : Bool}) -> {Nat.is_lt(value,limit) == True{} : Bool}:
  match value limit:
    case value 0n: Empty.absurd({Nat.is_lt(value,0n) == True{} : Bool},WordEncoding.false_true(order))
    case 0n 1n+limit: {==}
    case 1n+value 1n+limit: succ_le_lt(value,limit,order)

def above_sum(+residue: Nat,+line: Nat,+inner: Nat,+below: {Nat.is_lt(residue,inner) == True{} : Bool}) -> {Nat.is_lt(residue,Nat.add(line,inner)) == True{} : Bool}:
  succ_le_lt(residue,Nat.add(line,inner),Order.transitive(1n+residue,inner,Nat.add(line,inner),Order.strict_follow(residue,inner,below),Order.right_le(line,inner)))

# One column on from a column after the line.
def after_gap(+line: Nat,+inner: Nat,+residue: Nat,+residue_below: {Nat.is_lt(residue,inner) == True{} : Bool},+order: {Nat.is_le(residue,line) == False{} : Bool},inside: Bool,
  +next: {Nat.is_lt(1n+residue,inner) == inside : Bool}) -> {gap_at(line,inner,next_residue(residue,inner)) == gap_step(gap_at(line,inner,residue),inner) : Nat}:
  match inside:
    case True{}:
      %Equal.sym(Bool,Nat.is_le(residue,line),False{},order) :
    {gap_at(line,inner,next_residue(residue,inner)) == gap_step(Bool.pick(Nat,_,Nat.sub(line,residue),Nat.sub(Nat.add(line,inner),residue)),inner) : Nat}
      +less = le_false_lt(residue,line,order)
      %Equal.sym(Bool,Nat.is_lt(1n+residue,inner),True{},next) : {gap_at(line,inner,Bool.pick(Nat,_,1n+residue,0n)) == gap_step(Nat.sub(Nat.add(line,inner),residue),inner) : Nat}
      %Equal.sym(Bool,Nat.is_le(1n+residue,line),False{},lt_le_false(1n+residue,line,Order.lt_succ(line,residue,lt_le(line,residue,less)))) :
        {Bool.pick(Nat,_,Nat.sub(line,1n+residue),Nat.sub(Nat.add(line,inner),1n+residue)) == gap_step(Nat.sub(Nat.add(line,inner),residue),inner) : Nat}
      %Equal.sym(Nat,Nat.sub(Nat.add(line,inner),residue),1n+Nat.sub(Nat.add(line,inner),1n+residue),
        less_difference(Nat.add(line,inner),residue,above_sum(residue,line,inner,residue_below))) :
        {Nat.sub(Nat.add(line,inner),1n+residue) == gap_step(_,inner) : Nat}
      {==}
    case False{}:
      %Equal.sym(Bool,Nat.is_le(residue,line),False{},order) :
    {gap_at(line,inner,next_residue(residue,inner)) == gap_step(Bool.pick(Nat,_,Nat.sub(line,residue),Nat.sub(Nat.add(line,inner),residue)),inner) : Nat}
      +full = DivisionOrder.not_below(residue,inner,residue_below,next)
      %Equal.sym(Bool,Nat.is_lt(1n+residue,inner),False{},next) : {gap_at(line,inner,Bool.pick(Nat,_,1n+residue,0n)) == gap_step(Nat.sub(Nat.add(line,inner),residue),inner) : Nat}
      %Equal.sym(Bool,Nat.is_le(0n,line),True{},Order.zero_le(line)) :
        {Bool.pick(Nat,_,Nat.sub(line,0n),Nat.sub(Nat.add(line,inner),0n)) == gap_step(Nat.sub(Nat.add(line,inner),residue),inner) : Nat}
      %full : {Nat.sub(line,0n) == gap_step(Nat.sub(Nat.add(line,_),residue),inner) : Nat}
      %Equal.sym(Nat,Nat.add(line,1n+residue),1n+Nat.add(line,residue),Addition.successor_right(line,residue)) : {Nat.sub(line,0n) == gap_step(Nat.sub(_,residue),inner) : Nat}
      %Equal.sym(Nat,Nat.sub(1n+Nat.add(line,residue),residue),1n+Nat.sub(1n+Nat.add(line,residue),1n+residue),
        less_difference(1n+Nat.add(line,residue),residue,Order.lt_succ(residue,Nat.add(line,residue),Order.right_le(line,residue)))) :
        {Nat.sub(line,0n) == gap_step(_,inner) : Nat}
      %Equal.sym(Nat,Nat.sub(Nat.add(line,residue),residue),line,DivisionOrder.subtract_sum(line,residue)) : {Nat.sub(line,0n) == _ : Nat}
      Order.subtract_zero(line)

# A column on, the gap before the next cell of the line is one step on.
def step_gap_by(+line: Nat,+inner: Nat,+residue: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+line_below: {Nat.is_lt(line,inner) == True{} : Bool},
  +residue_below: {Nat.is_lt(residue,inner) == True{} : Bool},order: Bool,+ordered: {Nat.is_le(residue,line) == order : Bool},less: Bool,+classified: {Nat.is_lt(residue,line) == less : Bool}) ->
  {gap_at(line,inner,next_residue(residue,inner)) == gap_step(gap_at(line,inner,residue),inner) : Nat}:
  match order less:
    case True{} True{}: before_gap(line,inner,residue,line_below,ordered,classified)
    case True{} False{}:
      +same = Order.antisymmetric(residue,line,ordered,Order.not_less(residue,line,classified))
      %Equal.sym(Nat,residue,line,same) : {gap_at(line,inner,next_residue(_,inner)) == gap_step(gap_at(line,inner,_),inner) : Nat}
      same_gap(line,inner,positive,line_below,Nat.is_lt(1n+line,inner),{==})
    case False{} less: after_gap(line,inner,residue,residue_below,ordered,Nat.is_lt(1n+residue,inner),{==})

def step_gap(+line: Nat,+inner: Nat,+residue: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+line_below: {Nat.is_lt(line,inner) == True{} : Bool},
  +residue_below: {Nat.is_lt(residue,inner) == True{} : Bool}) -> {gap_at(line,inner,next_residue(residue,inner)) == gap_step(gap_at(line,inner,residue),inner) : Nat}:
  step_gap_by(line,inner,residue,positive,line_below,residue_below,Nat.is_le(residue,line),{==},Nat.is_lt(residue,line),{==})

def next_below_by(+residue: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},inside: Bool,+next: {Nat.is_lt(1n+residue,inner) == inside : Bool}) ->
  {Nat.is_lt(next_residue(residue,inner),inner) == True{} : Bool}:
  match inside:
    case True{}:
      %Equal.sym(Bool,Nat.is_lt(1n+residue,inner),True{},next) : {Nat.is_lt(Bool.pick(Nat,_,1n+residue,0n),inner) == True{} : Bool}
      next
    case False{}:
      %Equal.sym(Bool,Nat.is_lt(1n+residue,inner),False{},next) : {Nat.is_lt(Bool.pick(Nat,_,1n+residue,0n),inner) == True{} : Bool}
      positive

def next_below(+residue: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool}) -> {Nat.is_lt(next_residue(residue,inner),inner) == True{} : Bool}:
  next_below_by(residue,inner,positive,Nat.is_lt(1n+residue,inner),{==})

# count columns on, the gap is count steps on.
def gaps_after(count: Nat,+line: Nat,+inner: Nat,+residue: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+line_below: {Nat.is_lt(line,inner) == True{} : Bool},
  +residue_below: {Nat.is_lt(residue,inner) == True{} : Bool}) -> {gap_at(line,inner,residue_after(count,residue,inner)) == gap_after(count,gap_at(line,inner,residue),inner) : Nat}:
  match count:
    case 0n: {==}
    case 1n+rest:
      %step_gap(line,inner,residue,positive,line_below,residue_below) : {gap_at(line,inner,residue_after(rest,next_residue(residue,inner),inner)) == gap_after(rest,_,inner) : Nat}
      gaps_after(rest,line,inner,next_residue(residue,inner),positive,line_below,next_below(residue,inner,positive))

def small_remainder(+value: Nat,+inner: Nat,+below: {Nat.is_lt(value,inner) == True{} : Bool}) -> {Nat.mod(value,inner) == value : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(value,inner),(0n,value),Quotients.small(value,inner,below))

def next_mod_by(+column: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},which: Bool,+classified: {Nat.is_lt(1n+Nat.mod(column,inner),inner) == which : Bool}) ->
  {Nat.mod(1n+column,inner) == next_residue(Nat.mod(column,inner),inner) : Nat}:
  match which:
    case True{}:
      +moved = DivisionOrder.advanced_pair(column,inner,Nat.div(column,inner),Nat.mod(column,inner),True{},DivisionOrder.quotient_exact(column,inner,positive),
        DivisionOrder.remainder_small(column,inner,positive),positive,classified)
      %Equal.sym(Bool,Nat.is_lt(1n+Nat.mod(column,inner),inner),True{},classified) : {Nat.mod(1n+column,inner) == Bool.pick(Nat,_,1n+Nat.mod(column,inner),0n) : Nat}
      Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(1n+column,inner),(Nat.div(column,inner),1n+Nat.mod(column,inner)),moved)
    case False{}:
      +moved = DivisionOrder.advanced_pair(column,inner,Nat.div(column,inner),Nat.mod(column,inner),False{},DivisionOrder.quotient_exact(column,inner,positive),
        DivisionOrder.remainder_small(column,inner,positive),positive,classified)
      %Equal.sym(Bool,Nat.is_lt(1n+Nat.mod(column,inner),inner),False{},classified) : {Nat.mod(1n+column,inner) == Bool.pick(Nat,_,1n+Nat.mod(column,inner),0n) : Nat}
      Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(1n+column,inner),(1n+Nat.div(column,inner),0n),moved)

def next_mod(+column: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool}) -> {Nat.mod(1n+column,inner) == next_residue(Nat.mod(column,inner),inner) : Nat}:
  next_mod_by(column,inner,positive,Nat.is_lt(1n+Nat.mod(column,inner),inner),{==})

# count columns on, the residue is count residues on.
def mod_after(count: Nat,+column: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool}) ->
  {Nat.mod(Nat.add(column,count),inner) == residue_after(count,Nat.mod(column,inner),inner) : Nat}:
  match count:
    case 0n:
      %Equal.sym(Nat,Nat.add(column,0n),column,Addition.zero_right(column)) : {Nat.mod(_,inner) == Nat.mod(column,inner) : Nat}
      {==}
    case 1n+ +rest:
      %Equal.sym(Nat,Nat.add(column,1n+rest),1n+Nat.add(column,rest),Addition.successor_right(column,rest)) :
        {Nat.mod(_,inner) == residue_after(rest,next_residue(Nat.mod(column,inner),inner),inner) : Nat}
      %next_mod(column,inner,positive) : {Nat.mod(1n+Nat.add(column,rest),inner) == residue_after(rest,_,inner) : Nat}
      mod_after(rest,1n+column,inner,positive)

# The gap as the runtime computes it: line + inner - residue, modulo inner.
def mod_gap_by(+line: Nat,+inner: Nat,+residue: Nat,+line_below: {Nat.is_lt(line,inner) == True{} : Bool},+residue_below: {Nat.is_lt(residue,inner) == True{} : Bool},
  order: Bool,+ordered: {Nat.is_le(residue,line) == order : Bool}) -> {Nat.mod(Nat.sub(Nat.add(line,inner),residue),inner) == gap_at(line,inner,residue) : Nat}:
  match order:
    case True{}:
      +positive = Order.le_lt_trans(0n,line,inner,Order.zero_le(line),line_below)
      %Equal.sym(Bool,Nat.is_le(residue,line),True{},ordered) :
        {Nat.mod(Nat.sub(Nat.add(line,inner),residue),inner) == Bool.pick(Nat,_,Nat.sub(line,residue),Nat.sub(Nat.add(line,inner),residue)) : Nat}
      %Order.commutative(inner,line) : {Nat.mod(Nat.sub(_,residue),inner) == Nat.sub(line,residue) : Nat}
      %Equal.sym(Nat,Nat.sub(Nat.add(inner,line),residue),Nat.add(inner,Nat.sub(line,residue)),Order.subtract_from_sum(inner,line,residue,ordered)) :
        {Nat.mod(_,inner) == Nat.sub(line,residue) : Nat}
      %Addition.zero_right(inner) : {Nat.mod(Nat.add(_,Nat.sub(line,residue)),inner) == Nat.sub(line,residue) : Nat}
      %Equal.sym(Nat,Nat.mod(Nat.add(Nat.mul(1n,inner),Nat.sub(line,residue)),inner),Nat.mod(Nat.sub(line,residue),inner),DivisionOrder.remainder_shift(Nat.sub(line,residue),inner,1n,positive)) : {_ == Nat.sub(line,residue) : Nat}
      small_remainder(Nat.sub(line,residue),inner,Order.le_lt_trans(Nat.sub(line,residue),line,inner,Order.sub_le(line,residue),line_below))
    case False{}:
      %Equal.sym(Bool,Nat.is_le(residue,line),False{},ordered) :
        {Nat.mod(Nat.sub(Nat.add(line,inner),residue),inner) == Bool.pick(Nat,_,Nat.sub(line,residue),Nat.sub(Nat.add(line,inner),residue)) : Nat}
      small_remainder(Nat.sub(Nat.add(line,inner),residue),inner,
        Order.sub_lt(Nat.add(line,inner),residue,inner,lt_le(residue,Nat.add(line,inner),above_sum(residue,line,inner,residue_below)),
          Order.add_lt(line,residue,inner,le_false_lt(residue,line,ordered))))

def mod_gap(+line: Nat,+inner: Nat,+residue: Nat,+line_below: {Nat.is_lt(line,inner) == True{} : Bool},+residue_below: {Nat.is_lt(residue,inner) == True{} : Bool}) ->
  {Nat.mod(Nat.sub(Nat.add(line,inner),residue),inner) == gap_at(line,inner,residue) : Nat}:
  mod_gap_by(line,inner,residue,line_below,residue_below,Nat.is_le(residue,line),{==})

# The first cell of a line in a segment and how many it holds, as fold_lines
# computes them.
def segment_start(+line: U32,+inner: U32,+column: U32) -> U32:
  ((line + inner - column % inner) % inner : U32)

def segment_cells(+start: U32,+inner: U32,+count: U32) -> U32:
  Bool.pick(U32,U32.is_lt(start,count),((count - start + inner - 1) / inner : U32),0)

def double_room(+line: Nat,+inner: Nat,+cap: Nat,+line_below: {Nat.is_lt(line,inner) == True{} : Bool},+room: {Nat.is_le(Nat.double(inner),cap) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(line,inner),cap) == True{} : Bool}:
  Order.transitive(Nat.add(line,inner),Nat.add(inner,inner),cap,Order.add_le(line,inner,inner,lt_le(line,inner,line_below)),
    %Equal.sym(Nat,Nat.add(inner,inner),Nat.double(inner),Addition.double(inner)) : {Nat.is_le(_,cap) == True{} : Bool}
    room)

def start_exact(+line: U32,+inner: U32,+column: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +line_below: {Nat.is_lt(U32.to_nat(line),U32.to_nat(inner)) == True{} : Bool}) ->
  {U32.to_nat(segment_start(line,inner,column)) == gap_at(U32.to_nat(line),U32.to_nat(inner),Nat.mod(U32.to_nat(column),U32.to_nat(inner))) : Nat}:
  +lanes = U32.to_nat(inner)
  +residue = Nat.mod(U32.to_nat(column),lanes)
  +residue_below = DivisionOrder.remainder_small(U32.to_nat(column),lanes,positive)
  +sum_room = double_room(U32.to_nat(line),lanes,U32.to_nat(cap),line_below,room)
  +sum = Arithmetic.sum_exact(cap,cap_ok,line,inner,U32.to_nat(line),lanes,{==},{==},sum_room)
  +difference = Arithmetic.difference_exact(cap,cap_ok,(line + inner : U32),(column % inner : U32),Nat.add(U32.to_nat(line),lanes),residue,sum,
    U32Division.modulo(column,inner,cap,cap_ok,room,positive),
    Order.transitive(residue,lanes,Nat.add(U32.to_nat(line),lanes),lt_le(residue,lanes,residue_below),Order.right_le(U32.to_nat(line),lanes)),sum_room)
  Equal.trans(Nat,U32.to_nat(segment_start(line,inner,column)),Nat.mod(U32.to_nat((line + inner - column % inner : U32)),lanes),gap_at(U32.to_nat(line),lanes,residue),
    U32Division.modulo((line + inner - column % inner : U32),inner,cap,cap_ok,room,positive),
    Equal.trans(Nat,Nat.mod(U32.to_nat((line + inner - column % inner : U32)),lanes),Nat.mod(Nat.sub(Nat.add(U32.to_nat(line),lanes),residue),lanes),gap_at(U32.to_nat(line),lanes,residue),
      Equal.cong(Nat,Nat,value => Nat.mod(value,lanes),U32.to_nat((line + inner - column % inner : U32)),Nat.sub(Nat.add(U32.to_nat(line),lanes),residue),difference),
      mod_gap(U32.to_nat(line),lanes,residue,line_below,residue_below)))

def taken_count(+count: Nat,+inner: Nat,+gap: Nat,+below: {Nat.is_lt(gap,count) == True{} : Bool}) ->
  {Nat.sub(Nat.add(Nat.sub(count,gap),inner),1n) == Nat.sub(Nat.add(count,inner),1n+gap) : Nat}:
  %Equal.sym(Nat,Nat.sub(count,gap),1n+Nat.sub(count,1n+gap),less_difference(count,gap,below)) : {Nat.sub(Nat.add(_,inner),1n) == Nat.sub(Nat.add(count,inner),1n+gap) : Nat}
  %Equal.sym(Nat,Nat.sub(Nat.add(Nat.sub(count,1n+gap),inner),0n),Nat.add(Nat.sub(count,1n+gap),inner),Order.subtract_zero(Nat.add(Nat.sub(count,1n+gap),inner))) :
    {_ == Nat.sub(Nat.add(count,inner),1n+gap) : Nat}
  %Order.commutative(inner,count) : {Nat.add(Nat.sub(count,1n+gap),inner) == Nat.sub(_,1n+gap) : Nat}
  %Equal.sym(Nat,Nat.sub(Nat.add(inner,count),1n+gap),Nat.add(inner,Nat.sub(count,1n+gap)),Order.subtract_from_sum(inner,count,1n+gap,Order.strict_follow(gap,count,below))) :
    {Nat.add(Nat.sub(count,1n+gap),inner) == _ : Nat}
  Order.commutative(Nat.sub(count,1n+gap),inner)

def cells_exact_by(+start: U32,+inner: U32,+count: U32,+gap: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +start_value: {U32.to_nat(start) == gap : Nat},+count_room: {Nat.is_le(Nat.add(U32.to_nat(count),U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},
  inside: Bool,+classified: {Nat.is_lt(gap,U32.to_nat(count)) == inside : Bool}) ->
  {U32.to_nat(segment_cells(start,inner,count)) == cells_from(gap,U32.to_nat(inner),U32.to_nat(count)) : Nat}:
  match inside:
    case True{}:
      %Equal.sym(Bool,U32.is_lt(start,count),Nat.is_lt(U32.to_nat(start),U32.to_nat(count)),Comparison.lt(start,count)) :
        {U32.to_nat(Bool.pick(U32,_,((count - start + inner - 1) / inner : U32),0)) == cells_from(gap,U32.to_nat(inner),U32.to_nat(count)) : Nat}
      %Equal.sym(Nat,U32.to_nat(start),gap,start_value) :
        {U32.to_nat(Bool.pick(U32,Nat.is_lt(_,U32.to_nat(count)),((count - start + inner - 1) / inner : U32),0)) == cells_from(gap,U32.to_nat(inner),U32.to_nat(count)) : Nat}
      %Equal.sym(Bool,Nat.is_lt(gap,U32.to_nat(count)),True{},classified) :
        {U32.to_nat(Bool.pick(U32,_,((count - start + inner - 1) / inner : U32),0)) == Bool.pick(Nat,_,Nat.div(Nat.sub(Nat.add(U32.to_nat(count),U32.to_nat(inner)),1n+gap),U32.to_nat(inner)),0n) : Nat}
      +lanes = U32.to_nat(inner)
      +total = U32.to_nat(count)
      +left = Nat.sub(total,gap)
      +left_le = Order.sub_le(total,gap)
      +total_room = Order.transitive(total,Nat.add(total,lanes),U32.to_nat(cap),Order.le_add(total,lanes),count_room)
      +difference = Arithmetic.difference_exact(cap,cap_ok,count,start,total,gap,{==},start_value,lt_le(gap,total,classified),total_room)
      +sum_room = Order.transitive(Nat.add(left,lanes),Nat.add(total,lanes),U32.to_nat(cap),Order.add_le(left,total,lanes,left_le),count_room)
      +sum = Arithmetic.sum_exact(cap,cap_ok,(count - start : U32),inner,left,lanes,difference,{==},sum_room)
      +one = Order.transitive(1n,lanes,Nat.add(left,lanes),Order.strict_follow(0n,lanes,positive),Order.right_le(left,lanes))
      +less = Arithmetic.difference_exact(cap,cap_ok,(count - start + inner : U32),1,Nat.add(left,lanes),1n,sum,{==},one,sum_room)
      Equal.trans(Nat,U32.to_nat(((count - start + inner - 1) / inner : U32)),Nat.div(U32.to_nat((count - start + inner - 1 : U32)),lanes),
        Nat.div(Nat.sub(Nat.add(total,lanes),1n+gap),lanes),
        U32Division.divide((count - start + inner - 1 : U32),inner,cap,cap_ok,room,positive),
        Equal.cong(Nat,Nat,value => Nat.div(value,lanes),U32.to_nat((count - start + inner - 1 : U32)),Nat.sub(Nat.add(total,lanes),1n+gap),
          Equal.trans(Nat,U32.to_nat((count - start + inner - 1 : U32)),Nat.sub(Nat.add(left,lanes),1n),Nat.sub(Nat.add(total,lanes),1n+gap),less,
            taken_count(total,lanes,gap,classified))))
    case False{}:
      %Equal.sym(Bool,U32.is_lt(start,count),Nat.is_lt(U32.to_nat(start),U32.to_nat(count)),Comparison.lt(start,count)) :
        {U32.to_nat(Bool.pick(U32,_,((count - start + inner - 1) / inner : U32),0)) == cells_from(gap,U32.to_nat(inner),U32.to_nat(count)) : Nat}
      %Equal.sym(Nat,U32.to_nat(start),gap,start_value) :
        {U32.to_nat(Bool.pick(U32,Nat.is_lt(_,U32.to_nat(count)),((count - start + inner - 1) / inner : U32),0)) == cells_from(gap,U32.to_nat(inner),U32.to_nat(count)) : Nat}
      %Equal.sym(Bool,Nat.is_lt(gap,U32.to_nat(count)),False{},classified) :
        {U32.to_nat(Bool.pick(U32,_,((count - start + inner - 1) / inner : U32),0)) == Bool.pick(Nat,_,Nat.div(Nat.sub(Nat.add(U32.to_nat(count),U32.to_nat(inner)),1n+gap),U32.to_nat(inner)),0n) : Nat}
      {==}

def cells_exact(+start: U32,+inner: U32,+count: U32,+gap: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +start_value: {U32.to_nat(start) == gap : Nat},+count_room: {Nat.is_le(Nat.add(U32.to_nat(count),U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(segment_cells(start,inner,count)) == cells_from(gap,U32.to_nat(inner),U32.to_nat(count)) : Nat}:
  cells_exact_by(start,inner,count,gap,cap,cap_ok,room,positive,start_value,count_room,Nat.is_lt(gap,U32.to_nat(count)),{==})

def line_gap_below(+line: Nat,+inner: Nat,+residue: Nat,+line_below: {Nat.is_lt(line,inner) == True{} : Bool},+residue_below: {Nat.is_lt(residue,inner) == True{} : Bool}) ->
  {Nat.is_lt(gap_at(line,inner,residue),inner) == True{} : Bool}:
  +positive = Order.le_lt_trans(0n,line,inner,Order.zero_le(line),line_below)
  %mod_gap(line,inner,residue,line_below,residue_below) : {Nat.is_lt(_,inner) == True{} : Bool}
  DivisionOrder.remainder_small(Nat.sub(Nat.add(line,inner),residue),inner,positive)

# One line of a segment, as fold_lines folds it, takes its cells from the
# gap of the line at the column.
def segment_line(~Model: Data,~Cell: Data,~State: Data,~cell: Model -> U32 -> Cell,~step: State -> Cell -> State,+line: U32,+inner: U32,+column: U32,+count: U32,+offset: U32,+initial: State,+model: Model,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +line_below: {Nat.is_lt(U32.to_nat(line),U32.to_nat(inner)) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(U32.to_nat(count),U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool}) ->
  {strided(~Model,~Cell,~State,~cell,~step,U32.to_nat(segment_cells(segment_start(line,inner,column),inner,count)),(offset + segment_start(line,inner,column) : U32),inner,initial,model) ==
    taken(~Model,~Cell,~State,~cell,~step,U32.to_nat(count),gap_at(U32.to_nat(line),U32.to_nat(inner),Nat.mod(U32.to_nat(column),U32.to_nat(inner))),U32.to_nat(inner),offset,initial,model) : State}:
  +lanes = U32.to_nat(inner)
  +gap = gap_at(U32.to_nat(line),lanes,Nat.mod(U32.to_nat(column),lanes))
  +start_value = start_exact(line,inner,column,cap,cap_ok,room,positive,line_below)
  %Equal.sym(Nat,U32.to_nat(segment_cells(segment_start(line,inner,column),inner,count)),cells_from(gap,lanes,U32.to_nat(count)),
    cells_exact(segment_start(line,inner,column),inner,count,gap,cap,cap_ok,room,positive,start_value,count_room)) :
    {strided(~Model,~Cell,~State,~cell,~step,_,(offset + segment_start(line,inner,column) : U32),inner,initial,model) == taken(~Model,~Cell,~State,~cell,~step,U32.to_nat(count),gap,lanes,offset,initial,model) : State}
  %Arithmetic.from_nat_exact(segment_start(line,inner,column),gap,start_value) :
    {strided(~Model,~Cell,~State,~cell,~step,cells_from(gap,lanes,U32.to_nat(count)),(offset + _ : U32),inner,initial,model) == taken(~Model,~Cell,~State,~cell,~step,U32.to_nat(count),gap,lanes,offset,initial,model) : State}
  %Arithmetic.native_roundtrip(inner) :
    {strided(~Model,~Cell,~State,~cell,~step,cells_from(gap,lanes,U32.to_nat(count)),(offset + U32.from_nat(gap) : U32),_,initial,model) == taken(~Model,~Cell,~State,~cell,~step,U32.to_nat(count),gap,lanes,offset,initial,model) : State}
  gapped(~Model,~Cell,~State,~cell,~step,U32.to_nat(count),gap,lanes,offset,initial,model,
    line_gap_below(U32.to_nat(line),lanes,Nat.mod(U32.to_nat(column),lanes),line_below,DivisionOrder.remainder_small(U32.to_nat(column),lanes,positive)))

# count cells of a model from position on.
def cells(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,count: Nat,+position: U32,+model: Model) -> List<&2,Cell>:
  match count:
    case 0n: []
    case 1n+rest: cell(model,position) <> cells(~Model,~Cell,~cell,rest,(position + 1 : U32),model)

# The cells of a list taken from gap on, inner apart, folded in order.
def taken_list(~Cell: Data,~State: Data,~step: State -> Cell -> State,list: List<&2,Cell>,+gap: Nat,+inner: Nat,+initial: State) -> State:
  match list:
    case Nil{}: initial
    case Con{+head,tail}:
      match gap:
        case 0n: taken_list(~Cell,~State,~step,tail,Nat.sub(inner,1n),inner,step(initial,head))
        case 1n+ +before: taken_list(~Cell,~State,~step,tail,before,inner,initial)

def taken_cells(~Model: Data,~Cell: Data,~State: Data,~cell: Model -> U32 -> Cell,~step: State -> Cell -> State,count: Nat,+gap: Nat,+inner: Nat,+offset: U32,+initial: State,+model: Model) ->
  {taken(~Model,~Cell,~State,~cell,~step,count,gap,inner,offset,initial,model) == taken_list(~Cell,~State,~step,cells(~Model,~Cell,~cell,count,offset,model),gap,inner,initial) : State}:
  match count:
    case 0n: {==}
    case 1n+rest:
      match gap:
        case 0n: taken_cells(~Model,~Cell,~State,~cell,~step,rest,Nat.sub(inner,1n),inner,(offset + 1 : U32),step(initial,cell(model,offset)),model)
        case 1n+ +before: taken_cells(~Model,~Cell,~State,~cell,~step,rest,before,inner,(offset + 1 : U32),initial,model)

def taken_list_append(~Cell: Data,~State: Data,~step: State -> Cell -> State,first: List<&2,Cell>,+second: List<&2,Cell>,+gap: Nat,+inner: Nat,+initial: State) ->
  {taken_list(~Cell,~State,~step,List.append(&2,Cell,first,second),gap,inner,initial) ==
    taken_list(~Cell,~State,~step,second,gap_after(List.length(&2,Cell,first),gap,inner),inner,taken_list(~Cell,~State,~step,first,gap,inner,initial)) : State}:
  match first:
    case Nil{}: {==}
    case Con{+head,+tail}:
      match gap:
        case 0n: taken_list_append(~Cell,~State,~step,tail,second,Nat.sub(inner,1n),inner,step(initial,head))
        case 1n+ +before: taken_list_append(~Cell,~State,~step,tail,second,before,inner,initial)

def gap_zero(+line: Nat,+inner: Nat) -> {gap_at(line,inner,0n) == line : Nat}:
  %Equal.sym(Bool,Nat.is_le(0n,line),True{},Order.zero_le(line)) : {Bool.pick(Nat,_,Nat.sub(line,0n),Nat.sub(Nat.add(line,inner),0n)) == line : Nat}
  Order.subtract_zero(line)

# The gap of a line column cells into its block.
def column_gap(+column: Nat,+line: Nat,+inner: Nat,+positive: {Nat.is_lt(0n,inner) == True{} : Bool},+line_below: {Nat.is_lt(line,inner) == True{} : Bool}) ->
  {gap_after(column,line,inner) == gap_at(line,inner,Nat.mod(column,inner)) : Nat}:
  %Equal.sym(Nat,Nat.mod(column,inner),residue_after(column,Nat.mod(0n,inner),inner),mod_after(column,0n,inner,positive)) :
    {gap_after(column,line,inner) == gap_at(line,inner,_) : Nat}
  %Equal.sym(Nat,Nat.mod(0n,inner),0n,small_remainder(0n,inner,positive)) : {gap_after(column,line,inner) == gap_at(line,inner,residue_after(column,_,inner)) : Nat}
  %Equal.sym(Nat,gap_at(line,inner,residue_after(column,0n,inner)),gap_after(column,gap_at(line,inner,0n),inner),gaps_after(column,line,inner,0n,positive,line_below,positive)) :
    {gap_after(column,line,inner) == _ : Nat}
  %Equal.sym(Nat,gap_at(line,inner,0n),line,gap_zero(line,inner)) : {gap_after(column,line,inner) == gap_after(column,_,inner) : Nat}
  {==}

# One line of a segment takes its cells after the cells of the block before.
def line_progressed(~Model: Data,~Cell: Data,~State: Data,~cell: Model -> U32 -> Cell,~step: State -> Cell -> State,+line: U32,+position: Nat,+inner: U32,+column: U32,+count: U32,+offset: U32,+group: List<&2,Cell>,+initial: State,+model: Model,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(U32.to_nat(count),U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},
  +line_value: {U32.to_nat(line) == position : Nat},+line_below: {Nat.is_lt(position,U32.to_nat(inner)) == True{} : Bool},
  +column_value: {U32.to_nat(column) == List.length(&2,Cell,group) : Nat}) ->
  {strided(~Model,~Cell,~State,~cell,~step,U32.to_nat(segment_cells(segment_start(line,inner,column),inner,count)),(offset + segment_start(line,inner,column) : U32),inner,
      taken_list(~Cell,~State,~step,group,position,U32.to_nat(inner),initial),model) ==
    taken_list(~Cell,~State,~step,List.append(&2,Cell,group,cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model)),position,U32.to_nat(inner),initial) : State}:
  +lanes = U32.to_nat(inner)
  +before = taken_list(~Cell,~State,~step,group,position,lanes,initial)
  +segment = cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model)
  +start = segment_start(line,inner,column)
  +taken_cells_count = U32.to_nat(segment_cells(start,inner,count))
  %Equal.sym(State,taken_list(~Cell,~State,~step,List.append(&2,Cell,group,segment),position,lanes,initial),
    taken_list(~Cell,~State,~step,segment,gap_after(List.length(&2,Cell,group),position,lanes),lanes,before),
    taken_list_append(~Cell,~State,~step,group,segment,position,lanes,initial)) :
    {strided(~Model,~Cell,~State,~cell,~step,taken_cells_count,(offset + start : U32),inner,before,model) == _ : State}
  %Equal.sym(Nat,gap_after(List.length(&2,Cell,group),position,lanes),gap_at(position,lanes,Nat.mod(List.length(&2,Cell,group),lanes)),
    column_gap(List.length(&2,Cell,group),position,lanes,positive,line_below)) :
    {strided(~Model,~Cell,~State,~cell,~step,taken_cells_count,(offset + start : U32),inner,before,model) == taken_list(~Cell,~State,~step,segment,_,lanes,before) : State}
  %column_value : {strided(~Model,~Cell,~State,~cell,~step,taken_cells_count,(offset + start : U32),inner,before,model) == taken_list(~Cell,~State,~step,segment,gap_at(position,lanes,Nat.mod(_,lanes)),lanes,before) : State}
  %taken_cells(~Model,~Cell,~State,~cell,~step,U32.to_nat(count),gap_at(position,lanes,Nat.mod(U32.to_nat(column),lanes)),lanes,offset,before,model) :
    {strided(~Model,~Cell,~State,~cell,~step,taken_cells_count,(offset + start : U32),inner,before,model) == _ : State}
  %line_value : {strided(~Model,~Cell,~State,~cell,~step,taken_cells_count,(offset + start : U32),inner,before,model) ==
    taken(~Model,~Cell,~State,~cell,~step,U32.to_nat(count),gap_at(_,lanes,Nat.mod(U32.to_nat(column),lanes)),lanes,offset,before,model) : State}
  segment_line(~Model,~Cell,~State,~cell,~step,line,inner,column,count,offset,before,model,cap,cap_ok,room,positive,
    %Equal.sym(Nat,U32.to_nat(line),position,line_value) : {Nat.is_lt(_,lanes) == True{} : Bool}
    line_below,count_room)

def lanes_room(+lanes: Nat,+cap: Nat,+room: {Nat.is_le(Nat.double(lanes),cap) == True{} : Bool}) -> {Nat.is_le(lanes,cap) == True{} : Bool}:
  Order.transitive(lanes,Nat.add(lanes,lanes),cap,Order.le_add(lanes,lanes),
    %Equal.sym(Nat,Nat.add(lanes,lanes),Nat.double(lanes),Addition.double(lanes)) : {Nat.is_le(_,cap) == True{} : Bool}
    room)

# strided as the fold of a block model.
def strided_model(~Model: Data,~Cell: Data,~State: Data,~cell: Model -> U32 -> Cell,~step: State -> Cell -> State,count: Nat,position: U32,stride: U32,initial: State,model: Model) -> State:
  strided(~Model,~Cell,~State,~cell,~step,count,position,stride,initial,model)

def cells_split(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,first: Nat,+second: Nat,+position: U32,+model: Model) ->
  {cells(~Model,~Cell,~cell,Nat.add(first,second),position,model) ==
    List.append(&2,Cell,cells(~Model,~Cell,~cell,first,position,model),cells(~Model,~Cell,~cell,second,(position + U32.from_nat(first) : U32),model)) : List<&2,Cell>}:
  match first:
    case 0n:
      %Equal.sym(U32,(position + 0 : U32),position,Arithmetic.add_zero(position)) :
        {cells(~Model,~Cell,~cell,second,position,model) == cells(~Model,~Cell,~cell,second,_,model) : List<&2,Cell>}
      {==}
    case 1n+ +rest:
      %Equal.sym(U32,(position + U32.from_nat(1n+rest) : U32),((position + 1 : U32) + U32.from_nat(rest) : U32),next_position(position,rest)) :
        {cells(~Model,~Cell,~cell,1n+Nat.add(rest,second),position,model) ==
          cell(model,position) <> List.append(&2,Cell,cells(~Model,~Cell,~cell,rest,(position + 1 : U32),model),cells(~Model,~Cell,~cell,second,_,model)) : List<&2,Cell>}
      %Equal.sym(List<&2,Cell>,cells(~Model,~Cell,~cell,Nat.add(rest,second),(position + 1 : U32),model),
        List.append(&2,Cell,cells(~Model,~Cell,~cell,rest,(position + 1 : U32),model),cells(~Model,~Cell,~cell,second,((position + 1 : U32) + U32.from_nat(rest) : U32),model)),
        cells_split(~Model,~Cell,~cell,rest,second,(position + 1 : U32),model)) :
        {cell(model,position) <> _ ==
          cell(model,position) <> List.append(&2,Cell,cells(~Model,~Cell,~cell,rest,(position + 1 : U32),model),cells(~Model,~Cell,~cell,second,((position + 1 : U32) + U32.from_nat(rest) : U32),model)) :
          List<&2,Cell>}
      {==}

def append_one(~Element: Data,+items: List<&2,Element>,+item: Element,+rest: List<&2,Element>) ->
  {List.append(&2,Element,List.append(&2,Element,items,[item]),rest) == List.append(&2,Element,items,item <> rest) : List<&2,Element>}:
  match items:
    case Nil{}: {==}
    case Con{+head,+tail}:
      %Equal.sym(List<&2,Element>,List.append(&2,Element,List.append(&2,Element,tail,[item]),rest),List.append(&2,Element,tail,item <> rest),append_one(~Element,tail,item,rest)) :
        {head <> _ == head <> List.append(&2,Element,tail,item <> rest) : List<&2,Element>}
      {==}

def two_le(+value: Nat,+rest: Nat) -> {Nat.is_le(2n+value,Nat.add(value,2n+rest)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(value,2n+rest),1n+Nat.add(value,1n+rest),Addition.successor_right(value,1n+rest)) : {Nat.is_le(2n+value,_) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(value,1n+rest),1n+Nat.add(value,rest),Addition.successor_right(value,rest)) : {Nat.is_le(2n+value,1n+_) == True{} : Bool}
  Order.le_add(value,rest)

def one_more_below(+size: Nat,+total: Nat,+grown: Nat,+grown_length: {grown == Nat.add(size,1n) : Nat},+inside: {Nat.is_lt(1n+size,total) == True{} : Bool}) ->
  {Nat.is_lt(grown,total) == True{} : Bool}:
  %Equal.sym(Nat,grown,Nat.add(size,1n),grown_length) : {Nat.is_lt(_,total) == True{} : Bool}
  %Order.commutative(1n,size) : {Nat.is_lt(_,total) == True{} : Bool}
  inside

def one_more_fits(+size: Nat,+rest: Nat,+total: Nat,+grown: Nat,+grown_length: {grown == Nat.add(size,1n) : Nat},+fits: {Nat.is_le(Nat.add(size,2n+rest),total) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(grown,1n+rest),total) == True{} : Bool}:
  %Equal.sym(Nat,grown,Nat.add(size,1n),grown_length) : {Nat.is_le(Nat.add(_,1n+rest),total) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.add(size,1n),1n+rest),Nat.add(size,2n+rest),Addition.associative(size,1n,1n+rest)) : {Nat.is_le(_,total) == True{} : Bool}
  fits

def cells_length(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,count: Nat,+position: U32,+model: Model) ->
  {List.length(&2,Cell,cells(~Model,~Cell,~cell,count,position,model)) == count : Nat}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Nat,List.length(&2,Cell,cells(~Model,~Cell,~cell,rest,(position + 1 : U32),model)),rest,cells_length(~Model,~Cell,~cell,rest,(position + 1 : U32),model)) :
        {1n+_ == 1n+rest : Nat}
      {==}

# The cells of remaining words from offset on are count of them, then the
# rest from offset + count on.
def cells_taken(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,+remaining: U32,+count: U32,+offset: U32,+model: Model,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(count),U32.to_nat(remaining)) == True{} : Bool},+bounded: {Nat.is_le(U32.to_nat(remaining),U32.to_nat(cap)) == True{} : Bool}) ->
  {cells(~Model,~Cell,~cell,U32.to_nat(remaining),offset,model) ==
    List.append(&2,Cell,cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model),cells(~Model,~Cell,~cell,U32.to_nat((remaining - count : U32)),(offset + count : U32),model)) :
    List<&2,Cell>}:
  +taken = U32.to_nat(count)
  +left = Nat.sub(U32.to_nat(remaining),taken)
  %Order.restore_shift(taken,U32.to_nat(remaining),ordered) :
    {cells(~Model,~Cell,~cell,_,offset,model) ==
      List.append(&2,Cell,cells(~Model,~Cell,~cell,taken,offset,model),cells(~Model,~Cell,~cell,U32.to_nat((remaining - count : U32)),(offset + count : U32),model)) : List<&2,Cell>}
  %Equal.sym(Nat,U32.to_nat((remaining - count : U32)),left,Arithmetic.difference_exact(cap,cap_ok,remaining,count,U32.to_nat(remaining),taken,{==},{==},ordered,bounded)) :
    {cells(~Model,~Cell,~cell,Nat.add(taken,left),offset,model) ==
      List.append(&2,Cell,cells(~Model,~Cell,~cell,taken,offset,model),cells(~Model,~Cell,~cell,_,(offset + count : U32),model)) : List<&2,Cell>}
  %Arithmetic.native_roundtrip(count) :
    {cells(~Model,~Cell,~cell,Nat.add(taken,left),offset,model) ==
      List.append(&2,Cell,cells(~Model,~Cell,~cell,taken,offset,model),cells(~Model,~Cell,~cell,left,(offset + _ : U32),model)) : List<&2,Cell>}
  cells_split(~Model,~Cell,~cell,taken,left,offset,model)

def one_times(+value: Nat) -> {Nat.mul(1n,value) == value : Nat}:
  Equal.trans(Nat,Nat.mul(1n,value),Nat.mul(value,1n),value,RowMajor.multiply_commutative(1n,value),Order.times_one(value))

def zero_difference(+small: Nat,+large: Nat,ordered: {Nat.is_le(small,large) == True{} : Bool}) -> {Nat.is_eq(Nat.sub(large,small),0n) == Nat.is_eq(small,large) : Bool}:
  match small large:
    case 0n 0n: {==}
    case 0n 1n+large: {==}
    case 1n+small 0n: Empty.absurd({Nat.is_eq(Nat.sub(0n,1n+small),0n) == Nat.is_eq(1n+small,0n) : Bool},WordEncoding.false_true(ordered))
    case 1n+ +small 1n+ +large: zero_difference(small,large,ordered)

# The rest of the cells after a block filled in turns one block fewer.
def turned_quotient(+size: Nat,+remaining: Nat,+total: Nat,+filled: {Nat.is_lt(0n,total) == True{} : Bool},+short: {Nat.is_le(size,total) == True{} : Bool},
  +over: {Nat.is_le(Nat.sub(total,size),remaining) == True{} : Bool}) ->
  {Nat.div(Nat.add(size,remaining),total) == 1n+Nat.div(Nat.sub(remaining,Nat.sub(total,size)),total) : Nat}:
  +left = Nat.sub(total,size)
  +rest = Nat.sub(remaining,left)
  %Order.restore_shift(left,remaining,over) : {Nat.div(Nat.add(size,_),total) == 1n+Nat.div(rest,total) : Nat}
  %Addition.associative(size,left,rest) : {Nat.div(_,total) == 1n+Nat.div(rest,total) : Nat}
  %Equal.sym(Nat,Nat.add(size,left),total,Order.restore_shift(size,total,short)) : {Nat.div(Nat.add(_,rest),total) == 1n+Nat.div(rest,total) : Nat}
  %one_times(total) : {Nat.div(Nat.add(_,rest),total) == 1n+Nat.div(rest,total) : Nat}
  DivisionOrder.quotient_shift(rest,total,1n,filled)

def inside_sum(+size: Nat,+remaining: Nat,+total: Nat,+short: {Nat.is_le(size,total) == True{} : Bool},+below: {Nat.is_lt(remaining,Nat.sub(total,size)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(size,remaining),total) == True{} : Bool}:
  %Order.restore_shift(size,total,short) : {Nat.is_lt(Nat.add(size,remaining),_) == True{} : Bool}
  %Order.commutative(remaining,size) : {Nat.is_lt(_,Nat.add(size,Nat.sub(total,size))) == True{} : Bool}
  %Order.commutative(Nat.sub(total,size),size) : {Nat.is_lt(Nat.add(remaining,size),_) == True{} : Bool}
  Order.add_lt(remaining,Nat.sub(total,size),size,below)

def lt_by(+value: Nat,+same: Nat,+limit: Nat,+equal: {value == same : Nat},+below: {Nat.is_lt(same,limit) == True{} : Bool}) -> {Nat.is_lt(value,limit) == True{} : Bool}:
  %Equal.sym(Nat,value,same,equal) : {Nat.is_lt(_,limit) == True{} : Bool}
  below

def le_by(+value: Nat,+same: Nat,+limit: Nat,+equal: {value == same : Nat},+below: {Nat.is_le(same,limit) == True{} : Bool}) -> {Nat.is_le(value,limit) == True{} : Bool}:
  %Equal.sym(Nat,value,same,equal) : {Nat.is_le(_,limit) == True{} : Bool}
  below

def not_lt_by(+value: Nat,+same: Nat,+limit: Nat,+equal: {value == same : Nat},+above: {Nat.is_lt(same,limit) == False{} : Bool}) -> {Nat.is_lt(value,limit) == False{} : Bool}:
  %Equal.sym(Nat,value,same,equal) : {Nat.is_lt(_,limit) == False{} : Bool}
  above

def word_equal_false(+left: U32,+right: U32,+value: Nat,+exact: {U32.to_nat(left) == value : Nat},+less: {Nat.is_lt(value,U32.to_nat(right)) == True{} : Bool}) ->
  {U32.is_eq(left,right) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(left,right),Nat.is_eq(U32.to_nat(left),U32.to_nat(right)),Comparison.eq(left,right)) : {_ == False{} : Bool}
  %Equal.sym(Nat,U32.to_nat(left),value,exact) : {Nat.is_eq(_,U32.to_nat(right)) == False{} : Bool}
  Order.less_not_equal(value,U32.to_nat(right),less)

def word_equal_true(+left: U32,+right: U32,+value: Nat,+left_exact: {U32.to_nat(left) == value : Nat},+right_exact: {U32.to_nat(right) == value : Nat}) ->
  {U32.is_eq(left,right) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(left,right),Nat.is_eq(U32.to_nat(left),U32.to_nat(right)),Comparison.eq(left,right)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(left),value,left_exact) : {Nat.is_eq(_,U32.to_nat(right)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(right),value,right_exact) : {Nat.is_eq(value,_) == True{} : Bool}
  Order.equal_reflexive(value)

def word_is_zero(+value: U32,+zero: {U32.to_nat(value) == 0n : Nat}) -> {U32.is_zero(value) == True{} : Bool}:
  word_equal_true(value,0,0n,zero,{==})

def block_shift(+count: Nat,+total: Nat,+filled: {Nat.is_lt(0n,total) == True{} : Bool}) -> {Nat.div(Nat.add(total,count),total) == 1n+Nat.div(count,total) : Nat}:
  %one_times(total) : {Nat.div(Nat.add(_,count),total) == 1n+Nat.div(count,total) : Nat}
  DivisionOrder.quotient_shift(count,total,1n,filled)

# Cells after a block in progress cross at most one block more than the cells
# alone.
def quotient_step(+size: Nat,+count: Nat,+total: Nat,+filled: {Nat.is_lt(0n,total) == True{} : Bool},+short: {Nat.is_le(size,total) == True{} : Bool}) ->
  {Nat.is_le(Nat.div(Nat.add(size,count),total),1n+Nat.div(count,total)) == True{} : Bool}:
  %block_shift(count,total,filled) : {Nat.is_le(Nat.div(Nat.add(size,count),total),_) == True{} : Bool}
  DivisionOrder.quotient_le(Nat.add(size,count),Nat.add(total,count),total,filled,Order.add_le(size,total,count,short))

# The cells of the blocks of a tree, in order.
def stream(~Model: Data,~Cell: Data,~cell: Model -> U32 -> Cell,tree: Partition.Tree<&2,Model,Unit>) -> List<&2,Cell>:
  match tree:
    case Partition.Leaf{model,first,+count,details}: cells(~Model,~Cell,~cell,U32.to_nat(count),0,model)
    case Partition.Fork{left,right}: List.append(&2,Cell,stream(~Model,~Cell,~cell,left),stream(~Model,~Cell,~cell,right))

def take_drop(~Element: Data,+items: List<&2,Element>,+count: Nat) ->
  {List.append(&2,Element,List.take(&2,Element,items,count),List.drop(&2,Element,items,count)) == items : List<&2,Element>}:
  match items count:
    case [] count: {==}
    case +head <> +tail 0n: {==}
    case +head <> +tail 1n+ +previous:
      Equal.cong(List<&2,Element>,List<&2,Element>,others => head <> others,
        List.append(&2,Element,List.take(&2,Element,tail,previous),List.drop(&2,Element,tail,previous)),tail,take_drop(~Element,tail,previous))

def take_length(~Element: Data,+items: List<&2,Element>,+count: Nat,+inside: {Nat.is_le(count,List.length(&2,Element,items)) == True{} : Bool}) ->
  {List.length(&2,Element,List.take(&2,Element,items,count)) == count : Nat}:
  match items count:
    case [] 0n: {==}
    case [] 1n+previous: Empty.absurd({List.length(&2,Element,List.take(&2,Element,[],1n+previous)) == 1n+previous : Nat},WordEncoding.false_true(inside))
    case +head <> +tail 0n: {==}
    case +head <> +tail 1n+ +previous:
      Equal.cong(Nat,Nat,value => 1n+value,List.length(&2,Element,List.take(&2,Element,tail,previous)),previous,take_length(~Element,tail,previous,inside))

def drop_length(~Element: Data,+items: List<&2,Element>,+count: Nat) -> {List.length(&2,Element,List.drop(&2,Element,items,count)) == Nat.sub(List.length(&2,Element,items),count) : Nat}:
  match items count:
    case [] count: Equal.sym(Nat,Nat.sub(0n,count),0n,Order.zero_subtract(count))
    case +head <> +tail 0n: Equal.sym(Nat,Nat.sub(1n+List.length(&2,Element,tail),0n),1n+List.length(&2,Element,tail),Order.subtract_zero(1n+List.length(&2,Element,tail)))
    case +head <> +tail 1n+ +previous: drop_length(~Element,tail,previous)

def empty_of_length(~Element: Data,+items: List<&2,Element>,+zero: {List.length(&2,Element,items) == 0n : Nat}) -> {items == [] : List<&2,Element>}:
  match items:
    case []: {==}
    case +head <> +tail:
      Empty.absurd({head <> tail == [] : List<&2,Element>},
        WordEncoding.false_true(Equal.cong(Nat,Bool,value => Nat.is_eq(value,0n),1n+List.length(&2,Element,tail),0n,zero)))

def le_up(+value: Nat,+limit: Nat,+same: Nat,+equal: {limit == same : Nat},+below: {Nat.is_le(value,same) == True{} : Bool}) -> {Nat.is_le(value,limit) == True{} : Bool}:
  %Equal.sym(Nat,limit,same,equal) : {Nat.is_le(value,_) == True{} : Bool}
  below

def lt_up(+value: Nat,+limit: Nat,+same: Nat,+equal: {limit == same : Nat},+below: {Nat.is_lt(value,same) == True{} : Bool}) -> {Nat.is_lt(value,limit) == True{} : Bool}:
  %Equal.sym(Nat,limit,same,equal) : {Nat.is_lt(value,_) == True{} : Bool}
  below

def at_or(~Context: Data,list: List<&2,Context>,index: Nat,+fallback: Context) -> Context:
  match list index:
    case Nil{} index: fallback
    case Con{head,tail} 0n: head
    case Con{head,tail} 1n+previous: at_or(~Context,tail,previous,fallback)

def drop_zero(~Element: Data,+items: List<&2,Element>) -> {List.drop(&2,Element,items,0n) == items : List<&2,Element>}:
  match items:
    case []: {==}
    case +head <> +tail: {==}

def drop_cons(~Element: Data,+items: List<&2,Element>,+index: Nat,+fallback: Element,+inside: {Nat.is_lt(index,List.length(&2,Element,items)) == True{} : Bool}) ->
  {List.drop(&2,Element,items,index) == at_or(~Element,items,index,fallback) <> List.drop(&2,Element,items,1n+index) : List<&2,Element>}:
  match items index:
    case [] index:
      Empty.absurd({List.drop(&2,Element,[],index) == at_or(~Element,[],index,fallback) <> List.drop(&2,Element,[],1n+index) : List<&2,Element>},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(index,0n),True{},Equal.sym(Bool,Nat.is_lt(index,0n),False{},Quotients.lt_zero(index)),inside)))
    case +head <> +tail 0n:
      %Equal.sym(List<&2,Element>,List.drop(&2,Element,tail,0n),tail,drop_zero(~Element,tail)) : {head <> tail == head <> _ : List<&2,Element>}
      {==}
    case +head <> +tail 1n+ +previous: drop_cons(~Element,tail,previous,fallback,inside)

def drop_beyond(~Element: Data,+items: List<&2,Element>,+index: Nat,+beyond: {Nat.is_le(List.length(&2,Element,items),index) == True{} : Bool}) ->
  {List.drop(&2,Element,items,index) == [] : List<&2,Element>}:
  empty_of_length(~Element,List.drop(&2,Element,items,index),
    Equal.trans(Nat,List.length(&2,Element,List.drop(&2,Element,items,index)),Nat.sub(List.length(&2,Element,items),index),0n,drop_length(~Element,items,index),
      Order.subtract_below(List.length(&2,Element,items),index,beyond)))

# ---- The lines of a block, by values ----

# The values of remaining lines from line on, each folding the cells of group
# taken from its line on, inner apart, from its start.
def lines_taken(~Cell: Data,~Value: Data,~step: Value -> Cell -> Value,remaining: Nat,+line: Nat,+inner: Nat,+group: List<&2,Cell>,+starts: List<&2,Value>) -> List<&2,Value>:
  match remaining starts:
    case 0n starts: []
    case 1n+rest Nil{}: []
    case 1n+rest Con{+start,+tail}: taken_list(~Cell,~Value,~step,group,line,inner,start) <> lines_taken(~Cell,~Value,~step,rest,1n+line,inner,group,tail)

def turned_values(~Value: Data,~Final: Data,~turn_value: Value -> Final,values: List<&2,Value>) -> List<&2,Final>:
  match values:
    case Nil{}: []
    case Con{value,rest}: turn_value(value) <> turned_values(~Value,~Final,~turn_value,rest)

# The finals of the lines of a block.
def group_lines(~Cell: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~turn_value: Value -> Final,remaining: Nat,+line: Nat,+inner: Nat,+group: List<&2,Cell>,
  +starts: List<&2,Value>) -> List<&2,Final>:
  turned_values(~Value,~Final,~turn_value,lines_taken(~Cell,~Value,~step,remaining,line,inner,group,starts))

# The values the next remaining lines begin from: the contexts taken in turn,
# all again once they run out.
def drawn(~Context: Data,~Value: Data,~begin_value: Context -> Value,remaining: Nat,+all: List<&2,Context>,+upcoming: List<&2,Context>) -> List<&2,Value>:
  match remaining upcoming:
    case 1n+rest Con{+context,+later}: begin_value(context) <> drawn(~Context,~Value,~begin_value,rest,all,LineChain.refilled(~Context,all,later))
    case remaining upcoming: []

def after_drawn(~Context: Data,remaining: Nat,+all: List<&2,Context>,+upcoming: List<&2,Context>) -> List<&2,Context>:
  match remaining upcoming:
    case 1n+rest Con{+context,+later}: after_drawn(~Context,rest,all,LineChain.refilled(~Context,all,later))
    case remaining upcoming: upcoming

# The fold after the lines of a segment: taken the values they folded, folded
# the lines carried before them, rest the starts after them.
def lines_end(~Context: Data,~Value: Data,~Final: Data,~turn_value: Value -> Final,ends: Bool,+next: U32,+upcoming: List<&2,Context>,+taken: List<&2,Value>,+folded: List<&2,Value>,
  +rest: List<&2,Value>,+finals: List<&2,Final>) -> Folded<Context,Value,Final>:
  match ends:
    case True{}: Folded{next,upcoming,onto_values(~Value,folded,rest),onto_values(~Final,turned_values(~Value,~Final,~turn_value,taken),finals)}
    case False{}: Folded{next,upcoming,onto_values(~Value,folded,List.append(&2,Value,taken,rest)),finals}

def lines_end_step(~Context: Data,~Value: Data,~Final: Data,~turn_value: Value -> Final,ends: Bool,+next: U32,+upcoming: List<&2,Context>,+value: Value,+taken: List<&2,Value>,
  +folded: List<&2,Value>,+rest: List<&2,Value>,+finals: List<&2,Final>) ->
  {lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming,taken,placed_folded(~Value,ends,folded,value),rest,placed_finals(~Value,~Final,~turn_value,ends,finals,value)) ==
    lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming,value <> taken,folded,rest,finals) : Folded<Context,Value,Final>}:
  match ends:
    case True{}: {==}
    case False{}: {==}

def lines_end_none(~Context: Data,~Value: Data,~Final: Data,~turn_value: Value -> Final,ends: Bool,+next: U32,+upcoming: List<&2,Context>,+folded: List<&2,Value>,
  +rest: List<&2,Value>,+finals: List<&2,Final>) ->
  {lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming,[],folded,rest,finals) == Folded{next,upcoming,onto_values(~Value,folded,rest),finals} : Folded<Context,Value,Final>}:
  match ends:
    case True{}: {==}
    case False{}: {==}

# The lines of a block begun before take the cells of a segment after the
# cells of the block before it.
def lines_carried(~Model: Data,~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~cell: Model -> U32 -> Cell,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,remaining: Nat,+ends: Bool,+line: U32,+position: Nat,+inner: U32,+column: U32,+offset: U32,+count: U32,+next: U32,+all: List<&2,Context>,
  +upcoming: List<&2,Context>,+group: List<&2,Cell>,+starts: List<&2,Value>,+folded: List<&2,Value>,+finals: List<&2,Final>,+model: Model,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(U32.to_nat(count),U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},+line_value: {U32.to_nat(line) == position : Nat},
  +lines_below: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(inner)) == True{} : Bool},+column_value: {U32.to_nat(column) == List.length(&2,Cell,group) : Nat}) ->
  {lines_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,remaining,False{},ends,line,inner,column,offset,count,next,all,upcoming,lines_taken(~Cell,~Value,~step,remaining,position,U32.to_nat(inner),group,starts),folded,finals,
      model) ==
    lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming,
      lines_taken(~Cell,~Value,~step,remaining,position,U32.to_nat(inner),List.append(&2,Cell,group,cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model)),starts),folded,[],finals) : Folded<Context,Value,Final>}:
  match remaining starts:
    case 0n starts:
      Equal.sym(Folded<Context,Value,Final>,lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming,[],folded,[],finals),Folded{next,upcoming,onto_values(~Value,folded,[]),finals},
        lines_end_none(~Context,~Value,~Final,~turn_value,ends,next,upcoming,folded,[],finals))
    case 1n+rest Nil{}:
      Equal.sym(Folded<Context,Value,Final>,lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming,[],folded,[],finals),Folded{next,upcoming,onto_values(~Value,folded,[]),finals},
        lines_end_none(~Context,~Value,~Final,~turn_value,ends,next,upcoming,folded,[],finals))
    case 1n+ +rest Con{+start,+tail}:
      +lanes = U32.to_nat(inner)
      +extended = List.append(&2,Cell,group,cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model))
      +value = taken_list(~Cell,~Value,~step,extended,position,lanes,start)
      +taken = lines_taken(~Cell,~Value,~step,rest,1n+position,lanes,extended,tail)
      +inside = Order.transitive(1n+position,Nat.add(position,1n+rest),lanes,
        %Equal.sym(Nat,Nat.add(position,1n+rest),1n+Nat.add(position,rest),Addition.successor_right(position,rest)) : {Nat.is_le(1n+position,_) == True{} : Bool}
        Order.le_add(position,rest),lines_below)
      +next_value = Equal.trans(Nat,U32.to_nat((line + 1 : U32)),Nat.add(position,1n),1n+position,
        Arithmetic.sum_exact(cap,cap_ok,line,1,position,1n,line_value,{==},
          %Order.commutative(1n,position) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
          Order.transitive(1n+position,lanes,U32.to_nat(cap),inside,lanes_room(lanes,U32.to_nat(cap),room))),
        Order.commutative(position,1n))
      %Equal.sym(Value,
        strided(~Model,~Cell,~Value,~cell,~step,U32.to_nat(segment_cells(segment_start(line,inner,column),inner,count)),(offset + segment_start(line,inner,column) : U32),inner,
          taken_list(~Cell,~Value,~step,group,position,lanes,start),model),value,
        line_progressed(~Model,~Cell,~Value,~cell,~step,line,position,inner,column,count,offset,group,start,model,cap,cap_ok,room,positive,count_room,line_value,
          succ_le_lt(position,lanes,inside),column_value)) :
        {lines_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,lines_taken(~Cell,~Value,~step,rest,1n+position,lanes,group,tail),
            placed_folded(~Value,ends,folded,_),placed_finals(~Value,~Final,~turn_value,ends,finals,_),model) ==
          lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming,value <> taken,folded,[],finals) : Folded<Context,Value,Final>}
      Equal.trans(Folded<Context,Value,Final>,
        lines_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,rest,False{},ends,(line + 1 : U32),inner,column,offset,count,next,all,upcoming,lines_taken(~Cell,~Value,~step,rest,1n+position,lanes,group,tail),
          placed_folded(~Value,ends,folded,value),placed_finals(~Value,~Final,~turn_value,ends,finals,value),model),
        lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming,taken,placed_folded(~Value,ends,folded,value),[],placed_finals(~Value,~Final,~turn_value,ends,finals,value)),
        lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming,value <> taken,folded,[],finals),
        lines_carried(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,rest,ends,(line + 1 : U32),1n+position,inner,column,offset,count,next,all,upcoming,group,tail,placed_folded(~Value,ends,folded,value),
          placed_finals(~Value,~Final,~turn_value,ends,finals,value),model,cap,cap_ok,room,positive,count_room,next_value,
          %Addition.successor_right(position,rest) : {Nat.is_le(_,lanes) == True{} : Bool}
          lines_below,column_value),
        lines_end_step(~Context,~Value,~Final,~turn_value,ends,next,upcoming,value,taken,folded,[],finals))

# The lines of a block that begins in a segment take their values from the
# contexts and fold the cells of the segment.
def lines_begun(~Model: Data,~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~cell: Model -> U32 -> Cell,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,remaining: Nat,+ends: Bool,+line: U32,+position: Nat,+inner: U32,+column: U32,+offset: U32,+count: U32,+next: U32,+all: List<&2,Context>,
  +upcoming: List<&2,Context>,+starts: List<&2,Value>,+folded: List<&2,Value>,+finals: List<&2,Final>,+model: Model,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(inner)) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(U32.to_nat(count),U32.to_nat(inner)),U32.to_nat(cap)) == True{} : Bool},+line_value: {U32.to_nat(line) == position : Nat},
  +lines_below: {Nat.is_le(Nat.add(position,remaining),U32.to_nat(inner)) == True{} : Bool},+column_value: {U32.to_nat(column) == 0n : Nat}) ->
  {lines_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,remaining,True{},ends,line,inner,column,offset,count,next,all,upcoming,starts,folded,finals,model) ==
    lines_end(~Context,~Value,~Final,~turn_value,ends,next,after_drawn(~Context,remaining,all,upcoming),
      lines_taken(~Cell,~Value,~step,remaining,position,U32.to_nat(inner),cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model),
        drawn(~Context,~Value,~begin_value,remaining,all,upcoming)),folded,starts,finals) : Folded<Context,Value,Final>}:
  match remaining upcoming:
    case 0n upcoming:
      Equal.sym(Folded<Context,Value,Final>,lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming,[],folded,starts,finals),Folded{next,upcoming,onto_values(~Value,folded,starts),finals},
        lines_end_none(~Context,~Value,~Final,~turn_value,ends,next,upcoming,folded,starts,finals))
    case 1n+rest Nil{}:
      Equal.sym(Folded<Context,Value,Final>,lines_end(~Context,~Value,~Final,~turn_value,ends,next,[],[],folded,starts,finals),Folded{next,[],onto_values(~Value,folded,starts),finals},
        lines_end_none(~Context,~Value,~Final,~turn_value,ends,next,[],folded,starts,finals))
    case 1n+ +rest Con{+context,+later}:
      +lanes = U32.to_nat(inner)
      +segment = cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model)
      +refilled = LineChain.refilled(~Context,all,later)
      +value = taken_list(~Cell,~Value,~step,segment,position,lanes,begin_value(context))
      +taken = lines_taken(~Cell,~Value,~step,rest,1n+position,lanes,segment,drawn(~Context,~Value,~begin_value,rest,all,refilled))
      +after = after_drawn(~Context,rest,all,refilled)
      +inside = Order.transitive(1n+position,Nat.add(position,1n+rest),lanes,
        %Equal.sym(Nat,Nat.add(position,1n+rest),1n+Nat.add(position,rest),Addition.successor_right(position,rest)) : {Nat.is_le(1n+position,_) == True{} : Bool}
        Order.le_add(position,rest),lines_below)
      +next_value = Equal.trans(Nat,U32.to_nat((line + 1 : U32)),Nat.add(position,1n),1n+position,
        Arithmetic.sum_exact(cap,cap_ok,line,1,position,1n,line_value,{==},
          %Order.commutative(1n,position) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
          Order.transitive(1n+position,lanes,U32.to_nat(cap),inside,lanes_room(lanes,U32.to_nat(cap),room))),
        Order.commutative(position,1n))
      %Equal.sym(Value,
        strided(~Model,~Cell,~Value,~cell,~step,U32.to_nat(segment_cells(segment_start(line,inner,column),inner,count)),(offset + segment_start(line,inner,column) : U32),inner,
          begin_value(context),model),value,
        line_progressed(~Model,~Cell,~Value,~cell,~step,line,position,inner,column,count,offset,[],begin_value(context),model,cap,cap_ok,room,positive,count_room,line_value,
          succ_le_lt(position,lanes,inside),column_value)) :
        {lines_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,refilled,starts,
            placed_folded(~Value,ends,folded,_),placed_finals(~Value,~Final,~turn_value,ends,finals,_),model) ==
          lines_end(~Context,~Value,~Final,~turn_value,ends,next,after,value <> taken,folded,starts,finals) : Folded<Context,Value,Final>}
      Equal.trans(Folded<Context,Value,Final>,
        lines_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,rest,True{},ends,(line + 1 : U32),inner,column,offset,count,next,all,refilled,starts,
          placed_folded(~Value,ends,folded,value),placed_finals(~Value,~Final,~turn_value,ends,finals,value),model),
        lines_end(~Context,~Value,~Final,~turn_value,ends,next,after,taken,placed_folded(~Value,ends,folded,value),starts,placed_finals(~Value,~Final,~turn_value,ends,finals,value)),
        lines_end(~Context,~Value,~Final,~turn_value,ends,next,after,value <> taken,folded,starts,finals),
        lines_begun(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,rest,ends,(line + 1 : U32),1n+position,inner,column,offset,count,next,all,refilled,starts,placed_folded(~Value,ends,folded,value),
          placed_finals(~Value,~Final,~turn_value,ends,finals,value),model,cap,cap_ok,room,positive,count_room,next_value,
          %Addition.successor_right(position,rest) : {Nat.is_le(_,lanes) == True{} : Bool}
          lines_below,column_value),
        lines_end_step(~Context,~Value,~Final,~turn_value,ends,next,after,value,taken,folded,starts,finals))

# ---- A walk, one cell at a time ----

# The cells of the block in progress, the contexts of the lines after it, the
# values its lines began from and the finals of the lines done, last first.
type Fed<-Cell: Data,-Context: Data,-Value: Data,-Final: Data> is Data:
  Fed{group: List<&2,Cell>,upcoming: List<&2,Context>,current: List<&2,Value>,finals: List<&2,Final>}

def fed_group(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,fed: Fed<Cell,Context,Value,Final>) -> List<&2,Cell>:
  Fed{group,upcoming,current,finals} = fed
  group

# The values the lines of the block in progress folded its cells into.
def started(~Cell: Data,~Value: Data,~step: Value -> Cell -> Value,+lanes: Nat,group: List<&2,Cell>,+current: List<&2,Value>) -> List<&2,Value>:
  match group:
    case Nil{}: []
    case Con{+held,+others}: lines_taken(~Cell,~Value,~step,lanes,0n,lanes,held <> others,current)

def started_some(~Cell: Data,~Value: Data,~step: Value -> Cell -> Value,+lanes: Nat,+group: List<&2,Cell>,+current: List<&2,Value>,
  +nonempty: {Nat.is_lt(0n,List.length(&2,Cell,group)) == True{} : Bool}) ->
  {started(~Cell,~Value,~step,lanes,group,current) == lines_taken(~Cell,~Value,~step,lanes,0n,lanes,group,current) : List<&2,Value>}:
  match group:
    case Nil{}: Empty.absurd({started(~Cell,~Value,~step,lanes,[],current) == lines_taken(~Cell,~Value,~step,lanes,0n,lanes,[],current) : List<&2,Value>},WordEncoding.false_true(nonempty))
    case Con{+held,+others}: {==}

def folded_of(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+line: LineChain.Line,fed: Fed<Cell,Context,Value,Final>) -> Folded<Context,Value,Final>:
  Fed{+group,+upcoming,+current,+finals} = fed
  Folded{U32.from_nat(List.length(&2,Cell,group)),upcoming,started(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),group,current),finals}

# A block that begins draws the values of its lines.
def begun_upcoming(~Cell: Data,~Context: Data,group: List<&2,Cell>,+lanes: Nat,+all: List<&2,Context>,+upcoming: List<&2,Context>) -> List<&2,Context>:
  match group:
    case Nil{}: after_drawn(~Context,lanes,all,upcoming)
    case Con{held,others}: upcoming

def begun_current(~Cell: Data,~Context: Data,~Value: Data,~begin_value: Context -> Value,group: List<&2,Cell>,+lanes: Nat,+all: List<&2,Context>,+upcoming: List<&2,Context>,
  +current: List<&2,Value>) -> List<&2,Value>:
  match group:
    case Nil{}: drawn(~Context,~Value,~begin_value,lanes,all,upcoming)
    case Con{held,others}: current

def fed_begun(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+all: List<&2,Context>,+line: LineChain.Line,fed: Fed<Cell,Context,Value,Final>) -> Fed<Cell,Context,Value,Final>:
  Fed{+group,+upcoming,+current,+finals} = fed
  Fed{group,begun_upcoming(~Cell,~Context,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming),begun_current(~Cell,~Context,~Value,~begin_value,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming,current),finals}

# The block in progress after a cell joined it: kept while short of a whole
# block, turned when whole.
def fed_after(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+line: LineChain.Line,below: Bool,+group: List<&2,Cell>,+upcoming: List<&2,Context>,+current: List<&2,Value>,+finals: List<&2,Final>) -> Fed<Cell,Context,Value,Final>:
  match below:
    case True{}: Fed{group,upcoming,current,finals}
    case False{}: Fed{[],upcoming,[],onto_values(~Final,group_lines(~Cell,~Value,~Final,~step,~turn_value,U32.to_nat(LineChain.inner_of(line)),0n,U32.to_nat(LineChain.inner_of(line)),group,current),finals)}

def fed_cell(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+line: LineChain.Line,+head: Cell,fed: Fed<Cell,Context,Value,Final>) -> Fed<Cell,Context,Value,Final>:
  Fed{+group,+upcoming,+current,+finals} = fed
  fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,List.append(&2,Cell,group,[head])),U32.to_nat(LineChain.cells_of(line))),List.append(&2,Cell,group,[head]),upcoming,current,finals)

# The cells of list after the cells walked, one at a time.
def feed(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+all: List<&2,Context>,+line: LineChain.Line,list: List<&2,Cell>,fed: Fed<Cell,Context,Value,Final>) -> Fed<Cell,Context,Value,Final>:
  match list:
    case Nil{}: fed
    case Con{+head,tail}: feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,tail,fed_cell(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,head,fed_begun(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,fed)))

def feed_append(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+all: List<&2,Context>,+line: LineChain.Line,first: List<&2,Cell>,+second: List<&2,Cell>,+fed: Fed<Cell,Context,Value,Final>) ->
  {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,List.append(&2,Cell,first,second),fed) == feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,second,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,first,fed)) : Fed<Cell,Context,Value,Final>}:
  match first:
    case Nil{}: {==}
    case Con{+head,tail}: feed_append(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,tail,second,fed_cell(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,head,fed_begun(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,fed)))

# A segment that fits in the block in progress joins it, and turns it when it
# fills it. A block that begins draws first.
def feed_segment(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+all: List<&2,Context>,+line: LineChain.Line,+rest: List<&2,Cell>,+group: List<&2,Cell>,+upcoming: List<&2,Context>,+current: List<&2,Value>,
  +finals: List<&2,Final>,+head: Cell,+fits: {Nat.is_le(Nat.add(List.length(&2,Cell,group),1n+List.length(&2,Cell,rest)),U32.to_nat(LineChain.cells_of(line))) == True{} : Bool}) ->
  {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,head <> rest,Fed{group,upcoming,current,finals}) ==
    fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,List.append(&2,Cell,group,head <> rest)),U32.to_nat(LineChain.cells_of(line))),List.append(&2,Cell,group,head <> rest),begun_upcoming(~Cell,~Context,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming),begun_current(~Cell,~Context,~Value,~begin_value,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming,current),finals) :
    Fed<Cell,Context,Value,Final>}:
  match rest group:
    case Nil{} Nil{}: {==}
    case Nil{} Con{+held,+others}: {==}
    case Con{+next,+tail} Nil{}:
      +total = U32.to_nat(LineChain.cells_of(line))
      +inside = succ_le_lt(1n,total,Order.transitive(2n,2n+List.length(&2,Cell,tail),total,Order.le_add(2n,List.length(&2,Cell,tail)),fits))
      %Equal.sym(Bool,Nat.is_lt(1n,total),True{},inside) :
        {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,next <> tail,fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,_,[head],after_drawn(~Context,U32.to_nat(LineChain.inner_of(line)),all,upcoming),drawn(~Context,~Value,~begin_value,U32.to_nat(LineChain.inner_of(line)),all,upcoming),finals)) ==
          fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,head <> next <> tail),total),head <> next <> tail,after_drawn(~Context,U32.to_nat(LineChain.inner_of(line)),all,upcoming),
            drawn(~Context,~Value,~begin_value,U32.to_nat(LineChain.inner_of(line)),all,upcoming),finals) : Fed<Cell,Context,Value,Final>}
      feed_segment(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,tail,[head],after_drawn(~Context,U32.to_nat(LineChain.inner_of(line)),all,upcoming),drawn(~Context,~Value,~begin_value,U32.to_nat(LineChain.inner_of(line)),all,upcoming),finals,next,fits)
    case Con{+next,+tail} Con{+held,+others}:
      +total = U32.to_nat(LineChain.cells_of(line))
      +before = {held <> others : List<&2,Cell>}
      +size = List.length(&2,Cell,before)
      +grown = List.append(&2,Cell,before,[head])
      +grown_length = ListReadback.append_length(~Cell,before,[head])
      +inside = succ_le_lt(1n+size,total,Order.transitive(2n+size,Nat.add(size,2n+List.length(&2,Cell,tail)),total,two_le(size,List.length(&2,Cell,tail)),fits))
      +grown_below = one_more_below(size,total,List.length(&2,Cell,grown),grown_length,inside)
      +grown_fits = one_more_fits(size,List.length(&2,Cell,tail),total,List.length(&2,Cell,grown),grown_length,fits)
      %Equal.sym(Bool,Nat.is_lt(List.length(&2,Cell,grown),total),True{},grown_below) :
        {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,next <> tail,fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,_,grown,upcoming,current,finals)) ==
          fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,List.append(&2,Cell,before,head <> next <> tail)),total),List.append(&2,Cell,before,head <> next <> tail),upcoming,current,finals) :
          Fed<Cell,Context,Value,Final>}
      %append_one(~Cell,before,head,next <> tail) :
        {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,next <> tail,Fed{grown,upcoming,current,finals}) == fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,_),total),_,upcoming,current,finals) : Fed<Cell,Context,Value,Final>}
      feed_segment(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,tail,grown,upcoming,current,finals,next,grown_fits)

def feed_some(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+all: List<&2,Context>,+line: LineChain.Line,+group: List<&2,Cell>,+upcoming: List<&2,Context>,+current: List<&2,Value>,+finals: List<&2,Final>,
  +segment: List<&2,Cell>,+nonempty: {Nat.is_lt(0n,List.length(&2,Cell,segment)) == True{} : Bool},
  +fits: {Nat.is_le(Nat.add(List.length(&2,Cell,group),List.length(&2,Cell,segment)),U32.to_nat(LineChain.cells_of(line))) == True{} : Bool}) ->
  {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,segment,Fed{group,upcoming,current,finals}) ==
    fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,List.append(&2,Cell,group,segment)),U32.to_nat(LineChain.cells_of(line))),List.append(&2,Cell,group,segment),begun_upcoming(~Cell,~Context,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming),begun_current(~Cell,~Context,~Value,~begin_value,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming,current),finals) : Fed<Cell,Context,Value,Final>}:
  match segment:
    case Nil{}:
      Empty.absurd({feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,[],Fed{group,upcoming,current,finals}) ==
          fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,List.append(&2,Cell,group,[])),U32.to_nat(LineChain.cells_of(line))),List.append(&2,Cell,group,[]),begun_upcoming(~Cell,~Context,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming),begun_current(~Cell,~Context,~Value,~begin_value,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming,current),finals) : Fed<Cell,Context,Value,Final>},
        WordEncoding.false_true(nonempty))
    case Con{+head,+rest}: feed_segment(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,rest,group,upcoming,current,finals,head,fits)

def after_short(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+line: LineChain.Line,below: Bool,+group: List<&2,Cell>,+upcoming: List<&2,Context>,+current: List<&2,Value>,+finals: List<&2,Final>,
  +filled: {Nat.is_lt(0n,U32.to_nat(LineChain.cells_of(line))) == True{} : Bool},+classified: {Nat.is_lt(List.length(&2,Cell,group),U32.to_nat(LineChain.cells_of(line))) == below : Bool}) ->
  {Nat.is_lt(List.length(&2,Cell,fed_group(~Cell,~Context,~Value,~Final,fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,below,group,upcoming,current,finals))),U32.to_nat(LineChain.cells_of(line))) == True{} : Bool}:
  match below:
    case True{}: classified
    case False{}: filled

# The block in progress stays short of a whole block.
def feed_short(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+all: List<&2,Context>,+line: LineChain.Line,list: List<&2,Cell>,+fed: Fed<Cell,Context,Value,Final>,+filled: {Nat.is_lt(0n,U32.to_nat(LineChain.cells_of(line))) == True{} : Bool},
  +short: {Nat.is_lt(List.length(&2,Cell,fed_group(~Cell,~Context,~Value,~Final,fed)),U32.to_nat(LineChain.cells_of(line))) == True{} : Bool}) ->
  {Nat.is_lt(List.length(&2,Cell,fed_group(~Cell,~Context,~Value,~Final,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,list,fed))),U32.to_nat(LineChain.cells_of(line))) == True{} : Bool}:
  match list:
    case Nil{}: short
    case Con{+head,tail}:
      match fed:
        case Fed{+group,+upcoming,+current,+finals}:
          +grown = List.append(&2,Cell,group,[head])
          feed_short(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,tail,fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,grown),U32.to_nat(LineChain.cells_of(line))),grown,begun_upcoming(~Cell,~Context,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming),begun_current(~Cell,~Context,~Value,~begin_value,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming,current),finals),filled,
            after_short(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,grown),U32.to_nat(LineChain.cells_of(line))),grown,begun_upcoming(~Cell,~Context,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming),begun_current(~Cell,~Context,~Value,~begin_value,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming,current),finals,filled,{==}))

# ---- The segments of one block ----

def segments_done(~Model: Data,~Context: Data,~Value: Data,~Final: Data,~begin_value: Context -> Value,~fold_value: Nat -> U32 -> U32 -> Value -> Model -> Value,~turn_value: Value -> Final,fuel: Nat,+lines: LineChain.Lines<Context>,+offset: U32,+remaining: U32,+folded: Folded<Context,Value,Final>,+model: Model) ->
  {segments_model(~Model,~Context,~Value,~Final,~begin_value,~fold_value,~turn_value,fuel,True{},lines,offset,remaining,folded,model) == folded : Folded<Context,Value,Final>}:
  match fuel:
    case 0n: {==}
    case 1n+rest: {==}

def word_not_zero(+value: U32,+rest: Nat,+exact: {U32.to_nat(value) == 1n+rest : Nat}) -> {U32.is_zero(value) == False{} : Bool}:
  Arithmetic.word_positive_nonzero(value,
    %Equal.sym(Nat,U32.to_nat(value),1n+rest,exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    {==})

# The lines of a segment from the column of the block in progress, which they
# begin from the contexts when the block begins here.
def segment_lines(~Model: Data,~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~cell: Model -> U32 -> Cell,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+ends: Bool,+all: List<&2,Context>,+line: LineChain.Line,+column: U32,+offset: U32,+count: U32,+next: U32,+group: List<&2,Cell>,
  +upcoming: List<&2,Context>,+current: List<&2,Value>,+finals: List<&2,Final>,+model: Model,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(U32.to_nat(LineChain.inner_of(line))),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(LineChain.inner_of(line))) == True{} : Bool},
  +count_room: {Nat.is_le(Nat.add(U32.to_nat(count),U32.to_nat(LineChain.inner_of(line))),U32.to_nat(cap)) == True{} : Bool},+column_value: {U32.to_nat(column) == List.length(&2,Cell,group) : Nat}) ->
  {lines_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,U32.to_nat(LineChain.inner_of(line)),U32.is_zero(column),ends,0,LineChain.inner_of(line),column,offset,count,next,all,upcoming,started(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),group,current),[],finals,model) ==
    lines_end(~Context,~Value,~Final,~turn_value,ends,next,begun_upcoming(~Cell,~Context,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming),
      lines_taken(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),0n,U32.to_nat(LineChain.inner_of(line)),List.append(&2,Cell,group,cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model)),begun_current(~Cell,~Context,~Value,~begin_value,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming,current)),[],[],finals) : Folded<Context,Value,Final>}:
  match group:
    case Nil{}:
      %Equal.sym(Bool,U32.is_zero(column),True{},word_is_zero(column,column_value)) :
        {lines_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,U32.to_nat(LineChain.inner_of(line)),_,ends,0,LineChain.inner_of(line),column,offset,count,next,all,upcoming,[],[],finals,model) ==
          lines_end(~Context,~Value,~Final,~turn_value,ends,next,after_drawn(~Context,U32.to_nat(LineChain.inner_of(line)),all,upcoming),
            lines_taken(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),0n,U32.to_nat(LineChain.inner_of(line)),cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model),drawn(~Context,~Value,~begin_value,U32.to_nat(LineChain.inner_of(line)),all,upcoming)),[],[],finals) :
          Folded<Context,Value,Final>}
      lines_begun(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,U32.to_nat(LineChain.inner_of(line)),ends,0,0n,LineChain.inner_of(line),column,offset,count,next,all,upcoming,[],[],finals,model,cap,cap_ok,room,positive,count_room,{==},
        Order.reflexive(U32.to_nat(LineChain.inner_of(line))),column_value)
    case Con{+held,+others}:
      %Equal.sym(Bool,U32.is_zero(column),False{},word_not_zero(column,List.length(&2,Cell,others),column_value)) :
        {lines_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,U32.to_nat(LineChain.inner_of(line)),_,ends,0,LineChain.inner_of(line),column,offset,count,next,all,upcoming,
            lines_taken(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),0n,U32.to_nat(LineChain.inner_of(line)),held <> others,current),[],finals,model) ==
          lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming,
            lines_taken(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),0n,U32.to_nat(LineChain.inner_of(line)),List.append(&2,Cell,held <> others,cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model)),current),[],[],finals) : Folded<Context,Value,Final>}
      lines_carried(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,U32.to_nat(LineChain.inner_of(line)),ends,0,0n,LineChain.inner_of(line),column,offset,count,next,all,upcoming,held <> others,current,[],finals,model,cap,cap_ok,room,positive,count_room,
        {==},Order.reflexive(U32.to_nat(LineChain.inner_of(line))),column_value)

# The segments of the cells of a block from offset on, as fold_segments folds
# them, feed its cells one at a time. which tells whether the cells end before
# the block in progress.
def segments_fed(~Model: Data,~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~cell: Model -> U32 -> Cell,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,fuel: Nat,done: Bool,which: Bool,+all: List<&2,Context>,+line: LineChain.Line,+offset: U32,+remaining: U32,+column: U32,+group: List<&2,Cell>,
  +upcoming: List<&2,Context>,+current: List<&2,Value>,+finals: List<&2,Final>,+model: Model,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(U32.to_nat(LineChain.inner_of(line))),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(LineChain.inner_of(line))) == True{} : Bool},+block_room: {Nat.is_le(Nat.add(U32.to_nat(LineChain.cells_of(line)),U32.to_nat(LineChain.inner_of(line))),U32.to_nat(cap)) == True{} : Bool},+filled: {Nat.is_lt(0n,U32.to_nat(LineChain.cells_of(line))) == True{} : Bool},+cells_room: {Nat.is_le(Nat.double(U32.to_nat(LineChain.cells_of(line))),U32.to_nat(cap)) == True{} : Bool},
  +remaining_room: {Nat.is_le(U32.to_nat(remaining),U32.to_nat(cap)) == True{} : Bool},+column_value: {U32.to_nat(column) == List.length(&2,Cell,group) : Nat},
  +short: {Nat.is_lt(List.length(&2,Cell,group),U32.to_nat(LineChain.cells_of(line))) == True{} : Bool},+done_value: {U32.is_zero(remaining) == done : Bool},
  +classified: {Nat.is_lt(U32.to_nat(remaining),Nat.sub(U32.to_nat(LineChain.cells_of(line)),List.length(&2,Cell,group))) == which : Bool},
  +enough: {Nat.is_le(1n+Nat.div(Nat.add(List.length(&2,Cell,group),U32.to_nat(remaining)),U32.to_nat(LineChain.cells_of(line))),fuel) == True{} : Bool}) ->
  {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,done,LineChain.Lines{all,line},offset,remaining,Folded{column,upcoming,started(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),group,current),finals},model) ==
    folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,U32.to_nat(remaining),offset,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}:
  match fuel done which:
    case 0n done which:
      Empty.absurd({segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,0n,done,LineChain.Lines{all,line},offset,remaining,Folded{column,upcoming,started(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),group,current),finals},model) ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,U32.to_nat(remaining),offset,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>},
        WordEncoding.false_true(enough))
    case 1n+ +fuel True{} which:
      %Equal.sym(U32,remaining,0,Arithmetic.word_zero(remaining,done_value)) :
        {Folded{column,upcoming,started(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),group,current),finals} ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,U32.to_nat(_),offset,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}
      %Arithmetic.from_nat_exact(column,List.length(&2,Cell,group),column_value) :
        {Folded{_,upcoming,started(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),group,current),finals} == Folded{U32.from_nat(List.length(&2,Cell,group)),upcoming,started(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),group,current),finals} : Folded<Context,Value,Final>}
      {==}
    case 1n+ +fuel False{} True{}:
      +lanes = U32.to_nat(LineChain.inner_of(line))
      +total = U32.to_nat(LineChain.cells_of(line))
      +size = List.length(&2,Cell,group)
      +left = (LineChain.cells_of(line) - column : U32)
      +count = U32.min(remaining,left)
      +segment = cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model)
      +rest = cells(~Model,~Cell,~cell,U32.to_nat((remaining - count : U32)),(offset + count : U32),model)
      +grown = List.append(&2,Cell,group,segment)
      +taken = U32.to_nat(remaining)
      +ends = U32.is_eq((column + count : U32),LineChain.cells_of(line))
      +next = Bool.pick(U32,ends,0,(column + count : U32))
      +upcoming_after = begun_upcoming(~Cell,~Context,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming)
      +current_after = begun_current(~Cell,~Context,~Value,~begin_value,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming,current)
      +lined = lines_taken(~Cell,~Value,~step,lanes,0n,lanes,grown,current_after)
      +fits = lt_le(size,total,short)
      +total_room = Order.transitive(total,Nat.add(total,lanes),U32.to_nat(cap),Order.le_add(total,lanes),block_room)
      +left_value = Arithmetic.difference_exact(cap,cap_ok,LineChain.cells_of(line),column,total,size,{==},column_value,fits,total_room)
      +count_min = Equal.trans(Nat,U32.to_nat(count),Nat.min(taken,U32.to_nat(left)),Nat.min(taken,Nat.sub(total,size)),
        NativeExtrema.minimum(remaining,left),Equal.cong(Nat,Nat,value => Nat.min(taken,value),U32.to_nat(left),Nat.sub(total,size),left_value))
      +count_value = Equal.trans(Nat,U32.to_nat(count),Nat.min(taken,Nat.sub(total,size)),taken,count_min,
        Extrema.minimum_when_ordered(taken,Nat.sub(total,size),lt_le(taken,Nat.sub(total,size),classified)))
      +inside = inside_sum(size,taken,total,fits,classified)
      +below_total = Order.transitive(taken,Nat.sub(total,size),total,lt_le(taken,Nat.sub(total,size),classified),Order.sub_le(total,size))
      +sum_value = Arithmetic.sum_exact(cap,cap_ok,column,count,size,taken,column_value,count_value,
        Order.transitive(Nat.add(size,taken),total,U32.to_nat(cap),lt_le(Nat.add(size,taken),total,inside),total_room))
      +segment_length = Equal.trans(Nat,List.length(&2,Cell,segment),U32.to_nat(count),taken,cells_length(~Model,~Cell,~cell,U32.to_nat(count),offset,model),count_value)
      +grown_length = Equal.trans(Nat,List.length(&2,Cell,grown),Nat.add(size,List.length(&2,Cell,segment)),Nat.add(size,taken),
        ListReadback.append_length(~Cell,group,segment),Equal.cong(Nat,Nat,value => Nat.add(size,value),List.length(&2,Cell,segment),taken,segment_length))
      +grown_short = lt_by(List.length(&2,Cell,grown),Nat.add(size,taken),total,grown_length,inside)
      +column_next = Equal.trans(Nat,U32.to_nat((column + count : U32)),Nat.add(size,taken),List.length(&2,Cell,grown),sum_value,
        Equal.sym(Nat,List.length(&2,Cell,grown),Nat.add(size,taken),grown_length))
      +full_false = word_equal_false((column + count : U32),LineChain.cells_of(line),Nat.add(size,taken),sum_value,inside)
      +done_true = word_equal_true(count,remaining,taken,count_value,{==})
      +left_zero = Equal.trans(Nat,U32.to_nat((remaining - count : U32)),Nat.sub(taken,taken),0n,
        Arithmetic.difference_exact(cap,cap_ok,remaining,count,taken,taken,{==},count_value,Order.reflexive(taken),remaining_room),Order.subtract_below(taken,taken,Order.reflexive(taken)))
      +taken_positive = Arithmetic.word_nonzero_positive(remaining,done_value)
      +segment_some = {%Equal.sym(Nat,List.length(&2,Cell,segment),taken,segment_length) : {Nat.is_lt(0n,_) == True{} : Bool}
        taken_positive : {Nat.is_lt(0n,List.length(&2,Cell,segment)) == True{} : Bool}}
      +grown_some = {%Equal.sym(Nat,List.length(&2,Cell,grown),Nat.add(size,List.length(&2,Cell,segment)),ListReadback.append_length(~Cell,group,segment)) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.lt_le(0n,List.length(&2,Cell,segment),Nat.add(size,List.length(&2,Cell,segment)),segment_some,Order.right_le(size,List.length(&2,Cell,segment))) :
        {Nat.is_lt(0n,List.length(&2,Cell,grown)) == True{} : Bool}}
      +count_room = le_by(Nat.add(U32.to_nat(count),lanes),Nat.add(taken,lanes),U32.to_nat(cap),Equal.cong(Nat,Nat,value => Nat.add(value,lanes),U32.to_nat(count),taken,count_value),
        Order.transitive(Nat.add(taken,lanes),Nat.add(total,lanes),U32.to_nat(cap),Order.add_le(taken,total,lanes,below_total),block_room))
      %Equal.sym(Folded<Context,Value,Final>,lines_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,lanes,U32.is_zero(column),ends,0,LineChain.inner_of(line),column,offset,count,next,all,upcoming,started(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),group,current),[],finals,model),
        lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming_after,lined,[],[],finals),
        segment_lines(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,ends,all,line,column,offset,count,next,group,upcoming,current,finals,model,cap,cap_ok,room,positive,count_room,column_value)) :
        {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),_,model) ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,taken,offset,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}
      %Equal.sym(Bool,ends,False{},full_false) :
        {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),
            lines_end(~Context,~Value,~Final,~turn_value,_,Bool.pick(U32,_,0,(column + count : U32)),upcoming_after,lined,[],[],finals),model) ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,taken,offset,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}
      %Equal.sym(Bool,U32.is_eq(count,remaining),True{},done_true) :
        {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,_,LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),
            lines_end(~Context,~Value,~Final,~turn_value,False{},(column + count : U32),upcoming_after,lined,[],[],finals),model) ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,taken,offset,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}
      %Equal.sym(Folded<Context,Value,Final>,
        segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,True{},LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),
          lines_end(~Context,~Value,~Final,~turn_value,False{},(column + count : U32),upcoming_after,lined,[],[],finals),model),
        lines_end(~Context,~Value,~Final,~turn_value,False{},(column + count : U32),upcoming_after,lined,[],[],finals),
        segments_done(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),
          lines_end(~Context,~Value,~Final,~turn_value,False{},(column + count : U32),upcoming_after,lined,[],[],finals),model)) :
        {_ == folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,taken,offset,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}
      %Equal.sym(List<&2,Cell>,cells(~Model,~Cell,~cell,taken,offset,model),List.append(&2,Cell,segment,rest),
        cells_taken(~Model,~Cell,~cell,remaining,count,offset,model,cap,cap_ok,le_by(U32.to_nat(count),taken,taken,count_value,Order.reflexive(taken)),remaining_room)) :
        {Folded{(column + count : U32),upcoming_after,List.append(&2,Value,lined,[]),finals} == folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,_,Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}
      %Equal.sym(Fed<Cell,Context,Value,Final>,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,List.append(&2,Cell,segment,rest),Fed{group,upcoming,current,finals}),
        feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,rest,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,segment,Fed{group,upcoming,current,finals})),feed_append(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,segment,rest,Fed{group,upcoming,current,finals})) :
        {Folded{(column + count : U32),upcoming_after,List.append(&2,Value,lined,[]),finals} == folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,_) : Folded<Context,Value,Final>}
      %Equal.sym(Nat,U32.to_nat((remaining - count : U32)),0n,left_zero) :
        {Folded{(column + count : U32),upcoming_after,List.append(&2,Value,lined,[]),finals} ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,_,(offset + count : U32),model),feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,segment,Fed{group,upcoming,current,finals}))) : Folded<Context,Value,Final>}
      %Equal.sym(Fed<Cell,Context,Value,Final>,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,segment,Fed{group,upcoming,current,finals}),
        fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,grown),total),grown,upcoming_after,current_after,finals),
        feed_some(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,group,upcoming,current,finals,segment,segment_some,
          le_by(Nat.add(size,List.length(&2,Cell,segment)),Nat.add(size,taken),total,Equal.cong(Nat,Nat,value => Nat.add(size,value),List.length(&2,Cell,segment),taken,segment_length),
            lt_le(Nat.add(size,taken),total,inside)))) :
        {Folded{(column + count : U32),upcoming_after,List.append(&2,Value,lined,[]),finals} == folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,_) : Folded<Context,Value,Final>}
      %Equal.sym(Bool,Nat.is_lt(List.length(&2,Cell,grown),total),True{},grown_short) :
        {Folded{(column + count : U32),upcoming_after,List.append(&2,Value,lined,[]),finals} ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,_,grown,upcoming_after,current_after,finals)) : Folded<Context,Value,Final>}
      %Arithmetic.from_nat_exact((column + count : U32),List.length(&2,Cell,grown),column_next) :
        {Folded{_,upcoming_after,List.append(&2,Value,lined,[]),finals} ==
          Folded{U32.from_nat(List.length(&2,Cell,grown)),upcoming_after,started(~Cell,~Value,~step,lanes,grown,current_after),finals} : Folded<Context,Value,Final>}
      %Equal.sym(List<&2,Value>,started(~Cell,~Value,~step,lanes,grown,current_after),lined,started_some(~Cell,~Value,~step,lanes,grown,current_after,grown_some)) :
        {Folded{U32.from_nat(List.length(&2,Cell,grown)),upcoming_after,List.append(&2,Value,lined,[]),finals} == Folded{U32.from_nat(List.length(&2,Cell,grown)),upcoming_after,_,finals} : Folded<Context,Value,Final>}
      %Equal.sym(List<&2,Value>,List.append(&2,Value,lined,[]),lined,ListReadback.append_empty(~Value,lined)) :
        {Folded{U32.from_nat(List.length(&2,Cell,grown)),upcoming_after,_,finals} == Folded{U32.from_nat(List.length(&2,Cell,grown)),upcoming_after,lined,finals} : Folded<Context,Value,Final>}
      {==}
    case 1n+ +fuel False{} False{}:
      +lanes = U32.to_nat(LineChain.inner_of(line))
      +total = U32.to_nat(LineChain.cells_of(line))
      +size = List.length(&2,Cell,group)
      +left = (LineChain.cells_of(line) - column : U32)
      +count = U32.min(remaining,left)
      +segment = cells(~Model,~Cell,~cell,U32.to_nat(count),offset,model)
      +rest = cells(~Model,~Cell,~cell,U32.to_nat((remaining - count : U32)),(offset + count : U32),model)
      +grown = List.append(&2,Cell,group,segment)
      +taken = U32.to_nat(remaining)
      +gap = Nat.sub(total,size)
      +ends = U32.is_eq((column + count : U32),LineChain.cells_of(line))
      +next = Bool.pick(U32,ends,0,(column + count : U32))
      +upcoming_after = begun_upcoming(~Cell,~Context,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming)
      +current_after = begun_current(~Cell,~Context,~Value,~begin_value,group,U32.to_nat(LineChain.inner_of(line)),all,upcoming,current)
      +lined = lines_taken(~Cell,~Value,~step,lanes,0n,lanes,grown,current_after)
      +turned = onto_values(~Final,group_lines(~Cell,~Value,~Final,~step,~turn_value,lanes,0n,lanes,grown,current_after),finals)
      +fits = lt_le(size,total,short)
      +total_room = Order.transitive(total,Nat.add(total,lanes),U32.to_nat(cap),Order.le_add(total,lanes),block_room)
      +left_value = Arithmetic.difference_exact(cap,cap_ok,LineChain.cells_of(line),column,total,size,{==},column_value,fits,total_room)
      +over = Order.not_less(taken,gap,classified)
      +count_min = Equal.trans(Nat,U32.to_nat(count),Nat.min(taken,U32.to_nat(left)),Nat.min(taken,gap),
        NativeExtrema.minimum(remaining,left),Equal.cong(Nat,Nat,value => Nat.min(taken,value),U32.to_nat(left),gap,left_value))
      +count_value = Equal.trans(Nat,U32.to_nat(count),Nat.min(taken,gap),gap,count_min,
        Equal.trans(Nat,Nat.min(taken,gap),Nat.min(gap,taken),gap,Extrema.minimum_commutative(taken,gap),Extrema.minimum_when_ordered(gap,taken,over)))
      +whole = Order.restore_shift(size,total,fits)
      +sum_value = Equal.trans(Nat,U32.to_nat((column + count : U32)),Nat.add(size,gap),total,
        Arithmetic.sum_exact(cap,cap_ok,column,count,size,gap,column_value,count_value,le_by(Nat.add(size,gap),total,U32.to_nat(cap),whole,total_room)),whole)
      +segment_length = Equal.trans(Nat,List.length(&2,Cell,segment),U32.to_nat(count),gap,cells_length(~Model,~Cell,~cell,U32.to_nat(count),offset,model),count_value)
      +grown_length = Equal.trans(Nat,List.length(&2,Cell,grown),Nat.add(size,List.length(&2,Cell,segment)),total,
        ListReadback.append_length(~Cell,group,segment),
        Equal.trans(Nat,Nat.add(size,List.length(&2,Cell,segment)),Nat.add(size,gap),total,
          Equal.cong(Nat,Nat,value => Nat.add(size,value),List.length(&2,Cell,segment),gap,segment_length),whole))
      +grown_full = not_lt_by(List.length(&2,Cell,grown),total,total,grown_length,Order.not_below(total,total,Order.reflexive(total)))
      +full_true = word_equal_true((column + count : U32),LineChain.cells_of(line),total,sum_value,{==})
      +rest_value = Arithmetic.difference_exact(cap,cap_ok,remaining,count,taken,gap,{==},count_value,over,remaining_room)
      +done_same = Equal.trans(Bool,U32.is_zero((remaining - count : U32)),Nat.is_eq(U32.to_nat((remaining - count : U32)),0n),U32.is_eq(count,remaining),
        Comparison.eq((remaining - count : U32),0),
        Equal.trans(Bool,Nat.is_eq(U32.to_nat((remaining - count : U32)),0n),Nat.is_eq(Nat.sub(taken,gap),0n),U32.is_eq(count,remaining),
          Equal.cong(Nat,Bool,value => Nat.is_eq(value,0n),U32.to_nat((remaining - count : U32)),Nat.sub(taken,gap),rest_value),
          Equal.trans(Bool,Nat.is_eq(Nat.sub(taken,gap),0n),Nat.is_eq(gap,taken),U32.is_eq(count,remaining),zero_difference(gap,taken,over),
            Equal.sym(Bool,U32.is_eq(count,remaining),Nat.is_eq(gap,taken),
              Equal.trans(Bool,U32.is_eq(count,remaining),Nat.is_eq(U32.to_nat(count),taken),Nat.is_eq(gap,taken),Comparison.eq(count,remaining),
                Equal.cong(Nat,Bool,value => Nat.is_eq(value,taken),U32.to_nat(count),gap,count_value))))))
      +gap_positive = Order.positive_difference(total,size,short)
      +segment_some = {%Equal.sym(Nat,List.length(&2,Cell,segment),gap,segment_length) : {Nat.is_lt(0n,_) == True{} : Bool}
        gap_positive : {Nat.is_lt(0n,List.length(&2,Cell,segment)) == True{} : Bool}}
      +count_room = le_by(Nat.add(U32.to_nat(count),lanes),Nat.add(gap,lanes),U32.to_nat(cap),Equal.cong(Nat,Nat,value => Nat.add(value,lanes),U32.to_nat(count),gap,count_value),
        Order.transitive(Nat.add(gap,lanes),Nat.add(total,lanes),U32.to_nat(cap),Order.add_le(gap,total,lanes,Order.sub_le(total,size)),block_room))
      %Equal.sym(Folded<Context,Value,Final>,lines_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,lanes,U32.is_zero(column),ends,0,LineChain.inner_of(line),column,offset,count,next,all,upcoming,started(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),group,current),[],finals,model),
        lines_end(~Context,~Value,~Final,~turn_value,ends,next,upcoming_after,lined,[],[],finals),
        segment_lines(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,ends,all,line,column,offset,count,next,group,upcoming,current,finals,model,cap,cap_ok,room,positive,count_room,column_value)) :
        {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),_,model) ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,taken,offset,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}
      %Equal.sym(Bool,ends,True{},full_true) :
        {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),
            lines_end(~Context,~Value,~Final,~turn_value,_,Bool.pick(U32,_,0,(column + count : U32)),upcoming_after,lined,[],[],finals),model) ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,taken,offset,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}
      %Equal.sym(List<&2,Cell>,cells(~Model,~Cell,~cell,taken,offset,model),List.append(&2,Cell,segment,rest),
        cells_taken(~Model,~Cell,~cell,remaining,count,offset,model,cap,cap_ok,le_by(U32.to_nat(count),gap,taken,count_value,over),remaining_room)) :
        {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),Folded{0,upcoming_after,[],turned},model) ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,_,Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}
      %Equal.sym(Fed<Cell,Context,Value,Final>,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,List.append(&2,Cell,segment,rest),Fed{group,upcoming,current,finals}),
        feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,rest,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,segment,Fed{group,upcoming,current,finals})),feed_append(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,segment,rest,Fed{group,upcoming,current,finals})) :
        {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),Folded{0,upcoming_after,[],turned},model) ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,_) : Folded<Context,Value,Final>}
      %Equal.sym(Fed<Cell,Context,Value,Final>,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,segment,Fed{group,upcoming,current,finals}),
        fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,grown),total),grown,upcoming_after,current_after,finals),
        feed_some(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,group,upcoming,current,finals,segment,segment_some,
          le_by(Nat.add(size,List.length(&2,Cell,segment)),total,total,
            Equal.trans(Nat,Nat.add(size,List.length(&2,Cell,segment)),Nat.add(size,gap),total,
              Equal.cong(Nat,Nat,value => Nat.add(size,value),List.length(&2,Cell,segment),gap,segment_length),whole),
            Order.reflexive(total)))) :
        {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),Folded{0,upcoming_after,[],turned},model) ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,rest,_)) : Folded<Context,Value,Final>}
      %Equal.sym(Bool,Nat.is_lt(List.length(&2,Cell,grown),total),False{},grown_full) :
        {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,fuel,U32.is_eq(count,remaining),LineChain.Lines{all,line},(offset + count : U32),(remaining - count : U32),Folded{0,upcoming_after,[],turned},model) ==
          folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,rest,fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,_,grown,upcoming_after,current_after,finals))) : Folded<Context,Value,Final>}
      segments_fed(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,fuel,U32.is_eq(count,remaining),Nat.is_lt(U32.to_nat((remaining - count : U32)),Nat.sub(total,List.length(&2,Cell,[]))),all,line,(offset + count : U32),
        (remaining - count : U32),0,[],upcoming_after,[],turned,model,cap,cap_ok,room,positive,block_room,filled,cells_room,
        le_by(U32.to_nat((remaining - count : U32)),Nat.sub(taken,gap),U32.to_nat(cap),rest_value,Order.transitive(Nat.sub(taken,gap),taken,U32.to_nat(cap),Order.sub_le(taken,gap),remaining_room)),
        {==},filled,done_same,{==},
        %Equal.sym(Nat,U32.to_nat((remaining - count : U32)),Nat.sub(taken,gap),rest_value) : {Nat.is_le(1n+Nat.div(_,total),fuel) == True{} : Bool}
        %turned_quotient(size,taken,total,filled,fits,over) : {Nat.is_le(_,fuel) == True{} : Bool}
        enough)

# ---- One block and a walk over blocks ----

def or_false(+value: Bool) -> {value || False{} == value : Bool}:
  match value:
    case True{}: {==}
    case False{}: {==}

# One block of a walk, as fold_leaf folds it, feeds its cells one at a time.
def leaf_fed(~Model: Data,~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~cell: Model -> U32 -> Cell,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+all: List<&2,Context>,+line: LineChain.Line,+count: U32,+model: Model,+group: List<&2,Cell>,+upcoming: List<&2,Context>,+current: List<&2,Value>,
  +finals: List<&2,Final>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(U32.to_nat(LineChain.inner_of(line))),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(LineChain.inner_of(line))) == True{} : Bool},+block_room: {Nat.is_le(Nat.add(U32.to_nat(LineChain.cells_of(line)),U32.to_nat(LineChain.inner_of(line))),U32.to_nat(cap)) == True{} : Bool},+filled: {Nat.is_lt(0n,U32.to_nat(LineChain.cells_of(line))) == True{} : Bool},+cells_room: {Nat.is_le(Nat.double(U32.to_nat(LineChain.cells_of(line))),U32.to_nat(cap)) == True{} : Bool},+count_room: {Nat.is_le(Nat.add(U32.to_nat(count),2n),U32.to_nat(cap)) == True{} : Bool},
  +short: {Nat.is_lt(List.length(&2,Cell,group),U32.to_nat(LineChain.cells_of(line))) == True{} : Bool}) ->
  {leaf_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,LineChain.Lines{all,line},count,model,folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Fed{group,upcoming,current,finals})) ==
    folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,U32.to_nat(count),0,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}:
  +total = U32.to_nat(LineChain.cells_of(line))
  +size = List.length(&2,Cell,group)
  +taken = U32.to_nat(count)
  +quotient = Nat.div(taken,total)
  +widest = U32.max(LineChain.cells_of(line),1)
  +size_le = lt_le(size,total,short)
  +total_room = lanes_room(total,U32.to_nat(cap),cells_room)
  +column_value = NatWords.roundtrip(size,cap,cap_ok,Order.transitive(size,total,U32.to_nat(cap),size_le,total_room))
  +max_value = NativeExtrema.max_one_exact(LineChain.cells_of(line),total,{==},filled)
  +divided = Equal.trans(Nat,U32.to_nat((count / widest : U32)),Nat.div(taken,U32.to_nat(widest)),quotient,
    U32Division.divide(count,widest,cap,cap_ok,
      le_by(Nat.double(U32.to_nat(widest)),Nat.double(total),U32.to_nat(cap),Equal.cong(Nat,Nat,value => Nat.double(value),U32.to_nat(widest),total,max_value),cells_room),
      %Equal.sym(Nat,U32.to_nat(widest),total,max_value) : {Nat.is_lt(0n,_) == True{} : Bool}
      filled),
    Equal.cong(Nat,Nat,value => Nat.div(taken,value),U32.to_nat(widest),total,max_value))
  +quotient_le = Order.transitive(quotient,Nat.mul(quotient,total),taken,RowMajor.product_covers(quotient,total,filled),DivisionOrder.product_below(taken,total,filled))
  +fuel_value = Arithmetic.sum_exact(cap,cap_ok,(count / widest : U32),2,quotient,2n,divided,{==},
    Order.transitive(Nat.add(quotient,2n),Nat.add(taken,2n),U32.to_nat(cap),Order.add_le(quotient,taken,2n,quotient_le),count_room))
  +remaining_room = Order.transitive(taken,Nat.add(taken,2n),U32.to_nat(cap),Order.le_add(taken,2n),count_room)
  +enough = {%Equal.sym(Nat,U32.to_nat((count / widest + 2 : U32)),Nat.add(quotient,2n),fuel_value) : {Nat.is_le(1n+Nat.div(Nat.add(size,taken),total),_) == True{} : Bool}
    %Order.commutative(2n,quotient) : {Nat.is_le(1n+Nat.div(Nat.add(size,taken),total),_) == True{} : Bool}
    quotient_step(size,taken,total,filled,size_le) : {Nat.is_le(1n+Nat.div(Nat.add(size,taken),total),U32.to_nat((count / widest + 2 : U32))) == True{} : Bool}}
  %Equal.sym(Bool,U32.is_zero(LineChain.cells_of(line)),False{},Arithmetic.word_positive_nonzero(LineChain.cells_of(line),filled)) :
    {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,LineChain.segments(count,LineChain.cells_of(line)),U32.is_zero(count) || _,LineChain.Lines{all,line},0,count,
        Folded{U32.from_nat(size),upcoming,started(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),group,current),finals},model) ==
      folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,taken,0,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}
  %Equal.sym(Bool,U32.is_zero(count) || False{},U32.is_zero(count),or_false(U32.is_zero(count))) :
    {segments_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,LineChain.segments(count,LineChain.cells_of(line)),_,LineChain.Lines{all,line},0,count,Folded{U32.from_nat(size),upcoming,started(~Cell,~Value,~step,U32.to_nat(LineChain.inner_of(line)),group,current),finals},model) ==
      folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells(~Model,~Cell,~cell,taken,0,model),Fed{group,upcoming,current,finals})) : Folded<Context,Value,Final>}
  segments_fed(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,LineChain.segments(count,LineChain.cells_of(line)),U32.is_zero(count),Nat.is_lt(taken,Nat.sub(total,size)),all,line,0,count,U32.from_nat(size),group,upcoming,current,
    finals,model,cap,cap_ok,room,positive,block_room,filled,cells_room,remaining_room,column_value,short,{==},{==},enough)

def part_space(+part: Nat,+whole: Nat,+cap: Nat,+inside: {Nat.is_le(part,whole) == True{} : Bool},+room: {Nat.is_le(Nat.add(whole,2n),cap) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(part,2n),cap) == True{} : Bool}:
  Order.transitive(Nat.add(part,2n),Nat.add(whole,2n),cap,Order.add_le(part,whole,2n,inside),room)

# A walk over blocks in order, as walk_model walks them, feeds the cells of
# the blocks one at a time.
def walk_fed(~Model: Data,~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~cell: Model -> U32 -> Cell,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+all: List<&2,Context>,+line: LineChain.Line,+tree: Partition.Tree<&2,Model,Unit>,+fed: Fed<Cell,Context,Value,Final>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(U32.to_nat(LineChain.inner_of(line))),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(LineChain.inner_of(line))) == True{} : Bool},+block_room: {Nat.is_le(Nat.add(U32.to_nat(LineChain.cells_of(line)),U32.to_nat(LineChain.inner_of(line))),U32.to_nat(cap)) == True{} : Bool},+filled: {Nat.is_lt(0n,U32.to_nat(LineChain.cells_of(line))) == True{} : Bool},+cells_room: {Nat.is_le(Nat.double(U32.to_nat(LineChain.cells_of(line))),U32.to_nat(cap)) == True{} : Bool},
  +tree_room: {Nat.is_le(Nat.add(List.length(&2,Cell,stream(~Model,~Cell,~cell,tree)),2n),U32.to_nat(cap)) == True{} : Bool},
  +short: {Nat.is_lt(List.length(&2,Cell,fed_group(~Cell,~Context,~Value,~Final,fed)),U32.to_nat(LineChain.cells_of(line))) == True{} : Bool}) ->
  {walk_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,LineChain.Lines{all,line},tree,folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,fed)) == folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,stream(~Model,~Cell,~cell,tree),fed)) : Folded<Context,Value,Final>}:
  match tree:
    case Partition.Leaf{+model,+first,+count,+details}:
      match fed:
        case Fed{+group,+upcoming,+current,+finals}:
          leaf_fed(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,all,line,count,model,group,upcoming,current,finals,cap,cap_ok,room,positive,block_room,filled,cells_room,
            %cells_length(~Model,~Cell,~cell,U32.to_nat(count),0,model) : {Nat.is_le(Nat.add(_,2n),U32.to_nat(cap)) == True{} : Bool}
            tree_room,short)
    case Partition.Fork{+left,+right}:
      +first = stream(~Model,~Cell,~cell,left)
      +second = stream(~Model,~Cell,~cell,right)
      +after = feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,first,fed)
      +whole = ListReadback.append_length(~Cell,first,second)
      %Equal.sym(Folded<Context,Value,Final>,walk_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,LineChain.Lines{all,line},left,folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,fed)),folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,after),
        walk_fed(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,all,line,left,fed,cap,cap_ok,room,positive,block_room,filled,cells_room,
          part_space(List.length(&2,Cell,first),List.length(&2,Cell,List.append(&2,Cell,first,second)),U32.to_nat(cap),
            %Equal.sym(Nat,List.length(&2,Cell,List.append(&2,Cell,first,second)),Nat.add(List.length(&2,Cell,first),List.length(&2,Cell,second)),whole) :
              {Nat.is_le(List.length(&2,Cell,first),_) == True{} : Bool}
            Order.le_add(List.length(&2,Cell,first),List.length(&2,Cell,second)),
            tree_room),
          short)) :
        {walk_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,LineChain.Lines{all,line},right,_) == folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,List.append(&2,Cell,first,second),fed)) : Folded<Context,Value,Final>}
      %Equal.sym(Fed<Cell,Context,Value,Final>,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,List.append(&2,Cell,first,second),fed),feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,second,after),feed_append(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,first,second,fed)) :
        {walk_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,LineChain.Lines{all,line},right,folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,after)) == folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,_) : Folded<Context,Value,Final>}
      walk_fed(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,all,line,right,after,cap,cap_ok,room,positive,block_room,filled,cells_room,
        part_space(List.length(&2,Cell,second),List.length(&2,Cell,List.append(&2,Cell,first,second)),U32.to_nat(cap),
          %Equal.sym(Nat,List.length(&2,Cell,List.append(&2,Cell,first,second)),Nat.add(List.length(&2,Cell,first),List.length(&2,Cell,second)),whole) :
            {Nat.is_le(List.length(&2,Cell,second),_) == True{} : Bool}
          Order.right_le(List.length(&2,Cell,first),List.length(&2,Cell,second)),
          tree_room),
        feed_short(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,first,fed,filled,short))

# ---- Whole blocks of lines ----

def onto_append(~Element: Data,first: List<&2,Element>,+second: List<&2,Element>,+elements: List<&2,Element>) ->
  {onto_values(~Element,List.append(&2,Element,first,second),elements) == onto_values(~Element,second,onto_values(~Element,first,elements)) : List<&2,Element>}:
  match first:
    case Nil{}: {==}
    case Con{+head,+tail}: onto_append(~Element,tail,second,head <> elements)

def reverse_onto(~Element: Data,values: List<&2,Element>,+middle: List<&2,Element>,+tail: List<&2,Element>) ->
  {List.reverse.go(&2,Element,onto_values(~Element,values,middle),tail) == List.reverse.go(&2,Element,middle,List.append(&2,Element,values,tail)) : List<&2,Element>}:
  match values:
    case Nil{}: {==}
    case Con{+head,+rest}: reverse_onto(~Element,rest,head <> middle,tail)

# The finals of groups whole blocks of the cells of list, block by block, the
# lines of each beginning from the contexts drawn in turn from upcoming.
def grouped(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+all: List<&2,Context>,+line: LineChain.Line,groups: Nat,+list: List<&2,Cell>,+upcoming: List<&2,Context>) -> List<&2,Final>:
  match groups:
    case 0n: []
    case 1n+rest:
      List.append(&2,Final,
        group_lines(~Cell,~Value,~Final,~step,~turn_value,U32.to_nat(LineChain.inner_of(line)),0n,U32.to_nat(LineChain.inner_of(line)),List.take(&2,Cell,list,U32.to_nat(LineChain.cells_of(line))),drawn(~Context,~Value,~begin_value,U32.to_nat(LineChain.inner_of(line)),all,upcoming)),
        grouped(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,rest,List.drop(&2,Cell,list,U32.to_nat(LineChain.cells_of(line))),after_drawn(~Context,U32.to_nat(LineChain.inner_of(line)),all,upcoming)))

# The contexts after groups whole blocks.
def upcoming_after(~Context: Data,groups: Nat,+lanes: Nat,+all: List<&2,Context>,+upcoming: List<&2,Context>) -> List<&2,Context>:
  match groups:
    case 0n: upcoming
    case 1n+rest: upcoming_after(~Context,rest,lanes,all,after_drawn(~Context,lanes,all,upcoming))

# Whole blocks fed one cell at a time turn the lines of every block in order.
def feed_groups(~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+all: List<&2,Context>,+line: LineChain.Line,groups: Nat,+list: List<&2,Cell>,+upcoming: List<&2,Context>,+finals: List<&2,Final>,
  +filled: {Nat.is_lt(0n,U32.to_nat(LineChain.cells_of(line))) == True{} : Bool},+whole: {List.length(&2,Cell,list) == Nat.mul(groups,U32.to_nat(LineChain.cells_of(line))) : Nat}) ->
  {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,list,Fed{[],upcoming,[],finals}) ==
    Fed{[],upcoming_after(~Context,groups,U32.to_nat(LineChain.inner_of(line)),all,upcoming),[],onto_values(~Final,grouped(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,groups,list,upcoming),finals)} : Fed<Cell,Context,Value,Final>}:
  match groups:
    case 0n:
      %Equal.sym(List<&2,Cell>,list,[],empty_of_length(~Cell,list,whole)) : {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,_,Fed{[],upcoming,[],finals}) == Fed{[],upcoming,[],finals} : Fed<Cell,Context,Value,Final>}
      {==}
    case 1n+ +rest:
      +lanes = U32.to_nat(LineChain.inner_of(line))
      +total = U32.to_nat(LineChain.cells_of(line))
      +block = List.take(&2,Cell,list,total)
      +after = List.drop(&2,Cell,list,total)
      +begun = drawn(~Context,~Value,~begin_value,lanes,all,upcoming)
      +next = after_drawn(~Context,lanes,all,upcoming)
      +lines = group_lines(~Cell,~Value,~Final,~step,~turn_value,lanes,0n,lanes,block,begun)
      +later = grouped(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,rest,after,next)
      +inside = le_up(total,List.length(&2,Cell,list),Nat.add(total,Nat.mul(rest,total)),whole,Order.le_add(total,Nat.mul(rest,total)))
      +block_length = take_length(~Cell,list,total,inside)
      +after_length = Equal.trans(Nat,List.length(&2,Cell,after),Nat.sub(List.length(&2,Cell,list),total),Nat.mul(rest,total),drop_length(~Cell,list,total),
        Equal.trans(Nat,Nat.sub(List.length(&2,Cell,list),total),Nat.sub(Nat.add(total,Nat.mul(rest,total)),total),Nat.mul(rest,total),
          Equal.cong(Nat,Nat,value => Nat.sub(value,total),List.length(&2,Cell,list),Nat.add(total,Nat.mul(rest,total)),whole),Order.subtract_shift(total,Nat.mul(rest,total))))
      +block_some = {%Equal.sym(Nat,List.length(&2,Cell,block),total,block_length) : {Nat.is_lt(0n,_) == True{} : Bool}
        filled : {Nat.is_lt(0n,List.length(&2,Cell,block)) == True{} : Bool}}
      %take_drop(~Cell,list,total) :
        {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,_,Fed{[],upcoming,[],finals}) == Fed{[],upcoming_after(~Context,rest,lanes,all,next),[],onto_values(~Final,List.append(&2,Final,lines,later),finals)} : Fed<Cell,Context,Value,Final>}
      %Equal.sym(Fed<Cell,Context,Value,Final>,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,List.append(&2,Cell,block,after),Fed{[],upcoming,[],finals}),feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,after,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,block,Fed{[],upcoming,[],finals})),
        feed_append(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,block,after,Fed{[],upcoming,[],finals})) :
        {_ == Fed{[],upcoming_after(~Context,rest,lanes,all,next),[],onto_values(~Final,List.append(&2,Final,lines,later),finals)} : Fed<Cell,Context,Value,Final>}
      %Equal.sym(Fed<Cell,Context,Value,Final>,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,block,Fed{[],upcoming,[],finals}),fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,Nat.is_lt(List.length(&2,Cell,block),total),block,next,begun,finals),
        feed_some(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,[],upcoming,[],finals,block,block_some,le_by(List.length(&2,Cell,block),total,total,block_length,Order.reflexive(total)))) :
        {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,after,_) == Fed{[],upcoming_after(~Context,rest,lanes,all,next),[],onto_values(~Final,List.append(&2,Final,lines,later),finals)} : Fed<Cell,Context,Value,Final>}
      %Equal.sym(Bool,Nat.is_lt(List.length(&2,Cell,block),total),False{},not_lt_by(List.length(&2,Cell,block),total,total,block_length,Order.not_below(total,total,Order.reflexive(total)))) :
        {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,after,fed_after(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,_,block,next,begun,finals)) ==
          Fed{[],upcoming_after(~Context,rest,lanes,all,next),[],onto_values(~Final,List.append(&2,Final,lines,later),finals)} : Fed<Cell,Context,Value,Final>}
      %Equal.sym(List<&2,Final>,onto_values(~Final,List.append(&2,Final,lines,later),finals),onto_values(~Final,later,onto_values(~Final,lines,finals)),onto_append(~Final,lines,later,finals)) :
        {feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,after,Fed{[],next,[],onto_values(~Final,lines,finals)}) == Fed{[],upcoming_after(~Context,rest,lanes,all,next),[],_} : Fed<Cell,Context,Value,Final>}
      feed_groups(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,rest,after,next,onto_values(~Final,lines,finals),filled,after_length)

# A task over whole blocks from first, as task_model walks it, turns the lines
# of every block in order.
def task_lines(~Model: Data,~Cell: Data,~Context: Data,~Value: Data,~Final: Data,~cell: Model -> U32 -> Cell,~step: Value -> Cell -> Value,~begin_value: Context -> Value,~turn_value: Value -> Final,+all: List<&2,Context>,+line: LineChain.Line,+first: U32,+tree: Partition.Tree<&2,Model,Unit>,+groups: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(U32.to_nat(LineChain.inner_of(line))),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(LineChain.inner_of(line))) == True{} : Bool},+block_room: {Nat.is_le(Nat.add(U32.to_nat(LineChain.cells_of(line)),U32.to_nat(LineChain.inner_of(line))),U32.to_nat(cap)) == True{} : Bool},+filled: {Nat.is_lt(0n,U32.to_nat(LineChain.cells_of(line))) == True{} : Bool},+cells_room: {Nat.is_le(Nat.double(U32.to_nat(LineChain.cells_of(line))),U32.to_nat(cap)) == True{} : Bool},
  +tree_room: {Nat.is_le(Nat.add(List.length(&2,Cell,stream(~Model,~Cell,~cell,tree)),2n),U32.to_nat(cap)) == True{} : Bool},
  +whole: {List.length(&2,Cell,stream(~Model,~Cell,~cell,tree)) == Nat.mul(groups,U32.to_nat(LineChain.cells_of(line))) : Nat}) ->
  {folded_finals(~Context,~Value,~Final,walk_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,LineChain.Lines{all,line},tree,task_start(~Context,~Value,~Final,LineChain.Lines{all,line},first))) ==
    grouped(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,groups,stream(~Model,~Cell,~cell,tree),LineChain.upcoming_of(~Context,all,line,first)) : List<&2,Final>}:
  +upcoming = LineChain.upcoming_of(~Context,all,line,first)
  +cells_all = stream(~Model,~Cell,~cell,tree)
  +lines = grouped(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,groups,cells_all,upcoming)
  +fed = {Fed{[],upcoming,[],[]} : Fed<Cell,Context,Value,Final>}
  %Equal.sym(Folded<Context,Value,Final>,walk_model(~Model,~Context,~Value,~Final,~begin_value,~strided_model(~Model,~Cell,~Value,~cell,~step),~turn_value,LineChain.Lines{all,line},tree,folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,fed)),folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells_all,fed)),
    walk_fed(~Model,~Cell,~Context,~Value,~Final,~cell,~step,~begin_value,~turn_value,all,line,tree,fed,cap,cap_ok,room,positive,block_room,filled,cells_room,tree_room,filled)) :
    {folded_finals(~Context,~Value,~Final,_) == lines : List<&2,Final>}
  %Equal.sym(Fed<Cell,Context,Value,Final>,feed(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,cells_all,fed),Fed{[],upcoming_after(~Context,groups,U32.to_nat(LineChain.inner_of(line)),all,upcoming),[],onto_values(~Final,lines,[])},
    feed_groups(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,all,line,groups,cells_all,upcoming,[],filled,whole)) :
    {folded_finals(~Context,~Value,~Final,folded_of(~Cell,~Context,~Value,~Final,~step,~begin_value,~turn_value,line,_)) == lines : List<&2,Final>}
  %Equal.sym(List<&2,Final>,List.reverse.go(&2,Final,onto_values(~Final,lines,[]),[]),List.reverse.go(&2,Final,[],List.append(&2,Final,lines,[])),reverse_onto(~Final,lines,[],[])) :
    {_ == lines : List<&2,Final>}
  ListReadback.append_empty(~Final,lines)

# ---- The finals of the tasks in order ----

# The finals of the lines of every task, task by task.
def task_finals(~Model: Data,~Context: Data,~Value: Data,~Final: Data,tree: Partition.Tree<&2,Walked<Model,Context,Value,Final>,Unit>) -> List<&2,Final>:
  match tree:
    case Partition.Leaf{walked,first,count,details}:
      match walked:
        case Walked{blocks,folded}: folded_finals(~Context,~Value,~Final,folded)
    case Partition.Fork{left,right}: List.append(&2,Final,task_finals(~Model,~Context,~Value,~Final,left),task_finals(~Model,~Context,~Value,~Final,right))

# The finals unzipped from the tasks are the finals of every task in order.
def unzipped_in_order(~Model: Data,~Context: Data,~Value: Data,~Final: Data,tree: Partition.Tree<&2,Walked<Model,Context,Value,Final>,Unit>,+done: List<&2,Final>) ->
  {unzipped_finals(~Model,~Final,unzipped_model(~Model,~Context,~Value,~Final,tree,done)) == List.append(&2,Final,task_finals(~Model,~Context,~Value,~Final,tree),done) : List<&2,Final>}:
  match tree:
    case Partition.Leaf{walked,first,count,details}:
      match walked:
        case Walked{blocks,folded}: {==}
    case Partition.Fork{+left,+right}:
      +high = unzipped_finals(~Model,~Final,unzipped_model(~Model,~Context,~Value,~Final,right,done))
      +low_finals = task_finals(~Model,~Context,~Value,~Final,left)
      +high_finals = task_finals(~Model,~Context,~Value,~Final,right)
      %Equal.sym(List<&2,Final>,unzipped_finals(~Model,~Final,unzipped_model(~Model,~Context,~Value,~Final,left,high)),List.append(&2,Final,low_finals,high),
        unzipped_in_order(~Model,~Context,~Value,~Final,left,high)) :
        {_ == List.append(&2,Final,List.append(&2,Final,low_finals,high_finals),done) : List<&2,Final>}
      %Equal.sym(List<&2,Final>,high,List.append(&2,Final,high_finals,done),unzipped_in_order(~Model,~Context,~Value,~Final,right,done)) :
        {List.append(&2,Final,low_finals,_) == List.append(&2,Final,List.append(&2,Final,low_finals,high_finals),done) : List<&2,Final>}
      Equal.sym(List<&2,Final>,List.append(&2,Final,List.append(&2,Final,low_finals,high_finals),done),List.append(&2,Final,low_finals,List.append(&2,Final,high_finals,done)),
        ListReadback.append_associative(~Final,low_finals,high_finals,done))

# ---- The contexts lines begin from ----

# The values of remaining lines from residue on: a line begins from the
# context at its residue modulo the count of contexts.
def begins(~Context: Data,~Value: Data,~begin_value: Context -> Value,remaining: Nat,+residue: Nat,+all: List<&2,Context>,+fallback: Context) -> List<&2,Value>:
  match remaining:
    case 0n: []
    case 1n+rest:
      begin_value(at_or(~Context,all,residue,fallback)) <> begins(~Context,~Value,~begin_value,rest,next_residue(residue,List.length(&2,Context,all)),all,fallback)

# The contexts after the one at residue are those from the next residue on.
def refilled_next(~Context: Data,+all: List<&2,Context>,+residue: Nat,+fallback: Context,which: Bool,
  +classified: {Nat.is_lt(1n+residue,List.length(&2,Context,all)) == which : Bool}) ->
  {LineChain.refilled(~Context,all,List.drop(&2,Context,all,1n+residue)) == List.drop(&2,Context,all,next_residue(residue,List.length(&2,Context,all))) : List<&2,Context>}:
  match which:
    case True{}:
      +kinds = List.length(&2,Context,all)
      +split = drop_cons(~Context,all,1n+residue,fallback,classified)
      %Equal.sym(List<&2,Context>,List.drop(&2,Context,all,1n+residue),at_or(~Context,all,1n+residue,fallback) <> List.drop(&2,Context,all,2n+residue),split) :
        {LineChain.refilled(~Context,all,_) == List.drop(&2,Context,all,next_residue(residue,kinds)) : List<&2,Context>}
      %Equal.sym(Bool,Nat.is_lt(1n+residue,kinds),True{},classified) :
        {at_or(~Context,all,1n+residue,fallback) <> List.drop(&2,Context,all,2n+residue) == List.drop(&2,Context,all,Bool.pick(Nat,_,1n+residue,0n)) : List<&2,Context>}
      Equal.sym(List<&2,Context>,List.drop(&2,Context,all,1n+residue),at_or(~Context,all,1n+residue,fallback) <> List.drop(&2,Context,all,2n+residue),split)
    case False{}:
      +kinds = List.length(&2,Context,all)
      %Equal.sym(List<&2,Context>,List.drop(&2,Context,all,1n+residue),[],drop_beyond(~Context,all,1n+residue,Order.not_less(1n+residue,kinds,classified))) :
        {LineChain.refilled(~Context,all,_) == List.drop(&2,Context,all,next_residue(residue,kinds)) : List<&2,Context>}
      %Equal.sym(Bool,Nat.is_lt(1n+residue,kinds),False{},classified) :
        {all == List.drop(&2,Context,all,Bool.pick(Nat,_,1n+residue,0n)) : List<&2,Context>}
      Equal.sym(List<&2,Context>,List.drop(&2,Context,all,0n),all,drop_zero(~Context,all))

# Contexts drawn in turn from residue on, again from the first once they run
# out, begin each line from the context at its residue.
def drawn_begins(~Context: Data,~Value: Data,~begin_value: Context -> Value,remaining: Nat,+residue: Nat,+all: List<&2,Context>,+fallback: Context,
  +below: {Nat.is_lt(residue,List.length(&2,Context,all)) == True{} : Bool}) ->
  {drawn(~Context,~Value,~begin_value,remaining,all,List.drop(&2,Context,all,residue)) == begins(~Context,~Value,~begin_value,remaining,residue,all,fallback) : List<&2,Value>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +kinds = List.length(&2,Context,all)
      +first = at_or(~Context,all,residue,fallback)
      +positive = Order.le_lt_trans(0n,residue,kinds,Order.zero_le(residue),below)
      +next = next_residue(residue,kinds)
      %Equal.sym(List<&2,Context>,List.drop(&2,Context,all,residue),first <> List.drop(&2,Context,all,1n+residue),drop_cons(~Context,all,residue,fallback,below)) :
        {drawn(~Context,~Value,~begin_value,1n+rest,all,_) == begins(~Context,~Value,~begin_value,1n+rest,residue,all,fallback) : List<&2,Value>}
      %Equal.sym(List<&2,Context>,LineChain.refilled(~Context,all,List.drop(&2,Context,all,1n+residue)),List.drop(&2,Context,all,next),
        refilled_next(~Context,all,residue,fallback,Nat.is_lt(1n+residue,kinds),{==})) :
        {begin_value(first) <> drawn(~Context,~Value,~begin_value,rest,all,_) == begin_value(first) <> begins(~Context,~Value,~begin_value,rest,next,all,fallback) : List<&2,Value>}
      %Equal.sym(List<&2,Value>,drawn(~Context,~Value,~begin_value,rest,all,List.drop(&2,Context,all,next)),begins(~Context,~Value,~begin_value,rest,next,all,fallback),
        drawn_begins(~Context,~Value,~begin_value,rest,next,all,fallback,next_below(residue,kinds,positive))) :
        {begin_value(first) <> _ == begin_value(first) <> begins(~Context,~Value,~begin_value,rest,next,all,fallback) : List<&2,Value>}
      {==}

# The contexts after remaining lines drawn from residue on are those from the
# residue remaining lines on.
def after_drawn_residue(~Context: Data,remaining: Nat,+residue: Nat,+all: List<&2,Context>,+fallback: Context,
  +below: {Nat.is_lt(residue,List.length(&2,Context,all)) == True{} : Bool}) ->
  {after_drawn(~Context,remaining,all,List.drop(&2,Context,all,residue)) == List.drop(&2,Context,all,residue_after(remaining,residue,List.length(&2,Context,all))) : List<&2,Context>}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +kinds = List.length(&2,Context,all)
      +first = at_or(~Context,all,residue,fallback)
      +positive = Order.le_lt_trans(0n,residue,kinds,Order.zero_le(residue),below)
      +next = next_residue(residue,kinds)
      %Equal.sym(List<&2,Context>,List.drop(&2,Context,all,residue),first <> List.drop(&2,Context,all,1n+residue),drop_cons(~Context,all,residue,fallback,below)) :
        {after_drawn(~Context,1n+rest,all,_) == List.drop(&2,Context,all,residue_after(rest,next,kinds)) : List<&2,Context>}
      %Equal.sym(List<&2,Context>,LineChain.refilled(~Context,all,List.drop(&2,Context,all,1n+residue)),List.drop(&2,Context,all,next),
        refilled_next(~Context,all,residue,fallback,Nat.is_lt(1n+residue,kinds),{==})) :
        {after_drawn(~Context,rest,all,_) == List.drop(&2,Context,all,residue_after(rest,next,kinds)) : List<&2,Context>}
      after_drawn_residue(~Context,rest,next,all,fallback,next_below(residue,kinds,positive))

# The contexts a task begins from: those from the residue of the line of its
# first cell on.
def upcoming_residue(~Context: Data,+contexts: List<&2,Context>,+line: LineChain.Line,+first: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +positive: {Nat.is_lt(0n,List.length(&2,Context,contexts)) == True{} : Bool},+kinds_room: {Nat.is_le(Nat.double(List.length(&2,Context,contexts)),U32.to_nat(cap)) == True{} : Bool}) ->
  {LineChain.upcoming_of(~Context,contexts,line,first) ==
    List.drop(&2,Context,contexts,Nat.mod(U32.to_nat((first / U32.max(LineChain.cells_of(line),1) * LineChain.inner_of(line) : U32)),List.length(&2,Context,contexts))) :
    List<&2,Context>}:
  +kinds_count = List.length(&2,Context,contexts)
  +kinds = U32.max(U32.from_nat(kinds_count),1)
  +start = (first / U32.max(LineChain.cells_of(line),1) * LineChain.inner_of(line) : U32)
  +at = U32.to_nat(start)
  +kinds_value = NativeExtrema.max_one_exact(U32.from_nat(kinds_count),kinds_count,NatWords.roundtrip(kinds_count,cap,cap_ok,lanes_room(kinds_count,U32.to_nat(cap),kinds_room)),positive)
  +kinds_bound = le_by(Nat.double(U32.to_nat(kinds)),Nat.double(kinds_count),U32.to_nat(cap),Equal.cong(Nat,Nat,value => Nat.double(value),U32.to_nat(kinds),kinds_count,kinds_value),kinds_room)
  +kinds_positive = lt_up(0n,U32.to_nat(kinds),kinds_count,kinds_value,positive)
  +residue_value = Equal.trans(Nat,U32.to_nat((start % kinds : U32)),Nat.mod(at,U32.to_nat(kinds)),Nat.mod(at,kinds_count),
    U32Division.modulo(start,kinds,cap,cap_ok,kinds_bound,kinds_positive),Equal.cong(Nat,Nat,value => Nat.mod(at,value),U32.to_nat(kinds),kinds_count,kinds_value))
  Equal.cong(Nat,List<&2,Context>,index => List.drop(&2,Context,contexts,index),U32.to_nat((start % kinds : U32)),Nat.mod(at,kinds_count),residue_value)
