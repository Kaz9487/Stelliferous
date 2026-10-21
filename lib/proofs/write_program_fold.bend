# An indexed fold whose steps each apply a write list to the output is one
# write list: the steps' lists in order. A step's list is computed from the
# parameters the fold carries; advance says how they change from one step to
# the next.
import Base
import ../traversal_array.bend as Traversal
import ./storage_tree_model.bend as Tree
import ./write_program.bend as Writes
import ./write_program_every.bend as Every
import ./nat_to_u32_bounds.bend as Bounds
import ./write_program_readback.bend as Readback
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding

# The lists of steps index, index + 1, ... (count steps), in order.
def all(~Element: Data,~Context: Data,~Parameters: Data,
  ~writes: U32 -> Context -> Parameters -> List<&2,Writes.Assignment<Element>>,
  ~advance: U32 -> Context -> Parameters -> Parameters,
  count: Nat,+index: U32,+context: Context,+parameters: Parameters) -> List<&2,Writes.Assignment<Element>>:
  match count:
    case 0n: []
    case 1n+rest:
      List.append(&2,Writes.Assignment<Element>,writes(index,context,parameters),
        all(~Element,~Context,~Parameters,~writes,~advance,rest,(index + 1 : U32),context,advance(index,context,parameters)))

# The parameters after the fold.

def output(~State: Data,~Context: Data,~Element: Data,~Parameters: Data,
  ~update: U32 -> Context -> State -> State,
  ~parameters: State -> Parameters,
  ~written: State -> Tree.Tree<Element>,
  ~writes: U32 -> Context -> Parameters -> List<&2,Writes.Assignment<Element>>,
  ~advance: U32 -> Context -> Parameters -> Parameters,
  ~advances: @+index: U32 -> @+context: Context -> @+state: State ->
    {parameters(update(index,context,state)) == advance(index,context,parameters(state)) : Parameters},
  ~applies: @+index: U32 -> @+context: Context -> @+state: State ->
    {written(update(index,context,state)) == Writes.pure(Element,writes(index,context,parameters(state)),written(state)) : Tree.Tree<Element>},
  +count: Nat,+index: U32,+context: Context,+state: State) ->
  {written(Traversal.indexed_fold(~State,~Context,~update,count,index,context,state)) ==
    Writes.pure(Element,all(~Element,~Context,~Parameters,~writes,~advance,count,index,context,parameters(state)),written(state)) : Tree.Tree<Element>}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      +next = update(index,context,state)
      +here = writes(index,context,parameters(state))
      +later = all(~Element,~Context,~Parameters,~writes,~advance,rest,(index + 1 : U32),context,advance(index,context,parameters(state)))
      %Equal.sym(Tree.Tree<Element>,Writes.pure(Element,List.append(&2,Writes.Assignment<Element>,here,later),written(state)),
        Writes.pure(Element,later,Writes.pure(Element,here,written(state))),Writes.concatenated_writes(Element,here,later,written(state))) :
        {written(Traversal.indexed_fold(~State,~Context,~update,rest,(index + 1 : U32),context,next)) == _ : Tree.Tree<Element>}
      %applies(index,context,state) :
        {written(Traversal.indexed_fold(~State,~Context,~update,rest,(index + 1 : U32),context,next)) == Writes.pure(Element,later,_) : Tree.Tree<Element>}
      %advances(index,context,state) :
        {written(Traversal.indexed_fold(~State,~Context,~update,rest,(index + 1 : U32),context,next)) ==
          Writes.pure(Element,all(~Element,~Context,~Parameters,~writes,~advance,rest,(index + 1 : U32),context,_),written(next)) : Tree.Tree<Element>}
      output(~State,~Context,~Element,~Parameters,~update,~parameters,~written,~writes,~advance,~advances,~applies,rest,(index + 1 : U32),context,next)

# Every step's list has a property of a target, when an invariant on the
# remaining count, the position and the parameters gives it and survives the
# step. The invariant reads an environment passed at run time, and the target
# is computed from it. Step position p runs at index U32.from_nat(p).
def every(~Element: Data,~Context: Data,~Parameters: Data,
  ~writes: U32 -> Context -> Parameters -> List<&2,Writes.Assignment<Element>>,
  ~advance: U32 -> Context -> Parameters -> Parameters,
  ~Environment: Data,~Target: Data,~target: @+environment: Environment -> Target,
  ~Holds: @+target: Target -> U32 -> Element -> Data,
  ~Invariant: @+environment: Environment -> Nat -> Nat -> Context -> Parameters -> Data,
  ~step: @+environment: Environment -> @+rest: Nat -> @+position: Nat -> @+context: Context -> @+parameters: Parameters ->
    Invariant(environment,1n+rest,position,context,parameters) -> Every.Every(~Element,~Target,~Holds,target(environment),writes(U32.from_nat(position),context,parameters)),
  ~kept: @+environment: Environment -> @+rest: Nat -> @+position: Nat -> @+context: Context -> @+parameters: Parameters ->
    Invariant(environment,1n+rest,position,context,parameters) -> Invariant(environment,rest,1n+position,context,advance(U32.from_nat(position),context,parameters)),
  +environment: Environment,count: Nat,+position: Nat,+context: Context,+parameters: Parameters,+invariant: Invariant(environment,count,position,context,parameters)) ->
  Every.Every(~Element,~Target,~Holds,target(environment),all(~Element,~Context,~Parameters,~writes,~advance,count,U32.from_nat(position),context,parameters)):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +index = U32.from_nat(position)
      +next = advance(index,context,parameters)
      +goal = target(environment)
      %Equal.sym(U32,(index + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(index)) :
        Every.Every(~Element,~Target,~Holds,goal,List.append(&2,Writes.Assignment<Element>,writes(index,context,parameters),all(~Element,~Context,~Parameters,~writes,~advance,rest,_,context,next)))
      Every.appended(~Element,~Target,~Holds,goal,writes(index,context,parameters),all(~Element,~Context,~Parameters,~writes,~advance,rest,U32.from_nat(1n+position),context,next),
        step(environment,rest,position,context,parameters,invariant),
        every(~Element,~Context,~Parameters,~writes,~advance,~Environment,~Target,~target,~Holds,~Invariant,~step,~kept,environment,rest,1n+position,context,next,kept(environment,rest,position,context,parameters,invariant)))

# The lists contain the environment's address when the step at its target
# position contains it. The invariant locates the context and parameters of
# that step; gap is the target position minus the current one.
def contains_at(~Element: Data,~Context: Data,~Parameters: Data,
  ~writes: U32 -> Context -> Parameters -> List<&2,Writes.Assignment<Element>>,
  ~advance: U32 -> Context -> Parameters -> Parameters,
  ~Environment: Data,~address: @+environment: Environment -> U32,~target: @+environment: Environment -> Nat,
  ~Invariant: @+environment: Environment -> Nat -> Nat -> Context -> Parameters -> Data,
  ~kept: @+environment: Environment -> @+rest: Nat -> @+position: Nat -> @+context: Context -> @+parameters: Parameters ->
    Invariant(environment,1n+rest,position,context,parameters) -> Invariant(environment,rest,1n+position,context,advance(U32.from_nat(position),context,parameters)),
  ~found: @+environment: Environment -> @+rest: Nat -> @+context: Context -> @+parameters: Parameters ->
    Invariant(environment,1n+rest,target(environment),context,parameters) ->
    {Readback.contains(~Element,address(environment),writes(U32.from_nat(target(environment)),context,parameters)) == True{} : Bool},
  +environment: Environment,count: Nat,gap: Nat,+position: Nat,+context: Context,+parameters: Parameters,
  +invariant: Invariant(environment,count,position,context,parameters),
  +inside: {Nat.is_lt(gap,count) == True{} : Bool},+at: {Nat.add(position,gap) == target(environment) : Nat}) ->
  {Readback.contains(~Element,address(environment),all(~Element,~Context,~Parameters,~writes,~advance,count,U32.from_nat(position),context,parameters)) == True{} : Bool}:
  match count gap:
    case 0n 0n: Empty.absurd({Readback.contains(~Element,address(environment),[]) == True{} : Bool},WordEncoding.false_true(inside))
    case 0n 1n+g: Empty.absurd({Readback.contains(~Element,address(environment),[]) == True{} : Bool},WordEncoding.false_true(inside))
    case 1n+ +rest 0n:
      +index = U32.from_nat(position)
      +same = Equal.trans(Nat,position,Nat.add(position,0n),target(environment),Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)),at)
      Readback.covered_left(~Element,address(environment),writes(index,context,parameters),
        all(~Element,~Context,~Parameters,~writes,~advance,rest,(index + 1 : U32),context,advance(index,context,parameters)),
        %Equal.sym(Nat,position,target(environment),same) : {Readback.contains(~Element,address(environment),writes(U32.from_nat(_),context,parameters)) == True{} : Bool}
        found(environment,rest,context,parameters,%same : Invariant(environment,1n+rest,_,context,parameters) invariant))
    case 1n+ +rest 1n+ +g:
      +index = U32.from_nat(position)
      +next = advance(index,context,parameters)
      +moved = Equal.trans(Nat,1n+Nat.add(position,g),Nat.add(position,1n+g),target(environment),
        Equal.sym(Nat,Nat.add(position,1n+g),1n+Nat.add(position,g),Addition.successor_right(position,g)),at)
      Readback.covered_right(~Element,address(environment),writes(index,context,parameters),all(~Element,~Context,~Parameters,~writes,~advance,rest,(index + 1 : U32),context,next),
        %Equal.sym(U32,(index + 1 : U32),U32.from_nat(1n+position),Bounds.add_one(index)) :
          {Readback.contains(~Element,address(environment),all(~Element,~Context,~Parameters,~writes,~advance,rest,_,context,next)) == True{} : Bool}
        contains_at(~Element,~Context,~Parameters,~writes,~advance,~Environment,~address,~target,~Invariant,~kept,~found,
          environment,rest,g,1n+position,context,next,kept(environment,rest,position,context,parameters,invariant),inside,moved))
