# One induction for every state representation, output index and step count.
import Base
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ../traversal_loop.bend as Loop

def iterate(~State: Data,~step: State -> State,count: Nat,state: State) -> State:
  Loop.run(~State,~step,count,state)

def iteration_split(~State: Type,~step: State -> State,first: Nat,+second: Nat,-state: State) ->
  {Loop.run(~State,~step,Nat.add(first,second),state) == Loop.run(~State,~step,second,Loop.run(~State,~step,first,state)) : State}:
  match first:
    case 0n: {==}
    case 1n+rest: iteration_split(~State,~step,rest,second,step(state))

# count runs of a step that is eight steps are 8 * count steps.
def eights(~State: Type,~step: State -> State,~eight: State -> State,
  ~grouped: @-state: State -> {eight(state) == Loop.run(~State,~step,8n,state) : State},count: Nat,-state: State) ->
  {Loop.run(~State,~eight,count,state) == Loop.run(~State,~step,Nat.mul(count,8n),state) : State}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(State,eight(state),Loop.run(~State,~step,8n,state),grouped(state)) :
        {Loop.run(~State,~eight,rest,_) == Loop.run(~State,~step,Nat.mul(1n+rest,8n),state) : State}
      %Equal.sym(State,Loop.run(~State,~step,Nat.add(8n,Nat.mul(rest,8n)),state),Loop.run(~State,~step,Nat.mul(rest,8n),Loop.run(~State,~step,8n,state)),
        iteration_split(~State,~step,8n,Nat.mul(rest,8n),state)) :
        {Loop.run(~State,~eight,rest,Loop.run(~State,~step,8n,state)) == _ : State}
      eights(~State,~step,~eight,~grouped,rest,Loop.run(~State,~step,8n,state))

# The first count % 8 steps one by one, then the rest eight at a time.
def head_then_eights(~State: Type,~step: State -> State,~eight: State -> State,
  ~grouped: @-state: State -> {eight(state) == Loop.run(~State,~step,8n,state) : State},+count: Nat,-state: State) ->
  {Loop.run(~State,~eight,Nat.div(count,8n),Loop.run(~State,~step,Nat.mod(count,8n),state)) == Loop.run(~State,~step,count,state) : State}:
  +rounds = Nat.mul(Nat.div(count,8n),8n)
  +head = Nat.mod(count,8n)
  %Equal.sym(Nat,count,Nat.add(rounds,head),DivisionOrder.quotient_exact(count,8n,{==})) :
    {Loop.run(~State,~eight,Nat.div(count,8n),Loop.run(~State,~step,head,state)) == Loop.run(~State,~step,_,state) : State}
  %Order.commutative(head,rounds) :
    {Loop.run(~State,~eight,Nat.div(count,8n),Loop.run(~State,~step,head,state)) == Loop.run(~State,~step,_,state) : State}
  %Equal.sym(State,Loop.run(~State,~step,Nat.add(head,rounds),state),Loop.run(~State,~step,rounds,Loop.run(~State,~step,head,state)),
    iteration_split(~State,~step,head,rounds,state)) :
    {Loop.run(~State,~eight,Nat.div(count,8n),Loop.run(~State,~step,head,state)) == _ : State}
  eights(~State,~step,~eight,~grouped,Nat.div(count,8n),Loop.run(~State,~step,head,state))

# A single commuting step lifts to every count. RuntimeState may own arrays;
# only the model is duplicated in the proof. No runtime array is cloned.
def iteration_refines_model(
  ~RuntimeState: Type,~ModelState: Data,
  ~runtime_step: RuntimeState -> RuntimeState,~model_step: ModelState -> ModelState,
  ~represent: ModelState -> RuntimeState,
  ~step_matches: @+state: ModelState -> {runtime_step(represent(state)) == represent(model_step(state)) : RuntimeState},
  +count: Nat,+state: ModelState
) -> {Loop.run(~RuntimeState,~runtime_step,count,represent(state)) == represent(Loop.run(~ModelState,~model_step,count,state)) : RuntimeState}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(RuntimeState,runtime_step(represent(state)),represent(model_step(state)),step_matches(state)) :
        {Loop.run(~RuntimeState,~runtime_step,rest,_) == represent(Loop.run(~ModelState,~model_step,rest,model_step(state))) : RuntimeState}
      iteration_refines_model(~RuntimeState,~ModelState,~runtime_step,~model_step,~represent,~step_matches,rest,model_step(state))

# A read-only observation is preserved for every number of iterations.
def preserves_observation(
  ~State: Data,~Observation: Data,~step: State -> State,~observe: State -> Observation,
  ~preserved: @+state: State -> {observe(step(state)) == observe(state) : Observation},
  +count: Nat,+state: State
) -> {observe(iterate(~State,~step,count,state)) == observe(state) : Observation}:
  match count:
    case 0n: {==}
    case 1n+ +rest:
      %preserved(state) : {observe(iterate(~State,~step,rest,step(state))) == _ : Observation}
      preserves_observation(~State,~Observation,~step,~observe,~preserved,rest,step(state))

def iteration_matches_denotation(
  ~Observation: Data,~State: Data,
  ~Index: Data,
  ~step: State -> State,
  ~project: Index -> State -> Observation,
  ~denote: Index -> Nat -> State -> Observation,
  ~at_zero: @+index: Index -> @+state: State -> {project(index,state) == denote(index,0n,state) : Observation},
  ~after_step: @+index: Index -> @+count: Nat -> @+state: State -> {denote(index,count,step(state)) == denote(index,1n+count,state) : Observation},
  +count: Nat,+index: Index,+state: State
) -> {project(index,iterate(~State,~step,count,state)) == denote(index,count,state) : Observation}:
  match count:
    case 0n: at_zero(index,state)
    case 1n+ +rest:
      %after_step(index,rest,state) : {project(index,iterate(~State,~step,rest,step(state))) == _ : Observation}
      iteration_matches_denotation(~Observation,~State,~Index,~step,~project,~denote,~at_zero,~after_step,rest,index,step(state))

def indexed_successor(+position: U32,+offset: Nat) ->
  {(position + U32.from_nat(1n+offset) : U32) == ((position + 1 : U32) + U32.from_nat(offset) : U32) : U32}:
  %WordArithmetic.native_add(1n,offset) :
    {(position + _ : U32) == ((position + 1 : U32) + U32.from_nat(offset) : U32) : U32}
  Equal.sym(U32,((position + 1 : U32) + U32.from_nat(offset) : U32),(position + (1 + U32.from_nat(offset) : U32) : U32),
    Arithmetic.add_associative(position,1,U32.from_nat(offset)))

