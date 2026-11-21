# One sequential traversal for affine runtime owners and reusable proof states.
# The step is a compile-time template: no closure allocation in the loop.
import Base

def run(~State: Type,~step: State -> State,count: Nat,state: State) -> State:
  match count:
    case 0n: state
    case 1n+rest: run(~State,~step,rest,step(state))

