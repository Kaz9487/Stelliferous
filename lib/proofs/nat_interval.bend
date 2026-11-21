import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition

type Outside<-start: Nat,-length: Nat,-query: Nat> is Data:
  Before{bound: {Nat.is_lt(query,start) == True{} : Bool}}
  After{bound: {Nat.is_le(Nat.add(start,length),query) == True{} : Bool}}

def outside_head(+start: Nat,+remaining: Nat,+query: Nat,outside: Outside<start,1n+remaining,query>) ->
  {Nat.is_eq(query,start) == False{} : Bool}:
  match outside:
    case Before{bound}: Order.less_not_equal(query,start,bound)
    case After{bound}:
      %Equal.sym(Bool,Nat.is_eq(query,start),Nat.is_eq(start,query),Order.equal_symmetric(query,start)) : {_ == False{} : Bool}
      Order.less_not_equal(start,query,Order.room_index(start,remaining,query,bound))

def outside_tail(+start: Nat,+remaining: Nat,+query: Nat,outside: Outside<start,1n+remaining,query>) -> Outside<1n+start,remaining,query>:
  match outside:
    case Before{bound}: Before{Order.lt_succ(query,start,Bounds.strict_le(query,start,bound))}
    case After{bound}: After{Order.advance_room(start,remaining,query,bound)}
