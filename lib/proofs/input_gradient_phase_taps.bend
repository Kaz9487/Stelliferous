# Congruent taps keep their descending order across quotient cycles.
import Base
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./word_encoding.bend as Words
import ./nat_to_u32_bounds.bend as Bounds

def selected(enabled: Bool,value: Nat,+tail: List<&2,Nat>) -> List<&2,Nat>:
  Bool.pick(List<&2,Nat>,enabled,value <> tail,tail)

def descending(count: Nat,+residue: Nat,+stride: Nat) -> List<&2,Nat>:
  match count:
    case 0n: []
    case 1n+ +rest: Nat.add(residue,Nat.mul(rest,stride)) <> descending(rest,residue,stride)

def filtered(count: Nat,+residue: Nat,+stride: Nat) -> List<&2,Nat>:
  match count:
    case 0n: []
    case 1n+ +rest: selected(Nat.is_eq(residue,Nat.mod(rest,stride)),rest,filtered(rest,residue,stride))

def cyclic(+quotient: Nat,remainder: Nat,+residue: Nat,+stride: Nat) -> List<&2,Nat>:
  descending(Bool.pick(Nat,Nat.is_lt(residue,remainder),1n+quotient,quotient),residue,stride)

def witnessed(+value: Nat,+stride: Nat,+residue: Nat,found: DivisionOrder.Witness<value,stride>) -> List<&2,Nat>:
  DivisionOrder.Witness{quotient,remainder,exact,small} = found
  cyclic(quotient,remainder,residue,stride)

def threshold(+residue: Nat,+remainder: Nat) ->
  {Nat.is_lt(residue,1n+remainder) == Nat.is_lt(residue,remainder) || Nat.is_eq(residue,remainder) : Bool}:
  match residue remainder:
    case 0n 0n: {==}
    case 0n 1n+rest: {==}
    case 1n+ +rest 0n:
      %Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)) : {_ == Nat.is_lt(1n+rest,0n) || False{} : Bool}
      %Equal.sym(Bool,Nat.is_lt(1n+rest,0n),False{},Division.lt_zero(1n+rest)) : {False{} == _ || False{} : Bool}
      {==}
    case 1n+residue 1n+remainder: threshold(residue,remainder)

def append_choice(below: Bool,equal: Bool,+quotient: Nat,+remainder: Nat,+residue: Nat,+stride: Nat,+value: Nat,
  +below_compared: {Nat.is_lt(residue,remainder) == below : Bool},+equal_compared: {Nat.is_eq(residue,remainder) == equal : Bool},
  +value_exact: {value == Nat.add(Nat.mul(quotient,stride),remainder) : Nat}) ->
  {selected(equal,value,descending(Bool.pick(Nat,below,1n+quotient,quotient),residue,stride)) == descending(Bool.pick(Nat,below || equal,1n+quotient,quotient),residue,stride) : List<&2,Nat>}:
  match below equal:
    case False{} False{}: {==}
    case True{} False{}: {==}
    case True{} True{}:
      Empty.absurd({selected(True{},value,descending(1n+quotient,residue,stride)) == descending(1n+quotient,residue,stride) : List<&2,Nat>},
        Words.false_true(Equal.trans(Bool,False{},Nat.is_eq(residue,remainder),True{},
          Equal.sym(Bool,Nat.is_eq(residue,remainder),False{},Order.less_not_equal(residue,remainder,below_compared)),equal_compared)))
    case False{} True{}:
      Equal.cong(Nat,List<&2,Nat>,head => head <> descending(quotient,residue,stride),value,Nat.add(residue,Nat.mul(quotient,stride)),
        Equal.trans(Nat,value,Nat.add(Nat.mul(quotient,stride),remainder),Nat.add(residue,Nat.mul(quotient,stride)),value_exact,
          Equal.trans(Nat,Nat.add(Nat.mul(quotient,stride),remainder),Nat.add(Nat.mul(quotient,stride),residue),Nat.add(residue,Nat.mul(quotient,stride)),
            Equal.cong(Nat,Nat,remainder => Nat.add(Nat.mul(quotient,stride),remainder),remainder,residue,Equal.sym(Nat,residue,remainder,Order.equal_of_is_eq(residue,remainder,equal_compared))),
            Order.commutative(Nat.mul(quotient,stride),residue))))

def appended(+value: Nat,+stride: Nat,+residue: Nat,+quotient: Nat,+remainder: Nat,
  +positive: {Nat.is_lt(0n,stride) == True{} : Bool},+small: {Nat.is_lt(remainder,stride) == True{} : Bool},
  +exact: {value == Nat.add(Nat.mul(quotient,stride),remainder) : Nat},
  previous: {filtered(value,residue,stride) == cyclic(quotient,remainder,residue,stride) : List<&2,Nat>}) ->
  {filtered(1n+value,residue,stride) == cyclic(quotient,1n+remainder,residue,stride) : List<&2,Nat>}:
  %Equal.sym(Nat & Nat,Nat.divmod(value,stride),(quotient,remainder),Division.unique(value,stride,quotient,remainder,exact,positive,small)) :
    {selected(Nat.is_eq(residue,Pair.snd(Nat,Nat,_)),value,filtered(value,residue,stride)) == cyclic(quotient,1n+remainder,residue,stride) : List<&2,Nat>}
  %Equal.sym(List<&2,Nat>,filtered(value,residue,stride),cyclic(quotient,remainder,residue,stride),previous) :
    {selected(Nat.is_eq(residue,remainder),value,_) == cyclic(quotient,1n+remainder,residue,stride) : List<&2,Nat>}
  %Equal.sym(Bool,Nat.is_lt(residue,1n+remainder),Nat.is_lt(residue,remainder) || Nat.is_eq(residue,remainder),threshold(residue,remainder)) :
    {selected(Nat.is_eq(residue,remainder),value,cyclic(quotient,remainder,residue,stride)) == descending(Bool.pick(Nat,_,1n+quotient,quotient),residue,stride) : List<&2,Nat>}
  append_choice(Nat.is_lt(residue,remainder),Nat.is_eq(residue,remainder),quotient,remainder,residue,stride,value,{==},{==},exact)

def next_cycle(inside: Bool,+value: Nat,+stride: Nat,+residue: Nat,+quotient: Nat,+remainder: Nat,
  +positive: {Nat.is_lt(0n,stride) == True{} : Bool},+residue_small: {Nat.is_lt(residue,stride) == True{} : Bool},
  +small: {Nat.is_lt(remainder,stride) == True{} : Bool},+exact: {value == Nat.add(Nat.mul(quotient,stride),remainder) : Nat},
  +next: {Nat.is_lt(1n+remainder,stride) == inside : Bool},
  +previous: {filtered(value,residue,stride) == cyclic(quotient,remainder,residue,stride) : List<&2,Nat>}) ->
  {filtered(1n+value,residue,stride) == witnessed(1n+value,stride,residue,DivisionOrder.advanced(value,stride,quotient,remainder,inside,exact,small,positive,next)) : List<&2,Nat>}:
  match inside:
    case True{}: appended(value,stride,residue,quotient,remainder,positive,small,exact,previous)
    case False{}:
      %Equal.sym(Bool,Nat.is_lt(residue,0n),False{},Division.lt_zero(residue)) :
        {filtered(1n+value,residue,stride) == descending(Bool.pick(Nat,_,2n+quotient,1n+quotient),residue,stride) : List<&2,Nat>}
      %{
        %Equal.sym(Nat,1n+remainder,stride,DivisionOrder.not_below(remainder,stride,small,next)) : {Nat.is_lt(residue,_) == True{} : Bool} residue_small : {Nat.is_lt(residue,1n+remainder) == True{} : Bool}} :
        {filtered(1n+value,residue,stride) == descending(Bool.pick(Nat,_,1n+quotient,quotient),residue,stride) : List<&2,Nat>}
      appended(value,stride,residue,quotient,remainder,positive,small,exact,previous)

def successor(+value: Nat,+stride: Nat,+residue: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +residue_small: {Nat.is_lt(residue,stride) == True{} : Bool},+found: DivisionOrder.Witness<value,stride>,
  previous: {filtered(value,residue,stride) == witnessed(value,stride,residue,found) : List<&2,Nat>}) ->
  {filtered(1n+value,residue,stride) == witnessed(1n+value,stride,residue,DivisionOrder.successor(value,stride,positive,found)) : List<&2,Nat>}:
  DivisionOrder.Witness{+quotient,+remainder,+exact,+small} = found
  next_cycle(Nat.is_lt(1n+remainder,stride),value,stride,residue,quotient,remainder,positive,residue_small,small,exact,{==},previous)

def filtered_cycles(value: Nat,+stride: Nat,+residue: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +residue_small: {Nat.is_lt(residue,stride) == True{} : Bool}) ->
  {filtered(value,residue,stride) == witnessed(value,stride,residue,DivisionOrder.witness(value,stride,positive)) : List<&2,Nat>}:
  match value:
    case 0n:
      %Equal.sym(Bool,Nat.is_lt(residue,0n),False{},Division.lt_zero(residue)) :
        {[] == descending(Bool.pick(Nat,_,1n,0n),residue,stride) : List<&2,Nat>}
      {==}
    case 1n+ +rest:
      successor(rest,stride,residue,positive,residue_small,DivisionOrder.witness(rest,stride,positive),filtered_cycles(rest,stride,residue,positive,residue_small))

def count(+value: Nat,+residue: Nat,+stride: Nat) -> Nat:
  Bool.pick(Nat,Nat.is_lt(residue,value),1n+Nat.div(Nat.sub(Nat.sub(value,1n),residue),stride),0n)

def reflected_tail(+value: Nat,+residue: Nat,+prefix: Nat,+remainder: Nat,+local: Nat,
  +local_exact: {Nat.add(residue,1n+local) == remainder : Nat},
  +exact: {value == Nat.add(prefix,remainder) : Nat}) ->
  {Nat.sub(Nat.sub(value,1n),residue) == Nat.add(prefix,local) : Nat}:
  Order.reflected_complement(residue,Nat.add(prefix,local),value,
    Equal.trans(Nat,Nat.add(residue,1n+Nat.add(prefix,local)),Nat.add(residue,Nat.add(prefix,1n+local)),value,
      Equal.cong(Nat,Nat,tail => Nat.add(residue,tail),1n+Nat.add(prefix,local),Nat.add(prefix,1n+local),Equal.sym(Nat,Nat.add(prefix,1n+local),1n+Nat.add(prefix,local),Addition.successor_right(prefix,local))),
      Equal.trans(Nat,Nat.add(residue,Nat.add(prefix,1n+local)),Nat.add(prefix,Nat.add(residue,1n+local)),value,
        RowMajor.add_swap(residue,prefix,1n+local),
        Equal.trans(Nat,Nat.add(prefix,Nat.add(residue,1n+local)),Nat.add(prefix,remainder),value,
          Equal.cong(Nat,Nat,remainder => Nat.add(prefix,remainder),Nat.add(residue,1n+local),remainder,local_exact),Equal.sym(Nat,value,Nat.add(prefix,remainder),exact)))))

def tail_quotient(+value: Nat,+stride: Nat,+residue: Nat,+quotient: Nat,+remainder: Nat,+local: Nat,
  +positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +local_exact: {Nat.add(residue,1n+local) == remainder : Nat},
  +local_small: {Nat.is_lt(local,stride) == True{} : Bool},
  +exact: {value == Nat.add(Nat.mul(quotient,stride),remainder) : Nat}) ->
  {Nat.div(Nat.sub(Nat.sub(value,1n),residue),stride) == quotient : Nat}:
  %Equal.sym(Nat,Nat.sub(Nat.sub(value,1n),residue),Nat.add(Nat.mul(quotient,stride),local),reflected_tail(value,residue,Nat.mul(quotient,stride),remainder,local,local_exact,exact)) :
    {Nat.div(_,stride) == quotient : Nat}
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(quotient,stride),local),stride),(quotient,local),Division.quotient_remainder(quotient,local,stride,positive,local_small)) :
    {Pair.fst(Nat,Nat,_) == quotient : Nat}
  {==}

def previous_remainder_small(+stride: Nat,+remainder: Nat,+residue: Nat,
  +residue_small: {Nat.is_lt(residue,stride) == True{} : Bool},
  +ordered: {Nat.is_le(remainder,residue) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(Nat.add(stride,remainder),1n+residue),stride) == True{} : Bool}:
  +inside = Bounds.nat_trans(residue,stride,Nat.add(stride,remainder),residue_small,Order.le_add(stride,remainder))
  Order.sub_lt(Nat.add(stride,remainder),1n+residue,stride,Order.strict_follow(residue,Nat.add(stride,remainder),inside),
    %Order.commutative(remainder,stride) : {Nat.is_lt(_,Nat.add(1n+residue,stride)) == True{} : Bool}
    Order.add_lt(remainder,1n+residue,stride,Order.lt_succ(remainder,residue,ordered)))

def previous_value(+value: Nat,+stride: Nat,+quotient: Nat,+remainder: Nat,
  exact: {value == Nat.add(Nat.mul(1n+quotient,stride),remainder) : Nat}) ->
  {value == Nat.add(Nat.mul(quotient,stride),Nat.add(stride,remainder)) : Nat}:
  Equal.trans(Nat,value,Nat.add(Nat.add(stride,Nat.mul(quotient,stride)),remainder),Nat.add(Nat.mul(quotient,stride),Nat.add(stride,remainder)),exact,
    Equal.trans(Nat,Nat.add(Nat.add(stride,Nat.mul(quotient,stride)),remainder),Nat.add(stride,Nat.add(Nat.mul(quotient,stride),remainder)),Nat.add(Nat.mul(quotient,stride),Nat.add(stride,remainder)),
      Addition.associative(stride,Nat.mul(quotient,stride),remainder),RowMajor.add_swap(stride,Nat.mul(quotient,stride),remainder)))

def count_case(below: Bool,+value: Nat,+stride: Nat,+residue: Nat,quotient: Nat,+remainder: Nat,
  +positive: {Nat.is_lt(0n,stride) == True{} : Bool},+residue_small: {Nat.is_lt(residue,stride) == True{} : Bool},
  +small: {Nat.is_lt(remainder,stride) == True{} : Bool},
  +exact: {value == Nat.add(Nat.mul(quotient,stride),remainder) : Nat},
  +compared: {Nat.is_lt(residue,remainder) == below : Bool}) ->
  {count(value,residue,stride) == Bool.pick(Nat,below,1n+quotient,quotient) : Nat}:
  match below:
    case True{}:
      +quotient = quotient
      +local = Nat.sub(remainder,1n+residue)
      +within = Bounds.nat_trans(residue,remainder,value,compared,
        %Equal.sym(Nat,value,Nat.add(Nat.mul(quotient,stride),remainder),exact) : {Nat.is_le(remainder,_) == True{} : Bool}
        %Order.commutative(remainder,Nat.mul(quotient,stride)) : {Nat.is_le(remainder,_) == True{} : Bool} Order.le_add(remainder,Nat.mul(quotient,stride)))
      %Equal.sym(Bool,Nat.is_lt(residue,value),True{},within) : {Bool.pick(Nat,_,1n+Nat.div(Nat.sub(Nat.sub(value,1n),residue),stride),0n) == 1n+quotient : Nat}
      Equal.cong(Nat,Nat,quotient => 1n+quotient,Nat.div(Nat.sub(Nat.sub(value,1n),residue),stride),quotient,
        tail_quotient(value,stride,residue,quotient,remainder,local,positive,Order.split_at_element(remainder,residue,compared),
          Order.le_lt_trans(local,remainder,stride,Order.sub_le(remainder,1n+residue),small),exact))
    case False{}:
      match quotient:
        case 0n:
          %Equal.sym(Nat,value,remainder,exact) : {count(_,residue,stride) == 0n : Nat}
          %Equal.sym(Bool,Nat.is_lt(residue,remainder),False{},compared) : {Bool.pick(Nat,_,1n+Nat.div(Nat.sub(Nat.sub(remainder,1n),residue),stride),0n) == 0n : Nat}
          {==}
        case 1n+ +previous:
          +ordered = Order.not_less(residue,remainder,compared)
          +local = Nat.sub(Nat.add(stride,remainder),1n+residue)
          +within_local = Bounds.nat_trans(residue,stride,Nat.add(stride,remainder),residue_small,Order.le_add(stride,remainder))
          +within = Bounds.nat_trans(residue,stride,value,residue_small,
            %Equal.sym(Nat,value,Nat.add(Nat.mul(1n+previous,stride),remainder),exact) : {Nat.is_le(stride,_) == True{} : Bool}
            %Equal.sym(Nat,Nat.add(Nat.add(stride,Nat.mul(previous,stride)),remainder),Nat.add(stride,Nat.add(Nat.mul(previous,stride),remainder)),Addition.associative(stride,Nat.mul(previous,stride),remainder)) : {Nat.is_le(stride,_) == True{} : Bool}
            Order.le_add(stride,Nat.add(Nat.mul(previous,stride),remainder)))
          %Equal.sym(Bool,Nat.is_lt(residue,value),True{},within) : {Bool.pick(Nat,_,1n+Nat.div(Nat.sub(Nat.sub(value,1n),residue),stride),0n) == 1n+previous : Nat}
          Equal.cong(Nat,Nat,quotient => 1n+quotient,Nat.div(Nat.sub(Nat.sub(value,1n),residue),stride),previous,
            tail_quotient(value,stride,residue,previous,Nat.add(stride,remainder),local,positive,Order.split_at_element(Nat.add(stride,remainder),residue,within_local),previous_remainder_small(stride,remainder,residue,residue_small,ordered),previous_value(value,stride,previous,remainder,exact)))

def count_cycles(+value: Nat,+stride: Nat,+residue: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +residue_small: {Nat.is_lt(residue,stride) == True{} : Bool},found: DivisionOrder.Witness<value,stride>) ->
  {descending(count(value,residue,stride),residue,stride) == witnessed(value,stride,residue,found) : List<&2,Nat>}:
  DivisionOrder.Witness{+quotient,+remainder,+exact,+small} = found
  Equal.cong(Nat,List<&2,Nat>,count => descending(count,residue,stride),count(value,residue,stride),Bool.pick(Nat,Nat.is_lt(residue,remainder),1n+quotient,quotient),
    count_case(Nat.is_lt(residue,remainder),value,stride,residue,quotient,remainder,positive,residue_small,small,exact,{==}))

def filtered_count(+value: Nat,+stride: Nat,+residue: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  +residue_small: {Nat.is_lt(residue,stride) == True{} : Bool}) ->
  {filtered(value,residue,stride) == descending(count(value,residue,stride),residue,stride) : List<&2,Nat>}:
  Equal.trans(List<&2,Nat>,filtered(value,residue,stride),witnessed(value,stride,residue,DivisionOrder.witness(value,stride,positive)),descending(count(value,residue,stride),residue,stride),
    filtered_cycles(value,stride,residue,positive,residue_small),Equal.sym(List<&2,Nat>,descending(count(value,residue,stride),residue,stride),witnessed(value,stride,residue,DivisionOrder.witness(value,stride,positive)),count_cycles(value,stride,residue,positive,residue_small,DivisionOrder.witness(value,stride,positive))))
