# Natural division by a positive divisor: the quotient/remainder identity and
# monotone quotients, plus monotone truncated subtraction.
import Base
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./word_encoding.bend as WordEncoding

type Witness<-value: Nat,-divisor: Nat> is Data:
  Witness{quotient: Nat,remainder: Nat,exact: {value == Nat.add(Nat.mul(quotient,divisor),remainder) : Nat},
    small: {Nat.is_lt(remainder,divisor) == True{} : Bool}}

def not_below(+value: Nat,+limit: Nat,+below: {Nat.is_lt(value,limit) == True{} : Bool},+outside: {Nat.is_lt(1n+value,limit) == False{} : Bool}) ->
  {1n+value == limit : Nat}:
  match value limit:
    case 0n 0n: Empty.absurd({1n == 0n : Nat},WordEncoding.false_true(below))
    case 1n+rest 0n: Empty.absurd({2n+rest == 0n : Nat},WordEncoding.false_true(below))
    case 0n 1n+ +rest:
      match rest:
        case 0n: {==}
        case 1n+more: Empty.absurd({1n == 2n+more : Nat},WordEncoding.false_true(Equal.sym(Bool,True{},False{},outside)))
    case 1n+ +left 1n+ +right:
      Equal.cong(Nat,Nat,n => 1n+n,1n+left,right,not_below(left,right,below,outside))

def advanced(+value: Nat,+divisor: Nat,+quotient: Nat,+remainder: Nat,+inside: Bool,
  +exact: {value == Nat.add(Nat.mul(quotient,divisor),remainder) : Nat},+small: {Nat.is_lt(remainder,divisor) == True{} : Bool},
  +positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+next: {Nat.is_lt(1n+remainder,divisor) == inside : Bool}) -> Witness<1n+value,divisor>:
  match inside:
    case True{}:
      Witness{quotient,1n+remainder,
        %Equal.sym(Nat,value,Nat.add(Nat.mul(quotient,divisor),remainder),exact) : {1n+_ == Nat.add(Nat.mul(quotient,divisor),1n+remainder) : Nat}
        Equal.sym(Nat,Nat.add(Nat.mul(quotient,divisor),1n+remainder),1n+Nat.add(Nat.mul(quotient,divisor),remainder),
          Addition.successor_right(Nat.mul(quotient,divisor),remainder)),
        next}
    case False{}:
      +full = not_below(remainder,divisor,small,next)
      Witness{1n+quotient,0n,
        %Equal.sym(Nat,value,Nat.add(Nat.mul(quotient,divisor),remainder),exact) : {1n+_ == Nat.add(Nat.add(divisor,Nat.mul(quotient,divisor)),0n) : Nat}
        %Equal.sym(Nat,Nat.add(Nat.add(divisor,Nat.mul(quotient,divisor)),0n),Nat.add(divisor,Nat.mul(quotient,divisor)),Addition.zero_right(Nat.add(divisor,Nat.mul(quotient,divisor)))) :
          {1n+Nat.add(Nat.mul(quotient,divisor),remainder) == _ : Nat}
        %Order.commutative(Nat.mul(quotient,divisor),divisor) : {1n+Nat.add(Nat.mul(quotient,divisor),remainder) == _ : Nat}
        %full : {1n+Nat.add(Nat.mul(quotient,divisor),remainder) == Nat.add(Nat.mul(quotient,divisor),_) : Nat}
        Equal.sym(Nat,Nat.add(Nat.mul(quotient,divisor),1n+remainder),1n+Nat.add(Nat.mul(quotient,divisor),remainder),
          Addition.successor_right(Nat.mul(quotient,divisor),remainder)),
        positive}

def successor(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+previous: Witness<value,divisor>) -> Witness<1n+value,divisor>:
  Witness{+quotient,+remainder,+exact,+small} = previous
  advanced(value,divisor,quotient,remainder,Nat.is_lt(1n+remainder,divisor),exact,small,positive,{==})

def observed_quotient(-value: Nat,-divisor: Nat,found: Witness<value,divisor>) -> Nat:
  Witness{quotient,remainder,exact,small} = found
  quotient

def observed_remainder(-value: Nat,-divisor: Nat,found: Witness<value,divisor>) -> Nat:
  Witness{quotient,remainder,exact,small} = found
  remainder

def observed_pair(+value: Nat,+divisor: Nat,+found: Witness<value,divisor>,positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.divmod(value,divisor) == (observed_quotient(value,divisor,found),observed_remainder(value,divisor,found)) : Nat & Nat}:
  Witness{+quotient,+remainder,+exact,+small} = found
  Division.unique(value,divisor,quotient,remainder,exact,positive,small)

def advanced_pair(+value: Nat,+divisor: Nat,+quotient: Nat,+remainder: Nat,inside: Bool,
  +exact: {value == Nat.add(Nat.mul(quotient,divisor),remainder) : Nat},+small: {Nat.is_lt(remainder,divisor) == True{} : Bool},
  +positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+next: {Nat.is_lt(1n+remainder,divisor) == inside : Bool}) ->
  {Nat.divmod(1n+value,divisor) == Bool.pick(Nat & Nat,inside,(quotient,1n+remainder),(1n+quotient,0n)) : Nat & Nat}:
  match inside:
    case True{}: observed_pair(1n+value,divisor,advanced(value,divisor,quotient,remainder,True{},exact,small,positive,next),positive)
    case False{}: observed_pair(1n+value,divisor,advanced(value,divisor,quotient,remainder,False{},exact,small,positive,next),positive)

def witness(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) -> Witness<value,divisor>:
  match value:
    case 0n: Witness{0n,0n,{==},positive}
    case 1n+ +rest: successor(rest,divisor,positive,witness(rest,divisor,positive))

def quotient_exact_of(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+found: Witness<value,divisor>) ->
  {value == Nat.add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor)) : Nat}:
  Witness{+quotient,+remainder,+exact,+small} = found
  %Equal.sym(Nat & Nat,Nat.divmod(value,divisor),(quotient,remainder),Division.unique(value,divisor,quotient,remainder,exact,positive,small)) :
    {value == Nat.add(Nat.mul(Pair.fst(Nat,Nat,_),divisor),Pair.snd(Nat,Nat,_)) : Nat}
  exact

def quotient_exact(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {value == Nat.add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor)) : Nat}:
  quotient_exact_of(value,divisor,positive,witness(value,divisor,positive))

def remainder_small_of(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+found: Witness<value,divisor>) ->
  {Nat.is_lt(Nat.mod(value,divisor),divisor) == True{} : Bool}:
  Witness{+quotient,+remainder,+exact,+small} = found
  %Equal.sym(Nat & Nat,Nat.divmod(value,divisor),(quotient,remainder),Division.unique(value,divisor,quotient,remainder,exact,positive,small)) :
    {Nat.is_lt(Pair.snd(Nat,Nat,_),divisor) == True{} : Bool}
  small

def remainder_small(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mod(value,divisor),divisor) == True{} : Bool}:
  remainder_small_of(value,divisor,positive,witness(value,divisor,positive))

def remainder_le(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) -> {Nat.is_le(Nat.mod(value,divisor),value) == True{} : Bool}:
  +quotient = Nat.mul(Nat.div(value,divisor),divisor)
  %Equal.sym(Nat,value,Nat.add(quotient,Nat.mod(value,divisor)),quotient_exact(value,divisor,positive)) : {Nat.is_le(Nat.mod(value,divisor),_) == True{} : Bool}
  %Order.commutative(Nat.mod(value,divisor),quotient) : {Nat.is_le(Nat.mod(value,divisor),_) == True{} : Bool}
  Order.le_add(Nat.mod(value,divisor),quotient)

def product_below(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.div(value,divisor),divisor),value) == True{} : Bool}:
  %Equal.sym(Nat,value,Nat.add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor)),quotient_exact(value,divisor,positive)) :
    {Nat.is_le(Nat.mul(Nat.div(value,divisor),divisor),_) == True{} : Bool}
  Order.le_add(Nat.mul(Nat.div(value,divisor),divisor),Nat.mod(value,divisor))

def next_above(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.is_lt(value,Nat.mul(1n+Nat.div(value,divisor),divisor)) == True{} : Bool}:
  +quotient = Nat.div(value,divisor)
  +remainder = Nat.mod(value,divisor)
  %Equal.sym(Nat,value,Nat.add(Nat.mul(quotient,divisor),remainder),quotient_exact(value,divisor,positive)) :
    {Nat.is_lt(_,Nat.add(divisor,Nat.mul(quotient,divisor))) == True{} : Bool}
  %Order.commutative(Nat.mul(quotient,divisor),divisor) :
    {Nat.is_lt(Nat.add(Nat.mul(quotient,divisor),remainder),_) == True{} : Bool}
  %Order.commutative(remainder,Nat.mul(quotient,divisor)) : {Nat.is_lt(_,Nat.add(Nat.mul(quotient,divisor),divisor)) == True{} : Bool}
  %Order.commutative(divisor,Nat.mul(quotient,divisor)) : {Nat.is_lt(Nat.add(remainder,Nat.mul(quotient,divisor)),_) == True{} : Bool}
  Order.add_lt(remainder,divisor,Nat.mul(quotient,divisor),remainder_small(value,divisor,positive))

def quotient_below(+value: Nat,+limit: Nat,+divisor: Nat,positive: {Nat.is_lt(0n,divisor) == True{} : Bool},
  inside: {Nat.is_lt(value,Nat.mul(limit,divisor)) == True{} : Bool}) -> {Nat.is_lt(Nat.div(value,divisor),limit) == True{} : Bool}:
  RowMajor.cancel_multiply_lt(Nat.div(value,divisor),limit,divisor,
    Order.le_lt_trans(Nat.mul(Nat.div(value,divisor),divisor),value,Nat.mul(limit,divisor),product_below(value,divisor,positive),inside))

def quotient_le(+smaller: Nat,+larger: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},
  +ordered: {Nat.is_le(smaller,larger) == True{} : Bool}) -> {Nat.is_le(Nat.div(smaller,divisor),Nat.div(larger,divisor)) == True{} : Bool}:
  Division.lt_succ_le(Nat.div(smaller,divisor),Nat.div(larger,divisor),
    RowMajor.cancel_multiply_lt(Nat.div(smaller,divisor),1n+Nat.div(larger,divisor),divisor,
      Order.le_lt_trans(Nat.mul(Nat.div(smaller,divisor),divisor),larger,Nat.mul(1n+Nat.div(larger,divisor),divisor),
        Order.transitive(Nat.mul(Nat.div(smaller,divisor),divisor),smaller,larger,product_below(smaller,divisor,positive),ordered),
        next_above(larger,divisor,positive))))

def subtract_le(+smaller: Nat,+larger: Nat,+amount: Nat,+ordered: {Nat.is_le(smaller,larger) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(smaller,amount),Nat.sub(larger,amount)) == True{} : Bool}:
  match smaller larger amount:
    case 0n 0n 0n: {==}
    case 0n 0n 1n+rest: {==}
    case 0n 1n+above 0n: {==}
    case 0n 1n+ +above 1n+ +rest: Order.zero_le(Nat.sub(above,rest))
    case 1n+left 0n 0n: Empty.absurd({Nat.is_le(1n+left,0n) == True{} : Bool},WordEncoding.false_true(ordered))
    case 1n+left 0n 1n+rest: Empty.absurd({Nat.is_le(Nat.sub(left,rest),0n) == True{} : Bool},WordEncoding.false_true(ordered))
    case 1n+left 1n+right 0n: ordered
    case 1n+ +left 1n+ +right 1n+ +rest: subtract_le(left,right,rest,ordered)

# Adding whole rows changes the quotient and preserves the remainder.
def shifted_exact(+value: Nat,+divisor: Nat,+rows: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.add(Nat.mul(rows,divisor),value) == Nat.add(Nat.mul(Nat.add(rows,Nat.div(value,divisor)),divisor),Nat.mod(value,divisor)) : Nat}:
  +q = Nat.div(value,divisor)
  +r = Nat.mod(value,divisor)
  %Equal.sym(Nat,value,Nat.add(Nat.mul(q,divisor),r),quotient_exact(value,divisor,positive)) :
    {Nat.add(Nat.mul(rows,divisor),_) == Nat.add(Nat.mul(Nat.add(rows,q),divisor),r) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(rows,q),divisor),Nat.add(Nat.mul(rows,divisor),Nat.mul(q,divisor)),RowMajor.distrib(rows,q,divisor)) :
    {Nat.add(Nat.mul(rows,divisor),Nat.add(Nat.mul(q,divisor),r)) == Nat.add(_,r) : Nat}
  Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(rows,divisor),Nat.mul(q,divisor)),r),Nat.add(Nat.mul(rows,divisor),Nat.add(Nat.mul(q,divisor),r)),Addition.associative(Nat.mul(rows,divisor),Nat.mul(q,divisor),r))

def shifted_pair(+value: Nat,+divisor: Nat,+rows: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.divmod(Nat.add(Nat.mul(rows,divisor),value),divisor) == (Nat.add(rows,Nat.div(value,divisor)),Nat.mod(value,divisor)) : Nat & Nat}:
  Division.unique(Nat.add(Nat.mul(rows,divisor),value),divisor,Nat.add(rows,Nat.div(value,divisor)),Nat.mod(value,divisor),shifted_exact(value,divisor,rows,positive),positive,remainder_small(value,divisor,positive))

def quotient_shift(+value: Nat,+divisor: Nat,+rows: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.div(Nat.add(Nat.mul(rows,divisor),value),divisor) == Nat.add(rows,Nat.div(value,divisor)) : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(Nat.add(Nat.mul(rows,divisor),value),divisor),(Nat.add(rows,Nat.div(value,divisor)),Nat.mod(value,divisor)),shifted_pair(value,divisor,rows,positive))

def remainder_shift(+value: Nat,+divisor: Nat,+rows: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) ->
  {Nat.mod(Nat.add(Nat.mul(rows,divisor),value),divisor) == Nat.mod(value,divisor) : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(Nat.add(Nat.mul(rows,divisor),value),divisor),(Nat.add(rows,Nat.div(value,divisor)),Nat.mod(value,divisor)),shifted_pair(value,divisor,rows,positive))

def multiply_left_le(+factor: Nat,+smaller: Nat,+larger: Nat,+ordered: {Nat.is_le(smaller,larger) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(factor,smaller),Nat.mul(factor,larger)) == True{} : Bool}:
  %RowMajor.multiply_commutative(smaller,factor) : {Nat.is_le(_,Nat.mul(factor,larger)) == True{} : Bool}
  %RowMajor.multiply_commutative(larger,factor) : {Nat.is_le(Nat.mul(smaller,factor),_) == True{} : Bool}
  RowMajor.mul_le(smaller,larger,factor,ordered)

def right_factor_positive(+head: Nat,+tail: Nat,positive: {Nat.is_lt(0n,Nat.mul(head,tail)) == True{} : Bool}) -> {Nat.is_lt(0n,tail) == True{} : Bool}:
  RowMajor.left_factor_positive(tail,head,%RowMajor.multiply_commutative(head,tail) : {Nat.is_lt(0n,_) == True{} : Bool} positive)

def le_multiple(+images: Nat,+elements: Nat,positive: {Nat.is_lt(0n,images) == True{} : Bool}) ->
  {Nat.is_le(elements,Nat.mul(images,elements)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(images,elements),Nat.mul(elements,images),RowMajor.multiply_commutative(images,elements)) : {Nat.is_le(elements,_) == True{} : Bool}
  RowMajor.product_covers(elements,images,positive)

# A quotient is at most its value.
def quotient_at_most(+value: Nat,+divisor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool}) -> {Nat.is_le(Nat.div(value,divisor),value) == True{} : Bool}:
  Order.transitive(Nat.div(value,divisor),Nat.mul(Nat.div(value,divisor),divisor),value,
    %RowMajor.multiply_commutative(divisor,Nat.div(value,divisor)) : {Nat.is_le(Nat.div(value,divisor),_) == True{} : Bool}
      le_multiple(divisor,Nat.div(value,divisor),positive),
    product_below(value,divisor,positive))

def subtract_sum(+a: Nat,+b: Nat) -> {Nat.sub(Nat.add(a,b),b) == a : Nat}:
  match b:
    case 0n:
      %Equal.sym(Nat,Nat.add(a,0n),a,Addition.zero_right(a)) : {Nat.sub(_,0n) == a : Nat}
      Order.subtract_zero(a)
    case 1n+ +p:
      %Equal.sym(Nat,Nat.add(a,1n+p),1n+Nat.add(a,p),Addition.successor_right(a,p)) : {Nat.sub(_,1n+p) == a : Nat}
      subtract_sum(a,p)

# A value divided by a product of two positive numbers: the quotient of its
# quotient by the inner one, and the remainder of that quotient times the
# inner one plus its own remainder.
def product_divmod(+value: Nat,+outer: Nat,+inner: Nat,+outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool}) ->
  {Nat.divmod(value,Nat.mul(outer,inner)) == (Nat.div(Nat.div(value,inner),outer),Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner))) : Nat & Nat}:
  Division.unique(value,Nat.mul(outer,inner),Nat.div(Nat.div(value,inner),outer),Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner)),
    Equal.trans(Nat,value,Nat.add(Nat.mul(Nat.div(value,inner),inner),Nat.mod(value,inner)),Nat.add(Nat.mul(Nat.div(Nat.div(value,inner),outer),Nat.mul(outer,inner)),Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner))),quotient_exact(value,inner,inner_positive),
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.div(value,inner),inner),Nat.mod(value,inner)),Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.div(value,inner),outer),outer),Nat.mod(Nat.div(value,inner),outer)),inner),Nat.mod(value,inner)),Nat.add(Nat.mul(Nat.div(Nat.div(value,inner),outer),Nat.mul(outer,inner)),Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner))),Equal.cong(Nat,Nat,number => Nat.add(Nat.mul(number,inner),Nat.mod(value,inner)),Nat.div(value,inner),Nat.add(Nat.mul(Nat.div(Nat.div(value,inner),outer),outer),Nat.mod(Nat.div(value,inner),outer)),quotient_exact(Nat.div(value,inner),outer,outer_positive)),
    Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.div(value,inner),outer),outer),Nat.mod(Nat.div(value,inner),outer)),inner),Nat.mod(value,inner)),Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.div(Nat.div(value,inner),outer),outer),inner),Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner)),Nat.mod(value,inner)),Nat.add(Nat.mul(Nat.div(Nat.div(value,inner),outer),Nat.mul(outer,inner)),Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner))),Equal.cong(Nat,Nat,number => Nat.add(number,Nat.mod(value,inner)),Nat.mul(Nat.add(Nat.mul(Nat.div(Nat.div(value,inner),outer),outer),Nat.mod(Nat.div(value,inner),outer)),inner),Nat.add(Nat.mul(Nat.mul(Nat.div(Nat.div(value,inner),outer),outer),inner),Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner)),RowMajor.distrib(Nat.mul(Nat.div(Nat.div(value,inner),outer),outer),Nat.mod(Nat.div(value,inner),outer),inner)),
    Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(Nat.mul(Nat.div(Nat.div(value,inner),outer),outer),inner),Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner)),Nat.mod(value,inner)),Nat.add(Nat.add(Nat.mul(Nat.div(Nat.div(value,inner),outer),Nat.mul(outer,inner)),Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner)),Nat.mod(value,inner)),Nat.add(Nat.mul(Nat.div(Nat.div(value,inner),outer),Nat.mul(outer,inner)),Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner))),Equal.cong(Nat,Nat,number => Nat.add(Nat.add(number,Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner)),Nat.mod(value,inner)),Nat.mul(Nat.mul(Nat.div(Nat.div(value,inner),outer),outer),inner),Nat.mul(Nat.div(Nat.div(value,inner),outer),Nat.mul(outer,inner)),RowMajor.assoc(Nat.div(Nat.div(value,inner),outer),outer,inner)),
    Addition.associative(Nat.mul(Nat.div(Nat.div(value,inner),outer),Nat.mul(outer,inner)),Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner)))))),
    Order.product_positive(outer,inner,outer_positive,inner_positive),
    RowMajor.radix(Nat.mod(Nat.div(value,inner),outer),outer,Nat.mod(value,inner),inner,remainder_small(Nat.div(value,inner),outer,outer_positive),remainder_small(value,inner,inner_positive)))

def product_quotient(+value: Nat,+outer: Nat,+inner: Nat,+outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool}) -> {Nat.div(value,Nat.mul(outer,inner)) == Nat.div(Nat.div(value,inner),outer) : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(value,Nat.mul(outer,inner)),(Nat.div(Nat.div(value,inner),outer),Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner))),product_divmod(value,outer,inner,outer_positive,inner_positive)) : {Pair.fst(Nat,Nat,_) == Nat.div(Nat.div(value,inner),outer) : Nat}
  {==}

def product_remainder(+value: Nat,+outer: Nat,+inner: Nat,+outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool}) -> {Nat.mod(value,Nat.mul(outer,inner)) == Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner)) : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(value,Nat.mul(outer,inner)),(Nat.div(Nat.div(value,inner),outer),Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner))),product_divmod(value,outer,inner,outer_positive,inner_positive)) : {Pair.snd(Nat,Nat,_) == Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner)) : Nat}
  {==}

# The remainder by a product, divided by its inner factor: the remainder of
# the quotient by the inner factor, and the remainder by it.
def remainder_divmod(+value: Nat,+outer: Nat,+inner: Nat,+outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool}) ->
  {Nat.divmod(Nat.mod(value,Nat.mul(outer,inner)),inner) == (Nat.mod(Nat.div(value,inner),outer),Nat.mod(value,inner)) : Nat & Nat}:
  %Equal.sym(Nat,Nat.mod(value,Nat.mul(outer,inner)),Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner)),product_remainder(value,outer,inner,outer_positive,inner_positive)) : {Nat.divmod(_,inner) == (Nat.mod(Nat.div(value,inner),outer),Nat.mod(value,inner)) : Nat & Nat}
  Division.unique(Nat.add(Nat.mul(Nat.mod(Nat.div(value,inner),outer),inner),Nat.mod(value,inner)),inner,Nat.mod(Nat.div(value,inner),outer),Nat.mod(value,inner),{==},inner_positive,remainder_small(value,inner,inner_positive))

def remainder_quotient(+value: Nat,+outer: Nat,+inner: Nat,+outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool}) -> {Nat.div(Nat.mod(value,Nat.mul(outer,inner)),inner) == Nat.mod(Nat.div(value,inner),outer) : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.mod(value,Nat.mul(outer,inner)),inner),(Nat.mod(Nat.div(value,inner),outer),Nat.mod(value,inner)),remainder_divmod(value,outer,inner,outer_positive,inner_positive)) : {Pair.fst(Nat,Nat,_) == Nat.mod(Nat.div(value,inner),outer) : Nat}
  {==}

def remainder_remainder(+value: Nat,+outer: Nat,+inner: Nat,+outer_positive: {Nat.is_lt(0n,outer) == True{} : Bool},+inner_positive: {Nat.is_lt(0n,inner) == True{} : Bool}) -> {Nat.mod(Nat.mod(value,Nat.mul(outer,inner)),inner) == Nat.mod(value,inner) : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(Nat.mod(value,Nat.mul(outer,inner)),inner),(Nat.mod(Nat.div(value,inner),outer),Nat.mod(value,inner)),remainder_divmod(value,outer,inner,outer_positive,inner_positive)) : {Pair.snd(Nat,Nat,_) == Nat.mod(value,inner) : Nat}
  {==}

# Element first * width + local of rows rows of width elements from row first
# on, when it is element block * width + tap with tap < width, is at tap of
# row block - first of them.
type RowSplit<-first: Nat,-width: Nat,-local: Nat,-block: Nat,-tap: Nat,-rows: Nat> is Data:
  RowSplit{row: Nat,at: {local == Nat.add(Nat.mul(row,width),tap) : Nat},block_exact: {block == Nat.add(first,row) : Nat},below: {Nat.is_lt(row,rows) == True{} : Bool}}

def row_split(+first: Nat,+width: Nat,+local: Nat,+block: Nat,+tap: Nat,+rows: Nat,+tap_inside: {Nat.is_lt(tap,width) == True{} : Bool},
  +inside: {Nat.is_lt(local,Nat.mul(rows,width)) == True{} : Bool},+same: {Nat.add(Nat.mul(first,width),local) == Nat.add(Nat.mul(block,width),tap) : Nat}) ->
  RowSplit<first,width,local,block,tap,rows>:
  +positive = Order.le_lt_trans(0n,tap,width,Order.zero_le(tap),tap_inside)
  +row = Nat.div(local,width)
  +rest = Nat.mod(local,width)
  +before = Nat.mul(first,width)
  +local_exact = quotient_exact(local,width,positive)
  +joined = Equal.trans(Nat,Nat.add(before,local),Nat.add(before,Nat.add(Nat.mul(row,width),rest)),Nat.add(Nat.mul(Nat.add(first,row),width),rest),
    Equal.cong(Nat,Nat,value => Nat.add(before,value),local,Nat.add(Nat.mul(row,width),rest),local_exact),
    Equal.trans(Nat,Nat.add(before,Nat.add(Nat.mul(row,width),rest)),Nat.add(Nat.add(before,Nat.mul(row,width)),rest),Nat.add(Nat.mul(Nat.add(first,row),width),rest),
      Equal.sym(Nat,Nat.add(Nat.add(before,Nat.mul(row,width)),rest),Nat.add(before,Nat.add(Nat.mul(row,width),rest)),Addition.associative(before,Nat.mul(row,width),rest)),
      Equal.cong(Nat,Nat,value => Nat.add(value,rest),Nat.add(before,Nat.mul(row,width)),Nat.mul(Nat.add(first,row),width),
        Equal.sym(Nat,Nat.mul(Nat.add(first,row),width),Nat.add(before,Nat.mul(row,width)),RowMajor.distrib(first,row,width)))))
  +pairs = Equal.trans(Nat & Nat,(Nat.add(first,row),rest),Nat.divmod(Nat.add(before,local),width),(block,tap),
    Equal.sym(Nat & Nat,Nat.divmod(Nat.add(before,local),width),(Nat.add(first,row),rest),
      Division.unique(Nat.add(before,local),width,Nat.add(first,row),rest,joined,positive,remainder_small(local,width,positive))),
    %Equal.sym(Nat,Nat.add(before,local),Nat.add(Nat.mul(block,width),tap),same) : {Nat.divmod(_,width) == (block,tap) : Nat & Nat}
    Division.unique(Nat.add(Nat.mul(block,width),tap),width,block,tap,{==},positive,tap_inside))
  +rest_exact = Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),(Nat.add(first,row),rest),(block,tap),pairs)
  RowSplit{row,
    %rest_exact : {local == Nat.add(Nat.mul(row,width),_) : Nat} local_exact,
    Equal.sym(Nat,Nat.add(first,row),block,Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),(Nat.add(first,row),rest),(block,tap),pairs)),
    quotient_below(local,rows,width,positive,inside)}

# A value below the divisor divides to zero and leaves itself, also when it
# is shifted by whole rows of the divisor.
def small_quotient(+value: Nat,+divisor: Nat,+below: {Nat.is_lt(value,divisor) == True{} : Bool}) -> {Nat.div(value,divisor) == 0n : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.fst(Nat,Nat,pair),Nat.divmod(value,divisor),(0n,value),Division.small(value,divisor,below))

def small_remainder(+value: Nat,+divisor: Nat,+below: {Nat.is_lt(value,divisor) == True{} : Bool}) -> {Nat.mod(value,divisor) == value : Nat}:
  Equal.cong(Nat & Nat,Nat,pair => Pair.snd(Nat,Nat,pair),Nat.divmod(value,divisor),(0n,value),Division.small(value,divisor,below))

def shifted_small_quotient(+value: Nat,+divisor: Nat,+rows: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+below: {Nat.is_lt(value,divisor) == True{} : Bool}) ->
  {Nat.div(Nat.add(value,Nat.mul(rows,divisor)),divisor) == rows : Nat}:
  %Order.commutative(Nat.mul(rows,divisor),value) : {Nat.div(_,divisor) == rows : Nat}
  Equal.trans(Nat,Nat.div(Nat.add(Nat.mul(rows,divisor),value),divisor),Nat.add(rows,Nat.div(value,divisor)),rows,quotient_shift(value,divisor,rows,positive),
    %Equal.sym(Nat,Nat.div(value,divisor),0n,small_quotient(value,divisor,below)) : {Nat.add(rows,_) == rows : Nat} Addition.zero_right(rows))

def shifted_small_remainder(+value: Nat,+divisor: Nat,+rows: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},+below: {Nat.is_lt(value,divisor) == True{} : Bool}) ->
  {Nat.mod(Nat.add(value,Nat.mul(rows,divisor)),divisor) == value : Nat}:
  %Order.commutative(Nat.mul(rows,divisor),value) : {Nat.mod(_,divisor) == value : Nat}
  Equal.trans(Nat,Nat.mod(Nat.add(Nat.mul(rows,divisor),value),divisor),Nat.mod(value,divisor),value,remainder_shift(value,divisor,rows,positive),
    small_remainder(value,divisor,below))
