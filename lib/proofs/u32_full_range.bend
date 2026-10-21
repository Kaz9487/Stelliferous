# Exact conversions over the complete U32 range. Tensor stride-zero extents may
# exceed the physical array capacity, so storage-capacity bounds are insufficient.
import Base
import ./nat_order.bend as Order
import ./u32_successor.bend as U32Successor
import ./nat_to_u32_bounds.bend as Bounds
import ./word_encoding.bend as WordEncoding
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_multiplication.bend as Multiplication

def ones(width: Nat) -> Word(width):
  match width:
    case 0n: WNil{}
    case 1n+rest: WCon{True{},ones(rest)}

def equal_low_less(+comparison: Cmp) -> {Cmp.is_lt(Word.cmp.fin(True{},True{},comparison)) == Cmp.is_lt(comparison) : Bool}:
  match comparison:
    case LT{}: {==}
    case EQ{}: {==}
    case GT{}: {==}

def finish_bounded(+bit: Bool,+comparison: Cmp,ordered: {Cmp.is_le(comparison) == True{} : Bool}) ->
  {Cmp.is_le(Word.cmp.fin(bit,True{},comparison)) == True{} : Bool}:
  match bit comparison:
    case False{} LT{}: {==}
    case False{} EQ{}: {==}
    case False{} GT{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(ordered))
    case True{} LT{}: {==}
    case True{} EQ{}: {==}
    case True{} GT{}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(ordered))

def bounded(+width: Nat,+word: Word(width)) -> {Cmp.is_le(Word.cmp(width,word,ones(width))) == True{} : Bool}:
  match width:
    case 0n: {==}
    case 1n+ +rest:
      WCon{bit,+tail} = word
      finish_bounded(bit,Word.cmp(rest,tail,ones(rest)),bounded(rest,tail))

def increment(+width: Nat,+word: Word(width),below: {Cmp.is_lt(Word.cmp(width,word,ones(width))) == True{} : Bool}) ->
  {Word.to_nat(width,Word.inc(width,word)) == 1n+Word.to_nat(width,word) : Nat}:
  match width:
    case 0n: Empty.absurd({0n == 1n : Nat},WordEncoding.false_true(below))
    case 1n+ +rest:
      match word:
        case WCon{False{},tail}: {==}
        case WCon{True{},+tail}:
          %increment(rest,tail,{
            %equal_low_less(Word.cmp(rest,tail,ones(rest))) : {_ == True{} : Bool}
            below : {Cmp.is_lt(Word.cmp(rest,tail,ones(rest))) == True{} : Bool}}) :
            {Nat.double(Word.to_nat(rest,Word.inc(rest,tail))) == Nat.double(_) : Nat}
          {==}

def maximum(+value: U32) -> {U32.is_le(value,4294967295) == True{} : Bool}:
  U32{+word} = value
  bounded(32n,word)

def successor(+value: U32,below: {U32.is_lt(value,4294967295) == True{} : Bool}) ->
  {U32.to_nat(U32.add(value,1)) == 1n+U32.to_nat(value) : Nat}:
  U32{+word} = value
  %Equal.sym(Word(32n),Word.add(32n,word,Word.inc(32n,Word.zero(32n))),Word.inc(32n,word),U32Successor.add_one(32n,word)) :
    {Word.to_nat(32n,_) == 1n+Word.to_nat(32n,word) : Nat}
  increment(32n,word,below)

def roundtrip(+value: Nat,+capacity: U32,inside: {Nat.is_le(value,U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(U32.from_nat(value)) == value : Nat}:
  match value:
    case 0n: {==}
    case 1n+ +previous:
      +below = Bounds.predecessor_strict(previous,U32.to_nat(capacity),inside)
      +exact = roundtrip(previous,capacity,Bounds.strict_le(previous,U32.to_nat(capacity),below))
      %exact : {U32.to_nat(U32.inc(U32.from_nat(previous))) == 1n+_ : Nat}
      %Bounds.add_one(U32.from_nat(previous)) : {U32.to_nat(_) == 1n+U32.to_nat(U32.from_nat(previous)) : Nat}
      successor(U32.from_nat(previous),Bounds.uint_trans(U32.from_nat(previous),capacity,4294967295,
        Bounds.native_bound(previous,capacity,exact,below),maximum(capacity)))

def add(+left: U32,+right: U32,+capacity: U32,
  inside: {Nat.is_le(Nat.add(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(U32.add(left,right)) == Nat.add(U32.to_nat(left),U32.to_nat(right)) : Nat}:
  %Equal.sym(U32,U32.add(left,right),U32.from_nat(Nat.add(U32.to_nat(left),U32.to_nat(right))),Arithmetic.native_add(left,right)) :
    {U32.to_nat(_) == Nat.add(U32.to_nat(left),U32.to_nat(right)) : Nat}
  roundtrip(Nat.add(U32.to_nat(left),U32.to_nat(right)),capacity,inside)

def subtract(+left: U32,+right: U32,+capacity: U32,
  ordered: {Nat.is_le(U32.to_nat(right),U32.to_nat(left)) == True{} : Bool},
  inside: {Nat.is_le(Nat.sub(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(U32.sub(left,right)) == Nat.sub(U32.to_nat(left),U32.to_nat(right)) : Nat}:
  %Equal.sym(U32,U32.sub(left,right),U32.from_nat(Nat.sub(U32.to_nat(left),U32.to_nat(right))),Arithmetic.native_subtract(left,right,ordered)) :
    {U32.to_nat(_) == Nat.sub(U32.to_nat(left),U32.to_nat(right)) : Nat}
  roundtrip(Nat.sub(U32.to_nat(left),U32.to_nat(right)),capacity,inside)

def multiply(+left: U32,+right: U32,+capacity: U32,
  inside: {Nat.is_le(Nat.mul(U32.to_nat(left),U32.to_nat(right)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat(U32.mul(left,right)) == Nat.mul(U32.to_nat(left),U32.to_nat(right)) : Nat}:
  %Equal.sym(U32,U32.mul(left,right),U32.from_nat(Nat.mul(U32.to_nat(left),U32.to_nat(right))),Multiplication.native_multiply(left,right)) :
    {U32.to_nat(_) == Nat.mul(U32.to_nat(left),U32.to_nat(right)) : Nat}
  roundtrip(Nat.mul(U32.to_nat(left),U32.to_nat(right)),capacity,inside)

# Subtracting a common interval origin preserves its strict upper bound.
def interval_inside(+low: U32,+high: U32,+position: U32,
  +lower: {U32.is_le(low,position) == True{} : Bool},+upper: {U32.is_lt(position,high) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat((position - low : U32)),U32.to_nat((high - low : U32))) == True{} : Bool}:
  +ln = U32.to_nat(low)
  +hn = U32.to_nat(high)
  +pn = U32.to_nat(position)
  +ordered = Bounds.as_nat_le(low,position,lower)
  +before = Bounds.as_nat_lt(position,high,upper)
  +low_high = Order.le_lt_trans(ln,pn,hn,ordered,before)
  +local = Nat.sub(pn,ln)
  +relative_exact = subtract(position,low,position,ordered,Order.sub_le(pn,ln))
  +range_exact = subtract(high,low,high,Bounds.strict_le(ln,hn,low_high),Order.sub_le(hn,ln))
  %Equal.sym(Nat,U32.to_nat((position - low : U32)),local,relative_exact) : {Nat.is_lt(_,U32.to_nat((high - low : U32))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((high - low : U32)),Nat.sub(hn,ln),range_exact) : {Nat.is_lt(local,_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(local,Nat.sub(hn,ln)),Nat.is_lt(Nat.add(ln,local),hn),Order.subtraction_lt(hn,ln,local)) : {_ == True{} : Bool}
  %Equal.sym(Nat,Nat.add(ln,local),pn,Order.restore_shift(ln,pn,ordered)) : {Nat.is_lt(_,hn) == True{} : Bool} before
