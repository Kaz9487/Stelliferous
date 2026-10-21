# Machine operations -> unbounded Nat, with explicit no-overflow obligations.
import Base
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./word_subtraction.bend as Subtraction
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./u32_comparison.bend as Comparison
import ./row_major_index.bend as RowMajor

def count_within(+count: U32,+length: Nat,+capacity: U32,+upper: U32,exact: {U32.to_nat(count) == length : Nat},
  room: {Nat.is_le(length,U32.to_nat(capacity)) == True{} : Bool},capacity_bound: {U32.is_le(capacity,upper) == True{} : Bool}) ->
  {U32.is_le(count,upper) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_le(count,upper),Nat.is_le(U32.to_nat(count),U32.to_nat(upper)),Comparison.le(count,upper)) : {_ == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(count),length,exact) : {Nat.is_le(_,U32.to_nat(upper)) == True{} : Bool}
  Order.transitive(length,U32.to_nat(capacity),U32.to_nat(upper),room,Bounds.as_nat_le(capacity,upper,capacity_bound))

def native_roundtrip(+a: U32) -> {U32.from_nat(U32.to_nat(a)) == a : U32}:
  match a:
    case U32{+w}:
      %Equal.sym(U32,U32.from_nat(Word.to_nat(32n,w)),U32{WordArithmetic.encode(Word.to_nat(32n,w),32n)},
        WordArithmetic.native_encoding(Word.to_nat(32n,w))) : {_ == U32{w} : U32}
      %Equal.sym(Word(32n),WordArithmetic.encode(Word.to_nat(32n,w),32n),w,Multiplication.word_roundtrip(32n,w)) : {U32{_} == U32{w} : U32}
      {==}

def native_add(+a: U32,+b: U32) -> {U32.add(a,b) == U32.from_nat(Nat.add(U32.to_nat(a),U32.to_nat(b))) : U32}:
  %native_roundtrip(a) : {U32.add(_,b) == U32.from_nat(Nat.add(U32.to_nat(a),U32.to_nat(b))) : U32}
  %native_roundtrip(b) : {U32.add(U32.from_nat(U32.to_nat(a)),_) == U32.from_nat(Nat.add(U32.to_nat(a),U32.to_nat(b))) : U32}
  WordArithmetic.native_add(U32.to_nat(a),U32.to_nat(b))

def add_zero(+value: U32) -> {(value + 0 : U32) == value : U32}:
  %native_roundtrip(value) : {(value + 0 : U32) == _ : U32}
  %Addition.zero_right(U32.to_nat(value)) : {(value + 0 : U32) == U32.from_nat(_) : U32}
  native_add(value,0)

def add_commutative(+left: U32,+right: U32) -> {(left + right : U32) == (right + left : U32) : U32}:
  %Equal.sym(U32,(left + right : U32),U32.from_nat(Nat.add(U32.to_nat(left),U32.to_nat(right))),native_add(left,right)) :
    {_ == (right + left : U32) : U32}
  %Equal.sym(U32,(right + left : U32),U32.from_nat(Nat.add(U32.to_nat(right),U32.to_nat(left))),native_add(right,left)) :
    {U32.from_nat(Nat.add(U32.to_nat(left),U32.to_nat(right))) == _ : U32}
  %Order.commutative(U32.to_nat(left),U32.to_nat(right)) :
    {U32.from_nat(Nat.add(U32.to_nat(left),U32.to_nat(right))) == U32.from_nat(_) : U32}
  {==}

def add_zero_left(+value: U32) -> {(0 + value : U32) == value : U32}:
  Equal.trans(U32,(0 + value : U32),(value + 0 : U32),value,add_commutative(0,value),add_zero(value))

def multiply_zero_left(+value: U32) -> {(0 * value : U32) == 0 : U32}:
  Multiplication.native_multiply(0,value)

# These are modular identities, so they require no no-overflow assumption.
def encoded_add_associative(+a: Nat,+b: Nat,+c: Nat) ->
  {((U32.from_nat(a) + U32.from_nat(b) : U32) + U32.from_nat(c) : U32) == (U32.from_nat(a) + (U32.from_nat(b) + U32.from_nat(c) : U32) : U32) : U32}:
  %Equal.sym(U32,(U32.from_nat(a) + U32.from_nat(b) : U32),U32.from_nat(Nat.add(a,b)),WordArithmetic.native_add(a,b)) :
    {(_ + U32.from_nat(c) : U32) == (U32.from_nat(a) + (U32.from_nat(b) + U32.from_nat(c) : U32) : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(Nat.add(a,b)) + U32.from_nat(c) : U32),U32.from_nat(Nat.add(Nat.add(a,b),c)),WordArithmetic.native_add(Nat.add(a,b),c)) :
    {_ == (U32.from_nat(a) + (U32.from_nat(b) + U32.from_nat(c) : U32) : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(b) + U32.from_nat(c) : U32),U32.from_nat(Nat.add(b,c)),WordArithmetic.native_add(b,c)) : {U32.from_nat(Nat.add(Nat.add(a,b),c)) == (U32.from_nat(a) + _ : U32) : U32}
  %Equal.sym(U32,(U32.from_nat(a) + U32.from_nat(Nat.add(b,c)) : U32),U32.from_nat(Nat.add(a,Nat.add(b,c))),WordArithmetic.native_add(a,Nat.add(b,c))) : {U32.from_nat(Nat.add(Nat.add(a,b),c)) == _ : U32}
  %Addition.associative(a,b,c) : {U32.from_nat(Nat.add(Nat.add(a,b),c)) == U32.from_nat(_) : U32}
  {==}

def add_associative(+a: U32,+b: U32,+c: U32) -> {((a + b : U32) + c : U32) == (a + (b + c : U32) : U32) : U32}:
  %native_roundtrip(a) : {((_ + b : U32) + c : U32) == (_ + (b + c : U32) : U32) : U32}
  %native_roundtrip(b) : {((U32.from_nat(U32.to_nat(a)) + _ : U32) + c : U32) == (U32.from_nat(U32.to_nat(a)) + (_ + c : U32) : U32) : U32}
  %native_roundtrip(c) : {((U32.from_nat(U32.to_nat(a)) + U32.from_nat(U32.to_nat(b)) : U32) + _ : U32) == (U32.from_nat(U32.to_nat(a)) + (U32.from_nat(U32.to_nat(b)) + _ : U32) : U32) : U32}
  encoded_add_associative(U32.to_nat(a),U32.to_nat(b),U32.to_nat(c))

def subtract_from_sum(+a: U32,+b: U32,+c: U32) -> {((a + b : U32) - c : U32) == (a + (b - c : U32) : U32) : U32}:
  %Subtraction.add_subtract(b,c) : {((a + _ : U32) - c : U32) == (a + (b - c : U32) : U32) : U32}
  %add_associative(a,(b - c : U32),c) : {(_ - c : U32) == (a + (b - c : U32) : U32) : U32}
  Subtraction.subtract_add((a + (b - c : U32) : U32),c)

def subtract_first(+a: U32,+b: U32) -> {((a + b : U32) - a : U32) == b : U32}:
  %add_commutative(b,a) : {(_ - a : U32) == b : U32}
  Subtraction.subtract_add(b,a)

def add_after_subtract(+value: U32,+padding: U32,+offset: U32) ->
  {((value - padding : U32) + offset : U32) == ((value + offset : U32) - padding : U32) : U32}:
  %Equal.sym(U32,((value - padding : U32) + offset : U32),(offset + (value - padding : U32) : U32),add_commutative((value - padding : U32),offset)) :
    {_ == ((value + offset : U32) - padding : U32) : U32}
  %add_commutative(offset,value) : {(offset + (value - padding : U32) : U32) == (_ - padding : U32) : U32}
  Equal.sym(U32,((offset + value : U32) - padding : U32),(offset + (value - padding : U32) : U32),subtract_from_sum(offset,value,padding))

def subtract_sum(+value: U32,+left: U32,+right: U32) ->
  {((value - left : U32) - right : U32) == (value - (left + right : U32) : U32) : U32}:
  +rest = ((value - left : U32) - right : U32)
  +sum = (left + right : U32)
  +restored = Equal.trans(U32,(rest + sum : U32),(rest + (right + left : U32) : U32),value,
    Equal.cong(U32,U32,sum => (rest + sum : U32),sum,(right + left : U32),add_commutative(left,right)),
    Equal.trans(U32,(rest + (right + left : U32) : U32),((rest + right : U32) + left : U32),value,
      Equal.sym(U32,((rest + right : U32) + left : U32),(rest + (right + left : U32) : U32),add_associative(rest,right,left)),
      Equal.trans(U32,((rest + right : U32) + left : U32),((value - left : U32) + left : U32),value,
        Equal.cong(U32,U32,rest => (rest + left : U32),(rest + right : U32),(value - left : U32),Subtraction.add_subtract((value - left : U32),right)),
        Subtraction.add_subtract(value,left))))
  Equal.trans(U32,rest,((rest + sum : U32) - sum : U32),(value - sum : U32),
    Equal.sym(U32,((rest + sum : U32) - sum : U32),rest,Subtraction.subtract_add(rest,sum)),
    Equal.cong(U32,U32,value => (value - sum : U32),(rest + sum : U32),value,restored))

def subtract_difference(+value: U32,+tap: U32,+offset: U32) ->
  {((value - tap : U32) + offset : U32) == (value - (tap - offset : U32) : U32) : U32}:
  +result = ((value - tap : U32) + offset : U32)
  +difference = (tap - offset : U32)
  +restored = Equal.trans(U32,(result + difference : U32),((value - tap : U32) + (offset + difference : U32) : U32),value,
    add_associative((value - tap : U32),offset,difference),
    Equal.trans(U32,((value - tap : U32) + (offset + difference : U32) : U32),((value - tap : U32) + tap : U32),value,
      Equal.cong(U32,U32,tail => ((value - tap : U32) + tail : U32),(offset + difference : U32),tap,
        Equal.trans(U32,(offset + difference : U32),(difference + offset : U32),tap,add_commutative(offset,difference),Subtraction.add_subtract(tap,offset))),Subtraction.add_subtract(value,tap)))
  Equal.trans(U32,result,((result + difference : U32) - difference : U32),(value - difference : U32),
    Equal.sym(U32,((result + difference : U32) - difference : U32),result,Subtraction.subtract_add(result,difference)),
    Equal.cong(U32,U32,value => (value - difference : U32),(result + difference : U32),value,restored))

def native_subtract(+a: U32,+b: U32,h: {Nat.is_le(U32.to_nat(b),U32.to_nat(a)) == True{} : Bool}) -> {U32.sub(a,
    b) == U32.from_nat(Nat.sub(U32.to_nat(a),U32.to_nat(b))) : U32}:
  %native_roundtrip(a) : {U32.sub(_,b) == U32.from_nat(Nat.sub(U32.to_nat(a),U32.to_nat(b))) : U32}
  %native_roundtrip(b) : {U32.sub(U32.from_nat(U32.to_nat(a)),_) == U32.from_nat(Nat.sub(U32.to_nat(a),U32.to_nat(b))) : U32}
  Subtraction.encoded_subtract(U32.to_nat(a),U32.to_nat(b),h)

def add(+a: U32,+b: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},h: {Nat.is_le(Nat.add(U32.to_nat(a),U32.to_nat(b)),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(U32.add(a,b)) == Nat.add(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  %Equal.sym(U32,U32.add(a,b),U32.from_nat(Nat.add(U32.to_nat(a),U32.to_nat(b))),native_add(a,b)) : {U32.to_nat(_) == Nat.add(U32.to_nat(a),
      U32.to_nat(b)) : Nat}
  Bounds.roundtrip(Nat.add(U32.to_nat(a),U32.to_nat(b)),cap,cap_ok,h)

def subtract(+a: U32,+b: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},ordered: {Nat.is_le(U32.to_nat(b),
      U32.to_nat(a)) == True{} : Bool},h: {Nat.is_le(Nat.sub(U32.to_nat(a),U32.to_nat(b)),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(U32.sub(a,b)) == Nat.sub(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  %Equal.sym(U32,U32.sub(a,b),U32.from_nat(Nat.sub(U32.to_nat(a),U32.to_nat(b))),native_subtract(a,b,
      ordered)) : {U32.to_nat(_) == Nat.sub(U32.to_nat(a),U32.to_nat(b)) : Nat}
  Bounds.roundtrip(Nat.sub(U32.to_nat(a),U32.to_nat(b)),cap,cap_ok,h)

def remove_prefix(+value: U32,+base: U32,+local: Nat,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +exact: {U32.to_nat(value) == Nat.add(U32.to_nat(base),local) : Nat},room: {Nat.is_le(U32.to_nat(value),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((value - base : U32)) == local : Nat}:
  +local_room = Order.transitive(local,Nat.add(U32.to_nat(base),local),U32.to_nat(capacity),
    %Order.commutative(local,U32.to_nat(base)) : {Nat.is_le(local,_) == True{} : Bool} Order.le_add(local,U32.to_nat(base)),
    %exact : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} room)
  Equal.trans(Nat,U32.to_nat((value - base : U32)),Nat.sub(U32.to_nat(value),U32.to_nat(base)),local,
    subtract(value,base,capacity,capacity_ok,
      %Equal.sym(Nat,U32.to_nat(value),Nat.add(U32.to_nat(base),local),exact) : {Nat.is_le(U32.to_nat(base),_) == True{} : Bool} Order.le_add(U32.to_nat(base),local),
      %Equal.sym(Nat,U32.to_nat(value),Nat.add(U32.to_nat(base),local),exact) : {Nat.is_le(Nat.sub(_,U32.to_nat(base)),U32.to_nat(capacity)) == True{} : Bool}
      %Equal.sym(Nat,Nat.sub(Nat.add(U32.to_nat(base),local),U32.to_nat(base)),local,Order.subtract_shift(U32.to_nat(base),local)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} local_room),
    %Equal.sym(Nat,U32.to_nat(value),Nat.add(U32.to_nat(base),local),exact) : {Nat.sub(_,U32.to_nat(base)) == local : Nat} Order.subtract_shift(U32.to_nat(base),local))

def subtract_zero(+value: U32) -> {(value - 0 : U32) == value : U32}:
  Equal.trans(U32,(value - 0 : U32),U32.from_nat(Nat.sub(U32.to_nat(value),0n)),value,native_subtract(value,0,Order.zero_le(U32.to_nat(value))),
    Equal.trans(U32,U32.from_nat(Nat.sub(U32.to_nat(value),0n)),U32.from_nat(U32.to_nat(value)),value,
      Equal.cong(Nat,U32,U32.from_nat,Nat.sub(U32.to_nat(value),0n),U32.to_nat(value),Order.subtract_zero(U32.to_nat(value))),native_roundtrip(value)))

def multiply(+a: U32,+b: U32,+cap: U32,cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},h: {Nat.is_le(Nat.mul(U32.to_nat(a),U32.to_nat(b)),
      U32.to_nat(cap)) == True{} : Bool}) -> {U32.to_nat(U32.mul(a,b)) == Nat.mul(U32.to_nat(a),U32.to_nat(b)) : Nat}:
  Multiplication.bounded_multiply(a,b,cap,cap_ok,h)

def from_nat_exact(+count: U32,+length: Nat,exact: {U32.to_nat(count) == length : Nat}) -> {U32.from_nat(length) == count : U32}:
  %exact : {U32.from_nat(_) == count : U32}
  native_roundtrip(count)

# Each mixed-radix step is exact when its complete result fits. The product
# bound is derived here, rather than assumed for an intermediate machine word.
def multiply_add_exact(+a: U32,+radix: U32,+rest: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(a),U32.to_nat(radix)),U32.to_nat(rest)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {U32.to_nat((a * radix + rest : U32)) == Nat.add(Nat.mul(U32.to_nat(a),U32.to_nat(radix)),U32.to_nat(rest)) : Nat}:
  +product = Nat.mul(U32.to_nat(a),U32.to_nat(radix))
  +exact = multiply(a,radix,capacity,capacity_ok,
    Order.transitive(product,Nat.add(product,U32.to_nat(rest)),U32.to_nat(capacity),Order.le_add(product,U32.to_nat(rest)),room))
  %exact : {U32.to_nat((a * radix + rest : U32)) == Nat.add(_,U32.to_nat(rest)) : Nat}
  add((a * radix : U32),rest,capacity,capacity_ok,
    %Equal.sym(Nat,U32.to_nat((a * radix : U32)),product,exact) : {Nat.is_le(Nat.add(_,U32.to_nat(rest)),U32.to_nat(capacity)) == True{} : Bool}
    room)

def equal_of_is_eq(+left: U32,+right: U32,equal: {U32.is_eq(left,right) == True{} : Bool}) -> {left == right : U32}:
  %native_roundtrip(left) : {_ == right : U32}
  %native_roundtrip(right) : {U32.from_nat(U32.to_nat(left)) == _ : U32}
  Equal.cong(Nat,U32,value => U32.from_nat(value),U32.to_nat(left),U32.to_nat(right),Order.equal_of_is_eq(U32.to_nat(left),U32.to_nat(right),{
    %Comparison.eq(left,right) : {_ == True{} : Bool}
    equal : {Nat.is_eq(U32.to_nat(left),U32.to_nat(right)) == True{} : Bool}}))

def to_nat_injective(+left: U32,+right: U32,equal: {U32.to_nat(left) == U32.to_nat(right) : Nat}) -> {left == right : U32}:
  %native_roundtrip(left) : {_ == right : U32}
  %native_roundtrip(right) : {U32.from_nat(U32.to_nat(left)) == _ : U32}
  Equal.cong(Nat,U32,value => U32.from_nat(value),U32.to_nat(left),U32.to_nat(right),equal)

def word_zero(+value: U32,zero: {U32.is_zero(value) == True{} : Bool}) -> {value == 0 : U32}:
  to_nat_injective(value,0,Order.equal_of_is_eq(U32.to_nat(value),0n,{
    %Comparison.eq(value,0) : {_ == True{} : Bool}
    zero : {Nat.is_eq(U32.to_nat(value),0n) == True{} : Bool}}))

def word_nonzero_positive(+extent: U32,nonzero: {U32.is_zero(extent) == False{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}:
  Order.nonzero_positive(U32.to_nat(extent),{
    %Comparison.eq(extent,0) : {_ == False{} : Bool}
    nonzero : {Nat.is_eq(U32.to_nat(extent),0n) == False{} : Bool}})

# Zero channels, rows or width make the actual modular local count zero.
# These producers need no positive-width or whole-tensor indexing envelope.
def multiply_one(+word: U32) -> {(word * 1 : U32) == word : U32}:
  Equal.trans(U32,(word * 1 : U32),U32.from_nat(Nat.mul(U32.to_nat(word),1n)),word,Multiplication.native_multiply(word,1),
    Equal.trans(U32,U32.from_nat(Nat.mul(U32.to_nat(word),1n)),U32.from_nat(U32.to_nat(word)),word,
      Equal.cong(Nat,U32,U32.from_nat,Nat.mul(U32.to_nat(word),1n),U32.to_nat(word),Order.times_one(U32.to_nat(word))),native_roundtrip(word)))
def multiply_zero_right(+value: U32) -> {(value * 0 : U32) == 0 : U32}:
  Equal.trans(U32,(value * 0 : U32),U32.from_nat(Nat.mul(U32.to_nat(value),0n)),0,
    Multiplication.native_multiply(value,0),
    Equal.cong(Nat,U32,U32.from_nat,Nat.mul(U32.to_nat(value),0n),0n,RowMajor.multiply_zero_right(U32.to_nat(value))))

def multiply_commutative(+left: U32,+right: U32) -> {(left * right : U32) == (right * left : U32) : U32}:
  Equal.trans(U32,(left * right : U32),U32.from_nat(Nat.mul(U32.to_nat(left),U32.to_nat(right))),(right * left : U32),
    Multiplication.native_multiply(left,right),
    Equal.trans(U32,U32.from_nat(Nat.mul(U32.to_nat(left),U32.to_nat(right))),U32.from_nat(Nat.mul(U32.to_nat(right),U32.to_nat(left))),(right * left : U32),
      Equal.cong(Nat,U32,value => U32.from_nat(value),Nat.mul(U32.to_nat(left),U32.to_nat(right)),Nat.mul(U32.to_nat(right),U32.to_nat(left)),RowMajor.multiply_commutative(U32.to_nat(left),U32.to_nat(right))),
      Equal.sym(U32,(right * left : U32),U32.from_nat(Nat.mul(U32.to_nat(right),U32.to_nat(left))),Multiplication.native_multiply(right,left))))

def equal_reflexive(+word: U32) -> {U32.is_eq(word,word) == True{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(word,word),Nat.is_eq(U32.to_nat(word),U32.to_nat(word)),Comparison.eq(word,word)) : {_ == True{} : Bool}
  Order.equal_reflexive(U32.to_nat(word))

def add_zero_subtract(+value: U32,+padding: U32) -> {(value + (0 - padding : U32) : U32) == (value - padding : U32) : U32}:
  Equal.trans(U32,(value + (0 - padding : U32) : U32),((value + 0 : U32) - padding : U32),(value - padding : U32),
    Equal.sym(U32,((value + 0 : U32) - padding : U32),(value + (0 - padding : U32) : U32),subtract_from_sum(value,0,padding)),
    Equal.cong(U32,U32,n => (n - padding : U32),(value + 0 : U32),value,add_zero(value)))

def negate_twice(+value: U32) -> {(0 - (0 - value : U32) : U32) == value : U32}:
  +zero = Equal.trans(U32,(value + (0 - value : U32) : U32),((0 - value : U32) + value : U32),0,
    add_commutative(value,(0 - value : U32)),Subtraction.add_subtract(0,value))
  %zero : {(_ - (0 - value : U32) : U32) == value : U32}
  Subtraction.subtract_add(value,(0 - value : U32))

# A modular channel counter advances both its tile and source-plane strides.
def multiply_successor(+index: U32,+stride: U32) ->
  {((index + 1 : U32) * stride : U32) == (index * stride + stride : U32) : U32}:
  +a = U32.to_nat(index)
  +b = U32.to_nat(stride)
  +left = {
    %native_roundtrip(index) : {(_ + 1 : U32) == U32.from_nat(1n+a) : U32}
    %Equal.sym(U32,(U32.from_nat(a) + 1 : U32),(1 + U32.from_nat(a) : U32),add_commutative(U32.from_nat(a),1)) : {_ == U32.from_nat(1n+a) : U32}
    WordArithmetic.native_add(1n,a) : {(index + 1 : U32) == U32.from_nat(1n+a) : U32}}
  +right = Equal.trans(U32,(index * stride + stride : U32),(U32.from_nat(Nat.mul(a,b)) + stride : U32),U32.from_nat(Nat.add(Nat.mul(a,b),b)),
    Equal.cong(U32,U32,value => (value + stride : U32),(index * stride : U32),U32.from_nat(Nat.mul(a,b)),Multiplication.native_multiply(index,stride)),
    %native_roundtrip(stride) : {(U32.from_nat(Nat.mul(a,b)) + _ : U32) == U32.from_nat(Nat.add(Nat.mul(a,b),b)) : U32}
      WordArithmetic.native_add(Nat.mul(a,b),b))
  %Equal.sym(U32,(index * stride + stride : U32),U32.from_nat(Nat.add(Nat.mul(a,b),b)),right) :
    {((index + 1 : U32) * stride : U32) == _ : U32}
  %Order.commutative(b,Nat.mul(a,b)) : {((index + 1 : U32) * stride : U32) == U32.from_nat(_) : U32}
  %Equal.sym(U32,(index + 1 : U32),U32.from_nat(1n+a),left) : {(_ * stride : U32) == U32.from_nat(Nat.mul(1n+a,b)) : U32}
  %native_roundtrip(stride) : {(U32.from_nat(1n+a) * _ : U32) == U32.from_nat(Nat.mul(1n+a,b)) : U32}
  Multiplication.native_encoded_multiply(1n+a,b)

# Modular products preserve natural distributivity and associativity.
def multiply_additive(+left: U32,+right: U32,+factor: U32) ->
  {((left + right : U32) * factor : U32) == (left * factor + right * factor : U32) : U32}:
  +a = U32.to_nat(left)
  +b = U32.to_nat(right)
  +c = U32.to_nat(factor)
  +sum = Nat.add(a,b)
  +products = Nat.add(Nat.mul(a,c),Nat.mul(b,c))
  +left_exact = {
    %Equal.sym(U32,(left + right : U32),U32.from_nat(sum),native_add(left,right)) : {(_ * factor : U32) == U32.from_nat(Nat.mul(sum,c)) : U32}
    %native_roundtrip(factor) : {(U32.from_nat(sum) * _ : U32) == U32.from_nat(Nat.mul(sum,c)) : U32}
    Multiplication.native_encoded_multiply(sum,c) : {((left + right : U32) * factor : U32) == U32.from_nat(Nat.mul(sum,c)) : U32}}
  +right_exact = {
    %Equal.sym(U32,(left * factor : U32),U32.from_nat(Nat.mul(a,c)),Multiplication.native_multiply(left,factor)) : {(_ + right * factor : U32) == U32.from_nat(products) : U32}
    %Equal.sym(U32,(right * factor : U32),U32.from_nat(Nat.mul(b,c)),Multiplication.native_multiply(right,factor)) : {(U32.from_nat(Nat.mul(a,c)) + _ : U32) == U32.from_nat(products) : U32}
    WordArithmetic.native_add(Nat.mul(a,c),Nat.mul(b,c)) : {(left * factor + right * factor : U32) == U32.from_nat(products) : U32}}
  Equal.trans(U32,((left + right : U32) * factor : U32),U32.from_nat(Nat.mul(sum,c)),(left * factor + right * factor : U32),left_exact,
    Equal.trans(U32,U32.from_nat(Nat.mul(sum,c)),U32.from_nat(products),(left * factor + right * factor : U32),Equal.cong(Nat,U32,U32.from_nat,Nat.mul(sum,c),products,RowMajor.distrib(a,b,c)),Equal.sym(U32,(left * factor + right * factor : U32),U32.from_nat(products),right_exact)))
def multiply_associative(+left: U32,+middle: U32,+right: U32) ->
  {(left * middle * right : U32) == (left * (middle * right : U32) : U32) : U32}:
  +a = U32.to_nat(left)
  +b = U32.to_nat(middle)
  +c = U32.to_nat(right)
  +first = Nat.mul(Nat.mul(a,b),c)
  +last = Nat.mul(a,Nat.mul(b,c))
  +left_exact = {
    %Equal.sym(U32,(left * middle : U32),U32.from_nat(Nat.mul(a,b)),Multiplication.native_multiply(left,middle)) : {(_ * right : U32) == U32.from_nat(first) : U32}
    %native_roundtrip(right) : {(U32.from_nat(Nat.mul(a,b)) * _ : U32) == U32.from_nat(first) : U32}
    Multiplication.native_encoded_multiply(Nat.mul(a,b),c) : {(left * middle * right : U32) == U32.from_nat(first) : U32}}
  +right_exact = {
    %native_roundtrip(left) : {(_ * (middle * right : U32) : U32) == U32.from_nat(last) : U32}
    %Equal.sym(U32,(middle * right : U32),U32.from_nat(Nat.mul(b,c)),Multiplication.native_multiply(middle,right)) : {(U32.from_nat(a) * _ : U32) == U32.from_nat(last) : U32}
    Multiplication.native_encoded_multiply(a,Nat.mul(b,c)) : {(left * (middle * right : U32) : U32) == U32.from_nat(last) : U32}}
  Equal.trans(U32,(left * middle * right : U32),U32.from_nat(first),(left * (middle * right : U32) : U32),left_exact,
    Equal.trans(U32,U32.from_nat(first),U32.from_nat(last),(left * (middle * right : U32) : U32),Equal.cong(Nat,U32,U32.from_nat,first,last,RowMajor.assoc(a,b,c)),Equal.sym(U32,(left * (middle * right : U32) : U32),U32.from_nat(last),right_exact)))
