# The bit length of a word: a nonzero value lies in [2^(length - 1), 2^length).
import Base
import ../storage_buffer.bend as Storage
import ./nat_order.bend as Order
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_division.bend as Division
import ./word_encoding.bend as WordEncoding

def power(+exponent: Nat) -> Nat:
  match exponent:
    case 0n: 1n
    case 1n+ +rest: Nat.double(power(rest))

# A word padded with a zero on top keeps its value.
def pad_value(+width: Nat,+word: Word(width)) -> {Word.to_nat(1n+width,Word.shr.pad(width,word)) == Word.to_nat(width,word) : Nat}:
  match width:
    case 0n: {==}
    case 1n+ +rest:
      match word:
        case WCon{+bit,+tail}:
          match bit:
            case False{}: Equal.cong(Nat,Nat,value => Nat.double(value),Word.to_nat(1n+rest,Word.shr.pad(rest,tail)),Word.to_nat(rest,tail),pad_value(rest,tail))
            case True{}: Equal.cong(Nat,Nat,value => 1n+Nat.double(value),Word.to_nat(1n+rest,Word.shr.pad(rest,tail)),Word.to_nat(rest,tail),pad_value(rest,tail))

def halves_word(+bit: Bool,+tail: Word(31n)) -> {Nat.is_le(Nat.double(U32.to_nat(U32.shr(U32{WCon{bit,tail}}))),U32.to_nat(U32{WCon{bit,tail}})) == True{} : Bool}:
  match bit:
    case False{}:
      %Equal.sym(Nat,Word.to_nat(32n,Word.shr.pad(31n,tail)),Word.to_nat(31n,tail),pad_value(31n,tail)) : {Nat.is_le(Nat.double(_),U32.to_nat(U32{WCon{False{},tail}})) == True{} : Bool}
      Order.reflexive(Nat.double(Word.to_nat(31n,tail)))
    case True{}:
      %Equal.sym(Nat,Word.to_nat(32n,Word.shr.pad(31n,tail)),Word.to_nat(31n,tail),pad_value(31n,tail)) : {Nat.is_le(Nat.double(_),U32.to_nat(U32{WCon{True{},tail}})) == True{} : Bool}
      Order.successor(Nat.double(Word.to_nat(31n,tail)),Nat.double(Word.to_nat(31n,tail)),Order.reflexive(Nat.double(Word.to_nat(31n,tail))))

def halves(+value: U32) -> {Nat.is_le(Nat.double(U32.to_nat(U32.shr(value))),U32.to_nat(value)) == True{} : Bool}:
  match value:
    case U32{+word}:
      match word:
        case WCon{+bit,+tail}: halves_word(bit,tail)

def before_one(+count: Nat) -> Nat:
  match count:
    case 0n: 0n
    case 1n+ +rest: rest

def true_unless(+value: Bool,absurd: {value == False{} : Bool} -> Empty) -> {value == True{} : Bool}:
  match value:
    case True{}: {==}
    case False{}: Empty.absurd({False{} == True{} : Bool},absurd({==}))

def zero_length(+remaining: Nat) -> {Storage.bit_length(remaining,0) == 0n : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest: {==}

def nonzero_with(+value: U32,+zero: Bool,+test: {U32.is_zero(value) == zero : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(value)) == True{} : Bool}) -> {zero == False{} : Bool}:
  match zero:
    case True{}:
      Empty.absurd({True{} == False{} : Bool},Order.below_zero(0n,
        {%Arithmetic.word_zero(value,test) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool} positive : {Nat.is_lt(0n,U32.to_nat(0)) == True{} : Bool}}))
    case False{}: {==}

# A positive word is not zero.
def nonzero_word(+value: U32,+positive: {Nat.is_lt(0n,U32.to_nat(value)) == True{} : Bool}) -> {U32.is_zero(value) == False{} : Bool}:
  nonzero_with(value,U32.is_zero(value),{==},positive)

# A word whose shifted bits have a positive length is positive after the shift.
def shifted_positive_with(+rest: Nat,+value: U32,+more: Nat,+exact: {Storage.bit_length(rest,U32.shr(value)) == 1n+more : Nat},+choice: Bool,
  +test: {Nat.is_lt(0n,U32.to_nat(U32.shr(value))) == choice : Bool}) -> {choice == True{} : Bool}:
  match choice:
    case True{}: {==}
    case False{}:
      +zero = Arithmetic.to_nat_injective(U32.shr(value),0,
        Order.antisymmetric(U32.to_nat(U32.shr(value)),0n,Order.not_less(0n,U32.to_nat(U32.shr(value)),test),Order.zero_le(U32.to_nat(U32.shr(value)))))
      +lengths = Equal.trans(Nat,1n+more,Storage.bit_length(rest,U32.shr(value)),0n,Equal.sym(Nat,Storage.bit_length(rest,U32.shr(value)),1n+more,exact),
        Equal.trans(Nat,Storage.bit_length(rest,U32.shr(value)),Storage.bit_length(rest,0),0n,Equal.cong(U32,Nat,word => Storage.bit_length(rest,word),U32.shr(value),0,zero),zero_length(rest)))
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(Equal.cong(Nat,Bool,count => Nat.is_eq(count,0n),1n+more,0n,lengths)))

def lower_at(+remaining: Nat,+value: U32,+positive: {Nat.is_lt(0n,U32.to_nat(value)) == True{} : Bool},+length: Nat,
  +exact: {Storage.bit_length(remaining,value) == length : Nat}) -> {Nat.is_le(power(before_one(length)),U32.to_nat(value)) == True{} : Bool}:
  match remaining length:
    case 0n length:
      %exact : {Nat.is_le(power(before_one(_)),U32.to_nat(value)) == True{} : Bool}
      Order.positive_one(U32.to_nat(value),positive)
    case 1n+ +rest 0n: Order.positive_one(U32.to_nat(value),positive)
    case 1n+ +rest 1n+ +more:
      match more:
        case 0n: Order.positive_one(U32.to_nat(value),positive)
        case 1n+ +k:
          +picked = Equal.cong(Bool,Nat,zero => Bool.pick(Nat,zero,0n,1n+Storage.bit_length(rest,U32.shr(value))),U32.is_zero(value),False{},nonzero_word(value,positive))
          +shifted_exact = Equal.cong(Nat,Nat,count => before_one(count),1n+Storage.bit_length(rest,U32.shr(value)),2n+k,
            Equal.trans(Nat,1n+Storage.bit_length(rest,U32.shr(value)),Bool.pick(Nat,U32.is_zero(value),0n,1n+Storage.bit_length(rest,U32.shr(value))),2n+k,
              Equal.sym(Nat,Bool.pick(Nat,U32.is_zero(value),0n,1n+Storage.bit_length(rest,U32.shr(value))),1n+Storage.bit_length(rest,U32.shr(value)),picked),exact))
          +shifted = shifted_positive_with(rest,value,k,shifted_exact,Nat.is_lt(0n,U32.to_nat(U32.shr(value))),{==})
          Order.transitive(Nat.double(power(k)),Nat.double(U32.to_nat(U32.shr(value))),U32.to_nat(value),
            Order.double_le(power(k),U32.to_nat(U32.shr(value)),lower_at(rest,U32.shr(value),shifted,1n+k,shifted_exact)),halves(value))

# A positive word is at least two to its length less one.
def length_lower(+remaining: Nat,+value: U32,+positive: {Nat.is_lt(0n,U32.to_nat(value)) == True{} : Bool}) ->
  {Nat.is_le(power(before_one(Storage.bit_length(remaining,value))),U32.to_nat(value)) == True{} : Bool}:
  lower_at(remaining,value,positive,Storage.bit_length(remaining,value),{==})

def double_cancel_lt(+left: Nat,+right: Nat,+less: {Nat.is_lt(Nat.double(left),Nat.double(right)) == True{} : Bool}) -> {Nat.is_lt(left,right) == True{} : Bool}:
  true_unless(Nat.is_lt(left,right),outside => WordEncoding.false_true(
    Equal.trans(Bool,False{},Nat.is_lt(Nat.double(right),Nat.double(right)),True{},
      Equal.sym(Bool,Nat.is_lt(Nat.double(right),Nat.double(right)),False{},Order.not_below(Nat.double(right),Nat.double(right),Order.reflexive(Nat.double(right)))),
      Order.le_lt_trans(Nat.double(right),Nat.double(left),Nat.double(right),Order.double_le(right,left,Order.not_less(left,right,outside)),less))))

def pick_le(+zero: Bool,+length: Nat,+bound: Nat,+ordered: {Nat.is_le(length,bound) == True{} : Bool}) -> {Nat.is_le(Bool.pick(Nat,zero,0n,1n+length),1n+bound) == True{} : Bool}:
  match zero:
    case True{}: {==}
    case False{}: ordered

# A word below two to a bound has at most that many bits.
def length_upper(+remaining: Nat,+value: U32,+bound: Nat,+below: {Nat.is_lt(U32.to_nat(value),power(bound)) == True{} : Bool}) ->
  {Nat.is_le(Storage.bit_length(remaining,value),bound) == True{} : Bool}:
  match remaining bound:
    case 0n bound: Order.zero_le(bound)
    case 1n+ +rest 0n:
      +zero = Arithmetic.to_nat_injective(value,0,Order.antisymmetric(U32.to_nat(value),0n,Division.lt_succ_le(U32.to_nat(value),0n,below),Order.zero_le(U32.to_nat(value))))
      %Equal.sym(U32,value,0,zero) : {Nat.is_le(Storage.bit_length(1n+rest,_),0n) == True{} : Bool}
      {==}
    case 1n+ +rest 1n+ +more:
      pick_le(U32.is_zero(value),Storage.bit_length(rest,U32.shr(value)),more,
        length_upper(rest,U32.shr(value),more,
          double_cancel_lt(U32.to_nat(U32.shr(value)),power(more),Order.le_lt_trans(Nat.double(U32.to_nat(U32.shr(value))),U32.to_nat(value),Nat.double(power(more)),halves(value),below))))

def power_monotone(+lower: Nat,+upper: Nat,+ordered: {Nat.is_le(lower,upper) == True{} : Bool}) -> {Nat.is_le(power(lower),power(upper)) == True{} : Bool}:
  match lower upper:
    case 0n 0n: {==}
    case 0n 1n+ +rest: Order.transitive(1n,power(rest),Nat.double(power(rest)),power_monotone(0n,rest,Order.zero_le(rest)),Order.le_double(power(rest)))
    case 1n+ +rest 0n: Empty.absurd({Nat.is_le(power(1n+rest),power(0n)) == True{} : Bool},WordEncoding.false_true(ordered))
    case 1n+ +rest 1n+ +more: Order.double_le(power(rest),power(more),power_monotone(rest,more,ordered))
