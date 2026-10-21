# Selection and the tail's three masked passes preserve the chosen word.
import Base
import ../math_compare.bend as Compare
import ../math_fp32.bend as FP32
import ./math_fp32_bits.bend as Bits
import ./word_encoding.bend as WordEncoding
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./storage_shape.bend as Shape

def selector(+mask: F32) -> U32:
  (0 - U32.shrn((U32.and(FP32.to_bits(mask),2147483647) + 2147483647 : U32),31n) : U32)

type OneBit<-word: U32> is Data:
  BitZero{equal: {word == 0 : U32}}
  BitOne{equal: {word == 1 : U32}}

# The native word has a fixed width. Shifting by 31 retains exactly its last
# bit, independent of all lower bits, including any mask payload bits.
def top_bit(word: U32) -> OneBit<U32.shrn(word,31n)>:
  U32{WCon{b0,WCon{b1,WCon{b2,WCon{b3,WCon{b4,WCon{b5,WCon{b6,WCon{b7,WCon{b8,WCon{b9,WCon{b10,WCon{b11,WCon{b12,WCon{b13,WCon{b14,WCon{b15,WCon{b16,WCon{b17,WCon{b18,WCon{b19,WCon{b20,WCon{b21,WCon{b22,WCon{b23,WCon{b24,WCon{b25,WCon{b26,WCon{b27,WCon{b28,WCon{b29,WCon{b30,bits}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}} = word
  match bits:
    case WCon{False{},WNil{}}: BitZero{{==}}
    case WCon{True{},WNil{}}: BitOne{{==}}

type MaskWord<-word: U32> is Data:
  MaskNone{equal: {word == 0 : U32}}
  MaskAll{equal: {word == 4294967295 : U32}}

def negated(+word: U32,evidence: OneBit<word>) -> MaskWord<(0 - word : U32)>:
  match evidence:
    case BitZero{equal}:
      MaskNone{%Equal.sym(U32,word,0,equal) : {(0 - _ : U32) == 0 : U32} {==}}
    case BitOne{equal}:
      MaskAll{%Equal.sym(U32,word,1,equal) : {(0 - _ : U32) == 4294967295 : U32} {==}}

def mask_word(+mask: F32) -> MaskWord<selector(mask)>:
  negated(U32.shrn((U32.and(FP32.to_bits(mask),2147483647) + 2147483647 : U32),31n),top_bit((U32.and(FP32.to_bits(mask),2147483647) + 2147483647 : U32)))

# These four structural functions isolate the magnitude from the sign bit.
def cleared(width: Nat,bits: Word(1n+width)) -> Word(1n+width):
  match width:
    case 0n: WCon{False{},WNil{}}
    case 1n+rest:
      WCon{bit,tail} = bits
      WCon{bit,cleared(rest,tail)}

def lower_ones(width: Nat) -> Word(1n+width):
  match width:
    case 0n: WCon{False{},WNil{}}
    case 1n+rest: WCon{True{},lower_ones(rest)}

def highest(width: Nat,bits: Word(1n+width)) -> Bool:
  match width:
    case 0n:
      WCon{bit,tail} = bits
      bit
    case 1n+rest:
      WCon{bit,tail} = bits
      highest(rest,tail)

def magnitude_nonzero(width: Nat,bits: Word(1n+width)) -> Bool:
  match width:
    case 0n: False{}
    case 1n+rest:
      WCon{bit,tail} = bits
      bit || magnitude_nonzero(rest,tail)

def clear_mask(+width: Nat,+bits: Word(1n+width)) ->
  {Word.and(1n+width,bits,lower_ones(width)) == cleared(width,bits) : Word(1n+width)}:
  match width:
    case 0n:
      match bits:
        case WCon{False{},WNil{}}: {==}
        case WCon{True{},WNil{}}: {==}
    case 1n+ +rest:
      match bits:
        case WCon{False{},+tail}:
          %clear_mask(rest,tail) : {WCon{False{},Word.and(1n+rest,tail,lower_ones(rest))} == WCon{False{},_} : Word(1n+width)}
          {==}
        case WCon{True{},+tail}:
          %clear_mask(rest,tail) : {WCon{True{},Word.and(1n+rest,tail,lower_ones(rest))} == WCon{True{},_} : Word(1n+width)}
          {==}

# Adding the largest positive signed word carries into the sign position
# exactly when one of the magnitude bits or the incoming carry is set.
def magnitude_carry(+width: Nat,+bits: Word(1n+width),+carry: Bool) ->
  {highest(width,Word.adc(1n+width,cleared(width,bits),lower_ones(width),False{},carry)) == carry || magnitude_nonzero(width,bits) : Bool}:
  match width:
    case 0n:
      match bits carry:
        case WCon{bit,WNil{}} False{}: {==}
        case WCon{bit,WNil{}} True{}: {==}
    case 1n+ +rest:
      match bits carry:
        case WCon{False{},+tail} False{}: magnitude_carry(rest,tail,False{})
        case WCon{False{},+tail} True{}: magnitude_carry(rest,tail,True{})
        case WCon{True{},+tail} False{}: magnitude_carry(rest,tail,True{})
        case WCon{True{},+tail} True{}: magnitude_carry(rest,tail,True{})

def shifted_highest(bits: Word(32n)) -> {U32.shrn(U32{bits},31n) == Bool.pick(U32,highest(31n,bits),1,0) : U32}:
  WCon{b0,WCon{b1,WCon{b2,WCon{b3,WCon{b4,WCon{b5,WCon{b6,WCon{b7,WCon{b8,WCon{b9,WCon{b10,WCon{b11,WCon{b12,WCon{b13,WCon{b14,WCon{b15,WCon{b16,WCon{b17,WCon{b18,WCon{b19,WCon{b20,WCon{b21,WCon{b22,WCon{b23,WCon{b24,WCon{b25,WCon{b26,WCon{b27,WCon{b28,WCon{b29,WCon{b30,tail}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}} = bits
  match tail:
    case WCon{False{},WNil{}}: {==}
    case WCon{True{},WNil{}}: {==}

def negative_bit(+bit: Bool) -> {(0 - Bool.pick(U32,bit,1,0) : U32) == Bool.pick(U32,bit,4294967295,0) : U32}:
  match bit:
    case False{}: {==}
    case True{}: {==}

def mask_bits(mask: F32) -> Word(32n):
  F32{bits} = mask
  bits

def magnitude_selected(mask: F32) -> {selector(mask) == Bool.pick(U32,magnitude_nonzero(31n,mask_bits(mask)),4294967295,0) : U32}:
  F32{+bits} = mask
  %Equal.sym(Word(32n),Word.and(32n,bits,lower_ones(31n)),cleared(31n,bits),clear_mask(31n,bits)) : {(0 - U32.shrn(U32{Word.add(32n,_,lower_ones(31n))},31n) : U32) == Bool.pick(U32,magnitude_nonzero(31n,bits),4294967295,0) : U32}
  %Equal.sym(U32,U32.shrn(U32{Word.add(32n,cleared(31n,bits),lower_ones(31n))},31n),Bool.pick(U32,highest(31n,Word.add(32n,cleared(31n,bits),lower_ones(31n))),1,0),shifted_highest(Word.add(32n,cleared(31n,bits),lower_ones(31n)))) : {(0 - _ : U32) == Bool.pick(U32,magnitude_nonzero(31n,bits),4294967295,0) : U32}
  %Equal.sym(Bool,highest(31n,Word.add(32n,cleared(31n,bits),lower_ones(31n))),magnitude_nonzero(31n,bits),magnitude_carry(31n,bits,False{})) : {(0 - Bool.pick(U32,_,1,0) : U32) == Bool.pick(U32,magnitude_nonzero(31n,bits),4294967295,0) : U32}
  negative_bit(magnitude_nonzero(31n,bits))

def zero_finish(+bit: Bool,+order: Cmp) -> {Cmp.is_eq(Word.cmp.fin(bit,False{},order)) == Bool.not(bit) && Cmp.is_eq(order) : Bool}:
  match bit order:
    case False{} LT{}: {==}
    case False{} EQ{}: {==}
    case False{} GT{}: {==}
    case True{} LT{}: {==}
    case True{} EQ{}: {==}
    case True{} GT{}: {==}

def cleared_zero(+width: Nat,+bits: Word(1n+width)) -> {Cmp.is_eq(Word.cmp(1n+width,cleared(width,bits),Word.zero(1n+width))) == Bool.not(magnitude_nonzero(width,bits)) : Bool}:
  match width:
    case 0n: {==}
    case 1n+ +rest:
      match bits:
        case WCon{False{},+tail}:
          Equal.trans(Bool,Cmp.is_eq(Word.cmp.fin(False{},False{},Word.cmp(1n+rest,cleared(rest,tail),Word.zero(1n+rest)))),Cmp.is_eq(Word.cmp(1n+rest,cleared(rest,tail),Word.zero(1n+rest))),Bool.not(magnitude_nonzero(rest,tail)),zero_finish(False{},Word.cmp(1n+rest,cleared(rest,tail),Word.zero(1n+rest))),cleared_zero(rest,tail))
        case WCon{True{},+tail}: zero_finish(True{},Word.cmp(1n+rest,cleared(rest,tail),Word.zero(1n+rest)))

def and_all(+width: Nat,+word: Word(width)) -> {Word.and(width,word,Word.not(width,Word.zero(width))) == word : Word(width)}:
  match width:
    case 0n:
      WNil{} = word
      {==}
    case 1n+ +rest:
      match word:
        case WCon{False{},+tail}:
          %and_all(rest,tail) : {WCon{False{},Word.and(rest,tail,Word.not(rest,Word.zero(rest)))} == WCon{False{},_} : Word(1n+rest)}
          {==}
        case WCon{True{},+tail}:
          %and_all(rest,tail) : {WCon{True{},Word.and(rest,tail,Word.not(rest,Word.zero(rest)))} == WCon{True{},_} : Word(1n+rest)}
          {==}

def or_zero(+width: Nat,+word: Word(width)) -> {Word.or(width,word,Word.zero(width)) == word : Word(width)}:
  match width:
    case 0n:
      WNil{} = word
      {==}
    case 1n+ +rest:
      match word:
        case WCon{False{},+tail}:
          %or_zero(rest,tail) : {WCon{False{},Word.or(rest,tail,Word.zero(rest))} == WCon{False{},_} : Word(1n+rest)}
          {==}
        case WCon{True{},+tail}:
          %or_zero(rest,tail) : {WCon{True{},Word.or(rest,tail,Word.zero(rest))} == WCon{True{},_} : Word(1n+rest)}
          {==}

def zero_or(+width: Nat,+word: Word(width)) -> {Word.or(width,Word.zero(width),word) == word : Word(width)}:
  match width:
    case 0n:
      WNil{} = word
      {==}
    case 1n+ +rest:
      WCon{+bit,+tail} = word
      %zero_or(rest,tail) : {WCon{bit,Word.or(rest,Word.zero(rest),tail)} == WCon{bit,_} : Word(1n+rest)}
      {==}

def native_and_all(word: U32) -> {U32.and(word,4294967295) == word : U32}:
  U32{+bits} = word
  Equal.cong(Word(32n),U32,word => U32{word},Word.and(32n,bits,Word.not(32n,Word.zero(32n))),bits,and_all(32n,bits))

def native_and_zero(word: U32) -> {U32.and(word,0) == 0 : U32}:
  U32{+bits} = word
  Equal.cong(Word(32n),U32,word => U32{word},Word.and(32n,bits,Word.zero(32n)),Word.zero(32n),WordEncoding.and_zero(32n,bits))

def native_or_zero(word: U32) -> {U32.or(word,0) == word : U32}:
  U32{+bits} = word
  Equal.cong(Word(32n),U32,word => U32{word},Word.or(32n,bits,Word.zero(32n)),bits,or_zero(32n,bits))

def native_zero_or(word: U32) -> {U32.or(0,word) == word : U32}:
  U32{+bits} = word
  Equal.cong(Word(32n),U32,word => U32{word},Word.or(32n,Word.zero(32n),bits),bits,zero_or(32n,bits))

def select_reflected(+mask: F32,+yes: F32,+no: F32) ->
  {Compare.select(mask,yes,no) == FP32.from_bits((U32.and(FP32.to_bits(yes),selector(mask)) + FP32.to_bits(no) * (selector(mask) + 1 : U32) : U32)) : F32}:
  F32{mask_word} = mask
  F32{yes_word} = yes
  F32{no_word} = no
  {==}

def union_reflected(+left: F32,+right: F32) -> {Compare.union(left,right) == FP32.from_bits(U32.or(FP32.to_bits(left),FP32.to_bits(right))) : F32}:
  F32{left_word} = left
  F32{right_word} = right
  {==}

def select_no(+mask: F32,+yes: F32,+no: F32,equal: {selector(mask) == 0 : U32}) -> {Compare.select(mask,yes,no) == no : F32}:
  %Equal.sym(F32,Compare.select(mask,yes,no),FP32.from_bits((U32.and(FP32.to_bits(yes),selector(mask)) + FP32.to_bits(no) * (selector(mask) + 1 : U32) : U32)),select_reflected(mask,yes,no)) : {_ == no : F32}
  %Equal.sym(U32,selector(mask),0,equal) : {FP32.from_bits((U32.and(FP32.to_bits(yes),_) + FP32.to_bits(no) * (_ + 1 : U32) : U32)) == no : F32}
  %Equal.sym(U32,U32.and(FP32.to_bits(yes),0),0,native_and_zero(FP32.to_bits(yes))) : {FP32.from_bits((_ + FP32.to_bits(no) * 1 : U32)) == no : F32}
  %Equal.sym(U32,(FP32.to_bits(no) * 1 : U32),FP32.to_bits(no),Arithmetic.multiply_one(FP32.to_bits(no))) : {FP32.from_bits((0 + _ : U32)) == no : F32}
  %Equal.sym(U32,(0 + FP32.to_bits(no) : U32),FP32.to_bits(no),Equal.trans(U32,(0 + FP32.to_bits(no) : U32),(FP32.to_bits(no) + 0 : U32),FP32.to_bits(no),U32.add_comm(0,FP32.to_bits(no)),Arithmetic.add_zero(FP32.to_bits(no)))) : {FP32.from_bits(_) == no : F32}
  Bits.value_round_trip(no)

def select_yes(+mask: F32,+yes: F32,+no: F32,equal: {selector(mask) == 4294967295 : U32}) -> {Compare.select(mask,yes,no) == yes : F32}:
  %Equal.sym(F32,Compare.select(mask,yes,no),FP32.from_bits((U32.and(FP32.to_bits(yes),selector(mask)) + FP32.to_bits(no) * (selector(mask) + 1 : U32) : U32)),select_reflected(mask,yes,no)) : {_ == yes : F32}
  %Equal.sym(U32,selector(mask),4294967295,equal) : {FP32.from_bits((U32.and(FP32.to_bits(yes),_) + FP32.to_bits(no) * (_ + 1 : U32) : U32)) == yes : F32}
  %Equal.sym(U32,U32.and(FP32.to_bits(yes),4294967295),FP32.to_bits(yes),native_and_all(FP32.to_bits(yes))) : {FP32.from_bits((_ + FP32.to_bits(no) * 0 : U32)) == yes : F32}
  %Equal.sym(U32,(FP32.to_bits(no) * 0 : U32),0,Arithmetic.multiply_zero_right(FP32.to_bits(no))) : {FP32.from_bits((FP32.to_bits(yes) + _ : U32)) == yes : F32}
  %Equal.sym(U32,(FP32.to_bits(yes) + 0 : U32),FP32.to_bits(yes),Arithmetic.add_zero(FP32.to_bits(yes))) : {FP32.from_bits(_) == yes : F32}
  Bits.value_round_trip(yes)

def union_zero_right(+value: F32) -> {Compare.union(value,0.0) == value : F32}:
  %Equal.sym(F32,Compare.union(value,0.0),FP32.from_bits(U32.or(FP32.to_bits(value),0)),union_reflected(value,0.0)) : {_ == value : F32}
  %Equal.sym(U32,U32.or(FP32.to_bits(value),0),FP32.to_bits(value),native_or_zero(FP32.to_bits(value))) : {FP32.from_bits(_) == value : F32}
  Bits.value_round_trip(value)

def union_zero_left(+value: F32) -> {Compare.union(0.0,value) == value : F32}:
  %Equal.sym(F32,Compare.union(0.0,value),FP32.from_bits(U32.or(0,FP32.to_bits(value))),union_reflected(0.0,value)) : {_ == value : F32}
  %Equal.sym(U32,U32.or(0,FP32.to_bits(value)),FP32.to_bits(value),native_zero_or(FP32.to_bits(value))) : {FP32.from_bits(_) == value : F32}
  Bits.value_round_trip(value)

def joined(+mask: F32,+yes: F32,+no: F32,evidence: MaskWord<selector(mask)>) -> {Compare.union(Compare.selected(yes,mask),Compare.rejected(no,mask)) == Compare.select(mask,yes,no) : F32}:
  match evidence:
    case MaskNone{+equal}:
      %Equal.sym(F32,Compare.select(mask,yes,no),no,select_no(mask,yes,no,equal)) : {Compare.union(Compare.selected(yes,mask),Compare.rejected(no,mask)) == _ : F32}
      %Equal.sym(F32,Compare.selected(yes,mask),0.0,select_no(mask,yes,0.0,equal)) : {Compare.union(_,Compare.rejected(no,mask)) == no : F32}
      %Equal.sym(F32,Compare.rejected(no,mask),no,select_no(mask,0.0,no,equal)) : {Compare.union(0.0,_) == no : F32}
      union_zero_left(no)
    case MaskAll{+equal}:
      %Equal.sym(F32,Compare.select(mask,yes,no),yes,select_yes(mask,yes,no,equal)) : {Compare.union(Compare.selected(yes,mask),Compare.rejected(no,mask)) == _ : F32}
      %Equal.sym(F32,Compare.selected(yes,mask),yes,select_yes(mask,yes,0.0,equal)) : {Compare.union(_,Compare.rejected(no,mask)) == yes : F32}
      %Equal.sym(F32,Compare.rejected(no,mask),0.0,select_yes(mask,0.0,no,equal)) : {Compare.union(yes,_) == yes : F32}
      union_zero_right(yes)

def tail_join(+mask: F32,+yes: F32,+no: F32) -> {Compare.union(Compare.selected(yes,mask),Compare.rejected(no,mask)) == Compare.select(mask,yes,no) : F32}:
  joined(mask,yes,no,mask_word(mask))

# Structural F32 equality is equality of all bits, including signed zero and
# arbitrary NaN payloads. The chosen operand undergoes no FP arithmetic.
def ChosenBits(+mask: F32,+yes: F32,+no: F32) -> Data:
  Either<&2,&2,Shape.Both<{selector(mask) == 0 : U32},{FP32.to_bits(Compare.select(mask,yes,no)) == FP32.to_bits(no) : U32}>,
    Shape.Both<{selector(mask) == 4294967295 : U32},{FP32.to_bits(Compare.select(mask,yes,no)) == FP32.to_bits(yes) : U32}>>

def chosen_bits_when(+mask: F32,+yes: F32,+no: F32,evidence: MaskWord<selector(mask)>) -> ChosenBits(mask,yes,no):
  match evidence:
    case MaskNone{+equal}: Inl{Shape.Both{equal,Equal.cong(F32,U32,FP32.to_bits,Compare.select(mask,yes,no),no,select_no(mask,yes,no,equal))}}
    case MaskAll{+equal}: Inr{Shape.Both{equal,Equal.cong(F32,U32,FP32.to_bits,Compare.select(mask,yes,no),yes,select_yes(mask,yes,no,equal))}}

def chosen_bits(+mask: F32,+yes: F32,+no: F32) -> ChosenBits(mask,yes,no):
  chosen_bits_when(mask,yes,no,mask_word(mask))

def magnitude_choice(+enabled: Bool,+mask: F32,+yes: F32,+no: F32,equal: {selector(mask) == Bool.pick(U32,enabled,4294967295,0) : U32}) -> {Compare.select(mask,yes,no) == Bool.pick(F32,enabled,yes,no) : F32}:
  match enabled:
    case False{}: select_no(mask,yes,no,equal)
    case True{}: select_yes(mask,yes,no,equal)

def select_magnitude(+mask: F32,+yes: F32,+no: F32) -> {Compare.select(mask,yes,no) == Bool.pick(F32,magnitude_nonzero(31n,mask_bits(mask)),yes,no) : F32}:
  magnitude_choice(magnitude_nonzero(31n,mask_bits(mask)),mask,yes,no,magnitude_selected(mask))

def picked_word(+enabled: Bool,+yes: F32,+no: F32) -> {FP32.to_bits(Bool.pick(F32,enabled,yes,no)) == Bool.pick(U32,enabled,FP32.to_bits(yes),FP32.to_bits(no)) : U32}:
  match enabled:
    case False{}: {==}
    case True{}: {==}

def selected_word(+mask: F32,+yes: F32,+no: F32) -> {FP32.to_bits(Compare.select(mask,yes,no)) == Bool.pick(U32,magnitude_nonzero(31n,mask_bits(mask)),FP32.to_bits(yes),FP32.to_bits(no)) : U32}:
  Equal.trans(U32,FP32.to_bits(Compare.select(mask,yes,no)),FP32.to_bits(Bool.pick(F32,magnitude_nonzero(31n,mask_bits(mask)),yes,no)),Bool.pick(U32,magnitude_nonzero(31n,mask_bits(mask)),FP32.to_bits(yes),FP32.to_bits(no)),Equal.cong(F32,U32,FP32.to_bits,Compare.select(mask,yes,no),Bool.pick(F32,magnitude_nonzero(31n,mask_bits(mask)),yes,no),select_magnitude(mask,yes,no)),picked_word(magnitude_nonzero(31n,mask_bits(mask)),yes,no))
