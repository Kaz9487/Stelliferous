# Selection and the tail's three masked passes preserve the chosen word.
import Base
import ../math_compare.bend as Compare
import ../math_fp32.bend as FP32
import ./math_fp32_bits.bend as Bits
import ./bounded_u32_arithmetic.bend as Arithmetic
import ../kernel_gemm.bend as KernelGemm

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

def select_reflected(+mask: F32,+yes: F32,+no: F32) ->
  {Compare.select(mask,yes,no) == FP32.from_bits((U32.and(FP32.to_bits(yes),selector(mask)) + FP32.to_bits(no) * (selector(mask) + 1 : U32) : U32)) : F32}:
  F32{mask_word} = mask
  F32{yes_word} = yes
  F32{no_word} = no
  {==}

def select_no(+mask: F32,+yes: F32,+no: F32,equal: {selector(mask) == 0 : U32}) -> {Compare.select(mask,yes,no) == no : F32}:
  %Equal.sym(F32,Compare.select(mask,yes,no),FP32.from_bits((U32.and(FP32.to_bits(yes),selector(mask)) + FP32.to_bits(no) * (selector(mask) + 1 : U32) : U32)),select_reflected(mask,yes,no)) : {_ == no : F32}
  %Equal.sym(U32,selector(mask),0,equal) : {FP32.from_bits((U32.and(FP32.to_bits(yes),_) + FP32.to_bits(no) * (_ + 1 : U32) : U32)) == no : F32}
  %Equal.sym(U32,U32.and(FP32.to_bits(yes),0),0,Bits.native_and_zero(FP32.to_bits(yes))) : {FP32.from_bits((_ + FP32.to_bits(no) * 1 : U32)) == no : F32}
  %Equal.sym(U32,(FP32.to_bits(no) * 1 : U32),FP32.to_bits(no),Arithmetic.multiply_one(FP32.to_bits(no))) : {FP32.from_bits((0 + _ : U32)) == no : F32}
  %Equal.sym(U32,(0 + FP32.to_bits(no) : U32),FP32.to_bits(no),Equal.trans(U32,(0 + FP32.to_bits(no) : U32),(FP32.to_bits(no) + 0 : U32),FP32.to_bits(no),U32.add_comm(0,FP32.to_bits(no)),Arithmetic.add_zero(FP32.to_bits(no)))) : {FP32.from_bits(_) == no : F32}
  Bits.value_round_trip(no)

def select_yes(+mask: F32,+yes: F32,+no: F32,equal: {selector(mask) == 4294967295 : U32}) -> {Compare.select(mask,yes,no) == yes : F32}:
  %Equal.sym(F32,Compare.select(mask,yes,no),FP32.from_bits((U32.and(FP32.to_bits(yes),selector(mask)) + FP32.to_bits(no) * (selector(mask) + 1 : U32) : U32)),select_reflected(mask,yes,no)) : {_ == yes : F32}
  %Equal.sym(U32,selector(mask),4294967295,equal) : {FP32.from_bits((U32.and(FP32.to_bits(yes),_) + FP32.to_bits(no) * (_ + 1 : U32) : U32)) == yes : F32}
  %Equal.sym(U32,U32.and(FP32.to_bits(yes),4294967295),FP32.to_bits(yes),Bits.native_and_all(FP32.to_bits(yes))) : {FP32.from_bits((_ + FP32.to_bits(no) * 0 : U32)) == yes : F32}
  %Equal.sym(U32,(FP32.to_bits(no) * 0 : U32),0,Arithmetic.multiply_zero_right(FP32.to_bits(no))) : {FP32.from_bits((FP32.to_bits(yes) + _ : U32)) == yes : F32}
  %Equal.sym(U32,(FP32.to_bits(yes) + 0 : U32),FP32.to_bits(yes),Arithmetic.add_zero(FP32.to_bits(yes))) : {FP32.from_bits(_) == yes : F32}
  Bits.value_round_trip(yes)

# Structural F32 equality is equality of all bits, including signed zero and
# arbitrary NaN payloads. The chosen operand undergoes no FP arithmetic.
def ChosenBits(+mask: F32,+yes: F32,+no: F32) -> Data:
  Either<&2,&2,KernelGemm.Both<{selector(mask) == 0 : U32},{FP32.to_bits(Compare.select(mask,yes,no)) == FP32.to_bits(no) : U32}>,
    KernelGemm.Both<{selector(mask) == 4294967295 : U32},{FP32.to_bits(Compare.select(mask,yes,no)) == FP32.to_bits(yes) : U32}>>

def chosen_bits_when(+mask: F32,+yes: F32,+no: F32,evidence: MaskWord<selector(mask)>) -> ChosenBits(mask,yes,no):
  match evidence:
    case MaskNone{+equal}: Inl{KernelGemm.Both{equal,Equal.cong(F32,U32,FP32.to_bits,Compare.select(mask,yes,no),no,select_no(mask,yes,no,equal))}}
    case MaskAll{+equal}: Inr{KernelGemm.Both{equal,Equal.cong(F32,U32,FP32.to_bits,Compare.select(mask,yes,no),yes,select_yes(mask,yes,no,equal))}}

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
