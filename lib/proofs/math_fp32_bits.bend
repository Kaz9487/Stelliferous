# An FP32 value and its 32 bits are the same word: the value rebuilt from its
# bits is the value, NaN payloads included, and the bits of a value built from
# a word are that word. A word select with every bit set gives the first
# value, with none the second.
import Base
import ../math_fp32.bend as Fp32
import ./word_encoding.bend as WordEncoding

def value_round_trip(value: F32) -> {Fp32.from_bits(Fp32.to_bits(value)) == value : F32}:
  match value:
    case F32{bits}: {==}

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

def native_and_all(word: U32) -> {U32.and(word,4294967295) == word : U32}:
  U32{+bits} = word
  Equal.cong(Word(32n),U32,word => U32{word},Word.and(32n,bits,Word.not(32n,Word.zero(32n))),bits,and_all(32n,bits))

def native_and_zero(word: U32) -> {U32.and(word,0) == 0 : U32}:
  U32{+bits} = word
  Equal.cong(Word(32n),U32,word => U32{word},Word.and(32n,bits,Word.zero(32n)),Word.zero(32n),WordEncoding.and_zero(32n,bits))

def xor_zero(+width: Nat,+word: Word(width)) -> {Word.xor(width,word,Word.zero(width)) == word : Word(width)}:
  match width:
    case 0n:
      WNil{} = word
      {==}
    case 1n+ +rest:
      match word:
        case WCon{False{},+tail}:
          %xor_zero(rest,tail) : {WCon{False{},Word.xor(rest,tail,Word.zero(rest))} == WCon{False{},_} : Word(1n+rest)}
          {==}
        case WCon{True{},+tail}:
          %xor_zero(rest,tail) : {WCon{True{},Word.xor(rest,tail,Word.zero(rest))} == WCon{True{},_} : Word(1n+rest)}
          {==}

# x xor (y xor x) is y.
def xor_cancel(+width: Nat,+kept: Word(width),+other: Word(width)) -> {Word.xor(width,other,Word.xor(width,kept,other)) == kept : Word(width)}:
  match width:
    case 0n:
      WNil{} = kept
      WNil{} = other
      {==}
    case 1n+ +rest:
      match kept other:
        case WCon{False{},+k} WCon{False{},+o}:
          %xor_cancel(rest,k,o) : {WCon{False{},Word.xor(rest,o,Word.xor(rest,k,o))} == WCon{False{},_} : Word(1n+rest)}
          {==}
        case WCon{False{},+k} WCon{True{},+o}:
          %xor_cancel(rest,k,o) : {WCon{False{},Word.xor(rest,o,Word.xor(rest,k,o))} == WCon{False{},_} : Word(1n+rest)}
          {==}
        case WCon{True{},+k} WCon{False{},+o}:
          %xor_cancel(rest,k,o) : {WCon{True{},Word.xor(rest,o,Word.xor(rest,k,o))} == WCon{True{},_} : Word(1n+rest)}
          {==}
        case WCon{True{},+k} WCon{True{},+o}:
          %xor_cancel(rest,k,o) : {WCon{True{},Word.xor(rest,o,Word.xor(rest,k,o))} == WCon{True{},_} : Word(1n+rest)}
          {==}

def selected_all(+yes: F32,+no: F32) -> {Fp32.selected(4294967295,yes,no) == yes : F32}:
  F32{+yes_word} = yes
  F32{+no_word} = no
  %Equal.sym(U32,U32.and(U32.xor(U32{yes_word},U32{no_word}),4294967295),U32.xor(U32{yes_word},U32{no_word}),native_and_all(U32.xor(U32{yes_word},U32{no_word}))) :
    {Fp32.from_bits(U32.xor(U32{no_word},_)) == F32{yes_word} : F32}
  Equal.cong(Word(32n),F32,word => F32{word},Word.xor(32n,no_word,Word.xor(32n,yes_word,no_word)),yes_word,xor_cancel(32n,yes_word,no_word))

def selected_none(+yes: F32,+no: F32) -> {Fp32.selected(0,yes,no) == no : F32}:
  F32{+yes_word} = yes
  F32{+no_word} = no
  %Equal.sym(U32,U32.and(U32.xor(U32{yes_word},U32{no_word}),0),0,native_and_zero(U32.xor(U32{yes_word},U32{no_word}))) :
    {Fp32.from_bits(U32.xor(U32{no_word},_)) == F32{no_word} : F32}
  Equal.cong(Word(32n),F32,word => F32{word},Word.xor(32n,no_word,Word.zero(32n)),no_word,xor_zero(32n,no_word))
