# Stored element formats and the FP32 value of each. An element arrives as its
# bytes, least significant first, gathered into a low word and a high word.
import Base
import ./math_fp32.bend as Fp32

type Format is Data:
  Fp32{}
  Fp64{}
  Fp16{}
  BFloat16{}
  Int64{}
  Int32{}

# Bytes per element.
def width(format: Format) -> U32:
  match format:
    case Fp32{}: 4
    case Fp64{}: 8
    case Fp16{}: 2
    case BFloat16{}: 2
    case Int64{}: 8
    case Int32{}: 4

# IEEE half: a sign, five exponent bits and ten fraction bits. A normal value
# keeps its fraction and gains 112 on the exponent; infinities and NaNs keep
# their payload; a subnormal is normalized by doubling the fraction until its
# leading bit is at position ten, so widening is exact.

def normalized(fuel: Nat,done: Bool,+sign: U32,+exponent: U32,+mantissa: U32) -> F32:
  match fuel done:
    case fuel True{}: Fp32.from_bits((sign + exponent * 8388608 + (mantissa - 1024) * 8192 : U32))
    case 0n False{}: Fp32.from_bits(sign)
    case 1n+rest False{}:
      +doubled = (mantissa * 2 : U32)
      normalized(rest,U32.is_le(1024,doubled),sign,(exponent - 1 : U32),doubled)

def subnormal(zero: Bool,+sign: U32,+mantissa: U32) -> F32:
  match zero:
    case True{}: Fp32.from_bits(sign)
    case False{}: normalized(11n,False{},sign,113,mantissa)

def half_classified(lowest: Bool,highest: Bool,+sign: U32,+exponent: U32,+mantissa: U32) -> F32:
  match lowest highest:
    case True{} highest: subnormal(U32.is_zero(mantissa),sign,mantissa)
    case False{} True{}: Fp32.from_bits((sign + 2139095040 + mantissa * 8192 : U32))
    case False{} False{}: Fp32.from_bits((sign + (exponent + 112) * 8388608 + mantissa * 8192 : U32))

def half(+word: U32) -> F32:
  +exponent = (word / 1024 % 32 : U32)
  half_classified(U32.is_zero(exponent),U32.is_eq(exponent,31),(word / 32768 % 2 * 2147483648 : U32),exponent,(word % 1024 : U32))

# A two's complement integer rounds to the nearest FP32, ties to even, as a C
# conversion does. The magnitude is scaled until its leading bit is at position
# 23 of the low word: a small one is doubled, which is exact, and a large one is
# halved while the dropped bits are kept as a guard bit and a sticky bit.

def carried(full: Bool,+sign: U32,+exponent: U32,+low: U32) -> F32:
  match full:
    case True{}: Fp32.from_bits((sign + (exponent + 1) * 8388608 : U32))
    case False{}: Fp32.from_bits((sign + exponent * 8388608 + (low - 8388608) : U32))

def rounded(up: Bool,+sign: U32,+exponent: U32,+low: U32) -> F32:
  match up:
    case False{}: Fp32.from_bits((sign + exponent * 8388608 + (low - 8388608) : U32))
    case True{}: carried(U32.is_eq((low + 1 : U32),16777216),sign,exponent,(low + 1 : U32))

def halved(fuel: Nat,done: Bool,+sign: U32,+exponent: U32,+low: U32,+high: U32,+guard: U32,+sticky: U32) -> F32:
  match fuel done:
    case fuel True{}: rounded(U32.is_eq(guard,1) && (U32.is_eq(sticky,1) || U32.is_eq((low % 2 : U32),1)),sign,exponent,low)
    case 0n False{}: Fp32.from_bits(sign)
    case 1n+rest False{}:
      +next_low = (low / 2 + high % 2 * 2147483648 : U32)
      +next_high = (high / 2 : U32)
      halved(rest,U32.is_zero(next_high) && U32.is_lt(next_low,16777216),sign,(exponent + 1 : U32),next_low,next_high,(low % 2 : U32),U32.max(sticky,guard))

def doubled(fuel: Nat,done: Bool,+sign: U32,+exponent: U32,+low: U32,+high: U32) -> F32:
  match fuel done:
    case fuel True{}: halved(41n,U32.is_zero(high) && U32.is_lt(low,16777216),sign,exponent,low,high,0,0)
    case 0n False{}: Fp32.from_bits(sign)
    case 1n+rest False{}:
      +twice = (low * 2 : U32)
      doubled(rest,U32.is_le(8388608,twice),sign,(exponent - 1 : U32),twice,high)

def scaled(zero: Bool,+sign: U32,+low: U32,+high: U32) -> F32:
  match zero:
    case True{}: Fp32.from_bits(0)
    case False{}: doubled(24n,U32.is_le(8388608,low) || Bool.not(U32.is_zero(high)),sign,150,low,high)

def magnitude(negative: Bool,+low: U32,+high: U32) -> F32:
  match negative:
    case False{}: scaled(U32.is_zero(low) && U32.is_zero(high),0,low,high)
    case True{}: scaled(False{},2147483648,(0 - low : U32),(0 - high - Bool.pick(U32,U32.is_zero(low),0,1) : U32))

def integer(+low: U32,+high: U32) -> F32:
  magnitude(U32.is_le(2147483648,high),low,high)

def extended(+low: U32) -> U32:
  Bool.pick(U32,U32.is_le(2147483648,low),4294967295,0)

# IEEE double: a sign, eleven exponent bits and 52 fraction bits, the top 20 of
# them in the high word. It rounds to the nearest FP32, ties to even, as a C
# conversion does. With its leading bit restored, the fraction is a 53-bit
# integer significand: a normal FP32 is that integer halved as above, starting
# from the exponent that keeps the value; an FP32 subnormal is the significand
# shifted to the subnormal scale with the same guard and sticky bits, which
# carries into the smallest normal when it rounds up; anything smaller is a
# signed zero. An infinity stays infinite, and so does a value above the FP32
# range. A NaN stays a quiet NaN that keeps the top of its payload.

def shifted(fuel: Nat,done: Bool,+sign: U32,+remaining: U32,+low: U32,+high: U32,+guard: U32,+sticky: U32) -> F32:
  match fuel done:
    case fuel True{}:
      Fp32.from_bits((sign + low + Bool.pick(U32,U32.is_eq(guard,1) && (U32.is_eq(sticky,1) || U32.is_eq((low % 2 : U32),1)),1,0) : U32))
    case 0n False{}: Fp32.from_bits(sign)
    case 1n+rest False{}:
      shifted(rest,U32.is_eq(remaining,1),sign,(remaining - 1 : U32),(low / 2 + high % 2 * 2147483648 : U32),(high / 2 : U32),(low % 2 : U32),U32.max(sticky,guard))

def special(infinite: Bool,+sign: U32,+low: U32,+fraction: U32) -> F32:
  match infinite:
    case True{}: Fp32.from_bits((sign + 2139095040 : U32))
    case False{}: Fp32.from_bits((sign + 2143289344 + (fraction * 8 + low / 536870912) % 4194304 : U32))

def double_classified(nonfinite: Bool,large: Bool,normal: Bool,small: Bool,+sign: U32,+exponent: U32,+low: U32,+fraction: U32) -> F32:
  match nonfinite large normal small:
    case True{} large normal small: special(U32.is_zero(fraction) && U32.is_zero(low),sign,low,fraction)
    case False{} True{} normal small: Fp32.from_bits((sign + 2139095040 : U32))
    case False{} False{} True{} small: halved(41n,False{},sign,(exponent - 925 : U32),low,(fraction + 1048576 : U32),0,0)
    case False{} False{} False{} True{}: shifted(55n,False{},sign,(926 - exponent : U32),low,(fraction + 1048576 : U32),0,0)
    case False{} False{} False{} False{}: Fp32.from_bits(sign)

def double(+low: U32,+high: U32) -> F32:
  +exponent = (high / 1048576 % 2048 : U32)
  double_classified(U32.is_eq(exponent,2047),U32.is_le(1151,exponent),U32.is_le(897,exponent),U32.is_le(872,exponent),
    (high / 2147483648 * 2147483648 : U32),exponent,low,(high % 1048576 : U32))

# The value of the element whose first four bytes form `low` and whose later
# bytes form `high`.
def widen(format: Format,+low: U32,+high: U32) -> F32:
  match format:
    case Fp32{}: Fp32.from_bits(low)
    case Fp64{}: double(low,high)
    case Fp16{}: half(low)
    case BFloat16{}: Fp32.from_bits((low * 65536 : U32))
    case Int64{}: integer(low,high)
    case Int32{}: integer(low,extended(low))
