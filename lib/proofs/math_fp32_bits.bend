# An FP32 value and its 32 bits are the same word: the value rebuilt from its
# bits is the value, NaN payloads included.
import Base
import ../math_fp32.bend as Fp32

def value_round_trip(value: F32) -> {Fp32.from_bits(Fp32.to_bits(value)) == value : F32}:
  match value:
    case F32{bits}: {==}
