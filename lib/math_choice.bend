# One of two values by a flag, and the extrema built on it, compiled for each
# type (docs/toolchain_rules.md F16).
import Base

def chosen(~A: Type,flag: Bool,yes: A,no: A) -> A:
  match flag:
    case False{}: no
    case True{}: yes

def larger(+left: F32,+right: F32) -> F32:
  chosen(~F32,F32.is_lt(left,right),right,left)

def smaller(+left: F32,+right: F32) -> F32:
  chosen(~F32,F32.is_lt(left,right),left,right)

def larger_word(+left: U32,+right: U32) -> U32:
  chosen(~U32,U32.is_lt(left,right),right,left)
