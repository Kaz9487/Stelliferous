# The incrementing U32 store schedule is exact, in bounds and injective.
import Base
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./natural_addition.bend as Addition
import ../kernel_gemm.bend as KernelGemm

def encoded_distinct(+left: Nat,+right: Nat,+length: U32,
  +ordered: {Nat.is_lt(left,right) == True{} : Bool},+right_inside: {Nat.is_le(right,U32.to_nat(length)) == True{} : Bool}) ->
  {U32.is_eq(U32.from_nat(left),U32.from_nat(right)) == False{} : Bool}:
  %Equal.sym(Bool,U32.is_eq(U32.from_nat(left),U32.from_nat(right)),Nat.is_eq(U32.to_nat(U32.from_nat(left)),U32.to_nat(U32.from_nat(right))),Comparison.eq(U32.from_nat(left),U32.from_nat(right))) : {_ == False{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(left)),left,FullRange.roundtrip(left,length,Bounds.strict_le(left,U32.to_nat(length),Bounds.nat_trans(left,right,U32.to_nat(length),ordered,right_inside)))) :
    {Nat.is_eq(_,U32.to_nat(U32.from_nat(right))) == False{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(right)),right,FullRange.roundtrip(right,length,right_inside)) : {Nat.is_eq(left,_) == False{} : Bool}
  Order.less_not_equal(left,right,ordered)
