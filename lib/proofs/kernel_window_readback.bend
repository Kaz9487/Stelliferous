# A cell copied from a channel-group plane keeps the complete source cell.
import Base
import ./kernel_gather_blocks.bend as GatherBlocks
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./write_program.bend as Writes
import ./nat_order.bend as Order
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./natural_addition.bend as Addition

def plane_address(+group: Nat,+stride: U32,+start: U32,+column: Nat) ->
  {(U32.from_nat(group) * stride + start + U32.from_nat(column) : U32) ==
    U32.from_nat(Nat.add(Nat.add(Nat.mul(group,U32.to_nat(stride)),U32.to_nat(start)),column)) : U32}:
  +product = Nat.mul(group,U32.to_nat(stride))
  +base = Nat.add(product,U32.to_nat(start))
  %Arithmetic.native_roundtrip(stride) :
    {(U32.from_nat(group) * _ + start + U32.from_nat(column) : U32) == U32.from_nat(Nat.add(base,column)) : U32}
  %Equal.sym(U32,(U32.from_nat(group) * U32.from_nat(U32.to_nat(stride)) : U32),U32.from_nat(product),Multiplication.native_encoded_multiply(group,U32.to_nat(stride))) :
    {(_ + start + U32.from_nat(column) : U32) == U32.from_nat(Nat.add(base,column)) : U32}
  %Arithmetic.native_roundtrip(start) :
    {(U32.from_nat(product) + _ + U32.from_nat(column) : U32) == U32.from_nat(Nat.add(base,column)) : U32}
  %Equal.sym(U32,(U32.from_nat(product) + U32.from_nat(U32.to_nat(start)) : U32),U32.from_nat(base),WordArithmetic.native_add(product,U32.to_nat(start))) :
    {(_ + U32.from_nat(column) : U32) == U32.from_nat(Nat.add(base,column)) : U32}
  WordArithmetic.native_add(base,column)
