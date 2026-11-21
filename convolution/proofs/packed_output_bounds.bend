# All output-address bounds come from the unchanged runtime fast guard.
import Base
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ./convolution_guard_facts.bend as Guards
import ./concrete_support_guard.bend as Support
import ./supported_coordinate_bounds.bend as SupportedCoordinateBounds
import ./supported_padded_coordinates.bend as Padded
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/word_multiplication.bend as WordMultiplication

def channel_capacity_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,supported: Support.FastSupported<limits,problem>) ->
  {Nat.is_le(Guards.output_channel_count(problem),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(Guards.output_channel_count(problem),U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.capacity(limits)),
    SupportedCoordinateBounds.co_bound(limits,problem,Padded.scalar(limits,problem,supported)),
    Bounds.as_nat_le(Machine.dimension(limits),Machine.capacity(limits),Limits.dimension_within_capacity(limits)))

def padded_elements_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,supported: Support.FastSupported<limits,problem>) ->
  {Nat.is_le(Nat.mul(Support.padded(problem),Guards.output_channel_count(problem)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  ConvSpec.Problem{shape,input,weights,bias} = problem
  ConvSpec.Shape{input_channel_count,output_channel_count,height,width,kernel,stride,padding} = shape
  Support.FastSupported{scalar,packing,output,planes} = supported
  output

def elements_bound(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,supported: Support.FastSupported<limits,problem>,
  span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)),Nat.mul(U32.to_nat(span),Guards.output_channel_count(problem)),RowMajor.multiply_commutative(Guards.output_channel_count(problem),U32.to_nat(span))) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Order.transitive(Nat.mul(U32.to_nat(span),Guards.output_channel_count(problem)),Nat.mul(Support.padded(problem),Guards.output_channel_count(problem)),U32.to_nat(Machine.capacity(limits)),
    RowMajor.mul_le(U32.to_nat(span),Support.padded(problem),Guards.output_channel_count(problem),span_bound),padded_elements_bound(limits,problem,supported))

def count_exact(+limits: Machine.Limits,+problem: ConvSpec.Problem,+span: U32,+supported: Support.FastSupported<limits,problem>,
  span_bound: {Nat.is_le(U32.to_nat(span),Support.padded(problem)) == True{} : Bool}) ->
  {U32.to_nat((U32.from_nat(Guards.output_channel_count(problem)) * span : U32)) == Nat.mul(Guards.output_channel_count(problem),U32.to_nat(span)) : Nat}:
  +channels_exact = SupportedCoordinateBounds.co_exact(limits,problem,Padded.scalar(limits,problem,supported))
  %channels_exact : {U32.to_nat((U32.from_nat(Guards.output_channel_count(problem)) * span : U32)) == Nat.mul(_,U32.to_nat(span)) : Nat}
  WordMultiplication.bounded_multiply(U32.from_nat(Guards.output_channel_count(problem)),span,Machine.capacity(limits),Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(Guards.output_channel_count(problem))),Guards.output_channel_count(problem),channels_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(span)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    elements_bound(limits,problem,span,supported,span_bound))
