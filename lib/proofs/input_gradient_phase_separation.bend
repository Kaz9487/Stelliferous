# Channel block counts and bounds of the selected phase.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_extent.bend as Extent
import ./machine_limit_bounds.bend as Limits
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_division.bend as NativeDivision

def block_count(+geometry: Kernel.Geometry) -> U32:
  ((Domain.channel_word(geometry) + 7) / 8 : U32)

def block_count_exact(+call: Domain.Call) ->
  {U32.to_nat(block_count(Domain.geometry(call))) == Nat.div(Nat.add(Domain.channels(Domain.geometry(call)),7n),8n) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +sum_exact = Arithmetic.add(c,7,Machine.half_word(limits),Limits.half_word_fits(limits),Extent.channel_room(limits,U32.to_nat(c),Domain.channels_room(call)))
  %sum_exact : {U32.to_nat(block_count(geometry)) == Nat.div(_,8n) : Nat}
  NativeDivision.divide((c + 7 : U32),8,Machine.half_word(limits),Limits.half_word_fits(limits),Bounds.as_nat_le(16,Machine.half_word(limits),Limits.sixteen_within_half(limits)),{==})
