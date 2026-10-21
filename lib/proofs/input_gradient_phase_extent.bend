# The channel count plus padding stays inside the machine half-word.
import Base
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./machine_limit_bounds.bend as Limits
import ../machine_limits.bend as Machine
import ./encoded_coordinates.bend as Encoded

def channel_room(+limits: Machine.Limits,+channels: Nat,room: {Nat.is_le(channels,U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(channels,7n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +capacity = U32.to_nat(Machine.capacity(limits))
  %Equal.sym(Nat,U32.to_nat(Machine.half_word(limits)),Nat.double(capacity),Limits.doubled_capacity(limits)) :
    {Nat.is_le(Nat.add(channels,7n),_) == True{} : Bool}
  %Addition.double(capacity) : {Nat.is_le(Nat.add(channels,7n),_) == True{} : Bool}
  Encoded.sum_bounds(channels,7n,capacity,capacity,Limits.dimension_capacity(limits,channels,room),Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits)))
