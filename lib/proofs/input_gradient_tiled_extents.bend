# One axis of a validated call: the phases it folds over, and the count and
# taps of a phase.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_tiled_geometry.bend as Geometry
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_geometry.bend as Phase
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_order.bend as Order
import ./nat_extrema.bend as Extrema
import ./u32_extrema.bend as NativeExtrema
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./machine_limit_bounds.bend as Limits

type Axis is Data:
  Axis{call: Domain.Call,size: U32,size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},envelope: Geometry.WordEnvelope}
def call(axis: Axis) -> Domain.Call:
  Axis{call,size,room,envelope} = axis
  call
def geometry(axis: Axis) -> Kernel.Geometry:
  Domain.geometry(call(axis))
def limits(axis: Axis) -> Machine.Limits:
  Domain.limits(call(axis))
def extent(axis: Axis) -> U32:
  Axis{call,size,room,envelope} = axis
  size
def envelope(axis: Axis) -> Geometry.WordEnvelope:
  Axis{call,size,room,envelope} = axis
  envelope
def count(axis: Axis) -> Nat:
  +axis = axis
  U32.to_nat(U32.min(Domain.stride_word(geometry(axis)),extent(axis)))

# The rows and the columns of a call.
def height_room(+call: Domain.Call) ->
  {Nat.is_le(U32.to_nat(Domain.height_word(Domain.geometry(call))),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Domain.height_room(call)
def width_room(+call: Domain.Call) ->
  {Nat.is_le(U32.to_nat(Domain.width_word(Domain.geometry(call))),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Domain.width_room(call)
def height_bound(+geometry: Kernel.Geometry,+value: Nat,+bound: {Nat.is_le(value,Domain.height(geometry)) == True{} : Bool}) ->
  {Nat.is_le(value,U32.to_nat(Domain.height_word(geometry))) == True{} : Bool}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  bound
def rows(+call: Domain.Call,+envelope: Geometry.WordEnvelope) -> Axis:
  Axis{call,Domain.height_word(Domain.geometry(call)),height_room(call),envelope}
def columns(+call: Domain.Call,+envelope: Geometry.WordEnvelope) -> Axis:
  Axis{call,Domain.width_word(Domain.geometry(call)),width_room(call),envelope}

def count_size_room(+axis: Axis) -> {Nat.is_le(count(axis),U32.to_nat(extent(axis))) == True{} : Bool}:
  Axis{call,size,room,envelope} = axis
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  %Equal.sym(Nat,count(axis),Nat.min(Domain.stride(geometry),U32.to_nat(size)),NativeExtrema.minimum(Domain.stride_word(geometry),size)) : {Nat.is_le(_,U32.to_nat(size)) == True{} : Bool}
  Extrema.minimum_right(Domain.stride(geometry),U32.to_nat(size))
def count_stride_room(+axis: Axis) -> {Nat.is_le(count(axis),Domain.stride(geometry(axis))) == True{} : Bool}:
  Axis{call,size,room,envelope} = axis
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  %Equal.sym(Nat,count(axis),Nat.min(Domain.stride(geometry),U32.to_nat(size)),NativeExtrema.minimum(Domain.stride_word(geometry),size)) : {Nat.is_le(_,Domain.stride(geometry)) == True{} : Bool}
  Extrema.minimum_left(Domain.stride(geometry),U32.to_nat(size))
def count_room(+axis: Axis) -> {Nat.is_le(count(axis),U32.to_nat(Machine.dimension(limits(axis)))) == True{} : Bool}:
  Axis{call,size,room,envelope} = axis
  Order.transitive(count(axis),U32.to_nat(size),U32.to_nat(Machine.dimension(Domain.limits(call))),count_size_room(axis),room)
def index_exact(+axis: Axis,+index: Nat,+inside: {Nat.is_lt(index,count(axis)) == True{} : Bool}) -> {U32.to_nat(U32.from_nat(index)) == index : Nat}:
  Bounds.roundtrip(index,Machine.dimension(limits(axis)),Limits.dimension_fits(limits(axis)),Bounds.strict_le(index,U32.to_nat(Machine.dimension(limits(axis))),Bounds.nat_trans(index,count(axis),U32.to_nat(Machine.dimension(limits(axis))),inside,count_room(axis))))
def index_inside_extent(+axis: Axis,+index: Nat,+inside: {Nat.is_lt(index,count(axis)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(U32.from_nat(index)),U32.to_nat(extent(axis))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,index_exact(axis,index,inside)) : {Nat.is_lt(_,U32.to_nat(extent(axis))) == True{} : Bool}
  Bounds.nat_trans(index,count(axis),U32.to_nat(extent(axis)),inside,count_size_room(axis))

# A phase is used when it reads at least one tap.
def native_axis(axis: Axis,index: U32) -> Kernel.Axis:
  Axis{+call,size,room,envelope} = axis
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Kernel.axis(index,size,k,s,p)
def used(axis: Axis,index: U32) -> Bool:
  U32.is_lt(0,Phase.axis_taps(native_axis(axis,index)))
def axis_count(axis: Axis,index: U32) -> U32:
  Phase.axis_count(native_axis(axis,index))
def axis_taps(axis: Axis,index: U32) -> U32:
  Phase.axis_taps(native_axis(axis,index))

def count_sample_bound(+axis: Axis,+index: Nat,+inside: {Nat.is_lt(index,count(axis)) == True{} : Bool},enabled: {used(axis,U32.from_nat(index)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(axis_count(axis,U32.from_nat(index))),U32.to_nat(extent(axis))) == True{} : Bool}:
  Axis{call,size,room,envelope} = axis
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +stride_positive = Domain.stride_positive(call)
  +first = U32.from_nat(index)
  +first_inside = index_inside_extent(axis,index,inside)
  %Equal.sym(Nat,U32.to_nat(axis_count(axis,first)),1n+Nat.div(Phase.tail(size,first),U32.to_nat(s)),Phase.count_exact(first,size,k,s,p,Machine.coordinate(limits),Limits.coordinate_fits(limits),PhaseBounds.dimension_room(limits,U32.to_nat(size),room),PhaseBounds.stride_double_room(call),stride_positive,first_inside)) :
    {Nat.is_le(_,U32.to_nat(size)) == True{} : Bool}
  Phase.count_room(size,first,s,stride_positive,first_inside)

def count_positive(+axis: Axis,+index: Nat,+inside: {Nat.is_lt(index,count(axis)) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(axis_count(axis,U32.from_nat(index)))) == True{} : Bool}:
  Axis{call,size,room,envelope} = axis
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +first = U32.from_nat(index)
  %Equal.sym(Nat,U32.to_nat(axis_count(axis,first)),1n+Nat.div(Phase.tail(size,first),U32.to_nat(s)),Phase.count_exact(first,size,k,s,p,Machine.coordinate(limits),Limits.coordinate_fits(limits),PhaseBounds.dimension_room(limits,U32.to_nat(size),room),PhaseBounds.stride_double_room(call),Domain.stride_positive(call),index_inside_extent(axis,index,inside))) :
    {Nat.is_lt(0n,_) == True{} : Bool}
  {==}
