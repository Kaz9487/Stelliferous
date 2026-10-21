# Native band clipping reads back the same ceiling indices as the band model.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_tiled_geometry.bend as Geometry
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./nat_ceiling.bend as Ceiling
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./machine_limit_bounds.bend as Limits
import ./u32_comparison.bend as Comparison
import ./input_gradient_tiled_bands.bend as Bands
import ./storage_shape.bend as Shape
import ./u32_extrema.bend as NativeExtrema
import ./input_gradient_tiled_axes.bend as Axes
import ./boolean_logic.bend as Boolean
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_geometry.bend as Phase
import ./natural_addition.bend as Addition
import ./nat_division_order.bend as DivisionOrder
import ./u32_full_range.bend as FullRange
import ./word_subtraction.bend as Subtraction

def clipped(+phase: U32,+first: U32,+stride: U32) -> U32:
  Bool.pick(U32,U32.is_lt(phase,first),((first - phase + stride - 1) / stride : U32),0)

def zero_ceiling(+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) -> {Ceiling.ceiling(0n,stride) == 0n : Nat}:
  %Equal.sym(Nat & Nat,Nat.divmod(0n,stride),(0n,0n),Division.small(0n,stride,positive)) : {Ceiling.rounded(Pair.fst(Nat,Nat,_),Pair.snd(Nat,Nat,_)) == 0n : Nat}
  {==}

def clipped_matches_case(enabled: Bool,+phase: U32,+first: U32,+stride: U32,+limits: Machine.Limits,
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +stride_room: {Nat.is_le(U32.to_nat(stride),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +compared: {U32.is_lt(phase,first) == enabled : Bool}) ->
  {U32.to_nat(clipped(phase,first,stride)) == Ceiling.ceiling(Nat.sub(U32.to_nat(first),U32.to_nat(phase)),U32.to_nat(stride)) : Nat}:
  match enabled:
    case False{}:
      +ordered = Order.ge_true(U32.to_nat(phase),U32.to_nat(first),Order.comparison_not_less(Nat.cmp(U32.to_nat(phase),U32.to_nat(first)),
        %Comparison.lt(phase,first) : {_ == False{} : Bool} compared))
      +empty = Order.subtract_below(U32.to_nat(first),U32.to_nat(phase),ordered)
      %Equal.sym(Bool,U32.is_lt(phase,first),False{},compared) :
        {U32.to_nat(Bool.pick(U32,_,((first - phase + stride - 1) / stride : U32),0)) == Ceiling.ceiling(Nat.sub(U32.to_nat(first),U32.to_nat(phase)),U32.to_nat(stride)) : Nat}
      %Equal.sym(Nat,Nat.sub(U32.to_nat(first),U32.to_nat(phase)),0n,empty) : {0n == Ceiling.ceiling(_,U32.to_nat(stride)) : Nat}
      Equal.sym(Nat,Ceiling.ceiling(0n,U32.to_nat(stride)),0n,zero_ceiling(U32.to_nat(stride),positive))
    case True{}:
      +coordinate = Machine.coordinate(limits)
      +difference = (first - phase : U32)
      +natural = Nat.sub(U32.to_nat(first),U32.to_nat(phase))
      +room = Order.transitive(natural,U32.to_nat(first),U32.to_nat(Machine.dimension(limits)),Order.sub_le(U32.to_nat(first),U32.to_nat(phase)),first_room)
      +exact = Arithmetic.subtract(first,phase,coordinate,Limits.coordinate_fits(limits),Bounds.strict_le(U32.to_nat(phase),U32.to_nat(first),Bounds.as_nat_lt(phase,first,compared)),PhaseBounds.dimension_room(limits,natural,room))
      +native_room = {%Equal.sym(Nat,U32.to_nat(difference),natural,exact) : {Nat.is_le(_,U32.to_nat(Machine.dimension(limits))) == True{} : Bool} room : {Nat.is_le(U32.to_nat(difference),U32.to_nat(Machine.dimension(limits))) == True{} : Bool}}
      %Equal.sym(Bool,U32.is_lt(phase,first),True{},compared) :
        {U32.to_nat(Bool.pick(U32,_,((first - phase + stride - 1) / stride : U32),0)) == Ceiling.ceiling(natural,U32.to_nat(stride)) : Nat}
      Equal.trans(Nat,U32.to_nat(Kernel.taps(difference,stride)),Ceiling.ceiling(U32.to_nat(difference),U32.to_nat(stride)),Ceiling.ceiling(natural,U32.to_nat(stride)),
        Geometry.taps_exact(difference,stride,limits,native_room,stride_room,positive),
        Equal.cong(Nat,Nat,value => Ceiling.ceiling(value,U32.to_nat(stride)),U32.to_nat(difference),natural,exact))

def clipped_matches(+phase: U32,+first: U32,+stride: U32,+limits: Machine.Limits,
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +stride_room: {Nat.is_le(U32.to_nat(stride),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {U32.to_nat(clipped(phase,first,stride)) == Ceiling.ceiling(Nat.sub(U32.to_nat(first),U32.to_nat(phase)),U32.to_nat(stride)) : Nat}:
  clipped_matches_case(U32.is_lt(phase,first),phase,first,stride,limits,first_room,stride_room,positive,{==})

type Indices is Data:
  Indices{low: U32,high: U32}
def indices(+axis: Kernel.Axis,+stride: U32,+first: U32,+last: U32) -> Indices:
  Kernel.Axis{+start,+count,tap,taps,source} = axis
  Indices{clipped(start,first,stride),U32.min(count,clipped(start,last,stride))}
def low(indices: Indices) -> U32:
  Indices{low,high} = indices
  low
def high(indices: Indices) -> U32:
  Indices{low,high} = indices
  high

def model(+axis: Kernel.Axis,+stride: U32,+first: U32,+last: U32,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) -> Bands.Band:
  Kernel.Axis{+start,+count,tap,taps,source} = axis
  Bands.Band{U32.to_nat(start),U32.to_nat(count),U32.to_nat(stride),U32.to_nat(first),U32.to_nat(last),positive}

def indices_matches(+axis: Kernel.Axis,+stride: U32,+first: U32,+last: U32,+limits: Machine.Limits,
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +last_room: {Nat.is_le(U32.to_nat(last),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +stride_room: {Nat.is_le(U32.to_nat(stride),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  Shape.Both<{U32.to_nat(low(indices(axis,stride,first,last))) == Bands.low(model(axis,stride,first,last,positive)) : Nat},
    {U32.to_nat(high(indices(axis,stride,first,last))) == Bands.high(model(axis,stride,first,last,positive)) : Nat}>:
  Kernel.Axis{+start,+count,tap,taps,source} = axis
  +last_index = clipped(start,last,stride)
  Shape.Both{clipped_matches(start,first,stride,limits,first_room,stride_room,positive),
    Equal.trans(Nat,U32.to_nat(U32.min(count,last_index)),Nat.min(U32.to_nat(count),U32.to_nat(last_index)),Bands.high(model(axis,stride,first,last,positive)),
      NativeExtrema.minimum(count,last_index),
      Equal.cong(Nat,Nat,value => Nat.min(U32.to_nat(count),value),U32.to_nat(last_index),Ceiling.ceiling(Nat.sub(U32.to_nat(last),U32.to_nat(start)),U32.to_nat(stride)),clipped_matches(start,last,stride,limits,last_room,stride_room,positive)))}

# This is the exact data used by band_measure, not another clipping algorithm.
def native_used(+axis: Kernel.Axis,+stride: U32,+first: U32,+last: U32) -> Bool:
  Kernel.Axis{start,count,tap,+taps,source} = axis
  U32.is_lt(0,taps) && U32.is_lt(low(indices(axis,stride,first,last)),high(indices(axis,stride,first,last)))
def used_matches(+axis: Kernel.Axis,+kernel: U32,+stride: U32,+first: U32,+last: U32) ->
  {native_used(axis,stride,first,last) == Axes.band_sample_used(Axes.band_sample_from(axis,kernel,stride,first,last)) : Bool}:
  Kernel.Axis{start,count,tap,taps,source} = axis
  {==}

def nonempty_from(+axis: Kernel.Axis,+stride: U32,+first: U32,+last: U32,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +used: {native_used(axis,stride,first,last) == True{} : Bool},
  facts: Shape.Both<{U32.to_nat(low(indices(axis,stride,first,last))) == Bands.low(model(axis,stride,first,last,positive)) : Nat},
    {U32.to_nat(high(indices(axis,stride,first,last))) == Bands.high(model(axis,stride,first,last,positive)) : Nat}>) ->
  {Nat.is_lt(Bands.low(model(axis,stride,first,last,positive)),Bands.high(model(axis,stride,first,last,positive))) == True{} : Bool}:
  Kernel.Axis{start,count,tap,+taps,source} = axis
  Shape.Both{+low_exact,+high_exact} = facts
  %low_exact : {Nat.is_lt(_,Bands.high(model(axis,stride,first,last,positive))) == True{} : Bool}
  %high_exact : {Nat.is_lt(U32.to_nat(low(indices(axis,stride,first,last))),_) == True{} : Bool}
  Bounds.as_nat_lt(low(indices(axis,stride,first,last)),high(indices(axis,stride,first,last)),
    Boolean.right(U32.is_lt(0,taps),U32.is_lt(low(indices(axis,stride,first,last)),high(indices(axis,stride,first,last))),used))

def nonempty(+axis: Kernel.Axis,+stride: U32,+first: U32,+last: U32,+limits: Machine.Limits,
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +last_room: {Nat.is_le(U32.to_nat(last),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +stride_room: {Nat.is_le(U32.to_nat(stride),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+used: {native_used(axis,stride,first,last) == True{} : Bool}) ->
  {Nat.is_lt(Bands.low(model(axis,stride,first,last,positive)),Bands.high(model(axis,stride,first,last,positive))) == True{} : Bool}:
  nonempty_from(axis,stride,first,last,positive,used,indices_matches(axis,stride,first,last,limits,first_room,last_room,stride_room,positive))

# The clipped indices translate the phase origin to the actual output row.
def row(+geometry: Kernel.Geometry,+phase: U32,+position: U32) -> Nat:
  Nat.add(U32.to_nat(phase),Nat.mul(U32.to_nat(position),Domain.stride(geometry)))
def point(+geometry: Kernel.Geometry,+phase: U32,+position: U32) -> U32:
  U32.from_nat(row(geometry,phase,position))
def row_padding(+phase: Nat,+position: Nat,+stride: Nat,+padding: Nat) ->
  {Nat.add(Nat.add(phase,Nat.mul(position,stride)),padding) == Nat.add(Nat.mul(position,stride),Nat.add(phase,padding)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(phase,Nat.mul(position,stride)),padding),Nat.add(Nat.add(Nat.mul(position,stride),phase),padding),Nat.add(Nat.mul(position,stride),Nat.add(phase,padding)),
    Equal.cong(Nat,Nat,value => Nat.add(value,padding),Nat.add(phase,Nat.mul(position,stride)),Nat.add(Nat.mul(position,stride),phase),Order.commutative(phase,Nat.mul(position,stride))),Addition.associative(Nat.mul(position,stride),phase,padding))

def natural_shift(+phase: Nat,+position: Nat,+padding: Nat,+stride: Nat,+kernel: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {Nat.add(Nat.div(Nat.add(Nat.add(phase,Nat.mul(position,stride)),padding),stride),1n+kernel) == Nat.add(position,Nat.add(Nat.div(Nat.add(phase,padding),stride),1n+kernel)) : Nat}:
  +padded = Nat.add(Nat.add(phase,Nat.mul(position,stride)),padding)
  +shifted = Nat.add(Nat.mul(position,stride),Nat.add(phase,padding))
  +base = Nat.div(Nat.add(phase,padding),stride)
  Equal.trans(Nat,Nat.add(Nat.div(padded,stride),1n+kernel),Nat.add(Nat.add(position,base),1n+kernel),Nat.add(position,Nat.add(base,1n+kernel)),
    Equal.cong(Nat,Nat,value => Nat.add(value,1n+kernel),Nat.div(padded,stride),Nat.add(position,base),
      Equal.trans(Nat,Nat.div(padded,stride),Nat.div(shifted,stride),Nat.add(position,base),
        Equal.cong(Nat,Nat,value => Nat.div(value,stride),padded,shifted,row_padding(phase,position,stride,padding)),DivisionOrder.quotient_shift(Nat.add(phase,padding),stride,position,positive))),Addition.associative(position,base,1n+kernel))

def point_exact(+geometry: Kernel.Geometry,+limits: Machine.Limits,+phase: U32,+position: U32,
  room: {Nat.is_le(row(geometry,phase,position),U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) ->
  {U32.to_nat(point(geometry,phase,position)) == row(geometry,phase,position) : Nat}:
  Bounds.roundtrip(row(geometry,phase,position),Machine.dimension(limits),Limits.dimension_fits(limits),room)
def point_room(+geometry: Kernel.Geometry,+limits: Machine.Limits,+phase: U32,+position: U32,
  +room: {Nat.is_le(row(geometry,phase,position),U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(point(geometry,phase,position)),U32.to_nat(Machine.dimension(limits))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(point(geometry,phase,position)),row(geometry,phase,position),point_exact(geometry,limits,phase,position,room)) : {Nat.is_le(_,U32.to_nat(Machine.dimension(limits))) == True{} : Bool}
  room

def shifted_high_natural(+call: Domain.Call,+phase: U32,+position: U32,
  +room: {Nat.is_le(row(Domain.geometry(call),phase,position),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool}) ->
  {Geometry.high(Domain.geometry(call),point(Domain.geometry(call),phase,position)) == Nat.add(U32.to_nat(position),Geometry.high(Domain.geometry(call),phase)) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  %Equal.sym(Nat,U32.to_nat(point(geometry,phase,position)),row(geometry,phase,position),point_exact(geometry,limits,phase,position,room)) :
    {Nat.add(Nat.div(Nat.add(_,Domain.padding(geometry)),Domain.stride(geometry)),1n+Domain.kernel(geometry)) == Nat.add(U32.to_nat(position),Geometry.high(geometry,phase)) : Nat}
  natural_shift(U32.to_nat(phase),U32.to_nat(position),Domain.padding(geometry),Domain.stride(geometry),Domain.kernel(geometry),Domain.stride_positive(call))

def shifted_high(+call: Domain.Call,+phase: U32,+size: U32,+position: U32,+envelope: Geometry.WordEnvelope,
  +phase_room: {Nat.is_le(U32.to_nat(phase),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
  +room: {Nat.is_le(row(Domain.geometry(call),phase,position),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
  +active: {U32.is_lt(Phase.residue(phase,Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.kernel_word(Domain.geometry(call))) == True{} : Bool}) ->
  {U32.to_nat((position + Geometry.axis_high(Domain.geometry(call),phase,size) : U32)) == Geometry.high(Domain.geometry(call),point(Domain.geometry(call),phase,position)) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  +high_word = Geometry.axis_high(geometry,phase,size)
  +high = Geometry.high(geometry,phase)
  +exact = Geometry.validated_high_exact(call,phase,size,phase_room,envelope,active)
  +shifted = shifted_high_natural(call,phase,position,room)
  +sum_room = {%shifted : {Nat.is_le(_,U32.to_nat(Geometry.maximum(envelope))) == True{} : Bool}
    Geometry.natural_high_room(call,point(geometry,phase,position),point_room(geometry,limits,phase,position,room),envelope) : {Nat.is_le(Nat.add(U32.to_nat(position),high),U32.to_nat(Geometry.maximum(envelope))) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat((position + high_word : U32)),Nat.add(U32.to_nat(position),high),Geometry.high(geometry,point(geometry,phase,position)),
    Equal.trans(Nat,U32.to_nat((position + high_word : U32)),Nat.add(U32.to_nat(position),U32.to_nat(high_word)),Nat.add(U32.to_nat(position),high),
      FullRange.add(position,high_word,Geometry.maximum(envelope),
        %Equal.sym(Nat,U32.to_nat(high_word),high,exact) : {Nat.is_le(Nat.add(U32.to_nat(position),_),U32.to_nat(Geometry.maximum(envelope))) == True{} : Bool} sum_room),
      Equal.cong(Nat,Nat,value => Nat.add(U32.to_nat(position),value),U32.to_nat(high_word),high,exact)),
    Equal.sym(Nat,Geometry.high(geometry,point(geometry,phase,position)),Nat.add(U32.to_nat(position),high),shifted))

def shifted_low(+call: Domain.Call,+phase: U32,+size: U32,+position: U32,+envelope: Geometry.WordEnvelope,
  +phase_room: {Nat.is_le(U32.to_nat(phase),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
  +room: {Nat.is_le(row(Domain.geometry(call),phase,position),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
  +active: {U32.is_lt(Phase.residue(phase,Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.kernel_word(Domain.geometry(call))) == True{} : Bool}) ->
  {U32.to_nat((position + Geometry.axis_low(Domain.geometry(call),phase,size) : U32)) == Nat.sub(Geometry.high(Domain.geometry(call),point(Domain.geometry(call),phase,position)),U32.to_nat(Phase.axis_taps(Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)))))) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  +context = Kernel.axis(phase,size,Domain.kernel_word(geometry),Domain.stride_word(geometry),Domain.padding_word(geometry))
  +taps = Phase.axis_taps(context)
  +low = (position + Geometry.axis_low(geometry,phase,size) : U32)
  +high = (position + Geometry.axis_high(geometry,phase,size) : U32)
  +wanted = Geometry.high(geometry,point(geometry,phase,position))
  +kn = Domain.kernel(geometry)
  +exact = shifted_high(call,phase,size,position,envelope,phase_room,room,active)
  +ordered = {%Equal.sym(Nat,U32.to_nat(high),wanted,exact) : {Nat.is_le(U32.to_nat(taps),_) == True{} : Bool}
    Order.transitive(U32.to_nat(taps),kn,wanted,Geometry.axis_taps_kernel_bound(call,phase,size),
      Order.transitive(kn,1n+kn,wanted,Bounds.strict_le(kn,1n+kn,Order.lt_succ(kn,kn,Order.reflexive(kn))),
        %Order.commutative(1n+kn,Geometry.origin(geometry,point(geometry,phase,position))) : {Nat.is_le(1n+kn,_) == True{} : Bool} Order.le_add(1n+kn,Geometry.origin(geometry,point(geometry,phase,position))))) : {Nat.is_le(U32.to_nat(taps),U32.to_nat(high)) == True{} : Bool}}
  +capacity = Geometry.maximum(envelope)
  +difference_room = Order.transitive(Nat.sub(U32.to_nat(high),U32.to_nat(taps)),U32.to_nat(high),U32.to_nat(capacity),Order.sub_le(U32.to_nat(high),U32.to_nat(taps)),Geometry.maximum_covers(envelope,high))
  +span = Arithmetic.add_associative(position,Geometry.axis_low(geometry,phase,size),taps)
  +restored = Equal.trans(U32,(high - taps : U32),((low + taps : U32) - taps : U32),low,
    Equal.cong(U32,U32,value => (value - taps : U32),high,(low + taps : U32),Equal.sym(U32,(low + taps : U32),high,span)),Subtraction.subtract_add(low,taps))
  Equal.trans(Nat,U32.to_nat(low),U32.to_nat((high - taps : U32)),Nat.sub(wanted,U32.to_nat(taps)),
    Equal.cong(U32,Nat,U32.to_nat,low,(high - taps : U32),Equal.sym(U32,(high - taps : U32),low,restored)),
    Equal.trans(Nat,U32.to_nat((high - taps : U32)),Nat.sub(U32.to_nat(high),U32.to_nat(taps)),Nat.sub(wanted,U32.to_nat(taps)),
      FullRange.subtract(high,taps,capacity,ordered,difference_room),Equal.cong(Nat,Nat,value => Nat.sub(value,U32.to_nat(taps)),U32.to_nat(high),wanted,exact)))

def last_encoded(+value: U32,+positive: {Nat.is_lt(0n,U32.to_nat(value)) == True{} : Bool}) ->
  {U32.to_nat((value - 1 : U32)) == Nat.sub(U32.to_nat(value),1n) : Nat}:
  FullRange.subtract(value,1,value,Order.positive_one(U32.to_nat(value),positive),Order.sub_le(U32.to_nat(value),1n))

def positions_inside(+axis: Kernel.Axis,+stride: U32,+first: U32,+last: U32,+used: {native_used(axis,stride,first,last) == True{} : Bool}) ->
  Shape.Both<{Nat.is_lt(U32.to_nat(low(indices(axis,stride,first,last))),U32.to_nat(Phase.axis_count(axis))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat((high(indices(axis,stride,first,last)) - 1 : U32)),U32.to_nat(Phase.axis_count(axis))) == True{} : Bool}>:
  Kernel.Axis{+start,+count,tap,+taps,source} = axis
  +low = low(indices(axis,stride,first,last))
  +high = high(indices(axis,stride,first,last))
  +inside = Bounds.as_nat_lt(low,high,Boolean.right(U32.is_lt(0,taps),U32.is_lt(low,high),used))
  +room = Bounds.as_nat_le(high,count,NativeExtrema.minimum_left(count,clipped(start,last,stride)))
  +positive = Order.positive_bound(U32.to_nat(low),U32.to_nat(high),inside)
  Shape.Both{Bounds.nat_trans(U32.to_nat(low),U32.to_nat(high),U32.to_nat(count),inside,room),
    %Equal.sym(Nat,U32.to_nat((high - 1 : U32)),Nat.sub(U32.to_nat(high),1n),last_encoded(high,positive)) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool}
      Bounds.nat_trans(Nat.sub(U32.to_nat(high),1n),U32.to_nat(high),U32.to_nat(count),Order.last_below(U32.to_nat(high),positive),room)}

def row_room(+call: Domain.Call,+phase: U32,+size: U32,+position: U32,
  +size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
  +phase_inside: {Nat.is_lt(U32.to_nat(phase),U32.to_nat(size)) == True{} : Bool},
  +position_inside: {Nat.is_lt(U32.to_nat(position),U32.to_nat(Phase.axis_count(Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)))))) == True{} : Bool}) ->
  {Nat.is_le(row(Domain.geometry(call),phase,position),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Bounds.strict_le(row(geometry,phase,position),U32.to_nat(Machine.dimension(limits)),
    Bounds.nat_trans(row(geometry,phase,position),U32.to_nat(size),U32.to_nat(Machine.dimension(limits)),
      Phase.position_inside(phase,size,k,s,p,U32.to_nat(position),Machine.coordinate(limits),Limits.coordinate_fits(limits),PhaseBounds.dimension_room(limits,U32.to_nat(size),size_room),PhaseBounds.stride_double_room(call),Domain.stride_positive(call),phase_inside,position_inside),size_room))

def pure_low_affine(+band: Bands.Band,+padding: Nat,+kernel: Nat,+taps: Nat) ->
  {Bands.sample_low(band,padding,kernel,taps) == Nat.sub(Nat.add(Bands.low(band),Nat.add(Nat.div(Nat.add(Bands.phase(band),padding),Bands.stride(band)),1n+kernel)),taps) : Nat}:
  Equal.cong(Nat,Nat,value => Nat.sub(value,taps),Nat.add(Nat.div(Nat.add(Bands.start_point(band),padding),Bands.stride(band)),1n+kernel),
    Nat.add(Bands.low(band),Nat.add(Nat.div(Nat.add(Bands.phase(band),padding),Bands.stride(band)),1n+kernel)),natural_shift(Bands.phase(band),Bands.low(band),padding,Bands.stride(band),kernel,Bands.positive(band)))
def pure_high_affine(+band: Bands.Band,+padding: Nat,+kernel: Nat) ->
  {Bands.sample_high(band,padding,kernel) == Nat.add(Nat.sub(Bands.high(band),1n),Nat.add(Nat.div(Nat.add(Bands.phase(band),padding),Bands.stride(band)),1n+kernel)) : Nat}:
  natural_shift(Bands.phase(band),Nat.sub(Bands.high(band),1n),padding,Bands.stride(band),kernel,Bands.positive(band))

def grouped_low(+axis: Kernel.Axis,+kernel: U32,+stride: U32,+first: U32,+last: U32) ->
  {Axes.band_sample_low(Axes.band_sample_from(axis,kernel,stride,first,last)) == (low(indices(axis,stride,first,last)) + Phase.axis_source(axis) + kernel : U32) : U32}:
  Kernel.Axis{start,count,tap,taps,source} = axis
  {==}
def grouped_high(+axis: Kernel.Axis,+kernel: U32,+stride: U32,+first: U32,+last: U32) ->
  {Axes.band_sample_high(Axes.band_sample_from(axis,kernel,stride,first,last)) == ((high(indices(axis,stride,first,last)) - 1 : U32) + Phase.axis_source(axis) + kernel + Phase.axis_taps(axis) : U32) : U32}:
  Kernel.Axis{start,count,tap,taps,source} = axis
  {==}

def sample(axis: Kernel.Axis,kernel: U32,stride: U32,first: U32,last: U32) -> Axes.BandSample:
  Axes.band_sample_from(axis,kernel,stride,first,last)

def actual_low_readback(+call: Domain.Call,+phase: U32,+size: U32,+first: U32,+last: U32,+envelope: Geometry.WordEnvelope,
  +phase_room: {Nat.is_le(U32.to_nat(phase),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
  +room: {Nat.is_le(row(Domain.geometry(call),phase,low(indices(Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.stride_word(Domain.geometry(call)),first,last))),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
  +active: {U32.is_lt(Phase.residue(phase,Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.kernel_word(Domain.geometry(call))) == True{} : Bool}) ->
  {U32.to_nat(Axes.band_sample_low(sample(Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),first,last))) ==
    Nat.sub(Nat.add(U32.to_nat(low(indices(Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.stride_word(Domain.geometry(call)),first,last))),Geometry.high(Domain.geometry(call),phase)),U32.to_nat(Phase.axis_taps(Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)))))) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +axis = Kernel.axis(phase,size,k,s,p)
  +position = low(indices(axis,s,first,last))
  +value = Axes.band_sample_low(sample(axis,k,s,first,last))
  +wanted = Nat.sub(Nat.add(U32.to_nat(position),Geometry.high(geometry,phase)),U32.to_nat(Phase.axis_taps(axis)))
  +grouped = Equal.trans(U32,value,(position + Phase.axis_source(axis) + k : U32),(position + Geometry.axis_low(geometry,phase,size) : U32),grouped_low(axis,k,s,first,last),Arithmetic.add_associative(position,Phase.axis_source(axis),k))
  Equal.trans(Nat,U32.to_nat(value),U32.to_nat((position + Geometry.axis_low(geometry,phase,size) : U32)),wanted,Equal.cong(U32,Nat,U32.to_nat,value,(position + Geometry.axis_low(geometry,phase,size) : U32),grouped),
    Equal.trans(Nat,U32.to_nat((position + Geometry.axis_low(geometry,phase,size) : U32)),Nat.sub(Geometry.high(geometry,point(geometry,phase,position)),U32.to_nat(Phase.axis_taps(axis))),wanted,
      shifted_low(call,phase,size,position,envelope,phase_room,room,active),
      Equal.cong(Nat,Nat,value => Nat.sub(value,U32.to_nat(Phase.axis_taps(axis))),Geometry.high(geometry,point(geometry,phase,position)),Nat.add(U32.to_nat(position),Geometry.high(geometry,phase)),shifted_high_natural(call,phase,position,room))))

def actual_high_readback(+call: Domain.Call,+phase: U32,+size: U32,+first: U32,+last: U32,+envelope: Geometry.WordEnvelope,
  +phase_room: {Nat.is_le(U32.to_nat(phase),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
  +room: {Nat.is_le(row(Domain.geometry(call),phase,(high(indices(Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.stride_word(Domain.geometry(call)),first,last)) - 1 : U32)),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
  +active: {U32.is_lt(Phase.residue(phase,Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.kernel_word(Domain.geometry(call))) == True{} : Bool}) ->
  {U32.to_nat(Axes.band_sample_high(sample(Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),first,last))) ==
    Nat.add(U32.to_nat((high(indices(Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.stride_word(Domain.geometry(call)),first,last)) - 1 : U32)),Geometry.high(Domain.geometry(call),phase)) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +axis = Kernel.axis(phase,size,k,s,p)
  +position = (high(indices(axis,s,first,last)) - 1 : U32)
  +value = Axes.band_sample_high(sample(axis,k,s,first,last))
  +source = Phase.axis_source(axis)
  +taps = Phase.axis_taps(axis)
  +grouped = Equal.trans(U32,value,(position + source + k + taps : U32),(position + Geometry.axis_high(geometry,phase,size) : U32),grouped_high(axis,k,s,first,last),
    Equal.trans(U32,(position + source + k + taps : U32),((position + (source + k : U32) : U32) + taps : U32),(position + Geometry.axis_high(geometry,phase,size) : U32),
      Equal.cong(U32,U32,value => (value + taps : U32),(position + source + k : U32),(position + (source + k : U32) : U32),Arithmetic.add_associative(position,source,k)),Arithmetic.add_associative(position,(source + k : U32),taps)))
  Equal.trans(Nat,U32.to_nat(value),U32.to_nat((position + Geometry.axis_high(geometry,phase,size) : U32)),Nat.add(U32.to_nat(position),Geometry.high(geometry,phase)),
    Equal.cong(U32,Nat,U32.to_nat,value,(position + Geometry.axis_high(geometry,phase,size) : U32),grouped),
    Equal.trans(Nat,U32.to_nat((position + Geometry.axis_high(geometry,phase,size) : U32)),Geometry.high(geometry,point(geometry,phase,position)),Nat.add(U32.to_nat(position),Geometry.high(geometry,phase)),
      shifted_high(call,phase,size,position,envelope,phase_room,room,active),shifted_high_natural(call,phase,position,room)))

# Every field is available at an enabled band_measure call.
type Selection is Data:
  Selection{call: Domain.Call,phase: U32,size: U32,first: U32,last: U32,envelope: Geometry.WordEnvelope,
    size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
    phase_inside: {Nat.is_lt(U32.to_nat(phase),U32.to_nat(size)) == True{} : Bool},
    first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
    last_room: {Nat.is_le(U32.to_nat(last),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
    used: {native_used(Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.stride_word(Domain.geometry(call)),first,last) == True{} : Bool}}
def selection_axis(selection: Selection) -> Kernel.Axis:
  Selection{+call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)))
def selection_stride(selection: Selection) -> U32:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Domain.stride_word(Domain.geometry(call))
def selection_first(selection: Selection) -> U32:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  first
def selection_last(selection: Selection) -> U32:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  last
def selection_indices(selection: Selection) -> Indices:
  +selection = selection
  indices(selection_axis(selection),selection_stride(selection),selection_first(selection),selection_last(selection))
def selection_model(selection: Selection) -> Bands.Band:
  Selection{+call,+phase,+size,+first,+last,envelope,sr,pi,fr,lr,used} = selection
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  model(selection_axis(selection),s,first,last,Domain.stride_positive(call))
def selection_kernel(selection: Selection) -> Nat:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Domain.kernel(Domain.geometry(call))
def selection_padding(selection: Selection) -> Nat:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Domain.padding(Domain.geometry(call))
def selection_taps(selection: Selection) -> Nat:
  U32.to_nat(Phase.axis_taps(selection_axis(selection)))
def selection_sample(selection: Selection) -> Axes.BandSample:
  Selection{+call,+phase,+size,+first,+last,+envelope,+sr,+pi,+fr,+lr,+used} = selection
  sample(selection_axis(selection),Domain.kernel_word(Domain.geometry(call)),selection_stride(selection),first,last)

def readback_from(+selection: Selection,
  positions: Shape.Both<{Nat.is_lt(U32.to_nat(low(selection_indices(selection))),U32.to_nat(Phase.axis_count(selection_axis(selection)))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat((high(selection_indices(selection)) - 1 : U32)),U32.to_nat(Phase.axis_count(selection_axis(selection)))) == True{} : Bool}>,
  encoded: Shape.Both<{U32.to_nat(low(selection_indices(selection))) == Bands.low(selection_model(selection)) : Nat},
    {U32.to_nat(high(selection_indices(selection))) == Bands.high(selection_model(selection)) : Nat}>) ->
  Shape.Both<{U32.to_nat(Axes.band_sample_low(selection_sample(selection))) == Bands.sample_low(selection_model(selection),selection_padding(selection),selection_kernel(selection),selection_taps(selection)) : Nat},
    {U32.to_nat(Axes.band_sample_high(selection_sample(selection))) == Bands.sample_high(selection_model(selection),selection_padding(selection),selection_kernel(selection)) : Nat}>:
  Selection{call,phase,size,first,last,envelope,size_room,phase_inside,first_room,last_room,used} = selection
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Shape.Both{+low_inside,+high_inside} = positions
  Shape.Both{+low_exact,+high_exact} = encoded
  +axis = selection_axis(selection)
  +lo = low(selection_indices(selection))
  +hi = high(selection_indices(selection))
  +last_position = (hi - 1 : U32)
  +band = selection_model(selection)
  +taps = U32.to_nat(Phase.axis_taps(axis))
  +base = Geometry.high(geometry,phase)
  +phase_room = Bounds.strict_le(U32.to_nat(phase),U32.to_nat(Machine.dimension(limits)),Bounds.nat_trans(U32.to_nat(phase),U32.to_nat(size),U32.to_nat(Machine.dimension(limits)),phase_inside,size_room))
  +tap_used = Boolean.left(U32.is_lt(0,Phase.axis_taps(axis)),U32.is_lt(lo,hi),used)
  +active = Geometry.active_when_taps(phase,size,k,s,p,tap_used)
  +low_row_room = row_room(call,phase,size,lo,size_room,phase_inside,low_inside)
  +high_row_room = row_room(call,phase,size,last_position,size_room,phase_inside,high_inside)
  +high_positive = Order.positive_bound(U32.to_nat(lo),U32.to_nat(hi),Bounds.as_nat_lt(lo,hi,Boolean.right(U32.is_lt(0,Phase.axis_taps(axis)),U32.is_lt(lo,hi),used)))
  +last_exact = Equal.trans(Nat,U32.to_nat(last_position),Nat.sub(U32.to_nat(hi),1n),Nat.sub(Bands.high(band),1n),last_encoded(hi,high_positive),Equal.cong(Nat,Nat,value => Nat.sub(value,1n),U32.to_nat(hi),Bands.high(band),high_exact))
  Shape.Both{
    Equal.trans(Nat,U32.to_nat(Axes.band_sample_low(selection_sample(selection))),Nat.sub(Nat.add(U32.to_nat(lo),base),taps),Bands.sample_low(band,U32.to_nat(p),U32.to_nat(k),taps),
      actual_low_readback(call,phase,size,first,last,envelope,phase_room,low_row_room,active),
      Equal.trans(Nat,Nat.sub(Nat.add(U32.to_nat(lo),base),taps),Nat.sub(Nat.add(Bands.low(band),base),taps),Bands.sample_low(band,U32.to_nat(p),U32.to_nat(k),taps),
        Equal.cong(Nat,Nat,value => Nat.sub(Nat.add(value,base),taps),U32.to_nat(lo),Bands.low(band),low_exact),
        Equal.sym(Nat,Bands.sample_low(band,U32.to_nat(p),U32.to_nat(k),taps),Nat.sub(Nat.add(Bands.low(band),base),taps),pure_low_affine(band,U32.to_nat(p),U32.to_nat(k),taps)))),
    Equal.trans(Nat,U32.to_nat(Axes.band_sample_high(selection_sample(selection))),Nat.add(U32.to_nat(last_position),base),Bands.sample_high(band,U32.to_nat(p),U32.to_nat(k)),
      actual_high_readback(call,phase,size,first,last,envelope,phase_room,high_row_room,active),
      Equal.trans(Nat,Nat.add(U32.to_nat(last_position),base),Nat.add(Nat.sub(Bands.high(band),1n),base),Bands.sample_high(band,U32.to_nat(p),U32.to_nat(k)),
        Equal.cong(Nat,Nat,value => Nat.add(value,base),U32.to_nat(last_position),Nat.sub(Bands.high(band),1n),last_exact),
        Equal.sym(Nat,Bands.sample_high(band,U32.to_nat(p),U32.to_nat(k)),Nat.add(Nat.sub(Bands.high(band),1n),base),pure_high_affine(band,U32.to_nat(p),U32.to_nat(k)))))}

def selection_indices_exact(+selection: Selection) ->
  Shape.Both<{U32.to_nat(low(selection_indices(selection))) == Bands.low(selection_model(selection)) : Nat},
    {U32.to_nat(high(selection_indices(selection))) == Bands.high(selection_model(selection)) : Nat}>:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  indices_matches(selection_axis(selection),s,first,last,limits,fr,lr,Domain.stride_room(call),Domain.stride_positive(call))

def selection_bounds_from(+selection: Selection,+position: U32,
  +lower: {Nat.is_le(U32.to_nat(low(selection_indices(selection))),U32.to_nat(position)) == True{} : Bool},
  +upper: {Nat.is_lt(U32.to_nat(position),U32.to_nat(high(selection_indices(selection)))) == True{} : Bool},
  encoded: Shape.Both<{U32.to_nat(low(selection_indices(selection))) == Bands.low(selection_model(selection)) : Nat},
    {U32.to_nat(high(selection_indices(selection))) == Bands.high(selection_model(selection)) : Nat}>) ->
  Shape.Both<{Nat.is_le(Bands.low(selection_model(selection)),U32.to_nat(position)) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(position),Bands.high(selection_model(selection))) == True{} : Bool}>:
  Shape.Both{+lo,+hi} = encoded
  Shape.Both{%lo : {Nat.is_le(_,U32.to_nat(position)) == True{} : Bool} lower,
    %hi : {Nat.is_lt(U32.to_nat(position),_) == True{} : Bool} upper}
def selection_bounds(+selection: Selection,+position: U32,
  +lower: {Nat.is_le(U32.to_nat(low(selection_indices(selection))),U32.to_nat(position)) == True{} : Bool},
  +upper: {Nat.is_lt(U32.to_nat(position),U32.to_nat(high(selection_indices(selection)))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(Bands.low(selection_model(selection)),U32.to_nat(position)) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(position),Bands.high(selection_model(selection))) == True{} : Bool}>:
  selection_bounds_from(selection,position,lower,upper,selection_indices_exact(selection))

def readback(+selection: Selection) ->
  Shape.Both<{U32.to_nat(Axes.band_sample_low(selection_sample(selection))) == Bands.sample_low(selection_model(selection),selection_padding(selection),selection_kernel(selection),selection_taps(selection)) : Nat},
    {U32.to_nat(Axes.band_sample_high(selection_sample(selection))) == Bands.sample_high(selection_model(selection),selection_padding(selection),selection_kernel(selection)) : Nat}>:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  readback_from(selection,positions_inside(selection_axis(selection),s,first,last,used),selection_indices_exact(selection))

def selection_call(selection: Selection) -> Domain.Call:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  call
def selection_geometry(selection: Selection) -> Kernel.Geometry:
  Domain.geometry(selection_call(selection))
def selection_nonempty(+selection: Selection) -> {Nat.is_lt(Bands.low(selection_model(selection)),Bands.high(selection_model(selection))) == True{} : Bool}:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  nonempty(selection_axis(selection),s,first,last,limits,fr,lr,Domain.stride_room(call),Domain.stride_positive(call),used)

def sample_bounds_from(+selection: Selection,
  facts: Shape.Both<{U32.to_nat(Axes.band_sample_low(selection_sample(selection))) == Bands.sample_low(selection_model(selection),selection_padding(selection),selection_kernel(selection),selection_taps(selection)) : Nat},
    {U32.to_nat(Axes.band_sample_high(selection_sample(selection))) == Bands.sample_high(selection_model(selection),selection_padding(selection),selection_kernel(selection)) : Nat}>) ->
  Shape.Both<{Nat.is_le(Nat.sub(Geometry.high(selection_geometry(selection),selection_first(selection)),Geometry.halo(selection_geometry(selection))),U32.to_nat(Axes.band_sample_low(selection_sample(selection)))) == True{} : Bool},
    {Nat.is_le(U32.to_nat(Axes.band_sample_high(selection_sample(selection))),Geometry.end_high(selection_geometry(selection),selection_last(selection))) == True{} : Bool}>:
  Selection{call,phase,size,first,last,envelope,size_room,phase_inside,first_room,last_room,used} = selection
  Domain.Call{geometry,batch,limits,validated} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Shape.Both{+low_exact,+high_exact} = facts
  +band = selection_model(selection)
  +halo = Geometry.halo(geometry)
  +taps = selection_taps(selection)
  +taps_bound = Geometry.axis_taps_bounded(call,phase,size)
  +halo_bound = Ceiling.bounded(U32.to_nat(k),U32.to_nat(s),Domain.stride_positive(call))
  Shape.Both{
    %Equal.sym(Nat,U32.to_nat(Axes.band_sample_low(selection_sample(selection))),Bands.sample_low(band,U32.to_nat(p),U32.to_nat(k),taps),low_exact) :
      {Nat.is_le(Nat.sub(Geometry.high(geometry,first),halo),_) == True{} : Bool}
      Bands.sample_lower(band,U32.to_nat(p),U32.to_nat(k),taps,halo,taps_bound,halo_bound),
    %Equal.sym(Nat,U32.to_nat(Axes.band_sample_high(selection_sample(selection))),Bands.sample_high(band,U32.to_nat(p),U32.to_nat(k)),high_exact) :
      {Nat.is_le(_,Geometry.end_high(geometry,last)) == True{} : Bool}
      Bands.sample_upper(band,U32.to_nat(p),U32.to_nat(k),selection_nonempty(selection))}

def sample_bounds(+selection: Selection) ->
  Shape.Both<{Nat.is_le(Nat.sub(Geometry.high(selection_geometry(selection),selection_first(selection)),Geometry.halo(selection_geometry(selection))),U32.to_nat(Axes.band_sample_low(selection_sample(selection)))) == True{} : Bool},
    {Nat.is_le(U32.to_nat(Axes.band_sample_high(selection_sample(selection))),Geometry.end_high(selection_geometry(selection),selection_last(selection))) == True{} : Bool}>:
  sample_bounds_from(selection,readback(selection))

def sample_ordered_from(+selection: Selection,
  facts: Shape.Both<{U32.to_nat(Axes.band_sample_low(selection_sample(selection))) == Bands.sample_low(selection_model(selection),selection_padding(selection),selection_kernel(selection),selection_taps(selection)) : Nat},
    {U32.to_nat(Axes.band_sample_high(selection_sample(selection))) == Bands.sample_high(selection_model(selection),selection_padding(selection),selection_kernel(selection)) : Nat}>) ->
  {Nat.is_le(U32.to_nat(Axes.band_sample_low(selection_sample(selection))),U32.to_nat(Axes.band_sample_high(selection_sample(selection)))) == True{} : Bool}:
  Shape.Both{+lo,+hi} = facts
  +band = selection_model(selection)
  +padding = selection_padding(selection)
  +kernel = selection_kernel(selection)
  +taps = selection_taps(selection)
  %Equal.sym(Nat,U32.to_nat(Axes.band_sample_low(selection_sample(selection))),Bands.sample_low(band,padding,kernel,taps),lo) :
    {Nat.is_le(_,U32.to_nat(Axes.band_sample_high(selection_sample(selection)))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Axes.band_sample_high(selection_sample(selection))),Bands.sample_high(band,padding,kernel),hi) :
    {Nat.is_le(Bands.sample_low(band,padding,kernel,taps),_) == True{} : Bool}
  Bands.sample_ordered(band,padding,kernel,taps,selection_nonempty(selection))
def sample_ordered(+selection: Selection) ->
  {Nat.is_le(U32.to_nat(Axes.band_sample_low(selection_sample(selection))),U32.to_nat(Axes.band_sample_high(selection_sample(selection)))) == True{} : Bool}:
  sample_ordered_from(selection,readback(selection))

# Every phase row in the requested band activates its clipped native job range.
def member_indices_from(+axis: Kernel.Axis,+stride: U32,+first: U32,+last: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+index: Nat,
  +in_phase: {Nat.is_lt(index,U32.to_nat(Phase.axis_count(axis))) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(first),Bands.position(model(axis,stride,first,last,positive),index)) == True{} : Bool},
  +upper: {Nat.is_lt(Bands.position(model(axis,stride,first,last,positive),index),U32.to_nat(last)) == True{} : Bool},
  encoded: Shape.Both<{U32.to_nat(low(indices(axis,stride,first,last))) == Bands.low(model(axis,stride,first,last,positive)) : Nat},
    {U32.to_nat(high(indices(axis,stride,first,last))) == Bands.high(model(axis,stride,first,last,positive)) : Nat}>) ->
  Shape.Both<{Nat.is_le(U32.to_nat(low(indices(axis,stride,first,last))),index) == True{} : Bool},
    {Nat.is_lt(index,U32.to_nat(high(indices(axis,stride,first,last)))) == True{} : Bool}>:
  Kernel.Axis{start,count,tap,taps,source} = axis
  Shape.Both{+lo,+hi} = encoded
  +band = model(axis,stride,first,last,positive)
  Shape.Both{
    %Equal.sym(Nat,U32.to_nat(low(indices(axis,stride,first,last))),Bands.low(band),lo) : {Nat.is_le(_,index) == True{} : Bool} Bands.member_lower(band,index,lower),
    %Equal.sym(Nat,U32.to_nat(high(indices(axis,stride,first,last))),Bands.high(band),hi) : {Nat.is_lt(index,_) == True{} : Bool} Bands.member_upper(band,index,in_phase,upper)}
def member_indices(+axis: Kernel.Axis,+stride: U32,+first: U32,+last: U32,+limits: Machine.Limits,
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +last_room: {Nat.is_le(U32.to_nat(last),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +stride_room: {Nat.is_le(U32.to_nat(stride),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+index: Nat,
  +in_phase: {Nat.is_lt(index,U32.to_nat(Phase.axis_count(axis))) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(first),Bands.position(model(axis,stride,first,last,positive),index)) == True{} : Bool},
  +upper: {Nat.is_lt(Bands.position(model(axis,stride,first,last,positive),index),U32.to_nat(last)) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat(low(indices(axis,stride,first,last))),index) == True{} : Bool},
    {Nat.is_lt(index,U32.to_nat(high(indices(axis,stride,first,last)))) == True{} : Bool}>:
  member_indices_from(axis,stride,first,last,positive,index,in_phase,lower,upper,indices_matches(axis,stride,first,last,limits,first_room,last_room,stride_room,positive))
def member_used_from(+axis: Kernel.Axis,+stride: U32,+first: U32,+last: U32,+index: Nat,
  +taps_positive: {U32.is_lt(0,Phase.axis_taps(axis)) == True{} : Bool},
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(low(indices(axis,stride,first,last))),index) == True{} : Bool},
    {Nat.is_lt(index,U32.to_nat(high(indices(axis,stride,first,last)))) == True{} : Bool}>) ->
  {native_used(axis,stride,first,last) == True{} : Bool}:
  Kernel.Axis{start,count,tap,+taps,source} = axis
  Shape.Both{+lo,+hi} = bounds
  +left = low(indices(axis,stride,first,last))
  +right = high(indices(axis,stride,first,last))
  Boolean.and_true(U32.is_lt(0,taps),U32.is_lt(left,right),taps_positive,
    %Equal.sym(Bool,U32.is_lt(left,right),Nat.is_lt(U32.to_nat(left),U32.to_nat(right)),Comparison.lt(left,right)) : {_ == True{} : Bool}
    Order.le_lt_trans(U32.to_nat(left),index,U32.to_nat(right),lo,hi))
def member_used(+axis: Kernel.Axis,+stride: U32,+first: U32,+last: U32,+limits: Machine.Limits,
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +last_room: {Nat.is_le(U32.to_nat(last),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +stride_room: {Nat.is_le(U32.to_nat(stride),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+index: Nat,
  +in_phase: {Nat.is_lt(index,U32.to_nat(Phase.axis_count(axis))) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(first),Bands.position(model(axis,stride,first,last,positive),index)) == True{} : Bool},
  +upper: {Nat.is_lt(Bands.position(model(axis,stride,first,last,positive),index),U32.to_nat(last)) == True{} : Bool},
  +taps_positive: {U32.is_lt(0,Phase.axis_taps(axis)) == True{} : Bool}) -> {native_used(axis,stride,first,last) == True{} : Bool}:
  member_used_from(axis,stride,first,last,index,taps_positive,member_indices(axis,stride,first,last,limits,first_room,last_room,stride_room,positive,index,in_phase,lower,upper))

def selection_phase(selection: Selection) -> U32:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  phase
def selection_size(selection: Selection) -> U32:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  size
def phase_high(+selection: Selection) -> Nat:
  Geometry.high(selection_geometry(selection),selection_phase(selection))
def phase_base(+selection: Selection) -> Nat:
  Nat.sub(phase_high(selection),selection_taps(selection))
def phase_taps_room(+selection: Selection) -> {Nat.is_le(selection_taps(selection),phase_high(selection)) == True{} : Bool}:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Domain.Call{geometry,batch,limits,facts} = call
  +k = Domain.kernel(geometry)
  +high = phase_high(selection)
  Order.transitive(selection_taps(selection),k,high,Geometry.axis_taps_kernel_bound(call,phase,size),
    Order.transitive(k,1n+k,high,Bounds.strict_le(k,1n+k,Order.lt_succ(k,k,Order.reflexive(k))),
      %Order.commutative(1n+k,Geometry.origin(geometry,phase)) : {Nat.is_le(1n+k,_) == True{} : Bool} Order.le_add(1n+k,Geometry.origin(geometry,phase))))
def affine_readback_from(+selection: Selection,
  encoded: Shape.Both<{U32.to_nat(Axes.band_sample_low(selection_sample(selection))) == Bands.sample_low(selection_model(selection),selection_padding(selection),selection_kernel(selection),selection_taps(selection)) : Nat},
    {U32.to_nat(Axes.band_sample_high(selection_sample(selection))) == Bands.sample_high(selection_model(selection),selection_padding(selection),selection_kernel(selection)) : Nat}>) ->
  Shape.Both<{U32.to_nat(Axes.band_sample_low(selection_sample(selection))) == Nat.add(Bands.low(selection_model(selection)),phase_base(selection)) : Nat},
    {U32.to_nat(Axes.band_sample_high(selection_sample(selection))) == Nat.add(Nat.sub(Bands.high(selection_model(selection)),1n),phase_high(selection)) : Nat}>:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Shape.Both{+lo,+hi} = encoded
  +band = selection_model(selection)
  +padding = selection_padding(selection)
  +kernel = selection_kernel(selection)
  +taps = selection_taps(selection)
  Shape.Both{
    Equal.trans(Nat,U32.to_nat(Axes.band_sample_low(selection_sample(selection))),Bands.sample_low(band,padding,kernel,taps),Nat.add(Bands.low(band),phase_base(selection)),lo,
      Equal.trans(Nat,Bands.sample_low(band,padding,kernel,taps),Nat.sub(Nat.add(Bands.low(band),phase_high(selection)),taps),Nat.add(Bands.low(band),phase_base(selection)),pure_low_affine(band,padding,kernel,taps),Order.subtract_from_sum(Bands.low(band),phase_high(selection),taps,phase_taps_room(selection)))),
    Equal.trans(Nat,U32.to_nat(Axes.band_sample_high(selection_sample(selection))),Bands.sample_high(band,padding,kernel),Nat.add(Nat.sub(Bands.high(band),1n),phase_high(selection)),hi,pure_high_affine(band,padding,kernel))}
def affine_readback(+selection: Selection) ->
  Shape.Both<{U32.to_nat(Axes.band_sample_low(selection_sample(selection))) == Nat.add(Bands.low(selection_model(selection)),phase_base(selection)) : Nat},
    {U32.to_nat(Axes.band_sample_high(selection_sample(selection))) == Nat.add(Nat.sub(Bands.high(selection_model(selection)),1n),phase_high(selection)) : Nat}>:
  affine_readback_from(selection,readback(selection))

# A phase position and tap use the affine halo coordinate before subtracting k.
def tapped_coordinate(+selection: Selection,+position: U32,+tap: U32) -> Nat:
  Nat.add(Nat.add(U32.to_nat(position),phase_base(selection)),U32.to_nat(tap))
def tapped_lower(+selection: Selection,+position: U32,+tap: U32,
  +lower: {Nat.is_le(Bands.low(selection_model(selection)),U32.to_nat(position)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(Bands.low(selection_model(selection)),phase_base(selection)),tapped_coordinate(selection,position,tap)) == True{} : Bool}:
  +base = phase_base(selection)
  +coordinate = Nat.add(U32.to_nat(position),base)
  Order.transitive(Nat.add(Bands.low(selection_model(selection)),base),coordinate,tapped_coordinate(selection,position,tap),Order.add_le(Bands.low(selection_model(selection)),U32.to_nat(position),base,lower),Order.le_add(coordinate,U32.to_nat(tap)))
def tapped_upper(+selection: Selection,+position: U32,+tap: U32,
  +upper: {Nat.is_lt(U32.to_nat(position),Bands.high(selection_model(selection))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),selection_taps(selection)) == True{} : Bool}) ->
  {Nat.is_lt(tapped_coordinate(selection,position,tap),Nat.add(Nat.sub(Bands.high(selection_model(selection)),1n),phase_high(selection))) == True{} : Bool}:
  +last = Nat.sub(Bands.high(selection_model(selection)),1n)
  +base = phase_base(selection)
  +taps = selection_taps(selection)
  +restored = Equal.trans(Nat,Nat.add(base,taps),Nat.add(taps,base),phase_high(selection),Order.commutative(base,taps),Order.restore_shift(taps,phase_high(selection),phase_taps_room(selection)))
  %restored : {Nat.is_lt(tapped_coordinate(selection,position,tap),Nat.add(last,_)) == True{} : Bool}
  %Addition.associative(last,base,taps) : {Nat.is_lt(tapped_coordinate(selection,position,tap),_) == True{} : Bool}
  Order.le_lt_trans(tapped_coordinate(selection,position,tap),Nat.add(Nat.add(last,base),U32.to_nat(tap)),Nat.add(Nat.add(last,base),taps),Order.add_le(Nat.add(U32.to_nat(position),base),Nat.add(last,base),U32.to_nat(tap),Order.add_le(U32.to_nat(position),last,base,Order.before_last(U32.to_nat(position),Bands.high(selection_model(selection)),upper))),Order.offset_lt(Nat.add(last,base),U32.to_nat(tap),taps,tap_inside))
def tapped_bounds_from(+selection: Selection,+position: U32,+tap: U32,
  +lower: {Nat.is_le(Bands.low(selection_model(selection)),U32.to_nat(position)) == True{} : Bool},
  +upper: {Nat.is_lt(U32.to_nat(position),Bands.high(selection_model(selection))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),selection_taps(selection)) == True{} : Bool},
  encoded: Shape.Both<{U32.to_nat(Axes.band_sample_low(selection_sample(selection))) == Nat.add(Bands.low(selection_model(selection)),phase_base(selection)) : Nat},
    {U32.to_nat(Axes.band_sample_high(selection_sample(selection))) == Nat.add(Nat.sub(Bands.high(selection_model(selection)),1n),phase_high(selection)) : Nat}>) ->
  Shape.Both<{Nat.is_le(U32.to_nat(Axes.band_sample_low(selection_sample(selection))),tapped_coordinate(selection,position,tap)) == True{} : Bool},
    {Nat.is_lt(tapped_coordinate(selection,position,tap),U32.to_nat(Axes.band_sample_high(selection_sample(selection)))) == True{} : Bool}>:
  Shape.Both{+lo,+hi} = encoded
  Shape.Both{
    %Equal.sym(Nat,U32.to_nat(Axes.band_sample_low(selection_sample(selection))),Nat.add(Bands.low(selection_model(selection)),phase_base(selection)),lo) : {Nat.is_le(_,tapped_coordinate(selection,position,tap)) == True{} : Bool} tapped_lower(selection,position,tap,lower),
    %Equal.sym(Nat,U32.to_nat(Axes.band_sample_high(selection_sample(selection))),Nat.add(Nat.sub(Bands.high(selection_model(selection)),1n),phase_high(selection)),hi) : {Nat.is_lt(tapped_coordinate(selection,position,tap),_) == True{} : Bool} tapped_upper(selection,position,tap,upper,tap_inside)}
def tapped_bounds(+selection: Selection,+position: U32,+tap: U32,
  +lower: {Nat.is_le(Bands.low(selection_model(selection)),U32.to_nat(position)) == True{} : Bool},
  +upper: {Nat.is_lt(U32.to_nat(position),Bands.high(selection_model(selection))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),selection_taps(selection)) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat(Axes.band_sample_low(selection_sample(selection))),tapped_coordinate(selection,position,tap)) == True{} : Bool},
    {Nat.is_lt(tapped_coordinate(selection,position,tap),U32.to_nat(Axes.band_sample_high(selection_sample(selection)))) == True{} : Bool}>:
  tapped_bounds_from(selection,position,tap,lower,upper,tap_inside,affine_readback(selection))

def selection_envelope(selection: Selection) -> Geometry.WordEnvelope:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  envelope
def selection_limits(selection: Selection) -> Machine.Limits:
  Domain.limits(selection_call(selection))
def selection_phase_room(+selection: Selection) -> {Nat.is_le(U32.to_nat(selection_phase(selection)),U32.to_nat(Machine.dimension(selection_limits(selection)))) == True{} : Bool}:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Order.transitive(U32.to_nat(phase),U32.to_nat(size),U32.to_nat(Machine.dimension(Domain.limits(call))),Bounds.strict_le(U32.to_nat(phase),U32.to_nat(size),pi),sr)
def selection_active(+selection: Selection) ->
  {U32.is_lt(Phase.residue(selection_phase(selection),Domain.stride_word(selection_geometry(selection)),Domain.padding_word(selection_geometry(selection))),Domain.kernel_word(selection_geometry(selection))) == True{} : Bool}:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Domain.Call{geometry,batch,limits,facts} = call
  +axis = selection_axis(selection)
  Geometry.active_when_taps(phase,size,Domain.kernel_word(geometry),Domain.stride_word(geometry),Domain.padding_word(geometry),Boolean.left(U32.is_lt(0,Phase.axis_taps(axis)),U32.is_lt(low(indices(axis,Domain.stride_word(geometry),first,last)),high(indices(axis,Domain.stride_word(geometry),first,last))),used))
def position_base_exact(+selection: Selection,+position: U32,
  +in_phase: {Nat.is_lt(U32.to_nat(position),U32.to_nat(Phase.axis_count(selection_axis(selection)))) == True{} : Bool}) ->
  {U32.to_nat((position + Geometry.axis_low(selection_geometry(selection),selection_phase(selection),selection_size(selection)) : U32)) == Nat.add(U32.to_nat(position),phase_base(selection)) : Nat}:
  Selection{call,phase,size,first,last,envelope,sr,pi,fr,lr,used} = selection
  Domain.Call{geometry,batch,limits,facts} = call
  +row_room = row_room(call,phase,size,position,sr,pi,in_phase)
  +shifted = shifted_high_natural(call,phase,position,row_room)
  +taps = selection_taps(selection)
  +base = (position + Geometry.axis_low(geometry,phase,size) : U32)
  Equal.trans(Nat,U32.to_nat(base),Nat.sub(Geometry.high(geometry,point(geometry,phase,position)),taps),Nat.add(U32.to_nat(position),phase_base(selection)),shifted_low(call,phase,size,position,envelope,selection_phase_room(selection),row_room,selection_active(selection)),
    Equal.trans(Nat,Nat.sub(Geometry.high(geometry,point(geometry,phase,position)),taps),Nat.sub(Nat.add(U32.to_nat(position),phase_high(selection)),taps),Nat.add(U32.to_nat(position),phase_base(selection)),Equal.cong(Nat,Nat,value => Nat.sub(value,taps),Geometry.high(geometry,point(geometry,phase,position)),Nat.add(U32.to_nat(position),phase_high(selection)),shifted),Order.subtract_from_sum(U32.to_nat(position),phase_high(selection),taps,phase_taps_room(selection))))
def native_tapped(+selection: Selection,+position: U32,+tap: U32) -> U32:
  (position + Geometry.axis_low(selection_geometry(selection),selection_phase(selection),selection_size(selection)) + tap : U32)
def tapped_exact_finish(+selection: Selection,+position: U32,+tap: U32,+base: U32,
  +base_exact: {U32.to_nat(base) == Nat.add(U32.to_nat(position),phase_base(selection)) : Nat},
  +upper: {Nat.is_lt(tapped_coordinate(selection,position,tap),Nat.add(Nat.sub(Bands.high(selection_model(selection)),1n),phase_high(selection))) == True{} : Bool},
  encoded: Shape.Both<{U32.to_nat(Axes.band_sample_low(selection_sample(selection))) == Nat.add(Bands.low(selection_model(selection)),phase_base(selection)) : Nat},
    {U32.to_nat(Axes.band_sample_high(selection_sample(selection))) == Nat.add(Nat.sub(Bands.high(selection_model(selection)),1n),phase_high(selection)) : Nat}>) ->
  {U32.to_nat((base + tap : U32)) == tapped_coordinate(selection,position,tap) : Nat}:
  Shape.Both{lo,+hi} = encoded
  +maximum = Geometry.maximum(selection_envelope(selection))
  +coordinate = tapped_coordinate(selection,position,tap)
  +sample_high = Axes.band_sample_high(selection_sample(selection))
  +upper_high = {%hi : {Nat.is_le(_,U32.to_nat(maximum)) == True{} : Bool} Geometry.maximum_covers(selection_envelope(selection),sample_high) : {Nat.is_le(Nat.add(Nat.sub(Bands.high(selection_model(selection)),1n),phase_high(selection)),U32.to_nat(maximum)) == True{} : Bool}}
  +room = {%Equal.sym(Nat,U32.to_nat(base),Nat.add(U32.to_nat(position),phase_base(selection)),base_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(tap)),U32.to_nat(maximum)) == True{} : Bool}
    Bounds.strict_le(coordinate,U32.to_nat(maximum),Bounds.nat_trans(coordinate,Nat.add(Nat.sub(Bands.high(selection_model(selection)),1n),phase_high(selection)),U32.to_nat(maximum),upper,upper_high)) : {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(tap)),U32.to_nat(maximum)) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat((base + tap : U32)),Nat.add(U32.to_nat(base),U32.to_nat(tap)),coordinate,FullRange.add(base,tap,maximum,room),Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(tap)),U32.to_nat(base),Nat.add(U32.to_nat(position),phase_base(selection)),base_exact))
def native_tapped_exact(+selection: Selection,+position: U32,+tap: U32,
  +in_phase: {Nat.is_lt(U32.to_nat(position),U32.to_nat(Phase.axis_count(selection_axis(selection)))) == True{} : Bool},
  +upper: {Nat.is_lt(U32.to_nat(position),Bands.high(selection_model(selection))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),selection_taps(selection)) == True{} : Bool}) ->
  {U32.to_nat(native_tapped(selection,position,tap)) == tapped_coordinate(selection,position,tap) : Nat}:
  +base = (position + Geometry.axis_low(selection_geometry(selection),selection_phase(selection),selection_size(selection)) : U32)
  +exact = position_base_exact(selection,position,in_phase)
  +upper_bound = tapped_upper(selection,position,tap,upper,tap_inside)
  +readback = affine_readback(selection)
  tapped_exact_finish(selection,position,tap,base,exact,upper_bound,readback)
