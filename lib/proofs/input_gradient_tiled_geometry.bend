# The chunk halo uses the same native ceiling tap count as fits.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_phase_taps.bend as Taps
import ./input_gradient_phase_geometry.bend as Geometry
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./input_gradient_phase_domain.bend as Domain
import ./nat_ceiling.bend as Ceiling
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division_order.bend as Division
import ./u32_division.bend as NativeDivision
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition
import ./machine_limit_bounds.bend as Limits
import ./word_subtraction.bend as Subtraction
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./encoded_coordinates.bend as Encoded
import ./word_encoding.bend as WordEncoding

# A residue can only remove taps from the maximum ceil(kernel / stride).
def tap_ceiling_case(below: Bool,+kernel: Nat,+residue: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},
  compared: {Nat.is_lt(residue,kernel) == below : Bool}) ->
  {Nat.is_le(Bool.pick(Nat,below,1n+Nat.div(Nat.sub(Nat.sub(kernel,1n),residue),stride),0n),Ceiling.ceiling(kernel,stride)) == True{} : Bool}:
  match below:
    case False{}: Order.zero_le(Ceiling.ceiling(kernel,stride))
    case True{}:
      +last = Nat.sub(kernel,1n)
      +kp = Order.positive_bound(residue,kernel,compared)
      %Order.positive_has_last(kernel,kp) :
        {Nat.is_le(1n+Nat.div(Nat.sub(last,residue),stride),Ceiling.ceiling(_,stride)) == True{} : Bool}
      %Equal.sym(Nat,Ceiling.ceiling(1n+last,stride),1n+Nat.div(last,stride),Ceiling.successor_formula(last,stride,positive)) :
        {Nat.is_le(1n+Nat.div(Nat.sub(last,residue),stride),_) == True{} : Bool}
      Division.quotient_le(Nat.sub(last,residue),last,stride,positive,Order.sub_le(last,residue))

def tap_ceiling(+kernel: Nat,+residue: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) ->
  {Nat.is_le(Taps.count(kernel,residue,stride),Ceiling.ceiling(kernel,stride)) == True{} : Bool}:
  tap_ceiling_case(Nat.is_lt(residue,kernel),kernel,residue,stride,positive,{==})

def tap_numerator(+kernel: U32,+stride: U32,+limits: Machine.Limits,+kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +stride_room: {Nat.is_le(U32.to_nat(stride),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {U32.to_nat((kernel + stride - 1 : U32)) == Nat.add(U32.to_nat(kernel),Nat.sub(U32.to_nat(stride),1n)) : Nat}:
  +kn = U32.to_nat(kernel)
  +sn = U32.to_nat(stride)
  +capacity = Machine.coordinate(limits)
  +sum = Nat.add(kn,sn)
  +room = PhaseBounds.padded_room(limits,kn,sn,kernel_room,stride_room)
  +exact = Arithmetic.add(kernel,stride,capacity,Limits.coordinate_fits(limits),room)
  +sum_positive = Bounds.nat_trans(0n,sn,sum,positive,
    %Order.commutative(sn,kn) : {Nat.is_le(sn,_) == True{} : Bool} Order.le_add(sn,kn))
  Equal.trans(Nat,U32.to_nat((kernel + stride - 1 : U32)),Nat.sub(sum,1n),Nat.add(kn,Nat.sub(sn,1n)),
    %exact : {U32.to_nat((kernel + stride - 1 : U32)) == Nat.sub(_,1n) : Nat}
      Arithmetic.subtract((kernel + stride : U32),1,capacity,Limits.coordinate_fits(limits),
        %Equal.sym(Nat,U32.to_nat((kernel + stride : U32)),sum,exact) : {Nat.is_le(1n,_) == True{} : Bool} Order.positive_one(sum,sum_positive),
        %Equal.sym(Nat,U32.to_nat((kernel + stride : U32)),sum,exact) : {Nat.is_le(Nat.sub(_,1n),U32.to_nat(capacity)) == True{} : Bool}
          Order.transitive(Nat.sub(sum,1n),sum,U32.to_nat(capacity),Order.sub_le(sum,1n),room)),
    Order.subtract_from_sum(kn,sn,1n,Order.positive_one(sn,positive)))

def taps_exact(+kernel: U32,+stride: U32,+limits: Machine.Limits,+kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +stride_room: {Nat.is_le(U32.to_nat(stride),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {U32.to_nat(Kernel.taps(kernel,stride)) == Ceiling.ceiling(U32.to_nat(kernel),U32.to_nat(stride)) : Nat}:
  +numerator = (kernel + stride - 1 : U32)
  +natural = Nat.add(U32.to_nat(kernel),Nat.sub(U32.to_nat(stride),1n))
  Equal.trans(Nat,U32.to_nat(Kernel.taps(kernel,stride)),Nat.div(U32.to_nat(numerator),U32.to_nat(stride)),Ceiling.ceiling(U32.to_nat(kernel),U32.to_nat(stride)),
    NativeDivision.divide(numerator,stride,Machine.coordinate(limits),Limits.coordinate_fits(limits),PhaseBounds.double_room(limits,U32.to_nat(stride),stride_room),positive),
    Equal.trans(Nat,Nat.div(U32.to_nat(numerator),U32.to_nat(stride)),Nat.div(natural,U32.to_nat(stride)),Ceiling.ceiling(U32.to_nat(kernel),U32.to_nat(stride)),
      Equal.cong(Nat,Nat,value => Nat.div(value,U32.to_nat(stride)),U32.to_nat(numerator),natural,tap_numerator(kernel,stride,limits,kernel_room,stride_room,positive)),
      Ceiling.shifted_formula(U32.to_nat(kernel),U32.to_nat(stride),positive)))

def validated_taps_exact(+call: Domain.Call) ->
  {U32.to_nat(Kernel.taps(Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)))) == Ceiling.ceiling(Domain.kernel(Domain.geometry(call)),Domain.stride(Domain.geometry(call))) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  taps_exact(Domain.kernel_word(geometry),Domain.stride_word(geometry),limits,Domain.kernel_room(call),Domain.stride_room(call),Domain.stride_positive(call))

def taps_positive(+call: Domain.Call) ->
  {Nat.is_lt(0n,U32.to_nat(Kernel.taps(Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call))))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  +kn = Domain.kernel(geometry)
  +sn = Domain.stride(geometry)
  %Equal.sym(Nat,U32.to_nat(Kernel.taps(Domain.kernel_word(geometry),Domain.stride_word(geometry))),Ceiling.ceiling(kn,sn),validated_taps_exact(call)) : {Nat.is_lt(0n,_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(0n,Ceiling.ceiling(kn,sn)),Nat.is_lt(0n,kn),Ceiling.upper_comparison(kn,sn,0n,Domain.stride_positive(call))) : {_ == True{} : Bool}
  Domain.kernel_positive(call)

def fit_taps_exact(+call: Domain.Call) ->
  {Nat.div(Nat.add(Domain.kernel(Domain.geometry(call)),U32.to_nat((Domain.stride_word(Domain.geometry(call)) - 1 : U32))),Domain.stride(Domain.geometry(call))) == Ceiling.ceiling(Domain.kernel(Domain.geometry(call)),Domain.stride(Domain.geometry(call))) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +stride_positive = Domain.stride_positive(call)
  +stride = Domain.stride_word(geometry)
  +sn = Domain.stride(geometry)
  +capacity = Machine.coordinate(limits)
  %Equal.sym(Nat,U32.to_nat((stride - 1 : U32)),Nat.sub(sn,1n),Arithmetic.subtract(stride,1,capacity,Limits.coordinate_fits(limits),Order.positive_one(sn,stride_positive),
    Order.transitive(Nat.sub(sn,1n),sn,U32.to_nat(capacity),Order.sub_le(sn,1n),PhaseBounds.dimension_room(limits,sn,Domain.stride_room(call))))) :
    {Nat.div(Nat.add(Domain.kernel(geometry),_),sn) == Ceiling.ceiling(Domain.kernel(geometry),sn) : Nat}
  Ceiling.shifted_formula(Domain.kernel(geometry),sn,stride_positive)

def fit_size(+geometry: Kernel.Geometry) -> Nat:
  +taps = Ceiling.ceiling(Domain.kernel(geometry),Domain.stride(geometry))
  Nat.mul(taps,Nat.add(Nat.mul(Domain.outputs(geometry),Nat.add(taps,1n)),1n))

def fits_exact(+call: Domain.Call) ->
  {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == Nat.is_le(fit_size(Domain.geometry(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +native = Nat.div(Nat.add(Domain.kernel(geometry),U32.to_nat((s - 1 : U32))),Domain.stride(geometry))
  +taps = Ceiling.ceiling(Domain.kernel(geometry),Domain.stride(geometry))
  Equal.cong(Nat,Bool,taps => Nat.is_le(Nat.mul(taps,Nat.add(Nat.mul(Domain.outputs(geometry),Nat.add(taps,1n)),1n)),U32.to_nat(Machine.capacity(limits))),native,taps,fit_taps_exact(call))

def axis_taps_bounded(+call: Domain.Call,+first: U32,+size: U32) ->
  {Nat.is_le(U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))))),Ceiling.ceiling(Domain.kernel(Domain.geometry(call)),Domain.stride(Domain.geometry(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +stride_positive = Domain.stride_positive(call)
  +kernel = Domain.kernel_word(geometry)
  +stride = Domain.stride_word(geometry)
  +padding = Domain.padding_word(geometry)
  +residue = Geometry.residue(first,stride,padding)
  %Equal.sym(Nat,U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding))),Geometry.tap_count(kernel,residue,stride),
    Geometry.taps_exact(first,size,kernel,stride,padding,Machine.coordinate(limits),Limits.coordinate_fits(limits),PhaseBounds.kernel_coordinate_room(call),PhaseBounds.stride_double_room(call),stride_positive)) :
    {Nat.is_le(_,Ceiling.ceiling(Domain.kernel(geometry),Domain.stride(geometry))) == True{} : Bool}
  tap_ceiling(Domain.kernel(geometry),U32.to_nat(residue),Domain.stride(geometry),stride_positive)

# All residue origins differ from padding / stride by at most one.
def quotient_origin_upper(+padding: Nat,+first: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},inside: {Nat.is_lt(first,stride) == True{} : Bool}) ->
  {Nat.is_le(Nat.div(Nat.add(padding,first),stride),1n+Nat.div(padding,stride)) == True{} : Bool}:
  %Division.quotient_shift(padding,stride,1n,positive) :
    {Nat.is_le(Nat.div(Nat.add(padding,first),stride),_) == True{} : Bool}
  Division.quotient_le(Nat.add(padding,first),Nat.add(Nat.mul(1n,stride),padding),stride,positive,
    %Equal.sym(Nat,Nat.add(stride,0n),stride,Addition.zero_right(stride)) : {Nat.is_le(Nat.add(padding,first),Nat.add(_,padding)) == True{} : Bool}
    %Order.commutative(padding,stride) : {Nat.is_le(Nat.add(padding,first),_) == True{} : Bool}
    Order.offset_le(padding,first,stride,Bounds.strict_le(first,stride,inside)))

def tap_quotient_successor(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +active: {U32.is_lt(Geometry.residue(first,stride,padding),kernel) == True{} : Bool}) ->
  {((Geometry.axis_tap(Kernel.axis(first,size,kernel,stride,padding)) / stride : U32) + 1 : U32) == Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding)) : U32}:
  +context = Kernel.axis(first,size,kernel,stride,padding)
  +taps = Geometry.axis_taps(context)
  +quotient = (Geometry.axis_tap(context) / stride : U32)
  +natural = Nat.div(Geometry.tail(kernel,Geometry.residue(first,stride,padding)),U32.to_nat(stride))
  %Equal.sym(U32,(quotient + 1 : U32),U32.from_nat(Nat.add(U32.to_nat(quotient),1n)),Arithmetic.native_add(quotient,1)) : {_ == taps : U32}
  %Equal.sym(Nat,U32.to_nat(quotient),natural,Geometry.axis_tap_quotient(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive,active)) :
    {U32.from_nat(Nat.add(_,1n)) == taps : U32}
  %Equal.sym(Nat,Nat.add(natural,1n),1n+natural,Addition.successor_width(natural)) : {U32.from_nat(_) == taps : U32}
  Arithmetic.from_nat_exact(taps,1n+natural,Geometry.active_taps_exact(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive,active))

def source_plus_taps(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +active: {U32.is_lt(Geometry.residue(first,stride,padding),kernel) == True{} : Bool}) ->
  {(Geometry.axis_source(Kernel.axis(first,size,kernel,stride,padding)) + Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding)) : U32) == ((first + padding) / stride + 1 : U32) : U32}:
  +context = Kernel.axis(first,size,kernel,stride,padding)
  +origin = ((first + padding) / stride : U32)
  +quotient = (Geometry.axis_tap(context) / stride : U32)
  +source = (origin - quotient : U32)
  %tap_quotient_successor(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive,active) :
    {(source + _ : U32) == (origin + 1 : U32) : U32}
  Equal.trans(U32,(source + (quotient + 1 : U32) : U32),((source + quotient : U32) + 1 : U32),(origin + 1 : U32),
    Equal.sym(U32,((source + quotient : U32) + 1 : U32),(source + (quotient + 1 : U32) : U32),Arithmetic.add_associative(source,quotient,1)),
    Equal.cong(U32,U32,value => (value + 1 : U32),(source + quotient : U32),origin,Subtraction.add_subtract(origin,quotient)))

def axis_high_encoded(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +active: {U32.is_lt(Geometry.residue(first,stride,padding),kernel) == True{} : Bool}) ->
  {(Geometry.axis_source(Kernel.axis(first,size,kernel,stride,padding)) + kernel + Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding)) : U32) ==
    ((first + padding) / stride + kernel + 1 : U32) : U32}:
  +context = Kernel.axis(first,size,kernel,stride,padding)
  +source = Geometry.axis_source(context)
  +taps = Geometry.axis_taps(context)
  +origin = ((first + padding) / stride : U32)
  Equal.trans(U32,((source + kernel : U32) + taps : U32),((source + taps : U32) + kernel : U32),((origin + kernel : U32) + 1 : U32),
    Equal.trans(U32,((source + kernel : U32) + taps : U32),(source + (kernel + taps : U32) : U32),((source + taps : U32) + kernel : U32),Arithmetic.add_associative(source,kernel,taps),
      Equal.trans(U32,(source + (kernel + taps : U32) : U32),(source + (taps + kernel : U32) : U32),((source + taps : U32) + kernel : U32),
        Equal.cong(U32,U32,value => (source + value : U32),(kernel + taps : U32),(taps + kernel : U32),Arithmetic.add_commutative(kernel,taps)),
        Equal.sym(U32,((source + taps : U32) + kernel : U32),(source + (taps + kernel : U32) : U32),Arithmetic.add_associative(source,taps,kernel)))),
    Equal.trans(U32,((source + taps : U32) + kernel : U32),((origin + 1 : U32) + kernel : U32),((origin + kernel : U32) + 1 : U32),
      Equal.cong(U32,U32,value => (value + kernel : U32),(source + taps : U32),(origin + 1 : U32),source_plus_taps(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive,active)),
      Equal.trans(U32,((origin + 1 : U32) + kernel : U32),(origin + (1 + kernel : U32) : U32),((origin + kernel : U32) + 1 : U32),Arithmetic.add_associative(origin,1,kernel),
        Equal.trans(U32,(origin + (1 + kernel : U32) : U32),(origin + (kernel + 1 : U32) : U32),((origin + kernel : U32) + 1 : U32),
          Equal.cong(U32,U32,value => (origin + value : U32),(1 + kernel : U32),(kernel + 1 : U32),Arithmetic.add_commutative(1,kernel)),
          Equal.sym(U32,((origin + kernel : U32) + 1 : U32),(origin + (kernel + 1 : U32) : U32),Arithmetic.add_associative(origin,kernel,1))))))

# A binary complement supplies the full-word room without specializing a Nat
# theorem to the machine maximum. Axis and band origins use the same bound.
def complement_room(+left: Nat,+right: Nat,+limit: U32,+maximum: U32,
  +maximum_equal: {maximum == Machine.word_max() : U32},
  left_room: {Nat.is_le(left,U32.to_nat(limit)) == True{} : Bool},
  right_room: {Nat.is_le(right,U32.to_nat((maximum - limit : U32))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(left,right),U32.to_nat(maximum)) == True{} : Bool}:
  +ln = U32.to_nat(limit)
  +mn = U32.to_nat(maximum)
  +ordered = Bounds.as_nat_le(limit,maximum,
    %Equal.sym(U32,maximum,Machine.word_max(),maximum_equal) : {U32.is_le(limit,_) == True{} : Bool} FullRange.maximum(limit))
  +difference = FullRange.subtract(maximum,limit,maximum,ordered,Order.sub_le(mn,ln))
  %Order.restore_shift(ln,mn,ordered) :
    {Nat.is_le(Nat.add(left,right),_) == True{} : Bool}
  Encoded.sum_bounds(left,right,ln,Nat.sub(mn,ln),left_room,
    %difference : {Nat.is_le(right,_) == True{} : Bool} right_room)

def padded_origin_exact(+call: Domain.Call,+first: U32,
  first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool}) ->
  {U32.to_nat(((first + Domain.padding_word(Domain.geometry(call))) / Domain.stride_word(Domain.geometry(call)) : U32)) ==
    Nat.div(Nat.add(U32.to_nat(first),Domain.padding(Domain.geometry(call))),Domain.stride(Domain.geometry(call))) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +stride_positive = Domain.stride_positive(call)
  +coordinate = Machine.coordinate(limits)
  +padded = (first + p : U32)
  %Arithmetic.add(first,p,coordinate,Limits.coordinate_fits(limits),PhaseBounds.padded_room(limits,U32.to_nat(first),U32.to_nat(p),first_room,Domain.padding_room(call))) :
    {U32.to_nat((padded / s : U32)) == Nat.div(_,U32.to_nat(s)) : Nat}
  NativeDivision.divide(padded,s,coordinate,Limits.coordinate_fits(limits),PhaseBounds.stride_double_room(call),stride_positive)

def padded_origin_room(+call: Domain.Call,+first: U32,
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(((first + Domain.padding_word(Domain.geometry(call))) / Domain.stride_word(Domain.geometry(call)) : U32)),U32.to_nat(Machine.coordinate(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +stride_positive = Domain.stride_positive(call)
  +padded = Nat.add(U32.to_nat(first),Domain.padding(geometry))
  %Equal.sym(Nat,U32.to_nat(((first + p) / s : U32)),Nat.div(padded,Domain.stride(geometry)),padded_origin_exact(call,first,first_room)) :
    {Nat.is_le(_,U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
  Order.transitive(Nat.div(padded,Domain.stride(geometry)),padded,U32.to_nat(Machine.coordinate(limits)),Geometry.quotient_room(padded,Domain.stride(geometry),stride_positive),
    PhaseBounds.padded_room(limits,U32.to_nat(first),Domain.padding(geometry),first_room,Domain.padding_room(call)))

def kernel_successor_bound(+kernel: U32,+quarter: U32,+margin: U32,
  +quarter_equal: {quarter == 1073741824 : U32},+margin_equal: {margin == (quarter + 1 : U32) : U32},
  room: {U32.is_le(kernel,quarter) == True{} : Bool}) ->
  {Nat.is_le(1n+U32.to_nat(kernel),U32.to_nat(margin)) == True{} : Bool}:
  +exact = FullRange.successor(quarter,
    %Equal.sym(U32,quarter,1073741824,quarter_equal) : {U32.is_lt(_,4294967295) == True{} : Bool} {==})
  %Equal.sym(U32,margin,(quarter + 1 : U32),margin_equal) : {Nat.is_le(1n+U32.to_nat(kernel),U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat((quarter + 1 : U32)),1n+U32.to_nat(quarter),exact) : {Nat.is_le(1n+U32.to_nat(kernel),_) == True{} : Bool}
  Bounds.as_nat_le(kernel,quarter,room)

def origin_high_room(+call: Domain.Call,+first: U32,
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
  +maximum: U32,+half: U32,+quarter: U32,+margin: U32,
  +maximum_equal: {maximum == Machine.word_max() : U32},+half_equal: {half == 2147483648 : U32},+quarter_equal: {quarter == 1073741824 : U32},+margin_equal: {margin == (quarter + 1 : U32) : U32}) ->
  {Nat.is_le(Nat.add(U32.to_nat(((first + Domain.padding_word(Domain.geometry(call))) / Domain.stride_word(Domain.geometry(call)) : U32)),1n+Domain.kernel(Domain.geometry(call))),U32.to_nat(maximum)) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +origin = ((first + p) / s : U32)
  +kernel_quarter_room = Order.transitive(Domain.kernel(geometry),U32.to_nat(Machine.dimension(limits)),U32.to_nat(quarter),Domain.kernel_room(call),
    %Equal.sym(U32,quarter,1073741824,quarter_equal) : {Nat.is_le(U32.to_nat(Machine.dimension(limits)),U32.to_nat(_)) == True{} : Bool}
      Bounds.as_nat_le(Machine.dimension(limits),1073741824,Limits.dimension_allocatable(limits)))
  +kernel_bound = {%Equal.sym(Bool,U32.is_le(k,quarter),Nat.is_le(U32.to_nat(k),U32.to_nat(quarter)),Comparison.le(k,quarter)) : {_ == True{} : Bool} kernel_quarter_room : {U32.is_le(k,quarter) == True{} : Bool}}
  +successor_bound = kernel_successor_bound(k,quarter,margin,quarter_equal,margin_equal,kernel_bound)
  +origin_bound = Order.transitive(U32.to_nat(origin),U32.to_nat(Machine.coordinate(limits)),U32.to_nat(half),padded_origin_room(call,first,first_room),
    %Equal.sym(U32,half,2147483648,half_equal) : {Nat.is_le(U32.to_nat(Machine.coordinate(limits)),U32.to_nat(_)) == True{} : Bool}
      Bounds.as_nat_le(Machine.coordinate(limits),2147483648,Limits.coordinate_fits(limits)))
  +margin_guard = {%Equal.sym(U32,maximum,Machine.word_max(),maximum_equal) : {U32.is_le(margin,(_ - half : U32)) == True{} : Bool}
    %Equal.sym(U32,half,2147483648,half_equal) : {U32.is_le(margin,(Machine.word_max() - _ : U32)) == True{} : Bool}
    %Equal.sym(U32,margin,(quarter + 1 : U32),margin_equal) : {U32.is_le(_,(Machine.word_max() - 2147483648 : U32)) == True{} : Bool}
    %Equal.sym(U32,quarter,1073741824,quarter_equal) : {U32.is_le((_ + 1 : U32),(Machine.word_max() - 2147483648 : U32)) == True{} : Bool} {==} : {U32.is_le(margin,(maximum - half : U32)) == True{} : Bool}}
  complement_room(U32.to_nat(origin),1n+Domain.kernel(geometry),half,maximum,maximum_equal,origin_bound,
    Order.transitive(1n+Domain.kernel(geometry),U32.to_nat(margin),U32.to_nat((maximum - half : U32)),successor_bound,Bounds.as_nat_le(margin,(maximum - half : U32),margin_guard)))

def high_sum_exact(+left: U32,+right: U32,+maximum: U32,
  +room: {Nat.is_le(Nat.add(U32.to_nat(left),1n+U32.to_nat(right)),U32.to_nat(maximum)) == True{} : Bool}) ->
  {U32.to_nat((left + right + 1 : U32)) == Nat.add(U32.to_nat(left),1n+U32.to_nat(right)) : Nat}:
  +ln = U32.to_nat(left)
  +rn = U32.to_nat(right)
  +sum = Nat.add(ln,rn)
  +exact = FullRange.add(left,right,maximum,Order.transitive(sum,Nat.add(ln,1n+rn),U32.to_nat(maximum),Order.offset_le(ln,rn,1n+rn,Bounds.strict_le(rn,1n+rn,Order.lt_succ(rn,rn,Order.reflexive(rn)))),room))
  %Equal.trans(Nat,Nat.add(Nat.add(ln,rn),1n),Nat.add(ln,Nat.add(rn,1n)),Nat.add(ln,1n+rn),Addition.associative(ln,rn,1n),
      Equal.cong(Nat,Nat,value => Nat.add(ln,value),Nat.add(rn,1n),1n+rn,Addition.successor_width(rn))) :
    {U32.to_nat((left + right + 1 : U32)) == _ : Nat}
  %exact : {U32.to_nat((left + right + 1 : U32)) == Nat.add(_,1n) : Nat}
  FullRange.add((left + right : U32),1,maximum,
    %Equal.sym(Nat,U32.to_nat((left + right : U32)),sum,exact) : {Nat.is_le(Nat.add(_,1n),U32.to_nat(maximum)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(sum,1n),Nat.add(ln,1n+rn),Equal.trans(Nat,Nat.add(sum,1n),1n+sum,Nat.add(ln,1n+rn),Addition.successor_width(sum),Equal.sym(Nat,Nat.add(ln,1n+rn),1n+sum,Addition.successor_right(ln,rn)))) :
      {Nat.is_le(_,U32.to_nat(maximum)) == True{} : Bool} room)

# Keep binary constants inside a descriptor, not a closed large unary type.
type WordEnvelope is Data:
  WordEnvelope{maximum: U32,half: U32,quarter: U32,margin: U32,
    maximum_equal: {maximum == Machine.word_max() : U32},half_equal: {half == 2147483648 : U32},
    quarter_equal: {quarter == 1073741824 : U32},margin_equal: {margin == (quarter + 1 : U32) : U32}}
def word_envelope() -> WordEnvelope:
  WordEnvelope{Machine.word_max(),2147483648,1073741824,1073741825,{==},{==},{==},{==}}
def maximum(envelope: WordEnvelope) -> U32:
  WordEnvelope{maximum,half,quarter,margin,me,he,qe,re} = envelope
  maximum

def maximum_covers(+envelope: WordEnvelope,+value: U32) -> {Nat.is_le(U32.to_nat(value),U32.to_nat(maximum(envelope))) == True{} : Bool}:
  WordEnvelope{maximum,half,quarter,margin,me,he,qe,re} = envelope
  Bounds.as_nat_le(value,maximum,
    %Equal.sym(U32,maximum,Machine.word_max(),me) : {U32.is_le(value,_) == True{} : Bool} FullRange.maximum(value))

def origin(+geometry: Kernel.Geometry,+first: U32) -> Nat:
  Nat.div(Nat.add(U32.to_nat(first),Domain.padding(geometry)),Domain.stride(geometry))
def high(+geometry: Kernel.Geometry,+first: U32) -> Nat:
  Nat.add(origin(geometry,first),1n+Domain.kernel(geometry))
# One past the highest tile that the outputs below last read.
def end_high(+geometry: Kernel.Geometry,+last: U32) -> Nat:
  Nat.add(Nat.div(Nat.add(Nat.sub(U32.to_nat(last),1n),Domain.padding(geometry)),Domain.stride(geometry)),1n+Domain.kernel(geometry))
def axis_high(+geometry: Kernel.Geometry,+first: U32,+size: U32) -> U32:
  +context = Kernel.axis(first,size,Domain.kernel_word(geometry),Domain.stride_word(geometry),Domain.padding_word(geometry))
  (Geometry.axis_source(context) + Domain.kernel_word(geometry) + Geometry.axis_taps(context) : U32)
def axis_low(+geometry: Kernel.Geometry,+first: U32,+size: U32) -> U32:
  +context = Kernel.axis(first,size,Domain.kernel_word(geometry),Domain.stride_word(geometry),Domain.padding_word(geometry))
  (Geometry.axis_source(context) + Domain.kernel_word(geometry) : U32)

def validated_high_exact(+call: Domain.Call,+first: U32,+size: U32,
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},+envelope: WordEnvelope,
  active: {U32.is_lt(Geometry.residue(first,Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))),Domain.kernel_word(Domain.geometry(call))) == True{} : Bool}) ->
  {U32.to_nat(axis_high(Domain.geometry(call),first,size)) == high(Domain.geometry(call),first) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  WordEnvelope{+maximum,+half,+quarter,+margin,+me,+he,+qe,+re} = envelope
  +coordinate = Machine.coordinate(limits)
  +native_origin = ((first + p) / s : U32)
  +native_high = (native_origin + k + 1 : U32)
  +encoded = axis_high_encoded(first,size,k,s,p,coordinate,Limits.coordinate_fits(limits),PhaseBounds.kernel_coordinate_room(call),
    PhaseBounds.stride_double_room(call),Domain.stride_positive(call),active)
  Equal.trans(Nat,U32.to_nat(axis_high(geometry,first,size)),U32.to_nat(native_high),high(geometry,first),
    Equal.cong(U32,Nat,U32.to_nat,axis_high(geometry,first,size),native_high,encoded),
    Equal.trans(Nat,U32.to_nat(native_high),Nat.add(U32.to_nat(native_origin),1n+U32.to_nat(k)),high(geometry,first),
      high_sum_exact(native_origin,k,maximum,origin_high_room(call,first,first_room,maximum,half,quarter,margin,me,he,qe,re)),
      Equal.cong(Nat,Nat,value => Nat.add(value,1n+U32.to_nat(k)),U32.to_nat(native_origin),origin(geometry,first),padded_origin_exact(call,first,first_room))))

def axis_taps_kernel_bound(+call: Domain.Call,+first: U32,+size: U32) ->
  {Nat.is_le(U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call))))),Domain.kernel(Domain.geometry(call))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Order.transitive(U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,Domain.kernel_word(geometry),Domain.stride_word(geometry),Domain.padding_word(geometry)))),Ceiling.ceiling(Domain.kernel(geometry),Domain.stride(geometry)),Domain.kernel(geometry),
    axis_taps_bounded(call,first,size),Ceiling.bounded(Domain.kernel(geometry),Domain.stride(geometry),Domain.stride_positive(call)))

# A common interval envelope covers both phase shifts and a chunk halo.
def low_from_envelope(+origin: Nat,+high: Nat,+taps: Nat,+halo: Nat,
  +origin_lower: {Nat.is_le(origin,high) == True{} : Bool},
  +taps_bounded: {Nat.is_le(taps,halo) == True{} : Bool},+taps_inside: {Nat.is_le(taps,high) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(origin,halo),Nat.sub(high,taps)) == True{} : Bool}:
  +low = Nat.sub(high,taps)
  %Equal.sym(Bool,Nat.is_le(Nat.sub(origin,halo),low),Nat.is_le(origin,Nat.add(halo,low)),Order.subtraction_le(origin,halo,low)) : {_ == True{} : Bool}
  %Order.commutative(low,halo) : {Nat.is_le(origin,_) == True{} : Bool}
  Order.transitive(origin,high,Nat.add(low,halo),origin_lower,
    %Subtraction.restore(high,taps,taps_inside) : {Nat.is_le(_,Nat.add(low,halo)) == True{} : Bool}
    Order.offset_le(low,taps,halo,taps_bounded))

def envelope_span(+low: Nat,+high: Nat,+origin: Nat,+halo: Nat,+width: Nat,
  +halo_inside: {Nat.is_le(halo,origin) == True{} : Bool},
  +lower: {Nat.is_le(Nat.sub(origin,halo),low) == True{} : Bool},+upper: {Nat.is_le(high,Nat.add(origin,width)) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(high,low),Nat.add(halo,width)) == True{} : Bool}:
  %Equal.sym(Bool,Nat.is_le(Nat.sub(high,low),Nat.add(halo,width)),Nat.is_le(high,Nat.add(low,Nat.add(halo,width))),Order.subtraction_le(high,low,Nat.add(halo,width))) : {_ == True{} : Bool}
  %Addition.associative(low,halo,width) : {Nat.is_le(high,_) == True{} : Bool}
  Order.transitive(high,Nat.add(origin,width),Nat.add(Nat.add(low,halo),width),upper,
    Order.add_le(origin,Nat.add(low,halo),width,
      %Subtraction.restore(origin,halo,halo_inside) : {Nat.is_le(_,Nat.add(low,halo)) == True{} : Bool}
      Order.add_le(Nat.sub(origin,halo),low,halo,lower)))

def halo(+geometry: Kernel.Geometry) -> Nat:
  Ceiling.ceiling(Domain.kernel(geometry),Domain.stride(geometry))

def active_when_taps_from(enabled: Bool,+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,
  compared: {U32.is_lt(Geometry.residue(first,stride,padding),kernel) == enabled : Bool},
  used: {U32.is_lt(0,Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding))) == True{} : Bool}) ->
  {U32.is_lt(Geometry.residue(first,stride,padding),kernel) == True{} : Bool}:
  match enabled:
    case True{}: compared
    case False{}:
      Empty.absurd({U32.is_lt(Geometry.residue(first,stride,padding),kernel) == True{} : Bool},
        WordEncoding.false_true(
          %compared : {U32.is_lt(0,Bool.pick(U32,_,((kernel - 1 - Geometry.residue(first,stride,padding)) / stride + 1 : U32),0)) == True{} : Bool} used))

def active_when_taps(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,
  used: {U32.is_lt(0,Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding))) == True{} : Bool}) ->
  {U32.is_lt(Geometry.residue(first,stride,padding),kernel) == True{} : Bool}:
  active_when_taps_from(U32.is_lt(Geometry.residue(first,stride,padding),kernel),first,size,kernel,stride,padding,{==},used)

def natural_high_room(+call: Domain.Call,+first: U32,
  +first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},+envelope: WordEnvelope) ->
  {Nat.is_le(high(Domain.geometry(call),first),U32.to_nat(maximum(envelope))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  WordEnvelope{+maximum,+half,+quarter,+margin,+me,+he,+qe,+re} = envelope
  %padded_origin_exact(call,first,first_room) : {Nat.is_le(Nat.add(_,1n+Domain.kernel(geometry)),U32.to_nat(maximum)) == True{} : Bool}
  origin_high_room(call,first,first_room,maximum,half,quarter,margin,me,he,qe,re)

