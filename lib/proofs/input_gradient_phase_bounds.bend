# Phase packing metadata stays inside the validated machine envelope.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_geometry.bend as Geometry
import ./input_gradient_phase_packing.bend as Packing
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./u32_shift.bend as Shift
import ./u32_division.bend as NativeDivision
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding
import ./input_gradient_phase_weights.bend as Weights
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./encoded_coordinates.bend as Encoded
import ./word_arithmetic.bend as WordArithmetic
import ../kernel_pipeline.bend as Pipeline
import ../kernel_spatial_packing.bend as Spatial
import ./kernel_packing_window_coordinates.bend as Coordinates
import ./u32_comparison.bend as Comparison

def doubled_dimension(+limits: Machine.Limits) ->
  {Nat.is_le(Nat.double(U32.to_nat(Machine.dimension(limits))),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  %Shift.double(Machine.dimension(limits),Limits.dimension_below_half(limits)) :
    {Nat.is_le(_,U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
  Bounds.as_nat_le(U32.shl(Machine.dimension(limits)),Machine.coordinate(limits),Limits.doubled_dimension_within_coordinate(limits))

def dimension_room(+limits: Machine.Limits,+value: Nat,room: {Nat.is_le(value,U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) ->
  {Nat.is_le(value,U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Order.transitive(value,U32.to_nat(Machine.dimension(limits)),U32.to_nat(Machine.coordinate(limits)),room,
    Bounds.as_nat_le(Machine.dimension(limits),Machine.coordinate(limits),Limits.dimension_within_coordinate(limits)))

def double_room(+limits: Machine.Limits,+value: Nat,room: {Nat.is_le(value,U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(value),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  Order.transitive(Nat.double(value),Nat.double(U32.to_nat(Machine.dimension(limits))),U32.to_nat(Machine.coordinate(limits)),
    Order.double_le(value,U32.to_nat(Machine.dimension(limits)),room),doubled_dimension(limits))

def padded_room(+limits: Machine.Limits,+size: Nat,+padding: Nat,
  size_room: {Nat.is_le(size,U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  padding_room: {Nat.is_le(padding,U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(size,padding),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}:
  +dimension = U32.to_nat(Machine.dimension(limits))
  Order.transitive(Nat.add(size,padding),Nat.add(dimension,padding),U32.to_nat(Machine.coordinate(limits)),Order.add_le(size,dimension,padding,size_room),
    Order.transitive(Nat.add(dimension,padding),Nat.add(dimension,dimension),U32.to_nat(Machine.coordinate(limits)),Order.offset_le(dimension,padding,dimension,padding_room),
      %Equal.sym(Nat,Nat.add(dimension,dimension),Nat.double(dimension),Addition.double(dimension)) : {Nat.is_le(_,U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
      doubled_dimension(limits)))

# The sizes of a validated call fit the coordinate limit, and so do the
# doubled stride and each padded image side.
def height_coordinate_room(+call: Domain.Call) ->
  {Nat.is_le(Domain.height(Domain.geometry(call)),U32.to_nat(Machine.coordinate(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  dimension_room(limits,Domain.height(geometry),Domain.height_room(call))
def width_coordinate_room(+call: Domain.Call) ->
  {Nat.is_le(Domain.width(Domain.geometry(call)),U32.to_nat(Machine.coordinate(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  dimension_room(limits,Domain.width(geometry),Domain.width_room(call))
def kernel_coordinate_room(+call: Domain.Call) ->
  {Nat.is_le(Domain.kernel(Domain.geometry(call)),U32.to_nat(Machine.coordinate(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  dimension_room(limits,Domain.kernel(geometry),Domain.kernel_room(call))
def stride_double_room(+call: Domain.Call) ->
  {Nat.is_le(Nat.double(Domain.stride(Domain.geometry(call))),U32.to_nat(Machine.coordinate(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  double_room(limits,Domain.stride(geometry),Domain.stride_room(call))
def padded_height_room(+call: Domain.Call) ->
  {Nat.is_le(Nat.add(Domain.height(Domain.geometry(call)),Domain.padding(Domain.geometry(call))),U32.to_nat(Machine.coordinate(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  padded_room(limits,Domain.height(geometry),Domain.padding(geometry),Domain.height_room(call),Domain.padding_room(call))
def padded_width_room(+call: Domain.Call) ->
  {Nat.is_le(Nat.add(Domain.width(Domain.geometry(call)),Domain.padding(Domain.geometry(call))),U32.to_nat(Machine.coordinate(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  padded_room(limits,Domain.width(geometry),Domain.padding(geometry),Domain.width_room(call),Domain.padding_room(call))

def tap_count_room(choice: Bool,+kernel: U32,+residue: U32,+stride: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},compared: {Nat.is_lt(U32.to_nat(residue),U32.to_nat(kernel)) == choice : Bool}) ->
  {Nat.is_le(Bool.pick(Nat,choice,1n+Nat.div(Geometry.tail(kernel,residue),U32.to_nat(stride)),0n),U32.to_nat(kernel)) == True{} : Bool}:
  match choice:
    case False{}: Order.zero_le(U32.to_nat(kernel))
    case True{}: Geometry.count_room(kernel,residue,stride,positive,compared)

def taps_room(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding))),U32.to_nat(kernel)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding))),Geometry.tap_count(kernel,Geometry.residue(first,stride,padding),stride),Geometry.taps_exact(first,size,kernel,stride,padding,capacity,capacity_ok,kernel_room,stride_room,positive)) :
    {Nat.is_le(_,U32.to_nat(kernel)) == True{} : Bool}
  tap_count_room(Nat.is_lt(U32.to_nat(Geometry.residue(first,stride,padding)),U32.to_nat(kernel)),kernel,Geometry.residue(first,stride,padding),stride,positive,{==})

# The taps of a phase cover at most the kernel area.
def phase_area_room(+call: Domain.Call,+y: U32,+x: U32) ->
  {Nat.is_le(Packing.phase_area(Kernel.phase(Domain.geometry(call),y,x)),Nat.mul(Domain.kernel(Domain.geometry(call)),Domain.kernel(Domain.geometry(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +stride_positive = Domain.stride_positive(call)
  +kernel_coordinate_room = kernel_coordinate_room(call)
  +stride_double_room = stride_double_room(call)
  +rows = U32.to_nat(Geometry.axis_taps(Kernel.axis(y,h,k,s,p)))
  +columns = U32.to_nat(Geometry.axis_taps(Kernel.axis(x,w,k,s,p)))
  +kn = U32.to_nat(k)
  +coordinate = Machine.coordinate(limits)
  Order.transitive(Nat.mul(rows,columns),Nat.mul(kn,columns),Nat.mul(kn,kn),
    RowMajor.mul_le(rows,kn,columns,taps_room(y,h,k,s,p,coordinate,Limits.coordinate_fits(limits),kernel_coordinate_room,stride_double_room,stride_positive)),
    DivisionOrder.multiply_left_le(kn,columns,kn,taps_room(x,w,k,s,p,coordinate,Limits.coordinate_fits(limits),kernel_coordinate_room,stride_double_room,stride_positive)))

def phase_product_room(+call: Domain.Call,+y: U32,+x: U32) ->
  {Nat.is_le(Nat.mul(U32.to_nat(Packing.phase_outputs(Kernel.phase(Domain.geometry(call),y,x))),Packing.phase_area(Kernel.phase(Domain.geometry(call),y,x))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +area = Packing.phase_area(Kernel.phase(geometry,y,x))
  +square = Nat.mul(U32.to_nat(k),U32.to_nat(k))
  +full = Nat.mul(U32.to_nat(o),square)
  +channel_room = {
    %RowMajor.multiply_commutative(square,Domain.channels(geometry)) : {Nat.is_le(square,_) == True{} : Bool}
    RowMajor.product_covers(square,Domain.channels(geometry),Domain.channels_positive(call)) : {Nat.is_le(square,Nat.mul(Domain.channels(geometry),square)) == True{} : Bool}}
  Order.transitive(Nat.mul(U32.to_nat(o),area),full,U32.to_nat(Machine.capacity(limits)),DivisionOrder.multiply_left_le(U32.to_nat(o),area,square,phase_area_room(call,y,x)),
    Order.transitive(full,Nat.mul(U32.to_nat(o),Nat.mul(Domain.channels(geometry),square)),U32.to_nat(Machine.capacity(limits)),DivisionOrder.multiply_left_le(U32.to_nat(o),square,Nat.mul(Domain.channels(geometry),square),channel_room),Domain.weight_room(call)))

def phase_inner_exact(+call: Domain.Call,+y: U32,+x: U32) ->
  {U32.to_nat(Kernel.phase_inner(Kernel.phase(Domain.geometry(call),y,x))) == Nat.mul(U32.to_nat(Packing.phase_outputs(Kernel.phase(Domain.geometry(call),y,x))),Packing.phase_area(Kernel.phase(Domain.geometry(call),y,x))) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +yr = Geometry.axis_taps(Kernel.axis(y,h,k,s,p))
  +xr = Geometry.axis_taps(Kernel.axis(x,w,k,s,p))
  +product = Nat.mul(Nat.mul(U32.to_nat(o),U32.to_nat(yr)),U32.to_nat(xr))
  %RowMajor.assoc(U32.to_nat(o),U32.to_nat(yr),U32.to_nat(xr)) :
    {U32.to_nat(Kernel.phase_inner(Kernel.phase(geometry,y,x))) == _ : Nat}
  %Equal.sym(U32,(o * yr * xr : U32),U32.from_nat(product),Encoded.product_three_encoded(o,yr,xr)) :
    {U32.to_nat(_) == product : Nat}
  Bounds.roundtrip(product,Machine.capacity(limits),Limits.capacity_fits(limits),
    %Equal.sym(Nat,product,Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(yr),U32.to_nat(xr))),RowMajor.assoc(U32.to_nat(o),U32.to_nat(yr),U32.to_nat(xr))) :
      {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} phase_product_room(call,y,x))

def active_selected(choice: Bool,+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+offset: Nat,
  compared: {U32.is_lt(Geometry.residue(first,stride,padding),kernel) == choice : Bool},
  inside: {Nat.is_lt(offset,U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding)))) == True{} : Bool}) ->
  {U32.is_lt(Geometry.residue(first,stride,padding),kernel) == True{} : Bool}:
  match choice:
    case True{}: compared
    case False{}:
      Empty.absurd({U32.is_lt(Geometry.residue(first,stride,padding),kernel) == True{} : Bool},WordEncoding.false_true(Equal.sym(Bool,True{},False{},
        Equal.trans(Bool,True{},Nat.is_lt(offset,0n),False{},
          Equal.sym(Bool,Nat.is_lt(offset,0n),True{},
            %compared :
              {Nat.is_lt(offset,U32.to_nat(Bool.pick(U32,_,((kernel - 1 - Geometry.residue(first,stride,padding)) / stride + 1 : U32),0))) == True{} : Bool} inside),Division.lt_zero(offset)))))

def active(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+offset: Nat,
  inside: {Nat.is_lt(offset,U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding)))) == True{} : Bool}) ->
  {U32.is_lt(Geometry.residue(first,stride,padding),kernel) == True{} : Bool}:
  active_selected(U32.is_lt(Geometry.residue(first,stride,padding),kernel),first,size,kernel,stride,padding,offset,{==},inside)

def pixel(+first: U32,+stride: U32,+position: U32) -> Nat:
  Nat.add(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))
def pixel_quotient(+first: U32,+stride: U32,+padding: U32,+position: U32) -> U32:
  ((first + padding) / stride + position : U32)
def tap(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+offset: Nat) -> U32:
  (Geometry.axis_tap(Kernel.axis(first,size,kernel,stride,padding)) - U32.from_nat(Nat.mul(offset,U32.to_nat(stride))) : U32)
def tap_quotient(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+offset: Nat) -> U32:
  (tap(first,size,kernel,stride,padding,offset) / stride : U32)

type AxisFacts<-first: U32,-size: U32,-kernel: U32,-stride: U32,-padding: U32,-position: U32,-offset: Nat,-gradient: U32,-capacity: U32> is Data:
  AxisFacts{residue_inside: {Nat.is_lt(U32.to_nat(Geometry.residue(first,stride,padding)),U32.to_nat(stride)) == True{} : Bool},
    pixel_inside: {Nat.is_lt(pixel(first,stride,position),U32.to_nat(size)) == True{} : Bool},
    pixel_radix: {Nat.add(pixel(first,stride,position),U32.to_nat(padding)) == Nat.add(Nat.mul(U32.to_nat(pixel_quotient(first,stride,padding,position)),U32.to_nat(stride)),U32.to_nat(Geometry.residue(first,stride,padding))) : Nat},
    tap_inside: {Nat.is_lt(U32.to_nat(tap(first,size,kernel,stride,padding,offset)),U32.to_nat(kernel)) == True{} : Bool},
    tap_radix: {U32.to_nat(tap(first,size,kernel,stride,padding,offset)) == Nat.add(Nat.mul(U32.to_nat(tap_quotient(first,size,kernel,stride,padding,offset)),U32.to_nat(stride)),U32.to_nat(Geometry.residue(first,stride,padding))) : Nat},
    counter: {((Geometry.axis_source(Kernel.axis(first,size,kernel,stride,padding)) + position : U32) + U32.from_nat(offset) : U32) ==
      (pixel_quotient(first,stride,padding,position) - tap_quotient(first,size,kernel,stride,padding,offset) : U32) : U32},
    source_room: {Nat.is_le(Nat.add(U32.to_nat(gradient),U32.to_nat(tap_quotient(first,size,kernel,stride,padding,offset))),U32.to_nat(capacity)) == True{} : Bool}}

def axis_facts(+first: U32,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+position: U32,+offset: Nat,+gradient: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(capacity)) == True{} : Bool},
  +kernel_room: {Nat.is_le(U32.to_nat(kernel),U32.to_nat(capacity)) == True{} : Bool},
  +stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(capacity)) == True{} : Bool},
  +padded_room: {Nat.is_le(Nat.add(U32.to_nat(size),U32.to_nat(padding)),U32.to_nat(capacity)) == True{} : Bool},
  +source_room: {Nat.is_le(Nat.add(Nat.mul(U32.to_nat(gradient),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(capacity)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  +first_inside: {Nat.is_lt(U32.to_nat(first),U32.to_nat(size)) == True{} : Bool},
  +position_inside: {Nat.is_lt(U32.to_nat(position),U32.to_nat(Geometry.axis_count(Kernel.axis(first,size,kernel,stride,padding)))) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,kernel,stride,padding)))) == True{} : Bool}) -> AxisFacts<first,size,kernel,stride,padding,position,offset,gradient,capacity>:
  +selected = active(first,size,kernel,stride,padding,offset,offset_inside)
  +point = pixel(first,stride,position)
  +point_inside = Geometry.position_inside(first,size,kernel,stride,padding,U32.to_nat(position),capacity,capacity_ok,size_room,stride_room,positive,first_inside,position_inside)
  +point_room = Order.transitive(Nat.add(point,U32.to_nat(padding)),Nat.add(U32.to_nat(size),U32.to_nat(padding)),U32.to_nat(capacity),Order.add_le(point,U32.to_nat(size),U32.to_nat(padding),Bounds.strict_le(point,U32.to_nat(size),point_inside)),padded_room)
  +first_room = Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(padding)),Nat.add(U32.to_nat(size),U32.to_nat(padding)),U32.to_nat(capacity),Order.add_le(U32.to_nat(first),U32.to_nat(size),U32.to_nat(padding),Bounds.strict_le(U32.to_nat(first),U32.to_nat(size),first_inside)),padded_room)
  +base_exact = Arithmetic.add(first,padding,capacity,capacity_ok,first_room)
  +shift_room = {
    %Equal.sym(Nat,U32.to_nat((first + padding : U32)),Nat.add(U32.to_nat(first),U32.to_nat(padding)),base_exact) : {Nat.is_le(Nat.add(_,Nat.mul(U32.to_nat(position),U32.to_nat(stride))),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.add(U32.to_nat(first),U32.to_nat(padding)),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(U32.to_nat(first),Nat.add(U32.to_nat(padding),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))),Addition.associative(U32.to_nat(first),U32.to_nat(padding),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(U32.to_nat(padding),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),Nat.add(Nat.mul(U32.to_nat(position),U32.to_nat(stride)),U32.to_nat(padding)),Order.commutative(U32.to_nat(padding),Nat.mul(U32.to_nat(position),U32.to_nat(stride)))) : {Nat.is_le(Nat.add(U32.to_nat(first),_),U32.to_nat(capacity)) == True{} : Bool}
    %Addition.associative(U32.to_nat(first),Nat.mul(U32.to_nat(position),U32.to_nat(stride)),U32.to_nat(padding)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} point_room :
    {Nat.is_le(Nat.add(U32.to_nat((first + padding : U32)),Nat.mul(U32.to_nat(position),U32.to_nat(stride))),U32.to_nat(capacity)) == True{} : Bool}}
  +residue = U32.to_nat(Geometry.residue(first,stride,padding))
  +q = Nat.div(Geometry.tail(kernel,Geometry.residue(first,stride,padding)),U32.to_nat(stride))
  +remaining = Nat.sub(q,offset)
  +tap_value = tap(first,size,kernel,stride,padding,offset)
  +tap_exact = Geometry.axis_offset_exact(first,size,kernel,stride,padding,offset,capacity,capacity_ok,kernel_room,stride_room,positive,selected,offset_inside)
  +tap_last_room = Order.transitive(Nat.add(residue,Nat.mul(remaining,U32.to_nat(stride))),Geometry.last_tap(kernel,Geometry.residue(first,stride,padding),stride),Nat.sub(U32.to_nat(kernel),1n),
    Order.offset_le(residue,Nat.mul(remaining,U32.to_nat(stride)),Nat.mul(q,U32.to_nat(stride)),RowMajor.mul_le(remaining,q,U32.to_nat(stride),Order.sub_le(q,offset))),Geometry.last_tap_room(kernel,Geometry.residue(first,stride,padding),stride,positive,Bounds.as_nat_lt(Geometry.residue(first,stride,padding),kernel,selected)))
  +tap_inside = {
    %Equal.sym(Nat,U32.to_nat(tap_value),Nat.add(residue,Nat.mul(remaining,U32.to_nat(stride))),tap_exact) : {Nat.is_lt(_,U32.to_nat(kernel)) == True{} : Bool}
    Order.le_lt_trans(Nat.add(residue,Nat.mul(remaining,U32.to_nat(stride))),Nat.sub(U32.to_nat(kernel),1n),U32.to_nat(kernel),tap_last_room,Order.last_below(U32.to_nat(kernel),Order.positive_bound(U32.to_nat(Geometry.residue(first,stride,padding)),U32.to_nat(kernel),Bounds.as_nat_lt(Geometry.residue(first,stride,padding),kernel,selected)))) :
    {Nat.is_lt(U32.to_nat(tap_value),U32.to_nat(kernel)) == True{} : Bool}}
  +quotient = tap_quotient(first,size,kernel,stride,padding,offset)
  +quotient_exact = Geometry.axis_offset_quotient(first,size,kernel,stride,padding,offset,capacity,capacity_ok,kernel_room,stride_room,positive,selected,offset_inside)
  +quotient_room = Order.transitive(U32.to_nat(quotient),U32.to_nat(tap_value),U32.to_nat(kernel),
    %Equal.sym(Nat,U32.to_nat(quotient),Nat.div(U32.to_nat(tap_value),U32.to_nat(stride)),NativeDivision.divide(tap_value,stride,capacity,capacity_ok,stride_room,positive)) : {Nat.is_le(_,U32.to_nat(tap_value)) == True{} : Bool} Geometry.quotient_room(U32.to_nat(tap_value),U32.to_nat(stride),positive),Bounds.strict_le(U32.to_nat(tap_value),U32.to_nat(kernel),tap_inside))
  AxisFacts{Geometry.residue_inside(first,stride,padding,capacity,capacity_ok,stride_room,positive),point_inside,Geometry.axis_pixel_radix(first,stride,padding,position,capacity,capacity_ok,first_room,stride_room,positive,shift_room),tap_inside,
    %Equal.sym(Nat,U32.to_nat(quotient),remaining,quotient_exact) : {U32.to_nat(tap_value) == Nat.add(Nat.mul(_,U32.to_nat(stride)),residue) : Nat}
    %Order.commutative(residue,Nat.mul(remaining,U32.to_nat(stride))) : {U32.to_nat(tap_value) == _ : Nat} tap_exact,
    Geometry.axis_source_counter(first,size,kernel,stride,padding,position,offset,capacity,capacity_ok,kernel_room,stride_room,positive,selected,offset_inside),
    Order.transitive(Nat.add(U32.to_nat(gradient),U32.to_nat(quotient)),Nat.add(U32.to_nat(gradient),U32.to_nat(kernel)),U32.to_nat(capacity),Order.offset_le(U32.to_nat(gradient),U32.to_nat(quotient),U32.to_nat(kernel),quotient_room),
      Order.transitive(Nat.add(U32.to_nat(gradient),U32.to_nat(kernel)),Nat.add(Nat.mul(U32.to_nat(gradient),U32.to_nat(stride)),U32.to_nat(kernel)),U32.to_nat(capacity),Order.add_le(U32.to_nat(gradient),Nat.mul(U32.to_nat(gradient),U32.to_nat(stride)),U32.to_nat(kernel),RowMajor.product_covers(U32.to_nat(gradient),U32.to_nat(stride),positive)),source_room))}

def row_facts(+call: Domain.Call,+first: U32,+position: U32,+offset: Nat,
  first_inside: {Nat.is_lt(U32.to_nat(first),Domain.height(Domain.geometry(call))) == True{} : Bool},
  position_inside: {Nat.is_lt(U32.to_nat(position),U32.to_nat(Geometry.axis_count(Packing.phase_y(Kernel.phase(Domain.geometry(call),first,0))))) == True{} : Bool},
  offset_inside: {Nat.is_lt(offset,U32.to_nat(Packing.phase_rows(Kernel.phase(Domain.geometry(call),first,0)))) == True{} : Bool}) ->
  AxisFacts<first,Domain.height_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),position,offset,Domain.gradient_rows(Domain.geometry(call)),Machine.coordinate(Domain.limits(call))>:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  axis_facts(first,h,k,s,p,position,offset,gh,Machine.coordinate(limits),Limits.coordinate_fits(limits),
    height_coordinate_room(call),kernel_coordinate_room(call),stride_double_room(call),
    padded_height_room(call),Domain.row_source_room(call),Domain.stride_positive(call),first_inside,position_inside,offset_inside)

def column_facts(+call: Domain.Call,+first: U32,+position: U32,+offset: Nat,
  first_inside: {Nat.is_lt(U32.to_nat(first),Domain.width(Domain.geometry(call))) == True{} : Bool},
  position_inside: {Nat.is_lt(U32.to_nat(position),U32.to_nat(Geometry.axis_count(Packing.phase_x(Kernel.phase(Domain.geometry(call),0,first))))) == True{} : Bool},
  offset_inside: {Nat.is_lt(offset,U32.to_nat(Packing.phase_columns(Kernel.phase(Domain.geometry(call),0,first)))) == True{} : Bool}) ->
  AxisFacts<first,Domain.width_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),position,offset,Domain.gradient_columns(Domain.geometry(call)),Machine.coordinate(Domain.limits(call))>:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  axis_facts(first,w,k,s,p,position,offset,gw,Machine.coordinate(limits),Limits.coordinate_fits(limits),
    width_coordinate_room(call),kernel_coordinate_room(call),stride_double_room(call),
    padded_width_room(call),Domain.column_source_room(call),Domain.stride_positive(call),first_inside,position_inside,offset_inside)

def input_base_exact(+geometry: Kernel.Geometry,+y: U32,+x: U32,+image: U32) ->
  {Packing.input_base(Kernel.phase(geometry,y,x),image) == U32.from_nat(Nat.mul(U32.to_nat(image),Domain.gradient_size(geometry))) : U32}:
  Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
  %RowMajor.assoc(U32.to_nat(image),U32.to_nat(o),Nat.mul(U32.to_nat(gh),U32.to_nat(gw))) :
    {Packing.input_base(Kernel.phase(geometry,y,x),image) == U32.from_nat(_) : U32}
  %RowMajor.assoc(Nat.mul(U32.to_nat(image),U32.to_nat(o)),U32.to_nat(gh),U32.to_nat(gw)) :
    {Packing.input_base(Kernel.phase(geometry,y,x),image) == U32.from_nat(_) : U32}
  Encoded.product_four_encoded(image,o,gh,gw)

def gradient_step_exact(+call: Domain.Call,+y: U32,+x: U32) ->
  {U32.to_nat(Packing.gradient_step(Kernel.phase(Domain.geometry(call),y,x))) == Nat.mul(Domain.gradient_height(Domain.geometry(call)),Domain.gradient_width(Domain.geometry(call))) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Arithmetic.multiply(gh,gw,Machine.capacity(limits),Limits.capacity_fits(limits),Domain.gradient_area_room(call))

def gradient_base_exact(+call: Domain.Call,+y: U32,+x: U32,+image: U32,+output: Nat) ->
  {(Packing.input_base(Kernel.phase(Domain.geometry(call),y,x),image) + U32.from_nat(Nat.mul(output,U32.to_nat(Packing.gradient_step(Kernel.phase(Domain.geometry(call),y,x))))) : U32) ==
    U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),Domain.gradient_size(Domain.geometry(call))),Nat.mul(output,Nat.mul(Domain.gradient_height(Domain.geometry(call)),Domain.gradient_width(Domain.geometry(call)))))) : U32}:
  Domain.Call{geometry,batch,limits,facts} = call
  %gradient_step_exact(call,y,x) :
    {(Packing.input_base(Kernel.phase(geometry,y,x),image) + U32.from_nat(Nat.mul(output,U32.to_nat(Packing.gradient_step(Kernel.phase(geometry,y,x))))) : U32) == U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),Domain.gradient_size(geometry)),Nat.mul(output,_))) : U32}
  %Equal.sym(U32,Packing.input_base(Kernel.phase(geometry,y,x),image),U32.from_nat(Nat.mul(U32.to_nat(image),Domain.gradient_size(geometry))),input_base_exact(geometry,y,x,image)) :
    {(_ + U32.from_nat(Nat.mul(output,U32.to_nat(Packing.gradient_step(Kernel.phase(geometry,y,x))))) : U32) == U32.from_nat(Nat.add(Nat.mul(U32.to_nat(image),Domain.gradient_size(geometry)),Nat.mul(output,U32.to_nat(Packing.gradient_step(Kernel.phase(geometry,y,x)))))) : U32}
  WordArithmetic.native_add(Nat.mul(U32.to_nat(image),Domain.gradient_size(geometry)),Nat.mul(output,U32.to_nat(Packing.gradient_step(Kernel.phase(geometry,y,x)))))

def weight_step_exact(+call: Domain.Call,+y: U32,+x: U32) ->
  {U32.to_nat(Weights.output_step(Kernel.phase(Domain.geometry(call),y,x))) == Nat.mul(Domain.channels(Domain.geometry(call)),Nat.mul(Domain.kernel(Domain.geometry(call)),Domain.kernel(Domain.geometry(call)))) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  %RowMajor.assoc(U32.to_nat(c),U32.to_nat(k),U32.to_nat(k)) :
    {U32.to_nat(Weights.output_step(Kernel.phase(geometry,y,x))) == _ : Nat}
  Encoded.multiply_three(c,k,k,Machine.capacity(limits),Limits.capacity_fits(limits),Domain.kernel_positive(call),
    %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(c),U32.to_nat(k)),U32.to_nat(k)),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),U32.to_nat(k))),RowMajor.assoc(U32.to_nat(c),U32.to_nat(k),U32.to_nat(k))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} Domain.kernel_words_room(call))

def weight_square_exact(+call: Domain.Call,+y: U32,+x: U32) ->
  {U32.to_nat((Weights.kernel(Kernel.phase(Domain.geometry(call),y,x)) * Weights.kernel(Kernel.phase(Domain.geometry(call),y,x)) : U32)) == Nat.mul(Domain.kernel(Domain.geometry(call)),Domain.kernel(Domain.geometry(call))) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Arithmetic.multiply(k,k,Machine.capacity(limits),Limits.capacity_fits(limits),Domain.kernel_area_room(call))

def lane_position_exact(+call: Domain.Call,+first: U32,+position: U32,+lane: Nat,
  +first_inside: {Nat.is_lt(U32.to_nat(first),Domain.width(Domain.geometry(call))) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(U32.to_nat(position),lane),U32.to_nat(Geometry.axis_count(Packing.phase_x(Kernel.phase(Domain.geometry(call),0,first))))) == True{} : Bool}) ->
  {U32.to_nat((position + U32.from_nat(lane) : U32)) == Nat.add(U32.to_nat(position),lane) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +stride_positive = Domain.stride_positive(call)
  +width_coordinate_room = width_coordinate_room(call)
  +capacity = Machine.coordinate(limits)
  +sum = Nat.add(U32.to_nat(position),lane)
  +count = U32.to_nat(Geometry.axis_count(Kernel.axis(first,w,k,s,p)))
  +count_room = {
    %Equal.sym(Nat,count,1n+Nat.div(Geometry.tail(w,first),U32.to_nat(s)),Geometry.count_exact(first,w,k,s,p,capacity,Limits.coordinate_fits(limits),width_coordinate_room,stride_double_room(call),stride_positive,first_inside)) : {Nat.is_le(_,U32.to_nat(w)) == True{} : Bool}
    Geometry.count_room(w,first,s,stride_positive,first_inside) : {Nat.is_le(count,U32.to_nat(w)) == True{} : Bool}}
  +sum_room = Order.transitive(sum,count,U32.to_nat(capacity),Bounds.strict_le(sum,count,inside),Order.transitive(count,U32.to_nat(w),U32.to_nat(capacity),count_room,width_coordinate_room))
  %Equal.sym(U32,(position + U32.from_nat(lane) : U32),U32.from_nat(sum),
    %Arithmetic.native_roundtrip(position) : {(_ + U32.from_nat(lane) : U32) == U32.from_nat(sum) : U32}
    WordArithmetic.native_add(U32.to_nat(position),lane)) : {U32.to_nat(_) == sum : Nat}
  Bounds.roundtrip(sum,capacity,Limits.coordinate_fits(limits),sum_room)

def column_lane_facts(+call: Domain.Call,+first: U32,+position: U32,+lane: Nat,+offset: Nat,
  +first_inside: {Nat.is_lt(U32.to_nat(first),Domain.width(Domain.geometry(call))) == True{} : Bool},
  +position_inside: {Nat.is_lt(Nat.add(U32.to_nat(position),lane),U32.to_nat(Geometry.axis_count(Packing.phase_x(Kernel.phase(Domain.geometry(call),0,first))))) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(Packing.phase_columns(Kernel.phase(Domain.geometry(call),0,first)))) == True{} : Bool}) ->
  AxisFacts<first,Domain.width_word(Domain.geometry(call)),Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)),(position + U32.from_nat(lane) : U32),offset,Domain.gradient_columns(Domain.geometry(call)),Machine.coordinate(Domain.limits(call))>:
  column_facts(call,first,(position + U32.from_nat(lane) : U32),offset,first_inside,
    %Equal.sym(Nat,U32.to_nat((position + U32.from_nat(lane) : U32)),Nat.add(U32.to_nat(position),lane),lane_position_exact(call,first,position,lane,first_inside,position_inside)) :
      {Nat.is_lt(_,U32.to_nat(Geometry.axis_count(Packing.phase_x(Kernel.phase(Domain.geometry(call),0,first))))) == True{} : Bool} position_inside,offset_inside)

def lane_counter_order(+source: U32,+position: U32,+offset: U32,+lane: U32) ->
  {(((source + position : U32) + offset : U32) + lane : U32) == ((source + (position + lane : U32) : U32) + offset : U32) : U32}:
  %Arithmetic.add_associative(source,position,lane) :
    {(((source + position : U32) + offset : U32) + lane : U32) == (_ + offset : U32) : U32}
  %Equal.sym(U32,(((source + position : U32) + lane : U32) + offset : U32),((source + position : U32) + (lane + offset : U32) : U32),Arithmetic.add_associative((source + position : U32),lane,offset)) :
    {(((source + position : U32) + offset : U32) + lane : U32) == _ : U32}
  %Arithmetic.add_commutative(offset,lane) :
    {(((source + position : U32) + offset : U32) + lane : U32) == ((source + position : U32) + _ : U32) : U32}
  Arithmetic.add_associative((source + position : U32),offset,lane)

def column_lane_counter(+geometry: Kernel.Geometry,+y: U32,+x: U32,+position: U32,+lane: Nat,+offset: Nat,
  counter: {((Geometry.axis_source(Packing.phase_x(Kernel.phase(geometry,y,x))) + (position + U32.from_nat(lane) : U32) : U32) + U32.from_nat(offset) : U32) ==
    (pixel_quotient(x,Domain.stride_word(geometry),Domain.padding_word(geometry),(position + U32.from_nat(lane) : U32)) - tap_quotient(x,Domain.width_word(geometry),Domain.kernel_word(geometry),Domain.stride_word(geometry),Domain.padding_word(geometry),offset) : U32) : U32}) ->
  {((Packing.input_column(Kernel.phase(geometry,y,x),position) + U32.from_nat(offset) : U32) + U32.from_nat(lane) : U32) ==
    (pixel_quotient(x,Domain.stride_word(geometry),Domain.padding_word(geometry),(position + U32.from_nat(lane) : U32)) - tap_quotient(x,Domain.width_word(geometry),Domain.kernel_word(geometry),Domain.stride_word(geometry),Domain.padding_word(geometry),offset) : U32) : U32}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  %Equal.sym(U32,(((Geometry.axis_source(Packing.phase_x(Kernel.phase(geometry,y,x))) + position : U32) + U32.from_nat(offset) : U32) + U32.from_nat(lane) : U32),
    ((Geometry.axis_source(Packing.phase_x(Kernel.phase(geometry,y,x))) + (position + U32.from_nat(lane) : U32) : U32) + U32.from_nat(offset) : U32),
    lane_counter_order(Geometry.axis_source(Packing.phase_x(Kernel.phase(geometry,y,x))),position,U32.from_nat(offset),U32.from_nat(lane))) : {_ ==
      (pixel_quotient(x,Domain.stride_word(geometry),Domain.padding_word(geometry),(position + U32.from_nat(lane) : U32)) - tap_quotient(x,Domain.width_word(geometry),Domain.kernel_word(geometry),Domain.stride_word(geometry),Domain.padding_word(geometry),offset) : U32) : U32}
  counter

def phase_stencil(context: Kernel.Phase) -> Spatial.Stencil:
  Kernel.Phase{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},Kernel.Axis{yf,yc,yt,yr,ys},Kernel.Axis{xf,xc,xt,xr,xs},inner} = context
  Spatial.Stencil{0,gh,gw,yr,xr,1,ys,xs,xc,inner}

def phase_inner_positive(+context: Kernel.Phase,active: {U32.is_eq(Kernel.phase_inner(context),0) == False{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(Kernel.phase_inner(context))) == True{} : Bool}:
  Order.nonzero_positive(U32.to_nat(Kernel.phase_inner(context)),
    %Comparison.eq(Kernel.phase_inner(context),0) : {_ == False{} : Bool} active)

def window_geometry(+call: Domain.Call,+y: U32,+x: U32,
  y_inside: {Nat.is_lt(U32.to_nat(y),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +x_inside: {Nat.is_lt(U32.to_nat(x),Domain.width(Domain.geometry(call))) == True{} : Bool},
  active: {U32.is_eq(Kernel.phase_inner(Kernel.phase(Domain.geometry(call),y,x)),0) == False{} : Bool}) -> Coordinates.Geometry:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +stride_positive = Domain.stride_positive(call)
  +width_room = Domain.width_room(call)
  +kernel_room = Domain.kernel_room(call)
  +kernel_coordinate_room = kernel_coordinate_room(call)
  +stride_double_room = stride_double_room(call)
  +context = Kernel.phase(geometry,y,x)
  +window = phase_stencil(context)
  +row_word = Geometry.axis_taps(Packing.phase_y(context))
  +column_word = Geometry.axis_taps(Packing.phase_x(context))
  +rows = U32.to_nat(row_word)
  +columns = U32.to_nat(column_word)
  +area = Nat.mul(rows,columns)
  +inner_word = Kernel.phase_inner(context)
  +inner = U32.to_nat(inner_word)
  +width = U32.to_nat(Geometry.axis_count(Packing.phase_x(context)))
  +capacity = Machine.capacity(limits)
  +half = Machine.half_word(limits)
  +coordinate = Machine.coordinate(limits)
  +coordinate_ok = Limits.coordinate_fits(limits)
  +row_bound = taps_room(y,h,k,s,p,coordinate,coordinate_ok,kernel_coordinate_room,stride_double_room,stride_positive)
  +column_bound = taps_room(x,w,k,s,p,coordinate,coordinate_ok,kernel_coordinate_room,stride_double_room,stride_positive)
  +inner_exact = phase_inner_exact(call,y,x)
  +ip = phase_inner_positive(context,active)
  +product_positive = {
    %inner_exact : {Nat.is_lt(0n,_) == True{} : Bool} ip : {Nat.is_lt(0n,Nat.mul(U32.to_nat(o),area)) == True{} : Bool}}
  +area_positive = RowMajor.left_factor_positive(area,U32.to_nat(o),
    %RowMajor.multiply_commutative(U32.to_nat(o),area) : {Nat.is_lt(0n,_) == True{} : Bool} product_positive)
  +rp = RowMajor.left_factor_positive(rows,columns,area_positive)
  +cp = RowMajor.left_factor_positive(columns,rows,
    %RowMajor.multiply_commutative(rows,columns) : {Nat.is_lt(0n,_) == True{} : Bool} area_positive)
  +area_bound = Order.transitive(area,Nat.mul(U32.to_nat(k),U32.to_nat(k)),U32.to_nat(capacity),
    Order.transitive(area,Nat.mul(U32.to_nat(k),columns),Nat.mul(U32.to_nat(k),U32.to_nat(k)),RowMajor.mul_le(rows,U32.to_nat(k),columns,row_bound),DivisionOrder.multiply_left_le(U32.to_nat(k),columns,U32.to_nat(k),column_bound)),Domain.kernel_area_room(call))
  +area_word = (row_word * column_word : U32)
  +area_exact = Arithmetic.multiply(row_word,column_word,capacity,Limits.capacity_fits(limits),area_bound)
  +area_double = Limits.double_bound(limits,area,area_bound)
  +area_word_double = {%Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(half)) == True{} : Bool} area_double : {Nat.is_le(Nat.double(U32.to_nat(area_word)),U32.to_nat(half)) == True{} : Bool}}
  +area_word_positive = {%Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.is_lt(0n,_) == True{} : Bool} area_positive : {Nat.is_lt(0n,U32.to_nat(area_word)) == True{} : Bool}}
  +channels_exact = Equal.trans(Nat,U32.to_nat((inner_word / area_word : U32)),Nat.div(inner,U32.to_nat(area_word)),U32.to_nat(o),
    NativeDivision.divide(inner_word,area_word,half,Limits.half_word_fits(limits),area_word_double,area_word_positive),{
      %Equal.sym(Nat,inner,Nat.mul(U32.to_nat(o),area),inner_exact) : {Nat.div(_,U32.to_nat(area_word)) == U32.to_nat(o) : Nat}
      %Equal.sym(Nat,U32.to_nat(area_word),area,area_exact) : {Nat.div(Nat.mul(U32.to_nat(o),area),_) == U32.to_nat(o) : Nat}
      Division.exact_quotient(U32.to_nat(o),area,area_positive) : {Nat.div(inner,U32.to_nat(area_word)) == U32.to_nat(o) : Nat}})
  +width_exact = Geometry.count_exact(x,w,k,s,p,coordinate,coordinate_ok,width_coordinate_room(call),stride_double_room,stride_positive,x_inside)
  +wp = {%Equal.sym(Nat,width,1n+Nat.div(Geometry.tail(w,x),U32.to_nat(s)),width_exact) : {Nat.is_lt(0n,_) == True{} : Bool} {==} : {Nat.is_lt(0n,width) == True{} : Bool}}
  +width_bound = {%Equal.sym(Nat,width,1n+Nat.div(Geometry.tail(w,x),U32.to_nat(s)),width_exact) : {Nat.is_le(_,U32.to_nat(w)) == True{} : Bool} Geometry.count_room(w,x,s,stride_positive,x_inside) : {Nat.is_le(width,U32.to_nat(w)) == True{} : Bool}}
  +inner_bound = {%Equal.sym(Nat,inner,Nat.mul(U32.to_nat(o),area),inner_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} phase_product_room(call,y,x) : {Nat.is_le(inner,U32.to_nat(capacity)) == True{} : Bool}}
  Coordinates.Geometry{window,U32.to_nat(o),half,Limits.half_word_fits(limits),rp,cp,ip,wp,inner_exact,channels_exact,
    Limits.double_bound(limits,rows,Order.transitive(rows,U32.to_nat(k),U32.to_nat(capacity),row_bound,Limits.dimension_capacity(limits,U32.to_nat(k),kernel_room))),
    Limits.double_bound(limits,columns,Order.transitive(columns,U32.to_nat(k),U32.to_nat(capacity),column_bound,Limits.dimension_capacity(limits,U32.to_nat(k),kernel_room))),area_double,
    Limits.double_bound(limits,inner,inner_bound),Limits.double_bound(limits,width,Order.transitive(width,U32.to_nat(w),U32.to_nat(capacity),width_bound,Limits.dimension_capacity(limits,U32.to_nat(w),width_room)))}

