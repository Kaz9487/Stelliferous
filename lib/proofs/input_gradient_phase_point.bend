# Validated points select phases and row-major positions.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_geometry.bend as Geometry
import ./input_gradient_phase_packing.bend as Packing
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./kernel_accumulator.bend as Accumulator
import ./kernel_weight_gradient_packing.bend as Patches
import ./machine_limit_bounds.bend as Limits
import ./encoded_coordinates.bend as Encoded
import ./nat_order.bend as Order
import ./nat_division_order.bend as Division
import ./nat_division.bend as NaturalDivision
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_extrema.bend as Extrema
import ./nat_ceiling.bend as Ceiling
import ./u32_extrema.bend as NativeExtrema
import ./row_major_index.bend as RowMajor
import ./binary_radix.bend as Radix
import ./bounded_u32_arithmetic.bend as Arithmetic

def first(+position: Nat,stride: U32) -> U32:
  U32.from_nat(Nat.mod(position,U32.to_nat(stride)))
def local(+position: Nat,stride: U32) -> U32:
  U32.from_nat(Nat.div(position,U32.to_nat(stride)))
def phase(+geometry: Kernel.Geometry,+point: Domain.Point) -> Kernel.Phase:
  Kernel.phase(geometry,first(Domain.row(point),Domain.stride_word(geometry)),first(Domain.column(point),Domain.stride_word(geometry)))
def first_below(+position: Nat,+size: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},+inside: {Nat.is_lt(position,size) == True{} : Bool}) ->
  {Nat.is_lt(Nat.mod(position,stride),size) == True{} : Bool}:
  +q = Nat.div(position,stride)
  +r = Nat.mod(position,stride)
  Order.le_lt_trans(r,position,size,
    %Equal.sym(Nat,position,Nat.add(Nat.mul(q,stride),r),Division.quotient_exact(position,stride,positive)) : {Nat.is_le(r,_) == True{} : Bool}
    %Order.commutative(r,Nat.mul(q,stride)) : {Nat.is_le(r,_) == True{} : Bool}
    Order.le_add(r,Nat.mul(q,stride)),inside)

def natural_local_inside(+position: Nat,+size: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool},+inside: {Nat.is_lt(position,size) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(position,stride),1n+Nat.div(Nat.sub(Nat.sub(size,1n),Nat.mod(position,stride)),stride)) == True{} : Bool}:
  +q = Nat.div(position,stride)
  +r = Nat.mod(position,stride)
  +shifted = { %Equal.sym(Nat,position,Nat.add(Nat.mul(q,stride),r),Division.quotient_exact(position,stride,positive)) : {Nat.sub(_,r) == Nat.mul(q,stride) : Nat} Division.subtract_sum(Nat.mul(q,stride),r) : {Nat.sub(position,r) == Nat.mul(q,stride) : Nat} }
  Order.lt_succ(q,Nat.div(Nat.sub(Nat.sub(size,1n),r),stride),
    %Equal.trans(Nat,Nat.div(Nat.sub(position,r),stride),Nat.div(Nat.mul(q,stride),stride),q,Equal.cong(Nat,Nat,value => Nat.div(value,stride),Nat.sub(position,r),Nat.mul(q,stride),shifted),NaturalDivision.exact_quotient(q,stride,positive)) : {Nat.is_le(_,Nat.div(Nat.sub(Nat.sub(size,1n),r),stride)) == True{} : Bool}
    Division.quotient_le(Nat.sub(position,r),Nat.sub(Nat.sub(size,1n),r),stride,positive,Division.subtract_le(position,Nat.sub(size,1n),r,Order.before_last(position,size,inside))))

def local_le(+position: Nat,+stride: Nat,+positive: {Nat.is_lt(0n,stride) == True{} : Bool}) -> {Nat.is_le(Nat.div(position,stride),position) == True{} : Bool}:
  Order.transitive(Nat.div(position,stride),Nat.mul(Nat.div(position,stride),stride),position,RowMajor.product_covers(Nat.div(position,stride),stride,positive),Division.product_below(position,stride,positive))

def first_exact(+position: Nat,+size: U32,+stride: U32,+limits: Machine.Limits,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+inside: {Nat.is_lt(position,U32.to_nat(size)) == True{} : Bool},size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) ->
  {U32.to_nat(first(position,stride)) == Nat.mod(position,U32.to_nat(stride)) : Nat}:
  Bounds.roundtrip(Nat.mod(position,U32.to_nat(stride)),Machine.dimension(limits),Limits.dimension_fits(limits),Bounds.strict_le(Nat.mod(position,U32.to_nat(stride)),U32.to_nat(Machine.dimension(limits)),Bounds.nat_trans(Nat.mod(position,U32.to_nat(stride)),U32.to_nat(size),U32.to_nat(Machine.dimension(limits)),first_below(position,U32.to_nat(size),U32.to_nat(stride),positive,inside),size_room)))
def local_exact(+position: Nat,+size: U32,+stride: U32,+limits: Machine.Limits,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+inside: {Nat.is_lt(position,U32.to_nat(size)) == True{} : Bool},size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) ->
  {U32.to_nat(local(position,stride)) == Nat.div(position,U32.to_nat(stride)) : Nat}:
  Bounds.roundtrip(Nat.div(position,U32.to_nat(stride)),Machine.dimension(limits),Limits.dimension_fits(limits),Order.transitive(Nat.div(position,U32.to_nat(stride)),position,U32.to_nat(Machine.dimension(limits)),local_le(position,U32.to_nat(stride),positive),Bounds.strict_le(position,U32.to_nat(Machine.dimension(limits)),Bounds.nat_trans(position,U32.to_nat(size),U32.to_nat(Machine.dimension(limits)),inside,size_room))))

def first_loop_inside(+position: Nat,+size: U32,+stride: U32,+limits: Machine.Limits,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+inside: {Nat.is_lt(position,U32.to_nat(size)) == True{} : Bool},size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(first(position,stride)),U32.to_nat(U32.min(stride,size))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(first(position,stride)),Nat.mod(position,U32.to_nat(stride)),first_exact(position,size,stride,limits,positive,inside,size_room)) : {Nat.is_lt(_,U32.to_nat(U32.min(stride,size))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.min(stride,size)),Nat.min(U32.to_nat(stride),U32.to_nat(size)),NativeExtrema.minimum(stride,size)) : {Nat.is_lt(Nat.mod(position,U32.to_nat(stride)),_) == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(Nat.mod(position,U32.to_nat(stride)),Nat.min(U32.to_nat(stride),U32.to_nat(size))),Nat.is_lt(Nat.mod(position,U32.to_nat(stride)),U32.to_nat(stride)) && Nat.is_lt(Nat.mod(position,U32.to_nat(stride)),U32.to_nat(size)),Extrema.below_minimum(Nat.mod(position,U32.to_nat(stride)),U32.to_nat(stride),U32.to_nat(size))) : {_ == True{} : Bool}
  %Equal.sym(Bool,Nat.is_lt(Nat.mod(position,U32.to_nat(stride)),U32.to_nat(stride)),True{},Division.remainder_small(position,U32.to_nat(stride),positive)) : {_ && Nat.is_lt(Nat.mod(position,U32.to_nat(stride)),U32.to_nat(size)) == True{} : Bool}
  first_below(position,U32.to_nat(size),U32.to_nat(stride),positive,inside)

def first_inside(+position: Nat,+size: U32,+stride: U32,+limits: Machine.Limits,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+inside: {Nat.is_lt(position,U32.to_nat(size)) == True{} : Bool},size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.dimension(limits))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(first(position,stride)),U32.to_nat(size)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(first(position,stride)),Nat.mod(position,U32.to_nat(stride)),first_exact(position,size,stride,limits,positive,inside,size_room)) : {Nat.is_lt(_,U32.to_nat(size)) == True{} : Bool}
  first_below(position,U32.to_nat(size),U32.to_nat(stride),positive,inside)

def local_axis_inside(+position: Nat,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+limits: Machine.Limits,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+inside: {Nat.is_lt(position,U32.to_nat(size)) == True{} : Bool},
  +size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(local(position,stride)),U32.to_nat(Geometry.axis_count(Kernel.axis(first(position,stride),size,kernel,stride,padding)))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(local(position,stride)),Nat.div(position,U32.to_nat(stride)),local_exact(position,size,stride,limits,positive,inside,size_room)) : {Nat.is_lt(_,U32.to_nat(Geometry.axis_count(Kernel.axis(first(position,stride),size,kernel,stride,padding)))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Geometry.axis_count(Kernel.axis(first(position,stride),size,kernel,stride,padding))),1n+Nat.div(Geometry.tail(size,first(position,stride)),U32.to_nat(stride)),
    Geometry.count_exact(first(position,stride),size,kernel,stride,padding,Machine.coordinate(limits),Limits.coordinate_fits(limits),PhaseBounds.dimension_room(limits,U32.to_nat(size),size_room),stride_room,positive,first_inside(position,size,stride,limits,positive,inside,size_room))) : {Nat.is_lt(Nat.div(position,U32.to_nat(stride)),_) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(first(position,stride)),Nat.mod(position,U32.to_nat(stride)),first_exact(position,size,stride,limits,positive,inside,size_room)) : {Nat.is_lt(Nat.div(position,U32.to_nat(stride)),1n+Nat.div(Nat.sub(Nat.sub(U32.to_nat(size),1n),_),U32.to_nat(stride))) == True{} : Bool}
  natural_local_inside(position,U32.to_nat(size),U32.to_nat(stride),positive,inside)

def pixel_exact(+position: Nat,+size: U32,+kernel: U32,+stride: U32,+padding: U32,+limits: Machine.Limits,+positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+inside: {Nat.is_lt(position,U32.to_nat(size)) == True{} : Bool},
  +size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},+stride_room: {Nat.is_le(Nat.double(U32.to_nat(stride)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}) ->
  {U32.to_nat((first(position,stride) + local(position,stride) * stride : U32)) == position : Nat}:
  +q = Nat.div(position,U32.to_nat(stride))
  +r = Nat.mod(position,U32.to_nat(stride))
  Equal.trans(Nat,U32.to_nat((first(position,stride) + local(position,stride) * stride : U32)),Nat.add(U32.to_nat(first(position,stride)),Nat.mul(U32.to_nat(local(position,stride)),U32.to_nat(stride))),position,
    Geometry.position_exact(first(position,stride),size,kernel,stride,padding,local(position,stride),Machine.coordinate(limits),Limits.coordinate_fits(limits),PhaseBounds.dimension_room(limits,U32.to_nat(size),size_room),stride_room,positive,
      first_inside(position,size,stride,limits,positive,inside,size_room),local_axis_inside(position,size,kernel,stride,padding,limits,positive,inside,size_room,stride_room)),
    %Equal.sym(Nat,U32.to_nat(first(position,stride)),r,first_exact(position,size,stride,limits,positive,inside,size_room)) : {Nat.add(_,Nat.mul(U32.to_nat(local(position,stride)),U32.to_nat(stride))) == position : Nat}
    %Equal.sym(Nat,U32.to_nat(local(position,stride)),q,local_exact(position,size,stride,limits,positive,inside,size_room)) : {Nat.add(r,Nat.mul(_,U32.to_nat(stride))) == position : Nat}
    Equal.trans(Nat,Nat.add(r,Nat.mul(q,U32.to_nat(stride))),Nat.add(Nat.mul(q,U32.to_nat(stride)),r),position,Order.commutative(r,Nat.mul(q,U32.to_nat(stride))),Equal.sym(Nat,position,Nat.add(Nat.mul(q,U32.to_nat(stride)),r),Division.quotient_exact(position,U32.to_nat(stride),positive))))

type Axes<-geometry: Kernel.Geometry,-point: Domain.Point> is Data:
  Axes{first_y_exact: {U32.to_nat(first(Domain.row(point),Domain.stride_word(geometry))) == Nat.mod(Domain.row(point),Domain.stride(geometry)) : Nat},
    first_x_exact: {U32.to_nat(first(Domain.column(point),Domain.stride_word(geometry))) == Nat.mod(Domain.column(point),Domain.stride(geometry)) : Nat},
    first_y_loop: {Nat.is_lt(U32.to_nat(first(Domain.row(point),Domain.stride_word(geometry))),U32.to_nat(U32.min(Domain.stride_word(geometry),Domain.height_word(geometry)))) == True{} : Bool},
    first_x_loop: {Nat.is_lt(U32.to_nat(first(Domain.column(point),Domain.stride_word(geometry))),U32.to_nat(U32.min(Domain.stride_word(geometry),Domain.width_word(geometry)))) == True{} : Bool},
    row_exact: {U32.to_nat(local(Domain.row(point),Domain.stride_word(geometry))) == Nat.div(Domain.row(point),Domain.stride(geometry)) : Nat},
    column_exact: {U32.to_nat(local(Domain.column(point),Domain.stride_word(geometry))) == Nat.div(Domain.column(point),Domain.stride(geometry)) : Nat},
    row_inside: {Nat.is_lt(U32.to_nat(local(Domain.row(point),Domain.stride_word(geometry))),U32.to_nat(Geometry.axis_count(Kernel.axis(first(Domain.row(point),Domain.stride_word(geometry)),Domain.height_word(geometry),Domain.kernel_word(geometry),Domain.stride_word(geometry),Domain.padding_word(geometry))))) == True{} : Bool},
    column_inside: {Nat.is_lt(U32.to_nat(local(Domain.column(point),Domain.stride_word(geometry))),U32.to_nat(Geometry.axis_count(Kernel.axis(first(Domain.column(point),Domain.stride_word(geometry)),Domain.width_word(geometry),Domain.kernel_word(geometry),Domain.stride_word(geometry),Domain.padding_word(geometry))))) == True{} : Bool},
    row_pixel: {U32.to_nat((first(Domain.row(point),Domain.stride_word(geometry)) + local(Domain.row(point),Domain.stride_word(geometry)) * Domain.stride_word(geometry) : U32)) == Domain.row(point) : Nat},
    column_pixel: {U32.to_nat((first(Domain.column(point),Domain.stride_word(geometry)) + local(Domain.column(point),Domain.stride_word(geometry)) * Domain.stride_word(geometry) : U32)) == Domain.column(point) : Nat}}

def axes(+call: Domain.Call,+point: Domain.Point,+inside: Domain.Inside<Domain.geometry(call),Domain.batch(call),point>) -> Axes<Domain.geometry(call),point>:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Domain.Point{n,channel,+row,+column} = point
  Domain.Inside{ni,ci,+ri,+xi} = inside
  +stride_positive = Domain.stride_positive(call)
  +height_room = Domain.height_room(call)
  +width_room = Domain.width_room(call)
  +stride_double_room = PhaseBounds.stride_double_room(call)
  Axes{first_exact(row,h,s,limits,stride_positive,ri,height_room),first_exact(column,w,s,limits,stride_positive,xi,width_room),first_loop_inside(row,h,s,limits,stride_positive,ri,height_room),first_loop_inside(column,w,s,limits,stride_positive,xi,width_room),
    local_exact(row,h,s,limits,stride_positive,ri,height_room),local_exact(column,w,s,limits,stride_positive,xi,width_room),local_axis_inside(row,h,k,s,p,limits,stride_positive,ri,height_room,stride_double_room),local_axis_inside(column,w,k,s,p,limits,stride_positive,xi,width_room,stride_double_room),
    pixel_exact(row,h,k,s,p,limits,stride_positive,ri,height_room,stride_double_room),pixel_exact(column,w,k,s,p,limits,stride_positive,xi,width_room,stride_double_room)}

