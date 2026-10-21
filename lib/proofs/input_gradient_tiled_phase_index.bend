# The actual phase walk uses the same bounded row-major phase index as packing.
import Base
import ./input_gradient_tiled_chunk_reads.bend as Chunks
import ./input_gradient_tiled_chunk_geometry.bend as GeometryChunks
import ./nat_extrema.bend as NaturalExtrema
import ./nat_division_order.bend as Division
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_tiled_job_geometry.bend as Jobs
import ./input_gradient_tiled_packing_bounds.bend as PackingBounds
import ./input_gradient_phase_domain.bend as Domain
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./u32_extrema.bend as Extrema
import ./binary_radix.bend as Radix
import ./machine_limit_bounds.bend as Limits
import ./encoded_coordinates.bend as Encoded
import ./storage_shape.bend as Shape

def columns(+phase: Jobs.Phase) -> U32:
  U32.min(Domain.stride_word(Jobs.geometry(phase)),Domain.width_word(Jobs.geometry(phase)))
def rows(+phase: Jobs.Phase) -> U32:
  U32.min(Domain.stride_word(Jobs.geometry(phase)),Domain.height_word(Jobs.geometry(phase)))
def ordinal(+phase: Jobs.Phase) -> Nat:
  Nat.add(Nat.mul(Jobs.phase_y(phase),U32.to_nat(columns(phase))),Jobs.phase_x(phase))
def inside(+phase: Jobs.Phase) -> {Nat.is_lt(ordinal(phase),Kernel.window_jobs(Jobs.geometry(phase))) == True{} : Bool}:
  Jobs.Phase{+call,e,+y,+x,+yi,+xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  RowMajor.radix(y,U32.to_nat(rows(phase)),x,U32.to_nat(columns(phase)),yi,xi)
def count_room(+phase: Jobs.Phase) -> {Nat.is_le(Kernel.window_jobs(Jobs.geometry(phase)),U32.to_nat(Machine.capacity(Jobs.limits(phase)))) == True{} : Bool}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,+h,+w,k,+s,p,gh,gw} = g
  +rn = U32.to_nat(rows(phase))
  +cn = U32.to_nat(columns(phase))
  +hn = U32.to_nat(h)
  +wn = U32.to_nat(w)
  +area = Nat.mul(hn,wn)
  +row_room = Bounds.as_nat_le(rows(phase),h,Extrema.minimum_right(s,h))
  +column_room = Bounds.as_nat_le(columns(phase),w,Extrema.minimum_right(s,w))
  Order.transitive(Kernel.window_jobs(g),area,U32.to_nat(Machine.capacity(l)),
    Order.transitive(Nat.mul(rn,cn),Nat.mul(hn,cn),area,RowMajor.mul_le(rn,hn,cn,row_room),
      %Equal.sym(Nat,Nat.mul(hn,cn),Nat.mul(cn,hn),RowMajor.multiply_commutative(hn,cn)) : {Nat.is_le(_,area) == True{} : Bool}
      %RowMajor.multiply_commutative(wn,hn) : {Nat.is_le(Nat.mul(cn,hn),_) == True{} : Bool} RowMajor.mul_le(cn,wn,hn,column_room)),Domain.plane_room(call))
def ordinal_exact(+phase: Jobs.Phase) -> {U32.to_nat(U32.from_nat(ordinal(phase))) == ordinal(phase) : Nat}:
  Bounds.roundtrip(ordinal(phase),Machine.capacity(Jobs.limits(phase)),Limits.capacity_fits(Jobs.limits(phase)),Bounds.strict_le(ordinal(phase),U32.to_nat(Machine.capacity(Jobs.limits(phase))),Bounds.nat_trans(ordinal(phase),Kernel.window_jobs(Jobs.geometry(phase)),U32.to_nat(Machine.capacity(Jobs.limits(phase))),inside(phase),count_room(phase))))
def columns_double_room(+phase: Jobs.Phase) -> {Nat.is_le(Nat.double(U32.to_nat(columns(phase))),U32.to_nat(Machine.half_word(Jobs.limits(phase)))) == True{} : Bool}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,+w,k,+s,p,gh,gw} = g
  Limits.double_bound(l,U32.to_nat(columns(phase)),Order.transitive(U32.to_nat(columns(phase)),U32.to_nat(w),U32.to_nat(Machine.capacity(l)),Bounds.as_nat_le(columns(phase),w,Extrema.minimum_right(s,w)),Limits.dimension_capacity(l,U32.to_nat(w),Domain.width_room(call))))
def decoded(+phase: Jobs.Phase) ->
  Shape.Both<{(U32.from_nat(ordinal(phase)) / columns(phase) : U32) == U32.from_nat(Jobs.phase_y(phase)) : U32},
    {(U32.from_nat(ordinal(phase)) % columns(phase) : U32) == U32.from_nat(Jobs.phase_x(phase)) : U32}>:
  Jobs.Phase{+call,e,+y,+x,yi,+xi} = phase
  Domain.Call{+g,b,+l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +width = columns(phase)
  +positive = Order.positive_bound(x,U32.to_nat(width),xi)
  +cap = Machine.half_word(l)
  +ok = Limits.half_word_fits(l)
  +double_room = columns_double_room(phase)
  Shape.Both{Encoded.encoding((U32.from_nat(ordinal(phase)) / width : U32),y,Radix.quotient_word(U32.from_nat(ordinal(phase)),width,y,x,U32.to_nat(width),cap,ok,ordinal_exact(phase),{==},positive,xi,double_room)),
    Encoded.encoding((U32.from_nat(ordinal(phase)) % width : U32),x,Radix.remainder_word(U32.from_nat(ordinal(phase)),width,y,x,U32.to_nat(width),cap,ok,ordinal_exact(phase),{==},positive,xi,double_room))}
def phase_matches_from(+phase: Jobs.Phase,
  decoded: Shape.Both<{(U32.from_nat(ordinal(phase)) / columns(phase) : U32) == U32.from_nat(Jobs.phase_y(phase)) : U32},
    {(U32.from_nat(ordinal(phase)) % columns(phase) : U32) == U32.from_nat(Jobs.phase_x(phase)) : U32}>) ->
  {Kernel.phase_of(Jobs.geometry(phase),U32.from_nat(ordinal(phase))) == Jobs.context(phase) : Kernel.Phase}:
  Jobs.Phase{+call,e,+y,+x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Shape.Both{+ye,+xe} = decoded
  %Equal.sym(U32,(U32.from_nat(ordinal(phase)) / columns(phase) : U32),U32.from_nat(y),ye) : {Kernel.phase(g,_,(U32.from_nat(ordinal(phase)) % columns(phase) : U32)) == Jobs.context(phase) : Kernel.Phase}
  %Equal.sym(U32,(U32.from_nat(ordinal(phase)) % columns(phase) : U32),U32.from_nat(x),xe) : {Kernel.phase(g,U32.from_nat(y),_) == Jobs.context(phase) : Kernel.Phase}
  {==}
def phase_matches(+phase: Jobs.Phase) -> {Kernel.phase_of(Jobs.geometry(phase),U32.from_nat(ordinal(phase))) == Jobs.context(phase) : Kernel.Phase}:
  phase_matches_from(phase,decoded(phase))

def packing_bounds(phase: Jobs.Phase) -> PackingBounds.Query:
  PackingBounds.Query{False{},Jobs.call(phase)}
def packing_count_matches(+phase: Jobs.Phase) -> {Kernel.window_jobs(Jobs.geometry(phase)) == PackingBounds.count(packing_bounds(phase)) : Nat}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,+h,+w,k,+s,p,gh,gw} = g
  %Equal.sym(Nat,U32.to_nat(U32.min(s,h)),Nat.min(U32.to_nat(s),U32.to_nat(h)),Extrema.minimum(s,h)) : {Nat.mul(_,U32.to_nat(U32.min(s,w))) == PackingBounds.count(packing_bounds(phase)) : Nat}
  %Equal.sym(Nat,U32.to_nat(U32.min(s,w)),Nat.min(U32.to_nat(s),U32.to_nat(w)),Extrema.minimum(s,w)) : {Nat.mul(Nat.min(U32.to_nat(s),U32.to_nat(h)),_) == PackingBounds.count(packing_bounds(phase)) : Nat}
  {==}
def packing_inside(+phase: Jobs.Phase) -> {Nat.is_lt(ordinal(phase),PackingBounds.count(packing_bounds(phase))) == True{} : Bool}:
  %packing_count_matches(phase) : {Nat.is_lt(ordinal(phase),_) == True{} : Bool} inside(phase)

# Decode any ordinal visited by the actual phase fold.
def columns_positive(+phase: Jobs.Phase) -> {Nat.is_lt(0n,U32.to_nat(columns(phase))) == True{} : Bool}:
  Jobs.Phase{call,e,y,+x,yi,+xi} = phase
  Order.positive_bound(x,U32.to_nat(columns(phase)),xi)
def at_row_inside(+phase: Jobs.Phase,+index: Nat,+inside: {Nat.is_lt(index,Kernel.window_jobs(Jobs.geometry(phase))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(index,U32.to_nat(columns(phase))),U32.to_nat(rows(phase))) == True{} : Bool}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Division.quotient_below(index,U32.to_nat(rows(phase)),U32.to_nat(columns(phase)),columns_positive(phase),inside)
def at(+phase: Jobs.Phase,+index: Nat,+inside: {Nat.is_lt(index,Kernel.window_jobs(Jobs.geometry(phase))) == True{} : Bool}) -> Jobs.Phase:
  +width = U32.to_nat(columns(phase))
  Jobs.Phase{Jobs.call(phase),Jobs.envelope(phase),Nat.div(index,width),Nat.mod(index,width),
    at_row_inside(phase,index,inside),Division.remainder_small(index,width,columns_positive(phase))}
def at_exact(+phase: Jobs.Phase,+index: Nat,+inside: {Nat.is_lt(index,Kernel.window_jobs(Jobs.geometry(phase))) == True{} : Bool}) -> {ordinal(at(phase,index,inside)) == index : Nat}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Equal.sym(Nat,index,Nat.add(Nat.mul(Nat.div(index,U32.to_nat(columns(phase))),U32.to_nat(columns(phase))),Nat.mod(index,U32.to_nat(columns(phase)))),Division.quotient_exact(index,U32.to_nat(columns(phase)),columns_positive(phase)))
def at_matches(+phase: Jobs.Phase,+index: Nat,+inside: {Nat.is_lt(index,Kernel.window_jobs(Jobs.geometry(phase))) == True{} : Bool}) -> {Kernel.phase_of(Jobs.geometry(phase),U32.from_nat(index)) == Jobs.context(at(phase,index,inside)) : Kernel.Phase}:
  Jobs.Phase{+call,e,y,x,yi,xi} = phase
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  %Equal.cong(Nat,U32,U32.from_nat,ordinal(at(phase,index,inside)),index,at_exact(phase,index,inside)) : {Kernel.phase_of(g,_) == Jobs.context(at(phase,index,inside)) : Kernel.Phase}
  phase_matches(at(phase,index,inside))
def zero_row_inside(+grid: GeometryChunks.Grid) ->
  {Nat.is_lt(0n,U32.to_nat(U32.min(Domain.stride_word(GeometryChunks.grid_geometry(grid)),Domain.height_word(GeometryChunks.grid_geometry(grid))))) == True{} : Bool}:
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,+h,w,k,+s,p,gh,gw} = g
  %Equal.sym(Nat,U32.to_nat(U32.min(s,h)),Nat.min(U32.to_nat(s),U32.to_nat(h)),Extrema.minimum(s,h)) : {Nat.is_lt(0n,_) == True{} : Bool}
  NaturalExtrema.minimum_strict_lower(U32.to_nat(s),U32.to_nat(h),0n,Domain.stride_positive(call),Domain.height_positive(call))
def zero_column_inside(+grid: GeometryChunks.Grid) ->
  {Nat.is_lt(0n,U32.to_nat(U32.min(Domain.stride_word(GeometryChunks.grid_geometry(grid)),Domain.width_word(GeometryChunks.grid_geometry(grid))))) == True{} : Bool}:
  GeometryChunks.Grid{+call,e,count,range,rp,rr} = grid
  Domain.Call{+g,+b,+l,facts} = call
  Kernel.Geometry{c,o,h,+w,k,+s,p,gh,gw} = g
  %Equal.sym(Nat,U32.to_nat(U32.min(s,w)),Nat.min(U32.to_nat(s),U32.to_nat(w)),Extrema.minimum(s,w)) : {Nat.is_lt(0n,_) == True{} : Bool}
  NaturalExtrema.minimum_strict_lower(U32.to_nat(s),U32.to_nat(w),0n,Domain.stride_positive(call),Domain.width_positive(call))
def zero(+grid: GeometryChunks.Grid) -> Jobs.Phase:
  Jobs.Phase{GeometryChunks.grid_call(grid),GeometryChunks.grid_envelope(grid),0n,0n,
    zero_row_inside(grid),zero_column_inside(grid)}
def y_inside(+phase: Jobs.Phase) -> {Nat.is_lt(Jobs.phase_y(phase),U32.to_nat(U32.min(Domain.stride_word(Jobs.geometry(phase)),Domain.height_word(Jobs.geometry(phase))))) == True{} : Bool}:
  Jobs.Phase{call,e,y,x,yi,xi} = phase
  yi
def x_inside(+phase: Jobs.Phase) -> {Nat.is_lt(Jobs.phase_x(phase),U32.to_nat(U32.min(Domain.stride_word(Jobs.geometry(phase)),Domain.width_word(Jobs.geometry(phase))))) == True{} : Bool}:
  Jobs.Phase{call,e,y,x,yi,xi} = phase
  xi
def selected(+query: Chunks.Query,+index: Nat,+inside: {Nat.is_lt(index,Kernel.window_jobs(Chunks.geometry(query))) == True{} : Bool}) -> Chunks.Selection:
  +phase = at(zero(Chunks.grid(query)),index,inside)
  Chunks.Selection{query,Jobs.phase_y(phase),Jobs.phase_x(phase),y_inside(phase),x_inside(phase)}
def selected_matches(+query: Chunks.Query,+index: Nat,+inside: {Nat.is_lt(index,Kernel.window_jobs(Chunks.geometry(query))) == True{} : Bool}) -> {Kernel.phase_of(Chunks.geometry(query),U32.from_nat(index)) == Jobs.context(Chunks.phase(selected(query,index,inside))) : Kernel.Phase}:
  at_matches(zero(Chunks.grid(query)),index,inside)
def selected_ordinal(+query: Chunks.Query,+index: Nat,+inside: {Nat.is_lt(index,Kernel.window_jobs(Chunks.geometry(query))) == True{} : Bool}) -> {ordinal(Chunks.phase(selected(query,index,inside))) == index : Nat}:
  at_exact(zero(Chunks.grid(query)),index,inside)
