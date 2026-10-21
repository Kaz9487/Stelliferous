# Native leaf job counts and indices use the shared clipped row-major schedule.
import Base
import ./word_encoding.bend as WordEncoding
import ./nat_division.bend as NatDivision
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_geometry.bend as Geometry
import ./input_gradient_tiled_chunk_geometry.bend as Chunks
import ./input_gradient_tiled_capacity.bend as Capacity
import ./input_gradient_tiled_extents.bend as Extents
import ./input_gradient_tiled_columns.bend as Columns
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_band_geometry.bend as Samples
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./input_gradient_tiled_lookup.bend as Lookup
import ./nat_ceiling.bend as Ceiling
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./row_major_coordinates.bend as Grid
import ./binary_radix.bend as Radix
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_extrema.bend as NativeExtrema
import ./nat_extrema.bend as Extrema
import ./machine_limit_bounds.bend as Limits
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./input_gradient_phase_packing.bend as PhasePacking
import ./nat_division_order.bend as Division
import ./encoded_coordinates.bend as Encoded
import ./storage_shape.bend as Shape
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as Words
import ./nat_division.bend as NaturalDivision
import ./input_gradient_tiled_bands.bend as PureBands
import ./input_gradient_phase_geometry.bend as PhaseGeometry

def units(+geometry: Kernel.Geometry) -> U32:
  ((Domain.channel_word(geometry) + 3) / 4 : U32)
def four_quarter_room(+envelope: Geometry.WordEnvelope) -> {Nat.is_le(U32.to_nat(4),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}:
  Geometry.WordEnvelope{maximum,half,quarter,margin,me,he,qe,re} = envelope
  Bounds.as_nat_le(4,quarter,%Equal.sym(U32,quarter,1073741824,qe) : {U32.is_le(4,_) == True{} : Bool} {==})
def channels_quarter_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {Nat.is_le(Domain.channels(Domain.geometry(call)),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}:
  +limits = Domain.limits(call)
  Order.transitive(Domain.channels(Domain.geometry(call)),U32.to_nat(Machine.dimension(limits)),U32.to_nat(Capacity.quarter(envelope)),Domain.channels_room(call),Capacity.dimension_quarter_room(limits,envelope))
def units_expression(+geometry: Kernel.Geometry) -> {units(geometry) == ((Domain.channel_word(geometry) + 4 - 1) / 4 : U32) : U32}:
  +channels = Domain.channel_word(geometry)
  Equal.cong(U32,U32,value => (value / 4 : U32),(channels + 3 : U32),(channels + 4 - 1 : U32),Equal.sym(U32,(channels + 4 - 1 : U32),(channels + (4 - 1 : U32) : U32),Arithmetic.subtract_from_sum(channels,4,1)))
def units_exact(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {U32.to_nat(units(Domain.geometry(call))) == Ceiling.ceiling(Domain.channels(Domain.geometry(call)),4n) : Nat}:
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +geometry = Domain.geometry(call)
  Equal.trans(Nat,U32.to_nat(units(geometry)),U32.to_nat(((Domain.channel_word(geometry) + 4 - 1) / 4 : U32)),Ceiling.ceiling(Domain.channels(geometry),4n),Equal.cong(U32,Nat,U32.to_nat,units(geometry),((Domain.channel_word(geometry) + 4 - 1) / 4 : U32),units_expression(geometry)),Chunks.ceiling_exact(Domain.channel_word(geometry),4,envelope,channels_quarter_room(call,envelope),four_quarter_room(envelope),{==}))
def units_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {Nat.is_le(U32.to_nat(units(Domain.geometry(call))),Domain.channels(Domain.geometry(call))) == True{} : Bool}:
  +geometry = Domain.geometry(call)
  %Equal.sym(Nat,U32.to_nat(units(geometry)),Ceiling.ceiling(Domain.channels(geometry),4n),units_exact(call,envelope)) : {Nat.is_le(_,Domain.channels(geometry)) == True{} : Bool} Ceiling.bounded(Domain.channels(geometry),4n,{==})
def units_positive_from(+geometry: Kernel.Geometry,+channels_positive: {Nat.is_lt(0n,Domain.channels(geometry)) == True{} : Bool}) ->
  {Nat.is_lt(0n,Ceiling.ceiling(Domain.channels(geometry),4n)) == True{} : Bool}:
  RowMajor.left_factor_positive(Ceiling.ceiling(Domain.channels(geometry),4n),4n,Bounds.nat_trans(0n,Domain.channels(geometry),Nat.mul(Ceiling.ceiling(Domain.channels(geometry),4n),4n),channels_positive,Ceiling.covers(Domain.channels(geometry),4n,{==})))
def units_positive(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {Nat.is_lt(0n,U32.to_nat(units(Domain.geometry(call)))) == True{} : Bool}:
  +geometry = Domain.geometry(call)
  %Equal.sym(Nat,U32.to_nat(units(geometry)),Ceiling.ceiling(Domain.channels(geometry),4n),units_exact(call,envelope)) : {Nat.is_lt(0n,_) == True{} : Bool} units_positive_from(geometry,Domain.channels_positive(call))

# The phase's group count is bounded by the tiled maximum, even in tails.
def phase_groups(+query: Extents.Axis,+index: Nat) -> U32:
  ((Extents.axis_count(query,U32.from_nat(index)) + 7) / 8 : U32)
def phase_count_room(+query: Extents.Axis,+index: Nat,+inside: {Nat.is_lt(index,Extents.count(query)) == True{} : Bool},
  +enabled: {Extents.used(query,U32.from_nat(index)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Extents.axis_count(query,U32.from_nat(index))),U32.to_nat(Machine.capacity(Extents.limits(query)))) == True{} : Bool}:
  Extents.Axis{call,+size,+room,e} = query
  +limits = Domain.limits(call)
  Order.transitive(U32.to_nat(Extents.axis_count(query,U32.from_nat(index))),U32.to_nat(size),U32.to_nat(Machine.capacity(limits)),Extents.count_sample_bound(query,index,inside,enabled),Limits.dimension_capacity(limits,U32.to_nat(size),room))
def phase_groups_exact(+query: Extents.Axis,+index: Nat,+inside: {Nat.is_lt(index,Extents.count(query)) == True{} : Bool},
  +enabled: {Extents.used(query,U32.from_nat(index)) == True{} : Bool}) ->
  {U32.to_nat(phase_groups(query,index)) == Ceiling.ceiling(U32.to_nat(Extents.axis_count(query,U32.from_nat(index))),8n) : Nat}:
  Encoded.eight_blocks_exact(Extents.limits(query),Extents.axis_count(query,U32.from_nat(index)),phase_count_room(query,index,inside,enabled))
def phase_groups_bounded(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+index: Nat,
  +inside: {Nat.is_lt(index,Extents.count(Extents.columns(call,envelope))) == True{} : Bool},
  +enabled: {Extents.used(Extents.columns(call,envelope),U32.from_nat(index)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(phase_groups(Extents.columns(call,envelope),index)),U32.to_nat(Capacity.layout_groups(Kernel.tiling(Domain.geometry(call))))) == True{} : Bool}:
  +geometry = Domain.geometry(call)
  +query = Extents.columns(call,envelope)
  +phase = Extents.axis_count(query,U32.from_nat(index))
  +maximum = Bands.count_fold(Columns.whole(call,envelope))
  %Equal.sym(Nat,U32.to_nat(phase_groups(query,index)),Ceiling.ceiling(U32.to_nat(phase),8n),phase_groups_exact(query,index,inside,enabled)) : {Nat.is_le(_,U32.to_nat(Capacity.layout_groups(Kernel.tiling(geometry)))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Capacity.layout_groups(Kernel.tiling(geometry))),Ceiling.ceiling(U32.to_nat(maximum),8n),Chunks.groups_exact(call,envelope)) : {Nat.is_le(Ceiling.ceiling(U32.to_nat(phase),8n),_) == True{} : Bool}
  Ceiling.monotone(U32.to_nat(phase),U32.to_nat(maximum),8n,{==},Bounds.as_nat_le(phase,maximum,Columns.selected_count(call,envelope,index,inside,enabled)))
def phase_groups_extent_room(+query: Extents.Axis,+index: Nat,+inside: {Nat.is_lt(index,Extents.count(query)) == True{} : Bool},
  +enabled: {Extents.used(query,U32.from_nat(index)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(phase_groups(query,index)),U32.to_nat(Extents.extent(query))) == True{} : Bool}:
  +count = Extents.axis_count(query,U32.from_nat(index))
  %Equal.sym(Nat,U32.to_nat(phase_groups(query,index)),Ceiling.ceiling(U32.to_nat(count),8n),phase_groups_exact(query,index,inside,enabled)) : {Nat.is_le(_,U32.to_nat(Extents.extent(query))) == True{} : Bool}
  Order.transitive(Ceiling.ceiling(U32.to_nat(count),8n),U32.to_nat(count),U32.to_nat(Extents.extent(query)),Ceiling.bounded(U32.to_nat(count),8n,{==}),Extents.count_sample_bound(query,index,inside,enabled))

type Phase is Data:
  Phase{call: Domain.Call,envelope: Geometry.WordEnvelope,y: Nat,x: Nat,
    y_inside: {Nat.is_lt(y,U32.to_nat(U32.min(Domain.stride_word(Domain.geometry(call)),Domain.height_word(Domain.geometry(call))))) == True{} : Bool},
    x_inside: {Nat.is_lt(x,U32.to_nat(U32.min(Domain.stride_word(Domain.geometry(call)),Domain.width_word(Domain.geometry(call))))) == True{} : Bool}}
def call(phase: Phase) -> Domain.Call:
  Phase{call,e,y,x,yi,xi} = phase
  call
def geometry(phase: Phase) -> Kernel.Geometry:
  Domain.geometry(call(phase))
def limits(phase: Phase) -> Machine.Limits:
  Domain.limits(call(phase))
def envelope(phase: Phase) -> Geometry.WordEnvelope:
  Phase{call,e,y,x,yi,xi} = phase
  e
def phase_y(phase: Phase) -> Nat:
  Phase{call,e,y,x,yi,xi} = phase
  y
def phase_x(phase: Phase) -> Nat:
  Phase{call,e,y,x,yi,xi} = phase
  x
def context(+phase: Phase) -> Kernel.Phase:
  Kernel.phase(geometry(phase),U32.from_nat(phase_y(phase)),U32.from_nat(phase_x(phase)))
def columns(phase: Phase) -> Extents.Axis:
  Phase{+call,+e,y,x,yi,xi} = phase
  Extents.columns(call,e)
def rows(phase: Phase) -> Extents.Axis:
  Phase{+call,+e,y,x,yi,xi} = phase
  Extents.rows(call,e)
type Active<-phase: Phase> is Data:
  Active{y: {Extents.used(rows(phase),U32.from_nat(phase_y(phase))) == True{} : Bool},
    x: {Extents.used(columns(phase),U32.from_nat(phase_x(phase))) == True{} : Bool}}
def row_count(+phase: Phase) -> U32:
  Extents.axis_count(rows(phase),U32.from_nat(phase_y(phase)))
def column_count(+phase: Phase) -> U32:
  Extents.axis_count(columns(phase),U32.from_nat(phase_x(phase)))
def native_groups(+phase: Phase) -> U32:
  phase_groups(columns(phase),phase_x(phase))
def row_count_room(+phase: Phase,+active: Active<phase>) -> {Nat.is_le(U32.to_nat(row_count(phase)),Domain.height(geometry(phase))) == True{} : Bool}:
  Phase{call,e,y,x,yi,xi} = phase
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Active{+enabled,ex} = active
  Extents.count_sample_bound(rows(phase),y,yi,enabled)
def column_groups_room(+phase: Phase,+active: Active<phase>) -> {Nat.is_le(U32.to_nat(native_groups(phase)),Domain.width(geometry(phase))) == True{} : Bool}:
  Phase{call,e,y,x,yi,xi} = phase
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Active{ey,+enabled} = active
  phase_groups_extent_room(columns(phase),x,xi,enabled)
def phase_units_room(+phase: Phase) -> {Nat.is_le(U32.to_nat(units(geometry(phase))),Domain.channels(geometry(phase))) == True{} : Bool}:
  Phase{call,e,y,x,yi,xi} = phase
  units_room(call,e)
def phase_units_positive(+phase: Phase) -> {Nat.is_lt(0n,U32.to_nat(units(geometry(phase)))) == True{} : Bool}:
  Phase{call,e,y,x,yi,xi} = phase
  units_positive(call,e)

def group_span(+first: U32,+total: U32,+width: U32) -> U32:
  Bool.pick(U32,U32.is_lt(first,total),U32.min(width,(total - first : U32)),0)
def group_span_room_case(selected: Bool,+first: U32,+total: U32,+width: U32,+compared: {U32.is_lt(first,total) == selected : Bool}) ->
  {Nat.is_le(U32.to_nat(Bool.pick(U32,selected,U32.min(width,(total - first : U32)),0)),U32.to_nat(total)) == True{} : Bool}:
  match selected:
    case False{}: Order.zero_le(U32.to_nat(total))
    case True{}:
      +difference = (total - first : U32)
      +exact = FullRange.subtract(total,first,total,Bounds.strict_le(U32.to_nat(first),U32.to_nat(total),Bounds.as_nat_lt(first,total,compared)),Order.sub_le(U32.to_nat(total),U32.to_nat(first)))
      Order.transitive(U32.to_nat(U32.min(width,difference)),U32.to_nat(difference),U32.to_nat(total),Bounds.as_nat_le(U32.min(width,difference),difference,NativeExtrema.minimum_right(width,difference)),
        %Equal.sym(Nat,U32.to_nat(difference),Nat.sub(U32.to_nat(total),U32.to_nat(first)),exact) : {Nat.is_le(_,U32.to_nat(total)) == True{} : Bool} Order.sub_le(U32.to_nat(total),U32.to_nat(first)))
def group_span_room(+first: U32,+total: U32,+width: U32) -> {Nat.is_le(U32.to_nat(group_span(first,total,width)),U32.to_nat(total)) == True{} : Bool}:
  group_span_room_case(U32.is_lt(first,total),first,total,width,{==})
def group_span_selected(+first: U32,+total: U32,+width: U32,+position: Nat,
  +lower: {Nat.is_le(U32.to_nat(first),position) == True{} : Bool},+upper: {Nat.is_lt(position,U32.to_nat(total)) == True{} : Bool},
  +local_inside: {Nat.is_lt(Nat.sub(position,U32.to_nat(first)),U32.to_nat(width)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.sub(position,U32.to_nat(first)),U32.to_nat(group_span(first,total,width))) == True{} : Bool}:
  +low_high = Order.le_lt_trans(U32.to_nat(first),position,U32.to_nat(total),lower,upper)
  +selected = {%Equal.sym(Bool,U32.is_lt(first,total),Nat.is_lt(U32.to_nat(first),U32.to_nat(total)),Comparison.lt(first,total)) : {_ == True{} : Bool} low_high : {U32.is_lt(first,total) == True{} : Bool}}
  +difference = (total - first : U32)
  +exact = FullRange.subtract(total,first,total,Bounds.strict_le(U32.to_nat(first),U32.to_nat(total),low_high),Order.sub_le(U32.to_nat(total),U32.to_nat(first)))
  +local = Nat.sub(position,U32.to_nat(first))
  %Equal.sym(Bool,U32.is_lt(first,total),True{},selected) : {Nat.is_lt(local,U32.to_nat(Bool.pick(U32,_,U32.min(width,difference),0))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.min(width,difference)),Nat.min(U32.to_nat(width),U32.to_nat(difference)),NativeExtrema.minimum(width,difference)) : {Nat.is_lt(local,_) == True{} : Bool}
  Extrema.minimum_strict_lower(U32.to_nat(width),U32.to_nat(difference),local,local_inside,
    %Equal.sym(Nat,U32.to_nat(difference),Nat.sub(U32.to_nat(total),U32.to_nat(first)),exact) : {Nat.is_lt(local,_) == True{} : Bool}
    %Equal.sym(Bool,Nat.is_lt(local,Nat.sub(U32.to_nat(total),U32.to_nat(first))),Nat.is_lt(Nat.add(U32.to_nat(first),local),U32.to_nat(total)),Order.subtraction_lt(U32.to_nat(total),U32.to_nat(first),local)) : {_ == True{} : Bool}
    %Equal.sym(Nat,Nat.add(U32.to_nat(first),local),position,Order.restore_shift(U32.to_nat(first),position,lower)) : {Nat.is_lt(_,U32.to_nat(total)) == True{} : Bool} upper)

type Work is Data:
  Work{phase: Phase,chunk: Kernel.Chunk}
def phase(work: Work) -> Phase:
  Work{phase,chunk} = work
  phase
def chunk(work: Work) -> Kernel.Chunk:
  Work{phase,chunk} = work
  chunk
def chunk_row(chunk: Kernel.Chunk) -> U32:
  Kernel.Chunk{leaf,row,rows,low,group} = chunk
  row
def chunk_rows(chunk: Kernel.Chunk) -> U32:
  Kernel.Chunk{leaf,row,rows,low,group} = chunk
  rows
def chunk_group(chunk: Kernel.Chunk) -> U32:
  Kernel.Chunk{leaf,row,rows,low,group} = chunk
  group
def row_axis(+phase: Phase) -> Kernel.Axis:
  Extents.native_axis(rows(phase),U32.from_nat(phase_y(phase)))
def indices(+work: Work) -> Samples.Indices:
  +phase = phase(work)
  +first = chunk_row(chunk(work))
  Samples.indices(row_axis(phase),Domain.stride_word(geometry(phase)),first,(first + chunk_rows(chunk(work)) : U32))
def start(+work: Work) -> U32:
  Samples.low(indices(work))
def end(+work: Work) -> U32:
  Samples.high(indices(work))
def lines(+work: Work) -> U32:
  Bands.difference(start(work),end(work))
def groups(+work: Work) -> U32:
  group_span(chunk_group(chunk(work)),native_groups(phase(work)),Lookup.layout_groups(Lookup.chunk_layout(chunk(work))))
def native_units(+work: Work) -> U32:
  units(geometry(phase(work)))
def product_count(+work: Work) -> U32:
  (lines(work) * groups(work) * native_units(work) : U32)
def count(+work: Work) -> Nat:
  U32.to_nat(product_count(work))
def line_span_room_case(selected: Bool,+first: U32,+last: U32,+compared: {U32.is_lt(first,last) == selected : Bool}) ->
  {Nat.is_le(U32.to_nat(Bool.pick(U32,selected,(last - first : U32),0)),U32.to_nat(last)) == True{} : Bool}:
  match selected:
    case False{}: Order.zero_le(U32.to_nat(last))
    case True{}:
      %Equal.sym(Nat,U32.to_nat((last - first : U32)),Nat.sub(U32.to_nat(last),U32.to_nat(first)),FullRange.subtract(last,first,last,Bounds.strict_le(U32.to_nat(first),U32.to_nat(last),Bounds.as_nat_lt(first,last,compared)),Order.sub_le(U32.to_nat(last),U32.to_nat(first)))) : {Nat.is_le(_,U32.to_nat(last)) == True{} : Bool} Order.sub_le(U32.to_nat(last),U32.to_nat(first))
def line_span_room(+first: U32,+last: U32) -> {Nat.is_le(U32.to_nat(Bands.difference(first,last)),U32.to_nat(last)) == True{} : Bool}:
  line_span_room_case(U32.is_lt(first,last),first,last,{==})
def end_room(+work: Work) -> {Nat.is_le(U32.to_nat(end(work)),U32.to_nat(row_count(phase(work)))) == True{} : Bool}:
  Work{+phase,+chunk} = work
  Phase{call,e,y,x,yi,xi} = phase
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{leaf,+row,+count,low,group} = chunk
  +last = Samples.clipped(U32.from_nat(y),(row + count : U32),s)
  Bounds.as_nat_le(U32.min(row_count(phase),last),row_count(phase),NativeExtrema.minimum_left(row_count(phase),last))
def lines_room(+work: Work,+active: Active<phase(work)>) -> {Nat.is_le(U32.to_nat(lines(work)),Domain.height(geometry(phase(work)))) == True{} : Bool}:
  Order.transitive(U32.to_nat(lines(work)),U32.to_nat(end(work)),Domain.height(geometry(phase(work))),line_span_room(start(work),end(work)),Order.transitive(U32.to_nat(end(work)),U32.to_nat(row_count(phase(work))),Domain.height(geometry(phase(work))),end_room(work),row_count_room(phase(work),active)))
def groups_room(+work: Work,+active: Active<phase(work)>) -> {Nat.is_le(U32.to_nat(groups(work)),Domain.width(geometry(phase(work)))) == True{} : Bool}:
  Order.transitive(U32.to_nat(groups(work)),U32.to_nat(native_groups(phase(work))),Domain.width(geometry(phase(work))),group_span_room(chunk_group(chunk(work)),native_groups(phase(work)),Lookup.layout_groups(Lookup.chunk_layout(chunk(work)))),column_groups_room(phase(work),active))
def image_room(+phase: Phase) -> {Nat.is_le(Domain.image_size(geometry(phase)),U32.to_nat(Machine.capacity(limits(phase)))) == True{} : Bool}:
  Phase{call,e,y,x,yi,xi} = phase
  Domain.image_room(call)
def natural_count(+work: Work) -> Nat:
  Nat.mul(Nat.mul(U32.to_nat(lines(work)),U32.to_nat(groups(work))),U32.to_nat(native_units(work)))
def natural_count_room(+work: Work,+active: Active<phase(work)>) ->
  {Nat.is_le(natural_count(work),U32.to_nat(Machine.capacity(limits(phase(work))))) == True{} : Bool}:
  +phase = phase(work)
  +g = geometry(phase)
  +ln = U32.to_nat(lines(work))
  +gn = U32.to_nat(groups(work))
  +un = U32.to_nat(native_units(work))
  +h = Domain.height(g)
  +w = Domain.width(g)
  +c = Domain.channels(g)
  +plane = Nat.mul(h,w)
  +row_group_room = Order.transitive(Nat.mul(ln,gn),Nat.mul(h,gn),plane,RowMajor.mul_le(ln,h,gn,lines_room(work,active)),
    %Equal.sym(Nat,Nat.mul(h,gn),Nat.mul(gn,h),RowMajor.multiply_commutative(h,gn)) : {Nat.is_le(_,plane) == True{} : Bool}
    %RowMajor.multiply_commutative(w,h) : {Nat.is_le(Nat.mul(gn,h),_) == True{} : Bool} RowMajor.mul_le(gn,w,h,groups_room(work,active)))
  Order.transitive(natural_count(work),Domain.image_size(g),U32.to_nat(Machine.capacity(limits(phase))),
    Order.transitive(natural_count(work),Nat.mul(plane,un),Domain.image_size(g),RowMajor.mul_le(Nat.mul(ln,gn),plane,un,row_group_room),
      %Equal.sym(Nat,Nat.mul(plane,un),Nat.mul(un,plane),RowMajor.multiply_commutative(plane,un)) : {Nat.is_le(_,Domain.image_size(g)) == True{} : Bool} RowMajor.mul_le(un,c,plane,phase_units_room(phase))),image_room(phase))
def count_exact(+work: Work,+active: Active<phase(work)>) -> {count(work) == natural_count(work) : Nat}:
  Encoded.multiply_three(lines(work),groups(work),native_units(work),Machine.capacity(limits(phase(work))),Limits.capacity_fits(limits(phase(work))),phase_units_positive(phase(work)),natural_count_room(work,active))

def count_product_case(selected: Bool,+first: U32,+last: U32,+groups: U32,+units: U32) ->
  {U32.to_nat(Bool.pick(U32,selected,((last - first) * groups * units : U32),0)) == U32.to_nat((Bool.pick(U32,selected,(last - first : U32),0) * groups * units : U32)) : Nat}:
  match selected:
    case True{}: {==}
    case False{}:
      Equal.cong(U32,Nat,U32.to_nat,0,(0 * groups * units : U32),Equal.sym(U32,(0 * groups * units : U32),0,
        Equal.trans(U32,(0 * groups * units : U32),(0 * units : U32),0,Equal.cong(U32,U32,value => (value * units : U32),(0 * groups : U32),0,Arithmetic.multiply_zero_left(groups)),Arithmetic.multiply_zero_left(units))))
def count_product(+first: U32,+last: U32,+groups: U32,+units: U32) ->
  {U32.to_nat(Bool.pick(U32,U32.is_lt(first,last),((last - first) * groups * units : U32),0)) == U32.to_nat((Bands.difference(first,last) * groups * units : U32)) : Nat}:
  count_product_case(U32.is_lt(first,last),first,last,groups,units)
def work_count_matches(+work: Work,+wb: U32,+ob: U32) ->
  {Execution.work_count(Execution.work(context(phase(work)),chunk(work),wb,ob)) == count(work) : Nat}:
  Work{+phase,+chunk} = work
  Phase{call,e,y,x,yi,xi} = phase
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{geometry,+layout},first,count,target,plane},row,rows,low,group} = chunk
  Kernel.Tiling{layout_groups,shifts,outputs,column_low,kernel,height,width,cg,cr} = layout
  count_product(start(work),end(work),groups(work),native_units(work))
def native_count_exact(+work: Work,+active: Active<phase(work)>,+wb: U32,+ob: U32) ->
  {Execution.work_count(Execution.work(context(phase(work)),chunk(work),wb,ob)) == natural_count(work) : Nat}:
  Equal.trans(Nat,Execution.work_count(Execution.work(context(phase(work)),chunk(work),wb,ob)),count(work),natural_count(work),work_count_matches(work,wb,ob),count_exact(work,active))

# Units are innermost, then group, then phase line, as in the actual job loop.
def job_position(+work: Work,+line: Nat,+group: Nat,+unit: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(line,U32.to_nat(groups(work))),group),U32.to_nat(native_units(work))),unit)
def job_index(+work: Work,+line: Nat,+group: Nat,+unit: Nat) -> U32:
  U32.from_nat(job_position(work,line,group,unit))
type Inside<-work: Work,-line: Nat,-group: Nat,-unit: Nat> is Data:
  Inside{line: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool},group: {Nat.is_lt(group,U32.to_nat(groups(work))) == True{} : Bool},unit: {Nat.is_lt(unit,U32.to_nat(native_units(work))) == True{} : Bool}}
def unit_inside(+work: Work,+line: Nat,+group: Nat,+unit: Nat,inside: Inside<work,line,group,unit>) -> {Nat.is_lt(unit,U32.to_nat(native_units(work))) == True{} : Bool}:
  Inside{line_inside,group_inside,unit_inside} = inside
  unit_inside
def job_position_inside(+work: Work,+line: Nat,+group: Nat,+unit: Nat,+inside: Inside<work,line,group,unit>) -> {Nat.is_lt(job_position(work,line,group,unit),natural_count(work)) == True{} : Bool}:
  Inside{+li,+gi,+ui} = inside
  RowMajor.radix(Nat.add(Nat.mul(line,U32.to_nat(groups(work))),group),Nat.mul(U32.to_nat(lines(work)),U32.to_nat(groups(work))),unit,U32.to_nat(native_units(work)),RowMajor.radix(line,U32.to_nat(lines(work)),group,U32.to_nat(groups(work)),li,gi),ui)
def job_index_exact(+work: Work,+active: Active<phase(work)>,+line: Nat,+group: Nat,+unit: Nat,+inside: Inside<work,line,group,unit>) ->
  {U32.to_nat(job_index(work,line,group,unit)) == job_position(work,line,group,unit) : Nat}:
  +capacity = Machine.capacity(limits(phase(work)))
  Bounds.roundtrip(job_position(work,line,group,unit),capacity,Limits.capacity_fits(limits(phase(work))),Bounds.strict_le(job_position(work,line,group,unit),U32.to_nat(capacity),Bounds.nat_trans(job_position(work,line,group,unit),natural_count(work),U32.to_nat(capacity),job_position_inside(work,line,group,unit,inside),natural_count_room(work,active))))
def units_capacity_room(+phase: Phase) -> {Nat.is_le(U32.to_nat(units(geometry(phase))),U32.to_nat(Machine.capacity(limits(phase)))) == True{} : Bool}:
  Phase{call,e,y,x,yi,xi} = phase
  +geometry = Domain.geometry(call)
  Order.transitive(U32.to_nat(units(geometry)),Domain.channels(geometry),U32.to_nat(Machine.capacity(Domain.limits(call))),units_room(call,e),Domain.channels_capacity_room(call))
def units_double_room(+work: Work) -> {Nat.is_le(Nat.double(U32.to_nat(native_units(work))),U32.to_nat(Machine.half_word(limits(phase(work))))) == True{} : Bool}:
  Limits.double_bound(limits(phase(work)),U32.to_nat(native_units(work)),units_capacity_room(phase(work)))
def groups_capacity_room(+work: Work,+active: Active<phase(work)>) -> {Nat.is_le(U32.to_nat(groups(work)),U32.to_nat(Machine.capacity(limits(phase(work))))) == True{} : Bool}:
  Work{+phase,chunk} = work
  Phase{call,e,y,x,yi,xi} = phase
  +width = Domain.width(Domain.geometry(call))
  +limits = Domain.limits(call)
  Order.transitive(U32.to_nat(groups(work)),width,U32.to_nat(Machine.capacity(limits)),groups_room(work,active),Limits.dimension_capacity(limits,width,Domain.width_room(call)))
def groups_double_room(+work: Work,+active: Active<phase(work)>) -> {Nat.is_le(Nat.double(U32.to_nat(groups(work))),U32.to_nat(Machine.half_word(limits(phase(work))))) == True{} : Bool}:
  Limits.double_bound(limits(phase(work)),U32.to_nat(groups(work)),groups_capacity_room(work,active))
def unit_decode(+work: Work,+active: Active<phase(work)>,+line: Nat,+group: Nat,+unit: Nat,+inside: Inside<work,line,group,unit>) ->
  {U32.to_nat((job_index(work,line,group,unit) % native_units(work) : U32)) == unit : Nat}:
  Inside{li,gi,+ui} = inside
  Radix.remainder_word(job_index(work,line,group,unit),native_units(work),Nat.add(Nat.mul(line,U32.to_nat(groups(work))),group),unit,U32.to_nat(native_units(work)),Machine.half_word(limits(phase(work))),Limits.half_word_fits(limits(phase(work))),job_index_exact(work,active,line,group,unit,inside),{==},phase_units_positive(phase(work)),ui,units_double_room(work))
def line_group_decode(+work: Work,+active: Active<phase(work)>,+line: Nat,+group: Nat,+unit: Nat,+inside: Inside<work,line,group,unit>) ->
  {U32.to_nat((job_index(work,line,group,unit) / native_units(work) : U32)) == Nat.add(Nat.mul(line,U32.to_nat(groups(work))),group) : Nat}:
  Inside{li,gi,+ui} = inside
  Radix.quotient_word(job_index(work,line,group,unit),native_units(work),Nat.add(Nat.mul(line,U32.to_nat(groups(work))),group),unit,U32.to_nat(native_units(work)),Machine.half_word(limits(phase(work))),Limits.half_word_fits(limits(phase(work))),job_index_exact(work,active,line,group,unit,inside),{==},phase_units_positive(phase(work)),ui,units_double_room(work))
def group_decode(+work: Work,+active: Active<phase(work)>,+line: Nat,+group: Nat,+unit: Nat,+inside: Inside<work,line,group,unit>) ->
  {U32.to_nat((job_index(work,line,group,unit) / native_units(work) % groups(work) : U32)) == group : Nat}:
  Inside{li,+gi,ui} = inside
  Radix.remainder_word((job_index(work,line,group,unit) / native_units(work) : U32),groups(work),line,group,U32.to_nat(groups(work)),Machine.half_word(limits(phase(work))),Limits.half_word_fits(limits(phase(work))),line_group_decode(work,active,line,group,unit,inside),{==},Order.positive_bound(group,U32.to_nat(groups(work)),gi),gi,groups_double_room(work,active))
def line_decode(+work: Work,+active: Active<phase(work)>,+line: Nat,+group: Nat,+unit: Nat,+inside: Inside<work,line,group,unit>) ->
  {U32.to_nat((job_index(work,line,group,unit) / native_units(work) / groups(work) : U32)) == line : Nat}:
  Inside{li,+gi,ui} = inside
  Radix.quotient_word((job_index(work,line,group,unit) / native_units(work) : U32),groups(work),line,group,U32.to_nat(groups(work)),Machine.half_word(limits(phase(work))),Limits.half_word_fits(limits(phase(work))),line_group_decode(work,active,line,group,unit,inside),{==},Order.positive_bound(group,U32.to_nat(groups(work)),gi),gi,groups_double_room(work,active))

# Every ordinal in the actual job count has bounded shared coordinates.
def job_coordinates(+work: Work,+ordinal: Nat) -> Grid.Values:
  Grid.values(U32.to_nat(groups(work)),U32.to_nat(native_units(work)),ordinal)
def inside_coordinates(+work: Work,+value: Grid.Values,
  bounds: Grid.Bounded<U32.to_nat(lines(work)),U32.to_nat(groups(work)),U32.to_nat(native_units(work)),value>) ->
  Inside<work,Grid.channel(value),Grid.row(value),Grid.column(value)>:
  Grid.Values{line,group,unit} = value
  Grid.Bounded{li,gi,ui} = bounds
  Inside{li,gi,ui}
def job_coordinates_inside(+work: Work,+ordinal: Nat,+inside: {Nat.is_lt(ordinal,natural_count(work)) == True{} : Bool}) ->
  Inside<work,Grid.channel(job_coordinates(work,ordinal)),Grid.row(job_coordinates(work,ordinal)),Grid.column(job_coordinates(work,ordinal))>:
  +ln = U32.to_nat(lines(work))
  +gn = U32.to_nat(groups(work))
  +un = U32.to_nat(native_units(work))
  inside_coordinates(work,job_coordinates(work,ordinal),Grid.values_inside(ln,gn,un,ordinal,phase_units_positive(phase(work)),
    %RowMajor.assoc(ln,gn,un) : {Nat.is_lt(ordinal,_) == True{} : Bool} inside))
def job_coordinates_exact(+work: Work,+ordinal: Nat,+inside: {Nat.is_lt(ordinal,natural_count(work)) == True{} : Bool}) ->
  {ordinal == job_position(work,Grid.channel(job_coordinates(work,ordinal)),Grid.row(job_coordinates(work,ordinal)),Grid.column(job_coordinates(work,ordinal))) : Nat}:
  +ln = U32.to_nat(lines(work))
  +gn = U32.to_nat(groups(work))
  +un = U32.to_nat(native_units(work))
  +value = job_coordinates(work,ordinal)
  Equal.trans(Nat,ordinal,RowMajor.index(Grid.channel(value),gn,un,Grid.row(value),Grid.column(value)),job_position(work,Grid.channel(value),Grid.row(value),Grid.column(value)),Grid.values_exact(ln,gn,un,ordinal,phase_units_positive(phase(work)),%RowMajor.assoc(ln,gn,un) : {Nat.is_lt(ordinal,_) == True{} : Bool} inside),Equal.sym(Nat,job_position(work,Grid.channel(value),Grid.row(value),Grid.column(value)),RowMajor.index(Grid.channel(value),gn,un,Grid.row(value),Grid.column(value)),RowMajor.mixed(Grid.channel(value),gn,un,Grid.row(value),Grid.column(value))))
def encoded_coordinates_exact(+work: Work,+ordinal: Nat,+inside: {Nat.is_lt(ordinal,natural_count(work)) == True{} : Bool}) ->
  {job_index(work,Grid.channel(job_coordinates(work,ordinal)),Grid.row(job_coordinates(work,ordinal)),Grid.column(job_coordinates(work,ordinal))) == U32.from_nat(ordinal) : U32}:
  Equal.cong(Nat,U32,U32.from_nat,job_position(work,Grid.channel(job_coordinates(work,ordinal)),Grid.row(job_coordinates(work,ordinal)),Grid.column(job_coordinates(work,ordinal))),ordinal,Equal.sym(Nat,ordinal,job_position(work,Grid.channel(job_coordinates(work,ordinal)),Grid.row(job_coordinates(work,ordinal)),Grid.column(job_coordinates(work,ordinal))),job_coordinates_exact(work,ordinal,inside)))

def plan_units(plan: Kernel.Plan) -> U32:
  Kernel.Plan{units,groups,inner,source,row_step,group_step,target,target_row,step,plane,channels,columns,wb,ob} = plan
  units
def active_from_inner(+phase: Phase,
  +positive: {Nat.is_lt(0n,U32.to_nat(Kernel.phase_inner(context(phase)))) == True{} : Bool}) -> Active<phase>:
  Phase{call,e,y,x,yi,xi} = phase
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +ctx = context(phase)
  +yr = Extents.axis_taps(rows(phase),U32.from_nat(y))
  +xr = Extents.axis_taps(columns(phase),U32.from_nat(x))
  +inner_exact = PhaseBounds.phase_inner_exact(call,U32.from_nat(y),U32.from_nat(x))
  +product_positive = {%inner_exact : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(yr),U32.to_nat(xr)))) == True{} : Bool}}
  +area_positive = Division.right_factor_positive(U32.to_nat(o),Nat.mul(U32.to_nat(yr),U32.to_nat(xr)),product_positive)
  Active{
    %Equal.sym(Bool,U32.is_lt(0,yr),Nat.is_lt(0n,U32.to_nat(yr)),Comparison.lt(0,yr)) : {_ == True{} : Bool} RowMajor.left_factor_positive(U32.to_nat(yr),U32.to_nat(xr),area_positive),
    %Equal.sym(Bool,U32.is_lt(0,xr),Nat.is_lt(0n,U32.to_nat(xr)),Comparison.lt(0,xr)) : {_ == True{} : Bool} Division.right_factor_positive(U32.to_nat(yr),U32.to_nat(xr),area_positive)}
def active_phase(+phase: Phase,+selected: {U32.is_eq(Kernel.phase_inner(context(phase)),0) == False{} : Bool}) -> Active<phase>:
  active_from_inner(phase,PhaseBounds.phase_inner_positive(context(phase),selected))

def row_position(+work: Work,+line: Nat) -> Nat:
  Nat.add(U32.to_nat(start(work)),line)
def row_position_bounds_case(selected: Bool,+work: Work,+line: Nat,
  +same: {U32.is_lt(start(work),end(work)) == selected : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat(start(work)),row_position(work,line)) == True{} : Bool},
    {Nat.is_lt(row_position(work,line),U32.to_nat(end(work))) == True{} : Bool}>:
  match selected:
    case False{}:
      Empty.absurd(Shape.Both<{Nat.is_le(U32.to_nat(start(work)),row_position(work,line)) == True{} : Bool},
        {Nat.is_lt(row_position(work,line),U32.to_nat(end(work))) == True{} : Bool}>,Words.false_true(
        %NaturalDivision.lt_zero(line) : {_ == True{} : Bool}
        %same : {Nat.is_lt(line,U32.to_nat(Bool.pick(U32,_,(end(work) - start(work) : U32),0))) == True{} : Bool} inside))
    case True{}:
      +first = U32.to_nat(start(work))
      +last = U32.to_nat(end(work))
      +ordered = Bounds.strict_le(first,last,Bounds.as_nat_lt(start(work),end(work),same))
      +exact = FullRange.subtract(end(work),start(work),end(work),ordered,Order.sub_le(last,first))
      +local = {%same : {Nat.is_lt(line,U32.to_nat(Bool.pick(U32,_,(end(work) - start(work) : U32),0))) == True{} : Bool} inside : {Nat.is_lt(line,U32.to_nat((end(work) - start(work) : U32))) == True{} : Bool}}
      Shape.Both{Order.le_add(first,line),
        %Order.restore_shift(first,last,ordered) : {Nat.is_lt(row_position(work,line),_) == True{} : Bool}
        Order.offset_lt(first,line,Nat.sub(last,first),%exact : {Nat.is_lt(line,_) == True{} : Bool} local)}
def row_position_bounds(+work: Work,+line: Nat,+inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat(start(work)),row_position(work,line)) == True{} : Bool},
    {Nat.is_lt(row_position(work,line),U32.to_nat(end(work))) == True{} : Bool}>:
  row_position_bounds_case(U32.is_lt(start(work),end(work)),work,line,{==},inside)
def row_position_upper(+work: Work,+line: Nat,
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(start(work)),row_position(work,line)) == True{} : Bool},
    {Nat.is_lt(row_position(work,line),U32.to_nat(end(work))) == True{} : Bool}>) ->
  {Nat.is_lt(row_position(work,line),U32.to_nat(end(work))) == True{} : Bool}:
  Shape.Both{lo,hi} = bounds
  hi
def row_position_inside(+work: Work,+line: Nat,+inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  {Nat.is_lt(row_position(work,line),U32.to_nat(row_count(phase(work)))) == True{} : Bool}:
  Bounds.nat_trans(row_position(work,line),U32.to_nat(end(work)),U32.to_nat(row_count(phase(work))),row_position_upper(work,line,row_position_bounds(work,line,inside)),end_room(work))
def row_native(+work: Work,+line: Nat) -> U32:
  (start(work) + U32.from_nat(line) : U32)
def phase_height_room(+phase: Phase) -> {Nat.is_le(Domain.height(geometry(phase)),U32.to_nat(Machine.dimension(limits(phase)))) == True{} : Bool}:
  Phase{call,e,y,x,yi,xi} = phase
  Domain.height_room(call)
def row_native_exact(+work: Work,+active: Active<phase(work)>,+line: Nat,+inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  {U32.to_nat(row_native(work,line)) == row_position(work,line) : Nat}:
  +p = phase(work)
  +g = geometry(p)
  +l = limits(p)
  +coordinate = Machine.coordinate(l)
  +height = Domain.height(g)
  +row_room = Order.transitive(row_position(work,line),height,U32.to_nat(coordinate),
    Bounds.strict_le(row_position(work,line),height,Bounds.nat_trans(row_position(work,line),U32.to_nat(row_count(p)),height,row_position_inside(work,line,inside),row_count_room(p,active))),
    Order.transitive(height,U32.to_nat(Machine.dimension(l)),U32.to_nat(coordinate),phase_height_room(p),Bounds.as_nat_le(Machine.dimension(l),coordinate,Limits.dimension_within_coordinate(l))))
  +line_room = Order.transitive(line,row_position(work,line),U32.to_nat(coordinate),%Order.commutative(line,U32.to_nat(start(work))) : {Nat.is_le(line,_) == True{} : Bool} Order.le_add(line,U32.to_nat(start(work))),row_room)
  +line_exact = Bounds.roundtrip(line,coordinate,Limits.coordinate_fits(l),line_room)
  %line_exact : {U32.to_nat(row_native(work,line)) == Nat.add(U32.to_nat(start(work)),_) : Nat}
  Arithmetic.add(start(work),U32.from_nat(line),coordinate,Limits.coordinate_fits(l),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(line)),line,line_exact) : {Nat.is_le(Nat.add(U32.to_nat(start(work)),_),U32.to_nat(coordinate)) == True{} : Bool} row_room)

def work_band(+work: Work,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool}) -> Bands.Query:
  Work{Phase{+call,+e,y,x,yi,xi},+chunk} = work
  Bands.caller_query(call,e,chunk_row(chunk),chunk_rows(chunk),contained)
def work_band_rows(+work: Work,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool}) ->
  {Extents.extent(Bands.axis(work_band(work,contained))) == Domain.height_word(Bands.geometry(work_band(work,contained))) : U32}:
  Work{Phase{call,e,y,x,yi,xi},chunk} = work
  Domain.Call{g,b,l,facts} = call
  {==}
def work_phase_y_inside(+work: Work,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool}) ->
  {Nat.is_lt(phase_y(phase(work)),Bands.count(work_band(work,contained))) == True{} : Bool}:
  Work{Phase{call,e,y,x,yi,xi},chunk} = work
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  yi

def row_band_used_finish(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(start(work)),row_position(work,line)) == True{} : Bool},
    {Nat.is_lt(row_position(work,line),U32.to_nat(end(work))) == True{} : Bool}>) ->
  {Bands.used(work_band(work,contained),U32.from_nat(phase_y(phase(work)))) == True{} : Bool}:
  Work{+phase,+chunk} = work
  Phase{call,e,y,x,yi,xi} = phase
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Kernel.Chunk{leaf,+row,+rows,low,group} = chunk
  Active{+used,xused} = active
  +axis = Kernel.axis(U32.from_nat(y),h,k,s,p)
  %Samples.used_matches(axis,k,s,row,(row + rows : U32)) : {_ == True{} : Bool}
  Samples.member_used_from(axis,s,row,(row + rows : U32),row_position(work,line),used,bounds)
def row_band_used(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  {Bands.used(work_band(work,contained),U32.from_nat(phase_y(phase(work)))) == True{} : Bool}:
  row_band_used_finish(work,active,line,contained,row_position_bounds(work,line,inside))

def row_selection(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) -> Samples.Selection:
  Bands.selection(work_band(work,contained),phase_y(phase(work)),work_phase_y_inside(work,contained),row_band_used(work,active,line,contained,inside))
def row_selection_indices(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  {Samples.selection_indices(row_selection(work,active,line,contained,inside)) == indices(work) : Samples.Indices}:
  Work{Phase{call,e,y,x,yi,xi},chunk} = work
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}
def row_selection_bounds_finish(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool},
  native_bounds: Shape.Both<{Nat.is_le(U32.to_nat(start(work)),row_position(work,line)) == True{} : Bool},
    {Nat.is_lt(row_position(work,line),U32.to_nat(end(work))) == True{} : Bool}>,
  encoded: Shape.Both<{U32.to_nat(Samples.low(Samples.selection_indices(row_selection(work,active,line,contained,inside)))) == PureBands.low(Samples.selection_model(row_selection(work,active,line,contained,inside))) : Nat},
    {U32.to_nat(Samples.high(Samples.selection_indices(row_selection(work,active,line,contained,inside)))) == PureBands.high(Samples.selection_model(row_selection(work,active,line,contained,inside))) : Nat}>) ->
  Shape.Both<{Nat.is_le(PureBands.low(Samples.selection_model(row_selection(work,active,line,contained,inside))),U32.to_nat(row_native(work,line))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(row_native(work,line)),PureBands.high(Samples.selection_model(row_selection(work,active,line,contained,inside)))) == True{} : Bool}>:
  Shape.Both{+lo,+hi} = native_bounds
  +selected = row_selection(work,active,line,contained,inside)
  +same = row_selection_indices(work,active,line,contained,inside)
  +exact = row_native_exact(work,active,line,inside)
  Samples.selection_bounds_from(selected,row_native(work,line),
    %Equal.sym(Nat,U32.to_nat(row_native(work,line)),row_position(work,line),exact) : {Nat.is_le(U32.to_nat(Samples.low(Samples.selection_indices(selected))),_) == True{} : Bool}
    %Equal.sym(Samples.Indices,Samples.selection_indices(selected),indices(work),same) : {Nat.is_le(U32.to_nat(Samples.low(_)),row_position(work,line)) == True{} : Bool} lo,
    %Equal.sym(Nat,U32.to_nat(row_native(work,line)),row_position(work,line),exact) : {Nat.is_lt(_,U32.to_nat(Samples.high(Samples.selection_indices(selected)))) == True{} : Bool}
    %Equal.sym(Samples.Indices,Samples.selection_indices(selected),indices(work),same) : {Nat.is_lt(row_position(work,line),U32.to_nat(Samples.high(_))) == True{} : Bool} hi,encoded)
def row_selection_bounds(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(PureBands.low(Samples.selection_model(row_selection(work,active,line,contained,inside))),U32.to_nat(row_native(work,line))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(row_native(work,line)),PureBands.high(Samples.selection_model(row_selection(work,active,line,contained,inside)))) == True{} : Bool}>:
  row_selection_bounds_finish(work,active,line,contained,inside,row_position_bounds(work,line,inside),Samples.selection_indices_exact(row_selection(work,active,line,contained,inside)))

def row_selection_count(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  {PhaseGeometry.axis_count(Samples.selection_axis(row_selection(work,active,line,contained,inside))) == row_count(phase(work)) : U32}:
  Work{Phase{call,e,y,x,yi,xi},chunk} = work
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}
def row_selection_inside(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(row_native(work,line)),U32.to_nat(PhaseGeometry.axis_count(Samples.selection_axis(row_selection(work,active,line,contained,inside))))) == True{} : Bool}:
  %Equal.sym(U32,PhaseGeometry.axis_count(Samples.selection_axis(row_selection(work,active,line,contained,inside))),row_count(phase(work)),row_selection_count(work,active,line,contained,inside)) :
    {Nat.is_lt(U32.to_nat(row_native(work,line)),U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(row_native(work,line)),row_position(work,line),row_native_exact(work,active,line,inside)) : {Nat.is_lt(_,U32.to_nat(row_count(phase(work)))) == True{} : Bool}
  row_position_inside(work,line,inside)
def row_tapped_bounds_finish(+work: Work,+active: Active<phase(work)>,+line: Nat,+tap: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Samples.selection_taps(row_selection(work,active,line,contained,inside))) == True{} : Bool},
  bounds: Shape.Both<{Nat.is_le(PureBands.low(Samples.selection_model(row_selection(work,active,line,contained,inside))),U32.to_nat(row_native(work,line))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(row_native(work,line)),PureBands.high(Samples.selection_model(row_selection(work,active,line,contained,inside)))) == True{} : Bool}>) ->
  Shape.Both<{U32.is_le(Bands.low_fold(work_band(work,contained)),Samples.native_tapped(row_selection(work,active,line,contained,inside),row_native(work,line),tap)) == True{} : Bool},
    {U32.is_lt(Samples.native_tapped(row_selection(work,active,line,contained,inside),row_native(work,line),tap),Bands.high_fold(work_band(work,contained))) == True{} : Bool}>:
  Shape.Both{+lo,+hi} = bounds
  Bands.tapped_native_bounds(work_band(work,contained),phase_y(phase(work)),work_phase_y_inside(work,contained),row_band_used(work,active,line,contained,inside),row_native(work,line),tap,
    row_selection_inside(work,active,line,contained,inside),lo,hi,tap_inside)
def row_tapped_bounds(+work: Work,+active: Active<phase(work)>,+line: Nat,+tap: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Samples.selection_taps(row_selection(work,active,line,contained,inside))) == True{} : Bool}) ->
  Shape.Both<{U32.is_le(Bands.low_fold(work_band(work,contained)),Samples.native_tapped(row_selection(work,active,line,contained,inside),row_native(work,line),tap)) == True{} : Bool},
    {U32.is_lt(Samples.native_tapped(row_selection(work,active,line,contained,inside),row_native(work,line),tap),Bands.high_fold(work_band(work,contained))) == True{} : Bool}>:
  row_tapped_bounds_finish(work,active,line,tap,contained,inside,tap_inside,row_selection_bounds(work,active,line,contained,inside))
def row_tapped_relative(+work: Work,+active: Active<phase(work)>,+line: Nat,+tap: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Samples.selection_taps(row_selection(work,active,line,contained,inside))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(Bands.relative(work_band(work,contained),Samples.native_tapped(row_selection(work,active,line,contained,inside),row_native(work,line),tap))),U32.to_nat(Kernel.band_rows(geometry(phase(work)),chunk_row(chunk(work)),chunk_rows(chunk(work))))) == True{} : Bool}:
  Work{Phase{+call,+e,y,x,yi,xi},+chunk} = work
  Domain.Call{g,b,l,facts} = call
  +row = chunk_row(chunk)
  +rows = chunk_rows(chunk)
  %Bands.caller_count_matches(call,e,row,rows,contained) :
    {Nat.is_lt(U32.to_nat(Bands.relative(work_band(work,contained),Samples.native_tapped(row_selection(work,active,line,contained,inside),row_native(work,line),tap))),U32.to_nat(Kernel.band_rows(Domain.geometry(call),row,_))) == True{} : Bool}
  Bands.tapped_relative_finish(work_band(work,contained),work_band_rows(work,contained),phase_y(phase(work)),work_phase_y_inside(work,contained),row_band_used(work,active,line,contained,inside),row_native(work,line),tap,
    row_tapped_bounds(work,active,line,tap,contained,inside,tap_inside))

def line_inside(+work: Work,+line: Nat,+group: Nat,+unit: Nat,inside: Inside<work,line,group,unit>) -> {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}:
  Inside{li,gi,ui} = inside
  li

def row_selection_taps(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  {Samples.selection_taps(row_selection(work,active,line,contained,inside)) == U32.to_nat(PhasePacking.phase_rows(context(phase(work)))) : Nat}:
  Work{Phase{call,e,y,x,yi,xi},chunk} = work
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}

def column_active(+phase: Phase,active: Active<phase>) -> {Extents.used(columns(phase),U32.from_nat(phase_x(phase))) == True{} : Bool}:
  Active{ey,ex} = active
  ex

def group_span_monotone_case(selected: Bool,+first: U32,+total: U32,+width: U32,+full: U32,
  +same: {U32.is_lt(first,total) == selected : Bool},+ordered: {Nat.is_le(U32.to_nat(total),U32.to_nat(full)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Bool.pick(U32,selected,U32.min(width,(total - first : U32)),0)),U32.to_nat(U32.min(width,(full - first : U32)))) == True{} : Bool}:
  match selected:
    case False{}: Order.zero_le(U32.to_nat(U32.min(width,(full - first : U32))))
    case True{}:
      +fn = U32.to_nat(first)
      +tn = U32.to_nat(total)
      +gn = U32.to_nat(full)
      +wn = U32.to_nat(width)
      +lower = Bounds.strict_le(fn,tn,Bounds.as_nat_lt(first,total,same))
      +full_lower = Order.transitive(fn,tn,gn,lower,ordered)
      +difference = (total - first : U32)
      +full_difference = (full - first : U32)
      +exact = FullRange.subtract(total,first,total,lower,Order.sub_le(tn,fn))
      +full_exact = FullRange.subtract(full,first,full,full_lower,Order.sub_le(gn,fn))
      %Equal.sym(Nat,U32.to_nat(U32.min(width,difference)),Nat.min(wn,U32.to_nat(difference)),NativeExtrema.minimum(width,difference)) : {Nat.is_le(_,U32.to_nat(U32.min(width,full_difference))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(U32.min(width,full_difference)),Nat.min(wn,U32.to_nat(full_difference)),NativeExtrema.minimum(width,full_difference)) : {Nat.is_le(Nat.min(wn,U32.to_nat(difference)),_) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(difference),Nat.sub(tn,fn),exact) : {Nat.is_le(Nat.min(wn,_),Nat.min(wn,U32.to_nat(full_difference))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(full_difference),Nat.sub(gn,fn),full_exact) : {Nat.is_le(Nat.min(wn,Nat.sub(tn,fn)),Nat.min(wn,_)) == True{} : Bool}
      Extrema.minimum_monotone(Nat.sub(tn,fn),Nat.sub(gn,fn),wn,Division.subtract_le(tn,gn,fn,ordered))
def group_span_monotone(+first: U32,+total: U32,+width: U32,+full: U32,+ordered: {Nat.is_le(U32.to_nat(total),U32.to_nat(full)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(group_span(first,total,width)),U32.to_nat(U32.min(width,(full - first : U32)))) == True{} : Bool}:
  group_span_monotone_case(U32.is_lt(first,total),first,total,width,full,{==},ordered)

def physical_row(+work: Work,+line: Nat) -> Nat:
  PhaseBounds.pixel(U32.from_nat(phase_y(phase(work))),Domain.stride_word(geometry(phase(work))),row_native(work,line))
def row_selection_position(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  {PureBands.position(Samples.selection_model(row_selection(work,active,line,contained,inside)),U32.to_nat(row_native(work,line))) == physical_row(work,line) : Nat}:
  Work{Phase{call,e,y,x,yi,xi},chunk} = work
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  {==}
def physical_row_bounds_from(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool},
  interval: Shape.Both<{Nat.is_le(PureBands.low(Samples.selection_model(row_selection(work,active,line,contained,inside))),U32.to_nat(row_native(work,line))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(row_native(work,line)),PureBands.high(Samples.selection_model(row_selection(work,active,line,contained,inside)))) == True{} : Bool}>) ->
  Shape.Both<{Nat.is_le(U32.to_nat(chunk_row(chunk(work))),physical_row(work,line)) == True{} : Bool},
    {Nat.is_lt(physical_row(work,line),U32.to_nat((chunk_row(chunk(work)) + chunk_rows(chunk(work)) : U32))) == True{} : Bool}>:
  Work{Phase{call,e,y,x,yi,xi},chunk_value} = work
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Shape.Both{+lo,+hi} = interval
  +band = Samples.selection_model(row_selection(work,active,line,contained,inside))
  +index = U32.to_nat(row_native(work,line))
  +same = row_selection_position(work,active,line,contained,inside)
  Shape.Both{
    %same : {Nat.is_le(U32.to_nat(chunk_row(chunk(work))),_) == True{} : Bool} PureBands.position_lower(band,index,lo),
    %same : {Nat.is_lt(_,U32.to_nat((chunk_row(chunk(work)) + chunk_rows(chunk(work)) : U32))) == True{} : Bool} PureBands.position_upper(band,index,lo,hi)}
def physical_row_bounds(+work: Work,+active: Active<phase(work)>,+line: Nat,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool},
  +inside: {Nat.is_lt(line,U32.to_nat(lines(work))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat(chunk_row(chunk(work))),physical_row(work,line)) == True{} : Bool},
    {Nat.is_lt(physical_row(work,line),U32.to_nat((chunk_row(chunk(work)) + chunk_rows(chunk(work)) : U32))) == True{} : Bool}>:
  physical_row_bounds_from(work,active,line,contained,inside,row_selection_bounds(work,active,line,contained,inside))
def chunk_end_exact(+work: Work,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work)))) == True{} : Bool}) ->
  {U32.to_nat((chunk_row(chunk(work)) + chunk_rows(chunk(work)) : U32)) == Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))) : Nat}:
  Arithmetic.add(chunk_row(chunk(work)),chunk_rows(chunk(work)),Machine.dimension(limits(phase(work))),Limits.dimension_fits(limits(phase(work))),Order.transitive(Nat.add(U32.to_nat(chunk_row(chunk(work))),U32.to_nat(chunk_rows(chunk(work)))),Domain.height(geometry(phase(work))),U32.to_nat(Machine.dimension(limits(phase(work)))),contained,phase_height_room(phase(work))))

def row_active(+phase: Phase,active: Active<phase>) -> {Extents.used(rows(phase),U32.from_nat(phase_y(phase))) == True{} : Bool}:
  Active{ey,ex} = active
  ey

# A local column group lies inside the unclipped phase group interval.
def group_span_location_case(selected: Bool,+first: U32,+total: U32,+width: U32,+position: Nat,
  +compared: {U32.is_lt(first,total) == selected : Bool},
  +inside: {Nat.is_lt(position,U32.to_nat(Bool.pick(U32,selected,U32.min(width,(total - first : U32)),0))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(first),position),U32.to_nat(total)) == True{} : Bool}:
  match selected:
    case False{}:
      Empty.absurd({Nat.is_lt(Nat.add(U32.to_nat(first),position),U32.to_nat(total)) == True{} : Bool},WordEncoding.false_true(%NatDivision.lt_zero(position) : {_ == True{} : Bool} inside))
    case True{}:
      +difference = (total - first : U32)
      +exact = FullRange.subtract(total,first,total,Bounds.strict_le(U32.to_nat(first),U32.to_nat(total),Bounds.as_nat_lt(first,total,compared)),Order.sub_le(U32.to_nat(total),U32.to_nat(first)))
      %Order.subtraction_lt(U32.to_nat(total),U32.to_nat(first),position) : {_ == True{} : Bool}
      %exact : {Nat.is_lt(position,_) == True{} : Bool}
      Bounds.nat_trans(position,U32.to_nat(U32.min(width,difference)),U32.to_nat(difference),inside,Bounds.as_nat_le(U32.min(width,difference),difference,NativeExtrema.minimum_right(width,difference)))
def group_span_location(+first: U32,+total: U32,+width: U32,+position: Nat,
  +inside: {Nat.is_lt(position,U32.to_nat(group_span(first,total,width))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(first),position),U32.to_nat(total)) == True{} : Bool}:
  group_span_location_case(U32.is_lt(first,total),first,total,width,position,{==},inside)
