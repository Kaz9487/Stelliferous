# A band inside another band of the same image copies gradient rows inside the rows the outer band copies.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_axes.bend as Axes
import ./input_gradient_phase_geometry.bend as PhaseGeometry
import ./input_gradient_tiled_bands.bend as PureBands
import ./input_gradient_tiled_geometry.bend as Geometry
import ./input_gradient_tiled_extents.bend as Extents
import ./input_gradient_tiled_band_geometry.bend as Samples
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_copy_geometry.bend as Copy
import ./input_gradient_tiled_source_coverage.bend as Coverage
import ./input_gradient_tiled_source_rows.bend as Rows
import ./traversal_extrema.bend as Folds
import ./traversal_array_refinement.bend as Arrays
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./nat_extrema.bend as Extrema
import ./nat_ceiling.bend as Ceiling
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./boolean_logic.bend as Boolean
import ./u32_division.bend as U32Division
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./machine_limit_bounds.bend as Limits
import ./storage_shape.bend as Shape
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding

# The pure sample bounds of one phase move with the band ends.
def pure_low_monotone(+axis: Kernel.Axis,+stride: U32,+parent_first: U32,+parent_last: U32,+child_first: U32,+child_last: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+padding: Nat,+kernel: Nat,+tap_count: Nat,
  lower: {Nat.is_le(U32.to_nat(parent_first),U32.to_nat(child_first)) == True{} : Bool}) ->
  {Nat.is_le(PureBands.sample_low(Samples.model(axis,stride,parent_first,parent_last,positive),padding,kernel,tap_count),
    PureBands.sample_low(Samples.model(axis,stride,child_first,child_last,positive),padding,kernel,tap_count)) == True{} : Bool}:
  Kernel.Axis{+start,count,tap,taps,source} = axis
  +phase = U32.to_nat(start)
  +divisor = U32.to_nat(stride)
  +low_parent = Ceiling.ceiling(Nat.sub(U32.to_nat(parent_first),phase),divisor)
  +low_child = Ceiling.ceiling(Nat.sub(U32.to_nat(child_first),phase),divisor)
  +start_parent = Nat.add(phase,Nat.mul(low_parent,divisor))
  +start_child = Nat.add(phase,Nat.mul(low_child,divisor))
  +origin_parent = Nat.div(Nat.add(start_parent,padding),divisor)
  +origin_child = Nat.div(Nat.add(start_child,padding),divisor)
  DivisionOrder.subtract_le(Nat.add(origin_parent,1n+kernel),Nat.add(origin_child,1n+kernel),tap_count,
    Order.add_le(origin_parent,origin_child,1n+kernel,
      DivisionOrder.quotient_le(Nat.add(start_parent,padding),Nat.add(start_child,padding),divisor,positive,
        Order.add_le(start_parent,start_child,padding,
          Order.offset_le(phase,Nat.mul(low_parent,divisor),Nat.mul(low_child,divisor),
            RowMajor.mul_le(low_parent,low_child,divisor,
              Ceiling.monotone(Nat.sub(U32.to_nat(parent_first),phase),Nat.sub(U32.to_nat(child_first),phase),divisor,positive,
                DivisionOrder.subtract_le(U32.to_nat(parent_first),U32.to_nat(child_first),phase,lower))))))))
def pure_high_monotone(+axis: Kernel.Axis,+stride: U32,+parent_first: U32,+parent_last: U32,+child_first: U32,+child_last: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},+padding: Nat,+kernel: Nat,
  upper: {Nat.is_le(U32.to_nat(child_last),U32.to_nat(parent_last)) == True{} : Bool}) ->
  {Nat.is_le(PureBands.sample_high(Samples.model(axis,stride,child_first,child_last,positive),padding,kernel),
    PureBands.sample_high(Samples.model(axis,stride,parent_first,parent_last,positive),padding,kernel)) == True{} : Bool}:
  Kernel.Axis{+start,+count,tap,taps,source} = axis
  +phase = U32.to_nat(start)
  +limit = U32.to_nat(count)
  +divisor = U32.to_nat(stride)
  +ceiling_parent = Ceiling.ceiling(Nat.sub(U32.to_nat(parent_last),phase),divisor)
  +ceiling_child = Ceiling.ceiling(Nat.sub(U32.to_nat(child_last),phase),divisor)
  +high_parent = Nat.min(limit,ceiling_parent)
  +high_child = Nat.min(limit,ceiling_child)
  +last_parent = Nat.add(phase,Nat.mul(Nat.sub(high_parent,1n),divisor))
  +last_child = Nat.add(phase,Nat.mul(Nat.sub(high_child,1n),divisor))
  Order.add_le(Nat.div(Nat.add(last_child,padding),divisor),Nat.div(Nat.add(last_parent,padding),divisor),1n+kernel,
    DivisionOrder.quotient_le(Nat.add(last_child,padding),Nat.add(last_parent,padding),divisor,positive,
      Order.add_le(last_child,last_parent,padding,
        Order.offset_le(phase,Nat.mul(Nat.sub(high_child,1n),divisor),Nat.mul(Nat.sub(high_parent,1n),divisor),
          RowMajor.mul_le(Nat.sub(high_child,1n),Nat.sub(high_parent,1n),divisor,
            DivisionOrder.subtract_le(high_child,high_parent,1n,
              Extrema.minimum_monotone(ceiling_child,ceiling_parent,limit,
                Ceiling.monotone(Nat.sub(U32.to_nat(child_last),phase),Nat.sub(U32.to_nat(parent_last),phase),divisor,positive,
                  DivisionOrder.subtract_le(U32.to_nat(child_last),U32.to_nat(parent_last),phase,upper)))))))))

# A phase used by the inner band is used by the outer band.
def used_monotone_from(+axis: Kernel.Axis,+stride: U32,+parent_first: U32,+parent_last: U32,+child_first: U32,+child_last: U32,
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  lower: {Nat.is_le(U32.to_nat(parent_first),U32.to_nat(child_first)) == True{} : Bool},
  upper: {Nat.is_le(U32.to_nat(child_last),U32.to_nat(parent_last)) == True{} : Bool},
  +child_used: {Samples.native_used(axis,stride,child_first,child_last) == True{} : Bool},
  parent_facts: Shape.Both<{U32.to_nat(Samples.low(Samples.indices(axis,stride,parent_first,parent_last))) == PureBands.low(Samples.model(axis,stride,parent_first,parent_last,positive)) : Nat},
    {U32.to_nat(Samples.high(Samples.indices(axis,stride,parent_first,parent_last))) == PureBands.high(Samples.model(axis,stride,parent_first,parent_last,positive)) : Nat}>,
  child_facts: Shape.Both<{U32.to_nat(Samples.low(Samples.indices(axis,stride,child_first,child_last))) == PureBands.low(Samples.model(axis,stride,child_first,child_last,positive)) : Nat},
    {U32.to_nat(Samples.high(Samples.indices(axis,stride,child_first,child_last))) == PureBands.high(Samples.model(axis,stride,child_first,child_last,positive)) : Nat}>) ->
  {Samples.native_used(axis,stride,parent_first,parent_last) == True{} : Bool}:
  Kernel.Axis{+start,+count,tap,+taps,source} = axis
  Shape.Both{+parent_low,+parent_high} = parent_facts
  Shape.Both{+child_low,+child_high} = child_facts
  +phase = U32.to_nat(start)
  +divisor = U32.to_nat(stride)
  +low_parent = Samples.clipped(start,parent_first,stride)
  +high_parent = U32.min(count,Samples.clipped(start,parent_last,stride))
  +low_child = Samples.clipped(start,child_first,stride)
  +high_child = U32.min(count,Samples.clipped(start,child_last,stride))
  +pure_low_parent = Ceiling.ceiling(Nat.sub(U32.to_nat(parent_first),phase),divisor)
  +pure_high_parent = Nat.min(U32.to_nat(count),Ceiling.ceiling(Nat.sub(U32.to_nat(parent_last),phase),divisor))
  +pure_low_child = Ceiling.ceiling(Nat.sub(U32.to_nat(child_first),phase),divisor)
  +pure_high_child = Nat.min(U32.to_nat(count),Ceiling.ceiling(Nat.sub(U32.to_nat(child_last),phase),divisor))
  +child_nonempty = {%child_low : {Nat.is_lt(_,pure_high_child) == True{} : Bool}
    %child_high : {Nat.is_lt(U32.to_nat(low_child),_) == True{} : Bool}
    Bounds.as_nat_lt(low_child,high_child,Boolean.right(U32.is_lt(0,taps),U32.is_lt(low_child,high_child),child_used)) : {Nat.is_lt(pure_low_child,pure_high_child) == True{} : Bool}}
  +parent_nonempty = Order.le_lt_trans(pure_low_parent,pure_low_child,pure_high_parent,
    Ceiling.monotone(Nat.sub(U32.to_nat(parent_first),phase),Nat.sub(U32.to_nat(child_first),phase),divisor,positive,DivisionOrder.subtract_le(U32.to_nat(parent_first),U32.to_nat(child_first),phase,lower)),
    Bounds.nat_trans(pure_low_child,pure_high_child,pure_high_parent,child_nonempty,
      Extrema.minimum_monotone(Ceiling.ceiling(Nat.sub(U32.to_nat(child_last),phase),divisor),Ceiling.ceiling(Nat.sub(U32.to_nat(parent_last),phase),divisor),U32.to_nat(count),
        Ceiling.monotone(Nat.sub(U32.to_nat(child_last),phase),Nat.sub(U32.to_nat(parent_last),phase),divisor,positive,DivisionOrder.subtract_le(U32.to_nat(child_last),U32.to_nat(parent_last),phase,upper)))))
  +parent_native = {%Equal.sym(Nat,U32.to_nat(low_parent),pure_low_parent,parent_low) : {Nat.is_lt(_,U32.to_nat(high_parent)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(high_parent),pure_high_parent,parent_high) : {Nat.is_lt(pure_low_parent,_) == True{} : Bool}
    parent_nonempty : {Nat.is_lt(U32.to_nat(low_parent),U32.to_nat(high_parent)) == True{} : Bool}}
  Boolean.and_true(U32.is_lt(0,taps),U32.is_lt(low_parent,high_parent),Boolean.left(U32.is_lt(0,taps),U32.is_lt(low_child,high_child),child_used),
    U32Division.nat_lt_native(low_parent,high_parent,parent_native))
def used_monotone(+axis: Kernel.Axis,+stride: U32,+parent_first: U32,+parent_last: U32,+child_first: U32,+child_last: U32,+limits: Machine.Limits,
  +parent_first_room: {Nat.is_le(U32.to_nat(parent_first),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +parent_last_room: {Nat.is_le(U32.to_nat(parent_last),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +child_first_room: {Nat.is_le(U32.to_nat(child_first),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +child_last_room: {Nat.is_le(U32.to_nat(child_last),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +stride_room: {Nat.is_le(U32.to_nat(stride),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(stride)) == True{} : Bool},
  lower: {Nat.is_le(U32.to_nat(parent_first),U32.to_nat(child_first)) == True{} : Bool},
  upper: {Nat.is_le(U32.to_nat(child_last),U32.to_nat(parent_last)) == True{} : Bool},
  +child_used: {Samples.native_used(axis,stride,child_first,child_last) == True{} : Bool}) ->
  {Samples.native_used(axis,stride,parent_first,parent_last) == True{} : Bool}:
  used_monotone_from(axis,stride,parent_first,parent_last,child_first,child_last,positive,lower,upper,child_used,
    Samples.indices_matches(axis,stride,parent_first,parent_last,limits,parent_first_room,parent_last_room,stride_room,positive),
    Samples.indices_matches(axis,stride,child_first,child_last,limits,child_first_room,child_last_room,stride_room,positive))

# One phase of one call, and a band of it with the facts a selection needs.
type Shared is Data:
  Shared{call: Domain.Call,phase: U32,size: U32,envelope: Geometry.WordEnvelope,
    size_room: {Nat.is_le(U32.to_nat(size),U32.to_nat(Machine.dimension(Domain.limits(call)))) == True{} : Bool},
    phase_inside: {Nat.is_lt(U32.to_nat(phase),U32.to_nat(size)) == True{} : Bool}}
def shared_call(shared: Shared) -> Domain.Call:
  Shared{call,phase,size,envelope,size_room,phase_inside} = shared
  call
def shared_axis(shared: Shared) -> Kernel.Axis:
  Shared{+call,phase,size,envelope,size_room,phase_inside} = shared
  Kernel.axis(phase,size,Domain.kernel_word(Domain.geometry(call)),Domain.stride_word(Domain.geometry(call)),Domain.padding_word(Domain.geometry(call)))
def shared_stride(shared: Shared) -> U32:
  Shared{call,phase,size,envelope,size_room,phase_inside} = shared
  Domain.stride_word(Domain.geometry(call))
type Band<-shared: Shared> is Data:
  Band{first: U32,last: U32,
    first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(Domain.limits(shared_call(shared))))) == True{} : Bool},
    last_room: {Nat.is_le(U32.to_nat(last),U32.to_nat(Machine.dimension(Domain.limits(shared_call(shared))))) == True{} : Bool},
    used: {Samples.native_used(shared_axis(shared),shared_stride(shared),first,last) == True{} : Bool}}
def band_first(-shared: Shared,band: Band<shared>) -> U32:
  Band{first,last,first_room,last_room,used} = band
  first
def band_last(-shared: Shared,band: Band<shared>) -> U32:
  Band{first,last,first_room,last_room,used} = band
  last
def selected(+shared: Shared,band: Band<shared>) -> Samples.Selection:
  Shared{call,phase,size,envelope,size_room,phase_inside} = shared
  Band{first,last,first_room,last_room,used} = band
  Samples.Selection{call,phase,size,first,last,envelope,size_room,phase_inside,first_room,last_room,used}
def native_low(+shared: Shared,+band: Band<shared>) -> Nat:
  U32.to_nat(Axes.band_sample_low(Samples.selection_sample(selected(shared,band))))
def native_high(+shared: Shared,+band: Band<shared>) -> Nat:
  U32.to_nat(Axes.band_sample_high(Samples.selection_sample(selected(shared,band))))
def pure_low(+shared: Shared,+band: Band<shared>) -> Nat:
  PureBands.sample_low(Samples.selection_model(selected(shared,band)),Samples.selection_padding(selected(shared,band)),Samples.selection_kernel(selected(shared,band)),Samples.selection_taps(selected(shared,band)))
def pure_high(+shared: Shared,+band: Band<shared>) -> Nat:
  PureBands.sample_high(Samples.selection_model(selected(shared,band)),Samples.selection_padding(selected(shared,band)),Samples.selection_kernel(selected(shared,band)))
def Readback(+shared: Shared,+band: Band<shared>) -> Data:
  Shape.Both<{native_low(shared,band) == pure_low(shared,band) : Nat},{native_high(shared,band) == pure_high(shared,band) : Nat}>
def readback(+shared: Shared,+band: Band<shared>) -> Readback(shared,band):
  Samples.readback(selected(shared,band))

def phase_low_from(+shared: Shared,+parent: Band<shared>,+child: Band<shared>,
  lower: {Nat.is_le(U32.to_nat(band_first(shared,parent)),U32.to_nat(band_first(shared,child))) == True{} : Bool},
  parent_facts: Readback(shared,parent),child_facts: Readback(shared,child)) ->
  {Nat.is_le(native_low(shared,parent),native_low(shared,child)) == True{} : Bool}:
  Shared{call,+phase,+size,envelope,size_room,phase_inside} = shared
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,+k,+s,+p,gh,gw} = geometry
  Band{+parent_first,+parent_last,parent_first_room,parent_last_room,parent_used} = parent
  Band{+child_first,+child_last,child_first_room,child_last_room,child_used} = child
  Shape.Both{+parent_low,parent_high} = parent_facts
  Shape.Both{+child_low,child_high} = child_facts
  +axis = {Kernel.axis(phase,size,k,s,p) : Kernel.Axis}
  +positive = Domain.stride_positive(Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits,facts})
  +padding = U32.to_nat(p)
  +kernel = U32.to_nat(k)
  +tap_count = U32.to_nat(PhaseGeometry.axis_taps(axis))
  %Equal.sym(Nat,U32.to_nat(Axes.band_sample_low(Axes.band_sample_from(axis,k,s,parent_first,parent_last))),PureBands.sample_low(Samples.model(axis,s,parent_first,parent_last,positive),padding,kernel,tap_count),parent_low) :
    {Nat.is_le(_,U32.to_nat(Axes.band_sample_low(Axes.band_sample_from(axis,k,s,child_first,child_last)))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Axes.band_sample_low(Axes.band_sample_from(axis,k,s,child_first,child_last))),PureBands.sample_low(Samples.model(axis,s,child_first,child_last,positive),padding,kernel,tap_count),child_low) :
    {Nat.is_le(PureBands.sample_low(Samples.model(axis,s,parent_first,parent_last,positive),padding,kernel,tap_count),_) == True{} : Bool}
  pure_low_monotone(axis,s,parent_first,parent_last,child_first,child_last,positive,padding,kernel,tap_count,lower)
def phase_low(+shared: Shared,+parent: Band<shared>,+child: Band<shared>,
  lower: {Nat.is_le(U32.to_nat(band_first(shared,parent)),U32.to_nat(band_first(shared,child))) == True{} : Bool}) ->
  {Nat.is_le(native_low(shared,parent),native_low(shared,child)) == True{} : Bool}:
  phase_low_from(shared,parent,child,lower,readback(shared,parent),readback(shared,child))
def phase_high_from(+shared: Shared,+parent: Band<shared>,+child: Band<shared>,
  upper: {Nat.is_le(U32.to_nat(band_last(shared,child)),U32.to_nat(band_last(shared,parent))) == True{} : Bool},
  parent_facts: Readback(shared,parent),child_facts: Readback(shared,child)) ->
  {Nat.is_le(native_high(shared,child),native_high(shared,parent)) == True{} : Bool}:
  Shared{call,+phase,+size,envelope,size_room,phase_inside} = shared
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,+k,+s,+p,gh,gw} = geometry
  Band{+parent_first,+parent_last,parent_first_room,parent_last_room,parent_used} = parent
  Band{+child_first,+child_last,child_first_room,child_last_room,child_used} = child
  Shape.Both{parent_low,+parent_high} = parent_facts
  Shape.Both{child_low,+child_high} = child_facts
  +axis = {Kernel.axis(phase,size,k,s,p) : Kernel.Axis}
  +positive = Domain.stride_positive(Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits,facts})
  +padding = U32.to_nat(p)
  +kernel = U32.to_nat(k)
  %Equal.sym(Nat,U32.to_nat(Axes.band_sample_high(Axes.band_sample_from(axis,k,s,child_first,child_last))),PureBands.sample_high(Samples.model(axis,s,child_first,child_last,positive),padding,kernel),child_high) :
    {Nat.is_le(_,U32.to_nat(Axes.band_sample_high(Axes.band_sample_from(axis,k,s,parent_first,parent_last)))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Axes.band_sample_high(Axes.band_sample_from(axis,k,s,parent_first,parent_last))),PureBands.sample_high(Samples.model(axis,s,parent_first,parent_last,positive),padding,kernel),parent_high) :
    {Nat.is_le(PureBands.sample_high(Samples.model(axis,s,child_first,child_last,positive),padding,kernel),_) == True{} : Bool}
  pure_high_monotone(axis,s,parent_first,parent_last,child_first,child_last,positive,padding,kernel,upper)
def phase_high(+shared: Shared,+parent: Band<shared>,+child: Band<shared>,
  upper: {Nat.is_le(U32.to_nat(band_last(shared,child)),U32.to_nat(band_last(shared,parent))) == True{} : Bool}) ->
  {Nat.is_le(native_high(shared,child),native_high(shared,parent)) == True{} : Bool}:
  phase_high_from(shared,parent,child,upper,readback(shared,parent),readback(shared,child))

# An outer band and an inner band of one call, both as the kernel's callers build them.
type Nested is Data:
  Nested{call: Domain.Call,envelope: Geometry.WordEnvelope,
    parent_first: U32,parent_rows: U32,
    parent_room: {Nat.is_le(Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows)),Domain.height(Domain.geometry(call))) == True{} : Bool},
    child_first: U32,child_rows: U32,
    child_room: {Nat.is_le(Nat.add(U32.to_nat(child_first),U32.to_nat(child_rows)),Domain.height(Domain.geometry(call))) == True{} : Bool},
    lower: {Nat.is_le(U32.to_nat(parent_first),U32.to_nat(child_first)) == True{} : Bool},
    upper: {Nat.is_le(Nat.add(U32.to_nat(child_first),U32.to_nat(child_rows)),Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows))) == True{} : Bool}}
def parent(pair: Nested) -> Bands.Query:
  Nested{call,envelope,parent_first,parent_rows,parent_room,child_first,child_rows,child_room,lower,upper} = pair
  Bands.caller_query(call,envelope,parent_first,parent_rows,parent_room)
def child(pair: Nested) -> Bands.Query:
  Nested{call,envelope,parent_first,parent_rows,parent_room,child_first,child_rows,child_room,lower,upper} = pair
  Bands.caller_query(call,envelope,child_first,child_rows,child_room)
def phases(pair: Nested) -> Nat:
  Bands.count(child(pair))
def active(pair: Nested,index: U32) -> Bool:
  Bands.used(child(pair),index)
def low_sample(pair: Nested,index: U32) -> U32:
  Bands.sampled(~False{},child(pair),index)
def high_sample(pair: Nested,index: U32) -> U32:
  Bands.sampled(~True{},child(pair),index)
def low_boundary(pair: Nested) -> Nat:
  U32.to_nat(Bands.low_fold(parent(pair)))
def high_boundary(pair: Nested) -> Nat:
  U32.to_nat(Bands.high_fold(parent(pair)))
def child_context(pair: Nested) -> Kernel.Band:
  Bands.context(child(pair))
def child_higher_context(pair: Nested) -> Kernel.Band:
  Bands.higher_context(child(pair))
def lower_step(+index: U32,+pair: Nested,+value: U32) ->
  {Folds.filtered(~False{},~Nested,~active,~low_sample,index,pair,value) == Kernel.band_phase(index,child_context(pair),value) : U32}:
  Nested{call,envelope,parent_first,parent_rows,parent_room,child_first,child_rows,child_room,lower,upper} = pair
  Bands.lower_step_matches(index,Bands.caller_query(call,envelope,child_first,child_rows,child_room),value)
def higher_step(+index: U32,+pair: Nested,+value: U32) ->
  {Folds.filtered(~True{},~Nested,~active,~high_sample,index,pair,value) == Kernel.band_phase(index,child_higher_context(pair),value) : U32}:
  Nested{call,envelope,parent_first,parent_rows,parent_room,child_first,child_rows,child_room,lower,upper} = pair
  Bands.higher_step_matches(index,Bands.caller_query(call,envelope,child_first,child_rows,child_room),value)

def last_exact(+call: Domain.Call,+first: U32,+count: U32,
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool}) ->
  {U32.to_nat((first + count : U32)) == Nat.add(U32.to_nat(first),U32.to_nat(count)) : Nat}:
  +limits = Domain.limits(call)
  Arithmetic.add(first,count,Machine.dimension(limits),Limits.dimension_fits(limits),
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call)),U32.to_nat(Machine.dimension(limits)),room,Domain.height_room(call)))

# Every phase the inner band uses samples between the outer band's extrema.
def both_bounds(+pair: Nested,+index: Nat,inside: {Nat.is_lt(index,phases(pair)) == True{} : Bool},enabled: {active(pair,U32.from_nat(index)) == True{} : Bool}) ->
  Shape.Both<Folds.natural_bound(False{},low_sample(pair,U32.from_nat(index)),low_boundary(pair)),Folds.natural_bound(True{},high_sample(pair,U32.from_nat(index)),high_boundary(pair))>:
  Nested{call,+envelope,+parent_first,+parent_rows,+parent_room,+child_first,+child_rows,+child_room,+lower,+upper} = pair
  Domain.Call{geometry,+batch,+limits,+facts} = call
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +inside = inside
  +whole = {Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,limits,facts} : Domain.Call}
  +parent = {Bands.caller_query(whole,envelope,parent_first,parent_rows,parent_room) : Bands.Query}
  +child = {Bands.caller_query(whole,envelope,child_first,child_rows,child_room) : Bands.Query}
  +phase = U32.from_nat(index)
  +axis = {Kernel.axis(phase,h,k,s,p) : Kernel.Axis}
  +parent_last = (parent_first + parent_rows : U32)
  +child_last = (child_first + child_rows : U32)
  +child_end = {%Equal.sym(Nat,U32.to_nat(child_last),Nat.add(U32.to_nat(child_first),U32.to_nat(child_rows)),last_exact(whole,child_first,child_rows,child_room)) : {Nat.is_le(_,U32.to_nat(parent_last)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(parent_last),Nat.add(U32.to_nat(parent_first),U32.to_nat(parent_rows)),last_exact(whole,parent_first,parent_rows,parent_room)) : {Nat.is_le(Nat.add(U32.to_nat(child_first),U32.to_nat(child_rows)),_) == True{} : Bool}
    upper : {Nat.is_le(U32.to_nat(child_last),U32.to_nat(parent_last)) == True{} : Bool}}
  +child_native = {%Equal.sym(Bool,Samples.native_used(axis,s,child_first,child_last),Axes.band_sample_used(Axes.band_sample_from(axis,k,s,child_first,child_last)),Samples.used_matches(axis,k,s,child_first,child_last)) : {_ == True{} : Bool}
    enabled : {Samples.native_used(axis,s,child_first,child_last) == True{} : Bool}}
  +parent_native = used_monotone(axis,s,parent_first,parent_last,child_first,child_last,limits,Bands.first_room(parent),Bands.last_room(parent),Bands.first_room(child),Bands.last_room(child),
    Domain.stride_room(whole),Domain.stride_positive(whole),lower,child_end,child_native)
  +parent_enabled = {%Samples.used_matches(axis,k,s,parent_first,parent_last) : {_ == True{} : Bool}
    parent_native : {Axes.band_sample_used(Axes.band_sample_from(axis,k,s,parent_first,parent_last)) == True{} : Bool}}
  +shared = {Shared{whole,phase,h,envelope,Domain.height_room(whole),Extents.index_inside_extent(Extents.rows(whole,envelope),index,inside)} : Shared}
  +parent_band = {Band{parent_first,parent_last,Bands.first_room(parent),Bands.last_room(parent),parent_native} : Band<shared>}
  +child_band = {Band{child_first,child_last,Bands.first_room(child),Bands.last_room(child),child_native} : Band<shared>}
  Shape.Both{
    Order.transitive(U32.to_nat(Bands.low_fold(parent)),U32.to_nat(Axes.band_low(Bands.context(parent),phase)),U32.to_nat(Axes.band_low(Bands.context(child),phase)),
      Bounds.as_nat_le(Bands.low_fold(parent),Axes.band_low(Bands.context(parent),phase),Bands.contains_low(parent,index,inside,parent_enabled)),
      phase_low(shared,parent_band,child_band,lower)),
    Order.transitive(U32.to_nat(Axes.band_high(Bands.context(child),phase)),U32.to_nat(Axes.band_high(Bands.context(parent),phase)),U32.to_nat(Bands.high_fold(parent)),
      phase_high(shared,parent_band,child_band,child_end),
      Bounds.as_nat_le(Axes.band_high(Bands.context(parent),phase),Bands.high_fold(parent),Bands.contains_high(parent,index,inside,parent_enabled)))}
def low_bound(+pair: Nested,+index: Nat,inside: {Nat.is_lt(index,phases(pair)) == True{} : Bool},enabled: {active(pair,U32.from_nat(index)) == True{} : Bool}) ->
  Folds.natural_bound(False{},low_sample(pair,U32.from_nat(index)),low_boundary(pair)):
  Shape.first(Folds.natural_bound(False{},low_sample(pair,U32.from_nat(index)),low_boundary(pair)),Folds.natural_bound(True{},high_sample(pair,U32.from_nat(index)),high_boundary(pair)),both_bounds(pair,index,inside,enabled))
def high_bound(+pair: Nested,+index: Nat,inside: {Nat.is_lt(index,phases(pair)) == True{} : Bool},enabled: {active(pair,U32.from_nat(index)) == True{} : Bool}) ->
  Folds.natural_bound(True{},high_sample(pair,U32.from_nat(index)),high_boundary(pair)):
  Shape.second(Folds.natural_bound(False{},low_sample(pair,U32.from_nat(index)),low_boundary(pair)),Folds.natural_bound(True{},high_sample(pair,U32.from_nat(index)),high_boundary(pair)),both_bounds(pair,index,inside,enabled))

def initial_low(+pair: Nested) -> {Nat.is_le(low_boundary(pair),U32.to_nat(Bands.seed(child(pair)))) == True{} : Bool}:
  Nested{+call,+envelope,+parent_first,+parent_rows,+parent_room,child_first,child_rows,child_room,lower,upper} = pair
  Geometry.maximum_covers(envelope,Bands.low_fold(Bands.caller_query(call,envelope,parent_first,parent_rows,parent_room)))
def low_fold_monotone(+pair: Nested) -> {Nat.is_le(low_boundary(pair),U32.to_nat(Bands.low_fold(child(pair)))) == True{} : Bool}:
  %Arrays.indexed_fold_change_context(~U32,~Nested,~Kernel.Band,~child_context,~Folds.filtered(~False{},~Nested,~active,~low_sample),~Kernel.band_phase,~lower_step,phases(pair),0,pair,Bands.seed(child(pair))) :
    {Nat.is_le(low_boundary(pair),U32.to_nat(_)) == True{} : Bool}
  Folds.natural_fold(~False{},~Nested,~phases,~active,~low_sample,~low_boundary,~low_bound,pair,Bands.seed(child(pair)),initial_low(pair))
def high_fold_monotone(+pair: Nested) -> {Nat.is_le(U32.to_nat(Bands.high_fold(child(pair))),high_boundary(pair)) == True{} : Bool}:
  %Arrays.indexed_fold_change_context(~U32,~Nested,~Kernel.Band,~child_higher_context,~Folds.filtered(~True{},~Nested,~active,~high_sample),~Kernel.band_phase,~higher_step,phases(pair),0,pair,0) :
    {Nat.is_le(U32.to_nat(_),high_boundary(pair)) == True{} : Bool}
  Folds.natural_fold(~True{},~Nested,~phases,~active,~high_sample,~high_boundary,~high_bound,pair,0,Order.zero_le(high_boundary(pair)))

# The copied interval of the inner band lies inside the outer band's.
def begin_monotone(+pair: Nested) -> {Nat.is_le(U32.to_nat(Copy.begin(parent(pair))),U32.to_nat(Copy.begin(child(pair)))) == True{} : Bool}:
  Nested{+call,+envelope,+parent_first,+parent_rows,+parent_room,+child_first,+child_rows,+child_room,lower,upper} = pair
  +parent = {Bands.caller_query(call,envelope,parent_first,parent_rows,parent_room) : Bands.Query}
  +child = {Bands.caller_query(call,envelope,child_first,child_rows,child_room) : Bands.Query}
  +kernel = Domain.kernel(Domain.geometry(call))
  +parent_low = Kernel.band_low(Domain.geometry(call),parent_first,(parent_first + parent_rows : U32))
  +child_low = Kernel.band_low(Domain.geometry(call),child_first,(child_first + child_rows : U32))
  +ordered = low_fold_monotone(Nested{call,envelope,parent_first,parent_rows,parent_room,child_first,child_rows,child_room,lower,upper})
  %Equal.sym(Nat,U32.to_nat(Copy.begin(parent)),Nat.sub(U32.to_nat(parent_low),kernel),Copy.begin_exact(parent)) : {Nat.is_le(_,U32.to_nat(Copy.begin(child))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Copy.begin(child)),Nat.sub(U32.to_nat(child_low),kernel),Copy.begin_exact(child)) : {Nat.is_le(Nat.sub(U32.to_nat(parent_low),kernel),_) == True{} : Bool}
  %Equal.sym(U32,parent_low,Bands.low_fold(parent),Bands.actual_low_matches(parent,{==})) : {Nat.is_le(Nat.sub(U32.to_nat(_),kernel),Nat.sub(U32.to_nat(child_low),kernel)) == True{} : Bool}
  %Equal.sym(U32,child_low,Bands.low_fold(child),Bands.actual_low_matches(child,{==})) : {Nat.is_le(Nat.sub(U32.to_nat(Bands.low_fold(parent)),kernel),Nat.sub(U32.to_nat(_),kernel)) == True{} : Bool}
  DivisionOrder.subtract_le(U32.to_nat(Bands.low_fold(parent)),U32.to_nat(Bands.low_fold(child)),kernel,ordered)
def stop_monotone(+pair: Nested) -> {Nat.is_le(U32.to_nat(Copy.stop(child(pair))),U32.to_nat(Copy.stop(parent(pair)))) == True{} : Bool}:
  Nested{+call,+envelope,+parent_first,+parent_rows,+parent_room,+child_first,+child_rows,+child_room,lower,upper} = pair
  +parent = {Bands.caller_query(call,envelope,parent_first,parent_rows,parent_room) : Bands.Query}
  +child = {Bands.caller_query(call,envelope,child_first,child_rows,child_room) : Bands.Query}
  +kernel = Domain.kernel(Domain.geometry(call))
  +height = Domain.gradient_height(Domain.geometry(call))
  +parent_high = Kernel.band_high(Domain.geometry(call),parent_first,(parent_first + parent_rows : U32))
  +child_high = Kernel.band_high(Domain.geometry(call),child_first,(child_first + child_rows : U32))
  +child_end = Nat.sub(U32.to_nat(Bands.high_fold(child)),kernel)
  +parent_end = Nat.sub(U32.to_nat(Bands.high_fold(parent)),kernel)
  +ordered = high_fold_monotone(Nested{call,envelope,parent_first,parent_rows,parent_room,child_first,child_rows,child_room,lower,upper})
  %Equal.sym(Nat,U32.to_nat(Copy.stop(child)),Nat.min(Nat.sub(U32.to_nat(child_high),kernel),height),Copy.stop_exact(child)) : {Nat.is_le(_,U32.to_nat(Copy.stop(parent))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Copy.stop(parent)),Nat.min(Nat.sub(U32.to_nat(parent_high),kernel),height),Copy.stop_exact(parent)) : {Nat.is_le(Nat.min(Nat.sub(U32.to_nat(child_high),kernel),height),_) == True{} : Bool}
  %Equal.sym(U32,child_high,Bands.high_fold(child),Bands.actual_high_matches(child,{==})) : {Nat.is_le(Nat.min(Nat.sub(U32.to_nat(_),kernel),height),Nat.min(Nat.sub(U32.to_nat(parent_high),kernel),height)) == True{} : Bool}
  %Equal.sym(U32,parent_high,Bands.high_fold(parent),Bands.actual_high_matches(parent,{==})) : {Nat.is_le(Nat.min(child_end,height),Nat.min(Nat.sub(U32.to_nat(_),kernel),height)) == True{} : Bool}
  Extrema.minimum_lower(parent_end,height,Nat.min(child_end,height),
    Order.transitive(Nat.min(child_end,height),child_end,parent_end,Extrema.minimum_left(child_end,height),
      DivisionOrder.subtract_le(U32.to_nat(Bands.high_fold(child)),U32.to_nat(Bands.high_fold(parent)),kernel,ordered)),
    Extrema.minimum_right(child_end,height))
def end_bound(+pair: Nested,same: {U32.is_lt(Copy.begin(child(pair)),Copy.stop(child(pair))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Copy.begin(child(pair))),U32.to_nat(Copy.row_count(child(pair)))),Nat.add(U32.to_nat(Copy.begin(parent(pair))),U32.to_nat(Copy.row_count(parent(pair))))) == True{} : Bool}:
  +child_begin = U32.to_nat(Copy.begin(child(pair)))
  +child_stop = U32.to_nat(Copy.stop(child(pair)))
  +parent_begin = U32.to_nat(Copy.begin(parent(pair)))
  +parent_stop = U32.to_nat(Copy.stop(parent(pair)))
  +ordered = Bounds.strict_le(child_begin,child_stop,Bounds.as_nat_lt(Copy.begin(child(pair)),Copy.stop(child(pair)),same))
  %Equal.sym(Nat,U32.to_nat(Copy.row_count(child(pair))),Nat.sub(child_stop,child_begin),Copy.count_exact(child(pair))) : {Nat.is_le(Nat.add(child_begin,_),Nat.add(parent_begin,U32.to_nat(Copy.row_count(parent(pair))))) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(child_begin,Nat.sub(child_stop,child_begin)),child_stop,Order.restore_shift(child_begin,child_stop,ordered)) : {Nat.is_le(_,Nat.add(parent_begin,U32.to_nat(Copy.row_count(parent(pair))))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Copy.row_count(parent(pair))),Nat.sub(parent_stop,parent_begin),Copy.count_exact(parent(pair))) : {Nat.is_le(child_stop,Nat.add(parent_begin,_)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(parent_begin,Nat.sub(parent_stop,parent_begin)),Nat.max(parent_begin,parent_stop),Extrema.clipped_end(parent_begin,parent_stop)) : {Nat.is_le(child_stop,_) == True{} : Bool}
  Order.transitive(child_stop,parent_stop,Nat.max(parent_begin,parent_stop),stop_monotone(pair),Extrema.maximum_right(parent_begin,parent_stop))

# Rows that contain the outer band's copy contain the inner band's copy.
def restricted(selected: Bool,+pair: Nested,+first: U32,+rows: U32,same: {U32.is_lt(Copy.begin(child(pair)),Copy.stop(child(pair))) == selected : Bool},
  contains: Coverage.Contains<Copy.begin(parent(pair)),Copy.row_count(parent(pair)),first,rows,Domain.gradient_rows(Bands.geometry(parent(pair)))>) ->
  Coverage.Contains<Copy.begin(child(pair)),Copy.row_count(child(pair)),first,rows,Domain.gradient_rows(Bands.geometry(parent(pair)))>:
  match selected:
    case False{}:
      Coverage.Empty{%Equal.sym(Bool,U32.is_lt(Copy.begin(child(pair)),Copy.stop(child(pair))),False{},same) : {Bool.pick(U32,_,(Copy.stop(child(pair)) - Copy.begin(child(pair)) : U32),0) == 0 : U32} {==}}
    case True{}:
      match contains:
        case Coverage.Full{first_zero,whole}: Coverage.Full{first_zero,whole}
        case Coverage.Exact{+first_equal,+rows_equal}:
          Coverage.Superset{
            %Equal.sym(U32,first,Copy.begin(parent(pair)),first_equal) : {Nat.is_le(U32.to_nat(_),U32.to_nat(Copy.begin(child(pair)))) == True{} : Bool} begin_monotone(pair),
            %Equal.sym(U32,first,Copy.begin(parent(pair)),first_equal) : {Nat.is_le(Nat.add(U32.to_nat(Copy.begin(child(pair))),U32.to_nat(Copy.row_count(child(pair)))),Nat.add(U32.to_nat(_),U32.to_nat(rows))) == True{} : Bool}
            %Equal.sym(U32,rows,Copy.row_count(parent(pair)),rows_equal) : {Nat.is_le(Nat.add(U32.to_nat(Copy.begin(child(pair))),U32.to_nat(Copy.row_count(child(pair)))),Nat.add(U32.to_nat(Copy.begin(parent(pair))),U32.to_nat(_))) == True{} : Bool}
            end_bound(pair,same)}
        case Coverage.Superset{+lower,+upper}:
          Coverage.Superset{
            Order.transitive(U32.to_nat(first),U32.to_nat(Copy.begin(parent(pair))),U32.to_nat(Copy.begin(child(pair))),lower,begin_monotone(pair)),
            Order.transitive(Nat.add(U32.to_nat(Copy.begin(child(pair))),U32.to_nat(Copy.row_count(child(pair)))),Nat.add(U32.to_nat(Copy.begin(parent(pair))),U32.to_nat(Copy.row_count(parent(pair)))),Nat.add(U32.to_nat(first),U32.to_nat(rows)),end_bound(pair,same),upper)}
        case Coverage.Empty{+none}:
          +child_begin = U32.to_nat(Copy.begin(child(pair)))
          +child_stop = U32.to_nat(Copy.stop(child(pair)))
          +parent_begin = U32.to_nat(Copy.begin(parent(pair)))
          +parent_stop = U32.to_nat(Copy.stop(parent(pair)))
          +parent_lt = Order.le_lt_trans(parent_begin,child_begin,parent_stop,begin_monotone(pair),
            Bounds.nat_trans(child_begin,child_stop,parent_stop,Bounds.as_nat_lt(Copy.begin(child(pair)),Copy.stop(child(pair)),same),stop_monotone(pair)))
          +difference_zero = Equal.trans(Nat,Nat.sub(parent_stop,parent_begin),U32.to_nat(Copy.row_count(parent(pair))),0n,
            Equal.sym(Nat,U32.to_nat(Copy.row_count(parent(pair))),Nat.sub(parent_stop,parent_begin),Copy.count_exact(parent(pair))),
            Equal.cong(U32,Nat,U32.to_nat,Copy.row_count(parent(pair)),0,none))
          +padded = {%Equal.sym(Nat,Nat.sub(parent_stop,parent_begin),0n,difference_zero) : {parent_begin == Nat.add(parent_begin,_) : Nat}
            Equal.sym(Nat,Nat.add(parent_begin,0n),parent_begin,Addition.zero_right(parent_begin)) : {parent_begin == Nat.add(parent_begin,Nat.sub(parent_stop,parent_begin)) : Nat}}
          +collapsed = Equal.trans(Nat,parent_begin,Nat.add(parent_begin,Nat.sub(parent_stop,parent_begin)),parent_stop,padded,
            Order.restore_shift(parent_begin,parent_stop,Bounds.strict_le(parent_begin,parent_stop,parent_lt)))
          Empty.absurd(Coverage.Contains<Copy.begin(child(pair)),Copy.row_count(child(pair)),first,rows,Domain.gradient_rows(Bands.geometry(parent(pair)))>,
            WordEncoding.false_true(
              %Order.not_below(parent_stop,parent_stop,Order.reflexive(parent_stop)) : {_ == True{} : Bool}
              %collapsed : {Nat.is_lt(_,parent_stop) == True{} : Bool}
              parent_lt))
def restrict(+pair: Nested,+first: U32,+rows: U32,
  contains: Coverage.Contains<Copy.begin(parent(pair)),Copy.row_count(parent(pair)),first,rows,Domain.gradient_rows(Bands.geometry(parent(pair)))>) ->
  Coverage.Contains<Copy.begin(child(pair)),Copy.row_count(child(pair)),first,rows,Domain.gradient_rows(Bands.geometry(parent(pair)))>:
  restricted(U32.is_lt(Copy.begin(child(pair)),Copy.stop(child(pair))),pair,first,rows,{==},contains)

# A band copies rows exactly when its copy begins before it stops.
def positive_count_case(selected: Bool,+query: Bands.Query,+compared: {U32.is_lt(Copy.begin(query),Copy.stop(query)) == selected : Bool},
  positive: {Nat.is_lt(0n,U32.to_nat(Copy.row_count(query))) == True{} : Bool}) -> {selected == True{} : Bool}:
  match selected:
    case True{}: {==}
    case False{}:
      Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(
        %compared : {Nat.is_lt(0n,U32.to_nat(Bool.pick(U32,_,(Copy.stop(query) - Copy.begin(query) : U32),0))) == True{} : Bool}
        positive))
def positive_count(+query: Bands.Query,positive: {Nat.is_lt(0n,U32.to_nat(Copy.row_count(query))) == True{} : Bool}) ->
  {U32.is_lt(Copy.begin(query),Copy.stop(query)) == True{} : Bool}:
  positive_count_case(U32.is_lt(Copy.begin(query),Copy.stop(query)),query,{==},positive)

# The copied rows of a band end inside the gradient image.
def end_inside_from(+query: Bands.Query,same: {U32.is_lt(Copy.begin(query),Copy.stop(query)) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Copy.begin(query)),U32.to_nat(Copy.row_count(query))),Domain.gradient_height(Bands.geometry(query))) == True{} : Bool}:
  +begin = U32.to_nat(Copy.begin(query))
  +stop = U32.to_nat(Copy.stop(query))
  %Equal.sym(Nat,U32.to_nat(Copy.row_count(query)),Nat.sub(stop,begin),Copy.count_exact(query)) :
    {Nat.is_le(Nat.add(begin,_),Domain.gradient_height(Bands.geometry(query))) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(begin,Nat.sub(stop,begin)),stop,Order.restore_shift(begin,stop,Bounds.strict_le(begin,stop,Bounds.as_nat_lt(Copy.begin(query),Copy.stop(query),same)))) :
    {Nat.is_le(_,Domain.gradient_height(Bands.geometry(query))) == True{} : Bool}
  Copy.stop_bounded(query)
def end_inside(+query: Bands.Query,positive: {Nat.is_lt(0n,U32.to_nat(Copy.row_count(query))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(Copy.begin(query)),U32.to_nat(Copy.row_count(query))),Domain.gradient_height(Bands.geometry(query))) == True{} : Bool}:
  end_inside_from(query,positive_count(query,positive))

# The copied rows of a band are gradient rows, or there are none.
def copy_range_case(selected: Bool,+query: Bands.Query,compared: {U32.is_lt(Copy.begin(query),Copy.stop(query)) == selected : Bool}) ->
  Rows.Range<Copy.begin(query),Copy.row_count(query),Domain.gradient_rows(Bands.geometry(query))>:
  match selected:
    case True{}:
      Bands.Query{Extents.Axis{+call,+size,+size_room,+e},+first,+last,+ordered,+room} = query
      Domain.Call{+g,+b,+l,+facts} = call
      Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = g
      Rows.ImageRows{end_inside_from(Bands.Query{Extents.Axis{Domain.Call{Kernel.Geometry{c,o,h,w,k,s,p,gh,gw},b,l,facts},size,size_room,e},first,last,ordered,room},compared)}
    case False{}:
      Rows.EmptyRows{%Equal.sym(Bool,U32.is_lt(Copy.begin(query),Copy.stop(query)),False{},compared) :
        {Bool.pick(U32,_,(Copy.stop(query) - Copy.begin(query) : U32),0) == 0 : U32} {==}}
def copy_range(+query: Bands.Query) -> Rows.Range<Copy.begin(query),Copy.row_count(query),Domain.gradient_rows(Bands.geometry(query))>:
  copy_range_case(U32.is_lt(Copy.begin(query),Copy.stop(query)),query,{==})
