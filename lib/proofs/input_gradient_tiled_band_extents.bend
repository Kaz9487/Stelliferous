# Over the phases of an axis, the band extrema bound the tiles that outputs
# [first, last) read.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ../traversal_array.bend as Traversal
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_axes.bend as Axes
import ./input_gradient_phase_geometry.bend as Phase
import ./input_gradient_tiled_bands.bend as Bands
import ./input_gradient_tiled_geometry.bend as Geometry
import ./input_gradient_tiled_extents.bend as Extents
import ./input_gradient_tiled_band_geometry.bend as Samples
import ./traversal_extrema.bend as Folds
import ./traversal_array_refinement.bend as Arrays
import ./nat_order.bend as Order
import ./boolean_logic.bend as Boolean
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division_order.bend as Division
import ./storage_shape.bend as Shape
import ./u32_full_range.bend as FullRange
import ./nat_ceiling.bend as Ceiling
import ./natural_addition.bend as Addition
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction
import ./machine_limit_bounds.bend as Limits

type Query is Data:
  Query{axis: Extents.Axis,first: U32,last: U32,ordered: {Nat.is_le(U32.to_nat(first),U32.to_nat(last)) == True{} : Bool},
    last_room: {Nat.is_le(U32.to_nat(last),U32.to_nat(Extents.extent(axis))) == True{} : Bool}}
def axis(query: Query) -> Extents.Axis:
  Query{axis,first,last,ordered,room} = query
  axis
def first(query: Query) -> U32:
  Query{axis,first,last,ordered,room} = query
  first
def last(query: Query) -> U32:
  Query{axis,first,last,ordered,room} = query
  last
def geometry(query: Query) -> Kernel.Geometry:
  Extents.geometry(axis(query))
def envelope(query: Query) -> Geometry.WordEnvelope:
  Extents.envelope(axis(query))
def query_limits(query: Query) -> Machine.Limits:
  Extents.limits(axis(query))
def count(query: Query) -> Nat:
  Extents.count(axis(query))
def rows(query: Query) -> Nat:
  +query = query
  Nat.sub(U32.to_nat(last(query)),U32.to_nat(first(query)))

# The first output of every phase, and every output, of an axis.
def starts(+axis: Extents.Axis) -> Query:
  Query{axis,0,U32.min(Domain.stride_word(Extents.geometry(axis)),Extents.extent(axis)),Order.zero_le(Extents.count(axis)),Extents.count_size_room(axis)}
def whole(+axis: Extents.Axis) -> Query:
  Query{axis,0,Extents.extent(axis),Order.zero_le(U32.to_nat(Extents.extent(axis))),Order.reflexive(U32.to_nat(Extents.extent(axis)))}

def context(query: Query) -> Kernel.Band:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Kernel.Band{size,k,s,p,first,last,0}
def higher_context(query: Query) -> Kernel.Band:
  Axes.band_with_what(context(query),1)
def count_context(query: Query) -> Kernel.Band:
  Axes.band_with_what(context(query),2)
def lower(query: Query) -> Nat:
  +query = query
  Nat.sub(Geometry.high(geometry(query),first(query)),Geometry.halo(geometry(query)))
def upper(query: Query) -> Nat:
  +query = query
  Geometry.end_high(geometry(query),last(query))
def boundary(~higher: Bool,query: Query) -> Nat:
  +query = query
  Bool.pick(Nat,higher,upper(query),lower(query))
def used(query: Query,index: U32) -> Bool:
  Axes.band_used(context(query),index)
def sampled(~higher: Bool,query: Query,index: U32) -> U32:
  +query = query
  +index = index
  +context = context(query)
  Bool.pick(U32,higher,Axes.band_high(context,index),Axes.band_low(context,index))
def count_sampled(query: Query,index: U32) -> U32:
  Axes.band_count(context(query),index)
def count_boundary(query: Query) -> Nat:
  U32.to_nat(Extents.extent(axis(query)))

def first_room(+query: Query) -> {Nat.is_le(U32.to_nat(first(query)),U32.to_nat(Machine.dimension(query_limits(query)))) == True{} : Bool}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  +dimension = U32.to_nat(Machine.dimension(Domain.limits(call)))
  Order.transitive(U32.to_nat(first),U32.to_nat(last),dimension,ordered,Order.transitive(U32.to_nat(last),U32.to_nat(size),dimension,room,size_room))
def last_room(+query: Query) -> {Nat.is_le(U32.to_nat(last(query)),U32.to_nat(Machine.dimension(query_limits(query)))) == True{} : Bool}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Order.transitive(U32.to_nat(last),U32.to_nat(size),U32.to_nat(Machine.dimension(Domain.limits(call))),room,size_room)

def selection(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) -> Samples.Selection:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +phase = U32.from_nat(index)
  +exact = Extents.index_exact(axis(query),index,inside)
  +phase_inside = {%Equal.sym(Nat,U32.to_nat(phase),index,exact) : {Nat.is_lt(_,U32.to_nat(size)) == True{} : Bool}
    Bounds.nat_trans(index,count(query),U32.to_nat(size),inside,Extents.count_size_room(axis(query))) : {Nat.is_lt(U32.to_nat(phase),U32.to_nat(size)) == True{} : Bool}}
  +native_used = {%Equal.sym(Bool,Samples.native_used(Kernel.axis(phase,size,k,s,p),s,first,last),Axes.band_sample_used(Axes.band_sample_from(Kernel.axis(phase,size,k,s,p),k,s,first,last)),Samples.used_matches(Kernel.axis(phase,size,k,s,p),k,s,first,last)) : {_ == True{} : Bool} enabled : {Samples.native_used(Kernel.axis(phase,size,k,s,p),s,first,last) == True{} : Bool}}
  Samples.Selection{call,phase,size,first,last,envelope,size_room,phase_inside,first_room(query),last_room(query),native_used}

def sample_bounds(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(lower(query),U32.to_nat(Axes.band_low(context(query),U32.from_nat(index)))) == True{} : Bool},
    {Nat.is_le(U32.to_nat(Axes.band_high(context(query),U32.from_nat(index))),upper(query)) == True{} : Bool}>:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Samples.sample_bounds(selection(query,index,inside,enabled))

def chosen_bound(higher: Bool,+query: Query,+index: Nat,
  facts: Shape.Both<{Nat.is_le(lower(query),U32.to_nat(Axes.band_low(context(query),U32.from_nat(index)))) == True{} : Bool},
    {Nat.is_le(U32.to_nat(Axes.band_high(context(query),U32.from_nat(index))),upper(query)) == True{} : Bool}>) ->
  Folds.natural_bound(higher,Bool.pick(U32,higher,Axes.band_high(context(query),U32.from_nat(index)),Axes.band_low(context(query),U32.from_nat(index))),Bool.pick(Nat,higher,upper(query),lower(query))):
  match higher:
    case False{}:
      Shape.Both{lower,upper} = facts
      lower
    case True{}:
      Shape.Both{lower,upper} = facts
      upper

def sampled_bound(~higher: Bool,+query: Query,+index: Nat,inside: {Nat.is_lt(index,count(query)) == True{} : Bool},enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) ->
  Folds.natural_bound(higher,sampled(~higher,query,U32.from_nat(index)),boundary(~higher,query)):
  +inside = inside
  +enabled = enabled
  chosen_bound(higher,query,index,sample_bounds(query,index,inside,enabled))

# A used phase reads a tap.
def tapped(+query: Query,+index: Nat,+enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) ->
  {Extents.used(axis(query),U32.from_nat(index)) == True{} : Bool}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +native = Kernel.axis(U32.from_nat(index),size,k,s,p)
  Boolean.left(U32.is_lt(0,Phase.axis_taps(native)),U32.is_lt(Samples.low(Samples.indices(native,s,first,last)),Samples.high(Samples.indices(native,s,first,last))),enabled)
def count_sample_bound(+query: Query,+index: Nat,inside: {Nat.is_lt(index,count(query)) == True{} : Bool},enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) ->
  Folds.natural_bound(True{},count_sampled(query,U32.from_nat(index)),count_boundary(query)):
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Extents.count_sample_bound(axis(query),index,inside,tapped(query,index,enabled))

def lower_step_matches(+index: U32,+query: Query,+value: U32) ->
  {Folds.filtered(~False{},~Query,~used,~sampled(~False{}),index,query,value) == Kernel.band_phase(index,context(query),value) : U32}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}
def higher_step_matches(+index: U32,+query: Query,+value: U32) ->
  {Folds.filtered(~True{},~Query,~used,~sampled(~True{}),index,query,value) == Kernel.band_phase(index,higher_context(query),value) : U32}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}
def count_step_matches(+index: U32,+query: Query,+value: U32) ->
  {Folds.filtered(~True{},~Query,~used,~count_sampled,index,query,value) == Kernel.band_phase(index,count_context(query),value) : U32}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}

def seed(query: Query) -> U32:
  Geometry.maximum(envelope(query))
def seed_default(+query: Query) -> {seed(query) == Machine.word_max() : U32}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Geometry.WordEnvelope{maximum,half,quarter,margin,me,he,qe,re} = envelope
  me
def low_fold(query: Query) -> U32:
  +query = query
  Traversal.indexed_fold(~U32,~Kernel.Band,~Kernel.band_phase,count(query),0,context(query),seed(query))
def high_fold(query: Query) -> U32:
  +query = query
  Traversal.indexed_fold(~U32,~Kernel.Band,~Kernel.band_phase,count(query),0,higher_context(query),0)
def count_fold(query: Query) -> U32:
  +query = query
  Traversal.indexed_fold(~U32,~Kernel.Band,~Kernel.band_phase,count(query),0,count_context(query),0)

def initial_lower(+query: Query) -> {Nat.is_le(lower(query),U32.to_nat(seed(query))) == True{} : Bool}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Order.transitive(lower(query),Geometry.high(geometry,first),U32.to_nat(Geometry.maximum(envelope)),Order.sub_le(Geometry.high(geometry,first),Geometry.halo(geometry)),Geometry.natural_high_room(call,first,first_room(query),envelope))

def low_bound(+query: Query) -> {Nat.is_le(lower(query),U32.to_nat(low_fold(query))) == True{} : Bool}:
  %Arrays.indexed_fold_change_context(~U32,~Query,~Kernel.Band,~context,~Folds.filtered(~False{},~Query,~used,~sampled(~False{})),~Kernel.band_phase,~lower_step_matches,count(query),0,query,seed(query)) : {Nat.is_le(lower(query),U32.to_nat(_)) == True{} : Bool}
  Folds.natural_fold(~False{},~Query,~count,~used,~sampled(~False{}),~boundary(~False{}),~sampled_bound(~False{}),query,seed(query),initial_lower(query))
def high_bound(+query: Query) -> {Nat.is_le(U32.to_nat(high_fold(query)),upper(query)) == True{} : Bool}:
  %Arrays.indexed_fold_change_context(~U32,~Query,~Kernel.Band,~higher_context,~Folds.filtered(~True{},~Query,~used,~sampled(~True{})),~Kernel.band_phase,~higher_step_matches,count(query),0,query,0) : {Nat.is_le(U32.to_nat(_),upper(query)) == True{} : Bool}
  Folds.natural_fold(~True{},~Query,~count,~used,~sampled(~True{}),~boundary(~True{}),~sampled_bound(~True{}),query,0,Order.zero_le(upper(query)))
def count_bound(+query: Query) -> {Nat.is_le(U32.to_nat(count_fold(query)),count_boundary(query)) == True{} : Bool}:
  %Arrays.indexed_fold_change_context(~U32,~Query,~Kernel.Band,~count_context,~Folds.filtered(~True{},~Query,~used,~count_sampled),~Kernel.band_phase,~count_step_matches,count(query),0,query,0) : {Nat.is_le(U32.to_nat(_),count_boundary(query)) == True{} : Bool}
  Folds.natural_fold(~True{},~Query,~count,~used,~count_sampled,~count_boundary,~count_sample_bound,query,0,Order.zero_le(count_boundary(query)))

# The folds contain the extrema of every used phase.
def selected_query(query: Query,index: Nat) -> Folds.Query<Kernel.Band>:
  +query = query
  Folds.Query{context(query),count(query),index}
def contains_low(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) ->
  {U32.is_le(low_fold(query),Axes.band_low(context(query),U32.from_nat(index))) == True{} : Bool}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Axes.lower_band_contains(selected_query(query,index),seed(query),enabled,inside)
def contains_high(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) ->
  {U32.is_le(Axes.band_high(context(query),U32.from_nat(index)),high_fold(query)) == True{} : Bool}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Axes.higher_band_contains(selected_query(query,index),0,enabled,inside)
def contains_count(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) ->
  {U32.is_le(Axes.band_count(context(query),U32.from_nat(index)),count_fold(query)) == True{} : Bool}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Axes.count_band_contains(selected_query(query,index),0,enabled,inside)

def sample_ordered(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Axes.band_low(context(query),U32.from_nat(index))),U32.to_nat(Axes.band_high(context(query),U32.from_nat(index)))) == True{} : Bool}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Samples.sample_ordered(selection(query,index,inside,enabled))
def extrema_ordered(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(low_fold(query)),U32.to_nat(high_fold(query))) == True{} : Bool}:
  +low = Axes.band_low(context(query),U32.from_nat(index))
  +high = Axes.band_high(context(query),U32.from_nat(index))
  Order.transitive(U32.to_nat(low_fold(query)),U32.to_nat(low),U32.to_nat(high_fold(query)),Bounds.as_nat_le(low_fold(query),low,contains_low(query,index,inside,enabled)),
    Order.transitive(U32.to_nat(low),U32.to_nat(high),U32.to_nat(high_fold(query)),sample_ordered(query,index,inside,enabled),Bounds.as_nat_le(high,high_fold(query),contains_high(query,index,inside,enabled))))

# The tiles a band reads span at most width + halo rows when the end of the band
# is at most width rows past its origin.
def halo_inside_origin(+query: Query) -> {Nat.is_le(Geometry.halo(geometry(query)),Geometry.high(geometry(query),first(query))) == True{} : Bool}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  +kn = Domain.kernel(geometry)
  +origin = Geometry.origin(geometry,first)
  Order.transitive(Geometry.halo(geometry),kn,Geometry.high(geometry,first),Ceiling.bounded(kn,Domain.stride(geometry),Domain.stride_positive(call)),
    Order.transitive(kn,1n+kn,Geometry.high(geometry,first),Bounds.strict_le(kn,1n+kn,Order.lt_succ(kn,kn,Order.reflexive(kn))),
      %Order.commutative(1n+kn,origin) : {Nat.is_le(1n+kn,_) == True{} : Bool} Order.le_add(1n+kn,origin)))
def span_bounded(+query: Query,+width: Nat,
  +room: {Nat.is_le(upper(query),Nat.add(Geometry.high(geometry(query),first(query)),width)) == True{} : Bool},
  +ordered: {Nat.is_le(U32.to_nat(low_fold(query)),U32.to_nat(high_fold(query))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat((high_fold(query) - low_fold(query) : U32)),Nat.add(width,Geometry.halo(geometry(query)))) == True{} : Bool}:
  +low = low_fold(query)
  +high = high_fold(query)
  +halo = Geometry.halo(geometry(query))
  +exact = FullRange.subtract(high,low,Geometry.maximum(envelope(query)),ordered,
    Order.transitive(Nat.sub(U32.to_nat(high),U32.to_nat(low)),U32.to_nat(high),U32.to_nat(Geometry.maximum(envelope(query))),Order.sub_le(U32.to_nat(high),U32.to_nat(low)),Geometry.maximum_covers(envelope(query),high)))
  %Equal.sym(Nat,U32.to_nat((high - low : U32)),Nat.sub(U32.to_nat(high),U32.to_nat(low)),exact) : {Nat.is_le(_,Nat.add(width,halo)) == True{} : Bool}
  %Order.commutative(halo,width) : {Nat.is_le(Nat.sub(U32.to_nat(high),U32.to_nat(low)),_) == True{} : Bool}
  Geometry.envelope_span(U32.to_nat(low),U32.to_nat(high),Geometry.high(geometry(query),first(query)),halo,width,halo_inside_origin(query),low_bound(query),
    Order.transitive(U32.to_nat(high),upper(query),Nat.add(Geometry.high(geometry(query),first(query)),width),high_bound(query),room))

# A band ends at most rows - 1 rows past its origin; the first outputs of the
# phases end at most one row past theirs.
def rows_room(+query: Query) -> {Nat.is_le(upper(query),Nat.add(Geometry.high(geometry(query),first(query)),Nat.sub(rows(query),1n))) == True{} : Bool}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +start = U32.to_nat(first)
  +count = Nat.sub(U32.to_nat(last),start)
  +width = Nat.sub(count,1n)
  +quotient = Nat.div(Nat.add(start,U32.to_nat(p)),U32.to_nat(s))
  +origin = Nat.add(quotient,1n+U32.to_nat(k))
  %Order.commutative(width,origin) : {Nat.is_le(upper(query),_) == True{} : Bool}
  %Addition.associative(width,quotient,1n+U32.to_nat(k)) : {Nat.is_le(upper(query),_) == True{} : Bool}
  %Order.restore_shift(start,U32.to_nat(last),ordered) :
    {Nat.is_le(Nat.add(Nat.div(Nat.add(Nat.sub(_,1n),U32.to_nat(p)),U32.to_nat(s)),1n+U32.to_nat(k)),Nat.add(Nat.add(width,quotient),1n+U32.to_nat(k))) == True{} : Bool}
  Order.add_le(Nat.div(Nat.add(Nat.sub(Nat.add(start,count),1n),U32.to_nat(p)),U32.to_nat(s)),Nat.add(width,quotient),1n+U32.to_nat(k),
    Bands.end_quotient(start,count,U32.to_nat(p),U32.to_nat(s),Domain.stride_positive(call)))
def starts_room(+axis: Extents.Axis) -> {Nat.is_le(upper(starts(axis)),Nat.add(Geometry.high(Extents.geometry(axis),0),1n)) == True{} : Bool}:
  Extents.Axis{call,size,size_room,envelope} = axis
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +positive = Domain.stride_positive(call)
  +stride = U32.to_nat(s)
  +padding = U32.to_nat(p)
  +last = Nat.sub(Extents.count(axis),1n)
  +inside = Order.le_lt_trans(last,Nat.sub(stride,1n),stride,Division.subtract_le(Extents.count(axis),stride,1n,Extents.count_stride_room(axis)),Order.last_below(stride,positive))
  +origin = Nat.add(Nat.div(padding,stride),1n+U32.to_nat(k))
  %Order.commutative(1n,origin) : {Nat.is_le(upper(starts(axis)),_) == True{} : Bool}
  %Order.commutative(padding,last) : {Nat.is_le(Nat.add(Nat.div(_,stride),1n+U32.to_nat(k)),1n+origin) == True{} : Bool}
  Order.add_le(Nat.div(Nat.add(padding,last),stride),1n+Nat.div(padding,stride),1n+U32.to_nat(k),Geometry.quotient_origin_upper(padding,last,stride,positive,inside))

# A phase that reads a tap uses its first output and all its outputs.
def start_position(+axis: Extents.Axis,+index: Nat,+inside: {Nat.is_lt(index,Extents.count(axis)) == True{} : Bool}) ->
  {Nat.is_lt(Nat.add(U32.to_nat(U32.from_nat(index)),0n),Extents.count(axis)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(U32.to_nat(U32.from_nat(index)),0n),U32.to_nat(U32.from_nat(index)),Addition.zero_right(U32.to_nat(U32.from_nat(index)))) : {Nat.is_lt(_,Extents.count(axis)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,Extents.index_exact(axis,index,inside)) : {Nat.is_lt(_,Extents.count(axis)) == True{} : Bool}
  inside
def start_used(+axis: Extents.Axis,+index: Nat,+inside: {Nat.is_lt(index,Extents.count(axis)) == True{} : Bool},+active: {Extents.used(axis,U32.from_nat(index)) == True{} : Bool}) ->
  {used(starts(axis),U32.from_nat(index)) == True{} : Bool}:
  Extents.Axis{call,size,size_room,envelope} = axis
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +stride_positive = Domain.stride_positive(call)
  +native = Kernel.axis(U32.from_nat(index),size,k,s,p)
  +last = U32.min(s,size)
  %Equal.sym(Bool,Samples.native_used(native,s,0,last),Axes.band_sample_used(Axes.band_sample_from(native,k,s,0,last)),Samples.used_matches(native,k,s,0,last)) : {_ == True{} : Bool}
  Samples.member_used(native,s,0,last,limits,Order.zero_le(U32.to_nat(Machine.dimension(limits))),Extents.count_room(axis),Domain.stride_room(call),stride_positive,0n,
    Extents.count_positive(axis,index,inside),Order.zero_le(Bands.position(Samples.model(native,s,0,last,stride_positive),0n)),start_position(axis,index,inside),active)
def whole_used(+axis: Extents.Axis,+index: Nat,+inside: {Nat.is_lt(index,Extents.count(axis)) == True{} : Bool},+active: {Extents.used(axis,U32.from_nat(index)) == True{} : Bool}) ->
  {used(whole(axis),U32.from_nat(index)) == True{} : Bool}:
  Extents.Axis{call,size,size_room,envelope} = axis
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +stride_positive = Domain.stride_positive(call)
  +native = Kernel.axis(U32.from_nat(index),size,k,s,p)
  +position = {%Equal.sym(Nat,Nat.add(U32.to_nat(U32.from_nat(index)),0n),U32.to_nat(U32.from_nat(index)),Addition.zero_right(U32.to_nat(U32.from_nat(index)))) : {Nat.is_lt(_,U32.to_nat(size)) == True{} : Bool}
    Extents.index_inside_extent(axis,index,inside) : {Nat.is_lt(Nat.add(U32.to_nat(U32.from_nat(index)),0n),U32.to_nat(size)) == True{} : Bool}}
  %Equal.sym(Bool,Samples.native_used(native,s,0,size),Axes.band_sample_used(Axes.band_sample_from(native,k,s,0,size)),Samples.used_matches(native,k,s,0,size)) : {_ == True{} : Bool}
  Samples.member_used(native,s,0,size,limits,Order.zero_le(U32.to_nat(Machine.dimension(limits))),size_room,Domain.stride_room(call),stride_positive,0n,
    Extents.count_positive(axis,index,inside),Order.zero_le(Bands.position(Samples.model(native,s,0,size,stride_positive),0n)),position,active)

# A tapped coordinate of a used phase has an exact local tile index.
def relative(+query: Query,+position: U32) -> U32:
  (position - low_fold(query) : U32)
def difference(+low: U32,+high: U32) -> U32:
  Bool.pick(U32,U32.is_lt(low,high),(high - low : U32),0)
def value(query: Query) -> U32:
  +query = query
  difference(low_fold(query),high_fold(query))
def relative_inside(+query: Query,+position: U32,
  +lower: {U32.is_le(low_fold(query),position) == True{} : Bool},
  +upper: {U32.is_lt(position,high_fold(query)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(relative(query,position)),U32.to_nat(value(query))) == True{} : Bool}:
  +low = low_fold(query)
  +high = high_fold(query)
  +ordered = Order.le_lt_trans(U32.to_nat(low),U32.to_nat(position),U32.to_nat(high),Bounds.as_nat_le(low,position,lower),Bounds.as_nat_lt(position,high,upper))
  +active = {%Equal.sym(Bool,U32.is_lt(low,high),Nat.is_lt(U32.to_nat(low),U32.to_nat(high)),Comparison.lt(low,high)) : {_ == True{} : Bool} ordered : {U32.is_lt(low,high) == True{} : Bool}}
  %Equal.sym(Bool,U32.is_lt(low,high),True{},active) : {Nat.is_lt(U32.to_nat(relative(query,position)),U32.to_nat(Bool.pick(U32,_,(high - low : U32),0))) == True{} : Bool}
  FullRange.interval_inside(low,high,position,lower,upper)

def interval(+query: Query,+position: U32,
  bounds: Shape.Both<{U32.is_le(low_fold(query),position) == True{} : Bool},{U32.is_lt(position,high_fold(query)) == True{} : Bool}>) ->
  {Nat.is_lt(U32.to_nat((position - low_fold(query) : U32)),U32.to_nat((high_fold(query) - low_fold(query) : U32))) == True{} : Bool}:
  Shape.Both{+lower,+upper} = bounds
  FullRange.interval_inside(low_fold(query),high_fold(query),position,lower,upper)

def selection_low_matches(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) ->
  {Axes.band_sample_low(Samples.selection_sample(selection(query,index,inside,enabled))) == Axes.band_low(context(query),U32.from_nat(index)) : U32}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}
def selection_high_matches(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool}) ->
  {Axes.band_sample_high(Samples.selection_sample(selection(query,index,inside,enabled))) == Axes.band_high(context(query),U32.from_nat(index)) : U32}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}
def tapped_fold_bounds_from(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool},+position: U32,+tap: U32,
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(Axes.band_sample_low(Samples.selection_sample(selection(query,index,inside,enabled)))),Samples.tapped_coordinate(selection(query,index,inside,enabled),position,tap)) == True{} : Bool},
    {Nat.is_lt(Samples.tapped_coordinate(selection(query,index,inside,enabled),position,tap),U32.to_nat(Axes.band_sample_high(Samples.selection_sample(selection(query,index,inside,enabled))))) == True{} : Bool}>) ->
  Shape.Both<{Nat.is_le(U32.to_nat(low_fold(query)),Samples.tapped_coordinate(selection(query,index,inside,enabled),position,tap)) == True{} : Bool},
    {Nat.is_lt(Samples.tapped_coordinate(selection(query,index,inside,enabled),position,tap),U32.to_nat(high_fold(query))) == True{} : Bool}>:
  Shape.Both{+lo,+hi} = bounds
  +sample = Samples.selection_sample(selection(query,index,inside,enabled))
  Shape.Both{
    Order.transitive(U32.to_nat(low_fold(query)),U32.to_nat(Axes.band_sample_low(sample)),Samples.tapped_coordinate(selection(query,index,inside,enabled),position,tap),
      %Equal.sym(U32,Axes.band_sample_low(sample),Axes.band_low(context(query),U32.from_nat(index)),selection_low_matches(query,index,inside,enabled)) : {Nat.is_le(U32.to_nat(low_fold(query)),U32.to_nat(_)) == True{} : Bool}
        Bounds.as_nat_le(low_fold(query),Axes.band_low(context(query),U32.from_nat(index)),contains_low(query,index,inside,enabled)),lo),
    Bounds.nat_trans(Samples.tapped_coordinate(selection(query,index,inside,enabled),position,tap),U32.to_nat(Axes.band_sample_high(sample)),U32.to_nat(high_fold(query)),hi,
      %Equal.sym(U32,Axes.band_sample_high(sample),Axes.band_high(context(query),U32.from_nat(index)),selection_high_matches(query,index,inside,enabled)) : {Nat.is_le(U32.to_nat(_),U32.to_nat(high_fold(query))) == True{} : Bool}
        Bounds.as_nat_le(Axes.band_high(context(query),U32.from_nat(index)),high_fold(query),contains_high(query,index,inside,enabled)))}

def tapped_native_bounds_from(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool},+position: U32,+tap: U32,
  +exact: {U32.to_nat(Samples.native_tapped(selection(query,index,inside,enabled),position,tap)) == Samples.tapped_coordinate(selection(query,index,inside,enabled),position,tap) : Nat},
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(low_fold(query)),Samples.tapped_coordinate(selection(query,index,inside,enabled),position,tap)) == True{} : Bool},
    {Nat.is_lt(Samples.tapped_coordinate(selection(query,index,inside,enabled),position,tap),U32.to_nat(high_fold(query))) == True{} : Bool}>) ->
  Shape.Both<{U32.is_le(low_fold(query),Samples.native_tapped(selection(query,index,inside,enabled),position,tap)) == True{} : Bool},
    {U32.is_lt(Samples.native_tapped(selection(query,index,inside,enabled),position,tap),high_fold(query)) == True{} : Bool}>:
  Shape.Both{+lo,+hi} = bounds
  +word = Samples.native_tapped(selection(query,index,inside,enabled),position,tap)
  +natural = Samples.tapped_coordinate(selection(query,index,inside,enabled),position,tap)
  Shape.Both{
    %Equal.sym(Bool,U32.is_le(low_fold(query),word),Nat.is_le(U32.to_nat(low_fold(query)),U32.to_nat(word)),Comparison.le(low_fold(query),word)) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(word),natural,exact) : {Nat.is_le(U32.to_nat(low_fold(query)),_) == True{} : Bool} lo,
    %Equal.sym(Bool,U32.is_lt(word,high_fold(query)),Nat.is_lt(U32.to_nat(word),U32.to_nat(high_fold(query))),Comparison.lt(word,high_fold(query))) : {_ == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(word),natural,exact) : {Nat.is_lt(_,U32.to_nat(high_fold(query))) == True{} : Bool} hi}
def tapped_native_bounds(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool},+position: U32,+tap: U32,
  +in_phase: {Nat.is_lt(U32.to_nat(position),U32.to_nat(Phase.axis_count(Samples.selection_axis(selection(query,index,inside,enabled))))) == True{} : Bool},
  +lower: {Nat.is_le(Bands.low(Samples.selection_model(selection(query,index,inside,enabled))),U32.to_nat(position)) == True{} : Bool},
  +upper: {Nat.is_lt(U32.to_nat(position),Bands.high(Samples.selection_model(selection(query,index,inside,enabled)))) == True{} : Bool},
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),Samples.selection_taps(selection(query,index,inside,enabled))) == True{} : Bool}) ->
  Shape.Both<{U32.is_le(low_fold(query),Samples.native_tapped(selection(query,index,inside,enabled),position,tap)) == True{} : Bool},
    {U32.is_lt(Samples.native_tapped(selection(query,index,inside,enabled),position,tap),high_fold(query)) == True{} : Bool}>:
  +selected = selection(query,index,inside,enabled)
  tapped_native_bounds_from(query,index,inside,enabled,position,tap,Samples.native_tapped_exact(selected,position,tap,in_phase,upper,tap_inside),tapped_fold_bounds_from(query,index,inside,enabled,position,tap,Samples.tapped_bounds(selected,position,tap,lower,upper,tap_inside)))

# The row band of a call: output rows [first, first + count). The row lemmas
# take a band whose axis has the height of its geometry.
def caller_query(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool}) -> Query:
  +geometry = Domain.geometry(call)
  +limits = Domain.limits(call)
  +last = (first + count : U32)
  +sum = Nat.add(U32.to_nat(first),U32.to_nat(count))
  +exact = Arithmetic.add(first,count,Machine.dimension(limits),Limits.dimension_fits(limits),
    Order.transitive(sum,Domain.height(geometry),U32.to_nat(Machine.dimension(limits)),room,Domain.height_room(call)))
  Query{Extents.rows(call,envelope),first,last,
    %Equal.sym(Nat,U32.to_nat(last),sum,exact) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool} Order.le_add(U32.to_nat(first),U32.to_nat(count)),
    %Equal.sym(Nat,U32.to_nat(last),sum,exact) : {Nat.is_le(_,U32.to_nat(Domain.height_word(geometry))) == True{} : Bool} Extents.height_bound(geometry,sum,room)}

def actual_low_matches(+query: Query,+row: {Extents.extent(axis(query)) == Domain.height_word(geometry(query)) : U32}) ->
  {Kernel.band_low(geometry(query),first(query),last(query)) == low_fold(query) : U32}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  %row : {Kernel.band_fold(Kernel.Band{_,k,s,p,first,last,0},4294967295) == low_fold(query) : U32}
  %Equal.sym(U32,seed(query),Machine.word_max(),seed_default(query)) :
    {Kernel.band_fold(Kernel.Band{size,k,s,p,first,last,0},4294967295) == Traversal.indexed_fold(~U32,~Kernel.Band,~Kernel.band_phase,count(query),0,context(query),_) : U32}
  {==}
def actual_high_matches(+query: Query,+row: {Extents.extent(axis(query)) == Domain.height_word(geometry(query)) : U32}) ->
  {Kernel.band_high(geometry(query),first(query),last(query)) == high_fold(query) : U32}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  %row : {Kernel.band_fold(Kernel.Band{_,k,s,p,first,last,1},0) == high_fold(query) : U32}
  {==}
def rows_word(query: Query) -> U32:
  +query = query
  (last(query) - first(query) : U32)
def rows_exact(+query: Query) -> {U32.to_nat(rows_word(query)) == rows(query) : Nat}:
  Query{axis,+first,+last,+ordered,room} = query
  FullRange.subtract(last,first,last,ordered,Order.sub_le(U32.to_nat(last),U32.to_nat(first)))

def value_matches(+query: Query,+row: {Extents.extent(axis(query)) == Domain.height_word(geometry(query)) : U32}) ->
  {Kernel.band_rows(geometry(query),first(query),rows_word(query)) == value(query) : U32}:
  +geometry = geometry(query)
  +first = first(query)
  +last = last(query)
  +end = (first + rows_word(query) : U32)
  +end_equal = Equal.trans(U32,end,(rows_word(query) + first : U32),last,Arithmetic.add_commutative(first,rows_word(query)),Subtraction.add_subtract(last,first))
  +native_low = Kernel.band_low(geometry,first,last)
  +native_high = Kernel.band_high(geometry,first,last)
  Equal.trans(U32,Kernel.band_rows(geometry,first,rows_word(query)),difference(native_low,native_high),value(query),
    Equal.cong(U32,U32,end => difference(Kernel.band_low(geometry,first,end),Kernel.band_high(geometry,first,end)),end,last,end_equal),
    Equal.trans(U32,difference(native_low,native_high),difference(low_fold(query),native_high),value(query),
      Equal.cong(U32,U32,low => difference(low,native_high),native_low,low_fold(query),actual_low_matches(query,row)),
      Equal.cong(U32,U32,high => difference(low_fold(query),high),native_high,high_fold(query),actual_high_matches(query,row))))

def rows_value_bounded_case(ordered: Bool,+query: Query,+compared: {U32.is_lt(low_fold(query),high_fold(query)) == ordered : Bool}) ->
  {Nat.is_le(U32.to_nat(Bool.pick(U32,ordered,(high_fold(query) - low_fold(query) : U32),0)),Nat.add(Nat.sub(rows(query),1n),Geometry.halo(geometry(query)))) == True{} : Bool}:
  match ordered:
    case False{}: Order.zero_le(Nat.add(Nat.sub(rows(query),1n),Geometry.halo(geometry(query))))
    case True{}:
      span_bounded(query,Nat.sub(rows(query),1n),rows_room(query),Bounds.strict_le(U32.to_nat(low_fold(query)),U32.to_nat(high_fold(query)),Bounds.as_nat_lt(low_fold(query),high_fold(query),compared)))

# No used-phase witness is needed: an empty band has zero tile rows.
def band_rows_bounded(+query: Query,+row: {Extents.extent(axis(query)) == Domain.height_word(geometry(query)) : U32}) ->
  {Nat.is_le(U32.to_nat(Kernel.band_rows(geometry(query),first(query),rows_word(query))),Nat.add(Nat.sub(U32.to_nat(rows_word(query)),1n),Geometry.halo(geometry(query)))) == True{} : Bool}:
  %Equal.sym(U32,Kernel.band_rows(geometry(query),first(query),rows_word(query)),value(query),value_matches(query,row)) : {Nat.is_le(U32.to_nat(_),Nat.add(Nat.sub(U32.to_nat(rows_word(query)),1n),Geometry.halo(geometry(query)))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(rows_word(query)),rows(query),rows_exact(query)) : {Nat.is_le(U32.to_nat(value(query)),Nat.add(Nat.sub(_,1n),Geometry.halo(geometry(query)))) == True{} : Bool}
  rows_value_bounded_case(U32.is_lt(low_fold(query),high_fold(query)),query,{==})

def caller_count_matches(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool}) ->
  {rows_word(caller_query(call,envelope,first,count,room)) == count : U32}:
  Domain.Call{geometry,batch,limits,facts} = call
  Equal.trans(U32,((first + count : U32) - first : U32),((count + first : U32) - first : U32),count,
    Equal.cong(U32,U32,value => (value - first : U32),(first + count : U32),(count + first : U32),Arithmetic.add_commutative(first,count)),Subtraction.subtract_add(count,first))

def contained_band_rows(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+first: U32,+count: U32,
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Kernel.band_rows(Domain.geometry(call),first,count)),Nat.add(Nat.sub(U32.to_nat(count),1n),Geometry.halo(Domain.geometry(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  +query = caller_query(call,envelope,first,count,room)
  %caller_count_matches(call,envelope,first,count,room) :
    {Nat.is_le(U32.to_nat(Kernel.band_rows(geometry,first,_)),Nat.add(Nat.sub(U32.to_nat(_),1n),Geometry.halo(geometry))) == True{} : Bool}
  band_rows_bounded(query,{==})

def tapped_relative_finish(+query: Query,+row: {Extents.extent(axis(query)) == Domain.height_word(geometry(query)) : U32},
  +index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool},+position: U32,+tap: U32,
  bounds: Shape.Both<{U32.is_le(low_fold(query),Samples.native_tapped(selection(query,index,inside,enabled),position,tap)) == True{} : Bool},
    {U32.is_lt(Samples.native_tapped(selection(query,index,inside,enabled),position,tap),high_fold(query)) == True{} : Bool}>) ->
  {Nat.is_lt(U32.to_nat(relative(query,Samples.native_tapped(selection(query,index,inside,enabled),position,tap))),U32.to_nat(Kernel.band_rows(geometry(query),first(query),rows_word(query)))) == True{} : Bool}:
  Shape.Both{+lo,+hi} = bounds
  %Equal.sym(U32,Kernel.band_rows(geometry(query),first(query),rows_word(query)),value(query),value_matches(query,row)) : {Nat.is_lt(U32.to_nat(relative(query,Samples.native_tapped(selection(query,index,inside,enabled),position,tap))),U32.to_nat(_)) == True{} : Bool}
  relative_inside(query,Samples.native_tapped(selection(query,index,inside,enabled),position,tap),lo,hi)

def relative_native_matches(+query: Query,+row: {Extents.extent(axis(query)) == Domain.height_word(geometry(query)) : U32},
  +index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},+enabled: {used(query,U32.from_nat(index)) == True{} : Bool},+position: U32,+tap: U32) ->
  {(((position + Geometry.axis_low(geometry(query),U32.from_nat(index),Extents.extent(axis(query)))) - Kernel.band_low(geometry(query),first(query),last(query))) + tap : U32) == relative(query,Samples.native_tapped(selection(query,index,inside,enabled),position,tap)) : U32}:
  Query{Extents.Axis{call,size,size_room,envelope},first,last,ordered,room} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +coordinate = (position + Geometry.axis_low(geometry,U32.from_nat(index),size) : U32)
  +low = Kernel.band_low(geometry,first,last)
  Equal.trans(U32,((coordinate - low) + tap : U32),((coordinate + tap) - low : U32),relative(query,Samples.native_tapped(selection(query,index,inside,enabled),position,tap)),Arithmetic.add_after_subtract(coordinate,low,tap),
    Equal.cong(U32,U32,low => ((coordinate + tap) - low : U32),low,low_fold(query),actual_low_matches(query,row)))
