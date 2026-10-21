# The column origin, shifts and groups of the tiling are band extrema over the
# columns: the first output of every column phase, and every output column.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ../traversal_array.bend as Traversal
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_geometry.bend as Geometry
import ./input_gradient_tiled_extents.bend as Extents
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_band_geometry.bend as Samples
import ./input_gradient_tiled_bands.bend as BandModel
import ./nat_ceiling.bend as Ceiling
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./traversal_extrema.bend as Folds
import ./nat_order.bend as Order
import ./traversal_array_refinement.bend as Arrays

def layout_column_low(layout: Kernel.Tiling) -> U32:
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  column_low
def layout_shifts(layout: Kernel.Tiling) -> U32:
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  shifts

def starts(+call: Domain.Call,+envelope: Geometry.WordEnvelope) -> Bands.Query:
  Bands.starts(Extents.columns(call,envelope))
def whole(+call: Domain.Call,+envelope: Geometry.WordEnvelope) -> Bands.Query:
  Bands.whole(Extents.columns(call,envelope))

def column_low_matches(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {layout_column_low(Kernel.tiling(Domain.geometry(call))) == Bands.low_fold(starts(call,envelope)) : U32}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +query = starts(call,envelope)
  %Equal.sym(U32,Bands.seed(query),Machine.word_max(),Bands.seed_default(query)) :
    {layout_column_low(Kernel.tiling(geometry)) == Traversal.indexed_fold(~U32,~Kernel.Band,~Kernel.band_phase,Bands.count(query),0,Bands.context(query),_) : U32}
  {==}
def column_shifts_matches(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {layout_shifts(Kernel.tiling(Domain.geometry(call))) == (Bands.high_fold(starts(call,envelope)) - Bands.low_fold(starts(call,envelope)) : U32) : U32}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +query = starts(call,envelope)
  %Equal.sym(U32,Bands.seed(query),Machine.word_max(),Bands.seed_default(query)) :
    {layout_shifts(Kernel.tiling(geometry)) == (Bands.high_fold(query) - Traversal.indexed_fold(~U32,~Kernel.Band,~Kernel.band_phase,Bands.count(query),0,Bands.context(query),_) : U32) : U32}
  {==}

# The low and high folds of the first outputs through the filter of used phases.
def low_filtered(+call: Domain.Call,+envelope: Geometry.WordEnvelope) -> U32:
  Traversal.indexed_fold(~U32,~Bands.Query,~Folds.filtered(~False{},~Bands.Query,~Bands.used,~Bands.sampled(~False{})),Bands.count(starts(call,envelope)),0,starts(call,envelope),
    Bands.seed(starts(call,envelope)))
def high_filtered(+call: Domain.Call,+envelope: Geometry.WordEnvelope) -> U32:
  Traversal.indexed_fold(~U32,~Bands.Query,~Folds.filtered(~True{},~Bands.Query,~Bands.used,~Bands.sampled(~True{})),Bands.count(starts(call,envelope)),0,starts(call,envelope),0)

# A used column phase bounds the shifts by the band of the first outputs.
def shifts_enabled(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +found: Folds.Enabled<Bands.Query,Bands.used,starts(call,envelope),Bands.count(starts(call,envelope))>) ->
  {Nat.is_le(U32.to_nat(layout_shifts(Kernel.tiling(Domain.geometry(call)))),1n+Geometry.halo(Domain.geometry(call))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Folds.Enabled{+index,+inside,+enabled} = found
  +axis = Extents.columns(call,envelope)
  +query = Bands.starts(axis)
  %Equal.sym(U32,layout_shifts(Kernel.tiling(geometry)),(Bands.high_fold(query) - Bands.low_fold(query) : U32),column_shifts_matches(call,envelope)) :
    {Nat.is_le(U32.to_nat(_),1n+Geometry.halo(geometry)) == True{} : Bool}
  Bands.span_bounded(query,1n,Bands.starts_room(axis),Bands.extrema_ordered(query,index,inside,enabled))

# Without one both folds keep their seeds and the shifts are 1.
def shifts_initial(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +high: {high_filtered(call,envelope) == 0 : U32},+low: {low_filtered(call,envelope) == Bands.seed(starts(call,envelope)) : U32}) ->
  {Nat.is_le(U32.to_nat(layout_shifts(Kernel.tiling(Domain.geometry(call)))),1n+Geometry.halo(Domain.geometry(call))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  +query = starts(call,envelope)
  +halo = Geometry.halo(geometry)
  %Equal.sym(U32,layout_shifts(Kernel.tiling(geometry)),(Bands.high_fold(query) - Bands.low_fold(query) : U32),column_shifts_matches(call,envelope)) :
    {Nat.is_le(U32.to_nat(_),1n+halo) == True{} : Bool}
  %Arrays.indexed_fold_change_context(~U32,~Bands.Query,~Kernel.Band,~Bands.higher_context,~Folds.filtered(~True{},~Bands.Query,~Bands.used,~Bands.sampled(~True{})),~Kernel.band_phase,
    ~Bands.higher_step_matches,Bands.count(query),0,query,0) : {Nat.is_le(U32.to_nat((_ - Bands.low_fold(query) : U32)),1n+halo) == True{} : Bool}
  %Equal.sym(U32,high_filtered(call,envelope),0,high) : {Nat.is_le(U32.to_nat((_ - Bands.low_fold(query) : U32)),1n+halo) == True{} : Bool}
  %Arrays.indexed_fold_change_context(~U32,~Bands.Query,~Kernel.Band,~Bands.context,~Folds.filtered(~False{},~Bands.Query,~Bands.used,~Bands.sampled(~False{})),~Kernel.band_phase,
    ~Bands.lower_step_matches,Bands.count(query),0,query,Bands.seed(query)) : {Nat.is_le(U32.to_nat((0 - _ : U32)),1n+halo) == True{} : Bool}
  %Equal.sym(U32,low_filtered(call,envelope),Bands.seed(query),low) : {Nat.is_le(U32.to_nat((0 - _ : U32)),1n+halo) == True{} : Bool}
  %Equal.sym(U32,Bands.seed(query),Machine.word_max(),Bands.seed_default(query)) : {Nat.is_le(U32.to_nat((0 - _ : U32)),1n+halo) == True{} : Bool}
  Order.zero_le(halo)

def shifts_low(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+high: {high_filtered(call,envelope) == 0 : U32},
  low: Either<&2,&2,{low_filtered(call,envelope) == Bands.seed(starts(call,envelope)) : U32},Folds.Enabled<Bands.Query,Bands.used,starts(call,envelope),Bands.count(starts(call,envelope))>>) ->
  {Nat.is_le(U32.to_nat(layout_shifts(Kernel.tiling(Domain.geometry(call)))),1n+Geometry.halo(Domain.geometry(call))) == True{} : Bool}:
  match low:
    case Inl{kept}: shifts_initial(call,envelope,high,kept)
    case Inr{found}: shifts_enabled(call,envelope,found)

def shifts_high(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  high: Either<&2,&2,{high_filtered(call,envelope) == 0 : U32},Folds.Enabled<Bands.Query,Bands.used,starts(call,envelope),Bands.count(starts(call,envelope))>>) ->
  {Nat.is_le(U32.to_nat(layout_shifts(Kernel.tiling(Domain.geometry(call)))),1n+Geometry.halo(Domain.geometry(call))) == True{} : Bool}:
  match high:
    case Inl{kept}:
      shifts_low(call,envelope,kept,Folds.initial_or_enabled(~False{},~Bands.Query,~Bands.used,~Bands.sampled(~False{}),Bands.count(starts(call,envelope)),0n,starts(call,envelope),
        Bands.seed(starts(call,envelope))))
    case Inr{found}: shifts_enabled(call,envelope,found)

# The shifts are at most one more than the halo.
def tiling_shifts_bounded(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {Nat.is_le(U32.to_nat(layout_shifts(Kernel.tiling(Domain.geometry(call)))),1n+Geometry.halo(Domain.geometry(call))) == True{} : Bool}:
  shifts_high(call,envelope,Folds.initial_or_enabled(~True{},~Bands.Query,~Bands.used,~Bands.sampled(~True{}),Bands.count(starts(call,envelope)),0n,starts(call,envelope),0))

# A tap of a used column phase reads a shift inside the tiling.
def column_shift_inside(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+index: Nat,
  +inside: {Nat.is_lt(index,Extents.count(Extents.columns(call,envelope))) == True{} : Bool},
  +used: {Extents.used(Extents.columns(call,envelope),U32.from_nat(index)) == True{} : Bool},+tap: U32,
  +tap_inside: {Nat.is_lt(U32.to_nat(tap),U32.to_nat(Extents.axis_taps(Extents.columns(call,envelope),U32.from_nat(index)))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(((Geometry.axis_low(Domain.geometry(call),U32.from_nat(index),Domain.width_word(Domain.geometry(call))) - layout_column_low(Kernel.tiling(Domain.geometry(call)))) + tap : U32)),U32.to_nat(layout_shifts(Kernel.tiling(Domain.geometry(call))))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +axis = Extents.columns(call,envelope)
  +query = Bands.starts(axis)
  +enabled = Bands.start_used(axis,index,inside,used)
  +selected = Bands.selection(query,index,inside,enabled)
  +base = Geometry.axis_low(geometry,U32.from_nat(index),w)
  +low = Bands.low_fold(query)
  +high = Bands.high_fold(query)
  +lower = {%Equal.sym(Nat,Ceiling.ceiling(0n,U32.to_nat(s)),0n,Samples.zero_ceiling(U32.to_nat(s),Domain.stride_positive(call))) : {Nat.is_le(_,0n) == True{} : Bool} {==} :
    {Nat.is_le(BandModel.low(Samples.selection_model(selected)),U32.to_nat(0)) == True{} : Bool}}
  +upper = BandModel.member_upper(Samples.selection_model(selected),0n,Extents.count_positive(axis,index,inside),Bands.start_position(axis,index,inside))
  +bounds = Bands.tapped_native_bounds(query,index,inside,enabled,0,tap,Extents.count_positive(axis,index,inside),lower,upper,tap_inside)
  %Equal.sym(U32,layout_shifts(Kernel.tiling(geometry)),(high - low : U32),column_shifts_matches(call,envelope)) :
    {Nat.is_lt(U32.to_nat(((base - layout_column_low(Kernel.tiling(geometry))) + tap : U32)),U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(U32,layout_column_low(Kernel.tiling(geometry)),low,column_low_matches(call,envelope)) :
    {Nat.is_lt(U32.to_nat(((base - _) + tap : U32)),U32.to_nat((high - low : U32))) == True{} : Bool}
  %Equal.sym(U32,((base - low) + tap : U32),((base + tap) - low : U32),Arithmetic.add_after_subtract(base,low,tap)) :
    {Nat.is_lt(U32.to_nat(_),U32.to_nat((high - low : U32))) == True{} : Bool}
  %Arithmetic.add_zero_left(base) : {Nat.is_lt(U32.to_nat(((_ + tap) - low : U32)),U32.to_nat((high - low : U32))) == True{} : Bool}
  Bands.interval(query,Samples.native_tapped(selected,0,tap),bounds)

# The groups cover the most outputs of a used column phase.
def maximum_count_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {Nat.is_le(U32.to_nat(Bands.count_fold(whole(call,envelope))),Domain.width(Domain.geometry(call))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Bands.count_bound(whole(call,envelope))
def selected_count(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+index: Nat,
  +inside: {Nat.is_lt(index,Extents.count(Extents.columns(call,envelope))) == True{} : Bool},
  +used: {Extents.used(Extents.columns(call,envelope),U32.from_nat(index)) == True{} : Bool}) ->
  {U32.is_le(Extents.axis_count(Extents.columns(call,envelope),U32.from_nat(index)),Bands.count_fold(whole(call,envelope))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +axis = Extents.columns(call,envelope)
  Bands.contains_count(Bands.whole(axis),index,inside,Bands.whole_used(axis,index,inside,used))
