# Native group and chunk coordinates refine the shared ceiling partition.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ../kernel_pipeline.bend as Pipeline
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_extents.bend as Extents
import ./input_gradient_tiled_columns.bend as Columns
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_capacity.bend as Capacity
import ./input_gradient_tiled_geometry.bend as Geometry
import ./nat_division_order.bend as NaturalDivision
import ./input_gradient_tiled_chunks.bend as Chunks
import ./tensor_allocation_depth.bend as Depth
import ./input_gradient_tiled_leaf_execution.bend as Execution
import ./row_major_index.bend as RowMajor
import ./nat_ceiling.bend as Ceiling
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_division.bend as Division
import ./word_subtraction.bend as Subtraction
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./natural_addition.bend as Addition
import ./machine_limit_bounds.bend as Limits
import ./encoded_coordinates.bend as Encoded
import ./u32_extrema.bend as NativeExtrema
import ./nat_extrema.bend as Extrema

def groups_expression(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {Capacity.layout_groups(Kernel.tiling(Domain.geometry(call))) == ((Bands.count_fold(Columns.whole(call,envelope)) + 7) / 8 : U32) : U32}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}

def quarter_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {Nat.is_le(U32.to_nat(Bands.count_fold(Columns.whole(call,envelope))),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Order.transitive(U32.to_nat(Bands.count_fold(Columns.whole(call,envelope))),U32.to_nat(w),U32.to_nat(Capacity.quarter(envelope)),Columns.maximum_count_room(call,envelope),
    Order.transitive(U32.to_nat(w),U32.to_nat(Machine.dimension(limits)),U32.to_nat(Capacity.quarter(envelope)),Domain.width_room(call),Capacity.dimension_quarter_room(limits,envelope)))

def group_numerator_room(+value: U32,+envelope: Geometry.WordEnvelope,+room: {Nat.is_le(U32.to_nat(value),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(value),7n),U32.to_nat(Capacity.half(envelope))) == True{} : Bool}:
  Geometry.WordEnvelope{maximum,+half,+quarter,margin,me,he,+qe,re} = envelope
  +seven = Bounds.as_nat_le(7,quarter,%Equal.sym(U32,quarter,1073741824,qe) : {U32.is_le(7,_) == True{} : Bool} {==})
  %Equal.sym(Nat,U32.to_nat(half),Nat.double(U32.to_nat(quarter)),Capacity.quarter_doubled(envelope)) : {Nat.is_le(Nat.add(U32.to_nat(value),7n),_) == True{} : Bool}
  %Addition.double(U32.to_nat(quarter)) : {Nat.is_le(Nat.add(U32.to_nat(value),7n),_) == True{} : Bool}
  Encoded.sum_bounds(U32.to_nat(value),7n,U32.to_nat(quarter),U32.to_nat(quarter),room,seven)

def eight_double_room(+envelope: Geometry.WordEnvelope) -> {Nat.is_le(Nat.double(U32.to_nat(8)),U32.to_nat(Capacity.half(envelope))) == True{} : Bool}:
  Geometry.WordEnvelope{maximum,half,quarter,margin,me,he,qe,re} = envelope
  Bounds.as_nat_le(16,half,%Equal.sym(U32,half,2147483648,he) : {U32.is_le(16,_) == True{} : Bool} {==})

def groups_exact(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {U32.to_nat(Capacity.layout_groups(Kernel.tiling(Domain.geometry(call)))) == Ceiling.ceiling(U32.to_nat(Bands.count_fold(Columns.whole(call,envelope))),8n) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  +value = Bands.count_fold(Columns.whole(call,envelope))
  +half = Capacity.half(envelope)
  +numerator = (value + 7 : U32)
  +exact = Arithmetic.add(value,7,half,Capacity.half_fits(envelope),group_numerator_room(value,envelope,quarter_room(call,envelope)))
  +divisor_room = eight_double_room(envelope)
  Equal.trans(Nat,U32.to_nat(Capacity.layout_groups(Kernel.tiling(geometry))),U32.to_nat((numerator / 8 : U32)),Ceiling.ceiling(U32.to_nat(value),8n),Equal.cong(U32,Nat,U32.to_nat,Capacity.layout_groups(Kernel.tiling(geometry)),(numerator / 8 : U32),groups_expression(call,envelope)),
    Equal.trans(Nat,U32.to_nat((numerator / 8 : U32)),Nat.div(Nat.add(U32.to_nat(value),7n),8n),Ceiling.ceiling(U32.to_nat(value),8n),
      %exact : {U32.to_nat((numerator / 8 : U32)) == Nat.div(_,8n) : Nat} Division.divide(numerator,8,half,Capacity.half_fits(envelope),divisor_room,{==}),
      Ceiling.shifted_formula(U32.to_nat(value),8n,{==})))

def groups_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {Nat.is_le(U32.to_nat(Capacity.layout_groups(Kernel.tiling(Domain.geometry(call)))),Domain.width(Domain.geometry(call))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +value = Bands.count_fold(Columns.whole(call,envelope))
  %Equal.sym(Nat,U32.to_nat(Capacity.layout_groups(Kernel.tiling(geometry))),Ceiling.ceiling(U32.to_nat(value),8n),groups_exact(call,envelope)) : {Nat.is_le(_,U32.to_nat(w)) == True{} : Bool}
  Order.transitive(Ceiling.ceiling(U32.to_nat(value),8n),U32.to_nat(value),U32.to_nat(w),Ceiling.bounded(U32.to_nat(value),8n,{==}),Columns.maximum_count_room(call,envelope))

# Both arguments of a native ceiling fit inside the quarter-word envelope.
def ceiling_numerator(+value: U32,+span: U32,+envelope: Geometry.WordEnvelope,
  +value_room: {Nat.is_le(U32.to_nat(value),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool},
  +span_room: {Nat.is_le(U32.to_nat(span),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool}) ->
  {U32.to_nat((value + span - 1 : U32)) == Nat.add(U32.to_nat(value),Nat.sub(U32.to_nat(span),1n)) : Nat}:
  +quarter = Capacity.quarter(envelope)
  +half = Capacity.half(envelope)
  +sum = Nat.add(U32.to_nat(value),U32.to_nat(span))
  +room = {%Equal.sym(Nat,U32.to_nat(half),Nat.double(U32.to_nat(quarter)),Capacity.quarter_doubled(envelope)) : {Nat.is_le(sum,_) == True{} : Bool}
    %Addition.double(U32.to_nat(quarter)) : {Nat.is_le(sum,_) == True{} : Bool}
    Encoded.sum_bounds(U32.to_nat(value),U32.to_nat(span),U32.to_nat(quarter),U32.to_nat(quarter),value_room,span_room) : {Nat.is_le(sum,U32.to_nat(half)) == True{} : Bool}}
  +exact = Arithmetic.add(value,span,half,Capacity.half_fits(envelope),room)
  +sum_positive = Bounds.nat_trans(0n,U32.to_nat(span),sum,positive,
    %Order.commutative(U32.to_nat(span),U32.to_nat(value)) : {Nat.is_le(U32.to_nat(span),_) == True{} : Bool} Order.le_add(U32.to_nat(span),U32.to_nat(value)))
  Equal.trans(Nat,U32.to_nat((value + span - 1 : U32)),Nat.sub(sum,1n),Nat.add(U32.to_nat(value),Nat.sub(U32.to_nat(span),1n)),
    %exact : {U32.to_nat((value + span - 1 : U32)) == Nat.sub(_,1n) : Nat}
      Arithmetic.subtract((value + span : U32),1,half,Capacity.half_fits(envelope),
        %Equal.sym(Nat,U32.to_nat((value + span : U32)),sum,exact) : {Nat.is_le(1n,_) == True{} : Bool} Order.positive_one(sum,sum_positive),
        %Equal.sym(Nat,U32.to_nat((value + span : U32)),sum,exact) : {Nat.is_le(Nat.sub(_,1n),U32.to_nat(half)) == True{} : Bool} Order.transitive(Nat.sub(sum,1n),sum,U32.to_nat(half),Order.sub_le(sum,1n),room)),
    Order.subtract_from_sum(U32.to_nat(value),U32.to_nat(span),1n,Order.positive_one(U32.to_nat(span),positive)))

def ceiling_exact(+value: U32,+span: U32,+envelope: Geometry.WordEnvelope,
  +value_room: {Nat.is_le(U32.to_nat(value),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool},
  +span_room: {Nat.is_le(U32.to_nat(span),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(span)) == True{} : Bool}) ->
  {U32.to_nat(((value + span - 1) / span : U32)) == Ceiling.ceiling(U32.to_nat(value),U32.to_nat(span)) : Nat}:
  +numerator = (value + span - 1 : U32)
  +half = Capacity.half(envelope)
  +double_room = {%Equal.sym(Nat,U32.to_nat(half),Nat.double(U32.to_nat(Capacity.quarter(envelope))),Capacity.quarter_doubled(envelope)) : {Nat.is_le(Nat.double(U32.to_nat(span)),_) == True{} : Bool}
    Order.double_le(U32.to_nat(span),U32.to_nat(Capacity.quarter(envelope)),span_room) : {Nat.is_le(Nat.double(U32.to_nat(span)),U32.to_nat(half)) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat((numerator / span : U32)),Nat.div(U32.to_nat(numerator),U32.to_nat(span)),Ceiling.ceiling(U32.to_nat(value),U32.to_nat(span)),Division.divide(numerator,span,half,Capacity.half_fits(envelope),double_room,positive),
    Equal.trans(Nat,Nat.div(U32.to_nat(numerator),U32.to_nat(span)),Nat.div(Nat.add(U32.to_nat(value),Nat.sub(U32.to_nat(span),1n)),U32.to_nat(span)),Ceiling.ceiling(U32.to_nat(value),U32.to_nat(span)),
      Equal.cong(Nat,Nat,value => Nat.div(value,U32.to_nat(span)),U32.to_nat(numerator),Nat.add(U32.to_nat(value),Nat.sub(U32.to_nat(span),1n)),ceiling_numerator(value,span,envelope,value_room,span_room,positive)),Ceiling.shifted_formula(U32.to_nat(value),U32.to_nat(span),positive)))

def groups_quarter(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {Nat.is_le(U32.to_nat(Capacity.layout_groups(Kernel.tiling(Domain.geometry(call)))),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Order.transitive(U32.to_nat(Capacity.layout_groups(Kernel.tiling(geometry))),U32.to_nat(w),U32.to_nat(Capacity.quarter(envelope)),groups_room(call,envelope),
    Order.transitive(U32.to_nat(w),U32.to_nat(Machine.dimension(limits)),U32.to_nat(Capacity.quarter(envelope)),Domain.width_room(call),Capacity.dimension_quarter_room(limits,envelope)))

def chunk_rows_quarter(+native: Capacity.Native) -> {Nat.is_le(U32.to_nat(Capacity.native_rows(native)),U32.to_nat(Capacity.quarter(Capacity.native_envelope(native)))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(Capacity.native_rows(native)),Capacity.chunk_rows(Capacity.model(native)),Capacity.rows_exact(native)) : {Nat.is_le(_,U32.to_nat(Capacity.quarter(Capacity.native_envelope(native)))) == True{} : Bool}
  Order.transitive(Capacity.chunk_rows(Capacity.model(native)),Capacity.workspace(Capacity.model(native)),U32.to_nat(Capacity.quarter(Capacity.native_envelope(native))),Capacity.rows_room(Capacity.model(native)),
    %Capacity.workspace_exact(native) : {Nat.is_le(_,U32.to_nat(Capacity.quarter(Capacity.native_envelope(native)))) == True{} : Bool} Bounds.as_nat_le(Capacity.native_workspace(native),Capacity.quarter(Capacity.native_envelope(native)),Capacity.workspace_quarter(native)))

def clamped_groups_positive(+groups: U32,+quotient: U32) -> {Nat.is_lt(0n,U32.to_nat(U32.max(1,U32.min(groups,quotient)))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.max(1,U32.min(groups,quotient))),Nat.max(1n,U32.to_nat(U32.min(groups,quotient))),NativeExtrema.maximum(1,U32.min(groups,quotient))) : {Nat.is_lt(0n,_) == True{} : Bool}
  Bounds.nat_trans(0n,1n,Nat.max(1n,U32.to_nat(U32.min(groups,quotient))),{==},Extrema.maximum_left(1n,U32.to_nat(U32.min(groups,quotient))))
def clamped_groups_room(+groups: U32,+quotient: U32,+width: Nat,+positive: {Nat.is_lt(0n,width) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(groups),width) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(U32.max(1,U32.min(groups,quotient))),width) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(U32.max(1,U32.min(groups,quotient))),Nat.max(1n,U32.to_nat(U32.min(groups,quotient))),NativeExtrema.maximum(1,U32.min(groups,quotient))) : {Nat.is_le(_,width) == True{} : Bool}
  Extrema.maximum_upper(1n,U32.to_nat(U32.min(groups,quotient)),width,Order.positive_one(width,positive),Order.transitive(U32.to_nat(U32.min(groups,quotient)),U32.to_nat(groups),width,Bounds.as_nat_le(U32.min(groups,quotient),groups,NativeExtrema.minimum_left(groups,quotient)),room))

def group_clamp_expression(+geometry: Kernel.Geometry) ->
  {Capacity.layout_chunk_groups(Kernel.tiling(geometry)) == U32.max(1,U32.min(Capacity.layout_groups(Kernel.tiling(geometry)),
    (Pipeline.block_tiles() / (Kernel.taps(Domain.kernel_word(geometry),Domain.stride_word(geometry)) * (Capacity.outputs_word(geometry) * Columns.layout_shifts(Kernel.tiling(geometry)) + 1 : U32)) : U32))) : U32}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}
def actual_groups_positive(+geometry: Kernel.Geometry) -> {Nat.is_lt(0n,U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(geometry)))) == True{} : Bool}:
  +quotient = (Pipeline.block_tiles() / (Kernel.taps(Domain.kernel_word(geometry),Domain.stride_word(geometry)) * (Capacity.outputs_word(geometry) * Columns.layout_shifts(Kernel.tiling(geometry)) + 1 : U32)) : U32)
  %Equal.sym(U32,Capacity.layout_chunk_groups(Kernel.tiling(geometry)),U32.max(1,U32.min(Capacity.layout_groups(Kernel.tiling(geometry)),quotient)),group_clamp_expression(geometry)) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool}
  clamped_groups_positive(Capacity.layout_groups(Kernel.tiling(geometry)),quotient)

def actual_groups_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {Nat.is_le(U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(Domain.geometry(call)))),Domain.width(Domain.geometry(call))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  +quotient = (Pipeline.block_tiles() / (Kernel.taps(Domain.kernel_word(geometry),Domain.stride_word(geometry)) * (Capacity.outputs_word(geometry) * Columns.layout_shifts(Kernel.tiling(geometry)) + 1 : U32)) : U32)
  %Equal.sym(U32,Capacity.layout_chunk_groups(Kernel.tiling(geometry)),U32.max(1,U32.min(Capacity.layout_groups(Kernel.tiling(geometry)),quotient)),group_clamp_expression(geometry)) : {Nat.is_le(U32.to_nat(_),Domain.width(geometry)) == True{} : Bool}
  clamped_groups_room(Capacity.layout_groups(Kernel.tiling(geometry)),quotient,Domain.width(geometry),Domain.width_positive(call),groups_room(call,envelope))

def actual_groups_quarter(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {Nat.is_le(U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(Domain.geometry(call)))),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Order.transitive(U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(geometry))),U32.to_nat(w),U32.to_nat(Capacity.quarter(envelope)),actual_groups_room(call,envelope),
    Order.transitive(U32.to_nat(w),U32.to_nat(Machine.dimension(limits)),U32.to_nat(Capacity.quarter(envelope)),Domain.width_room(call),Capacity.dimension_quarter_room(limits,envelope)))

# The native leaf uses the same two ceiling factors as the clipped partition.
def group_chunks(+geometry: Kernel.Geometry) -> U32:
  +groups = Capacity.layout_groups(Kernel.tiling(geometry))
  +span = Capacity.layout_chunk_groups(Kernel.tiling(geometry))
  ((groups + span - 1) / span : U32)
def row_chunks(+geometry: Kernel.Geometry,+count: U32) -> U32:
  +span = Capacity.layout_chunk_rows(Kernel.tiling(geometry))
  ((count + span - 1) / span : U32)
def chunk_query(geometry: Kernel.Geometry,count: U32,row: Nat,group: Nat) -> Chunks.Query:
  +geometry = geometry
  Chunks.Query{U32.to_nat(count),U32.to_nat(Capacity.layout_groups(Kernel.tiling(geometry))),U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(geometry))),U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(geometry))),row,group}

def group_chunks_exact(+call: Domain.Call,+envelope: Geometry.WordEnvelope) ->
  {U32.to_nat(group_chunks(Domain.geometry(call))) == Ceiling.ceiling(U32.to_nat(Capacity.layout_groups(Kernel.tiling(Domain.geometry(call)))),U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(Domain.geometry(call))))) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  ceiling_exact(Capacity.layout_groups(Kernel.tiling(geometry)),Capacity.layout_chunk_groups(Kernel.tiling(geometry)),envelope,groups_quarter(call,envelope),actual_groups_quarter(call,envelope),actual_groups_positive(geometry))

def count_quarter(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+count: U32,+contained: {Nat.is_le(U32.to_nat(count),Domain.height(Domain.geometry(call))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(count),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Order.transitive(U32.to_nat(count),U32.to_nat(h),U32.to_nat(Capacity.quarter(envelope)),contained,
    Order.transitive(U32.to_nat(h),U32.to_nat(Machine.dimension(limits)),U32.to_nat(Capacity.quarter(envelope)),Domain.height_room(call),Capacity.dimension_quarter_room(limits,envelope)))

def row_chunks_exact(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+count: U32,
  +contained: {Nat.is_le(U32.to_nat(count),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call))))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call)))),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}) ->
  {U32.to_nat(row_chunks(Domain.geometry(call),count)) == Ceiling.ceiling(U32.to_nat(count),U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call))))) : Nat}:
  ceiling_exact(count,Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call))),envelope,count_quarter(call,envelope,count,contained),room,positive)

def grid_product_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+count: U32,
  +contained: {Nat.is_le(U32.to_nat(count),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call))))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call)))),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(row_chunks(Domain.geometry(call),count)),U32.to_nat(group_chunks(Domain.geometry(call)))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  +rc = Ceiling.ceiling(U32.to_nat(count),U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(geometry))))
  +gc = Ceiling.ceiling(U32.to_nat(Capacity.layout_groups(Kernel.tiling(geometry))),U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(geometry))))
  +rbound = Order.transitive(rc,U32.to_nat(count),Domain.height(geometry),Ceiling.bounded(U32.to_nat(count),U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(geometry))),positive),contained)
  +gbound = Order.transitive(gc,U32.to_nat(Capacity.layout_groups(Kernel.tiling(geometry))),Domain.width(geometry),Ceiling.bounded(U32.to_nat(Capacity.layout_groups(Kernel.tiling(geometry))),U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(geometry))),actual_groups_positive(geometry)),groups_room(call,envelope))
  +plane = Nat.mul(Domain.height(geometry),Domain.width(geometry))
  %Equal.sym(Nat,U32.to_nat(row_chunks(geometry,count)),rc,row_chunks_exact(call,envelope,count,contained,positive,room)) : {Nat.is_le(Nat.mul(_,U32.to_nat(group_chunks(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(group_chunks(geometry)),gc,group_chunks_exact(call,envelope)) : {Nat.is_le(Nat.mul(rc,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Order.transitive(Nat.mul(rc,gc),plane,U32.to_nat(Machine.capacity(limits)),
    Order.transitive(Nat.mul(rc,gc),Nat.mul(Domain.height(geometry),gc),plane,RowMajor.mul_le(rc,Domain.height(geometry),gc,rbound),
      %Equal.sym(Nat,Nat.mul(Domain.height(geometry),gc),Nat.mul(gc,Domain.height(geometry)),RowMajor.multiply_commutative(Domain.height(geometry),gc)) : {Nat.is_le(_,plane) == True{} : Bool}
      %RowMajor.multiply_commutative(Domain.width(geometry),Domain.height(geometry)) : {Nat.is_le(Nat.mul(gc,Domain.height(geometry)),_) == True{} : Bool} RowMajor.mul_le(gc,Domain.width(geometry),Domain.height(geometry),gbound)),
    Domain.plane_room(call))

def leaf_count_expression(+geometry: Kernel.Geometry,+count: U32,+first: U32,+target: U32,+plane: U32) ->
  {Execution.chunk_count(Kernel.LeafContext{Kernel.Task{geometry,Kernel.tiling(geometry)},first,count,target,plane}) == U32.to_nat((row_chunks(geometry,count) * group_chunks(geometry) : U32)) : Nat}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}

def leaf_count_exact(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+count: U32,+first: U32,+target: U32,+plane: U32,
  +contained: {Nat.is_le(U32.to_nat(count),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call))))) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call)))),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}) ->
  {Execution.chunk_count(Kernel.LeafContext{Kernel.Task{Domain.geometry(call),Kernel.tiling(Domain.geometry(call))},first,count,target,plane}) == Chunks.chunk_count(chunk_query(Domain.geometry(call),count,0n,0n)) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Equal.trans(Nat,Execution.chunk_count(Kernel.LeafContext{Kernel.Task{geometry,Kernel.tiling(geometry)},first,count,target,plane}),U32.to_nat((row_chunks(geometry,count) * group_chunks(geometry) : U32)),Chunks.chunk_count(chunk_query(geometry,count,0n,0n)),leaf_count_expression(geometry,count,first,target,plane),
    Equal.trans(Nat,U32.to_nat((row_chunks(geometry,count) * group_chunks(geometry) : U32)),Nat.mul(U32.to_nat(row_chunks(geometry,count)),U32.to_nat(group_chunks(geometry))),Chunks.chunk_count(chunk_query(geometry,count,0n,0n)),
      Arithmetic.multiply(row_chunks(geometry,count),group_chunks(geometry),Machine.capacity(limits),Limits.capacity_fits(limits),grid_product_room(call,envelope,count,contained,positive,room)),
      %row_chunks_exact(call,envelope,count,contained,positive,room) : {Nat.mul(U32.to_nat(row_chunks(geometry,count)),U32.to_nat(group_chunks(geometry))) == Nat.mul(_,Chunks.group_chunks(chunk_query(geometry,count,0n,0n))) : Nat}
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(row_chunks(geometry,count)),value),U32.to_nat(group_chunks(geometry)),Chunks.group_chunks(chunk_query(geometry,count,0n,0n)),group_chunks_exact(call,envelope))))

type Grid is Data:
  Grid{call: Domain.Call,envelope: Geometry.WordEnvelope,count: U32,
    contained: {Nat.is_le(U32.to_nat(count),Domain.height(Domain.geometry(call))) == True{} : Bool},
    rows_positive: {Nat.is_lt(0n,U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call))))) == True{} : Bool},
    rows_room: {Nat.is_le(U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call)))),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}}
def grid_call(grid: Grid) -> Domain.Call:
  Grid{call,e,c,contained,rp,rr} = grid
  call
def grid_geometry(grid: Grid) -> Kernel.Geometry:
  Domain.geometry(grid_call(grid))
def grid_batch(grid: Grid) -> Nat:
  Domain.batch(grid_call(grid))
def grid_limits(grid: Grid) -> Machine.Limits:
  Domain.limits(grid_call(grid))
def grid_envelope(grid: Grid) -> Geometry.WordEnvelope:
  Grid{call,envelope,c,contained,rp,rr} = grid
  envelope
def grid_count(grid: Grid) -> U32:
  Grid{call,e,count,contained,rp,rr} = grid
  count
def query(+grid: Grid,+row: Nat,+group: Nat) -> Chunks.Query:
  chunk_query(grid_geometry(grid),grid_count(grid),row,group)
def grid_groups(+grid: Grid) -> U32:
  group_chunks(grid_geometry(grid))
def grid_rows(+grid: Grid) -> U32:
  row_chunks(grid_geometry(grid),grid_count(grid))
def grid_groups_exact(+grid: Grid,+row: Nat,+group: Nat) -> {U32.to_nat(grid_groups(grid)) == Chunks.group_chunks(query(grid,row,group)) : Nat}:
  Grid{call,e,c,contained,rp,rr} = grid
  group_chunks_exact(call,e)
def grid_rows_exact(+grid: Grid,+row: Nat,+group: Nat) -> {U32.to_nat(grid_rows(grid)) == Chunks.row_chunks(query(grid,row,group)) : Nat}:
  Grid{call,e,c,contained,rp,rr} = grid
  row_chunks_exact(call,e,c,contained,rp,rr)
def grid_room(+grid: Grid) -> {Nat.is_le(Nat.mul(U32.to_nat(grid_rows(grid)),U32.to_nat(grid_groups(grid))),U32.to_nat(Machine.capacity(grid_limits(grid)))) == True{} : Bool}:
  Grid{call,e,c,contained,rp,rr} = grid
  grid_product_room(call,e,c,contained,rp,rr)
def grid_natural_room(+grid: Grid,+row: Nat,+group: Nat) -> {Nat.is_le(Chunks.chunk_count(query(grid,row,group)),U32.to_nat(Machine.capacity(grid_limits(grid)))) == True{} : Bool}:
  %grid_groups_exact(grid,row,group) : {Nat.is_le(Nat.mul(Chunks.row_chunks(query(grid,row,group)),_),U32.to_nat(Machine.capacity(grid_limits(grid)))) == True{} : Bool}
  %grid_rows_exact(grid,row,group) : {Nat.is_le(Nat.mul(_,U32.to_nat(grid_groups(grid))),U32.to_nat(Machine.capacity(grid_limits(grid)))) == True{} : Bool}
  grid_room(grid)
def native_producer(+grid: Grid,+row: Nat,+group: Nat) -> U32:
  U32.from_nat(Chunks.producing_chunk(query(grid,row,group)))
def producer_exact(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) ->
  {U32.to_nat(native_producer(grid,row,group)) == Chunks.producing_chunk(query(grid,row,group)) : Nat}:
  Bounds.roundtrip(Chunks.producing_chunk(query(grid,row,group)),Machine.capacity(grid_limits(grid)),Limits.capacity_fits(grid_limits(grid)),
    Order.transitive(Chunks.producing_chunk(query(grid,row,group)),Chunks.chunk_count(query(grid,row,group)),U32.to_nat(Machine.capacity(grid_limits(grid))),Bounds.strict_le(Chunks.producing_chunk(query(grid,row,group)),Chunks.chunk_count(query(grid,row,group)),Chunks.producer_inside(query(grid,row,group),inside)),grid_natural_room(grid,row,group)))
def group_chunks_double_room(+grid: Grid) -> {Nat.is_le(Nat.double(U32.to_nat(grid_groups(grid))),U32.to_nat(Capacity.half(grid_envelope(grid)))) == True{} : Bool}:
  Grid{call,e,c,contained,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  +bounded = {%Equal.sym(Nat,U32.to_nat(group_chunks(g)),Ceiling.ceiling(U32.to_nat(Capacity.layout_groups(Kernel.tiling(g))),U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(g)))),group_chunks_exact(call,e)) : {Nat.is_le(_,U32.to_nat(Capacity.quarter(e))) == True{} : Bool}
    Order.transitive(Ceiling.ceiling(U32.to_nat(Capacity.layout_groups(Kernel.tiling(g))),U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(g)))),U32.to_nat(Capacity.layout_groups(Kernel.tiling(g))),U32.to_nat(Capacity.quarter(e)),Ceiling.bounded(U32.to_nat(Capacity.layout_groups(Kernel.tiling(g))),U32.to_nat(Capacity.layout_chunk_groups(Kernel.tiling(g))),actual_groups_positive(g)),groups_quarter(call,e)) : {Nat.is_le(U32.to_nat(group_chunks(g)),U32.to_nat(Capacity.quarter(e))) == True{} : Bool}}
  %Equal.sym(Nat,U32.to_nat(Capacity.half(e)),Nat.double(U32.to_nat(Capacity.quarter(e))),Capacity.quarter_doubled(e)) : {Nat.is_le(Nat.double(U32.to_nat(group_chunks(g))),_) == True{} : Bool}
  Order.double_le(U32.to_nat(group_chunks(g)),U32.to_nat(Capacity.quarter(e)),bounded)
def grid_groups_positive(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) -> {Nat.is_lt(0n,U32.to_nat(grid_groups(grid))) == True{} : Bool}:
  Chunks.Inside{rp,+gp,ri,+gi} = inside
  +q = query(grid,row,group)
  %Equal.sym(Nat,U32.to_nat(grid_groups(grid)),Chunks.group_chunks(q),grid_groups_exact(grid,row,group)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.positive_bound(Chunks.group_chunk(q),Chunks.group_chunks(q),Chunks.chunks_inside(Chunks.group(q),Chunks.groups(q),Chunks.chunk_groups(q),gp,gi))
def native_producer_row(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) ->
  {U32.to_nat((native_producer(grid,row,group) / grid_groups(grid) : U32)) == Chunks.row_chunk(query(grid,row,group)) : Nat}:
  +q = query(grid,row,group)
  +e = grid_envelope(grid)
  Equal.trans(Nat,U32.to_nat((native_producer(grid,row,group) / grid_groups(grid) : U32)),Nat.div(U32.to_nat(native_producer(grid,row,group)),U32.to_nat(grid_groups(grid))),Chunks.row_chunk(q),
    Division.divide(native_producer(grid,row,group),grid_groups(grid),Capacity.half(e),Capacity.half_fits(e),group_chunks_double_room(grid),grid_groups_positive(grid,row,group,inside)),
    %Equal.sym(Nat,U32.to_nat(native_producer(grid,row,group)),Chunks.producing_chunk(q),producer_exact(grid,row,group,inside)) : {Nat.div(_,U32.to_nat(grid_groups(grid))) == Chunks.row_chunk(q) : Nat}
    %Equal.sym(Nat,U32.to_nat(grid_groups(grid)),Chunks.group_chunks(q),grid_groups_exact(grid,row,group)) : {Nat.div(Chunks.producing_chunk(q),_) == Chunks.row_chunk(q) : Nat}
    Chunks.producer_row(q,inside))
def native_producer_group(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) ->
  {U32.to_nat((native_producer(grid,row,group) % grid_groups(grid) : U32)) == Chunks.group_chunk(query(grid,row,group)) : Nat}:
  +q = query(grid,row,group)
  +e = grid_envelope(grid)
  Equal.trans(Nat,U32.to_nat((native_producer(grid,row,group) % grid_groups(grid) : U32)),Nat.mod(U32.to_nat(native_producer(grid,row,group)),U32.to_nat(grid_groups(grid))),Chunks.group_chunk(q),
    Division.modulo(native_producer(grid,row,group),grid_groups(grid),Capacity.half(e),Capacity.half_fits(e),group_chunks_double_room(grid),grid_groups_positive(grid,row,group,inside)),
    %Equal.sym(Nat,U32.to_nat(native_producer(grid,row,group)),Chunks.producing_chunk(q),producer_exact(grid,row,group,inside)) : {Nat.mod(_,U32.to_nat(grid_groups(grid))) == Chunks.group_chunk(q) : Nat}
    %Equal.sym(Nat,U32.to_nat(grid_groups(grid)),Chunks.group_chunks(q),grid_groups_exact(grid,row,group)) : {Nat.mod(Chunks.producing_chunk(q),_) == Chunks.group_chunk(q) : Nat}
    Chunks.producer_group(q,inside))

def tiled_rows_positive(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call))))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  +native = Capacity.tiled_native(call,envelope,fits)
  %Equal.sym(U32,Capacity.layout_chunk_rows(Kernel.tiling(geometry)),Capacity.native_rows(native),Capacity.tiled_rows_matches(call,envelope,fits)) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(Capacity.native_rows(native)),Capacity.chunk_rows(Capacity.model(native)),Capacity.rows_exact(native)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Capacity.rows_positive(Capacity.model(native))
def tiled_rows_quarter(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(Domain.geometry(call)))),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +native = Capacity.tiled_native(call,envelope,fits)
  %Equal.sym(U32,Capacity.layout_chunk_rows(Kernel.tiling(geometry)),Capacity.native_rows(native),Capacity.tiled_rows_matches(call,envelope,fits)) : {Nat.is_le(U32.to_nat(_),U32.to_nat(Capacity.quarter(envelope))) == True{} : Bool}
  chunk_rows_quarter(native)
def fitted_grid(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+count: U32,
  +contained: {Nat.is_le(U32.to_nat(count),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) -> Grid:
  Grid{call,envelope,count,contained,tiled_rows_positive(call,envelope,fits),tiled_rows_quarter(call,envelope,fits)}

# Native multiplication of the decoded chunk coordinate gives its first row.
def row_offset(+grid: Grid,+row: Nat,+group: Nat) -> U32:
  ((native_producer(grid,row,group) / grid_groups(grid)) * Capacity.layout_chunk_rows(Kernel.tiling(grid_geometry(grid))) : U32)
def group_offset(+grid: Grid,+row: Nat,+group: Nat) -> U32:
  ((native_producer(grid,row,group) % grid_groups(grid)) * Capacity.layout_chunk_groups(Kernel.tiling(grid_geometry(grid))) : U32)
def row_offset_natural(+grid: Grid,+row: Nat,+group: Nat) -> Nat:
  Nat.mul(Chunks.row_chunk(query(grid,row,group)),Chunks.chunk_rows(query(grid,row,group)))
def group_offset_natural(+grid: Grid,+row: Nat,+group: Nat) -> Nat:
  Nat.mul(Chunks.group_chunk(query(grid,row,group)),Chunks.chunk_groups(query(grid,row,group)))
def row_offset_room(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) ->
  {Nat.is_le(row_offset_natural(grid,row,group),U32.to_nat(Capacity.quarter(grid_envelope(grid)))) == True{} : Bool}:
  Grid{call,e,c,contained,rp,rr} = grid
  Chunks.Inside{+positive,gp,+ri,gi} = inside
  +q = query(grid,row,group)
  Order.transitive(row_offset_natural(grid,row,group),row,U32.to_nat(Capacity.quarter(e)),NaturalDivision.product_below(row,Chunks.chunk_rows(q),positive),
    Order.transitive(row,U32.to_nat(c),U32.to_nat(Capacity.quarter(e)),Bounds.strict_le(row,U32.to_nat(c),ri),count_quarter(call,e,c,contained)))
def group_offset_room(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) ->
  {Nat.is_le(group_offset_natural(grid,row,group),U32.to_nat(Capacity.quarter(grid_envelope(grid)))) == True{} : Bool}:
  Grid{call,e,c,contained,rows_positive,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Chunks.Inside{rp,+positive,ri,+gi} = inside
  +q = query(grid,row,group)
  Order.transitive(group_offset_natural(grid,row,group),group,U32.to_nat(Capacity.quarter(e)),NaturalDivision.product_below(group,Chunks.chunk_groups(q),positive),
    Order.transitive(group,U32.to_nat(Capacity.layout_groups(Kernel.tiling(g))),U32.to_nat(Capacity.quarter(e)),Bounds.strict_le(group,U32.to_nat(Capacity.layout_groups(Kernel.tiling(g))),gi),groups_quarter(call,e)))
def row_offset_exact(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) ->
  {U32.to_nat(row_offset(grid,row,group)) == row_offset_natural(grid,row,group) : Nat}:
  +g = grid_geometry(grid)
  +e = grid_envelope(grid)
  +coordinate = (native_producer(grid,row,group) / grid_groups(grid) : U32)
  %native_producer_row(grid,row,group,inside) : {U32.to_nat(row_offset(grid,row,group)) == Nat.mul(_,Chunks.chunk_rows(query(grid,row,group))) : Nat}
  Arithmetic.multiply(coordinate,Capacity.layout_chunk_rows(Kernel.tiling(g)),Capacity.half(e),Capacity.half_fits(e),
    %Equal.sym(Nat,U32.to_nat(coordinate),Chunks.row_chunk(query(grid,row,group)),native_producer_row(grid,row,group,inside)) : {Nat.is_le(Nat.mul(_,Chunks.chunk_rows(query(grid,row,group))),U32.to_nat(Capacity.half(e))) == True{} : Bool}
    Order.transitive(row_offset_natural(grid,row,group),U32.to_nat(Capacity.quarter(e)),U32.to_nat(Capacity.half(e)),row_offset_room(grid,row,group,inside),Bounds.as_nat_le(Capacity.quarter(e),Capacity.half(e),Capacity.quarter_half(e))))
def group_offset_exact(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) ->
  {U32.to_nat(group_offset(grid,row,group)) == group_offset_natural(grid,row,group) : Nat}:
  +g = grid_geometry(grid)
  +e = grid_envelope(grid)
  +coordinate = (native_producer(grid,row,group) % grid_groups(grid) : U32)
  %native_producer_group(grid,row,group,inside) : {U32.to_nat(group_offset(grid,row,group)) == Nat.mul(_,Chunks.chunk_groups(query(grid,row,group))) : Nat}
  Arithmetic.multiply(coordinate,Capacity.layout_chunk_groups(Kernel.tiling(g)),Capacity.half(e),Capacity.half_fits(e),
    %Equal.sym(Nat,U32.to_nat(coordinate),Chunks.group_chunk(query(grid,row,group)),native_producer_group(grid,row,group,inside)) : {Nat.is_le(Nat.mul(_,Chunks.chunk_groups(query(grid,row,group))),U32.to_nat(Capacity.half(e))) == True{} : Bool}
    Order.transitive(group_offset_natural(grid,row,group),U32.to_nat(Capacity.quarter(e)),U32.to_nat(Capacity.half(e)),group_offset_room(grid,row,group,inside),Bounds.as_nat_le(Capacity.quarter(e),Capacity.half(e),Capacity.quarter_half(e))))

def row_remainder(+grid: Grid,+row: Nat,+group: Nat) -> U32:
  (grid_count(grid) - row_offset(grid,row,group) : U32)
def group_remainder(+grid: Grid,+row: Nat,+group: Nat) -> U32:
  (Capacity.layout_groups(Kernel.tiling(grid_geometry(grid))) - group_offset(grid,row,group) : U32)
def selected_rows(+grid: Grid,+row: Nat,+group: Nat) -> U32:
  U32.min(Capacity.layout_chunk_rows(Kernel.tiling(grid_geometry(grid))),row_remainder(grid,row,group))
def selected_groups(+grid: Grid,+row: Nat,+group: Nat) -> U32:
  U32.min(Capacity.layout_chunk_groups(Kernel.tiling(grid_geometry(grid))),group_remainder(grid,row,group))
def selected_rows_natural(+grid: Grid,+row: Nat,+group: Nat) -> Nat:
  Nat.min(Chunks.chunk_rows(query(grid,row,group)),Nat.sub(U32.to_nat(grid_count(grid)),row_offset_natural(grid,row,group)))
def row_remainder_exact(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) ->
  {U32.to_nat(row_remainder(grid,row,group)) == Nat.sub(U32.to_nat(grid_count(grid)),row_offset_natural(grid,row,group)) : Nat}:
  Grid{call,e,c,contained,rp,rr} = grid
  Chunks.Inside{+positive,gp,+ri,gi} = inside
  +q = query(grid,row,group)
  %row_offset_exact(grid,row,group,inside) : {U32.to_nat(row_remainder(grid,row,group)) == Nat.sub(U32.to_nat(c),_) : Nat}
  Arithmetic.subtract(c,row_offset(grid,row,group),Capacity.half(e),Capacity.half_fits(e),
    %Equal.sym(Nat,U32.to_nat(row_offset(grid,row,group)),row_offset_natural(grid,row,group),row_offset_exact(grid,row,group,inside)) : {Nat.is_le(_,U32.to_nat(c)) == True{} : Bool}
    Order.transitive(row_offset_natural(grid,row,group),row,U32.to_nat(c),NaturalDivision.product_below(row,Chunks.chunk_rows(q),positive),Bounds.strict_le(row,U32.to_nat(c),ri)),
    Order.transitive(Nat.sub(U32.to_nat(c),U32.to_nat(row_offset(grid,row,group))),U32.to_nat(c),U32.to_nat(Capacity.half(e)),Order.sub_le(U32.to_nat(c),U32.to_nat(row_offset(grid,row,group))),
      Order.transitive(U32.to_nat(c),U32.to_nat(Capacity.quarter(e)),U32.to_nat(Capacity.half(e)),count_quarter(call,e,c,contained),Bounds.as_nat_le(Capacity.quarter(e),Capacity.half(e),Capacity.quarter_half(e)))))
def selected_rows_exact(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) ->
  {U32.to_nat(selected_rows(grid,row,group)) == selected_rows_natural(grid,row,group) : Nat}:
  %row_remainder_exact(grid,row,group,inside) : {U32.to_nat(selected_rows(grid,row,group)) == Nat.min(Chunks.chunk_rows(query(grid,row,group)),_) : Nat}
  NativeExtrema.minimum(Capacity.layout_chunk_rows(Kernel.tiling(grid_geometry(grid))),row_remainder(grid,row,group))
def selected_row_inside(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) ->
  {Nat.is_lt(Nat.mod(row,Chunks.chunk_rows(query(grid,row,group))),U32.to_nat(selected_rows(grid,row,group))) == True{} : Bool}:
  Chunks.Inside{+positive,gp,+ri,gi} = inside
  +q = query(grid,row,group)
  %Equal.sym(Nat,U32.to_nat(selected_rows(grid,row,group)),selected_rows_natural(grid,row,group),selected_rows_exact(grid,row,group,inside)) : {Nat.is_lt(Nat.mod(row,Chunks.chunk_rows(q)),_) == True{} : Bool}
  Chunks.clipped_local(row,U32.to_nat(grid_count(grid)),Chunks.chunk_rows(q),positive,ri)

# The physical row base cancels from the native last-chunk remainder.
def physical_rows(+grid: Grid,+first: U32,+row: Nat,+group: Nat) -> U32:
  U32.min(Capacity.layout_chunk_rows(Kernel.tiling(grid_geometry(grid))),((first + grid_count(grid)) - (first + row_offset(grid,row,group)) : U32))
def physical_remainder_matches(+grid: Grid,+first: U32,+row: Nat,+group: Nat) ->
  {((first + grid_count(grid)) - (first + row_offset(grid,row,group)) : U32) == row_remainder(grid,row,group) : U32}:
  +count = grid_count(grid)
  +offset = row_offset(grid,row,group)
  Equal.trans(U32,((first + count) - (first + offset) : U32),(((first + count) - first) - offset : U32),(count - offset : U32),
    Equal.sym(U32,(((first + count) - first) - offset : U32),((first + count) - (first + offset) : U32),Arithmetic.subtract_sum((first + count : U32),first,offset)),
    Equal.cong(U32,U32,value => (value - offset : U32),((first + count) - first : U32),count,
      %Arithmetic.add_commutative(count,first) : {(_ - first : U32) == count : U32} Subtraction.subtract_add(count,first)))
def physical_rows_matches(+grid: Grid,+first: U32,+row: Nat,+group: Nat) -> {physical_rows(grid,first,row,group) == selected_rows(grid,row,group) : U32}:
  Equal.cong(U32,U32,value => U32.min(Capacity.layout_chunk_rows(Kernel.tiling(grid_geometry(grid))),value),((first + grid_count(grid)) - (first + row_offset(grid,row,group)) : U32),row_remainder(grid,row,group),physical_remainder_matches(grid,first,row,group))
def selected_chunk(+grid: Grid,+first: U32,+target: U32,+plane: U32,+row: Nat,+group: Nat) -> Kernel.Chunk:
  +g = grid_geometry(grid)
  +start = (first + row_offset(grid,row,group) : U32)
  +count = selected_rows(grid,row,group)
  Kernel.Chunk{Kernel.LeafContext{Kernel.Task{g,Kernel.tiling(g)},first,grid_count(grid),target,plane},start,count,Kernel.band_low(g,start,(start + count : U32)),group_offset(grid,row,group)}
def selected_chunk_matches(+grid: Grid,+first: U32,+target: U32,+plane: U32,+row: Nat,+group: Nat,+source: Execution.Source) ->
  {Execution.shape_chunk(Execution.chunk_shape(native_producer(grid,row,group),Kernel.LeafContext{Kernel.Task{grid_geometry(grid),Kernel.tiling(grid_geometry(grid))},first,grid_count(grid),target,plane},source)) == selected_chunk(grid,first,target,plane,row,group) : Kernel.Chunk}:
  Grid{call,e,c,contained,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{cc,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,base,source_plane,source_first,source_rows} = source
  +start = (first + row_offset(grid,row,group) : U32)
  Equal.cong(U32,Kernel.Chunk,value => Kernel.Chunk{Kernel.LeafContext{Kernel.Task{g,Kernel.tiling(g)},first,grid_count(grid),target,plane},start,value,Kernel.band_low(g,start,(start + value : U32)),group_offset(grid,row,group)},physical_rows(grid,first,row,group),selected_rows(grid,row,group),physical_rows_matches(grid,first,row,group))

def grid_height_room(+grid: Grid) -> {Nat.is_le(Domain.height(grid_geometry(grid)),U32.to_nat(Machine.dimension(grid_limits(grid)))) == True{} : Bool}:
  Grid{call,e,c,contained,rp,rr} = grid
  Domain.height_room(call)
def selected_offset_below_count(+grid: Grid,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>) ->
  {Nat.is_le(row_offset_natural(grid,row,group),U32.to_nat(grid_count(grid))) == True{} : Bool}:
  Chunks.Inside{+rp,gp,+ri,gi} = inside
  Order.transitive(row_offset_natural(grid,row,group),row,U32.to_nat(grid_count(grid)),NaturalDivision.product_below(row,Chunks.chunk_rows(query(grid,row,group)),rp),Bounds.strict_le(row,U32.to_nat(grid_count(grid)),ri))
def selected_start(+grid: Grid,+first: U32,+row: Nat,+group: Nat) -> U32:
  (first + row_offset(grid,row,group) : U32)
def selected_start_exact(+grid: Grid,+first: U32,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(grid_count(grid))),Domain.height(grid_geometry(grid))) == True{} : Bool}) ->
  {U32.to_nat(selected_start(grid,first,row,group)) == Nat.add(U32.to_nat(first),row_offset_natural(grid,row,group)) : Nat}:
  +limits = grid_limits(grid)
  %row_offset_exact(grid,row,group,inside) : {U32.to_nat(selected_start(grid,first,row,group)) == Nat.add(U32.to_nat(first),_) : Nat}
  Arithmetic.add(first,row_offset(grid,row,group),Machine.dimension(limits),Limits.dimension_fits(limits),
    %Equal.sym(Nat,U32.to_nat(row_offset(grid,row,group)),row_offset_natural(grid,row,group),row_offset_exact(grid,row,group,inside)) : {Nat.is_le(Nat.add(U32.to_nat(first),_),U32.to_nat(Machine.dimension(limits))) == True{} : Bool}
    Order.transitive(Nat.add(U32.to_nat(first),row_offset_natural(grid,row,group)),Nat.add(U32.to_nat(first),U32.to_nat(grid_count(grid))),U32.to_nat(Machine.dimension(limits)),Order.offset_le(U32.to_nat(first),row_offset_natural(grid,row,group),U32.to_nat(grid_count(grid)),selected_offset_below_count(grid,row,group,inside)),
      Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(grid_count(grid))),Domain.height(grid_geometry(grid)),U32.to_nat(Machine.dimension(limits)),contained,grid_height_room(grid))))
def selected_band_end(+grid: Grid,+first: U32,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(grid_count(grid))),Domain.height(grid_geometry(grid))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(selected_start(grid,first,row,group)),U32.to_nat(selected_rows(grid,row,group))),Nat.add(U32.to_nat(first),U32.to_nat(grid_count(grid)))) == True{} : Bool}:
  +offset = row_offset_natural(grid,row,group)
  +rows = selected_rows_natural(grid,row,group)
  +count = U32.to_nat(grid_count(grid))
  +start = U32.to_nat(first)
  +local = Nat.sub(count,offset)
  +restored = Order.restore_shift(offset,count,selected_offset_below_count(grid,row,group,inside))
  %Equal.sym(Nat,U32.to_nat(selected_start(grid,first,row,group)),Nat.add(start,offset),selected_start_exact(grid,first,row,group,inside,contained)) : {Nat.is_le(Nat.add(_,U32.to_nat(selected_rows(grid,row,group))),Nat.add(start,count)) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(selected_rows(grid,row,group)),rows,selected_rows_exact(grid,row,group,inside)) : {Nat.is_le(Nat.add(Nat.add(start,offset),_),Nat.add(start,count)) == True{} : Bool}
  %Equal.sym(Nat,Nat.add(Nat.add(start,offset),rows),Nat.add(start,Nat.add(offset,rows)),Addition.associative(start,offset,rows)) : {Nat.is_le(_,Nat.add(start,count)) == True{} : Bool}
  %restored : {Nat.is_le(Nat.add(start,Nat.add(offset,rows)),Nat.add(start,_)) == True{} : Bool}
  Order.offset_le(start,Nat.add(offset,rows),Nat.add(offset,local),Order.offset_le(offset,rows,local,Extrema.minimum_right(Chunks.chunk_rows(query(grid,row,group)),local)))
def selected_band_lower(+grid: Grid,+first: U32,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(grid_count(grid))),Domain.height(grid_geometry(grid))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(first),U32.to_nat(selected_start(grid,first,row,group))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(selected_start(grid,first,row,group)),Nat.add(U32.to_nat(first),row_offset_natural(grid,row,group)),selected_start_exact(grid,first,row,group,inside,contained)) : {Nat.is_le(U32.to_nat(first),_) == True{} : Bool}
  Order.le_add(U32.to_nat(first),row_offset_natural(grid,row,group))
def selected_band_contains(+grid: Grid,+first: U32,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(grid_count(grid))),Domain.height(grid_geometry(grid))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(selected_start(grid,first,row,group)),U32.to_nat(selected_rows(grid,row,group))),Domain.height(grid_geometry(grid))) == True{} : Bool}:
  Order.transitive(Nat.add(U32.to_nat(selected_start(grid,first,row,group)),U32.to_nat(selected_rows(grid,row,group))),Nat.add(U32.to_nat(first),U32.to_nat(grid_count(grid))),Domain.height(grid_geometry(grid)),selected_band_end(grid,first,row,group,inside,contained),contained)
def selected_rows_bounded(+grid: Grid,+row: Nat,+group: Nat) -> {Nat.is_le(U32.to_nat(selected_rows(grid,row,group)),U32.to_nat(Capacity.layout_chunk_rows(Kernel.tiling(grid_geometry(grid))))) == True{} : Bool}:
  Bounds.as_nat_le(selected_rows(grid,row,group),Capacity.layout_chunk_rows(Kernel.tiling(grid_geometry(grid))),NativeExtrema.minimum_left(Capacity.layout_chunk_rows(Kernel.tiling(grid_geometry(grid))),row_remainder(grid,row,group)))

def selected_guarded(+grid: Grid,+first: U32,+row: Nat,+group: Nat,+inside: Chunks.Inside<query(grid,row,group)>,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(grid_count(grid))),Domain.height(grid_geometry(grid))) == True{} : Bool},
  +fits: {Kernel.fits(grid_geometry(grid),Machine.capacity(grid_limits(grid))) == True{} : Bool}) ->
  Depth.Guarded<Capacity.tiled_count(grid_geometry(grid),selected_start(grid,first,row,group),selected_rows(grid,row,group))>:
  Grid{call,e,c,range,rp,rr} = grid
  Capacity.tiled_guarded(call,e,fits,selected_start(grid,first,row,group),selected_rows(grid,row,group),selected_band_contains(grid,first,row,group,inside,contained),selected_rows_bounded(grid,row,group))
def selected_shape_tiles_matches(+grid: Grid,+first: U32,+target: U32,+plane: U32,+row: Nat,+group: Nat,+source: Execution.Source) ->
  {Execution.shape_tiles(Execution.chunk_shape(native_producer(grid,row,group),Kernel.LeafContext{Kernel.Task{grid_geometry(grid),Kernel.tiling(grid_geometry(grid))},first,grid_count(grid),target,plane},source)) == Capacity.tiled_count(grid_geometry(grid),selected_start(grid,first,row,group),selected_rows(grid,row,group)) : U32}:
  Grid{call,e,c,range,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{cc,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,base,source_plane,source_first,source_rows} = source
  Equal.cong(U32,U32,value => Capacity.tiled_count(g,selected_start(grid,first,row,group),value),physical_rows(grid,first,row,group),selected_rows(grid,row,group),physical_rows_matches(grid,first,row,group))

# An enabled phase's eight-lane group is contained in the tiled group extent.
def phase_group_inside(+grid: Grid,+index: Nat,
  +inside: {Nat.is_lt(index,Extents.count(Extents.columns(grid_call(grid),grid_envelope(grid)))) == True{} : Bool},
  +used: {Extents.used(Extents.columns(grid_call(grid),grid_envelope(grid)),U32.from_nat(index)) == True{} : Bool},
  +position: Nat,+position_inside: {Nat.is_lt(position,U32.to_nat(Extents.axis_count(Extents.columns(grid_call(grid),grid_envelope(grid)),U32.from_nat(index)))) == True{} : Bool}) ->
  {Nat.is_lt(Nat.div(position,8n),U32.to_nat(Capacity.layout_groups(Kernel.tiling(grid_geometry(grid))))) == True{} : Bool}:
  Grid{call,e,c,range,rp,rr} = grid
  +q = Extents.columns(call,e)
  +maximum = U32.to_nat(Bands.count_fold(Columns.whole(call,e)))
  %Equal.sym(Nat,U32.to_nat(Capacity.layout_groups(Kernel.tiling(Domain.geometry(call)))),Ceiling.ceiling(maximum,8n),groups_exact(call,e)) : {Nat.is_lt(Nat.div(position,8n),_) == True{} : Bool}
  Chunks.chunks_inside(position,maximum,8n,{==},Bounds.nat_trans(position,U32.to_nat(Extents.axis_count(q,U32.from_nat(index))),maximum,position_inside,Bounds.as_nat_le(Extents.axis_count(q,U32.from_nat(index)),Bands.count_fold(Columns.whole(call,e)),Columns.selected_count(call,e,index,inside,used))))
def phase_inside(+grid: Grid,+index: Nat,
  +inside: {Nat.is_lt(index,Extents.count(Extents.columns(grid_call(grid),grid_envelope(grid)))) == True{} : Bool},
  +used: {Extents.used(Extents.columns(grid_call(grid),grid_envelope(grid)),U32.from_nat(index)) == True{} : Bool},
  +row: Nat,+row_inside: {Nat.is_lt(row,U32.to_nat(grid_count(grid))) == True{} : Bool},
  +position: Nat,+position_inside: {Nat.is_lt(position,U32.to_nat(Extents.axis_count(Extents.columns(grid_call(grid),grid_envelope(grid)),U32.from_nat(index)))) == True{} : Bool}) ->
  Chunks.Inside<query(grid,row,Nat.div(position,8n))>:
  Grid{call,e,c,range,rp,rr} = grid
  Chunks.Inside{rp,actual_groups_positive(Domain.geometry(call)),row_inside,phase_group_inside(grid,index,inside,used,position,position_inside)}

def selected_filling(+grid: Grid,+first: U32,+row: Nat,+group: Nat,+source: Execution.Source) -> Kernel.Filling:
  +g = grid_geometry(grid)
  +start = selected_start(grid,first,row,group)
  +count = selected_rows(grid,row,group)
  Kernel.Filling{Kernel.tiling(g),Kernel.band_low(g,start,(start + count : U32)),
    Execution.source_base(source),Execution.source_plane(source),Execution.source_first(source),Execution.source_rows(source),group_offset(grid,row,group),selected_groups(grid,row,group)}
def selected_filling_matches(+grid: Grid,+first: U32,+target: U32,+plane: U32,+row: Nat,+group: Nat,+source: Execution.Source) ->
  {Execution.shape_filling(Execution.chunk_shape(native_producer(grid,row,group),Kernel.LeafContext{Kernel.Task{grid_geometry(grid),Kernel.tiling(grid_geometry(grid))},first,grid_count(grid),target,plane},source)) == selected_filling(grid,first,row,group,source) : Kernel.Filling}:
  Grid{call,e,c,range,rp,rr} = grid
  Domain.Call{g,b,l,facts} = call
  Kernel.Geometry{cc,o,h,w,k,s,p,gh,gw} = g
  Execution.Source{data,+base,+source_plane,+source_first,+source_rows} = source
  +start = selected_start(grid,first,row,group)
  Equal.cong(U32,Kernel.Filling,value => Kernel.Filling{Kernel.tiling(g),Kernel.band_low(g,start,(start + value : U32)),base,source_plane,source_first,source_rows,group_offset(grid,row,group),selected_groups(grid,row,group)},physical_rows(grid,first,row,group),selected_rows(grid,row,group),physical_rows_matches(grid,first,row,group))
