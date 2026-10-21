# Chunk dimensions use only the halo bound and the smallest-cell fits guard.
import Base
import ./nat_order.bend as Order
import ./nat_extrema.bend as Extrema
import ./nat_to_u32_bounds.bend as Bounds
import ./nat_division_order.bend as Division
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./word_subtraction.bend as Subtraction
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_tiled_geometry.bend as Geometry
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./machine_limit_bounds.bend as Limits
import ./u32_division.bend as NativeDivision
import ./u32_extrema.bend as NativeExtrema
import ./traversal_extrema.bend as Folds
import ./u32_shift.bend as Shift
import ./input_gradient_phase_geometry.bend as Phase
import ./u32_comparison.bend as Comparison
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_extents.bend as Extents
import ./input_gradient_tiled_columns.bend as Columns
import ../kernel_pipeline.bend as Pipeline
import ./input_gradient_tiled_band_extents.bend as Bands
import ./tensor_allocation_depth.bend as Depth

type Plan is Data:
  Plan{groups: Nat,taps: Nat,block: Nat,limit: Nat,capacity: Nat,
    taps_positive: {Nat.is_lt(0n,taps) == True{} : Bool},block_positive: {Nat.is_lt(0n,block) == True{} : Bool},
    fits: {Nat.is_le(Nat.mul(taps,block),capacity) == True{} : Bool}}
def groups(plan: Plan) -> Nat:
  Plan{groups,taps,block,limit,capacity,tp,bp,fits} = plan
  groups
def taps(plan: Plan) -> Nat:
  Plan{groups,taps,block,limit,capacity,tp,bp,fits} = plan
  taps
def block(plan: Plan) -> Nat:
  Plan{groups,taps,block,limit,capacity,tp,bp,fits} = plan
  block
def limit(plan: Plan) -> Nat:
  Plan{groups,taps,block,limit,capacity,tp,bp,fits} = plan
  limit
def capacity(plan: Plan) -> Nat:
  Plan{groups,taps,block,limit,capacity,tp,bp,fits} = plan
  capacity
def cell(+plan: Plan) -> Nat:
  Nat.mul(taps(plan),block(plan))
def workspace(+plan: Plan) -> Nat:
  Nat.max(limit(plan),capacity(plan))
def chunk_groups(+plan: Plan) -> Nat:
  Nat.max(1n,Nat.min(groups(plan),Nat.div(limit(plan),cell(plan))))
def row_span(+plan: Plan) -> Nat:
  Nat.mul(chunk_groups(plan),block(plan))
def fitting(+plan: Plan) -> Nat:
  1n+Nat.div(limit(plan),row_span(plan))
def chunk_rows(+plan: Plan) -> Nat:
  Bool.pick(Nat,Nat.is_lt(taps(plan),fitting(plan)),Nat.sub(fitting(plan),taps(plan)),1n)
def halo_rows(+plan: Plan) -> Nat:
  Nat.add(Nat.sub(chunk_rows(plan),1n),taps(plan))

def groups_positive(+plan: Plan) -> {Nat.is_lt(0n,chunk_groups(plan)) == True{} : Bool}:
  Bounds.nat_trans(0n,1n,chunk_groups(plan),{==},Extrema.maximum_left(1n,Nat.min(groups(plan),Nat.div(limit(plan),cell(plan)))))
def cell_positive(+plan: Plan) -> {Nat.is_lt(0n,cell(plan)) == True{} : Bool}:
  Plan{groups,+taps,+block,limit,capacity,+tp,+bp,fits} = plan
  Order.product_positive(taps,block,tp,bp)
def span_positive(+plan: Plan) -> {Nat.is_lt(0n,row_span(plan)) == True{} : Bool}:
  Plan{groups,taps,+block,limit,capacity,tp,+bp,fits} = plan
  Order.product_positive(chunk_groups(plan),block,groups_positive(plan),bp)

def clamped_cell_case(selected: Bool,+plan: Plan,
  compared: {Nat.is_lt(1n,Nat.min(groups(plan),Nat.div(limit(plan),cell(plan)))) == selected : Bool}) ->
  {Nat.is_le(Nat.mul(Bool.pick(Nat,selected,Nat.min(groups(plan),Nat.div(limit(plan),cell(plan))),1n),cell(plan)),workspace(plan)) == True{} : Bool}:
  match selected:
    case False{}:
      Plan{groups,+taps,+block,+limit,+capacity,tp,bp,+fits} = plan
      %Equal.sym(Nat,Nat.mul(1n,cell(plan)),cell(plan),Addition.zero_right(cell(plan))) : {Nat.is_le(_,workspace(plan)) == True{} : Bool}
      Order.transitive(cell(plan),capacity,workspace(plan),fits,Extrema.maximum_right(limit,capacity))
    case True{}:
      +count = Nat.min(groups(plan),Nat.div(limit(plan),cell(plan)))
      +quotient = Nat.div(limit(plan),cell(plan))
      Order.transitive(Nat.mul(count,cell(plan)),Nat.mul(quotient,cell(plan)),workspace(plan),RowMajor.mul_le(count,quotient,cell(plan),Extrema.minimum_right(groups(plan),quotient)),
        Order.transitive(Nat.mul(quotient,cell(plan)),limit(plan),workspace(plan),Division.product_below(limit(plan),cell(plan),cell_positive(plan)),Extrema.maximum_left(limit(plan),capacity(plan))))

def clamped_cell(+plan: Plan) -> {Nat.is_le(Nat.mul(chunk_groups(plan),cell(plan)),workspace(plan)) == True{} : Bool}:
  +minimum = Nat.min(groups(plan),Nat.div(limit(plan),cell(plan)))
  %Equal.sym(Nat,chunk_groups(plan),Bool.pick(Nat,Nat.is_lt(1n,minimum),minimum,1n),Extrema.maximum_choice(1n,minimum)) : {Nat.is_le(Nat.mul(_,cell(plan)),workspace(plan)) == True{} : Bool}
  clamped_cell_case(Nat.is_lt(1n,minimum),plan,{==})
def minimal_row_room(+plan: Plan) -> {Nat.is_le(Nat.mul(taps(plan),row_span(plan)),workspace(plan)) == True{} : Bool}:
  +t = taps(plan)
  +g = chunk_groups(plan)
  +b = block(plan)
  +same = Equal.trans(Nat,Nat.mul(t,Nat.mul(g,b)),Nat.mul(Nat.mul(t,g),b),Nat.mul(g,Nat.mul(t,b)),Equal.sym(Nat,Nat.mul(Nat.mul(t,g),b),Nat.mul(t,Nat.mul(g,b)),RowMajor.assoc(t,g,b)),
    Equal.trans(Nat,Nat.mul(Nat.mul(t,g),b),Nat.mul(Nat.mul(g,t),b),Nat.mul(g,Nat.mul(t,b)),Equal.cong(Nat,Nat,value => Nat.mul(value,b),Nat.mul(t,g),Nat.mul(g,t),RowMajor.multiply_commutative(t,g)),RowMajor.assoc(g,t,b)))
  %Equal.sym(Nat,Nat.mul(t,row_span(plan)),Nat.mul(g,cell(plan)),same) : {Nat.is_le(_,workspace(plan)) == True{} : Bool}
  clamped_cell(plan)

def fitting_tail(+quotient: Nat,+taps: Nat,+inside: {Nat.is_lt(taps,1n+quotient) == True{} : Bool}) ->
  {Nat.add(Nat.sub(Nat.sub(1n+quotient,taps),1n),taps) == quotient : Nat}:
  +rows = Nat.sub(1n+quotient,taps)
  +positive = Order.positive_difference(1n+quotient,taps,inside)
  +last = Nat.sub(rows,1n)
  Order.cancel_sum(1n,Nat.add(last,taps),quotient,
    Equal.trans(Nat,1n+Nat.add(last,taps),Nat.add(rows,taps),1n+quotient,
      Equal.cong(Nat,Nat,value => Nat.add(value,taps),1n+last,rows,Order.positive_has_last(rows,positive)),Subtraction.restore(1n+quotient,taps,Bounds.strict_le(taps,1n+quotient,inside))))

def halo_room_case(fits: Bool,+plan: Plan,+compared: {Nat.is_lt(taps(plan),fitting(plan)) == fits : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.add(Nat.sub(Bool.pick(Nat,fits,Nat.sub(fitting(plan),taps(plan)),1n),1n),taps(plan)),row_span(plan)),workspace(plan)) == True{} : Bool}:
  match fits:
    case False{}: minimal_row_room(plan)
    case True{}:
      +quotient = Nat.div(limit(plan),row_span(plan))
      %Equal.sym(Nat,Nat.add(Nat.sub(Nat.sub(fitting(plan),taps(plan)),1n),taps(plan)),quotient,fitting_tail(quotient,taps(plan),compared)) : {Nat.is_le(Nat.mul(_,row_span(plan)),workspace(plan)) == True{} : Bool}
      Order.transitive(Nat.mul(quotient,row_span(plan)),limit(plan),workspace(plan),Division.product_below(limit(plan),row_span(plan),span_positive(plan)),Extrema.maximum_left(limit(plan),capacity(plan)))

def halo_room(+plan: Plan) -> {Nat.is_le(Nat.mul(halo_rows(plan),row_span(plan)),workspace(plan)) == True{} : Bool}:
  halo_room_case(Nat.is_lt(taps(plan),fitting(plan)),plan,{==})
def rows_positive_case(fits: Bool,+plan: Plan,+compared: {Nat.is_lt(taps(plan),fitting(plan)) == fits : Bool}) ->
  {Nat.is_lt(0n,Bool.pick(Nat,fits,Nat.sub(fitting(plan),taps(plan)),1n)) == True{} : Bool}:
  match fits:
    case False{}: {==}
    case True{}: Order.positive_difference(fitting(plan),taps(plan),compared)
def rows_positive(+plan: Plan) -> {Nat.is_lt(0n,chunk_rows(plan)) == True{} : Bool}:
  rows_positive_case(Nat.is_lt(taps(plan),fitting(plan)),plan,{==})

def row_span_room(+plan: Plan) -> {Nat.is_le(row_span(plan),workspace(plan)) == True{} : Bool}:
  Plan{groups,+taps,block,limit,capacity,+tp,bp,fits} = plan
  Order.transitive(row_span(plan),Nat.mul(taps,row_span(plan)),workspace(plan),
    %RowMajor.multiply_commutative(row_span(plan),taps) : {Nat.is_le(row_span(plan),_) == True{} : Bool} RowMajor.product_covers(row_span(plan),taps,tp),minimal_row_room(plan))

def tile_room(+plan: Plan,+rows: Nat,+tile_rows: Nat,
  +rows_bounded: {Nat.is_le(rows,chunk_rows(plan)) == True{} : Bool},
  +halo_bounded: {Nat.is_le(tile_rows,Nat.add(Nat.sub(rows,1n),taps(plan))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(tile_rows,row_span(plan)),workspace(plan)) == True{} : Bool}:
  Order.transitive(Nat.mul(tile_rows,row_span(plan)),Nat.mul(halo_rows(plan),row_span(plan)),workspace(plan),
    RowMajor.mul_le(tile_rows,halo_rows(plan),row_span(plan),Order.transitive(tile_rows,Nat.add(Nat.sub(rows,1n),taps(plan)),halo_rows(plan),halo_bounded,
      Order.add_le(Nat.sub(rows,1n),Nat.sub(chunk_rows(plan),1n),taps(plan),Division.subtract_le(rows,chunk_rows(plan),1n,rows_bounded)))),halo_room(plan))

# Native words carry derived scalar readbacks; the geometry adapter supplies
# them from the actual tiling and fits, before any leaf addresses are used.
def quarter(envelope: Geometry.WordEnvelope) -> U32:
  Geometry.WordEnvelope{maximum,half,quarter,margin,me,he,qe,re} = envelope
  quarter
def half(envelope: Geometry.WordEnvelope) -> U32:
  Geometry.WordEnvelope{maximum,half,quarter,margin,me,he,qe,re} = envelope
  half
def margin(envelope: Geometry.WordEnvelope) -> U32:
  Geometry.WordEnvelope{maximum,half,quarter,margin,me,he,qe,re} = envelope
  margin

# The dimension limit fits a quarter word.
def dimension_quarter_room(+limits: Machine.Limits,+envelope: Geometry.WordEnvelope) ->
  {Nat.is_le(U32.to_nat(Machine.dimension(limits)),U32.to_nat(quarter(envelope))) == True{} : Bool}:
  Geometry.WordEnvelope{maximum,half,+quarter,margin,me,he,+qe,re} = envelope
  %Equal.sym(U32,quarter,1073741824,qe) : {Nat.is_le(U32.to_nat(Machine.dimension(limits)),U32.to_nat(_)) == True{} : Bool}
  Bounds.as_nat_le(Machine.dimension(limits),1073741824,Limits.dimension_allocatable(limits))

type Native is Data:
  Native{groups: U32,shifts: U32,outputs: U32,kernel: U32,stride: U32,limits: Machine.Limits,limit: U32,envelope: Geometry.WordEnvelope,taps: Nat,block: Nat,
    taps_positive: {Nat.is_lt(0n,taps) == True{} : Bool},block_positive: {Nat.is_lt(0n,block) == True{} : Bool},
    fits: {Nat.is_le(Nat.mul(taps,block),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    taps_exact: {U32.to_nat(Kernel.taps(kernel,stride)) == taps : Nat},
    block_exact: {U32.to_nat((outputs * shifts + 1 : U32)) == block : Nat},
    limit_bound: {U32.is_le(limit,quarter(envelope)) == True{} : Bool}}
def native_limits(native: Native) -> Machine.Limits:
  Native{g,s,o,k,stride,limits,limit,envelope,t,b,tp,bp,fits,te,be,lb} = native
  limits
def native_envelope(native: Native) -> Geometry.WordEnvelope:
  Native{g,s,o,k,stride,limits,limit,envelope,t,b,tp,bp,fits,te,be,lb} = native
  envelope
def native_limit(native: Native) -> U32:
  Native{g,s,o,k,stride,limits,limit,envelope,t,b,tp,bp,fits,te,be,lb} = native
  limit
def native_taps(native: Native) -> U32:
  Native{g,s,o,+k,+stride,limits,limit,envelope,t,b,tp,bp,fits,te,be,lb} = native
  Kernel.taps(k,stride)
def native_block(native: Native) -> U32:
  Native{g,+s,+o,k,stride,limits,limit,envelope,t,b,tp,bp,fits,te,be,lb} = native
  (o * s + 1 : U32)
def native_cell(native: Native) -> U32:
  +native = native
  (native_taps(native) * native_block(native) : U32)
def model(native: Native) -> Plan:
  Native{+g,s,o,k,stride,+limits,+limit,envelope,+t,+b,+tp,+bp,+fits,te,be,lb} = native
  Plan{U32.to_nat(g),t,b,U32.to_nat(limit),U32.to_nat(Machine.capacity(limits)),tp,bp,fits}
def native_groups(native: Native) -> U32:
  Native{+g,s,o,k,stride,limits,+limit,envelope,t,b,tp,bp,fits,te,be,lb} = native
  U32.max(1,U32.min(g,(limit / native_cell(native) : U32)))
def native_span(native: Native) -> U32:
  +native = native
  (native_groups(native) * native_block(native) : U32)
def native_workspace(native: Native) -> U32:
  +native = native
  U32.max(native_limit(native),Machine.capacity(native_limits(native)))

def cell_exact(+native: Native) -> {U32.to_nat(native_cell(native)) == cell(model(native)) : Nat}:
  Native{g,s,o,+k,+stride,+limits,limit,envelope,+t,+b,tp,bp,+fits,+te,+be,lb} = native
  +tap_word = Kernel.taps(k,stride)
  +block_word = native_block(native)
  Equal.trans(Nat,U32.to_nat((tap_word * block_word : U32)),Nat.mul(U32.to_nat(tap_word),U32.to_nat(block_word)),Nat.mul(t,b),
    Arithmetic.multiply(tap_word,block_word,Machine.capacity(limits),Limits.capacity_fits(limits),
      %Equal.sym(Nat,U32.to_nat(tap_word),t,te) : {Nat.is_le(Nat.mul(_,U32.to_nat(block_word)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(block_word),b,be) : {Nat.is_le(Nat.mul(t,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} fits),
    Equal.trans(Nat,Nat.mul(U32.to_nat(tap_word),U32.to_nat(block_word)),Nat.mul(t,U32.to_nat(block_word)),Nat.mul(t,b),
      Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(block_word)),U32.to_nat(tap_word),t,te),Equal.cong(Nat,Nat,value => Nat.mul(t,value),U32.to_nat(block_word),b,be)))

def cell_word_room(+native: Native) -> {Nat.is_le(U32.to_nat(native_cell(native)),U32.to_nat(Machine.capacity(native_limits(native)))) == True{} : Bool}:
  Native{g,s,o,k,stride,+limits,limit,envelope,t,b,tp,bp,+fits,te,be,lb} = native
  %Equal.sym(Nat,U32.to_nat(native_cell(native)),cell(model(native)),cell_exact(native)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  fits

def cell_division(+native: Native) ->
  {U32.to_nat((native_limit(native) / native_cell(native) : U32)) == Nat.div(limit(model(native)),cell(model(native))) : Nat}:
  Native{g,s,o,k,stride,+limits,+budget,envelope,t,b,tp,bp,fits,te,be,lb} = native
  +denominator = native_cell(native)
  %cell_exact(native) : {U32.to_nat((native_limit(native) / denominator : U32)) == Nat.div(limit(model(native)),_) : Nat}
  NativeDivision.divide(native_limit(native),denominator,Machine.half_word(limits),Limits.half_word_fits(limits),Limits.double_bound(limits,U32.to_nat(denominator),cell_word_room(native)),
    %Equal.sym(Nat,U32.to_nat(denominator),cell(model(native)),cell_exact(native)) : {Nat.is_lt(0n,_) == True{} : Bool} cell_positive(model(native)))

def groups_exact(+native: Native) -> {U32.to_nat(native_groups(native)) == chunk_groups(model(native)) : Nat}:
  Native{+g,s,o,k,stride,limits,+limit,envelope,t,b,tp,bp,fits,te,be,lb} = native
  +quotient = (limit / native_cell(native) : U32)
  Equal.trans(Nat,U32.to_nat(U32.max(1,U32.min(g,quotient))),Nat.max(1n,U32.to_nat(U32.min(g,quotient))),chunk_groups(model(native)),NativeExtrema.maximum(1,U32.min(g,quotient)),
    Equal.cong(Nat,Nat,value => Nat.max(1n,value),U32.to_nat(U32.min(g,quotient)),Nat.min(U32.to_nat(g),Nat.div(U32.to_nat(limit),cell(model(native)))),
      Equal.trans(Nat,U32.to_nat(U32.min(g,quotient)),Nat.min(U32.to_nat(g),U32.to_nat(quotient)),Nat.min(U32.to_nat(g),Nat.div(U32.to_nat(limit),cell(model(native)))),NativeExtrema.minimum(g,quotient),
        Equal.cong(Nat,Nat,value => Nat.min(U32.to_nat(g),value),U32.to_nat(quotient),Nat.div(U32.to_nat(limit),cell(model(native))),cell_division(native)))))

def workspace_exact(+native: Native) -> {U32.to_nat(native_workspace(native)) == workspace(model(native)) : Nat}:
  Native{g,s,o,k,stride,+limits,+limit,envelope,t,b,tp,bp,fits,te,be,lb} = native
  NativeExtrema.maximum(limit,Machine.capacity(limits))

def capacity_quarter(+limits: Machine.Limits,+envelope: Geometry.WordEnvelope) -> {U32.is_le(Machine.capacity(limits),quarter(envelope)) == True{} : Bool}:
  Machine.Limits{d,+c,k,h,Machine.Facts{df,cf,kf,hf,da,+ca,db,cb,dc,dk,ddk,ch,sc,sh,chb}} = limits
  Geometry.WordEnvelope{maximum,half,quarter,margin,me,he,qe,re} = envelope
  %Equal.sym(U32,quarter,1073741824,qe) : {U32.is_le(c,_) == True{} : Bool}
  ca

def quarter_half(+envelope: Geometry.WordEnvelope) -> {U32.is_le(quarter(envelope),half(envelope)) == True{} : Bool}:
  Geometry.WordEnvelope{maximum,+half,+quarter,margin,me,+he,+qe,re} = envelope
  %Equal.sym(U32,quarter,1073741824,qe) : {U32.is_le(_,half) == True{} : Bool}
  %Equal.sym(U32,half,2147483648,he) : {U32.is_le(1073741824,_) == True{} : Bool}
  {==}
def half_fits(+envelope: Geometry.WordEnvelope) -> {U32.is_le(half(envelope),2147483648) == True{} : Bool}:
  Geometry.WordEnvelope{maximum,half,quarter,margin,me,he,qe,re} = envelope
  %Equal.sym(U32,half,2147483648,he) : {U32.is_le(_,2147483648) == True{} : Bool}
  {==}
def workspace_quarter(+native: Native) -> {U32.is_le(native_workspace(native),quarter(native_envelope(native))) == True{} : Bool}:
  Native{g,s,o,k,stride,+limits,+limit,+envelope,t,b,tp,bp,fits,te,be,+lb} = native
  NativeExtrema.maximum_upper(limit,Machine.capacity(limits),quarter(envelope),lb,capacity_quarter(limits,envelope))
def workspace_fits(+native: Native) -> {U32.is_le(native_workspace(native),2147483648) == True{} : Bool}:
  +envelope = native_envelope(native)
  Folds.transitive(native_workspace(native),half(envelope),2147483648,
    Folds.transitive(native_workspace(native),quarter(envelope),half(envelope),workspace_quarter(native),quarter_half(envelope)),half_fits(envelope))

def quarter_doubled(+envelope: Geometry.WordEnvelope) -> {U32.to_nat(half(envelope)) == Nat.double(U32.to_nat(quarter(envelope))) : Nat}:
  Geometry.WordEnvelope{maximum,+half,+quarter,margin,me,+he,+qe,re} = envelope
  +same = {%Equal.sym(U32,quarter,1073741824,qe) : {U32.shl(_) == half : U32}
    %Equal.sym(U32,half,2147483648,he) : {U32.shl(1073741824) == _ : U32} {==} : {U32.shl(quarter) == half : U32}}
  Equal.trans(Nat,U32.to_nat(half),U32.to_nat(U32.shl(quarter)),Nat.double(U32.to_nat(quarter)),Equal.cong(U32,Nat,U32.to_nat,half,U32.shl(quarter),Equal.sym(U32,U32.shl(quarter),half,same)),
    Shift.double(quarter,%Equal.sym(U32,quarter,1073741824,qe) : {U32.is_lt(_,2147483648) == True{} : Bool} {==}))

def workspace_double_room(+native: Native) -> {Nat.is_le(Nat.double(workspace(model(native))),U32.to_nat(half(native_envelope(native)))) == True{} : Bool}:
  +envelope = native_envelope(native)
  %Equal.sym(Nat,U32.to_nat(half(envelope)),Nat.double(U32.to_nat(quarter(envelope))),quarter_doubled(envelope)) : {Nat.is_le(Nat.double(workspace(model(native))),_) == True{} : Bool}
  Order.double_le(workspace(model(native)),U32.to_nat(quarter(envelope)),
    %workspace_exact(native) : {Nat.is_le(_,U32.to_nat(quarter(envelope))) == True{} : Bool} Bounds.as_nat_le(native_workspace(native),quarter(envelope),workspace_quarter(native)))

def span_exact(+native: Native) -> {U32.to_nat(native_span(native)) == row_span(model(native)) : Nat}:
  Native{g,s,o,k,stride,limits,limit,envelope,t,+b,tp,bp,fits,te,+be,lb} = native
  +groups = native_groups(native)
  +block = native_block(native)
  +natural_groups = chunk_groups(model(native))
  +groups_exact = groups_exact(native)
  Equal.trans(Nat,U32.to_nat((groups * block : U32)),Nat.mul(U32.to_nat(groups),U32.to_nat(block)),Nat.mul(natural_groups,b),
    Arithmetic.multiply(groups,block,native_workspace(native),workspace_fits(native),
      %Equal.sym(Nat,U32.to_nat(groups),natural_groups,groups_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(block)),U32.to_nat(native_workspace(native))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(block),b,be) : {Nat.is_le(Nat.mul(natural_groups,_),U32.to_nat(native_workspace(native))) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(native_workspace(native)),workspace(model(native)),workspace_exact(native)) : {Nat.is_le(row_span(model(native)),_) == True{} : Bool} row_span_room(model(native))),
    Equal.trans(Nat,Nat.mul(U32.to_nat(groups),U32.to_nat(block)),Nat.mul(natural_groups,U32.to_nat(block)),Nat.mul(natural_groups,b),
      Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(block)),U32.to_nat(groups),natural_groups,groups_exact),Equal.cong(Nat,Nat,value => Nat.mul(natural_groups,value),U32.to_nat(block),b,be)))

def span_division(+native: Native) -> {U32.to_nat((native_limit(native) / native_span(native) : U32)) == Nat.div(limit(model(native)),row_span(model(native))) : Nat}:
  Native{g,s,o,k,stride,limits,+budget,envelope,t,b,tp,bp,fits,te,be,lb} = native
  +denominator = native_span(native)
  +envelope = native_envelope(native)
  %span_exact(native) : {U32.to_nat((native_limit(native) / denominator : U32)) == Nat.div(limit(model(native)),_) : Nat}
  NativeDivision.divide(native_limit(native),denominator,half(envelope),half_fits(envelope),
    %Equal.sym(Nat,U32.to_nat(denominator),row_span(model(native)),span_exact(native)) : {Nat.is_le(Nat.double(_),U32.to_nat(half(envelope))) == True{} : Bool}
      Order.transitive(Nat.double(row_span(model(native))),Nat.double(workspace(model(native))),U32.to_nat(half(envelope)),Order.double_le(row_span(model(native)),workspace(model(native)),row_span_room(model(native))),workspace_double_room(native)),
    %Equal.sym(Nat,U32.to_nat(denominator),row_span(model(native)),span_exact(native)) : {Nat.is_lt(0n,_) == True{} : Bool} span_positive(model(native)))

def native_fitting(native: Native) -> U32:
  +native = native
  (native_limit(native) / native_span(native) + 1 : U32)
def native_rows(native: Native) -> U32:
  +native = native
  Bool.pick(U32,U32.is_lt(native_taps(native),native_fitting(native)),(native_fitting(native) - native_taps(native) : U32),1)

def limit_successor_room(+native: Native) -> {Nat.is_le(Nat.add(U32.to_nat(native_limit(native)),1n),U32.to_nat(half(native_envelope(native)))) == True{} : Bool}:
  Native{g,s,o,k,stride,limits,+limit,+envelope,t,b,tp,bp,fits,te,be,+lb} = native
  Geometry.WordEnvelope{maximum,+half,+quarter,+margin,me,+he,+qe,+re} = envelope
  +bound = Geometry.kernel_successor_bound(limit,quarter,margin,qe,re,lb)
  +margin_half = {%Equal.sym(U32,margin,(quarter + 1 : U32),re) : {U32.is_le(_,half) == True{} : Bool}
    %Equal.sym(U32,quarter,1073741824,qe) : {U32.is_le((_ + 1 : U32),half) == True{} : Bool}
    %Equal.sym(U32,half,2147483648,he) : {U32.is_le((1073741824 + 1 : U32),_) == True{} : Bool} {==} : {U32.is_le(margin,half) == True{} : Bool}}
  %Equal.sym(Nat,Nat.add(U32.to_nat(limit),1n),1n+U32.to_nat(limit),Addition.successor_width(U32.to_nat(limit))) : {Nat.is_le(_,U32.to_nat(half)) == True{} : Bool}
  Order.transitive(1n+U32.to_nat(limit),U32.to_nat(margin),U32.to_nat(half),bound,Bounds.as_nat_le(margin,half,margin_half))

def quotient_successor_room(+native: Native) -> {Nat.is_le(Nat.add(U32.to_nat((native_limit(native) / native_span(native) : U32)),1n),U32.to_nat(half(native_envelope(native)))) == True{} : Bool}:
  Native{g,s,o,k,stride,limits,+limit,envelope,t,b,tp,bp,fits,te,be,lb} = native
  +quotient = (limit / native_span(native) : U32)
  +qroom = {%Equal.sym(Nat,U32.to_nat(quotient),Nat.div(U32.to_nat(limit),row_span(model(native))),span_division(native)) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
    Phase.quotient_room(U32.to_nat(limit),row_span(model(native)),span_positive(model(native))) : {Nat.is_le(U32.to_nat(quotient),U32.to_nat(limit)) == True{} : Bool}}
  Order.transitive(Nat.add(U32.to_nat(quotient),1n),Nat.add(U32.to_nat(limit),1n),U32.to_nat(half(native_envelope(native))),Order.add_le(U32.to_nat(quotient),U32.to_nat(limit),1n,qroom),limit_successor_room(native))

def fitting_exact(+native: Native) -> {U32.to_nat(native_fitting(native)) == fitting(model(native)) : Nat}:
  +quotient = (native_limit(native) / native_span(native) : U32)
  +q = Nat.div(limit(model(native)),row_span(model(native)))
  Equal.trans(Nat,U32.to_nat(native_fitting(native)),Nat.add(U32.to_nat(quotient),1n),1n+q,
    Arithmetic.add(quotient,1,half(native_envelope(native)),half_fits(native_envelope(native)),quotient_successor_room(native)),
    Equal.trans(Nat,Nat.add(U32.to_nat(quotient),1n),Nat.add(q,1n),1n+q,Equal.cong(Nat,Nat,value => Nat.add(value,1n),U32.to_nat(quotient),q,span_division(native)),Addition.successor_width(q)))

def fitting_word_room(+native: Native) -> {Nat.is_le(U32.to_nat(native_fitting(native)),U32.to_nat(half(native_envelope(native)))) == True{} : Bool}:
  +quotient = (native_limit(native) / native_span(native) : U32)
  +exact = Arithmetic.add(quotient,1,half(native_envelope(native)),half_fits(native_envelope(native)),quotient_successor_room(native))
  %Equal.sym(Nat,U32.to_nat(native_fitting(native)),Nat.add(U32.to_nat(quotient),1n),exact) : {Nat.is_le(_,U32.to_nat(half(native_envelope(native)))) == True{} : Bool}
  quotient_successor_room(native)

def rows_compared(+native: Native) -> {U32.is_lt(native_taps(native),native_fitting(native)) == Nat.is_lt(taps(model(native)),fitting(model(native))) : Bool}:
  Native{g,s,o,k,stride,limits,limit,envelope,+t,b,tp,bp,fits,+te,be,lb} = native
  Equal.trans(Bool,U32.is_lt(native_taps(native),native_fitting(native)),Nat.is_lt(U32.to_nat(native_taps(native)),U32.to_nat(native_fitting(native))),Nat.is_lt(t,fitting(model(native))),Comparison.lt(native_taps(native),native_fitting(native)),
    Equal.trans(Bool,Nat.is_lt(U32.to_nat(native_taps(native)),U32.to_nat(native_fitting(native))),Nat.is_lt(t,U32.to_nat(native_fitting(native))),Nat.is_lt(t,fitting(model(native))),
      Equal.cong(Nat,Bool,value => Nat.is_lt(value,U32.to_nat(native_fitting(native))),U32.to_nat(native_taps(native)),t,te),Equal.cong(Nat,Bool,value => Nat.is_lt(t,value),U32.to_nat(native_fitting(native)),fitting(model(native)),fitting_exact(native))))

def rows_exact_case(selected: Bool,+native: Native,+compared: {U32.is_lt(native_taps(native),native_fitting(native)) == selected : Bool}) ->
  {U32.to_nat(Bool.pick(U32,selected,(native_fitting(native) - native_taps(native) : U32),1)) == Bool.pick(Nat,selected,Nat.sub(fitting(model(native)),taps(model(native))),1n) : Nat}:
  match selected:
    case False{}: {==}
    case True{}:
      Native{g,s,o,k,stride,limits,limit,envelope,+t,b,tp,bp,fits,+te,be,lb} = native
      +fit_word = native_fitting(native)
      +tap = native_taps(native)
      +ordered = Bounds.strict_le(U32.to_nat(tap),U32.to_nat(fit_word),Bounds.as_nat_lt(tap,fit_word,compared))
      Equal.trans(Nat,U32.to_nat((fit_word - tap : U32)),Nat.sub(U32.to_nat(fit_word),U32.to_nat(tap)),Nat.sub(fitting(model(native)),t),
        Arithmetic.subtract(fit_word,tap,half(native_envelope(native)),half_fits(native_envelope(native)),ordered,
          Order.transitive(Nat.sub(U32.to_nat(fit_word),U32.to_nat(tap)),U32.to_nat(fit_word),U32.to_nat(half(native_envelope(native))),Order.sub_le(U32.to_nat(fit_word),U32.to_nat(tap)),fitting_word_room(native))),
        Equal.trans(Nat,Nat.sub(U32.to_nat(fit_word),U32.to_nat(tap)),Nat.sub(fitting(model(native)),U32.to_nat(tap)),Nat.sub(fitting(model(native)),t),
          Equal.cong(Nat,Nat,value => Nat.sub(value,U32.to_nat(tap)),U32.to_nat(fit_word),fitting(model(native)),fitting_exact(native)),Equal.cong(Nat,Nat,value => Nat.sub(fitting(model(native)),value),U32.to_nat(tap),t,te)))

def rows_exact(+native: Native) -> {U32.to_nat(native_rows(native)) == chunk_rows(model(native)) : Nat}:
  +selected = U32.is_lt(native_taps(native),native_fitting(native))
  Equal.trans(Nat,U32.to_nat(native_rows(native)),Bool.pick(Nat,selected,Nat.sub(fitting(model(native)),taps(model(native))),1n),chunk_rows(model(native)),rows_exact_case(selected,native,{==}),
    Equal.cong(Bool,Nat,value => Bool.pick(Nat,value,Nat.sub(fitting(model(native)),taps(model(native))),1n),selected,Nat.is_lt(taps(model(native)),fitting(model(native))),rows_compared(native)))

def layout_groups(layout: Kernel.Tiling) -> U32:
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  groups
def outputs_word(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  o
def block_size(+geometry: Kernel.Geometry) -> Nat:
  Nat.add(Nat.mul(Domain.outputs(geometry),U32.to_nat(Columns.layout_shifts(Kernel.tiling(geometry)))),1n)
def maximum_block(+geometry: Kernel.Geometry) -> Nat:
  Nat.add(Nat.mul(Domain.outputs(geometry),1n+Geometry.halo(geometry)),1n)
def fits_room(+call: Domain.Call,+fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Geometry.halo(Domain.geometry(call)),maximum_block(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  %Addition.successor_width(Geometry.halo(geometry)) : {Nat.is_le(Nat.mul(Geometry.halo(geometry),Nat.add(Nat.mul(Domain.outputs(geometry),_),1n)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  %Geometry.fits_exact(call) : {_ == True{} : Bool}
  fits

def halo_positive(+call: Domain.Call) -> {Nat.is_lt(0n,Geometry.halo(Domain.geometry(call))) == True{} : Bool}:
  %Geometry.validated_taps_exact(call) : {Nat.is_lt(0n,_) == True{} : Bool}
  Geometry.taps_positive(call)
def block_positive(+geometry: Kernel.Geometry) -> {Nat.is_lt(0n,block_size(geometry)) == True{} : Bool}:
  +product = Nat.mul(Domain.outputs(geometry),U32.to_nat(Columns.layout_shifts(Kernel.tiling(geometry))))
  %Equal.sym(Nat,Nat.add(product,1n),1n+product,Addition.successor_width(product)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.lt_succ(0n,product,Order.zero_le(product))

def geometry_cell_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Geometry.halo(Domain.geometry(call)),block_size(Domain.geometry(call))),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  +shifts = U32.to_nat(Columns.layout_shifts(Kernel.tiling(geometry)))
  +outputs = Domain.outputs(geometry)
  +halo = Geometry.halo(geometry)
  +block_bound = Order.add_le(Nat.mul(outputs,shifts),Nat.mul(outputs,1n+halo),1n,Division.multiply_left_le(outputs,shifts,1n+halo,Columns.tiling_shifts_bounded(call,envelope)))
  Order.transitive(Nat.mul(halo,block_size(geometry)),Nat.mul(halo,maximum_block(geometry)),U32.to_nat(Machine.capacity(limits)),Division.multiply_left_le(halo,block_size(geometry),maximum_block(geometry),block_bound),fits_room(call,fits))

def geometry_block_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) ->
  {Nat.is_le(block_size(Domain.geometry(call)),U32.to_nat(Machine.capacity(Domain.limits(call)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Order.transitive(block_size(geometry),Nat.mul(Geometry.halo(geometry),block_size(geometry)),U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(block_size(geometry),Geometry.halo(geometry)) : {Nat.is_le(block_size(geometry),_) == True{} : Bool} RowMajor.product_covers(block_size(geometry),Geometry.halo(geometry),halo_positive(call)),
    geometry_cell_room(call,envelope,fits))

def geometry_block_exact(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) ->
  {U32.to_nat((outputs_word(Domain.geometry(call)) * Columns.layout_shifts(Kernel.tiling(Domain.geometry(call))) + 1 : U32)) == block_size(Domain.geometry(call)) : Nat}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Arithmetic.multiply_add_exact(o,Columns.layout_shifts(Kernel.tiling(geometry)),1,Machine.capacity(limits),Limits.capacity_fits(limits),geometry_block_room(call,envelope,fits))

# The scalar readback fields above are derived from the tiling and the fit.
def fitted_native(+call: Domain.Call,+envelope: Geometry.WordEnvelope,+limit: U32,
  +limit_bound: {U32.is_le(limit,quarter(envelope)) == True{} : Bool},
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) -> Native:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Native{layout_groups(Kernel.tiling(geometry)),Columns.layout_shifts(Kernel.tiling(geometry)),o,k,s,limits,limit,envelope,Geometry.halo(geometry),block_size(geometry),
    halo_positive(call),block_positive(geometry),geometry_cell_room(call,envelope,fits),Geometry.validated_taps_exact(call),
    geometry_block_exact(call,envelope,fits),limit_bound}

def layout_chunk_groups(layout: Kernel.Tiling) -> U32:
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  chunk_groups
def layout_chunk_rows(layout: Kernel.Tiling) -> U32:
  Kernel.Tiling{groups,shifts,outputs,column_low,kernel,height,width,chunk_groups,chunk_rows} = layout
  chunk_rows
def pipeline_bound(+envelope: Geometry.WordEnvelope) -> {U32.is_le(Pipeline.block_tiles(),quarter(envelope)) == True{} : Bool}:
  Geometry.WordEnvelope{maximum,half,quarter,margin,me,he,qe,re} = envelope
  %Equal.sym(U32,quarter,1073741824,qe) : {U32.is_le(Pipeline.block_tiles(),_) == True{} : Bool}
  {==}

def tiled_native(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) -> Native:
  fitted_native(call,envelope,Pipeline.block_tiles(),pipeline_bound(envelope),fits)

def tiled_groups_matches(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) ->
  {layout_chunk_groups(Kernel.tiling(Domain.geometry(call))) == native_groups(tiled_native(call,envelope,fits)) : U32}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}
def tiled_rows_matches(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool}) ->
  {layout_chunk_rows(Kernel.tiling(Domain.geometry(call))) == native_rows(tiled_native(call,envelope,fits)) : U32}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}

# The actual number of tile rows is supplied by band_rows, not an agreement
# assumption about a finished chunk. Only the chunk's contained band is needed.
def contained_tile_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +rows_bounded: {Nat.is_le(U32.to_nat(count),U32.to_nat(layout_chunk_rows(Kernel.tiling(Domain.geometry(call))))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(U32.to_nat(Kernel.band_rows(Domain.geometry(call),first,count)),row_span(model(tiled_native(call,envelope,fits)))),workspace(model(tiled_native(call,envelope,fits)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +native = tiled_native(call,envelope,fits)
  +plan = model(native)
  +count_bounded = {%rows_exact(native) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool}
    %tiled_rows_matches(call,envelope,fits) : {Nat.is_le(U32.to_nat(count),U32.to_nat(_)) == True{} : Bool} rows_bounded : {Nat.is_le(U32.to_nat(count),chunk_rows(plan)) == True{} : Bool}}
  tile_room(plan,U32.to_nat(count),U32.to_nat(Kernel.band_rows(Domain.geometry(call),first,count)),count_bounded,
    Bands.contained_band_rows(call,envelope,first,count,contained))

def tile_product(+native: Native,+rows: U32) -> U32:
  (rows * native_groups(native) * native_block(native) : U32)
def tile_product_exact(+native: Native,+rows: U32,
  +room: {Nat.is_le(Nat.mul(U32.to_nat(rows),row_span(model(native))),workspace(model(native))) == True{} : Bool}) ->
  {U32.to_nat(tile_product(native,rows)) == Nat.mul(U32.to_nat(rows),row_span(model(native))) : Nat}:
  Native{g,s,o,k,stride,limits,limit,envelope,t,+b,tp,+bp,fits,te,+be,lb} = native
  +groups = native_groups(native)
  +gn = chunk_groups(model(native))
  +block = native_block(native)
  +rn = U32.to_nat(rows)
  +product = Nat.mul(rn,gn)
  +gn_exact = groups_exact(native)
  +product_room = Order.transitive(product,Nat.mul(rn,row_span(model(native))),workspace(model(native)),
    %RowMajor.assoc(rn,gn,b) : {Nat.is_le(product,_) == True{} : Bool} RowMajor.product_covers(product,b,bp),room)
  +first_exact = Equal.trans(Nat,U32.to_nat((rows * groups : U32)),Nat.mul(rn,U32.to_nat(groups)),product,
    Arithmetic.multiply(rows,groups,native_workspace(native),workspace_fits(native),
    %Equal.sym(Nat,U32.to_nat(groups),gn,gn_exact) : {Nat.is_le(Nat.mul(rn,_),U32.to_nat(native_workspace(native))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(native_workspace(native)),workspace(model(native)),workspace_exact(native)) : {Nat.is_le(product,_) == True{} : Bool} product_room),
    Equal.cong(Nat,Nat,value => Nat.mul(rn,value),U32.to_nat(groups),gn,gn_exact))
  +full_room = {%Equal.sym(Nat,Nat.mul(product,b),Nat.mul(rn,row_span(model(native))),RowMajor.assoc(rn,gn,b)) : {Nat.is_le(_,workspace(model(native))) == True{} : Bool} room : {Nat.is_le(Nat.mul(product,b),workspace(model(native))) == True{} : Bool}}
  Equal.trans(Nat,U32.to_nat(tile_product(native,rows)),Nat.mul(product,b),Nat.mul(rn,row_span(model(native))),
    Equal.trans(Nat,U32.to_nat(tile_product(native,rows)),Nat.mul(U32.to_nat((rows * groups : U32)),U32.to_nat(block)),Nat.mul(product,b),
      Arithmetic.multiply((rows * groups : U32),block,native_workspace(native),workspace_fits(native),
        %Equal.sym(Nat,U32.to_nat((rows * groups : U32)),product,first_exact) : {Nat.is_le(Nat.mul(_,U32.to_nat(block)),U32.to_nat(native_workspace(native))) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(block),b,be) : {Nat.is_le(Nat.mul(product,_),U32.to_nat(native_workspace(native))) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(native_workspace(native)),workspace(model(native)),workspace_exact(native)) : {Nat.is_le(Nat.mul(product,b),_) == True{} : Bool} full_room),
      Equal.trans(Nat,Nat.mul(U32.to_nat((rows * groups : U32)),U32.to_nat(block)),Nat.mul(product,U32.to_nat(block)),Nat.mul(product,b),
        Equal.cong(Nat,Nat,value => Nat.mul(value,U32.to_nat(block)),U32.to_nat((rows * groups : U32)),product,first_exact),Equal.cong(Nat,Nat,value => Nat.mul(product,value),U32.to_nat(block),b,be))),RowMajor.assoc(rn,gn,b))

def tiled_product_matches(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},+first: U32,+count: U32) ->
  {(Kernel.band_rows(Domain.geometry(call),first,count) * layout_chunk_groups(Kernel.tiling(Domain.geometry(call))) * (outputs_word(Domain.geometry(call)) * Columns.layout_shifts(Kernel.tiling(Domain.geometry(call))) + 1 : U32) : U32) ==
    tile_product(tiled_native(call,envelope,fits),Kernel.band_rows(Domain.geometry(call),first,count)) : U32}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +native = tiled_native(call,envelope,fits)
  +rows = Kernel.band_rows(Domain.geometry(call),first,count)
  Equal.cong(U32,U32,groups => (rows * groups * (outputs_word(Domain.geometry(call)) * Columns.layout_shifts(Kernel.tiling(Domain.geometry(call))) + 1 : U32) : U32),layout_chunk_groups(Kernel.tiling(Domain.geometry(call))),native_groups(native),tiled_groups_matches(call,envelope,fits))

def tiled_count(+geometry: Kernel.Geometry,+first: U32,+count: U32) -> U32:
  (Kernel.band_rows(geometry,first,count) * layout_chunk_groups(Kernel.tiling(geometry)) * (outputs_word(geometry) * Columns.layout_shifts(Kernel.tiling(geometry)) + 1 : U32) : U32)

def tiled_count_room(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +rows_bounded: {Nat.is_le(U32.to_nat(count),U32.to_nat(layout_chunk_rows(Kernel.tiling(Domain.geometry(call))))) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(tiled_count(Domain.geometry(call),first,count)),U32.to_nat(native_workspace(tiled_native(call,envelope,fits)))) == True{} : Bool}:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +native = tiled_native(call,envelope,fits)
  +rows = Kernel.band_rows(Domain.geometry(call),first,count)
  +room = contained_tile_room(call,envelope,fits,first,count,contained,rows_bounded)
  +exact = tile_product_exact(native,rows,room)
  %Equal.sym(U32,tiled_count(Domain.geometry(call),first,count),tile_product(native,rows),tiled_product_matches(call,envelope,fits,first,count)) :
    {Nat.is_le(U32.to_nat(_),U32.to_nat(native_workspace(native))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(tile_product(native,rows)),Nat.mul(U32.to_nat(rows),row_span(model(native))),exact) : {Nat.is_le(_,U32.to_nat(native_workspace(native))) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(native_workspace(native)),workspace(model(native)),workspace_exact(native)) : {Nat.is_le(Nat.mul(U32.to_nat(rows),row_span(model(native))),_) == True{} : Bool}
  room

def quarter_below_half(+envelope: Geometry.WordEnvelope) -> {U32.is_lt(quarter(envelope),half(envelope)) == True{} : Bool}:
  Geometry.WordEnvelope{maximum,+half,+quarter,margin,me,+he,+qe,re} = envelope
  %Equal.sym(U32,quarter,1073741824,qe) : {U32.is_lt(_,half) == True{} : Bool}
  %Equal.sym(U32,half,2147483648,he) : {U32.is_lt(1073741824,_) == True{} : Bool}
  {==}

def tiled_guarded(+call: Domain.Call,+envelope: Geometry.WordEnvelope,
  +fits: {Kernel.fits(Domain.geometry(call),Machine.capacity(Domain.limits(call))) == True{} : Bool},+first: U32,+count: U32,
  +contained: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Domain.height(Domain.geometry(call))) == True{} : Bool},
  +rows_bounded: {Nat.is_le(U32.to_nat(count),U32.to_nat(layout_chunk_rows(Kernel.tiling(Domain.geometry(call))))) == True{} : Bool}) -> Depth.Guarded<tiled_count(Domain.geometry(call),first,count)>:
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  +native = tiled_native(call,envelope,fits)
  +count_word = tiled_count(Domain.geometry(call),first,count)
  +capacity = native_workspace(native)
  +quarter = quarter(envelope)
  +half = half(envelope)
  +room = tiled_count_room(call,envelope,fits,first,count,contained,rows_bounded)
  +below = {%Equal.sym(Bool,U32.is_lt(count_word,half),Nat.is_lt(U32.to_nat(count_word),U32.to_nat(half)),Comparison.lt(count_word,half)) : {_ == True{} : Bool}
    Order.le_lt_trans(U32.to_nat(count_word),U32.to_nat(quarter),U32.to_nat(half),Order.transitive(U32.to_nat(count_word),U32.to_nat(capacity),U32.to_nat(quarter),room,Bounds.as_nat_le(capacity,quarter,workspace_quarter(native))),Bounds.as_nat_lt(quarter,half,quarter_below_half(envelope))) : {U32.is_lt(count_word,half) == True{} : Bool}}
  Depth.guarded(count_word,Bounds.uint_trans(count_word,half,2147483648,below,half_fits(envelope)))

def rows_room(+plan: Plan) -> {Nat.is_le(chunk_rows(plan),workspace(plan)) == True{} : Bool}:
  Plan{groups,+taps,block,limit,capacity,+tp,bp,fits} = plan
  +rows = chunk_rows(plan)
  +last = Nat.sub(rows,1n)
  +halo = halo_rows(plan)
  +rows_halo = {%Order.positive_has_last(rows,rows_positive(plan)) : {Nat.is_le(_,halo) == True{} : Bool}
    %Addition.successor_width(last) : {Nat.is_le(_,halo) == True{} : Bool}
    Order.offset_le(last,1n,taps,Order.positive_one(taps,tp)) : {Nat.is_le(rows,halo) == True{} : Bool}}
  Order.transitive(rows,halo,workspace(plan),rows_halo,
    Order.transitive(halo,Nat.mul(halo,row_span(plan)),workspace(plan),RowMajor.product_covers(halo,row_span(plan),span_positive(plan)),halo_room(plan)))
