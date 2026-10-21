# The actual gradient copy interval contains every valid tapped halo row.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_tiled_band_extents.bend as Bands
import ./input_gradient_tiled_extents.bend as Extents
import ./nat_order.bend as Order
import ./row_major_index.bend as RowMajor
import ./machine_limit_bounds.bend as Limits
import ./nat_division_order.bend as Division
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./nat_extrema.bend as Extrema
import ./u32_extrema.bend as NativeExtrema
import ./u32_full_range.bend as FullRange
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_subtraction.bend as Subtraction
import ./padded_interval.bend as Padded
import ./storage_shape.bend as Shape
import ./word_encoding.bend as WordEncoding

def clipped_origin(+coordinate: U32,+kernel: U32) -> U32:
  Bool.pick(U32,U32.is_lt(kernel,coordinate),(coordinate - kernel : U32),0)
def clipped_origin_case(selected: Bool,+coordinate: U32,+kernel: U32,+compared: {U32.is_lt(kernel,coordinate) == selected : Bool}) ->
  {U32.to_nat(Bool.pick(U32,selected,(coordinate - kernel : U32),0)) == Nat.sub(U32.to_nat(coordinate),U32.to_nat(kernel)) : Nat}:
  match selected:
    case True{}: FullRange.subtract(coordinate,kernel,coordinate,Bounds.strict_le(U32.to_nat(kernel),U32.to_nat(coordinate),Bounds.as_nat_lt(kernel,coordinate,compared)),Order.sub_le(U32.to_nat(coordinate),U32.to_nat(kernel)))
    case False{}:
      +not_less = {%Comparison.lt(kernel,coordinate) : {_ == False{} : Bool} compared : {Nat.is_lt(U32.to_nat(kernel),U32.to_nat(coordinate)) == False{} : Bool}}
      Equal.sym(Nat,Nat.sub(U32.to_nat(coordinate),U32.to_nat(kernel)),0n,Order.subtract_below(U32.to_nat(coordinate),U32.to_nat(kernel),Order.not_less(U32.to_nat(kernel),U32.to_nat(coordinate),not_less)))
def clipped_origin_exact(+coordinate: U32,+kernel: U32) ->
  {U32.to_nat(clipped_origin(coordinate,kernel)) == Nat.sub(U32.to_nat(coordinate),U32.to_nat(kernel)) : Nat}:
  clipped_origin_case(U32.is_lt(kernel,coordinate),coordinate,kernel,{==})
def clipped_stop(+coordinate: U32,+kernel: U32,+height: U32) -> U32:
  Bool.pick(U32,U32.is_lt(kernel,coordinate),U32.min((coordinate - kernel : U32),height),0)
def clipped_stop_case(selected: Bool,+coordinate: U32,+kernel: U32,+height: U32,+compared: {U32.is_lt(kernel,coordinate) == selected : Bool}) ->
  {U32.to_nat(Bool.pick(U32,selected,U32.min((coordinate - kernel : U32),height),0)) == Nat.min(Nat.sub(U32.to_nat(coordinate),U32.to_nat(kernel)),U32.to_nat(height)) : Nat}:
  match selected:
    case True{}:
      Equal.trans(Nat,U32.to_nat(U32.min((coordinate - kernel : U32),height)),Nat.min(U32.to_nat((coordinate - kernel : U32)),U32.to_nat(height)),Nat.min(Nat.sub(U32.to_nat(coordinate),U32.to_nat(kernel)),U32.to_nat(height)),NativeExtrema.minimum((coordinate - kernel : U32),height),
        Equal.cong(Nat,Nat,value => Nat.min(value,U32.to_nat(height)),U32.to_nat((coordinate - kernel : U32)),Nat.sub(U32.to_nat(coordinate),U32.to_nat(kernel)),FullRange.subtract(coordinate,kernel,coordinate,Bounds.strict_le(U32.to_nat(kernel),U32.to_nat(coordinate),Bounds.as_nat_lt(kernel,coordinate,compared)),Order.sub_le(U32.to_nat(coordinate),U32.to_nat(kernel)))))
    case False{}:
      +not_less = {%Comparison.lt(kernel,coordinate) : {_ == False{} : Bool} compared : {Nat.is_lt(U32.to_nat(kernel),U32.to_nat(coordinate)) == False{} : Bool}}
      +difference = Order.subtract_below(U32.to_nat(coordinate),U32.to_nat(kernel),Order.not_less(U32.to_nat(kernel),U32.to_nat(coordinate),not_less))
      %Equal.sym(Nat,Nat.sub(U32.to_nat(coordinate),U32.to_nat(kernel)),0n,difference) : {0n == Nat.min(_,U32.to_nat(height)) : Nat}
      Equal.sym(Nat,Nat.min(0n,U32.to_nat(height)),0n,Extrema.minimum_when_ordered(0n,U32.to_nat(height),Order.zero_le(U32.to_nat(height))))
def clipped_stop_exact(+coordinate: U32,+kernel: U32,+height: U32) ->
  {U32.to_nat(clipped_stop(coordinate,kernel,height)) == Nat.min(Nat.sub(U32.to_nat(coordinate),U32.to_nat(kernel)),U32.to_nat(height)) : Nat}:
  clipped_stop_case(U32.is_lt(kernel,coordinate),coordinate,kernel,height,{==})

def begin(+query: Bands.Query) -> U32:
  clipped_origin(Kernel.band_low(Bands.geometry(query),Bands.first(query),Bands.last(query)),Domain.kernel_word(Bands.geometry(query)))
def stop(+query: Bands.Query) -> U32:
  clipped_stop(Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query)),Domain.kernel_word(Bands.geometry(query)),Domain.gradient_rows(Bands.geometry(query)))
def row_count(+query: Bands.Query) -> U32:
  clipped_origin(stop(query),begin(query))
def band_end_matches(+query: Bands.Query) -> {(Bands.first(query) + Bands.rows_word(query) : U32) == Bands.last(query) : U32}:
  Equal.trans(U32,(Bands.first(query) + Bands.rows_word(query) : U32),(Bands.rows_word(query) + Bands.first(query) : U32),Bands.last(query),Arithmetic.add_commutative(Bands.first(query),Bands.rows_word(query)),Subtraction.add_subtract(Bands.last(query),Bands.first(query)))
def begin_matches(+query: Bands.Query) -> {Kernel.copy_start(Bands.geometry(query),Bands.first(query),Bands.rows_word(query)) == begin(query) : U32}:
  Bands.Query{Extents.Axis{+call,size,size_room,e},+first,+last,ordered,room} = query
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw} = g
  Equal.cong(U32,U32,last => clipped_origin(Kernel.band_low(g,first,last),k),(first + Bands.rows_word(query) : U32),last,band_end_matches(query))
def count_matches(+query: Bands.Query) -> {Kernel.copy_count(Bands.geometry(query),Bands.first(query),Bands.rows_word(query)) == row_count(query) : U32}:
  Bands.Query{Extents.Axis{+call,size,size_room,e},+first,+last,ordered,room} = query
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,+k,s,p,+gh,gw} = g
  Equal.cong(U32,U32,last => clipped_origin(clipped_stop(Kernel.band_high(g,first,last),k,gh),clipped_origin(Kernel.band_low(g,first,last),k)),(first + Bands.rows_word(query) : U32),last,band_end_matches(query))
def begin_exact(+query: Bands.Query) ->
  {U32.to_nat(begin(query)) == Nat.sub(U32.to_nat(Kernel.band_low(Bands.geometry(query),Bands.first(query),Bands.last(query))),Domain.kernel(Bands.geometry(query))) : Nat}:
  Bands.Query{Extents.Axis{+call,size,size_room,e},first,last,ordered,room} = query
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  clipped_origin_exact(Kernel.band_low(g,first,last),k)
def stop_exact(+query: Bands.Query) ->
  {U32.to_nat(stop(query)) == Nat.min(Nat.sub(U32.to_nat(Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query))),Domain.kernel(Bands.geometry(query))),Domain.gradient_height(Bands.geometry(query))) : Nat}:
  Bands.Query{Extents.Axis{+call,size,size_room,e},first,last,ordered,room} = query
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  clipped_stop_exact(Kernel.band_high(g,first,last),k,gh)
def count_exact(+query: Bands.Query) -> {U32.to_nat(row_count(query)) == Nat.sub(U32.to_nat(stop(query)),U32.to_nat(begin(query))) : Nat}:
  clipped_origin_exact(stop(query),begin(query))

def gradient_kernel_room(+call: Domain.Call) ->
  {Nat.is_le(Nat.add(Domain.gradient_height(Domain.geometry(call)),Domain.kernel(Domain.geometry(call))),U32.to_nat(Machine.coordinate(Domain.limits(call)))) == True{} : Bool}:
  +geometry = Domain.geometry(call)
  +limits = Domain.limits(call)
  +rows = Domain.gradient_height(geometry)
  +stride = Domain.stride(geometry)
  +kernel = Domain.kernel(geometry)
  Order.transitive(Nat.add(rows,kernel),Nat.add(Nat.mul(rows,stride),kernel),U32.to_nat(Machine.coordinate(limits)),Order.add_le(rows,Nat.mul(rows,stride),kernel,RowMajor.product_covers(rows,stride,Domain.stride_positive(call))),Domain.row_source_room(call))
def query_gradient_kernel_room(+query: Bands.Query) ->
  {Nat.is_le(Nat.add(Domain.gradient_height(Bands.geometry(query)),Domain.kernel(Bands.geometry(query))),U32.to_nat(Machine.coordinate(Bands.query_limits(query)))) == True{} : Bool}:
  Bands.Query{Extents.Axis{+call,size,size_room,e},first,last,ordered,room} = query
  gradient_kernel_room(call)
def copied_row_bounds_from(+query: Bands.Query,+coordinate: U32,
  +lower: {U32.is_le(Kernel.band_low(Bands.geometry(query),Bands.first(query),Bands.last(query)),coordinate) == True{} : Bool},
  +upper: {U32.is_lt(coordinate,Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query))) == True{} : Bool},
  +row_inside: {U32.is_lt((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32),Domain.gradient_rows(Bands.geometry(query))) == True{} : Bool},
  +kernel_before: {Nat.is_le(Domain.kernel(Bands.geometry(query)),U32.to_nat(coordinate)) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat(begin(query)),U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32)),U32.to_nat(stop(query))) == True{} : Bool}>:
  Bands.Query{Extents.Axis{+call,size,size_room,e},+first,+last,ordered,room} = query
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,+k,s,p,+gh,gw} = g
  +low = Kernel.band_low(g,first,last)
  +high = Kernel.band_high(g,first,last)
  +row = (coordinate - k : U32)
  +exact = FullRange.subtract(coordinate,k,coordinate,kernel_before,Order.sub_le(U32.to_nat(coordinate),U32.to_nat(k)))
  +kernel_high = Order.le_lt_trans(U32.to_nat(k),U32.to_nat(coordinate),U32.to_nat(high),kernel_before,Bounds.as_nat_lt(coordinate,high,upper))
  +high_exact = FullRange.subtract(high,k,high,Bounds.strict_le(U32.to_nat(k),U32.to_nat(high),kernel_high),Order.sub_le(U32.to_nat(high),U32.to_nat(k)))
  +kernel_word_before = {%Equal.sym(Bool,U32.is_le(k,coordinate),Nat.is_le(U32.to_nat(k),U32.to_nat(coordinate)),Comparison.le(k,coordinate)) : {_ == True{} : Bool} kernel_before : {U32.is_le(k,coordinate) == True{} : Bool}}
  Shape.Both{
    %Equal.sym(Nat,U32.to_nat(begin(query)),Nat.sub(U32.to_nat(low),U32.to_nat(k)),begin_exact(query)) : {Nat.is_le(_,U32.to_nat(row)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(row),Nat.sub(U32.to_nat(coordinate),U32.to_nat(k)),exact) : {Nat.is_le(Nat.sub(U32.to_nat(low),U32.to_nat(k)),_) == True{} : Bool} Division.subtract_le(U32.to_nat(low),U32.to_nat(coordinate),U32.to_nat(k),Bounds.as_nat_le(low,coordinate,lower)),
    %Equal.sym(Nat,U32.to_nat(stop(query)),Nat.min(Nat.sub(U32.to_nat(high),U32.to_nat(k)),U32.to_nat(gh)),stop_exact(query)) : {Nat.is_lt(U32.to_nat(row),_) == True{} : Bool}
    Extrema.minimum_strict_lower(Nat.sub(U32.to_nat(high),U32.to_nat(k)),U32.to_nat(gh),U32.to_nat(row),
      %high_exact : {Nat.is_lt(U32.to_nat(row),_) == True{} : Bool} FullRange.interval_inside(k,high,coordinate,kernel_word_before,upper),Bounds.as_nat_lt(row,gh,row_inside))}
def copied_row_lower_finish(+query: Bands.Query,+coordinate: U32,
  +lower: {U32.is_le(Kernel.band_low(Bands.geometry(query),Bands.first(query),Bands.last(query)),coordinate) == True{} : Bool},
  +upper: {U32.is_lt(coordinate,Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query))) == True{} : Bool},
  +row_inside: {U32.is_lt((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32),Domain.gradient_rows(Bands.geometry(query))) == True{} : Bool},
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(Domain.kernel_word(Bands.geometry(query))),U32.to_nat(coordinate)) == True{} : Bool},
    {Nat.is_lt(U32.to_nat(coordinate),Nat.add(U32.to_nat(Domain.kernel_word(Bands.geometry(query))),U32.to_nat(Domain.gradient_rows(Bands.geometry(query))))) == True{} : Bool}>) ->
  Shape.Both<{Nat.is_le(U32.to_nat(begin(query)),U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32)),U32.to_nat(stop(query))) == True{} : Bool}>:
  Bands.Query{Extents.Axis{+call,size,size_room,e},first,last,ordered,room} = query
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  Shape.Both{+kernel_before,hi} = bounds
  copied_row_bounds_from(query,coordinate,lower,upper,row_inside,kernel_before)
def copied_row_bounds(+query: Bands.Query,+coordinate: U32,
  +lower: {U32.is_le(Kernel.band_low(Bands.geometry(query),Bands.first(query),Bands.last(query)),coordinate) == True{} : Bool},
  +upper: {U32.is_lt(coordinate,Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query))) == True{} : Bool},
  +row_inside: {U32.is_lt((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32),Domain.gradient_rows(Bands.geometry(query))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat(begin(query)),U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32)),U32.to_nat(stop(query))) == True{} : Bool}>:
  Bands.Query{Extents.Axis{+call,size,size_room,e},first,last,ordered,room} = query
  Domain.Call{+g,b,l,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +kernel = Domain.kernel_word(g)
  +height = Domain.gradient_rows(g)
  +limits = Bands.query_limits(query)
  copied_row_lower_finish(query,coordinate,lower,upper,row_inside,Padded.forward(coordinate,kernel,height,Machine.coordinate(limits),Limits.coordinate_fits(limits),
    query_gradient_kernel_room(query),row_inside))

def copied_interval_from(+query: Bands.Query,+coordinate: U32,
  bounds: Shape.Both<{Nat.is_le(U32.to_nat(begin(query)),U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32)),U32.to_nat(stop(query))) == True{} : Bool}>) ->
  Shape.Both<{Nat.is_le(U32.to_nat(begin(query)),U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32)),Nat.add(U32.to_nat(begin(query)),U32.to_nat(row_count(query)))) == True{} : Bool}>:
  Shape.Both{+lo,+hi} = bounds
  +row = (coordinate - Domain.kernel_word(Bands.geometry(query)) : U32)
  +start = U32.to_nat(begin(query))
  +finish = U32.to_nat(stop(query))
  +ordered = Bounds.strict_le(start,finish,Order.le_lt_trans(start,U32.to_nat(row),finish,lo,hi))
  +restored = Order.restore_shift(start,finish,ordered)
  Shape.Both{lo,
    %Equal.sym(Nat,U32.to_nat(row_count(query)),Nat.sub(finish,start),count_exact(query)) : {Nat.is_lt(U32.to_nat(row),Nat.add(start,_)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(start,Nat.sub(finish,start)),finish,restored) : {Nat.is_lt(U32.to_nat(row),_) == True{} : Bool} hi}
def copied_interval(+query: Bands.Query,+coordinate: U32,
  +lower: {U32.is_le(Kernel.band_low(Bands.geometry(query),Bands.first(query),Bands.last(query)),coordinate) == True{} : Bool},
  +upper: {U32.is_lt(coordinate,Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query))) == True{} : Bool},
  +row_inside: {U32.is_lt((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32),Domain.gradient_rows(Bands.geometry(query))) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat(begin(query)),U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32))) == True{} : Bool},
    {Nat.is_lt(U32.to_nat((coordinate - Domain.kernel_word(Bands.geometry(query)) : U32)),Nat.add(U32.to_nat(begin(query)),U32.to_nat(row_count(query)))) == True{} : Bool}>:
  copied_interval_from(query,coordinate,copied_row_bounds(query,coordinate,lower,upper,row_inside))

def stop_bounded(+query: Bands.Query) -> {Nat.is_le(U32.to_nat(stop(query)),Domain.gradient_height(Bands.geometry(query))) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(stop(query)),Nat.min(Nat.sub(U32.to_nat(Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query))),Domain.kernel(Bands.geometry(query))),Domain.gradient_height(Bands.geometry(query))),stop_exact(query)) :
    {Nat.is_le(_,Domain.gradient_height(Bands.geometry(query))) == True{} : Bool}
  Extrema.minimum_right(Nat.sub(U32.to_nat(Kernel.band_high(Bands.geometry(query),Bands.first(query),Bands.last(query))),Domain.kernel(Bands.geometry(query))),Domain.gradient_height(Bands.geometry(query)))
def required_row_inside_case(selected: Bool,+query: Bands.Query,+row: U32,
  +same: {U32.is_lt(begin(query),stop(query)) == selected : Bool},
  +upper: {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(begin(query)),U32.to_nat(row_count(query)))) == True{} : Bool},
  +lower: {Nat.is_le(U32.to_nat(begin(query)),U32.to_nat(row)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(row),Domain.gradient_height(Bands.geometry(query))) == True{} : Bool}:
  match selected:
    case False{}:
      Empty.absurd({Nat.is_lt(U32.to_nat(row),Domain.gradient_height(Bands.geometry(query))) == True{} : Bool},
        WordEncoding.false_true(%Order.not_below(U32.to_nat(begin(query)),U32.to_nat(row),lower) : {_ == True{} : Bool}
          %Addition.zero_right(U32.to_nat(begin(query))) : {Nat.is_lt(U32.to_nat(row),_) == True{} : Bool}
          %same : {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(begin(query)),U32.to_nat(Bool.pick(U32,_,(stop(query) - begin(query) : U32),0)))) == True{} : Bool} upper))
    case True{}:
      +first = U32.to_nat(begin(query))
      +last = U32.to_nat(stop(query))
      +ordered = Bounds.strict_le(first,last,Bounds.as_nat_lt(begin(query),stop(query),same))
      +count_read = count_exact(query)
      Bounds.nat_trans(U32.to_nat(row),last,Domain.gradient_height(Bands.geometry(query)),
        %Order.restore_shift(first,last,ordered) : {Nat.is_lt(U32.to_nat(row),_) == True{} : Bool}
        %count_read : {Nat.is_lt(U32.to_nat(row),Nat.add(first,_)) == True{} : Bool} upper,stop_bounded(query))
def required_row_inside(+query: Bands.Query,+row: U32,
  +lower: {Nat.is_le(U32.to_nat(begin(query)),U32.to_nat(row)) == True{} : Bool},
  +upper: {Nat.is_lt(U32.to_nat(row),Nat.add(U32.to_nat(begin(query)),U32.to_nat(row_count(query)))) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat(row),Domain.gradient_height(Bands.geometry(query))) == True{} : Bool}:
  required_row_inside_case(U32.is_lt(begin(query),stop(query)),query,row,{==},upper,lower)
