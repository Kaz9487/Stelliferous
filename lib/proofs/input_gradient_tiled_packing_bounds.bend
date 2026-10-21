# The actual phase traversal fits the weight and offset allocations.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../machine_limits.bend as Machine
import ../traversal_array.bend as Traversal
import ./input_gradient_phase_domain.bend as Domain
import ./input_gradient_phase_geometry.bend as Geometry
import ./input_gradient_phase_bounds.bend as PhaseBounds
import ./input_gradient_phase_separation.bend as Separation
import ./input_gradient_phase_window_weights.bend as PhaseWeights
import ./input_gradient_tiled_totals.bend as Totals
import ./natural_range_sums.bend as Sums
import ./machine_limit_bounds.bend as Limits
import ./nat_extrema.bend as Extrema
import ./u32_extrema.bend as NativeExtrema
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as Division
import ./u32_division.bend as NativeDivision
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_ceiling.bend as Ceiling
import ./kernel_weights_refinement.bend as Mapped
import ./storage_shape.bend as Shape
import ./natural_addition.bend as Addition
import ./input_gradient_phase_taps.bend as Taps
import ./traversal_array_refinement.bend as Arrays

type Query is Data:
  Query{weights: Bool,call: Domain.Call}

def call(query: Query) -> Domain.Call:
  Query{weights,call} = query
  call
def geometry(query: Query) -> Kernel.Geometry:
  Domain.geometry(call(query))
def limits(query: Query) -> Machine.Limits:
  Domain.limits(call(query))
def rows(query: Query) -> Nat:
  Query{weights,+call} = query
  +geometry = Domain.geometry(call)
  Nat.min(Domain.stride(geometry),Domain.height(geometry))
def columns(query: Query) -> Nat:
  Query{weights,+call} = query
  +geometry = Domain.geometry(call)
  Nat.min(Domain.stride(geometry),Domain.width(geometry))
def columns_word(query: Query) -> U32:
  Query{weights,+call} = query
  +geometry = Domain.geometry(call)
  U32.min(Domain.stride_word(geometry),Domain.width_word(geometry))
def count(query: Query) -> Nat:
  +query = query
  Nat.mul(rows(query),columns(query))

def columns_positive(+query: Query) -> {Nat.is_lt(0n,columns(query)) == True{} : Bool}:
  Query{weights,+call} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Extrema.minimum_strict_lower(Domain.stride(geometry),Domain.width(geometry),0n,Domain.stride_positive(call),Domain.width_positive(call))

def count_room(+query: Query) -> {Nat.is_le(count(query),U32.to_nat(Machine.capacity(limits(query)))) == True{} : Bool}:
  Query{weights,+call} = query
  Domain.Call{geometry,batch,limits,facts} = call
  +h = Domain.height(geometry)
  +w = Domain.width(geometry)
  +plane = Nat.mul(h,w)
  Order.transitive(count(query),plane,U32.to_nat(Machine.capacity(limits)),
    Order.transitive(count(query),Nat.mul(h,columns(query)),plane,
      RowMajor.mul_le(rows(query),h,columns(query),Extrema.minimum_right(Domain.stride(geometry),h)),Division.multiply_left_le(h,columns(query),w,Extrema.minimum_right(Domain.stride(geometry),w))),
    Domain.plane_room(call))

def index_exact(+query: Query,+index: Nat,inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) -> {U32.to_nat(U32.from_nat(index)) == index : Nat}:
  Bounds.roundtrip(index,Machine.capacity(limits(query)),Limits.capacity_fits(limits(query)),Bounds.strict_le(index,U32.to_nat(Machine.capacity(limits(query))),Bounds.nat_trans(index,count(query),U32.to_nat(Machine.capacity(limits(query))),inside,count_room(query))))

def column_word_exact(+query: Query) -> {U32.to_nat(columns_word(query)) == columns(query) : Nat}:
  Query{weights,+call} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  NativeExtrema.minimum(Domain.stride_word(geometry),Domain.width_word(geometry))

def column_double_room(+query: Query) -> {Nat.is_le(Nat.double(U32.to_nat(columns_word(query))),U32.to_nat(Machine.coordinate(limits(query)))) == True{} : Bool}:
  Query{weights,+call} = query
  Domain.Call{geometry,batch,limits,facts} = call
  %Equal.sym(Nat,U32.to_nat(columns_word(query)),columns(query),column_word_exact(query)) : {Nat.is_le(Nat.double(_),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}
  PhaseBounds.double_room(limits,columns(query),Order.transitive(columns(query),Domain.width(geometry),U32.to_nat(Machine.dimension(limits)),Extrema.minimum_right(Domain.stride(geometry),Domain.width(geometry)),Domain.width_room(call)))

def window_jobs_exact(+query: Query) -> {Kernel.window_jobs(geometry(query)) == count(query) : Nat}:
  Query{weights,+call} = query
  Domain.Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,+h,+w,k,+s,p,gh,gw} = geometry
  Equal.trans(Nat,Kernel.window_jobs(geometry),Nat.mul(rows(query),U32.to_nat(U32.min(s,w))),count(query),
    Equal.cong(Nat,Nat,rows => Nat.mul(rows,U32.to_nat(U32.min(s,w))),U32.to_nat(U32.min(s,h)),rows(query),NativeExtrema.minimum(s,h)),
    Equal.cong(Nat,Nat,columns => Nat.mul(rows(query),columns),U32.to_nat(U32.min(s,w)),columns(query),NativeExtrema.minimum(s,w)))

def factor(query: Query) -> Nat:
  Query{weights,+call} = query
  +geometry = Domain.geometry(call)
  Bool.pick(Nat,weights,Nat.mul(Ceiling.ceiling(Domain.channels(geometry),8n),Domain.outputs(geometry)),Domain.outputs(geometry))

def model(query: Query) -> Totals.Weighted:
  Query{+weights,+call} = query
  +geometry = Domain.geometry(call)
  Totals.Weighted{Totals.Matrix{Totals.Axis{Domain.kernel(geometry),Domain.stride(geometry),Domain.padding(geometry),Domain.stride_positive(call)},columns(query),columns_positive(query)},factor(query)}

def pure(query: Query,index: Nat) -> Nat:
  Sums.scaled(~Totals.Weighted,~Totals.weighted_plain,~Totals.weighted_factor,model(query),index)

def pure_same(+query: Query,+index: Nat) ->
  {pure(query,index) == Sums.scaled(~Totals.Weighted,~Totals.weighted_plain,~Totals.weighted_factor,model(query),index) : Nat}:
  {==}

def factor_room(+query: Query) ->
  {Nat.is_le(factor(query),Nat.mul(Domain.outputs(geometry(query)),Domain.channels(geometry(query)))) == True{} : Bool}:
  Query{weights,+call} = query
  match weights:
    case False{}:
      Domain.Call{g,batch,limits,facts} = call
      RowMajor.product_covers(Domain.outputs(g),Domain.channels(g),Domain.channels_positive(call))
    case True{}:
      Domain.Call{g,batch,limits,facts} = call
      %Equal.sym(Nat,Nat.mul(Domain.outputs(g),Domain.channels(g)),Nat.mul(Domain.channels(g),Domain.outputs(g)),RowMajor.multiply_commutative(Domain.outputs(g),Domain.channels(g))) :
        {Nat.is_le(Nat.mul(Ceiling.ceiling(Domain.channels(g),8n),Domain.outputs(g)),_) == True{} : Bool}
      RowMajor.mul_le(Ceiling.ceiling(Domain.channels(g),8n),Domain.channels(g),Domain.outputs(g),Ceiling.bounded(Domain.channels(g),8n,{==}))

def total_room(+query: Query,+prefix: Nat,inside: {Nat.is_le(prefix,count(query)) == True{} : Bool}) ->
  {Nat.is_le(Sums.sum(~Query,~pure,prefix,0n,query),U32.to_nat(Machine.capacity(limits(query)))) == True{} : Bool}:
  Query{weights,+call} = query
  Domain.Call{g,batch,limits,facts} = call
  +weighted = model(query)
  +kernel = Domain.kernel(g)
  +square = Nat.mul(kernel,kernel)
  +coefficient = Nat.mul(Domain.outputs(g),Domain.channels(g))
  %Equal.sym(Nat,Sums.sum(~Query,~pure,prefix,0n,query),Sums.sum(~Totals.Weighted,~Sums.scaled(~Totals.Weighted,~Totals.weighted_plain,~Totals.weighted_factor),prefix,0n,weighted),
    Sums.project_context(~Query,~Totals.Weighted,~model,~pure,~Sums.scaled(~Totals.Weighted,~Totals.weighted_plain,~Totals.weighted_factor),~pure_same,prefix,0n,query)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  Order.transitive(Sums.sum(~Totals.Weighted,~Sums.scaled(~Totals.Weighted,~Totals.weighted_plain,~Totals.weighted_factor),prefix,0n,weighted),Nat.mul(factor(query),square),U32.to_nat(Machine.capacity(limits)),
    Totals.weighted_prefix(prefix,rows(query),weighted,inside,Extrema.minimum_left(Domain.stride(g),Domain.height(g)),Extrema.minimum_left(Domain.stride(g),Domain.width(g))),
    Order.transitive(Nat.mul(factor(query),square),Nat.mul(coefficient,square),U32.to_nat(Machine.capacity(limits)),RowMajor.mul_le(factor(query),coefficient,square,factor_room(query)),
      %Equal.sym(Nat,Nat.mul(coefficient,square),Domain.weight_size(g),RowMajor.assoc(Domain.outputs(g),Domain.channels(g),square)) :
        {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} Domain.weight_room(call)))

def quotient_exact(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {U32.to_nat((U32.from_nat(index) / columns_word(query) : U32)) == Nat.div(index,columns(query)) : Nat}:
  %column_word_exact(query) : {U32.to_nat((U32.from_nat(index) / columns_word(query) : U32)) == Nat.div(index,_) : Nat}
  %index_exact(query,index,inside) : {U32.to_nat((U32.from_nat(index) / columns_word(query) : U32)) == Nat.div(_,U32.to_nat(columns_word(query))) : Nat}
  NativeDivision.divide(U32.from_nat(index),columns_word(query),Machine.coordinate(limits(query)),Limits.coordinate_fits(limits(query)),column_double_room(query),
    %Equal.sym(Nat,U32.to_nat(columns_word(query)),columns(query),column_word_exact(query)) : {Nat.is_lt(0n,_) == True{} : Bool} columns_positive(query))

def remainder_exact(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {U32.to_nat((U32.from_nat(index) % columns_word(query) : U32)) == Nat.mod(index,columns(query)) : Nat}:
  %column_word_exact(query) : {U32.to_nat((U32.from_nat(index) % columns_word(query) : U32)) == Nat.mod(index,_) : Nat}
  %index_exact(query,index,inside) : {U32.to_nat((U32.from_nat(index) % columns_word(query) : U32)) == Nat.mod(_,U32.to_nat(columns_word(query))) : Nat}
  NativeDivision.modulo(U32.from_nat(index),columns_word(query),Machine.coordinate(limits(query)),Limits.coordinate_fits(limits(query)),column_double_room(query),
    %Equal.sym(Nat,U32.to_nat(columns_word(query)),columns(query),column_word_exact(query)) : {Nat.is_lt(0n,_) == True{} : Bool} columns_positive(query))

def native_taps(+query: Query,+first: U32,+size: U32,
  first_room: {Nat.is_le(U32.to_nat(first),U32.to_nat(Machine.dimension(limits(query)))) == True{} : Bool}) ->
  {U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,Domain.kernel_word(geometry(query)),Domain.stride_word(geometry(query)),Domain.padding_word(geometry(query))))) ==
    Taps.count(Domain.kernel(geometry(query)),Nat.mod(Nat.add(U32.to_nat(first),Domain.padding(geometry(query))),Domain.stride(geometry(query))),Domain.stride(geometry(query))) : Nat}:
  Query{weights,+call} = query
  Domain.Call{g,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,+k,+s,+p,gh,gw} = g
  +stride_positive = Domain.stride_positive(call)
  +stride_double_room = PhaseBounds.stride_double_room(call)
  +coordinate = Machine.coordinate(limits)
  +residue = Geometry.residue(first,s,p)
  Equal.trans(Nat,U32.to_nat(Geometry.axis_taps(Kernel.axis(first,size,k,s,p))),Taps.count(U32.to_nat(k),U32.to_nat(residue),U32.to_nat(s)),
    Taps.count(U32.to_nat(k),Nat.mod(Nat.add(U32.to_nat(first),U32.to_nat(p)),U32.to_nat(s)),U32.to_nat(s)),
    Geometry.taps_exact(first,size,k,s,p,coordinate,Limits.coordinate_fits(limits),PhaseBounds.kernel_coordinate_room(call),stride_double_room,stride_positive),
    Equal.cong(Nat,Nat,residue => Taps.count(U32.to_nat(k),residue,U32.to_nat(s)),U32.to_nat(residue),Nat.mod(Nat.add(U32.to_nat(first),U32.to_nat(p)),U32.to_nat(s)),
      Geometry.residue_exact(first,s,p,coordinate,Limits.coordinate_fits(limits),PhaseBounds.padded_room(limits,U32.to_nat(first),U32.to_nat(p),first_room,Domain.padding_room(call)),stride_double_room,stride_positive)))

def first_rooms(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  Shape.Both<{Nat.is_le(U32.to_nat((U32.from_nat(index) / columns_word(query) : U32)),U32.to_nat(Machine.dimension(limits(query)))) == True{} : Bool},
    {Nat.is_le(U32.to_nat((U32.from_nat(index) % columns_word(query) : U32)),U32.to_nat(Machine.dimension(limits(query)))) == True{} : Bool}>:
  Query{weights,+call} = query
  Domain.Call{g,batch,limits,facts} = call
  +height_room = Domain.height_room(call)
  +width_room = Domain.width_room(call)
  Shape.Both{
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(index) / columns_word(query) : U32)),Nat.div(index,columns(query)),quotient_exact(query,index,inside)) :
      {Nat.is_le(_,U32.to_nat(Machine.dimension(limits))) == True{} : Bool}
      Order.transitive(Nat.div(index,columns(query)),Domain.height(g),U32.to_nat(Machine.dimension(limits)),
        Bounds.strict_le(Nat.div(index,columns(query)),Domain.height(g),Bounds.nat_trans(Nat.div(index,columns(query)),rows(query),Domain.height(g),
          Division.quotient_below(index,rows(query),columns(query),columns_positive(query),inside),Extrema.minimum_right(Domain.stride(g),Domain.height(g)))),height_room),
    %Equal.sym(Nat,U32.to_nat((U32.from_nat(index) % columns_word(query) : U32)),Nat.mod(index,columns(query)),remainder_exact(query,index,inside)) :
      {Nat.is_le(_,U32.to_nat(Machine.dimension(limits))) == True{} : Bool}
      Order.transitive(Nat.mod(index,columns(query)),Domain.width(g),U32.to_nat(Machine.dimension(limits)),
        Bounds.strict_le(Nat.mod(index,columns(query)),Domain.width(g),Bounds.nat_trans(Nat.mod(index,columns(query)),columns(query),Domain.width(g),
          Division.remainder_small(index,columns(query),columns_positive(query)),Extrema.minimum_right(Domain.stride(g),Domain.width(g)))),width_room)}

def phase_exact_with_rooms(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool},
  evidence: Shape.Both<{Nat.is_le(U32.to_nat((U32.from_nat(index) / columns_word(query) : U32)),U32.to_nat(Machine.dimension(limits(query)))) == True{} : Bool},
    {Nat.is_le(U32.to_nat((U32.from_nat(index) % columns_word(query) : U32)),U32.to_nat(Machine.dimension(limits(query)))) == True{} : Bool}>) ->
  {U32.to_nat(Kernel.phase_inner(Kernel.phase_of(geometry(query),U32.from_nat(index)))) ==
    Nat.mul(Domain.outputs(geometry(query)),Totals.matrix_sample(Totals.weighted_matrix(model(query)),index)) : Nat}:
  Query{weights,+call} = query
  Domain.Call{g,batch,limits,facts} = call
  Kernel.Geometry{c,+o,+h,+w,+k,+s,+p,gh,gw} = g
  Shape.Both{yr,xr} = evidence
  +y = (U32.from_nat(index) / columns_word(query) : U32)
  +x = (U32.from_nat(index) % columns_word(query) : U32)
  +native_y = U32.to_nat(Geometry.axis_taps(Kernel.axis(y,h,k,s,p)))
  +native_x = U32.to_nat(Geometry.axis_taps(Kernel.axis(x,w,k,s,p)))
  +matrix = Totals.weighted_matrix(model(query))
  +axis = Totals.matrix_axis(matrix)
  +natural_y = Totals.cyclic_taps(axis,Nat.div(index,columns(query)))
  +natural_x = Totals.cyclic_taps(axis,Nat.mod(index,columns(query)))
  +y_exact = {
    %quotient_exact(query,index,inside) : {native_y == Taps.count(U32.to_nat(k),Nat.mod(Nat.add(_,U32.to_nat(p)),U32.to_nat(s)),U32.to_nat(s)) : Nat}
    native_taps(query,y,h,yr) : {native_y == natural_y : Nat}}
  +x_exact = {
    %remainder_exact(query,index,inside) : {native_x == Taps.count(U32.to_nat(k),Nat.mod(Nat.add(_,U32.to_nat(p)),U32.to_nat(s)),U32.to_nat(s)) : Nat}
    native_taps(query,x,w,xr) : {native_x == natural_x : Nat}}
  Equal.trans(Nat,U32.to_nat(Kernel.phase_inner(Kernel.phase_of(g,U32.from_nat(index)))),Nat.mul(U32.to_nat(o),Nat.mul(native_y,native_x)),Nat.mul(U32.to_nat(o),Nat.mul(natural_y,natural_x)),
    PhaseBounds.phase_inner_exact(call,y,x),
    Equal.trans(Nat,Nat.mul(U32.to_nat(o),Nat.mul(native_y,native_x)),Nat.mul(U32.to_nat(o),Nat.mul(natural_y,native_x)),Nat.mul(U32.to_nat(o),Nat.mul(natural_y,natural_x)),
      Equal.cong(Nat,Nat,y => Nat.mul(U32.to_nat(o),Nat.mul(y,native_x)),native_y,natural_y,y_exact),
      Equal.cong(Nat,Nat,x => Nat.mul(U32.to_nat(o),Nat.mul(natural_y,x)),native_x,natural_x,x_exact)))

def phase_exact(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {U32.to_nat(Kernel.phase_inner(Kernel.phase_of(geometry(query),U32.from_nat(index)))) ==
    Nat.mul(Domain.outputs(geometry(query)),Totals.matrix_sample(Totals.weighted_matrix(model(query)),index)) : Nat}:
  phase_exact_with_rooms(query,index,inside,first_rooms(query,index,inside))

def native(query: Query,index: U32) -> U32:
  Query{weights,+call} = query
  +g = Domain.geometry(call)
  +inner = Kernel.phase_inner(Kernel.phase_of(g,index))
  Bool.pick(U32,weights,(Separation.block_count(g) * inner : U32),inner)

def native_match(+query: Query,+index: Nat,inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {native(query,U32.from_nat(index)) == U32.from_nat(pure(query,index)) : U32}:
  Query{weights,+call} = query
  match weights:
    case False{}:
      Domain.Call{g,batch,limits,facts} = call
      Equal.sym(U32,U32.from_nat(pure(query,index)),Kernel.phase_inner(Kernel.phase_of(g,U32.from_nat(index))),
        Arithmetic.from_nat_exact(Kernel.phase_inner(Kernel.phase_of(g,U32.from_nat(index))),pure(query,index),phase_exact(query,index,inside)))
    case True{}:
      Domain.Call{g,batch,limits,facts} = call
      Kernel.Geometry{+c,+o,h,w,k,s,p,gh,gw} = g
      +inside = inside
      +y = (U32.from_nat(index) / columns_word(query) : U32)
      +x = (U32.from_nat(index) % columns_word(query) : U32)
      +inner = Kernel.phase_inner(Kernel.phase_of(g,U32.from_nat(index)))
      +blocks = Ceiling.ceiling(U32.to_nat(c),8n)
      +area = Totals.matrix_sample(Totals.weighted_matrix(model(query)),index)
      +exact = {
        %Equal.sym(Nat,Nat.mul(Nat.mul(blocks,U32.to_nat(o)),area),Nat.mul(blocks,Nat.mul(U32.to_nat(o),area)),RowMajor.assoc(blocks,U32.to_nat(o),area)) :
          {U32.to_nat(native(query,U32.from_nat(index))) == _ : Nat}
        %phase_exact(query,index,inside) : {U32.to_nat(native(query,U32.from_nat(index))) == Nat.mul(blocks,_) : Nat}
        %PhaseWeights.block_count_exact(call,y,x) :
          {U32.to_nat(native(query,U32.from_nat(index))) == Nat.mul(_,U32.to_nat(inner)) : Nat}
        PhaseWeights.packed_exact(call,y,x) : {U32.to_nat(native(query,U32.from_nat(index))) == pure(query,index) : Nat}}
      Equal.sym(U32,U32.from_nat(pure(query,index)),native(query,U32.from_nat(index)),Arithmetic.from_nat_exact(native(query,U32.from_nat(index)),pure(query,index),exact))

def update(index: U32,query: Query,value: U32) -> U32:
  Query{weights,+call} = query
  +g = Domain.geometry(call)
  +value = value
  +index = index
  Bool.pick(U32,weights,Kernel.weight_total(index,g,value),Kernel.inner_total(index,g,value))

def identity(value: U32) -> U32:
  value

def step_matches(+index: U32,+query: Query,+value: U32) ->
  {update(index,query,value) == Sums.encoded_update(~Query,~native,index,query,value) : U32}:
  Query{weights,call} = query
  match weights:
    case False{}:
      Domain.Call{g,batch,limits,facts} = call
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
      {==}
    case True{}:
      Domain.Call{g,batch,limits,facts} = call
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
      {==}

def prefix_word(+query: Query,prefix: Nat) -> U32:
  Traversal.indexed_fold(~U32,~Query,~update,prefix,0,query,0)

def prefix_encoded(+query: Query,+prefix: Nat,room: {Nat.is_le(prefix,count(query)) == True{} : Bool}) ->
  {prefix_word(query,prefix) == U32.from_nat(Sums.sum(~Query,~pure,prefix,0n,query)) : U32}:
  +encoded = Traversal.indexed_fold(~U32,~Query,~Sums.encoded_update(~Query,~native),prefix,0,query,0)
  Equal.trans(U32,prefix_word(query,prefix),encoded,U32.from_nat(Sums.sum(~Query,~pure,prefix,0n,query)),
    Arrays.indexed_fold_refines_model(~U32,~U32,~Query,~update,~Sums.encoded_update(~Query,~native),~identity,~step_matches,prefix,0,query,0),
    Sums.encoded(~Query,~count,~pure,~native,~native_match,prefix,0n,query,0n,room))

def prefix_exact(+query: Query,+prefix: Nat,+room: {Nat.is_le(prefix,count(query)) == True{} : Bool}) ->
  {U32.to_nat(prefix_word(query,prefix)) == Sums.sum(~Query,~pure,prefix,0n,query) : Nat}:
  %Equal.sym(U32,prefix_word(query,prefix),U32.from_nat(Sums.sum(~Query,~pure,prefix,0n,query)),prefix_encoded(query,prefix,room)) :
    {U32.to_nat(_) == Sums.sum(~Query,~pure,prefix,0n,query) : Nat}
  Bounds.roundtrip(Sums.sum(~Query,~pure,prefix,0n,query),Machine.capacity(limits(query)),Limits.capacity_fits(limits(query)),total_room(query,prefix,room))

def weights(query: Query) -> Bool:
  Query{weights,call} = query
  weights
def weights_selected(query: Query) -> Data:
  {weights(query) == True{} : Bool}
def entries_selected(query: Query) -> Data:
  {weights(query) == False{} : Bool}

def weights_step(+index: U32,+query: Query,+value: U32,selected: weights_selected(query)) ->
  {update(index,query,value) == Kernel.weight_total(index,geometry(query),value) : U32}:
  Query{+flag,call} = query
  Domain.Call{g,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  %Equal.sym(Bool,flag,True{},selected) : {Bool.pick(U32,_,Kernel.weight_total(index,g,value),Kernel.inner_total(index,g,value)) == Kernel.weight_total(index,g,value) : U32}
  {==}

def entries_step(+index: U32,+query: Query,+value: U32,selected: entries_selected(query)) ->
  {update(index,query,value) == Kernel.inner_total(index,geometry(query),value) : U32}:
  Query{+flag,call} = query
  Domain.Call{g,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  %Equal.sym(Bool,flag,False{},selected) : {Bool.pick(U32,_,Kernel.weight_total(index,g,value),Kernel.inner_total(index,g,value)) == Kernel.inner_total(index,g,value) : U32}
  {==}

def prefix_program(+query: Query,+prefix: Nat) ->
  {prefix_word(query,prefix) == Bool.pick(U32,weights(query),Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.weight_total,prefix,0,geometry(query),0),
    Traversal.indexed_fold(~U32,~Kernel.Geometry,~Kernel.inner_total,prefix,0,geometry(query),0)) : U32}:
  Query{flag,call} = query
  match flag:
    case True{}:
      Arrays.indexed_fold_change_context_under(~U32,~Query,~Kernel.Geometry,~geometry,~weights_selected,~update,~Kernel.weight_total,~weights_step,prefix,0,query,0,{==})
    case False{}:
      Arrays.indexed_fold_change_context_under(~U32,~Query,~Kernel.Geometry,~geometry,~entries_selected,~update,~Kernel.inner_total,~entries_step,prefix,0,query,0,{==})

def prefix_step(+query: Query,+index: Nat) ->
  {prefix_word(query,1n+index) == update(U32.from_nat(index),query,prefix_word(query,index)) : U32}:
  +size = Equal.trans(Nat,Nat.add(index,1n),1n+Nat.add(index,0n),1n+index,Addition.successor_right(index,0n),
    Equal.cong(Nat,Nat,value => 1n+value,Nat.add(index,0n),index,Addition.zero_right(index)))
  %size : {Traversal.indexed_fold(~U32,~Query,~update,_,0,query,0) == update(U32.from_nat(index),query,prefix_word(query,index)) : U32}
  Arrays.indexed_fold_split(~U32,~Query,~update,index,1n,0n,query,0)

def pure_prefix_step(+query: Query,+index: Nat) ->
  {Sums.sum(~Query,~pure,1n+index,0n,query) == Nat.add(Sums.sum(~Query,~pure,index,0n,query),pure(query,index)) : Nat}:
  +size = Equal.trans(Nat,Nat.add(index,1n),1n+Nat.add(index,0n),1n+index,Addition.successor_right(index,0n),
    Equal.cong(Nat,Nat,value => 1n+value,Nat.add(index,0n),index,Addition.zero_right(index)))
  %size : {Sums.sum(~Query,~pure,_,0n,query) == Nat.add(Sums.sum(~Query,~pure,index,0n,query),pure(query,index)) : Nat}
  Sums.split(~Query,~pure,index,1n,0n,query)

def sample_room(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {Nat.is_le(pure(query,index),U32.to_nat(Machine.capacity(limits(query)))) == True{} : Bool}:
  +previous = Sums.sum(~Query,~pure,index,0n,query)
  +next = Sums.sum(~Query,~pure,1n+index,0n,query)
  Order.transitive(pure(query,index),next,U32.to_nat(Machine.capacity(limits(query))),
    %Equal.sym(Nat,next,Nat.add(previous,pure(query,index)),pure_prefix_step(query,index)) : {Nat.is_le(pure(query,index),_) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(previous,pure(query,index)),Nat.add(pure(query,index),previous),Order.commutative(previous,pure(query,index))) :
      {Nat.is_le(pure(query,index),_) == True{} : Bool}
      Order.le_add(pure(query,index),previous),
    total_room(query,1n+index,Order.strict_follow(index,count(query),inside)))

def native_exact(+query: Query,+index: Nat,+inside: {Nat.is_lt(index,count(query)) == True{} : Bool}) ->
  {U32.to_nat(native(query,U32.from_nat(index))) == pure(query,index) : Nat}:
  %Equal.sym(U32,native(query,U32.from_nat(index)),U32.from_nat(pure(query,index)),native_match(query,index,inside)) :
    {U32.to_nat(_) == pure(query,index) : Nat}
  Bounds.roundtrip(pure(query,index),Machine.capacity(limits(query)),Limits.capacity_fits(limits(query)),sample_room(query,index,inside))

def weight_span_exact(+query: Query,+index: Nat,selected: weights_selected(query)) ->
  {U32.to_nat(native(query,U32.from_nat(index))) == Nat.mul(Ceiling.ceiling(Domain.channels(geometry(query)),8n),U32.to_nat(Kernel.phase_inner(Kernel.phase_of(geometry(query),U32.from_nat(index))))) : Nat}:
  Query{+flag,+call} = query
  Domain.Call{g,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = g
  +y = (U32.from_nat(index) / columns_word(query) : U32)
  +x = (U32.from_nat(index) % columns_word(query) : U32)
  +inner = Kernel.phase_inner(Kernel.phase(g,y,x))
  +blocks = Mapped.mapped_blocks(~Kernel.Phase,PhaseWeights.layout(g,y,x))
  %Equal.sym(Bool,flag,True{},selected) :
    {U32.to_nat(Bool.pick(U32,_,(Separation.block_count(g) * inner : U32),inner)) == Nat.mul(Ceiling.ceiling(Domain.channels(g),8n),U32.to_nat(inner)) : Nat}
  %PhaseWeights.block_count_exact(call,y,x) :
    {U32.to_nat((Separation.block_count(g) * inner : U32)) == Nat.mul(_,U32.to_nat(inner)) : Nat}
  PhaseWeights.packed_exact(call,y,x)

def weight_span_positive(+query: Query,+index: Nat,+selected: weights_selected(query),
  active: {U32.is_eq(Kernel.phase_inner(Kernel.phase_of(geometry(query),U32.from_nat(index))),0) == False{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(native(query,U32.from_nat(index)))) == True{} : Bool}:
  Query{weights,+call} = query
  Domain.Call{g,batch,limits,facts} = call
  +blocks = Ceiling.ceiling(Domain.channels(g),8n)
  +inner = U32.to_nat(Kernel.phase_inner(Kernel.phase_of(g,U32.from_nat(index))))
  %Equal.sym(Nat,U32.to_nat(native(query,U32.from_nat(index))),Nat.mul(blocks,inner),weight_span_exact(query,index,selected)) : {Nat.is_lt(0n,_) == True{} : Bool}
  Order.product_positive(blocks,inner,
    %Equal.sym(Bool,Nat.is_lt(0n,blocks),Nat.is_lt(0n,Domain.channels(g)),Ceiling.upper_comparison(Domain.channels(g),8n,0n,{==})) : {_ == True{} : Bool} Domain.channels_positive(call),
    Order.nonzero_positive(inner,%Comparison.eq(Kernel.phase_inner(Kernel.phase_of(g,U32.from_nat(index))),0) : {_ == False{} : Bool} active))
