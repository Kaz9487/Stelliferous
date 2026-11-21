# The shapes of an input gradient call, and the facts of a call that passes
# its checks.
import Base
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_shape.bend as Shape
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/tensor_convolution.bend as Conv
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_grid.bend as Grid
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/nat_to_u32_bounds.bend as Bounds
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/natural_addition.bend as Addition
import ../../lib/proofs/row_major_index.bend as RowMajor
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_tiled_public.bend as TiledPublic
import ../../lib/proofs/word_multiplication.bend as WordMultiplication
import ../../lib/tensor_tiled_cells.bend as Cells

# ---- The shapes of a call ----

# CHW has exactly one image. NCHW keeps its explicit batch extent.
type Mode is Data:
  Single{}
  Batched{batch: U32}
def images(mode: Mode) -> U32:
  match mode:
    case Single{}: 1
    case Batched{batch}: batch
def batched(mode: Mode) -> Bool:
  match mode:
    case Single{}: False{}
    case Batched{batch}: True{}
def arguments(+mode: Mode,geometry: Kernel.Geometry) -> Gradient.Arguments:
  Gradient.Arguments{geometry,images(mode),batched(mode)}

# The output gradient [N,O,gh,gw] or [O,gh,gw] and the weights [O,C,k,k].
def input_shape(+mode: Mode,geometry: Kernel.Geometry) -> List<&2,U32>:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Conv.output_shape(batched(mode),images(mode),o,gh,gw)
def weight_shape(geometry: Kernel.Geometry) -> List<&2,U32>:
  Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw} = geometry
  [o,c,k,k]

def stride(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  s
def padding(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  p
def height(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  h
def width(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  w

def outputs_word(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  o

def geometry(arguments: Gradient.Arguments) -> Kernel.Geometry:
  Gradient.Arguments{geometry,batch,batched} = arguments
  geometry
def batch(arguments: Gradient.Arguments) -> Nat:
  Gradient.Arguments{geometry,batch,batched} = arguments
  U32.to_nat(batch)

# ---- The checks of a call ----

# The checks of a valid shape and of its direct bounds, last first.
def valid_conditions(shape: Shape.Dimensions,input_length: Nat,weight_length: Nat,bias_length: Nat) -> List<&2,Bool>:
  Shape.Dimensions{+ci,+co,+h,+w,+k,+s,+p} = shape
  [Nat.is_eq(bias_length,co),Nat.is_eq(weight_length,Nat.mul(co,Nat.mul(ci,Nat.mul(k,k)))),Nat.is_eq(input_length,Nat.mul(ci,Nat.mul(h,w))),
    Nat.is_le(k,Nat.add(w,Nat.mul(2n,p))),Nat.is_le(k,Nat.add(h,Nat.mul(2n,p))),Nat.is_lt(0n,s),Nat.is_lt(0n,k),Nat.is_lt(0n,w),Nat.is_lt(0n,h),Nat.is_lt(0n,co),Nat.is_lt(0n,ci)]
def shape_valid_condition(+shape: Shape.Dimensions,+input_length: Nat,+weight_length: Nat,+bias_length: Nat,+index: Nat,
  accepted: {Shape.valid(shape,input_length,weight_length,bias_length) == True{} : Bool}) ->
  {Boolean.condition_at(valid_conditions(shape,input_length,weight_length,bias_length),index) == True{} : Bool}:
  Shape.Dimensions{+ci,+co,+h,+w,+k,+s,+p} = shape
  Boolean.every_condition_holds(valid_conditions(shape,input_length,weight_length,bias_length),index,accepted)

def direct_conditions(+dimension: U32,+capacity: U32,+coordinate: U32,shape: Shape.Dimensions,input_length: Nat,weight_length: Nat) -> List<&2,Bool>:
  Shape.Dimensions{+ci,+co,+h,+w,+k,+s,+p} = shape
  +oh = Shape.axis_size(h,k,s,p)
  +ow = Shape.axis_size(w,k,s,p)
  [Nat.is_le(Nat.add(Nat.mul(ow,s),k),U32.to_nat(coordinate)),Nat.is_le(Nat.add(Nat.mul(oh,s),k),U32.to_nat(coordinate)),
    Nat.is_le(Nat.mul(co,Nat.mul(oh,ow)),U32.to_nat(capacity)),Nat.is_le(weight_length,U32.to_nat(capacity)),Nat.is_le(input_length,U32.to_nat(capacity)),
    Nat.is_le(p,U32.to_nat(dimension)),Nat.is_le(s,U32.to_nat(dimension)),Nat.is_le(k,U32.to_nat(dimension)),Nat.is_le(w,U32.to_nat(dimension)),
    Nat.is_le(h,U32.to_nat(dimension)),Nat.is_le(co,U32.to_nat(dimension)),Nat.is_le(ci,U32.to_nat(dimension))]
def shape_direct_condition(+dimension: U32,+capacity: U32,+coordinate: U32,+shape: Shape.Dimensions,+input_length: Nat,+weight_length: Nat,+index: Nat,
  accepted: {Shape.direct_bounds(dimension,capacity,coordinate,shape,input_length,weight_length) == True{} : Bool}) ->
  {Boolean.condition_at(direct_conditions(dimension,capacity,coordinate,shape,input_length,weight_length),index) == True{} : Bool}:
  Shape.Dimensions{+ci,+co,+h,+w,+k,+s,+p} = shape
  Boolean.every_condition_holds(direct_conditions(dimension,capacity,coordinate,shape,input_length,weight_length),index,accepted)

# The checks of the call, last first.
def guard_conditions(+arguments: Gradient.Arguments,+d_cells: U32,+weight_cells: U32,+limits: Machine.Limits) -> List<&2,Bool>:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+batch,batched} = arguments
  +dimensions = Gradient.input_dimensions(arguments)
  +input_length = Domain.image_size(geometry(arguments))
  +logical_weights = Domain.weight_size(geometry(arguments))
  [Kernel.fits(geometry(arguments),Machine.capacity(limits)),
    Nat.is_le(Nat.mul(U32.to_nat(batch),input_length),U32.to_nat(Machine.capacity(limits))),
    Nat.is_le(Nat.mul(U32.to_nat(batch),Domain.gradient_size(geometry(arguments))),U32.to_nat(Machine.capacity(limits))),
    Nat.is_eq(U32.to_nat(weight_cells),Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(Tiled.groups(c)),Nat.mul(U32.to_nat(k),U32.to_nat(k))))),
    Nat.is_eq(U32.to_nat(d_cells),Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(Tiled.groups(o)),Nat.mul(U32.to_nat(gh),U32.to_nat(gw))))),
    Nat.is_eq(U32.to_nat(gw),Shape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))),
    Nat.is_eq(U32.to_nat(gh),Shape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))),
    Shape.direct_bounds(Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits),dimensions,input_length,logical_weights),
    Shape.valid(dimensions,input_length,logical_weights,U32.to_nat(o))]

def guard_condition(+arguments: Gradient.Arguments,+d_cells: U32,+weight_cells: U32,+limits: Machine.Limits,+index: Nat,
  accepted: {Gradient.input_valid_with(arguments,d_cells,weight_cells,limits) == True{} : Bool}) ->
  {Boolean.condition_at(guard_conditions(arguments,d_cells,weight_cells,limits),index) == True{} : Bool}:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+batch,batched} = arguments
  Machine.Limits{+dimension,+capacity,+coordinate,half_word,facts} = limits
  Boolean.every_condition_holds(guard_conditions(arguments,d_cells,weight_cells,limits),index,accepted)

def valid_condition(+arguments: Gradient.Arguments,+d_cells: U32,+weight_cells: U32,+limits: Machine.Limits,+index: Nat,
  accepted: {Gradient.input_valid_with(arguments,d_cells,weight_cells,limits) == True{} : Bool}) ->
  {Boolean.condition_at(valid_conditions(Gradient.input_dimensions(arguments),Domain.image_size(geometry(arguments)),Domain.weight_size(geometry(arguments)),Domain.outputs(geometry(arguments))),index) == True{} : Bool}:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+batch,batched} = arguments
  shape_valid_condition(Gradient.input_dimensions(arguments),Domain.image_size(geometry(arguments)),Domain.weight_size(geometry(arguments)),U32.to_nat(o),index,
    guard_condition(arguments,d_cells,weight_cells,limits,8n,accepted))

def direct_condition(+arguments: Gradient.Arguments,+d_cells: U32,+weight_cells: U32,+limits: Machine.Limits,+index: Nat,
  accepted: {Gradient.input_valid_with(arguments,d_cells,weight_cells,limits) == True{} : Bool}) ->
  {Boolean.condition_at(direct_conditions(Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits),Gradient.input_dimensions(arguments),
    Domain.image_size(geometry(arguments)),Domain.weight_size(geometry(arguments))),index) == True{} : Bool}:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+batch,batched} = arguments
  shape_direct_condition(Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits),Gradient.input_dimensions(arguments),
    Domain.image_size(geometry(arguments)),Domain.weight_size(geometry(arguments)),index,guard_condition(arguments,d_cells,weight_cells,limits,7n,accepted))

# The facts the kernel proofs read from a call that passes its checks.
def facts(+arguments: Gradient.Arguments,+d_cells: U32,+weight_cells: U32,+limits: Machine.Limits,
  +accepted: {Gradient.input_valid_with(arguments,d_cells,weight_cells,limits) == True{} : Bool}) -> Domain.Facts<geometry(arguments),batch(arguments),limits>:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+n,batched} = arguments
  Machine.Limits{+dimension,+capacity,+coordinate,+half,+limit_facts} = limits
  +rows = Order.equal_of_is_eq(U32.to_nat(gh),Shape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),guard_condition(arguments,d_cells,weight_cells,limits,6n,accepted))
  +columns = Order.equal_of_is_eq(U32.to_nat(gw),Shape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),guard_condition(arguments,d_cells,weight_cells,limits,5n,accepted))
  Domain.Facts{
    Domain.Positive{valid_condition(arguments,d_cells,weight_cells,limits,10n,accepted),valid_condition(arguments,d_cells,weight_cells,limits,9n,accepted),
      valid_condition(arguments,d_cells,weight_cells,limits,8n,accepted),valid_condition(arguments,d_cells,weight_cells,limits,7n,accepted),
      valid_condition(arguments,d_cells,weight_cells,limits,6n,accepted),valid_condition(arguments,d_cells,weight_cells,limits,5n,accepted)},
    Domain.Dimensions{direct_condition(arguments,d_cells,weight_cells,limits,11n,accepted),direct_condition(arguments,d_cells,weight_cells,limits,10n,accepted),
      direct_condition(arguments,d_cells,weight_cells,limits,9n,accepted),direct_condition(arguments,d_cells,weight_cells,limits,8n,accepted),
      direct_condition(arguments,d_cells,weight_cells,limits,7n,accepted),direct_condition(arguments,d_cells,weight_cells,limits,6n,accepted),
      direct_condition(arguments,d_cells,weight_cells,limits,5n,accepted)},
    Domain.Capacities{direct_condition(arguments,d_cells,weight_cells,limits,4n,accepted),direct_condition(arguments,d_cells,weight_cells,limits,3n,accepted),
      %Equal.sym(Nat,U32.to_nat(gh),Shape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),rows) :
        {Nat.is_le(Nat.mul(U32.to_nat(o),Nat.mul(_,U32.to_nat(gw))),U32.to_nat(capacity)) == True{} : Bool}
      %Equal.sym(Nat,U32.to_nat(gw),Shape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),columns) :
        {Nat.is_le(Nat.mul(U32.to_nat(o),Nat.mul(Shape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),_)),U32.to_nat(capacity)) == True{} : Bool}
      direct_condition(arguments,d_cells,weight_cells,limits,2n,accepted),
      guard_condition(arguments,d_cells,weight_cells,limits,2n,accepted)},
    Domain.GradientAxes{rows,columns,
      %Equal.sym(Nat,U32.to_nat(gh),Shape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),rows) :
        {Nat.is_le(Nat.add(Nat.mul(_,U32.to_nat(s)),U32.to_nat(k)),U32.to_nat(coordinate)) == True{} : Bool}
      direct_condition(arguments,d_cells,weight_cells,limits,1n,accepted),
      %Equal.sym(Nat,U32.to_nat(gw),Shape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),columns) :
        {Nat.is_le(Nat.add(Nat.mul(_,U32.to_nat(s)),U32.to_nat(k)),U32.to_nat(coordinate)) == True{} : Bool}
      direct_condition(arguments,d_cells,weight_cells,limits,0n,accepted)}}

# The call a check passes for.
def call(+arguments: Gradient.Arguments,+d_cells: U32,+weight_cells: U32,+limits: Machine.Limits,
  +accepted: {Gradient.input_valid_with(arguments,d_cells,weight_cells,limits) == True{} : Bool}) -> Domain.Call:
  Domain.Call{geometry(arguments),batch(arguments),limits,facts(arguments,d_cells,weight_cells,limits,accepted)}

# The kernel fits the capacity.
def fits(+arguments: Gradient.Arguments,+d_cells: U32,+weight_cells: U32,+limits: Machine.Limits,
  +accepted: {Gradient.input_valid_with(arguments,d_cells,weight_cells,limits) == True{} : Bool}) ->
  {Kernel.fits(geometry(arguments),Machine.capacity(limits)) == True{} : Bool}:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+n,batched} = arguments
  guard_condition(arguments,d_cells,weight_cells,limits,0n,accepted)

# The values of the input gradient fit the capacity.
def values_room(+arguments: Gradient.Arguments,+d_cells: U32,+weight_cells: U32,+limits: Machine.Limits,
  +accepted: {Gradient.input_valid_with(arguments,d_cells,weight_cells,limits) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(batch(arguments),Domain.image_size(geometry(arguments))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+n,batched} = arguments
  guard_condition(arguments,d_cells,weight_cells,limits,1n,accepted)

def checks(+mode: Mode,+geometry: Kernel.Geometry) -> Bool:
  Gradient.input_valid_with(arguments(mode,geometry),Tensors.count_or_zero(Tiled.cell_shape(input_shape(mode,geometry),Conv.lane_of(batched(mode)),Grid.Innermost{})),
    Tensors.count_or_zero(Tiled.cell_shape(weight_shape(geometry),1n,Grid.Outermost{})),Default.limits())

# ---- The cells of the operands of a call ----

# The cells of the output gradient and of the weights of a call.
def gradient_cell_shape(+mode: Mode,+geometry: Kernel.Geometry) -> List<&2,U32>:
  Tiled.cell_shape(input_shape(mode,geometry),Conv.lane_of(batched(mode)),Grid.Innermost{})

def weight_cell_shape(+geometry: Kernel.Geometry) -> List<&2,U32>:
  Tiled.cell_shape(weight_shape(geometry),1n,Grid.Outermost{})

# The lane axes of the output gradient and of the weights lie inside their
# shapes.
def gradient_within(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Nat.is_lt(Conv.lane_of(batched(mode)),List.length(&2,U32,input_shape(mode,geometry))) == True{} : Bool}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

def weight_within(+geometry: Kernel.Geometry) -> {Nat.is_lt(1n,List.length(&2,U32,weight_shape(geometry))) == True{} : Bool}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}

def gradient_axis_inside(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Nat.is_le(Conv.lane_of(batched(mode)),List.length(&2,U32,input_shape(mode,geometry))) == True{} : Bool}:
  Bounds.strict_le(Conv.lane_of(batched(mode)),List.length(&2,U32,input_shape(mode,geometry)),gradient_within(mode,geometry))

def weight_axis_inside(+geometry: Kernel.Geometry) -> {Nat.is_le(1n,List.length(&2,U32,weight_shape(geometry))) == True{} : Bool}:
  Bounds.strict_le(1n,List.length(&2,U32,weight_shape(geometry)),weight_within(geometry))

# The output gradient and the weights have at most four axes, and four axes
# fit the capacity.
def gradient_rank(+mode: Mode,+geometry: Kernel.Geometry) -> {Nat.is_le(List.length(&2,U32,input_shape(mode,geometry)),4n) == True{} : Bool}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

def weight_rank(+geometry: Kernel.Geometry) -> {Nat.is_le(List.length(&2,U32,weight_shape(geometry)),4n) == True{} : Bool}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}

def four_room(+limits: Machine.Limits) -> {Nat.is_le(4n,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Order.transitive(4n,7n,U32.to_nat(Machine.capacity(limits)),{==},Bounds.as_nat_le(7,Machine.capacity(limits),Limits.seven_within_capacity(limits)))

# The values of the output gradient and of the weights of a call.
def gradient_count(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Nat.mul(U32.to_nat(images(mode)),Domain.gradient_size(geometry)) == ShapeLaws.elements(input_shape(mode,geometry)) : Nat}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
      +size = Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(gh),U32.to_nat(gw)))
      Equal.trans(Nat,Nat.add(size,0n),size,ShapeLaws.elements([o,gh,gw]),Addition.zero_right(size),
        Equal.sym(Nat,ShapeLaws.elements([o,gh,gw]),size,ShapeLaws.elements_three(o,gh,gw)))
    case Batched{+batch}:
      Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
      +size = Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(gh),U32.to_nat(gw)))
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(batch),value),size,ShapeLaws.elements([o,gh,gw]),
        Equal.sym(Nat,ShapeLaws.elements([o,gh,gw]),size,ShapeLaws.elements_three(o,gh,gw)))

def weight_count(+geometry: Kernel.Geometry) -> {Domain.weight_size(geometry) == ShapeLaws.elements(weight_shape(geometry)) : Nat}:
  Kernel.Geometry{+c,+o,h,w,+k,s,p,gh,gw} = geometry
  Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(k),value))),U32.to_nat(k),Nat.mul(U32.to_nat(k),1n),
    Equal.sym(Nat,Nat.mul(U32.to_nat(k),1n),U32.to_nat(k),Order.times_one(U32.to_nat(k))))

# The cells of the output gradient of a call are the cells its rows count.
def gradient_cells_count(+mode: Mode,+geometry: Kernel.Geometry) ->
  {ShapeLaws.elements(gradient_cell_shape(mode,geometry)) ==
    Nat.mul(Nat.mul(Nat.mul(U32.to_nat(images(mode)),Domain.gradient_height(geometry)),Domain.gradient_width(geometry)),U32.to_nat(Tiled.groups(outputs_word(geometry)))) : Nat}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
      +groups = Tiled.groups(o)
      +rows = U32.to_nat(gh)
      +columns = U32.to_nat(gw)
      +count = U32.to_nat(groups)
      Equal.trans(Nat,ShapeLaws.elements([gh,gw,groups]),Nat.mul(Nat.mul(rows,columns),count),Nat.mul(Nat.mul(Nat.add(rows,0n),columns),count),
        Equal.trans(Nat,ShapeLaws.elements([gh,gw,groups]),Nat.mul(rows,Nat.mul(columns,count)),Nat.mul(Nat.mul(rows,columns),count),ShapeLaws.elements_three(gh,gw,groups),
          Equal.sym(Nat,Nat.mul(Nat.mul(rows,columns),count),Nat.mul(rows,Nat.mul(columns,count)),RowMajor.assoc(rows,columns,count))),
        Equal.cong(Nat,Nat,value => Nat.mul(Nat.mul(value,columns),count),rows,Nat.add(rows,0n),Equal.sym(Nat,Nat.add(rows,0n),rows,Addition.zero_right(rows))))
    case Batched{+batch}:
      Kernel.Geometry{c,+o,h,w,k,s,p,+gh,+gw} = geometry
      +groups = Tiled.groups(o)
      +images = U32.to_nat(batch)
      +rows = U32.to_nat(gh)
      +columns = U32.to_nat(gw)
      +count = U32.to_nat(groups)
      Equal.trans(Nat,ShapeLaws.elements([batch,gh,gw,groups]),Nat.mul(images,Nat.mul(Nat.mul(rows,columns),count)),Nat.mul(Nat.mul(Nat.mul(images,rows),columns),count),
        Equal.cong(Nat,Nat,value => Nat.mul(images,value),ShapeLaws.elements([gh,gw,groups]),Nat.mul(Nat.mul(rows,columns),count),
          Equal.trans(Nat,ShapeLaws.elements([gh,gw,groups]),Nat.mul(rows,Nat.mul(columns,count)),Nat.mul(Nat.mul(rows,columns),count),ShapeLaws.elements_three(gh,gw,groups),
            Equal.sym(Nat,Nat.mul(Nat.mul(rows,columns),count),Nat.mul(rows,Nat.mul(columns,count)),RowMajor.assoc(rows,columns,count)))),
        Equal.trans(Nat,Nat.mul(images,Nat.mul(Nat.mul(rows,columns),count)),Nat.mul(Nat.mul(images,Nat.mul(rows,columns)),count),Nat.mul(Nat.mul(Nat.mul(images,rows),columns),count),
          Equal.sym(Nat,Nat.mul(Nat.mul(images,Nat.mul(rows,columns)),count),Nat.mul(images,Nat.mul(Nat.mul(rows,columns),count)),RowMajor.assoc(images,Nat.mul(rows,columns),count)),
          Equal.cong(Nat,Nat,value => Nat.mul(value,count),Nat.mul(images,Nat.mul(rows,columns)),Nat.mul(Nat.mul(images,rows),columns),
            Equal.sym(Nat,Nat.mul(Nat.mul(images,rows),columns),Nat.mul(images,Nat.mul(rows,columns)),RowMajor.assoc(images,rows,columns)))))

# The lane axis of the output gradient, read from its shape or its cells.
def shape_axis(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Tiled.lane_axis(List.length(&2,U32,input_shape(mode,geometry))) == Conv.lane_of(batched(mode)) : Nat}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

def cells_axis(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Tiled.lane_axis(List.length(&2,U32,gradient_cell_shape(mode,geometry))) == Conv.lane_of(batched(mode)) : Nat}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

# ---- The cells of the input and of the output gradient ----

# The words of the cells of the input: the values after its channels and
# the values of its images.
def image_after(+geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,+h,+w,k,s,p,gh,gw} = geometry
  Grid.product([h,w])

def image_outer(+mode: Mode,+geometry: Kernel.Geometry) -> U32:
  match mode:
    case Single{}: (1 * image_after(geometry) : U32)
    case Batched{+batch}: ((1 * batch : U32) * image_after(geometry) : U32)

# The values after the channels of an image.
def image_after_word(+limits: Machine.Limits,+geometry: Kernel.Geometry,
  +room: {Nat.is_le(Nat.mul(Domain.height(geometry),Domain.width(geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(image_after(geometry)) == Nat.mul(Domain.height(geometry),Domain.width(geometry)) : Nat}:
  Kernel.Geometry{c,o,+h,+w,k,s,p,gh,gw} = geometry
  +rows = U32.to_nat(h)
  +columns = U32.to_nat(w)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +exact = TiledPublic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),[h,w],
    %Equal.sym(Nat,Nat.mul(columns,1n),columns,Order.times_one(columns)) : {Nat.is_le(Nat.mul(rows,_),capacity) == True{} : Bool} room)
  Equal.trans(Nat,U32.to_nat(Grid.product([h,w])),Nat.mul(rows,Nat.mul(columns,1n)),Nat.mul(rows,columns),exact,
    Equal.cong(Nat,Nat,value => Nat.mul(rows,value),Nat.mul(columns,1n),columns,Order.times_one(columns)))

# The outer word of the input counts the values of its images.
def image_outer_exact(+limits: Machine.Limits,+mode: Mode,+geometry: Kernel.Geometry,
  +room: {Nat.is_le(Nat.mul(Domain.height(geometry),Domain.width(geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outer_room: {Nat.is_le(Nat.mul(U32.to_nat(images(mode)),Nat.mul(Domain.height(geometry),Domain.width(geometry))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(image_outer(mode,geometry)) == Nat.mul(U32.to_nat(images(mode)),Nat.mul(Domain.height(geometry),Domain.width(geometry))) : Nat}:
  match mode:
    case Single{}:
      +after = image_after(geometry)
      +plane = Nat.mul(Domain.height(geometry),Domain.width(geometry))
      %Equal.sym(U32,(1 * after : U32),after,TiledPublic.one_times(after)) : {U32.to_nat(_) == Nat.add(plane,0n) : Nat}
      Equal.trans(Nat,U32.to_nat(after),plane,Nat.add(plane,0n),image_after_word(limits,geometry,room),Equal.sym(Nat,Nat.add(plane,0n),plane,Addition.zero_right(plane)))
    case Batched{+batch}:
      +after = image_after(geometry)
      +plane = Nat.mul(Domain.height(geometry),Domain.width(geometry))
      +images = U32.to_nat(batch)
      +exact = image_after_word(limits,geometry,room)
      %Equal.sym(U32,(1 * batch : U32),batch,TiledPublic.one_times(batch)) : {U32.to_nat((_ * after : U32)) == Nat.mul(images,plane) : Nat}
      Equal.trans(Nat,U32.to_nat((batch * after : U32)),Nat.mul(images,U32.to_nat(after)),Nat.mul(images,plane),
        WordMultiplication.bounded_multiply(batch,after,Machine.capacity(limits),Limits.capacity_fits(limits),
          %Equal.sym(Nat,U32.to_nat(after),plane,exact) : {Nat.is_le(Nat.mul(images,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} outer_room),
        Equal.cong(Nat,Nat,value => Nat.mul(images,value),U32.to_nat(after),plane,exact))

# The lane axis of the input lies inside its shape.
def image_within(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Nat.is_lt(Conv.lane_of(batched(mode)),List.length(&2,U32,Gradient.input_shape(arguments(mode,geometry)))) == True{} : Bool}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

# The values and the cells of the input of a call.
def image_count(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Nat.mul(U32.to_nat(images(mode)),Domain.image_size(geometry)) == ShapeLaws.elements(Gradient.input_shape(arguments(mode,geometry))) : Nat}:
  match mode:
    case Single{}:
      Kernel.Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
      +size = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w)))
      Equal.trans(Nat,Nat.add(size,0n),size,ShapeLaws.elements([c,h,w]),Addition.zero_right(size),
        Equal.sym(Nat,ShapeLaws.elements([c,h,w]),size,ShapeLaws.elements_three(c,h,w)))
    case Batched{+batch}:
      Kernel.Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
      +size = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w)))
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(batch),value),size,ShapeLaws.elements([c,h,w]),
        Equal.sym(Nat,ShapeLaws.elements([c,h,w]),size,ShapeLaws.elements_three(c,h,w)))

def image_cells_count(+mode: Mode,+geometry: Kernel.Geometry) ->
  {ShapeLaws.elements(Gradient.input_cell_shape(arguments(mode,geometry))) ==
    Nat.mul(Nat.mul(U32.to_nat(images(mode)),Domain.height(geometry)),Nat.mul(Domain.width(geometry),U32.to_nat(Tiled.groups(Domain.channel_word(geometry))))) : Nat}:
  match mode:
    case Single{}:
      Kernel.Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
      +groups = Tiled.groups(c)
      +rest = Nat.mul(U32.to_nat(w),U32.to_nat(groups))
      Equal.trans(Nat,ShapeLaws.elements([h,w,groups]),Nat.mul(U32.to_nat(h),rest),Nat.mul(Nat.add(U32.to_nat(h),0n),rest),ShapeLaws.elements_three(h,w,groups),
        Equal.cong(Nat,Nat,value => Nat.mul(value,rest),U32.to_nat(h),Nat.add(U32.to_nat(h),0n),Equal.sym(Nat,Nat.add(U32.to_nat(h),0n),U32.to_nat(h),Addition.zero_right(U32.to_nat(h)))))
    case Batched{+batch}:
      Kernel.Geometry{+c,o,+h,+w,k,s,p,gh,gw} = geometry
      +groups = Tiled.groups(c)
      +rest = Nat.mul(U32.to_nat(w),U32.to_nat(groups))
      Equal.trans(Nat,ShapeLaws.elements([batch,h,w,groups]),Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(h),rest)),Nat.mul(Nat.mul(U32.to_nat(batch),U32.to_nat(h)),rest),
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(batch),value),ShapeLaws.elements([h,w,groups]),Nat.mul(U32.to_nat(h),rest),ShapeLaws.elements_three(h,w,groups)),
        Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(batch),U32.to_nat(h)),rest),Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(h),rest)),RowMajor.assoc(U32.to_nat(batch),U32.to_nat(h),rest)))

# The lane axis of the input lies at most at its rank, it has at most four
# axes, and its tiled cells are the cells of a call.
def image_axis_inside(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Nat.is_le(Conv.lane_of(batched(mode)),List.length(&2,U32,Gradient.input_shape(arguments(mode,geometry)))) == True{} : Bool}:
  Bounds.strict_le(Conv.lane_of(batched(mode)),List.length(&2,U32,Gradient.input_shape(arguments(mode,geometry))),image_within(mode,geometry))

def image_rank(+mode: Mode,+geometry: Kernel.Geometry) -> {Nat.is_le(List.length(&2,U32,Gradient.input_shape(arguments(mode,geometry))),4n) == True{} : Bool}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

def image_shape_axis(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Tiled.lane_axis(List.length(&2,U32,Gradient.input_shape(arguments(mode,geometry)))) == Conv.lane_of(batched(mode)) : Nat}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

def image_cell_shape(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Tiled.cell_shape(Gradient.input_shape(arguments(mode,geometry)),Conv.lane_of(batched(mode)),Grid.Innermost{}) == Gradient.input_cell_shape(arguments(mode,geometry)) :
    List<&2,U32>}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

# The input lies in the cells of its image geometry, with its channels in
# the lanes.
def image_cells_geometry(+mode: Mode,+geometry: Kernel.Geometry) ->
  {TiledPublic.geometry_of(Gradient.input_shape(arguments(mode,geometry)),Conv.lane_of(batched(mode)),Grid.Innermost{}) ==
    Cells.Cells{image_outer(mode,geometry),image_after(geometry),Domain.channel_word(geometry),Tiled.groups(Domain.channel_word(geometry)),Grid.Innermost{}} : Cells.Cells}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

def image_lanes(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Grid.lanes_at(Gradient.input_shape(arguments(mode,geometry)),Conv.lane_of(batched(mode))) == Domain.channel_word(geometry) : U32}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

# The words of the cells of the output gradient: the values after its
# outputs and the values of its images.
def gradient_after(+geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,+gh,+gw} = geometry
  Grid.product([gh,gw])

def gradient_outer(+mode: Mode,+geometry: Kernel.Geometry) -> U32:
  match mode:
    case Single{}: (1 * gradient_after(geometry) : U32)
    case Batched{+batch}: ((1 * batch : U32) * gradient_after(geometry) : U32)

# The output gradient of a call lies in the cells of its image geometry, with
# its outputs in the lanes.
def gradient_geometry(+mode: Mode,+geometry: Kernel.Geometry) ->
  {TiledPublic.geometry_of(input_shape(mode,geometry),Conv.lane_of(batched(mode)),Grid.Innermost{}) ==
    Cells.Cells{gradient_outer(mode,geometry),gradient_after(geometry),outputs_word(geometry),Tiled.groups(outputs_word(geometry)),Grid.Innermost{}} : Cells.Cells}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

def gradient_lanes(+mode: Mode,+geometry: Kernel.Geometry) ->
  {Grid.lanes_at(input_shape(mode,geometry),Conv.lane_of(batched(mode))) == outputs_word(geometry) : U32}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

# The words of the cells of the output gradient count its values.
def gradient_after_word(+limits: Machine.Limits,+geometry: Kernel.Geometry,
  +room: {Nat.is_le(Nat.mul(Domain.gradient_height(geometry),Domain.gradient_width(geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(gradient_after(geometry)) == Nat.mul(Domain.gradient_height(geometry),Domain.gradient_width(geometry)) : Nat}:
  Kernel.Geometry{c,o,h,w,k,s,p,+gh,+gw} = geometry
  +rows = U32.to_nat(gh)
  +columns = U32.to_nat(gw)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +exact = TiledPublic.product_exact(Machine.capacity(limits),Limits.capacity_fits(limits),[gh,gw],
    %Equal.sym(Nat,Nat.mul(columns,1n),columns,Order.times_one(columns)) : {Nat.is_le(Nat.mul(rows,_),capacity) == True{} : Bool} room)
  Equal.trans(Nat,U32.to_nat(Grid.product([gh,gw])),Nat.mul(rows,Nat.mul(columns,1n)),Nat.mul(rows,columns),exact,
    Equal.cong(Nat,Nat,value => Nat.mul(rows,value),Nat.mul(columns,1n),columns,Order.times_one(columns)))

def gradient_outer_exact(+limits: Machine.Limits,+mode: Mode,+geometry: Kernel.Geometry,
  +room: {Nat.is_le(Nat.mul(Domain.gradient_height(geometry),Domain.gradient_width(geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +outer_room: {Nat.is_le(Nat.mul(U32.to_nat(images(mode)),Nat.mul(Domain.gradient_height(geometry),Domain.gradient_width(geometry))),U32.to_nat(Machine.capacity(limits))) ==
    True{} : Bool}) ->
  {U32.to_nat(gradient_outer(mode,geometry)) == Nat.mul(U32.to_nat(images(mode)),Nat.mul(Domain.gradient_height(geometry),Domain.gradient_width(geometry))) : Nat}:
  match mode:
    case Single{}:
      +after = gradient_after(geometry)
      +plane = Nat.mul(Domain.gradient_height(geometry),Domain.gradient_width(geometry))
      %Equal.sym(U32,(1 * after : U32),after,TiledPublic.one_times(after)) : {U32.to_nat(_) == Nat.add(plane,0n) : Nat}
      Equal.trans(Nat,U32.to_nat(after),plane,Nat.add(plane,0n),gradient_after_word(limits,geometry,room),Equal.sym(Nat,Nat.add(plane,0n),plane,Addition.zero_right(plane)))
    case Batched{+batch}:
      +after = gradient_after(geometry)
      +plane = Nat.mul(Domain.gradient_height(geometry),Domain.gradient_width(geometry))
      +images = U32.to_nat(batch)
      +exact = gradient_after_word(limits,geometry,room)
      %Equal.sym(U32,(1 * batch : U32),batch,TiledPublic.one_times(batch)) : {U32.to_nat((_ * after : U32)) == Nat.mul(images,plane) : Nat}
      Equal.trans(Nat,U32.to_nat((batch * after : U32)),Nat.mul(images,U32.to_nat(after)),Nat.mul(images,plane),
        WordMultiplication.bounded_multiply(batch,after,Machine.capacity(limits),Limits.capacity_fits(limits),
          %Equal.sym(Nat,U32.to_nat(after),plane,exact) : {Nat.is_le(Nat.mul(images,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} outer_room),
        Equal.cong(Nat,Nat,value => Nat.mul(images,value),U32.to_nat(after),plane,exact))
