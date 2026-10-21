# Input-gradient validation uses the shared row-major index and natural bounds.
import Base
import ../kernel_input_gradient.bend as Kernel
import ../kernel_shape.bend as Shape
import ../machine_limits.bend as Machine
import ./nat_order.bend as Order
import ./nat_division.bend as NaturalDivision
import ./row_major_index.bend as RowMajor
import ./row_major_coordinates.bend as Grid
import ./machine_limit_bounds.bend as Limits
import ./encoded_coordinates.bend as Encoded

def channels(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  U32.to_nat(c)
def height(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  U32.to_nat(h)
def width(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  U32.to_nat(w)
def image_size(+geometry: Kernel.Geometry) -> Nat:
  Nat.mul(channels(geometry),Nat.mul(height(geometry),width(geometry)))

type Point is Data:
  Point{image: Nat,channel: Nat,row: Nat,column: Nat}
def image(point: Point) -> Nat:
  Point{image,channel,row,column} = point
  image
def channel(point: Point) -> Nat:
  Point{image,channel,row,column} = point
  channel
def row(point: Point) -> Nat:
  Point{image,channel,row,column} = point
  row
def column(point: Point) -> Nat:
  Point{image,channel,row,column} = point
  column
def from_coordinates(image: Nat,value: Grid.Values) -> Point:
  Grid.Values{channel,row,column} = value
  Point{image,channel,row,column}
def point(+geometry: Kernel.Geometry,+index: Nat) -> Point:
  from_coordinates(Nat.div(index,image_size(geometry)),Grid.values(height(geometry),width(geometry),Nat.mod(index,image_size(geometry))))

type Positive<-geometry: Kernel.Geometry> is Data:
  Positive{channels: {Nat.is_lt(0n,channels(geometry)) == True{} : Bool},outputs: {Nat.is_lt(0n,outputs(geometry)) == True{} : Bool},
    height: {Nat.is_lt(0n,height(geometry)) == True{} : Bool},width: {Nat.is_lt(0n,width(geometry)) == True{} : Bool},
    kernel: {Nat.is_lt(0n,kernel(geometry)) == True{} : Bool},stride: {Nat.is_lt(0n,stride(geometry)) == True{} : Bool}}
type Inside<-geometry: Kernel.Geometry,-batch: Nat,-point: Point> is Data:
  Inside{image: {Nat.is_lt(image(point),batch) == True{} : Bool},channel: {Nat.is_lt(channel(point),channels(geometry)) == True{} : Bool},row: {Nat.is_lt(row(point),height(geometry)) == True{} : Bool},column: {Nat.is_lt(column(point),width(geometry)) == True{} : Bool}}
def channel_inside(+geometry: Kernel.Geometry,+batch: Nat,+point: Point,inside: Inside<geometry,batch,point>) -> {Nat.is_lt(channel(point),channels(geometry)) == True{} : Bool}:
  Inside{i,c,r,x} = inside
  c
def column_inside(+geometry: Kernel.Geometry,+batch: Nat,+point: Point,inside: Inside<geometry,batch,point>) -> {Nat.is_lt(column(point),width(geometry)) == True{} : Bool}:
  Inside{i,c,r,x} = inside
  x

def outputs(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  U32.to_nat(o)
def kernel(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  U32.to_nat(k)
def stride(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  U32.to_nat(s)
def padding(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  U32.to_nat(p)
def gradient_height(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  U32.to_nat(gh)
def gradient_width(geometry: Kernel.Geometry) -> Nat:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  U32.to_nat(gw)
def gradient_size(+geometry: Kernel.Geometry) -> Nat:
  Nat.mul(outputs(geometry),Nat.mul(gradient_height(geometry),gradient_width(geometry)))
def weight_size(+geometry: Kernel.Geometry) -> Nat:
  Nat.mul(outputs(geometry),Nat.mul(channels(geometry),Nat.mul(kernel(geometry),kernel(geometry))))

# The facts validation proves about a call: positive sizes, sizes within the
# dimension limit, buffers within the capacity, and the gradient axes.
type Dimensions<-geometry: Kernel.Geometry,-limits: Machine.Limits> is Data:
  Dimensions{channels: {Nat.is_le(channels(geometry),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},outputs: {Nat.is_le(outputs(geometry),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
    height: {Nat.is_le(height(geometry),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},width: {Nat.is_le(width(geometry),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
    kernel: {Nat.is_le(kernel(geometry),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},stride: {Nat.is_le(stride(geometry),U32.to_nat(Machine.dimension(limits))) == True{} : Bool},
    padding: {Nat.is_le(padding(geometry),U32.to_nat(Machine.dimension(limits))) == True{} : Bool}}
type Capacities<-geometry: Kernel.Geometry,-batch: Nat,-limits: Machine.Limits> is Data:
  Capacities{image: {Nat.is_le(image_size(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},weight: {Nat.is_le(weight_size(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    gradient_image: {Nat.is_le(gradient_size(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    gradient: {Nat.is_le(Nat.mul(batch,gradient_size(geometry)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
type GradientAxes<-geometry: Kernel.Geometry,-limits: Machine.Limits> is Data:
  GradientAxes{rows: {gradient_height(geometry) == Shape.axis_size(height(geometry),kernel(geometry),stride(geometry),padding(geometry)) : Nat},
    columns: {gradient_width(geometry) == Shape.axis_size(width(geometry),kernel(geometry),stride(geometry),padding(geometry)) : Nat},
    row_source: {Nat.is_le(Nat.add(Nat.mul(gradient_height(geometry),stride(geometry)),kernel(geometry)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool},
    column_source: {Nat.is_le(Nat.add(Nat.mul(gradient_width(geometry),stride(geometry)),kernel(geometry)),U32.to_nat(Machine.coordinate(limits))) == True{} : Bool}}
type Facts<-geometry: Kernel.Geometry,-batch: Nat,-limits: Machine.Limits> is Data:
  Facts{positive: Positive<geometry>,dimensions: Dimensions<geometry,limits>,capacities: Capacities<geometry,batch,limits>,gradient: GradientAxes<geometry,limits>}

# A validated call: its geometry, batch and machine limits with their facts.
type Call is Data:
  Call{geometry: Kernel.Geometry,batch: Nat,limits: Machine.Limits,facts: Facts<geometry,batch,limits>}
def geometry(call: Call) -> Kernel.Geometry:
  Call{geometry,batch,limits,facts} = call
  geometry
def batch(call: Call) -> Nat:
  Call{geometry,batch,limits,facts} = call
  batch
def limits(call: Call) -> Machine.Limits:
  Call{geometry,batch,limits,facts} = call
  limits

def channels_positive(call: Call) -> {Nat.is_lt(0n,channels(geometry(call))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{Positive{channels,outputs,height,width,kernel,stride},dimensions,capacities,gradient}} = call
  channels
def outputs_positive(call: Call) -> {Nat.is_lt(0n,outputs(geometry(call))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{Positive{channels,outputs,height,width,kernel,stride},dimensions,capacities,gradient}} = call
  outputs
def height_positive(call: Call) -> {Nat.is_lt(0n,height(geometry(call))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{Positive{channels,outputs,height,width,kernel,stride},dimensions,capacities,gradient}} = call
  height
def width_positive(call: Call) -> {Nat.is_lt(0n,width(geometry(call))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{Positive{channels,outputs,height,width,kernel,stride},dimensions,capacities,gradient}} = call
  width
def kernel_positive(call: Call) -> {Nat.is_lt(0n,kernel(geometry(call))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{Positive{channels,outputs,height,width,kernel,stride},dimensions,capacities,gradient}} = call
  kernel
def stride_positive(call: Call) -> {Nat.is_lt(0n,stride(geometry(call))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{Positive{channels,outputs,height,width,kernel,stride},dimensions,capacities,gradient}} = call
  stride

def channels_room(call: Call) -> {Nat.is_le(channels(geometry(call)),U32.to_nat(Machine.dimension(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,Dimensions{channels,outputs,height,width,kernel,stride,padding},capacities,gradient}} = call
  channels
def outputs_room(call: Call) -> {Nat.is_le(outputs(geometry(call)),U32.to_nat(Machine.dimension(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,Dimensions{channels,outputs,height,width,kernel,stride,padding},capacities,gradient}} = call
  outputs
def height_room(call: Call) -> {Nat.is_le(height(geometry(call)),U32.to_nat(Machine.dimension(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,Dimensions{channels,outputs,height,width,kernel,stride,padding},capacities,gradient}} = call
  height
def width_room(call: Call) -> {Nat.is_le(width(geometry(call)),U32.to_nat(Machine.dimension(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,Dimensions{channels,outputs,height,width,kernel,stride,padding},capacities,gradient}} = call
  width
def kernel_room(call: Call) -> {Nat.is_le(kernel(geometry(call)),U32.to_nat(Machine.dimension(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,Dimensions{channels,outputs,height,width,kernel,stride,padding},capacities,gradient}} = call
  kernel
def stride_room(call: Call) -> {Nat.is_le(stride(geometry(call)),U32.to_nat(Machine.dimension(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,Dimensions{channels,outputs,height,width,kernel,stride,padding},capacities,gradient}} = call
  stride
def padding_room(call: Call) -> {Nat.is_le(padding(geometry(call)),U32.to_nat(Machine.dimension(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,Dimensions{channels,outputs,height,width,kernel,stride,padding},capacities,gradient}} = call
  padding

def image_room(call: Call) -> {Nat.is_le(image_size(geometry(call)),U32.to_nat(Machine.capacity(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,dimensions,Capacities{image,weight,gradient_image,gradient},axes}} = call
  image
def weight_room(call: Call) -> {Nat.is_le(weight_size(geometry(call)),U32.to_nat(Machine.capacity(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,dimensions,Capacities{image,weight,gradient_image,gradient},axes}} = call
  weight
def gradient_image_room(call: Call) -> {Nat.is_le(gradient_size(geometry(call)),U32.to_nat(Machine.capacity(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,dimensions,Capacities{image,weight,gradient_image,gradient},axes}} = call
  gradient_image

def rows_exact(call: Call) -> {gradient_height(geometry(call)) == Shape.axis_size(height(geometry(call)),kernel(geometry(call)),stride(geometry(call)),padding(geometry(call))) : Nat}:
  Call{geometry,batch,limits,Facts{positive,dimensions,capacities,GradientAxes{rows,columns,row_source,column_source}}} = call
  rows
def columns_exact(call: Call) -> {gradient_width(geometry(call)) == Shape.axis_size(width(geometry(call)),kernel(geometry(call)),stride(geometry(call)),padding(geometry(call))) : Nat}:
  Call{geometry,batch,limits,Facts{positive,dimensions,capacities,GradientAxes{rows,columns,row_source,column_source}}} = call
  columns
def row_source_room(call: Call) -> {Nat.is_le(Nat.add(Nat.mul(gradient_height(geometry(call)),stride(geometry(call))),kernel(geometry(call))),U32.to_nat(Machine.coordinate(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,dimensions,capacities,GradientAxes{rows,columns,row_source,column_source}}} = call
  row_source
def column_source_room(call: Call) -> {Nat.is_le(Nat.add(Nat.mul(gradient_width(geometry(call)),stride(geometry(call))),kernel(geometry(call))),U32.to_nat(Machine.coordinate(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{positive,dimensions,capacities,GradientAxes{rows,columns,row_source,column_source}}} = call
  column_source

def kernel_words_room(+call: Call) -> {Nat.is_le(Nat.mul(channels(geometry(call)),Nat.mul(kernel(geometry(call)),kernel(geometry(call)))),U32.to_nat(Machine.capacity(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,facts} = call
  +words = Nat.mul(channels(geometry),Nat.mul(kernel(geometry),kernel(geometry)))
  Order.transitive(words,weight_size(geometry),U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(words,outputs(geometry)) : {Nat.is_le(words,_) == True{} : Bool} RowMajor.product_covers(words,outputs(geometry),outputs_positive(call)),weight_room(call))
def kernel_area_room(+call: Call) -> {Nat.is_le(Nat.mul(kernel(geometry(call)),kernel(geometry(call))),U32.to_nat(Machine.capacity(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,facts} = call
  +area = Nat.mul(kernel(geometry),kernel(geometry))
  Order.transitive(area,Nat.mul(channels(geometry),area),U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(area,channels(geometry)) : {Nat.is_le(area,_) == True{} : Bool} RowMajor.product_covers(area,channels(geometry),channels_positive(call)),kernel_words_room(call))
def channels_capacity_room(+call: Call) -> {Nat.is_le(channels(geometry(call)),U32.to_nat(Machine.capacity(limits(call)))) == True{} : Bool}:
  Limits.dimension_capacity(limits(call),channels(geometry(call)),channels_room(call))
def plane_room(+call: Call) -> {Nat.is_le(Nat.mul(height(geometry(call)),width(geometry(call))),U32.to_nat(Machine.capacity(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,facts} = call
  +plane = Nat.mul(height(geometry),width(geometry))
  Order.transitive(plane,image_size(geometry),U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(plane,channels(geometry)) : {Nat.is_le(plane,_) == True{} : Bool} RowMajor.product_covers(plane,channels(geometry),channels_positive(call)),image_room(call))
def gradient_area_room(+call: Call) -> {Nat.is_le(Nat.mul(gradient_height(geometry(call)),gradient_width(geometry(call))),U32.to_nat(Machine.capacity(limits(call)))) == True{} : Bool}:
  Call{geometry,batch,limits,facts} = call
  +area = Nat.mul(gradient_height(geometry),gradient_width(geometry))
  Order.transitive(area,gradient_size(geometry),U32.to_nat(Machine.capacity(limits)),
    %RowMajor.multiply_commutative(area,outputs(geometry)) : {Nat.is_le(area,_) == True{} : Bool} RowMajor.product_covers(area,outputs(geometry),outputs_positive(call)),gradient_image_room(call))

def channel_word(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  c
def kernel_word(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  k
def gradient_rows(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  gh
def gradient_columns(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  gw

def height_word(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  h
def width_word(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  w
def stride_word(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  s
def padding_word(geometry: Kernel.Geometry) -> U32:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  p

def image_positive(+call: Call) -> {Nat.is_lt(0n,image_size(geometry(call))) == True{} : Bool}:
  Call{geometry,batch,limits,Facts{Positive{cp,op,hp,wp,kp,sp},dimensions,capacities,gradient}} = call
  Order.product_positive(channels(geometry),Nat.mul(height(geometry),width(geometry)),cp,Order.product_positive(height(geometry),width(geometry),hp,wp))
def image_words_exact(+call: Call) ->
  {U32.to_nat((channel_word(geometry(call)) * height_word(geometry(call)) * width_word(geometry(call)) : U32)) == image_size(geometry(call)) : Nat}:
  Call{geometry,batch,limits,facts} = call
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  %RowMajor.assoc(U32.to_nat(c),U32.to_nat(h),U32.to_nat(w)) : {U32.to_nat((c * h * w : U32)) == _ : Nat}
  Encoded.multiply_three(c,h,w,Machine.capacity(limits),Limits.capacity_fits(limits),width_positive(call),
    %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(c),U32.to_nat(h)),U32.to_nat(w)),image_size(geometry),RowMajor.assoc(U32.to_nat(c),U32.to_nat(h),U32.to_nat(w))) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} image_room(call))

def run_images(+call: Call,+count: Nat,+count_exact: {count == Nat.mul(batch(call),image_size(geometry(call))) : Nat}) ->
  {Nat.div(count,U32.to_nat((channel_word(geometry(call)) * height_word(geometry(call)) * width_word(geometry(call)) : U32))) == batch(call) : Nat}:
  Call{geometry,batch,limits,facts} = call
  %Equal.sym(Nat,U32.to_nat((channel_word(geometry) * height_word(geometry) * width_word(geometry) : U32)),image_size(geometry),image_words_exact(call)) : {Nat.div(count,_) == batch : Nat}
  %Equal.sym(Nat,count,Nat.mul(batch,image_size(geometry)),count_exact) : {Nat.div(_,image_size(geometry)) == batch : Nat}
  NaturalDivision.exact_quotient(batch,image_size(geometry),image_positive(call))
