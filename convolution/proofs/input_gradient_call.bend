# A ready result of input implies the bounds of its validation and allocation.
import Base
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_layout.bend as Layout
import ../../lib/tensor_convolution.bend as Conv
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_shape.bend as Shape
import ../../lib/default_machine_limits.bend as Default
import ../../lib/machine_limits.bend as Machine
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_storage_count.bend as StorageCount
import ../../lib/proofs/tensor_whole_leaf.bend as WholeLeaf
import ../../lib/tensor_band.bend as Bands
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/input_gradient_phase_domain.bend as Coordinates
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/word_encoding.bend as Words
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic

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

def geometry(arguments: Gradient.Arguments) -> Kernel.Geometry:
  Gradient.Arguments{geometry,batch,batched} = arguments
  geometry
def batch(arguments: Gradient.Arguments) -> Nat:
  Gradient.Arguments{geometry,batch,batched} = arguments
  U32.to_nat(batch)
def ready(pair: Tensors.Tensor<F32> & Tensors.Tensor<F32>) -> Bool:
  Contract.tensor_ready(~F32,Tensors.result_of(~F32,pair))
def validation_conditions(+arguments: Gradient.Arguments,d_length: Nat,+weight_length: Nat,+limits: Machine.Limits) -> List<&2,Bool>:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+batch,batched} = arguments
  +dimensions = Gradient.input_dimensions(arguments)
  +input_length = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w)))
  [Kernel.fits(geometry(arguments),Machine.capacity(limits)),Nat.is_eq(d_length,Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(gh),U32.to_nat(gw))))),
    Nat.is_eq(U32.to_nat(gw),Shape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))),
    Nat.is_eq(U32.to_nat(gh),Shape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))),
    Shape.supports_direct(limits,dimensions,input_length,weight_length),Shape.valid(dimensions,input_length,weight_length,U32.to_nat(o))]
def validated_condition(+arguments: Gradient.Arguments,+d_length: Nat,+weight_length: Nat,+limits: Machine.Limits,+index: Nat,
  accepted: {Gradient.input_valid_with(arguments,d_length,weight_length,limits) == True{} : Bool}) ->
  {Boolean.condition_at(validation_conditions(arguments,d_length,weight_length,limits),index) == True{} : Bool}:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+batch,batched} = arguments
  Boolean.every_condition_holds(validation_conditions(arguments,d_length,weight_length,limits),index,accepted)
def valid_condition(+arguments: Gradient.Arguments,+d_length: Nat,+weight_length: Nat,+limits: Machine.Limits,+index: Nat,
  accepted: {Gradient.input_valid_with(arguments,d_length,weight_length,limits) == True{} : Bool}) ->
  {Boolean.condition_at(valid_conditions(Gradient.input_dimensions(arguments),Coordinates.image_size(geometry(arguments)),weight_length,Coordinates.outputs(geometry(arguments))),index) == True{} : Bool}:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+batch,batched} = arguments
  shape_valid_condition(Gradient.input_dimensions(arguments),Coordinates.image_size(geometry(arguments)),weight_length,U32.to_nat(o),index,validated_condition(arguments,d_length,weight_length,limits,5n,accepted))
def direct_condition(+arguments: Gradient.Arguments,+d_length: Nat,+weight_length: Nat,+limits: Machine.Limits,+index: Nat,
  accepted: {Gradient.input_valid_with(arguments,d_length,weight_length,limits) == True{} : Bool}) ->
  {Boolean.condition_at(direct_conditions(Machine.dimension(limits),Machine.capacity(limits),Machine.coordinate(limits),Gradient.input_dimensions(arguments),Coordinates.image_size(geometry(arguments)),weight_length),index) == True{} : Bool}:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+batch,batched} = arguments
  Machine.Limits{+dimension,+capacity,+coordinate,half,facts} = limits
  shape_direct_condition(dimension,capacity,coordinate,Gradient.input_dimensions(arguments),Coordinates.image_size(geometry(arguments)),weight_length,index,
    validated_condition(arguments,d_length,weight_length,limits,4n,accepted))

def checked_guard(accepted: Bool,+arguments: Gradient.Arguments,+count: U32,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+workers: U32,+weight_shape: List<&2,U32>,+weight_fill: F32,+weight_workers: U32,
  accepted_result: {ready(Gradient.input_checked(accepted,arguments,count,Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_shape,weight_fill,weight_workers)) == True{} : Bool}) -> {accepted == True{} : Bool}:
  match accepted:
    case True{}: {==}
    case False{}: Empty.absurd({False{} == True{} : Bool},Words.false_true(accepted_result))

type Selected<-arguments: Gradient.Arguments,-gradient: Buffer.Model<F32>,-weights: Buffer.Model<F32>> is Data:
  Selected{count: U32,counted: {Tensors.count_of(Gradient.input_shape(arguments)) == Some{count} : Maybe<&2,U32>},valid: {Gradient.input_valid(arguments,Buffer.length(~F32,gradient),Buffer.length(~F32,weights)) == True{} : Bool}}
def counted(result: Maybe<&2,U32>,+arguments: Gradient.Arguments,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+workers: U32,+weight_shape: List<&2,U32>,+weight_fill: F32,+weight_workers: U32,
  +equation: {Tensors.count_of(Gradient.input_shape(arguments)) == result : Maybe<&2,U32>},
  accepted_result: {ready(Gradient.input_count(result,arguments,Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_shape,weight_fill,weight_workers)) == True{} : Bool}) -> Selected<arguments,gradient,weights>:
  match result:
    case None{}: Empty.absurd(Selected<arguments,gradient,weights>,Words.false_true(accepted_result))
    case Some{+count}:
      Buffer.Model{+dt,+dl,+dc} = gradient
      Buffer.Model{+wt,+wl,+wc} = weights
      Selected{count,equation,checked_guard(Gradient.input_valid(arguments,dl,wl),arguments,count,gradient,weights,workers,weight_shape,weight_fill,weight_workers,accepted_result)}

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
def input_shape(+mode: Mode,geometry: Kernel.Geometry) -> List<&2,U32>:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Conv.output_shape(batched(mode),images(mode),o,gh,gw)
def weight_shape(geometry: Kernel.Geometry) -> List<&2,U32>:
  Kernel.Geometry{c,o,h,w,+k,s,p,gh,gw} = geometry
  [o,c,k,k]
def parsed_shape(+mode: Mode,geometry: Kernel.Geometry) -> Conv.InputShape:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  Conv.InputShape{images(mode),batched(mode),o,gh,gw}
def parse(+mode: Mode,+geometry: Kernel.Geometry) -> {Conv.input_shape(input_shape(mode,geometry)) == Some{parsed_shape(mode,geometry)} : Maybe<&2,Conv.InputShape>}:
  match mode:
    case Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
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
def parsed_call(+mode: Mode,+geometry: Kernel.Geometry,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+workers: U32,+weight_fill: F32,+weight_workers: U32) ->
  {Gradient.input_geometry(stride(geometry),padding(geometry),height(geometry),width(geometry),Some{parsed_shape(mode,geometry)},weight_shape(geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_fill,weight_workers) ==
    Gradient.input_count(Tensors.count_of(Gradient.input_shape(arguments(mode,geometry))),arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_shape(geometry),weight_fill,weight_workers) : Tensors.Tensor<F32> & Tensors.Tensor<F32>}:
  Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  %Equal.sym(Bool,U32.is_eq(o,o),True{},Arithmetic.equal_reflexive(o)) :
    {Gradient.input_count(Bool.pick(Maybe<&2,U32>,_ && U32.is_eq(k,k),Tensors.count_of(Gradient.input_shape(arguments(mode,geometry))),None{}),arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_shape(geometry),weight_fill,weight_workers) ==
      Gradient.input_count(Tensors.count_of(Gradient.input_shape(arguments(mode,geometry))),arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_shape(geometry),weight_fill,weight_workers) : Tensors.Tensor<F32> & Tensors.Tensor<F32>}
  %Equal.sym(Bool,U32.is_eq(k,k),True{},Arithmetic.equal_reflexive(k)) :
    {Gradient.input_count(Bool.pick(Maybe<&2,U32>,_,Tensors.count_of(Gradient.input_shape(arguments(mode,geometry))),None{}),arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_shape(geometry),weight_fill,weight_workers) ==
      Gradient.input_count(Tensors.count_of(Gradient.input_shape(arguments(mode,geometry))),arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_shape(geometry),weight_fill,weight_workers) : Tensors.Tensor<F32> & Tensors.Tensor<F32>}
  {==}
def public_call(+mode: Mode,+geometry: Kernel.Geometry,+gradient: Buffer.Model<F32>,+d_fill: F32,+workers: U32,+weights: Buffer.Model<F32>,+weight_fill: F32,+weight_workers: U32) ->
  {Gradient.input(Tensors.whole(~F32,Buffer.pack(~F32,gradient),input_shape(mode,geometry),d_fill,Tensors.Usable{},workers),Tensors.whole(~F32,Buffer.pack(~F32,weights),weight_shape(geometry),weight_fill,Tensors.Usable{},weight_workers),stride(geometry),padding(geometry),height(geometry),width(geometry)) ==
    Gradient.input_count(Tensors.count_of(Gradient.input_shape(arguments(mode,geometry))),arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_shape(geometry),weight_fill,weight_workers) : Tensors.Tensor<F32> & Tensors.Tensor<F32>}:
  +gradient_whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(input_shape(mode,geometry))} : Bands.Geometry}
  +weight_whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(weight_shape(geometry))} : Bands.Geometry}
  %Equal.sym(Storage.Buffer<F32>,Tensors.whole_storage(~F32,gradient_whole,d_fill,Tensors.arrive(~F32,gradient_whole,Bands.unsplit(gradient_whole),
      Tensors.whole(~F32,Buffer.pack(~F32,gradient),input_shape(mode,geometry),d_fill,Tensors.Usable{},workers))),Buffer.pack(~F32,gradient),
      WholeLeaf.arrived_whole(~F32,Buffer.pack(~F32,gradient),input_shape(mode,geometry),d_fill,Tensors.Usable{},workers)) :
    {Gradient.input_status(Tensors.Usable{},stride(geometry),padding(geometry),height(geometry),width(geometry),_,input_shape(mode,geometry),d_fill,workers,
      Tensors.whole_storage(~F32,weight_whole,weight_fill,Tensors.arrive(~F32,weight_whole,Bands.unsplit(weight_whole),
      Tensors.whole(~F32,Buffer.pack(~F32,weights),weight_shape(geometry),weight_fill,Tensors.Usable{},weight_workers))),weight_shape(geometry),weight_fill,Tensors.Usable{},weight_workers) ==
      Gradient.input_count(Tensors.count_of(Gradient.input_shape(arguments(mode,geometry))),arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_shape(geometry),weight_fill,weight_workers) : Tensors.Tensor<F32> & Tensors.Tensor<F32>}
  %Equal.sym(Storage.Buffer<F32>,Tensors.whole_storage(~F32,weight_whole,weight_fill,Tensors.arrive(~F32,weight_whole,Bands.unsplit(weight_whole),
      Tensors.whole(~F32,Buffer.pack(~F32,weights),weight_shape(geometry),weight_fill,Tensors.Usable{},weight_workers))),Buffer.pack(~F32,weights),
      WholeLeaf.arrived_whole(~F32,Buffer.pack(~F32,weights),weight_shape(geometry),weight_fill,Tensors.Usable{},weight_workers)) :
    {Gradient.input_status(Tensors.Usable{},stride(geometry),padding(geometry),height(geometry),width(geometry),Buffer.pack(~F32,gradient),input_shape(mode,geometry),d_fill,workers,
      _,weight_shape(geometry),weight_fill,Tensors.Usable{},weight_workers) ==
      Gradient.input_count(Tensors.count_of(Gradient.input_shape(arguments(mode,geometry))),arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_shape(geometry),weight_fill,weight_workers) : Tensors.Tensor<F32> & Tensors.Tensor<F32>}
  %Equal.sym(Maybe<&2,Conv.InputShape>,Conv.input_shape(input_shape(mode,geometry)),Some{parsed_shape(mode,geometry)},parse(mode,geometry)) :
    {Gradient.input_geometry(stride(geometry),padding(geometry),height(geometry),width(geometry),_,weight_shape(geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_fill,weight_workers) ==
      Gradient.input_count(Tensors.count_of(Gradient.input_shape(arguments(mode,geometry))),arguments(mode,geometry),Buffer.pack(~F32,gradient),Buffer.pack(~F32,weights),workers,weight_shape(geometry),weight_fill,weight_workers) : Tensors.Tensor<F32> & Tensors.Tensor<F32>}
  parsed_call(mode,geometry,gradient,weights,workers,weight_fill,weight_workers)

# The facts the kernel proofs read from a call that passed validation.
def facts(+arguments: Gradient.Arguments,+gradient: Buffer.Model<F32>,+weights: Buffer.Model<F32>,+limits: Machine.Limits,+limits_equal: {limits == Default.limits() : Machine.Limits},
  +accepted: {Gradient.input_valid_with(arguments,Buffer.length(~F32,gradient),Buffer.length(~F32,weights),limits) == True{} : Bool},gradient_ready: Buffer.ContentReady<F32,Buffer.content(~F32,gradient)>) -> Coordinates.Facts<geometry(arguments),batch(arguments),limits>:
  Gradient.Arguments{Kernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw},+n,batched} = arguments
  Buffer.Model{+dt,+dlen,+dc} = gradient
  Machine.Limits{+dimension,+capacity,+coordinate,+half,+facts} = limits
  +dl = Buffer.length(~F32,gradient)
  +wl = Buffer.length(~F32,weights)
  +height = Order.equal_of_is_eq(U32.to_nat(gh),Shape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),validated_condition(arguments,dl,wl,limits,3n,accepted))
  +width = Order.equal_of_is_eq(U32.to_nat(gw),Shape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),validated_condition(arguments,dl,wl,limits,2n,accepted))
  Coordinates.Facts{
    Coordinates.Positive{valid_condition(arguments,dl,wl,limits,10n,accepted),valid_condition(arguments,dl,wl,limits,9n,accepted),valid_condition(arguments,dl,wl,limits,8n,accepted),valid_condition(arguments,dl,wl,limits,7n,accepted),valid_condition(arguments,dl,wl,limits,6n,accepted),valid_condition(arguments,dl,wl,limits,5n,accepted)},
    Coordinates.Dimensions{direct_condition(arguments,dl,wl,limits,11n,accepted),direct_condition(arguments,dl,wl,limits,10n,accepted),direct_condition(arguments,dl,wl,limits,9n,accepted),direct_condition(arguments,dl,wl,limits,8n,accepted),direct_condition(arguments,dl,wl,limits,7n,accepted),direct_condition(arguments,dl,wl,limits,6n,accepted),direct_condition(arguments,dl,wl,limits,5n,accepted)},
    Coordinates.Capacities{direct_condition(arguments,dl,wl,limits,4n,accepted),
    %Order.equal_of_is_eq(wl,Coordinates.weight_size(geometry(arguments)),valid_condition(arguments,dl,wl,limits,1n,accepted)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    direct_condition(arguments,dl,wl,limits,3n,accepted),
    %Equal.sym(Nat,U32.to_nat(gh),Shape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),height) : {Nat.is_le(Nat.mul(U32.to_nat(o),Nat.mul(_,U32.to_nat(gw))),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(gw),Shape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),width) : {Nat.is_le(Nat.mul(U32.to_nat(o),Nat.mul(Shape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),_)),U32.to_nat(capacity)) == True{} : Bool}
    direct_condition(arguments,dl,wl,limits,2n,accepted),
    %Order.equal_of_is_eq(dl,Nat.mul(U32.to_nat(n),Coordinates.gradient_size(geometry(arguments))),validated_condition(arguments,dl,wl,limits,1n,accepted)) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    StorageCount.content_room(~F32,Buffer.content(~F32,gradient),capacity,Equal.cong(Machine.Limits,U32,Machine.capacity,limits,Default.limits(),limits_equal),gradient_ready)},
    Coordinates.GradientAxes{height,width,
    %Equal.sym(Nat,U32.to_nat(gh),Shape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),height) : {Nat.is_le(Nat.add(Nat.mul(_,U32.to_nat(s)),U32.to_nat(k)),U32.to_nat(coordinate)) == True{} : Bool}
    direct_condition(arguments,dl,wl,limits,1n,accepted),
    %Equal.sym(Nat,U32.to_nat(gw),Shape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)),width) : {Nat.is_le(Nat.add(Nat.mul(_,U32.to_nat(s)),U32.to_nat(k)),U32.to_nat(coordinate)) == True{} : Bool}
    direct_condition(arguments,dl,wl,limits,0n,accepted)}}

