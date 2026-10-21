# Gradients of conv2d in a fixed FP32 order. Every function gives the
# tensors it only reads back to its caller.
import Base
import ./tensor.bend as Tensors
import ./tensor_band.bend as Bands
import ./tensor_convolution.bend as TensorConvolution
import ./kernel_input_gradient.bend as InputKernel
import ./kernel_shape.bend as KernelShape
import ./storage_buffer.bend as Storage
import ./default_machine_limits.bend as Default
import ./machine_limits.bend as Machine
import ./tensor_parameter_gradient.bend as ParameterGradient

def Tensor() -> Type:
  Tensors.Tensor<F32>

type Arguments is Data:
  Arguments{geometry: InputKernel.Geometry,batch: U32,batched: Bool}

def input_dimensions(arguments: Arguments) -> KernelShape.Dimensions:
  Arguments{InputKernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,batched} = arguments
  KernelShape.Dimensions{U32.to_nat(c),U32.to_nat(o),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)}

# The bands of the input gradient: rows of every image of the batch.
def input_band_geometry(arguments: Arguments) -> Bands.Geometry:
  Arguments{InputKernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,batched} = arguments
  Bands.Geometry{batch,c,h,w}

def input_shape(arguments: Arguments) -> List<&2,U32>:
  Arguments{InputKernel.Geometry{c,o,h,w,k,s,p,gh,gw},batch,batched} = arguments
  TensorConvolution.output_shape(batched,batch,c,h,w)

# The limits are an argument so that a proof can keep them symbolic.
def input_valid_with(+arguments: Arguments,d_length: Nat,+weight_length: Nat,+limits: Machine.Limits) -> Bool:
  Arguments{+geometry,+batch,batched} = arguments
  InputKernel.Geometry{+c,+o,+h,+w,+k,+s,+p,+gh,+gw} = geometry
  +dimensions = input_dimensions(arguments)
  +input_length = Nat.mul(U32.to_nat(c),Nat.mul(U32.to_nat(h),U32.to_nat(w)))
  KernelShape.valid(dimensions,input_length,weight_length,U32.to_nat(o)) &&
    KernelShape.supports_direct(limits,dimensions,input_length,weight_length) &&
    Nat.is_eq(U32.to_nat(gh),KernelShape.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))) &&
    Nat.is_eq(U32.to_nat(gw),KernelShape.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))) &&
    Nat.is_eq(d_length,Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(o),Nat.mul(U32.to_nat(gh),U32.to_nat(gw))))) &&
    InputKernel.fits(geometry,Machine.capacity(limits))

def input_valid(arguments: Arguments,d_length: Nat,weight_length: Nat) -> Bool:
  input_valid_with(arguments,d_length,weight_length,Default.limits())

def input_result(+arguments: Arguments,+workers: U32,+weight_shape: List<&2,U32>,+weight_fill: F32,+weight_workers: U32,result: InputKernel.Outcome) -> Tensor() & Tensor():
  InputKernel.Outcome{weights,output} = result
  (Tensors.Tensor{output,input_band_geometry(arguments),input_shape(arguments),0.0,Tensors.Usable{},workers},Tensors.whole(~F32,weights,weight_shape,weight_fill,Tensors.Usable{},weight_workers))

def input_checked(accepted: Bool,+arguments: Arguments,+count: U32,gradient: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,+workers: U32,+weight_shape: List<&2,U32>,+weight_fill: F32,+weight_workers: U32) -> Tensor() & Tensor():
  match accepted:
    case False{}:
      (Tensors.failed(~F32,"conv2d_input_gradient",Tensors.InvalidArgument{input_shape(arguments)},input_shape(arguments),0.0,workers),Tensors.whole(~F32,weights,weight_shape,weight_fill,Tensors.Usable{},weight_workers))
    case True{}:
      Arguments{+geometry,batch,batched} = arguments
      input_result(arguments,workers,weight_shape,weight_fill,weight_workers,InputKernel.run(geometry,U32.to_nat(count),U32.max(workers,U32.max(weight_workers,1)),gradient,weights))

def input_count(count: Maybe<&2,U32>,+arguments: Arguments,gradient: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,+workers: U32,+weight_shape: List<&2,U32>,+weight_fill: F32,+weight_workers: U32) -> Tensor() & Tensor():
  match count:
    case None{}:
      (Tensors.failed(~F32,"conv2d_input_gradient",Tensors.InvalidShape{input_shape(arguments)},input_shape(arguments),0.0,workers),Tensors.whole(~F32,weights,weight_shape,weight_fill,Tensors.Usable{},weight_workers))
    case Some{+count}:
      Storage.Buffer{d,+d_length,d_certificate} = gradient
      Storage.Buffer{w,+weight_length,w_certificate} = weights
      input_checked(input_valid(arguments,d_length,weight_length),arguments,count,gradient,weights,workers,weight_shape,weight_fill,weight_workers)

def input_geometry(+stride: U32,+padding: U32,+height: U32,+width: U32,shape: Maybe<&2,TensorConvolution.InputShape>,+weight_shape: List<&2,U32>,gradient: Storage.Buffer<F32>,weights: Storage.Buffer<F32>,+workers: U32,+weight_fill: F32,+weight_workers: U32) -> Tensor() & Tensor():
  match shape weight_shape:
    case Some{TensorConvolution.InputShape{+batch,+batched,+outputs,+gh,+gw}} Con{wo,Con{+channels,Con{+kernel,Con{kw,Nil{}}}}}:
      +arguments = {Arguments{InputKernel.Geometry{channels,outputs,height,width,kernel,stride,padding,gh,gw},batch,batched} : Arguments}
      input_count(Bool.pick(Maybe<&2,U32>,U32.is_eq(outputs,wo) && U32.is_eq(kernel,kw),Tensors.count_of(input_shape(arguments)),None{}),arguments,gradient,weights,workers,weight_shape,weight_fill,weight_workers)
    case shape weight_shape:
      (Tensors.failed(~F32,"conv2d_input_gradient",Tensors.InvalidShape{weight_shape},weight_shape,0.0,workers),Tensors.whole(~F32,weights,weight_shape,weight_fill,Tensors.Usable{},weight_workers))

def input_status(status: Tensors.Status,+stride: U32,+padding: U32,+height: U32,+width: U32,gradient: Storage.Buffer<F32>,+d_shape: List<&2,U32>,+d_fill: F32,+workers: U32,weights: Storage.Buffer<F32>,+weight_shape: List<&2,U32>,+weight_fill: F32,+weight_status: Tensors.Status,+weight_workers: U32) -> Tensor() & Tensor():
  match status:
    case Tensors.Failed{problem}:
      (Tensors.whole(~F32,gradient,d_shape,d_fill,Tensors.Failed{problem},workers),Tensors.whole(~F32,weights,weight_shape,weight_fill,weight_status,weight_workers))
    case Tensors.Usable{}:
      input_geometry(stride,padding,height,width,TensorConvolution.input_shape(d_shape),weight_shape,gradient,weights,workers,weight_fill,weight_workers)

# d [N,O,Ho,Wo] or [O,Ho,Wo] and weights [O,C,k,k] of a convolution of an
# input of the given height and width: (the gradient of that input, the
# weights). Several input sizes give the same output size when the stride is
# above 1, so height and width are given.
def input(d: Tensor(),weights: Tensor(),+stride: U32,+padding: U32,+height: U32,+width: U32) -> Tensor() & Tensor():
  Tensors.with_whole(~F32,~(Tensor() & Tensor()),d,gradient => d_shape => d_fill => d_status => workers =>
    Tensors.with_whole(~F32,~(Tensor() & Tensor()),weights,weights => weight_shape => weight_fill => weight_status => weight_workers =>
      +weight_status = weight_status
      input_status(Tensors.first_failure(d_status,weight_status),stride,padding,height,width,gradient,d_shape,d_fill,workers,weights,weight_shape,weight_fill,weight_status,weight_workers)))

# The group length of the weight and bias gradient contract.
def group_length() -> U32:
  256

# x [N,C,H,W], d [N,O,Ho,Wo], or both without their batch axes.
# Returns (weight gradient [O,C,k,k], (x,d)); all sources are read only.
def weight(x: Tensor(),d: Tensor(),kernel: U32,stride: U32,padding: U32) -> Tensor() & (Tensor() & Tensor()):
  ParameterGradient.weight(x,d,kernel,stride,padding,group_length())

# d [N,O,Ho,Wo] or [O,Ho,Wo]: (bias gradient [O], d).
def bias(d: Tensor()) -> Tensor() & Tensor():
  ParameterGradient.bias(d,group_length())
