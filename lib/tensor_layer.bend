# A convolution followed by an activation, as a training layer: forward keeps
# what backward reads, backward gives the gradients in a fixed FP32 order.
import Base
import ./math_activation.bend as Activations
import ./tensor.bend as Tensors
import ./tensor_f32.bend as F
import ./tensor_convolution_gradient.bend as Gradient

def Tensor() -> Type:
  Tensors.Tensor<F32>

type Settings is Data:
  Settings{stride: U32,padding: U32,activation: Activations.Activation}

def settings(stride: U32,padding: U32,activation: Activations.Activation) -> Settings:
  Settings{stride,padding,activation}

# The layer input and the convolution result before the activation. One
# backward call uses it.
def Saved() -> Type:
  Tensor() & Tensor()

def forward_activated(x: Tensor(),weights: Tensor(),bias: Tensor(),pair: Tensor() & Tensor()) -> Tensor() & (Saved() & (Tensor() & Tensor())):
  (output,z) = pair
  (output,((x,z),(weights,bias)))

def forward_copied(settings: Settings,xs: Tensor() & Tensor(),ws: Tensor() & Tensor(),bs: Tensor() & Tensor()) -> Tensor() & (Saved() & (Tensor() & Tensor())):
  Settings{stride,padding,+activation} = settings
  (x,x_kept) = xs
  (weights,weights_kept) = ws
  (bias,bias_kept) = bs
  forward_activated(x_kept,weights_kept,bias_kept,F.activate_forward(F.conv2d(x,weights,bias,stride,padding,Activations.identity()),activation))

# x [N,C,H,W] or [C,H,W], weights [O,C,k,k] and bias [O]:
# (the activated output, (what backward reads, (weights, bias))).
def forward(x: Tensor(),weights: Tensor(),bias: Tensor(),settings: Settings) -> Tensor() & (Saved() & (Tensor() & Tensor())):
  forward_copied(settings,F.copy(x),F.copy(weights),F.copy(bias))

def kernel(shape: List<&2,U32>) -> U32:
  match shape:
    case [outputs,channels,kernel,width]: kernel
    case shape: 0

# The gradients of the parameters with what the input gradient reads:
# (weight gradient, (bias gradient, (dz, (weights, the shape of x)))).
def Parameters() -> Type:
  Tensor() & (Tensor() & (Tensor() & (Tensor() & List<&2,U32>)))

def biased(weight_gradient: Tensor(),weights: Tensor(),shape: List<&2,U32>,pair: Tensor() & Tensor()) -> Parameters():
  (bias_gradient,dz) = pair
  (weight_gradient,(bias_gradient,(dz,(weights,shape))))

def measured(weight_gradient: Tensor(),weights: Tensor(),dz: Tensor(),input: Tensor() & List<&2,U32>) -> Parameters():
  (x,shape) = input
  biased(weight_gradient,weights,shape,Gradient.bias(dz))

def weighted(weights: Tensor(),result: Tensor() & (Tensor() & Tensor())) -> Parameters():
  (weight_gradient,(x,dz)) = result
  measured(weight_gradient,weights,dz,Tensors.shape(~F32,x))

def shaped(stride: U32,padding: U32,x: Tensor(),dz: Tensor(),measured: Tensor() & List<&2,U32>) -> Parameters():
  (weights,shape) = measured
  weighted(weights,Gradient.weight(x,dz,kernel(shape),stride,padding))

# dz = dy * activation'(z), then the gradients of the weights and the bias.
def parameters(dy: Tensor(),saved: Saved(),weights: Tensor(),settings: Settings) -> Parameters():
  (x,z) = saved
  Settings{stride,padding,+activation} = settings
  shaped(stride,padding,x,F.activate_backward(dy,z,activation),Tensors.shape(~F32,weights))

def finished(weight_gradient: Tensor(),bias_gradient: Tensor(),pair: Tensor() & Tensor()) -> Tensor() & (Tensor() & (Tensor() & Tensor())):
  (dx,weights) = pair
  (dx,(weight_gradient,(bias_gradient,weights)))

# The height and width of [N,C,H,W] or [C,H,W]; zero for another rank.
def image(shape: List<&2,U32>) -> U32 & U32:
  match shape:
    case [images,channels,height,width]: (height,width)
    case [channels,height,width]: (height,width)
    case shape: (0,0)

def sized(+stride: U32,+padding: U32,weight_gradient: Tensor(),bias_gradient: Tensor(),dz: Tensor(),weights: Tensor(),size: U32 & U32) -> Tensor() & (Tensor() & (Tensor() & Tensor())):
  (height,width) = size
  finished(weight_gradient,bias_gradient,Gradient.input(dz,weights,stride,padding,height,width))

def propagated(+stride: U32,+padding: U32,result: Parameters()) -> Tensor() & (Tensor() & (Tensor() & Tensor())):
  (weight_gradient,(bias_gradient,(dz,(weights,shape)))) = result
  sized(stride,padding,weight_gradient,bias_gradient,dz,weights,image(shape))

# dy, the gradient of the loss with respect to the layer output:
# (the gradient of the layer input, (weight gradient, (bias gradient, weights))).
def backward(dy: Tensor(),saved: Saved(),weights: Tensor(),+settings: Settings) -> Tensor() & (Tensor() & (Tensor() & Tensor())):
  Settings{+stride,+padding,activation} = settings
  propagated(stride,padding,parameters(dy,saved,weights,settings))

def without_input(result: Parameters()) -> Tensor() & (Tensor() & Tensor()):
  (weight_gradient,(bias_gradient,(dz,(weights,shape)))) = result
  (weight_gradient,(bias_gradient,weights))

# backward for a layer whose input needs no gradient:
# (weight gradient, (bias gradient, weights)).
def backward_parameters(dy: Tensor(),saved: Saved(),weights: Tensor(),settings: Settings) -> Tensor() & (Tensor() & Tensor()):
  without_input(parameters(dy,saved,weights,settings))
