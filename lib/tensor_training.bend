# Stelliferous training: convolution, batch norm, activation and pool layers.
# A layer is a value. forward returns the output, what backward reads and the
# layer, backward the input gradient and the layer holding its parameter
# gradients, evaluate the output in inference mode, and sgd_step the layer with
# its gradients applied. Every step is a public operation in a fixed FP32
# order, so results do not depend on the worker count.
import Base
import ./math_activation.bend as Activations
import ./tensor.bend as Tensors
import ./tensor_f32.bend as F
import ./tensor_layer.bend as ConvLayer
import ./tensor_batch_norm.bend as Norm
import ./tensor_loss.bend as Loss
import ./tensor_optimizer.bend as Optimizer

def Tensor() -> Type:
  Tensors.Tensor<F32>

# The gradient of one parameter: none before backward and after sgd_step.
type Gradient is Type:
  Empty{}
  Held{tensor: Tensors.Tensor<F32>}

# A convolution of weights [O,C,k,k] and bias [O]; a batch normalization of
# gamma, beta and the running statistics, each [C]; an activation; the global
# average pool of [N,C,H,W] to [N,C,1,1].
type Layer is Type:
  Convolution{weights: Tensors.Tensor<F32>,bias: Tensors.Tensor<F32>,stride: U32,padding: U32,weight_gradient: Gradient,bias_gradient: Gradient}
  BatchNorm{gamma: Tensors.Tensor<F32>,beta: Tensors.Tensor<F32>,running_mean: Tensors.Tensor<F32>,running_variance: Tensors.Tensor<F32>,
    epsilon: F32,momentum: F32,gamma_gradient: Gradient,beta_gradient: Gradient}
  Activation{activation: Activations.Activation}
  AveragePool{}

# What one backward call of a layer reads.
type Saved is Type:
  Convolved{saved: ConvLayer.Saved()}
  Normalized{saved: Norm.Saved()}
  Activated{input: Tensors.Tensor<F32>}
  Pooled{shape: List<&2,U32>}

# ---- Layers ----

def conv2d_layer(weights: Tensor(),bias: Tensor(),stride: U32,padding: U32) -> Layer:
  Convolution{weights,bias,stride,padding,Empty{},Empty{}}

# gamma 1, beta 0, running mean 0 and running variance 1, as in PyTorch.
def batch_norm_layer_with(+channels: U32,epsilon: F32,momentum: F32) -> Layer:
  BatchNorm{F.ones([channels]),F.zeros([channels]),F.zeros([channels]),F.ones([channels]),epsilon,momentum,Empty{},Empty{}}

# epsilon 1e-5 and momentum 0.1, as in PyTorch.
def batch_norm_layer(+channels: U32) -> Layer:
  batch_norm_layer_with(channels,0.00001,0.1)

# A batch normalization with the given parameters and running statistics.
def batch_norm_layer_from(gamma: Tensor(),beta: Tensor(),running_mean: Tensor(),running_variance: Tensor(),epsilon: F32,momentum: F32) -> Layer:
  BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,Empty{},Empty{}}

def activation_layer(activation: Activations.Activation) -> Layer:
  Activation{activation}

def average_pool_layer() -> Layer:
  AveragePool{}

# ---- Forward ----

def convolved(+stride: U32,+padding: U32,weight_gradient: Gradient,bias_gradient: Gradient,
  result: Tensor() & (ConvLayer.Saved() & (Tensor() & Tensor()))) -> Tensor() & (Saved & Layer):
  (output,(saved,(weights,bias))) = result
  (output,(Convolved{saved},Convolution{weights,bias,stride,padding,weight_gradient,bias_gradient}))

def normalized(+epsilon: F32,+momentum: F32,gamma_gradient: Gradient,beta_gradient: Gradient,result: Norm.Forward()) -> Tensor() & (Saved & Layer):
  (output,(saved,((gamma,beta),(running_mean,running_variance)))) = result
  (output,(Normalized{saved},BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient}))

def activated(+activation: Activations.Activation,pair: Tensor() & Tensor()) -> Tensor() & (Saved & Layer):
  (output,input) = pair
  (output,(Activated{input},Activation{activation}))

# [N,C,H,W] to [N,C,1,1]; another rank gives the extents unchanged.
def pooled_extents(shape: List<&2,U32>) -> List<&2,U32>:
  match shape:
    case [images,channels,height,width]: [images,channels,1,1]
    case shape: shape

def pooled_forward(measured: Tensor() & List<&2,U32>) -> Tensor() & (Saved & Layer):
  (x,+shape) = measured
  (F.reshape(F.mean_axes(x,[2,3]),pooled_extents(shape)),(Pooled{shape},AveragePool{}))

# (the output, (what backward reads, the layer)). A convolution keeps its
# input and output; a batch normalization normalizes with the statistics of
# the batch and moves its running statistics; the pool averages each channel
# of each image.
def forward(x: Tensor(),layer: Layer) -> Tensor() & (Saved & Layer):
  match layer:
    case Convolution{weights,bias,+stride,+padding,weight_gradient,bias_gradient}:
      convolved(stride,padding,weight_gradient,bias_gradient,ConvLayer.forward(x,weights,bias,ConvLayer.settings(stride,padding,Activations.identity())))
    case BatchNorm{gamma,beta,running_mean,running_variance,+epsilon,+momentum,gamma_gradient,beta_gradient}:
      normalized(epsilon,momentum,gamma_gradient,beta_gradient,Norm.forward(x,(gamma,beta),(running_mean,running_variance),Norm.settings(epsilon,momentum)))
    case Activation{+activation}: activated(activation,F.activate_forward(x,activation))
    case AveragePool{}: pooled_forward(Tensors.shape(~F32,x))

# ---- Backward ----

def convolution_back(+stride: U32,+padding: U32,bias: Tensor(),result: Tensor() & (Tensor() & (Tensor() & Tensor()))) -> Tensor() & Layer:
  (dx,(weight_gradient,(bias_gradient,weights))) = result
  (dx,Convolution{weights,bias,stride,padding,Held{weight_gradient},Held{bias_gradient}})

def norm_back(beta: Tensor(),running_mean: Tensor(),running_variance: Tensor(),+epsilon: F32,+momentum: F32,result: Norm.Backward()) -> Tensor() & Layer:
  (dx,(gamma_gradient,(beta_gradient,gamma))) = result
  (dx,BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,Held{gamma_gradient},Held{beta_gradient}})

# The average of H * W positions spreads its gradient over them.
def pooled_backward(dy: Tensor(),shape: List<&2,U32>) -> Tensor():
  match shape:
    case [images,channels,+height,+width]:
      F.broadcast_to(F.div_scalar(dy,U32.to_f32((height * width : U32))),[images,channels,height,width])
    case shape: Tensors.rejected(~F32,"backward",dy)

# A saved value of another kind of layer.
def unmatched(dy: Tensor(),saved: Saved,layer: Layer) -> Tensor() & Layer:
  (Tensors.rejected(~F32,"backward",dy),layer)

def convolution_backward(dy: Tensor(),saved: Saved,weights: Tensor(),bias: Tensor(),+stride: U32,+padding: U32,weight_gradient: Gradient,bias_gradient: Gradient) -> Tensor() & Layer:
  match saved:
    case Convolved{kept}:
      convolution_back(stride,padding,bias,ConvLayer.backward(dy,kept,weights,ConvLayer.settings(stride,padding,Activations.identity())))
    case saved: unmatched(dy,saved,Convolution{weights,bias,stride,padding,weight_gradient,bias_gradient})

def norm_backward(dy: Tensor(),saved: Saved,gamma: Tensor(),beta: Tensor(),running_mean: Tensor(),running_variance: Tensor(),+epsilon: F32,+momentum: F32,
  gamma_gradient: Gradient,beta_gradient: Gradient) -> Tensor() & Layer:
  match saved:
    case Normalized{kept}: norm_back(beta,running_mean,running_variance,epsilon,momentum,Norm.backward(dy,kept,gamma))
    case saved: unmatched(dy,saved,BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient})

def activation_backward(dy: Tensor(),saved: Saved,+activation: Activations.Activation) -> Tensor() & Layer:
  match saved:
    case Activated{input}: (F.activate_backward(dy,input,activation),Activation{activation})
    case saved: unmatched(dy,saved,Activation{activation})

def pool_backward(dy: Tensor(),saved: Saved) -> Tensor() & Layer:
  match saved:
    case Pooled{shape}: (pooled_backward(dy,shape),AveragePool{})
    case saved: unmatched(dy,saved,AveragePool{})

# dy, the gradient of the loss with respect to the layer output, and what
# forward saved: (the gradient of the layer input, the layer holding the
# gradients of its parameters). A saved value of another kind of layer gives
# a failed input gradient.
def backward(dy: Tensor(),saved: Saved,layer: Layer) -> Tensor() & Layer:
  match layer:
    case Convolution{weights,bias,+stride,+padding,weight_gradient,bias_gradient}:
      convolution_backward(dy,saved,weights,bias,stride,padding,weight_gradient,bias_gradient)
    case BatchNorm{gamma,beta,running_mean,running_variance,+epsilon,+momentum,gamma_gradient,beta_gradient}:
      norm_backward(dy,saved,gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient)
    case Activation{+activation}: activation_backward(dy,saved,activation)
    case AveragePool{}: pool_backward(dy,saved)

# ---- Evaluation ----

def convolution_evaluated(x: Tensor(),+stride: U32,+padding: U32,weight_gradient: Gradient,bias_gradient: Gradient,
  weight_copies: Tensor() & Tensor(),bias_copies: Tensor() & Tensor()) -> Tensor() & Layer:
  (weights,weights_kept) = weight_copies
  (bias,bias_kept) = bias_copies
  (F.conv2d(x,weights,bias,stride,padding,Activations.identity()),Convolution{weights_kept,bias_kept,stride,padding,weight_gradient,bias_gradient})

def norm_evaluated(+epsilon: F32,+momentum: F32,gamma_gradient: Gradient,beta_gradient: Gradient,result: Norm.Evaluated()) -> Tensor() & Layer:
  (output,((gamma,beta),(running_mean,running_variance))) = result
  (output,BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient})

def pooled_evaluated(measured: Tensor() & List<&2,U32>) -> Tensor() & Layer:
  (x,shape) = measured
  (F.reshape(F.mean_axes(x,[2,3]),pooled_extents(shape)),AveragePool{})

# The output in inference mode: a batch normalization uses its running
# statistics and leaves them unchanged.
def evaluate(x: Tensor(),layer: Layer) -> Tensor() & Layer:
  match layer:
    case Convolution{weights,bias,+stride,+padding,weight_gradient,bias_gradient}:
      convolution_evaluated(x,stride,padding,weight_gradient,bias_gradient,F.copy(weights),F.copy(bias))
    case BatchNorm{gamma,beta,running_mean,running_variance,+epsilon,+momentum,gamma_gradient,beta_gradient}:
      norm_evaluated(epsilon,momentum,gamma_gradient,beta_gradient,Norm.evaluate(x,(gamma,beta),(running_mean,running_variance),Norm.settings(epsilon,momentum)))
    case Activation{+activation}: (F.activate(x,activation),Activation{activation})
    case AveragePool{}: pooled_evaluated(Tensors.shape(~F32,x))

# ---- Updates ----

def stepped(parameter: Tensor(),gradient: Gradient,+rate: F32) -> Tensor():
  match gradient:
    case Empty{}: parameter
    case Held{tensor}: Optimizer.sgd(parameter,tensor,rate)

# Every parameter with a gradient becomes parameter - gradient * rate, and the
# gradients are cleared. A layer without gradients comes back unchanged.
def sgd_step(layer: Layer,+rate: F32) -> Layer:
  match layer:
    case Convolution{weights,bias,stride,padding,weight_gradient,bias_gradient}:
      Convolution{stepped(weights,weight_gradient,rate),stepped(bias,bias_gradient,rate),stride,padding,Empty{},Empty{}}
    case BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient}:
      BatchNorm{stepped(gamma,gamma_gradient,rate),stepped(beta,beta_gradient,rate),running_mean,running_variance,epsilon,momentum,Empty{},Empty{}}
    case Activation{activation}: Activation{activation}
    case AveragePool{}: AveragePool{}

# ---- Losses ----

# The mean squared error of prediction against a target of its shape:
# (the loss, (its gradient with respect to prediction, the target)).
def mse(prediction: Tensor(),target: Tensor()) -> Tensor() & (Tensor() & Tensor()):
  Loss.mse(prediction,target,Loss.mean())

# The mean cross entropy of logits [N,K] against a target [N,K] whose rows
# are class probabilities: (the loss, (its gradient with respect to logits,
# the target)).
def cross_entropy(logits: Tensor(),target: Tensor()) -> Tensor() & (Tensor() & Tensor()):
  Loss.cross_entropy(logits,target)

# The class numbers [N], each a float, as one-hot rows [N,classes].
def one_hot(labels: Tensor(),+classes: U32) -> Tensor():
  Loss.one_hot(labels,classes)
