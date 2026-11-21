# Stelliferous training: layers, residual networks, losses and SGD.
# The layers are convolution, batch norm, layer norm, activation and pool.
# A layer is a value. forward returns the output, what backward reads and the
# layer, backward the input gradient and the layer holding its parameter
# gradients, evaluate the output in inference mode, and sgd_step the layer with
# its gradients applied. A network runs layers and residual blocks in order
# through the same steps. Every step is a public operation in a fixed FP32
# order, so results do not depend on the worker count.
import Base
import ./math_activation.bend as Activations
import ./tensor_f32.bend as F
import ./tensor_tiled.bend as Tiled
import ./tensor_layer.bend as ConvLayer
import ./tensor_batch_norm.bend as Norm
import ./tensor_layer_norm.bend as LineNorm
import ./tensor_loss.bend as Loss
import ./tensor_optimizer.bend as Optimizer

def Tensor() -> Type:
  F.Tensor()

# The gradient of one parameter: none before backward and after sgd_step.
type Gradient is Type:
  Empty{}
  Held{tensor: Tensor()}

# A convolution of weights [O,C,k,k] and bias [O]; a batch normalization of
# gamma, beta and the running statistics, each [C]; a layer normalization of
# gamma and beta, each [D]; an activation; the global average pool of [N,C,H,W]
# to [N,C,1,1]. A parameter has its gradient and its velocity.
type Layer is Type:
  Convolution{weights: Tensor(),bias: Tensor(),stride: U32,padding: U32,weight_gradient: Gradient,bias_gradient: Gradient,
    weight_velocity: Optimizer.Velocity,bias_velocity: Optimizer.Velocity}
  BatchNorm{gamma: Tensor(),beta: Tensor(),running_mean: Tensor(),running_variance: Tensor(),
    epsilon: F32,momentum: F32,gamma_gradient: Gradient,beta_gradient: Gradient,
    gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity}
  LayerNorm{gamma: Tensor(),beta: Tensor(),epsilon: F32,gamma_gradient: Gradient,beta_gradient: Gradient,
    gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity}
  Activation{activation: Activations.Activation}
  AveragePool{}

# What one backward call of a layer reads.
type Saved is Type:
  Convolved{saved: ConvLayer.Saved()}
  Normalized{saved: Norm.Saved()}
  LineNormalized{saved: LineNorm.Saved()}
  Activated{input: Tensor()}
  Pooled{shape: List<&2,U32>}

# ---- Layers ----

def conv2d_layer(weights: Tensor(),bias: Tensor(),stride: U32,padding: U32) -> Layer:
  Convolution{Tiled.along(weights,0n),Tiled.along(bias,0n),stride,padding,Empty{},Empty{},Optimizer.Still{},Optimizer.Still{}}

# gamma 1, beta 0, running mean 0 and running variance 1, as in PyTorch.
def batch_norm_layer_with(+channels: U32,epsilon: F32,momentum: F32) -> Layer:
  BatchNorm{F.ones([channels]),F.zeros([channels]),F.zeros([channels]),F.ones([channels]),epsilon,momentum,Empty{},Empty{},Optimizer.Still{},Optimizer.Still{}}

# epsilon 1e-5 and momentum 0.1, as in PyTorch.
def batch_norm_layer(+channels: U32) -> Layer:
  batch_norm_layer_with(channels,0.00001,0.1)

# A batch normalization with the given parameters and running statistics.
def batch_norm_layer_from(gamma: Tensor(),beta: Tensor(),running_mean: Tensor(),running_variance: Tensor(),epsilon: F32,momentum: F32) -> Layer:
  BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,Empty{},Empty{},Optimizer.Still{},Optimizer.Still{}}

# gamma 1 and beta 0 for D features over the last axis, as in PyTorch.
def layer_norm_layer_with(+features: U32,epsilon: F32) -> Layer:
  LayerNorm{F.ones([features]),F.zeros([features]),epsilon,Empty{},Empty{},Optimizer.Still{},Optimizer.Still{}}

# epsilon 1e-5, as in PyTorch.
def layer_norm_layer(+features: U32) -> Layer:
  layer_norm_layer_with(features,0.00001)

# A layer normalization with the given parameters.
def layer_norm_layer_from(gamma: Tensor(),beta: Tensor(),epsilon: F32) -> Layer:
  LayerNorm{gamma,beta,epsilon,Empty{},Empty{},Optimizer.Still{},Optimizer.Still{}}

def activation_layer(activation: Activations.Activation) -> Layer:
  Activation{activation}

def average_pool_layer() -> Layer:
  AveragePool{}

# ---- Forward ----

def convolved(+stride: U32,+padding: U32,weight_gradient: Gradient,bias_gradient: Gradient,weight_velocity: Optimizer.Velocity,bias_velocity: Optimizer.Velocity,
  result: Tensor() & (ConvLayer.Saved() & (Tensor() & Tensor()))) -> Tensor() & (Saved & Layer):
  (output,(saved,(weights,bias))) = result
  (output,(Convolved{saved},Convolution{weights,bias,stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}))

def normalized(+epsilon: F32,+momentum: F32,gamma_gradient: Gradient,beta_gradient: Gradient,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity,result: Norm.Forward()) -> Tensor() & (Saved & Layer):
  (output,(saved,((gamma,beta),(running_mean,running_variance)))) = result
  (output,(Normalized{saved},BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}))

def line_normalized(+epsilon: F32,gamma_gradient: Gradient,beta_gradient: Gradient,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity,result: LineNorm.Forward()) -> Tensor() & (Saved & Layer):
  (output,(saved,(gamma,beta))) = result
  (output,(LineNormalized{saved},LayerNorm{gamma,beta,epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}))

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
# the batch and moves its running statistics; a layer normalization normalizes
# each line of the last axis; the pool averages each channel of each image.
def forward(x: Tensor(),layer: Layer) -> Tensor() & (Saved & Layer):
  match layer:
    case Convolution{weights,bias,+stride,+padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      convolved(stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity,ConvLayer.forward(x,weights,bias,ConvLayer.settings(stride,padding,Activations.identity())))
    case BatchNorm{gamma,beta,running_mean,running_variance,+epsilon,+momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      normalized(epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Norm.forward(x,(gamma,beta),(running_mean,running_variance),Norm.settings(epsilon,momentum)))
    case LayerNorm{gamma,beta,+epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      line_normalized(epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,LineNorm.forward(x,(gamma,beta),epsilon))
    case Activation{+activation}: activated(activation,F.activate_forward(x,activation))
    case AveragePool{}: pooled_forward(F.shape(x))

# ---- Backward ----

def convolution_back(+stride: U32,+padding: U32,bias: Tensor(),weight_velocity: Optimizer.Velocity,bias_velocity: Optimizer.Velocity,result: Tensor() & (Tensor() & (Tensor() & Tensor()))) -> Tensor() & Layer:
  (dx,(weight_gradient,(bias_gradient,weights))) = result
  (dx,Convolution{Tiled.along(weights,0n),bias,stride,padding,Held{weight_gradient},Held{bias_gradient},weight_velocity,bias_velocity})

def norm_back(beta: Tensor(),running_mean: Tensor(),running_variance: Tensor(),+epsilon: F32,+momentum: F32,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity,result: Norm.Backward()) -> Tensor() & Layer:
  (dx,(gamma_gradient,(beta_gradient,gamma))) = result
  (dx,BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,Held{gamma_gradient},Held{beta_gradient},gamma_velocity,beta_velocity})

def line_norm_back(beta: Tensor(),+epsilon: F32,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity,result: LineNorm.Backward()) -> Tensor() & Layer:
  (dx,(gamma_gradient,(beta_gradient,gamma))) = result
  (dx,LayerNorm{gamma,beta,epsilon,Held{gamma_gradient},Held{beta_gradient},gamma_velocity,beta_velocity})

# The average of H * W positions spreads its gradient over them.
def pooled_backward(dy: Tensor(),shape: List<&2,U32>) -> Tensor():
  match shape:
    case [images,channels,+height,+width]:
      F.broadcast_to(F.div_scalar(dy,U32.to_f32((height * width : U32))),[images,channels,height,width])
    case shape: Tiled.rejected("backward",dy)

# A saved value of another kind of layer.
def unmatched(dy: Tensor(),saved: Saved,layer: Layer) -> Tensor() & Layer:
  (Tiled.rejected("backward",dy),layer)

def convolution_backward(dy: Tensor(),saved: Saved,weights: Tensor(),bias: Tensor(),+stride: U32,+padding: U32,weight_gradient: Gradient,bias_gradient: Gradient,weight_velocity: Optimizer.Velocity,bias_velocity: Optimizer.Velocity) -> Tensor() & Layer:
  match saved:
    case Convolved{kept}:
      convolution_back(stride,padding,bias,weight_velocity,bias_velocity,ConvLayer.backward(dy,kept,weights,ConvLayer.settings(stride,padding,Activations.identity())))
    case saved: unmatched(dy,saved,Convolution{weights,bias,stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity})

def norm_backward(dy: Tensor(),saved: Saved,gamma: Tensor(),beta: Tensor(),running_mean: Tensor(),running_variance: Tensor(),+epsilon: F32,+momentum: F32,
  gamma_gradient: Gradient,beta_gradient: Gradient,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity) -> Tensor() & Layer:
  match saved:
    case Normalized{kept}: norm_back(beta,running_mean,running_variance,epsilon,momentum,gamma_velocity,beta_velocity,Norm.backward(dy,kept,gamma))
    case saved: unmatched(dy,saved,BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity})

def line_norm_backward(dy: Tensor(),saved: Saved,gamma: Tensor(),beta: Tensor(),+epsilon: F32,gamma_gradient: Gradient,beta_gradient: Gradient,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity) -> Tensor() & Layer:
  match saved:
    case LineNormalized{kept}: line_norm_back(beta,epsilon,gamma_velocity,beta_velocity,LineNorm.backward(dy,kept,gamma))
    case saved: unmatched(dy,saved,LayerNorm{gamma,beta,epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity})

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
    case Convolution{weights,bias,+stride,+padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      convolution_backward(dy,saved,weights,bias,stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity)
    case BatchNorm{gamma,beta,running_mean,running_variance,+epsilon,+momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      norm_backward(dy,saved,gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity)
    case LayerNorm{gamma,beta,+epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      line_norm_backward(dy,saved,gamma,beta,epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity)
    case Activation{+activation}: activation_backward(dy,saved,activation)
    case AveragePool{}: pool_backward(dy,saved)

# ---- Evaluation ----

def convolution_evaluated(x: Tensor(),+stride: U32,+padding: U32,weight_gradient: Gradient,bias_gradient: Gradient,weight_velocity: Optimizer.Velocity,bias_velocity: Optimizer.Velocity,
  weight_copies: Tensor() & Tensor(),bias_copies: Tensor() & Tensor()) -> Tensor() & Layer:
  (weights,weights_kept) = weight_copies
  (bias,bias_kept) = bias_copies
  (F.conv2d(x,weights,bias,stride,padding,Activations.identity()),Convolution{weights_kept,bias_kept,stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity})

def norm_evaluated(+epsilon: F32,+momentum: F32,gamma_gradient: Gradient,beta_gradient: Gradient,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity,result: Norm.Evaluated()) -> Tensor() & Layer:
  (output,((gamma,beta),(running_mean,running_variance))) = result
  (output,BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity})

def line_norm_evaluated(+epsilon: F32,gamma_gradient: Gradient,beta_gradient: Gradient,gamma_velocity: Optimizer.Velocity,beta_velocity: Optimizer.Velocity,result: LineNorm.Evaluated()) -> Tensor() & Layer:
  (output,(gamma,beta)) = result
  (output,LayerNorm{gamma,beta,epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity})

def pooled_evaluated(measured: Tensor() & List<&2,U32>) -> Tensor() & Layer:
  (x,shape) = measured
  (F.reshape(F.mean_axes(x,[2,3]),pooled_extents(shape)),AveragePool{})

# The output in inference mode: a batch normalization uses its running
# statistics and leaves them unchanged; a layer normalization has none.
def evaluate(x: Tensor(),layer: Layer) -> Tensor() & Layer:
  match layer:
    case Convolution{weights,bias,+stride,+padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      convolution_evaluated(x,stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity,F.copy(weights),F.copy(bias))
    case BatchNorm{gamma,beta,running_mean,running_variance,+epsilon,+momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      norm_evaluated(epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,Norm.evaluate(x,(gamma,beta),(running_mean,running_variance),Norm.settings(epsilon,momentum)))
    case LayerNorm{gamma,beta,+epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      line_norm_evaluated(epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity,LineNorm.evaluate(x,(gamma,beta),epsilon))
    case Activation{+activation}: (F.activate(x,activation),Activation{activation})
    case AveragePool{}: pooled_evaluated(F.shape(x))

# ---- Updates ----

# What one step does to every parameter that has a gradient.
type Descent is Data:
  Descent{rate: F32,momentum: F32,decay: F32}

def descended(parameter: Tensor(),tensor: Tensor(),velocity: Optimizer.Velocity,+descent: Descent) -> Tensor() & Optimizer.Velocity:
  Descent{+rate,+momentum,+decay} = descent
  Optimizer.sgd_with(parameter,tensor,velocity,rate,momentum,decay)

def stepped(parameter: Tensor(),gradient: Gradient,velocity: Optimizer.Velocity,+descent: Descent) -> Tensor() & Optimizer.Velocity:
  match gradient:
    case Empty{}: (parameter,velocity)
    case Held{tensor}: descended(parameter,tensor,velocity,descent)

def convolution_stepped(+stride: U32,+padding: U32,weights: Tensor() & Optimizer.Velocity,bias: Tensor() & Optimizer.Velocity) -> Layer:
  (weights,weight_velocity) = weights
  (bias,bias_velocity) = bias
  Convolution{weights,bias,stride,padding,Empty{},Empty{},weight_velocity,bias_velocity}

def norm_stepped(running_mean: Tensor(),running_variance: Tensor(),+epsilon: F32,+momentum: F32,gamma: Tensor() & Optimizer.Velocity,beta: Tensor() & Optimizer.Velocity) -> Layer:
  (gamma,gamma_velocity) = gamma
  (beta,beta_velocity) = beta
  BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,Empty{},Empty{},gamma_velocity,beta_velocity}

def line_norm_stepped(+epsilon: F32,gamma: Tensor() & Optimizer.Velocity,beta: Tensor() & Optimizer.Velocity) -> Layer:
  (gamma,gamma_velocity) = gamma
  (beta,beta_velocity) = beta
  LayerNorm{gamma,beta,epsilon,Empty{},Empty{},gamma_velocity,beta_velocity}

def layer_stepped(layer: Layer,+descent: Descent) -> Layer:
  match layer:
    case Convolution{weights,bias,+stride,+padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      convolution_stepped(stride,padding,stepped(weights,weight_gradient,weight_velocity,descent),stepped(bias,bias_gradient,bias_velocity,descent))
    case BatchNorm{gamma,beta,running_mean,running_variance,+epsilon,+norm_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      norm_stepped(running_mean,running_variance,epsilon,norm_momentum,stepped(gamma,gamma_gradient,gamma_velocity,descent),stepped(beta,beta_gradient,beta_velocity,descent))
    case LayerNorm{gamma,beta,+epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      line_norm_stepped(epsilon,stepped(gamma,gamma_gradient,gamma_velocity,descent),stepped(beta,beta_gradient,beta_velocity,descent))
    case Activation{activation}: Activation{activation}
    case AveragePool{}: AveragePool{}

# Every parameter with a gradient takes one step of Optimizer.sgd_with, and the
# gradients are cleared. The layer keeps the velocities for the next step. A
# layer without gradients comes back unchanged.
def sgd_step_with(layer: Layer,rate: F32,momentum: F32,weight_decay: F32) -> Layer:
  layer_stepped(layer,Descent{rate,momentum,weight_decay})

# Without momentum and weight decay: every parameter with a gradient becomes
# parameter - gradient * rate.
def sgd_step(layer: Layer,rate: F32) -> Layer:
  sgd_step_with(layer,rate,0.0,0.0)

# ---- Networks ----

# Layers in order. A residual block adds the output of its branch to the
# output of its shortcut, both on the block's input, and gives the sum to
# rest; an empty shortcut passes the input on.
type Network is Type:
  End{}
  Then{layer: Layer,rest: Network}
  Residual{branch: Network,shortcut: Network,rest: Network}

# What backward_network reads, the last layer first.
type Tape is Type:
  Blank{}
  Recorded{saved: Saved,layer: Layer,rest: Tape}
  Joined{branch: Tape,shortcut: Tape,rest: Tape}

def split(~Result: Type,copies: Tensor() & Tensor(),next: Tensor() -> Tensor() -> Result) -> Result:
  (first,second) = copies
  next(first,second)

def recorded(result: Tensor() & (Saved & Layer),next: Tensor() -> Saved -> Layer -> Tensor() & Tape) -> Tensor() & Tape:
  (y,(saved,layer)) = result
  next(y,saved,layer)

def joined(branch: Tensor() & Tape,shortcut: Tensor() & Tape,next: Tensor() -> Tensor() -> Tape -> Tape -> Tensor() & Tape) -> Tensor() & Tape:
  (left,branch_tape) = branch
  (right,shortcut_tape) = shortcut
  next(left,right,branch_tape,shortcut_tape)

def network_forward(network: Network,x: Tensor(),tape: Tape) -> Tensor() & Tape:
  match network:
    case End{}: (x,tape)
    case Then{layer,rest}: recorded(forward(x,layer),y => saved => layer => network_forward(rest,y,Recorded{saved,layer,tape}))
    case Residual{branch,shortcut,rest}:
      split(~(Tensor() & Tape),F.copy(x),first => second =>
        joined(network_forward(branch,first,Blank{}),network_forward(shortcut,second,Blank{}),
          left => right => branch_tape => shortcut_tape => network_forward(rest,F.add(left,right),Joined{branch_tape,shortcut_tape,tape})))

# (the output, what backward_network reads). Batch normalizations use the
# statistics of the batch and move their running statistics.
def forward_network(x: Tensor(),network: Network) -> Tensor() & Tape:
  network_forward(network,x,Blank{})

def returned(result: Tensor() & Layer,next: Tensor() -> Layer -> Tensor() & Network) -> Tensor() & Network:
  (dx,layer) = result
  next(dx,layer)

def met(branch: Tensor() & Network,shortcut: Tensor() & Network,next: Tensor() -> Tensor() -> Network -> Network -> Tensor() & Network) -> Tensor() & Network:
  (left,branch_network) = branch
  (right,shortcut_network) = shortcut
  next(left,right,branch_network,shortcut_network)

def network_backward(tape: Tape,dy: Tensor(),network: Network) -> Tensor() & Network:
  match tape:
    case Blank{}: (dy,network)
    case Recorded{saved,layer,rest}: returned(backward(dy,saved,layer),dx => layer => network_backward(rest,dx,Then{layer,network}))
    case Joined{branch,shortcut,rest}:
      split(~(Tensor() & Network),F.copy(dy),first => second =>
        met(network_backward(branch,first,End{}),network_backward(shortcut,second,End{}),
          left => right => branch_network => shortcut_network => network_backward(rest,F.add(left,right),Residual{branch_network,shortcut_network,network})))

# dy, the gradient of the loss with respect to the network output, and what
# forward_network saved: (the gradient of the network input, the network whose
# layers hold the gradients of their parameters). A residual block gives its
# input the sum of the gradients of its branch and of its shortcut.
def backward_network(dy: Tensor(),tape: Tape) -> Tensor() & Network:
  network_backward(tape,dy,End{})

def in_front(layer: Layer,result: Tensor() & Network) -> Tensor() & Network:
  (y,rest) = result
  (y,Then{layer,rest})

def block_in_front(branch: Network,shortcut: Network,result: Tensor() & Network) -> Tensor() & Network:
  (y,rest) = result
  (y,Residual{branch,shortcut,rest})

def network_evaluated(network: Network,x: Tensor()) -> Tensor() & Network:
  match network:
    case End{}: (x,End{})
    case Then{layer,rest}: returned(evaluate(x,layer),y => layer => in_front(layer,network_evaluated(rest,y)))
    case Residual{branch,shortcut,rest}:
      split(~(Tensor() & Network),F.copy(x),first => second =>
        met(network_evaluated(branch,first),network_evaluated(shortcut,second),
          left => right => branch_network => shortcut_network => block_in_front(branch_network,shortcut_network,network_evaluated(rest,F.add(left,right)))))

# The output in inference mode and the network: batch normalizations use
# their running statistics and leave them unchanged.
def evaluate_network(x: Tensor(),network: Network) -> Tensor() & Network:
  network_evaluated(network,x)

def network_stepped(network: Network,+descent: Descent) -> Network:
  match network:
    case End{}: End{}
    case Then{layer,rest}: Then{layer_stepped(layer,descent),network_stepped(rest,descent)}
    case Residual{branch,shortcut,rest}: Residual{network_stepped(branch,descent),network_stepped(shortcut,descent),network_stepped(rest,descent)}

# sgd_step_with on every layer, in order.
def sgd_network_with(network: Network,rate: F32,momentum: F32,weight_decay: F32) -> Network:
  network_stepped(network,Descent{rate,momentum,weight_decay})

def sgd_network(network: Network,rate: F32) -> Network:
  sgd_network_with(network,rate,0.0,0.0)

# ---- Checkpoints ----

# A parameter at prefix.bin and its velocity: prefix_moving.bin holds 1 when
# prefix_velocity.bin holds it, 0 before the first step with momentum.
def velocity_saved(+prefix: String,velocity: Optimizer.Velocity) -> IO(Optimizer.Velocity):
  match velocity:
    case Optimizer.Still{}:
      do IO<Optimizer.Velocity>:
        moving : Tensor() <- F.save_raw(prefix ++ "_moving.bin",F.zeros([1]))
        IO.pure(Optimizer.Velocity,Optimizer.Still{})
    case Optimizer.Moving{tensor,rest}:
      do IO<Optimizer.Velocity>:
        moving : Tensor() <- F.save_raw(prefix ++ "_moving.bin",F.ones([1]))
        tensor : Tensor() <- F.save_raw(prefix ++ "_velocity.bin",tensor)
        IO.pure(Optimizer.Velocity,Optimizer.Moving{tensor,rest})

def parameter_saved(+prefix: String,parameter: Tensor(),velocity: Optimizer.Velocity) -> IO(Tensor() & Optimizer.Velocity):
  do IO<Tensor() & Optimizer.Velocity>:
    parameter : Tensor() <- F.save_raw(prefix ++ ".bin",parameter)
    velocity : Optimizer.Velocity <- velocity_saved(prefix,velocity)
    IO.pure(Tensor() & Optimizer.Velocity,(parameter,velocity))

def layer_saved(+prefix: String,layer: Layer) -> IO(Layer):
  match layer:
    case Convolution{weights,bias,+stride,+padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      do IO<Layer>:
        weights : Tensor() & Optimizer.Velocity <- parameter_saved(prefix ++ "_weights",weights,weight_velocity)
        bias : Tensor() & Optimizer.Velocity <- parameter_saved(prefix ++ "_bias",bias,bias_velocity)
        IO.pure(Layer,convolution_stepped(stride,padding,weights,bias))
    case BatchNorm{gamma,beta,running_mean,running_variance,+epsilon,+norm_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      do IO<Layer>:
        gamma : Tensor() & Optimizer.Velocity <- parameter_saved(prefix ++ "_gamma",gamma,gamma_velocity)
        beta : Tensor() & Optimizer.Velocity <- parameter_saved(prefix ++ "_beta",beta,beta_velocity)
        running_mean : Tensor() <- F.save_raw(prefix ++ "_mean.bin",running_mean)
        running_variance : Tensor() <- F.save_raw(prefix ++ "_variance.bin",running_variance)
        IO.pure(Layer,norm_stepped(running_mean,running_variance,epsilon,norm_momentum,gamma,beta))
    case LayerNorm{gamma,beta,+epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      do IO<Layer>:
        gamma : Tensor() & Optimizer.Velocity <- parameter_saved(prefix ++ "_gamma",gamma,gamma_velocity)
        beta : Tensor() & Optimizer.Velocity <- parameter_saved(prefix ++ "_beta",beta,beta_velocity)
        IO.pure(Layer,line_norm_stepped(epsilon,gamma,beta))
    case Activation{activation}: IO.pure(Layer,Activation{activation})
    case AveragePool{}: IO.pure(Layer,AveragePool{})

def loaded_on(+path: String,+shape: List<&2,U32>,+workers: U32) -> IO(Tensor()):
  do IO<Tensor()>:
    tensor : Tensor() <- F.load_raw(path,shape)
    IO.pure(Tensor(),F.with_workers(tensor,workers))

def velocity_loaded(moving: Bool,+path: String,+shape: List<&2,U32>,+workers: U32) -> IO(Optimizer.Velocity):
  match moving:
    case False{}: IO.pure(Optimizer.Velocity,Optimizer.Still{})
    case True{}:
      do IO<Optimizer.Velocity>:
        tensor : Tensor() <- loaded_on(path,shape,workers)
        IO.pure(Optimizer.Velocity,Optimizer.Moving{tensor,Optimizer.Still{}})

def parameter_loaded(+prefix: String,+shape: List<&2,U32>,counted: Tensor() & U32) -> IO(Tensor() & Optimizer.Velocity):
  (replaced,+workers) = counted
  do IO<Tensor() & Optimizer.Velocity>:
    parameter : Tensor() <- loaded_on(prefix ++ ".bin",shape,workers)
    moving : Tensor() <- F.load_raw(prefix ++ "_moving.bin",[1])
    velocity : Optimizer.Velocity <- velocity_loaded(F32.is_eq(F.item_or(moving,0.0),1.0),prefix ++ "_velocity.bin",shape,workers)
    IO.pure(Tensor() & Optimizer.Velocity,(parameter,velocity))

# The parameter saved under prefix, in the shape and on the workers of the
# one it replaces.
def parameter_restored(+prefix: String,measured: Tensor() & List<&2,U32>) -> IO(Tensor() & Optimizer.Velocity):
  (parameter,+shape) = measured
  parameter_loaded(prefix,shape,F.workers(parameter))

def statistic_loaded(+path: String,+shape: List<&2,U32>,counted: Tensor() & U32) -> IO(Tensor()):
  (replaced,+workers) = counted
  loaded_on(path,shape,workers)

def statistic_restored(+path: String,measured: Tensor() & List<&2,U32>) -> IO(Tensor()):
  (statistic,+shape) = measured
  statistic_loaded(path,shape,F.workers(statistic))

def aligned(parameter: Tensor() & Optimizer.Velocity) -> Tensor() & Optimizer.Velocity:
  (tensor,velocity) = parameter
  (Tiled.along(tensor,0n),velocity)

def layer_restored(+prefix: String,layer: Layer) -> IO(Layer):
  match layer:
    case Convolution{weights,bias,+stride,+padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      do IO<Layer>:
        weights : Tensor() & Optimizer.Velocity <- parameter_restored(prefix ++ "_weights",F.shape(weights))
        bias : Tensor() & Optimizer.Velocity <- parameter_restored(prefix ++ "_bias",F.shape(bias))
        IO.pure(Layer,convolution_stepped(stride,padding,aligned(weights),aligned(bias)))
    case BatchNorm{gamma,beta,running_mean,running_variance,+epsilon,+norm_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      do IO<Layer>:
        gamma : Tensor() & Optimizer.Velocity <- parameter_restored(prefix ++ "_gamma",F.shape(gamma))
        beta : Tensor() & Optimizer.Velocity <- parameter_restored(prefix ++ "_beta",F.shape(beta))
        running_mean : Tensor() <- statistic_restored(prefix ++ "_mean.bin",F.shape(running_mean))
        running_variance : Tensor() <- statistic_restored(prefix ++ "_variance.bin",F.shape(running_variance))
        IO.pure(Layer,norm_stepped(running_mean,running_variance,epsilon,norm_momentum,gamma,beta))
    case LayerNorm{gamma,beta,+epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      do IO<Layer>:
        gamma : Tensor() & Optimizer.Velocity <- parameter_restored(prefix ++ "_gamma",F.shape(gamma))
        beta : Tensor() & Optimizer.Velocity <- parameter_restored(prefix ++ "_beta",F.shape(beta))
        IO.pure(Layer,line_norm_stepped(epsilon,gamma,beta))
    case Activation{activation}: IO.pure(Layer,Activation{activation})
    case AveragePool{}: IO.pure(Layer,AveragePool{})

# The layers in order, each under its number.
type Numbered is Type:
  Numbered{next: U32,network: Network}

def numbered(+next: U32,network: Network) -> IO(Numbered):
  IO.pure(Numbered,Numbered{next,network})

def numbered_then(result: Numbered,next: U32 -> Network -> IO(Numbered)) -> IO(Numbered):
  Numbered{+after,network} = result
  next(after,network)

def visited(saving: Bool,+prefix: String,layer: Layer) -> IO(Layer):
  match saving:
    case True{}: layer_saved(prefix,layer)
    case False{}: layer_restored(prefix,layer)

def walked(network: Network,+folder: String,+first: U32,+saving: Bool) -> IO(Numbered):
  match network:
    case End{}: numbered(first,End{})
    case Then{layer,rest}:
      do IO<Numbered>:
        layer : Layer <- visited(saving,folder ++ "layer" ++ U32.show(first),layer)
        after : Numbered <- walked(rest,folder,(first + 1 : U32),saving)
        numbered_then(after,next => rest => numbered(next,Then{layer,rest}))
    case Residual{branch,shortcut,rest}:
      IO.bind(Numbered,Numbered,walked(branch,folder,first,saving),branch => numbered_then(branch,after_branch => branch =>
        IO.bind(Numbered,Numbered,walked(shortcut,folder,after_branch,saving),shortcut => numbered_then(shortcut,after_shortcut => shortcut =>
          IO.bind(Numbered,Numbered,walked(rest,folder,after_shortcut,saving),rest => numbered_then(rest,next => rest =>
            numbered(next,Residual{branch,shortcut,rest})))))))

def walked_network(result: Numbered) -> IO(Network):
  Numbered{+next,network} = result
  IO.pure(Network,network)

# The parameters, running statistics and velocities of every layer in order,
# as raw FP32 files folder/layer0_weights.bin and so on. Gradients are not
# saved and the network comes back without them. A tensor that cannot be
# written is failed in the network.
def save_network(+folder: String,network: Network) -> IO(Network):
  IO.bind(Numbered,Network,walked(network,folder,0,True{}),walked_network)

# The network that save_network wrote into folder, read in the shapes and on
# the workers of the layers of network, which has the same layers.
def restore_network(+folder: String,network: Network) -> IO(Network):
  IO.bind(Numbered,Network,walked(network,folder,0,False{}),walked_network)

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
