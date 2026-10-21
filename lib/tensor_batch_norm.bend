# Batch normalization of [N,C,H,W] over the images and positions of each
# channel, as a training layer: forward normalizes with the statistics of the
# batch and moves the running statistics, backward gives the gradients,
# evaluate normalizes with the running statistics. Every step is a public
# operation in a fixed FP32 order; the sums are those of sum_axes.
import Base
import ./tensor.bend as Tensors
import ./tensor_f32.bend as F

def Tensor() -> Type:
  Tensors.Tensor<F32>

type Settings is Data:
  Settings{epsilon: F32,momentum: F32}

def settings(epsilon: F32,momentum: F32) -> Settings:
  Settings{epsilon,momentum}

# gamma and beta, each [C].
def Parameters() -> Type:
  Tensor() & Tensor()

# The running mean and the running variance, each [C].
def Running() -> Type:
  Tensor() & Tensor()

# The normalized input and 1 / sqrt(variance + epsilon). One backward call
# uses it.
def Saved() -> Type:
  Tensor() & Tensor()

def channels(shape: List<&2,U32>) -> U32:
  match shape:
    case [images,channels,height,width]: channels
    case shape: 0

# The number of values of each channel, N * H * W.
def count(shape: List<&2,U32>) -> U32:
  match shape:
    case [images,channels,height,width]: (images * (height * width : U32) : U32)
    case shape: 0

# A per-channel tensor [C] seen as [1,C,1,1], to broadcast over [N,C,H,W].
def spread(tensor: Tensor(),+channels: U32) -> Tensor():
  F.reshape(tensor,[1,channels,1,1])

# 1 / sqrt(variance + epsilon).
def inverse_deviation(variance: Tensor(),+channels: U32,+epsilon: F32) -> Tensor():
  F.div(F.ones([channels]),F.sqrt(F.add_scalar(variance,epsilon)))

# running * (1 - momentum) + statistic * momentum.
def moved(running: Tensor(),statistic: Tensor(),+momentum: F32) -> Tensor():
  F.add(F.mul_scalar(running,(1.0 - momentum : F32)),F.mul_scalar(statistic,momentum))

# xhat * gamma + beta, and the parameters back.
def affine(xhat: Tensor(),+channels: U32,gammas: Tensor() & Tensor(),betas: Tensor() & Tensor()) -> Tensor() & Parameters():
  (gamma,gamma_kept) = gammas
  (beta,beta_kept) = betas
  (F.add(F.mul(xhat,spread(gamma,channels)),spread(beta,channels)),(gamma_kept,beta_kept))

# ---- Training ----

# (the output, (what backward reads, (parameters, running statistics))).
def Forward() -> Type:
  Tensor() & (Saved() & (Parameters() & Running()))

def finished(xhat: Tensor(),deviation: Tensor(),running_mean: Tensor(),running_variance: Tensor(),result: Tensor() & Parameters()) -> Forward():
  (output,parameters) = result
  (output,((xhat,deviation),(parameters,(running_mean,running_variance))))

def normalized(+channels: U32,+momentum: F32,mean: Tensor(),unbiased: Tensor(),deviation: Tensor(),xhats: Tensor() & Tensor(),parameters: Parameters(),
  running: Running()) -> Forward():
  (xhat,xhat_kept) = xhats
  (gamma,beta) = parameters
  (running_mean,running_variance) = running
  finished(xhat_kept,deviation,moved(running_mean,mean,momentum),moved(running_variance,unbiased,momentum),affine(xhat,channels,F.copy(gamma),F.copy(beta)))

def scaled(+channels: U32,+momentum: F32,mean: Tensor(),difference: Tensor(),unbiased: Tensor(),deviations: Tensor() & Tensor(),parameters: Parameters(),
  running: Running()) -> Forward():
  (deviation,deviation_kept) = deviations
  normalized(channels,momentum,mean,unbiased,deviation_kept,F.copy(F.mul(difference,spread(deviation,channels))),parameters,running)

def deviated(+settings: Settings,+channels: U32,+values: U32,mean: Tensor(),difference: Tensor(),sums: Tensor() & Tensor(),parameters: Parameters(),
  running: Running()) -> Forward():
  Settings{+epsilon,+momentum} = settings
  (squares,squares_kept) = sums
  scaled(channels,momentum,mean,difference,F.div_scalar(squares_kept,U32.to_f32((values - 1 : U32))),
    F.copy(inverse_deviation(F.div_scalar(squares,U32.to_f32(values)),channels,epsilon)),parameters,running)

def squared(+settings: Settings,+channels: U32,+values: U32,mean: Tensor(),difference: Tensor(),pair: Tensor() & Tensor(),parameters: Parameters(),
  running: Running()) -> Forward():
  (left,right) = pair
  deviated(settings,channels,values,mean,difference,F.copy(F.sum_axes(F.mul(left,right),[0,2,3])),parameters,running)

def differenced(+settings: Settings,+channels: U32,+values: U32,mean: Tensor(),differences: Tensor() & Tensor(),parameters: Parameters(),
  running: Running()) -> Forward():
  (difference,difference_kept) = differences
  squared(settings,channels,values,mean,difference_kept,F.copy(difference),parameters,running)

def centered(+settings: Settings,+channels: U32,+values: U32,means: Tensor() & Tensor(),x: Tensor(),parameters: Parameters(),running: Running()) -> Forward():
  (mean,mean_kept) = means
  differenced(settings,channels,values,mean_kept,F.copy(F.sub(x,spread(mean,channels))),parameters,running)

def split(+settings: Settings,+channels: U32,+values: U32,xs: Tensor() & Tensor(),parameters: Parameters(),running: Running()) -> Forward():
  (x,x_kept) = xs
  centered(settings,channels,values,F.copy(F.div_scalar(F.sum_axes(x,[0,2,3]),U32.to_f32(values))),x_kept,parameters,running)

def shaped(+settings: Settings,measured: Tensor() & List<&2,U32>,parameters: Parameters(),running: Running()) -> Forward():
  (x,+shape) = measured
  split(settings,channels(shape),count(shape),F.copy(x),parameters,running)

# x [N,C,H,W], with M = N * H * W, mean = sum_axes(x,[0,2,3]) / M, d = x - mean
# and s = sum_axes(d * d,[0,2,3]): xhat = d * (1 / sqrt(s / M + epsilon)), the
# output xhat * gamma + beta, the running mean moved towards mean and the
# running variance towards s / (M - 1).
def forward(x: Tensor(),parameters: Parameters(),running: Running(),+settings: Settings) -> Forward():
  shaped(settings,F.shape(x),parameters,running)

# (dx, (dgamma, (dbeta, gamma))).
def Backward() -> Type:
  Tensor() & (Tensor() & (Tensor() & Tensor()))

def propagated(+channels: U32,+values: U32,dy: Tensor(),xhat: Tensor(),scale: Tensor(),dbetas: Tensor() & Tensor(),dgammas: Tensor() & Tensor(),
  gamma: Tensor()) -> Backward():
  (dbeta,dbeta_kept) = dbetas
  (dgamma,dgamma_kept) = dgammas
  +total = U32.to_f32(values)
  (F.mul(F.sub(F.sub(F.mul_scalar(dy,total),spread(dbeta,channels)),F.mul(xhat,spread(dgamma,channels))),spread(F.div_scalar(scale,total),channels)),
    (dgamma_kept,(dbeta_kept,gamma)))

def sums(+channels: U32,+values: U32,dys: Tensor() & Tensor(),dy: Tensor(),xhat: Tensor(),xhat_kept: Tensor(),scale: Tensor(),gamma: Tensor()) -> Backward():
  (for_beta,for_gamma) = dys
  propagated(channels,values,dy,xhat_kept,scale,F.copy(F.sum_axes(for_beta,[0,2,3])),F.copy(F.sum_axes(F.mul(for_gamma,xhat),[0,2,3])),gamma)

def copied(+channels: U32,+values: U32,dys: Tensor() & Tensor(),xhats: Tensor() & Tensor(),deviation: Tensor(),gammas: Tensor() & Tensor()) -> Backward():
  (dy,dy_kept) = dys
  (xhat,xhat_kept) = xhats
  (gamma,gamma_kept) = gammas
  sums(channels,values,F.copy(dy),dy_kept,xhat,xhat_kept,F.mul(gamma,deviation),gamma_kept)

def backward_shaped(measured: Tensor() & List<&2,U32>,saved: Saved(),gamma: Tensor()) -> Backward():
  (dy,+shape) = measured
  (xhat,deviation) = saved
  copied(channels(shape),count(shape),F.copy(dy),F.copy(xhat),deviation,F.copy(gamma))

# dy is the gradient of the loss with respect to the output. With
# dbeta = sum_axes(dy,[0,2,3]) and dgamma = sum_axes(dy * xhat,[0,2,3]):
# dx = ((dy * M - dbeta) - xhat * dgamma) * ((gamma * deviation) / M), where
# deviation is the saved 1 / sqrt(variance + epsilon).
def backward(dy: Tensor(),saved: Saved(),gamma: Tensor()) -> Backward():
  backward_shaped(F.shape(dy),saved,gamma)

# ---- Evaluation ----

# (the output, (parameters, running statistics)).
def Evaluated() -> Type:
  Tensor() & (Parameters() & Running())

def evaluation(mean: Tensor(),variance: Tensor(),result: Tensor() & Parameters()) -> Evaluated():
  (output,parameters) = result
  (output,(parameters,(mean,variance)))

def evaluated(+channels: U32,+epsilon: F32,x: Tensor(),parameters: Parameters(),means: Tensor() & Tensor(),variances: Tensor() & Tensor()) -> Evaluated():
  (gamma,beta) = parameters
  (mean,mean_kept) = means
  (variance,variance_kept) = variances
  evaluation(mean_kept,variance_kept,
    affine(F.mul(F.sub(x,spread(mean,channels)),spread(inverse_deviation(variance,channels,epsilon),channels)),channels,F.copy(gamma),F.copy(beta)))

def evaluate_shaped(+settings: Settings,measured: Tensor() & List<&2,U32>,parameters: Parameters(),running: Running()) -> Evaluated():
  Settings{+epsilon,momentum} = settings
  (x,+shape) = measured
  (running_mean,running_variance) = running
  evaluated(channels(shape),epsilon,x,parameters,F.copy(running_mean),F.copy(running_variance))

# The output with the running statistics:
# (x - running mean) * (1 / sqrt(running variance + epsilon)) * gamma + beta.
def evaluate(x: Tensor(),parameters: Parameters(),running: Running(),+settings: Settings) -> Evaluated():
  evaluate_shaped(settings,F.shape(x),parameters,running)
