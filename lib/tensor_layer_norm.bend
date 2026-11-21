# Layer normalization over the last axis, composed of public operations in a
# fixed FP32 order. Gamma and beta have the extent of that axis. Parameters
# return, and backward reads xhat and the deviation that forward saved. A
# failed input, or gamma of another extent, fails every result computed.
import Base
import ./tensor.bend as Tensors
import ./tensor_f32.bend as F
import ./tensor_layout.bend as Layout
import ./tensor_shape.bend as Shapes
import ./tensor_tiled.bend as Tiled
import ./tensor_tiled_map.bend as TiledMap

def Tensor() -> Type:
  F.Tensor()

def Saved() -> Type:
  Tensor() & Tensor()

def Parameters() -> Type:
  Tensor() & Tensor()

def Forward() -> Type:
  Tensor() & (Saved() & Parameters())

def Backward() -> Type:
  Tensor() & (Tensor() & (Tensor() & Tensor()))

def Evaluated() -> Type:
  Tensor() & Parameters()

# 1 / sqrt(variance + epsilon).
def inverse_deviation(epsilon: F32,variance: F32) -> F32:
  (1.0 / F32.sqrt((variance + epsilon : F32)) : F32)

def square(value: F32) -> F32:
  +value = value
  (value * value : F32)

# The extents of a shape with only the last one left, and the axes before it.
def last_of(shape: List<&2,U32>) -> List<&2,U32>:
  match shape:
    case []: []
    case [extent]: [extent]
    case extent <> rest: last_of(rest)

def before_last(remaining: Nat,+axis: U32) -> List<&2,U32>:
  match remaining:
    case 0n: []
    case 1n: []
    case 2n+rest: axis <> before_last(1n+rest,(axis + 1 : U32))

def before(count: Nat) -> Nat:
  match count:
    case 0n: 0n
    case 1n+rest: rest

def last_axis(+shape: List<&2,U32>) -> U32:
  U32.from_nat(before(List.length(&2,U32,shape)))

def unit_extents(+shape: List<&2,U32>) -> List<&2,U32>:
  Layout.with_extent(shape,before(List.length(&2,U32,shape)),1)

# The mean along the last axis, with that axis of extent 1.
def line_mean(tensor: Tensor(),+shape: List<&2,U32>) -> Tensor():
  F.reshape(F.mean_axes(tensor,[last_axis(shape)]),unit_extents(shape))

def deviation_of(+epsilon: F32,+shape: List<&2,U32>,difference: Tensor()) -> Tensor():
  F.tiled_map(~F32,~TiledMap.lanes(~F32,~inverse_deviation),epsilon,1.25,line_mean(F.tiled_lanes(~square,1.0,difference),shape))

def scaled_by(difference: Tensor(),deviations: Tensor() & Tensor()) -> Saved():
  (deviation,kept) = deviations
  (F.mul(difference,deviation),kept)

def scaled(+epsilon: F32,+shape: List<&2,U32>,copies: Tensor() & Tensor()) -> Saved():
  (difference,for_variance) = copies
  scaled_by(difference,F.copy(deviation_of(epsilon,shape,for_variance)))

def centered(+epsilon: F32,+shape: List<&2,U32>,copies: Tensor() & Tensor()) -> Saved():
  (x,for_mean) = copies
  scaled(epsilon,shape,F.copy(F.sub(x,line_mean(for_mean,shape))))

# (xhat, deviation) over each line of the last axis: xhat = (x - mean) *
# deviation, with deviation = 1 / sqrt(mean((x - mean)^2) + epsilon) in the
# shape of x with the last extent 1.
def normalized(x: Tensor(),+epsilon: F32,+shape: List<&2,U32>) -> Saved():
  centered(epsilon,shape,F.copy(x))

def affine(xhat: Tensor(),gamma: Tensor(),beta: Tensor()) -> Tensor():
  F.add(F.mul(xhat,gamma),beta)

# ---- Forward ----

def forward_kept(xhats: Tensor() & Tensor(),deviation: Tensor(),gammas: Tensor() & Tensor(),betas: Tensor() & Tensor()) -> Forward():
  (xhat,xhat_kept) = xhats
  (gamma,gamma_kept) = gammas
  (beta,beta_kept) = betas
  (affine(xhat,gamma,beta),((xhat_kept,deviation),(gamma_kept,beta_kept)))

def forward_saved(saved: Saved(),gamma: Tensor(),beta: Tensor()) -> Forward():
  (xhat,deviation) = saved
  forward_kept(F.copy(xhat),deviation,F.copy(gamma),F.copy(beta))

# A failed forward fails every result it computes and returns gamma and beta.
def forward_failed(+problem: Tensors.Problem,+shape: List<&2,U32>,+workers: U32,parameters: Parameters()) -> Forward():
  (Tiled.failed_as(problem,shape,workers),((Tiled.failed_as(problem,shape,workers),Tiled.failed_as(problem,unit_extents(shape),workers)),parameters))

# The problem of a gamma that is not one per element of the last axis.
def mismatched(+shape: List<&2,U32>) -> Tensors.Problem:
  Tensors.Problem{"layer_norm",Tensors.InvalidArgument{shape}}

def forward_when(same: Bool,x: Tensor(),+shape: List<&2,U32>,+workers: U32,gamma: Tensor(),beta: Tensor(),+epsilon: F32) -> Forward():
  match same:
    case True{}: forward_saved(normalized(x,epsilon,shape),gamma,beta)
    case False{}: forward_failed(mismatched(shape),shape,workers,(gamma,beta))

# One gamma per element of the last axis of x.
def matches(+shape: List<&2,U32>,+gamma_shape: List<&2,U32>) -> Bool:
  match gamma_shape:
    case [features]: Shapes.same_extents(last_of(shape),[features])
    case gamma_shape: False{}

def forward_shapes(x: Tensor(),+shape: List<&2,U32>,+workers: U32,gamma_measured: Tensor() & List<&2,U32>,beta: Tensor(),+epsilon: F32) -> Forward():
  (gamma,gamma_shape) = gamma_measured
  forward_when(matches(shape,gamma_shape),x,shape,workers,gamma,beta,epsilon)

def forward_status(+status: Tensors.Status,+shape: List<&2,U32>,+workers: U32,+epsilon: F32,x: Tensor(),parameters: Parameters()) -> Forward():
  match status:
    case Tensors.Usable{}:
      (gamma,beta) = parameters
      forward_shapes(x,shape,workers,F.shape(gamma),beta,epsilon)
    case Tensors.Failed{+problem}: forward_failed(problem,shape,workers,parameters)

# x in [..., D], gamma and beta in [D]: (xhat * gamma + beta, ((xhat, deviation),
# (gamma, beta))).
def forward(x: Tensor(),parameters: Parameters(),+epsilon: F32) -> Forward():
  Tiled.with_metadata(~Forward(),x,x => shape => status => workers => forward_status(status,shape,workers,epsilon,x,parameters))

# ---- Backward ----

def three(pair: Tensor() & Tensor()) -> Tensor() & (Tensor() & Tensor()):
  (first,second) = pair
  (first,F.copy(second))

def input_gradient(+shape: List<&2,U32>,dzs: Tensor() & (Tensor() & Tensor()),xhat: Tensor(),xhat_product: Tensor(),deviation: Tensor()) -> Tensor():
  (dz,(for_mean,for_product)) = dzs
  F.mul(F.sub(F.sub(dz,line_mean(for_mean,shape)),F.mul(xhat,line_mean(F.mul(for_product,xhat_product),shape))),deviation)

def back_parts(+shape: List<&2,U32>,dys: Tensor() & (Tensor() & Tensor()),xhats: Tensor() & (Tensor() & Tensor()),deviation: Tensor(),gammas: Tensor() & Tensor()) -> Backward():
  (dy,(dy_gamma,dy_beta)) = dys
  (xhat,(xhat_gamma,xhat_product)) = xhats
  (gamma,kept) = gammas
  +axes = before_last(List.length(&2,U32,shape),0)
  (input_gradient(shape,three(F.copy(F.mul(dy,gamma))),xhat,xhat_product,deviation),
    (F.sum_axes(F.mul(dy_gamma,xhat_gamma),axes),(F.sum_axes(dy_beta,axes),kept)))

def back_shaped(dy: Tensor(),+shape: List<&2,U32>,saved: Saved(),gamma: Tensor()) -> Backward():
  (xhat,deviation) = saved
  back_parts(shape,three(F.copy(dy)),three(F.copy(xhat)),deviation,F.copy(gamma))

# A failed dy fails every gradient and returns gamma.
def backward_failed(+problem: Tensors.Problem,+shape: List<&2,U32>,+workers: U32,gamma: Tensor()) -> Backward():
  (Tiled.failed_as(problem,shape,workers),(Tiled.failed_as(problem,last_of(shape),workers),(Tiled.failed_as(problem,last_of(shape),workers),gamma)))

def back_status(+status: Tensors.Status,+shape: List<&2,U32>,+workers: U32,dy: Tensor(),saved: Saved(),gamma: Tensor()) -> Backward():
  match status:
    case Tensors.Usable{}: back_shaped(dy,shape,saved,gamma)
    case Tensors.Failed{+problem}: backward_failed(problem,shape,workers,gamma)

# dy and what forward saved: (dx, (dgamma, (dbeta, gamma))). With dz = dy * gamma:
# dx = (dz - mean(dz) - xhat * mean(dz * xhat)) * deviation along the last axis,
# dgamma the sum of dy * xhat and dbeta the sum of dy over the other axes.
def backward(dy: Tensor(),saved: Saved(),gamma: Tensor()) -> Backward():
  Tiled.with_metadata(~Backward(),dy,dy => shape => status => workers => back_status(status,shape,workers,dy,saved,gamma))

# ---- Evaluate ----

def evaluated_kept(saved: Saved(),gammas: Tensor() & Tensor(),betas: Tensor() & Tensor()) -> Evaluated():
  (xhat,deviation) = saved
  (gamma,gamma_kept) = gammas
  (beta,beta_kept) = betas
  (affine(xhat,gamma,beta),(gamma_kept,beta_kept))

def evaluate_failed(+problem: Tensors.Problem,+shape: List<&2,U32>,+workers: U32,parameters: Parameters()) -> Evaluated():
  (Tiled.failed_as(problem,shape,workers),parameters)

def evaluate_when(same: Bool,x: Tensor(),+shape: List<&2,U32>,+workers: U32,gamma: Tensor(),beta: Tensor(),+epsilon: F32) -> Evaluated():
  match same:
    case True{}: evaluated_kept(normalized(x,epsilon,shape),F.copy(gamma),F.copy(beta))
    case False{}: evaluate_failed(mismatched(shape),shape,workers,(gamma,beta))

def evaluate_shapes(x: Tensor(),+shape: List<&2,U32>,+workers: U32,gamma_measured: Tensor() & List<&2,U32>,beta: Tensor(),+epsilon: F32) -> Evaluated():
  (gamma,gamma_shape) = gamma_measured
  evaluate_when(matches(shape,gamma_shape),x,shape,workers,gamma,beta,epsilon)

def evaluate_status(+status: Tensors.Status,+shape: List<&2,U32>,+workers: U32,+epsilon: F32,x: Tensor(),parameters: Parameters()) -> Evaluated():
  match status:
    case Tensors.Usable{}:
      (gamma,beta) = parameters
      evaluate_shapes(x,shape,workers,F.shape(gamma),beta,epsilon)
    case Tensors.Failed{+problem}: evaluate_failed(problem,shape,workers,parameters)

# The same output as forward, without what backward reads.
def evaluate(x: Tensor(),parameters: Parameters(),+epsilon: F32) -> Evaluated():
  Tiled.with_metadata(~Evaluated(),x,x => shape => status => workers => evaluate_status(status,shape,workers,epsilon,x,parameters))
