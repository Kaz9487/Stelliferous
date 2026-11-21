# Parameter updates in a fixed FP32 order. The parameter's storage is reused.
import Base
import ./tensor_f32.bend as F

def Tensor() -> Type:
  F.Tensor()

# parameter - gradient * rate, for a gradient of the parameter's shape.
def sgd(parameter: Tensor(),gradient: Tensor(),+rate: F32) -> Tensor():
  F.sub(parameter,F.mul_scalar(gradient,rate))

# The velocity of one parameter: none before its first step with momentum. The
# rest is always Still: a recursive type is one box, so a layer stays narrow
# however many tensors it holds.
type Velocity is Type:
  Still{}
  Moving{tensor: Tensor(),rest: Velocity}

# gradient + parameter * decay. A decay of zero leaves the gradient as it is.
def decayed_copies(gradient: Tensor(),+decay: F32,copies: Tensor() & Tensor()) -> Tensor() & Tensor():
  (parameter,factor) = copies
  (parameter,F.add(gradient,F.mul_scalar(factor,decay)))

def decayed(zero: Bool,parameter: Tensor(),gradient: Tensor(),+decay: F32) -> Tensor() & Tensor():
  match zero:
    case True{}: (parameter,gradient)
    case False{}: decayed_copies(gradient,decay,F.copy(parameter))

def kept_velocity(copies: Tensor() & Tensor()) -> Tensor() & Velocity:
  (applied,kept) = copies
  (applied,Moving{kept,Still{}})

# velocity * momentum + direction; the first step takes the direction. A
# momentum of zero takes the direction and keeps no velocity.
def accumulated(velocity: Velocity,direction: Tensor(),+momentum: F32) -> Tensor() & Velocity:
  match velocity:
    case Still{}: kept_velocity(F.copy(direction))
    case Moving{tensor,rest}: kept_velocity(F.copy(F.add(F.mul_scalar(tensor,momentum),direction)))

def moved(zero: Bool,velocity: Velocity,direction: Tensor(),+momentum: F32) -> Tensor() & Velocity:
  match zero:
    case True{}: (direction,Still{})
    case False{}: accumulated(velocity,direction,momentum)

def descended(parameter: Tensor(),+rate: F32,applied: Tensor() & Velocity) -> Tensor() & Velocity:
  (direction,velocity) = applied
  (sgd(parameter,direction,rate),velocity)

def stepped(velocity: Velocity,+rate: F32,+momentum: F32,pair: Tensor() & Tensor()) -> Tensor() & Velocity:
  (parameter,direction) = pair
  descended(parameter,rate,moved(F32.is_eq(momentum,0.0),velocity,direction,momentum))

# (parameter - velocity * rate, velocity), as PyTorch's SGD without dampening
# computes them: direction = gradient + parameter * decay and velocity =
# velocity * momentum + direction.
def sgd_with(parameter: Tensor(),gradient: Tensor(),velocity: Velocity,+rate: F32,+momentum: F32,+decay: F32) -> Tensor() & Velocity:
  stepped(velocity,rate,momentum,decayed(F32.is_eq(decay,0.0),parameter,gradient,decay))
