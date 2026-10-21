# Parameter updates in a fixed FP32 order. The parameter's storage is reused.
import Base
import ./tensor.bend as Tensors
import ./tensor_f32.bend as F

def Tensor() -> Type:
  Tensors.Tensor<F32>

# parameter - gradient * rate, for a gradient of the parameter's shape.
def sgd(parameter: Tensor(),gradient: Tensor(),+rate: F32) -> Tensor():
  F.sub(parameter,F.mul_scalar(gradient,rate))
