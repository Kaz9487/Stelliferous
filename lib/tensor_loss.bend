# Losses with their gradient, in a fixed FP32 order. The prediction becomes
# the gradient, and the target returns.
import Base
import ./tensor.bend as Tensors
import ./tensor_shape.bend as Shapes
import ./tensor_layout.bend as Layout
import ./tensor_operations.bend as Operations
import ./math_sum.bend as Sum
import ./tensor_f32.bend as F
import ./math_fp32.bend as Arithmetic

def Tensor() -> Type:
  Tensors.Tensor<F32>

type Reduction is Data:
  Mean{}
  Sum{}

def mean() -> Reduction:
  Mean{}

def sum() -> Reduction:
  Sum{}

def square(value: F32) -> F32:
  +value = value
  (value * value : F32)

def reduced(reduction: Reduction,+count: F32,total: Tensor()) -> Tensor():
  match reduction:
    case Mean{}: Tensors.combine_scalar(~F32,~F32.div,1.0,total,count)
    case Sum{}: total

def mse_squares(+reduction: Reduction,+count: F32,target: Tensor(),pair: Tensor() & Tensor()) -> Tensor() & (Tensor() & Tensor()):
  (difference,squares) = pair
  (reduced(reduction,count,Layout.reduce_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"mse_loss",6.0,Sum.zero(),
      Tensors.apply(~F32,~Operations.mapped(~F32,~square),1.0,squares))),
    (reduced(reduction,count,Tensors.combine_scalar(~F32,~F32.mul,1.0,difference,2.0)),target))

def mse_counted(+reduction: Reduction,target: Tensor(),shaped: Tensor() & List<&2,U32>) -> Tensor() & (Tensor() & Tensor()):
  (difference,+shape) = shaped
  mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(shape)),target,Layout.copy(~F32,difference))

def mse_difference(same: Bool,prediction: Tensor(),kept: Tensor()) -> Tensor():
  match same:
    case True{}: Tensors.combine(~F32,~F32.sub,"mse_loss",1.0,prediction,kept)
    case False{}: Tensors.rejected(~F32,"mse_loss",prediction)

def mse_kept(+reduction: Reduction,same: Bool,prediction: Tensor(),pair: Tensor() & Tensor()) -> Tensor() & (Tensor() & Tensor()):
  (target,kept) = pair
  mse_counted(reduction,target,Tensors.shape(~F32,mse_difference(same,prediction,kept)))

def mse_shaped(+reduction: Reduction,shaped: Tensor() & List<&2,U32>,measured: Tensor() & List<&2,U32>) -> Tensor() & (Tensor() & Tensor()):
  (prediction,shape) = shaped
  (target,target_shape) = measured
  mse_kept(reduction,Shapes.same_extents(shape,target_shape),prediction,Layout.copy(~F32,target))

# The loss of prediction against a target of the same shape, the gradient of
# the loss with respect to prediction, and the target. With d = prediction -
# target: Mean gives sum(d * d) / count and (d * 2) / count, Sum gives
# sum(d * d) and d * 2; the squares are added in row-major order from +0.
def mse(prediction: Tensor(),target: Tensor(),+reduction: Reduction) -> Tensor() & (Tensor() & Tensor()):
  mse_shaped(reduction,Tensors.shape(~F32,prediction),Tensors.shape(~F32,target))

# ---- Cross entropy ----

# The class numbers [N] as rows [N,K] of a target: 1 where the column equals
# the class number, 0 elsewhere.
def one_hot_rows(shape: List<&2,U32>,labels: Tensor(),+classes: U32) -> Tensor():
  match shape:
    case [images]: F.equal(F.reshape(labels,[images,1]),F.reshape(F.arange(0.0,classes),[1,classes]))
    case shape: Tensors.rejected(~F32,"one_hot",labels)

def one_hot_measured(measured: Tensor() & List<&2,U32>,+classes: U32) -> Tensor():
  (labels,shape) = measured
  one_hot_rows(shape,labels,classes)

def one_hot(labels: Tensor(),+classes: U32) -> Tensor():
  one_hot_measured(Tensors.shape(~F32,labels),classes)

# The exponentials of a row added in the lanes of math_sum.bend, and the
# logarithm of their sum.
def exponential_step(lanes: Sum.Lanes,value: F32) -> Sum.Lanes:
  Sum.step(lanes,Arithmetic.exp(value))

def logarithm_total(lanes: Sum.Lanes) -> F32:
  Arithmetic.log(Sum.total(lanes))

def entropy_gradient(+images: U32,log_probabilities: Tensor() & Tensor(),targets: Tensor() & Tensor(),target: Tensor()) -> Tensor() & (Tensor() & Tensor()):
  (for_loss,for_gradient) = log_probabilities
  (for_product,for_difference) = targets
  (F.div_scalar(F.sum(F.mul(for_product,for_loss)),F32.neg(U32.to_f32(images))),
    (F.div_scalar(F.sub(F.exp(for_gradient),for_difference),U32.to_f32(images)),target))

def entropy_shifted(+images: U32,shifted: Tensor() & Tensor(),targets: Tensor() & Tensor()) -> Tensor() & (Tensor() & Tensor()):
  (values,for_sum) = shifted
  (target,target_kept) = targets
  entropy_gradient(images,
    F.copy(F.sub(values,F.reshape(Layout.reduce_axes(~F32,~Sum.Lanes,~exponential_step,~logarithm_total,"cross_entropy",10.5,Sum.zero(),for_sum,[1]),[images,1]))),
    F.copy(target),target_kept)

def entropy_rows(+images: U32,logits: Tensor() & Tensor(),target: Tensor()) -> Tensor() & (Tensor() & Tensor()):
  (values,for_maximum) = logits
  entropy_shifted(images,
    F.copy(F.sub(values,F.reshape(Layout.reduce_axes(~F32,~F32,~F32.max,~Tensors.same(~F32),"cross_entropy",1.0,F32.div(F32.neg(1.0),0.0),for_maximum,[1]),[images,1]))),
    F.copy(target))

def entropy_failed(shape: List<&2,U32>,logits: Tensor(),target: Tensor()) -> Tensor() & (Tensor() & Tensor()):
  (Tensors.failed(~F32,"cross_entropy",Tensors.InvalidShape{shape},[],0.0,1),(Tensors.rejected(~F32,"cross_entropy",logits),target))

def entropy_ranked(shape: List<&2,U32>,logits: Tensor(),target: Tensor()) -> Tensor() & (Tensor() & Tensor()):
  match shape:
    case [images,classes]: entropy_rows(images,F.copy(logits),target)
    case shape: entropy_failed(shape,logits,target)

def entropy_shaped(same: Bool,logits: Tensor(),shape: List<&2,U32>,target: Tensor()) -> Tensor() & (Tensor() & Tensor()):
  match same:
    case True{}: entropy_ranked(shape,logits,target)
    case False{}: entropy_failed(shape,logits,target)

def entropy_measured(measured_logits: Tensor() & List<&2,U32>,measured_target: Tensor() & List<&2,U32>) -> Tensor() & (Tensor() & Tensor()):
  (logits,+shape) = measured_logits
  (target,target_shape) = measured_target
  entropy_shaped(Shapes.same_extents(shape,target_shape),logits,shape,target)

# logits [N,K] and a target [N,K] whose rows are class probabilities, such as
# one_hot of the class numbers: the mean loss, its gradient with respect to
# logits, and the target. With m the maximum of a row and the log
# probabilities p = (logits - m) - log(sum of exp(logits - m) over the row),
# the sum over the row in the lanes of math_sum.bend: the loss is
# sum(target * p) / -N, the sum of math_sum.bend in row-major order, and the
# gradient (exp(p) - target) / N.
def cross_entropy(logits: Tensor(),target: Tensor()) -> Tensor() & (Tensor() & Tensor()):
  entropy_measured(Tensors.shape(~F32,logits),Tensors.shape(~F32,target))
