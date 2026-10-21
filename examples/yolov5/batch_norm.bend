# The batch normalization after each convolution of a YOLOv5 checkpoint,
# folded into the convolution's weights and bias once, before the network runs.
import Base
import ../../lib/tensor_f32.bend as F

# The batch normalization epsilon YOLOv5 sets (initialize_weights in utils/torch_utils.py).
def epsilon() -> F32:
  0.001

# [O] as [O,1,1,1], one value for each output of the weights.
def column(pair: F.Tensor() & List<&2,U32>) -> F.Tensor():
  (tensor,shape) = pair
  F.reshape(tensor,List.append(&2,U32,shape,[1,1,1]))

def folded(weights: F.Tensor(),beta: F.Tensor(),mean: F.Tensor(),gammas: F.Tensor() & F.Tensor(),roots: F.Tensor() & F.Tensor()) -> F.Tensor() & F.Tensor():
  (gamma,gamma_bias) = gammas
  (root,root_bias) = roots
  (F.mul(weights,column(F.shape(F.div(gamma,root)))),F.sub(beta,F.div(F.mul(gamma_bias,mean),root_bias)))

def kept(+name: String,pair: F.Tensor() & F.Tensor()) -> F.Model(Unit):
  (weights,bias) = pair
  do F.Model<Unit>:
    weights : F.Tensor() <- F.keep(name ++ ".conv.weight",weights)
    bias : F.Tensor() <- F.keep(name ++ ".conv.bias",bias)
    F.Model.pure(Unit,Unit{})

# In the order of PyTorch's fuse_conv_and_bn: scale = gamma / sqrt(epsilon + variance),
# the weights times the scale of their output, bias = beta - gamma * mean / sqrt(variance + epsilon).
def fold(+name: String) -> F.Model(Unit):
  do F.Model<Unit>:
    weights : F.Tensor() <- F.take(name ++ ".conv.weight")
    gamma : F.Tensor() <- F.take(name ++ ".bn.weight")
    beta : F.Tensor() <- F.take(name ++ ".bn.bias")
    mean : F.Tensor() <- F.take(name ++ ".bn.running_mean")
    variance : F.Tensor() <- F.take(name ++ ".bn.running_var")
    batches : F.Tensor() <- F.take(name ++ ".bn.num_batches_tracked")
    kept(name,folded(weights,beta,mean,F.copy(gamma),F.copy(F.sqrt(F.add_scalar(variance,epsilon())))))

def fold_when(found: Bool,+name: String) -> F.Model(Unit):
  match found:
    case True{}: fold(String.take(name,Nat.sub(String.length(name),15n)))
    case False{}: F.Model.pure(Unit,Unit{})

def fold_each(names: List<&2,String>) -> F.Model(Unit):
  match names:
    case []: F.Model.pure(Unit,Unit{})
    case +name <> rest:
      do F.Model<Unit>:
        done : Unit <- fold_when(String.ends_with(name,".bn.running_var"),name)
        fold_each(rest)

# Every <name>.bn.* folded into <name>.conv.weight and a new <name>.conv.bias.
def fold_all() -> F.Model(Unit):
  do F.Model<Unit>:
    names : List<&2,String> <- F.weight_names()
    fold_each(names)

def folded_weights(pair: Unit & F.Weights()) -> F.Weights():
  (done,weights) = pair
  weights

# The checkpoint with its batch normalizations folded.
def load(+path: String) -> IO(F.Weights()):
  do IO<F.Weights()>:
    weights : F.Weights() <- F.load_checkpoint(path)
    IO.pure(F.Weights(),folded_weights(F.run_keeping(Unit,fold_all(),weights)))
