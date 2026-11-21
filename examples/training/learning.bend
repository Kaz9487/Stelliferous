# Two Conv-BatchNorm-SiLU layers learn the output of a teacher network: 40
# steps of SGD on the mean squared error. python run.py gradients writes the
# inputs, runs this program on 1, 2, 4 and 8 threads and compares the losses
# and the learned layers with the reference and with PyTorch.
import Base
import ../../lib/math_activation.bend as Act
import ../../lib/tensor_f32.bend as F
import ../../lib/tensor_training.bend as Train

def folder() -> String:
  "out/results/training/"

def updated(result: F.Tensor() & Train.Network,+rate: F32) -> Train.Network:
  (dx,network) = result
  Train.sgd_network(network,rate)

# The loss of one step, the input and the target back, and the updated network.
def Stepped() -> Type:
  F32 & (F.Tensor() & (F.Tensor() & Train.Network))

def lost(x: F.Tensor(),tape: Train.Tape,result: F.Tensor() & (F.Tensor() & F.Tensor())) -> Stepped():
  (loss,(dy,target)) = result
  (F.item_or(loss,0.0),(x,(target,updated(Train.backward_network(dy,tape),0.5))))

def predicted(x: F.Tensor(),target: F.Tensor(),result: F.Tensor() & Train.Tape) -> Stepped():
  (y,tape) = result
  lost(x,tape,Train.mse(y,target))

def step(xs: F.Tensor() & F.Tensor(),target: F.Tensor(),network: Train.Network) -> Stepped():
  (x,kept) = xs
  predicted(kept,target,Train.forward_network(x,network))

def stepped(result: Stepped(),next: F32 -> F.Tensor() -> F.Tensor() -> Train.Network -> List<&2,F32> & Train.Network) -> List<&2,F32> & Train.Network:
  (loss,(x,(target,network))) = result
  next(loss,x,target,network)

def continued(+loss: F32,result: List<&2,F32> & Train.Network) -> List<&2,F32> & Train.Network:
  (losses,network) = result
  (loss <> losses,network)

# The loss of every step and the network after the last one.
def trained(remaining: Nat,x: F.Tensor(),target: F.Tensor(),network: Train.Network) -> List<&2,F32> & Train.Network:
  match remaining:
    case 0n: ([],network)
    case 1n+rest: stepped(step(F.copy(x),target,network),loss => x => target => network => continued(loss,trained(rest,x,target,network)))

def load(+name: String,+shape: List<&2,U32>,+workers: U32) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    tensor : F.Tensor() <- F.load_raw(folder() ++ "learn_" ++ name ++ ".bin",shape)
    IO.pure(F.Tensor(),F.with_workers(tensor,workers))

# A convolution of 32 channels, its batch normalization and SiLU.
def block(+index: String,+workers: U32,rest: Train.Network) -> IO(Train.Network):
  do IO<Train.Network>:
    weights : F.Tensor() <- load("weights" ++ index,[32,32,3,3],workers)
    bias : F.Tensor() <- load("bias" ++ index,[32],workers)
    gamma : F.Tensor() <- load("gamma" ++ index,[32],workers)
    beta : F.Tensor() <- load("beta" ++ index,[32],workers)
    mean : F.Tensor() <- load("mean" ++ index,[32],workers)
    variance : F.Tensor() <- load("variance" ++ index,[32],workers)
    IO.pure(Train.Network,Train.Then{Train.conv2d_layer(weights,bias,1,1),
      Train.Then{Train.batch_norm_layer_from(gamma,beta,mean,variance,0.00001,0.1),Train.Then{Train.activation_layer(Act.silu()),rest}}})

def save(+name: String,+tag: String,tensor: F.Tensor()) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    tensor : F.Tensor() <- F.save_raw(folder() ++ "learned_" ++ name ++ "_" ++ tag ++ ".bin",tensor)
    F.expect(tensor)

# The parameters of a layer under the number of its block; a convolution
# starts the next block.
def saved_layer(layer: Train.Layer,+block: Nat,+tag: String) -> IO(Nat):
  match layer:
    case Train.Convolution{weights,bias,stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      do IO<Nat>:
        weights : F.Tensor() <- save("weights" ++ Nat.show(Nat.add(block,1n)),tag,weights)
        bias : F.Tensor() <- save("bias" ++ Nat.show(Nat.add(block,1n)),tag,bias)
        IO.pure(Nat,Nat.add(block,1n))
    case Train.BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      do IO<Nat>:
        gamma : F.Tensor() <- save("gamma" ++ Nat.show(block),tag,gamma)
        beta : F.Tensor() <- save("beta" ++ Nat.show(block),tag,beta)
        mean : F.Tensor() <- save("mean" ++ Nat.show(block),tag,running_mean)
        variance : F.Tensor() <- save("variance" ++ Nat.show(block),tag,running_variance)
        IO.pure(Nat,block)
    case Train.LayerNorm{gamma,beta,epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}: IO.pure(Nat,block)
    case Train.Activation{activation}: IO.pure(Nat,block)
    case Train.AveragePool{}: IO.pure(Nat,block)

# The parameters of every layer in order; the number of the last block.
def saved_network(network: Train.Network,+block: Nat,+tag: String) -> IO(Nat):
  match network:
    case Train.End{}: IO.pure(Nat,block)
    case Train.Then{layer,rest}:
      do IO<Nat>:
        next : Nat <- saved_layer(layer,block,tag)
        saved_network(rest,next,tag)
    case Train.Residual{branch,shortcut,rest}:
      do IO<Nat>:
        after_branch : Nat <- saved_network(branch,block,tag)
        after_shortcut : Nat <- saved_network(shortcut,after_branch,tag)
        saved_network(rest,after_shortcut,tag)

def saved(+tag: String,result: List<&2,F32> & Train.Network) -> IO(Unit):
  (losses,network) = result
  do IO<Unit>:
    losses : F.Tensor() <- save("losses",tag,F.from_list([40],losses))
    blocks : Nat <- saved_network(network,0n,tag)
    IO.pure(Unit,Unit{})

def learned(+workers: U32) -> IO(Unit):
  do IO<Unit>:
    x : F.Tensor() <- load("x",[2,32,40,40],workers)
    target : F.Tensor() <- load("target",[2,32,40,40],workers)
    second : Train.Network <- block("2",workers,Train.End{})
    network : Train.Network <- block("1",workers,second)
    saved(U32.show(workers),trained(40n,x,target,network))

def main() -> IO(Unit):
  do IO<Unit>:
    workers : U32 <- IO.thread_count()
    learned(workers)
