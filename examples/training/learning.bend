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

# The layers of the network in order.
type Network is Type:
  End{}
  Then{layer: Train.Layer,rest: Network}

# Each layer with what its backward reads, the last layer first.
type Tape is Type:
  Blank{}
  Recorded{saved: Train.Saved,layer: Train.Layer,rest: Tape}

def recorded(result: F.Tensor() & (Train.Saved & Train.Layer),next: F.Tensor() -> Train.Saved -> Train.Layer -> F.Tensor() & Tape) -> F.Tensor() & Tape:
  (y,(saved,layer)) = result
  next(y,saved,layer)

# The output of the network and the tape that holds its layers.
def forward(network: Network,x: F.Tensor(),tape: Tape) -> F.Tensor() & Tape:
  match network:
    case End{}: (x,tape)
    case Then{layer,rest}: recorded(Train.forward(x,layer),y => saved => layer => forward(rest,y,Recorded{saved,layer,tape}))

def returned(result: F.Tensor() & Train.Layer,next: F.Tensor() -> Train.Layer -> Network) -> Network:
  (dx,layer) = result
  next(dx,layer)

# The layers in order again, each holding its gradients.
def backward(tape: Tape,dy: F.Tensor(),network: Network) -> Network:
  match tape:
    case Blank{}: network
    case Recorded{saved,layer,rest}: returned(Train.backward(dy,saved,layer),dx => layer => backward(rest,dx,Then{layer,network}))

def updated(network: Network,+rate: F32) -> Network:
  match network:
    case End{}: End{}
    case Then{layer,rest}: Then{Train.sgd_step(layer,rate),updated(rest,rate)}

# The loss of one step, the input and the target back, and the updated network.
def Stepped() -> Type:
  F32 & (F.Tensor() & (F.Tensor() & Network))

def lost(x: F.Tensor(),tape: Tape,result: F.Tensor() & (F.Tensor() & F.Tensor())) -> Stepped():
  (loss,(dy,target)) = result
  (F.item_or(loss,0.0),(x,(target,updated(backward(tape,dy,End{}),0.5))))

def predicted(x: F.Tensor(),target: F.Tensor(),result: F.Tensor() & Tape) -> Stepped():
  (y,tape) = result
  lost(x,tape,Train.mse(y,target))

def step(xs: F.Tensor() & F.Tensor(),target: F.Tensor(),network: Network) -> Stepped():
  (x,kept) = xs
  predicted(kept,target,forward(network,x,Blank{}))

def stepped(result: Stepped(),next: F32 -> F.Tensor() -> F.Tensor() -> Network -> List<&2,F32> & Network) -> List<&2,F32> & Network:
  (loss,(x,(target,network))) = result
  next(loss,x,target,network)

def continued(+loss: F32,result: List<&2,F32> & Network) -> List<&2,F32> & Network:
  (losses,network) = result
  (loss <> losses,network)

# The loss of every step and the network after the last one.
def trained(remaining: Nat,x: F.Tensor(),target: F.Tensor(),network: Network) -> List<&2,F32> & Network:
  match remaining:
    case 0n: ([],network)
    case 1n+rest: stepped(step(F.copy(x),target,network),loss => x => target => network => continued(loss,trained(rest,x,target,network)))

def load(+name: String,+shape: List<&2,U32>,+workers: U32) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    tensor : F.Tensor() <- F.load_raw(folder() ++ "learn_" ++ name ++ ".bin",shape)
    IO.pure(F.Tensor(),F.with_workers(tensor,workers))

# A convolution of 32 channels, its batch normalization and SiLU.
def block(+index: String,+workers: U32,rest: Network) -> IO(Network):
  do IO<Network>:
    weights : F.Tensor() <- load("weights" ++ index,[32,32,3,3],workers)
    bias : F.Tensor() <- load("bias" ++ index,[32],workers)
    gamma : F.Tensor() <- load("gamma" ++ index,[32],workers)
    beta : F.Tensor() <- load("beta" ++ index,[32],workers)
    mean : F.Tensor() <- load("mean" ++ index,[32],workers)
    variance : F.Tensor() <- load("variance" ++ index,[32],workers)
    IO.pure(Network,Then{Train.conv2d_layer(weights,bias,1,1),
      Then{Train.batch_norm_layer_from(gamma,beta,mean,variance,0.00001,0.1),Then{Train.activation_layer(Act.silu()),rest}}})

def save(+name: String,+tag: String,tensor: F.Tensor()) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    tensor : F.Tensor() <- F.save_raw(folder() ++ "learned_" ++ name ++ "_" ++ tag ++ ".bin",tensor)
    F.expect(tensor)

# The parameters of a layer under the number of its block; a convolution
# starts the next block.
def saved_layer(layer: Train.Layer,+block: Nat,+tag: String) -> IO(Nat):
  match layer:
    case Train.Convolution{weights,bias,stride,padding,weight_gradient,bias_gradient}:
      do IO<Nat>:
        weights : F.Tensor() <- save("weights" ++ Nat.show(Nat.add(block,1n)),tag,weights)
        bias : F.Tensor() <- save("bias" ++ Nat.show(Nat.add(block,1n)),tag,bias)
        IO.pure(Nat,Nat.add(block,1n))
    case Train.BatchNorm{gamma,beta,running_mean,running_variance,epsilon,momentum,gamma_gradient,beta_gradient}:
      do IO<Nat>:
        gamma : F.Tensor() <- save("gamma" ++ Nat.show(block),tag,gamma)
        beta : F.Tensor() <- save("beta" ++ Nat.show(block),tag,beta)
        mean : F.Tensor() <- save("mean" ++ Nat.show(block),tag,running_mean)
        variance : F.Tensor() <- save("variance" ++ Nat.show(block),tag,running_variance)
        IO.pure(Nat,block)
    case Train.Activation{activation}: IO.pure(Nat,block)
    case Train.AveragePool{}: IO.pure(Nat,block)

def saved_network(network: Network,+block: Nat,+tag: String) -> IO(Unit):
  match network:
    case End{}: IO.pure(Unit,Unit{})
    case Then{layer,rest}:
      do IO<Unit>:
        next : Nat <- saved_layer(layer,block,tag)
        saved_network(rest,next,tag)

def saved(+tag: String,result: List<&2,F32> & Network) -> IO(Unit):
  (losses,network) = result
  do IO<Unit>:
    losses : F.Tensor() <- save("losses",tag,F.from_list([40],losses))
    saved_network(network,0n,tag)

def learned(+workers: U32) -> IO(Unit):
  do IO<Unit>:
    x : F.Tensor() <- load("x",[2,32,40,40],workers)
    target : F.Tensor() <- load("target",[2,32,40,40],workers)
    second : Network <- block("2",workers,End{})
    network : Network <- block("1",workers,second)
    saved(U32.show(workers),trained(40n,x,target,network))

def main() -> IO(Unit):
  do IO<Unit>:
    workers : U32 <- IO.thread_count()
    learned(workers)
