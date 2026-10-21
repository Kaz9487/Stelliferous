# One epoch of a ResNet-20-shaped network on CIFAR-10-shaped images through
# the public training API: 19 Conv-BatchNorm-ReLU layers, the average pool and
# a 1x1 head of 10 classes, cross entropy and SGD, 390 steps of batch 128.
# The residual additions and projections are left out. One batch of fixed
# images is reused, so reading data is not timed. epoch_torch.py runs the same
# network in PyTorch.
import Base
import ../../lib/math_activation.bend as Act
import ../../lib/tensor_f32.bend as F
import ../../lib/tensor_training.bend as Train

type Network is Type:
  End{}
  Then{layer: Train.Layer,rest: Network}

type Tape is Type:
  Blank{}
  Recorded{saved: Train.Saved,layer: Train.Layer,rest: Tape}

def recorded(result: F.Tensor() & (Train.Saved & Train.Layer),next: F.Tensor() -> Train.Saved -> Train.Layer -> F.Tensor() & Tape) -> F.Tensor() & Tape:
  (y,(saved,layer)) = result
  next(y,saved,layer)

def forward(network: Network,x: F.Tensor(),tape: Tape) -> F.Tensor() & Tape:
  match network:
    case End{}: (x,tape)
    case Then{layer,rest}: recorded(Train.forward(x,layer),y => saved => layer => forward(rest,y,Recorded{saved,layer,tape}))

def returned(result: F.Tensor() & Train.Layer,next: F.Tensor() -> Train.Layer -> Network) -> Network:
  (dx,layer) = result
  next(dx,layer)

def backward(tape: Tape,dy: F.Tensor(),network: Network) -> Network:
  match tape:
    case Blank{}: network
    case Recorded{saved,layer,rest}: returned(Train.backward(dy,saved,layer),dx => layer => backward(rest,dx,Then{layer,network}))

def updated(network: Network,+rate: F32) -> Network:
  match network:
    case End{}: End{}
    case Then{layer,rest}: Then{Train.sgd_step(layer,rate),updated(rest,rate)}

def Stepped() -> Type:
  F32 & (F.Tensor() & (F.Tensor() & Network))

def lost(x: F.Tensor(),tape: Tape,result: F.Tensor() & (F.Tensor() & F.Tensor())) -> Stepped():
  (loss,(dy,target)) = result
  (F.item_or(loss,0.0),(x,(target,updated(backward(tape,F.reshape(dy,[128,10,1,1]),End{}),0.1))))

def predicted(x: F.Tensor(),target: F.Tensor(),result: F.Tensor() & Tape) -> Stepped():
  (y,tape) = result
  lost(x,tape,Train.cross_entropy(F.reshape(y,[128,10]),target))

def step(xs: F.Tensor() & F.Tensor(),target: F.Tensor(),network: Network) -> Stepped():
  (x,kept) = xs
  predicted(kept,target,forward(network,x,Blank{}))

def stepped(result: Stepped(),next: F32 -> F.Tensor() -> F.Tensor() -> Network -> List<&2,F32> & Network) -> List<&2,F32> & Network:
  (loss,(x,(target,network))) = result
  next(loss,x,target,network)

def continued(+loss: F32,result: List<&2,F32> & Network) -> List<&2,F32> & Network:
  (losses,network) = result
  (loss <> losses,network)

def trained(remaining: Nat,x: F.Tensor(),target: F.Tensor(),network: Network) -> List<&2,F32> & Network:
  match remaining:
    case 0n: ([],network)
    case 1n+rest: stepped(step(F.copy(x),target,network),loss => x => target => network => continued(loss,trained(rest,x,target,network)))

# Conv 3x3 (padding 1), batch normalization and ReLU in front of rest.
def block(+inputs: U32,+outputs: U32,+stride: U32,+workers: U32,rest: Network) -> Network:
  Then{Train.conv2d_layer(F.with_workers(F.full([outputs,inputs,3,3],0.01),workers),F.with_workers(F.zeros([outputs]),workers),stride,1),
    Then{Train.batch_norm_layer(outputs),Then{Train.activation_layer(Act.relu()),rest}}}

def repeated(count: Nat,+channels: U32,+workers: U32,rest: Network) -> Network:
  match count:
    case 0n: rest
    case 1n+more: block(channels,channels,1,workers,repeated(more,channels,workers,rest))

def stage(+inputs: U32,+outputs: U32,+stride: U32,+workers: U32,rest: Network) -> Network:
  later = repeated(5n,outputs,workers,rest)
  block(inputs,outputs,stride,workers,later)

def head(+workers: U32) -> Network:
  Then{Train.average_pool_layer(),Then{Train.conv2d_layer(F.with_workers(F.full([10,64,1,1],0.01),workers),F.with_workers(F.zeros([10]),workers),1,0),End{}}}

def network(+workers: U32) -> Network:
  third = stage(32,64,2,workers,head(workers))
  second = stage(16,32,2,workers,third)
  first = stage(16,16,1,workers,second)
  block(3,16,1,workers,first)

def first_last(losses: List<&2,F32>,+first: F32,+last: F32) -> String:
  match losses:
    case []: F32.show(first) ++ " ... " ++ F32.show(last)
    case +loss <> rest: first_last(rest,first,loss)

def losses_text(losses: List<&2,F32>) -> String:
  match losses:
    case []: "none"
    case +loss <> rest: first_last(rest,loss,loss)

def reported(+workers: U32,+start: Nat,result: List<&2,F32> & Network) -> IO(Unit):
  (losses,network) = result
  do IO<Unit>:
    end : Nat <- IO.now()
    IO.print("Stelliferous epoch of 390 steps, batch 128, workers " ++ U32.show(workers) ++ ": " ++ Nat.show(Nat.sub(end,start)) ++ " ms, loss " ++ losses_text(losses))

def timed(+workers: U32) -> IO(Unit):
  do IO<Unit>:
    start : Nat <- IO.now()
    reported(workers,start,trained(390n,F.with_workers(F.mul_scalar(F.reshape(F.arange(0.0,393216),[128,3,32,32]),0.00001),workers),
      Train.one_hot(F.with_workers(F.zeros([128]),workers),10),network(workers)))

def main() -> IO(Unit):
  do IO<Unit>:
    workers : U32 <- IO.thread_count()
    timed(workers)
