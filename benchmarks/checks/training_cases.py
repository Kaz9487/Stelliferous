"""Run the training operations natively and compare them bit for bit with training_reference.py.

Every case runs with 1, 2, 4 and 8 workers. A tensor that an operation only
reads must come back with the bits it received. The steps of learning_case
also run in PyTorch, which must give the same losses, layers and running
statistics to rounding.
"""
from pathlib import Path
import os
import shutil
import subprocess
import numpy as np
import torch
from training_reference import (EPSILON, F32, MOMENTUM, batch_norm_backward, batch_norm_evaluate, batch_norm_forward, learning_case,
                                squared_error, step, train, two_layer_case)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'out/checks/training.bend'
EXAMPLE = ROOT / 'examples/training/learning.bend'
BUILD = ROOT / 'out/build/training'
REPORT = ROOT / 'out/results/training'
WORKERS = [1, 2, 4, 8]
STEPS = 40

# name, shape, mean
LOSSES = [
    ('loss_mean', [2, 3, 5, 7], True),
    ('loss_sum', [2, 3, 5, 7], False),
    ('loss_scalar', [], True),
    ('loss_large', [3, 87383], True),
]


# One training step of two_layer_case: every tensor the reference names is saved as it appears.
STEP = '''
def first() -> Layer.Settings:
  Layer.settings(FIRST,Act.silu())

def second() -> Layer.Settings:
  Layer.settings(SECOND,Act.silu())

def kept(+name: String,+tag: String,tensor: F.Tensor()) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    tensor : F.Tensor() <- F.save_raw(folder() ++ "step_" ++ name ++ "_" ++ tag ++ ".bin",tensor)
    F.expect(tensor)

def step_first(+tag: String,bias1: F.Tensor(),result: F.Tensor() & (F.Tensor() & (F.Tensor() & F.Tensor()))) -> IO(F.Tensor()):
  (dx,(dweights1,(dbias1,weights1))) = result
  do IO<F.Tensor()>:
    dx : F.Tensor() <- kept("dx",tag,dx)
    dweights1 : F.Tensor() <- kept("dweights1",tag,dweights1)
    dbias1 : F.Tensor() <- kept("dbias1",tag,dbias1)
    weights1 : F.Tensor() <- kept("weights1",tag,Optimizer.sgd(weights1,dweights1,RATE))
    kept("bias1",tag,Optimizer.sgd(bias1,dbias1,RATE))

def step_second(+tag: String,saved1: Layer.Saved(),weights1: F.Tensor(),bias1: F.Tensor(),bias2: F.Tensor(),result: F.Tensor() & (F.Tensor() & (F.Tensor() & F.Tensor()))) -> IO(F.Tensor()):
  (dy1,(dweights2,(dbias2,weights2))) = result
  do IO<F.Tensor()>:
    dy1 : F.Tensor() <- kept("dy1",tag,dy1)
    dweights2 : F.Tensor() <- kept("dweights2",tag,dweights2)
    dbias2 : F.Tensor() <- kept("dbias2",tag,dbias2)
    weights2 : F.Tensor() <- kept("weights2",tag,Optimizer.sgd(weights2,dweights2,RATE))
    bias2 : F.Tensor() <- kept("bias2",tag,Optimizer.sgd(bias2,dbias2,RATE))
    step_first(tag,bias1,Layer.backward(dy1,saved1,weights1,first()))

def step_loss(+tag: String,saved1: Layer.Saved(),weights1: F.Tensor(),bias1: F.Tensor(),saved2: Layer.Saved(),weights2: F.Tensor(),bias2: F.Tensor(),result: F.Tensor() & (F.Tensor() & F.Tensor())) -> IO(F.Tensor()):
  (loss,(dy2,target)) = result
  do IO<F.Tensor()>:
    loss : F.Tensor() <- kept("loss",tag,loss)
    step_second(tag,saved1,weights1,bias1,bias2,Layer.backward(dy2,saved2,weights2,second()))

def step_output(+tag: String,target: F.Tensor(),saved1: Layer.Saved(),weights1: F.Tensor(),bias1: F.Tensor(),result: F.Tensor() & (Layer.Saved() & (F.Tensor() & F.Tensor()))) -> IO(F.Tensor()):
  (y2,(saved2,(weights2,bias2))) = result
  step_loss(tag,saved1,weights1,bias1,saved2,weights2,bias2,Loss.mse(y2,target,Loss.mean()))

def step_hidden(+tag: String,target: F.Tensor(),weights2: F.Tensor(),bias2: F.Tensor(),result: F.Tensor() & (Layer.Saved() & (F.Tensor() & F.Tensor()))) -> IO(F.Tensor()):
  (y1,(saved1,(weights1,bias1))) = result
  step_output(tag,target,saved1,weights1,bias1,Layer.forward(y1,weights2,bias2,second()))

def step_case(+tag: String,+workers: U32) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    x : F.Tensor() <- F.load_raw(folder() ++ "step_x.bin",X_SHAPE)
    target : F.Tensor() <- F.load_raw(folder() ++ "step_target.bin",TARGET_SHAPE)
    weights1 : F.Tensor() <- F.load_raw(folder() ++ "step_weights1.bin",WEIGHTS1)
    bias1 : F.Tensor() <- F.load_raw(folder() ++ "step_bias1.bin",BIAS1)
    weights2 : F.Tensor() <- F.load_raw(folder() ++ "step_weights2.bin",WEIGHTS2)
    bias2 : F.Tensor() <- F.load_raw(folder() ++ "step_bias2.bin",BIAS2)
    step_hidden(tag,F.with_workers(target,workers),F.with_workers(weights2,workers),F.with_workers(bias2,workers),
      Layer.forward(F.with_workers(x,workers),F.with_workers(weights1,workers),F.with_workers(bias1,workers),first()))
'''
# One batch normalization case: forward, backward and evaluate, every tensor saved as it appears.
NORM = '''
def normalization() -> Norm.Settings:
  Norm.settings(EPSILON,MOMENTUM)

def norm_kept(+file: String,+tag: String,tensor: F.Tensor()) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    tensor : F.Tensor() <- F.save_raw(folder() ++ file ++ "_" ++ tag ++ ".bin",tensor)
    F.expect(tensor)

def norm_backward(+name: String,+tag: String,result: Norm.Backward()) -> IO(F.Tensor()):
  (dx,(dgamma,(dbeta,gamma))) = result
  do IO<F.Tensor()>:
    dx : F.Tensor() <- norm_kept(name ++ "_dx",tag,dx)
    dgamma : F.Tensor() <- norm_kept(name ++ "_dgamma",tag,dgamma)
    dbeta : F.Tensor() <- norm_kept(name ++ "_dbeta",tag,dbeta)
    norm_kept(name ++ "_gamma_back",tag,gamma)

def norm_forward(+name: String,+tag: String,dy: F.Tensor(),result: Norm.Forward()) -> IO(F.Tensor()):
  (y,((xhat,deviation),((gamma,beta),(mean,variance)))) = result
  do IO<F.Tensor()>:
    y : F.Tensor() <- norm_kept(name ++ "_y",tag,y)
    xhat : F.Tensor() <- norm_kept(name ++ "_xhat",tag,xhat)
    deviation : F.Tensor() <- norm_kept(name ++ "_deviation",tag,deviation)
    gamma : F.Tensor() <- norm_kept(name ++ "_gamma",tag,gamma)
    beta : F.Tensor() <- norm_kept(name ++ "_beta",tag,beta)
    mean : F.Tensor() <- norm_kept(name ++ "_mean",tag,mean)
    variance : F.Tensor() <- norm_kept(name ++ "_variance",tag,variance)
    norm_backward(name,tag,Norm.backward(dy,(xhat,deviation),gamma))

def norm_evaluated(+name: String,+tag: String,result: Norm.Evaluated()) -> IO(F.Tensor()):
  (y,(parameters,running)) = result
  norm_kept(name ++ "_evaluated",tag,y)

def norm_case(+name: String,+tag: String,+shape: List<&2,U32>,+channels: U32,+workers: U32) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    x : F.Tensor() <- F.load_raw(folder() ++ name ++ "_x.bin",shape)
    dy : F.Tensor() <- F.load_raw(folder() ++ name ++ "_dy.bin",shape)
    gamma : F.Tensor() <- F.load_raw(folder() ++ name ++ "_gamma.bin",[channels])
    beta : F.Tensor() <- F.load_raw(folder() ++ name ++ "_beta.bin",[channels])
    mean : F.Tensor() <- F.load_raw(folder() ++ name ++ "_mean.bin",[channels])
    variance : F.Tensor() <- F.load_raw(folder() ++ name ++ "_variance.bin",[channels])
    done : F.Tensor() <- norm_forward(name,tag,F.with_workers(dy,workers),
      Norm.forward(F.with_workers(x,workers),(F.with_workers(gamma,workers),F.with_workers(beta,workers)),
        (F.with_workers(mean,workers),F.with_workers(variance,workers)),normalization()))
    x : F.Tensor() <- F.load_raw(folder() ++ name ++ "_x.bin",shape)
    gamma : F.Tensor() <- F.load_raw(folder() ++ name ++ "_gamma.bin",[channels])
    beta : F.Tensor() <- F.load_raw(folder() ++ name ++ "_beta.bin",[channels])
    mean : F.Tensor() <- F.load_raw(folder() ++ name ++ "_mean.bin",[channels])
    variance : F.Tensor() <- F.load_raw(folder() ++ name ++ "_variance.bin",[channels])
    norm_evaluated(name,tag,Norm.evaluate(F.with_workers(x,workers),(F.with_workers(gamma,workers),F.with_workers(beta,workers)),
      (F.with_workers(mean,workers),F.with_workers(variance,workers)),normalization()))
'''
# What examples/training/learning.bend saves: the loss of every step and the layers after the last one.
LEARNED = ['losses'] + [f'{kind}{index}' for index in (1, 2) for kind in ('weights', 'bias', 'gamma', 'beta', 'mean', 'variance')]
CLASSIFY = '''
# CLASSIFY_STEPS steps of classify_case through the training layers: a
# Conv-BatchNorm-ReLU block, the global average pool and a 1x1 convolution to
# logits, cross entropy, SGD; then the logits of evaluate.
type Network is Type:
  End{}
  Then{layer: Train.Layer,rest: Network}

type Tape is Type:
  Blank{}
  Recorded{saved: Train.Saved,layer: Train.Layer,rest: Tape}

def recorded(result: F.Tensor() & (Train.Saved & Train.Layer),next: F.Tensor() -> Train.Saved -> Train.Layer -> F.Tensor() & Tape) -> F.Tensor() & Tape:
  (y,(saved,layer)) = result
  next(y,saved,layer)

def network_forward(network: Network,x: F.Tensor(),tape: Tape) -> F.Tensor() & Tape:
  match network:
    case End{}: (x,tape)
    case Then{layer,rest}: recorded(Train.forward(x,layer),y => saved => layer => network_forward(rest,y,Recorded{saved,layer,tape}))

def returned(result: F.Tensor() & Train.Layer,next: F.Tensor() -> Train.Layer -> Network) -> Network:
  (dx,layer) = result
  next(dx,layer)

def network_backward(tape: Tape,dy: F.Tensor(),network: Network) -> Network:
  match tape:
    case Blank{}: network
    case Recorded{saved,layer,rest}: returned(Train.backward(dy,saved,layer),dx => layer => network_backward(rest,dx,Then{layer,network}))

def network_updated(network: Network,+rate: F32) -> Network:
  match network:
    case End{}: End{}
    case Then{layer,rest}: Then{Train.sgd_step(layer,rate),network_updated(rest,rate)}

def prepended(layer: Train.Layer,result: F.Tensor() & Network) -> F.Tensor() & Network:
  (y,rest) = result
  (y,Then{layer,rest})

def evaluated(result: F.Tensor() & Train.Layer,next: F.Tensor() -> Train.Layer -> F.Tensor() & Network) -> F.Tensor() & Network:
  (y,layer) = result
  next(y,layer)

def network_evaluate(network: Network,x: F.Tensor()) -> F.Tensor() & Network:
  match network:
    case End{}: (x,End{})
    case Then{layer,rest}: evaluated(Train.evaluate(x,layer),y => layer => prepended(layer,network_evaluate(rest,y)))

def Classified() -> Type:
  F32 & (F.Tensor() & (F.Tensor() & Network))

def classify_lost(x: F.Tensor(),tape: Tape,result: F.Tensor() & (F.Tensor() & F.Tensor())) -> Classified():
  (loss,(dlogits,target)) = result
  (F.item_or(loss,0.0),(x,(target,network_updated(network_backward(tape,F.reshape(dlogits,[IMAGES,CLASSES,1,1]),End{}),CLASSIFY_RATE))))

def classify_predicted(x: F.Tensor(),target: F.Tensor(),result: F.Tensor() & Tape) -> Classified():
  (logits,tape) = result
  classify_lost(x,tape,Train.cross_entropy(F.reshape(logits,[IMAGES,CLASSES]),target))

def classify_step(xs: F.Tensor() & F.Tensor(),target: F.Tensor(),network: Network) -> Classified():
  (x,x_kept) = xs
  classify_predicted(x_kept,target,network_forward(network,x,Blank{}))

def classified_then(result: Classified(),next: F32 -> F.Tensor() -> F.Tensor() -> Network -> List<&2,F32> & (F.Tensor() & Network)) ->
  List<&2,F32> & (F.Tensor() & Network):
  (loss,(x,(target,network))) = result
  next(loss,x,target,network)

def classify_continued(+loss: F32,result: List<&2,F32> & (F.Tensor() & Network)) -> List<&2,F32> & (F.Tensor() & Network):
  (losses,rest) = result
  (loss <> losses,rest)

# The loss of every step, then the logits of evaluate and the network.
def classify_trained(remaining: Nat,x: F.Tensor(),target: F.Tensor(),network: Network) -> List<&2,F32> & (F.Tensor() & Network):
  match remaining:
    case 0n: ([],network_evaluate(network,x))
    case 1n+rest:
      classified_then(classify_step(F.copy(x),target,network),loss => x => target => network => classify_continued(loss,classify_trained(rest,x,target,network)))

def classify_saved(+tag: String,result: List<&2,F32> & (F.Tensor() & Network)) -> IO(F.Tensor()):
  (losses,(logits,network)) = result
  do IO<F.Tensor()>:
    losses : F.Tensor() <- norm_kept("classify_losses",tag,F.from_list([CLASSIFY_STEPS],losses))
    norm_kept("classify_logits",tag,logits)

def classify_load(+name: String,+shape: List<&2,U32>,+workers: U32) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    tensor : F.Tensor() <- F.load_raw(folder() ++ "classify_" ++ name ++ ".bin",shape)
    IO.pure(F.Tensor(),F.with_workers(tensor,workers))

def classify_case(+tag: String,+workers: U32) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    x : F.Tensor() <- classify_load("x",X_SHAPE,workers)
    labels : F.Tensor() <- classify_load("labels",[IMAGES],workers)
    weights : F.Tensor() <- classify_load("weights",WEIGHTS,workers)
    bias : F.Tensor() <- classify_load("bias",[CHANNELS],workers)
    head : F.Tensor() <- classify_load("head",HEAD,workers)
    head_bias : F.Tensor() <- classify_load("head_bias",[CLASSES],workers)
    classify_saved(tag,classify_trained(CLASSIFY_STEPSn,x,Train.one_hot(labels,CLASSES),
      Then{Train.conv2d_layer(weights,bias,1,1),Then{Train.batch_norm_layer(CHANNELS),Then{Train.activation_layer(Act.relu()),
        Then{Train.average_pool_layer(),Then{Train.conv2d_layer(head,head_bias,1,0),End{}}}}}}))

# The backward of a layer with what another kind of layer saved fails: it reads as the fallback.
def saved_mismatch(result: F.Tensor() & Train.Layer) -> F.Tensor():
  (dx,layer) = result
  F.from_list([1],[F.item_or(F.sum(dx),12345.0)])

def mismatch_forwarded(result: F.Tensor() & (Train.Saved & Train.Layer)) -> F.Tensor():
  (y,(saved,layer)) = result
  saved_mismatch(Train.backward(y,saved,Train.conv2d_layer(F.zeros([1,1,1,1]),F.zeros([1]),1,0)))
'''
CLASSIFY_STEPS = 5
CLASSIFY_RATE = 0.1
CLASSIFY_X, CLASSIFY_WEIGHTS, CLASSIFY_HEAD = (4, 3, 8, 8), (8, 3, 3, 3), (5, 8, 1, 1)


def classify_case():
    """Four 8x8 images of three channels and their class numbers, a 3x3 convolution of eight channels and a 1x1 head of five classes."""
    rng = np.random.default_rng(20261022)
    return dict(x=rng.normal(0, 1, CLASSIFY_X), labels=np.array([0, 3, 1, 4]), weights=rng.normal(0, 0.3, CLASSIFY_WEIGHTS),
                bias=rng.normal(0, 0.1, CLASSIFY_WEIGHTS[0]), head=rng.normal(0, 0.3, CLASSIFY_HEAD), head_bias=rng.normal(0, 0.1, CLASSIFY_HEAD[0]))


def classify_torch():
    """The losses of the same steps and the logits of eval mode with torch.nn.functional and torch.optim.SGD in FP32."""
    case = {name: torch.tensor(value.astype(F32)) for name, value in classify_case().items()}
    parameters = [case[name].clone().requires_grad_(True) for name in ('weights', 'bias', 'head', 'head_bias')]
    gamma, beta = torch.ones(CLASSIFY_WEIGHTS[0], requires_grad=True), torch.zeros(CLASSIFY_WEIGHTS[0], requires_grad=True)
    mean, variance = torch.zeros(CLASSIFY_WEIGHTS[0]), torch.ones(CLASSIFY_WEIGHTS[0])
    optimizer = torch.optim.SGD(parameters + [gamma, beta], lr=CLASSIFY_RATE)
    functional = torch.nn.functional

    def logits(training):
        weights, bias, head, head_bias = parameters
        y = functional.batch_norm(functional.conv2d(case['x'], weights, bias, padding=1), mean, variance, gamma, beta, training=training, momentum=MOMENTUM, eps=EPSILON)
        return functional.conv2d(functional.adaptive_avg_pool2d(functional.relu(y), 1), head, head_bias).reshape(CLASSIFY_X[0], CLASSIFY_HEAD[0])
    losses = []
    for _ in range(CLASSIFY_STEPS):
        loss = functional.cross_entropy(logits(True), case['labels'].long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    with torch.no_grad():
        return np.array(losses, F32), logits(False).numpy()
STEP_VALUES = ['loss', 'dx', 'dy1', 'dweights1', 'dbias1', 'dweights2', 'dbias2', 'weights1', 'bias1', 'weights2', 'bias2']
# name, shape of x
NORMS = [
    ('norm_small', [2, 3, 5, 7]),
    ('norm_row', [1, 4, 1, 9]),
    ('norm_wide', [3, 16, 24, 24]),
]
NORM_VALUES = ['y', 'xhat', 'deviation', 'gamma', 'beta', 'mean', 'variance', 'dx', 'dgamma', 'dbeta', 'gamma_back', 'evaluated']


def norm_inputs(rng, shape):
    channels = shape[1]
    return dict(x=rng.normal(0.3, 1.5, shape).astype(F32), dy=rng.normal(0, 1, shape).astype(F32),
                gamma=rng.normal(1, 0.2, channels).astype(F32), beta=rng.normal(0, 0.2, channels).astype(F32),
                mean=rng.normal(0, 0.1, channels).astype(F32), variance=rng.uniform(0.5, 1.5, channels).astype(F32))


def norm_expected(inputs):
    y, xhat, deviation, mean, variance = batch_norm_forward(inputs['x'], inputs['gamma'], inputs['beta'], inputs['mean'], inputs['variance'], EPSILON, MOMENTUM)
    dx, dgamma, dbeta = batch_norm_backward(inputs['dy'], xhat, deviation, inputs['gamma'])
    evaluated = batch_norm_evaluate(inputs['x'], inputs['gamma'], inputs['beta'], inputs['mean'], inputs['variance'], EPSILON)
    return dict(y=y, xhat=xhat, deviation=deviation, gamma=inputs['gamma'], beta=inputs['beta'], mean=mean, variance=variance,
                dx=dx, dgamma=dgamma, dbeta=dbeta, gamma_back=inputs['gamma'], evaluated=evaluated)


def generate():
    rng = np.random.default_rng(20261014)
    steps = []
    for name, shape, mean in LOSSES:
        for kind in ('y', 'target'):
            rng.normal(size=shape).astype('<f4').tofile(REPORT / f'{name}_{kind}.bin')
        steps += [f'    {name}_{workers} : F.Tensor() <- loss_case("{name}","{workers}",{shape},Loss.{"mean" if mean else "sum"}(),{workers})\n'
                  for workers in WORKERS]
    x, layers, target, rate = two_layer_case()
    for name, value in dict(x=x, target=target, weights1=layers[0][0], bias1=layers[0][1], weights2=layers[1][0], bias2=layers[1][1]).items():
        value.astype('<f4').tofile(REPORT / f'step_{name}.bin')
    steps += [f'    step_{workers} : F.Tensor() <- step_case("{workers}",{workers})\n' for workers in WORKERS]
    rng = np.random.default_rng(20261021)
    for name, shape in NORMS:
        for kind, value in norm_inputs(rng, shape).items():
            value.tofile(REPORT / f'{name}_{kind}.bin')
        steps += [f'    {name}_{workers} : F.Tensor() <- norm_case("{name}","{workers}",{shape},{shape[1]},{workers})\n' for workers in WORKERS]
    for name, value in classify_case().items():
        value.astype('<f4').tofile(REPORT / f'classify_{name}.bin')
    steps += [f'    classify_{workers} : F.Tensor() <- classify_case("{workers}",{workers})\n' for workers in WORKERS]
    learning_x, student, learning_target, _ = learning_case()
    learning = dict(x=learning_x, target=learning_target)
    for index, (weights, bias, stride, padding, (gamma, beta, mean, variance)) in enumerate(student, 1):
        learning.update({f'weights{index}': weights, f'bias{index}': bias, f'gamma{index}': gamma, f'beta{index}': beta,
                         f'mean{index}': mean, f'variance{index}': variance})
    for name, value in learning.items():
        value.astype('<f4').tofile(REPORT / f'learn_{name}.bin')
    SOURCE.write_text('''import Base
import ../../lib/math_activation.bend as Act
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_batch_norm.bend as Norm
import ../../lib/tensor_f32.bend as F
import ../../lib/tensor_loss.bend as Loss
import ../../lib/tensor_layer.bend as Layer
import ../../lib/tensor_optimizer.bend as Optimizer
import ../../lib/tensor_training.bend as Train

def folder() -> String:
  "out/results/training/"

def loss_saved(+name: String,+tag: String,result: F.Tensor() & (F.Tensor() & F.Tensor())) -> IO(F.Tensor()):
  (loss,(gradient,target)) = result
  do IO<F.Tensor()>:
    loss : F.Tensor() <- F.save_raw(folder() ++ name ++ "_loss_" ++ tag ++ ".bin",loss)
    loss : F.Tensor() <- F.expect(loss)
    gradient : F.Tensor() <- F.save_raw(folder() ++ name ++ "_gradient_" ++ tag ++ ".bin",gradient)
    gradient : F.Tensor() <- F.expect(gradient)
    target : F.Tensor() <- F.save_raw(folder() ++ name ++ "_returned_" ++ tag ++ ".bin",target)
    F.expect(target)

def loss_case(+name: String,+tag: String,+shape: List<&2,U32>,reduction: Loss.Reduction,+workers: U32) -> IO(F.Tensor()):
  do IO<F.Tensor()>:
    y : F.Tensor() <- F.load_raw(folder() ++ name ++ "_y.bin",shape)
    target : F.Tensor() <- F.load_raw(folder() ++ name ++ "_target.bin",shape)
    loss_saved(name,tag,Loss.mse(F.with_workers(y,workers),F.with_workers(target,workers),reduction))

# A target of another shape fails the loss and the gradient: both read as the fallback.
def mismatched(result: F.Tensor() & (F.Tensor() & F.Tensor())) -> F.Tensor():
  (loss,(gradient,target)) = result
  F.from_list([2],[F.item_or(loss,12345.0),F.item_or(F.sum(gradient),12345.0)])

''' + STEP.replace('X_SHAPE', str(list(x.shape))).replace('TARGET_SHAPE', str(list(target.shape))).replace('RATE', repr(rate))
        .replace('FIRST', f'{layers[0][2]},{layers[0][3]}').replace('SECOND', f'{layers[1][2]},{layers[1][3]}')
        .replace('WEIGHTS1', str(list(layers[0][0].shape))).replace('WEIGHTS2', str(list(layers[1][0].shape)))
        .replace('BIAS1', str(list(layers[0][1].shape))).replace('BIAS2', str(list(layers[1][1].shape)))
        + NORM.replace('EPSILON', np.format_float_positional(F32(EPSILON))).replace('MOMENTUM', np.format_float_positional(F32(MOMENTUM)))
        + CLASSIFY.replace('CLASSIFY_STEPS', str(CLASSIFY_STEPS)).replace('CLASSIFY_RATE', repr(CLASSIFY_RATE)).replace('X_SHAPE', str(list(CLASSIFY_X)))
        .replace('IMAGES', str(CLASSIFY_X[0])).replace('CLASSES', str(CLASSIFY_HEAD[0])).replace('CHANNELS', str(CLASSIFY_WEIGHTS[0]))
        .replace('WEIGHTS', str(list(CLASSIFY_WEIGHTS))).replace('HEAD', str(list(CLASSIFY_HEAD)))
        + '''
def main() -> IO(Unit):
  do IO<Unit>:
''' + ''.join(steps) + '''    failed : F.Tensor() <- F.save_raw(folder() ++ "loss_mismatch.bin",mismatched(Loss.mse(F.zeros([2,3]),F.zeros([3]),Loss.mean())))
    failed : F.Tensor() <- F.expect(failed)
    failed : F.Tensor() <- F.save_raw(folder() ++ "saved_mismatch.bin",mismatch_forwarded(Train.forward(F.zeros([2]),Train.activation_layer(Act.relu()))))
    failed : F.Tensor() <- F.expect(failed)
    IO.print("done")
''')


def compare():
    failures = []
    for name, shape, mean in LOSSES:
        y, target = (np.fromfile(REPORT / f'{name}_{kind}.bin', '<f4').reshape(shape) for kind in ('y', 'target'))
        loss, gradient = squared_error(y, target, mean)
        expected = dict(loss=np.asarray(loss, F32), gradient=gradient, returned=target)
        for workers in WORKERS:
            for kind, value in expected.items():
                actual = np.fromfile(REPORT / f'{name}_{kind}_{workers}.bin', np.uint32)
                if not np.array_equal(actual, np.asarray(value, '<f4').view(np.uint32).ravel()):
                    failures.append(f'{name} {kind} with {workers} workers')
    x, layers, target, rate = two_layer_case()
    values = step(x, layers, target, rate)
    for workers in WORKERS:
        for name in STEP_VALUES:
            actual = np.fromfile(REPORT / f'step_{name}_{workers}.bin', np.uint32)
            if not np.array_equal(actual, np.asarray(values[name], '<f4').view(np.uint32).ravel()):
                failures.append(f'step {name} with {workers} workers')
    if not np.array_equal(np.fromfile(REPORT / 'loss_mismatch.bin', '<f4'), np.full(2, 12345, F32)):
        failures.append('loss_mismatch')
    for name, shape in NORMS:
        inputs = {kind: np.fromfile(REPORT / f'{name}_{kind}.bin', '<f4').reshape(shape if kind in ('x', 'dy') else -1)
                  for kind in ('x', 'dy', 'gamma', 'beta', 'mean', 'variance')}
        expected = norm_expected(inputs)
        for workers in WORKERS:
            for kind in NORM_VALUES:
                actual = np.fromfile(REPORT / f'{name}_{kind}_{workers}.bin', np.uint32)
                if not np.array_equal(actual, np.asarray(expected[kind], '<f4').view(np.uint32).ravel()):
                    failures.append(f'{name} {kind} with {workers} workers')
    x, student, target, rate = learning_case()
    losses, learned = train(x, student, target, rate, STEPS)
    expected = dict(zip(LEARNED, [losses] + [value for weights, bias, stride, padding, statistics in learned for value in (weights, bias, *statistics)]))
    for workers in WORKERS:
        for name, value in expected.items():
            if not np.array_equal(np.fromfile(REPORT / f'learned_{name}_{workers}.bin', np.uint32), value.view(np.uint32).ravel()):
                failures.append(f'learning {name} with {workers} workers')
    if not losses[-1] < losses[0]:
        failures.append('learning does not lower the loss')
    if np.fromfile(REPORT / 'saved_mismatch.bin', '<f4')[0] != 12345:
        failures.append('saved_mismatch')
    classified = {name: [np.fromfile(REPORT / f'classify_{name}_{workers}.bin', np.uint32) for workers in WORKERS] for name in ('losses', 'logits')}
    if not all(np.array_equal(values[0], other) for values in classified.values() for other in values[1:]):
        failures.append('classify differs between worker counts')
    theirs = classify_torch()
    if not all(np.allclose(ours[0].view(F32).reshape(their.shape), their, rtol=1e-4, atol=1e-5) for ours, their in zip(classified.values(), theirs)):
        failures.append('classify differs from PyTorch')
    if not torch_agrees(x, student, target, rate, losses, learned):
        failures.append('learning differs from PyTorch')
    return failures, losses


def torch_agrees(x, student, target, rate, losses, learned):
    """The same steps with torch.nn.functional and torch.optim.SGD in FP32."""
    parameters = [torch.tensor(value, requires_grad=True)
                  for weights, bias, stride, padding, (gamma, beta, mean, variance) in student for value in (weights, bias, gamma, beta)]
    running = [torch.tensor(value) for weights, bias, stride, padding, (gamma, beta, mean, variance) in student for value in (mean, variance)]
    optimizer = torch.optim.SGD(parameters, lr=rate)
    theirs = []
    for _ in range(STEPS):
        y = torch.tensor(x)
        for index, (weights, bias, stride, padding, statistics) in enumerate(student):
            weights, bias, gamma, beta = parameters[4 * index:4 * index + 4]
            z = torch.nn.functional.conv2d(y, weights, bias, stride=stride, padding=padding)
            y = torch.nn.functional.silu(torch.nn.functional.batch_norm(z, running[2 * index], running[2 * index + 1], gamma, beta,
                                                                        training=True, momentum=MOMENTUM, eps=EPSILON))
        loss = torch.nn.functional.mse_loss(y, torch.tensor(target))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        theirs.append(loss.item())
    ours = [value for weights, bias, stride, padding, (gamma, beta, mean, variance) in learned for value in (weights, bias, gamma, beta)]
    statistics = [value for weights, bias, stride, padding, (gamma, beta, mean, variance) in learned for value in (mean, variance)]
    return np.allclose(losses, theirs, rtol=1e-5, atol=0) and all(
        np.allclose(value, parameter.detach().numpy(), rtol=1e-5, atol=1e-6) for value, parameter in zip(ours, parameters)) and all(
        np.allclose(value, theirs.numpy(), rtol=1e-5, atol=1e-6) for value, theirs in zip(statistics, running))


def main():
    for folder in (SOURCE.parent, BUILD, REPORT):
        folder.mkdir(parents=True, exist_ok=True)
    generate()
    generated = BUILD / 'training.c'
    subprocess.run(['node', 'scripts/bend_launcher.mjs', str(SOURCE), '-o', str(generated)], cwd=ROOT, env=dict(os.environ, BEND_NO_TELEMETRY='1'), check=True)
    cc = os.environ.get('CC') or shutil.which('clang-19') or shutil.which('clang')
    if not cc:
        raise SystemExit('The training cases need clang on PATH or CC.')
    subprocess.run([cc, '-O3', '-ffp-contract=off', '-std=c11', str(generated), '-lpthread', '-lm', '-o', str(BUILD / 'training')], check=True)
    subprocess.run([str(BUILD / 'training'), '--threads', str(max(WORKERS)), '--gpu', 'off'], cwd=ROOT, check=True, capture_output=True)
    learning = BUILD / 'learning.c'
    subprocess.run(['node', 'scripts/bend_launcher.mjs', str(EXAMPLE), '-o', str(learning)], cwd=ROOT, env=dict(os.environ, BEND_NO_TELEMETRY='1'), check=True)
    subprocess.run([cc, '-O3', '-ffp-contract=off', '-std=c11', str(learning), '-lpthread', '-lm', '-o', str(BUILD / 'learning')], check=True)
    for workers in WORKERS:
        subprocess.run([str(BUILD / 'learning'), '--threads', str(workers), '--gpu', 'off'], cwd=ROOT, check=True, capture_output=True)
    failures, losses = compare()
    assert not failures, failures
    print(f'{len(LOSSES)} loss cases, one training step of two layers, {len(NORMS)} batch normalization cases and {STEPS} learning steps'
          f' equal the reference bit for bit with {WORKERS} workers.')
    print(f'The {STEPS} learning steps lower the loss from {losses[0]:.6g} to {losses[-1]:.6g} and agree with PyTorch to rounding.')
    print(f'{CLASSIFY_STEPS} classifier steps of cross entropy through the average pool, and the logits of evaluate, agree with PyTorch'
          f' to rounding and are equal bit for bit with {WORKERS} workers.')


if __name__ == '__main__':
    main()
