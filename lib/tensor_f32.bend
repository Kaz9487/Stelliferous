# The everyday F32 tensor module: import this one module and call F.add,
# F.concat, F.check, ... without element-type arguments. A tensor is held in
# its tiled form: one axis blocked by eight, the lanes of one Tile8, the rest
# row-major. Values enter and leave as F32 at the boundary: construction,
# files, storage and the reads. Base determines operand order, rounding, NaN
# and signed-zero behavior.
import Base
import ./default_machine_limits.bend as Default
import ./machine_limits.bend as Machine
import ./storage_buffer.bend as Buffers
import ./math_activation.bend as Activations
import ./math_fp32.bend as Arithmetic
import ./math_sum.bend as Sum
import ./math_largest.bend as Largest
import ./math_random.bend as Random
import ./math_activation_lanes.bend as ActivationLanes
import ./tensor_access.bend as Access
import ./math_compare.bend as Compare
import ./tensor.bend as Tensors
import ./tensor_shape.bend as Shapes
import ./tensor_select.bend as Select
import ./tensor_layout.bend as Layout
import ./tensor_tiled.bend as Tiled
import ./tensor_grid.bend as Grid
import ./tensor_tiled_map.bend as TiledMap
import ./tensor_tiled_layout.bend as TiledLayout
import ./tensor_tiled_concat.bend as TiledConcat
import ./tensor_tiled_pool.bend as TiledPool
import ./tensor_tiled_reduce.bend as TiledReduce
import ./tensor_tiled_broadcast.bend as TiledBroadcast
import ./tensor_tiled_depthwise.bend as TiledDepthwise
import ./kernel_tile.bend as Tile
import ./math_gelu_lanes.bend as GeluLanes
import ./math_choice.bend as Choice
import ./tensor_convolution.bend as TensorConvolution
import ./tensor_matrix_products.bend as TiledProducts
import ./tensor_file.bend as Files
import ./tensor_model.bend as Models
import ./tensor_checkpoint.bend as Checkpoints

def Tensor() -> Type:
  Tiled.Tiled

def Problem() -> Data:
  Tensors.Problem

# Storage exchanged with foreign I/O; see from_storage and into_storage.
def Storage() -> Type:
  Buffers.Buffer<F32>

# Construction

def full(+shape: List<&2,U32>,+value: F32) -> Tensor():
  TiledMap.full(shape,value)

def zeros(+shape: List<&2,U32>) -> Tensor():
  Tiled.zeros(shape)

def ones(+shape: List<&2,U32>) -> Tensor():
  full(shape,1.0)

def from_list(+shape: List<&2,U32>,+values: List<&2,F32>) -> Tensor():
  Tiled.tiled(Tensors.values_of(~F32,shape,values,0.0))

def from_storage(+shape: List<&2,U32>,storage: Storage()) -> Tensor():
  Tiled.tiled(Tensors.from_storage(~F32,shape,storage,0.0))

def loaded(read: IO(Files.Tensor())) -> IO(Tensor()):
  do IO<Tensor()>:
    tensor : Files.Tensor() <- read
    IO.pure(Tensor(),Tiled.tiled(tensor))

# Raw little-endian FP32 files: the values of the shape in row-major order.
# A file that cannot be read, or does not match the shape, gives a failed tensor.
def load_raw(+path: String,+shape: List<&2,U32>) -> IO(Tensor()):
  loaded(Files.load(path,shape))

def save_raw(+path: String,tensor: Tensor()) -> IO(Tensor()):
  loaded(Files.save(path,Tiled.untiled(tensor)))

# One byte per value in row-major order, each an FP32 from 0 to 255, such as
# the records of CIFAR-10. A file that cannot be read, or does not match the
# shape, gives a failed tensor.
def load_u8(+path: String,+shape: List<&2,U32>) -> IO(Tensor()):
  loaded(Files.load_u8(path,shape))

# NumPy .npy files of FP32 values; the file carries the shape.
def load_npy(+path: String) -> IO(Tensor()):
  loaded(Files.load_npy(path))

def save_npy(+path: String,tensor: Tensor()) -> IO(Tensor()):
  loaded(Files.save_npy(path,Tiled.untiled(tensor)))

# Models: tensors by name, loaded once. In `do Model<T>:` a step takes a tensor
# from the weights with take, or puts a copy there with keep for a later take.
def Weights() -> Type:
  Models.Store

def Model(A: Type) -> Type:
  Models.Model(A)

def Model.pure(-A: Type,value: A) -> Model(A):
  Models.Model.pure(A,value)

def Model.bind(-A: Type,-B: Type,step: Model(A),next: A -> Model(B)) -> Model(B):
  Models.Model.bind(A,B,step,next)

def take(+name: String) -> Model(Tensor()):
  Models.take(name)

def keep(+name: String,tensor: Tensor()) -> Model(Tensor()):
  Models.keep(name,tensor)

# The names of the tensors in the weights, in their order; the weights stay.
def weight_names() -> Model(List<&2,String>):
  Models.listed_names()

# folder/names.txt lists one name per line; folder/<name>.npy holds its tensor.
def load_weights(+folder: String) -> IO(Weights()):
  Models.load(folder)

# A PyTorch checkpoint, the ZIP form torch.save writes: one tensor per
# state-dict entry under its PyTorch name. FP16 and BFloat16 values widen
# exactly, integers round to the nearest FP32. A file that cannot be read gives
# weights whose every take reports why.
def load_checkpoint(+path: String) -> IO(Weights()):
  Checkpoints.load(path)

def copy_weights(weights: Weights()) -> Weights() & Weights():
  Models.copy(weights)

def run(-A: Type,model: Model(A),weights: Weights()) -> A:
  Models.result(A,model,weights)

# The result with what remains of the weights, kept tensors included.
def run_keeping(-A: Type,model: Model(A),weights: Weights()) -> A & Weights():
  Models.run(A,model,weights)

def save_weights(+folder: String,weights: Weights()) -> IO(Weights()):
  Models.save(folder,weights)

# Execution context: operations may use count workers, like a tensor's device.
# Read the runtime's pool once with IO.thread_count(); results keep the count.
def with_workers(tensor: Tensor(),+count: U32) -> Tensor():
  Tiled.with_workers(tensor,count)

def workers(tensor: Tensor()) -> Tensor() & U32:
  Tiled.workers(tensor)

# Observation and exits

def shape(tensor: Tensor()) -> Tensor() & List<&2,U32>:
  Tiled.shape(tensor)

def check(tensor: Tensor()) -> Result<&1,&1,Tensor() & Problem(),Tensor()>:
  Tiled.check(tensor)

def describe(problem: Problem()) -> String:
  Tensors.describe(problem)

def expected(result: Result<&1,&1,Tensor() & Problem(),Tensor()>) -> IO(Tensor()):
  match result:
    case Done{tensor}: IO.pure(Tensor(),tensor)
    case Fail{(tensor,problem)}: IO.die(Tensor(),1,Tensors.describe(problem))

# Stops the program with the recorded problem, for scripts and examples.
def expect(tensor: Tensor()) -> IO(Tensor()):
  expected(Tiled.check(tensor))

def to_list(tensor: Tensor()) -> Result<&2,&2,Problem(),List<&2,F32>>:
  Tensors.to_list(~F32,Tiled.untiled(tensor))

def item(tensor: Tensor()) -> Result<&2,&2,Problem(),F32>:
  Tensors.item(~F32,Tiled.untiled(tensor))

# Reads with a fallback for a failed tensor, for callers that do not inspect
# the problem.
def to_list_or(tensor: Tensor(),fallback: List<&2,F32>) -> List<&2,F32>:
  Tensors.value_or(~List<&2,F32>,to_list(tensor),fallback)

def item_or(tensor: Tensor(),fallback: F32) -> F32:
  Tensors.value_or(~F32,item(tensor),fallback)

def into_storage(tensor: Tensor()) -> Result<&2,&1,Problem(),Storage()>:
  Tensors.into_storage(~F32,Tiled.untiled(tensor))

def expect_storage(tensor: Tensor()) -> IO(Storage()):
  Tensors.expect_storage(~F32,Tiled.untiled(tensor))

# The value at index of a storage and the storage back: 0.0 past its end.
def read_storage(storage: Storage(),+index: U32) -> Storage() & F32:
  Buffers.Buffer{data,+length,certificate} = storage
  Buffers.read_masked(~F32,0.0,U32.is_lt(index,U32.from_nat(length)),Buffers.Buffer{data,length,certificate},index)

# Operations consume their tensor arguments and return one result. Keep a
# tensor by copying it first; copy uses then, and shape uses then_shape.

def then(pair: Tensor() & Tensor(),body: Tensor() -> Tensor() -> Tensor()) -> Tensor():
  Buffers.bind_pair(~Tensor(),~Tensor(),~Tensor(),pair,body)

def then_shape(pair: Tensor() & List<&2,U32>,body: Tensor() -> List<&2,U32> -> Tensor()) -> Tensor():
  Buffers.bind_pair(~Tensor(),~List<&2,U32>,~Tensor(),pair,body)

def copy(tensor: Tensor()) -> Tensor() & Tensor():
  Tiled.copy(tensor)

# Binary elementwise operations reuse the left tensor's storage. The number
# after the name is the operation's scalar work for the band schedule.
# Operands of one shape combine cell by cell, with the lanes of the right on
# the axis of the left; operands of other extents broadcast on the cells
# (TiledBroadcast).
def binary_usable(~operation: F32 -> F32 -> F32,name: String,operations: F32,same: Bool,+ls: List<&2,U32>,+rs: List<&2,U32>,left: Tensor(),right: Tensor()) -> Tensor():
  match same:
    case True{}: TiledBroadcast.same(~operation,name,operations,left,right)
    case False{}: TiledBroadcast.combine(~operation,name,operations,ls,rs,left,right)

# A failed operand fails the call and keeps the left tensor.
def binary_status(~operation: F32 -> F32 -> F32,name: String,operations: F32,status: Tensors.Status,+ls: List<&2,U32>,+rs: List<&2,U32>,+workers: U32,
  left: Tensor(),right: Tensor()) -> Tensor():
  match status:
    case Tensors.Usable{}: binary_usable(~operation,name,operations,Shapes.same_extents(ls,rs),ls,rs,left,right)
    case Tensors.Failed{problem}: Tiled.with_workers(Tiled.with_status(Tensors.Failed{problem},left),workers)

# operation(left element, right element); operations counts its scalar work.
def binary(~operation: F32 -> F32 -> F32,name: String,operations: F32,left: Tensor(),right: Tensor()) -> Tensor():
  Tiled.with_metadata(~Tensor(),left,left => ls => lt => lw =>
    Tiled.with_metadata(~Tensor(),right,right => rs => rt => rw =>
      binary_status(~operation,name,operations,Tensors.first_failure(lt,rt),ls,rs,U32.max(lw,rw),left,right)))

def add(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~F32.add,"add",1.0,left,right)

def sub(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~F32.sub,"sub",1.0,left,right)

def mul(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~F32.mul,"mul",1.0,left,right)

# Base division: a zero divisor gives an infinity or NaN, never a failure.
def div(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~F32.div,"div",1.0,left,right)

def maximum(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~F32.max,"maximum",2.0,left,right)

def minimum(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~F32.min,"minimum",2.0,left,right)

# Elementwise functions of one tensor run on its tiled form. The number is
# the scalar work of one value for the band schedule.
def tiled_map(~Context: Data,~transform: Context -> Tile.Tile8 -> Tile.Tile8,context: Context,operations: F32,tensor: Tensor()) -> Tensor():
  TiledMap.map(~Context,~transform,context,operations,tensor)

def tiled_lanes(~operation: F32 -> F32,operations: F32,tensor: Tensor()) -> Tensor():
  tiled_map(~Unit,~TiledMap.lanes(~Unit,~Access.apply_element(~F32,~operation)),Unit{},operations,tensor)

# The operation of an element and the value.
def scalar_value(~operation: F32 -> F32 -> F32,value: F32,element: F32) -> F32:
  operation(element,value)

def tiled_scalar(~operation: F32 -> F32 -> F32,operations: F32,tensor: Tensor(),value: F32) -> Tensor():
  tiled_map(~F32,~TiledMap.lanes(~F32,~scalar_value(~operation)),value,operations,tensor)

def add_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  tiled_scalar(~F32.add,1.0,tensor,value)

def sub_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  tiled_scalar(~F32.sub,1.0,tensor,value)

def mul_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  tiled_scalar(~F32.mul,1.0,tensor,value)

def div_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  tiled_scalar(~F32.div,1.0,tensor,value)

def less(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~Compare.less_elements,"less",2.625,left,right)

def less_equal(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~Compare.less_equal_elements,"less_equal",2.625,left,right)

def greater(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~Compare.greater_elements,"greater",2.625,left,right)

def greater_equal(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~Compare.greater_equal_elements,"greater_equal",2.625,left,right)

def equal(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~Compare.equal_elements,"equal",1.375,left,right)

def not_equal(left: Tensor(),right: Tensor()) -> Tensor():
  binary(~Compare.not_equal_elements,"not_equal",1.375,left,right)

# Scalar comparisons share the eight-lane map and preserve the tensor layout.
# Costs count AVX2 arithmetic, comparison, selection and value-bit instructions
# in one eight-element round, divided by eight; moves and addressing are excluded.
def less_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  tiled_map(~F32,~TiledMap.lanes(~F32,~Compare.less),value,2.75,tensor)

def less_equal_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  tiled_map(~F32,~TiledMap.lanes(~F32,~Compare.less_equal),value,2.625,tensor)

def greater_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  tiled_map(~F32,~TiledMap.lanes(~F32,~Compare.greater),value,2.625,tensor)

def greater_equal_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  tiled_map(~F32,~TiledMap.lanes(~F32,~Compare.greater_equal),value,2.75,tensor)

def equal_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  tiled_map(~F32,~TiledMap.lanes(~F32,~Compare.equal),value,1.5,tensor)

def not_equal_scalar(tensor: Tensor(),+value: F32) -> Tensor():
  tiled_map(~F32,~TiledMap.lanes(~F32,~Compare.not_equal),value,1.5,tensor)

def select(mask: Tensor(),yes: Tensor(),no: Tensor()) -> Tensor():
  Select.select(mask,yes,no)

def clamp(tensor: Tensor(),+low: F32,+high: F32) -> Tensor():
  tiled_map(~Compare.Bounds,~TiledMap.lanes(~Compare.Bounds,~Compare.clamp),Compare.Bounds{low,high},0.5,tensor)

# Activations: Act.silu() and the other values of lib/math_activation.bend
# carry their own planner cost. activated applies one to a tiled form, lane by
# lane: Identity leaves the cells as they are, and Custom, which carries only
# a cost, reaches no caller of it. activate applies one to a tensor; relu,
# silu, ... below are its shorthands. The function of Custom reaches
# conv2d_custom or apply, and activate fails on it.
def lanes_of(~operation: F32 -> F32,operations: F32,tiled: Tiled.Tiled) -> Tiled.Tiled:
  TiledMap.map(~Unit,~TiledMap.lanes(~Unit,~Access.apply_element(~F32,~operation)),Unit{},operations,tiled)

def activated(tiled: Tiled.Tiled,+activation: Activations.Activation) -> Tiled.Tiled:
  match activation:
    case Activations.Identity{}: tiled
    case Activations.ReLU{}: lanes_of(~Activations.relu_value,Activations.operations(Activations.ReLU{}),tiled)
    case Activations.ReLU6{}: lanes_of(~Activations.relu6_value,Activations.operations(Activations.ReLU6{}),tiled)
    case Activations.LeakyReLU{+slope}:
      TiledMap.map(~F32,~TiledMap.lanes(~F32,~scalar_value(~Activations.leaky_relu_value)),slope,Activations.operations(Activations.LeakyReLU{slope}),tiled)
    case Activations.Sigmoid{}: lanes_of(~Activations.sigmoid_value,Activations.operations(Activations.Sigmoid{}),tiled)
    case Activations.SiLU{}: lanes_of(~Activations.silu_value,Activations.operations(Activations.SiLU{}),tiled)
    case Activations.Tanh{}: lanes_of(~Activations.tanh_value,Activations.operations(Activations.Tanh{}),tiled)
    case Activations.HardSigmoid{}: lanes_of(~Activations.hard_sigmoid_value,Activations.operations(Activations.HardSigmoid{}),tiled)
    case Activations.HardSwish{}: lanes_of(~Activations.hard_swish_value,Activations.operations(Activations.HardSwish{}),tiled)
    case Activations.GELUTanh{}: TiledMap.map(~Unit,~GeluLanes.tile,Unit{},Activations.operations(Activations.GELUTanh{}),tiled)
    case Activations.GELU{}: lanes_of(~Activations.gelu_value,Activations.operations(Activations.GELU{}),tiled)
    case Activations.Custom{count}: tiled

def activate(tensor: Tensor(),+activation: Activations.Activation) -> Tensor():
  match activation:
    case Activations.Identity{}: tensor
    case Activations.Custom{count}: Tiled.rejected("activate",tensor)
    case activation: activated(tensor,activation)

def relu(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.relu())

def relu6(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.relu6())

def leaky_relu(tensor: Tensor(),slope: F32) -> Tensor():
  activate(tensor,Activations.leaky_relu(slope))

def sigmoid(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.sigmoid())

def silu(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.silu())

def tanh(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.tanh())

def hard_sigmoid(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.hard_sigmoid())

def hard_swish(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.hard_swish())

# PyTorch's gelu(approximate='tanh').
def gelu_tanh(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.gelu_tanh())

# PyTorch's gelu with approximate='none'.
def gelu(tensor: Tensor()) -> Tensor():
  activate(tensor,Activations.gelu())

# The derivative of an activation at every element, by its own arithmetic.
def activate_derivative(tensor: Tensor(),+activation: Activations.Activation) -> Tensor():
  match activation:
    case Activations.Identity{}: tiled_lanes(~Activations.identity_derivative_value,Activations.derivative_operations(activation),tensor)
    case Activations.ReLU{}: tiled_lanes(~Activations.relu_derivative_value,Activations.derivative_operations(activation),tensor)
    case Activations.ReLU6{}: tiled_lanes(~Activations.relu6_derivative_value,Activations.derivative_operations(activation),tensor)
    case Activations.LeakyReLU{+slope}: tiled_map(~F32,~TiledMap.lanes(~F32,~(slope => value => Activations.leaky_relu_derivative_value(value,slope))),slope,Activations.derivative_operations(Activations.LeakyReLU{slope}),tensor)
    case Activations.HardSigmoid{}: tiled_lanes(~Activations.hard_sigmoid_derivative_value,Activations.derivative_operations(activation),tensor)
    case Activations.HardSwish{}: tiled_lanes(~Activations.hard_swish_derivative_value,Activations.derivative_operations(activation),tensor)
    case Activations.Sigmoid{}: tiled_lanes(~Activations.sigmoid_derivative_value,Activations.derivative_operations(activation),tensor)
    case Activations.Tanh{}: tiled_lanes(~Activations.tanh_derivative_value,Activations.derivative_operations(activation),tensor)
    case Activations.SiLU{}: tiled_map(~Unit,~ActivationLanes.silu,Unit{},Activations.derivative_operations(activation),tensor)
    case Activations.GELUTanh{}: tiled_map(~Unit,~ActivationLanes.gelu_tanh,Unit{},Activations.derivative_operations(activation),tensor)
    case Activations.GELU{}: tiled_lanes(~Activations.gelu_derivative_value,Activations.derivative_operations(activation),tensor)
    case activation: Tiled.rejected("activate_derivative",tensor)

def sigmoid_derivative(tensor: Tensor()) -> Tensor():
  activate_derivative(tensor,Activations.sigmoid())

def tanh_derivative(tensor: Tensor()) -> Tensor():
  activate_derivative(tensor,Activations.tanh())

def silu_derivative(tensor: Tensor()) -> Tensor():
  activate_derivative(tensor,Activations.silu())

def gelu_tanh_derivative(tensor: Tensor()) -> Tensor():
  activate_derivative(tensor,Activations.gelu_tanh())

def gelu_derivative(tensor: Tensor()) -> Tensor():
  activate_derivative(tensor,Activations.gelu())

def activated_pair(+activation: Activations.Activation,pair: Tensor() & Tensor()) -> Tensor() & Tensor():
  (first,second) = pair
  (activate(first,activation),second)

# activate beside its input, which activate_backward reads: (output, input).
def activate_forward(tensor: Tensor(),+activation: Activations.Activation) -> Tensor() & Tensor():
  activated_pair(activation,copy(tensor))

def backward_when(same: Bool,+activation: Activations.Activation,gradient: Tensor(),input: Tensor()) -> Tensor():
  match same:
    case True{}: mul(gradient,activate_derivative(input,activation))
    case False{}: Tiled.rejected("activate_backward",gradient)

def backward_shaped(+activation: Activations.Activation,gradient: Tensor() & List<&2,U32>,input: Tensor() & List<&2,U32>) -> Tensor():
  (gradient,shape) = gradient
  (input,input_shape) = input
  backward_when(Shapes.same_extents(shape,input_shape),activation,gradient,input)

# The gradient before an activation from the gradient after it and the
# activation's input of the same shape: gradient * activation'(input).
def activate_backward(gradient: Tensor(),input: Tensor(),+activation: Activations.Activation) -> Tensor():
  backward_shaped(activation,Tiled.shape(gradient),Tiled.shape(input))

# Ordered convolution over CHW or NCHW; the activation runs on each finished
# leaf while it is in cache.
def conv2d(input: Tensor(),weights: Tensor(),bias: Tensor(),stride: U32,padding: U32,activation: Activations.Activation) -> Tensor():
  TensorConvolution.conv2d(~Activations.identity_value,activation,input,weights,bias,stride,padding)

# Every channel of an input of shape [N,]C,H,W convolves with its own weights
# of shape 1,K,K (weights of shape C,1,K,K), adds its own bias (bias of shape
# C) and goes through the activation, as conv2d of that channel alone does.
# A Custom activation fails.
def depthwise_conv2d(input: Tensor(),weights: Tensor(),bias: Tensor(),stride: U32,padding: U32,+activation: Activations.Activation) -> Tensor():
  match activation:
    case Activations.Custom{count}: Tiled.rejected("depthwise_conv2d",input)
    case activation: activated(TiledDepthwise.depthwise_conv2d(input,weights,bias,stride,padding),activation)

# conv2d with a custom activation of the given scalar operations per element.
def conv2d_custom(~activation: F32 -> F32,operations: F32,input: Tensor(),weights: Tensor(),bias: Tensor(),stride: U32,padding: U32) -> Tensor():
  TensorConvolution.conv2d(~activation,Activations.Custom{operations},input,weights,bias,stride,padding)

# Custom elementwise functions are passed as direct templates, never records,
# with their scalar operations per element.
def apply(~operation: F32 -> F32,operations: F32,tensor: Tensor()) -> Tensor():
  tiled_lanes(~operation,operations,tensor)

# The function also receives the element's row-major index.
def apply_indexed(~operation: U32 -> F32 -> F32,operations: F32,tensor: Tensor()) -> Tensor():
  TiledMap.indexed(~operation,operations,tensor)

# start, start + 1, ... as a rank-1 tensor of count elements.
def arange(+start: F32,+count: U32) -> Tensor():
  add_scalar(apply_indexed(~(index => value => U32.to_f32(index)),1.0,zeros([count])),start)

# Random values by counter: element i depends only on the seed, the stream and
# i, in row-major order, so results do not depend on the worker count. Use one
# stream per purpose or step. Twenty rounds of five operations and twelve key
# additions make 112 a word, counted by eight lanes.
def random_uniform(+shape: List<&2,U32>,+seed: U32,+stream: U32) -> Tensor():
  tiled_map(~Random.Key,~TiledMap.lanes(~Random.Key,~Random.uniform_of_counter),Random.Seeded{seed,stream},14.375,
    apply_indexed(~Random.counter,1.0,zeros(shape)))

# Whole numbers below the bound as FP32 values, by the top of the product of
# the word and the bound. The bound is at most 2^24 for exact values.
def random_below(+shape: List<&2,U32>,+bound: U32,+seed: U32,+stream: U32) -> Tensor():
  tiled_map(~Random.Bounded,~TiledMap.lanes(~Random.Bounded,~Random.below_of_counter),Random.Bounded{bound,Random.Seeded{seed,stream}},15.875,
    apply_indexed(~Random.counter,1.0,zeros(shape)))

# Reductions over listed axes remove those axes; the reductions of all
# elements return a rank-0 tensor, so failures propagate like any result. A
# sum adds in the order of tensor_partials.bend: every row of the last axis
# folds with the sum of math_sum.bend (eight lanes of TwoSum and a fixed
# merge tree), and the rows merge in one tree over the reduced axes. The
# maximum and the minimum take the elements of the reduced axes in row-major
# order, from -inf and +inf. An axis is listed once.
def sum_axes(tensor: Tensor(),+axes: List<&2,U32>) -> Tensor():
  TiledReduce.sum_axes("sum_axes",tensor,axes)

def max_axes(tensor: Tensor(),+axes: List<&2,U32>) -> Tensor():
  TiledReduce.reduce_axes(~F32,~F32.max,~Tensors.same(~F32),"max_axes",1.0,F32.div(F32.neg(1.0),0.0),tensor,axes)

def min_axes(tensor: Tensor(),+axes: List<&2,U32>) -> Tensor():
  TiledReduce.reduce_axes(~F32,~F32.min,~Tensors.same(~F32),"min_axes",1.0,F32.div(1.0,0.0),tensor,axes)

# sum_axes divided by the number of elements it adds; no element gives NaN.
def mean_axes(tensor: Tensor(),+axes: List<&2,U32>) -> Tensor():
  then_shape(shape(tensor),tensor => extents =>
    +extents = extents
    div_scalar(TiledReduce.sum_axes("mean_axes",tensor,axes),
      U32.to_f32(Tensors.count_or_zero(Layout.extents_at(extents,Layout.axes_where(List.length(&2,U32,extents),0n,Layout.naturals(axes),True{}))))))

# The place along the axis of the first largest element, as an FP32; a NaN
# counts as the largest. The axis is removed.
def argmax(tensor: Tensor(),+axis: U32) -> Tensor():
  TiledReduce.reduce_axes(~Largest.Largest,~Largest.step,~Largest.place,"argmax",6.0,Largest.start(),tensor,[axis])

def sum(tensor: Tensor()) -> Tensor():
  TiledReduce.sum_all("sum",tensor)

def max(tensor: Tensor()) -> Tensor():
  TiledReduce.reduce_all(~F32,~F32.max,~Tensors.same(~F32),"max",1.0,F32.div(F32.neg(1.0),0.0),tensor)

def min(tensor: Tensor()) -> Tensor():
  TiledReduce.reduce_all(~F32,~F32.min,~Tensors.same(~F32),"min",1.0,F32.div(1.0,0.0),tensor)

# sum divided by the number of elements; an empty tensor gives NaN.
def mean(tensor: Tensor()) -> Tensor():
  then_shape(shape(tensor),tensor => extents =>
    +extents = extents
    div_scalar(TiledReduce.sum_all("mean",tensor),
      U32.to_f32(Tensors.count_or_zero(extents))))

# Layout: results are new tensors, except reshape, which relabels.

def reshape(tensor: Tensor(),+shape: List<&2,U32>) -> Tensor():
  TiledLayout.reshape(tensor,shape)

# The result's axis i is the argument's axis order[i].
def transpose(tensor: Tensor(),order: List<&2,U32>) -> Tensor():
  TiledLayout.transpose(tensor,order)

# The elements in reverse order along every listed axis.
def flip(tensor: Tensor(),axes: List<&2,U32>) -> Tensor():
  TiledLayout.flip(tensor,axes)

# count elements from start along axis, every step-th one.
def slice(tensor: Tensor(),axis: U32,start: U32,count: U32,step: U32) -> Tensor():
  TiledLayout.slice(tensor,axis,start,count,step)

# target with part written at start along axis.
def set_slice(target: Tensor(),axis: U32,start: U32,part: Tensor()) -> Tensor():
  TiledLayout.set_slice(target,axis,start,part)

def Parts() -> Type:
  TiledConcat.Parts

# Build the parts of concat: F.cons(a, F.cons(b, F.single(c))).
def single(tensor: Tensor()) -> Parts():
  TiledConcat.Single{tensor}

def cons(tensor: Tensor(),rest: Parts()) -> Parts():
  TiledConcat.Cons{tensor,rest}

def concat(parts: Parts(),+axis: U32) -> Tensor():
  TiledConcat.concat(parts,axis)

# Repeats each element of the last two axes scale times in each direction.
def upsample_nearest(tensor: Tensor(),+scale: U32) -> Tensor():
  TiledLayout.upsample_nearest(tensor,scale)

# Over the last two axes; positions outside the input count as -inf.
def max_pool2d_with(tensor: Tensor(),+kernel: U32,+stride: U32,+padding: U32) -> Tensor():
  TiledPool.pool2d(~F32.max,"max_pool2d",F32.div(F32.neg(1.0),0.0),tensor,kernel,stride,padding)

# Windows of kernel by kernel that do not overlap: stride kernel, padding 0.
def max_pool2d(tensor: Tensor(),+kernel: U32) -> Tensor():
  max_pool2d_with(tensor,kernel,kernel,0)

# Broadcasting and axis operations.

# The tensor repeated to shape under NumPy broadcasting; add, sub, mul, max
# and min broadcast their operands in the same way.
def broadcast_to(tensor: Tensor(),+shape: List<&2,U32>) -> Tensor():
  TiledLayout.broadcast_to(tensor,shape)

# Elementwise functions share the tiled map of the activations.
def exp(tensor: Tensor()) -> Tensor():
  tiled_lanes(~Arithmetic.exp,4.5,tensor)

def log(tensor: Tensor()) -> Tensor():
  tiled_lanes(~Arithmetic.log,8.25,tensor)

# One square-root instruction per eight lanes.
def sqrt(tensor: Tensor()) -> Tensor():
  tiled_lanes(~Arithmetic.sqrt,0.125,tensor)

# ---- Softmax ----

def exp_centered(value: F32,maximum: F32) -> F32:
  Arithmetic.exp(F32.sub(value,maximum))

def softmax_maximum(maximum: F32,value: F32) -> F32:
  Choice.larger(maximum,value)

# The exponentials of the values less the maximum of their line.
def softmax_exponentials(+axis: U32,+unit: List<&2,U32>,pair: Tensor() & Tensor()) -> Tensor():
  (values,for_maximum) = pair
  binary(~exp_centered,"sub",5.5,values,reshape(TiledReduce.reduce_axes(~F32,~softmax_maximum,~Tensors.same(~F32),"softmax",1.0,F32.div(F32.neg(1.0),0.0),for_maximum,[axis]),unit))

# The exponentials divided by the sum of their line.
def softmax_divided(+axis: U32,+unit: List<&2,U32>,pair: Tensor() & Tensor()) -> Tensor():
  (values,for_sum) = pair
  div(values,reshape(TiledReduce.reduce_axes(~Sum.Lanes,~Sum.step,~Sum.total,"softmax",6.0,Sum.zero(),for_sum,[axis]),unit))

def softmax_lines(+axis: U32,+shape: List<&2,U32>,tensor: Tensor()) -> Tensor():
  +unit = Layout.with_extent(shape,U32.to_nat(axis),1)
  softmax_divided(axis,unit,copy(softmax_exponentials(axis,unit,copy(tensor))))

def softmax_valid(valid: Bool,empty: Bool,+shape: List<&2,U32>,+axis: U32,tensor: Tensor()) -> Tensor():
  match valid empty:
    case True{} False{}: softmax_lines(axis,shape,tensor)
    case True{} True{}: tensor
    case False{} empty: Tiled.rejected("softmax",tensor)

# Along the axis, every line of values folds twice and maps once: the maximum
# from -inf, then the sum of math_sum.bend of the exponentials of the values
# less the maximum, then each such exponential divided by the sum.
def softmax(tensor: Tensor(),+axis: U32) -> Tensor():
  Tiled.with_metadata(~Tensor(),tensor,tensor => shape => status => workers =>
    +shape = shape
    softmax_valid(Tensors.usable(status) && Nat.is_lt(U32.to_nat(axis),List.length(&2,U32,shape)),Shapes.has_zero(shape),shape,axis,tensor))

# before[i] new elements in front of axis i and after[i] behind it, all equal
# to value.
def pad(tensor: Tensor(),+before: List<&2,U32>,+after: List<&2,U32>,+value: F32) -> Tensor():
  TiledLayout.pad(Layout.Constant{value},tensor,before,after)

# The same, each new element repeating the nearest element of the tensor.
def pad_edge(tensor: Tensor(),+before: List<&2,U32>,+after: List<&2,U32>) -> Tensor():
  TiledLayout.pad(Layout.Edge{},tensor,before,after)

# ---- Matrix products ----

# The extents before the last two, and the last two.
type Factors is Data:
  Factors{leading: List<&2,U32>,rows: U32,inner: U32}

def factors_extended(+extent: U32,found: Maybe<&2,Factors>) -> Maybe<&2,Factors>:
  match found:
    case None{}: None{}
    case Some{Factors{leading,rows,inner}}: Some{Factors{extent <> leading,rows,inner}}

def factors(shape: List<&2,U32>) -> Maybe<&2,Factors>:
  match shape:
    case []: None{}
    case [extent]: None{}
    case [rows,inner]: Some{Factors{[],rows,inner}}
    case extent <> rest: factors_extended(extent,factors(rest))

# left [count,rows,inner], right with its outputs at right_axis and bias [outputs]:
# the products [count,rows,outputs] of each image, each output adding them
# from +0.0 and then its bias.
def products(left: Tensor(),right: Tensor(),bias: Tensor(),+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32,+right_axis: Nat) -> Tensor():
  TiledProducts.lines(count,rows,inner,outputs,workers,right_axis,left,right,bias)

# One right for all of left: its matrices are the rows of one matrix.
def products_shared(left: Tensor(),right: Tensor(),bias: Tensor(),+count: U32,+rows: U32,+inner: U32,+outputs: U32,+workers: U32) -> Tensor():
  +all_rows = (count * rows : U32)
  products(reshape(left,[1,all_rows,inner]),right,bias,1,all_rows,inner,outputs,workers,0n)

# images of image_rows rows each: one image of all rows when one right is shared.
def matmul_inner(valid: Bool,left: Tensor(),right: Tensor(),+images: U32,+image_rows: U32,+leading: List<&2,U32>,+rows: U32,+inner: U32,+outputs: U32,+workers: U32) -> Tensor():
  match valid:
    case False{}: Tiled.rejected("matmul",left)
    case True{}:
      reshape(products(reshape(left,[images,image_rows,inner]),reshape(right,[images,inner,outputs]),zeros([outputs]),images,image_rows,inner,outputs,workers,2n),
        List.append(&2,U32,leading,[rows,outputs]))

def matmul_found(left_factors: Maybe<&2,Factors>,right_factors: Maybe<&2,Factors>,+workers: U32,+limits: Machine.Limits,left: Tensor(),right: Tensor()) -> Tensor():
  match left_factors right_factors:
    case Some{Factors{+leading,+rows,+inner}} Some{Factors{+right_leading,+right_inner,+outputs}}:
      +count = Tensors.count_or_zero(leading)
      +shared = List.is_empty(&2,U32,right_leading)
      +images = Bool.pick(U32,shared,1,count)
      +image_rows = Bool.pick(U32,shared,(count * rows : U32),rows)
      matmul_inner(U32.is_eq(inner,right_inner) && (Shapes.same_extents(right_leading,leading) || shared) && TiledProducts.supported(images,image_rows,inner,outputs,limits),
        left,right,images,image_rows,leading,rows,inner,outputs,workers)
    case left_factors right_factors: Tiled.rejected("matmul",left)

def matmul_status(status: Tensors.Status,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+workers: U32,left: Tensor(),right: Tensor()) -> Tensor():
  match status:
    case Tensors.Usable{}: matmul_found(factors(left_shape),factors(right_shape),workers,Default.limits(),left,right)
    case Tensors.Failed{problem}: Tiled.with_workers(Tiled.with_status(Tensors.Failed{problem},left),workers)

# left [...,m,k] and right [...,k,n], with the same axes in front or none on the
# right: at each row i and column j, the products left[i,k] * right[k,j] added in
# increasing k from +0.0, as a convolution adds them.
def matmul(left: Tensor(),right: Tensor()) -> Tensor():
  Tiled.with_metadata(~Tensor(),left,left => ls => lt => lw =>
    Tiled.with_metadata(~Tensor(),right,right => rs => rt => rw =>
      matmul_status(Tensors.first_failure(lt,rt),ls,rs,U32.max(lw,rw),left,right)))

def linear_shaped(valid: Bool,x: Tensor(),weight: Tensor(),bias: Tensor(),+leading: List<&2,U32>,+rows: U32,+inputs: U32,+outputs: U32,+workers: U32) -> Tensor():
  match valid:
    case False{}: Tiled.rejected("linear",x)
    case True{}:
      reshape(products_shared(x,weight,bias,Tensors.count_or_zero(leading),rows,inputs,outputs,workers),List.append(&2,U32,leading,[rows,outputs]))

def linear_found(found: Maybe<&2,Factors>,+inputs: U32,+outputs: U32,+bias_shape: List<&2,U32>,+workers: U32,+limits: Machine.Limits,x: Tensor(),weight: Tensor(),bias: Tensor()) -> Tensor():
  match found:
    case None{}: Tiled.rejected("linear",x)
    case Some{Factors{+leading,+rows,+own}}:
      linear_shaped(U32.is_eq(own,inputs) && Shapes.same_extents(bias_shape,[outputs]) && TiledProducts.supported(1,(Tensors.count_or_zero(leading) * rows : U32),inputs,outputs,limits),
        x,weight,bias,leading,rows,inputs,outputs,workers)

def linear_weight(+weight_shape: List<&2,U32>,+shape: List<&2,U32>,+bias_shape: List<&2,U32>,+workers: U32,x: Tensor(),weight: Tensor(),bias: Tensor()) -> Tensor():
  match weight_shape:
    case [outputs,inputs]: linear_found(factors(shape),inputs,outputs,bias_shape,workers,Default.limits(),x,weight,bias)
    case other: Tiled.rejected("linear",x)

def linear_status(status: Tensors.Status,+weight_shape: List<&2,U32>,+shape: List<&2,U32>,+bias_shape: List<&2,U32>,+workers: U32,x: Tensor(),weight: Tensor(),bias: Tensor()) -> Tensor():
  match status:
    case Tensors.Usable{}: linear_weight(weight_shape,shape,bias_shape,workers,x,weight,bias)
    case Tensors.Failed{problem}: Tiled.with_workers(Tiled.with_status(Tensors.Failed{problem},x),workers)

# x [...,rows,inputs], weight [outputs,inputs] and bias [outputs], as PyTorch stores
# them: x weight^T + bias, each output adding its products in increasing input
# from +0.0 and then the bias.
def linear(x: Tensor(),weight: Tensor(),bias: Tensor()) -> Tensor():
  Tiled.with_metadata(~Tensor(),x,x => shape => status => workers =>
    Tiled.with_metadata(~Tensor(),weight,weight => weight_shape => weight_status => weight_workers =>
      Tiled.with_metadata(~Tensor(),bias,bias => bias_shape => bias_status => bias_workers =>
        linear_status(Tensors.first_failure(status,Tensors.first_failure(weight_status,bias_status)),weight_shape,shape,bias_shape,U32.max(workers,U32.max(weight_workers,bias_workers)),x,weight,bias))))
