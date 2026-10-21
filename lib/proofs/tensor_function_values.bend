# The public eight-lane functions of one tensor are the ordered map of their
# scalar function, in any layout. The mapper law is discharged here, not left
# as a premise of the public result. F.activate and F.activate_derivative of
# every activation hold the scalar activation or derivative of each value,
# F.activate_forward and F.activate_backward the pair a training layer uses.
import Base
import ../tensor_f32.bend as F32Tensor
import ../tensor.bend as Tensors
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../tensor_band.bend as Bands
import ../storage_buffer.bend as Storage
import ../math_activation.bend as Activations
import ../math_fp32.bend as Arithmetic
import ../math_activation_lanes.bend as ActivationLanes
import ../kernel_tile.bend as Tile
import ./tensor_operations_refinement.bend as OperationLaws
import ./tensor_band_map.bend as BandMap
import ./tensor_band_model.bend as Model
import ./tensor_band_normal_unary.bend as NormalUnary
import ./tensor_global_shape_count.bend as Global
import ../tensor_gelu.bend as Gelu
import ./tensor_gelu_values.bend as GeluValues
import ./tensor_contract.bend as Contract
import ./kernel_pipeline_activation.bend as ActivationValues
import ./word_encoding.bend as Words
import ../tensor_shape.bend as Shapes
import ./storage_shape.bend as Shape
import ./tensor_copy_contract.bend as Copy
import ./tensor_band_combine_leaf.bend as Leaf
import ./tensor_combine_public.bend as CombinePublic
import ./tensor_ready_flow.bend as Flow
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_unit_broadcast.bend as UnitBroadcast

# A fixed operation context is arbitrary, including every leaky slope.
def ContextPublicLaw(~Context: Data,~context: Context,~Logical: Data,~original: Logical -> U32 -> F32,~operation: Context -> F32 -> F32,~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32>) -> Type:
  @+logical: Logical -> @+bands: Model.Model<F32> -> @+geometry: Bands.Geometry -> @+shape: List<&2,U32> -> @+fill: F32 -> @+workers: U32 ->
  @contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}) ->
  Global.Contract(~F32,~BandMap.Context<Context,Logical>,~BandMap.value(~F32,~Context,~Logical,~operation,~original),BandMap.Context{context,logical},
    call(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}))

def public_context(~Context: Data,~context: Context,~Logical: Data,~original: Logical -> U32 -> F32,~operation: Context -> F32 -> F32,~mapper: @+count: Nat -> @+position: U32 -> @+context: Context -> Storage.Buffer<F32> -> Storage.Buffer<F32>,~lawful: OperationLaws.MapLaw_context(~F32,~Context,~operation,~mapper),~cost: F32) ->
  ContextPublicLaw(~Context,~context,~Logical,~original,~operation,~Tensors.apply_context(~F32,~Context,~mapper,context,cost)):
  logical => bands => geometry => shape => fill => workers => contract =>
    NormalUnary.apply_context(~F32,~Context,~Logical,~operation,~mapper,~lawful,~original,context,logical,cost,bands,geometry,shape,fill,workers,contract)

def PublicLaw(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32,~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32>,~name: String) -> Type:
  ContextPublicLaw(~Unit,~Unit{},~Logical,~original,~Access.apply_element(~F32,~operation),~call)

def public(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32,~mapper: @+count: Nat -> @+position: U32 -> Storage.Buffer<F32> -> Storage.Buffer<F32>,~lawful: OperationLaws.MapLaw(~F32,~operation,~mapper),~name: String,~cost: F32) ->
  PublicLaw(~Logical,~original,~operation,~Tensors.apply(~F32,~mapper,cost),~name):
  public_context(~Unit,~Unit{},~Logical,~original,~Access.apply_element(~F32,~operation),~Operations.without_context(~F32,~mapper),~OperationLaws.unit_map_law(~F32,~operation,~mapper,~lawful),~cost)

def sigmoid(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.sigmoid_value,~F32Tensor.sigmoid,~"sigmoid"):
  public(~Logical,~original,~Activations.sigmoid_value,~Operations.mapped_lanes(~Activations.sigmoid_value),~OperationLaws.mapped_lanes(~Activations.sigmoid_value),~"sigmoid",~Activations.operations(Activations.Sigmoid{}))

def silu(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.silu_value,~F32Tensor.silu,~"silu"):
  public(~Logical,~original,~Activations.silu_value,~Operations.mapped_lanes(~Activations.silu_value),~OperationLaws.mapped_lanes(~Activations.silu_value),~"silu",~Activations.operations(Activations.SiLU{}))

def tanh(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.tanh_value,~F32Tensor.tanh,~"tanh"):
  public(~Logical,~original,~Activations.tanh_value,~Operations.mapped_lanes(~Activations.tanh_value),~OperationLaws.mapped_lanes(~Activations.tanh_value),~"tanh",~Activations.operations(Activations.Tanh{}))

def gelu_tanh(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.gelu_tanh_value,~F32Tensor.gelu_tanh,~"gelu_tanh"):
  public(~Logical,~original,~Activations.gelu_tanh_value,~Gelu.mapped,~GeluValues.mapped,~"gelu_tanh",~Activations.operations(Activations.GELUTanh{}))

def exp(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Arithmetic.exp,~F32Tensor.exp,~"exp"):
  public(~Logical,~original,~Arithmetic.exp,~Operations.mapped_lanes(~Arithmetic.exp),~OperationLaws.mapped_lanes(~Arithmetic.exp),~"exp",~4.5)

def log(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Arithmetic.log,~F32Tensor.log,~"log"):
  public(~Logical,~original,~Arithmetic.log,~Operations.mapped_lanes(~Arithmetic.log),~OperationLaws.mapped_lanes(~Arithmetic.log),~"log",~8.25)

def sqrt(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Arithmetic.sqrt,~F32Tensor.sqrt,~"sqrt"):
  public(~Logical,~original,~Arithmetic.sqrt,~Operations.mapped_lanes(~Arithmetic.sqrt),~OperationLaws.mapped_lanes(~Arithmetic.sqrt),~"sqrt",~0.125)

def identity_derivative(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.identity_derivative_value,~(tensor => F32Tensor.activate_derivative(tensor,Activations.Identity{})),~"identity_derivative"):
  public(~Logical,~original,~Activations.identity_derivative_value,~Operations.mapped_lanes(~Activations.identity_derivative_value),~OperationLaws.mapped_lanes(~Activations.identity_derivative_value),~"identity_derivative",~Activations.derivative_operations(Activations.Identity{}))

def relu_derivative(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.relu_derivative_value,~(tensor => F32Tensor.activate_derivative(tensor,Activations.ReLU{})),~"relu_derivative"):
  public(~Logical,~original,~Activations.relu_derivative_value,~Operations.mapped_lanes(~Activations.relu_derivative_value),~OperationLaws.mapped_lanes(~Activations.relu_derivative_value),~"relu_derivative",~Activations.derivative_operations(Activations.ReLU{}))

def relu6_derivative(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.relu6_derivative_value,~(tensor => F32Tensor.activate_derivative(tensor,Activations.ReLU6{})),~"relu6_derivative"):
  public(~Logical,~original,~Activations.relu6_derivative_value,~Operations.mapped_lanes(~Activations.relu6_derivative_value),~OperationLaws.mapped_lanes(~Activations.relu6_derivative_value),~"relu6_derivative",~Activations.derivative_operations(Activations.ReLU6{}))

def hard_sigmoid_derivative(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.hard_sigmoid_derivative_value,~(tensor => F32Tensor.activate_derivative(tensor,Activations.HardSigmoid{})),~"hard_sigmoid_derivative"):
  public(~Logical,~original,~Activations.hard_sigmoid_derivative_value,~Operations.mapped_lanes(~Activations.hard_sigmoid_derivative_value),~OperationLaws.mapped_lanes(~Activations.hard_sigmoid_derivative_value),~"hard_sigmoid_derivative",~Activations.derivative_operations(Activations.HardSigmoid{}))

def hard_swish_derivative(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.hard_swish_derivative_value,~(tensor => F32Tensor.activate_derivative(tensor,Activations.HardSwish{})),~"hard_swish_derivative"):
  public(~Logical,~original,~Activations.hard_swish_derivative_value,~Operations.mapped_lanes(~Activations.hard_swish_derivative_value),~OperationLaws.mapped_lanes(~Activations.hard_swish_derivative_value),~"hard_swish_derivative",~Activations.derivative_operations(Activations.HardSwish{}))

def leaky_relu_derivative(~slope: F32,~Logical: Data,~original: Logical -> U32 -> F32) ->
  ContextPublicLaw(~F32,~slope,~Logical,~original,~(slope => value => Activations.leaky_relu_derivative_value(value,slope)),~(tensor => F32Tensor.activate_derivative(tensor,Activations.LeakyReLU{slope}))):
  public_context(~F32,~slope,~Logical,~original,~(slope => value => Activations.leaky_relu_derivative_value(value,slope)),~Operations.lanes(~F32,~(slope => value => Activations.leaky_relu_derivative_value(value,slope))),~OperationLaws.lanes(~F32,~(slope => value => Activations.leaky_relu_derivative_value(value,slope))),~Activations.derivative_operations(Activations.LeakyReLU{slope}))

def sigmoid_derivative(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.sigmoid_derivative_value,~F32Tensor.sigmoid_derivative,~"sigmoid_derivative"):
  public(~Logical,~original,~Activations.sigmoid_derivative_value,~Operations.mapped_lanes(~Activations.sigmoid_derivative_value),~OperationLaws.mapped_lanes(~Activations.sigmoid_derivative_value),~"sigmoid_derivative",~Activations.derivative_operations(Activations.Sigmoid{}))

def tanh_derivative(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.tanh_derivative_value,~F32Tensor.tanh_derivative,~"tanh_derivative"):
  public(~Logical,~original,~Activations.tanh_derivative_value,~Operations.mapped_lanes(~Activations.tanh_derivative_value),~OperationLaws.mapped_lanes(~Activations.tanh_derivative_value),~"tanh_derivative",~Activations.derivative_operations(Activations.Tanh{}))

# The staged arithmetic changes only the implementation of one eight-value cell.
def silu_derivative_tile(+context: Unit,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,+g: F32,+h: F32) ->
  {ActivationLanes.silu(context,Tile.Tile{a,b,c,d,e,f,g,h}) == Tile.Tile{Activations.silu_derivative_value(a),Activations.silu_derivative_value(b),Activations.silu_derivative_value(c),Activations.silu_derivative_value(d),Activations.silu_derivative_value(e),Activations.silu_derivative_value(f),Activations.silu_derivative_value(g),Activations.silu_derivative_value(h)} : Tile.Tile8}:
  {==}

def silu_derivative(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.silu_derivative_value,~F32Tensor.silu_derivative,~"silu_derivative"):
  public_context(~Unit,~Unit{},~Logical,~original,~Access.apply_element(~F32,~Activations.silu_derivative_value),~Operations.lanes_tiles(~Unit,~Access.apply_element(~F32,~Activations.silu_derivative_value),~ActivationLanes.silu),~OperationLaws.lanes_tiles(~Unit,~Access.apply_element(~F32,~Activations.silu_derivative_value),~ActivationLanes.silu,~silu_derivative_tile),~Activations.derivative_operations(Activations.SiLU{}))

def gelu_tanh_derivative_tile(+context: Unit,+a: F32,+b: F32,+c: F32,+d: F32,+e: F32,+f: F32,+g: F32,+h: F32) ->
  {ActivationLanes.gelu_tanh(context,Tile.Tile{a,b,c,d,e,f,g,h}) == Tile.Tile{Activations.gelu_tanh_derivative_value(a),Activations.gelu_tanh_derivative_value(b),Activations.gelu_tanh_derivative_value(c),Activations.gelu_tanh_derivative_value(d),Activations.gelu_tanh_derivative_value(e),Activations.gelu_tanh_derivative_value(f),Activations.gelu_tanh_derivative_value(g),Activations.gelu_tanh_derivative_value(h)} : Tile.Tile8}:
  {==}

def gelu_tanh_derivative(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Activations.gelu_tanh_derivative_value,~F32Tensor.gelu_tanh_derivative,~"gelu_tanh_derivative"):
  public_context(~Unit,~Unit{},~Logical,~original,~Access.apply_element(~F32,~Activations.gelu_tanh_derivative_value),~Operations.lanes_tiles(~Unit,~Access.apply_element(~F32,~Activations.gelu_tanh_derivative_value),~ActivationLanes.gelu_tanh),~OperationLaws.lanes_tiles(~Unit,~Access.apply_element(~F32,~Activations.gelu_tanh_derivative_value),~ActivationLanes.gelu_tanh,~gelu_tanh_derivative_tile),~Activations.derivative_operations(Activations.GELUTanh{}))

# ---- Every activation and its derivative ----

# The activation and the reader of the values it is applied to.
type ActivationRead<-Logical: Data> is Data:
  ActivationRead{activation: Activations.Activation,logical: Logical}

# The activation of the reader's value, as the fused convolution activation
# computes it.
def activated_value(~Logical: Data,~original: Logical -> U32 -> F32,read: ActivationRead<Logical>,index: U32) -> F32:
  ActivationRead{+activation,+logical} = read
  ActivationValues.activation_value(~Activations.identity_value,activation,original(logical,index))

# The scalar derivative of each activation. A custom activation has none:
# its call is rejected.
def derivative_value(activation: Activations.Activation,value: F32) -> F32:
  match activation:
    case Activations.Identity{}: Activations.identity_derivative_value(value)
    case Activations.ReLU{}: Activations.relu_derivative_value(value)
    case Activations.ReLU6{}: Activations.relu6_derivative_value(value)
    case Activations.LeakyReLU{slope}: Activations.leaky_relu_derivative_value(value,slope)
    case Activations.Sigmoid{}: Activations.sigmoid_derivative_value(value)
    case Activations.SiLU{}: Activations.silu_derivative_value(value)
    case Activations.Tanh{}: Activations.tanh_derivative_value(value)
    case Activations.HardSigmoid{}: Activations.hard_sigmoid_derivative_value(value)
    case Activations.HardSwish{}: Activations.hard_swish_derivative_value(value)
    case Activations.GELUTanh{}: Activations.gelu_tanh_derivative_value(value)
    case Activations.Custom{operations}: value

def derivative_at(~Logical: Data,~original: Logical -> U32 -> F32,read: ActivationRead<Logical>,index: U32) -> F32:
  ActivationRead{+activation,+logical} = read
  derivative_value(activation,original(logical,index))

# A Ready tensor of the contract of a reader, read through another reader
# that agrees with it at every index.
def read_as(~Before: Data,~After: Data,~before_value: Before -> U32 -> F32,~after_value: After -> U32 -> F32,+before: Before,+after: After,
  agrees: @+index: U32 -> {before_value(before,index) == after_value(after,index) : F32},+shape: List<&2,U32>,tensor: Tensors.Tensor<F32>,
  ready: {Contract.tensor_ready(~F32,tensor) == True{} : Bool},extents: {Contract.extents(~F32,tensor) == shape : List<&2,U32>},
  contract: Global.Contract(~F32,~Before,~before_value,before,tensor)) -> Global.Shaped(~F32,~After,~after_value,after,shape,tensor):
  Global.restated(~F32,~Before,~After,~before_value,~after_value,before,after,agrees,shape,tensor,
    Global.shaped(~F32,~Before,~before_value,before,shape,tensor,ready,extents,contract))

def Activated(~Logical: Data,~original: Logical -> U32 -> F32,+activation: Activations.Activation,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensors.Tensor<F32>) -> Data:
  Global.Shaped(~F32,~ActivationRead<Logical>,~activated_value(~Logical,~original),ActivationRead{activation,logical},shape,tensor)

# The activation of a tensor of the contract, of any layout. A Ready result
# is a tensor of its shape that holds the activation of the value at every
# index.
def activate(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  ready: {Contract.tensor_ready(~F32,F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},activation)) == True{} : Bool}) ->
  Activated(~Logical,~original,activation,logical,shape,F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},activation)):
  match activation:
    case Activations.Identity{}:
      read_as(~Logical,~ActivationRead<Logical>,~original,~activated_value(~Logical,~original),logical,ActivationRead{Activations.Identity{},logical},index => {==},shape,
        Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},{==},{==},contract)
    case Activations.ReLU{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.relu_value),~original),~activated_value(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.ReLU{},logical},index => {==},shape,
        F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.ReLU{}),{==},{==},
        public(~Logical,~original,~Activations.relu_value,~Operations.mapped(~F32,~Activations.relu_value),~OperationLaws.mapped(~F32,~Activations.relu_value),~"relu",~Activations.operations(Activations.ReLU{}))(
          logical,bands,geometry,shape,fill,workers,contract))
    case Activations.ReLU6{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.relu6_value),~original),~activated_value(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.ReLU6{},logical},index => {==},shape,
        F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.ReLU6{}),{==},{==},
        public(~Logical,~original,~Activations.relu6_value,~Operations.mapped(~F32,~Activations.relu6_value),~OperationLaws.mapped(~F32,~Activations.relu6_value),~"relu6",~Activations.operations(Activations.ReLU6{}))(
          logical,bands,geometry,shape,fill,workers,contract))
    case Activations.LeakyReLU{+slope}:
      read_as(~BandMap.Context<F32,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~Activations.leaky_relu_value),~original),~activated_value(~Logical,~original),
        BandMap.Context{slope,logical},ActivationRead{Activations.LeakyReLU{slope},logical},index => {==},shape,
        F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.LeakyReLU{slope}),{==},{==},
        NormalUnary.combine_scalar(~F32,~Logical,~Activations.leaky_relu_value,~original,logical,Activations.operations(Activations.LeakyReLU{slope}),slope,bands,geometry,shape,fill,workers,contract))
    case Activations.Sigmoid{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.sigmoid_value),~original),~activated_value(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.Sigmoid{},logical},index => {==},shape,
        F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.Sigmoid{}),{==},{==},
        sigmoid(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.SiLU{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.silu_value),~original),~activated_value(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.SiLU{},logical},index => {==},shape,
        F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.SiLU{}),{==},{==},
        silu(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.Tanh{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.tanh_value),~original),~activated_value(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.Tanh{},logical},index => {==},shape,
        F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.Tanh{}),{==},{==},
        tanh(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.HardSigmoid{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.hard_sigmoid_value),~original),~activated_value(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.HardSigmoid{},logical},index => {==},shape,
        F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.HardSigmoid{}),{==},{==},
        public(~Logical,~original,~Activations.hard_sigmoid_value,~Operations.mapped(~F32,~Activations.hard_sigmoid_value),~OperationLaws.mapped(~F32,~Activations.hard_sigmoid_value),~"hard_sigmoid",
          ~Activations.operations(Activations.HardSigmoid{}))(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.HardSwish{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.hard_swish_value),~original),~activated_value(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.HardSwish{},logical},index => {==},shape,
        F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.HardSwish{}),{==},{==},
        public(~Logical,~original,~Activations.hard_swish_value,~Operations.mapped(~F32,~Activations.hard_swish_value),~OperationLaws.mapped(~F32,~Activations.hard_swish_value),~"hard_swish",
          ~Activations.operations(Activations.HardSwish{}))(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.GELUTanh{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.gelu_tanh_value),~original),~activated_value(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.GELUTanh{},logical},index => {==},shape,
        F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.GELUTanh{}),{==},{==},
        gelu_tanh(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.Custom{+operations}:
      Empty.absurd(Activated(~Logical,~original,Activations.Custom{operations},logical,shape,
          F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.Custom{operations})),
        Words.false_true(ready))

# The activations activate computes: every one but Custom, whose function
# reaches only conv2d_custom and apply.
def built_in(activation: Activations.Activation) -> Bool:
  match activation:
    case Activations.Custom{operations}: False{}
    case activation: True{}

# A built-in activation of a Usable tensor is Ready.
def activated_ready(+activation: Activations.Activation,+bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  built: {built_in(activation) == True{} : Bool}) ->
  {Contract.tensor_ready(~F32,F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},activation)) == True{} : Bool}:
  match activation:
    case Activations.Custom{+operations}:
      Empty.absurd({Contract.tensor_ready(~F32,F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.Custom{operations})) == True{} : Bool},
        Words.false_true(built))
    case Activations.Identity{}: {==}
    case Activations.ReLU{}: {==}
    case Activations.ReLU6{}: {==}
    case Activations.LeakyReLU{slope}: {==}
    case Activations.Sigmoid{}: {==}
    case Activations.SiLU{}: {==}
    case Activations.Tanh{}: {==}
    case Activations.HardSigmoid{}: {==}
    case Activations.HardSwish{}: {==}
    case Activations.GELUTanh{}: {==}

# The output of activate_forward holds the activation of the values and the
# input comes back with them, both tensors of the shape.
def ActivatedForward(~Logical: Data,~original: Logical -> U32 -> F32,+activation: Activations.Activation,+logical: Logical,+shape: List<&2,U32>,
  result: Tensors.Tensor<F32> & Tensors.Tensor<F32>) -> Data:
  (output,input) = result
  Shape.Both<Activated(~Logical,~original,activation,logical,shape,output),Global.Shaped(~F32,~Logical,~original,logical,shape,input)>

def forward_pair(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +first: Model.Model<F32>,+second: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  output: Activated(~Logical,~original,activation,logical,shape,F32Tensor.activate(Tensors.Tensor{Model.pack(~F32,first),geometry,shape,fill,Tensors.Usable{},workers},activation)),
  second_contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,second),geometry,shape,fill,Tensors.Usable{},workers})) ->
  ActivatedForward(~Logical,~original,activation,logical,shape,
    F32Tensor.activated_pair(activation,(Tensors.Tensor{Model.pack(~F32,first),geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{Model.pack(~F32,second),geometry,shape,fill,Tensors.Usable{},workers}))):
  Shape.Both{output,Shape.Both{Global.Usable{second,geometry,fill,workers,{==}},second_contract}}

# One copy is activated, the other comes back.
def forward_copies(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +model: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,evidence: Copy.Copied<F32,model>,
  +contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,model),geometry,shape,fill,Tensors.Usable{},workers}),
  +built: {built_in(activation) == True{} : Bool}) ->
  ActivatedForward(~Logical,~original,activation,logical,shape,F32Tensor.activate_forward(Tensors.Tensor{Model.pack(~F32,model),geometry,shape,fill,Tensors.Usable{},workers},activation)):
  Copy.Copied{+first,+second,+equation,+first_content,+second_content} = evidence
  %equation : ActivatedForward(~Logical,~original,activation,logical,shape,
    F32Tensor.activated_pair(activation,Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),_,
      left => right => (Tensors.Tensor{left,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{right,geometry,shape,fill,Tensors.Usable{},workers}))))
  forward_pair(~Logical,~original,logical,activation,first,second,geometry,shape,fill,workers,
    activate(~Logical,~original,logical,activation,first,geometry,shape,fill,workers,
      Copy.kept(~F32,~Logical,~original,logical,model,first,geometry,shape,fill,workers,first_content,contract),
      activated_ready(activation,first,geometry,shape,fill,workers,built)),
    Copy.kept(~F32,~Logical,~original,logical,model,second,geometry,shape,fill,workers,second_content,contract))

# activate_forward of a tensor of the contract, of any layout, with a
# built-in activation: the output is Ready in the shape and holds the
# activation of the value at every index, and the input comes back with its
# values.
def activate_forward(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +model: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,model),geometry,shape,fill,Tensors.Usable{},workers}),
  +built: {built_in(activation) == True{} : Bool}) ->
  ActivatedForward(~Logical,~original,activation,logical,shape,F32Tensor.activate_forward(Tensors.Tensor{Model.pack(~F32,model),geometry,shape,fill,Tensors.Usable{},workers},activation)):
  forward_copies(~Logical,~original,logical,activation,model,geometry,shape,fill,workers,Copy.copied(~F32,model),contract,built)

def Derived(~Logical: Data,~original: Logical -> U32 -> F32,+activation: Activations.Activation,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensors.Tensor<F32>) -> Data:
  Global.Shaped(~F32,~ActivationRead<Logical>,~derivative_at(~Logical,~original),ActivationRead{activation,logical},shape,tensor)

# The derivative of an activation of a tensor of the contract, of any layout.
# A Ready result is a tensor of its shape that holds the derivative at the
# value of every index.
def activate_derivative(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,
  +bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  ready: {Contract.tensor_ready(~F32,F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},activation)) == True{} : Bool}) ->
  Derived(~Logical,~original,activation,logical,shape,F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},activation)):
  match activation:
    case Activations.Identity{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.identity_derivative_value),~original),~derivative_at(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.Identity{},logical},index => {==},shape,
        F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.Identity{}),{==},{==},
        identity_derivative(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.ReLU{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.relu_derivative_value),~original),~derivative_at(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.ReLU{},logical},index => {==},shape,
        F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.ReLU{}),{==},{==},
        relu_derivative(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.ReLU6{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.relu6_derivative_value),~original),~derivative_at(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.ReLU6{},logical},index => {==},shape,
        F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.ReLU6{}),{==},{==},
        relu6_derivative(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.LeakyReLU{+slope}:
      read_as(~BandMap.Context<F32,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~F32,~Logical,~(slope => value => Activations.leaky_relu_derivative_value(value,slope)),~original),
        ~derivative_at(~Logical,~original),BandMap.Context{slope,logical},ActivationRead{Activations.LeakyReLU{slope},logical},index => {==},shape,
        F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.LeakyReLU{slope}),{==},{==},
        NormalUnary.apply_context(~F32,~F32,~Logical,~(slope => value => Activations.leaky_relu_derivative_value(value,slope)),
          ~Operations.lanes(~F32,~(slope => value => Activations.leaky_relu_derivative_value(value,slope))),~OperationLaws.lanes(~F32,~(slope => value => Activations.leaky_relu_derivative_value(value,slope))),
          ~original,slope,logical,Activations.derivative_operations(Activations.LeakyReLU{slope}),bands,geometry,shape,fill,workers,contract))
    case Activations.Sigmoid{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.sigmoid_derivative_value),~original),~derivative_at(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.Sigmoid{},logical},index => {==},shape,
        F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.Sigmoid{}),{==},{==},
        sigmoid_derivative(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.SiLU{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.silu_derivative_value),~original),~derivative_at(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.SiLU{},logical},index => {==},shape,
        F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.SiLU{}),{==},{==},
        silu_derivative(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.Tanh{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.tanh_derivative_value),~original),~derivative_at(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.Tanh{},logical},index => {==},shape,
        F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.Tanh{}),{==},{==},
        tanh_derivative(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.HardSigmoid{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.hard_sigmoid_derivative_value),~original),~derivative_at(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.HardSigmoid{},logical},index => {==},shape,
        F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.HardSigmoid{}),{==},{==},
        hard_sigmoid_derivative(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.HardSwish{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.hard_swish_derivative_value),~original),~derivative_at(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.HardSwish{},logical},index => {==},shape,
        F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.HardSwish{}),{==},{==},
        hard_swish_derivative(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.GELUTanh{}:
      read_as(~BandMap.Context<Unit,Logical>,~ActivationRead<Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Activations.gelu_tanh_derivative_value),~original),~derivative_at(~Logical,~original),
        BandMap.Context{Unit{},logical},ActivationRead{Activations.GELUTanh{},logical},index => {==},shape,
        F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.GELUTanh{}),{==},{==},
        gelu_tanh_derivative(~Logical,~original)(logical,bands,geometry,shape,fill,workers,contract))
    case Activations.Custom{+operations}:
      Empty.absurd(Derived(~Logical,~original,Activations.Custom{operations},logical,shape,
          F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},Activations.Custom{operations})),
        Words.false_true(ready))

# The gradient before an activation holds at every index the gradient after
# it times the derivative at the activation's input.
def ActivatedBackward(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,
  +activation: Activations.Activation,+gradient: Gradient,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensors.Tensor<F32>) -> Data:
  Global.Shaped(~F32,~Leaf.Context<Gradient,ActivationRead<Logical>>,~Leaf.value(~F32,~Gradient,~ActivationRead<Logical>,~F32.mul,~gradient_value,~derivative_at(~Logical,~original)),
    Leaf.Context{gradient,ActivationRead{activation,logical}},shape,tensor)

# Of operands of one shape, activate_backward is the gradient times the
# derivative at the input.
def backward_same(+activation: Activations.Activation,gradient_bands: Bands.Bands(~F32),+gradient_geometry: Bands.Geometry,+gradient_fill: F32,+gradient_status: Tensors.Status,+gradient_workers: U32,
  input_bands: Bands.Bands(~F32),+input_geometry: Bands.Geometry,+input_fill: F32,+input_status: Tensors.Status,+input_workers: U32,+shape: List<&2,U32>) ->
  {F32Tensor.activate_backward(Tensors.Tensor{gradient_bands,gradient_geometry,shape,gradient_fill,gradient_status,gradient_workers},
      Tensors.Tensor{input_bands,input_geometry,shape,input_fill,input_status,input_workers},activation) ==
    F32Tensor.mul(Tensors.Tensor{gradient_bands,gradient_geometry,shape,gradient_fill,gradient_status,gradient_workers},
      F32Tensor.activate_derivative(Tensors.Tensor{input_bands,input_geometry,shape,input_fill,input_status,input_workers},activation)) : Tensors.Tensor<F32>}:
  %Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},ShapeLaws.same_extents_reflexive(shape)) :
    {F32Tensor.backward_when(_,activation,Tensors.Tensor{gradient_bands,gradient_geometry,shape,gradient_fill,gradient_status,gradient_workers},
        Tensors.Tensor{input_bands,input_geometry,shape,input_fill,input_status,input_workers}) ==
      F32Tensor.mul(Tensors.Tensor{gradient_bands,gradient_geometry,shape,gradient_fill,gradient_status,gradient_workers},
        F32Tensor.activate_derivative(Tensors.Tensor{input_bands,input_geometry,shape,input_fill,input_status,input_workers},activation)) : Tensors.Tensor<F32>}
  {==}

# The product of the gradient and a derivative of the shape.
def derived_product(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,
  +activation: Activations.Activation,+gradient: Gradient,+logical: Logical,+shape: List<&2,U32>,
  +gradient_bands: Model.Model<F32>,+gradient_geometry: Bands.Geometry,+gradient_fill: F32,+gradient_workers: U32,
  +gradient_contract: Global.Contract(~F32,~Gradient,~gradient_value,gradient,
    Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers}),
  -derivative: Tensors.Tensor<F32>,derived: Derived(~Logical,~original,activation,logical,shape,derivative),
  ready: {Contract.tensor_ready(~F32,F32Tensor.mul(Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers},derivative)) == True{} : Bool}) ->
  ActivatedBackward(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,shape,
    F32Tensor.mul(Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers},derivative)):
  Shape.Both{usable,contract} = derived
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},derivative,equation)
  %equation : ActivatedBackward(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,shape,
    F32Tensor.mul(Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers},_))
  Global.shaped(~F32,~Leaf.Context<Gradient,ActivationRead<Logical>>,~Leaf.value(~F32,~Gradient,~ActivationRead<Logical>,~F32.mul,~gradient_value,~derivative_at(~Logical,~original)),
    Leaf.Context{gradient,ActivationRead{activation,logical}},shape,
    F32Tensor.mul(Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers},
      Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}),
    %back : {Contract.tensor_ready(~F32,F32Tensor.mul(Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers},_)) == True{} : Bool}
      ready,
    Flow.combined_extents(~F32,~Operations.combine_kernel(~F32,~F32.mul),"mul",1.0,Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers,
      Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers,UnitBroadcast.self_broadcast(shape)),
    CombinePublic.combine(~F32,~Gradient,~ActivationRead<Logical>,~F32.mul,~Operations.combine_kernel(~F32,~F32.mul),~OperationLaws.scalar_kernel(~F32,~F32.mul),
      ~gradient_value,~derivative_at(~Logical,~original),gradient,ActivationRead{activation,logical},"mul",1.0,shape,gradient_geometry,geometry,gradient_fill,fill,gradient_workers,workers,
      Model.pack(~F32,gradient_bands),Model.pack(~F32,bands),gradient_contract,
      %back : Global.Contract(~F32,~ActivationRead<Logical>,~derivative_at(~Logical,~original),ActivationRead{activation,logical},_) contract))

# activate_backward of a gradient and an activation's input of the
# contracts and one shape, of any layout. A Ready result is a tensor of the
# shape that holds at every index the gradient times the derivative at the
# input.
def activate_backward(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,
  +activation: Activations.Activation,+gradient: Gradient,+logical: Logical,+shape: List<&2,U32>,
  +gradient_bands: Model.Model<F32>,+gradient_geometry: Bands.Geometry,+gradient_fill: F32,+gradient_workers: U32,
  +gradient_contract: Global.Contract(~F32,~Gradient,~gradient_value,gradient,
    Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers}),
  +bands: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  +ready: {Contract.tensor_ready(~F32,F32Tensor.activate_backward(Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers},
    Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},activation)) == True{} : Bool}) ->
  ActivatedBackward(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,shape,
    F32Tensor.activate_backward(Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers},
      Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},activation)):
  +same = backward_same(activation,Model.pack(~F32,gradient_bands),gradient_geometry,gradient_fill,Tensors.Usable{},gradient_workers,
    Model.pack(~F32,bands),geometry,fill,Tensors.Usable{},workers,shape)
  %Equal.sym(Tensors.Tensor<F32>,
    F32Tensor.activate_backward(Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers},
      Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},activation),
    F32Tensor.mul(Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers},
      F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},activation)),same) :
    ActivatedBackward(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,shape,_)
  derived_product(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,shape,gradient_bands,gradient_geometry,gradient_fill,gradient_workers,gradient_contract,
    F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},activation),
    activate_derivative(~Logical,~original,logical,activation,bands,geometry,shape,fill,workers,contract,
      Flow.combine_right(~F32,~F32.mul,Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers},
        F32Tensor.activate_derivative(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},activation),"mul",1.0,%same : {Contract.tensor_ready(~F32,_) == True{} : Bool} ready)),
    %same : {Contract.tensor_ready(~F32,_) == True{} : Bool} ready)

# The activation of a failed tensor stays failed.
def activate_failed(+activation: Activations.Activation,bands: Bands.Bands(~F32),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+problem: Tensors.Problem,+workers: U32) ->
  {Contract.tensor_ready(~F32,F32Tensor.activate(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers},activation)) == False{} : Bool}:
  match activation:
    case Activations.Identity{}: {==}
    case Activations.ReLU{}: {==}
    case Activations.ReLU6{}: {==}
    case Activations.LeakyReLU{slope}: {==}
    case Activations.Sigmoid{}: {==}
    case Activations.SiLU{}: {==}
    case Activations.Tanh{}: {==}
    case Activations.HardSigmoid{}: {==}
    case Activations.HardSwish{}: {==}
    case Activations.GELUTanh{}: {==}
    case Activations.Custom{operations}: {==}
