# The elementwise functions of one tiled tensor hold the function of the value
# at every index. F.activate and F.activate_derivative of every built-in
# activation hold the scalar activation or derivative of each value, and
# F.activate_backward the gradient times the derivative at the input.
# F.activate_forward keeps the input beside the activation.
import Base
import ../tensor_f32.bend as F32Tensor
import ../tensor.bend as Tensors
import ../tensor_access.bend as Access
import ../math_activation.bend as Activations
import ../math_fp32.bend as Arithmetic
import ../math_activation_lanes.bend as ActivationLanes
import ../math_gelu_lanes.bend as GeluLanes
import ../kernel_tile.bend as Tile
import ../kernel_gemm.bend as KernelGemm
import ../tensor_grid.bend as Grid
import ../tensor_shape.bend as Shapes
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_map.bend as TiledMap
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./kernel_pipeline_activation.bend as ActivationValues
import ./word_encoding.bend as Words
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_pieces_model.bend as PiecesModel
import ./tensor_pieces_map.bend as PiecesMap
import ./tensor_pieces_combine_leaf.bend as Leaf
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_map.bend as TiledMaps
import ./tensor_binary_values.bend as Binary
import ./tensor_copy_contract.bend as Copy

# ---- A function at every lane ----

# A map with an operation and its context holds the operation of the value at
# every index.
def ContextPublicLaw(~Context: Data,~context: Context,~Logical: Data,~original: Logical -> U32 -> F32,~operation: Context -> F32 -> F32,~call: Tiled.Tiled -> Tiled.Tiled) -> Type:
  @+logical: Logical -> @+values: List<&2,U32> -> @tiled: Tiled.Tiled -> @+contract: Public.Contract(~Logical,~original,logical,values,tiled) ->
  Public.Contract(~PiecesMap.Context<Context,Logical>,~PiecesMap.value(~F32,~Context,~Logical,~operation,~original),PiecesMap.Context{context,logical},values,call(tiled))

# A map whose tile function is the operation at every lane.
def public_context(~Context: Data,~Logical: Data,~original: Logical -> U32 -> F32,~operation: Context -> F32 -> F32,~transform: Context -> Tile.Tile8 -> Tile.Tile8,
  ~lane_law: TiledMaps.LaneLaw(~Context,~transform,~operation),+context: Context,+cost: F32) ->
  ContextPublicLaw(~Context,~context,~Logical,~original,~operation,~F32Tensor.tiled_map(~Context,~transform,context,cost)):
  logical => values => tiled => contract =>
    TiledMaps.map(~Context,~Logical,~operation,~transform,~lane_law,~original,context,cost,logical,values,tiled,contract)

def PublicLaw(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32,~call: Tiled.Tiled -> Tiled.Tiled) -> Type:
  ContextPublicLaw(~Unit,~Unit{},~Logical,~original,~Access.apply_element(~F32,~operation),~call)

def public(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32,+cost: F32) -> PublicLaw(~Logical,~original,~operation,~F32Tensor.tiled_lanes(~operation,cost)):
  public_context(~Unit,~Logical,~original,~Access.apply_element(~F32,~operation),~TiledMap.lanes(~Unit,~Access.apply_element(~F32,~operation)),
    ~TiledMaps.lanes_law(~Unit,~Access.apply_element(~F32,~operation)),Unit{},cost)

# An operation with a scalar at every lane.
def scalar_public(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32 -> F32,+scalar: F32,+cost: F32) ->
  ContextPublicLaw(~F32,~scalar,~Logical,~original,~F32Tensor.scalar_value(~operation),~(tensor => F32Tensor.tiled_scalar(~operation,cost,tensor,scalar))):
  public_context(~F32,~Logical,~original,~F32Tensor.scalar_value(~operation),~TiledMap.lanes(~F32,~F32Tensor.scalar_value(~operation)),
    ~TiledMaps.lanes_law(~F32,~F32Tensor.scalar_value(~operation)),scalar,cost)

def exp(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Arithmetic.exp,~F32Tensor.exp):
  public(~Logical,~original,~Arithmetic.exp,4.5)

def log(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Arithmetic.log,~F32Tensor.log):
  public(~Logical,~original,~Arithmetic.log,8.25)

def sqrt(~Logical: Data,~original: Logical -> U32 -> F32) -> PublicLaw(~Logical,~original,~Arithmetic.sqrt,~F32Tensor.sqrt):
  public(~Logical,~original,~Arithmetic.sqrt,0.125)

# ---- Activations and their derivatives ----

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
    case Activations.GELU{}: Activations.gelu_derivative_value(value)
    case Activations.Custom{operations}: value

def derivative_at(~Logical: Data,~original: Logical -> U32 -> F32,read: ActivationRead<Logical>,index: U32) -> F32:
  ActivationRead{+activation,+logical} = read
  derivative_value(activation,original(logical,index))

def Activated(~Logical: Data,~original: Logical -> U32 -> F32,+activation: Activations.Activation,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled) -> Data:
  Public.Contract(~ActivationRead<Logical>,~activated_value(~Logical,~original),ActivationRead{activation,logical},values,tiled)

def Derived(~Logical: Data,~original: Logical -> U32 -> F32,+activation: Activations.Activation,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled) -> Data:
  Public.Contract(~ActivationRead<Logical>,~derivative_at(~Logical,~original),ActivationRead{activation,logical},values,tiled)

# The activations activate computes: every one but Custom, whose function
# reaches only conv2d_custom and apply.
def built_in(activation: Activations.Activation) -> Bool:
  match activation:
    case Activations.Custom{operations}: False{}
    case activation: True{}

# ---- Reading a map as an activation ----

# The reader of a map whose context names the activation.
def MappedRead(~Context: Data,~Logical: Data,~build: Context -> Activations.Activation,mapped: PiecesMap.Context<Context,Logical>,read: ActivationRead<Logical>) -> Data:
  PiecesMap.Context{+context,+logical} = mapped
  {read == ActivationRead{build(context),logical} : ActivationRead<Logical>}

def mapped_agree(~Context: Data,~Logical: Data,~original: Logical -> U32 -> F32,~operation: Context -> F32 -> F32,~value: ActivationRead<Logical> -> U32 -> F32,
  ~build: Context -> Activations.Activation,
  ~same: @+context: Context -> @+logical: Logical -> @+index: U32 -> {operation(context,original(logical,index)) == value(ActivationRead{build(context),logical},index) : F32},
  +mapped: PiecesMap.Context<Context,Logical>,+read: ActivationRead<Logical>,+related: MappedRead(~Context,~Logical,~build,mapped,read),+index: U32) ->
  {PiecesMap.value(~F32,~Context,~Logical,~operation,~original,mapped,index) == value(read,index) : F32}:
  match mapped:
    case PiecesMap.Context{+context,+logical}:
      %Equal.sym(ActivationRead<Logical>,read,ActivationRead{build(context),logical},related) : {operation(context,original(logical,index)) == value(_,index) : F32}
      same(context,logical,index)

# A map whose operation is the activation at its context holds the activation
# of the value at every index.
def mapped_read(~Context: Data,~Logical: Data,~original: Logical -> U32 -> F32,~operation: Context -> F32 -> F32,~transform: Context -> Tile.Tile8 -> Tile.Tile8,
  ~lane_law: TiledMaps.LaneLaw(~Context,~transform,~operation),~value: ActivationRead<Logical> -> U32 -> F32,~build: Context -> Activations.Activation,
  ~same: @+context: Context -> @+logical: Logical -> @+index: U32 -> {operation(context,original(logical,index)) == value(ActivationRead{build(context),logical},index) : F32},
  +context: Context,+activation: Activations.Activation,+built: {activation == build(context) : Activations.Activation},+cost: F32,+logical: Logical,+values: List<&2,U32>,
  tiled: Tiled.Tiled,+contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~ActivationRead<Logical>,~value,ActivationRead{activation,logical},values,TiledMap.map(~Context,~transform,context,cost,tiled)):
  Public.by_packed(~Logical,~original,(tiled => Public.Contract(~ActivationRead<Logical>,~value,ActivationRead{activation,logical},values,TiledMap.map(~Context,~transform,context,cost,tiled))),
    logical,values,tiled,contract,
    model => shape => fill => workers => axis => lanes => position => packed =>
      Public.reread(~PiecesMap.Context<Context,Logical>,~ActivationRead<Logical>,~PiecesMap.value(~F32,~Context,~Logical,~operation,~original),~value,
        ~MappedRead(~Context,~Logical,~build),~mapped_agree(~Context,~Logical,~original,~operation,~value,~build,~same),PiecesMap.Context{context,logical},ActivationRead{activation,logical},
        Equal.cong(Activations.Activation,ActivationRead<Logical>,chosen => ActivationRead{chosen,logical},activation,build(context),built),values,
        TiledMap.map(~Context,~transform,context,cost,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
        TiledMaps.map(~Context,~Logical,~operation,~transform,~lane_law,~original,context,cost,logical,values,
          Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position},packed)))

# A function of the value alone at every lane.
def lanes_read(~Logical: Data,~original: Logical -> U32 -> F32,~function: F32 -> F32,~value: ActivationRead<Logical> -> U32 -> F32,~build: Unit -> Activations.Activation,
  ~same: @+context: Unit -> @+logical: Logical -> @+index: U32 -> {Access.apply_element(~F32,~function,context,original(logical,index)) == value(ActivationRead{build(context),logical},index) : F32},
  +activation: Activations.Activation,+built: {activation == build(Unit{}) : Activations.Activation},+cost: F32,+logical: Logical,+values: List<&2,U32>,
  tiled: Tiled.Tiled,+contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  Public.Contract(~ActivationRead<Logical>,~value,ActivationRead{activation,logical},values,F32Tensor.lanes_of(~function,cost,tiled)):
  mapped_read(~Unit,~Logical,~original,~Access.apply_element(~F32,~function),~TiledMap.lanes(~Unit,~Access.apply_element(~F32,~function)),
    ~TiledMaps.lanes_law(~Unit,~Access.apply_element(~F32,~function)),~value,~build,~same,Unit{},activation,built,cost,logical,values,tiled,contract)

# The reader of values an activation keeps.
def KeptRead(~Logical: Data,+activation: Activations.Activation,logical: Logical,read: ActivationRead<Logical>) -> Data:
  {read == ActivationRead{activation,logical} : ActivationRead<Logical>}

def identity_agree(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+read: ActivationRead<Logical>,
  +related: KeptRead(~Logical,Activations.Identity{},logical,read),+index: U32) -> {original(logical,index) == activated_value(~Logical,~original,read,index) : F32}:
  %Equal.sym(ActivationRead<Logical>,read,ActivationRead{Activations.Identity{},logical},related) : {original(logical,index) == activated_value(~Logical,~original,_,index) : F32}
  {==}

def custom_agree(~Logical: Data,~original: Logical -> U32 -> F32,+count: F32,+logical: Logical,+read: ActivationRead<Logical>,
  +related: KeptRead(~Logical,Activations.Custom{count},logical,read),+index: U32) -> {original(logical,index) == activated_value(~Logical,~original,read,index) : F32}:
  %Equal.sym(ActivationRead<Logical>,read,ActivationRead{Activations.Custom{count},logical},related) : {original(logical,index) == activated_value(~Logical,~original,_,index) : F32}
  {==}

# ---- The activation of a tiled tensor ----

# The activation of tiled values, as a convolution leaf applies it, holds the
# activation of the value at every index. Custom keeps the values.
def activated(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+values: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled)) -> Activated(~Logical,~original,activation,logical,values,F32Tensor.activated(tiled,activation)):
  match activation:
    case Activations.Identity{}:
      Public.reread(~Logical,~ActivationRead<Logical>,~original,~activated_value(~Logical,~original),~KeptRead(~Logical,Activations.Identity{}),~identity_agree(~Logical,~original),
        logical,ActivationRead{Activations.Identity{},logical},{==},values,tiled,contract)
    case Activations.ReLU{}:
      lanes_read(~Logical,~original,~Activations.relu_value,~activated_value(~Logical,~original),~(context => Activations.ReLU{}),~(context => base => index => {==}),
        Activations.ReLU{},{==},Activations.operations(Activations.ReLU{}),logical,values,tiled,contract)
    case Activations.ReLU6{}:
      lanes_read(~Logical,~original,~Activations.relu6_value,~activated_value(~Logical,~original),~(context => Activations.ReLU6{}),~(context => base => index => {==}),
        Activations.ReLU6{},{==},Activations.operations(Activations.ReLU6{}),logical,values,tiled,contract)
    case Activations.LeakyReLU{+slope}:
      mapped_read(~F32,~Logical,~original,~F32Tensor.scalar_value(~Activations.leaky_relu_value),~TiledMap.lanes(~F32,~F32Tensor.scalar_value(~Activations.leaky_relu_value)),
        ~TiledMaps.lanes_law(~F32,~F32Tensor.scalar_value(~Activations.leaky_relu_value)),~activated_value(~Logical,~original),~(context => Activations.LeakyReLU{context}),
        ~(context => base => index => {==}),slope,Activations.LeakyReLU{slope},{==},Activations.operations(Activations.LeakyReLU{slope}),logical,values,tiled,contract)
    case Activations.Sigmoid{}:
      lanes_read(~Logical,~original,~Activations.sigmoid_value,~activated_value(~Logical,~original),~(context => Activations.Sigmoid{}),~(context => base => index => {==}),
        Activations.Sigmoid{},{==},Activations.operations(Activations.Sigmoid{}),logical,values,tiled,contract)
    case Activations.SiLU{}:
      lanes_read(~Logical,~original,~Activations.silu_value,~activated_value(~Logical,~original),~(context => Activations.SiLU{}),~(context => base => index => {==}),
        Activations.SiLU{},{==},Activations.operations(Activations.SiLU{}),logical,values,tiled,contract)
    case Activations.Tanh{}:
      lanes_read(~Logical,~original,~Activations.tanh_value,~activated_value(~Logical,~original),~(context => Activations.Tanh{}),~(context => base => index => {==}),
        Activations.Tanh{},{==},Activations.operations(Activations.Tanh{}),logical,values,tiled,contract)
    case Activations.HardSigmoid{}:
      lanes_read(~Logical,~original,~Activations.hard_sigmoid_value,~activated_value(~Logical,~original),~(context => Activations.HardSigmoid{}),~(context => base => index => {==}),
        Activations.HardSigmoid{},{==},Activations.operations(Activations.HardSigmoid{}),logical,values,tiled,contract)
    case Activations.HardSwish{}:
      lanes_read(~Logical,~original,~Activations.hard_swish_value,~activated_value(~Logical,~original),~(context => Activations.HardSwish{}),~(context => base => index => {==}),
        Activations.HardSwish{},{==},Activations.operations(Activations.HardSwish{}),logical,values,tiled,contract)
    case Activations.GELUTanh{}:
      mapped_read(~Unit,~Logical,~original,~Access.apply_element(~F32,~Activations.gelu_tanh_value),~GeluLanes.tile,~ActivationValues.gelu_tanh_tile,
        ~activated_value(~Logical,~original),~(context => Activations.GELUTanh{}),~(context => base => index => {==}),
        Unit{},Activations.GELUTanh{},{==},Activations.operations(Activations.GELUTanh{}),logical,values,tiled,contract)
    case Activations.GELU{}:
      lanes_read(~Logical,~original,~Activations.gelu_value,~activated_value(~Logical,~original),~(context => Activations.GELU{}),~(context => base => index => {==}),
        Activations.GELU{},{==},Activations.operations(Activations.GELU{}),logical,values,tiled,contract)
    case Activations.Custom{+count}:
      Public.reread(~Logical,~ActivationRead<Logical>,~original,~activated_value(~Logical,~original),~KeptRead(~Logical,Activations.Custom{count}),~custom_agree(~Logical,~original,count),
        logical,ActivationRead{Activations.Custom{count},logical},{==},values,tiled,contract)

# activate of a tensor of the contract with a built-in activation holds the
# activation of the value at every index.
def activate(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+values: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled),+built: {built_in(activation) == True{} : Bool}) ->
  Activated(~Logical,~original,activation,logical,values,F32Tensor.activate(tiled,activation)):
  match activation:
    case Activations.Identity{}: activated(~Logical,~original,logical,Activations.Identity{},values,tiled,contract)
    case Activations.ReLU{}: activated(~Logical,~original,logical,Activations.ReLU{},values,tiled,contract)
    case Activations.ReLU6{}: activated(~Logical,~original,logical,Activations.ReLU6{},values,tiled,contract)
    case Activations.LeakyReLU{+slope}: activated(~Logical,~original,logical,Activations.LeakyReLU{slope},values,tiled,contract)
    case Activations.Sigmoid{}: activated(~Logical,~original,logical,Activations.Sigmoid{},values,tiled,contract)
    case Activations.SiLU{}: activated(~Logical,~original,logical,Activations.SiLU{},values,tiled,contract)
    case Activations.Tanh{}: activated(~Logical,~original,logical,Activations.Tanh{},values,tiled,contract)
    case Activations.HardSigmoid{}: activated(~Logical,~original,logical,Activations.HardSigmoid{},values,tiled,contract)
    case Activations.HardSwish{}: activated(~Logical,~original,logical,Activations.HardSwish{},values,tiled,contract)
    case Activations.GELUTanh{}: activated(~Logical,~original,logical,Activations.GELUTanh{},values,tiled,contract)
    case Activations.GELU{}: activated(~Logical,~original,logical,Activations.GELU{},values,tiled,contract)
    case Activations.Custom{+count}: Empty.absurd(Activated(~Logical,~original,Activations.Custom{count},logical,values,F32Tensor.activate(tiled,Activations.Custom{count})),Words.false_true(built))

# The activated copy beside the input.
def ActivatedForward(~Logical: Data,~original: Logical -> U32 -> F32,+activation: Activations.Activation,+logical: Logical,+values: List<&2,U32>,
  result: Tiled.Tiled & Tiled.Tiled) -> Data:
  (output,input) = result
  KernelGemm.Both<Activated(~Logical,~original,activation,logical,values,output),Public.Contract(~Logical,~original,logical,values,input)>

def forward_pair(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+values: List<&2,U32>,
  pair: Tiled.Tiled & Tiled.Tiled,copies: Copy.Copies(~Logical,~original,logical,values,pair),+built: {built_in(activation) == True{} : Bool}) ->
  ActivatedForward(~Logical,~original,activation,logical,values,F32Tensor.activated_pair(activation,pair)):
  match pair:
    case (first,second):
      KernelGemm.Both{+first_contract,+second_contract} = copies
      KernelGemm.Both{activate(~Logical,~original,logical,activation,values,first,first_contract,built),second_contract}

# activate_forward of a tensor of the contract with a built-in activation: the
# output holds the activation of the value at every index, and the input
# comes back with its values.
def activate_forward(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+values: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled),+built: {built_in(activation) == True{} : Bool}) ->
  ActivatedForward(~Logical,~original,activation,logical,values,F32Tensor.activate_forward(tiled,activation)):
  Public.by_packed(~Logical,~original,(that => ActivatedForward(~Logical,~original,activation,logical,values,F32Tensor.activate_forward(that,activation))),
    logical,values,tiled,contract,
    model => shape => fill => workers => axis => lanes => position => packed =>
      forward_pair(~Logical,~original,logical,activation,values,
        Tiled.copy(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position}),
        Copy.tiled(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,model),shape,fill,Tensors.Usable{},workers},axis,lanes,position},packed),
        built))

# ---- The derivative of an activation ----

def silu_derivative_tile(+context: Unit,+tile: Tile.Tile8) ->
  {ActivationLanes.silu(context,tile) == TiledMap.lanes(~Unit,~Access.apply_element(~F32,~Activations.silu_derivative_value),context,tile) : Tile.Tile8}:
  match tile:
    case Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}

def gelu_tanh_derivative_tile(+context: Unit,+tile: Tile.Tile8) ->
  {ActivationLanes.gelu_tanh(context,tile) == TiledMap.lanes(~Unit,~Access.apply_element(~F32,~Activations.gelu_tanh_derivative_value),context,tile) : Tile.Tile8}:
  match tile:
    case Tile.Tile{+a,+b,+c,+d,+e,+f,+g,+h}: {==}

# activate_derivative of a tensor of the contract with a built-in activation
# holds the derivative at the value at every index.
def activate_derivative(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+values: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~Logical,~original,logical,values,tiled),+built: {built_in(activation) == True{} : Bool}) ->
  Derived(~Logical,~original,activation,logical,values,F32Tensor.activate_derivative(tiled,activation)):
  match activation:
    case Activations.Identity{}:
      lanes_read(~Logical,~original,~Activations.identity_derivative_value,~derivative_at(~Logical,~original),~(context => Activations.Identity{}),~(context => base => index => {==}),
        Activations.Identity{},{==},Activations.derivative_operations(Activations.Identity{}),logical,values,tiled,contract)
    case Activations.ReLU{}:
      lanes_read(~Logical,~original,~Activations.relu_derivative_value,~derivative_at(~Logical,~original),~(context => Activations.ReLU{}),~(context => base => index => {==}),
        Activations.ReLU{},{==},Activations.derivative_operations(Activations.ReLU{}),logical,values,tiled,contract)
    case Activations.ReLU6{}:
      lanes_read(~Logical,~original,~Activations.relu6_derivative_value,~derivative_at(~Logical,~original),~(context => Activations.ReLU6{}),~(context => base => index => {==}),
        Activations.ReLU6{},{==},Activations.derivative_operations(Activations.ReLU6{}),logical,values,tiled,contract)
    case Activations.LeakyReLU{+slope}:
      mapped_read(~F32,~Logical,~original,~(slope => value => Activations.leaky_relu_derivative_value(value,slope)),
        ~TiledMap.lanes(~F32,~(slope => value => Activations.leaky_relu_derivative_value(value,slope))),
        ~TiledMaps.lanes_law(~F32,~(slope => value => Activations.leaky_relu_derivative_value(value,slope))),~derivative_at(~Logical,~original),
        ~(context => Activations.LeakyReLU{context}),~(context => base => index => {==}),slope,Activations.LeakyReLU{slope},{==},
        Activations.derivative_operations(Activations.LeakyReLU{slope}),logical,values,tiled,contract)
    case Activations.Sigmoid{}:
      lanes_read(~Logical,~original,~Activations.sigmoid_derivative_value,~derivative_at(~Logical,~original),~(context => Activations.Sigmoid{}),~(context => base => index => {==}),
        Activations.Sigmoid{},{==},Activations.derivative_operations(Activations.Sigmoid{}),logical,values,tiled,contract)
    case Activations.SiLU{}:
      mapped_read(~Unit,~Logical,~original,~Access.apply_element(~F32,~Activations.silu_derivative_value),~ActivationLanes.silu,~silu_derivative_tile,
        ~derivative_at(~Logical,~original),~(context => Activations.SiLU{}),~(context => base => index => {==}),
        Unit{},Activations.SiLU{},{==},Activations.derivative_operations(Activations.SiLU{}),logical,values,tiled,contract)
    case Activations.Tanh{}:
      lanes_read(~Logical,~original,~Activations.tanh_derivative_value,~derivative_at(~Logical,~original),~(context => Activations.Tanh{}),~(context => base => index => {==}),
        Activations.Tanh{},{==},Activations.derivative_operations(Activations.Tanh{}),logical,values,tiled,contract)
    case Activations.HardSigmoid{}:
      lanes_read(~Logical,~original,~Activations.hard_sigmoid_derivative_value,~derivative_at(~Logical,~original),~(context => Activations.HardSigmoid{}),
        ~(context => base => index => {==}),Activations.HardSigmoid{},{==},Activations.derivative_operations(Activations.HardSigmoid{}),logical,values,tiled,contract)
    case Activations.HardSwish{}:
      lanes_read(~Logical,~original,~Activations.hard_swish_derivative_value,~derivative_at(~Logical,~original),~(context => Activations.HardSwish{}),
        ~(context => base => index => {==}),Activations.HardSwish{},{==},Activations.derivative_operations(Activations.HardSwish{}),logical,values,tiled,contract)
    case Activations.GELUTanh{}:
      mapped_read(~Unit,~Logical,~original,~Access.apply_element(~F32,~Activations.gelu_tanh_derivative_value),~ActivationLanes.gelu_tanh,~gelu_tanh_derivative_tile,
        ~derivative_at(~Logical,~original),~(context => Activations.GELUTanh{}),~(context => base => index => {==}),
        Unit{},Activations.GELUTanh{},{==},Activations.derivative_operations(Activations.GELUTanh{}),logical,values,tiled,contract)
    case Activations.GELU{}:
      lanes_read(~Logical,~original,~Activations.gelu_derivative_value,~derivative_at(~Logical,~original),~(context => Activations.GELU{}),~(context => base => index => {==}),
        Activations.GELU{},{==},Activations.derivative_operations(Activations.GELU{}),logical,values,tiled,contract)
    case Activations.Custom{+count}:
      Empty.absurd(Derived(~Logical,~original,Activations.Custom{count},logical,values,F32Tensor.activate_derivative(tiled,Activations.Custom{count})),Words.false_true(built))

# ---- The gradient before an activation ----

# The gradient after an activation times the derivative at its input.
def ActivatedBackward(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,
  +activation: Activations.Activation,+gradient: Gradient,+logical: Logical,+values: List<&2,U32>,tiled: Tiled.Tiled) -> Data:
  Public.Contract(~Leaf.Context<Gradient,ActivationRead<Logical>>,~Leaf.value(~F32,~Gradient,~ActivationRead<Logical>,~F32.mul,~gradient_value,~derivative_at(~Logical,~original)),
    Leaf.Context{gradient,ActivationRead{activation,logical}},values,tiled)

# Packed operands of one shape.
def backward_packed(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,
  +activation: Activations.Activation,+gradient: Gradient,+logical: Logical,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +values: List<&2,U32>,+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+built: {built_in(activation) == True{} : Bool},
  +gm: PiecesModel.Model<Tile.Tile8>,+gc: List<&2,U32>,+gf: Tile.Tile8,+gw: U32,+go: Nat,+gl: U32,+gp: Grid.Position,
  +gradient_contract: Public.Contract(~Gradient,~gradient_value,gradient,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gc,gf,Tensors.Usable{},gw},go,gl,gp}),
  +im: PiecesModel.Model<Tile.Tile8>,+ic: List<&2,U32>,+input_fill: Tile.Tile8,+iw: U32,+io: Nat,+il: U32,+ip: Grid.Position,
  +contract: Public.Contract(~Logical,~original,logical,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,im),ic,input_fill,Tensors.Usable{},iw},io,il,ip})) ->
  ActivatedBackward(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,values,
    F32Tensor.activate_backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gc,gf,Tensors.Usable{},gw},go,gl,gp},
      Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,im),ic,input_fill,Tensors.Usable{},iw},io,il,ip},activation)):
  KernelGemm.Both{+gradient_global,Public.Laid{+gradient_inside,+gradient_shaped,+gradient_restored,+gradient_counted,+gradient_pinned}} = gradient_contract
  KernelGemm.Both{+global,Public.Laid{+inside,+shaped,+restored,+counted,+pinned}} = contract
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(gc,go,gl,gp),values,gradient_restored) :
    ActivatedBackward(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,values,
      F32Tensor.backward_when(Shapes.same_extents(_,Tiled.logical_shape(ic,io,il,ip)),activation,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gc,gf,Tensors.Usable{},gw},go,gl,gp},
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,im),ic,input_fill,Tensors.Usable{},iw},io,il,ip}))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(ic,io,il,ip),values,restored) :
    ActivatedBackward(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,values,
      F32Tensor.backward_when(Shapes.same_extents(values,_),activation,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gc,gf,Tensors.Usable{},gw},go,gl,gp},
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,im),ic,input_fill,Tensors.Usable{},iw},io,il,ip}))
  %Equal.sym(Bool,Shapes.same_extents(values,values),True{},ShapeLaws.same_extents_reflexive(values)) :
    ActivatedBackward(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,values,
      F32Tensor.backward_when(_,activation,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gc,gf,Tensors.Usable{},gw},go,gl,gp},
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,im),ic,input_fill,Tensors.Usable{},iw},io,il,ip}))
  Binary.same(~Gradient,~gradient_value,~ActivationRead<Logical>,~derivative_at(~Logical,~original),~F32.mul,gradient,ActivationRead{activation,logical},"mul",1.0,
    limits,default_capacity,values,nonempty,rank_room,
    Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gc,gf,Tensors.Usable{},gw},go,gl,gp},gradient_contract,
    F32Tensor.activate_derivative(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,im),ic,input_fill,Tensors.Usable{},iw},io,il,ip},activation),
    activate_derivative(~Logical,~original,logical,activation,values,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,im),ic,input_fill,Tensors.Usable{},iw},io,il,ip},contract,built))

# activate_backward of a gradient and an activation's input of the contracts
# and one shape without a zero extent holds at every index the gradient times
# the derivative at the input.
def activate_backward(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,
  +activation: Activations.Activation,+gradient: Gradient,+logical: Logical,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +values: List<&2,U32>,+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+built: {built_in(activation) == True{} : Bool},
  gradient_tiled: Tiled.Tiled,+gradient_contract: Public.Contract(~Gradient,~gradient_value,gradient,values,gradient_tiled),
  tiled: Tiled.Tiled,+contract: Public.Contract(~Logical,~original,logical,values,tiled)) ->
  ActivatedBackward(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,values,F32Tensor.activate_backward(gradient_tiled,tiled,activation)):
  Public.by_packed(~Gradient,~gradient_value,
    (that => ActivatedBackward(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,values,F32Tensor.activate_backward(that,tiled,activation))),
    gradient,values,gradient_tiled,gradient_contract,
    gm => gc => gf => gw => go => gl => gp => packed_gradient => Public.by_packed(~Logical,~original,
      (that => ActivatedBackward(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,values,
        F32Tensor.activate_backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,gm),gc,gf,Tensors.Usable{},gw},go,gl,gp},that,activation))),
      logical,values,tiled,contract,
      im => ic => input_fill => iw => io => il => ip => packed => backward_packed(~Gradient,~gradient_value,~Logical,~original,activation,gradient,logical,limits,default_capacity,
        values,nonempty,rank_room,built,gm,gc,gf,gw,go,gl,gp,packed_gradient,im,ic,input_fill,iw,io,il,ip,packed)))
