# The convolution layer of tiled tensors. Forward holds the activation of the
# ordered convolution value and keeps x, that value, the weights and the bias
# with their readers. Backward holds the input, weight and bias gradients of
# x, dz = dy * activation'(z) and the weights, and gives the weights back.
import Base
import ../../lib/tensor_f32.bend as F
import ../../lib/tensor_layer.bend as Layer
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/tensor_parameter_gradient.bend as Parameter
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/math_activation.bend as Activations
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/tensor.bend as Tensors
import ../../lib/kernel_tile.bend as Tile
import ../../lib/tensor_grid.bend as Grid
import ../../lib/proofs/tensor_tiled_public.bend as TiledPublic
import ../../lib/proofs/tensor_pieces_model.bend as PiecesModel
import ../../lib/proofs/tensor_pieces_combine_leaf.bend as Leaf
import ../../lib/proofs/tensor_function_values.bend as FunctionValues
import ../../lib/proofs/tensor_copy_contract.bend as Copy
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../specification.bend as ConvSpec
import ./image_problems.bend as ImageProblems
import ./convolution_call.bend as ConvCall
import ./input_gradient_call.bend as Call
import ./input_gradient_public.bend as InputPublic
import ./parameter_gradient_terms.bend as Terms
import ./parameter_gradient_public.bend as ParameterPublic

# ---- Forward ----

# The convolution value of the layer before its activation.
def convolution_context(~Logical: Data,~Weights: Data,~weight_value: Weights -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +logical: Logical,+weight_logical: Weights,+bias_logical: Bias,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32) ->
  ImageProblems.Context<Logical>:
  ImageProblems.Context{logical,Activations.identity(),U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),
    U32.to_nat(stride),U32.to_nat(padding),Contract.listed(~F32,~Weights,~weight_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,weight_logical),
    Contract.listed(~F32,~Bias,~bias_value,ShapeLaws.elements([outputs]),0n,bias_logical)}

# The output extents of the call.
def output_extents(+batched: Bool,+count: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32) -> List<&2,U32>:
  TensorConvolution.output_shape(batched,count,outputs,U32.from_nat(KernelShape.axis_size(U32.to_nat(height),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding))),
    U32.from_nat(KernelShape.axis_size(U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding))))

# The output holds the activation of the convolution value, z holds that
# value, and x, the weights and the bias come back holding their readers.
def Forward(~Logical: Data,~original: Logical -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +logical: Logical,+weight_logical: Weights,+bias_logical: Bias,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,
  +activation: Activations.Activation,+count: U32,+batched: Bool,result: Layer.Tensor() & (Layer.Saved() & (Layer.Tensor() & Layer.Tensor()))) -> Data:
  (output,(saved,parameters)) = result
  (x,z) = saved
  (weights,bias) = parameters
  KernelGemm.Both<FunctionValues.Activated(~ImageProblems.Context<Logical>,~ImageProblems.value(~Activations.identity_value,~Logical,~original),activation,convolution_context(~Logical,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding),output_extents(batched,count,outputs,height,width,kernel,stride,padding),output),
    KernelGemm.Both<KernelGemm.Both<TiledPublic.Contract(~Logical,~original,logical,TensorConvolution.output_shape(batched,count,channels,height,width),x),
        TiledPublic.Contract(~ImageProblems.Context<Logical>,~ImageProblems.value(~Activations.identity_value,~Logical,~original),convolution_context(~Logical,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding),output_extents(batched,count,outputs,height,width,kernel,stride,padding),z)>,
      KernelGemm.Both<TiledPublic.Contract(~Weights,~weight_value,weight_logical,[outputs,channels,kernel,kernel],weights),
        TiledPublic.Contract(~Bias,~bias_value,bias_logical,[outputs],bias)>>>

# The activated convolution value with the second copies of x, the weights and
# the bias.
def activated_holds(~Logical: Data,~original: Logical -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +logical: Logical,+weight_logical: Weights,+bias_logical: Bias,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,
  +activation: Activations.Activation,+count: U32,+batched: Bool,x: Layer.Tensor(),+x_contract: TiledPublic.Contract(~Logical,~original,logical,TensorConvolution.output_shape(batched,count,channels,height,width),x),
  weights: Layer.Tensor(),+w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_logical,[outputs,channels,kernel,kernel],weights),
  bias: Layer.Tensor(),+b_contract: TiledPublic.Contract(~Bias,~bias_value,bias_logical,[outputs],bias),
  pair: Layer.Tensor() & Layer.Tensor(),+activated: FunctionValues.ActivatedForward(~ImageProblems.Context<Logical>,~ImageProblems.value(~Activations.identity_value,~Logical,~original),activation,convolution_context(~Logical,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding),output_extents(batched,count,outputs,height,width,kernel,stride,padding),pair)) ->
  Forward(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,Layer.forward_activated(x,weights,bias,pair)):
  (output,z) = pair
  KernelGemm.Both{+output_held,+z_contract} = activated
  KernelGemm.Both{output_held,KernelGemm.Both{KernelGemm.Both{x_contract,z_contract},KernelGemm.Both{w_contract,b_contract}}}

# The convolution and its copies, as the cells of models.
def forward_packed(~Logical: Data,~original: Logical -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +logical: Logical,+weight_logical: Weights,+bias_logical: Bias,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,
  +activation: Activations.Activation,+count: U32,+batched: Bool,x2: Layer.Tensor(),+x2_contract: TiledPublic.Contract(~Logical,~original,logical,TensorConvolution.output_shape(batched,count,channels,height,width),x2),
  w2: Layer.Tensor(),+w2_contract: TiledPublic.Contract(~Weights,~weight_value,weight_logical,[outputs,channels,kernel,kernel],w2),
  b2: Layer.Tensor(),+b2_contract: TiledPublic.Contract(~Bias,~bias_value,bias_logical,[outputs],b2),
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: TiledPublic.Contract(~Logical,~original,logical,TensorConvolution.output_shape(batched,count,channels,height,width),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}),
  +wm: PiecesModel.Model<Tile.Tile8>,+ws: List<&2,U32>,+wf: Tile.Tile8,+ww: U32,+wa: Nat,+wl: U32,+wp: Grid.Position,+w_packed: TiledPublic.Contract(~Weights,~weight_value,weight_logical,[outputs,channels,kernel,kernel],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp}),
  +bm: PiecesModel.Model<Tile.Tile8>,+bs: List<&2,U32>,+bf: Tile.Tile8,+bw: U32,+ba: Nat,+bl: U32,+bp: Grid.Position,+b_packed: TiledPublic.Contract(~Bias,~bias_value,bias_logical,[outputs],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),
  +valid: {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding),Contract.listed(~F32,~Weights,~weight_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,weight_logical),Contract.listed(~F32,~Bias,~bias_value,ShapeLaws.elements([outputs]),0n,bias_logical),0n)) == True{} : Bool},
  +machine: {TensorConvolution.machine_support(TensorConvolution.Geometry{ConvCall.images(batched,count),batched,KernelShape.Dimensions{U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)}},Default.limits()) == True{} : Bool},
  +count_positive: {Nat.is_lt(0n,U32.to_nat(ConvCall.images(batched,count))) == True{} : Bool},
  +built: {FunctionValues.built_in(activation) == True{} : Bool}) ->
  Forward(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,Layer.forward_activated(x2,w2,b2,F.activate_forward(F.conv2d(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp},stride,padding,Activations.identity()),activation))):
  TiledPublic.by_packed(~ImageProblems.Context<Logical>,~ImageProblems.value(~Activations.identity_value,~Logical,~original),(that => Forward(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,Layer.forward_activated(x2,w2,b2,F.activate_forward(that,activation)))),convolution_context(~Logical,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding),output_extents(batched,count,outputs,height,width,kernel,stride,padding),
    F.conv2d(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp},stride,padding,Activations.identity()),
    ConvCall.conv2d_contract(~Activations.identity_value,~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,
      channels,outputs,height,width,kernel,stride,padding,Activations.identity(),count,batched,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp},x_packed,w_packed,b_packed,valid,machine,count_positive),
    cm => cs => cf => cw => ca => cl => cp => c_packed => activated_holds(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,x2,x2_contract,w2,w2_contract,b2,b2_contract,F.activate_forward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,cm),cs,cf,Tensors.Usable{},cw},ca,cl,cp},activation),
      FunctionValues.activate_forward(~ImageProblems.Context<Logical>,~ImageProblems.value(~Activations.identity_value,~Logical,~original),convolution_context(~Logical,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding),activation,output_extents(batched,count,outputs,height,width,kernel,stride,padding),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,cm),cs,cf,Tensors.Usable{},cw},ca,cl,cp},c_packed,built)))

# Forward of x, weights and bias of the contract of the shapes of a valid
# problem that the default machine supports, with a built-in activation.
def forward(~Logical: Data,~original: Logical -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,~Bias: Data,~bias_value: Bias -> U32 -> F32,
  +logical: Logical,+weight_logical: Weights,+bias_logical: Bias,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,
  +activation: Activations.Activation,+count: U32,+batched: Bool,x: Layer.Tensor(),+x_contract: TiledPublic.Contract(~Logical,~original,logical,TensorConvolution.output_shape(batched,count,channels,height,width),x),
  weights: Layer.Tensor(),+w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_logical,[outputs,channels,kernel,kernel],weights),
  bias: Layer.Tensor(),+b_contract: TiledPublic.Contract(~Bias,~bias_value,bias_logical,[outputs],bias),
  +valid: {ConvSpec.valid(ImageProblems.problem(~Logical,~original,logical,U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding),Contract.listed(~F32,~Weights,~weight_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,weight_logical),Contract.listed(~F32,~Bias,~bias_value,ShapeLaws.elements([outputs]),0n,bias_logical),0n)) == True{} : Bool},
  +machine: {TensorConvolution.machine_support(TensorConvolution.Geometry{ConvCall.images(batched,count),batched,KernelShape.Dimensions{U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)}},Default.limits()) == True{} : Bool},
  +count_positive: {Nat.is_lt(0n,U32.to_nat(ConvCall.images(batched,count))) == True{} : Bool},
  +built: {FunctionValues.built_in(activation) == True{} : Bool}) ->
  Forward(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,Layer.forward(x,weights,bias,Layer.settings(stride,padding,activation))):
  Copy.tiled_copied(~Logical,~original,~(pair => Forward(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,Layer.forward_copied(Layer.settings(stride,padding,activation),pair,Tiled.copy(weights),Tiled.copy(bias)))),
    logical,TensorConvolution.output_shape(batched,count,channels,height,width),x,x_contract,
    x1 => x2 => x1_contract => x2_contract => Copy.tiled_copied(~Weights,~weight_value,
      ~(pair => Forward(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,Layer.forward_copied(Layer.settings(stride,padding,activation),(x1,x2),pair,Tiled.copy(bias)))),
      weight_logical,[outputs,channels,kernel,kernel],weights,w_contract,
      w1 => w2 => w1_contract => w2_contract => Copy.tiled_copied(~Bias,~bias_value,
        ~(pair => Forward(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,Layer.forward_copied(Layer.settings(stride,padding,activation),(x1,x2),(w1,w2),pair))),
        bias_logical,[outputs],bias,b_contract,
        b1 => b2 => b1_contract => b2_contract => TiledPublic.by_packed(~Logical,~original,
          (that => Forward(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,Layer.forward_activated(x2,w2,b2,F.activate_forward(F.conv2d(that,w1,b1,stride,padding,Activations.identity()),activation)))),
          logical,TensorConvolution.output_shape(batched,count,channels,height,width),x1,x1_contract,
          xm => xs => xf => xw => xa => xl => xp => x_packed => TiledPublic.by_packed(~Weights,~weight_value,
            (that => Forward(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,Layer.forward_activated(x2,w2,b2,F.activate_forward(F.conv2d(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},that,b1,stride,padding,Activations.identity()),activation)))),
            weight_logical,[outputs,channels,kernel,kernel],w1,w1_contract,
            wm => ws => wf => ww => wa => wl => wp => w_packed => TiledPublic.by_packed(~Bias,~bias_value,
              (that => Forward(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,Layer.forward_activated(x2,w2,b2,F.activate_forward(F.conv2d(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},that,stride,padding,Activations.identity()),activation)))),
              bias_logical,[outputs],b1,b1_contract,
              bm => bs => bf => bw => ba => bl => bp => b_packed => forward_packed(~Logical,~original,~Weights,~weight_value,~Bias,~bias_value,logical,weight_logical,bias_logical,channels,outputs,height,width,kernel,stride,padding,activation,count,batched,x2,x2_contract,w2,w2_contract,b2,b2_contract,xm,xs,xf,xw,xa,xl,xp,x_packed,wm,ws,wf,ww,wa,wl,wp,w_packed,bm,bs,bf,bw,ba,bl,bp,b_packed,
                valid,machine,count_positive,built)))))))

# ---- Backward ----

# The input, weight and bias gradients of a call geometry, as the input and
# parameter gradient specifications read them from x, dz and the weights.
def InputGradient(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,tensor: Layer.Tensor()) -> Data:
  TiledPublic.Contract(~InputPublic.InputRead<Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,Weights>,~InputPublic.input_at(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),~Weights,~weight_value),InputPublic.InputRead{Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},weight_context,geometry},
    Gradient.input_shape(Call.arguments(mode,geometry)),tensor)

def WeightGradient(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,tensor: Layer.Tensor()) -> Data:
  TiledPublic.Contract(~Terms.WeightRead<Input,Leaf.Context<DY,FunctionValues.ActivationRead<Z>>>,~Terms.weight_at(~Input,~input_value,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value))),
    ParameterPublic.weight_read(~Input,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,input_context,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},mode,geometry),Call.weight_shape(geometry),tensor)

def BiasGradient(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,tensor: Layer.Tensor()) -> Data:
  TiledPublic.Contract(~Terms.BiasRead<Leaf.Context<DY,FunctionValues.ActivationRead<Z>>>,~Terms.bias_at(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value))),ParameterPublic.bias_read(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},mode,geometry),[Call.outputs_word(geometry)],tensor)

# backward_parameters: the weight and bias gradients, and the weights back.
def BackwardParameters(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,result: Layer.Tensor() & (Layer.Tensor() & Layer.Tensor())) -> Data:
  (weight_gradient,(bias_gradient,weights)) = result
  KernelGemm.Both<WeightGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,weight_gradient),KernelGemm.Both<BiasGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,bias_gradient),
    TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights)>>

# backward: the input gradient, then what backward_parameters gives.
def Backward(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,result: Layer.Tensor() & (Layer.Tensor() & (Layer.Tensor() & Layer.Tensor()))) -> Data:
  (dx,gradients) = result
  KernelGemm.Both<InputGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,dx),BackwardParameters(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,gradients)>

# The gradients and dz of the parameter step, with the shape of x it measured.
def ParametersCall(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,result: Layer.Parameters()) -> Data:
  (weight_gradient,(bias_gradient,(dz,(weights,shape)))) = result
  KernelGemm.Both<WeightGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,weight_gradient),KernelGemm.Both<BiasGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,bias_gradient),
    KernelGemm.Both<TiledPublic.Contract(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Call.input_shape(mode,geometry),dz),
      KernelGemm.Both<TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights),{shape == Gradient.input_shape(Call.arguments(mode,geometry)) : List<&2,U32>}>>>>

# The bias gradient of dz, with dz back.
def biased_holds(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,-weight_gradient: Layer.Tensor(),+weight_held: WeightGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,weight_gradient),
  -weights: Layer.Tensor(),+w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights),
  pair: Layer.Tensor() & Layer.Tensor(),+call: ParameterPublic.TiledBiasCall(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},mode,geometry,pair)) ->
  ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.biased(weight_gradient,weights,Gradient.input_shape(Call.arguments(mode,geometry)),pair)):
  (bias_gradient,dz) = pair
  KernelGemm.Both{+bias_held,+dz_contract} = call
  KernelGemm.Both{weight_held,KernelGemm.Both{bias_held,KernelGemm.Both{dz_contract,KernelGemm.Both{w_contract,{==}}}}}

# dz and x, as the cells of models: x measures the shape of the call and dz
# gives the bias gradient.
def measured_packed(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,-weight_gradient: Layer.Tensor(),+weight_held: WeightGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,weight_gradient),
  -weights: Layer.Tensor(),+w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights),
  +dm: PiecesModel.Model<Tile.Tile8>,+ds: List<&2,U32>,+df: Tile.Tile8,+dw: U32,+da: Nat,+dl: U32,+dp: Grid.Position,+d_packed: TiledPublic.Contract(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Call.input_shape(mode,geometry),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp}),
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: TiledPublic.Contract(~Input,~input_value,input_context,Gradient.input_shape(Call.arguments(mode,geometry)),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}),
  +positive: {Parameter.positive(ParameterPublic.gradient_image(mode,geometry)) == True{} : Bool}) ->
  ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.measured(weight_gradient,weights,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},F.shape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}))):
  KernelGemm.Both{+global,TiledPublic.Laid{+inside,+shaped,+restored,+counted,+pinned}} = x_packed
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(xs,xa,xl,xp),Gradient.input_shape(Call.arguments(mode,geometry)),restored) :
    ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.biased(weight_gradient,weights,_,Gradient.bias(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp})))
  biased_holds(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,weight_gradient,weight_held,weights,w_contract,Gradient.bias(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp}),
    ParameterPublic.tiled_bias(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},mode,geometry,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},d_packed,positive))

# The weight gradient of x and dz, with x and dz back.
def weighted_holds(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,-weights: Layer.Tensor(),+w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights),
  result: Layer.Tensor() & (Layer.Tensor() & Layer.Tensor()),
  +call: ParameterPublic.TiledWeightCall(~Input,~input_value,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),input_context,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},mode,geometry,result),
  +positive: {Parameter.positive(ParameterPublic.gradient_image(mode,geometry)) == True{} : Bool}) ->
  ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.weighted(weights,result)):
  (weight_gradient,(x,dz)) = result
  KernelGemm.Both{+weight_held,KernelGemm.Both{+x_contract,+dz_contract}} = call
  TiledPublic.by_packed(~Input,~input_value,(that => ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.measured(weight_gradient,weights,dz,F.shape(that)))),input_context,Gradient.input_shape(Call.arguments(mode,geometry)),x,x_contract,
    xm => xs => xf => xw => xa => xl => xp => x_packed => TiledPublic.by_packed(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),(that => ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.measured(weight_gradient,weights,that,F.shape(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp})))),Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Call.input_shape(mode,geometry),dz,dz_contract,
      dm => ds => df => dw => da => dl => dp => d_packed => measured_packed(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,weight_gradient,weight_held,weights,w_contract,dm,ds,df,dw,da,dl,dp,d_packed,xm,xs,xf,xw,xa,xl,xp,x_packed,positive)))

# The kernel the step reads from the shape of the weights is the kernel of
# the call geometry.
def kernel_exact(+geometry: Kernel.Geometry) -> {Layer.kernel(Call.weight_shape(geometry)) == Domain.kernel_word(geometry) : U32}:
  Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
  {==}

# The parameter step of dy, x, z and the weights as the cells of models.
def parameters_packed(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,+ym: PiecesModel.Model<Tile.Tile8>,+ys: List<&2,U32>,+yf: Tile.Tile8,+yw: U32,+ya: Nat,+yl: U32,+yp: Grid.Position,+y_packed: TiledPublic.Contract(~DY,~dy_value,dy_context,Call.input_shape(mode,geometry),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp}),
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: TiledPublic.Contract(~Input,~input_value,input_context,Gradient.input_shape(Call.arguments(mode,geometry)),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}),
  +zm: PiecesModel.Model<Tile.Tile8>,+zs: List<&2,U32>,+zf: Tile.Tile8,+zw: U32,+za: Nat,+zl: U32,+zp: Grid.Position,+z_packed: TiledPublic.Contract(~Z,~z_value,z_context,Call.input_shape(mode,geometry),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp}),
  +wm: PiecesModel.Model<Tile.Tile8>,+ws: List<&2,U32>,+wf: Tile.Tile8,+ww: U32,+wa: Nat,+wl: U32,+wp: Grid.Position,+w_packed: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp}),
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +valid: {Parameter.valid(ParameterPublic.input_image(mode,geometry),ParameterPublic.gradient_image(mode,geometry),Domain.kernel_word(geometry),Call.stride(geometry),
    Call.padding(geometry),Gradient.group_length()) == True{} : Bool},
  +built: {FunctionValues.built_in(activation) == True{} : Bool}) ->
  ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.parameters(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp}),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Layer.settings(Call.stride(geometry),Call.padding(geometry),activation))):
  KernelGemm.Both{+global,TiledPublic.Laid{+inside,+shaped,+restored,+counted,+pinned}} = w_packed
  +w_own = {KernelGemm.Both{global,TiledPublic.Laid{inside,shaped,restored,counted,pinned}} : TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp})}
  +positive = ParameterPublic.gradient_positive(mode,geometry,valid)
  +rank_room = Order.transitive(List.length(&2,U32,Call.input_shape(mode,geometry)),4n,U32.to_nat(Machine.capacity(limits)),Call.gradient_rank(mode,geometry),Call.four_room(limits))
  %Equal.sym(List<&2,U32>,Tiled.logical_shape(ws,wa,wl,wp),Call.weight_shape(geometry),restored) :
    ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.weighted(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Gradient.weight(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},F.activate_backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp},activation),Layer.kernel(_),Call.stride(geometry),Call.padding(geometry))))
  %Equal.sym(U32,Layer.kernel(Call.weight_shape(geometry)),Domain.kernel_word(geometry),kernel_exact(geometry)) :
    ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.weighted(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Gradient.weight(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},F.activate_backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp},activation),_,Call.stride(geometry),Call.padding(geometry))))
  weighted_holds(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},w_own,
    Gradient.weight(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},F.activate_backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp},activation),Domain.kernel_word(geometry),Call.stride(geometry),Call.padding(geometry)),
    ParameterPublic.tiled_weight(~Input,~input_value,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),input_context,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},mode,geometry,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},x_packed,F.activate_backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp},activation),
      FunctionValues.activate_backward(~DY,~dy_value,~Z,~z_value,activation,dy_context,z_context,limits,default_capacity,Call.input_shape(mode,geometry),
        ParameterPublic.gradient_nonempty(mode,geometry,positive),rank_room,built,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},y_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp},z_packed),
      valid),
    positive)

# What holds of the parameter step of dy, x, z and the weights as the cells of
# models holds of their parameter step.
def parameters_of(~Goal: Layer.Parameters() -> Data,~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,
  dy: Layer.Tensor(),+dy_contract: TiledPublic.Contract(~DY,~dy_value,dy_context,Call.input_shape(mode,geometry),dy),
  x: Layer.Tensor(),+x_contract: TiledPublic.Contract(~Input,~input_value,input_context,Gradient.input_shape(Call.arguments(mode,geometry)),x),
  z: Layer.Tensor(),+z_contract: TiledPublic.Contract(~Z,~z_value,z_context,Call.input_shape(mode,geometry),z),
  weights: Layer.Tensor(),+w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights),
  +valid: {Parameter.valid(ParameterPublic.input_image(mode,geometry),ParameterPublic.gradient_image(mode,geometry),Domain.kernel_word(geometry),Call.stride(geometry),
    Call.padding(geometry),Gradient.group_length()) == True{} : Bool},
  +built: {FunctionValues.built_in(activation) == True{} : Bool},
  body: @result: Layer.Parameters() -> @+call: ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,result) -> Goal(result)) ->
  Goal(Layer.parameters(dy,(x,z),weights,Layer.settings(Call.stride(geometry),Call.padding(geometry),activation))):
  TiledPublic.by_packed(~DY,~dy_value,(that => Goal(Layer.parameters(that,(x,z),weights,Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)))),dy_context,Call.input_shape(mode,geometry),dy,dy_contract,
    ym => ys => yf => yw => ya => yl => yp => y_packed => TiledPublic.by_packed(~Input,~input_value,(that => Goal(Layer.parameters(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(that,z),weights,Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)))),input_context,Gradient.input_shape(Call.arguments(mode,geometry)),x,x_contract,
      xm => xs => xf => xw => xa => xl => xp => x_packed => TiledPublic.by_packed(~Z,~z_value,(that => Goal(Layer.parameters(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},that),weights,Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)))),z_context,Call.input_shape(mode,geometry),z,z_contract,
        zm => zs => zf => zw => za => zl => zp => z_packed => TiledPublic.by_packed(~Weights,~weight_value,(that => Goal(Layer.parameters(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp}),that,Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)))),weight_context,Call.weight_shape(geometry),weights,w_contract,
          wm => ws => wf => ww => wa => wl => wp => w_packed => body(Layer.parameters(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp}),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)),
            parameters_packed(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,ym,ys,yf,yw,ya,yl,yp,y_packed,xm,xs,xf,xw,xa,xl,xp,x_packed,zm,zs,zf,zw,za,zl,zp,z_packed,wm,ws,wf,ww,wa,wl,wp,w_packed,Default.limits(),{==},valid,built))))))

# The height and width backward reads from the shape of x are those of the
# call geometry.
def image_size(+mode: Call.Mode,+geometry: Kernel.Geometry) -> {Layer.image(Gradient.input_shape(Call.arguments(mode,geometry))) == (Call.height(geometry),Call.width(geometry)) : U32 & U32}:
  match mode:
    case Call.Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Call.Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

# The input gradient of dz and the weights, after the parameter gradients.
def finished_holds(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,-weight_gradient: Layer.Tensor(),-bias_gradient: Layer.Tensor(),pair: Layer.Tensor() & Layer.Tensor(),
  +weight_held: WeightGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,weight_gradient),+bias_held: BiasGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,bias_gradient),
  +call: InputPublic.TiledInputCall(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),~Weights,~weight_value,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},weight_context,mode,geometry,pair)) ->
  Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.finished(weight_gradient,bias_gradient,pair)):
  (dx,weights) = pair
  KernelGemm.Both{+dx_held,+w_contract} = call
  KernelGemm.Both{dx_held,KernelGemm.Both{weight_held,KernelGemm.Both{bias_held,w_contract}}}

# The input gradient step reads the height and width of the shape of x.
def propagated_holds(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,result: Layer.Parameters(),+call: ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,result),
  +checks: {Call.checks(mode,geometry) == True{} : Bool},+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool}) ->
  Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.propagated(Call.stride(geometry),Call.padding(geometry),result)):
  (weight_gradient,(bias_gradient,(dz,(weights,shape)))) = result
  KernelGemm.Both{+weight_held,KernelGemm.Both{+bias_held,KernelGemm.Both{+dz_contract,KernelGemm.Both{+w_contract,+shape_exact}}}} = call
  %Equal.sym(List<&2,U32>,shape,Gradient.input_shape(Call.arguments(mode,geometry)),shape_exact) :
    Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.sized(Call.stride(geometry),Call.padding(geometry),weight_gradient,bias_gradient,dz,weights,Layer.image(_)))
  %Equal.sym(U32 & U32,Layer.image(Gradient.input_shape(Call.arguments(mode,geometry))),(Call.height(geometry),Call.width(geometry)),image_size(mode,geometry)) :
    Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.sized(Call.stride(geometry),Call.padding(geometry),weight_gradient,bias_gradient,dz,weights,_))
  TiledPublic.by_packed(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),(that => Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.sized(Call.stride(geometry),Call.padding(geometry),weight_gradient,bias_gradient,that,weights,
      (Call.height(geometry),Call.width(geometry))))),Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Call.input_shape(mode,geometry),dz,dz_contract,
    dm => ds => df => dw => da => dl => dp => d_packed => TiledPublic.by_packed(~Weights,~weight_value,(that => Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.sized(Call.stride(geometry),Call.padding(geometry),weight_gradient,bias_gradient,
        Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},that,(Call.height(geometry),Call.width(geometry))))),weight_context,Call.weight_shape(geometry),weights,w_contract,
      wm => ws => wf => ww => wa => wl => wp => w_packed => finished_holds(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,weight_gradient,bias_gradient,
        Gradient.input(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry)),weight_held,bias_held,
        InputPublic.tiled_input(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),~Weights,~weight_value,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},weight_context,mode,geometry,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,Tensors.Usable{},dw},da,dl,dp},d_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},w_packed,checks,images_positive))))

# Backward of dy, the saved x and z and the weights of their contracts in the
# shapes of a valid call, with a built-in activation: the input, weight and
# bias gradients hold the specifications of x, dz = dy * activation'(z) and
# the weights, and the weights come back.
def backward(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,dy: Layer.Tensor(),+dy_contract: TiledPublic.Contract(~DY,~dy_value,dy_context,Call.input_shape(mode,geometry),dy),
  x: Layer.Tensor(),+x_contract: TiledPublic.Contract(~Input,~input_value,input_context,Gradient.input_shape(Call.arguments(mode,geometry)),x),
  z: Layer.Tensor(),+z_contract: TiledPublic.Contract(~Z,~z_value,z_context,Call.input_shape(mode,geometry),z),
  weights: Layer.Tensor(),+w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights),
  +checks: {Call.checks(mode,geometry) == True{} : Bool},+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  +valid: {Parameter.valid(ParameterPublic.input_image(mode,geometry),ParameterPublic.gradient_image(mode,geometry),Domain.kernel_word(geometry),Call.stride(geometry),
    Call.padding(geometry),Gradient.group_length()) == True{} : Bool},
  +built: {FunctionValues.built_in(activation) == True{} : Bool}) ->
  Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.backward(dy,(x,z),weights,Layer.settings(Call.stride(geometry),Call.padding(geometry),activation))):
  parameters_of(~(result => Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.propagated(Call.stride(geometry),Call.padding(geometry),result))),~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    dy,dy_contract,x,x_contract,z,z_contract,weights,w_contract,valid,built,
    result => call => propagated_holds(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,result,call,checks,images_positive))

# The parameter gradients without the input gradient.
def without_input_holds(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,result: Layer.Parameters(),+call: ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,result)) -> BackwardParameters(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.without_input(result)):
  (weight_gradient,(bias_gradient,(dz,(weights,shape)))) = result
  KernelGemm.Both{+weight_held,KernelGemm.Both{+bias_held,KernelGemm.Both{+dz_contract,KernelGemm.Both{+w_contract,+shape_exact}}}} = call
  KernelGemm.Both{weight_held,KernelGemm.Both{bias_held,w_contract}}

# backward_parameters of dy, the saved x and z and the weights of their
# contracts: the weight and bias gradients, and the weights back.
def backward_parameters(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,
  ~weight_value: Weights -> U32 -> F32,+input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,
  +mode: Call.Mode,+geometry: Kernel.Geometry,dy: Layer.Tensor(),+dy_contract: TiledPublic.Contract(~DY,~dy_value,dy_context,Call.input_shape(mode,geometry),dy),
  x: Layer.Tensor(),+x_contract: TiledPublic.Contract(~Input,~input_value,input_context,Gradient.input_shape(Call.arguments(mode,geometry)),x),
  z: Layer.Tensor(),+z_contract: TiledPublic.Contract(~Z,~z_value,z_context,Call.input_shape(mode,geometry),z),
  weights: Layer.Tensor(),+w_contract: TiledPublic.Contract(~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights),
  +valid: {Parameter.valid(ParameterPublic.input_image(mode,geometry),ParameterPublic.gradient_image(mode,geometry),Domain.kernel_word(geometry),Call.stride(geometry),
    Call.padding(geometry),Gradient.group_length()) == True{} : Bool},
  +built: {FunctionValues.built_in(activation) == True{} : Bool}) ->
  BackwardParameters(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.backward_parameters(dy,(x,z),weights,Layer.settings(Call.stride(geometry),Call.padding(geometry),activation))):
  parameters_of(~(result => BackwardParameters(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,Layer.without_input(result))),~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,dy,dy_contract,x,x_contract,z,z_contract,weights,w_contract,valid,built,
    result => call => without_input_holds(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,result,call))
