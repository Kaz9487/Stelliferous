# Train.forward, Train.evaluate, Train.backward and Train.sgd_step_with of a
# convolution layer: each result holds what the theorems of the convolution
# layer state, and the layer comes back as its description states.
import Base
import ../../lib/tensor_training.bend as Train
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_layer.bend as Layer
import ../../lib/tensor_optimizer.bend as Optimizer
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/tensor_parameter_gradient.bend as Parameter
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/kernel_tile.bend as Tile
import ../../lib/tensor_grid.bend as Grid
import ../../lib/math_activation.bend as Activations
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/tensor_tiled_public.bend as Public
import ../../lib/proofs/tensor_pieces_model.bend as PiecesModel
import ../../lib/proofs/tensor_copy_contract.bend as Copy
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_tiled_relane.bend as Relane
import ../../lib/proofs/tensor_optimizer_values.bend as OptimizerValues
import ../../lib/proofs/tensor_pieces_combine_leaf.bend as Leaf
import ../../lib/proofs/tensor_function_values.bend as FunctionValues
import ../../lib/proofs/tensor_training_layer_values.bend as Layers
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../../lib/proofs/word_encoding.bend as Words
import ../specification.bend as ConvSpec
import ./image_problems.bend as ImageProblems
import ./convolution_call.bend as ConvCall
import ./input_gradient_call.bend as Call
import ./layer_public.bend as LayerPublic
import ./parameter_gradient_public.bend as ParameterPublic

# ---- Description ----

# The geometry of the gradient kernels for a convolution of the description.
def geometry_of(+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32) -> Kernel.Geometry:
  Kernel.Geometry{channels,outputs,height,width,kernel,stride,padding,
    U32.from_nat(KernelShape.axis_size(U32.to_nat(height),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding))),
    U32.from_nat(KernelShape.axis_size(U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)))}

# A positive number of images is a positive number of convolution images.
def count_positive(+mode: Call.Mode,+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(ConvCall.images(Call.batched(mode),Call.images(mode)))) == True{} : Bool}:
  match mode:
    case Call.Single{}: images_positive
    case Call.Batched{+batch}: images_positive

# A convolution of [O, C, k, k] weights and [O] bias in their slots, with its
# stride and padding.
def ConvolutionLayer(~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+channels: U32,+outputs: U32,+kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,layer: Train.Layer) -> Data:
  match layer:
    case Train.Convolution{weight_tensor,bias_tensor,layer_stride,layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      KernelGemm.Both<{layer_stride == stride : U32},KernelGemm.Both<{layer_padding == padding : U32},
        KernelGemm.Both<Layers.Slot(~P,~p_value,~WeightGraded,weights,[outputs,channels,kernel,kernel],weight_tensor,weight_gradient,weight_velocity),
          Layers.Slot(~P,~p_value,~BiasGraded,bias,[outputs],bias_tensor,bias_gradient,bias_velocity)>>>
    case layer: {False{} == True{} : Bool}

# The velocities of the weights and the bias, kept.
def ConvolutionKept(~P: Data,~p_value: P -> U32 -> F32,+channels: U32,+outputs: U32,+kernel: U32,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,
  weight_velocity: Optimizer.Velocity,bias_velocity: Optimizer.Velocity) -> Data:
  KernelGemm.Both<OptimizerValues.Kept(~P,~p_value,Layers.parameter_velocity(~P,weights),[outputs,channels,kernel,kernel],Layers.parameter_moving(~P,weights),weight_velocity),
    OptimizerValues.Kept(~P,~p_value,Layers.parameter_velocity(~P,bias),[outputs],Layers.parameter_moving(~P,bias),bias_velocity)>

# ---- Forward ----

# forward: what the layer theorem states of the output, x, z, the weights and
# the bias, the gradients and velocities as they were.
def ConvolutionForward(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,result: Tiled.Tiled & (Train.Saved & Train.Layer)) -> Data:
  (output,rest) = result
  (saved,layer) = rest
  match saved:
    case Train.Convolved{kept}:
      match layer:
        case Train.Convolution{weight_tensor,bias_tensor,layer_stride,layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
          KernelGemm.Both<{layer_stride == stride : U32},KernelGemm.Both<{layer_padding == padding : U32},
            KernelGemm.Both<LayerPublic.Forward(~X,~x_value,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding,Activations.identity(),Call.images(mode),Call.batched(mode),(output,(kept,(weight_tensor,bias_tensor)))),
              KernelGemm.Both<WeightGraded(weight_gradient),KernelGemm.Both<BiasGraded(bias_gradient),ConvolutionKept(~P,~p_value,channels,outputs,kernel,weights,bias,weight_velocity,bias_velocity)>>>>>
        case layer: {False{} == True{} : Bool}
    case saved: {False{} == True{} : Bool}

def forward_holds(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,-weight_gradient: Train.Gradient,-bias_gradient: Train.Gradient,-weight_velocity: Optimizer.Velocity,-bias_velocity: Optimizer.Velocity,result: Layer.Tensor() & (Layer.Saved() & (Layer.Tensor() & Layer.Tensor())),+held: LayerPublic.Forward(~X,~x_value,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding,Activations.identity(),Call.images(mode),Call.batched(mode),result),
  +weight_graded: WeightGraded(weight_gradient),+bias_graded: BiasGraded(bias_gradient),+kept: ConvolutionKept(~P,~p_value,channels,outputs,kernel,weights,bias,weight_velocity,bias_velocity)) ->
  ConvolutionForward(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.convolved(stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity,result)):
  (output,(saved,(weight_tensor,bias_tensor))) = result
  KernelGemm.Both{{==},KernelGemm.Both{{==},KernelGemm.Both{held,KernelGemm.Both{weight_graded,KernelGemm.Both{bias_graded,kept}}}}}

def forward_packed(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+valid: {ConvSpec.valid(ImageProblems.problem(~X,~x_value,x,U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding),Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,Layers.parameter_value(~P,weights)),Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs]),0n,Layers.parameter_value(~P,bias)),0n)) == True{} : Bool},
  +machine: {TensorConvolution.machine_support(TensorConvolution.Geometry{ConvCall.images(Call.batched(mode),Call.images(mode)),Call.batched(mode),KernelShape.Dimensions{U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)}},Default.limits()) == True{} : Bool},
  +images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},-weight_gradient: Train.Gradient,-bias_gradient: Train.Gradient,-weight_velocity: Optimizer.Velocity,-bias_velocity: Optimizer.Velocity,+weight_graded: WeightGraded(weight_gradient),+bias_graded: BiasGraded(bias_gradient),
  +kept: ConvolutionKept(~P,~p_value,channels,outputs,kernel,weights,bias,weight_velocity,bias_velocity),
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: Public.Contract(~X,~x_value,x,TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}),+wm: PiecesModel.Model<Tile.Tile8>,+ws: List<&2,U32>,+wf: Tile.Tile8,+ww: U32,+wa: Nat,+wl: U32,+wp: Grid.Position,+w_packed: Public.Contract(~P,~p_value,Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp}),
  +bm: PiecesModel.Model<Tile.Tile8>,+bs: List<&2,U32>,+bf: Tile.Tile8,+bw: U32,+ba: Nat,+bl: U32,+bp: Grid.Position,+b_packed: Public.Contract(~P,~p_value,Layers.parameter_value(~P,bias),[outputs],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp})) ->
  ConvolutionForward(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.convolved(stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity,Layer.forward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp},Layer.settings(stride,padding,Activations.identity())))):
  forward_holds(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,weight_gradient,bias_gradient,weight_velocity,bias_velocity,Layer.forward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp},Layer.settings(stride,padding,Activations.identity())),
    LayerPublic.forward(~X,~x_value,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding,Activations.identity(),Call.images(mode),Call.batched(mode),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},x_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},w_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp},b_packed,valid,machine,count_positive(mode,images_positive),{==}),
    weight_graded,bias_graded,kept)

# forward of a convolution of the description on its tensors.
def ForwardOf(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,weight_gradient: Train.Gradient,bias_gradient: Train.Gradient,weight_velocity: Optimizer.Velocity,bias_velocity: Optimizer.Velocity,
  input: Tiled.Tiled,weight_tensor: Tiled.Tiled,bias_tensor: Tiled.Tiled) -> Data:
  ConvolutionForward(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.convolved(stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity,Layer.forward(input,weight_tensor,bias_tensor,Layer.settings(stride,padding,Activations.identity()))))

# forward of a convolution of the description on x of the contract.
def convolution_forward(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+valid: {ConvSpec.valid(ImageProblems.problem(~X,~x_value,x,U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding),Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,Layers.parameter_value(~P,weights)),Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs]),0n,Layers.parameter_value(~P,bias)),0n)) == True{} : Bool},
  +machine: {TensorConvolution.machine_support(TensorConvolution.Geometry{ConvCall.images(Call.batched(mode),Call.images(mode)),Call.batched(mode),KernelShape.Dimensions{U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)}},Default.limits()) == True{} : Bool},
  +images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width),tiled),
  layer: Train.Layer,+held: ConvolutionLayer(~P,~p_value,~WeightGraded,~BiasGraded,channels,outputs,kernel,stride,padding,weights,bias,layer)) ->
  ConvolutionForward(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.forward(tiled,layer)):
  match layer:
    case Train.Convolution{weight_tensor,bias_tensor,+layer_stride,+layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      KernelGemm.Both{+stride_exact,KernelGemm.Both{+padding_exact,KernelGemm.Both{KernelGemm.Both{+weight_contract,KernelGemm.Both{+weight_graded,+weight_kept}},
        KernelGemm.Both{+bias_contract,KernelGemm.Both{+bias_graded,+bias_kept}}}}} = held
      %Equal.sym(U32,layer_stride,stride,stride_exact) :
        ConvolutionForward(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.forward(tiled,Train.Convolution{weight_tensor,bias_tensor,_,layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}))
      %Equal.sym(U32,layer_padding,padding,padding_exact) :
        ConvolutionForward(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.forward(tiled,Train.Convolution{weight_tensor,bias_tensor,stride,_,weight_gradient,bias_gradient,weight_velocity,bias_velocity}))
      Public.by_packed(~X,~x_value,(that => ForwardOf(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,weight_gradient,bias_gradient,weight_velocity,bias_velocity,that,weight_tensor,bias_tensor)),x,TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width),tiled,contract,
        xm => xs => xf => xw => xa => xl => xp => x_packed => Public.by_packed(~P,~p_value,(that => ForwardOf(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,weight_gradient,bias_gradient,weight_velocity,bias_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},that,bias_tensor)),Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],weight_tensor,weight_contract,
          wm => ws => wf => ww => wa => wl => wp => w_packed => Public.by_packed(~P,~p_value,(that => ForwardOf(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,weight_gradient,bias_gradient,weight_velocity,bias_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},that)),Layers.parameter_value(~P,bias),[outputs],bias_tensor,bias_contract,
            bm => bs => bf => bw => ba => bl => bp => b_packed => forward_packed(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,valid,machine,images_positive,weight_gradient,bias_gradient,weight_velocity,bias_velocity,weight_graded,bias_graded,KernelGemm.Both{weight_kept,bias_kept},
              xm,xs,xf,xw,xa,xl,xp,x_packed,wm,ws,wf,ww,wa,wl,wp,w_packed,bm,bs,bf,bw,ba,bl,bp,b_packed))))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(ConvolutionForward(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.forward(tiled,Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(ConvolutionForward(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.forward(tiled,Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(ConvolutionForward(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.forward(tiled,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(ConvolutionForward(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.forward(tiled,Train.AveragePool{})),Words.false_true(held))

# ---- Evaluate ----

# evaluate: the output holds the convolution value, and the weights, the bias,
# the gradients and velocities come back as they were.
def ConvolutionEvaluate(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,result: Tiled.Tiled & Train.Layer) -> Data:
  (output,layer) = result
  match layer:
    case Train.Convolution{weight_tensor,bias_tensor,layer_stride,layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      KernelGemm.Both<{layer_stride == stride : U32},KernelGemm.Both<{layer_padding == padding : U32},
        KernelGemm.Both<Public.Contract(~ImageProblems.Context<X>,~ImageProblems.value(~Activations.identity_value,~X,~x_value),LayerPublic.convolution_context(~X,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding),LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding),output),
          KernelGemm.Both<Public.Contract(~P,~p_value,Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],weight_tensor),KernelGemm.Both<Public.Contract(~P,~p_value,Layers.parameter_value(~P,bias),[outputs],bias_tensor),
            KernelGemm.Both<WeightGraded(weight_gradient),KernelGemm.Both<BiasGraded(bias_gradient),ConvolutionKept(~P,~p_value,channels,outputs,kernel,weights,bias,weight_velocity,bias_velocity)>>>>>>>
    case layer: {False{} == True{} : Bool}

def evaluate_packed(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+valid: {ConvSpec.valid(ImageProblems.problem(~X,~x_value,x,U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding),Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,Layers.parameter_value(~P,weights)),Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs]),0n,Layers.parameter_value(~P,bias)),0n)) == True{} : Bool},
  +machine: {TensorConvolution.machine_support(TensorConvolution.Geometry{ConvCall.images(Call.batched(mode),Call.images(mode)),Call.batched(mode),KernelShape.Dimensions{U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)}},Default.limits()) == True{} : Bool},
  +images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},-weight_gradient: Train.Gradient,-bias_gradient: Train.Gradient,-weight_velocity: Optimizer.Velocity,-bias_velocity: Optimizer.Velocity,+weight_graded: WeightGraded(weight_gradient),+bias_graded: BiasGraded(bias_gradient),
  +kept: ConvolutionKept(~P,~p_value,channels,outputs,kernel,weights,bias,weight_velocity,bias_velocity),
  +xm: PiecesModel.Model<Tile.Tile8>,+xs: List<&2,U32>,+xf: Tile.Tile8,+xw: U32,+xa: Nat,+xl: U32,+xp: Grid.Position,+x_packed: Public.Contract(~X,~x_value,x,TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp}),+wm: PiecesModel.Model<Tile.Tile8>,+ws: List<&2,U32>,+wf: Tile.Tile8,+ww: U32,+wa: Nat,+wl: U32,+wp: Grid.Position,+w_packed: Public.Contract(~P,~p_value,Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp}),
  +bm: PiecesModel.Model<Tile.Tile8>,+bs: List<&2,U32>,+bf: Tile.Tile8,+bw: U32,+ba: Nat,+bl: U32,+bp: Grid.Position,+b_packed: Public.Contract(~P,~p_value,Layers.parameter_value(~P,bias),[outputs],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp}),
  -weights_kept: Tiled.Tiled,+weights_contract: Public.Contract(~P,~p_value,Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],weights_kept),
  -bias_kept: Tiled.Tiled,+bias_contract: Public.Contract(~P,~p_value,Layers.parameter_value(~P,bias),[outputs],bias_kept)) ->
  ConvolutionEvaluate(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.convolution_evaluated(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity,(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},weights_kept),(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp},bias_kept))):
  KernelGemm.Both{{==},KernelGemm.Both{{==},KernelGemm.Both{
    ConvCall.conv2d_contract(~Activations.identity_value,~X,~x_value,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding,
      Activations.identity(),Call.images(mode),Call.batched(mode),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,bm),bs,bf,Tensors.Usable{},bw},ba,bl,bp},x_packed,w_packed,b_packed,valid,machine,count_positive(mode,images_positive)),
    KernelGemm.Both{weights_contract,KernelGemm.Both{bias_contract,KernelGemm.Both{weight_graded,KernelGemm.Both{bias_graded,kept}}}}}}}

# evaluate of a convolution of the description on its tensors and copies.
def EvaluateOf(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,weight_gradient: Train.Gradient,bias_gradient: Train.Gradient,weight_velocity: Optimizer.Velocity,bias_velocity: Optimizer.Velocity,
  input: Tiled.Tiled,weight_copies: Tiled.Tiled & Tiled.Tiled,bias_copies: Tiled.Tiled & Tiled.Tiled) -> Data:
  ConvolutionEvaluate(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.convolution_evaluated(input,stride,padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity,weight_copies,bias_copies))

# evaluate of a convolution of the description on x of the contract.
def convolution_evaluate(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+valid: {ConvSpec.valid(ImageProblems.problem(~X,~x_value,x,U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding),Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,Layers.parameter_value(~P,weights)),Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs]),0n,Layers.parameter_value(~P,bias)),0n)) == True{} : Bool},
  +machine: {TensorConvolution.machine_support(TensorConvolution.Geometry{ConvCall.images(Call.batched(mode),Call.images(mode)),Call.batched(mode),KernelShape.Dimensions{U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)}},Default.limits()) == True{} : Bool},
  +images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},tiled: Tiled.Tiled,+contract: Public.Contract(~X,~x_value,x,TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width),tiled),
  layer: Train.Layer,+held: ConvolutionLayer(~P,~p_value,~WeightGraded,~BiasGraded,channels,outputs,kernel,stride,padding,weights,bias,layer)) ->
  ConvolutionEvaluate(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.evaluate(tiled,layer)):
  match layer:
    case Train.Convolution{weight_tensor,bias_tensor,+layer_stride,+layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      KernelGemm.Both{+stride_exact,KernelGemm.Both{+padding_exact,KernelGemm.Both{KernelGemm.Both{+weight_contract,KernelGemm.Both{+weight_graded,+weight_kept}},
        KernelGemm.Both{+bias_contract,KernelGemm.Both{+bias_graded,+bias_kept}}}}} = held
      %Equal.sym(U32,layer_stride,stride,stride_exact) :
        ConvolutionEvaluate(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.evaluate(tiled,Train.Convolution{weight_tensor,bias_tensor,_,layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}))
      %Equal.sym(U32,layer_padding,padding,padding_exact) :
        ConvolutionEvaluate(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.evaluate(tiled,Train.Convolution{weight_tensor,bias_tensor,stride,_,weight_gradient,bias_gradient,weight_velocity,bias_velocity}))
      Copy.tiled_copied(~P,~p_value,~(pair => EvaluateOf(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,weight_gradient,bias_gradient,weight_velocity,bias_velocity,tiled,pair,Tiled.copy(bias_tensor))),Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],weight_tensor,weight_contract,
        w1 => w2 => w1_contract => w2_contract => Copy.tiled_copied(~P,~p_value,~(pair => EvaluateOf(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,weight_gradient,bias_gradient,weight_velocity,bias_velocity,tiled,(w1,w2),pair)),Layers.parameter_value(~P,bias),[outputs],bias_tensor,
          bias_contract,
          b1 => b2 => b1_contract => b2_contract => Public.by_packed(~X,~x_value,(that => EvaluateOf(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,weight_gradient,bias_gradient,weight_velocity,bias_velocity,that,(w1,w2),(b1,b2))),x,TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width),tiled,contract,
            xm => xs => xf => xw => xa => xl => xp => x_packed => Public.by_packed(~P,~p_value,(that => EvaluateOf(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,weight_gradient,bias_gradient,weight_velocity,bias_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(that,w2),(b1,b2))),Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],w1,w1_contract,
              wm => ws => wf => ww => wa => wl => wp => w_packed => Public.by_packed(~P,~p_value,(that => EvaluateOf(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,weight_gradient,bias_gradient,weight_velocity,bias_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,xm),xs,xf,Tensors.Usable{},xw},xa,xl,xp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},w2),(that,b2))),Layers.parameter_value(~P,bias),[outputs],b1,b1_contract,
                bm => bs => bf => bw => ba => bl => bp => b_packed => evaluate_packed(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,valid,machine,images_positive,weight_gradient,bias_gradient,weight_velocity,bias_velocity,weight_graded,bias_graded,KernelGemm.Both{weight_kept,bias_kept},
                  xm,xs,xf,xw,xa,xl,xp,x_packed,wm,ws,wf,ww,wa,wl,wp,w_packed,bm,bs,bf,bw,ba,bl,bp,b_packed,w2,w2_contract,b2,b2_contract))))))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(ConvolutionEvaluate(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.evaluate(tiled,Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(ConvolutionEvaluate(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.evaluate(tiled,Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(ConvolutionEvaluate(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.evaluate(tiled,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(ConvolutionEvaluate(~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,Train.evaluate(tiled,Train.AveragePool{})),Words.false_true(held))

# ---- Backward ----

# dz = dy * activation'(z), with z the convolution of x: the context and
# reader that the gradients read.
def Delta(~DY: Data,~X: Data) -> Data:
  Leaf.Context<DY,FunctionValues.ActivationRead<ImageProblems.Context<X>>>

def delta_value(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32) -> Delta(~DY,~X) -> U32 -> F32:
  Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<ImageProblems.Context<X>>,~F32.mul,~dy_value,
    ~FunctionValues.derivative_at(~ImageProblems.Context<X>,~ImageProblems.value(~Activations.identity_value,~X,~x_value)))

# What backward reads: x and z.
def ConvolutionPair(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,pair: Tiled.Tiled & Tiled.Tiled) -> Data:
  (input,value) = pair
  KernelGemm.Both<Public.Contract(~X,~x_value,x,TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width),input),Public.Contract(~ImageProblems.Context<X>,~ImageProblems.value(~Activations.identity_value,~X,~x_value),LayerPublic.convolution_context(~X,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding),LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding),value)>

def ConvolutionSaved(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,saved: Train.Saved) -> Data:
  match saved:
    case Train.Convolved{pair}: ConvolutionPair(~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,pair)
    case saved: {False{} == True{} : Bool}

# backward: what the layer theorem states of dx and the weight and bias
# gradients of dz = dy * activation'(z), the weights and the bias, and the
# velocities as they were.
def ConvolutionBackward(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+dy: DY,result: Tiled.Tiled & Train.Layer) -> Data:
  (dx,layer) = result
  match layer:
    case Train.Convolution{weight_tensor,bias_tensor,layer_stride,layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      match weight_gradient:
        case Train.Held{dweights}:
          match bias_gradient:
            case Train.Held{dbias}:
              KernelGemm.Both<{layer_stride == stride : U32},KernelGemm.Both<{layer_padding == padding : U32},
                KernelGemm.Both<LayerPublic.InputGradient(~X,~x_value,~DY,~dy_value,~ImageProblems.Context<X>,~ImageProblems.value(~Activations.identity_value,~X,~x_value),~P,~p_value,x,dy,LayerPublic.convolution_context(~X,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding),Layers.parameter_value(~P,weights),Activations.identity(),mode,geometry_of(channels,outputs,height,width,kernel,stride,padding),dx),KernelGemm.Both<LayerPublic.WeightGradient(~X,~x_value,~DY,~dy_value,~ImageProblems.Context<X>,~ImageProblems.value(~Activations.identity_value,~X,~x_value),~P,~p_value,x,dy,LayerPublic.convolution_context(~X,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding),Layers.parameter_value(~P,weights),Activations.identity(),mode,geometry_of(channels,outputs,height,width,kernel,stride,padding),dweights),KernelGemm.Both<LayerPublic.BiasGradient(~X,~x_value,~DY,~dy_value,~ImageProblems.Context<X>,~ImageProblems.value(~Activations.identity_value,~X,~x_value),~P,~p_value,x,dy,LayerPublic.convolution_context(~X,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding),Layers.parameter_value(~P,weights),Activations.identity(),mode,geometry_of(channels,outputs,height,width,kernel,stride,padding),dbias),
                  KernelGemm.Both<Public.Contract(~P,~p_value,Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],weight_tensor),KernelGemm.Both<Public.Contract(~P,~p_value,Layers.parameter_value(~P,bias),[outputs],bias_tensor),
                    ConvolutionKept(~P,~p_value,channels,outputs,kernel,weights,bias,weight_velocity,bias_velocity)>>>>>>>
            case Train.Empty{}: {False{} == True{} : Bool}
        case Train.Empty{}: {False{} == True{} : Bool}
    case layer: {False{} == True{} : Bool}

def back_holds(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+dy: DY,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+weights_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([outputs,channels,kernel,kernel])) == True{} : Bool},
  -bias_tensor: Tiled.Tiled,-weight_velocity: Optimizer.Velocity,-bias_velocity: Optimizer.Velocity,
  result: Layer.Tensor() & (Layer.Tensor() & (Layer.Tensor() & Layer.Tensor())),+held: LayerPublic.Backward(~X,~x_value,~DY,~dy_value,~ImageProblems.Context<X>,~ImageProblems.value(~Activations.identity_value,~X,~x_value),~P,~p_value,x,dy,LayerPublic.convolution_context(~X,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding),Layers.parameter_value(~P,weights),Activations.identity(),mode,geometry_of(channels,outputs,height,width,kernel,stride,padding),result),
  +bias_contract: Public.Contract(~P,~p_value,Layers.parameter_value(~P,bias),[outputs],bias_tensor),+kept: ConvolutionKept(~P,~p_value,channels,outputs,kernel,weights,bias,weight_velocity,bias_velocity)) ->
  ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.convolution_back(stride,padding,bias_tensor,weight_velocity,bias_velocity,result)):
  (dx,(dweights,(dbias,weight_tensor))) = result
  KernelGemm.Both{+input_held,KernelGemm.Both{+weight_held,KernelGemm.Both{+bias_held,+weight_contract}}} = held
  KernelGemm.Both{{==},KernelGemm.Both{{==},KernelGemm.Both{input_held,KernelGemm.Both{weight_held,KernelGemm.Both{bias_held,
    KernelGemm.Both{Relane.tiled_along(~P,~p_value,Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],weight_tensor,0n,limits,default_capacity,{==},weights_nonempty,Layers.four_room(limits),weight_contract),
      KernelGemm.Both{bias_contract,kept}}}}}}}

def backward_packed(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+dy: DY,+checks: {Call.checks(mode,geometry_of(channels,outputs,height,width,kernel,stride,padding)) == True{} : Bool},+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  +valid: {Parameter.valid(ParameterPublic.input_image(mode,geometry_of(channels,outputs,height,width,kernel,stride,padding)),ParameterPublic.gradient_image(mode,geometry_of(channels,outputs,height,width,kernel,stride,padding)),Domain.kernel_word(geometry_of(channels,outputs,height,width,kernel,stride,padding)),Call.stride(geometry_of(channels,outputs,height,width,kernel,stride,padding)),Call.padding(geometry_of(channels,outputs,height,width,kernel,stride,padding)),Gradient.group_length()) == True{} : Bool},
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+weights_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([outputs,channels,kernel,kernel])) == True{} : Bool},
  -bias_tensor: Tiled.Tiled,-weight_velocity: Optimizer.Velocity,-bias_velocity: Optimizer.Velocity,
  +bias_contract: Public.Contract(~P,~p_value,Layers.parameter_value(~P,bias),[outputs],bias_tensor),+kept: ConvolutionKept(~P,~p_value,channels,outputs,kernel,weights,bias,weight_velocity,bias_velocity),
  +ym: PiecesModel.Model<Tile.Tile8>,+ys: List<&2,U32>,+yf: Tile.Tile8,+yw: U32,+ya: Nat,+yl: U32,+yp: Grid.Position,+y_packed: Public.Contract(~DY,~dy_value,dy,LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp}),+nm: PiecesModel.Model<Tile.Tile8>,+ns: List<&2,U32>,+nf: Tile.Tile8,+nw: U32,+na: Nat,+nl: U32,+np: Grid.Position,+i_packed: Public.Contract(~X,~x_value,x,TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,nm),ns,nf,Tensors.Usable{},nw},na,nl,np}),
  +zm: PiecesModel.Model<Tile.Tile8>,+zs: List<&2,U32>,+zf: Tile.Tile8,+zw: U32,+za: Nat,+zl: U32,+zp: Grid.Position,+z_packed: Public.Contract(~ImageProblems.Context<X>,~ImageProblems.value(~Activations.identity_value,~X,~x_value),LayerPublic.convolution_context(~X,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding),LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp}),+wm: PiecesModel.Model<Tile.Tile8>,+ws: List<&2,U32>,+wf: Tile.Tile8,+ww: U32,+wa: Nat,+wl: U32,+wp: Grid.Position,+w_packed: Public.Contract(~P,~p_value,Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp})) ->
  ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.convolution_back(stride,padding,bias_tensor,weight_velocity,bias_velocity,
    Layer.backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,nm),ns,nf,Tensors.Usable{},nw},na,nl,np},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp}),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Layer.settings(stride,padding,Activations.identity())))):
  back_holds(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,limits,default_capacity,weights_nonempty,bias_tensor,weight_velocity,bias_velocity,
    Layer.backward(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,nm),ns,nf,Tensors.Usable{},nw},na,nl,np},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp}),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},Layer.settings(stride,padding,Activations.identity())),
    LayerPublic.backward(~X,~x_value,~DY,~dy_value,~ImageProblems.Context<X>,~ImageProblems.value(~Activations.identity_value,~X,~x_value),~P,~p_value,x,dy,LayerPublic.convolution_context(~X,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding),Layers.parameter_value(~P,weights),Activations.identity(),mode,geometry_of(channels,outputs,height,width,kernel,stride,padding),Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},y_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,nm),ns,nf,Tensors.Usable{},nw},na,nl,np},i_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp},z_packed,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,wm),ws,wf,Tensors.Usable{},ww},wa,wl,wp},w_packed,checks,images_positive,valid,{==}),bias_contract,kept)

# backward of a convolution of the description on its tensors.
def BackwardOf(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+dy: DY,bias_tensor: Tiled.Tiled,weight_velocity: Optimizer.Velocity,bias_velocity: Optimizer.Velocity,
  gradient: Tiled.Tiled,input: Tiled.Tiled,value: Tiled.Tiled,weight_tensor: Tiled.Tiled) -> Data:
  ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.convolution_back(stride,padding,bias_tensor,weight_velocity,bias_velocity,
    Layer.backward(gradient,(input,value),weight_tensor,Layer.settings(stride,padding,Activations.identity()))))

def backward_pair(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+dy: DY,+checks: {Call.checks(mode,geometry_of(channels,outputs,height,width,kernel,stride,padding)) == True{} : Bool},+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  +valid: {Parameter.valid(ParameterPublic.input_image(mode,geometry_of(channels,outputs,height,width,kernel,stride,padding)),ParameterPublic.gradient_image(mode,geometry_of(channels,outputs,height,width,kernel,stride,padding)),Domain.kernel_word(geometry_of(channels,outputs,height,width,kernel,stride,padding)),Call.stride(geometry_of(channels,outputs,height,width,kernel,stride,padding)),Call.padding(geometry_of(channels,outputs,height,width,kernel,stride,padding)),Gradient.group_length()) == True{} : Bool},
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+weights_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([outputs,channels,kernel,kernel])) == True{} : Bool},gradient: Tiled.Tiled,+dy_contract: Public.Contract(~DY,~dy_value,dy,LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding),gradient),
  pair: Tiled.Tiled & Tiled.Tiled,+kept: ConvolutionPair(~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,pair),layer: Train.Layer,+held: ConvolutionLayer(~P,~p_value,~WeightGraded,~BiasGraded,channels,outputs,kernel,stride,padding,weights,bias,layer)) ->
  ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,Train.Convolved{pair},layer)):
  (input,value) = pair
  KernelGemm.Both{+input_contract,+value_contract} = kept
  match layer:
    case Train.Convolution{weight_tensor,bias_tensor,+layer_stride,+layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      KernelGemm.Both{+stride_exact,KernelGemm.Both{+padding_exact,KernelGemm.Both{KernelGemm.Both{+weight_contract,KernelGemm.Both{+weight_graded,+weight_kept}},
        KernelGemm.Both{+bias_contract,KernelGemm.Both{+bias_graded,+bias_kept}}}}} = held
      %Equal.sym(U32,layer_stride,stride,stride_exact) :
        ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,Train.Convolved{(input,value)},
          Train.Convolution{weight_tensor,bias_tensor,_,layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}))
      %Equal.sym(U32,layer_padding,padding,padding_exact) :
        ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,Train.Convolved{(input,value)},
          Train.Convolution{weight_tensor,bias_tensor,stride,_,weight_gradient,bias_gradient,weight_velocity,bias_velocity}))
      Public.by_packed(~DY,~dy_value,(that => BackwardOf(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,bias_tensor,weight_velocity,bias_velocity,that,input,value,weight_tensor)),dy,LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding),gradient,dy_contract,
        ym => ys => yf => yw => ya => yl => yp => y_packed => Public.by_packed(~X,~x_value,(that => BackwardOf(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,bias_tensor,weight_velocity,bias_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},that,value,weight_tensor)),x,TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width),input,input_contract,
          nm => ns => nf => nw => na => nl => np => i_packed => Public.by_packed(~ImageProblems.Context<X>,~ImageProblems.value(~Activations.identity_value,~X,~x_value),(that => BackwardOf(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,bias_tensor,weight_velocity,bias_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,nm),ns,nf,Tensors.Usable{},nw},na,nl,np},that,weight_tensor)),LayerPublic.convolution_context(~X,~P,~p_value,~P,~p_value,x,Layers.parameter_value(~P,weights),Layers.parameter_value(~P,bias),channels,outputs,height,width,kernel,stride,padding),LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding),value,value_contract,
            zm => zs => zf => zw => za => zl => zp => z_packed => Public.by_packed(~P,~p_value,(that => BackwardOf(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,bias_tensor,weight_velocity,bias_velocity,Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,ym),ys,yf,Tensors.Usable{},yw},ya,yl,yp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,nm),ns,nf,Tensors.Usable{},nw},na,nl,np},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,zm),zs,zf,Tensors.Usable{},zw},za,zl,zp},that)),Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],weight_tensor,weight_contract,
              wm => ws => wf => ww => wa => wl => wp => w_packed => backward_packed(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,checks,images_positive,valid,limits,default_capacity,weights_nonempty,bias_tensor,weight_velocity,bias_velocity,bias_contract,
                KernelGemm.Both{weight_kept,bias_kept},ym,ys,yf,yw,ya,yl,yp,y_packed,nm,ns,nf,nw,na,nl,np,i_packed,zm,zs,zf,zw,za,zl,zp,z_packed,wm,ws,wf,ww,wa,wl,wp,w_packed)))))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,Train.Convolved{(input,value)},Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,Train.Convolved{(input,value)},Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,Train.Convolved{(input,value)},Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,Train.Convolved{(input,value)},Train.AveragePool{})),Words.false_true(held))

# backward of a convolution of the description, for dy of the contract and
# what forward saved.
def convolution_backward(~DY: Data,~dy_value: DY -> U32 -> F32,~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,~WeightGraded: Train.Gradient -> Data,~BiasGraded: Train.Gradient -> Data,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+x: X,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+dy: DY,+checks: {Call.checks(mode,geometry_of(channels,outputs,height,width,kernel,stride,padding)) == True{} : Bool},+images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
  +valid: {Parameter.valid(ParameterPublic.input_image(mode,geometry_of(channels,outputs,height,width,kernel,stride,padding)),ParameterPublic.gradient_image(mode,geometry_of(channels,outputs,height,width,kernel,stride,padding)),Domain.kernel_word(geometry_of(channels,outputs,height,width,kernel,stride,padding)),Call.stride(geometry_of(channels,outputs,height,width,kernel,stride,padding)),Call.padding(geometry_of(channels,outputs,height,width,kernel,stride,padding)),Gradient.group_length()) == True{} : Bool},
  +limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+weights_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([outputs,channels,kernel,kernel])) == True{} : Bool},gradient: Tiled.Tiled,+dy_contract: Public.Contract(~DY,~dy_value,dy,LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding),gradient),
  saved: Train.Saved,+kept: ConvolutionSaved(~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,saved),layer: Train.Layer,+held: ConvolutionLayer(~P,~p_value,~WeightGraded,~BiasGraded,channels,outputs,kernel,stride,padding,weights,bias,layer)) ->
  ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,saved,layer)):
  match saved:
    case Train.Convolved{pair}:
      backward_pair(~DY,~dy_value,~X,~x_value,~P,~p_value,~WeightGraded,~BiasGraded,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,checks,images_positive,valid,limits,default_capacity,weights_nonempty,gradient,dy_contract,pair,kept,layer,held)
    case Train.Normalized{other_pair}:
      Empty.absurd(ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,Train.Normalized{other_pair},layer)),Words.false_true(kept))
    case Train.LineNormalized{other_saved}:
      Empty.absurd(ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,Train.LineNormalized{other_saved},layer)),Words.false_true(kept))
    case Train.Activated{other_input}:
      Empty.absurd(ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,Train.Activated{other_input},layer)),Words.false_true(kept))
    case Train.Pooled{other_shape}:
      Empty.absurd(ConvolutionBackward(~DY,~dy_value,~X,~x_value,~P,~p_value,mode,channels,outputs,height,width,kernel,stride,padding,x,weights,bias,dy,Train.backward(gradient,Train.Pooled{other_shape},layer)),Words.false_true(kept))

# ---- One step ----

# After a step: the weights and the bias stepped, the gradients cleared, the
# stride and padding as they were.
def ConvolutionStepped(~P: Data,~p_value: P -> U32 -> F32,~GG: Data,~gg_value: GG -> U32 -> F32,~BG: Data,~bg_value: BG -> U32 -> F32,+channels: U32,+outputs: U32,+kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+weight_moved: GG,+bias_moved: BG,+rate: F32,+momentum: F32,+decay: F32,layer: Train.Layer) -> Data:
  match layer:
    case Train.Convolution{weight_tensor,bias_tensor,layer_stride,layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      KernelGemm.Both<{layer_stride == stride : U32},KernelGemm.Both<{layer_padding == padding : U32},
        KernelGemm.Both<Layers.Cleared(weight_gradient),KernelGemm.Both<Layers.Cleared(bias_gradient),
          KernelGemm.Both<Layers.ParameterStepped(~P,~p_value,~GG,~gg_value,weights,weight_moved,rate,momentum,decay,[outputs,channels,kernel,kernel],(weight_tensor,weight_velocity)),
            Layers.ParameterStepped(~P,~p_value,~BG,~bg_value,bias,bias_moved,rate,momentum,decay,[outputs],(bias_tensor,bias_velocity))>>>>>
    case layer: {False{} == True{} : Bool}

def stepped_holds(~P: Data,~p_value: P -> U32 -> F32,~GG: Data,~gg_value: GG -> U32 -> F32,~BG: Data,~bg_value: BG -> U32 -> F32,+channels: U32,+outputs: U32,+kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+weight_moved: GG,+bias_moved: BG,+rate: F32,+momentum: F32,+decay: F32,+layer_stride: U32,+layer_padding: U32,
  +stride_exact: {layer_stride == stride : U32},+padding_exact: {layer_padding == padding : U32},
  weight_result: Tiled.Tiled & Optimizer.Velocity,+weight_held: Layers.ParameterStepped(~P,~p_value,~GG,~gg_value,weights,weight_moved,rate,momentum,decay,[outputs,channels,kernel,kernel],weight_result),
  bias_result: Tiled.Tiled & Optimizer.Velocity,+bias_held: Layers.ParameterStepped(~P,~p_value,~BG,~bg_value,bias,bias_moved,rate,momentum,decay,[outputs],bias_result)) ->
  ConvolutionStepped(~P,~p_value,~GG,~gg_value,~BG,~bg_value,channels,outputs,kernel,stride,padding,weights,bias,weight_moved,bias_moved,rate,momentum,decay,Train.convolution_stepped(layer_stride,layer_padding,weight_result,bias_result)):
  (weight_tensor,weight_velocity) = weight_result
  (bias_tensor,bias_velocity) = bias_result
  KernelGemm.Both{stride_exact,KernelGemm.Both{padding_exact,KernelGemm.Both{{==},KernelGemm.Both{{==},KernelGemm.Both{weight_held,bias_held}}}}}

# Train.sgd_step_with of a convolution whose gradients were computed.
def convolution_step(~P: Data,~p_value: P -> U32 -> F32,~GG: Data,~gg_value: GG -> U32 -> F32,~BG: Data,~bg_value: BG -> U32 -> F32,+channels: U32,+outputs: U32,+kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+weight_moved: GG,+bias_moved: BG,+rate: F32,+momentum: F32,+decay: F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +weights_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([outputs,channels,kernel,kernel])) == True{} : Bool},+bias_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([outputs])) == True{} : Bool},
  layer: Train.Layer,+held: ConvolutionLayer(~P,~p_value,~(gradient => Layers.Computed(~GG,~gg_value,weight_moved,[outputs,channels,kernel,kernel],gradient)),
    ~(gradient => Layers.Computed(~BG,~bg_value,bias_moved,[outputs],gradient)),channels,outputs,kernel,stride,padding,weights,bias,layer)) ->
  ConvolutionStepped(~P,~p_value,~GG,~gg_value,~BG,~bg_value,channels,outputs,kernel,stride,padding,weights,bias,weight_moved,bias_moved,rate,momentum,decay,Train.sgd_step_with(layer,rate,momentum,decay)):
  match layer:
    case Train.Convolution{weight_tensor,bias_tensor,+layer_stride,+layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      KernelGemm.Both{+stride_exact,KernelGemm.Both{+padding_exact,KernelGemm.Both{KernelGemm.Both{+weight_contract,KernelGemm.Both{+weight_graded,+weight_kept}},
        KernelGemm.Both{+bias_contract,KernelGemm.Both{+bias_graded,+bias_kept}}}}} = held
      Layers.slot_stepped(~P,~p_value,~GG,~gg_value,~(stepped => ConvolutionStepped(~P,~p_value,~GG,~gg_value,~BG,~bg_value,channels,outputs,kernel,stride,padding,weights,bias,weight_moved,bias_moved,rate,momentum,decay,
          Train.convolution_stepped(layer_stride,layer_padding,stepped,Train.stepped(bias_tensor,bias_gradient,bias_velocity,Train.Descent{rate,momentum,decay})))),
        weights,weight_moved,rate,momentum,decay,[outputs,channels,kernel,kernel],limits,default_capacity,weights_nonempty,Layers.four_room(limits),weight_tensor,weight_contract,weight_gradient,weight_graded,
        weight_velocity,weight_kept,
        tm => ts => tf => tt => tw => ta => tl => tp => dm => ds => df => dt => dw => da => dl => dp => weight_model => weight_held =>
          Layers.slot_stepped(~P,~p_value,~BG,~bg_value,~(stepped => ConvolutionStepped(~P,~p_value,~GG,~gg_value,~BG,~bg_value,channels,outputs,kernel,stride,padding,weights,bias,weight_moved,bias_moved,rate,momentum,decay,
              Train.convolution_stepped(layer_stride,layer_padding,Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},OptimizerValues.velocity_of(weight_model),rate,momentum,decay),stepped))),
            bias,bias_moved,rate,momentum,decay,[outputs],limits,default_capacity,bias_nonempty,Layers.one_room(limits,outputs),bias_tensor,bias_contract,bias_gradient,bias_graded,
            bias_velocity,bias_kept,
            um => us => uf => ut => uw => ua => ul => up => em => es => ef => et => ew => ea => el => ep => bias_model => bias_held =>
              stepped_holds(~P,~p_value,~GG,~gg_value,~BG,~bg_value,channels,outputs,kernel,stride,padding,weights,bias,weight_moved,bias_moved,rate,momentum,decay,layer_stride,layer_padding,stride_exact,padding_exact,
                Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,tm),ts,tf,tt,tw},ta,tl,tp},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,dm),ds,df,dt,dw},da,dl,dp},OptimizerValues.velocity_of(weight_model),rate,momentum,decay),weight_held,
                Optimizer.sgd_with(Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,um),us,uf,ut,uw},ua,ul,up},Tiled.Tiled{Tensors.Tensor{PiecesModel.pack(~Tile.Tile8,em),es,ef,et,ew},ea,el,ep},OptimizerValues.velocity_of(bias_model),rate,momentum,decay),bias_held)))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(ConvolutionStepped(~P,~p_value,~GG,~gg_value,~BG,~bg_value,channels,outputs,kernel,stride,padding,weights,bias,weight_moved,bias_moved,rate,momentum,decay,Train.sgd_step_with(Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(ConvolutionStepped(~P,~p_value,~GG,~gg_value,~BG,~bg_value,channels,outputs,kernel,stride,padding,weights,bias,weight_moved,bias_moved,rate,momentum,decay,Train.sgd_step_with(Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(ConvolutionStepped(~P,~p_value,~GG,~gg_value,~BG,~bg_value,channels,outputs,kernel,stride,padding,weights,bias,weight_moved,bias_moved,rate,momentum,decay,Train.sgd_step_with(Train.Activation{other_activation},rate,momentum,decay)),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(ConvolutionStepped(~P,~p_value,~GG,~gg_value,~BG,~bg_value,channels,outputs,kernel,stride,padding,weights,bias,weight_moved,bias_moved,rate,momentum,decay,Train.sgd_step_with(Train.AveragePool{},rate,momentum,decay)),Words.false_true(held))

def cleared_slots(~P: Data,~p_value: P -> U32 -> F32,+channels: U32,+outputs: U32,+kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+rate: F32,+momentum: F32,+decay: F32,
  -weight_tensor: Tiled.Tiled,weight_gradient: Train.Gradient,-weight_velocity: Optimizer.Velocity,
  -bias_tensor: Tiled.Tiled,bias_gradient: Train.Gradient,-bias_velocity: Optimizer.Velocity,+layer_stride: U32,+layer_padding: U32,
  +stride_exact: {layer_stride == stride : U32},+padding_exact: {layer_padding == padding : U32},
  +weight_contract: Public.Contract(~P,~p_value,Layers.parameter_value(~P,weights),[outputs,channels,kernel,kernel],weight_tensor),+weight_graded: Layers.Cleared(weight_gradient),
  +weight_kept: OptimizerValues.Kept(~P,~p_value,Layers.parameter_velocity(~P,weights),[outputs,channels,kernel,kernel],Layers.parameter_moving(~P,weights),weight_velocity),
  +bias_contract: Public.Contract(~P,~p_value,Layers.parameter_value(~P,bias),[outputs],bias_tensor),+bias_graded: Layers.Cleared(bias_gradient),
  +bias_kept: OptimizerValues.Kept(~P,~p_value,Layers.parameter_velocity(~P,bias),[outputs],Layers.parameter_moving(~P,bias),bias_velocity)) ->
  ConvolutionLayer(~P,~p_value,~Layers.Cleared,~Layers.Cleared,channels,outputs,kernel,stride,padding,weights,bias,
    Train.convolution_stepped(layer_stride,layer_padding,Train.stepped(weight_tensor,weight_gradient,weight_velocity,Train.Descent{rate,momentum,decay}),
      Train.stepped(bias_tensor,bias_gradient,bias_velocity,Train.Descent{rate,momentum,decay}))):
  match weight_gradient:
    case Train.Empty{}:
      match bias_gradient:
        case Train.Empty{}:
          KernelGemm.Both{stride_exact,KernelGemm.Both{padding_exact,KernelGemm.Both{KernelGemm.Both{weight_contract,KernelGemm.Both{weight_graded,weight_kept}},
            KernelGemm.Both{bias_contract,KernelGemm.Both{bias_graded,bias_kept}}}}}
        case Train.Held{other}:
          Empty.absurd(ConvolutionLayer(~P,~p_value,~Layers.Cleared,~Layers.Cleared,channels,outputs,kernel,stride,padding,weights,bias,
            Train.convolution_stepped(layer_stride,layer_padding,(weight_tensor,weight_velocity),Train.stepped(bias_tensor,Train.Held{other},bias_velocity,Train.Descent{rate,momentum,decay}))),
            Words.false_true(bias_graded))
    case Train.Held{other}:
      Empty.absurd(ConvolutionLayer(~P,~p_value,~Layers.Cleared,~Layers.Cleared,channels,outputs,kernel,stride,padding,weights,bias,
        Train.convolution_stepped(layer_stride,layer_padding,Train.stepped(weight_tensor,Train.Held{other},weight_velocity,Train.Descent{rate,momentum,decay}),
          Train.stepped(bias_tensor,bias_gradient,bias_velocity,Train.Descent{rate,momentum,decay}))),Words.false_true(weight_graded))

# Train.sgd_step_with of a convolution without gradients: the layer as it was.
def convolution_step_cleared(~P: Data,~p_value: P -> U32 -> F32,+channels: U32,+outputs: U32,+kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<P>,+bias: Layers.Parameter<P>,+rate: F32,+momentum: F32,+decay: F32,
  layer: Train.Layer,+held: ConvolutionLayer(~P,~p_value,~Layers.Cleared,~Layers.Cleared,channels,outputs,kernel,stride,padding,weights,bias,layer)) ->
  ConvolutionLayer(~P,~p_value,~Layers.Cleared,~Layers.Cleared,channels,outputs,kernel,stride,padding,weights,bias,Train.sgd_step_with(layer,rate,momentum,decay)):
  match layer:
    case Train.Convolution{weight_tensor,bias_tensor,+layer_stride,+layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      KernelGemm.Both{+stride_exact,KernelGemm.Both{+padding_exact,KernelGemm.Both{KernelGemm.Both{+weight_contract,KernelGemm.Both{+weight_graded,+weight_kept}},
        KernelGemm.Both{+bias_contract,KernelGemm.Both{+bias_graded,+bias_kept}}}}} = held
      cleared_slots(~P,~p_value,channels,outputs,kernel,stride,padding,weights,bias,rate,momentum,decay,weight_tensor,weight_gradient,weight_velocity,bias_tensor,bias_gradient,bias_velocity,layer_stride,layer_padding,
        stride_exact,padding_exact,weight_contract,weight_graded,weight_kept,bias_contract,bias_graded,bias_kept)
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(ConvolutionLayer(~P,~p_value,~Layers.Cleared,~Layers.Cleared,channels,outputs,kernel,stride,padding,weights,bias,Train.sgd_step_with(Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(ConvolutionLayer(~P,~p_value,~Layers.Cleared,~Layers.Cleared,channels,outputs,kernel,stride,padding,weights,bias,Train.sgd_step_with(Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity},rate,momentum,decay)),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(ConvolutionLayer(~P,~p_value,~Layers.Cleared,~Layers.Cleared,channels,outputs,kernel,stride,padding,weights,bias,Train.sgd_step_with(Train.Activation{other_activation},rate,momentum,decay)),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(ConvolutionLayer(~P,~p_value,~Layers.Cleared,~Layers.Cleared,channels,outputs,kernel,stride,padding,weights,bias,Train.sgd_step_with(Train.AveragePool{},rate,momentum,decay)),Words.false_true(held))
