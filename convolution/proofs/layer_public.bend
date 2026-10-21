# The values of a training layer. Forward returns the activation of the
# ordered convolution value at every index, and keeps the input, the
# convolution value, the weights and the bias with their values. Backward
# forms dz = dy * activation'(z) and returns the input, weight and bias
# gradient specifications of x, dz and the weights, with the weights back.
import Base
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_f32.bend as F
import ../../lib/tensor_layer.bend as Layer
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/tensor_parameter_gradient.bend as Parameter
import ../../lib/kernel_input_gradient.bend as Kernel
import ../../lib/math_activation.bend as Activations
import ../../lib/storage_buffer.bend as Storage
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_combine_leaf.bend as Leaf
import ../../lib/proofs/tensor_copy_contract.bend as Copy
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_function_values.bend as FunctionValues
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/word_encoding.bend as Words
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_public_frame.bend as PublicFrame
import ./convolution_public.bend as ConvPublic
import ./parameter_gradient_readers.bend as Readers
import ./parameter_gradient_public.bend as ParameterPublic
import ./parameter_gradient_ready.bend as ParameterReady
import ./input_gradient_public.bend as InputPublic
import ./input_gradient_call.bend as Call

# The convolution value of the layer before its activation.
def convolution_context(~Logical: Data,~WeightContext: Data,~weight_value: WeightContext -> U32 -> F32,~BiasContext: Data,~bias_value: BiasContext -> U32 -> F32,
  +logical: Logical,+weight_context: WeightContext,+bias_context: BiasContext,+input: PublicFrame.NativeInput) -> BandConvolutionValues.Context<Logical>:
  PublicFrame.reference_context(~Logical,logical,Activations.identity(),input,ConvPublic.weight_values(~WeightContext,~weight_value,weight_context,input),
    ConvPublic.bias_values(~BiasContext,~bias_value,bias_context,input))

# The output holds the activation of the convolution value and the saved z
# the convolution value, both Ready in the output shape of the call; x, the
# weights and the bias come back with their values.
def Forward(~Logical: Data,~original: Logical -> U32 -> F32,~WeightContext: Data,~weight_value: WeightContext -> U32 -> F32,~BiasContext: Data,~bias_value: BiasContext -> U32 -> F32,
  +logical: Logical,+weight_context: WeightContext,+bias_context: BiasContext,+activation: Activations.Activation,+context: BandConvolutionValues.Context<Logical>,+shape: List<&2,U32>,
  result: Layer.Tensor() & (Layer.Saved() & (Layer.Tensor() & Layer.Tensor()))) -> Data:
  (output,((x,z),(weights,bias))) = result
  Shape.Both<
    FunctionValues.Activated(~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~Activations.identity_value,~Logical,~original),activation,context,shape,output),
    Shape.Both<Global.Contract(~F32,~Logical,~original,logical,x),
      Shape.Both<ConvPublic.Convolved(~Activations.identity_value,~Logical,~original,context,shape,z),
        Shape.Both<Global.Contract(~F32,~WeightContext,~weight_value,weight_context,weights),Global.Contract(~F32,~BiasContext,~bias_value,bias_context,bias)>>>>

# The activated convolution value is the output, the convolution value
# itself is saved.
def activated(~Logical: Data,~original: Logical -> U32 -> F32,~WeightContext: Data,~weight_value: WeightContext -> U32 -> F32,~BiasContext: Data,~bias_value: BiasContext -> U32 -> F32,
  +logical: Logical,+weight_context: WeightContext,+bias_context: BiasContext,+activation: Activations.Activation,+context: BandConvolutionValues.Context<Logical>,+shape: List<&2,U32>,
  -x: Layer.Tensor(),-weights: Layer.Tensor(),-bias: Layer.Tensor(),pair: Layer.Tensor() & Layer.Tensor(),
  x_contract: Global.Contract(~F32,~Logical,~original,logical,x),
  weight_contract: Global.Contract(~F32,~WeightContext,~weight_value,weight_context,weights),
  bias_contract: Global.Contract(~F32,~BiasContext,~bias_value,bias_context,bias),
  forwarded: FunctionValues.ActivatedForward(~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~Activations.identity_value,~Logical,~original),activation,context,shape,pair)) ->
  Forward(~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,context,shape,
    Layer.forward_activated(x,weights,bias,pair)):
  (output,z) = pair
  Shape.Both{output_values,z_values} = forwarded
  Shape.Both{output_values,Shape.Both{x_contract,Shape.Both{z_values,Shape.Both{weight_contract,bias_contract}}}}

# The convolution value is Ready in its shape: its bands are those of a model.
def convolution_kept(~Logical: Data,~original: Logical -> U32 -> F32,~WeightContext: Data,~weight_value: WeightContext -> U32 -> F32,~BiasContext: Data,~bias_value: BiasContext -> U32 -> F32,
  +logical: Logical,+weight_context: WeightContext,+bias_context: BiasContext,+activation: Activations.Activation,+context: BandConvolutionValues.Context<Logical>,
  -x: Layer.Tensor(),-weights: Layer.Tensor(),-bias: Layer.Tensor(),-z: Layer.Tensor(),+shape: List<&2,U32>,
  x_contract: Global.Contract(~F32,~Logical,~original,logical,x),
  weight_contract: Global.Contract(~F32,~WeightContext,~weight_value,weight_context,weights),
  bias_contract: Global.Contract(~F32,~BiasContext,~bias_value,bias_context,bias),
  convolved: ConvPublic.Convolved(~Activations.identity_value,~Logical,~original,context,shape,z),
  +built: {FunctionValues.built_in(activation) == True{} : Bool}) ->
  Forward(~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,context,shape,
    Layer.forward_activated(x,weights,bias,F.activate_forward(z,activation))):
  Shape.Both{Global.Usable{+model,+geometry,+fill,+workers,+equation},contract} = convolved
  %equation : Forward(~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,context,shape,
    Layer.forward_activated(x,weights,bias,F.activate_forward(_,activation)))
  activated(~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,context,shape,x,weights,bias,
    F.activate_forward(Tensors.Tensor{Model.pack(~F32,model),geometry,shape,fill,Tensors.Usable{},workers},activation),x_contract,weight_contract,bias_contract,
    FunctionValues.activate_forward(~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~Activations.identity_value,~Logical,~original),
      context,activation,model,geometry,shape,fill,workers,
      %Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,model),geometry,shape,fill,Tensors.Usable{},workers},z,equation) :
        Global.Contract(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~Activations.identity_value,~Logical,~original),context,_) contract,
      built))

# The input, the weights and the bias are copied: the first copies meet in
# the convolution, the second copies come back.
def input_copies(~Logical: Data,~original: Logical -> U32 -> F32,~WeightContext: Data,~weight_value: WeightContext -> U32 -> F32,~BiasContext: Data,~bias_value: BiasContext -> U32 -> F32,
  +logical: Logical,+weight_context: WeightContext,+bias_context: BiasContext,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +x_bands: Model.Model<F32>,+x_geometry: Bands.Geometry,+x_fill: F32,+x_workers: U32,x_evidence: Copy.Copied<F32,x_bands>,
  +x_contract: Global.Contract(~F32,~Logical,~original,logical,
    Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers}),
  +weight_bands: Model.Model<F32>,+weight_geometry: Bands.Geometry,+weight_fill: F32,+weight_workers: U32,weight_evidence: Copy.Copied<F32,weight_bands>,
  +weight_contract: Global.Contract(~F32,~WeightContext,~weight_value,weight_context,
    Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,ConvPublic.weight_extents(input),weight_fill,Tensors.Usable{},weight_workers}),
  +bias_bands: Model.Model<F32>,+bias_geometry: Bands.Geometry,+bias_fill: F32,+bias_workers: U32,bias_evidence: Copy.Copied<F32,bias_bands>,
  +bias_contract: Global.Contract(~F32,~BiasContext,~bias_value,bias_context,
    Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,ConvPublic.bias_extents(input),bias_fill,Tensors.Usable{},bias_workers}),
  +supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +validated: {PublicFrame.frame_valid(input,ConvPublic.weight_values(~WeightContext,~weight_value,weight_context,input),
    ConvPublic.bias_values(~BiasContext,~bias_value,bias_context,input)) == True{} : Bool},
  +built: {FunctionValues.built_in(activation) == True{} : Bool}) ->
  Forward(~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,
    convolution_context(~Logical,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,input),ConvPublic.output_extents(input),
    Layer.forward(
      Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers},
      Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,ConvPublic.weight_extents(input),weight_fill,Tensors.Usable{},weight_workers},
      Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,ConvPublic.bias_extents(input),bias_fill,Tensors.Usable{},bias_workers},
      Layer.settings(ConvPublic.stride(input),ConvPublic.padding(input),activation))):
  Copy.Copied{+x_first,+x_second,+x_equation,+x_first_content,+x_second_content} = x_evidence
  Copy.Copied{+weight_first,+weight_second,+weight_equation,+weight_first_content,+weight_second_content} = weight_evidence
  Copy.Copied{+bias_first,+bias_second,+bias_equation,+bias_first_content,+bias_second_content} = bias_evidence
  +context = convolution_context(~Logical,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,input)
  +shape = ConvPublic.output_extents(input)
  +settings = Layer.settings(ConvPublic.stride(input),ConvPublic.padding(input),activation)
  %x_equation : Forward(~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,context,shape,
    Layer.forward_copied(settings,
      Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),_,left => right =>
        (Tensors.Tensor{left,x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers},Tensors.Tensor{right,x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers})),
      F.copy(Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,ConvPublic.weight_extents(input),weight_fill,Tensors.Usable{},weight_workers}),
      F.copy(Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,ConvPublic.bias_extents(input),bias_fill,Tensors.Usable{},bias_workers})))
  %weight_equation : Forward(~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,context,shape,
    Layer.forward_copied(settings,
      (Tensors.Tensor{Model.pack(~F32,x_first),x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers},
        Tensors.Tensor{Model.pack(~F32,x_second),x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers}),
      Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),_,left => right =>
        (Tensors.Tensor{left,weight_geometry,ConvPublic.weight_extents(input),weight_fill,Tensors.Usable{},weight_workers},
          Tensors.Tensor{right,weight_geometry,ConvPublic.weight_extents(input),weight_fill,Tensors.Usable{},weight_workers})),
      F.copy(Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,ConvPublic.bias_extents(input),bias_fill,Tensors.Usable{},bias_workers})))
  %bias_equation : Forward(~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,context,shape,
    Layer.forward_copied(settings,
      (Tensors.Tensor{Model.pack(~F32,x_first),x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers},
        Tensors.Tensor{Model.pack(~F32,x_second),x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers}),
      (Tensors.Tensor{Model.pack(~F32,weight_first),weight_geometry,ConvPublic.weight_extents(input),weight_fill,Tensors.Usable{},weight_workers},
        Tensors.Tensor{Model.pack(~F32,weight_second),weight_geometry,ConvPublic.weight_extents(input),weight_fill,Tensors.Usable{},weight_workers}),
      Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),_,left => right =>
        (Tensors.Tensor{left,bias_geometry,ConvPublic.bias_extents(input),bias_fill,Tensors.Usable{},bias_workers},
          Tensors.Tensor{right,bias_geometry,ConvPublic.bias_extents(input),bias_fill,Tensors.Usable{},bias_workers}))))
  convolution_kept(~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,context,
    Tensors.Tensor{Model.pack(~F32,x_second),x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers},
    Tensors.Tensor{Model.pack(~F32,weight_second),weight_geometry,ConvPublic.weight_extents(input),weight_fill,Tensors.Usable{},weight_workers},
    Tensors.Tensor{Model.pack(~F32,bias_second),bias_geometry,ConvPublic.bias_extents(input),bias_fill,Tensors.Usable{},bias_workers},
    F.conv2d(
      Tensors.Tensor{Model.pack(~F32,x_first),x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers},
      Tensors.Tensor{Model.pack(~F32,weight_first),weight_geometry,ConvPublic.weight_extents(input),weight_fill,Tensors.Usable{},weight_workers},
      Tensors.Tensor{Model.pack(~F32,bias_first),bias_geometry,ConvPublic.bias_extents(input),bias_fill,Tensors.Usable{},bias_workers},
      ConvPublic.stride(input),ConvPublic.padding(input),Activations.identity()),
    shape,
    Copy.kept(~F32,~Logical,~original,logical,x_bands,x_second,x_geometry,ConvPublic.input_extents(input),x_fill,x_workers,x_second_content,x_contract),
    Copy.kept(~F32,~WeightContext,~weight_value,weight_context,weight_bands,weight_second,weight_geometry,ConvPublic.weight_extents(input),weight_fill,weight_workers,
      weight_second_content,weight_contract),
    Copy.kept(~F32,~BiasContext,~bias_value,bias_context,bias_bands,bias_second,bias_geometry,ConvPublic.bias_extents(input),bias_fill,bias_workers,
      bias_second_content,bias_contract),
    ConvPublic.conv2d(~Activations.identity_value,~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,
      Activations.identity(),input,Model.pack(~F32,x_first),x_geometry,x_fill,x_workers,
      Copy.kept(~F32,~Logical,~original,logical,x_bands,x_first,x_geometry,ConvPublic.input_extents(input),x_fill,x_workers,x_first_content,x_contract),
      weight_first,weight_geometry,weight_fill,weight_workers,
      Copy.kept(~F32,~WeightContext,~weight_value,weight_context,weight_bands,weight_first,weight_geometry,ConvPublic.weight_extents(input),weight_fill,weight_workers,
        weight_first_content,weight_contract),
      bias_first,bias_geometry,bias_fill,bias_workers,
      Copy.kept(~F32,~BiasContext,~bias_value,bias_context,bias_bands,bias_first,bias_geometry,ConvPublic.bias_extents(input),bias_fill,bias_workers,
        bias_first_content,bias_contract),
      supported,validated),
    built)

# Forward of an input, weights and bias of the convolution contract, in any
# layout, with a built-in activation. The output is Ready in the output shape
# of the call and holds the activation of the ordered convolution value at
# every index; the saved z is that convolution value in the same shape.
def forward(~Logical: Data,~original: Logical -> U32 -> F32,~WeightContext: Data,~weight_value: WeightContext -> U32 -> F32,~BiasContext: Data,~bias_value: BiasContext -> U32 -> F32,
  +logical: Logical,+weight_context: WeightContext,+bias_context: BiasContext,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +x_bands: Model.Model<F32>,+x_geometry: Bands.Geometry,+x_fill: F32,+x_workers: U32,
  +x_contract: Global.Contract(~F32,~Logical,~original,logical,
    Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers}),
  +weight_bands: Model.Model<F32>,+weight_geometry: Bands.Geometry,+weight_fill: F32,+weight_workers: U32,
  +weight_contract: Global.Contract(~F32,~WeightContext,~weight_value,weight_context,
    Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,ConvPublic.weight_extents(input),weight_fill,Tensors.Usable{},weight_workers}),
  +bias_bands: Model.Model<F32>,+bias_geometry: Bands.Geometry,+bias_fill: F32,+bias_workers: U32,
  +bias_contract: Global.Contract(~F32,~BiasContext,~bias_value,bias_context,
    Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,ConvPublic.bias_extents(input),bias_fill,Tensors.Usable{},bias_workers}),
  +supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +validated: {PublicFrame.frame_valid(input,ConvPublic.weight_values(~WeightContext,~weight_value,weight_context,input),
    ConvPublic.bias_values(~BiasContext,~bias_value,bias_context,input)) == True{} : Bool},
  +built: {FunctionValues.built_in(activation) == True{} : Bool}) ->
  Forward(~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,
    convolution_context(~Logical,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,input),ConvPublic.output_extents(input),
    Layer.forward(
      Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,ConvPublic.input_extents(input),x_fill,Tensors.Usable{},x_workers},
      Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,ConvPublic.weight_extents(input),weight_fill,Tensors.Usable{},weight_workers},
      Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,ConvPublic.bias_extents(input),bias_fill,Tensors.Usable{},bias_workers},
      Layer.settings(ConvPublic.stride(input),ConvPublic.padding(input),activation))):
  input_copies(~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,input,
    x_bands,x_geometry,x_fill,x_workers,Copy.copied(~F32,x_bands),x_contract,
    weight_bands,weight_geometry,weight_fill,weight_workers,Copy.copied(~F32,weight_bands),weight_contract,
    bias_bands,bias_geometry,bias_fill,bias_workers,Copy.copied(~F32,bias_bands),bias_contract,supported,validated,built)

# dz, the gradient before the activation, at an index: dy times the
# derivative of the activation at z.
def delta(~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,+dy_context: DY,+z_context: Z,+activation: Activations.Activation,+index: U32) ->
  {Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value),
      Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},index) ==
    F32.mul(dy_value(dy_context,index),FunctionValues.derivative_value(activation,z_value(z_context,index))) : F32}:
  {==}

# The gradients of a layer of a call geometry, as the parameter and input
# gradient specifications read them from x, dz and the weights.
def WeightGradient(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,-tensor: Layer.Tensor()) -> Data:
  Global.Contract(~F32,~Readers.WeightRead<Input,Leaf.Context<DY,FunctionValues.ActivationRead<Z>>>,
    ~Readers.weight_at(~Input,~input_value,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
      ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value))),
    Readers.WeightRead{input_context,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},
      Parameter.image(Gradient.input_shape(Call.arguments(mode,geometry))),Parameter.image(Call.input_shape(mode,geometry)),
      Layer.kernel(Call.weight_shape(geometry)),Call.stride(geometry),Call.padding(geometry),Gradient.group_length()},tensor)

def BiasGradient(~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,
  +dy_context: DY,+z_context: Z,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,-tensor: Layer.Tensor()) -> Data:
  Global.Contract(~F32,~Readers.BiasRead<Leaf.Context<DY,FunctionValues.ActivationRead<Z>>>,
    ~Readers.bias_at(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
      ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value))),
    Readers.BiasRead{Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Parameter.image(Call.input_shape(mode,geometry)),Gradient.group_length()},tensor)

def InputGradient(~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+geometry: Kernel.Geometry,-tensor: Layer.Tensor()) -> Data:
  Global.Contract(~F32,~InputPublic.InputRead<Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,Weights>,
    ~InputPublic.input_at(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
      ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),~Weights,~weight_value),
    InputPublic.InputRead{Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},weight_context,geometry},tensor)

# backward_parameters: the weight and bias gradients, and the weights back.
def BackwardParameters(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  result: Layer.Tensor() & (Layer.Tensor() & Layer.Tensor())) -> Data:
  (weight_gradient,(bias_gradient,weights)) = result
  Shape.Both<WeightGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,input_context,dy_context,z_context,activation,mode,geometry,weight_gradient),
    Shape.Both<BiasGradient(~DY,~dy_value,~Z,~z_value,dy_context,z_context,activation,mode,geometry,bias_gradient),
      Global.Shaped(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights)>>

# backward: the input gradient, then what backward_parameters gives.
def Backward(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  result: Layer.Tensor() & (Layer.Tensor() & (Layer.Tensor() & Layer.Tensor()))) -> Data:
  (dx,gradients) = result
  Shape.Both<InputGradient(~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,dy_context,z_context,weight_context,activation,geometry,dx),
    BackwardParameters(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,gradients)>

# The gradients and dz of the parameter step, with the shape of x it measured.
def ParametersCall(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  result: Layer.Parameters()) -> Data:
  (weight_gradient,(bias_gradient,(dz,(weights,shape)))) = result
  Shape.Both<WeightGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,input_context,dy_context,z_context,activation,mode,geometry,weight_gradient),
    Shape.Both<BiasGradient(~DY,~dy_value,~Z,~z_value,dy_context,z_context,activation,mode,geometry,bias_gradient),
      Shape.Both<Global.Shaped(~F32,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
          ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
          Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Call.input_shape(mode,geometry),dz),
        Shape.Both<Global.Shaped(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights),
          {shape == Gradient.input_shape(Call.arguments(mode,geometry)) : List<&2,U32>}>>>>

# Calls whose gradients are Ready.
def GradientsReady(result: Layer.Tensor() & (Layer.Tensor() & Layer.Tensor())) -> Data:
  (weight_gradient,(bias_gradient,weights)) = result
  Shape.Both<{Contract.tensor_ready(~F32,weight_gradient) == True{} : Bool},{Contract.tensor_ready(~F32,bias_gradient) == True{} : Bool}>

def BackwardReady(result: Layer.Tensor() & (Layer.Tensor() & (Layer.Tensor() & Layer.Tensor()))) -> Data:
  (dx,gradients) = result
  Shape.Both<{Contract.tensor_ready(~F32,dx) == True{} : Bool},GradientsReady(gradients)>

def ParametersReady(result: Layer.Parameters()) -> Data:
  (weight_gradient,(bias_gradient,rest)) = result
  Shape.Both<{Contract.tensor_ready(~F32,weight_gradient) == True{} : Bool},{Contract.tensor_ready(~F32,bias_gradient) == True{} : Bool}>

# The readiness of each call of the step, read back from the readiness of
# what the step returns.
def biased_ready(-weight_gradient: Layer.Tensor(),-weights: Layer.Tensor(),-shape: List<&2,U32>,pair: Layer.Tensor() & Layer.Tensor(),
  ready: ParametersReady(Layer.biased(weight_gradient,weights,shape,pair))) ->
  Shape.Both<{Contract.tensor_ready(~F32,weight_gradient) == True{} : Bool},{Contract.tensor_ready(~F32,ParameterPublic.bias_result(pair)) == True{} : Bool}>:
  (bias_gradient,dz) = pair
  ready

def measured_ready(-weight_gradient: Layer.Tensor(),-weights: Layer.Tensor(),dz: Layer.Tensor(),x: Layer.Tensor(),
  ready: ParametersReady(Layer.measured(weight_gradient,weights,dz,Tensors.shape(~F32,x)))) ->
  Shape.Both<{Contract.tensor_ready(~F32,weight_gradient) == True{} : Bool},{Contract.tensor_ready(~F32,ParameterPublic.bias_result(Gradient.bias(dz))) == True{} : Bool}>:
  Tensors.Tensor{bands,geometry,shape,fill,status,workers} = x
  biased_ready(weight_gradient,weights,shape,Gradient.bias(dz),ready)

def BiasLater(result: Layer.Tensor() & (Layer.Tensor() & Layer.Tensor())) -> Data:
  (weight_gradient,(x,dz)) = result
  {Contract.tensor_ready(~F32,ParameterPublic.bias_result(Gradient.bias(dz))) == True{} : Bool}

def WeightedReady(result: Layer.Tensor() & (Layer.Tensor() & Layer.Tensor())) -> Data:
  Shape.Both<{Contract.tensor_ready(~F32,ParameterPublic.weight_result(result)) == True{} : Bool},BiasLater(result)>

def weighted_ready(-weights: Layer.Tensor(),result: Layer.Tensor() & (Layer.Tensor() & Layer.Tensor()),ready: ParametersReady(Layer.weighted(weights,result))) ->
  WeightedReady(result):
  (weight_gradient,(x,dz)) = result
  measured_ready(weight_gradient,weights,dz,x,ready)

def finished_ready(-weight_gradient: Layer.Tensor(),-bias_gradient: Layer.Tensor(),pair: Layer.Tensor() & Layer.Tensor(),
  ready: BackwardReady(Layer.finished(weight_gradient,bias_gradient,pair))) ->
  Shape.Both<{Call.ready(pair) == True{} : Bool},Shape.Both<{Contract.tensor_ready(~F32,weight_gradient) == True{} : Bool},{Contract.tensor_ready(~F32,bias_gradient) == True{} : Bool}>>:
  (dx,weights) = pair
  ready

def SizedReady(dz: Layer.Tensor(),weights: Layer.Tensor(),+stride: U32,+padding: U32,size: U32 & U32) -> Data:
  (height,width) = size
  {Call.ready(Gradient.input(dz,weights,stride,padding,height,width)) == True{} : Bool}

def InputReady(+stride: U32,+padding: U32,result: Layer.Parameters()) -> Data:
  (weight_gradient,(bias_gradient,(dz,(weights,shape)))) = result
  SizedReady(dz,weights,stride,padding,Layer.image(shape))

def sized_ready(+stride: U32,+padding: U32,-weight_gradient: Layer.Tensor(),-bias_gradient: Layer.Tensor(),dz: Layer.Tensor(),weights: Layer.Tensor(),size: U32 & U32,
  ready: BackwardReady(Layer.sized(stride,padding,weight_gradient,bias_gradient,dz,weights,size))) ->
  Shape.Both<SizedReady(dz,weights,stride,padding,size),Shape.Both<{Contract.tensor_ready(~F32,weight_gradient) == True{} : Bool},{Contract.tensor_ready(~F32,bias_gradient) == True{} : Bool}>>:
  (height,width) = size
  finished_ready(weight_gradient,bias_gradient,Gradient.input(dz,weights,stride,padding,height,width),ready)

def propagated_ready(+stride: U32,+padding: U32,result: Layer.Parameters(),ready: BackwardReady(Layer.propagated(stride,padding,result))) ->
  Shape.Both<InputReady(stride,padding,result),ParametersReady(result)>:
  (weight_gradient,(bias_gradient,(dz,(weights,shape)))) = result
  sized_ready(stride,padding,weight_gradient,bias_gradient,dz,weights,Layer.image(shape),ready)

def without_input_ready(result: Layer.Parameters(),ready: GradientsReady(Layer.without_input(result))) -> ParametersReady(result):
  (weight_gradient,(bias_gradient,(dz,(weights,shape)))) = result
  ready

# A failed output gradient gives a failed weight gradient.
def weight_shaped_failed(xi: Maybe<&2,Parameter.ImageGeometry>,di: Maybe<&2,Parameter.ImageGeometry>,+k: U32,+s: U32,+p: U32,+g: U32,
  -x: Storage.Buffer<F32>,+xs: List<&2,U32>,-xf: F32,+xw: U32,-d: Storage.Buffer<F32>,-ds: List<&2,U32>,-df: F32,+problem: Tensors.Problem,-dw: U32) ->
  {Contract.tensor_ready(~F32,ParameterPublic.weight_result(Parameter.weight_shaped(xi,di,Tensors.Usable{},Tensors.Failed{problem},k,s,p,g,x,xs,xf,xw,d,ds,df,dw))) == False{} : Bool}:
  match xi di:
    case Some{+xi} Some{+di}: {==}
    case Some{+xi} None{}: {==}
    case None{} Some{+di}: {==}
    case None{} None{}: {==}

def weight_failed(-x_bands: Bands.Bands(~F32),+x_geometry: Bands.Geometry,+xs: List<&2,U32>,+x_fill: F32,+x_workers: U32,
  -d_bands: Bands.Bands(~F32),+d_geometry: Bands.Geometry,+ds: List<&2,U32>,+d_fill: F32,+problem: Tensors.Problem,+d_workers: U32,+k: U32,+s: U32,+p: U32) ->
  {Contract.tensor_ready(~F32,ParameterPublic.weight_result(Gradient.weight(Tensors.Tensor{x_bands,x_geometry,xs,x_fill,Tensors.Usable{},x_workers},
    Tensors.Tensor{d_bands,d_geometry,ds,d_fill,Tensors.Failed{problem},d_workers},k,s,p))) == False{} : Bool}:
  +x_whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(xs)} : Bands.Geometry}
  +d_whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(ds)} : Bands.Geometry}
  weight_shaped_failed(Parameter.image(xs),Parameter.image(ds),k,s,p,Gradient.group_length(),
    Tensors.whole_storage(~F32,x_whole,x_fill,Tensors.arrive(~F32,x_whole,Bands.unsplit(x_whole),Tensors.Tensor{x_bands,x_geometry,xs,x_fill,Tensors.Usable{},x_workers})),
    xs,x_fill,x_workers,
    Tensors.whole_storage(~F32,d_whole,d_fill,Tensors.arrive(~F32,d_whole,Bands.unsplit(d_whole),Tensors.Tensor{d_bands,d_geometry,ds,d_fill,Tensors.Failed{problem},d_workers})),
    ds,d_fill,problem,d_workers)

def delta_status(-x_bands: Bands.Bands(~F32),+x_geometry: Bands.Geometry,+xs: List<&2,U32>,+x_fill: F32,+x_workers: U32,
  -bands: Bands.Bands(~F32),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,status: Tensors.Status,+workers: U32,+k: U32,+s: U32,+p: U32,
  accepted: {Contract.tensor_ready(~F32,ParameterPublic.weight_result(Gradient.weight(Tensors.Tensor{x_bands,x_geometry,xs,x_fill,Tensors.Usable{},x_workers},
    Tensors.Tensor{bands,geometry,shape,fill,status,workers},k,s,p))) == True{} : Bool}) ->
  {Contract.tensor_ready(~F32,Tensors.Tensor{bands,geometry,shape,fill,status,workers}) == True{} : Bool}:
  match status:
    case Tensors.Usable{}: {==}
    case Tensors.Failed{+problem}:
      Empty.absurd({Contract.tensor_ready(~F32,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}) == True{} : Bool},
        Words.false_true(%weight_failed(x_bands,x_geometry,xs,x_fill,x_workers,bands,geometry,shape,fill,problem,workers,k,s,p) : {_ == True{} : Bool} accepted))

# A Ready weight gradient has a Ready output gradient.
def delta_ready(-x_bands: Bands.Bands(~F32),+x_geometry: Bands.Geometry,+xs: List<&2,U32>,+x_fill: F32,+x_workers: U32,dz: Layer.Tensor(),+k: U32,+s: U32,+p: U32,
  accepted: {Contract.tensor_ready(~F32,ParameterPublic.weight_result(Gradient.weight(Tensors.Tensor{x_bands,x_geometry,xs,x_fill,Tensors.Usable{},x_workers},dz,k,s,p))) == True{} : Bool}) ->
  {Contract.tensor_ready(~F32,dz) == True{} : Bool}:
  Tensors.Tensor{bands,+geometry,+shape,+fill,status,+workers} = dz
  delta_status(x_bands,x_geometry,xs,x_fill,x_workers,bands,geometry,shape,fill,status,workers,k,s,p,accepted)

# The weight gradient of x and dz, with x and dz back.
def delta_weight(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+activation: Activations.Activation,
  +x_bands: Model.Model<F32>,+x_geometry: Bands.Geometry,+xs: List<&2,U32>,+x_fill: F32,+x_workers: U32,
  +x_contract: Global.Contract(~F32,~Input,~input_value,input_context,Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,x_fill,Tensors.Usable{},x_workers}),
  +shape: List<&2,U32>,-dz: Layer.Tensor(),
  dz_values: FunctionValues.ActivatedBackward(~DY,~dy_value,~Z,~z_value,activation,dy_context,z_context,shape,dz),
  +k: U32,+s: U32,+p: U32,
  accepted: {Contract.tensor_ready(~F32,ParameterPublic.weight_result(Gradient.weight(
    Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,x_fill,Tensors.Usable{},x_workers},dz,k,s,p))) == True{} : Bool}) ->
  ParameterPublic.WeightCall(~Input,~input_value,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
    ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
    input_context,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Parameter.image(xs),Parameter.image(shape),k,s,p,Gradient.group_length(),xs,shape,
    Gradient.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,x_fill,Tensors.Usable{},x_workers},dz,k,s,p)):
  Shape.Both{usable,contract} = dz_values
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Layer.Tensor(),Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},dz,equation)
  %equation : ParameterPublic.WeightCall(~Input,~input_value,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
    ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
    input_context,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Parameter.image(xs),Parameter.image(shape),k,s,p,Gradient.group_length(),xs,shape,
    Gradient.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,x_fill,Tensors.Usable{},x_workers},_,k,s,p))
  ParameterReady.public_weight(~Input,~input_value,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
    ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
    input_context,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},x_bands,x_geometry,xs,x_fill,x_workers,x_contract,
    bands,geometry,shape,fill,workers,
    %back : Global.Contract(~F32,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
        ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
        Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},_)
    contract,
    k,s,p,
    %back : {Contract.tensor_ready(~F32,ParameterPublic.weight_result(Gradient.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,x_fill,Tensors.Usable{},x_workers},
        _,k,s,p))) == True{} : Bool}
    accepted)

# A Ready weight gradient of x and dz = activate_backward(dy, z): dz is
# Ready, so it holds dy times the derivative at z.
def weight_call(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+activation: Activations.Activation,
  +x_bands: Model.Model<F32>,+x_geometry: Bands.Geometry,+xs: List<&2,U32>,+x_fill: F32,+x_workers: U32,
  +x_contract: Global.Contract(~F32,~Input,~input_value,input_context,Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,x_fill,Tensors.Usable{},x_workers}),
  +shape: List<&2,U32>,+dy_bands: Model.Model<F32>,+dy_geometry: Bands.Geometry,+dy_fill: F32,+dy_workers: U32,
  +dy_contract: Global.Contract(~F32,~DY,~dy_value,dy_context,Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,shape,dy_fill,Tensors.Usable{},dy_workers}),
  +z_bands: Model.Model<F32>,+z_geometry: Bands.Geometry,+z_fill: F32,+z_workers: U32,
  +z_contract: Global.Contract(~F32,~Z,~z_value,z_context,Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,shape,z_fill,Tensors.Usable{},z_workers}),
  +k: U32,+s: U32,+p: U32,
  +accepted: {Contract.tensor_ready(~F32,ParameterPublic.weight_result(Gradient.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,x_fill,Tensors.Usable{},x_workers},
    F.activate_backward(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,shape,dy_fill,Tensors.Usable{},dy_workers},
      Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,shape,z_fill,Tensors.Usable{},z_workers},activation),k,s,p))) == True{} : Bool}) ->
  ParameterPublic.WeightCall(~Input,~input_value,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
    ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
    input_context,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Parameter.image(xs),Parameter.image(shape),k,s,p,Gradient.group_length(),xs,shape,
    Gradient.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,x_fill,Tensors.Usable{},x_workers},
      F.activate_backward(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,shape,dy_fill,Tensors.Usable{},dy_workers},
        Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,shape,z_fill,Tensors.Usable{},z_workers},activation),k,s,p)):
  delta_weight(~Input,~input_value,~DY,~dy_value,~Z,~z_value,input_context,dy_context,z_context,activation,x_bands,x_geometry,xs,x_fill,x_workers,x_contract,shape,
    F.activate_backward(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,shape,dy_fill,Tensors.Usable{},dy_workers},
      Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,shape,z_fill,Tensors.Usable{},z_workers},activation),
    FunctionValues.activate_backward(~DY,~dy_value,~Z,~z_value,activation,dy_context,z_context,shape,dy_bands,dy_geometry,dy_fill,dy_workers,dy_contract,
      z_bands,z_geometry,z_fill,z_workers,z_contract,
      delta_ready(Model.pack(~F32,x_bands),x_geometry,xs,x_fill,x_workers,
        F.activate_backward(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,shape,dy_fill,Tensors.Usable{},dy_workers},
          Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,shape,z_fill,Tensors.Usable{},z_workers},activation),
        k,s,p,accepted)),
    k,s,p,accepted)

# The bias gradient of dz, with dz back.
def biased_holds(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  -weight_gradient: Layer.Tensor(),weight_contract: WeightGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,input_context,dy_context,z_context,activation,mode,geometry,weight_gradient),
  +w_bands: Model.Model<F32>,+w_geometry: Bands.Geometry,+w_fill: F32,+w_workers: U32,
  +w_contract: Global.Contract(~F32,~Weights,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers}),
  pair: Layer.Tensor() & Layer.Tensor(),
  call: ParameterPublic.BiasCall(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
    ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
    Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Parameter.image(Call.input_shape(mode,geometry)),Gradient.group_length(),Call.input_shape(mode,geometry),pair)) ->
  ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.biased(weight_gradient,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Gradient.input_shape(Call.arguments(mode,geometry)),pair)):
  (bias_gradient,dz) = pair
  Shape.Both{bias_contract,dz_shaped} = call
  Shape.Both{weight_contract,Shape.Both{bias_contract,Shape.Both{dz_shaped,Shape.Both{
    Global.shaped(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),
      Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},{==},{==},w_contract),{==}}}}}

# dz comes back from the weight gradient as a tensor of its shape.
def measured_holds(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  -weight_gradient: Layer.Tensor(),weight_contract: WeightGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,input_context,dy_context,z_context,activation,mode,geometry,weight_gradient),
  +w_bands: Model.Model<F32>,+w_geometry: Bands.Geometry,+w_fill: F32,+w_workers: U32,
  +w_contract: Global.Contract(~F32,~Weights,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers}),
  -dz: Layer.Tensor(),
  dz_shaped: Global.Shaped(~F32,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
    ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
    Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Call.input_shape(mode,geometry),dz),
  later: {Contract.tensor_ready(~F32,ParameterPublic.bias_result(Gradient.bias(dz))) == True{} : Bool}) ->
  ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.biased(weight_gradient,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Gradient.input_shape(Call.arguments(mode,geometry)),Gradient.bias(dz))):
  Shape.Both{usable,contract} = dz_shaped
  Global.Usable{+dz_bands,+dz_geometry,+dz_fill,+dz_workers,+equation} = usable
  +back = Equal.sym(Layer.Tensor(),Tensors.Tensor{Model.pack(~F32,dz_bands),dz_geometry,Call.input_shape(mode,geometry),dz_fill,Tensors.Usable{},dz_workers},dz,equation)
  %equation : ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.biased(weight_gradient,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Gradient.input_shape(Call.arguments(mode,geometry)),Gradient.bias(_)))
  biased_holds(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    weight_gradient,weight_contract,w_bands,w_geometry,w_fill,w_workers,w_contract,
    Gradient.bias(Tensors.Tensor{Model.pack(~F32,dz_bands),dz_geometry,Call.input_shape(mode,geometry),dz_fill,Tensors.Usable{},dz_workers}),
    ParameterReady.public_bias(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
      ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
      Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},dz_bands,dz_geometry,Call.input_shape(mode,geometry),dz_fill,dz_workers,
      %back : Global.Contract(~F32,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
          ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
          Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},_)
        contract,
      %back : {Contract.tensor_ready(~F32,ParameterPublic.bias_result(Gradient.bias(_))) == True{} : Bool} later))

# x comes back from the weight gradient as a tensor of its shape, which is
# the shape the step measures.
def weighted_holds(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  +w_bands: Model.Model<F32>,+w_geometry: Bands.Geometry,+w_fill: F32,+w_workers: U32,
  +w_contract: Global.Contract(~F32,~Weights,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers}),
  result: Layer.Tensor() & (Layer.Tensor() & Layer.Tensor()),
  call: ParameterPublic.WeightCall(~Input,~input_value,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
    ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
    input_context,Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},
    Parameter.image(Gradient.input_shape(Call.arguments(mode,geometry))),Parameter.image(Call.input_shape(mode,geometry)),
    Layer.kernel(Call.weight_shape(geometry)),Call.stride(geometry),Call.padding(geometry),Gradient.group_length(),
    Gradient.input_shape(Call.arguments(mode,geometry)),Call.input_shape(mode,geometry),result),
  later: BiasLater(result)) ->
  ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.weighted(Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},result)):
  (weight_gradient,(x,dz)) = result
  Shape.Both{weight_contract,Shape.Both{x_shaped,dz_shaped}} = call
  Shape.Both{x_usable,x_contract} = x_shaped
  Global.Usable{+x_bands,+x_geometry,+x_fill,+x_workers,+equation} = x_usable
  %equation : ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.measured(weight_gradient,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},dz,Tensors.shape(~F32,_)))
  measured_holds(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    weight_gradient,weight_contract,w_bands,w_geometry,w_fill,w_workers,w_contract,dz,dz_shaped,later)

def weight_split(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  +dy_bands: Model.Model<F32>,+dy_geometry: Bands.Geometry,+dy_fill: F32,+dy_workers: U32,
  +dy_contract: Global.Contract(~F32,~DY,~dy_value,dy_context,Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers}),
  +x_bands: Model.Model<F32>,+x_geometry: Bands.Geometry,+x_fill: F32,+x_workers: U32,
  +x_contract: Global.Contract(~F32,~Input,~input_value,input_context,
    Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers}),
  +z_bands: Model.Model<F32>,+z_geometry: Bands.Geometry,+z_fill: F32,+z_workers: U32,
  +z_contract: Global.Contract(~F32,~Z,~z_value,z_context,Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
  +w_bands: Model.Model<F32>,+w_geometry: Bands.Geometry,+w_fill: F32,+w_workers: U32,
  +w_contract: Global.Contract(~F32,~Weights,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers}),
  readiness: WeightedReady(Gradient.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
    F.activate_backward(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
      Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers},activation),
    Layer.kernel(Call.weight_shape(geometry)),Call.stride(geometry),Call.padding(geometry)))) ->
  ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.weighted(Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Gradient.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
        F.activate_backward(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
          Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers},activation),
        Layer.kernel(Call.weight_shape(geometry)),Call.stride(geometry),Call.padding(geometry)))):
  Shape.Both{+accepted,later} = readiness
  weighted_holds(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    w_bands,w_geometry,w_fill,w_workers,w_contract,
    Gradient.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
      F.activate_backward(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
        Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers},activation),
      Layer.kernel(Call.weight_shape(geometry)),Call.stride(geometry),Call.padding(geometry)),
    weight_call(~Input,~input_value,~DY,~dy_value,~Z,~z_value,input_context,dy_context,z_context,activation,
      x_bands,x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,x_workers,x_contract,
      Call.input_shape(mode,geometry),dy_bands,dy_geometry,dy_fill,dy_workers,dy_contract,z_bands,z_geometry,z_fill,z_workers,z_contract,
      Layer.kernel(Call.weight_shape(geometry)),Call.stride(geometry),Call.padding(geometry),accepted),
    later)

# The parameter step: dz, the weight gradient of x and dz, the bias gradient
# of dz, with dz, the weights and the shape of x back.
def parameters_call(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  +dy_bands: Model.Model<F32>,+dy_geometry: Bands.Geometry,+dy_fill: F32,+dy_workers: U32,
  +dy_contract: Global.Contract(~F32,~DY,~dy_value,dy_context,Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers}),
  +x_bands: Model.Model<F32>,+x_geometry: Bands.Geometry,+x_fill: F32,+x_workers: U32,
  +x_contract: Global.Contract(~F32,~Input,~input_value,input_context,
    Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers}),
  +z_bands: Model.Model<F32>,+z_geometry: Bands.Geometry,+z_fill: F32,+z_workers: U32,
  +z_contract: Global.Contract(~F32,~Z,~z_value,z_context,Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
  +w_bands: Model.Model<F32>,+w_geometry: Bands.Geometry,+w_fill: F32,+w_workers: U32,
  +w_contract: Global.Contract(~F32,~Weights,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers}),
  ready: ParametersReady(Layer.parameters(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
    (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
      Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
    Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
    Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)))) ->
  ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.parameters(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
      (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
        Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
      Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Layer.settings(Call.stride(geometry),Call.padding(geometry),activation))):
  weight_split(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    dy_bands,dy_geometry,dy_fill,dy_workers,dy_contract,x_bands,x_geometry,x_fill,x_workers,x_contract,z_bands,z_geometry,z_fill,z_workers,z_contract,
    w_bands,w_geometry,w_fill,w_workers,w_contract,
    weighted_ready(Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Gradient.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
        F.activate_backward(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
          Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers},activation),
        Layer.kernel(Call.weight_shape(geometry)),Call.stride(geometry),Call.padding(geometry)),
      ready))

# The height and width backward reads from the shape of x are those of the
# call geometry.
def image_size(+mode: Call.Mode,+geometry: Kernel.Geometry) ->
  {Layer.image(Gradient.input_shape(Call.arguments(mode,geometry))) == (Call.height(geometry),Call.width(geometry)) : U32 & U32}:
  match mode:
    case Call.Single{}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}
    case Call.Batched{batch}:
      Kernel.Geometry{c,o,h,w,k,s,p,gh,gw} = geometry
      {==}

def finished_holds(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  -weight_gradient: Layer.Tensor(),-bias_gradient: Layer.Tensor(),
  weight_contract: WeightGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,input_context,dy_context,z_context,activation,mode,geometry,weight_gradient),
  bias_contract: BiasGradient(~DY,~dy_value,~Z,~z_value,dy_context,z_context,activation,mode,geometry,bias_gradient),
  pair: Layer.Tensor() & Layer.Tensor(),
  call: InputPublic.InputCall(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
    ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),~Weights,~weight_value,
    Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},weight_context,geometry,pair)) ->
  Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.finished(weight_gradient,bias_gradient,pair)):
  (dx,weights) = pair
  Shape.Both{dx_contract,weights_shaped} = call
  Shape.Both{dx_contract,Shape.Both{weight_contract,Shape.Both{bias_contract,weights_shaped}}}

# The input gradient of dz and the weights, both tensors of their shapes.
def input_shaped(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  -weight_gradient: Layer.Tensor(),-bias_gradient: Layer.Tensor(),
  weight_contract: WeightGradient(~Input,~input_value,~DY,~dy_value,~Z,~z_value,input_context,dy_context,z_context,activation,mode,geometry,weight_gradient),
  bias_contract: BiasGradient(~DY,~dy_value,~Z,~z_value,dy_context,z_context,activation,mode,geometry,bias_gradient),
  -dz: Layer.Tensor(),
  dz_shaped: Global.Shaped(~F32,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
    ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
    Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},Call.input_shape(mode,geometry),dz),
  -weights: Layer.Tensor(),weights_shaped: Global.Shaped(~F32,~Weights,~weight_value,weight_context,Call.weight_shape(geometry),weights),
  accepted: {Call.ready(Gradient.input(dz,weights,Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))) == True{} : Bool}) ->
  Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.finished(weight_gradient,bias_gradient,Gradient.input(dz,weights,Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry)))):
  Shape.Both{dz_usable,dz_contract} = dz_shaped
  Global.Usable{+dz_bands,+dz_geometry,+dz_fill,+dz_workers,+dz_equation} = dz_usable
  Shape.Both{weights_usable,weights_contract} = weights_shaped
  Global.Usable{+w_bands,+w_geometry,+w_fill,+w_workers,+w_equation} = weights_usable
  +dz_back = Equal.sym(Layer.Tensor(),Tensors.Tensor{Model.pack(~F32,dz_bands),dz_geometry,Call.input_shape(mode,geometry),dz_fill,Tensors.Usable{},dz_workers},dz,dz_equation)
  +w_back = Equal.sym(Layer.Tensor(),Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},weights,w_equation)
  %dz_equation : Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.finished(weight_gradient,bias_gradient,Gradient.input(_,weights,Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))))
  %w_equation : Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.finished(weight_gradient,bias_gradient,
      Gradient.input(Tensors.Tensor{Model.pack(~F32,dz_bands),dz_geometry,Call.input_shape(mode,geometry),dz_fill,Tensors.Usable{},dz_workers},_,
        Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))))
  finished_holds(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    weight_gradient,bias_gradient,weight_contract,bias_contract,
    Gradient.input(Tensors.Tensor{Model.pack(~F32,dz_bands),dz_geometry,Call.input_shape(mode,geometry),dz_fill,Tensors.Usable{},dz_workers},
      Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry)),
    InputPublic.input(~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
      ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),~Weights,~weight_value,
      Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},weight_context,mode,geometry,dz_bands,dz_geometry,dz_fill,dz_workers,
      %dz_back : Global.Contract(~F32,~Leaf.Context<DY,FunctionValues.ActivationRead<Z>>,
          ~Leaf.value(~F32,~DY,~FunctionValues.ActivationRead<Z>,~F32.mul,~dy_value,~FunctionValues.derivative_at(~Z,~z_value)),
          Leaf.Context{dy_context,FunctionValues.ActivationRead{activation,z_context}},_)
        dz_contract,
      w_bands,w_geometry,w_fill,w_workers,
      %w_back : Global.Contract(~F32,~Weights,~weight_value,weight_context,_) weights_contract,
      %w_back : {Call.ready(Gradient.input(Tensors.Tensor{Model.pack(~F32,dz_bands),dz_geometry,Call.input_shape(mode,geometry),dz_fill,Tensors.Usable{},dz_workers},_,
          Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))) == True{} : Bool}
      %dz_back : {Call.ready(Gradient.input(_,weights,Call.stride(geometry),Call.padding(geometry),Call.height(geometry),Call.width(geometry))) == True{} : Bool}
      accepted))

# The input gradient step reads the height and width of the shape of x.
def propagated_holds(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  result: Layer.Parameters(),
  call: ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,result),
  ready: InputReady(Call.stride(geometry),Call.padding(geometry),result)) ->
  Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.propagated(Call.stride(geometry),Call.padding(geometry),result)):
  (weight_gradient,(bias_gradient,(dz,(weights,shape)))) = result
  Shape.Both{weight_contract,Shape.Both{bias_contract,Shape.Both{dz_shaped,Shape.Both{weights_shaped,same}}}} = call
  +size = Equal.trans(U32 & U32,Layer.image(shape),Layer.image(Gradient.input_shape(Call.arguments(mode,geometry))),(Call.height(geometry),Call.width(geometry)),
    Equal.cong(List<&2,U32>,U32 & U32,extents => Layer.image(extents),shape,Gradient.input_shape(Call.arguments(mode,geometry)),same),image_size(mode,geometry))
  %Equal.sym(U32 & U32,Layer.image(shape),(Call.height(geometry),Call.width(geometry)),size) :
    Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
      Layer.sized(Call.stride(geometry),Call.padding(geometry),weight_gradient,bias_gradient,dz,weights,_))
  input_shaped(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    weight_gradient,bias_gradient,weight_contract,bias_contract,dz,dz_shaped,weights,weights_shaped,
    %size : SizedReady(dz,weights,Call.stride(geometry),Call.padding(geometry),_) ready)

def backward_split(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  +dy_bands: Model.Model<F32>,+dy_geometry: Bands.Geometry,+dy_fill: F32,+dy_workers: U32,
  +dy_contract: Global.Contract(~F32,~DY,~dy_value,dy_context,Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers}),
  +x_bands: Model.Model<F32>,+x_geometry: Bands.Geometry,+x_fill: F32,+x_workers: U32,
  +x_contract: Global.Contract(~F32,~Input,~input_value,input_context,
    Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers}),
  +z_bands: Model.Model<F32>,+z_geometry: Bands.Geometry,+z_fill: F32,+z_workers: U32,
  +z_contract: Global.Contract(~F32,~Z,~z_value,z_context,Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
  +w_bands: Model.Model<F32>,+w_geometry: Bands.Geometry,+w_fill: F32,+w_workers: U32,
  +w_contract: Global.Contract(~F32,~Weights,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers}),
  readiness: Shape.Both<
    InputReady(Call.stride(geometry),Call.padding(geometry),
      Layer.parameters(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
        (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
          Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
        Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
        Layer.settings(Call.stride(geometry),Call.padding(geometry),activation))),
    ParametersReady(Layer.parameters(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
      (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
        Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
      Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)))>) ->
  Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.propagated(Call.stride(geometry),Call.padding(geometry),
      Layer.parameters(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
        (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
          Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
        Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
        Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)))):
  Shape.Both{input_ready,parameters_ready} = readiness
  propagated_holds(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.parameters(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
      (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
        Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
      Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)),
    parameters_call(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
      dy_bands,dy_geometry,dy_fill,dy_workers,dy_contract,x_bands,x_geometry,x_fill,x_workers,x_contract,z_bands,z_geometry,z_fill,z_workers,z_contract,
      w_bands,w_geometry,w_fill,w_workers,w_contract,parameters_ready),
    input_ready)

# Backward of dy, the saved input and convolution value, and the weights, of
# the readers' contracts in any layout and the shapes of a call geometry. A
# call that returns Ready gradients returns at every index the input, weight
# and bias gradient specifications of x, dz = dy * activation'(z) and the
# weights, and the weights back.
def backward(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  +dy_bands: Model.Model<F32>,+dy_geometry: Bands.Geometry,+dy_fill: F32,+dy_workers: U32,
  +dy_contract: Global.Contract(~F32,~DY,~dy_value,dy_context,Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers}),
  +x_bands: Model.Model<F32>,+x_geometry: Bands.Geometry,+x_fill: F32,+x_workers: U32,
  +x_contract: Global.Contract(~F32,~Input,~input_value,input_context,
    Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers}),
  +z_bands: Model.Model<F32>,+z_geometry: Bands.Geometry,+z_fill: F32,+z_workers: U32,
  +z_contract: Global.Contract(~F32,~Z,~z_value,z_context,Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
  +w_bands: Model.Model<F32>,+w_geometry: Bands.Geometry,+w_fill: F32,+w_workers: U32,
  +w_contract: Global.Contract(~F32,~Weights,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers}),
  ready: BackwardReady(Layer.backward(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
    (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
      Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
    Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
    Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)))) ->
  Backward(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.backward(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
      (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
        Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
      Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Layer.settings(Call.stride(geometry),Call.padding(geometry),activation))):
  backward_split(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    dy_bands,dy_geometry,dy_fill,dy_workers,dy_contract,x_bands,x_geometry,x_fill,x_workers,x_contract,z_bands,z_geometry,z_fill,z_workers,z_contract,
    w_bands,w_geometry,w_fill,w_workers,w_contract,
    propagated_ready(Call.stride(geometry),Call.padding(geometry),
      Layer.parameters(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
        (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
          Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
        Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
        Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)),
      ready))

def without_input_call(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  result: Layer.Parameters(),
  call: ParametersCall(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,result)) ->
  BackwardParameters(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.without_input(result)):
  (weight_gradient,(bias_gradient,(dz,(weights,shape)))) = result
  Shape.Both{weight_contract,Shape.Both{bias_contract,Shape.Both{dz_shaped,Shape.Both{weights_shaped,same}}}} = call
  Shape.Both{weight_contract,Shape.Both{bias_contract,weights_shaped}}

# backward_parameters, for a layer whose input needs no gradient: Ready
# gradients hold the weight and bias gradient specifications of x and dz.
def backward_parameters(~Input: Data,~input_value: Input -> U32 -> F32,~DY: Data,~dy_value: DY -> U32 -> F32,~Z: Data,~z_value: Z -> U32 -> F32,~Weights: Data,~weight_value: Weights -> U32 -> F32,
  +input_context: Input,+dy_context: DY,+z_context: Z,+weight_context: Weights,+activation: Activations.Activation,+mode: Call.Mode,+geometry: Kernel.Geometry,
  +dy_bands: Model.Model<F32>,+dy_geometry: Bands.Geometry,+dy_fill: F32,+dy_workers: U32,
  +dy_contract: Global.Contract(~F32,~DY,~dy_value,dy_context,Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers}),
  +x_bands: Model.Model<F32>,+x_geometry: Bands.Geometry,+x_fill: F32,+x_workers: U32,
  +x_contract: Global.Contract(~F32,~Input,~input_value,input_context,
    Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers}),
  +z_bands: Model.Model<F32>,+z_geometry: Bands.Geometry,+z_fill: F32,+z_workers: U32,
  +z_contract: Global.Contract(~F32,~Z,~z_value,z_context,Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
  +w_bands: Model.Model<F32>,+w_geometry: Bands.Geometry,+w_fill: F32,+w_workers: U32,
  +w_contract: Global.Contract(~F32,~Weights,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers}),
  ready: GradientsReady(Layer.backward_parameters(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
    (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
      Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
    Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
    Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)))) ->
  BackwardParameters(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.backward_parameters(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
      (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
        Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
      Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Layer.settings(Call.stride(geometry),Call.padding(geometry),activation))):
  without_input_call(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
    Layer.parameters(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
      (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
        Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
      Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
      Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)),
    parameters_call(~Input,~input_value,~DY,~dy_value,~Z,~z_value,~Weights,~weight_value,input_context,dy_context,z_context,weight_context,activation,mode,geometry,
      dy_bands,dy_geometry,dy_fill,dy_workers,dy_contract,x_bands,x_geometry,x_fill,x_workers,x_contract,z_bands,z_geometry,z_fill,z_workers,z_contract,
      w_bands,w_geometry,w_fill,w_workers,w_contract,
      without_input_ready(Layer.parameters(Tensors.Tensor{Model.pack(~F32,dy_bands),dy_geometry,Call.input_shape(mode,geometry),dy_fill,Tensors.Usable{},dy_workers},
          (Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,Gradient.input_shape(Call.arguments(mode,geometry)),x_fill,Tensors.Usable{},x_workers},
            Tensors.Tensor{Model.pack(~F32,z_bands),z_geometry,Call.input_shape(mode,geometry),z_fill,Tensors.Usable{},z_workers}),
          Tensors.Tensor{Model.pack(~F32,w_bands),w_geometry,Call.weight_shape(geometry),w_fill,Tensors.Usable{},w_workers},
          Layer.settings(Call.stride(geometry),Call.padding(geometry),activation)),
        ready)))
