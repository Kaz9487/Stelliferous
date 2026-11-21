# forward_network, evaluate_network, backward_network and sgd_network_with of
# Stelliferous training hold the values of their layers in order: each layer
# reads the value the one before it gives, and a residual block adds the
# values of its branch and its shortcut.
import Base
import ../../lib/math_activation.bend as Activations
import ../../lib/math_sum.bend as Sum
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_f32.bend as F
import ../../lib/tensor_tiled.bend as Tiled
import ../../lib/tensor_training.bend as Train
import ../../lib/tensor_view.bend as Views
import ../../lib/tensor_grid.bend as Grid
import ../../lib/kernel_gemm.bend as KernelGemm
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/tensor_parameter_gradient.bend as Parameter
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/input_gradient_phase_domain.bend as Domain
import ../specification.bend as ConvSpec
import ./convolution_call.bend as ConvCall
import ./parameter_gradient_public.bend as ParameterPublic
import ../../lib/proofs/tensor_tiled_public.bend as Public
import ../../lib/proofs/tensor_pieces_map.bend as PiecesMap
import ../../lib/proofs/tensor_pieces_combine_leaf.bend as Leaf
import ../../lib/proofs/tensor_partials_values.bend as PartialsValues
import ../../lib/proofs/tensor_function_values.bend as FunctionValues
import ../../lib/proofs/tensor_optimizer_values.bend as OptimizerValues
import ../../lib/proofs/tensor_batch_norm_public.bend as NormPublic
import ../../lib/proofs/tensor_layer_norm_values.bend as LineValues
import ../../lib/proofs/tensor_training_layer_values.bend as Layers
import ./image_problems.bend as ImageProblems
import ./layer_public.bend as LayerPublic
import ./input_gradient_call.bend as Call
import ./input_gradient_public.bend as InputPublic
import ./parameter_gradient_terms.bend as Terms
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/nat_order.bend as Order
import ../../lib/proofs/boolean_logic.bend as Boolean
import ../../lib/proofs/word_encoding.bend as Words
import ../../lib/proofs/tensor_copy_contract.bend as Copy
import ../../lib/proofs/tensor_binary_values.bend as Binary
import ../../lib/proofs/tensor_view_gather.bend as ViewGather
import ../../lib/tensor_optimizer.bend as Optimizer
import ./training_convolution_public.bend as TrainingConvolution

# ---- Values ----

# xhat of a batch normalization and the inverse deviation it saves.
def NormXhat(~V: Data) -> Data:
  NormPublic.Channels<V,NormPublic.Quotient(),NormPublic.Deviation(~NormPublic.Quotient()),V,V>

def NormInverse() -> Data:
  NormPublic.Deviation(~NormPublic.Quotient())

def inverse_value() -> NormInverse() -> U32 -> F32:
  NormPublic.deviation_value(~NormPublic.Quotient(),~NormPublic.quotient_value())

def xhat_value(~V: Data,~v_value: V -> U32 -> F32) -> NormXhat(~V) -> U32 -> F32:
  NormPublic.normalized_value(~V,~v_value,~NormPublic.Quotient(),~NormPublic.quotient_value(),~NormInverse(),~inverse_value(),~V,~V)

# The update of a step and its velocity.
def StepApplied(~V: Data) -> Data:
  OptimizerValues.Applied<OptimizerValues.Direction<V,V>,V>

def step_applied_value(~V: Data,~v_value: V -> U32 -> F32) -> StepApplied(~V) -> U32 -> F32:
  OptimizerValues.applied_value(~OptimizerValues.Direction<V,V>,~OptimizerValues.direction_value(~V,~v_value,~V,~v_value),~V,~v_value)

# The value of a network: the given input and parameters are read through
# x_value, and each result of a layer reads the values below it with its
# own reader.
type Value<-X: Data> is Data:
  Given{x: X}
  Activated{context: FunctionValues.ActivationRead<Value<X>>}
  ActivationGradient{context: Leaf.Context<Value<X>,FunctionValues.ActivationRead<Value<X>>>}
  Pooled{context: Layers.PoolMean()}
  PoolGradient{context: Layers.PoolSpreadValue(~Value<X>)}
  NormOutput{context: NormXhat(~Value<X>)}
  NormMoved{context: NormPublic.Moved(~Value<X>)}
  NormEvaluated{context: NormPublic.Channels<Value<X>,Value<X>,NormPublic.Deviation(~Value<X>),Value<X>,Value<X>>}
  NormGradient{context: NormPublic.BackChannels<Value<X>,NormXhat(~Value<X>),PartialsValues.Box<Sum.Parts>,PartialsValues.Box<Sum.Parts>,NormPublic.Scale(~Value<X>,~NormInverse())>}
  Summed{context: PartialsValues.Box<Sum.Parts>}
  LineOutput{context: LineValues.Output(~Value<X>,~Value<X>,~Value<X>)}
  LineGradient{context: LineValues.InputGradient(~Value<X>,~Value<X>,~Value<X>)}
  Convolved{context: ImageProblems.Context<Value<X>>}
  ConvolutionInput{context: InputPublic.InputRead<TrainingConvolution.Delta(~Value<X>,~Value<X>),Value<X>>}
  ConvolutionWeight{context: Terms.WeightRead<Value<X>,TrainingConvolution.Delta(~Value<X>,~Value<X>)>}
  ConvolutionBias{context: Terms.BiasRead<TrainingConvolution.Delta(~Value<X>,~Value<X>)>}
  Stepped{context: Leaf.Context<Value<X>,PiecesMap.Context<F32,StepApplied(~Value<X>)>>}
  Velocity{context: StepApplied(~Value<X>)}
  Added{context: Leaf.Context<Value<X>,Value<X>>}

def interpret(~X: Data,~x_value: X -> U32 -> F32,value: Value<X>,index: U32) -> F32:
  match value:
    case Given{x}: x_value(x,index)
    case Activated{context}: FunctionValues.activated_value(~Value<X>,~interpret(~X,~x_value),context,index)
    case ActivationGradient{context}:
      Leaf.value(~F32,~Value<X>,~FunctionValues.ActivationRead<Value<X>>,~F32.mul,~interpret(~X,~x_value),~FunctionValues.derivative_at(~Value<X>,~interpret(~X,~x_value)),
        context,index)
    case Pooled{context}: Layers.pool_mean_value()(context,index)
    case PoolGradient{context}: Layers.pool_spread_value(~Value<X>,~interpret(~X,~x_value))(context,index)
    case NormOutput{context}:
      NormPublic.output_value(~Value<X>,~interpret(~X,~x_value),~NormPublic.Quotient(),~NormPublic.quotient_value(),~NormInverse(),~inverse_value(),
        ~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),context,index)
    case NormMoved{context}: NormPublic.moved_value(~Value<X>,~interpret(~X,~x_value))(context,index)
    case NormEvaluated{context}:
      NormPublic.output_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~NormPublic.Deviation(~Value<X>),
        ~NormPublic.deviation_value(~Value<X>,~interpret(~X,~x_value)),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),context,index)
    case NormGradient{context}:
      NormPublic.back_value(~Value<X>,~interpret(~X,~x_value),~NormXhat(~Value<X>),~xhat_value(~Value<X>,~interpret(~X,~x_value)),
        ~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),
        ~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),
        ~NormPublic.Scale(~Value<X>,~NormInverse()),~NormPublic.scale_value(~Value<X>,~interpret(~X,~x_value),~NormInverse(),~inverse_value()),context,index)
    case Summed{context}: PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded,context,index)
    case LineOutput{context}:
      LineValues.output_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value))(context,index)
    case LineGradient{context}:
      LineValues.input_gradient_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value))(context,index)
    case Convolved{context}: ImageProblems.value(~Activations.identity_value,~Value<X>,~interpret(~X,~x_value),context,index)
    case ConvolutionInput{context}:
      InputPublic.input_at(~TrainingConvolution.Delta(~Value<X>,~Value<X>),~TrainingConvolution.delta_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value)),
        ~Value<X>,~interpret(~X,~x_value),context,index)
    case ConvolutionWeight{context}:
      Terms.weight_at(~Value<X>,~interpret(~X,~x_value),~TrainingConvolution.Delta(~Value<X>,~Value<X>),
        ~TrainingConvolution.delta_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value)),context,index)
    case ConvolutionBias{context}:
      Terms.bias_at(~TrainingConvolution.Delta(~Value<X>,~Value<X>),~TrainingConvolution.delta_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value)),
        context,index)
    case Stepped{context}:
      Leaf.value(~F32,~Value<X>,~PiecesMap.Context<F32,StepApplied(~Value<X>)>,~F32.sub,~interpret(~X,~x_value),
        ~PiecesMap.value(~F32,~F32,~StepApplied(~Value<X>),~F.scalar_value(~F32.mul),~step_applied_value(~Value<X>,~interpret(~X,~x_value))),context,index)
    case Velocity{context}: step_applied_value(~Value<X>,~interpret(~X,~x_value))(context,index)
    case Added{context}: Leaf.value(~F32,~Value<X>,~Value<X>,~F32.add,~interpret(~X,~x_value),~interpret(~X,~x_value),context,index)

# ---- Results as values ----

# A value made of the context of a result.
def Wrapped(~X: Data,~C: Data,~make: C -> Value<X>,context: C,value: Value<X>) -> Data:
  {value == make(context) : Value<X>}

def wrapped_agree(~X: Data,~x_value: X -> U32 -> F32,~C: Data,~reader: C -> U32 -> F32,~make: C -> Value<X>,
  ~same: @+context: C -> @+index: U32 -> {reader(context,index) == interpret(~X,~x_value,make(context),index) : F32},
  +context: C,+value: Value<X>,+related: Wrapped(~X,~C,~make,context,value),+index: U32) -> {reader(context,index) == interpret(~X,~x_value,value,index) : F32}:
  %Equal.sym(Value<X>,value,make(context),related) : {reader(context,index) == interpret(~X,~x_value,_,index) : F32}
  same(context,index)

# A result whose reader agrees with the value made of its context holds that
# value.
def wrapped(~X: Data,~x_value: X -> U32 -> F32,~C: Data,~reader: C -> U32 -> F32,~make: C -> Value<X>,
  ~same: @+context: C -> @+index: U32 -> {reader(context,index) == interpret(~X,~x_value,make(context),index) : F32},
  +context: C,+shape: List<&2,U32>,tiled: Tiled.Tiled,+contract: Public.Contract(~C,~reader,context,shape,tiled)) ->
  Public.Contract(~Value<X>,~interpret(~X,~x_value),make(context),shape,tiled):
  Public.reread(~C,~Value<X>,~reader,~interpret(~X,~x_value),~Wrapped(~X,~C,~make),~wrapped_agree(~X,~x_value,~C,~reader,~make,~same),
    context,make(context),{==},shape,tiled,contract)

# The activation of a value holds the activated value.
def activated(~X: Data,~x_value: X -> U32 -> F32,+context: FunctionValues.ActivationRead<Value<X>>,+shape: List<&2,U32>,tiled: Tiled.Tiled,
  +contract: Public.Contract(~FunctionValues.ActivationRead<Value<X>>,~FunctionValues.activated_value(~Value<X>,~interpret(~X,~x_value)),context,shape,tiled)) ->
  Public.Contract(~Value<X>,~interpret(~X,~x_value),Activated{context},shape,tiled):
  wrapped(~X,~x_value,~FunctionValues.ActivationRead<Value<X>>,~FunctionValues.activated_value(~Value<X>,~interpret(~X,~x_value)),~(read => Activated{read}),
    ~(read => index => {==}),context,shape,tiled,contract)

# ---- Descriptions ----

# The gradients a layer holds of its two parameters: cleared, or computed in
# tensors that hold the two values.
type Grading<-X: Data> is Data:
  Ungraded{}
  Graded{first: Value<X>,second: Value<X>}

def graded_first(~X: Data,~x_value: X -> U32 -> F32,+grading: Grading<X>,+shape: List<&2,U32>) -> Train.Gradient -> Data:
  match grading:
    case Ungraded{}: Layers.Cleared
    case Graded{+first,second}: gradient => Layers.Computed(~Value<X>,~interpret(~X,~x_value),first,shape,gradient)

def graded_second(~X: Data,~x_value: X -> U32 -> F32,+grading: Grading<X>,+shape: List<&2,U32>) -> Train.Gradient -> Data:
  match grading:
    case Ungraded{}: Layers.Cleared
    case Graded{first,+second}: gradient => Layers.Computed(~Value<X>,~interpret(~X,~x_value),second,shape,gradient)

# A layer of the network: a convolution of [C,H,W] or [N,C,H,W], a batch
# normalization of [N,C,H,W], a layer normalization along the last axis with
# the views that stretch its statistics and parameters, an activation, or
# the average pool of [N,C,H,W] with the view its gradient spreads through.
type LayerValues<-X: Data> is Data:
  ConvolutionOf{mode: Call.Mode,channels: U32,outputs: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,
    weights: Layers.Parameter<Value<X>>,bias: Layers.Parameter<Value<X>>,grading: Grading<X>}
  BatchNormOf{images: U32,channels: U32,height: U32,width: U32,epsilon: F32,momentum: F32,gamma: Layers.Parameter<Value<X>>,beta: Layers.Parameter<Value<X>>,
    mean: Value<X>,variance: Value<X>,grading: Grading<X>}
  LayerNormOf{features: U32,epsilon: F32,laxes: List<&2,Views.Axis>,loffset: U32,raxes: List<&2,Views.Axis>,roffset: U32,gaxes: List<&2,Views.Axis>,goffset: U32,
    gamma: Layers.Parameter<Value<X>>,beta: Layers.Parameter<Value<X>>,grading: Grading<X>}
  ActivationOf{activation: Activations.Activation}
  PoolOf{images: U32,channels: U32,height: U32,width: U32,axes: List<&2,Views.Axis>,offset: U32}

type NetworkValues<-X: Data> is Data:
  Last{}
  Next{layer: LayerValues<X>,rest: NetworkValues<X>}
  Block{branch: NetworkValues<X>,shortcut: NetworkValues<X>,rest: NetworkValues<X>}

# The convolution of an input with the parameters of a layer.
def convolution_context(~X: Data,~x_value: X -> U32 -> F32,+input: Value<X>,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,
  +padding: U32,+weights: Layers.Parameter<Value<X>>,+bias: Layers.Parameter<Value<X>>) -> ImageProblems.Context<Value<X>>:
  LayerPublic.convolution_context(~Value<X>,~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),input,Layers.parameter_value(~Value<X>,weights),
    Layers.parameter_value(~Value<X>,bias),channels,outputs,height,width,kernel,stride,padding)

# The output and xhat of a batch normalization in training mode.
def norm_context(~X: Data,~x_value: X -> U32 -> F32,+input: Value<X>,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,
  +gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>) -> NormXhat(~Value<X>):
  NormPublic.Channels{input,NormPublic.channel_means(~Value<X>,~interpret(~X,~x_value),input,images,channels,height,width),
    NormPublic.deviation_context(~NormPublic.Quotient(),NormPublic.channel_variances(~Value<X>,~interpret(~X,~x_value),input,images,channels,height,width),epsilon),
    Layers.parameter_value(~Value<X>,gamma),Layers.parameter_value(~Value<X>,beta),Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})}

# The value a layer gives of an input of the extents in training mode.
def layer_value(~X: Data,~x_value: X -> U32 -> F32,+layer: LayerValues<X>,+input: Value<X>,+shape: List<&2,U32>) -> Value<X>:
  match layer:
    case ConvolutionOf{mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,grading}:
      Activated{FunctionValues.ActivationRead{Activations.identity(),
        Convolved{convolution_context(~X,~x_value,input,channels,outputs,height,width,kernel,stride,padding,weights,bias)}}}
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,momentum,+gamma,+beta,mean,variance,grading}:
      NormOutput{norm_context(~X,~x_value,input,images,channels,height,width,epsilon,gamma,beta)}
    case LayerNormOf{features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,grading}:
      LineOutput{LineValues.output_context(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),input,
        Layers.parameter_value(~Value<X>,gamma),Layers.parameter_value(~Value<X>,beta),shape,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset)}
    case ActivationOf{+activation}: Activated{FunctionValues.ActivationRead{activation,input}}
    case PoolOf{+images,+channels,+height,+width,axes,offset}: Pooled{Layers.pool_context(~Value<X>,~interpret(~X,~x_value),input,images,channels,height,width)}

# The value of a layer in inference mode: a batch normalization reads its
# running statistics, and a convolution gives the convolution value.
def evaluated_value(~X: Data,~x_value: X -> U32 -> F32,+layer: LayerValues<X>,+input: Value<X>,+shape: List<&2,U32>) -> Value<X>:
  match layer:
    case ConvolutionOf{mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,grading}:
      Convolved{convolution_context(~X,~x_value,input,channels,outputs,height,width,kernel,stride,padding,weights,bias)}
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,momentum,+gamma,+beta,+mean,+variance,grading}:
      NormEvaluated{NormPublic.Channels{input,mean,NormPublic.deviation_context(~Value<X>,variance,epsilon),Layers.parameter_value(~Value<X>,gamma),
        Layers.parameter_value(~Value<X>,beta),Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})}}
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,+grading}:
      layer_value(~X,~x_value,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,shape)
    case ActivationOf{+activation}: layer_value(~X,~x_value,ActivationOf{activation},input,shape)
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}: layer_value(~X,~x_value,PoolOf{images,channels,height,width,axes,offset},input,shape)

# The extents a layer gives of an input of the extents.
def layer_shape(~X: Data,+layer: LayerValues<X>,+shape: List<&2,U32>) -> List<&2,U32>:
  match layer:
    case ConvolutionOf{+mode,channels,+outputs,+height,+width,+kernel,+stride,+padding,weights,bias,grading}:
      LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding)
    case BatchNormOf{+images,+channels,+height,+width,epsilon,momentum,gamma,beta,mean,variance,grading}: [images,channels,height,width]
    case LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading}: shape
    case ActivationOf{activation}: shape
    case PoolOf{+images,+channels,height,width,axes,offset}: [images,channels,1,1]

# A layer as forward of an input leaves it: a batch normalization moves its
# running statistics toward the statistics of the batch.
def layer_forwarded(~X: Data,~x_value: X -> U32 -> F32,+layer: LayerValues<X>,+input: Value<X>) -> LayerValues<X>:
  match layer:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,+grading}:
      ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading}
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,+momentum,+gamma,+beta,+mean,+variance,+grading}:
      BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,
        NormMoved{NormPublic.moved_context(~Value<X>,mean,NormPublic.channel_means(~Value<X>,~interpret(~X,~x_value),input,images,channels,height,width),momentum)},
        NormMoved{NormPublic.moved_context(~Value<X>,variance,NormPublic.unbiased_variances(~Value<X>,~interpret(~X,~x_value),input,images,channels,height,width),momentum)},
        grading}
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,+grading}:
      LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading}
    case ActivationOf{+activation}: ActivationOf{activation}
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}: PoolOf{images,channels,height,width,axes,offset}

# ---- Traces ----

# A value with its extents.
type Valued<-X: Data> is Data:
  Valued{value: Value<X>,shape: List<&2,U32>}

def valued_value(~X: Data,+valued: Valued<X>) -> Value<X>:
  match valued:
    case Valued{+value,shape}: value

def valued_shape(~X: Data,+valued: Valued<X>) -> List<&2,U32>:
  match valued:
    case Valued{value,+shape}: shape

# What a layer gives of an input in training mode and in inference mode.
def layer_valued(~X: Data,~x_value: X -> U32 -> F32,+layer: LayerValues<X>,+input: Valued<X>) -> Valued<X>:
  Valued{layer_value(~X,~x_value,layer,valued_value(~X,input),valued_shape(~X,input)),layer_shape(~X,layer,valued_shape(~X,input))}

def layer_evaluated(~X: Data,~x_value: X -> U32 -> F32,+layer: LayerValues<X>,+input: Valued<X>) -> Valued<X>:
  Valued{evaluated_value(~X,~x_value,layer,valued_value(~X,input),valued_shape(~X,input)),layer_shape(~X,layer,valued_shape(~X,input))}

# The sum of what a branch and a shortcut give, in the extents of the branch.
def sum_valued(~X: Data,+branch: Valued<X>,+shortcut: Valued<X>) -> Valued<X>:
  Valued{Added{Leaf.Context{valued_value(~X,branch),valued_value(~X,shortcut)}},valued_shape(~X,branch)}

# What forward_network records, the last layer first: each layer as forward
# left it reads what the rest gives of the input of the trace, and the branch
# and the shortcut of a block read the input of the block.
type Trace<-X: Data> is Data:
  Unwritten{}
  Written{layer: LayerValues<X>,rest: Trace<X>}
  Branched{branch: Trace<X>,shortcut: Trace<X>,rest: Trace<X>}

# What a trace gives of its input.
def trace_valued(~X: Data,~x_value: X -> U32 -> F32,+trace: Trace<X>,+input: Valued<X>) -> Valued<X>:
  match trace:
    case Unwritten{}: input
    case Written{+layer,+rest}: layer_valued(~X,~x_value,layer,trace_valued(~X,~x_value,rest,input))
    case Branched{+branch,+shortcut,+rest}:
      sum_valued(~X,trace_valued(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input)),trace_valued(~X,~x_value,shortcut,trace_valued(~X,~x_value,rest,input)))

# What forward of a network records after a prefix of the input.
def network_trace(~X: Data,~x_value: X -> U32 -> F32,+network: NetworkValues<X>,+prefix: Trace<X>,+input: Valued<X>) -> Trace<X>:
  match network:
    case Last{}: prefix
    case Next{+layer,+rest}:
      network_trace(~X,~x_value,rest,Written{layer_forwarded(~X,~x_value,layer,valued_value(~X,trace_valued(~X,~x_value,prefix,input))),prefix},input)
    case Block{+branch,+shortcut,+rest}:
      network_trace(~X,~x_value,rest,Branched{network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),
        network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),prefix},input)

# What a network gives of an input in inference mode: a block adds what its
# branch and its shortcut give of its input.
def evaluated_valued(~X: Data,~x_value: X -> U32 -> F32,+network: NetworkValues<X>,+input: Valued<X>) -> Valued<X>:
  match network:
    case Last{}: input
    case Next{+layer,+rest}: evaluated_valued(~X,~x_value,rest,layer_evaluated(~X,~x_value,layer,input))
    case Block{+branch,+shortcut,+rest}:
      evaluated_valued(~X,~x_value,rest,sum_valued(~X,evaluated_valued(~X,~x_value,branch,input),evaluated_valued(~X,~x_value,shortcut,input)))

# ---- What a description asks of a network ----

# The dimensions of a valid convolution problem: every extent positive and
# the kernel inside the padded image.
def dimensions_valid(+channels: Nat,+outputs: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat) -> Bool:
  Nat.is_lt(0n,channels) && Nat.is_lt(0n,outputs) && Nat.is_lt(0n,height) && Nat.is_lt(0n,width) && Nat.is_lt(0n,kernel) && Nat.is_lt(0n,stride) &&
    Nat.is_le(kernel,Nat.add(height,Nat.mul(2n,padding))) && Nat.is_le(kernel,Nat.add(width,Nat.mul(2n,padding)))

# A convolution of [C,H,W] or [N,C,H,W] that runtime accepts forward and
# backward: the dimensions of a valid problem, machine support, a nonempty
# batch, the checks of the gradients and nonempty parameters.
type ConvolutionFits<-shape: List<&2,U32>,-mode: Call.Mode,-channels: U32,-outputs: U32,-height: U32,-width: U32,-kernel: U32,-stride: U32,-padding: U32> is Data:
  ConvolutionFits{same: {shape == TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width) : List<&2,U32>},
    dimensions: {dimensions_valid(U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),
      U32.to_nat(stride),U32.to_nat(padding)) == True{} : Bool},
    machine: {TensorConvolution.machine_support(TensorConvolution.Geometry{ConvCall.images(Call.batched(mode),Call.images(mode)),Call.batched(mode),
      KernelShape.Dimensions{U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)}},
      Default.limits()) == True{} : Bool},
    images_positive: {Nat.is_lt(0n,U32.to_nat(Call.images(mode))) == True{} : Bool},
    checks: {Call.checks(mode,TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding)) == True{} : Bool},
    gradients_valid: {Parameter.valid(ParameterPublic.input_image(mode,TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding)),
      ParameterPublic.gradient_image(mode,TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding)),
      Domain.kernel_word(TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding)),
      Call.stride(TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding)),
      Call.padding(TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding)),Gradient.group_length()) == True{} : Bool},
    weights_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([outputs,channels,kernel,kernel])) == True{} : Bool},
    bias_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([outputs])) == True{} : Bool}}

# A batch normalization of [N,C,H,W] with room for its sums.
type NormFits<-shape: List<&2,U32>,-images: U32,-channels: U32,-height: U32,-width: U32> is Data:
  NormFits{same: {shape == [images,channels,height,width] : List<&2,U32>},room: Layers.NormRoom<images,channels,height,width>,
    channels_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([channels])) == True{} : Bool}}

# A layer normalization of the features along the last axis of the extents.
type LineFits<-limits: Machine.Limits,-shape: List<&2,U32>,-features: U32,-laxes: List<&2,Views.Axis>,-loffset: U32,-raxes: List<&2,Views.Axis>,-roffset: U32,
  -gaxes: List<&2,Views.Axis>,-goffset: U32> is Data:
  LineFits{facts: Layers.LineFacts<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,
    features_nonempty: {Nat.is_lt(0n,ShapeLaws.elements([features])) == True{} : Bool}}

# Nonempty extents that fit the machine: what an activation and the sum of a
# block need.
type Spread<-limits: Machine.Limits,-shape: List<&2,U32>> is Data:
  Spread{nonempty: {Nat.is_lt(0n,ShapeLaws.elements(shape)) == True{} : Bool},
    rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

# The average pool of [N,C,H,W] and the view its gradient spreads through.
type PoolFits<-limits: Machine.Limits,-shape: List<&2,U32>,-images: U32,-channels: U32,-height: U32,-width: U32,-axes: List<&2,Views.Axis>,-offset: U32> is Data:
  PoolFits{same: {shape == [images,channels,height,width] : List<&2,U32>},facts: Layers.PoolFacts<limits,images,channels,height,width>,
    spread: Layers.PoolSpread<images,channels,height,width,axes,offset>}

# What a layer of the description asks of the runtime layer and of the
# extents of its input.
def LayerDescribed(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+values: LayerValues<X>,+shape: List<&2,U32>,layer: Train.Layer) -> Data:
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,+grading}:
      KernelGemm.Both<ConvolutionFits<shape,mode,channels,outputs,height,width,kernel,stride,padding>,
        TrainingConvolution.ConvolutionLayer(~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[outputs,channels,kernel,kernel]),
          ~graded_second(~X,~x_value,grading,[outputs]),channels,outputs,kernel,stride,padding,weights,bias,layer)>
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,+momentum,+gamma,+beta,+mean,+variance,+grading}:
      KernelGemm.Both<NormFits<shape,images,channels,height,width>,
        Layers.NormLayer(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[channels]),
          ~graded_second(~X,~x_value,grading,[channels]),channels,epsilon,momentum,gamma,beta,mean,variance,layer)>
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,+grading}:
      KernelGemm.Both<LineFits<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,
        Layers.LineLayer(~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[features]),
          ~graded_second(~X,~x_value,grading,[features]),features,epsilon,gamma,beta,layer)>
    case ActivationOf{+activation}:
      KernelGemm.Both<KernelGemm.Both<{FunctionValues.built_in(activation) == True{} : Bool},Spread<limits,shape>>,Layers.ActivationLayer(activation,layer)>
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}:
      KernelGemm.Both<PoolFits<limits,shape,images,channels,height,width,axes,offset>,Layers.PoolLayer(layer)>

# The extents a network gives of an input of the extents.
def network_shape(~X: Data,+network: NetworkValues<X>,+shape: List<&2,U32>) -> List<&2,U32>:
  match network:
    case Last{}: shape
    case Next{+layer,+rest}: network_shape(~X,rest,layer_shape(~X,layer,shape))
    case Block{+branch,shortcut,+rest}: network_shape(~X,rest,network_shape(~X,branch,shape))

# The branch and the shortcut of a block give the same extents, nonempty and
# fitting the machine, of an input of extents that fit it.
def BlockFits(~X: Data,+limits: Machine.Limits,+branch: NetworkValues<X>,+shortcut: NetworkValues<X>,+shape: List<&2,U32>) -> Data:
  KernelGemm.Both<{network_shape(~X,branch,shape) == network_shape(~X,shortcut,shape) : List<&2,U32>},
    KernelGemm.Both<Spread<limits,network_shape(~X,branch,shape)>,Spread<limits,shape>>>

# A network the description describes on an input of the extents.
def Described(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+description: NetworkValues<X>,+shape: List<&2,U32>,network: Train.Network) -> Data:
  match description:
    case Last{}:
      match network:
        case Train.End{}: {True{} == True{} : Bool}
        case Train.Then{layer,others}: {False{} == True{} : Bool}
        case Train.Residual{branch_network,shortcut_network,others}: {False{} == True{} : Bool}
    case Next{+values,+rest}:
      match network:
        case Train.End{}: {False{} == True{} : Bool}
        case Train.Then{layer,others}:
          KernelGemm.Both<LayerDescribed(~X,~x_value,limits,values,shape,layer),Described(~X,~x_value,limits,rest,layer_shape(~X,values,shape),others)>
        case Train.Residual{branch_network,shortcut_network,others}: {False{} == True{} : Bool}
    case Block{+branch,+shortcut,+rest}:
      match network:
        case Train.End{}: {False{} == True{} : Bool}
        case Train.Then{layer,others}: {False{} == True{} : Bool}
        case Train.Residual{branch_network,shortcut_network,others}:
          KernelGemm.Both<BlockFits(~X,limits,branch,shortcut,shape),KernelGemm.Both<Described(~X,~x_value,limits,branch,shape,branch_network),
            KernelGemm.Both<Described(~X,~x_value,limits,shortcut,shape,shortcut_network),Described(~X,~x_value,limits,rest,network_shape(~X,branch,shape),others)>>>

# ---- Extents ----

# Forward leaves the extents a layer gives.
def forwarded_shape(~X: Data,~x_value: X -> U32 -> F32,+values: LayerValues<X>,+input: Value<X>,+shape: List<&2,U32>) ->
  {layer_shape(~X,layer_forwarded(~X,~x_value,values,input),shape) == layer_shape(~X,values,shape) : List<&2,U32>}:
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,+grading}: {==}
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,+momentum,+gamma,+beta,+mean,+variance,+grading}: {==}
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,+grading}: {==}
    case ActivationOf{+activation}: {==}
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}: {==}

# What the trace of a network after a prefix gives has the extents the
# network gives of what the prefix gives.
def traced_shape(~X: Data,~x_value: X -> U32 -> F32,+description: NetworkValues<X>,+prefix: Trace<X>,+input: Valued<X>) ->
  {valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,description,prefix,input),input)) ==
    network_shape(~X,description,valued_shape(~X,trace_valued(~X,~x_value,prefix,input))) : List<&2,U32>}:
  match description:
    case Last{}: {==}
    case Next{+values,+rest}:
      %forwarded_shape(~X,~x_value,values,valued_value(~X,trace_valued(~X,~x_value,prefix,input)),valued_shape(~X,trace_valued(~X,~x_value,prefix,input))) :
        {valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,rest,Written{layer_forwarded(~X,~x_value,values,valued_value(~X,trace_valued(~X,~x_value,prefix,input))),prefix},input),input)) ==
          network_shape(~X,rest,_) : List<&2,U32>}
      traced_shape(~X,~x_value,rest,Written{layer_forwarded(~X,~x_value,values,valued_value(~X,trace_valued(~X,~x_value,prefix,input))),prefix},input)
    case Block{+branch,+shortcut,+rest}:
      %traced_shape(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)) :
        {valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,rest,Branched{network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),
          network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),prefix},input),input)) == network_shape(~X,rest,_) : List<&2,U32>}
      traced_shape(~X,~x_value,rest,Branched{network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),
        network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),prefix},input)

# ---- Convolution problems ----

# The values of a parameter listed in the elements of its extents.
def listed_elements(~P: Data,~p_value: P -> U32 -> F32,+shape: List<&2,U32>,+value: P) ->
  {ConvSpec.length(Contract.listed(~F32,~P,~p_value,ShapeLaws.elements(shape),0n,value)) == ShapeLaws.elements(shape) : Nat}:
  %ImageProblems.listed(~P,~p_value,ShapeLaws.elements(shape),0n,value) : {ConvSpec.length(_) == ShapeLaws.elements(shape) : Nat}
  ImageProblems.image_length(~P,~p_value,ShapeLaws.elements(shape),0n,value)

def input_listed(~X: Data,~x_value: X -> U32 -> F32,+x: X,+channels: U32,+height: U32,+width: U32) ->
  {Nat.is_eq(ConvSpec.length(ImageProblems.image_values(~X,~x_value,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))),
    Nat.mul(0n,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),x)),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))) == True{} : Bool}:
  %Equal.sym(Nat,ConvSpec.length(ImageProblems.image_values(~X,~x_value,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))),
      Nat.mul(0n,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),x)),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))),
    ImageProblems.image_length(~X,~x_value,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))),
      Nat.mul(0n,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),x)) :
    {Nat.is_eq(_,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))) == True{} : Bool}
  Order.equal_reflexive(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))))

def weights_listed(~P: Data,~p_value: P -> U32 -> F32,+weights: P,+outputs: U32,+channels: U32,+kernel: U32) ->
  {Nat.is_eq(ConvSpec.length(Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,weights)),
    Nat.mul(U32.to_nat(outputs),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel))))) == True{} : Bool}:
  %Equal.sym(Nat,ConvSpec.length(Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,weights)),ShapeLaws.elements([outputs,channels,kernel,kernel]),
    listed_elements(~P,~p_value,[outputs,channels,kernel,kernel],weights)) :
    {Nat.is_eq(_,Nat.mul(U32.to_nat(outputs),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel))))) == True{} : Bool}
  %ShapeLaws.elements_four_exact(outputs,channels,kernel,kernel,U32.to_nat(outputs),U32.to_nat(channels),U32.to_nat(kernel),U32.to_nat(kernel),{==},{==},{==},{==}) :
    {Nat.is_eq(ShapeLaws.elements([outputs,channels,kernel,kernel]),_) == True{} : Bool}
  Order.equal_reflexive(ShapeLaws.elements([outputs,channels,kernel,kernel]))

def bias_listed(~P: Data,~p_value: P -> U32 -> F32,+bias: P,+outputs: U32) ->
  {Nat.is_eq(ConvSpec.length(Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs]),0n,bias)),U32.to_nat(outputs)) == True{} : Bool}:
  %Equal.sym(Nat,ConvSpec.length(Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs]),0n,bias)),ShapeLaws.elements([outputs]),
    listed_elements(~P,~p_value,[outputs],bias)) : {Nat.is_eq(_,U32.to_nat(outputs)) == True{} : Bool}
  %Order.times_one(U32.to_nat(outputs)) : {Nat.is_eq(ShapeLaws.elements([outputs]),_) == True{} : Bool}
  Order.equal_reflexive(ShapeLaws.elements([outputs]))

# A convolution of valid dimensions over listed parameters is a valid problem
# of its first image: the problem depends on the input only through lengths.
def problem_valid(~X: Data,~x_value: X -> U32 -> F32,~P: Data,~p_value: P -> U32 -> F32,+x: X,+channels: U32,+outputs: U32,+height: U32,+width: U32,+kernel: U32,
  +stride: U32,+padding: U32,+weights: P,+bias: P,
  +dimensions: {dimensions_valid(U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)) == True{} : Bool}) ->
  {ConvSpec.valid(ImageProblems.problem(~X,~x_value,x,U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding),
    Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,weights),Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs]),0n,bias),0n)) == True{} : Bool}:
  Boolean.and_true(
    dimensions_valid(U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)) &&
      Nat.is_eq(ConvSpec.length(ImageProblems.image_values(~X,~x_value,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))),
        Nat.mul(0n,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),x)),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))) &&
      Nat.is_eq(ConvSpec.length(Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,weights)),
        Nat.mul(U32.to_nat(outputs),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel))))),
    Nat.is_eq(ConvSpec.length(Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs]),0n,bias)),U32.to_nat(outputs)),
    Boolean.and_true(
      dimensions_valid(U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)) &&
        Nat.is_eq(ConvSpec.length(ImageProblems.image_values(~X,~x_value,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))),
          Nat.mul(0n,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),x)),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),
      Nat.is_eq(ConvSpec.length(Contract.listed(~F32,~P,~p_value,ShapeLaws.elements([outputs,channels,kernel,kernel]),0n,weights)),
        Nat.mul(U32.to_nat(outputs),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(kernel),U32.to_nat(kernel))))),
      Boolean.and_true(
        dimensions_valid(U32.to_nat(channels),U32.to_nat(outputs),U32.to_nat(height),U32.to_nat(width),U32.to_nat(kernel),U32.to_nat(stride),U32.to_nat(padding)),
        Nat.is_eq(ConvSpec.length(ImageProblems.image_values(~X,~x_value,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))),
          Nat.mul(0n,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),x)),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),
        dimensions,input_listed(~X,~x_value,x,channels,height,width)),
      weights_listed(~P,~p_value,weights,outputs,channels,kernel)),
    bias_listed(~P,~p_value,bias,outputs))

# ---- Forward of a layer ----

# What forward of a layer saves for backward, as the theorem of its forward
# states it: saved tensors read their own contexts.
def LayerSaved(~X: Data,~x_value: X -> U32 -> F32,+values: LayerValues<X>,+input: Valued<X>,saved: Train.Saved) -> Data:
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,grading}:
      TrainingConvolution.ConvolutionSaved(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),mode,channels,outputs,height,width,kernel,stride,padding,
        valued_value(~X,input),weights,bias,saved)
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,momentum,+gamma,+beta,mean,variance,grading}:
      Layers.NormSaved(~NormXhat(~Value<X>),~xhat_value(~Value<X>,~interpret(~X,~x_value)),~NormInverse(),~inverse_value(),
        norm_context(~X,~x_value,valued_value(~X,input),images,channels,height,width,epsilon,gamma,beta),
        NormPublic.deviation_context(~NormPublic.Quotient(),NormPublic.channel_variances(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width),epsilon),
        images,channels,height,width,saved)
    case LayerNormOf{features,+epsilon,+laxes,+loffset,+raxes,+roffset,gaxes,goffset,gamma,beta,grading}:
      Layers.LineSaved(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),epsilon,laxes,loffset,raxes,roffset,saved)
    case ActivationOf{activation}: Layers.ActivationSaved(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),saved)
    case PoolOf{+images,+channels,+height,+width,axes,offset}: {saved == Train.Pooled{[images,channels,height,width]} : Train.Saved}

# The output of forward holds what the layer as forward leaves it gives of
# the input, what it saves holds what backward reads, and the layer comes
# back as the description of that layer states.
def ForwardOutput(~X: Data,~x_value: X -> U32 -> F32,+values: LayerValues<X>,+input: Valued<X>,output: Tiled.Tiled) -> Data:
  Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,layer_valued(~X,~x_value,layer_forwarded(~X,~x_value,values,valued_value(~X,input)),input)),
    valued_shape(~X,layer_valued(~X,~x_value,layer_forwarded(~X,~x_value,values,valued_value(~X,input)),input)),output)

def ForwardSaved(~X: Data,~x_value: X -> U32 -> F32,+values: LayerValues<X>,+input: Valued<X>,saved: Train.Saved) -> Data:
  LayerSaved(~X,~x_value,layer_forwarded(~X,~x_value,values,valued_value(~X,input)),input,saved)

def ForwardLayer(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+values: LayerValues<X>,+input: Valued<X>,layer: Train.Layer) -> Data:
  LayerDescribed(~X,~x_value,limits,layer_forwarded(~X,~x_value,values,valued_value(~X,input)),valued_shape(~X,input),layer)

def LayerForward(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+values: LayerValues<X>,+input: Valued<X>,result: Tiled.Tiled & (Train.Saved & Train.Layer)) -> Data:
  Layers.Forwarded(~(output => ForwardOutput(~X,~x_value,values,input,output)),~(saved => ForwardSaved(~X,~x_value,values,input,saved)),
    ~(layer => ForwardLayer(~X,~x_value,limits,values,input,layer)),result)

# A result of forward that holds one statement of each part holds another.
def forwarded_mapped(~First: Tiled.Tiled -> Data,~FirstSaved: Train.Saved -> Data,~FirstLayer: Train.Layer -> Data,
  ~Output: Tiled.Tiled -> Data,~Saved: Train.Saved -> Data,~Layer: Train.Layer -> Data,
  result: Tiled.Tiled & (Train.Saved & Train.Layer),+held: Layers.Forwarded(~First,~FirstSaved,~FirstLayer,result),
  output_map: @output: Tiled.Tiled -> @+output_held: First(output) -> Output(output),
  saved_map: @-saved: Train.Saved -> @+saved_held: FirstSaved(saved) -> Saved(saved),
  layer_map: @-layer: Train.Layer -> @+layer_held: FirstLayer(layer) -> Layer(layer)) -> Layers.Forwarded(~Output,~Saved,~Layer,result):
  (output,rest) = result
  (saved,layer) = rest
  KernelGemm.Both{+output_held,KernelGemm.Both{+saved_held,+layer_held}} = held
  KernelGemm.Both{output_map(output,output_held),KernelGemm.Both{saved_map(saved,saved_held),layer_map(layer,layer_held)}}

# The convolution value under an activation, as a value.
def convolved_make(~X: Data,+read: FunctionValues.ActivationRead<ImageProblems.Context<Value<X>>>) -> Value<X>:
  match read:
    case FunctionValues.ActivationRead{+activation,+logical}: Activated{FunctionValues.ActivationRead{activation,Convolved{logical}}}

def convolved_same(~X: Data,~x_value: X -> U32 -> F32,+read: FunctionValues.ActivationRead<ImageProblems.Context<Value<X>>>,+index: U32) ->
  {FunctionValues.activated_value(~ImageProblems.Context<Value<X>>,~ImageProblems.value(~Activations.identity_value,~Value<X>,~interpret(~X,~x_value)),read,index) ==
    interpret(~X,~x_value,convolved_make(~X,read),index) : F32}:
  match read:
    case FunctionValues.ActivationRead{+activation,+logical}: {==}

# forward of an activation.
def activation_forward_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+activation: Activations.Activation,+input: Valued<X>,+xm: Layers.TensorModel,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,ActivationOf{activation},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerForward(~X,~x_value,limits,ActivationOf{activation},input,Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm))):
  KernelGemm.Both{KernelGemm.Both{+built,+spread},+layered} = described
  forwarded_mapped(~(output => FunctionValues.Activated(~Value<X>,~interpret(~X,~x_value),activation,valued_value(~X,input),valued_shape(~X,input),output)),
    ~(saved => Layers.ActivationSaved(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),saved)),~(layer => Layers.ActivationLayer(activation,layer)),
    ~(output => ForwardOutput(~X,~x_value,ActivationOf{activation},input,output)),~(saved => ForwardSaved(~X,~x_value,ActivationOf{activation},input,saved)),
    ~(layer => ForwardLayer(~X,~x_value,limits,ActivationOf{activation},input,layer)),
    Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm)),
    Layers.activation_forward(~Value<X>,~interpret(~X,~x_value),activation,valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm),contract,Layers.layer_of(lm),
      layered,built),
    (output => output_held => activated(~X,~x_value,FunctionValues.ActivationRead{activation,valued_value(~X,input)},valued_shape(~X,input),output,output_held)),
    (saved => saved_held => saved_held),
    (layer => layer_held => KernelGemm.Both{KernelGemm.Both{built,spread},layer_held}))

# forward of the average pool.
def pool_forward_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +images: U32,+channels: U32,+height: U32,+width: U32,+axes: List<&2,Views.Axis>,+offset: U32,+input: Valued<X>,+xm: Layers.TensorModel,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,PoolOf{images,channels,height,width,axes,offset},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerForward(~X,~x_value,limits,PoolOf{images,channels,height,width,axes,offset},input,Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  PoolFits{+same,+facts,+spread} = fits
  forwarded_mapped(~(output => Layers.PoolOutput(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width,output)),
    ~(saved => {saved == Train.Pooled{[images,channels,height,width]} : Train.Saved}),~(layer => Layers.PoolLayer(layer)),
    ~(output => ForwardOutput(~X,~x_value,PoolOf{images,channels,height,width,axes,offset},input,output)),
    ~(saved => ForwardSaved(~X,~x_value,PoolOf{images,channels,height,width,axes,offset},input,saved)),
    ~(layer => ForwardLayer(~X,~x_value,limits,PoolOf{images,channels,height,width,axes,offset},input,layer)),
    Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm)),
    Layers.pool_forward(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width,limits,default_capacity,facts,Layers.tensor_of(xm),
      (%same : Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),_,Layers.tensor_of(xm)) contract),Layers.layer_of(lm),layered),
    (output => output_held => wrapped(~X,~x_value,~Layers.PoolMean(),~Layers.pool_mean_value(),~(context => Pooled{context}),~(context => index => {==}),
      Layers.pool_context(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width),[images,channels,1,1],output,output_held)),
    (saved => saved_held => saved_held),
    (layer => layer_held => KernelGemm.Both{fits,layer_held}))

# A result of forward of a batch normalization, as the network states it.
def norm_converted(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,
  +gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,+mean: Value<X>,+variance: Value<X>,+grading: Grading<X>,+input: Valued<X>,
  +fits: NormFits<valued_shape(~X,input),images,channels,height,width>,result: Tiled.Tiled & (Train.Saved & Train.Layer),
  +held: Layers.NormForward(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[channels]),
    ~graded_second(~X,~x_value,grading,[channels]),images,channels,height,width,epsilon,momentum,gamma,beta,valued_value(~X,input),mean,variance,result)) ->
  LayerForward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,result):
  (output,rest) = result
  (saved,layer) = rest
  match saved:
    case Train.Normalized{pair}:
      (normalized,inverse) = pair
      match layer:
        case Train.BatchNorm{gammas,betas,means,variances,+layer_epsilon,+layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
          KernelGemm.Both{+epsilon_exact,KernelGemm.Both{+momentum_exact,KernelGemm.Both{
            KernelGemm.Both{+output_contract,KernelGemm.Both{+xhat_contract,KernelGemm.Both{+inverse_contract,KernelGemm.Both{+gamma_contract,KernelGemm.Both{+beta_contract,
              KernelGemm.Both{+means_contract,+variances_contract}}}}}},
            KernelGemm.Both{+gamma_graded,KernelGemm.Both{+beta_graded,KernelGemm.Both{+gamma_kept,+beta_kept}}}}}} = held
          KernelGemm.Both{wrapped(~X,~x_value,~NormXhat(~Value<X>),~NormPublic.output_value(~Value<X>,~interpret(~X,~x_value),~NormPublic.Quotient(),~NormPublic.quotient_value(),
              ~NormInverse(),~inverse_value(),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value)),~(context => NormOutput{context}),~(context => index => {==}),
              norm_context(~X,~x_value,valued_value(~X,input),images,channels,height,width,epsilon,gamma,beta),[images,channels,height,width],output,output_contract),
            KernelGemm.Both{KernelGemm.Both{xhat_contract,inverse_contract},
              KernelGemm.Both{fits,KernelGemm.Both{epsilon_exact,KernelGemm.Both{momentum_exact,
                KernelGemm.Both{KernelGemm.Both{gamma_contract,KernelGemm.Both{gamma_graded,gamma_kept}},
                  KernelGemm.Both{KernelGemm.Both{beta_contract,KernelGemm.Both{beta_graded,beta_kept}},
                    KernelGemm.Both{wrapped(~X,~x_value,~NormPublic.Moved(~Value<X>),~NormPublic.moved_value(~Value<X>,~interpret(~X,~x_value)),~(context => NormMoved{context}),
                        ~(context => index => {==}),
                        NormPublic.moved_context(~Value<X>,mean,NormPublic.channel_means(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width),momentum),
                        [channels],means,means_contract),
                      wrapped(~X,~x_value,~NormPublic.Moved(~Value<X>),~NormPublic.moved_value(~Value<X>,~interpret(~X,~x_value)),~(context => NormMoved{context}),
                        ~(context => index => {==}),
                        NormPublic.moved_context(~Value<X>,variance,NormPublic.unbiased_variances(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width),momentum),
                        [channels],variances,variances_contract)}}}}}}}}
        case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
          Empty.absurd(LayerForward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,(Train.Normalized{(normalized,inverse)},Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}))),Words.false_true(held))
        case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
          Empty.absurd(LayerForward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,(Train.Normalized{(normalized,inverse)},Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}))),Words.false_true(held))
        case Train.Activation{other_activation}:
          Empty.absurd(LayerForward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,(Train.Normalized{(normalized,inverse)},Train.Activation{other_activation}))),Words.false_true(held))
        case Train.AveragePool{}:
          Empty.absurd(LayerForward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,(Train.Normalized{(normalized,inverse)},Train.AveragePool{}))),Words.false_true(held))
    case Train.Convolved{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,(Train.Convolved{other},layer))),Words.false_true(held))
    case Train.LineNormalized{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,(Train.LineNormalized{other},layer))),Words.false_true(held))
    case Train.Activated{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,(Train.Activated{other},layer))),Words.false_true(held))
    case Train.Pooled{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,(Train.Pooled{other},layer))),Words.false_true(held))

# forward of a batch normalization.
def norm_forward_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,
  +mean: Value<X>,+variance: Value<X>,+grading: Grading<X>,+input: Valued<X>,+xm: Layers.TensorModel,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerForward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,
    Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  NormFits{+same,+room,+channels_nonempty} = fits
  norm_converted(~X,~x_value,limits,images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading,input,fits,
    Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm)),
    Layers.norm_forward(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[channels]),
      ~graded_second(~X,~x_value,grading,[channels]),images,channels,height,width,epsilon,momentum,gamma,beta,valued_value(~X,input),mean,variance,
      limits,default_capacity,room,Layers.tensor_of(xm),
      (%same : Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),_,Layers.tensor_of(xm)) contract),Layers.layer_of(lm),layered))

# A result of forward of a layer normalization, as the network states it.
def line_converted(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,
  +raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,
  +grading: Grading<X>,+input: Valued<X>,+fits: LineFits<limits,valued_shape(~X,input),features,laxes,loffset,raxes,roffset,gaxes,goffset>,
  result: Tiled.Tiled & (Train.Saved & Train.Layer),
  +held: Layers.LineForward(~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[features]),
    ~graded_second(~X,~x_value,grading,[features]),valued_shape(~X,input),features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,
    valued_value(~X,input),result)) ->
  LayerForward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,result):
  (output,rest) = result
  (saved,layer) = rest
  match saved:
    case Train.LineNormalized{kept}:
      match layer:
        case Train.LayerNorm{gammas,betas,+layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
          KernelGemm.Both{+epsilon_exact,KernelGemm.Both{KernelGemm.Both{+output_contract,KernelGemm.Both{+saved_held,KernelGemm.Both{+gamma_contract,+beta_contract}}},
            KernelGemm.Both{+gamma_graded,KernelGemm.Both{+beta_graded,KernelGemm.Both{+gamma_kept,+beta_kept}}}}} = held
          KernelGemm.Both{wrapped(~X,~x_value,~LineValues.Output(~Value<X>,~Value<X>,~Value<X>),
              ~LineValues.output_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value)),~(context => LineOutput{context}),
              ~(context => index => {==}),
              LineValues.output_context(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),
                Layers.parameter_value(~Value<X>,gamma),Layers.parameter_value(~Value<X>,beta),valued_shape(~X,input),epsilon,laxes,loffset,raxes,roffset,gaxes,goffset),
              valued_shape(~X,input),output,output_contract),
            KernelGemm.Both{saved_held,
              KernelGemm.Both{fits,KernelGemm.Both{epsilon_exact,
                KernelGemm.Both{KernelGemm.Both{gamma_contract,KernelGemm.Both{gamma_graded,gamma_kept}},
                  KernelGemm.Both{beta_contract,KernelGemm.Both{beta_graded,beta_kept}}}}}}}
        case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
          Empty.absurd(LayerForward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,(Train.LineNormalized{kept},Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}))),Words.false_true(held))
        case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
          Empty.absurd(LayerForward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,(Train.LineNormalized{kept},Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}))),Words.false_true(held))
        case Train.Activation{other_activation}:
          Empty.absurd(LayerForward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,(Train.LineNormalized{kept},Train.Activation{other_activation}))),Words.false_true(held))
        case Train.AveragePool{}:
          Empty.absurd(LayerForward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,(Train.LineNormalized{kept},Train.AveragePool{}))),Words.false_true(held))
    case Train.Convolved{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,(Train.Convolved{other},layer))),Words.false_true(held))
    case Train.Normalized{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,(Train.Normalized{other},layer))),Words.false_true(held))
    case Train.Activated{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,(Train.Activated{other},layer))),Words.false_true(held))
    case Train.Pooled{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,(Train.Pooled{other},layer))),Words.false_true(held))

# forward of a layer normalization.
def line_forward_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,
  +gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,+grading: Grading<X>,+input: Valued<X>,+xm: Layers.TensorModel,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerForward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,
    Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  LineFits{+facts,+features_nonempty} = fits
  line_converted(~X,~x_value,limits,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading,input,fits,
    Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm)),
    Layers.line_forward(~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[features]),
      ~graded_second(~X,~x_value,grading,[features]),valued_shape(~X,input),features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,
      valued_value(~X,input),limits,default_capacity,facts,Layers.tensor_of(xm),contract,Layers.layer_of(lm),layered))

# A result of forward of a convolution, as the network states it.
def convolution_converted(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,
  +kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<Value<X>>,+bias: Layers.Parameter<Value<X>>,+grading: Grading<X>,+input: Valued<X>,
  +fits: ConvolutionFits<valued_shape(~X,input),mode,channels,outputs,height,width,kernel,stride,padding>,result: Tiled.Tiled & (Train.Saved & Train.Layer),
  +held: TrainingConvolution.ConvolutionForward(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),
    ~graded_first(~X,~x_value,grading,[outputs,channels,kernel,kernel]),~graded_second(~X,~x_value,grading,[outputs]),
    mode,channels,outputs,height,width,kernel,stride,padding,valued_value(~X,input),weights,bias,result)) ->
  LayerForward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,result):
  (output,rest) = result
  (saved,layer) = rest
  match saved:
    case Train.Convolved{kept}:
      (input_kept,value_kept) = kept
      match layer:
        case Train.Convolution{weight_tensor,bias_tensor,+layer_stride,+layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
          KernelGemm.Both{+stride_exact,KernelGemm.Both{+padding_exact,KernelGemm.Both{
            KernelGemm.Both{+output_held,KernelGemm.Both{KernelGemm.Both{+input_contract,+value_contract},KernelGemm.Both{+weight_contract,+bias_contract}}},
            KernelGemm.Both{+weight_graded,KernelGemm.Both{+bias_graded,KernelGemm.Both{+weight_kept,+bias_kept}}}}}} = held
          KernelGemm.Both{wrapped(~X,~x_value,~FunctionValues.ActivationRead<ImageProblems.Context<Value<X>>>,
              ~FunctionValues.activated_value(~ImageProblems.Context<Value<X>>,~ImageProblems.value(~Activations.identity_value,~Value<X>,~interpret(~X,~x_value))),
              ~(read => convolved_make(~X,read)),~(read => index => convolved_same(~X,~x_value,read,index)),
              FunctionValues.ActivationRead{Activations.identity(),convolution_context(~X,~x_value,valued_value(~X,input),channels,outputs,height,width,kernel,stride,padding,weights,bias)},
              LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding),output,output_held),
            KernelGemm.Both{KernelGemm.Both{input_contract,value_contract},
              KernelGemm.Both{fits,KernelGemm.Both{stride_exact,KernelGemm.Both{padding_exact,
                KernelGemm.Both{KernelGemm.Both{weight_contract,KernelGemm.Both{weight_graded,weight_kept}},
                  KernelGemm.Both{bias_contract,KernelGemm.Both{bias_graded,bias_kept}}}}}}}}
        case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
          Empty.absurd(LayerForward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,(Train.Convolved{(input_kept,value_kept)},Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}))),Words.false_true(held))
        case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
          Empty.absurd(LayerForward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,(Train.Convolved{(input_kept,value_kept)},Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}))),Words.false_true(held))
        case Train.Activation{other_activation}:
          Empty.absurd(LayerForward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,(Train.Convolved{(input_kept,value_kept)},Train.Activation{other_activation}))),Words.false_true(held))
        case Train.AveragePool{}:
          Empty.absurd(LayerForward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,(Train.Convolved{(input_kept,value_kept)},Train.AveragePool{}))),Words.false_true(held))
    case Train.Normalized{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,(Train.Normalized{other},layer))),Words.false_true(held))
    case Train.LineNormalized{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,(Train.LineNormalized{other},layer))),Words.false_true(held))
    case Train.Activated{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,(Train.Activated{other},layer))),Words.false_true(held))
    case Train.Pooled{other}:
      Empty.absurd(LayerForward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,(Train.Pooled{other},layer))),Words.false_true(held))

# forward of a convolution.
def convolution_forward_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,
  +kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<Value<X>>,+bias: Layers.Parameter<Value<X>>,+grading: Grading<X>,+input: Valued<X>,
  +xm: Layers.TensorModel,+contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),
  +lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},valued_shape(~X,input),
    Layers.layer_of(lm))) ->
  LayerForward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,
    Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  ConvolutionFits{+same,+dimensions,+machine,+images_positive,+checks,+gradients_valid,+weights_nonempty,+bias_nonempty} = fits
  convolution_converted(~X,~x_value,limits,mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading,input,fits,
    Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm)),
    TrainingConvolution.convolution_forward(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),
      ~graded_first(~X,~x_value,grading,[outputs,channels,kernel,kernel]),~graded_second(~X,~x_value,grading,[outputs]),
      mode,channels,outputs,height,width,kernel,stride,padding,valued_value(~X,input),weights,bias,
      problem_valid(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),channels,outputs,height,width,kernel,stride,padding,
        Layers.parameter_value(~Value<X>,weights),Layers.parameter_value(~Value<X>,bias),dimensions),
      machine,images_positive,Layers.tensor_of(xm),
      (%same : Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),_,Layers.tensor_of(xm)) contract),Layers.layer_of(lm),layered))

# forward of a layer of the description on a tensor and a layer of models.
def layer_forward(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +values: LayerValues<X>,+input: Valued<X>,+xm: Layers.TensorModel,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,values,valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerForward(~X,~x_value,limits,values,input,Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm))):
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,+grading}:
      convolution_forward_step(~X,~x_value,limits,mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading,input,xm,contract,lm,described)
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,+momentum,+gamma,+beta,+mean,+variance,+grading}:
      norm_forward_step(~X,~x_value,limits,default_capacity,images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading,input,xm,contract,lm,described)
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,+grading}:
      line_forward_step(~X,~x_value,limits,default_capacity,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading,input,xm,contract,lm,described)
    case ActivationOf{+activation}: activation_forward_step(~X,~x_value,limits,activation,input,xm,contract,lm,described)
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}:
      pool_forward_step(~X,~x_value,limits,default_capacity,images,channels,height,width,axes,offset,input,xm,contract,lm,described)

# ---- Forward of a network ----

# The traces of the branch and the shortcut of a block give the same
# extents, nonempty and fitting the machine, of an input of extents that fit
# it.
def BlockTaped(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+branch: Trace<X>,+shortcut: Trace<X>,+input: Valued<X>) -> Data:
  KernelGemm.Both<{valued_shape(~X,trace_valued(~X,~x_value,branch,input)) == valued_shape(~X,trace_valued(~X,~x_value,shortcut,input)) : List<&2,U32>},
    KernelGemm.Both<Spread<limits,valued_shape(~X,trace_valued(~X,~x_value,branch,input))>,Spread<limits,valued_shape(~X,input)>>>

# The tape holds the trace of an input: each record what its layer saved of
# what the rest gives and the layer as its description states, and each
# block the tapes of its branch and its shortcut on the input of the block.
def Taped(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+trace: Trace<X>,+input: Valued<X>,tape: Train.Tape) -> Data:
  match trace:
    case Unwritten{}:
      match tape:
        case Train.Blank{}: {True{} == True{} : Bool}
        case Train.Recorded{saved,layer,others}: {False{} == True{} : Bool}
        case Train.Joined{branch_tape,shortcut_tape,others}: {False{} == True{} : Bool}
    case Written{+values,+rest}:
      match tape:
        case Train.Blank{}: {False{} == True{} : Bool}
        case Train.Recorded{saved,layer,others}:
          KernelGemm.Both<KernelGemm.Both<LayerSaved(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),saved),
            LayerDescribed(~X,~x_value,limits,values,valued_shape(~X,trace_valued(~X,~x_value,rest,input)),layer)>,Taped(~X,~x_value,limits,rest,input,others)>
        case Train.Joined{branch_tape,shortcut_tape,others}: {False{} == True{} : Bool}
    case Branched{+branch,+shortcut,+rest}:
      match tape:
        case Train.Blank{}: {False{} == True{} : Bool}
        case Train.Recorded{saved,layer,others}: {False{} == True{} : Bool}
        case Train.Joined{branch_tape,shortcut_tape,others}:
          KernelGemm.Both<Taped(~X,~x_value,limits,branch,trace_valued(~X,~x_value,rest,input),branch_tape),
            KernelGemm.Both<Taped(~X,~x_value,limits,shortcut,trace_valued(~X,~x_value,rest,input),shortcut_tape),
              KernelGemm.Both<BlockTaped(~X,~x_value,limits,branch,shortcut,trace_valued(~X,~x_value,rest,input)),Taped(~X,~x_value,limits,rest,input,others)>>>

# What forward of a network gives: the output holds what the trace gives of
# the input, and the tape holds the trace.
def NetworkRan(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+trace: Trace<X>,+input: Valued<X>,result: Tiled.Tiled & Train.Tape) -> Data:
  (output,tape) = result
  KernelGemm.Both<Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,trace_valued(~X,~x_value,trace,input)),valued_shape(~X,trace_valued(~X,~x_value,trace,input)),output),
    Taped(~X,~x_value,limits,trace,input,tape)>

# forward of a layer and then of the rest, on the input and the layer as
# models.
def layer_seen(~X: Data,~x_value: X -> U32 -> F32,~P: Tiled.Tiled & Train.Tape -> Data,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+values: LayerValues<X>,+prefix: Trace<X>,+input: Valued<X>,
  -layer: Train.Layer,+layer_observed: Layers.LayerObserved<layer>,
  +layer_described: LayerDescribed(~X,~x_value,limits,values,valued_shape(~X,trace_valued(~X,~x_value,prefix,input)),layer),-others: Train.Network,
  -x: Tiled.Tiled,+x_observed: Layers.TensorObserved<x>,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,trace_valued(~X,~x_value,prefix,input)),valued_shape(~X,trace_valued(~X,~x_value,prefix,input)),x),
  tape: Train.Tape,+taped: Taped(~X,~x_value,limits,prefix,input,tape),
  next: @output: Tiled.Tiled -> @after: Train.Tape ->
    @+output_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),
      valued_value(~X,trace_valued(~X,~x_value,Written{layer_forwarded(~X,~x_value,values,valued_value(~X,trace_valued(~X,~x_value,prefix,input))),prefix},input)),
      valued_shape(~X,trace_valued(~X,~x_value,Written{layer_forwarded(~X,~x_value,values,valued_value(~X,trace_valued(~X,~x_value,prefix,input))),prefix},input)),output) ->
    @+after_taped: Taped(~X,~x_value,limits,Written{layer_forwarded(~X,~x_value,values,valued_value(~X,trace_valued(~X,~x_value,prefix,input))),prefix},input,after) ->
    P(Train.network_forward(others,output,after))) ->
  P(Train.network_forward(Train.Then{layer,others},x,tape)):
  Layers.LayerObserved{+lm,+l_equation} = layer_observed
  Layers.TensorObserved{+xm,+x_equation} = x_observed
  %x_equation : P(Train.network_forward(Train.Then{layer,others},_,tape))
  %l_equation : P(Train.network_forward(Train.Then{_,others},Layers.tensor_of(xm),tape))
  Layers.recorded_holds(~(output => ForwardOutput(~X,~x_value,values,trace_valued(~X,~x_value,prefix,input),output)),
    ~(saved => ForwardSaved(~X,~x_value,values,trace_valued(~X,~x_value,prefix,input),saved)),
    ~(layer => ForwardLayer(~X,~x_value,limits,values,trace_valued(~X,~x_value,prefix,input),layer)),~P,
    ~(output => saved => layer => Train.network_forward(others,output,Train.Recorded{saved,layer,tape})),
    Train.forward(Layers.tensor_of(xm),Layers.layer_of(lm)),
    layer_forward(~X,~x_value,limits,default_capacity,values,trace_valued(~X,~x_value,prefix,input),xm,
      (%Equal.sym(Tiled.Tiled,Layers.tensor_of(xm),x,x_equation) :
        Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,trace_valued(~X,~x_value,prefix,input)),valued_shape(~X,trace_valued(~X,~x_value,prefix,input)),_) contract),
      lm,
      (%Equal.sym(Train.Layer,Layers.layer_of(lm),layer,l_equation) :
        LayerDescribed(~X,~x_value,limits,values,valued_shape(~X,trace_valued(~X,~x_value,prefix,input)),_) layer_described)),
    output => saved => layer => output_held => saved_held => layer_held =>
      next(output,Train.Recorded{saved,layer,tape},output_held,KernelGemm.Both{KernelGemm.Both{saved_held,layer_held},taped}))

# The sum of what the branch and the shortcut of a block give, then the rest,
# on the two outputs as models.
def joined_seen(~X: Data,~x_value: X -> U32 -> F32,~P: Tiled.Tiled & Train.Tape -> Data,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+branch: NetworkValues<X>,+shortcut: NetworkValues<X>,+prefix: Trace<X>,
  +input: Valued<X>,+fits: BlockFits(~X,limits,branch,shortcut,valued_shape(~X,trace_valued(~X,~x_value,prefix,input))),-others: Train.Network,
  -left: Tiled.Tiled,+left_observed: Layers.TensorObserved<left>,branch_tape: Train.Tape,
  +branch_ran: NetworkRan(~X,~x_value,limits,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input),
    (left,branch_tape)),
  -right: Tiled.Tiled,+right_observed: Layers.TensorObserved<right>,shortcut_tape: Train.Tape,
  +shortcut_ran: NetworkRan(~X,~x_value,limits,network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input),
    (right,shortcut_tape)),
  tape: Train.Tape,+taped: Taped(~X,~x_value,limits,prefix,input,tape),
  next: @output: Tiled.Tiled -> @after: Train.Tape ->
    @+output_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),
      valued_value(~X,trace_valued(~X,~x_value,Branched{network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),
        network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),prefix},input)),
      valued_shape(~X,trace_valued(~X,~x_value,Branched{network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),
        network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),prefix},input)),output) ->
    @+after_taped: Taped(~X,~x_value,limits,Branched{network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),
      network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),prefix},input,after) ->
    P(Train.network_forward(others,output,after))) ->
  P(Train.network_forward(others,F.add(left,right),Train.Joined{branch_tape,shortcut_tape,tape})):
  KernelGemm.Both{+same_shapes,KernelGemm.Both{+sum_spread,+input_spread}} = fits
  Spread{+nonempty,+rank_room} = sum_spread
  Layers.TensorObserved{+lm,+l_equation} = left_observed
  KernelGemm.Both{+left_contract,+branch_taped} = branch_ran
  Layers.TensorObserved{+rm,+r_equation} = right_observed
  KernelGemm.Both{+right_contract,+shortcut_taped} = shortcut_ran
  +branch_shape = traced_shape(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input))
  +shortcut_shape = traced_shape(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input))
  +shapes = Equal.trans(List<&2,U32>,
    valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
    network_shape(~X,branch,valued_shape(~X,trace_valued(~X,~x_value,prefix,input))),
    valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
    branch_shape,
    Equal.trans(List<&2,U32>,network_shape(~X,branch,valued_shape(~X,trace_valued(~X,~x_value,prefix,input))),
      network_shape(~X,shortcut,valued_shape(~X,trace_valued(~X,~x_value,prefix,input))),
      valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
      same_shapes,
      Equal.sym(List<&2,U32>,
        valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
        network_shape(~X,shortcut,valued_shape(~X,trace_valued(~X,~x_value,prefix,input))),shortcut_shape)))
  %l_equation : P(Train.network_forward(others,F.add(_,right),Train.Joined{branch_tape,shortcut_tape,tape}))
  %r_equation : P(Train.network_forward(others,F.add(Layers.tensor_of(lm),_),Train.Joined{branch_tape,shortcut_tape,tape}))
  next(F.add(Layers.tensor_of(lm),Layers.tensor_of(rm)),Train.Joined{branch_tape,shortcut_tape,tape},
    wrapped(~X,~x_value,~Leaf.Context<Value<X>,Value<X>>,~Leaf.value(~F32,~Value<X>,~Value<X>,~F32.add,~interpret(~X,~x_value),~interpret(~X,~x_value)),
      ~(context => Added{context}),~(context => index => {==}),
      Leaf.Context{valued_value(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
        valued_value(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input)))},
      valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
      F.add(Layers.tensor_of(lm),Layers.tensor_of(rm)),
      Binary.same(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~F32.add,
        valued_value(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
        valued_value(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
        "add",1.0,limits,default_capacity,
        valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
        (%Equal.sym(List<&2,U32>,
            valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
            network_shape(~X,branch,valued_shape(~X,trace_valued(~X,~x_value,prefix,input))),branch_shape) :
          {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} nonempty),
        (%Equal.sym(List<&2,U32>,
            valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
            network_shape(~X,branch,valued_shape(~X,trace_valued(~X,~x_value,prefix,input))),branch_shape) :
          {Nat.is_le(List.length(&2,U32,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room),
        Layers.tensor_of(lm),
        (%Equal.sym(Tiled.Tiled,Layers.tensor_of(lm),left,l_equation) :
          Public.Contract(~Value<X>,~interpret(~X,~x_value),
            valued_value(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
            valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),_)
          left_contract),
        Layers.tensor_of(rm),
        (%Equal.sym(Tiled.Tiled,Layers.tensor_of(rm),right,r_equation) :
          Public.Contract(~Value<X>,~interpret(~X,~x_value),
            valued_value(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
            valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),_)
          (%Equal.sym(List<&2,U32>,
              valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
              valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
              shapes) :
            Public.Contract(~Value<X>,~interpret(~X,~x_value),
              valued_value(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
              _,right)
            right_contract)))),
    KernelGemm.Both{branch_taped,KernelGemm.Both{shortcut_taped,KernelGemm.Both{KernelGemm.Both{shapes,KernelGemm.Both{
      (%Equal.sym(List<&2,U32>,
          valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),trace_valued(~X,~x_value,prefix,input))),
          network_shape(~X,branch,valued_shape(~X,trace_valued(~X,~x_value,prefix,input))),branch_shape) :
        Spread<limits,_> sum_spread),input_spread}},taped}}})

# forward of a network after a prefix whose trace the tape holds: whatever
# holds of every output and tape that hold the trace of the network after
# the prefix holds of what forward gives.
def forward_holds(~X: Data,~x_value: X -> U32 -> F32,~P: Tiled.Tiled & Train.Tape -> Data,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+description: NetworkValues<X>,+prefix: Trace<X>,+input: Valued<X>,
  network: Train.Network,+described: Described(~X,~x_value,limits,description,valued_shape(~X,trace_valued(~X,~x_value,prefix,input)),network),
  x: Tiled.Tiled,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,trace_valued(~X,~x_value,prefix,input)),valued_shape(~X,trace_valued(~X,~x_value,prefix,input)),x),
  tape: Train.Tape,+taped: Taped(~X,~x_value,limits,prefix,input,tape),
  done: @output: Tiled.Tiled -> @after: Train.Tape -> @+ran: NetworkRan(~X,~x_value,limits,network_trace(~X,~x_value,description,prefix,input),input,(output,after)) ->
    P((output,after))) ->
  P(Train.network_forward(network,x,tape)):
  match description:
    case Last{}:
      match network:
        case Train.End{}: done(x,tape,KernelGemm.Both{contract,taped})
        case Train.Then{other_layer,other_rest}:
          Empty.absurd(P(Train.network_forward(Train.Then{other_layer,other_rest},x,tape)),Words.false_true(described))
        case Train.Residual{other_branch,other_shortcut,other_rest}:
          Empty.absurd(P(Train.network_forward(Train.Residual{other_branch,other_shortcut,other_rest},x,tape)),Words.false_true(described))
    case Next{+values,+rest}:
      match network:
        case Train.Then{layer,others}:
          KernelGemm.Both{+layer_described,+rest_described} = described
          layer_seen(~X,~x_value,~P,limits,default_capacity,values,prefix,input,layer,Layers.observe_layer(layer),layer_described,others,x,Layers.observe_tensor(x),
            contract,tape,taped,
            output => after => output_contract => after_taped =>
              forward_holds(~X,~x_value,~P,limits,default_capacity,rest,
                Written{layer_forwarded(~X,~x_value,values,valued_value(~X,trace_valued(~X,~x_value,prefix,input))),prefix},input,others,
                (%Equal.sym(List<&2,U32>,
                    layer_shape(~X,layer_forwarded(~X,~x_value,values,valued_value(~X,trace_valued(~X,~x_value,prefix,input))),valued_shape(~X,trace_valued(~X,~x_value,prefix,input))),
                    layer_shape(~X,values,valued_shape(~X,trace_valued(~X,~x_value,prefix,input))),
                    forwarded_shape(~X,~x_value,values,valued_value(~X,trace_valued(~X,~x_value,prefix,input)),valued_shape(~X,trace_valued(~X,~x_value,prefix,input)))) :
                  Described(~X,~x_value,limits,rest,_,others) rest_described),
                output,output_contract,after,after_taped,done))
        case Train.End{}:
          Empty.absurd(P(Train.network_forward(Train.End{},x,tape)),Words.false_true(described))
        case Train.Residual{other_branch,other_shortcut,other_rest}:
          Empty.absurd(P(Train.network_forward(Train.Residual{other_branch,other_shortcut,other_rest},x,tape)),Words.false_true(described))
    case Block{+branch,+shortcut,+rest}:
      match network:
        case Train.Residual{branch_network,shortcut_network,others}:
          KernelGemm.Both{+fits,KernelGemm.Both{+branch_described,KernelGemm.Both{+shortcut_described,+rest_described}}} = described
          Copy.tiled_copied(~Value<X>,~interpret(~X,~x_value),
            ~(copies => P(Train.split(~(Tiled.Tiled & Train.Tape),copies,first => second =>
              Train.joined(Train.network_forward(branch_network,first,Train.Blank{}),Train.network_forward(shortcut_network,second,Train.Blank{}),
                left => right => branch_tape => shortcut_tape => Train.network_forward(others,F.add(left,right),Train.Joined{branch_tape,shortcut_tape,tape}))))),
            valued_value(~X,trace_valued(~X,~x_value,prefix,input)),valued_shape(~X,trace_valued(~X,~x_value,prefix,input)),x,contract,
            first => second => first_contract => second_contract =>
              forward_holds(~X,~x_value,
                ~(branch_result => P(Train.joined(branch_result,Train.network_forward(shortcut_network,second,Train.Blank{}),
                  left => right => branch_tape => shortcut_tape => Train.network_forward(others,F.add(left,right),Train.Joined{branch_tape,shortcut_tape,tape})))),
                limits,default_capacity,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input),branch_network,branch_described,first,first_contract,Train.Blank{},{==},
                left => branch_tape => branch_ran =>
                  forward_holds(~X,~x_value,
                    ~(shortcut_result => P(Train.joined((left,branch_tape),shortcut_result,
                      left => right => branch_tape => shortcut_tape => Train.network_forward(others,F.add(left,right),Train.Joined{branch_tape,shortcut_tape,tape})))),
                    limits,default_capacity,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input),shortcut_network,shortcut_described,second,second_contract,Train.Blank{},{==},
                    right => shortcut_tape => shortcut_ran =>
                      joined_seen(~X,~x_value,~P,limits,default_capacity,branch,shortcut,prefix,input,fits,others,left,Layers.observe_tensor(left),branch_tape,branch_ran,
                        right,Layers.observe_tensor(right),shortcut_tape,shortcut_ran,tape,taped,
                        output => after => output_contract => after_taped =>
                          forward_holds(~X,~x_value,~P,limits,default_capacity,rest,
                            Branched{network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),
                              network_trace(~X,~x_value,shortcut,Unwritten{},trace_valued(~X,~x_value,prefix,input)),prefix},input,others,
                            (%Equal.sym(List<&2,U32>,
                                valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input)),
                                  trace_valued(~X,~x_value,prefix,input))),
                                network_shape(~X,branch,valued_shape(~X,trace_valued(~X,~x_value,prefix,input))),
                                traced_shape(~X,~x_value,branch,Unwritten{},trace_valued(~X,~x_value,prefix,input))) :
                              Described(~X,~x_value,limits,rest,_,others) rest_described),
                            output,output_contract,after,after_taped,done)))))
        case Train.End{}:
          Empty.absurd(P(Train.network_forward(Train.End{},x,tape)),Words.false_true(described))
        case Train.Then{other_layer,other_rest}:
          Empty.absurd(P(Train.network_forward(Train.Then{other_layer,other_rest},x,tape)),Words.false_true(described))

# forward_network of the description on an input: the output holds what the
# trace of the network gives of the input, and the tape holds that trace.
def NetworkForward(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+description: NetworkValues<X>,+input: Valued<X>,result: Tiled.Tiled & Train.Tape) -> Data:
  NetworkRan(~X,~x_value,limits,network_trace(~X,~x_value,description,Unwritten{},input),input,result)

def forward(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +description: NetworkValues<X>,+input: Valued<X>,network: Train.Network,+described: Described(~X,~x_value,limits,description,valued_shape(~X,input),network),
  x: Tiled.Tiled,+contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),x)) ->
  NetworkForward(~X,~x_value,limits,description,input,Train.forward_network(x,network)):
  forward_holds(~X,~x_value,~(result => NetworkForward(~X,~x_value,limits,description,input,result)),limits,default_capacity,description,Unwritten{},input,network,described,
    x,contract,Train.Blank{},{==},output => after => ran => ran)

# ---- Evaluate of a layer ----

# What a network gives in inference mode has the extents the network gives.
def evaluated_shape(~X: Data,~x_value: X -> U32 -> F32,+description: NetworkValues<X>,+input: Valued<X>) ->
  {valued_shape(~X,evaluated_valued(~X,~x_value,description,input)) == network_shape(~X,description,valued_shape(~X,input)) : List<&2,U32>}:
  match description:
    case Last{}: {==}
    case Next{+values,+rest}: evaluated_shape(~X,~x_value,rest,layer_evaluated(~X,~x_value,values,input))
    case Block{+branch,+shortcut,+rest}:
      %evaluated_shape(~X,~x_value,branch,input) :
        {valued_shape(~X,evaluated_valued(~X,~x_value,rest,sum_valued(~X,evaluated_valued(~X,~x_value,branch,input),evaluated_valued(~X,~x_value,shortcut,input)))) ==
          network_shape(~X,rest,_) : List<&2,U32>}
      evaluated_shape(~X,~x_value,rest,sum_valued(~X,evaluated_valued(~X,~x_value,branch,input),evaluated_valued(~X,~x_value,shortcut,input)))

# The output of evaluate holds what the layer gives of the input in inference
# mode, and the layer comes back as the description states.
def EvaluateOutput(~X: Data,~x_value: X -> U32 -> F32,+values: LayerValues<X>,+input: Valued<X>,output: Tiled.Tiled) -> Data:
  Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,layer_evaluated(~X,~x_value,values,input)),valued_shape(~X,layer_evaluated(~X,~x_value,values,input)),output)

def EvaluateLayer(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+values: LayerValues<X>,+input: Valued<X>,layer: Train.Layer) -> Data:
  LayerDescribed(~X,~x_value,limits,values,valued_shape(~X,input),layer)

def LayerEvaluate(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+values: LayerValues<X>,+input: Valued<X>,result: Tiled.Tiled & Train.Layer) -> Data:
  Layers.Answered(~(output => EvaluateOutput(~X,~x_value,values,input,output)),~(layer => EvaluateLayer(~X,~x_value,limits,values,input,layer)),result)

# A result of evaluate that holds one statement of each part holds another.
def answered_mapped(~First: Tiled.Tiled -> Data,~FirstLayer: Train.Layer -> Data,~Output: Tiled.Tiled -> Data,~Layer: Train.Layer -> Data,
  result: Tiled.Tiled & Train.Layer,+held: Layers.Answered(~First,~FirstLayer,result),
  output_map: @output: Tiled.Tiled -> @+output_held: First(output) -> Output(output),
  layer_map: @-layer: Train.Layer -> @+layer_held: FirstLayer(layer) -> Layer(layer)) -> Layers.Answered(~Output,~Layer,result):
  (output,layer) = result
  KernelGemm.Both{+output_held,+layer_held} = held
  KernelGemm.Both{output_map(output,output_held),layer_map(layer,layer_held)}

# evaluate of an activation.
def activation_evaluate_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+activation: Activations.Activation,+input: Valued<X>,+xm: Layers.TensorModel,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,ActivationOf{activation},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerEvaluate(~X,~x_value,limits,ActivationOf{activation},input,Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm))):
  KernelGemm.Both{KernelGemm.Both{+built,+spread},+layered} = described
  answered_mapped(~(output => FunctionValues.Activated(~Value<X>,~interpret(~X,~x_value),activation,valued_value(~X,input),valued_shape(~X,input),output)),
    ~(layer => Layers.ActivationLayer(activation,layer)),~(output => EvaluateOutput(~X,~x_value,ActivationOf{activation},input,output)),
    ~(layer => EvaluateLayer(~X,~x_value,limits,ActivationOf{activation},input,layer)),
    Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm)),
    Layers.activation_evaluate(~Value<X>,~interpret(~X,~x_value),activation,valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm),contract,Layers.layer_of(lm),
      layered,built),
    (output => output_held => activated(~X,~x_value,FunctionValues.ActivationRead{activation,valued_value(~X,input)},valued_shape(~X,input),output,output_held)),
    (layer => layer_held => KernelGemm.Both{KernelGemm.Both{built,spread},layer_held}))

# evaluate of the average pool.
def pool_evaluate_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +images: U32,+channels: U32,+height: U32,+width: U32,+axes: List<&2,Views.Axis>,+offset: U32,+input: Valued<X>,+xm: Layers.TensorModel,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,PoolOf{images,channels,height,width,axes,offset},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerEvaluate(~X,~x_value,limits,PoolOf{images,channels,height,width,axes,offset},input,Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  PoolFits{+same,+facts,+spread} = fits
  answered_mapped(~(output => Layers.PoolOutput(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width,output)),
    ~(layer => Layers.PoolLayer(layer)),~(output => EvaluateOutput(~X,~x_value,PoolOf{images,channels,height,width,axes,offset},input,output)),
    ~(layer => EvaluateLayer(~X,~x_value,limits,PoolOf{images,channels,height,width,axes,offset},input,layer)),
    Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm)),
    Layers.pool_evaluate(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width,limits,default_capacity,facts,Layers.tensor_of(xm),
      (%same : Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),_,Layers.tensor_of(xm)) contract),Layers.layer_of(lm),layered),
    (output => output_held => wrapped(~X,~x_value,~Layers.PoolMean(),~Layers.pool_mean_value(),~(context => Pooled{context}),~(context => index => {==}),
      Layers.pool_context(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width),[images,channels,1,1],output,output_held)),
    (layer => layer_held => KernelGemm.Both{fits,layer_held}))

# A result of evaluate of a batch normalization, as the network states it.
def norm_evaluated_converted(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,
  +momentum: F32,+gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,+mean: Value<X>,+variance: Value<X>,+grading: Grading<X>,+input: Valued<X>,
  +fits: NormFits<valued_shape(~X,input),images,channels,height,width>,result: Tiled.Tiled & Train.Layer,
  +held: Layers.NormEvaluate(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[channels]),
    ~graded_second(~X,~x_value,grading,[channels]),images,channels,height,width,epsilon,momentum,gamma,beta,valued_value(~X,input),mean,variance,result)) ->
  LayerEvaluate(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,result):
  (output,layer) = result
  match layer:
    case Train.BatchNorm{gammas,betas,means,variances,+layer_epsilon,+layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{+momentum_exact,KernelGemm.Both{
        KernelGemm.Both{+output_contract,KernelGemm.Both{+gamma_contract,KernelGemm.Both{+beta_contract,KernelGemm.Both{+means_contract,+variances_contract}}}},
        KernelGemm.Both{+gamma_graded,KernelGemm.Both{+beta_graded,KernelGemm.Both{+gamma_kept,+beta_kept}}}}}} = held
      KernelGemm.Both{wrapped(~X,~x_value,~NormPublic.Channels<Value<X>,Value<X>,NormPublic.Deviation(~Value<X>),Value<X>,Value<X>>,
          ~NormPublic.output_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~NormPublic.Deviation(~Value<X>),
            ~NormPublic.deviation_value(~Value<X>,~interpret(~X,~x_value)),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value)),
          ~(context => NormEvaluated{context}),~(context => index => {==}),
          NormPublic.Channels{valued_value(~X,input),mean,NormPublic.deviation_context(~Value<X>,variance,epsilon),Layers.parameter_value(~Value<X>,gamma),
            Layers.parameter_value(~Value<X>,beta),Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})},
          [images,channels,height,width],output,output_contract),
        KernelGemm.Both{fits,KernelGemm.Both{epsilon_exact,KernelGemm.Both{momentum_exact,
          KernelGemm.Both{KernelGemm.Both{gamma_contract,KernelGemm.Both{gamma_graded,gamma_kept}},
            KernelGemm.Both{KernelGemm.Both{beta_contract,KernelGemm.Both{beta_graded,beta_kept}},KernelGemm.Both{means_contract,variances_contract}}}}}}}
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity})),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,(output,Train.AveragePool{})),Words.false_true(held))

# evaluate of a batch normalization.
def norm_evaluate_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,
  +mean: Value<X>,+variance: Value<X>,+grading: Grading<X>,+input: Valued<X>,+xm: Layers.TensorModel,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerEvaluate(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,
    Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  NormFits{+same,Layers.NormRoom{+nonempty,+cap,+cap_ok,+count_room},+channels_nonempty} = fits
  norm_evaluated_converted(~X,~x_value,limits,images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading,input,fits,
    Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm)),
    Layers.norm_evaluate(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[channels]),
      ~graded_second(~X,~x_value,grading,[channels]),images,channels,height,width,epsilon,momentum,gamma,beta,valued_value(~X,input),mean,variance,
      limits,default_capacity,nonempty,Layers.tensor_of(xm),
      (%same : Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),_,Layers.tensor_of(xm)) contract),Layers.layer_of(lm),layered))

# A result of evaluate of a layer normalization, as the network states it.
def line_evaluated_converted(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,
  +raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,
  +grading: Grading<X>,+input: Valued<X>,+fits: LineFits<limits,valued_shape(~X,input),features,laxes,loffset,raxes,roffset,gaxes,goffset>,
  result: Tiled.Tiled & Train.Layer,
  +held: Layers.LineEvaluate(~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[features]),
    ~graded_second(~X,~x_value,grading,[features]),valued_shape(~X,input),features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,
    valued_value(~X,input),result)) ->
  LayerEvaluate(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,result):
  (output,layer) = result
  match layer:
    case Train.LayerNorm{gammas,betas,+layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{KernelGemm.Both{+output_contract,KernelGemm.Both{+gamma_contract,+beta_contract}},
        KernelGemm.Both{+gamma_graded,KernelGemm.Both{+beta_graded,KernelGemm.Both{+gamma_kept,+beta_kept}}}}} = held
      KernelGemm.Both{wrapped(~X,~x_value,~LineValues.Output(~Value<X>,~Value<X>,~Value<X>),
          ~LineValues.output_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value)),~(context => LineOutput{context}),
          ~(context => index => {==}),
          LineValues.output_context(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),
            Layers.parameter_value(~Value<X>,gamma),Layers.parameter_value(~Value<X>,beta),valued_shape(~X,input),epsilon,laxes,loffset,raxes,roffset,gaxes,goffset),
          valued_shape(~X,input),output,output_contract),
        KernelGemm.Both{fits,KernelGemm.Both{epsilon_exact,
          KernelGemm.Both{KernelGemm.Both{gamma_contract,KernelGemm.Both{gamma_graded,gamma_kept}},
            KernelGemm.Both{beta_contract,KernelGemm.Both{beta_graded,beta_kept}}}}}}
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity})),Words.false_true(held))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,(output,Train.AveragePool{})),Words.false_true(held))

# evaluate of a layer normalization.
def line_evaluate_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,
  +gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,+grading: Grading<X>,+input: Valued<X>,+xm: Layers.TensorModel,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerEvaluate(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,
    Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  LineFits{+facts,+features_nonempty} = fits
  line_evaluated_converted(~X,~x_value,limits,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading,input,fits,
    Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm)),
    Layers.line_evaluate(~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[features]),
      ~graded_second(~X,~x_value,grading,[features]),valued_shape(~X,input),features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,
      valued_value(~X,input),limits,default_capacity,facts,Layers.tensor_of(xm),contract,Layers.layer_of(lm),layered))

# A result of evaluate of a convolution, as the network states it.
def convolution_evaluated_converted(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,
  +kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<Value<X>>,+bias: Layers.Parameter<Value<X>>,+grading: Grading<X>,+input: Valued<X>,
  +fits: ConvolutionFits<valued_shape(~X,input),mode,channels,outputs,height,width,kernel,stride,padding>,result: Tiled.Tiled & Train.Layer,
  +held: TrainingConvolution.ConvolutionEvaluate(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),
    ~graded_first(~X,~x_value,grading,[outputs,channels,kernel,kernel]),~graded_second(~X,~x_value,grading,[outputs]),
    mode,channels,outputs,height,width,kernel,stride,padding,valued_value(~X,input),weights,bias,result)) ->
  LayerEvaluate(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,result):
  (output,layer) = result
  match layer:
    case Train.Convolution{weight_tensor,bias_tensor,+layer_stride,+layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      KernelGemm.Both{+stride_exact,KernelGemm.Both{+padding_exact,KernelGemm.Both{+output_contract,KernelGemm.Both{+weight_contract,KernelGemm.Both{+bias_contract,
        KernelGemm.Both{+weight_graded,KernelGemm.Both{+bias_graded,KernelGemm.Both{+weight_kept,+bias_kept}}}}}}}} = held
      KernelGemm.Both{wrapped(~X,~x_value,~ImageProblems.Context<Value<X>>,~ImageProblems.value(~Activations.identity_value,~Value<X>,~interpret(~X,~x_value)),
          ~(context => Convolved{context}),~(context => index => {==}),
          convolution_context(~X,~x_value,valued_value(~X,input),channels,outputs,height,width,kernel,stride,padding,weights,bias),
          LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding),output,output_contract),
        KernelGemm.Both{fits,KernelGemm.Both{stride_exact,KernelGemm.Both{padding_exact,
          KernelGemm.Both{KernelGemm.Both{weight_contract,KernelGemm.Both{weight_graded,weight_kept}},
            KernelGemm.Both{bias_contract,KernelGemm.Both{bias_graded,bias_kept}}}}}}}
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LayerEvaluate(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,(output,Train.AveragePool{})),Words.false_true(held))

# evaluate of a convolution.
def convolution_evaluate_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,
  +kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<Value<X>>,+bias: Layers.Parameter<Value<X>>,+grading: Grading<X>,+input: Valued<X>,
  +xm: Layers.TensorModel,+contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),
  +lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},valued_shape(~X,input),
    Layers.layer_of(lm))) ->
  LayerEvaluate(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,
    Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  ConvolutionFits{+same,+dimensions,+machine,+images_positive,+checks,+gradients_valid,+weights_nonempty,+bias_nonempty} = fits
  convolution_evaluated_converted(~X,~x_value,limits,mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading,input,fits,
    Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm)),
    TrainingConvolution.convolution_evaluate(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),
      ~graded_first(~X,~x_value,grading,[outputs,channels,kernel,kernel]),~graded_second(~X,~x_value,grading,[outputs]),
      mode,channels,outputs,height,width,kernel,stride,padding,valued_value(~X,input),weights,bias,
      problem_valid(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),channels,outputs,height,width,kernel,stride,padding,
        Layers.parameter_value(~Value<X>,weights),Layers.parameter_value(~Value<X>,bias),dimensions),
      machine,images_positive,Layers.tensor_of(xm),
      (%same : Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),_,Layers.tensor_of(xm)) contract),Layers.layer_of(lm),layered))

# evaluate of a layer of the description on a tensor and a layer of models.
def layer_evaluate(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +values: LayerValues<X>,+input: Valued<X>,+xm: Layers.TensorModel,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),Layers.tensor_of(xm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,values,valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerEvaluate(~X,~x_value,limits,values,input,Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm))):
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,+grading}:
      convolution_evaluate_step(~X,~x_value,limits,mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading,input,xm,contract,lm,described)
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,+momentum,+gamma,+beta,+mean,+variance,+grading}:
      norm_evaluate_step(~X,~x_value,limits,default_capacity,images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading,input,xm,contract,lm,described)
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,+grading}:
      line_evaluate_step(~X,~x_value,limits,default_capacity,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading,input,xm,contract,lm,described)
    case ActivationOf{+activation}: activation_evaluate_step(~X,~x_value,limits,activation,input,xm,contract,lm,described)
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}:
      pool_evaluate_step(~X,~x_value,limits,default_capacity,images,channels,height,width,axes,offset,input,xm,contract,lm,described)

# ---- Evaluate of a network ----

# What evaluate of a network gives: the output holds what the network gives
# of the input in inference mode, and the network comes back as described.
def EvaluateRan(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+description: NetworkValues<X>,+input: Valued<X>,result: Tiled.Tiled & Train.Network) -> Data:
  (output,network) = result
  KernelGemm.Both<Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,evaluated_valued(~X,~x_value,description,input)),
      valued_shape(~X,evaluated_valued(~X,~x_value,description,input)),output),
    Described(~X,~x_value,limits,description,valued_shape(~X,input),network)>

# evaluate of a layer and then of the rest, on the input and the layer as
# models.
def layer_evaluated_seen(~X: Data,~x_value: X -> U32 -> F32,~P: Tiled.Tiled & Train.Network -> Data,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+values: LayerValues<X>,+input: Valued<X>,
  -layer: Train.Layer,+layer_observed: Layers.LayerObserved<layer>,+layer_described: LayerDescribed(~X,~x_value,limits,values,valued_shape(~X,input),layer),
  -others: Train.Network,-x: Tiled.Tiled,+x_observed: Layers.TensorObserved<x>,
  +contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),x),
  next: @output: Tiled.Tiled -> @after: Train.Layer -> @+output_contract: EvaluateOutput(~X,~x_value,values,input,output) ->
    @+after_described: EvaluateLayer(~X,~x_value,limits,values,input,after) -> P(Train.in_front(after,Train.network_evaluated(others,output)))) ->
  P(Train.network_evaluated(Train.Then{layer,others},x)):
  Layers.LayerObserved{+lm,+l_equation} = layer_observed
  Layers.TensorObserved{+xm,+x_equation} = x_observed
  %x_equation : P(Train.network_evaluated(Train.Then{layer,others},_))
  %l_equation : P(Train.network_evaluated(Train.Then{_,others},Layers.tensor_of(xm)))
  Layers.returned_holds(~(output => EvaluateOutput(~X,~x_value,values,input,output)),~(layer => EvaluateLayer(~X,~x_value,limits,values,input,layer)),~P,
    ~(output => layer => Train.in_front(layer,Train.network_evaluated(others,output))),
    Train.evaluate(Layers.tensor_of(xm),Layers.layer_of(lm)),
    layer_evaluate(~X,~x_value,limits,default_capacity,values,input,xm,
      (%Equal.sym(Tiled.Tiled,Layers.tensor_of(xm),x,x_equation) : Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),_) contract),
      lm,(%Equal.sym(Train.Layer,Layers.layer_of(lm),layer,l_equation) : LayerDescribed(~X,~x_value,limits,values,valued_shape(~X,input),_) layer_described)),
    next)

# A layer in front of the rest of a network.
def in_front_ran(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+values: LayerValues<X>,+rest: NetworkValues<X>,+input: Valued<X>,-layer: Train.Layer,
  +layer_described: EvaluateLayer(~X,~x_value,limits,values,input,layer),-output: Tiled.Tiled,-after: Train.Network,
  +ran: EvaluateRan(~X,~x_value,limits,rest,layer_evaluated(~X,~x_value,values,input),(output,after))) ->
  EvaluateRan(~X,~x_value,limits,Next{values,rest},input,(output,Train.Then{layer,after})):
  KernelGemm.Both{+output_contract,+rest_described} = ran
  KernelGemm.Both{output_contract,KernelGemm.Both{layer_described,rest_described}}

# The sum of what the branch and the shortcut of a block give in inference
# mode, then the rest, on the two outputs as models.
def joined_evaluated_seen(~X: Data,~x_value: X -> U32 -> F32,~P: Tiled.Tiled & Train.Network -> Data,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+branch: NetworkValues<X>,+shortcut: NetworkValues<X>,+input: Valued<X>,
  +fits: BlockFits(~X,limits,branch,shortcut,valued_shape(~X,input)),-others: Train.Network,
  -left: Tiled.Tiled,+left_observed: Layers.TensorObserved<left>,-branch_network: Train.Network,+branch_ran: EvaluateRan(~X,~x_value,limits,branch,input,(left,branch_network)),
  -right: Tiled.Tiled,+right_observed: Layers.TensorObserved<right>,-shortcut_network: Train.Network,
  +shortcut_ran: EvaluateRan(~X,~x_value,limits,shortcut,input,(right,shortcut_network)),
  next: @output: Tiled.Tiled ->
    @+output_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),
      valued_value(~X,sum_valued(~X,evaluated_valued(~X,~x_value,branch,input),evaluated_valued(~X,~x_value,shortcut,input))),
      valued_shape(~X,sum_valued(~X,evaluated_valued(~X,~x_value,branch,input),evaluated_valued(~X,~x_value,shortcut,input))),output) ->
    P(Train.block_in_front(branch_network,shortcut_network,Train.network_evaluated(others,output)))) ->
  P(Train.block_in_front(branch_network,shortcut_network,Train.network_evaluated(others,F.add(left,right)))):
  KernelGemm.Both{+same_shapes,KernelGemm.Both{Spread{+nonempty,+rank_room},+input_spread}} = fits
  Layers.TensorObserved{+lm,+l_equation} = left_observed
  KernelGemm.Both{+left_contract,+branch_described} = branch_ran
  Layers.TensorObserved{+rm,+r_equation} = right_observed
  KernelGemm.Both{+right_contract,+shortcut_described} = shortcut_ran
  +branch_shape = evaluated_shape(~X,~x_value,branch,input)
  +shortcut_shape = evaluated_shape(~X,~x_value,shortcut,input)
  +shapes = Equal.trans(List<&2,U32>,valued_shape(~X,evaluated_valued(~X,~x_value,branch,input)),network_shape(~X,branch,valued_shape(~X,input)),
    valued_shape(~X,evaluated_valued(~X,~x_value,shortcut,input)),branch_shape,
    Equal.trans(List<&2,U32>,network_shape(~X,branch,valued_shape(~X,input)),network_shape(~X,shortcut,valued_shape(~X,input)),
      valued_shape(~X,evaluated_valued(~X,~x_value,shortcut,input)),same_shapes,
      Equal.sym(List<&2,U32>,valued_shape(~X,evaluated_valued(~X,~x_value,shortcut,input)),network_shape(~X,shortcut,valued_shape(~X,input)),shortcut_shape)))
  %l_equation : P(Train.block_in_front(branch_network,shortcut_network,Train.network_evaluated(others,F.add(_,right))))
  %r_equation : P(Train.block_in_front(branch_network,shortcut_network,Train.network_evaluated(others,F.add(Layers.tensor_of(lm),_))))
  next(F.add(Layers.tensor_of(lm),Layers.tensor_of(rm)),
    wrapped(~X,~x_value,~Leaf.Context<Value<X>,Value<X>>,~Leaf.value(~F32,~Value<X>,~Value<X>,~F32.add,~interpret(~X,~x_value),~interpret(~X,~x_value)),
      ~(context => Added{context}),~(context => index => {==}),
      Leaf.Context{valued_value(~X,evaluated_valued(~X,~x_value,branch,input)),valued_value(~X,evaluated_valued(~X,~x_value,shortcut,input))},
      valued_shape(~X,evaluated_valued(~X,~x_value,branch,input)),F.add(Layers.tensor_of(lm),Layers.tensor_of(rm)),
      Binary.same(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~F32.add,
        valued_value(~X,evaluated_valued(~X,~x_value,branch,input)),valued_value(~X,evaluated_valued(~X,~x_value,shortcut,input)),
        "add",1.0,limits,default_capacity,valued_shape(~X,evaluated_valued(~X,~x_value,branch,input)),
        (%Equal.sym(List<&2,U32>,valued_shape(~X,evaluated_valued(~X,~x_value,branch,input)),network_shape(~X,branch,valued_shape(~X,input)),branch_shape) :
          {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} nonempty),
        (%Equal.sym(List<&2,U32>,valued_shape(~X,evaluated_valued(~X,~x_value,branch,input)),network_shape(~X,branch,valued_shape(~X,input)),branch_shape) :
          {Nat.is_le(List.length(&2,U32,_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room),
        Layers.tensor_of(lm),
        (%Equal.sym(Tiled.Tiled,Layers.tensor_of(lm),left,l_equation) :
          Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,evaluated_valued(~X,~x_value,branch,input)),
            valued_shape(~X,evaluated_valued(~X,~x_value,branch,input)),_) left_contract),
        Layers.tensor_of(rm),
        (%Equal.sym(Tiled.Tiled,Layers.tensor_of(rm),right,r_equation) :
          Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,evaluated_valued(~X,~x_value,shortcut,input)),
            valued_shape(~X,evaluated_valued(~X,~x_value,branch,input)),_)
          (%Equal.sym(List<&2,U32>,valued_shape(~X,evaluated_valued(~X,~x_value,branch,input)),valued_shape(~X,evaluated_valued(~X,~x_value,shortcut,input)),shapes) :
            Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,evaluated_valued(~X,~x_value,shortcut,input)),_,right) right_contract)))))

# A residual block in front of the rest of a network.
def block_ran(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+branch: NetworkValues<X>,+shortcut: NetworkValues<X>,+rest: NetworkValues<X>,+input: Valued<X>,
  +fits: BlockFits(~X,limits,branch,shortcut,valued_shape(~X,input)),
  -left: Tiled.Tiled,-branch_network: Train.Network,+branch_ran: EvaluateRan(~X,~x_value,limits,branch,input,(left,branch_network)),
  -right: Tiled.Tiled,-shortcut_network: Train.Network,+shortcut_ran: EvaluateRan(~X,~x_value,limits,shortcut,input,(right,shortcut_network)),
  -output: Tiled.Tiled,-after: Train.Network,
  +rest_ran: EvaluateRan(~X,~x_value,limits,rest,sum_valued(~X,evaluated_valued(~X,~x_value,branch,input),evaluated_valued(~X,~x_value,shortcut,input)),(output,after))) ->
  EvaluateRan(~X,~x_value,limits,Block{branch,shortcut,rest},input,(output,Train.Residual{branch_network,shortcut_network,after})):
  KernelGemm.Both{+left_contract,+branch_described} = branch_ran
  KernelGemm.Both{+right_contract,+shortcut_described} = shortcut_ran
  KernelGemm.Both{+output_contract,+rest_described} = rest_ran
  KernelGemm.Both{output_contract,KernelGemm.Both{fits,KernelGemm.Both{branch_described,KernelGemm.Both{shortcut_described,
    (%evaluated_shape(~X,~x_value,branch,input) : Described(~X,~x_value,limits,rest,_,after) rest_described)}}}}

# evaluate of a network: whatever holds of every output and network that
# hold what the network gives in inference mode holds of what evaluate gives.
def evaluate_holds(~X: Data,~x_value: X -> U32 -> F32,~P: Tiled.Tiled & Train.Network -> Data,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+description: NetworkValues<X>,+input: Valued<X>,
  network: Train.Network,+described: Described(~X,~x_value,limits,description,valued_shape(~X,input),network),
  x: Tiled.Tiled,+contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),x),
  done: @output: Tiled.Tiled -> @after: Train.Network -> @+ran: EvaluateRan(~X,~x_value,limits,description,input,(output,after)) -> P((output,after))) ->
  P(Train.network_evaluated(network,x)):
  match description:
    case Last{}:
      match network:
        case Train.End{}: done(x,Train.End{},KernelGemm.Both{contract,{==}})
        case Train.Then{other_layer,other_rest}:
          Empty.absurd(P(Train.network_evaluated(Train.Then{other_layer,other_rest},x)),Words.false_true(described))
        case Train.Residual{other_branch,other_shortcut,other_rest}:
          Empty.absurd(P(Train.network_evaluated(Train.Residual{other_branch,other_shortcut,other_rest},x)),Words.false_true(described))
    case Next{+values,+rest}:
      match network:
        case Train.Then{layer,others}:
          KernelGemm.Both{+layer_described,+rest_described} = described
          layer_evaluated_seen(~X,~x_value,~P,limits,default_capacity,values,input,layer,Layers.observe_layer(layer),layer_described,others,x,Layers.observe_tensor(x),contract,
            output => after => output_contract => after_described =>
              evaluate_holds(~X,~x_value,~(result => P(Train.in_front(after,result))),limits,default_capacity,rest,layer_evaluated(~X,~x_value,values,input),others,
                rest_described,output,output_contract,
                rest_output => rest_after => rest_ran =>
                  done(rest_output,Train.Then{after,rest_after},in_front_ran(~X,~x_value,limits,values,rest,input,after,after_described,rest_output,rest_after,rest_ran))))
        case Train.End{}:
          Empty.absurd(P(Train.network_evaluated(Train.End{},x)),Words.false_true(described))
        case Train.Residual{other_branch,other_shortcut,other_rest}:
          Empty.absurd(P(Train.network_evaluated(Train.Residual{other_branch,other_shortcut,other_rest},x)),Words.false_true(described))
    case Block{+branch,+shortcut,+rest}:
      match network:
        case Train.Residual{branch_network,shortcut_network,others}:
          KernelGemm.Both{+fits,KernelGemm.Both{+branch_described,KernelGemm.Both{+shortcut_described,+rest_described}}} = described
          Copy.tiled_copied(~Value<X>,~interpret(~X,~x_value),
            ~(copies => P(Train.split(~(Tiled.Tiled & Train.Network),copies,first => second =>
              Train.met(Train.network_evaluated(branch_network,first),Train.network_evaluated(shortcut_network,second),
                left => right => branch_result => shortcut_result => Train.block_in_front(branch_result,shortcut_result,Train.network_evaluated(others,F.add(left,right))))))),
            valued_value(~X,input),valued_shape(~X,input),x,contract,
            first => second => first_contract => second_contract =>
              evaluate_holds(~X,~x_value,
                ~(branch_result => P(Train.met(branch_result,Train.network_evaluated(shortcut_network,second),
                  left => right => branch_result => shortcut_result => Train.block_in_front(branch_result,shortcut_result,Train.network_evaluated(others,F.add(left,right)))))),
                limits,default_capacity,branch,input,branch_network,branch_described,first,first_contract,
                left => branch_after => branch_ran =>
                  evaluate_holds(~X,~x_value,
                    ~(shortcut_result => P(Train.met((left,branch_after),shortcut_result,
                      left => right => branch_result => shortcut_result => Train.block_in_front(branch_result,shortcut_result,Train.network_evaluated(others,F.add(left,right)))))),
                    limits,default_capacity,shortcut,input,shortcut_network,shortcut_described,second,second_contract,
                    right => shortcut_after => shortcut_ran =>
                      joined_evaluated_seen(~X,~x_value,~P,limits,default_capacity,branch,shortcut,input,fits,others,left,Layers.observe_tensor(left),branch_after,branch_ran,
                        right,Layers.observe_tensor(right),shortcut_after,shortcut_ran,
                        output => output_contract =>
                          evaluate_holds(~X,~x_value,~(result => P(Train.block_in_front(branch_after,shortcut_after,result))),limits,default_capacity,rest,
                            sum_valued(~X,evaluated_valued(~X,~x_value,branch,input),evaluated_valued(~X,~x_value,shortcut,input)),others,
                            (%Equal.sym(List<&2,U32>,valued_shape(~X,evaluated_valued(~X,~x_value,branch,input)),network_shape(~X,branch,valued_shape(~X,input)),
                                evaluated_shape(~X,~x_value,branch,input)) :
                              Described(~X,~x_value,limits,rest,_,others) rest_described),
                            output,output_contract,
                            rest_output => rest_after => rest_ran =>
                              done(rest_output,Train.Residual{branch_after,shortcut_after,rest_after},
                                block_ran(~X,~x_value,limits,branch,shortcut,rest,input,fits,left,branch_after,branch_ran,right,shortcut_after,shortcut_ran,
                                  rest_output,rest_after,rest_ran)))))))
        case Train.End{}:
          Empty.absurd(P(Train.network_evaluated(Train.End{},x)),Words.false_true(described))
        case Train.Then{other_layer,other_rest}:
          Empty.absurd(P(Train.network_evaluated(Train.Then{other_layer,other_rest},x)),Words.false_true(described))

# evaluate_network of the description on an input: the output holds what the
# network gives of the input in inference mode, and the network comes back as
# described.
def NetworkEvaluate(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+description: NetworkValues<X>,+input: Valued<X>,result: Tiled.Tiled & Train.Network) -> Data:
  EvaluateRan(~X,~x_value,limits,description,input,result)

def evaluate(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +description: NetworkValues<X>,+input: Valued<X>,network: Train.Network,+described: Described(~X,~x_value,limits,description,valued_shape(~X,input),network),
  x: Tiled.Tiled,+contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),valued_shape(~X,input),x)) ->
  NetworkEvaluate(~X,~x_value,limits,description,input,Train.evaluate_network(x,network)):
  evaluate_holds(~X,~x_value,~(result => NetworkEvaluate(~X,~x_value,limits,description,input,result)),limits,default_capacity,description,input,network,described,
    x,contract,output => after => ran => ran)

# ---- Gradients ----

# The gradient a layer gives its input from the gradient of its output, as
# the layer forward left it reads the input.
def layer_gradient(~X: Data,~x_value: X -> U32 -> F32,+values: LayerValues<X>,+input: Valued<X>,+gradient: Value<X>) -> Value<X>:
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,grading}:
      ConvolutionInput{InputPublic.InputRead{Leaf.Context{gradient,FunctionValues.ActivationRead{Activations.identity(),
        convolution_context(~X,~x_value,valued_value(~X,input),channels,outputs,height,width,kernel,stride,padding,weights,bias)}},
        Layers.parameter_value(~Value<X>,weights),TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding)}}
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,momentum,+gamma,+beta,mean,variance,grading}:
      NormGradient{NormPublic.BackChannels{gradient,norm_context(~X,~x_value,valued_value(~X,input),images,channels,height,width,epsilon,gamma,beta),
        NormPublic.shift_sums(~Value<X>,~interpret(~X,~x_value),gradient,images,channels,height,width),
        NormPublic.product_sums(~Value<X>,~interpret(~X,~x_value),~NormXhat(~Value<X>),~xhat_value(~Value<X>,~interpret(~X,~x_value)),gradient,
          norm_context(~X,~x_value,valued_value(~X,input),images,channels,height,width,epsilon,gamma,beta),images,channels,height,width),
        NormPublic.scale_context(~Value<X>,~NormInverse(),Layers.parameter_value(~Value<X>,gamma),
          NormPublic.deviation_context(~NormPublic.Quotient(),NormPublic.channel_variances(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width),
            epsilon),images,height,width),
        U32.to_f32((images * (height * width : U32) : U32)),Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})}}
    case LayerNormOf{features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,beta,grading}:
      LineGradient{LineValues.input_gradient_context(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),
        valued_value(~X,input),Layers.parameter_value(~Value<X>,gamma),gradient,valued_shape(~X,input),epsilon,laxes,loffset,raxes,roffset,gaxes,goffset)}
    case ActivationOf{+activation}: ActivationGradient{Leaf.Context{gradient,FunctionValues.ActivationRead{activation,valued_value(~X,input)}}}
    case PoolOf{images,channels,+height,+width,+axes,+offset}:
      PoolGradient{ViewGather.Viewing{PiecesMap.Context{U32.to_f32((height * width : U32)),gradient},axes,offset,[]}}

# A layer with the gradients of its parameters from the gradient of its
# output.
def layer_graded(~X: Data,~x_value: X -> U32 -> F32,+values: LayerValues<X>,+input: Valued<X>,+gradient: Value<X>) -> LayerValues<X>:
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,grading}:
      ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,
        Graded{ConvolutionWeight{ParameterPublic.weight_read(~Value<X>,~TrainingConvolution.Delta(~Value<X>,~Value<X>),valued_value(~X,input),
            Leaf.Context{gradient,FunctionValues.ActivationRead{Activations.identity(),
              convolution_context(~X,~x_value,valued_value(~X,input),channels,outputs,height,width,kernel,stride,padding,weights,bias)}},
            mode,TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding))},
          ConvolutionBias{ParameterPublic.bias_read(~TrainingConvolution.Delta(~Value<X>,~Value<X>),
            Leaf.Context{gradient,FunctionValues.ActivationRead{Activations.identity(),
              convolution_context(~X,~x_value,valued_value(~X,input),channels,outputs,height,width,kernel,stride,padding,weights,bias)}},
            mode,TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding))}}}
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,+momentum,+gamma,+beta,+mean,+variance,grading}:
      BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,
        Graded{Summed{NormPublic.product_sums(~Value<X>,~interpret(~X,~x_value),~NormXhat(~Value<X>),~xhat_value(~Value<X>,~interpret(~X,~x_value)),gradient,
            norm_context(~X,~x_value,valued_value(~X,input),images,channels,height,width,epsilon,gamma,beta),images,channels,height,width)},
          Summed{NormPublic.shift_sums(~Value<X>,~interpret(~X,~x_value),gradient,images,channels,height,width)}}}
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,grading}:
      LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,
        Graded{Summed{LineValues.gamma_gradient_context(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),gradient,
            valued_shape(~X,input),epsilon,laxes,loffset,raxes,roffset)},
          Summed{LineValues.beta_gradient_context(~Value<X>,~interpret(~X,~x_value),gradient,valued_shape(~X,input))}}}
    case ActivationOf{+activation}: ActivationOf{activation}
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}: PoolOf{images,channels,height,width,axes,offset}

# The gradient a trace gives its input from the gradient of its output: each
# layer from the gradient the layers after it give, and a block the sum of
# what its branch and its shortcut give.
def trace_gradient(~X: Data,~x_value: X -> U32 -> F32,+trace: Trace<X>,+input: Valued<X>,+gradient: Value<X>) -> Value<X>:
  match trace:
    case Unwritten{}: gradient
    case Written{+values,+rest}:
      trace_gradient(~X,~x_value,rest,input,layer_gradient(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient))
    case Branched{+branch,+shortcut,+rest}:
      trace_gradient(~X,~x_value,rest,input,Added{Leaf.Context{trace_gradient(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input),gradient),
        trace_gradient(~X,~x_value,shortcut,trace_valued(~X,~x_value,rest,input),gradient)}})

# The network backward gives, in front of the later layers: every layer of
# the trace with the gradients of its parameters.
def trace_network(~X: Data,~x_value: X -> U32 -> F32,+trace: Trace<X>,+input: Valued<X>,+gradient: Value<X>,+later: NetworkValues<X>) -> NetworkValues<X>:
  match trace:
    case Unwritten{}: later
    case Written{+values,+rest}:
      trace_network(~X,~x_value,rest,input,layer_gradient(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient),
        Next{layer_graded(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient),later})
    case Branched{+branch,+shortcut,+rest}:
      trace_network(~X,~x_value,rest,input,Added{Leaf.Context{trace_gradient(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input),gradient),
          trace_gradient(~X,~x_value,shortcut,trace_valued(~X,~x_value,rest,input),gradient)}},
        Block{trace_network(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input),gradient,Last{}),
          trace_network(~X,~x_value,shortcut,trace_valued(~X,~x_value,rest,input),gradient,Last{}),later})

# Gradients leave the extents a layer gives.
def graded_shape(~X: Data,~x_value: X -> U32 -> F32,+values: LayerValues<X>,+input: Valued<X>,+gradient: Value<X>,+shape: List<&2,U32>) ->
  {layer_shape(~X,layer_graded(~X,~x_value,values,input,gradient),shape) == layer_shape(~X,values,shape) : List<&2,U32>}:
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,+grading}: {==}
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,+momentum,+gamma,+beta,+mean,+variance,+grading}: {==}
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,+grading}: {==}
    case ActivationOf{+activation}: {==}
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}: {==}

# The network backward gives of a trace has the extents of the trace.
def gradient_network_shape(~X: Data,~x_value: X -> U32 -> F32,+trace: Trace<X>,+input: Valued<X>,+gradient: Value<X>,+later: NetworkValues<X>) ->
  {network_shape(~X,trace_network(~X,~x_value,trace,input,gradient,later),valued_shape(~X,input)) ==
    network_shape(~X,later,valued_shape(~X,trace_valued(~X,~x_value,trace,input))) : List<&2,U32>}:
  match trace:
    case Unwritten{}: {==}
    case Written{+values,+rest}:
      %graded_shape(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient,valued_shape(~X,trace_valued(~X,~x_value,rest,input))) :
        {network_shape(~X,trace_network(~X,~x_value,rest,input,layer_gradient(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient),
          Next{layer_graded(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient),later}),valued_shape(~X,input)) == network_shape(~X,later,_) : List<&2,U32>}
      gradient_network_shape(~X,~x_value,rest,input,layer_gradient(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient),
        Next{layer_graded(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient),later})
    case Branched{+branch,+shortcut,+rest}:
      %gradient_network_shape(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input),gradient,Last{}) :
        {network_shape(~X,trace_network(~X,~x_value,rest,input,Added{Leaf.Context{trace_gradient(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input),gradient),
            trace_gradient(~X,~x_value,shortcut,trace_valued(~X,~x_value,rest,input),gradient)}},
          Block{trace_network(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input),gradient,Last{}),
            trace_network(~X,~x_value,shortcut,trace_valued(~X,~x_value,rest,input),gradient,Last{}),later}),valued_shape(~X,input)) == network_shape(~X,later,_) : List<&2,U32>}
      gradient_network_shape(~X,~x_value,rest,input,Added{Leaf.Context{trace_gradient(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input),gradient),
          trace_gradient(~X,~x_value,shortcut,trace_valued(~X,~x_value,rest,input),gradient)}},
        Block{trace_network(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input),gradient,Last{}),
          trace_network(~X,~x_value,shortcut,trace_valued(~X,~x_value,rest,input),gradient,Last{}),later})

# ---- Backward of a layer ----

# The output of backward holds the gradient the layer gives its input, and
# the layer comes back with the gradients of its parameters.
def BackwardOutput(~X: Data,~x_value: X -> U32 -> F32,+values: LayerValues<X>,+input: Valued<X>,+gradient: Value<X>,output: Tiled.Tiled) -> Data:
  Public.Contract(~Value<X>,~interpret(~X,~x_value),layer_gradient(~X,~x_value,values,input,gradient),valued_shape(~X,input),output)

def BackwardLayer(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+values: LayerValues<X>,+input: Valued<X>,+gradient: Value<X>,layer: Train.Layer) -> Data:
  LayerDescribed(~X,~x_value,limits,layer_graded(~X,~x_value,values,input,gradient),valued_shape(~X,input),layer)

def LayerBackward(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+values: LayerValues<X>,+input: Valued<X>,+gradient: Value<X>,
  result: Tiled.Tiled & Train.Layer) -> Data:
  Layers.Answered(~(output => BackwardOutput(~X,~x_value,values,input,gradient,output)),~(layer => BackwardLayer(~X,~x_value,limits,values,input,gradient,layer)),result)

# backward of an activation.
def activation_backward_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +activation: Activations.Activation,+input: Valued<X>,+gradient: Value<X>,+ym: Layers.TensorModel,
  +dy_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),gradient,valued_shape(~X,input),Layers.tensor_of(ym)),+sm: Layers.SavedModel,
  +kept: LayerSaved(~X,~x_value,ActivationOf{activation},input,Layers.saved_of(sm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,ActivationOf{activation},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerBackward(~X,~x_value,limits,ActivationOf{activation},input,gradient,Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm))):
  KernelGemm.Both{KernelGemm.Both{+built,Spread{+nonempty,+rank_room}},+layered} = described
  answered_mapped(~(output => FunctionValues.ActivatedBackward(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),activation,gradient,valued_value(~X,input),
      valued_shape(~X,input),output)),~(layer => Layers.ActivationLayer(activation,layer)),
    ~(output => BackwardOutput(~X,~x_value,ActivationOf{activation},input,gradient,output)),~(layer => BackwardLayer(~X,~x_value,limits,ActivationOf{activation},input,gradient,layer)),
    Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm)),
    Layers.activation_backward(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),activation,gradient,valued_value(~X,input),limits,default_capacity,
      valued_shape(~X,input),nonempty,rank_room,built,Layers.tensor_of(ym),dy_contract,Layers.saved_of(sm),kept,Layers.layer_of(lm),layered),
    (output => output_held => wrapped(~X,~x_value,~Leaf.Context<Value<X>,FunctionValues.ActivationRead<Value<X>>>,
      ~Leaf.value(~F32,~Value<X>,~FunctionValues.ActivationRead<Value<X>>,~F32.mul,~interpret(~X,~x_value),~FunctionValues.derivative_at(~Value<X>,~interpret(~X,~x_value))),
      ~(context => ActivationGradient{context}),~(context => index => {==}),Leaf.Context{gradient,FunctionValues.ActivationRead{activation,valued_value(~X,input)}},
      valued_shape(~X,input),output,output_held)),
    (layer => layer_held => KernelGemm.Both{KernelGemm.Both{built,Spread{nonempty,rank_room}},layer_held}))

# backward of the average pool.
def pool_backward_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +images: U32,+channels: U32,+height: U32,+width: U32,+axes: List<&2,Views.Axis>,+offset: U32,+input: Valued<X>,+gradient: Value<X>,+ym: Layers.TensorModel,
  +dy_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),gradient,[images,channels,1,1],Layers.tensor_of(ym)),+sm: Layers.SavedModel,
  +kept: LayerSaved(~X,~x_value,PoolOf{images,channels,height,width,axes,offset},input,Layers.saved_of(sm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,PoolOf{images,channels,height,width,axes,offset},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerBackward(~X,~x_value,limits,PoolOf{images,channels,height,width,axes,offset},input,gradient,
    Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  PoolFits{+same,+facts,+spread} = fits
  answered_mapped(~(output => Layers.PoolGradient(~Value<X>,~interpret(~X,~x_value),gradient,images,channels,height,width,axes,offset,output)),
    ~(layer => Layers.PoolLayer(layer)),~(output => BackwardOutput(~X,~x_value,PoolOf{images,channels,height,width,axes,offset},input,gradient,output)),
    ~(layer => BackwardLayer(~X,~x_value,limits,PoolOf{images,channels,height,width,axes,offset},input,gradient,layer)),
    Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm)),
    Layers.pool_backward(~Value<X>,~interpret(~X,~x_value),gradient,images,channels,height,width,axes,offset,limits,default_capacity,spread,Layers.tensor_of(ym),dy_contract,
      Layers.saved_of(sm),kept,Layers.layer_of(lm),layered),
    (output => output_held =>
      (%Equal.sym(List<&2,U32>,valued_shape(~X,input),[images,channels,height,width],same) :
        Public.Contract(~Value<X>,~interpret(~X,~x_value),PoolGradient{ViewGather.Viewing{PiecesMap.Context{U32.to_f32((height * width : U32)),gradient},axes,offset,[]}},_,output)
        wrapped(~X,~x_value,~Layers.PoolSpreadValue(~Value<X>),~Layers.pool_spread_value(~Value<X>,~interpret(~X,~x_value)),~(context => PoolGradient{context}),
          ~(context => index => {==}),ViewGather.Viewing{PiecesMap.Context{U32.to_f32((height * width : U32)),gradient},axes,offset,[]},[images,channels,height,width],output,
          output_held))),
    (layer => layer_held => KernelGemm.Both{fits,layer_held}))

# A result of backward of a batch normalization, as the network states it.
def norm_backward_converted(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,
  +momentum: F32,+gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,+mean: Value<X>,+variance: Value<X>,+grading: Grading<X>,+input: Valued<X>,
  +gradient: Value<X>,+fits: NormFits<valued_shape(~X,input),images,channels,height,width>,result: Tiled.Tiled & Train.Layer,
  +held: Layers.NormBackward(~Value<X>,~interpret(~X,~x_value),~NormXhat(~Value<X>),~xhat_value(~Value<X>,~interpret(~X,~x_value)),~NormInverse(),~inverse_value(),
    ~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),images,channels,height,width,epsilon,momentum,gamma,beta,gradient,
    norm_context(~X,~x_value,valued_value(~X,input),images,channels,height,width,epsilon,gamma,beta),
    NormPublic.deviation_context(~NormPublic.Quotient(),NormPublic.channel_variances(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width),epsilon),
    mean,variance,result)) ->
  LayerBackward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,gradient,result):
  NormFits{+same,+room,+channels_nonempty} = fits
  (dx,layer) = result
  match layer:
    case Train.BatchNorm{gammas,betas,means,variances,+layer_epsilon,+layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      match gamma_gradient:
        case Train.Held{dgamma}:
          match beta_gradient:
            case Train.Held{dbeta}:
              KernelGemm.Both{+epsilon_exact,KernelGemm.Both{+momentum_exact,KernelGemm.Both{
                KernelGemm.Both{+dx_contract,KernelGemm.Both{+dgamma_contract,KernelGemm.Both{+dbeta_contract,+gamma_contract}}},
                KernelGemm.Both{+beta_contract,KernelGemm.Both{+mean_contract,KernelGemm.Both{+variance_contract,KernelGemm.Both{+gamma_kept,+beta_kept}}}}}}} = held
              KernelGemm.Both{
                (%Equal.sym(List<&2,U32>,valued_shape(~X,input),[images,channels,height,width],same) :
                  Public.Contract(~Value<X>,~interpret(~X,~x_value),layer_gradient(~X,~x_value,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},
                    input,gradient),_,dx)
                  wrapped(~X,~x_value,~NormPublic.BackChannels<Value<X>,NormXhat(~Value<X>),PartialsValues.Box<Sum.Parts>,PartialsValues.Box<Sum.Parts>,NormPublic.Scale(~Value<X>,~NormInverse())>,
                    ~NormPublic.back_value(~Value<X>,~interpret(~X,~x_value),~NormXhat(~Value<X>),~xhat_value(~Value<X>,~interpret(~X,~x_value)),
                      ~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),
                      ~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),
                      ~NormPublic.Scale(~Value<X>,~NormInverse()),~NormPublic.scale_value(~Value<X>,~interpret(~X,~x_value),~NormInverse(),~inverse_value())),
                    ~(context => NormGradient{context}),~(context => index => {==}),
                    NormPublic.BackChannels{gradient,norm_context(~X,~x_value,valued_value(~X,input),images,channels,height,width,epsilon,gamma,beta),
                      NormPublic.shift_sums(~Value<X>,~interpret(~X,~x_value),gradient,images,channels,height,width),
                      NormPublic.product_sums(~Value<X>,~interpret(~X,~x_value),~NormXhat(~Value<X>),~xhat_value(~Value<X>,~interpret(~X,~x_value)),gradient,
                        norm_context(~X,~x_value,valued_value(~X,input),images,channels,height,width,epsilon,gamma,beta),images,channels,height,width),
                      NormPublic.scale_context(~Value<X>,~NormInverse(),Layers.parameter_value(~Value<X>,gamma),
                        NormPublic.deviation_context(~NormPublic.Quotient(),
                          NormPublic.channel_variances(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width),epsilon),images,height,width),
                      U32.to_f32((images * (height * width : U32) : U32)),Public.geometry_of([images,channels,height,width],1n,Grid.Outermost{})},
                    [images,channels,height,width],dx,dx_contract)),
                KernelGemm.Both{fits,KernelGemm.Both{epsilon_exact,KernelGemm.Both{momentum_exact,
                  KernelGemm.Both{KernelGemm.Both{gamma_contract,KernelGemm.Both{
                      wrapped(~X,~x_value,~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),~(context => Summed{context}),
                        ~(context => index => {==}),
                        NormPublic.product_sums(~Value<X>,~interpret(~X,~x_value),~NormXhat(~Value<X>),~xhat_value(~Value<X>,~interpret(~X,~x_value)),gradient,
                          norm_context(~X,~x_value,valued_value(~X,input),images,channels,height,width,epsilon,gamma,beta),images,channels,height,width),[channels],dgamma,dgamma_contract),
                      gamma_kept}},
                    KernelGemm.Both{KernelGemm.Both{beta_contract,KernelGemm.Both{
                        wrapped(~X,~x_value,~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),~(context => Summed{context}),
                          ~(context => index => {==}),NormPublic.shift_sums(~Value<X>,~interpret(~X,~x_value),gradient,images,channels,height,width),[channels],dbeta,dbeta_contract),
                        beta_kept}},
                      KernelGemm.Both{mean_contract,variance_contract}}}}}}}
            case Train.Empty{}:
              Empty.absurd(LayerBackward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,gradient,
                (dx,Train.BatchNorm{gammas,betas,means,variances,layer_epsilon,layer_momentum,Train.Held{dgamma},Train.Empty{},gamma_velocity,beta_velocity})),Words.false_true(held))
        case Train.Empty{}:
          Empty.absurd(LayerBackward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,gradient,
            (dx,Train.BatchNorm{gammas,betas,means,variances,layer_epsilon,layer_momentum,Train.Empty{},beta_gradient,gamma_velocity,beta_velocity})),Words.false_true(held))
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,gradient,(dx,Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity})),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,gradient,(dx,Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,gradient,(dx,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,gradient,(dx,Train.AveragePool{})),Words.false_true(held))

# backward of a batch normalization.
def norm_backward_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,+gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,
  +mean: Value<X>,+variance: Value<X>,+grading: Grading<X>,+input: Valued<X>,+gradient: Value<X>,+ym: Layers.TensorModel,
  +dy_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),gradient,[images,channels,height,width],Layers.tensor_of(ym)),+sm: Layers.SavedModel,
  +kept: LayerSaved(~X,~x_value,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,Layers.saved_of(sm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerBackward(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading},input,gradient,
    Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  NormFits{+same,+room,+channels_nonempty} = fits
  norm_backward_converted(~X,~x_value,limits,images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading,input,gradient,fits,
    Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm)),
    Layers.norm_backward(~Value<X>,~interpret(~X,~x_value),~NormXhat(~Value<X>),~xhat_value(~Value<X>,~interpret(~X,~x_value)),~NormInverse(),~inverse_value(),
      ~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[channels]),
      ~graded_second(~X,~x_value,grading,[channels]),images,channels,height,width,epsilon,momentum,gamma,beta,gradient,
      norm_context(~X,~x_value,valued_value(~X,input),images,channels,height,width,epsilon,gamma,beta),
      NormPublic.deviation_context(~NormPublic.Quotient(),NormPublic.channel_variances(~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),images,channels,height,width),epsilon),
      mean,variance,limits,default_capacity,room,Layers.tensor_of(ym),dy_contract,Layers.saved_of(sm),kept,Layers.layer_of(lm),layered))

# A result of backward of a layer normalization, as the network states it.
def line_backward_converted(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,
  +raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,+gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,
  +grading: Grading<X>,+input: Valued<X>,+gradient: Value<X>,+fits: LineFits<limits,valued_shape(~X,input),features,laxes,loffset,raxes,roffset,gaxes,goffset>,
  result: Tiled.Tiled & Train.Layer,
  +held: Layers.LineBackward(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),valued_shape(~X,input),features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,
    gamma,beta,valued_value(~X,input),gradient,result)) ->
  LayerBackward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,gradient,result):
  (dx,layer) = result
  match layer:
    case Train.LayerNorm{gammas,betas,+layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      match gamma_gradient:
        case Train.Held{dgamma}:
          match beta_gradient:
            case Train.Held{dbeta}:
              KernelGemm.Both{+epsilon_exact,KernelGemm.Both{KernelGemm.Both{+dx_contract,KernelGemm.Both{+dgamma_contract,KernelGemm.Both{+dbeta_contract,+gamma_contract}}},
                KernelGemm.Both{+beta_contract,KernelGemm.Both{+gamma_kept,+beta_kept}}}} = held
              KernelGemm.Both{
                wrapped(~X,~x_value,~LineValues.InputGradient(~Value<X>,~Value<X>,~Value<X>),
                  ~LineValues.input_gradient_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value)),
                  ~(context => LineGradient{context}),~(context => index => {==}),
                  LineValues.input_gradient_context(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),
                    valued_value(~X,input),Layers.parameter_value(~Value<X>,gamma),gradient,valued_shape(~X,input),epsilon,laxes,loffset,raxes,roffset,gaxes,goffset),
                  valued_shape(~X,input),dx,dx_contract),
                KernelGemm.Both{fits,KernelGemm.Both{epsilon_exact,
                  KernelGemm.Both{KernelGemm.Both{gamma_contract,KernelGemm.Both{
                      wrapped(~X,~x_value,~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),~(context => Summed{context}),
                        ~(context => index => {==}),
                        LineValues.gamma_gradient_context(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),valued_value(~X,input),gradient,
                          valued_shape(~X,input),epsilon,laxes,loffset,raxes,roffset),[features],dgamma,dgamma_contract),
                      gamma_kept}},
                    KernelGemm.Both{beta_contract,KernelGemm.Both{
                      wrapped(~X,~x_value,~PartialsValues.Box<Sum.Parts>,~PartialsValues.box_value(~F32,~Sum.Parts,~Sum.merge,~Sum.rounded),~(context => Summed{context}),
                        ~(context => index => {==}),LineValues.beta_gradient_context(~Value<X>,~interpret(~X,~x_value),gradient,valued_shape(~X,input)),[features],dbeta,dbeta_contract),
                      beta_kept}}}}}}
            case Train.Empty{}:
              Empty.absurd(LayerBackward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,gradient,
                (dx,Train.LayerNorm{gammas,betas,layer_epsilon,Train.Held{dgamma},Train.Empty{},gamma_velocity,beta_velocity})),Words.false_true(held))
        case Train.Empty{}:
          Empty.absurd(LayerBackward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,gradient,
            (dx,Train.LayerNorm{gammas,betas,layer_epsilon,Train.Empty{},beta_gradient,gamma_velocity,beta_velocity})),Words.false_true(held))
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,gradient,(dx,Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity})),Words.false_true(held))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,gradient,(dx,Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,gradient,(dx,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,gradient,(dx,Train.AveragePool{})),Words.false_true(held))

# backward of a layer normalization.
def line_backward_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +features: U32,+epsilon: F32,+laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,
  +gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,+grading: Grading<X>,+input: Valued<X>,+gradient: Value<X>,+ym: Layers.TensorModel,
  +dy_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),gradient,valued_shape(~X,input),Layers.tensor_of(ym)),+sm: Layers.SavedModel,
  +kept: LayerSaved(~X,~x_value,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,Layers.saved_of(sm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerBackward(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading},input,gradient,
    Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  LineFits{+facts,+features_nonempty} = fits
  line_backward_converted(~X,~x_value,limits,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading,input,gradient,fits,
    Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm)),
    Layers.line_backward(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~graded_first(~X,~x_value,grading,[features]),
      ~graded_second(~X,~x_value,grading,[features]),valued_shape(~X,input),features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,
      valued_value(~X,input),gradient,limits,default_capacity,facts,Layers.tensor_of(ym),dy_contract,Layers.saved_of(sm),kept,Layers.layer_of(lm),layered))

# A result of backward of a convolution, as the network states it.
def convolution_backward_converted(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,
  +kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<Value<X>>,+bias: Layers.Parameter<Value<X>>,+grading: Grading<X>,+input: Valued<X>,
  +gradient: Value<X>,+fits: ConvolutionFits<valued_shape(~X,input),mode,channels,outputs,height,width,kernel,stride,padding>,result: Tiled.Tiled & Train.Layer,
  +held: TrainingConvolution.ConvolutionBackward(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),
    mode,channels,outputs,height,width,kernel,stride,padding,valued_value(~X,input),weights,bias,gradient,result)) ->
  LayerBackward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,gradient,result):
  ConvolutionFits{+same,+dimensions,+machine,+images_positive,+checks,+gradients_valid,+weights_nonempty,+bias_nonempty} = fits
  (dx,layer) = result
  match layer:
    case Train.Convolution{weight_tensor,bias_tensor,+layer_stride,+layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      match weight_gradient:
        case Train.Held{dweights}:
          match bias_gradient:
            case Train.Held{dbias}:
              KernelGemm.Both{+stride_exact,KernelGemm.Both{+padding_exact,KernelGemm.Both{+dx_contract,KernelGemm.Both{+dweights_contract,KernelGemm.Both{+dbias_contract,
                KernelGemm.Both{+weight_contract,KernelGemm.Both{+bias_contract,KernelGemm.Both{+weight_kept,+bias_kept}}}}}}}} = held
              KernelGemm.Both{
                (%Equal.sym(List<&2,U32>,valued_shape(~X,input),TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width),same) :
                  Public.Contract(~Value<X>,~interpret(~X,~x_value),layer_gradient(~X,~x_value,
                    ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,gradient),_,dx)
                  wrapped(~X,~x_value,~InputPublic.InputRead<TrainingConvolution.Delta(~Value<X>,~Value<X>),Value<X>>,
                    ~InputPublic.input_at(~TrainingConvolution.Delta(~Value<X>,~Value<X>),
                      ~TrainingConvolution.delta_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value)),~Value<X>,~interpret(~X,~x_value)),
                    ~(context => ConvolutionInput{context}),~(context => index => {==}),
                    InputPublic.InputRead{Leaf.Context{gradient,FunctionValues.ActivationRead{Activations.identity(),
                      convolution_context(~X,~x_value,valued_value(~X,input),channels,outputs,height,width,kernel,stride,padding,weights,bias)}},
                      Layers.parameter_value(~Value<X>,weights),TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding)},
                    TensorConvolution.output_shape(Call.batched(mode),Call.images(mode),channels,height,width),dx,dx_contract)),
                KernelGemm.Both{fits,KernelGemm.Both{stride_exact,KernelGemm.Both{padding_exact,
                  KernelGemm.Both{KernelGemm.Both{weight_contract,KernelGemm.Both{
                      wrapped(~X,~x_value,~Terms.WeightRead<Value<X>,TrainingConvolution.Delta(~Value<X>,~Value<X>)>,
                        ~Terms.weight_at(~Value<X>,~interpret(~X,~x_value),~TrainingConvolution.Delta(~Value<X>,~Value<X>),
                          ~TrainingConvolution.delta_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value))),
                        ~(context => ConvolutionWeight{context}),~(context => index => {==}),
                        ParameterPublic.weight_read(~Value<X>,~TrainingConvolution.Delta(~Value<X>,~Value<X>),valued_value(~X,input),
                          Leaf.Context{gradient,FunctionValues.ActivationRead{Activations.identity(),
                            convolution_context(~X,~x_value,valued_value(~X,input),channels,outputs,height,width,kernel,stride,padding,weights,bias)}},
                          mode,TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding)),
                        [outputs,channels,kernel,kernel],dweights,dweights_contract),
                      weight_kept}},
                    KernelGemm.Both{bias_contract,KernelGemm.Both{
                      wrapped(~X,~x_value,~Terms.BiasRead<TrainingConvolution.Delta(~Value<X>,~Value<X>)>,
                        ~Terms.bias_at(~TrainingConvolution.Delta(~Value<X>,~Value<X>),
                          ~TrainingConvolution.delta_value(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value))),
                        ~(context => ConvolutionBias{context}),~(context => index => {==}),
                        ParameterPublic.bias_read(~TrainingConvolution.Delta(~Value<X>,~Value<X>),
                          Leaf.Context{gradient,FunctionValues.ActivationRead{Activations.identity(),
                            convolution_context(~X,~x_value,valued_value(~X,input),channels,outputs,height,width,kernel,stride,padding,weights,bias)}},
                          mode,TrainingConvolution.geometry_of(channels,outputs,height,width,kernel,stride,padding)),
                        [outputs],dbias,dbias_contract),
                      bias_kept}}}}}}}
            case Train.Empty{}:
              Empty.absurd(LayerBackward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,gradient,
                (dx,Train.Convolution{weight_tensor,bias_tensor,layer_stride,layer_padding,Train.Held{dweights},Train.Empty{},weight_velocity,bias_velocity})),Words.false_true(held))
        case Train.Empty{}:
          Empty.absurd(LayerBackward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,gradient,
            (dx,Train.Convolution{weight_tensor,bias_tensor,layer_stride,layer_padding,Train.Empty{},bias_gradient,weight_velocity,bias_velocity})),Words.false_true(held))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,gradient,(dx,Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,gradient,(dx,Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity})),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,gradient,(dx,Train.Activation{other_activation})),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LayerBackward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,gradient,(dx,Train.AveragePool{})),Words.false_true(held))

# backward of a convolution.
def convolution_backward_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+mode: Call.Mode,+channels: U32,+outputs: U32,+height: U32,+width: U32,
  +kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<Value<X>>,+bias: Layers.Parameter<Value<X>>,+grading: Grading<X>,+input: Valued<X>,+gradient: Value<X>,
  +ym: Layers.TensorModel,
  +dy_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),gradient,LayerPublic.output_extents(Call.batched(mode),Call.images(mode),outputs,height,width,kernel,stride,padding),
    Layers.tensor_of(ym)),+sm: Layers.SavedModel,
  +kept: LayerSaved(~X,~x_value,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,Layers.saved_of(sm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},valued_shape(~X,input),
    Layers.layer_of(lm))) ->
  LayerBackward(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading},input,gradient,
    Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm))):
  KernelGemm.Both{+fits,+layered} = described
  ConvolutionFits{+same,+dimensions,+machine,+images_positive,+checks,+gradients_valid,+weights_nonempty,+bias_nonempty} = fits
  convolution_backward_converted(~X,~x_value,limits,mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading,input,gradient,fits,
    Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm)),
    TrainingConvolution.convolution_backward(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),
      ~graded_first(~X,~x_value,grading,[outputs,channels,kernel,kernel]),~graded_second(~X,~x_value,grading,[outputs]),
      mode,channels,outputs,height,width,kernel,stride,padding,valued_value(~X,input),weights,bias,gradient,checks,images_positive,gradients_valid,limits,default_capacity,
      weights_nonempty,Layers.tensor_of(ym),dy_contract,Layers.saved_of(sm),kept,Layers.layer_of(lm),layered))

# backward of a layer of the description on a gradient, what forward saved
# and a layer of models.
def layer_backward(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +values: LayerValues<X>,+input: Valued<X>,+gradient: Value<X>,+ym: Layers.TensorModel,
  +dy_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),gradient,valued_shape(~X,layer_valued(~X,~x_value,values,input)),Layers.tensor_of(ym)),
  +sm: Layers.SavedModel,+kept: LayerSaved(~X,~x_value,values,input,Layers.saved_of(sm)),+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,values,valued_shape(~X,input),Layers.layer_of(lm))) ->
  LayerBackward(~X,~x_value,limits,values,input,gradient,Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm))):
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,+grading}:
      convolution_backward_step(~X,~x_value,limits,default_capacity,mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,grading,input,gradient,ym,dy_contract,
        sm,kept,lm,described)
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,+momentum,+gamma,+beta,+mean,+variance,+grading}:
      norm_backward_step(~X,~x_value,limits,default_capacity,images,channels,height,width,epsilon,momentum,gamma,beta,mean,variance,grading,input,gradient,ym,dy_contract,
        sm,kept,lm,described)
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,+grading}:
      line_backward_step(~X,~x_value,limits,default_capacity,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,grading,input,gradient,ym,dy_contract,
        sm,kept,lm,described)
    case ActivationOf{+activation}: activation_backward_step(~X,~x_value,limits,default_capacity,activation,input,gradient,ym,dy_contract,sm,kept,lm,described)
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}:
      pool_backward_step(~X,~x_value,limits,default_capacity,images,channels,height,width,axes,offset,input,gradient,ym,dy_contract,sm,kept,lm,described)

# ---- Backward of a network ----

# What backward of a tape gives in front of the later layers: the gradient of
# the input holds what the trace gives it, and the network holds every layer
# of the trace with the gradients of its parameters.
def BackwardRan(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+trace: Trace<X>,+input: Valued<X>,+gradient: Value<X>,+later: NetworkValues<X>,
  result: Tiled.Tiled & Train.Network) -> Data:
  (output,network) = result
  KernelGemm.Both<Public.Contract(~Value<X>,~interpret(~X,~x_value),trace_gradient(~X,~x_value,trace,input,gradient),valued_shape(~X,input),output),
    Described(~X,~x_value,limits,trace_network(~X,~x_value,trace,input,gradient,later),valued_shape(~X,input),network)>

# backward of a record and then of the rest of the tape, on the gradient, what
# forward saved and the layer as models.
def layer_backward_seen(~X: Data,~x_value: X -> U32 -> F32,~P: Tiled.Tiled & Train.Network -> Data,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+values: LayerValues<X>,+input: Valued<X>,+gradient: Value<X>,
  -saved: Train.Saved,+saved_observed: Layers.SavedObserved<saved>,-layer: Train.Layer,+layer_observed: Layers.LayerObserved<layer>,
  +recorded: KernelGemm.Both<LayerSaved(~X,~x_value,values,input,saved),LayerDescribed(~X,~x_value,limits,values,valued_shape(~X,input),layer)>,
  -others: Train.Tape,-dy: Tiled.Tiled,+dy_observed: Layers.TensorObserved<dy>,
  +dy_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),gradient,valued_shape(~X,layer_valued(~X,~x_value,values,input)),dy),-network: Train.Network,
  next: @output: Tiled.Tiled -> @after: Train.Layer -> @+output_contract: BackwardOutput(~X,~x_value,values,input,gradient,output) ->
    @+after_described: BackwardLayer(~X,~x_value,limits,values,input,gradient,after) -> P(Train.network_backward(others,output,Train.Then{after,network}))) ->
  P(Train.network_backward(Train.Recorded{saved,layer,others},dy,network)):
  Layers.SavedObserved{+sm,+s_equation} = saved_observed
  Layers.LayerObserved{+lm,+l_equation} = layer_observed
  KernelGemm.Both{+kept,+layered} = recorded
  Layers.TensorObserved{+ym,+y_equation} = dy_observed
  %y_equation : P(Train.network_backward(Train.Recorded{saved,layer,others},_,network))
  %s_equation : P(Train.network_backward(Train.Recorded{_,layer,others},Layers.tensor_of(ym),network))
  %l_equation : P(Train.network_backward(Train.Recorded{Layers.saved_of(sm),_,others},Layers.tensor_of(ym),network))
  Layers.returned_holds(~(output => BackwardOutput(~X,~x_value,values,input,gradient,output)),~(layer => BackwardLayer(~X,~x_value,limits,values,input,gradient,layer)),~P,
    ~(output => layer => Train.network_backward(others,output,Train.Then{layer,network})),
    Train.backward(Layers.tensor_of(ym),Layers.saved_of(sm),Layers.layer_of(lm)),
    layer_backward(~X,~x_value,limits,default_capacity,values,input,gradient,ym,
      (%Equal.sym(Tiled.Tiled,Layers.tensor_of(ym),dy,y_equation) :
        Public.Contract(~Value<X>,~interpret(~X,~x_value),gradient,valued_shape(~X,layer_valued(~X,~x_value,values,input)),_) dy_contract),
      sm,(%Equal.sym(Train.Saved,Layers.saved_of(sm),saved,s_equation) : LayerSaved(~X,~x_value,values,input,_) kept),
      lm,(%Equal.sym(Train.Layer,Layers.layer_of(lm),layer,l_equation) : LayerDescribed(~X,~x_value,limits,values,valued_shape(~X,input),_) layered)),
    next)

# The sum of the gradients the branch and the shortcut of a block give its
# input, then the rest of the tape, on the two gradients as models.
def joined_backward_seen(~X: Data,~x_value: X -> U32 -> F32,~P: Tiled.Tiled & Train.Network -> Data,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+branch: Trace<X>,+shortcut: Trace<X>,+input: Valued<X>,+gradient: Value<X>,
  +later: NetworkValues<X>,
  +same_shapes: {valued_shape(~X,trace_valued(~X,~x_value,branch,input)) == valued_shape(~X,trace_valued(~X,~x_value,shortcut,input)) : List<&2,U32>},
  +branch_spread: Spread<limits,valued_shape(~X,trace_valued(~X,~x_value,branch,input))>,+input_spread: Spread<limits,valued_shape(~X,input)>,-others: Train.Tape,
  -left: Tiled.Tiled,+left_observed: Layers.TensorObserved<left>,-branch_network: Train.Network,
  +branch_ran: BackwardRan(~X,~x_value,limits,branch,input,gradient,Last{},(left,branch_network)),
  -right: Tiled.Tiled,+right_observed: Layers.TensorObserved<right>,-shortcut_network: Train.Network,
  +shortcut_ran: BackwardRan(~X,~x_value,limits,shortcut,input,gradient,Last{},(right,shortcut_network)),
  -network: Train.Network,+network_described: Described(~X,~x_value,limits,later,valued_shape(~X,trace_valued(~X,~x_value,branch,input)),network),
  next: @output: Tiled.Tiled ->
    @+output_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),
      Added{Leaf.Context{trace_gradient(~X,~x_value,branch,input,gradient),trace_gradient(~X,~x_value,shortcut,input,gradient)}},valued_shape(~X,input),output) ->
    @+next_described: Described(~X,~x_value,limits,Block{trace_network(~X,~x_value,branch,input,gradient,Last{}),trace_network(~X,~x_value,shortcut,input,gradient,Last{}),later},
      valued_shape(~X,input),Train.Residual{branch_network,shortcut_network,network}) ->
    P(Train.network_backward(others,output,Train.Residual{branch_network,shortcut_network,network}))) ->
  P(Train.network_backward(others,F.add(left,right),Train.Residual{branch_network,shortcut_network,network})):
  Spread{+nonempty,+rank_room} = input_spread
  Layers.TensorObserved{+lm,+l_equation} = left_observed
  KernelGemm.Both{+left_contract,+branch_described} = branch_ran
  Layers.TensorObserved{+rm,+r_equation} = right_observed
  KernelGemm.Both{+right_contract,+shortcut_described} = shortcut_ran
  +branch_shape = gradient_network_shape(~X,~x_value,branch,input,gradient,Last{})
  +shortcut_shape = gradient_network_shape(~X,~x_value,shortcut,input,gradient,Last{})
  +shapes = Equal.trans(List<&2,U32>,network_shape(~X,trace_network(~X,~x_value,branch,input,gradient,Last{}),valued_shape(~X,input)),
    valued_shape(~X,trace_valued(~X,~x_value,branch,input)),network_shape(~X,trace_network(~X,~x_value,shortcut,input,gradient,Last{}),valued_shape(~X,input)),branch_shape,
    Equal.trans(List<&2,U32>,valued_shape(~X,trace_valued(~X,~x_value,branch,input)),valued_shape(~X,trace_valued(~X,~x_value,shortcut,input)),
      network_shape(~X,trace_network(~X,~x_value,shortcut,input,gradient,Last{}),valued_shape(~X,input)),same_shapes,
      Equal.sym(List<&2,U32>,network_shape(~X,trace_network(~X,~x_value,shortcut,input,gradient,Last{}),valued_shape(~X,input)),
        valued_shape(~X,trace_valued(~X,~x_value,shortcut,input)),shortcut_shape)))
  %l_equation : P(Train.network_backward(others,F.add(_,right),Train.Residual{branch_network,shortcut_network,network}))
  %r_equation : P(Train.network_backward(others,F.add(Layers.tensor_of(lm),_),Train.Residual{branch_network,shortcut_network,network}))
  next(F.add(Layers.tensor_of(lm),Layers.tensor_of(rm)),
    wrapped(~X,~x_value,~Leaf.Context<Value<X>,Value<X>>,~Leaf.value(~F32,~Value<X>,~Value<X>,~F32.add,~interpret(~X,~x_value),~interpret(~X,~x_value)),
      ~(context => Added{context}),~(context => index => {==}),
      Leaf.Context{trace_gradient(~X,~x_value,branch,input,gradient),trace_gradient(~X,~x_value,shortcut,input,gradient)},valued_shape(~X,input),
      F.add(Layers.tensor_of(lm),Layers.tensor_of(rm)),
      Binary.same(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~F32.add,trace_gradient(~X,~x_value,branch,input,gradient),
        trace_gradient(~X,~x_value,shortcut,input,gradient),"add",1.0,limits,default_capacity,valued_shape(~X,input),nonempty,rank_room,
        Layers.tensor_of(lm),
        (%Equal.sym(Tiled.Tiled,Layers.tensor_of(lm),left,l_equation) :
          Public.Contract(~Value<X>,~interpret(~X,~x_value),trace_gradient(~X,~x_value,branch,input,gradient),valued_shape(~X,input),_) left_contract),
        Layers.tensor_of(rm),
        (%Equal.sym(Tiled.Tiled,Layers.tensor_of(rm),right,r_equation) :
          Public.Contract(~Value<X>,~interpret(~X,~x_value),trace_gradient(~X,~x_value,shortcut,input,gradient),valued_shape(~X,input),_) right_contract))),
    KernelGemm.Both{KernelGemm.Both{shapes,KernelGemm.Both{
        (%Equal.sym(List<&2,U32>,network_shape(~X,trace_network(~X,~x_value,branch,input,gradient,Last{}),valued_shape(~X,input)),
            valued_shape(~X,trace_valued(~X,~x_value,branch,input)),branch_shape) :
          Spread<limits,_> branch_spread),Spread{nonempty,rank_room}}},
      KernelGemm.Both{branch_described,KernelGemm.Both{shortcut_described,
        (%Equal.sym(List<&2,U32>,network_shape(~X,trace_network(~X,~x_value,branch,input,gradient,Last{}),valued_shape(~X,input)),
            valued_shape(~X,trace_valued(~X,~x_value,branch,input)),branch_shape) :
          Described(~X,~x_value,limits,later,_,network) network_described)}}})

# backward of a tape that holds a trace, in front of later layers: whatever
# holds of every gradient and network that hold what the trace gives holds of
# what backward gives.
def backward_holds(~X: Data,~x_value: X -> U32 -> F32,~P: Tiled.Tiled & Train.Network -> Data,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+trace: Trace<X>,+input: Valued<X>,+gradient: Value<X>,+later: NetworkValues<X>,
  tape: Train.Tape,+taped: Taped(~X,~x_value,limits,trace,input,tape),
  dy: Tiled.Tiled,+dy_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),gradient,valued_shape(~X,trace_valued(~X,~x_value,trace,input)),dy),
  network: Train.Network,+network_described: Described(~X,~x_value,limits,later,valued_shape(~X,trace_valued(~X,~x_value,trace,input)),network),
  done: @output: Tiled.Tiled -> @after: Train.Network -> @+ran: BackwardRan(~X,~x_value,limits,trace,input,gradient,later,(output,after)) -> P((output,after))) ->
  P(Train.network_backward(tape,dy,network)):
  match trace:
    case Unwritten{}:
      match tape:
        case Train.Blank{}: done(dy,network,KernelGemm.Both{dy_contract,network_described})
        case Train.Recorded{other_saved,other_layer,other_rest}:
          Empty.absurd(P(Train.network_backward(Train.Recorded{other_saved,other_layer,other_rest},dy,network)),Words.false_true(taped))
        case Train.Joined{other_branch,other_shortcut,other_rest}:
          Empty.absurd(P(Train.network_backward(Train.Joined{other_branch,other_shortcut,other_rest},dy,network)),Words.false_true(taped))
    case Written{+values,+rest}:
      match tape:
        case Train.Recorded{saved,layer,others}:
          KernelGemm.Both{+recorded,+rest_taped} = taped
          layer_backward_seen(~X,~x_value,~P,limits,default_capacity,values,trace_valued(~X,~x_value,rest,input),gradient,saved,Layers.observe_saved(saved),
            layer,Layers.observe_layer(layer),recorded,others,dy,Layers.observe_tensor(dy),dy_contract,network,
            output => after => output_contract => after_described =>
              backward_holds(~X,~x_value,~P,limits,default_capacity,rest,input,layer_gradient(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient),
                Next{layer_graded(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient),later},others,rest_taped,output,output_contract,
                Train.Then{after,network},
                KernelGemm.Both{after_described,
                  (%Equal.sym(List<&2,U32>,
                      layer_shape(~X,layer_graded(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient),valued_shape(~X,trace_valued(~X,~x_value,rest,input))),
                      layer_shape(~X,values,valued_shape(~X,trace_valued(~X,~x_value,rest,input))),
                      graded_shape(~X,~x_value,values,trace_valued(~X,~x_value,rest,input),gradient,valued_shape(~X,trace_valued(~X,~x_value,rest,input)))) :
                    Described(~X,~x_value,limits,later,_,network) network_described)},
                done))
        case Train.Blank{}:
          Empty.absurd(P(Train.network_backward(Train.Blank{},dy,network)),Words.false_true(taped))
        case Train.Joined{other_branch,other_shortcut,other_rest}:
          Empty.absurd(P(Train.network_backward(Train.Joined{other_branch,other_shortcut,other_rest},dy,network)),Words.false_true(taped))
    case Branched{+branch,+shortcut,+rest}:
      match tape:
        case Train.Joined{branch_tape,shortcut_tape,others}:
          KernelGemm.Both{+branch_taped,KernelGemm.Both{+shortcut_taped,KernelGemm.Both{KernelGemm.Both{+same_shapes,KernelGemm.Both{+branch_spread,+input_spread}},
            +rest_taped}}} = taped
          Copy.tiled_copied(~Value<X>,~interpret(~X,~x_value),
            ~(copies => P(Train.split(~(Tiled.Tiled & Train.Network),copies,first => second =>
              Train.met(Train.network_backward(branch_tape,first,Train.End{}),Train.network_backward(shortcut_tape,second,Train.End{}),
                left => right => branch_network => shortcut_network =>
                  Train.network_backward(others,F.add(left,right),Train.Residual{branch_network,shortcut_network,network}))))),
            gradient,valued_shape(~X,trace_valued(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input))),dy,dy_contract,
            first => second => first_contract => second_contract =>
              backward_holds(~X,~x_value,
                ~(branch_result => P(Train.met(branch_result,Train.network_backward(shortcut_tape,second,Train.End{}),
                  left => right => branch_network => shortcut_network =>
                    Train.network_backward(others,F.add(left,right),Train.Residual{branch_network,shortcut_network,network})))),
                limits,default_capacity,branch,trace_valued(~X,~x_value,rest,input),gradient,Last{},branch_tape,branch_taped,first,first_contract,Train.End{},{==},
                left => branch_network => branch_ran =>
                  backward_holds(~X,~x_value,
                    ~(shortcut_result => P(Train.met((left,branch_network),shortcut_result,
                      left => right => branch_network => shortcut_network =>
                        Train.network_backward(others,F.add(left,right),Train.Residual{branch_network,shortcut_network,network})))),
                    limits,default_capacity,shortcut,trace_valued(~X,~x_value,rest,input),gradient,Last{},shortcut_tape,shortcut_taped,second,
                    (%same_shapes : Public.Contract(~Value<X>,~interpret(~X,~x_value),gradient,_,second) second_contract),Train.End{},{==},
                    right => shortcut_network => shortcut_ran =>
                      joined_backward_seen(~X,~x_value,~P,limits,default_capacity,branch,shortcut,trace_valued(~X,~x_value,rest,input),gradient,later,
                        same_shapes,branch_spread,input_spread,others,left,Layers.observe_tensor(left),branch_network,branch_ran,
                        right,Layers.observe_tensor(right),shortcut_network,shortcut_ran,network,network_described,
                        output => output_contract => next_described =>
                          backward_holds(~X,~x_value,~P,limits,default_capacity,rest,input,
                            Added{Leaf.Context{trace_gradient(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input),gradient),
                              trace_gradient(~X,~x_value,shortcut,trace_valued(~X,~x_value,rest,input),gradient)}},
                            Block{trace_network(~X,~x_value,branch,trace_valued(~X,~x_value,rest,input),gradient,Last{}),
                              trace_network(~X,~x_value,shortcut,trace_valued(~X,~x_value,rest,input),gradient,Last{}),later},
                            others,rest_taped,output,output_contract,Train.Residual{branch_network,shortcut_network,network},next_described,done)))))
        case Train.Blank{}:
          Empty.absurd(P(Train.network_backward(Train.Blank{},dy,network)),Words.false_true(taped))
        case Train.Recorded{other_saved,other_layer,other_rest}:
          Empty.absurd(P(Train.network_backward(Train.Recorded{other_saved,other_layer,other_rest},dy,network)),Words.false_true(taped))

# backward_network of a tape that holds the trace of the description on an
# input, for a gradient of the output: the gradient of the input holds what
# the trace gives it, and the network holds every layer with the gradients of
# its parameters.
def NetworkBackward(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+description: NetworkValues<X>,+input: Valued<X>,+gradient: Value<X>,
  result: Tiled.Tiled & Train.Network) -> Data:
  BackwardRan(~X,~x_value,limits,network_trace(~X,~x_value,description,Unwritten{},input),input,gradient,Last{},result)

def backward(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +description: NetworkValues<X>,+input: Valued<X>,+gradient: Value<X>,tape: Train.Tape,
  +taped: Taped(~X,~x_value,limits,network_trace(~X,~x_value,description,Unwritten{},input),input,tape),dy: Tiled.Tiled,
  +dy_contract: Public.Contract(~Value<X>,~interpret(~X,~x_value),gradient,valued_shape(~X,trace_valued(~X,~x_value,network_trace(~X,~x_value,description,Unwritten{},input),input)),dy)) ->
  NetworkBackward(~X,~x_value,limits,description,input,gradient,Train.backward_network(dy,tape)):
  backward_holds(~X,~x_value,~(result => NetworkBackward(~X,~x_value,limits,description,input,gradient,result)),limits,default_capacity,
    network_trace(~X,~x_value,description,Unwritten{},input),input,gradient,Last{},tape,taped,dy,dy_contract,Train.End{},{==},output => after => ran => ran)

# ---- A step of SGD ----

# What a step of SGD with the gradient applies to a parameter.
def step_applied(~X: Data,+parameter: Layers.Parameter<Value<X>>,+moved: Value<X>,+momentum: F32,+decay: F32) -> StepApplied(~Value<X>):
  OptimizerValues.Applied{F32.is_eq(momentum,0.0),Layers.parameter_moving(~Value<X>,parameter),momentum,
    OptimizerValues.Direction{F32.is_eq(decay,0.0),decay,Layers.parameter_value(~Value<X>,parameter),moved},Layers.parameter_velocity(~Value<X>,parameter)}

# A parameter after a step of SGD with the gradient: the value the step
# leaves and the velocity it applied.
def parameter_descended(~X: Data,+parameter: Layers.Parameter<Value<X>>,+moved: Value<X>,+rate: F32,+momentum: F32,+decay: F32) -> Layers.Parameter<Value<X>>:
  Layers.Parameter{Stepped{OptimizerValues.step_context(~Value<X>,~StepApplied(~Value<X>),Layers.parameter_value(~Value<X>,parameter),
      step_applied(~X,parameter,moved,momentum,decay),rate)},Velocity{step_applied(~X,parameter,moved,momentum,decay)},Bool.not(F32.is_eq(momentum,0.0))}

# A layer after a step of SGD: every parameter with a gradient steps, and
# the gradients are cleared.
def layer_descended(~X: Data,+values: LayerValues<X>,+rate: F32,+momentum: F32,+decay: F32) -> LayerValues<X>:
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,+grading}:
      match grading:
        case Ungraded{}: ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,Ungraded{}}
        case Graded{+first,+second}:
          ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,parameter_descended(~X,weights,first,rate,momentum,decay),
            parameter_descended(~X,bias,second,rate,momentum,decay),Ungraded{}}
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,+norm_momentum,+gamma,+beta,+mean,+variance,+grading}:
      match grading:
        case Ungraded{}: BatchNormOf{images,channels,height,width,epsilon,norm_momentum,gamma,beta,mean,variance,Ungraded{}}
        case Graded{+first,+second}:
          BatchNormOf{images,channels,height,width,epsilon,norm_momentum,parameter_descended(~X,gamma,first,rate,momentum,decay),
            parameter_descended(~X,beta,second,rate,momentum,decay),mean,variance,Ungraded{}}
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,+grading}:
      match grading:
        case Ungraded{}: LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,Ungraded{}}
        case Graded{+first,+second}:
          LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,parameter_descended(~X,gamma,first,rate,momentum,decay),
            parameter_descended(~X,beta,second,rate,momentum,decay),Ungraded{}}
    case ActivationOf{+activation}: ActivationOf{activation}
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}: PoolOf{images,channels,height,width,axes,offset}

def network_descended(~X: Data,+network: NetworkValues<X>,+rate: F32,+momentum: F32,+decay: F32) -> NetworkValues<X>:
  match network:
    case Last{}: Last{}
    case Next{+layer,+rest}: Next{layer_descended(~X,layer,rate,momentum,decay),network_descended(~X,rest,rate,momentum,decay)}
    case Block{+branch,+shortcut,+rest}:
      Block{network_descended(~X,branch,rate,momentum,decay),network_descended(~X,shortcut,rate,momentum,decay),network_descended(~X,rest,rate,momentum,decay)}

# A step leaves the extents a layer and a network give.
def descended_shape(~X: Data,+values: LayerValues<X>,+rate: F32,+momentum: F32,+decay: F32,+shape: List<&2,U32>) ->
  {layer_shape(~X,layer_descended(~X,values,rate,momentum,decay),shape) == layer_shape(~X,values,shape) : List<&2,U32>}:
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,+grading}:
      match grading:
        case Ungraded{}: {==}
        case Graded{+first,+second}: {==}
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,+norm_momentum,+gamma,+beta,+mean,+variance,+grading}:
      match grading:
        case Ungraded{}: {==}
        case Graded{+first,+second}: {==}
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,+grading}:
      match grading:
        case Ungraded{}: {==}
        case Graded{+first,+second}: {==}
    case ActivationOf{+activation}: {==}
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}: {==}

def network_descended_shape(~X: Data,+network: NetworkValues<X>,+rate: F32,+momentum: F32,+decay: F32,+shape: List<&2,U32>) ->
  {network_shape(~X,network_descended(~X,network,rate,momentum,decay),shape) == network_shape(~X,network,shape) : List<&2,U32>}:
  match network:
    case Last{}: {==}
    case Next{+layer,+rest}:
      %Equal.sym(List<&2,U32>,layer_shape(~X,layer_descended(~X,layer,rate,momentum,decay),shape),layer_shape(~X,layer,shape),descended_shape(~X,layer,rate,momentum,decay,shape)) :
        {network_shape(~X,network_descended(~X,rest,rate,momentum,decay),_) == network_shape(~X,rest,layer_shape(~X,layer,shape)) : List<&2,U32>}
      network_descended_shape(~X,rest,rate,momentum,decay,layer_shape(~X,layer,shape))
    case Block{+branch,+shortcut,+rest}:
      %Equal.sym(List<&2,U32>,network_shape(~X,network_descended(~X,branch,rate,momentum,decay),shape),network_shape(~X,branch,shape),
          network_descended_shape(~X,branch,rate,momentum,decay,shape)) :
        {network_shape(~X,network_descended(~X,rest,rate,momentum,decay),_) == network_shape(~X,rest,network_shape(~X,branch,shape)) : List<&2,U32>}
      network_descended_shape(~X,rest,rate,momentum,decay,network_shape(~X,branch,shape))

# A velocity kept for a reader is kept for the value made of its context.
def kept_wrapped(~X: Data,~x_value: X -> U32 -> F32,~C: Data,~reader: C -> U32 -> F32,~make: C -> Value<X>,
  ~same: @+context: C -> @+index: U32 -> {reader(context,index) == interpret(~X,~x_value,make(context),index) : F32},
  +context: C,+shape: List<&2,U32>,+moving: Bool,velocity: Optimizer.Velocity,+kept: OptimizerValues.Kept(~C,~reader,context,shape,moving,velocity)) ->
  OptimizerValues.Kept(~Value<X>,~interpret(~X,~x_value),make(context),shape,moving,velocity):
  match velocity:
    case Optimizer.Still{}: kept
    case Optimizer.Moving{tensor,rest}:
      KernelGemm.Both{+moving_exact,+contract} = kept
      KernelGemm.Both{moving_exact,wrapped(~X,~x_value,~C,~reader,~make,~same,context,shape,tensor,contract)}

# A parameter after a step holds the stepped value, its gradient is cleared
# and its velocity holds what the step applied.
def parameter_slot(~X: Data,~x_value: X -> U32 -> F32,+parameter: Layers.Parameter<Value<X>>,+moved: Value<X>,+rate: F32,+momentum: F32,+decay: F32,
  +shape: List<&2,U32>,tensor: Tiled.Tiled,velocity: Optimizer.Velocity,-gradient: Train.Gradient,+cleared: Layers.Cleared(gradient),
  +held: Layers.ParameterStepped(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),parameter,moved,rate,momentum,decay,shape,(tensor,velocity))) ->
  Layers.Slot(~Value<X>,~interpret(~X,~x_value),~Layers.Cleared,parameter_descended(~X,parameter,moved,rate,momentum,decay),shape,tensor,gradient,velocity):
  KernelGemm.Both{+contract,+kept} = held
  KernelGemm.Both{wrapped(~X,~x_value,~Leaf.Context<Value<X>,PiecesMap.Context<F32,StepApplied(~Value<X>)>>,
      ~Leaf.value(~F32,~Value<X>,~PiecesMap.Context<F32,StepApplied(~Value<X>)>,~F32.sub,~interpret(~X,~x_value),
        ~PiecesMap.value(~F32,~F32,~StepApplied(~Value<X>),~F.scalar_value(~F32.mul),~step_applied_value(~Value<X>,~interpret(~X,~x_value)))),
      ~(context => Stepped{context}),~(context => index => {==}),
      OptimizerValues.step_context(~Value<X>,~StepApplied(~Value<X>),Layers.parameter_value(~Value<X>,parameter),step_applied(~X,parameter,moved,momentum,decay),rate),
      shape,tensor,contract),
    KernelGemm.Both{cleared,kept_wrapped(~X,~x_value,~StepApplied(~Value<X>),~step_applied_value(~Value<X>,~interpret(~X,~x_value)),~(context => Velocity{context}),
      ~(context => index => {==}),step_applied(~X,parameter,moved,momentum,decay),shape,Bool.not(F32.is_eq(momentum,0.0)),velocity,kept)}}

# A step of a convolution with gradients, as the network states it.
def convolution_descended(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+rate: F32,+momentum: F32,+decay: F32,+mode: Call.Mode,+channels: U32,
  +outputs: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+weights: Layers.Parameter<Value<X>>,+bias: Layers.Parameter<Value<X>>,
  +first: Value<X>,+second: Value<X>,+shape: List<&2,U32>,+fits: ConvolutionFits<shape,mode,channels,outputs,height,width,kernel,stride,padding>,result: Train.Layer,
  +held: TrainingConvolution.ConvolutionStepped(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),channels,outputs,kernel,
    stride,padding,weights,bias,first,second,rate,momentum,decay,result)) ->
  LayerDescribed(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,parameter_descended(~X,weights,first,rate,momentum,decay),
    parameter_descended(~X,bias,second,rate,momentum,decay),Ungraded{}},shape,result):
  match result:
    case Train.Convolution{weight_tensor,bias_tensor,+layer_stride,+layer_padding,weight_gradient,bias_gradient,weight_velocity,bias_velocity}:
      KernelGemm.Both{+stride_exact,KernelGemm.Both{+padding_exact,KernelGemm.Both{+weight_cleared,KernelGemm.Both{+bias_cleared,
        KernelGemm.Both{+weight_stepped,+bias_stepped}}}}} = held
      KernelGemm.Both{fits,KernelGemm.Both{stride_exact,KernelGemm.Both{padding_exact,KernelGemm.Both{
        parameter_slot(~X,~x_value,weights,first,rate,momentum,decay,[outputs,channels,kernel,kernel],weight_tensor,weight_velocity,weight_gradient,weight_cleared,weight_stepped),
        parameter_slot(~X,~x_value,bias,second,rate,momentum,decay,[outputs],bias_tensor,bias_velocity,bias_gradient,bias_cleared,bias_stepped)}}}}
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,parameter_descended(~X,weights,first,rate,momentum,decay),parameter_descended(~X,bias,second,rate,momentum,decay),Ungraded{}},shape,Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,parameter_descended(~X,weights,first,rate,momentum,decay),parameter_descended(~X,bias,second,rate,momentum,decay),Ungraded{}},shape,Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,parameter_descended(~X,weights,first,rate,momentum,decay),parameter_descended(~X,bias,second,rate,momentum,decay),Ungraded{}},shape,Train.Activation{other_activation}),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,ConvolutionOf{mode,channels,outputs,height,width,kernel,stride,padding,parameter_descended(~X,weights,first,rate,momentum,decay),parameter_descended(~X,bias,second,rate,momentum,decay),Ungraded{}},shape,Train.AveragePool{}),Words.false_true(held))

# A step of a batch normalization with gradients, as the network states it.
def norm_descended(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+rate: F32,+momentum: F32,+decay: F32,+images: U32,+channels: U32,+height: U32,
  +width: U32,+epsilon: F32,+norm_momentum: F32,+gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,+mean: Value<X>,+variance: Value<X>,
  +first: Value<X>,+second: Value<X>,+shape: List<&2,U32>,+fits: NormFits<shape,images,channels,height,width>,result: Train.Layer,
  +held: Layers.NormStepped(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),
    channels,epsilon,norm_momentum,gamma,beta,mean,variance,first,second,rate,momentum,decay,result)) ->
  LayerDescribed(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,norm_momentum,parameter_descended(~X,gamma,first,rate,momentum,decay),
    parameter_descended(~X,beta,second,rate,momentum,decay),mean,variance,Ungraded{}},shape,result):
  match result:
    case Train.BatchNorm{gammas,betas,means,variances,+layer_epsilon,+layer_momentum,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{+momentum_exact,KernelGemm.Both{+gamma_cleared,KernelGemm.Both{+beta_cleared,
        KernelGemm.Both{+gamma_stepped,KernelGemm.Both{+beta_stepped,KernelGemm.Both{+mean_contract,+variance_contract}}}}}}} = held
      KernelGemm.Both{fits,KernelGemm.Both{epsilon_exact,KernelGemm.Both{momentum_exact,KernelGemm.Both{
        parameter_slot(~X,~x_value,gamma,first,rate,momentum,decay,[channels],gammas,gamma_velocity,gamma_gradient,gamma_cleared,gamma_stepped),
        KernelGemm.Both{parameter_slot(~X,~x_value,beta,second,rate,momentum,decay,[channels],betas,beta_velocity,beta_gradient,beta_cleared,beta_stepped),
          KernelGemm.Both{mean_contract,variance_contract}}}}}}
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,norm_momentum,parameter_descended(~X,gamma,first,rate,momentum,decay),parameter_descended(~X,beta,second,rate,momentum,decay),mean,variance,Ungraded{}},shape,Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}),Words.false_true(held))
    case Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,norm_momentum,parameter_descended(~X,gamma,first,rate,momentum,decay),parameter_descended(~X,beta,second,rate,momentum,decay),mean,variance,Ungraded{}},shape,Train.LayerNorm{other_gamma,other_beta,other_epsilon,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,norm_momentum,parameter_descended(~X,gamma,first,rate,momentum,decay),parameter_descended(~X,beta,second,rate,momentum,decay),mean,variance,Ungraded{}},shape,Train.Activation{other_activation}),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,BatchNormOf{images,channels,height,width,epsilon,norm_momentum,parameter_descended(~X,gamma,first,rate,momentum,decay),parameter_descended(~X,beta,second,rate,momentum,decay),mean,variance,Ungraded{}},shape,Train.AveragePool{}),Words.false_true(held))

# A step of a layer normalization with gradients, as the network states it.
def line_descended(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+rate: F32,+momentum: F32,+decay: F32,+features: U32,+epsilon: F32,
  +laxes: List<&2,Views.Axis>,+loffset: U32,+raxes: List<&2,Views.Axis>,+roffset: U32,+gaxes: List<&2,Views.Axis>,+goffset: U32,
  +gamma: Layers.Parameter<Value<X>>,+beta: Layers.Parameter<Value<X>>,+first: Value<X>,+second: Value<X>,+shape: List<&2,U32>,
  +fits: LineFits<limits,shape,features,laxes,loffset,raxes,roffset,gaxes,goffset>,result: Train.Layer,
  +held: Layers.LineStepped(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),features,epsilon,gamma,beta,first,second,
    rate,momentum,decay,result)) ->
  LayerDescribed(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,parameter_descended(~X,gamma,first,rate,momentum,decay),
    parameter_descended(~X,beta,second,rate,momentum,decay),Ungraded{}},shape,result):
  match result:
    case Train.LayerNorm{gammas,betas,+layer_epsilon,gamma_gradient,beta_gradient,gamma_velocity,beta_velocity}:
      KernelGemm.Both{+epsilon_exact,KernelGemm.Both{+gamma_cleared,KernelGemm.Both{+beta_cleared,KernelGemm.Both{+gamma_stepped,+beta_stepped}}}} = held
      KernelGemm.Both{fits,KernelGemm.Both{epsilon_exact,KernelGemm.Both{
        parameter_slot(~X,~x_value,gamma,first,rate,momentum,decay,[features],gammas,gamma_velocity,gamma_gradient,gamma_cleared,gamma_stepped),
        parameter_slot(~X,~x_value,beta,second,rate,momentum,decay,[features],betas,beta_velocity,beta_gradient,beta_cleared,beta_stepped)}}}
    case Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,parameter_descended(~X,gamma,first,rate,momentum,decay),parameter_descended(~X,beta,second,rate,momentum,decay),Ungraded{}},shape,Train.Convolution{other_weights,other_bias,other_stride,other_padding,other_weight_gradient,other_bias_gradient,other_weight_velocity,other_bias_velocity}),Words.false_true(held))
    case Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,parameter_descended(~X,gamma,first,rate,momentum,decay),parameter_descended(~X,beta,second,rate,momentum,decay),Ungraded{}},shape,Train.BatchNorm{other_gamma,other_beta,other_mean,other_variance,other_epsilon,other_momentum,other_gamma_gradient,other_beta_gradient,other_gamma_velocity,other_beta_velocity}),Words.false_true(held))
    case Train.Activation{other_activation}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,parameter_descended(~X,gamma,first,rate,momentum,decay),parameter_descended(~X,beta,second,rate,momentum,decay),Ungraded{}},shape,Train.Activation{other_activation}),Words.false_true(held))
    case Train.AveragePool{}:
      Empty.absurd(LayerDescribed(~X,~x_value,limits,LayerNormOf{features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,parameter_descended(~X,gamma,first,rate,momentum,decay),parameter_descended(~X,beta,second,rate,momentum,decay),Ungraded{}},shape,Train.AveragePool{}),Words.false_true(held))

# An activation and the average pool have nothing to step.
def unchanged_stepped(-layer: Train.Layer,-fixed: Train.Layer,+same: {layer == fixed : Train.Layer},+descent: Train.Descent,
  +kept: {Train.layer_stepped(fixed,descent) == fixed : Train.Layer}) -> {Train.layer_stepped(layer,descent) == fixed : Train.Layer}:
  %Equal.sym(Train.Layer,layer,fixed,same) : {Train.layer_stepped(_,descent) == fixed : Train.Layer}
  kept

# A step of SGD of a layer of the description on a layer of a model.
def layer_step(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rate: F32,+momentum: F32,+decay: F32,+values: LayerValues<X>,+shape: List<&2,U32>,+lm: Layers.LayerModel,
  +described: LayerDescribed(~X,~x_value,limits,values,shape,Layers.layer_of(lm))) ->
  LayerDescribed(~X,~x_value,limits,layer_descended(~X,values,rate,momentum,decay),shape,Train.layer_stepped(Layers.layer_of(lm),Train.Descent{rate,momentum,decay})):
  match values:
    case ConvolutionOf{+mode,+channels,+outputs,+height,+width,+kernel,+stride,+padding,+weights,+bias,+grading}:
      match grading:
        case Ungraded{}:
          KernelGemm.Both{+fits,+layered} = described
          KernelGemm.Both{fits,TrainingConvolution.convolution_step_cleared(~Value<X>,~interpret(~X,~x_value),channels,outputs,kernel,stride,padding,weights,bias,
            rate,momentum,decay,Layers.layer_of(lm),layered)}
        case Graded{+first,+second}:
          KernelGemm.Both{+fits,+layered} = described
          ConvolutionFits{+same,+dimensions,+machine,+images_positive,+checks,+gradients_valid,+weights_nonempty,+bias_nonempty} = fits
          convolution_descended(~X,~x_value,limits,rate,momentum,decay,mode,channels,outputs,height,width,kernel,stride,padding,weights,bias,first,second,shape,fits,
            Train.sgd_step_with(Layers.layer_of(lm),rate,momentum,decay),
            TrainingConvolution.convolution_step(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),channels,outputs,
              kernel,stride,padding,weights,bias,first,second,rate,momentum,decay,limits,default_capacity,weights_nonempty,bias_nonempty,Layers.layer_of(lm),layered))
    case BatchNormOf{+images,+channels,+height,+width,+epsilon,+norm_momentum,+gamma,+beta,+mean,+variance,+grading}:
      match grading:
        case Ungraded{}:
          KernelGemm.Both{+fits,+layered} = described
          KernelGemm.Both{fits,Layers.norm_step_cleared(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),channels,epsilon,norm_momentum,gamma,beta,mean,variance,
            rate,momentum,decay,Layers.layer_of(lm),layered)}
        case Graded{+first,+second}:
          KernelGemm.Both{+fits,+layered} = described
          NormFits{+same,+room,+channels_nonempty} = fits
          norm_descended(~X,~x_value,limits,rate,momentum,decay,images,channels,height,width,epsilon,norm_momentum,gamma,beta,mean,variance,first,second,shape,fits,
            Train.sgd_step_with(Layers.layer_of(lm),rate,momentum,decay),
            Layers.norm_step(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),
              channels,epsilon,norm_momentum,gamma,beta,mean,variance,first,second,rate,momentum,decay,limits,default_capacity,channels_nonempty,Layers.layer_of(lm),layered))
    case LayerNormOf{+features,+epsilon,+laxes,+loffset,+raxes,+roffset,+gaxes,+goffset,+gamma,+beta,+grading}:
      match grading:
        case Ungraded{}:
          KernelGemm.Both{+fits,+layered} = described
          KernelGemm.Both{fits,Layers.line_step_cleared(~Value<X>,~interpret(~X,~x_value),features,epsilon,gamma,beta,rate,momentum,decay,Layers.layer_of(lm),layered)}
        case Graded{+first,+second}:
          KernelGemm.Both{+fits,+layered} = described
          LineFits{+facts,+features_nonempty} = fits
          line_descended(~X,~x_value,limits,rate,momentum,decay,features,epsilon,laxes,loffset,raxes,roffset,gaxes,goffset,gamma,beta,first,second,shape,fits,
            Train.sgd_step_with(Layers.layer_of(lm),rate,momentum,decay),
            Layers.line_step(~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),~Value<X>,~interpret(~X,~x_value),features,epsilon,gamma,beta,first,second,
              rate,momentum,decay,limits,default_capacity,features_nonempty,Layers.layer_of(lm),layered))
    case ActivationOf{+activation}:
      KernelGemm.Both{+fits,+layered} = described
      KernelGemm.Both{fits,unchanged_stepped(Layers.layer_of(lm),Train.Activation{activation},layered,Train.Descent{rate,momentum,decay},{==})}
    case PoolOf{+images,+channels,+height,+width,+axes,+offset}:
      KernelGemm.Both{+fits,+layered} = described
      KernelGemm.Both{fits,unchanged_stepped(Layers.layer_of(lm),Train.AveragePool{},layered,Train.Descent{rate,momentum,decay},{==})}

# A step of SGD of a layer, on the layer as a model.
def layer_stepped_seen(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},+rate: F32,+momentum: F32,+decay: F32,+values: LayerValues<X>,
  +shape: List<&2,U32>,-layer: Train.Layer,+layer_observed: Layers.LayerObserved<layer>,+described: LayerDescribed(~X,~x_value,limits,values,shape,layer)) ->
  LayerDescribed(~X,~x_value,limits,layer_descended(~X,values,rate,momentum,decay),shape,Train.layer_stepped(layer,Train.Descent{rate,momentum,decay})):
  Layers.LayerObserved{+lm,+l_equation} = layer_observed
  %l_equation : LayerDescribed(~X,~x_value,limits,layer_descended(~X,values,rate,momentum,decay),shape,Train.layer_stepped(_,Train.Descent{rate,momentum,decay}))
  layer_step(~X,~x_value,limits,default_capacity,rate,momentum,decay,values,shape,lm,
    (%Equal.sym(Train.Layer,Layers.layer_of(lm),layer,l_equation) : LayerDescribed(~X,~x_value,limits,values,shape,_) described))

# A block after a step gives the extents it gave.
def block_descended_fits(~X: Data,+limits: Machine.Limits,+rate: F32,+momentum: F32,+decay: F32,+branch: NetworkValues<X>,+shortcut: NetworkValues<X>,
  +shape: List<&2,U32>,+fits: BlockFits(~X,limits,branch,shortcut,shape)) ->
  BlockFits(~X,limits,network_descended(~X,branch,rate,momentum,decay),network_descended(~X,shortcut,rate,momentum,decay),shape):
  KernelGemm.Both{+same,KernelGemm.Both{+branch_spread,+input_spread}} = fits
  KernelGemm.Both{
    Equal.trans(List<&2,U32>,network_shape(~X,network_descended(~X,branch,rate,momentum,decay),shape),network_shape(~X,branch,shape),
      network_shape(~X,network_descended(~X,shortcut,rate,momentum,decay),shape),network_descended_shape(~X,branch,rate,momentum,decay,shape),
      Equal.trans(List<&2,U32>,network_shape(~X,branch,shape),network_shape(~X,shortcut,shape),network_shape(~X,network_descended(~X,shortcut,rate,momentum,decay),shape),
        same,Equal.sym(List<&2,U32>,network_shape(~X,network_descended(~X,shortcut,rate,momentum,decay),shape),network_shape(~X,shortcut,shape),
          network_descended_shape(~X,shortcut,rate,momentum,decay,shape)))),
    KernelGemm.Both{
      (%Equal.sym(List<&2,U32>,network_shape(~X,network_descended(~X,branch,rate,momentum,decay),shape),network_shape(~X,branch,shape),
          network_descended_shape(~X,branch,rate,momentum,decay,shape)) :
        Spread<limits,_> branch_spread),input_spread}}

# A step of SGD of every layer of a network.
def sgd_holds(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rate: F32,+momentum: F32,+decay: F32,+description: NetworkValues<X>,+shape: List<&2,U32>,network: Train.Network,
  +described: Described(~X,~x_value,limits,description,shape,network)) ->
  Described(~X,~x_value,limits,network_descended(~X,description,rate,momentum,decay),shape,Train.network_stepped(network,Train.Descent{rate,momentum,decay})):
  match description:
    case Last{}:
      match network:
        case Train.End{}: {==}
        case Train.Then{other_layer,other_rest}:
          Empty.absurd(Described(~X,~x_value,limits,Last{},shape,Train.network_stepped(Train.Then{other_layer,other_rest},Train.Descent{rate,momentum,decay})),Words.false_true(described))
        case Train.Residual{other_branch,other_shortcut,other_rest}:
          Empty.absurd(Described(~X,~x_value,limits,Last{},shape,Train.network_stepped(Train.Residual{other_branch,other_shortcut,other_rest},Train.Descent{rate,momentum,decay})),Words.false_true(described))
    case Next{+values,+rest}:
      match network:
        case Train.Then{layer,others}:
          KernelGemm.Both{+layer_described,+rest_described} = described
          KernelGemm.Both{layer_stepped_seen(~X,~x_value,limits,default_capacity,rate,momentum,decay,values,shape,layer,Layers.observe_layer(layer),layer_described),
            (%Equal.sym(List<&2,U32>,layer_shape(~X,layer_descended(~X,values,rate,momentum,decay),shape),layer_shape(~X,values,shape),
                descended_shape(~X,values,rate,momentum,decay,shape)) :
              Described(~X,~x_value,limits,network_descended(~X,rest,rate,momentum,decay),_,Train.network_stepped(others,Train.Descent{rate,momentum,decay}))
              sgd_holds(~X,~x_value,limits,default_capacity,rate,momentum,decay,rest,layer_shape(~X,values,shape),others,rest_described))}
        case Train.End{}:
          Empty.absurd(Described(~X,~x_value,limits,network_descended(~X,Next{values,rest},rate,momentum,decay),shape,Train.network_stepped(Train.End{},Train.Descent{rate,momentum,decay})),Words.false_true(described))
        case Train.Residual{other_branch,other_shortcut,other_rest}:
          Empty.absurd(Described(~X,~x_value,limits,network_descended(~X,Next{values,rest},rate,momentum,decay),shape,Train.network_stepped(Train.Residual{other_branch,other_shortcut,other_rest},Train.Descent{rate,momentum,decay})),Words.false_true(described))
    case Block{+branch,+shortcut,+rest}:
      match network:
        case Train.Residual{branch_network,shortcut_network,others}:
          KernelGemm.Both{+fits,KernelGemm.Both{+branch_described,KernelGemm.Both{+shortcut_described,+rest_described}}} = described
          KernelGemm.Both{block_descended_fits(~X,limits,rate,momentum,decay,branch,shortcut,shape,fits),
            KernelGemm.Both{sgd_holds(~X,~x_value,limits,default_capacity,rate,momentum,decay,branch,shape,branch_network,branch_described),
              KernelGemm.Both{sgd_holds(~X,~x_value,limits,default_capacity,rate,momentum,decay,shortcut,shape,shortcut_network,shortcut_described),
                (%Equal.sym(List<&2,U32>,network_shape(~X,network_descended(~X,branch,rate,momentum,decay),shape),network_shape(~X,branch,shape),
                    network_descended_shape(~X,branch,rate,momentum,decay,shape)) :
                  Described(~X,~x_value,limits,network_descended(~X,rest,rate,momentum,decay),_,Train.network_stepped(others,Train.Descent{rate,momentum,decay}))
                  sgd_holds(~X,~x_value,limits,default_capacity,rate,momentum,decay,rest,network_shape(~X,branch,shape),others,rest_described))}}}
        case Train.End{}:
          Empty.absurd(Described(~X,~x_value,limits,network_descended(~X,Block{branch,shortcut,rest},rate,momentum,decay),shape,Train.network_stepped(Train.End{},Train.Descent{rate,momentum,decay})),Words.false_true(described))
        case Train.Then{other_layer,other_rest}:
          Empty.absurd(Described(~X,~x_value,limits,network_descended(~X,Block{branch,shortcut,rest},rate,momentum,decay),shape,Train.network_stepped(Train.Then{other_layer,other_rest},Train.Descent{rate,momentum,decay})),Words.false_true(described))

# sgd_network_with of a network the description describes: every layer with
# gradients steps, and the network holds the description after the step.
def sgd(~X: Data,~x_value: X -> U32 -> F32,+limits: Machine.Limits,+default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rate: F32,+momentum: F32,+decay: F32,+description: NetworkValues<X>,+shape: List<&2,U32>,network: Train.Network,
  +described: Described(~X,~x_value,limits,description,shape,network)) ->
  Described(~X,~x_value,limits,network_descended(~X,description,rate,momentum,decay),shape,Train.sgd_network_with(network,rate,momentum,decay)):
  sgd_holds(~X,~x_value,limits,default_capacity,rate,momentum,decay,description,shape,network,described)
