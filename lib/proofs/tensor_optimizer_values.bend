# An SGD step composes the scaling of the gradient by the rate and the
# subtraction from the parameter: every element is parameter - gradient * rate
# in FP32, for a parameter and a gradient of one shape in any layout.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_band_schedule.bend as Schedule
import ../tensor_operations.bend as Operations
import ../tensor_optimizer.bend as Optimizer
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_model.bend as Model
import ./tensor_band_map.bend as BandMap
import ./tensor_band_normal_unary.bend as NormalUnary
import ./tensor_band_combine_leaf.bend as Leaf
import ./tensor_combine_public.bend as CombinePublic
import ./tensor_operations_refinement.bend as OperationsLaws

# The parameter, and the rate with the gradient.
def step_context(~Parameter: Data,~Gradient: Data,+parameter: Parameter,+gradient: Gradient,+rate: F32) -> Leaf.Context<Parameter,BandMap.Context<F32,Gradient>>:
  Leaf.Context{parameter,BandMap.Context{rate,gradient}}

# The reader of the result at every index: the parameter minus the product of
# the gradient and the rate.
def stepped(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+parameter: Parameter,+gradient: Gradient,+rate: F32,+index: U32) ->
  {Leaf.value(~F32,~Parameter,~BandMap.Context<F32,Gradient>,~F32.sub,~parameter_value,~BandMap.value(~F32,~F32,~Gradient,~Bands.scalar_value(~F32,~F32.mul),~gradient_value),
      step_context(~Parameter,~Gradient,parameter,gradient,rate),index) ==
    F32.sub(parameter_value(parameter,index),F32.mul(gradient_value(gradient,index),rate)) : F32}:
  {==}

# A parameter and a gradient of one shape, each in any layout and holding the
# values of its own reader: the result of sgd is Normal and holds the value
# of stepped at every index.
def sgd(~Parameter: Data,~parameter_value: Parameter -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+parameter: Parameter,+gradient: Gradient,+rate: F32,+shape: List<&2,U32>,
  +parameter_bands: Model.Model<F32>,+parameter_geometry: Bands.Geometry,+parameter_fill: F32,+parameter_workers: U32,
  +gradient_bands: Model.Model<F32>,+gradient_geometry: Bands.Geometry,+gradient_fill: F32,+gradient_workers: U32,
  parameter_contract: Global.Contract(~F32,~Parameter,~parameter_value,parameter,
    Tensors.Tensor{Model.pack(~F32,parameter_bands),parameter_geometry,shape,parameter_fill,Tensors.Usable{},parameter_workers}),
  gradient_contract: Global.Contract(~F32,~Gradient,~gradient_value,gradient,
    Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers})) ->
  Global.Contract(~F32,~Leaf.Context<Parameter,BandMap.Context<F32,Gradient>>,
    ~Leaf.value(~F32,~Parameter,~BandMap.Context<F32,Gradient>,~F32.sub,~parameter_value,~BandMap.value(~F32,~F32,~Gradient,~Bands.scalar_value(~F32,~F32.mul),~gradient_value)),
    step_context(~Parameter,~Gradient,parameter,gradient,rate),
    Optimizer.sgd(Tensors.Tensor{Model.pack(~F32,parameter_bands),parameter_geometry,shape,parameter_fill,Tensors.Usable{},parameter_workers},
      Tensors.Tensor{Model.pack(~F32,gradient_bands),gradient_geometry,shape,gradient_fill,Tensors.Usable{},gradient_workers},rate)):
  CombinePublic.combine(~F32,~Parameter,~BandMap.Context<F32,Gradient>,~F32.sub,~Operations.combine_kernel(~F32,~F32.sub),~OperationsLaws.scalar_kernel(~F32,~F32.sub),
    ~parameter_value,~BandMap.value(~F32,~F32,~Gradient,~Bands.scalar_value(~F32,~F32.mul),~gradient_value),parameter,BandMap.Context{rate,gradient},"sub",1.0,shape,
    parameter_geometry,gradient_geometry,parameter_fill,gradient_fill,parameter_workers,gradient_workers,
    Model.pack(~F32,parameter_bands),Schedule.scalar(~F32,~F32.mul,gradient_geometry,gradient_workers,1.0,rate,Model.pack(~F32,gradient_bands)),
    parameter_contract,
    NormalUnary.combine_scalar(~F32,~Gradient,~F32.mul,~gradient_value,gradient,1.0,rate,gradient_bands,gradient_geometry,shape,gradient_fill,gradient_workers,gradient_contract))
