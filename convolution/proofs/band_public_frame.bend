# The public convolution frame: the native record, its checked guards and the
# reference context.
import Base
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_tensor_execution.bend as TensorExecution
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./band_leaf_caller.bend as Leaf
import ../specification.bend as ConvSpec
import ../../lib/machine_limits.bend as Machine
import ../../lib/math_activation.bend as Activations
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_model.bend as Model
import ./band_caller_guards.bend as Caller
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_leaf_values.bend as LeafValues
import ./band_native_source_domain.bend as BandNativeSourceDomain
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ./band_frame_configuration.bend as Configuration
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/tensor_band_map.bend as BandMap

# Public native dimensions and the validation and support guards construct the
# convolution frame. The input record keeps the complete machine predicate
# opaque until its generic machine-limits consumer inspects the dimensions.
type NativeInput is Data:
  NativeInput{input_images: U32,batched: Bool,input_channels: U32,output_channels: U32,input_height: U32,input_width: U32,kernel: U32,stride: U32,padding: U32}

# The images of the record: its count for NCHW extents, one for CHW extents,
# as TensorConvolution.input_shape reads them.
def image_count(batched: Bool,count: U32) -> U32:
  match batched:
    case True{}: count
    case False{}: 1

def input_images(input: NativeInput) -> U32:
  NativeInput{input_images,batched,ci,co,h,w,k,s,p} = input
  image_count(batched,input_images)

def frame_dimensions(input: NativeInput) -> KernelShape.Dimensions:
  NativeInput{input_images,batched,ci,co,h,w,k,s,p} = input
  BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p)

def geometry(input: NativeInput) -> TensorConvolution.Geometry:
  NativeInput{input_images,+batched,ci,co,h,w,k,s,p} = input
  TensorConvolution.Geometry{image_count(batched,input_images),batched,BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p)}

def source_geometry(input: NativeInput) -> Bands.Geometry:
  NativeInput{input_images,batched,ci,co,h,w,k,s,p} = input
  Bands.Geometry{image_count(batched,input_images),ci,h,w}

def frame_valid(input: NativeInput,weights: List<&2,F32>,bias: List<&2,F32>) -> Bool:
  NativeInput{input_images,batched,ci,co,h,w,k,s,p} = input
  +shape = BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p)
  KernelShape.valid(shape,Caller.input_length(shape),ConvSpec.length(weights),ConvSpec.length(bias))

def machine(input: NativeInput,+limits: Machine.Limits) -> Bool:
  NativeInput{+input_images,batched,ci,co,h,w,k,s,p} = input
  +shape = BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p)
  KernelShape.supports_packed(limits,shape,Caller.input_length(shape),Configuration.weight_size(shape)) &&
    Nat.is_le(LeafValues.output_elements(image_count(batched,input_images),shape),U32.to_nat(Machine.capacity(limits)))

def configured_native(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: NativeInput,+parallel: Bool,+limits: Machine.Limits,+capacity: U32,
  +weights: List<&2,F32>,+bias: List<&2,F32>,domain: BandGather.Domain<Leaf.input_geometry(input_images(input),frame_dimensions(input)),limits,capacity>,
  weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  frame_accepted: {frame_valid(input,weights,bias) == True{} : Bool},supported: {machine(input,limits) == True{} : Bool}) -> PipelineExecution.Frame<Logical>:
  NativeInput{+input_images,batched,+ci,+co,+h,+w,+k,+s,+p} = input
  Configuration.configured(~Logical,~original,logical,activation,image_count(batched,input_images),BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p),parallel,limits,capacity,weights,bias,
    domain,weight_evidence,bias_evidence,frame_accepted,supported)

# Public frame construction retains the incoming logical context and parameter
# sequences. Inspect native dimensions only inside the generic configuration
# proof; the actual Default caller keeps the input record opaque.
def frame_images(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> U32:
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  images

def bias_content(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Buffer.Content<F32>:
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,be),ConvSpec.length(bias)}

def reference_context(~Logical: Data,logical: Logical,activation: Activations.Activation,input: NativeInput,weights: List<&2,F32>,bias: List<&2,F32>) -> BandConvolutionValues.Context<Logical>:
  NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  BandConvolutionValues.Context{logical,activation,U32.to_nat(ci),U32.to_nat(co),U32.to_nat(h),U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p),weights,bias}

type Fields<-Logical: Data,-logical: Logical,-activation: Activations.Activation,-input: NativeInput,-weights: List<&2,F32>,-bias: List<&2,F32>,
  -weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,-bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,-frame: PipelineExecution.Frame<Logical>,-limits: Machine.Limits,-capacity: U32> is Data:
  Fields{logical_exact: {PipelineExecution.convolution_logical(~Logical,frame) == logical : Logical},
    activation_exact: {TensorExecution.execution_activation(~Logical,frame) == activation : Activations.Activation},
    images_exact: {frame_images(~Logical,frame) == input_images(input) : U32},
    dimensions_exact: {PipelineExecution.dimensions(~Logical,frame) == frame_dimensions(input) : KernelShape.Dimensions},
    input_geometry_exact: {PipelineExecution.input_geometry(~Logical,frame) == source_geometry(input) : Bands.Geometry},
    weights_exact: {PipelineExecution.source_tree(~Logical,frame) == StorageLaws.sequence_storage(F32,ConvSpec.length(weights),weights,weight_evidence) : Tree.Tree<F32>},
    bias_exact: {bias_content(~Logical,frame) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>},
    context_exact: {PipelineExecution.output_context(~Logical,frame) == reference_context(~Logical,logical,activation,input,weights,bias) : BandConvolutionValues.Context<Logical>},
    limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == limits : Machine.Limits},
    capacity_exact: {PipelineExecution.convolution_capacity(~Logical,frame) == capacity : U32}}

def configured(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: NativeInput,+parallel: Bool,+limits: Machine.Limits,+capacity: U32,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+domain: BandGather.Domain<Leaf.input_geometry(input_images(input),frame_dimensions(input)),limits,capacity>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +validated: {frame_valid(input,weights,bias) == True{} : Bool},+supported: {machine(input,limits) == True{} : Bool}) ->
  Fields<Logical,logical,activation,input,weights,bias,weight_evidence,bias_evidence,
    configured_native(~Logical,~original,logical,activation,input,parallel,limits,capacity,weights,bias,domain,weight_evidence,bias_evidence,validated,supported),limits,capacity>:
  NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  Fields{{==},{==},{==},{==},BandNativeSourceDomain.geometry_roundtrip(image_count(batched,images),ci,co,h,w,k,s,p),{==},{==},{==},{==},{==}}

def settings_bias_matches(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>) ->
  {PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) == bias_content(~Logical,frame) : Buffer.Content<F32>}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  {==}

def settings_bias(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,+frame: PipelineExecution.Frame<Logical>,+limits: Machine.Limits,+capacity: U32,
  fields: Fields<Logical,logical,activation,input,weights,bias,weight_evidence,bias_evidence,frame,limits,capacity>) ->
  {PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)} : Buffer.Content<F32>}:
  Fields{logical_exact,activation_exact,images_exact,dimensions_exact,input_geometry_exact,weights_exact,bias_exact,context_exact,limits_exact,capacity_exact} = fields
  Equal.trans(Buffer.Content<F32>,PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),bias_content(~Logical,frame),
    Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias),bias,bias_evidence),ConvSpec.length(bias)},settings_bias_matches(~Logical,~original,frame),bias_exact)

def valid_parameters(dimensions: KernelShape.Dimensions,weights: Buffer.Model<F32>,bias: Buffer.Model<F32>) -> Bool:
  KernelShape.Dimensions{+ci,co,+h,+w,k,s,p} = dimensions
  KernelShape.valid(dimensions,Nat.mul(ci,Nat.mul(h,w)),Buffer.length(~F32,weights),Buffer.length(~F32,bias))

def allocation_room(~Element: Data,+model: Buffer.Model<Element>,ready: Buffer.Sized<Element,model>) ->
  {Nat.is_le(Buffer.length(~Element,model),U32.to_nat(U32.shln(1,Buffer.depth(~Element,model)))) == True{} : Bool}:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,equation}} = model
  Buffer.Sized{depth_ok,room} = ready
  %Shape.capacity(Element,depth,tree,Buffer.balanced(~Element,model)) : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool}
  room

def weight_length(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Nat:
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  ConvSpec.length(weights)

def bias_length(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Nat:
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  ConvSpec.length(bias)

def parameters(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  weight_exact: {Buffer.length(~F32,weights) == weight_length(~Logical,frame) : Nat},
  bias_exact: {Buffer.length(~F32,bias) == bias_length(~Logical,frame) : Nat}) ->
  {valid_parameters(PipelineExecution.dimensions(~Logical,frame),weights,bias) == True{} : Bool}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{+ci,co,+h,+w,k,s,p},images,parallel,limits,capacity,+weight_list,+bias_list,we,be,domain,valid,supported,room} = frame
  %Equal.sym(Nat,Buffer.length(~F32,weights),ConvSpec.length(weight_list),weight_exact) :
    {KernelShape.valid(KernelShape.Dimensions{ci,co,h,w,k,s,p},Nat.mul(ci,Nat.mul(h,w)),_,Buffer.length(~F32,bias)) == True{} : Bool}
  %Equal.sym(Nat,Buffer.length(~F32,bias),ConvSpec.length(bias_list),bias_exact) :
    {KernelShape.valid(KernelShape.Dimensions{ci,co,h,w,k,s,p},Nat.mul(ci,Nat.mul(h,w)),ConvSpec.length(weight_list),_) == True{} : Bool}
  valid

# Public caller results use the ordered convolution specification context.
# Planner parallelism changes ownership scheduling while preserving this value.
def Contract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,tensor: Tensors.Tensor<F32>) -> Data:
  Shape.Both<BandMap.LogicalValues(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,tensor),BandMap.WellFormed(~F32,tensor)>

def Related(~Logical: Data,frame: PipelineExecution.Frame<Logical>,context: BandConvolutionValues.Context<Logical>) -> Data:
  {PipelineExecution.output_context(~Logical,frame) == context : BandConvolutionValues.Context<Logical>}

def frame_value(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+index: U32) ->
  {PipelineExecution.result_value(~custom,~Logical,~original,frame,index) == BandConvolutionValues.value(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),index) : F32}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room} = frame
  {==}

def agreement(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+context: BandConvolutionValues.Context<Logical>,+related: Related(~Logical,frame,context),+index: U32) ->
  {PipelineExecution.result_value(~custom,~Logical,~original,frame,index) == BandConvolutionValues.value(~custom,~Logical,~original,context,index) : F32}:
  Equal.trans(F32,PipelineExecution.result_value(~custom,~Logical,~original,frame,index),BandConvolutionValues.value(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),index),BandConvolutionValues.value(~custom,~Logical,~original,context,index),
    frame_value(~custom,~Logical,~original,frame,index),Equal.cong(BandConvolutionValues.Context<Logical>,F32,value => BandConvolutionValues.value(~custom,~Logical,~original,value,index),PipelineExecution.output_context(~Logical,frame),context,related))

def settings_logical(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>) ->
  {PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) == frame : PipelineExecution.Frame<Logical>}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room} = frame
  {==}

def values(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+context: BandConvolutionValues.Context<Logical>,+related: Related(~Logical,frame,context),tensor: Tensors.Tensor<F32>,
  values: TensorExecution.TensorValues(~custom,~Logical,~original,frame,tensor)) ->
  BandMap.LogicalValues(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,tensor):
  match tensor:
    case Tensors.Tensor{bands,+geometry,shape,fill,status,workers}:
      BandValues.values(~F32,~PipelineExecution.Frame<Logical>,~BandConvolutionValues.Context<Logical>,~PipelineExecution.result_value(~custom,~Logical,~original),~BandConvolutionValues.value(~custom,~Logical,~original),~Related(~Logical),~agreement(~custom,~Logical,~original),
        frame,context,related,geometry,Model.contents(~F32,bands),
        %settings_logical(~Logical,~original,frame) : BandValues.Values(~F32,~PipelineExecution.Frame<Logical>,~PipelineExecution.result_value(~custom,~Logical,~original),_,geometry,Model.contents(~F32,bands))
        values)

def from_frame(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+context: BandConvolutionValues.Context<Logical>,+related: Related(~Logical,frame,context),tensor: Tensors.Tensor<F32>,
  evidence: TensorExecution.OwnersContract(~custom,~Logical,~original,frame,tensor)) -> Contract(~custom,~Logical,~original,context,tensor):
  Shape.Both{frame_values,well_formed} = evidence
  Shape.Both{values(~custom,~Logical,~original,frame,context,related,tensor,frame_values),well_formed}

