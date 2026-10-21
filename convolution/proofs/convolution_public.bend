# The public conv2d. A Normal input of any layout arrives in the layout the
# planner chooses for it, the band pipeline runs there, and every output cell
# holds the ordered convolution value. Weights and bias of any layout are read
# as their whole tensors.
import Base
import ../specification.bend as ConvSpec
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/tensor_band_window.bend as BandWindow
import ../../lib/tensor_band_schedule.bend as BandSchedule
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/kernel_tile.bend as Tile
import ../../lib/kernel_band_planning.bend as BandPlanning
import ../../lib/default_kernel_cost.bend as DefaultCost
import ../../lib/math_activation.bend as Activations
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/storage_certificate.bend as Certificate
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/storage_refinement.bend as StorageLaws
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/proofs/tensor_band_inspection.bend as Inspection
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_count.bend as BandCount
import ../../lib/proofs/tensor_band_from_whole.bend as FromWhole
import ../../lib/proofs/tensor_band_plan.bend as BandPlan
import ../../lib/proofs/tensor_band_target.bend as Target
import ../../lib/proofs/tensor_band_arrive.bend as Arrive
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_whole_leaf.bend as WholeLeaf
import ../../lib/proofs/tensor_gather_values.bend as GatherValues
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_band_concat_execution.bend as BandConcatExecution
import ../../lib/proofs/kernel_band_planning_refinement.bend as Planning
import ../../lib/proofs/kernel_band_planning_observation.bend as Observation
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ../../lib/proofs/bounded_u32_arithmetic.bend as Arithmetic
import ../../lib/proofs/boolean_logic.bend as Boolean
import ./convolution_guard_facts.bend as Guards
import ./band_caller_guards.bend as Caller
import ./band_leaf_caller.bend as Leaf
import ./band_leaf_values.bend as LeafValues
import ./band_frame_configuration.bend as Configuration
import ./band_native_source_domain.bend as BandNativeSourceDomain
import ./band_convolution_values.bend as BandConvolutionValues
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_tensor_execution.bend as TensorExecution
import ./band_output_metadata.bend as OutputMetadata
import ./band_output_normal.bend as BandOutputNormal
import ./band_public_frame.bend as PublicFrame
import ./band_parameter_owners.bend as Parameters

# A conv2d result: Usable in its output shape, with the values and the Normal
# envelope of the contract.
def Convolved(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+shape: List<&2,U32>,-result: Tensors.Tensor<F32>) -> Data:
  Global.Shaped(~F32,~BandConvolutionValues.Context<Logical>,~BandConvolutionValues.value(~custom,~Logical,~original),context,shape,result)

# The values the pipeline stores, with the Normal envelope of the result.
def convolved(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+context: BandConvolutionValues.Context<Logical>,+shape: List<&2,U32>,-result: Tensors.Tensor<F32>,
  usable: Global.Usable<F32,shape,result>,contract: PublicFrame.Contract(~custom,~Logical,~original,context,result),normal: Global.Normal(~F32,result)) ->
  Convolved(~custom,~Logical,~original,context,shape,result):
  Shape.Both{values,well_formed} = contract
  Shape.Both{usable,Shape.Both{normal,values}}

# The output extents of a frame: its images, output channels and output
# rows and columns.
def frame_extents(~Logical: Data,frame: PipelineExecution.Frame<Logical>,batched: Bool) -> List<&2,U32>:
  BandOutputNormal.extents(batched,OutputMetadata.bounds_geometry(~Logical,frame))

def dimension_extents(batched: Bool,images: U32,dimensions: KernelShape.Dimensions) -> List<&2,U32>:
  KernelShape.Dimensions{ci,co,+h,+w,+k,+s,+p} = dimensions
  TensorConvolution.output_shape(batched,images,U32.from_nat(co),U32.from_nat(ConvSpec.axis_size(h,k,s,p)),U32.from_nat(ConvSpec.axis_size(w,k,s,p)))

# The public extents of a native input record.
def batched(input: PublicFrame.NativeInput) -> Bool:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  batched

def input_extents(input: PublicFrame.NativeInput) -> List<&2,U32>:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  TensorConvolution.output_shape(batched,images,ci,h,w)

def weight_extents(input: PublicFrame.NativeInput) -> List<&2,U32>:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,+k,s,p} = input
  [co,ci,k,k]

def bias_extents(input: PublicFrame.NativeInput) -> List<&2,U32>:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  [co]

# The extents of the result of a native input record.
def output_extents(input: PublicFrame.NativeInput) -> List<&2,U32>:
  PublicFrame.NativeInput{images,batched,ci,co,+h,+w,+k,+s,+p} = input
  TensorConvolution.output_shape(batched,images,co,
    U32.from_nat(ConvSpec.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))),
    U32.from_nat(ConvSpec.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p))))

def stride(input: PublicFrame.NativeInput) -> U32:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  s

def padding(input: PublicFrame.NativeInput) -> U32:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  p

def input_shape_value(input: PublicFrame.NativeInput) -> TensorConvolution.InputShape:
  PublicFrame.NativeInput{images,+batched,ci,co,h,w,k,s,p} = input
  TensorConvolution.InputShape{PublicFrame.image_count(batched,images),batched,ci,h,w}

def input_shape_matches(+input: PublicFrame.NativeInput) ->
  {TensorConvolution.input_shape(input_extents(input)) == Some{input_shape_value(input)} : Maybe<&2,TensorConvolution.InputShape>}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  match batched:
    case True{}: {==}
    case False{}: {==}

def parameter_geometry_matches(+input: PublicFrame.NativeInput) ->
  {TensorConvolution.geometry(Some{input_shape_value(input)},weight_extents(input),bias_extents(input),stride(input),padding(input)) == Some{PublicFrame.geometry(input)} : Maybe<&2,TensorConvolution.Geometry>}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  %Equal.sym(Bool,U32.is_eq(ci,ci),True{},BandShape.equal_words(ci,ci,{==})) :
    {TensorConvolution.matched(_ && U32.is_eq(k,k) && U32.is_eq(co,co),TensorConvolution.Geometry{PublicFrame.image_count(batched,images),batched,PublicFrame.frame_dimensions(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})}) ==
      Some{PublicFrame.geometry(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})} : Maybe<&2,TensorConvolution.Geometry>}
  %Equal.sym(Bool,U32.is_eq(k,k),True{},BandShape.equal_words(k,k,{==})) :
    {TensorConvolution.matched(True{} && _ && U32.is_eq(co,co),TensorConvolution.Geometry{PublicFrame.image_count(batched,images),batched,PublicFrame.frame_dimensions(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})}) ==
      Some{PublicFrame.geometry(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})} : Maybe<&2,TensorConvolution.Geometry>}
  %Equal.sym(Bool,U32.is_eq(co,co),True{},BandShape.equal_words(co,co,{==})) :
    {TensorConvolution.matched(True{} && True{} && _,TensorConvolution.Geometry{PublicFrame.image_count(batched,images),batched,PublicFrame.frame_dimensions(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})}) ==
      Some{PublicFrame.geometry(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p})} : Maybe<&2,TensorConvolution.Geometry>}
  {==}

def public_geometry(+input: PublicFrame.NativeInput) ->
  {TensorConvolution.geometry(TensorConvolution.input_shape(input_extents(input)),weight_extents(input),bias_extents(input),stride(input),padding(input)) == Some{PublicFrame.geometry(input)} : Maybe<&2,TensorConvolution.Geometry>}:
  %Equal.sym(Maybe<&2,TensorConvolution.InputShape>,TensorConvolution.input_shape(input_extents(input)),Some{input_shape_value(input)},input_shape_matches(input)) :
    {TensorConvolution.geometry(_,weight_extents(input),bias_extents(input),stride(input),padding(input)) == Some{PublicFrame.geometry(input)} : Maybe<&2,TensorConvolution.Geometry>}
  parameter_geometry_matches(input)

# The public CHW or NCHW extents of an input are its image rows.
def source_rows(+input: PublicFrame.NativeInput) ->
  {Bands.image_rows(input_extents(input),PublicFrame.source_geometry(input)) == True{} : Bool}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  match batched:
    case True{}: BandOutputNormal.coherent(True{},Bands.Geometry{images,ci,h,w},{==})
    case False{}: BandOutputNormal.coherent(False{},Bands.Geometry{1,ci,h,w},{==})

def geometry_parts(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,batched: Bool) ->
  {TensorExecution.execution_geometry(~Logical,frame,batched) == TensorConvolution.Geometry{PublicFrame.frame_images(~Logical,frame),batched,PipelineExecution.dimensions(~Logical,frame)} : TensorConvolution.Geometry}:
  PipelineExecution.Frame{logical,activation,dimensions,images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  {==}

def input_geometry_parts(+input: PublicFrame.NativeInput) ->
  {PublicFrame.geometry(input) == TensorConvolution.Geometry{PublicFrame.input_images(input),batched(input),PublicFrame.frame_dimensions(input)} : TensorConvolution.Geometry}:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  {==}

def geometry_fields(~Logical: Data,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+weights: List<&2,F32>,+bias: List<&2,F32>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,+frame: PipelineExecution.Frame<Logical>,+limits: Machine.Limits,+capacity: U32,
  fields: PublicFrame.Fields<Logical,logical,activation,input,weights,bias,weight_evidence,bias_evidence,frame,limits,capacity>) ->
  {TensorExecution.execution_geometry(~Logical,frame,batched(input)) == PublicFrame.geometry(input) : TensorConvolution.Geometry}:
  PublicFrame.Fields{logical_exact,activation_exact,+images_exact,+dimensions_exact,input_geometry_exact,weights_exact,bias_exact,context_exact,limits_exact,capacity_exact} = fields
  Equal.trans(TensorConvolution.Geometry,TensorExecution.execution_geometry(~Logical,frame,batched(input)),
    TensorConvolution.Geometry{PublicFrame.frame_images(~Logical,frame),batched(input),PipelineExecution.dimensions(~Logical,frame)},PublicFrame.geometry(input),
    geometry_parts(~Logical,frame,batched(input)),
    %Equal.sym(U32,PublicFrame.frame_images(~Logical,frame),PublicFrame.input_images(input),images_exact) :
      {TensorConvolution.Geometry{_,batched(input),PipelineExecution.dimensions(~Logical,frame)} == PublicFrame.geometry(input) : TensorConvolution.Geometry}
    %Equal.sym(KernelShape.Dimensions,PipelineExecution.dimensions(~Logical,frame),PublicFrame.frame_dimensions(input),dimensions_exact) :
      {TensorConvolution.Geometry{PublicFrame.input_images(input),batched(input),_} == PublicFrame.geometry(input) : TensorConvolution.Geometry}
    Equal.sym(TensorConvolution.Geometry,PublicFrame.geometry(input),TensorConvolution.Geometry{PublicFrame.input_images(input),batched(input),PublicFrame.frame_dimensions(input)},input_geometry_parts(input)))

# Keep limits generic when exposing the factory's parameter sequences.
def configured_lengths(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,+parallel: Bool,+limits: Machine.Limits,+capacity: U32,
  +weights: List<&2,F32>,+bias: List<&2,F32>,+domain: BandGather.Domain<Leaf.input_geometry(PublicFrame.input_images(input),PublicFrame.frame_dimensions(input)),limits,capacity>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weights),weights>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias),bias>,
  +validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool},+supported: {PublicFrame.machine(input,limits) == True{} : Bool}) ->
  Shape.Both<
    {PublicFrame.weight_length(~Logical,PublicFrame.configured_native(~Logical,~original,logical,activation,input,parallel,limits,capacity,weights,bias,domain,weight_evidence,bias_evidence,validated,supported)) == ConvSpec.length(weights) : Nat},
    {PublicFrame.bias_length(~Logical,PublicFrame.configured_native(~Logical,~original,logical,activation,input,parallel,limits,capacity,weights,bias,domain,weight_evidence,bias_evidence,validated,supported)) == ConvSpec.length(bias) : Nat}>:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  Shape.Both{{==},{==}}

# The geometry of the bands the convolution reads.
def arrival(~Logical: Data,frame: PipelineExecution.Frame<Logical>,+batched: Bool) -> Bands.Geometry:
  TensorConvolution.input_geometry(TensorExecution.execution_geometry(~Logical,frame,batched))

def arrival_exact(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool) ->
  {arrival(~Logical,frame,batched) == PipelineExecution.input_geometry(~Logical,frame) : Bands.Geometry}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  {==}

# The planner reads the window of the same input geometry.
def window_input(+geometry: TensorConvolution.Geometry,+parallel: Bool,+activation: Activations.Activation) ->
  {BandWindow.input_geometry(TensorConvolution.context_window(TensorConvolution.band_context(geometry,parallel,activation))) == TensorConvolution.input_geometry(geometry) : Bands.Geometry}:
  TensorConvolution.Geometry{count,batched,KernelShape.Dimensions{ci,rows,h,w,k,s,p}} = geometry
  {==}

# What the call checks of its parameters.
def parameters(+geometry: TensorConvolution.Geometry,weights: Buffer.Model<F32>,bias: Buffer.Model<F32>) -> Bool:
  TensorConvolution.Geometry{count,batched,+dimensions} = geometry
  Buffer.Model{weight_tree,+weight_length,Certificate.Certificate{+weight_depth,weight_equation}} = weights
  Buffer.Model{bias_tree,+bias_length,Certificate.Certificate{+bias_depth,bias_equation}} = bias
  TensorConvolution.parameters_valid(dimensions,weight_length,weight_depth,bias_length,bias_depth)

def checked_matches(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: TensorConvolution.Geometry,tensor: Tensors.Tensor<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +shape: List<&2,U32>,fill: F32,workers: U32) ->
  {TensorConvolution.band_checked(~custom,activation,Some{geometry},tensor,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) ==
    TensorConvolution.band_supported(~custom,activation,parameters(geometry,weights,bias),geometry,tensor,Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers) : Tensors.Tensor<F32>}:
  TensorConvolution.Geometry{count,batched,dimensions} = geometry
  Buffer.Model{weight_tree,weight_length,Certificate.Certificate{weight_depth,weight_equation}} = weights
  Buffer.Model{bias_tree,bias_length,Certificate.Certificate{bias_depth,bias_equation}} = bias
  {==}

def parameters_held(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  valid: {PublicFrame.valid_parameters(PipelineExecution.dimensions(~Logical,frame),weights,bias) == True{} : Bool},
  weight_room: {Nat.is_le(Buffer.length(~F32,weights),U32.to_nat(U32.shln(1,Buffer.depth(~F32,weights)))) == True{} : Bool},
  bias_room: {Nat.is_le(Buffer.length(~F32,bias),U32.to_nat(U32.shln(1,Buffer.depth(~F32,bias)))) == True{} : Bool}) ->
  {parameters(TensorExecution.execution_geometry(~Logical,frame,batched),weights,bias) == True{} : Bool}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding},images,parallel,limits,capacity,sw,sb,we,be,domain,frame_valid,supported,room} = frame
  Buffer.Model{weight_tree,+weight_length,Certificate.Certificate{+weight_depth,weight_equation}} = weights
  Buffer.Model{bias_tree,+bias_length,Certificate.Certificate{+bias_depth,bias_equation}} = bias
  +valid_shape = KernelShape.valid(KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},Nat.mul(ci,Nat.mul(height,width)),weight_length,bias_length)
  +weights_fit = Nat.is_le(weight_length,U32.to_nat(U32.shln(1,weight_depth)))
  Boolean.and_true(valid_shape && weights_fit,Nat.is_le(bias_length,U32.to_nat(U32.shln(1,bias_depth))),Boolean.and_true(valid_shape,weights_fit,valid,weight_room),bias_room)

# The checked parameter lengths fit the shape and their allocations.
def parameters_accepted(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  weight_length: {Buffer.length(~F32,weights) == PublicFrame.weight_length(~Logical,frame) : Nat},bias_length: {Buffer.length(~F32,bias) == PublicFrame.bias_length(~Logical,frame) : Nat},
  weight_ready: Buffer.Sized<F32,weights>,bias_ready: Buffer.Sized<F32,bias>) ->
  {parameters(TensorExecution.execution_geometry(~Logical,frame,batched),weights,bias) == True{} : Bool}:
  parameters_held(~Logical,frame,batched,weights,bias,PublicFrame.parameters(~Logical,frame,weights,bias,weight_length,bias_length),
    PublicFrame.allocation_room(~F32,weights,weight_ready),PublicFrame.allocation_room(~F32,bias,bias_ready))

# The frame's checked support and output room are the machine guard.
def machine_accepted(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool) ->
  {TensorConvolution.machine_support(TensorExecution.execution_geometry(~Logical,frame,batched),PipelineExecution.convolution_limits(~Logical,frame)) == True{} : Bool}:
  PipelineExecution.Frame{+logical,activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,parallel,+limits,capacity,+weights,+bias,we,be,domain,+valid,supported,room} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +length_exact = Configuration.weights_length(Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n),
    Guards.valid(Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n),Caller.valid(~Logical,~original,logical,dimensions,weights,bias,0n,valid)))
  %length_exact : {KernelShape.supports_packed(limits,dimensions,Caller.input_length(dimensions),_) &&
    Nat.is_le(LeafValues.output_elements(images,dimensions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  %Equal.sym(Bool,KernelShape.supports_packed(limits,dimensions,Caller.input_length(dimensions),ConvSpec.length(weights)),True{},supported) :
    {_ && Nat.is_le(LeafValues.output_elements(images,dimensions),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  room

def covered_at(+geometry: TensorConvolution.Geometry,inspection: Bands.Inspection) -> Bool:
  TensorConvolution.Geometry{+count,batched,+dimensions} = geometry
  TensorConvolution.covered(dimensions,count,inspection)

def arrived_matches(~custom: F32 -> F32,+activation: Activations.Activation,+geometry: TensorConvolution.Geometry,parallel: Bool,weights: Storage.Buffer<Tile.Tile8>,bias: Storage.Buffer<F32>,
  +shape: List<&2,U32>,fill: F32,workers: U32,input: Bands.Bands(~F32),+inspection: Bands.Inspection) ->
  {TensorConvolution.arrived(~custom,activation,geometry,parallel,weights,bias,shape,fill,workers,(input,inspection)) ==
    TensorConvolution.arrived_covered(~custom,activation,covered_at(geometry,inspection),geometry,parallel,input,weights,bias,shape,fill,workers) : Tensors.Tensor<F32>}:
  TensorConvolution.Geometry{count,batched,dimensions} = geometry
  {==}

def covered_rows(+count: U32,+ci: Nat,+co: Nat,+h: Nat,+w: Nat,+k: Nat,+s: Nat,+p: Nat,+inspection: Bands.Inspection,words: LeafValues.Words<ci,co,h,w,k,s,p>,
  checked: {Inspection.valid(inspection) && U32.is_zero(Inspection.first(inspection)) &&
    Nat.is_eq(U32.to_nat(Inspection.end(inspection)),Inspection.total_rows(Leaf.input_geometry(count,KernelShape.Dimensions{ci,co,h,w,k,s,p}))) == True{} : Bool}) ->
  {TensorConvolution.covered(KernelShape.Dimensions{ci,co,h,w,k,s,p},count,inspection) == True{} : Bool}:
  Bands.Inspection{+valid,+first,+end,leaves} = inspection
  LeafValues.Words{ci_word,co_word,h_word,w_word,k_word,s_word,p_word,oh_word,ow_word} = words
  %h_word : {valid && U32.is_zero(first) && Nat.is_eq(U32.to_nat(end),Nat.mul(U32.to_nat(count),_)) == True{} : Bool}
  checked

# Well formed bands of the input geometry cover every row of its images.
def covered(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+source: Model.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>) ->
  {covered_at(TensorExecution.execution_geometry(~Logical,frame,batched),Inspection.model(~F32,arrival(~Logical,frame,batched),source)) == True{} : Bool}:
  PipelineExecution.Frame{+logical,activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,parallel,+limits,+capacity,+weights,+bias,we,be,+domain,+valid,+supported,room} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,h,w,k,s,p} : KernelShape.Dimensions}
  +geometry = Leaf.input_geometry(images,dimensions)
  covered_rows(images,ci,co,h,w,k,s,p,Inspection.model(~F32,geometry,source),
    Caller.words(limits,Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n),
      Caller.valid(~Logical,~original,logical,dimensions,weights,bias,0n,valid),Caller.supported(~Logical,~original,logical,limits,dimensions,weights,bias,0n,supported)),
    BandShape.well_formed_checked(~F32,geometry,source,Machine.capacity(limits),Limits.capacity_fits(limits),BandGather.rows_room(geometry,limits,capacity,domain),shape))

# The band pipeline on arrived bands, with the parallelism of their layout.
def parallel_run(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+parallel: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),
    TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),parallel,Model.pack(~F32,source),
      TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)):
  PipelineExecution.Frame{+logical,+activation,KernelShape.Dimensions{+ci,+co,+h,+w,+k,+s,+p},+images,previous,+limits,+capacity,+sw,+sb,+we,+be,+domain,+valid,+supported,+room} = frame
  +frame = {PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,sw,sb,we,be,domain,valid,supported,room} : PipelineExecution.Frame<Logical>}
  +geometry = {TensorConvolution.Geometry{images,batched,KernelShape.Dimensions{ci,co,h,w,k,s,p}} : TensorConvolution.Geometry}
  PublicFrame.from_frame(~custom,~Logical,~original,frame,PipelineExecution.output_context(~Logical,frame),{==},
    TensorConvolution.band_output(~custom,activation,geometry,parallel,Model.pack(~F32,source),TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    TensorExecution.band_output_contract(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,shape,values,weight_exact,bias_exact))

def run(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+parallel: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +matched: {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  +limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),parallel,Model.pack(~F32,source),
      TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)):
  +output = OutputMetadata.bounds_geometry(~Logical,frame)
  +contract = parallel_run(~custom,~Logical,~original,frame,batched,parallel,workers,source,weights,bias,shape,values,weight_exact,bias_exact)
  +header = BandOutputNormal.band_output_header(~custom,~Logical,frame,batched,parallel,workers,Model.pack(~F32,source),Buffer.pack(~F32,weights),Buffer.pack(~F32,bias))
  convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),parallel,Model.pack(~F32,source),
      TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),
    BandOutputNormal.usable_from_header(batched,output,
      TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),parallel,Model.pack(~F32,source),
        TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),header),
    contract,
    BandOutputNormal.normal_from_header(batched,output,
      TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),parallel,Model.pack(~F32,source),
        TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),header,
      BandOutputNormal.completed_counted(BandOutputNormal.extents(batched,output),Bands.elements(output),BandOutputNormal.counted(~Logical,~original,frame,batched,matched,limits_exact)),
      matched,BandOutputNormal.contract_shape(~custom,~Logical,~original,frame,
        TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),parallel,Model.pack(~F32,source),
          TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers),contract)))

# Arrived bands are read through their model, and they cover every row.
def inspected(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+parallel: Bool,+workers: U32,
  -bands: Bands.Bands(~F32),+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+layout: Bands.Plan,
  observation: Model.Observation<F32,bands>,
  arrived: Arrive.Arrived(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),arrival(~Logical,frame,batched),layout,bands),
  +matched: {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  +limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.arrived(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),parallel,
      TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,
      Bands.inspect(~F32,arrival(~Logical,frame,batched),bands))):
  Model.Observation{+model,+equation} = observation
  Shape.Both{same,Shape.Both{well_formed,values}} = arrived
  +contents = Model.observed_contents(~F32,model,bands,equation)
  +exact = arrival_exact(~Logical,frame,batched)
  +context = PipelineExecution.output_context(~Logical,frame)
  +activation = TensorExecution.execution_activation(~Logical,frame)
  +geometry = TensorExecution.execution_geometry(~Logical,frame,batched)
  +input = arrival(~Logical,frame,batched)
  +inspection = Inspection.model(~F32,input,model)
  +owned = {
    %exact : BandShape.WellFormed<F32,_,Model.content(~F32,model)>
    %contents : BandShape.WellFormed<F32,input,_>
    well_formed : BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,model)>}
  +read = {
    %exact : BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),_,Model.content(~F32,model))
    %contents : BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),input,_)
    values : BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,model))}
  %equation : Convolved(~custom,~Logical,~original,context,frame_extents(~Logical,frame,batched),
    TensorConvolution.arrived(~custom,activation,geometry,parallel,TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,Bands.inspect(~F32,input,_)))
  %Equal.sym(Bands.Bands(~F32) & Bands.Inspection,Bands.inspect(~F32,input,Model.pack(~F32,model)),(Model.pack(~F32,model),inspection),Inspection.inspect(~F32,input,model)) :
    Convolved(~custom,~Logical,~original,context,frame_extents(~Logical,frame,batched),
      TensorConvolution.arrived(~custom,activation,geometry,parallel,TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,_))
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.arrived(~custom,activation,geometry,parallel,TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,(Model.pack(~F32,model),inspection)),
    TensorConvolution.arrived_covered(~custom,activation,covered_at(geometry,inspection),geometry,parallel,Model.pack(~F32,model),TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers),
    arrived_matches(~custom,activation,geometry,parallel,TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,Model.pack(~F32,model),inspection)) :
    Convolved(~custom,~Logical,~original,context,frame_extents(~Logical,frame,batched),_)
  %Equal.sym(Bool,covered_at(geometry,inspection),True{},covered(~Logical,~original,frame,batched,model,owned)) :
    Convolved(~custom,~Logical,~original,context,frame_extents(~Logical,frame,batched),
      TensorConvolution.arrived_covered(~custom,activation,_,geometry,parallel,Model.pack(~F32,model),TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers))
  run(~custom,~Logical,~original,frame,batched,parallel,workers,model,weights,bias,matched,limits_exact,owned,read,weight_exact,bias_exact)

# The input arrives in the selected layout.
def selected(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+model: Model.Model<F32>,+other: Bands.Geometry,
  +weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,+selection: BandPlanning.LayoutPlan,
  arrival_facts: Target.Arrival<shape,arrival(~Logical,frame,batched),Planning.layout(selection)>,
  well_formed: BandShape.WellFormed<F32,other,Model.content(~F32,model)>,other_coherent: Global.Coherent(shape,other),
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),other,Model.content(~F32,model)),
  +matched: {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  +limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.selected(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),
      TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,
      Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},selection)):
  BandPlanning.LayoutPlan{+layout,+parallel,time,repartition} = selection
  +input = arrival(~Logical,frame,batched)
  inspected(~custom,~Logical,~original,frame,batched,parallel,workers,
    Tensors.arrive(~F32,input,layout,Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers}),weights,bias,shape,fill,layout,
    Model.observe(~F32,Tensors.arrive(~F32,input,layout,Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers})),
    Arrive.arrived(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),shape,input,layout,other,fill,Tensors.Usable{},workers,model,arrival_facts,well_formed,other_coherent,values),
    matched,limits_exact,weight_exact,bias_exact)

# Well formed bands of a geometry fit it and cover its rows.
def layout_facts(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+model: Model.Model<F32>,+domain: BandGather.Domain<geometry,limits,capacity>,
  well_formed: BandShape.WellFormed<F32,geometry,Model.content(~F32,model)>) -> Planning.LayoutFacts(geometry,limits,capacity,Model.layout(~F32,model)):
  BandShape.WellFormed{metadata,ready,coverage} = well_formed
  Shape.Both{FromWhole.target(~F32,geometry,limits,capacity,model,domain,metadata,Model.ready(~F32,model,ready)),
    %Equal.sym(Bands.Plan,Model.layout(~F32,model),Model.content_layout(~F32,Model.content(~F32,model)),Model.layout_matches_content(~F32,model)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    coverage}

# The planner chooses again from the plan of an input of the geometry it reads.
def replanned(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+model: Model.Model<F32>,+other: Bands.Geometry,
  +weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  observation: Observation.ReplanObservation<Model.layout(~F32,model),
    TensorConvolution.band_work(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame)),U32.shln(1,Buffer.depth(~F32,bias))),
    TensorConvolution.context_window(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame))),
    workers,DefaultCost.configuration(),PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame)>,
  +counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,arrival(~Logical,frame,batched)),
  well_formed: BandShape.WellFormed<F32,other,Model.content(~F32,model)>,other_coherent: Global.Coherent(shape,other),
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),other,Model.content(~F32,model)),
  +matched: {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  +limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.selected(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),
      TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,
      Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},
      BandPlanning.replan(Model.layout(~F32,model),
        TensorConvolution.band_work(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame)),U32.shln(1,Buffer.depth(~F32,bias))),
        TensorConvolution.context_window(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame))),
        workers,DefaultCost.configuration()))):
  Observation.ReplanObservation{+selection,exact,facts} = observation
  +geometry = TensorExecution.execution_geometry(~Logical,frame,batched)
  +activation = TensorExecution.execution_activation(~Logical,frame)
  +window = TensorConvolution.context_window(TensorConvolution.band_context(geometry,True{},activation))
  %exact : Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.selected(~custom,activation,geometry,TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,
      Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},_))
  selected(~custom,~Logical,~original,frame,batched,model,other,weights,bias,shape,fill,workers,selection,
    Target.Arrival{counted,coherent,{
      %window_input(geometry,True{},activation) : Target.Planned(_,Planning.layout(selection))
      Target.fitted_planned(BandWindow.input_geometry(window),PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame),Planning.layout(selection),facts) :
      Target.Planned(arrival(~Logical,frame,batched),Planning.layout(selection))}},
    well_formed,other_coherent,values,matched,limits_exact,weight_exact,bias_exact)

# The first split of every image, when the input holds another geometry.
def first(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+model: Model.Model<F32>,+other: Bands.Geometry,
  +weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,+plan: BandPlanning.Plan,+window: BandWindow.Geometry,
  +window_exact: {BandWindow.input_geometry(window) == arrival(~Logical,frame,batched) : Bands.Geometry},
  +domain: BandGather.Domain<arrival(~Logical,frame,batched),PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame)>,
  +counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,arrival(~Logical,frame,batched)),
  well_formed: BandShape.WellFormed<F32,other,Model.content(~F32,model)>,other_coherent: Global.Coherent(shape,other),
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),other,Model.content(~F32,model)),
  +matched: {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  +limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.selected(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),
      TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,
      Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},BandPlanning.first_split(plan,window))):
  BandPlanning.Plan{+depth,+parallel,+time} = plan
  BandWindow.Geometry{Bands.Geometry{+batch,+channels,+height,+width},kernel,stride,padding,oh,ow} = window
  +split = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +limits = PipelineExecution.convolution_limits(~Logical,frame)
  +capacity = PipelineExecution.convolution_capacity(~Logical,frame)
  +layout = Bands.image_plan(U32.to_nat(batch),0,U32.to_nat(depth),height)
  selected(~custom,~Logical,~original,frame,batched,model,other,weights,bias,shape,fill,workers,BandPlanning.LayoutPlan{layout,parallel,time,0.0},
    Target.Arrival{counted,coherent,{
      %window_exact : Target.Planned(_,layout)
      Target.split_planned(split,limits,capacity,U32.to_nat(depth),
        %Equal.sym(Bands.Geometry,split,arrival(~Logical,frame,batched),window_exact) : BandGather.Domain<_,limits,capacity> domain) :
      Target.Planned(arrival(~Logical,frame,batched),layout)}},
    well_formed,other_coherent,values,matched,limits_exact,weight_exact,bias_exact)

# The layout the call chooses: again from the input's plan when the input
# holds the geometry the convolution reads, otherwise the first split.
def planned(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,same: Bool,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+model: Model.Model<F32>,+other: Bands.Geometry,
  +weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +compared: {Bands.same_geometry(arrival(~Logical,frame,batched),other) == same : Bool},
  +domain: BandGather.Domain<arrival(~Logical,frame,batched),PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame)>,
  +counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,arrival(~Logical,frame,batched)),
  +well_formed: BandShape.WellFormed<F32,other,Model.content(~F32,model)>,other_coherent: Global.Coherent(shape,other),
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),other,Model.content(~F32,model)),
  +matched: {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  +limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.selected(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),
      TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,
      Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},
      BandPlanning.layout_for(same,Model.layout(~F32,model),
        TensorConvolution.band_work(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame)),U32.shln(1,Buffer.depth(~F32,bias))),
        TensorConvolution.context_window(TensorConvolution.band_context(TensorExecution.execution_geometry(~Logical,frame,batched),True{},TensorExecution.execution_activation(~Logical,frame))),
        workers,DefaultCost.configuration()))):
  match same:
    case True{}:
      +geometry = TensorExecution.execution_geometry(~Logical,frame,batched)
      +activation = TensorExecution.execution_activation(~Logical,frame)
      +context = TensorConvolution.band_context(geometry,True{},activation)
      +window = TensorConvolution.context_window(context)
      +input = arrival(~Logical,frame,batched)
      +limits = PipelineExecution.convolution_limits(~Logical,frame)
      +capacity = PipelineExecution.convolution_capacity(~Logical,frame)
      +into = Equal.sym(Bands.Geometry,BandWindow.input_geometry(window),input,window_input(geometry,True{},activation))
      +owned = {
        %Equal.sym(Bands.Geometry,input,other,BandCount.same_geometry_equal(input,other,compared)) : BandShape.WellFormed<F32,_,Model.content(~F32,model)>
        well_formed : BandShape.WellFormed<F32,input,Model.content(~F32,model)>}
      replanned(~custom,~Logical,~original,frame,batched,model,other,weights,bias,shape,fill,workers,
        Observation.observe(Model.layout(~F32,model),TensorConvolution.band_work(context,U32.shln(1,Buffer.depth(~F32,bias))),window,workers,DefaultCost.configuration(),limits,capacity,
          %into : BandGather.Domain<_,limits,capacity> domain,
          %into : Planning.LayoutFacts(_,limits,capacity,Model.layout(~F32,model)) layout_facts(input,limits,capacity,model,domain,owned)),
        counted,coherent,well_formed,other_coherent,values,matched,limits_exact,weight_exact,bias_exact)
    case False{}:
      +geometry = TensorExecution.execution_geometry(~Logical,frame,batched)
      +activation = TensorExecution.execution_activation(~Logical,frame)
      +context = TensorConvolution.band_context(geometry,True{},activation)
      first(~custom,~Logical,~original,frame,batched,model,other,weights,bias,shape,fill,workers,
        BandPlanning.select(TensorConvolution.band_work(context,U32.shln(1,Buffer.depth(~F32,bias))),TensorConvolution.context_window(context),workers,DefaultCost.configuration()),
        TensorConvolution.context_window(context),window_input(geometry,True{},activation),
        domain,counted,coherent,well_formed,other_coherent,values,matched,limits_exact,weight_exact,bias_exact)

# A Normal input offers its geometry and plan, and the bias certificate sizes
# the work the planner reads.
def offered(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+model: Model.Model<F32>,+other: Bands.Geometry,
  +weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +domain: BandGather.Domain<arrival(~Logical,frame,batched),PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame)>,
  +counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,arrival(~Logical,frame,batched)),
  +well_formed: BandShape.WellFormed<F32,other,Model.content(~F32,model)>,other_coherent: Global.Coherent(shape,other),
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),other,Model.content(~F32,model)),
  +matched: {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  +limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.offered(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),
      TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,
      Tensors.offer(~F32,Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers}))):
  Buffer.Model{bias_tree,bias_length,Certificate.Certificate{bias_depth,bias_equation}} = bias
  +bias = {Buffer.Model{bias_tree,bias_length,Certificate.Certificate{bias_depth,bias_equation}} : Buffer.Model<F32>}
  +geometry = TensorExecution.execution_geometry(~Logical,frame,batched)
  +layout = Model.layout(~F32,model)
  %Equal.sym(Tensors.Tensor<F32> & BandSchedule.Target,Tensors.offer(~F32,Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers}),
    (Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},BandSchedule.Target{other,layout,BandSchedule.splits(layout)}),
    Target.offer(~F32,model,other,shape,fill,workers)) :
    Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
      TensorConvolution.offered(~custom,TensorExecution.execution_activation(~Logical,frame),geometry,TensorConvolution.packed(geometry,Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),shape,fill,workers,_))
  planned(~custom,~Logical,~original,Bands.same_geometry(arrival(~Logical,frame,batched),other),frame,batched,model,other,weights,bias,shape,fill,workers,{==},
    domain,counted,coherent,well_formed,other_coherent,values,matched,limits_exact,weight_exact,bias_exact)

# The checks of the call accept the frame's parameters and machine guard.
def checked(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+model: Model.Model<F32>,+other: Bands.Geometry,
  +weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +domain: BandGather.Domain<arrival(~Logical,frame,batched),PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame)>,
  +counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,arrival(~Logical,frame,batched)),
  +well_formed: BandShape.WellFormed<F32,other,Model.content(~F32,model)>,other_coherent: Global.Coherent(shape,other),
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),other,Model.content(~F32,model)),
  +matched: {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  +limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits},
  weight_length: {Buffer.length(~F32,weights) == PublicFrame.weight_length(~Logical,frame) : Nat},bias_length: {Buffer.length(~F32,bias) == PublicFrame.bias_length(~Logical,frame) : Nat},
  weight_ready: Buffer.Sized<F32,weights>,bias_ready: Buffer.Sized<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{TensorExecution.execution_geometry(~Logical,frame,batched)},
      Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers)):
  +context = PipelineExecution.output_context(~Logical,frame)
  +activation = TensorExecution.execution_activation(~Logical,frame)
  +geometry = TensorExecution.execution_geometry(~Logical,frame,batched)
  +machine = {
    %limits_exact : {TensorConvolution.machine_support(geometry,_) == True{} : Bool}
    machine_accepted(~Logical,~original,frame,batched) : {TensorConvolution.machine_support(geometry,Default.limits()) == True{} : Bool}}
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.band_checked(~custom,activation,Some{geometry},Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    TensorConvolution.band_supported(~custom,activation,parameters(geometry,weights,bias),geometry,Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},
      Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers),
    checked_matches(~custom,activation,geometry,Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},weights,bias,shape,fill,workers)) :
    Convolved(~custom,~Logical,~original,context,frame_extents(~Logical,frame,batched),_)
  %Equal.sym(Bool,parameters(geometry,weights,bias),True{},parameters_accepted(~Logical,frame,batched,weights,bias,weight_length,bias_length,weight_ready,bias_ready)) :
    Convolved(~custom,~Logical,~original,context,frame_extents(~Logical,frame,batched),
      TensorConvolution.band_supported(~custom,activation,_,geometry,Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  %Equal.sym(Bool,TensorConvolution.machine_support(geometry,Default.limits()),True{},machine) :
    Convolved(~custom,~Logical,~original,context,frame_extents(~Logical,frame,batched),
      TensorConvolution.band_machine(~custom,activation,_,geometry,Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  offered(~custom,~Logical,~original,frame,batched,model,other,weights,bias,shape,fill,workers,domain,counted,coherent,well_formed,other_coherent,values,matched,limits_exact,weight_exact,bias_exact)

def frame_output_images(~Logical: Data,+frame: PipelineExecution.Frame<Logical>) ->
  {BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)) == PublicFrame.frame_images(~Logical,frame) : U32}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  {==}

def native_output_match(+input: PublicFrame.NativeInput) ->
  {BandOutputNormal.shape_matched(PublicFrame.input_images(input),batched(input)) == True{} : Bool}:
  PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p} = input
  match batched:
    case True{}: {==}
    case False{}: {==}

def output_matched(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+input: PublicFrame.NativeInput,
  images_exact: {PublicFrame.frame_images(~Logical,frame) == PublicFrame.input_images(input) : U32}) ->
  {BandOutputNormal.shape_matched(BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),batched(input)) == True{} : Bool}:
  %Equal.sym(U32,BandOutputNormal.images(OutputMetadata.bounds_geometry(~Logical,frame)),PublicFrame.frame_images(~Logical,frame),frame_output_images(~Logical,frame)) :
    {BandOutputNormal.shape_matched(_,batched(input)) == True{} : Bool}
  %Equal.sym(U32,PublicFrame.frame_images(~Logical,frame),PublicFrame.input_images(input),images_exact) :
    {BandOutputNormal.shape_matched(_,batched(input)) == True{} : Bool}
  native_output_match(input)

def frame_dimension_extents(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool) ->
  {frame_extents(~Logical,frame,batched) == dimension_extents(batched,PublicFrame.frame_images(~Logical,frame),PipelineExecution.dimensions(~Logical,frame)) : List<&2,U32>}:
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,h,w,k,s,p},images,parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  {==}

def image_extents(+batched: Bool,+images: U32,+channels: U32,+height: U32,+width: U32) ->
  {TensorConvolution.output_shape(batched,PublicFrame.image_count(batched,images),channels,height,width) == TensorConvolution.output_shape(batched,images,channels,height,width) : List<&2,U32>}:
  match batched:
    case True{}: {==}
    case False{}: {==}

def native_extents(+input: PublicFrame.NativeInput) ->
  {dimension_extents(batched(input),PublicFrame.input_images(input),PublicFrame.frame_dimensions(input)) == output_extents(input) : List<&2,U32>}:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  +height = U32.from_nat(ConvSpec.axis_size(U32.to_nat(h),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)))
  +width = U32.from_nat(ConvSpec.axis_size(U32.to_nat(w),U32.to_nat(k),U32.to_nat(s),U32.to_nat(p)))
  %Equal.sym(U32,U32.from_nat(U32.to_nat(co)),co,Arithmetic.native_roundtrip(co)) :
    {TensorConvolution.output_shape(batched,PublicFrame.image_count(batched,images),_,height,width) == TensorConvolution.output_shape(batched,images,co,height,width) : List<&2,U32>}
  image_extents(batched,images,co,height,width)

# The output extents of the frame of a call are those of its native input.
def framed_extents(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+input: PublicFrame.NativeInput,
  images_exact: {PublicFrame.frame_images(~Logical,frame) == PublicFrame.input_images(input) : U32},
  dimensions_exact: {PipelineExecution.dimensions(~Logical,frame) == PublicFrame.frame_dimensions(input) : KernelShape.Dimensions}) ->
  {frame_extents(~Logical,frame,batched(input)) == output_extents(input) : List<&2,U32>}:
  Equal.trans(List<&2,U32>,frame_extents(~Logical,frame,batched(input)),
    dimension_extents(batched(input),PublicFrame.frame_images(~Logical,frame),PipelineExecution.dimensions(~Logical,frame)),output_extents(input),
    frame_dimension_extents(~Logical,frame,batched(input)),{
      %Equal.sym(U32,PublicFrame.frame_images(~Logical,frame),PublicFrame.input_images(input),images_exact) :
        {dimension_extents(batched(input),_,PipelineExecution.dimensions(~Logical,frame)) == output_extents(input) : List<&2,U32>}
      %Equal.sym(KernelShape.Dimensions,PipelineExecution.dimensions(~Logical,frame),PublicFrame.frame_dimensions(input),dimensions_exact) :
        {dimension_extents(batched(input),PublicFrame.input_images(input),_) == output_extents(input) : List<&2,U32>}
      native_extents(input) :
        {dimension_extents(batched(input),PublicFrame.frame_images(~Logical,frame),PipelineExecution.dimensions(~Logical,frame)) == output_extents(input) : List<&2,U32>}})

# The frame of the call fixes the logical context, the parameter sequences and
# the native geometry.
def framed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +frame: PipelineExecution.Frame<Logical>,+model: Model.Model<F32>,+other: Bands.Geometry,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+workers: U32,
  +fields: PublicFrame.Fields<Logical,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits())>,
  lengths: Shape.Both<{PublicFrame.weight_length(~Logical,frame) == ConvSpec.length(weight_list) : Nat},{PublicFrame.bias_length(~Logical,frame) == ConvSpec.length(bias_list) : Nat}>,
  +counted: Global.Counted<input_extents(input)>,
  well_formed: BandShape.WellFormed<F32,other,Model.contents(~F32,Model.pack(~F32,model))>,other_coherent: Global.Coherent(input_extents(input),other),
  values: BandValues.Values(~F32,~Logical,~original,logical,other,Model.contents(~F32,Model.pack(~F32,model))),
  weight_length: {Buffer.length(~F32,weights) == ConvSpec.length(weight_list) : Nat},bias_length: {Buffer.length(~F32,bias) == ConvSpec.length(bias_list) : Nat},
  weight_ready: Buffer.Sized<F32,weights>,bias_ready: Buffer.Sized<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  Convolved(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,weight_list,bias_list),output_extents(input),
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},Tensors.Tensor{Model.pack(~F32,model),other,input_extents(input),fill,Tensors.Usable{},workers},
      Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),input_extents(input),fill,workers)):
  PublicFrame.Fields{+logical_exact,+activation_exact,+images_exact,dimensions_exact,+input_geometry_exact,+weights_exact,bias_content_exact,+context_exact,+limits_exact,capacity_exact} = fields
  Shape.Both{weight_length_exact,bias_length_exact} = lengths
  +batched = batched(input)
  +shape = input_extents(input)
  +geometry_exact = geometry_fields(~Logical,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits()),fields)
  +bias_settings = PublicFrame.settings_bias(~Logical,~original,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,frame,Default.limits(),Machine.capacity(Default.limits()),fields)
  +to_arrival = Equal.sym(Bands.Geometry,arrival(~Logical,frame,batched),PipelineExecution.input_geometry(~Logical,frame),arrival_exact(~Logical,frame,batched))
  +from_source = Equal.sym(Bands.Geometry,PipelineExecution.input_geometry(~Logical,frame),PublicFrame.source_geometry(input),input_geometry_exact)
  +coherent = {
    %to_arrival : Global.Coherent(shape,_)
    %from_source : Global.Coherent(shape,_)
    BandCount.shape_coherence_elements(shape,PublicFrame.source_geometry(input),source_rows(input)) : Global.Coherent(shape,arrival(~Logical,frame,batched))}
  +domain = {
    %to_arrival : BandGather.Domain<_,PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame)>
    PipelineExecution.convolution_domain(~Logical,frame) :
    BandGather.Domain<arrival(~Logical,frame,batched),PipelineExecution.convolution_limits(~Logical,frame),PipelineExecution.convolution_capacity(~Logical,frame)>}
  %framed_extents(~Logical,frame,input,images_exact,dimensions_exact) : Convolved(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,weight_list,bias_list),_,
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  %context_exact : Convolved(~custom,~Logical,~original,_,frame_extents(~Logical,frame,batched),
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  %activation_exact : Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.band_checked(~custom,_,Some{PublicFrame.geometry(input)},Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  %geometry_exact : Convolved(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),frame_extents(~Logical,frame,batched),
    TensorConvolution.band_checked(~custom,TensorExecution.execution_activation(~Logical,frame),Some{_},Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers},
      Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),shape,fill,workers))
  checked(~custom,~Logical,~original,frame,batched,model,other,weights,bias,shape,fill,workers,domain,counted,coherent,
    %Model.packed_contents(~F32,model) : BandShape.WellFormed<F32,other,_> well_formed,
    other_coherent,
    %Equal.sym(Logical,PipelineExecution.convolution_logical(~Logical,frame),logical,logical_exact) : BandValues.Values(~F32,~Logical,~original,_,other,Model.content(~F32,model))
    %Model.packed_contents(~F32,model) : BandValues.Values(~F32,~Logical,~original,logical,other,_)
    values,
    output_matched(~Logical,frame,input,images_exact),limits_exact,
    Equal.trans(Nat,Buffer.length(~F32,weights),ConvSpec.length(weight_list),PublicFrame.weight_length(~Logical,frame),weight_length,Equal.sym(Nat,PublicFrame.weight_length(~Logical,frame),ConvSpec.length(weight_list),weight_length_exact)),
    Equal.trans(Nat,Buffer.length(~F32,bias),ConvSpec.length(bias_list),PublicFrame.bias_length(~Logical,frame),bias_length,Equal.sym(Nat,PublicFrame.bias_length(~Logical,frame),ConvSpec.length(bias_list),bias_length_exact)),
    weight_ready,bias_ready,
    Equal.trans(Tree.Tree<F32>,Buffer.tree(~F32,weights),StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence),PipelineExecution.source_tree(~Logical,frame),weight_exact,
      Equal.sym(Tree.Tree<F32>,PipelineExecution.source_tree(~Logical,frame),StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence),weights_exact)),
    Equal.trans(Buffer.Content<F32>,Buffer.content(~F32,bias),Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)},
      PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),bias_exact,
      Equal.sym(Buffer.Content<F32>,PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)),Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)},bias_settings)))

# The existing convolution valid guard supplies positive C/H/W. The shape's
# actual Counted receipt supplies the whole-tensor bound, including batch N.
def counted_domain(+shape: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+counted: Global.Counted<shape>,
  coherent: Global.Coherent(shape,Bands.Geometry{images,channels,height,width}),
  positives: BandNativeSourceDomain.Positive<U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width)>) ->
  BandGather.Domain<Bands.Geometry{images,channels,height,width},Default.limits(),Machine.capacity(Default.limits())>:
  BandNativeSourceDomain.Positive{cp,hp,wp} = positives
  Global.domain(shape,Bands.Geometry{images,channels,height,width},counted,coherent,cp,hp,wp)

def source_domain(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+input: PublicFrame.NativeInput,
  +weights: List<&2,F32>,+bias: List<&2,F32>,
  shape: Shape.Both<Global.Counted<input_extents(input)>,Global.Coherent(input_extents(input),PublicFrame.source_geometry(input))>,
  validated: {PublicFrame.frame_valid(input,weights,bias) == True{} : Bool}) ->
  BandGather.Domain<Leaf.input_geometry(PublicFrame.input_images(input),PublicFrame.frame_dimensions(input)),Default.limits(),Machine.capacity(Default.limits())>:
  PublicFrame.NativeInput{+images,+batched,+ci,+co,+h,+w,+k,+s,+p} = input
  Shape.Both{+counted,coherent} = shape
  +dimensions = BandNativeSourceDomain.dimensions(ci,co,h,w,k,s,p)
  +problem = Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n)
  +count = PublicFrame.image_count(batched,images)
  BandNativeSourceDomain.converted(Bands.Geometry{count,ci,h,w},Leaf.input_geometry(count,dimensions),Default.limits(),Machine.capacity(Default.limits()),
    Equal.sym(Bands.Geometry,Leaf.input_geometry(count,dimensions),Bands.Geometry{count,ci,h,w},BandNativeSourceDomain.geometry_roundtrip(count,ci,co,h,w,k,s,p)),
    counted_domain(input_extents(PublicFrame.NativeInput{images,batched,ci,co,h,w,k,s,p}),count,ci,h,w,counted,coherent,
      BandNativeSourceDomain.positive(problem,Guards.valid(problem,Caller.valid(~Logical,~original,logical,dimensions,weights,bias,0n,validated)))))

# A Normal input and whole parameters of the record's extents.
def native(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +weight_list: List<&2,F32>,+bias_list: List<&2,F32>,
  +weight_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(weight_list),weight_list>,+bias_evidence: StorageLaws.SequenceEvidence<F32,ConvSpec.length(bias_list),bias_list>,
  +validated: {PublicFrame.frame_valid(input,weight_list,bias_list) == True{} : Bool},+supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +model: Model.Model<F32>,+other: Bands.Geometry,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+workers: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,model),other,input_extents(input),fill,Tensors.Usable{},workers}),
  weight_length: {Buffer.length(~F32,weights) == ConvSpec.length(weight_list) : Nat},bias_length: {Buffer.length(~F32,bias) == ConvSpec.length(bias_list) : Nat},
  weight_ready: Buffer.Sized<F32,weights>,bias_ready: Buffer.Sized<F32,bias>,
  weight_exact: {Buffer.tree(~F32,weights) == StorageLaws.sequence_storage(F32,ConvSpec.length(weight_list),weight_list,weight_evidence) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == Buffer.Content{StorageLaws.sequence_storage(F32,ConvSpec.length(bias_list),bias_list,bias_evidence),ConvSpec.length(bias_list)} : Buffer.Content<F32>}) ->
  Convolved(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,weight_list,bias_list),output_extents(input),
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},Tensors.Tensor{Model.pack(~F32,model),other,input_extents(input),fill,Tensors.Usable{},workers},
      Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),input_extents(input),fill,workers)):
  Shape.Both{Shape.Both{well_formed,Shape.Both{+counted,other_coherent}},values} = contract
  +limits = Default.limits()
  +capacity = Machine.capacity(Default.limits())
  +domain = source_domain(~Logical,~original,logical,input,weight_list,bias_list,
    Shape.Both{counted,BandCount.shape_coherence_elements(input_extents(input),PublicFrame.source_geometry(input),source_rows(input))},validated)
  framed(~custom,~Logical,~original,logical,activation,input,weight_list,bias_list,weight_evidence,bias_evidence,
    PublicFrame.configured_native(~Logical,~original,logical,activation,input,True{},limits,capacity,weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported),
    model,other,weights,bias,fill,workers,
    PublicFrame.configured(~Logical,~original,logical,activation,input,True{},limits,capacity,weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported),
    configured_lengths(~Logical,~original,logical,activation,input,True{},limits,capacity,weight_list,bias_list,domain,weight_evidence,bias_evidence,validated,supported),
    counted,well_formed,other_coherent,values,weight_length,bias_length,weight_ready,bias_ready,weight_exact,bias_exact)

# Weights and bias arrive as one leaf each, and the input keeps its layout
# and the larger worker count.
def layout_matches(~custom: F32 -> F32,+activation: Activations.Activation,+input: PublicFrame.NativeInput,-bands: Bands.Bands(~F32),+other: Bands.Geometry,
  +weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,+fill: F32,+input_workers: U32,+parameter_workers: U32) ->
  {TensorConvolution.input_layout(~custom,activation,Tensors.Tensor{bands,other,input_extents(input),fill,Tensors.Usable{},input_workers},
      Buffer.pack(~F32,weights),weight_extents(input),Buffer.pack(~F32,bias),bias_extents(input),Tensors.Usable{},parameter_workers,stride(input),padding(input)) ==
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},
      Tensors.Tensor{bands,other,input_extents(input),fill,Tensors.Usable{},U32.max(input_workers,parameter_workers)},
      Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),input_extents(input),fill,U32.max(input_workers,parameter_workers)) : Tensors.Tensor<F32>}:
  +workers = U32.max(input_workers,parameter_workers)
  %Equal.sym(Maybe<&2,TensorConvolution.Geometry>,
    TensorConvolution.geometry(TensorConvolution.input_shape(input_extents(input)),weight_extents(input),bias_extents(input),stride(input),padding(input)),Some{PublicFrame.geometry(input)},public_geometry(input)) :
    {TensorConvolution.band_checked(~custom,activation,_,Tensors.Tensor{bands,other,input_extents(input),fill,Tensors.Usable{},workers},Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),input_extents(input),fill,workers) ==
      TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},Tensors.Tensor{bands,other,input_extents(input),fill,Tensors.Usable{},workers},
        Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),input_extents(input),fill,workers) : Tensors.Tensor<F32>}
  {==}

# The input as the tensor the call reads.
def source_tensor(model: Model.Model<F32>,other: Bands.Geometry,shape: List<&2,U32>,fill: F32,workers: U32) -> Tensors.Tensor<F32>:
  Tensors.Tensor{Model.pack(~F32,model),other,shape,fill,Tensors.Usable{},workers}

def parameters_contract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+activation: Activations.Activation,+input: PublicFrame.NativeInput,
  +model: Model.Model<F32>,+other: Bands.Geometry,+fill: F32,+input_workers: U32,+parameter_workers: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,model),other,input_extents(input),fill,Tensors.Usable{},input_workers}),
  +supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  -weight_buffer: Storage.Buffer<F32>,-bias_buffer: Storage.Buffer<F32>,+weight_parameter: Parameters.Parameter<weight_buffer>,+bias_parameter: Parameters.Parameter<bias_buffer>,
  +validated: {PublicFrame.frame_valid(input,Parameters.parameter_values(weight_buffer,weight_parameter),Parameters.parameter_values(bias_buffer,bias_parameter)) == True{} : Bool}) ->
  Convolved(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,Parameters.parameter_values(weight_buffer,weight_parameter),Parameters.parameter_values(bias_buffer,bias_parameter)),output_extents(input),
    TensorConvolution.input_layout(~custom,activation,source_tensor(model,other,input_extents(input),fill,input_workers),
      weight_buffer,weight_extents(input),bias_buffer,bias_extents(input),Tensors.Usable{},parameter_workers,stride(input),padding(input))):
  Parameters.Parameter{+weight_model,+weight_equation,+weight_ready,+weight_values,+weight_length,+weight_evidence,+weight_exact} = weight_parameter
  Parameters.Parameter{+bias_model,+bias_equation,+bias_ready,+bias_values,+bias_length,+bias_evidence,+bias_exact} = bias_parameter
  +context = PublicFrame.reference_context(~Logical,logical,activation,input,weight_values,bias_values)
  +workers = U32.max(input_workers,parameter_workers)
  %weight_equation :
    Convolved(~custom,~Logical,~original,context,output_extents(input),TensorConvolution.input_layout(~custom,activation,source_tensor(model,other,input_extents(input),fill,input_workers),
      _,weight_extents(input),bias_buffer,bias_extents(input),Tensors.Usable{},parameter_workers,stride(input),padding(input)))
  %bias_equation :
    Convolved(~custom,~Logical,~original,context,output_extents(input),TensorConvolution.input_layout(~custom,activation,source_tensor(model,other,input_extents(input),fill,input_workers),
      Buffer.pack(~F32,weight_model),weight_extents(input),_,bias_extents(input),Tensors.Usable{},parameter_workers,stride(input),padding(input)))
  %Equal.sym(Tensors.Tensor<F32>,
    TensorConvolution.input_layout(~custom,activation,source_tensor(model,other,input_extents(input),fill,input_workers),
      Buffer.pack(~F32,weight_model),weight_extents(input),Buffer.pack(~F32,bias_model),bias_extents(input),Tensors.Usable{},parameter_workers,stride(input),padding(input)),
    TensorConvolution.band_checked(~custom,activation,Some{PublicFrame.geometry(input)},Tensors.Tensor{Model.pack(~F32,model),other,input_extents(input),fill,Tensors.Usable{},workers},
      Buffer.pack(~F32,weight_model),Buffer.pack(~F32,bias_model),input_extents(input),fill,workers),
    layout_matches(~custom,activation,input,Model.pack(~F32,model),other,weight_model,bias_model,fill,input_workers,parameter_workers)) :
    Convolved(~custom,~Logical,~original,context,output_extents(input),_)
  native(~custom,~Logical,~original,logical,activation,input,weight_values,bias_values,weight_evidence,bias_evidence,validated,supported,
    model,other,weight_model,bias_model,fill,workers,contract,
    Equal.sym(Nat,ConvSpec.length(weight_values),Buffer.length(~F32,weight_model),weight_length),Equal.sym(Nat,ConvSpec.length(bias_values),Buffer.length(~F32,bias_model),bias_length),
    weight_ready,bias_ready,weight_exact,Parameters.parameter_content(bias_buffer,bias_parameter))

# The values the specification reads from weights and bias: their readers'
# values at every index of the parameter extents.
def weight_values(~Context: Data,~value: Context -> U32 -> F32,+context: Context,+input: PublicFrame.NativeInput) -> List<&2,F32>:
  Contract.listed(~F32,~Context,~value,ShapeLaws.elements(weight_extents(input)),0n,context)

def bias_values(~Context: Data,~value: Context -> U32 -> F32,+context: Context,+input: PublicFrame.NativeInput) -> List<&2,F32>:
  Contract.listed(~F32,~Context,~value,ShapeLaws.elements(bias_extents(input)),0n,context)

# Storage that holds a reader's values over the extents is the parameter of
# the reader's values.
def parameter_of(~Context: Data,~value: Context -> U32 -> F32,+context: Context,+shape: List<&2,U32>,+model: Buffer.Model<F32>,
  +facts: GatherValues.Facts(~F32,~Context,~value,context,shape,Buffer.content(~F32,model))) ->
  Parameters.Parameter<Buffer.pack(~F32,model)>:
  Shape.Both{Contract.WellFormed{length_exact,ready},values} = facts
  Parameters.observed_parameter(Buffer.pack(~F32,model),
    %Equal.sym(Buffer.Content<F32>,Buffer.contents(~F32,Buffer.pack(~F32,model)),Buffer.content(~F32,model),Buffer.packed_contents(~F32,model)) : Buffer.ContentReady<F32,_> ready,
    Buffer.Observation{model,{==}})

def parameter_listed(~Context: Data,~value: Context -> U32 -> F32,+context: Context,+shape: List<&2,U32>,+model: Buffer.Model<F32>,
  +facts: GatherValues.Facts(~F32,~Context,~value,context,shape,Buffer.content(~F32,model))) ->
  {Parameters.parameter_values(Buffer.pack(~F32,model),parameter_of(~Context,~value,context,shape,model,facts)) ==
    Contract.listed(~F32,~Context,~value,ShapeLaws.elements(shape),0n,context) : List<&2,F32>}:
  Shape.Both{Contract.WellFormed{+length_exact,+ready},+values} = facts
  %length_exact :
    {Parameters.parameter_values(Buffer.pack(~F32,model),parameter_of(~Context,~value,context,shape,model,Shape.Both{Contract.WellFormed{length_exact,ready},values})) ==
      Contract.listed(~F32,~Context,~value,_,0n,context) : List<&2,F32>}
  Parameters.listed_sequence(~Context,~value,context,Buffer.content(~F32,model),values)

# With the weights and bias as storage holding their readers' values, the
# call is the native convolution of those values.
def parameterized(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,~WeightContext: Data,~weight_value: WeightContext -> U32 -> F32,
  ~BiasContext: Data,~bias_value: BiasContext -> U32 -> F32,+logical: Logical,+weight_context: WeightContext,+bias_context: BiasContext,
  +activation: Activations.Activation,+input: PublicFrame.NativeInput,+model: Model.Model<F32>,+other: Bands.Geometry,+fill: F32,+input_workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,model),other,input_extents(input),fill,Tensors.Usable{},input_workers}),
  +supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +weights: Buffer.Model<F32>,+weight_facts: GatherValues.Facts(~F32,~WeightContext,~weight_value,weight_context,weight_extents(input),Buffer.content(~F32,weights)),+weight_workers: U32,
  +bias: Buffer.Model<F32>,+bias_facts: GatherValues.Facts(~F32,~BiasContext,~bias_value,bias_context,bias_extents(input),Buffer.content(~F32,bias)),+bias_workers: U32,
  +validated: {PublicFrame.frame_valid(input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)) == True{} : Bool}) ->
  Convolved(~custom,~Logical,~original,
    PublicFrame.reference_context(~Logical,logical,activation,input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)),output_extents(input),
    TensorConvolution.input_layout(~custom,activation,source_tensor(model,other,input_extents(input),fill,input_workers),
      Buffer.pack(~F32,weights),weight_extents(input),Buffer.pack(~F32,bias),bias_extents(input),Tensors.first_failure(Tensors.Usable{},Tensors.Usable{}),U32.max(weight_workers,bias_workers),stride(input),padding(input))):
  +weight_parameter = parameter_of(~WeightContext,~weight_value,weight_context,weight_extents(input),weights,weight_facts)
  +bias_parameter = parameter_of(~BiasContext,~bias_value,bias_context,bias_extents(input),bias,bias_facts)
  +weight_listed = parameter_listed(~WeightContext,~weight_value,weight_context,weight_extents(input),weights,weight_facts)
  +bias_listed = parameter_listed(~BiasContext,~bias_value,bias_context,bias_extents(input),bias,bias_facts)
  %weight_listed :
    Convolved(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,_,bias_values(~BiasContext,~bias_value,bias_context,input)),output_extents(input),
      TensorConvolution.input_layout(~custom,activation,source_tensor(model,other,input_extents(input),fill,input_workers),
        Buffer.pack(~F32,weights),weight_extents(input),Buffer.pack(~F32,bias),bias_extents(input),Tensors.Usable{},U32.max(weight_workers,bias_workers),stride(input),padding(input)))
  %bias_listed :
    Convolved(~custom,~Logical,~original,PublicFrame.reference_context(~Logical,logical,activation,input,Parameters.parameter_values(Buffer.pack(~F32,weights),weight_parameter),_),output_extents(input),
      TensorConvolution.input_layout(~custom,activation,source_tensor(model,other,input_extents(input),fill,input_workers),
        Buffer.pack(~F32,weights),weight_extents(input),Buffer.pack(~F32,bias),bias_extents(input),Tensors.Usable{},U32.max(weight_workers,bias_workers),stride(input),padding(input)))
  parameters_contract(~custom,~Logical,~original,logical,activation,input,model,other,fill,input_workers,U32.max(weight_workers,bias_workers),contract,supported,
    Buffer.pack(~F32,weights),Buffer.pack(~F32,bias),weight_parameter,bias_parameter,
    %Equal.sym(List<&2,F32>,Parameters.parameter_values(Buffer.pack(~F32,weights),weight_parameter),weight_values(~WeightContext,~weight_value,weight_context,input),weight_listed) :
      {PublicFrame.frame_valid(input,_,Parameters.parameter_values(Buffer.pack(~F32,bias),bias_parameter)) == True{} : Bool}
    %Equal.sym(List<&2,F32>,Parameters.parameter_values(Buffer.pack(~F32,bias),bias_parameter),bias_values(~BiasContext,~bias_value,bias_context,input),bias_listed) :
      {PublicFrame.frame_valid(input,weight_values(~WeightContext,~weight_value,weight_context,input),_) == True{} : Bool}
    validated)

# The bias arrives with the weights already in storage.
def weighted(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,~WeightContext: Data,~weight_value: WeightContext -> U32 -> F32,
  ~BiasContext: Data,~bias_value: BiasContext -> U32 -> F32,+logical: Logical,+weight_context: WeightContext,+bias_context: BiasContext,
  +activation: Activations.Activation,+input: PublicFrame.NativeInput,+model: Model.Model<F32>,+other: Bands.Geometry,+fill: F32,+input_workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,model),other,input_extents(input),fill,Tensors.Usable{},input_workers}),
  +supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +weights: Buffer.Model<F32>,+weight_facts: GatherValues.Facts(~F32,~WeightContext,~weight_value,weight_context,weight_extents(input),Buffer.content(~F32,weights)),+weight_workers: U32,
  +bias_bands: Model.Model<F32>,+bias_geometry: Bands.Geometry,+bias_fill: F32,+bias_workers: U32,
  +bias_contract: Global.Contract(~F32,~BiasContext,~bias_value,bias_context,Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,bias_extents(input),bias_fill,Tensors.Usable{},bias_workers}),
  +validated: {PublicFrame.frame_valid(input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)) == True{} : Bool}) ->
  Convolved(~custom,~Logical,~original,
    PublicFrame.reference_context(~Logical,logical,activation,input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)),output_extents(input),
    Tensors.with_whole(~F32,~Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,bias_extents(input),bias_fill,Tensors.Usable{},bias_workers},
      bias => bias_shape => bias_fill => bias_status => bias_workers =>
        TensorConvolution.input_layout(~custom,activation,source_tensor(model,other,input_extents(input),fill,input_workers),
          Buffer.pack(~F32,weights),weight_extents(input),bias,bias_shape,Tensors.first_failure(Tensors.Usable{},bias_status),U32.max(weight_workers,bias_workers),stride(input),padding(input)))):
  WholeLeaf.with_whole(~F32,~BiasContext,~bias_value,~Tensors.Tensor<F32>,
    ~(result => Convolved(~custom,~Logical,~original,
      PublicFrame.reference_context(~Logical,logical,activation,input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)),output_extents(input),result)),
    ~(bias => bias_shape => bias_fill => bias_status => bias_workers =>
      TensorConvolution.input_layout(~custom,activation,source_tensor(model,other,input_extents(input),fill,input_workers),
        Buffer.pack(~F32,weights),weight_extents(input),bias,bias_shape,Tensors.first_failure(Tensors.Usable{},bias_status),U32.max(weight_workers,bias_workers),stride(input),padding(input))),
    bias_context,bias_bands,bias_geometry,bias_extents(input),bias_fill,bias_workers,bias_contract,
    bias => bias_facts => parameterized(~custom,~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,
      activation,input,model,other,fill,input_workers,contract,supported,weights,weight_facts,weight_workers,bias,bias_facts,bias_workers,validated))

def parameters_arrived(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,~WeightContext: Data,~weight_value: WeightContext -> U32 -> F32,
  ~BiasContext: Data,~bias_value: BiasContext -> U32 -> F32,+logical: Logical,+weight_context: WeightContext,+bias_context: BiasContext,
  +activation: Activations.Activation,+input: PublicFrame.NativeInput,+model: Model.Model<F32>,+other: Bands.Geometry,+fill: F32,+input_workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{Model.pack(~F32,model),other,input_extents(input),fill,Tensors.Usable{},input_workers}),
  +weight_bands: Model.Model<F32>,+weight_geometry: Bands.Geometry,+weight_fill: F32,+weight_workers: U32,
  +weight_contract: Global.Contract(~F32,~WeightContext,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,weight_extents(input),weight_fill,Tensors.Usable{},weight_workers}),
  +bias_bands: Model.Model<F32>,+bias_geometry: Bands.Geometry,+bias_fill: F32,+bias_workers: U32,
  +bias_contract: Global.Contract(~F32,~BiasContext,~bias_value,bias_context,Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,bias_extents(input),bias_fill,Tensors.Usable{},bias_workers}),
  +supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +validated: {PublicFrame.frame_valid(input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)) == True{} : Bool}) ->
  Convolved(~custom,~Logical,~original,
    PublicFrame.reference_context(~Logical,logical,activation,input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)),output_extents(input),
    TensorConvolution.conv2d(~custom,activation,source_tensor(model,other,input_extents(input),fill,input_workers),
      Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,weight_extents(input),weight_fill,Tensors.Usable{},weight_workers},
      Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,bias_extents(input),bias_fill,Tensors.Usable{},bias_workers},stride(input),padding(input))):
  WholeLeaf.with_whole(~F32,~WeightContext,~weight_value,~Tensors.Tensor<F32>,
    ~(result => Convolved(~custom,~Logical,~original,
      PublicFrame.reference_context(~Logical,logical,activation,input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)),output_extents(input),result)),
    ~(weights => weight_shape => weight_fill => weight_status => weight_workers =>
      Tensors.with_whole(~F32,~Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,bias_extents(input),bias_fill,Tensors.Usable{},bias_workers},
        bias => bias_shape => bias_fill => bias_status => bias_workers =>
          TensorConvolution.input_layout(~custom,activation,source_tensor(model,other,input_extents(input),fill,input_workers),
            weights,weight_shape,bias,bias_shape,Tensors.first_failure(weight_status,bias_status),U32.max(weight_workers,bias_workers),stride(input),padding(input)))),
    weight_context,weight_bands,weight_geometry,weight_extents(input),weight_fill,weight_workers,weight_contract,
    weights => weight_facts => weighted(~custom,~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,
      activation,input,model,other,fill,input_workers,contract,supported,weights,weight_facts,weight_workers,bias_bands,bias_geometry,bias_fill,bias_workers,bias_contract,validated))

def observed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,~WeightContext: Data,~weight_value: WeightContext -> U32 -> F32,
  ~BiasContext: Data,~bias_value: BiasContext -> U32 -> F32,+logical: Logical,+weight_context: WeightContext,+bias_context: BiasContext,
  +activation: Activations.Activation,+input: PublicFrame.NativeInput,-bands: Bands.Bands(~F32),+layout: Bands.Geometry,+fill: F32,+input_workers: U32,
  observation: Model.Observation<F32,bands>,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{bands,layout,input_extents(input),fill,Tensors.Usable{},input_workers}),
  +weight_bands: Model.Model<F32>,+weight_geometry: Bands.Geometry,+weight_fill: F32,+weight_workers: U32,
  +weight_contract: Global.Contract(~F32,~WeightContext,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,weight_extents(input),weight_fill,Tensors.Usable{},weight_workers}),
  +bias_bands: Model.Model<F32>,+bias_geometry: Bands.Geometry,+bias_fill: F32,+bias_workers: U32,
  +bias_contract: Global.Contract(~F32,~BiasContext,~bias_value,bias_context,Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,bias_extents(input),bias_fill,Tensors.Usable{},bias_workers}),
  +supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +validated: {PublicFrame.frame_valid(input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)) == True{} : Bool}) ->
  Convolved(~custom,~Logical,~original,
    PublicFrame.reference_context(~Logical,logical,activation,input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)),output_extents(input),
    TensorConvolution.conv2d(~custom,activation,Tensors.Tensor{bands,layout,input_extents(input),fill,Tensors.Usable{},input_workers},
      Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,weight_extents(input),weight_fill,Tensors.Usable{},weight_workers},
      Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,bias_extents(input),bias_fill,Tensors.Usable{},bias_workers},stride(input),padding(input))):
  Model.Observation{+model,+equation} = observation
  %equation : Convolved(~custom,~Logical,~original,
    PublicFrame.reference_context(~Logical,logical,activation,input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)),output_extents(input),
    TensorConvolution.conv2d(~custom,activation,Tensors.Tensor{_,layout,input_extents(input),fill,Tensors.Usable{},input_workers},
      Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,weight_extents(input),weight_fill,Tensors.Usable{},weight_workers},
      Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,bias_extents(input),bias_fill,Tensors.Usable{},bias_workers},stride(input),padding(input)))
  parameters_arrived(~custom,~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,input,model,layout,fill,input_workers,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,model),bands,equation) :
      Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{_,layout,input_extents(input),fill,Tensors.Usable{},input_workers}) contract,
    weight_bands,weight_geometry,weight_fill,weight_workers,weight_contract,bias_bands,bias_geometry,bias_fill,bias_workers,bias_contract,supported,validated)

# The public conv2d of a Ready input with Normal state and complete values,
# whatever its layout, and weights and bias of any layout with their own
# values. When the native record names the call, the result is Ready in the
# record's output extents and Normal, and every output cell holds the ordered convolution value of those
# weights and bias.
def conv2d(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,~WeightContext: Data,~weight_value: WeightContext -> U32 -> F32,
  ~BiasContext: Data,~bias_value: BiasContext -> U32 -> F32,+logical: Logical,+weight_context: WeightContext,+bias_context: BiasContext,
  +activation: Activations.Activation,+input: PublicFrame.NativeInput,bands: Bands.Bands(~F32),+layout: Bands.Geometry,+fill: F32,+input_workers: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{bands,layout,input_extents(input),fill,Tensors.Usable{},input_workers}),
  +weight_bands: Model.Model<F32>,+weight_geometry: Bands.Geometry,+weight_fill: F32,+weight_workers: U32,
  +weight_contract: Global.Contract(~F32,~WeightContext,~weight_value,weight_context,Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,weight_extents(input),weight_fill,Tensors.Usable{},weight_workers}),
  +bias_bands: Model.Model<F32>,+bias_geometry: Bands.Geometry,+bias_fill: F32,+bias_workers: U32,
  +bias_contract: Global.Contract(~F32,~BiasContext,~bias_value,bias_context,Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,bias_extents(input),bias_fill,Tensors.Usable{},bias_workers}),
  +supported: {PublicFrame.machine(input,Default.limits()) == True{} : Bool},
  +validated: {PublicFrame.frame_valid(input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)) == True{} : Bool}) ->
  Convolved(~custom,~Logical,~original,
    PublicFrame.reference_context(~Logical,logical,activation,input,weight_values(~WeightContext,~weight_value,weight_context,input),bias_values(~BiasContext,~bias_value,bias_context,input)),output_extents(input),
    TensorConvolution.conv2d(~custom,activation,Tensors.Tensor{bands,layout,input_extents(input),fill,Tensors.Usable{},input_workers},
      Tensors.Tensor{Model.pack(~F32,weight_bands),weight_geometry,weight_extents(input),weight_fill,Tensors.Usable{},weight_workers},
      Tensors.Tensor{Model.pack(~F32,bias_bands),bias_geometry,bias_extents(input),bias_fill,Tensors.Usable{},bias_workers},stride(input),padding(input))):
  observed(~custom,~Logical,~original,~WeightContext,~weight_value,~BiasContext,~bias_value,logical,weight_context,bias_context,activation,input,bands,layout,fill,input_workers,
    Model.observe(~F32,bands),contract,weight_bands,weight_geometry,weight_fill,weight_workers,weight_contract,bias_bands,bias_geometry,bias_fill,bias_workers,bias_contract,supported,validated)

# A failed input or parameter stays failed, with the input's owner and the
# larger worker count.
def failed(~custom: F32 -> F32,+activation: Activations.Activation,input: Bands.Bands(~F32),+layout: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+input_status: Tensors.Status,+input_workers: U32,
  -weights: Bands.Bands(~F32),+weight_geometry: Bands.Geometry,+weight_shape: List<&2,U32>,+weight_fill: F32,+weight_status: Tensors.Status,+weight_workers: U32,
  -bias: Bands.Bands(~F32),+bias_geometry: Bands.Geometry,+bias_shape: List<&2,U32>,+bias_fill: F32,+bias_status: Tensors.Status,+bias_workers: U32,+stride: U32,+padding: U32,+problem: Tensors.Problem,
  failure: {Tensors.first_failure(input_status,Tensors.first_failure(weight_status,bias_status)) == Tensors.Failed{problem} : Tensors.Status}) ->
  {TensorConvolution.conv2d(~custom,activation,Tensors.Tensor{input,layout,shape,fill,input_status,input_workers},
      Tensors.Tensor{weights,weight_geometry,weight_shape,weight_fill,weight_status,weight_workers},Tensors.Tensor{bias,bias_geometry,bias_shape,bias_fill,bias_status,bias_workers},stride,padding) ==
    Tensors.Tensor{input,layout,shape,fill,Tensors.Failed{problem},U32.max(input_workers,U32.max(weight_workers,bias_workers))} : Tensors.Tensor<F32>}:
  %Equal.sym(Tensors.Status,Tensors.first_failure(input_status,Tensors.first_failure(weight_status,bias_status)),Tensors.Failed{problem},failure) :
    {TensorConvolution.band_status(~custom,activation,_,layout,input,shape,fill,
        Tensors.whole_storage(~F32,Bands.Geometry{1,1,1,Tensors.count_or_zero(weight_shape)},weight_fill,
          Tensors.arrive(~F32,Bands.Geometry{1,1,1,Tensors.count_or_zero(weight_shape)},Bands.unsplit(Bands.Geometry{1,1,1,Tensors.count_or_zero(weight_shape)}),
            Tensors.Tensor{weights,weight_geometry,weight_shape,weight_fill,weight_status,weight_workers})),weight_shape,
        Tensors.whole_storage(~F32,Bands.Geometry{1,1,1,Tensors.count_or_zero(bias_shape)},bias_fill,
          Tensors.arrive(~F32,Bands.Geometry{1,1,1,Tensors.count_or_zero(bias_shape)},Bands.unsplit(Bands.Geometry{1,1,1,Tensors.count_or_zero(bias_shape)}),
            Tensors.Tensor{bias,bias_geometry,bias_shape,bias_fill,bias_status,bias_workers})),bias_shape,
        stride,padding,U32.max(input_workers,U32.max(weight_workers,bias_workers))) ==
      Tensors.Tensor{input,layout,shape,fill,Tensors.Failed{problem},U32.max(input_workers,U32.max(weight_workers,bias_workers))} : Tensors.Tensor<F32>}
  {==}

# A shape that names no convolution, or parameters that do not fit it, fail
# as an invalid argument.
def invalid_geometry(~custom: F32 -> F32,+activation: Activations.Activation,input: Bands.Bands(~F32),+layout: Bands.Geometry,weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,
  +shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {TensorConvolution.band_checked(~custom,activation,None{},Tensors.Tensor{input,layout,shape,fill,Tensors.Usable{},workers},weights,bias,shape,fill,workers) ==
    Tensors.Tensor{input,layout,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidArgument{shape}}},workers} : Tensors.Tensor<F32>}:
  {==}

def invalid_parameters(~custom: F32 -> F32,+activation: Activations.Activation,geometry: TensorConvolution.Geometry,input: Bands.Bands(~F32),+layout: Bands.Geometry,
  weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {TensorConvolution.band_supported(~custom,activation,False{},geometry,Tensors.Tensor{input,layout,shape,fill,Tensors.Usable{},workers},weights,bias,shape,fill,workers) ==
    Tensors.Tensor{input,layout,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidArgument{shape}}},workers} : Tensors.Tensor<F32>}:
  {==}

# A shape the machine guards reject fails as an invalid shape.
def rejected_machine(~custom: F32 -> F32,+activation: Activations.Activation,geometry: TensorConvolution.Geometry,input: Bands.Bands(~F32),+layout: Bands.Geometry,
  weights: Storage.Buffer<F32>,bias: Storage.Buffer<F32>,+shape: List<&2,U32>,+fill: F32,+workers: U32) ->
  {TensorConvolution.band_machine(~custom,activation,False{},geometry,Tensors.Tensor{input,layout,shape,fill,Tensors.Usable{},workers},weights,bias,shape,fill,workers) ==
    Tensors.Tensor{input,layout,shape,fill,Tensors.Failed{Tensors.Problem{"conv2d",Tensors.InvalidShape{shape}}},workers} : Tensors.Tensor<F32>}:
  {==}
