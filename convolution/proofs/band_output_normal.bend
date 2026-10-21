# The actual band output retains every public axis and the checked batch room.
# Count completion is shared with the other tensor producers.
import Base
import ../specification.bend as ConvSpec
import ../../lib/tensor_convolution.bend as TensorConvolution
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_band.bend as Bands
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/machine_limits.bend as Machine
import ../../lib/default_machine_limits.bend as Default
import ../../lib/proofs/tensor_band_gather.bend as BandGather
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_count.bend as BandCount
import ../../lib/proofs/tensor_shape_refinement.bend as ShapeLaws
import ../../lib/proofs/tensor_count_refinement.bend as Count
import ../../lib/proofs/tensor_count_completion.bend as CountCompletion
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_band_map.bend as BandMap
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/storage_buffer.bend as Storage
import ../../lib/proofs/machine_limit_bounds.bend as Limits
import ./band_pipeline_execution.bend as PipelineExecution
import ./band_output_metadata.bend as OutputMetadata
import ./band_leaf_values.bend as LeafValues
import ./band_caller_guards.bend as Caller
import ./band_tensor_execution.bend as TensorExecution
import ./band_public_frame.bend as PublicFrame

def images(geometry: Bands.Geometry) -> U32:
  Bands.Geometry{images,channels,height,width} = geometry
  images

def extents(batched: Bool,geometry: Bands.Geometry) -> List<&2,U32>:
  Bands.Geometry{images,channels,height,width} = geometry
  TensorConvolution.output_shape(batched,images,channels,height,width)

def shape_matched(+images: U32,batched: Bool) -> Bool:
  match batched:
    case True{}: True{}
    case False{}: U32.is_eq(images,1)

def coherent(+batched: Bool,+geometry: Bands.Geometry,
  matched: {shape_matched(images(geometry),batched) == True{} : Bool}) ->
  {Bands.image_rows(extents(batched,geometry),geometry) == True{} : Bool}:
  match batched geometry:
    case True{} Bands.Geometry{+images,+channels,+height,+width}:
      %Equal.sym(Bool,U32.is_eq(images,images),True{},BandShape.equal_words(images,images,{==})) : {_ && U32.is_eq(channels,channels) && U32.is_eq(height,height) && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(channels,channels),True{},BandShape.equal_words(channels,channels,{==})) : {True{} && _ && U32.is_eq(height,height) && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(height,height),True{},BandShape.equal_words(height,height,{==})) : {True{} && True{} && _ && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(width,width),True{},BandShape.equal_words(width,width,{==})) : {True{} && True{} && True{} && _ == True{} : Bool}
      {==}
    case False{} Bands.Geometry{+images,+channels,+height,+width}:
      %Equal.sym(Bool,U32.is_eq(images,1),True{},matched) : {_ && U32.is_eq(channels,channels) && U32.is_eq(height,height) && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(channels,channels),True{},BandShape.equal_words(channels,channels,{==})) : {True{} && _ && U32.is_eq(height,height) && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(height,height),True{},BandShape.equal_words(height,height,{==})) : {True{} && True{} && _ && U32.is_eq(width,width) == True{} : Bool}
      %Equal.sym(Bool,U32.is_eq(width,width),True{},BandShape.equal_words(width,width,{==})) : {True{} && True{} && True{} && _ == True{} : Bool}
      {==}

def domain_from_words(+images: U32,+ci: Nat,+co: Nat,+height: Nat,+width: Nat,+kernel: Nat,+stride: Nat,+padding: Nat,+limits: Machine.Limits,
  words: LeafValues.Words<ci,co,height,width,kernel,stride,padding>,
  co_positive: {Nat.is_lt(0n,co) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(ConvSpec.axis_size(height,kernel,stride,padding),ConvSpec.axis_size(width,kernel,stride,padding)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  BandGather.Domain<Bands.Geometry{images,U32.from_nat(co),U32.from_nat(ConvSpec.axis_size(height,kernel,stride,padding)),U32.from_nat(ConvSpec.axis_size(width,kernel,stride,padding))},limits,Machine.capacity(limits)>:
  LeafValues.Words{ci_word,+co_word,height_word,width_word,kernel_word,stride_word,padding_word,+oh_word,+ow_word} = words
  +oh = ConvSpec.axis_size(height,kernel,stride,padding)
  +ow = ConvSpec.axis_size(width,kernel,stride,padding)
  +native_room = {
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(co)),co,co_word) : {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(_,Nat.mul(U32.to_nat(U32.from_nat(oh)),U32.to_nat(U32.from_nat(ow))))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(oh)),oh,oh_word) : {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(_,U32.to_nat(U32.from_nat(ow))))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ow)),ow,ow_word) : {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(co,Nat.mul(oh,_))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
    room : {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(U32.from_nat(co)),Nat.mul(U32.to_nat(U32.from_nat(oh)),U32.to_nat(U32.from_nat(ow))))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  BandGather.Domain{Limits.capacity_fits(limits),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(co)),co,co_word) : {Nat.is_lt(0n,_) == True{} : Bool} co_positive,
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(oh)),oh,oh_word) : {Nat.is_lt(0n,_) == True{} : Bool} {==},
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(ow)),ow,ow_word) : {Nat.is_lt(0n,_) == True{} : Bool} {==},
    native_room,native_room}

def domain(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>) ->
  BandGather.Domain<OutputMetadata.bounds_geometry(~Logical,frame),PipelineExecution.convolution_limits(~Logical,frame),Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame))>:
  PipelineExecution.Frame{+logical,activation,KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding},+images,parallel,+limits,capacity,+weights,+bias,we,be,input_domain,+valid,+supported,room} = frame
  +dimensions = {KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding} : KernelShape.Dimensions}
  +problem = Caller.problem(~Logical,~original,logical,dimensions,weights,bias,0n)
  +problem_valid = Caller.valid(~Logical,~original,logical,dimensions,weights,bias,0n,valid)
  domain_from_words(images,ci,co,height,width,kernel,stride,padding,limits,
    Caller.words(limits,problem,problem_valid,Caller.supported(~Logical,~original,logical,limits,dimensions,weights,bias,0n,supported)),
    Caller.outputs_positive(problem,problem_valid),room)

def elements(~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,
  matched: {shape_matched(images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool}) ->
  {ShapeLaws.elements(extents(batched,OutputMetadata.bounds_geometry(~Logical,frame))) == BandGather.elements(OutputMetadata.bounds_geometry(~Logical,frame)) : Nat}:
  BandCount.shape_coherence_elements(extents(batched,OutputMetadata.bounds_geometry(~Logical,frame)),OutputMetadata.bounds_geometry(~Logical,frame),coherent(batched,OutputMetadata.bounds_geometry(~Logical,frame),matched))

def counted(~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,
  matched: {shape_matched(images(OutputMetadata.bounds_geometry(~Logical,frame)),batched) == True{} : Bool},
  limits_exact: {PipelineExecution.convolution_limits(~Logical,frame) == Default.limits() : Machine.Limits}) ->
  Count.Count<ShapeLaws.elements(extents(batched,OutputMetadata.bounds_geometry(~Logical,frame))),Bands.elements(OutputMetadata.bounds_geometry(~Logical,frame))>:
  +output_domain = domain(~Logical,~original,frame)
  Count.Count{
    Equal.trans(Nat,ShapeLaws.elements(extents(batched,OutputMetadata.bounds_geometry(~Logical,frame))),BandGather.elements(OutputMetadata.bounds_geometry(~Logical,frame)),U32.to_nat(Bands.elements(OutputMetadata.bounds_geometry(~Logical,frame))),
      elements(~Logical,frame,batched,matched),
      Equal.sym(Nat,U32.to_nat(Bands.elements(OutputMetadata.bounds_geometry(~Logical,frame))),BandGather.elements(OutputMetadata.bounds_geometry(~Logical,frame)),
        BandCount.exact(OutputMetadata.bounds_geometry(~Logical,frame),PipelineExecution.convolution_limits(~Logical,frame),Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame)),output_domain))),
    %limits_exact : {U32.is_le(Bands.elements(OutputMetadata.bounds_geometry(~Logical,frame)),Machine.capacity(_)) == True{} : Bool}
    BandCount.within(OutputMetadata.bounds_geometry(~Logical,frame),PipelineExecution.convolution_limits(~Logical,frame),Machine.capacity(PipelineExecution.convolution_limits(~Logical,frame)),output_domain)}

# Header evidence is produced from the real constructors before attaching
# the existing owner WellFormed proof. It does not inspect convolution values.
def Header(+batched: Bool,+expected_geometry: Bands.Geometry,tensor: Tensors.Tensor<F32>) -> Data:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,status,workers}:
      Shape.Both<{geometry == expected_geometry : Bands.Geometry},
        Shape.Both<{shape == extents(batched,expected_geometry) : List<&2,U32>},{status == Tensors.Usable{} : Tensors.Status}>>

def band_output_header(~custom: F32 -> F32,~Logical: Data,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+parallel: Bool,+workers: U32,
  -source: Bands.Bands(~F32),-weights: Storage.Buffer<F32>,-bias: Storage.Buffer<F32>) ->
  Header(batched,OutputMetadata.bounds_geometry(~Logical,frame),
    TensorConvolution.band_output(~custom,TensorExecution.execution_activation(~Logical,frame),TensorExecution.execution_geometry(~Logical,frame,batched),parallel,source,TensorConvolution.packed(TensorExecution.execution_geometry(~Logical,frame,batched),weights),bias,workers)):
  PipelineExecution.Frame{logical,activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,old_parallel,limits,capacity,weight_list,bias_list,we,be,input_domain,valid,supported,room} = frame
  Shape.Both{{==},Shape.Both{{==},{==}}}

def shape_receipt(+batched: Bool,+expected_geometry: Bands.Geometry,+geometry: Bands.Geometry,+shape: List<&2,U32>,
  geometry_exact: {geometry == expected_geometry : Bands.Geometry},shape_exact: {shape == extents(batched,expected_geometry) : List<&2,U32>},
  counted: Global.Counted<extents(batched,expected_geometry)>,matched: {shape_matched(images(expected_geometry),batched) == True{} : Bool}) ->
  Shape.Both<Global.Counted<shape>,Global.Coherent(shape,geometry)>:
  %Equal.sym(List<&2,U32>,shape,extents(batched,expected_geometry),shape_exact) : Shape.Both<Global.Counted<_>,Global.Coherent(_,geometry)>
  %Equal.sym(Bands.Geometry,geometry,expected_geometry,geometry_exact) : Shape.Both<Global.Counted<extents(batched,expected_geometry)>,Global.Coherent(extents(batched,expected_geometry),_)>
  Shape.Both{counted,BandCount.shape_coherence_elements(extents(batched,expected_geometry),expected_geometry,coherent(batched,expected_geometry,matched))}

def normal_from_header(+batched: Bool,+expected_geometry: Bands.Geometry,tensor: Tensors.Tensor<F32>,
  header: Header(batched,expected_geometry,tensor),counted: Global.Counted<extents(batched,expected_geometry)>,
  matched: {shape_matched(images(expected_geometry),batched) == True{} : Bool},well_formed: BandMap.WellFormed(~F32,tensor)) -> Global.Normal(~F32,tensor):
  match tensor:
    case Tensors.Tensor{bands,+geometry,+shape,fill,+status,workers}:
      Shape.Both{geometry_exact,Shape.Both{shape_exact,status_exact}} = header
      %Equal.sym(Tensors.Status,status,Tensors.Usable{},status_exact) : Global.Normal(~F32,Tensors.Tensor{bands,geometry,shape,fill,_,workers})
      Shape.Both{well_formed,shape_receipt(batched,expected_geometry,geometry,shape,geometry_exact,shape_exact,counted,matched)}

def usable_from_header(+batched: Bool,+expected_geometry: Bands.Geometry,tensor: Tensors.Tensor<F32>,
  header: Header(batched,expected_geometry,tensor)) -> Global.Usable<F32,extents(batched,expected_geometry),tensor>:
  match tensor:
    case Tensors.Tensor{bands,+geometry,+shape,+fill,+status,+workers}:
      Shape.Both{geometry_exact,Shape.Both{shape_exact,status_exact}} = header
      %shape_exact : Global.Usable<F32,_,Tensors.Tensor{bands,geometry,shape,fill,status,workers}>
      %Equal.sym(Tensors.Status,status,Tensors.Usable{},status_exact) : Global.Usable<F32,shape,Tensors.Tensor{bands,geometry,shape,fill,_,workers}>
      Global.usable(~F32,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},{==})

def completed_counted(+extents: List<&2,U32>,+count: U32,evidence: Count.Count<ShapeLaws.elements(extents),count>) -> Global.Counted<extents>:
  Count.Count{exact,bounded} = evidence
  Global.Counted{count,CountCompletion.count_of(extents,count,exact,bounded)}

def contract_shape(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,-tensor: Tensors.Tensor<F32>,
  contract: PublicFrame.Contract(~custom,~Logical,~original,PipelineExecution.output_context(~Logical,frame),tensor)) -> BandMap.WellFormed(~F32,tensor):
  Shape.Both{values,well_formed} = contract
  well_formed

