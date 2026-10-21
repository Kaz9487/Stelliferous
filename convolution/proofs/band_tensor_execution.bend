# Tensor-level band convolution: the values and the well formed output of the
# band pipeline.
import Base
import ./band_pipeline_execution.bend as PipelineExecution
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/proofs/storage_shape.bend as Shape
import ../../lib/proofs/storage_tree_model.bend as Tree
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/tensor_band_shape_refinement.bend as BandShape
import ../../lib/proofs/tensor_band_values.bend as BandValues
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_convolution.bend as TensorConvolution
import ./band_pipeline_well_formed.bend as PipelineWellFormed
import ../../lib/kernel_shape.bend as KernelShape
import ../../lib/proofs/tensor_band_map.bend as BandMap
import ../../lib/math_activation.bend as Activations

# The Tensor-valued band_output wrapper reads the leaf schedule.
def TensorValues(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,frame: PipelineExecution.Frame<Logical>,tensor: Tensors.Tensor<F32>) -> Data:
  +configuration = PipelineExecution.settings(~Logical,~original,frame)
  BandMap.LogicalValues(~F32,~PipelineExecution.Frame<Logical>,~PipelineExecution.result_value(~custom,~Logical,~original),PipelineExecution.values_logical(~PipelineExecution.Frame<Logical>,configuration),tensor)

def execution_activation(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Activations.Activation:
  PipelineExecution.Frame{logical,execution_activation,shape,images,execution_parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  execution_activation

def execution_geometry(~Logical: Data,frame: PipelineExecution.Frame<Logical>,batched: Bool) -> TensorConvolution.Geometry:
  PipelineExecution.Frame{logical,execution_activation,shape,images,execution_parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  TensorConvolution.Geometry{images,batched,shape}

def execution_parallel(~Logical: Data,frame: PipelineExecution.Frame<Logical>) -> Bool:
  PipelineExecution.Frame{logical,execution_activation,shape,images,execution_parallel,limits,capacity,weights,bias,we,be,domain,valid,supported,room} = frame
  execution_parallel

def band_output(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  TensorValues(~custom,~Logical,~original,frame,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),Model.pack(~F32,source),TensorConvolution.packed(execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)):
  PipelineExecution.Frame{+logical,+execution_activation,KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding},+images,+execution_parallel,+limits,+capacity,+weight_list,+bias_list,+we,+be,+domain,+valid,+supported,+room} = frame
  +frame = {PipelineExecution.Frame{logical,execution_activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,execution_parallel,limits,capacity,weight_list,bias_list,we,be,domain,valid,supported,room} : PipelineExecution.Frame<Logical>}
  PipelineExecution.execute(~custom,~Logical,~original,frame,source,weights,bias,shape,values,weight_exact,bias_exact)

def band_output_well_formed(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source))) ->
  BandMap.WellFormed(~F32,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),Model.pack(~F32,source),TensorConvolution.packed(execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)):
  PipelineExecution.Frame{+logical,+execution_activation,KernelShape.Dimensions{+ci,+co,+height,+width,+kernel,+stride,+padding},+images,+execution_parallel,+limits,+capacity,+weight_list,+bias_list,+we,+be,+domain,+valid,+supported,+room} = frame
  +frame = {PipelineExecution.Frame{logical,execution_activation,KernelShape.Dimensions{ci,co,height,width,kernel,stride,padding},images,execution_parallel,limits,capacity,weight_list,bias_list,we,be,domain,valid,supported,room} : PipelineExecution.Frame<Logical>}
  PipelineWellFormed.formed_execute(~custom,~Logical,~original,frame,source,weights,bias,shape,values)

def band_output_contract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,+batched: Bool,+workers: U32,
  +source: Model.Model<F32>,+weights: Buffer.Model<F32>,+bias: Buffer.Model<F32>,
  +shape: BandShape.WellFormed<F32,PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)>,
  +values: BandValues.Values(~F32,~Logical,~original,PipelineExecution.convolution_logical(~Logical,frame),PipelineExecution.input_geometry(~Logical,frame),Model.content(~F32,source)),
  weight_exact: {Buffer.tree(~F32,weights) == PipelineExecution.source_tree(~Logical,frame) : Tree.Tree<F32>},
  bias_exact: {Buffer.content(~F32,bias) == PipelineExecution.values_bias(~PipelineExecution.Frame<Logical>,PipelineExecution.settings(~Logical,~original,frame)) : Buffer.Content<F32>}) ->
  Shape.Both<TensorValues(~custom,~Logical,~original,frame,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),Model.pack(~F32,source),TensorConvolution.packed(execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers)),
    BandMap.WellFormed(~F32,TensorConvolution.band_output(~custom,execution_activation(~Logical,frame),execution_geometry(~Logical,frame,batched),execution_parallel(~Logical,frame),Model.pack(~F32,source),TensorConvolution.packed(execution_geometry(~Logical,frame,batched),Buffer.pack(~F32,weights)),Buffer.pack(~F32,bias),workers))>:
  Shape.Both{band_output(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,shape,values,weight_exact,bias_exact),
    band_output_well_formed(~custom,~Logical,~original,frame,batched,workers,source,weights,bias,shape,values)}

# The values and the well formed bands of a band convolution result.
def OwnersContract(~custom: F32 -> F32,~Logical: Data,~original: Logical -> U32 -> F32,+frame: PipelineExecution.Frame<Logical>,-tensor: Tensors.Tensor<F32>) -> Data:
  Shape.Both<TensorValues(~custom,~Logical,~original,frame,tensor),BandMap.WellFormed(~F32,tensor)>
