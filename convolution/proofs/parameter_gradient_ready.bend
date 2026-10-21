# The public weight and bias gradients of tensors of any layout that hold
# the values of their readers. A Ready result has the Normal state and holds
# the readers' gradient contract at every element, and the tensors the call
# only reads come back holding the same values.
import Base
import ../../lib/tensor.bend as Tensors
import ../../lib/tensor_convolution_gradient.bend as Gradient
import ../../lib/tensor_parameter_gradient.bend as Parameter
import ../../lib/proofs/tensor_contract.bend as Contract
import ../../lib/proofs/tensor_whole_leaf.bend as WholeLeaf
import ../../lib/proofs/tensor_gather_values.bend as GatherValues
import ../../lib/proofs/tensor_global_shape_count.bend as Global
import ../../lib/proofs/tensor_band_model.bend as Model
import ../../lib/proofs/storage_buffer_model.bend as Buffer
import ../../lib/tensor_band.bend as Bands
import ./parameter_gradient_readers.bend as Readers
import ./parameter_gradient_public.bend as Public

def WeightHolds(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+input_context: Input,+gradient_context: Gradient,
  +xs: List<&2,U32>,+ds: List<&2,U32>,+k: U32,+s: U32,+p: U32,+g: U32,result: Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>)) -> Type:
  @accepted: {Contract.tensor_ready(~F32,Public.weight_result(result)) == True{} : Bool} ->
    Public.WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,Parameter.image(xs),Parameter.image(ds),k,s,p,g,xs,ds,result)

# The output gradient arrives with the input already in storage.
def weight_input_stored(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+input_context: Input,+gradient_context: Gradient,
  +x: Buffer.Model<F32>,+xs: List<&2,U32>,+xf: F32,+xw: U32,+x_facts: GatherValues.Facts(~F32,~Input,~input_value,input_context,xs,Buffer.content(~F32,x)),
  +d_bands: Model.Model<F32>,+d_geometry: Bands.Geometry,+ds: List<&2,U32>,+df: F32,+dw: U32,
  +d_contract: Global.Contract(~F32,~Gradient,~gradient_value,gradient_context,Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,ds,df,Tensors.Usable{},dw}),
  +k: U32,+s: U32,+p: U32,+g: U32) ->
  WeightHolds(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,xs,ds,k,s,p,g,
    Tensors.with_whole(~F32,~(Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>)),Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,ds,df,Tensors.Usable{},dw},
      d => ds => df => d_status => dw =>
        +ds = ds
        Parameter.weight_shaped(Parameter.image(xs),Parameter.image(ds),Tensors.Usable{},d_status,k,s,p,g,Buffer.pack(~F32,x),xs,xf,xw,d,ds,df,dw))):
  WholeLeaf.with_whole(~F32,~Gradient,~gradient_value,~(Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>)),
    ~(result => WeightHolds(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,xs,ds,k,s,p,g,result)),
    ~(d => ds => df => d_status => dw =>
      +ds = ds
      Parameter.weight_shaped(Parameter.image(xs),Parameter.image(ds),Tensors.Usable{},d_status,k,s,p,g,Buffer.pack(~F32,x),xs,xf,xw,d,ds,df,dw)),
    gradient_context,d_bands,d_geometry,ds,df,dw,d_contract,
    d => d_facts => accepted => Public.weight_shaped(~Input,~input_value,~Gradient,~gradient_value,Parameter.image(xs),Parameter.image(ds),input_context,gradient_context,k,s,p,g,
      x,xs,xf,xw,x_facts,Readers.image_cells(xs),d,ds,df,dw,d_facts,Readers.image_cells(ds),accepted))

# The weight gradient of an input and an output gradient of any layout. A
# Ready result holds, at every weight, the ordered sum the contract gives
# for the values of the two readers.
def weight(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+input_context: Input,+gradient_context: Gradient,
  +x_bands: Model.Model<F32>,+x_geometry: Bands.Geometry,+xs: List<&2,U32>,+xf: F32,+xw: U32,
  +x_contract: Global.Contract(~F32,~Input,~input_value,input_context,Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,xf,Tensors.Usable{},xw}),
  +d_bands: Model.Model<F32>,+d_geometry: Bands.Geometry,+ds: List<&2,U32>,+df: F32,+dw: U32,
  +d_contract: Global.Contract(~F32,~Gradient,~gradient_value,gradient_context,Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,ds,df,Tensors.Usable{},dw}),
  +k: U32,+s: U32,+p: U32,+g: U32,
  accepted: {Contract.tensor_ready(~F32,Public.weight_result(Parameter.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,xf,Tensors.Usable{},xw},
    Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,ds,df,Tensors.Usable{},dw},k,s,p,g))) == True{} : Bool}) ->
  Public.WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,Parameter.image(xs),Parameter.image(ds),k,s,p,g,xs,ds,
    Parameter.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,xf,Tensors.Usable{},xw},Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,ds,df,Tensors.Usable{},dw},k,s,p,g)):
  holds = WholeLeaf.with_whole(~F32,~Input,~input_value,~(Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>)),
    ~(result => WeightHolds(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,xs,ds,k,s,p,g,result)),
    ~(x => xs => xf => status => xw =>
      +xs = xs
      Tensors.with_whole(~F32,~(Tensors.Tensor<F32> & (Tensors.Tensor<F32> & Tensors.Tensor<F32>)),Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,ds,df,Tensors.Usable{},dw},
        d => ds => df => d_status => dw =>
          +ds = ds
          Parameter.weight_shaped(Parameter.image(xs),Parameter.image(ds),status,d_status,k,s,p,g,x,xs,xf,xw,d,ds,df,dw))),
    input_context,x_bands,x_geometry,xs,xf,xw,x_contract,
    x => x_facts => weight_input_stored(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,x,xs,xf,xw,x_facts,
      d_bands,d_geometry,ds,df,dw,d_contract,k,s,p,g))
  holds(accepted)

def BiasHolds(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+gradient_context: Gradient,+shape: List<&2,U32>,+g: U32,
  result: Tensors.Tensor<F32> & Tensors.Tensor<F32>) -> Type:
  @accepted: {Contract.tensor_ready(~F32,Public.bias_result(result)) == True{} : Bool} ->
    Public.BiasCall(~Gradient,~gradient_value,gradient_context,Parameter.image(shape),g,shape,result)

# The bias gradient of an output gradient of any layout. A Ready result
# holds, at every output, the ordered sum the contract gives for the values
# of the reader.
def bias(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+gradient_context: Gradient,
  +bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Gradient,~gradient_value,gradient_context,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  +g: U32,
  accepted: {Contract.tensor_ready(~F32,Public.bias_result(Parameter.bias(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},g))) == True{} : Bool}) ->
  Public.BiasCall(~Gradient,~gradient_value,gradient_context,Parameter.image(shape),g,shape,Parameter.bias(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},g)):
  holds = WholeLeaf.with_whole(~F32,~Gradient,~gradient_value,~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),
    ~(result => BiasHolds(~Gradient,~gradient_value,gradient_context,shape,g,result)),
    ~(d => shape => fill => status => workers =>
      +shape = shape
      Parameter.bias_shaped(Parameter.image(shape),g,d,shape,fill,status,workers)),
    gradient_context,bands,geometry,shape,fill,workers,contract,
    d => facts => accepted => Public.bias_shaped(~Gradient,~gradient_value,Parameter.image(shape),gradient_context,g,d,shape,fill,workers,facts,Readers.image_cells(shape),accepted))
  holds(accepted)

# The public functions use the group length of the contract.
def public_weight(~Input: Data,~input_value: Input -> U32 -> F32,~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+input_context: Input,+gradient_context: Gradient,
  +x_bands: Model.Model<F32>,+x_geometry: Bands.Geometry,+xs: List<&2,U32>,+xf: F32,+xw: U32,
  +x_contract: Global.Contract(~F32,~Input,~input_value,input_context,Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,xf,Tensors.Usable{},xw}),
  +d_bands: Model.Model<F32>,+d_geometry: Bands.Geometry,+ds: List<&2,U32>,+df: F32,+dw: U32,
  +d_contract: Global.Contract(~F32,~Gradient,~gradient_value,gradient_context,Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,ds,df,Tensors.Usable{},dw}),
  +k: U32,+s: U32,+p: U32,
  accepted: {Contract.tensor_ready(~F32,Public.weight_result(Gradient.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,xf,Tensors.Usable{},xw},
    Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,ds,df,Tensors.Usable{},dw},k,s,p))) == True{} : Bool}) ->
  Public.WeightCall(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,Parameter.image(xs),Parameter.image(ds),k,s,p,Gradient.group_length(),xs,ds,
    Gradient.weight(Tensors.Tensor{Model.pack(~F32,x_bands),x_geometry,xs,xf,Tensors.Usable{},xw},Tensors.Tensor{Model.pack(~F32,d_bands),d_geometry,ds,df,Tensors.Usable{},dw},k,s,p)):
  weight(~Input,~input_value,~Gradient,~gradient_value,input_context,gradient_context,x_bands,x_geometry,xs,xf,xw,x_contract,d_bands,d_geometry,ds,df,dw,d_contract,
    k,s,p,Gradient.group_length(),accepted)

def public_bias(~Gradient: Data,~gradient_value: Gradient -> U32 -> F32,+gradient_context: Gradient,
  +bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Gradient,~gradient_value,gradient_context,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~F32,Public.bias_result(Gradient.bias(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}))) == True{} : Bool}) ->
  Public.BiasCall(~Gradient,~gradient_value,gradient_context,Parameter.image(shape),Gradient.group_length(),shape,
    Gradient.bias(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers})):
  bias(~Gradient,~gradient_value,gradient_context,bands,geometry,shape,fill,workers,contract,Gradient.group_length(),accepted)
