# The global average pool of training. forward and evaluate hold at each
# image and channel the sum of its H * W positions in eight TwoSum lanes
# divided by H * W; backward holds dy / (H * W) read at each position through
# the view the extents of dy stretch to.
import Base
import ../tensor_f32.bend as F
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_band.bend as Bands
import ../tensor_view.bend as Views
import ../tensor_training.bend as Train
import ../math_sum.bend as Sum
import ./tensor_contract.bend as Contract
import ./tensor_ready_flow.bend as Flow
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_model.bend as Model
import ./tensor_band_map.bend as BandMap
import ./tensor_band_normal_unary.bend as NormalUnary
import ./tensor_broadcast_public.bend as Broadcast
import ./tensor_combine_public.bend as CombinePublic
import ./tensor_reduce_values.bend as Reduce
import ./tensor_reshape_values.bend as ReshapeValues
import ./tensor_shape_refinement.bend as ShapeLaws
import ./word_encoding.bend as Words

def Tensor() -> Type:
  Tensors.Tensor<F32>

# ---- Forward and evaluate ----

# The sums of the positions of each image and channel, each divided by H * W.
def mean_context(~Logical: Data,+logical: Logical,+shape: List<&2,U32>) -> BandMap.Context<F32,Reduce.Lines<Sum.Lanes,Reduce.Ordered<Logical>>>:
  BandMap.Context{U32.to_f32(Tensors.count_or_zero(Layout.extents_at(shape,Reduce.reduced_axes(shape,[2,3])))),
    Reduce.Lines{Sum.zero(),Reduce.source(~Logical,logical,shape,Reduce.kept_axes(shape,[2,3]),Reduce.reduced_axes(shape,[2,3])),
      ShapeLaws.elements(Layout.extents_at(shape,Reduce.reduced_axes(shape,[2,3])))}}

def MeanContract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensor()) -> Data:
  Global.Contract(~F32,~BandMap.Context<F32,Reduce.Lines<Sum.Lanes,Reduce.Ordered<Logical>>>,~BandMap.value(~F32,~F32,~Reduce.Lines<Sum.Lanes,Reduce.Ordered<Logical>>,~Bands.scalar_value(~F32,~F32.div),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Logical>,~Reduce.ordered_value(~F32,~Logical,~original),~Sum.step,~Sum.total)),mean_context(~Logical,logical,shape),tensor)

def MeanShaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+target: List<&2,U32>,-tensor: Tensor()) -> Data:
  Global.Shaped(~F32,~BandMap.Context<F32,Reduce.Lines<Sum.Lanes,Reduce.Ordered<Logical>>>,~BandMap.value(~F32,~F32,~Reduce.Lines<Sum.Lanes,Reduce.Ordered<Logical>>,~Bands.scalar_value(~F32,~F32.div),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Logical>,~Reduce.ordered_value(~F32,~Logical,~original),~Sum.step,~Sum.total)),mean_context(~Logical,logical,shape),target,tensor)

# The means, as bands of a model, reshaped.
def pooled_reshaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+target: List<&2,U32>,-mean: Tensor(),+extents: List<&2,U32>,
  usable: Global.Usable<F32,extents,mean>,contract: MeanContract(~Logical,~original,logical,shape,mean),
  accepted: {Contract.tensor_ready(~F32,F.reshape(mean,target)) == True{} : Bool}) ->
  MeanShaped(~Logical,~original,logical,shape,target,F.reshape(mean,target)):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensor(),Tensors.Tensor{Model.pack(~F32,bands),geometry,extents,fill,Tensors.Usable{},workers},mean,equation)
  %equation : MeanShaped(~Logical,~original,logical,shape,target,F.reshape(_,target))
  ReshapeValues.reshape(~F32,~BandMap.Context<F32,Reduce.Lines<Sum.Lanes,Reduce.Ordered<Logical>>>,~BandMap.value(~F32,~F32,~Reduce.Lines<Sum.Lanes,Reduce.Ordered<Logical>>,~Bands.scalar_value(~F32,~F32.div),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Logical>,~Reduce.ordered_value(~F32,~Logical,~original),~Sum.step,~Sum.total)),mean_context(~Logical,logical,shape),target,bands,geometry,extents,fill,workers,
    %back : MeanContract(~Logical,~original,logical,shape,_) contract,
    %back : {Contract.tensor_ready(~F32,F.reshape(_,target)) == True{} : Bool} accepted)

# A Ready pool of a tensor of the contract holds the means.
def pooled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),
  +accepted: {Contract.tensor_ready(~F32,F.reshape(F.mean_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),[2,3]),Train.pooled_extents(shape))) == True{} : Bool}) ->
  MeanShaped(~Logical,~original,logical,shape,Train.pooled_extents(shape),F.reshape(F.mean_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),[2,3]),Train.pooled_extents(shape))):
  +mean_ready = Flow.reshaped(~F32,F.mean_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),[2,3]),Train.pooled_extents(shape),accepted)
  pooled_reshaped(~Logical,~original,logical,shape,Train.pooled_extents(shape),F.mean_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),[2,3]),Contract.extents(~F32,F.mean_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),[2,3])),Global.usable(~F32,F.mean_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),[2,3]),mean_ready),
    Reduce.mean_axes(~Logical,~original,logical,[2,3],bands,geometry,shape,fill,workers,contract,mean_ready),accepted)

def ForwardReady(result: Tensor() & (Train.Saved & Train.Layer)) -> Data:
  (output,rest) = result
  {Contract.tensor_ready(~F32,output) == True{} : Bool}

def Forward(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,result: Tensor() & (Train.Saved & Train.Layer)) -> Data:
  (output,rest) = result
  MeanShaped(~Logical,~original,logical,shape,Train.pooled_extents(shape),output)

def EvaluateReady(result: Tensor() & Train.Layer) -> Data:
  (output,layer) = result
  {Contract.tensor_ready(~F32,output) == True{} : Bool}

def Evaluate(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,result: Tensor() & Train.Layer) -> Data:
  (output,layer) = result
  MeanShaped(~Logical,~original,logical,shape,Train.pooled_extents(shape),output)

def forward_observed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,-bands: Bands.Bands(~F32),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  observation: Model.Observation<F32,bands>,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  ready: ForwardReady(Train.forward(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{}))) ->
  Forward(~Logical,~original,logical,shape,Train.forward(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{})):
  Model.Observation{+model,+equation} = observation
  %equation : Forward(~Logical,~original,logical,shape,Train.forward(Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{}))
  pooled(~Logical,~original,logical,model,geometry,shape,fill,workers,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,model),bands,equation) :
      Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers}) contract,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,model),bands,equation) :
      ForwardReady(Train.forward(Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{})) ready)

# forward of the pool on a tensor of the contract in any layout: a Ready output
# holds at each image and channel the mean of its positions.
def forward(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,bands: Bands.Bands(~F32),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  ready: ForwardReady(Train.forward(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{}))) ->
  Forward(~Logical,~original,logical,shape,Train.forward(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{})):
  forward_observed(~Logical,~original,logical,bands,geometry,shape,fill,workers,Model.observe(~F32,bands),contract,ready)

def evaluate_observed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,-bands: Bands.Bands(~F32),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  observation: Model.Observation<F32,bands>,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  ready: EvaluateReady(Train.evaluate(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{}))) ->
  Evaluate(~Logical,~original,logical,shape,Train.evaluate(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{})):
  Model.Observation{+model,+equation} = observation
  %equation : Evaluate(~Logical,~original,logical,shape,Train.evaluate(Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{}))
  pooled(~Logical,~original,logical,model,geometry,shape,fill,workers,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,model),bands,equation) :
      Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers}) contract,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,model),bands,equation) :
      EvaluateReady(Train.evaluate(Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{})) ready)

# evaluate of the pool holds the same means.
def evaluate(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,bands: Bands.Bands(~F32),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  ready: EvaluateReady(Train.evaluate(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{}))) ->
  Evaluate(~Logical,~original,logical,shape,Train.evaluate(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},Train.AveragePool{})):
  evaluate_observed(~Logical,~original,logical,bands,geometry,shape,fill,workers,Model.observe(~F32,bands),contract,ready)

# ---- Backward ----

# dy / (H * W) read at [N,C,H,W] through the view the extents of dy stretch to.
def Spread(~Logical: Data) -> Data:
  Broadcast.Expanded<BandMap.Context<F32,Logical>>

def spread_context(~Logical: Data,+logical: Logical,+source: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32) -> Spread(~Logical):
  Broadcast.Expanded{BandMap.Context{U32.to_f32((height * width : U32)),logical},Tensors.stretched(Tensors.view_of(source),[images,channels,height,width])}

def SpreadShaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,-tensor: Tensor()) -> Data:
  Global.Shaped(~F32,~Spread(~Logical),~Broadcast.value(~F32,~BandMap.Context<F32,Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~F32.div),~original)),spread_context(~Logical,logical,source,images,channels,height,width),[images,channels,height,width],tensor)

# A Ready broadcast had a Ready source.
def broadcast_source(tensor: Tensor(),+target: List<&2,U32>,ready: {Contract.tensor_ready(~F32,F.broadcast_to(tensor,target)) == True{} : Bool}) ->
  {Contract.tensor_ready(~F32,tensor) == True{} : Bool}:
  Tensors.Tensor{bands,+geometry,+shape,+fill,status,+workers} = tensor
  match status:
    case Tensors.Usable{}: {==}
    case Tensors.Failed{+problem}:
      Empty.absurd({Contract.tensor_ready(~F32,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}) == True{} : Bool},Words.false_true(ready))

# The view a Ready broadcast reads through, and its counted extents.
type Found<-source: List<&2,U32>,-target: List<&2,U32>> is Data:
  Found{view: Views.View,stretched: {Tensors.stretched(Tensors.view_of(source),target) == Some{view} : Maybe<&2,Views.View>},counted: Global.Counted<target>}

def count_found(count: Maybe<&2,U32>,+target: List<&2,U32>,+axes: List<&2,Views.Axis>,+offset: U32,-bands: Bands.Bands(~F32),+geometry: Bands.Geometry,
  +source: List<&2,U32>,+fill: F32,+workers: U32,+equation: {Tensors.count_of(target) == count : Maybe<&2,U32>},
  ready: {Contract.tensor_ready(~F32,Tensors.gather_axes(~F32,"broadcast_to",count,axes,offset,[],bands,geometry,source,target,fill,workers)) == True{} : Bool}) ->
  Global.Counted<target>:
  match count:
    case Some{+found}: Global.Counted{found,equation}
    case None{}: Empty.absurd(Global.Counted<target>,Words.false_true(ready))

def view_found(view: Maybe<&2,Views.View>,+source: List<&2,U32>,+target: List<&2,U32>,-bands: Bands.Bands(~F32),+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  +equation: {Tensors.stretched(Tensors.view_of(source),target) == view : Maybe<&2,Views.View>},
  ready: {Contract.tensor_ready(~F32,Tensors.broadcast_view(~F32,Tensors.Usable{},view,bands,geometry,source,target,fill,workers)) == True{} : Bool}) ->
  Found<source,target>:
  match view:
    case Some{+found}:
      match found:
        case Views.View{+axes,+offset}:
          Found{Views.View{axes,offset},equation,count_found(Tensors.count_of(target),target,axes,offset,bands,geometry,source,fill,workers,{==},ready)}
    case None{}: Empty.absurd(Found<source,target>,Words.false_true(ready))

# dy / (H * W), as bands of a model, broadcast through the view it was found to stretch to.
def spread_found(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+scaled: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~BandMap.Context<F32,Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~F32.div),~original),BandMap.Context{U32.to_f32((height * width : U32)),logical},
    Tensors.Tensor{Model.pack(~F32,scaled),geometry,source,fill,Tensors.Usable{},workers}),
  found: Found<source,[images,channels,height,width]>) ->
  SpreadShaped(~Logical,~original,logical,source,images,channels,height,width,F.broadcast_to(Tensors.Tensor{Model.pack(~F32,scaled),geometry,source,fill,Tensors.Usable{},workers},[images,channels,height,width])):
  Found{+view,+stretched,+counted} = found
  %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(source),[images,channels,height,width]),Some{view},stretched) :
    Global.Shaped(~F32,~Spread(~Logical),~Broadcast.value(~F32,~BandMap.Context<F32,Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~F32.div),~original)),Broadcast.Expanded{BandMap.Context{U32.to_f32((height * width : U32)),logical},_},[images,channels,height,width],
      F.broadcast_to(Tensors.Tensor{Model.pack(~F32,scaled),geometry,source,fill,Tensors.Usable{},workers},[images,channels,height,width]))
  Broadcast.broadcast_to(~F32,~BandMap.Context<F32,Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~F32.div),~original),BandMap.Context{U32.to_f32((height * width : U32)),logical},source,[images,channels,height,width],view,fill,workers,scaled,geometry,
    contract,counted,stretched)

def spread_scaled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,-scaled: Tensor(),usable: Global.Usable<F32,source,scaled>,
  contract: Global.Contract(~F32,~BandMap.Context<F32,Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~F32.div),~original),BandMap.Context{U32.to_f32((height * width : U32)),logical},scaled),
  ready: {Contract.tensor_ready(~F32,F.broadcast_to(scaled,[images,channels,height,width])) == True{} : Bool}) ->
  SpreadShaped(~Logical,~original,logical,source,images,channels,height,width,F.broadcast_to(scaled,[images,channels,height,width])):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensor(),Tensors.Tensor{Model.pack(~F32,bands),geometry,source,fill,Tensors.Usable{},workers},scaled,equation)
  %equation : SpreadShaped(~Logical,~original,logical,source,images,channels,height,width,F.broadcast_to(_,[images,channels,height,width]))
  spread_found(~Logical,~original,logical,source,images,channels,height,width,bands,geometry,fill,workers,
    %back : Global.Contract(~F32,~BandMap.Context<F32,Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~F32.div),~original),BandMap.Context{U32.to_f32((height * width : U32)),logical},_) contract,
    view_found(Tensors.stretched(Tensors.view_of(source),[images,channels,height,width]),source,[images,channels,height,width],Model.pack(~F32,bands),geometry,fill,workers,{==},
      %back : {Contract.tensor_ready(~F32,F.broadcast_to(_,[images,channels,height,width])) == True{} : Bool} ready))

# A Ready pool gradient of dy of the contract, as bands of a model.
def spread(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,+dy: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,dy,geometry,source,fill,workers)),
  +ready: {Contract.tensor_ready(~F32,F.broadcast_to(F.div_scalar(CombinePublic.tensor(~F32,dy,geometry,source,fill,workers),U32.to_f32((height * width : U32))),[images,channels,height,width])) == True{} : Bool}) ->
  SpreadShaped(~Logical,~original,logical,source,images,channels,height,width,F.broadcast_to(F.div_scalar(CombinePublic.tensor(~F32,dy,geometry,source,fill,workers),U32.to_f32((height * width : U32))),[images,channels,height,width])):
  +scaled_ready = broadcast_source(F.div_scalar(CombinePublic.tensor(~F32,dy,geometry,source,fill,workers),U32.to_f32((height * width : U32))),[images,channels,height,width],ready)
  spread_scaled(~Logical,~original,logical,source,images,channels,height,width,F.div_scalar(CombinePublic.tensor(~F32,dy,geometry,source,fill,workers),U32.to_f32((height * width : U32))),Global.usable(~F32,F.div_scalar(CombinePublic.tensor(~F32,dy,geometry,source,fill,workers),U32.to_f32((height * width : U32))),scaled_ready),
    NormalUnary.combine_scalar(~F32,~Logical,~F32.div,~original,logical,1.0,U32.to_f32((height * width : U32)),dy,geometry,source,fill,workers,contract),ready)

def BackwardReady(result: Tensor() & Train.Layer) -> Data:
  (gradient,layer) = result
  {Contract.tensor_ready(~F32,gradient) == True{} : Bool}

def Backward(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,result: Tensor() & Train.Layer) -> Data:
  (gradient,layer) = result
  SpreadShaped(~Logical,~original,logical,source,images,channels,height,width,gradient)

def backward_observed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,-dy: Bands.Bands(~F32),+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  observation: Model.Observation<F32,dy>,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{dy,geometry,source,fill,Tensors.Usable{},workers}),
  ready: BackwardReady(Train.backward(Tensors.Tensor{dy,geometry,source,fill,Tensors.Usable{},workers},Train.Pooled{[images,channels,height,width]},Train.AveragePool{}))) ->
  Backward(~Logical,~original,logical,source,images,channels,height,width,Train.backward(Tensors.Tensor{dy,geometry,source,fill,Tensors.Usable{},workers},Train.Pooled{[images,channels,height,width]},Train.AveragePool{})):
  Model.Observation{+model,+equation} = observation
  %equation : Backward(~Logical,~original,logical,source,images,channels,height,width,Train.backward(Tensors.Tensor{_,geometry,source,fill,Tensors.Usable{},workers},Train.Pooled{[images,channels,height,width]},Train.AveragePool{}))
  spread(~Logical,~original,logical,source,images,channels,height,width,model,geometry,fill,workers,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,model),dy,equation) :
      Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{_,geometry,source,fill,Tensors.Usable{},workers}) contract,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,model),dy,equation) :
      BackwardReady(Train.backward(Tensors.Tensor{_,geometry,source,fill,Tensors.Usable{},workers},Train.Pooled{[images,channels,height,width]},Train.AveragePool{})) ready)

# backward of the pool, after forward saved [N,C,H,W], on dy of the contract in
# any layout: a Ready input gradient holds at each position dy / (H * W) read
# through the view the extents of dy stretch to.
def backward(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: List<&2,U32>,+images: U32,+channels: U32,+height: U32,+width: U32,dy: Bands.Bands(~F32),+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{dy,geometry,source,fill,Tensors.Usable{},workers}),
  ready: BackwardReady(Train.backward(Tensors.Tensor{dy,geometry,source,fill,Tensors.Usable{},workers},Train.Pooled{[images,channels,height,width]},Train.AveragePool{}))) ->
  Backward(~Logical,~original,logical,source,images,channels,height,width,Train.backward(Tensors.Tensor{dy,geometry,source,fill,Tensors.Usable{},workers},Train.Pooled{[images,channels,height,width]},Train.AveragePool{})):
  backward_observed(~Logical,~original,logical,source,images,channels,height,width,dy,geometry,fill,workers,Model.observe(~F32,dy),contract,ready)
