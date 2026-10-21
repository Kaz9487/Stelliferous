# Source-level FP32 softmax: ordered max, library exp, the sum of math_sum.bend, division.
import Base
import ../tensor_f32.bend as F
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../tensor_band.bend as Bands
import ../math_fp32.bend as Arithmetic
import ../math_sum.bend as Sum
import ../storage_buffer.bend as Storage
import ./tensor_contract.bend as Contract
import ./tensor_copy_contract.bend as CopyContract
import ./tensor_reshape_values.bend as ReshapeValues
import ./tensor_reduce_values.bend as Reduce
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_operations_refinement.bend as OperationLaws
import ./tensor_ready_flow.bend as ReadyFlow
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_model.bend as Model
import ./tensor_band_map.bend as BandMap
import ./tensor_band_normal_unary.bend as NormalUnary
import ./tensor_band_combine_leaf.bend as Leaf
import ./tensor_broadcast_public.bend as Broadcast
import ./tensor_combine_public.bend as CombinePublic
import ./tensor_combine_broadcast.bend as CombineBroadcast
import ./tensor_unit_broadcast.bend as UnitBroadcast
import ./storage_shape.bend as Shape
import ./tensor_band_concat_execution.bend as Observation

type Softmax<-Logical: Data> is Data:
  Softmax{logical: Logical,shape: List<&2,U32>,axis: U32}

# The extents of a reduction that keeps its axis.
def unit(+shape: List<&2,U32>,+axis: U32) -> List<&2,U32>:
  Layout.with_extent(shape,U32.to_nat(axis),1)

# The reduction of one axis: the other axes kept, the line of the axis folded.
def Fold(~Accumulator: Data,~Context: Data) -> Data:
  Reduce.Lines<Accumulator,Reduce.Ordered<Context>>

def fold_context(~Accumulator: Data,~Context: Data,+initial: Accumulator,+context: Context,+shape: List<&2,U32>,+axis: U32) -> Fold(~Accumulator,~Context):
  +kept = Reduce.kept_axes(shape,[axis])
  +reduced = Reduce.reduced_axes(shape,[axis])
  Reduce.Lines{initial,Reduce.source(~Context,context,shape,kept,reduced),ShapeLaws.elements(Layout.extents_at(shape,reduced))}

def fold_value(~Accumulator: Data,~Context: Data,~reader: Context -> U32 -> F32,~fold: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,
  context: Fold(~Accumulator,~Context),index: U32) -> F32:
  Reduce.value(~F32,~Accumulator,~Reduce.Ordered<Context>,~Reduce.ordered_value(~F32,~Context,~reader),~fold,~finish,context,index)

def same_fold(~Accumulator: Data,~Context: Data,before: Fold(~Accumulator,~Context),after: Fold(~Accumulator,~Context)) -> Data:
  {before == after : Fold(~Accumulator,~Context)}

def same_fold_agrees(~Accumulator: Data,~Context: Data,~reader: Context -> U32 -> F32,~fold: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,
  +before: Fold(~Accumulator,~Context),+after: Fold(~Accumulator,~Context),+related: same_fold(~Accumulator,~Context,before,after),+index: U32) ->
  {Reduce.value(~F32,~Accumulator,~Reduce.Ordered<Context>,~Reduce.ordered_value(~F32,~Context,~reader),~fold,~finish,before,index) ==
    fold_value(~Accumulator,~Context,~reader,~fold,~finish,after,index) : F32}:
  %Equal.sym(Fold(~Accumulator,~Context),before,after,related) : {Reduce.value(~F32,~Accumulator,~Reduce.Ordered<Context>,~Reduce.ordered_value(~F32,~Context,~reader),~fold,~finish,_,index) ==
    fold_value(~Accumulator,~Context,~reader,~fold,~finish,after,index) : F32}
  {==}

def Centered(~Logical: Data) -> Data:
  Leaf.Context<Broadcast.Expanded<Logical>,Broadcast.Expanded<Fold(~F32,~Logical)>>

def Exponential(~Logical: Data) -> Data:
  BandMap.Context<Unit,Centered(~Logical)>

def Normalized(~Logical: Data) -> Data:
  Leaf.Context<Broadcast.Expanded<Exponential(~Logical)>,Broadcast.Expanded<Fold(~Sum.Lanes,~Exponential(~Logical))>>

def centered_context(~Logical: Data,context: Softmax<Logical>) -> Centered(~Logical):
  Softmax{+logical,+shape,+axis} = context
  Leaf.Context{Broadcast.Expanded{logical,Broadcast.expansion(shape,shape)},
    Broadcast.Expanded{fold_context(~F32,~Logical,F32.div(F32.neg(1.0),0.0),logical,shape,axis),Broadcast.expansion(unit(shape,axis),shape)}}

def exponential_context(~Logical: Data,context: Softmax<Logical>) -> Exponential(~Logical):
  BandMap.Context{Unit{},centered_context(~Logical,context)}

def normalized_context(~Logical: Data,context: Softmax<Logical>) -> Normalized(~Logical):
  Softmax{+logical,+shape,+axis} = context
  Leaf.Context{Broadcast.Expanded{exponential_context(~Logical,Softmax{logical,shape,axis}),Broadcast.expansion(shape,shape)},
    Broadcast.Expanded{fold_context(~Sum.Lanes,~Exponential(~Logical),Sum.zero(),exponential_context(~Logical,Softmax{logical,shape,axis}),shape,axis),Broadcast.expansion(unit(shape,axis),shape)}}

# At i: exp(x[i] - m(i)) / s(i). m folds the line of i in increasing index
# from -infinity; s adds the exponentials of the line in eight TwoSum lanes.
def value(~Logical: Data,~original: Logical -> U32 -> F32,context: Softmax<Logical>,index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<Exponential(~Logical)>,~Broadcast.Expanded<Fold(~Sum.Lanes,~Exponential(~Logical))>,~F32.div,~Broadcast.value(~F32,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32)))))),~Broadcast.value(~F32,~Fold(~Sum.Lanes,~Exponential(~Logical)),~fold_value(~Sum.Lanes,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32))))),~Sum.step,~Sum.total)),normalized_context(~Logical,context),index)

# Names for the steps of the public pipeline.
def reduced(~Accumulator: Data,~fold: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,+reduction: String,+operations: F32,+initial: Accumulator,tensor: Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32) -> Tensors.Tensor<F32>:
  F.reshape(Layout.reduce_axes(~F32,~Accumulator,~fold,~finish,reduction,operations,initial,tensor,[axis]),unit(shape,axis))

def centered_call(left: Tensors.Tensor<F32>,right: Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32) -> Tensors.Tensor<F32>:
  F.sub(left,reduced(~F32,~F32.max,~Tensors.same(~F32),"softmax",1.0,F32.div(F32.neg(1.0),0.0),right,shape,axis))

def exponential_call(left: Tensors.Tensor<F32>,right: Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32) -> Tensors.Tensor<F32>:
  F.exp(centered_call(left,right,shape,axis))

def normalized(pair: Tensors.Tensor<F32> & Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32) -> Tensors.Tensor<F32>:
  (left,right) = pair
  F.div(left,reduced(~Sum.Lanes,~Sum.step,~Sum.total,"softmax",6.0,Sum.zero(),right,shape,axis))

def after_first(pair: Tensors.Tensor<F32> & Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32) -> Tensors.Tensor<F32>:
  (left,right) = pair
  normalized(F.copy(exponential_call(left,right,shape,axis)),shape,axis)

def normalized_public(pair: Tensors.Tensor<F32> & Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32) -> Tensors.Tensor<F32>:
  F.then(pair,left => right => F.div(left,reduced(~Sum.Lanes,~Sum.step,~Sum.total,"softmax",6.0,Sum.zero(),right,shape,axis)))

def normalized_call(pair: Tensors.Tensor<F32> & Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32) -> {normalized(pair,shape,axis) == normalized_public(pair,shape,axis) : Tensors.Tensor<F32>}:
  (left,right) = pair
  {==}

def after_first_public(pair: Tensors.Tensor<F32> & Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32) -> Tensors.Tensor<F32>:
  F.then(pair,left => right => normalized_public(F.copy(exponential_call(left,right,shape,axis)),shape,axis))

def after_first_call(pair: Tensors.Tensor<F32> & Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32) -> {after_first(pair,shape,axis) == after_first_public(pair,shape,axis) : Tensors.Tensor<F32>}:
  (left,right) = pair
  normalized_call(F.copy(exponential_call(left,right,shape,axis)),shape,axis)

# A Ready result had Ready steps.
def normalized_owners(+left: Observation.TensorOwnerModel<F32>,+right: Observation.TensorOwnerModel<F32>,+shape: List<&2,U32>,+axis: U32,
  ready: {Contract.tensor_ready(~F32,normalized((Observation.tensor_owner_pack(~F32,left),Observation.tensor_owner_pack(~F32,right)),shape,axis)) == True{} : Bool}) -> {Contract.tensor_ready(~F32,Observation.tensor_owner_pack(~F32,right)) == True{} : Bool}:
  ReadyFlow.reduced(~F32,~Sum.Lanes,~Sum.step,~Sum.total,Observation.tensor_owner_pack(~F32,right),"softmax",6.0,Sum.zero(),[axis],
    ReadyFlow.reshaped(~F32,Layout.reduce_axes(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"softmax",6.0,Sum.zero(),Observation.tensor_owner_pack(~F32,right),[axis]),unit(shape,axis),
      ReadyFlow.combine_right(~F32,~F32.div,Observation.tensor_owner_pack(~F32,left),reduced(~Sum.Lanes,~Sum.step,~Sum.total,"softmax",6.0,Sum.zero(),Observation.tensor_owner_pack(~F32,right),shape,axis),"div",1.0,ready)))

def normalized_observed(-left: Tensors.Tensor<F32>,-right: Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32,
  left_observation: Observation.TensorObservation<F32,left>,right_observation: Observation.TensorObservation<F32,right>,
  ready: {Contract.tensor_ready(~F32,normalized((left,right),shape,axis)) == True{} : Bool}) -> {Contract.tensor_ready(~F32,right) == True{} : Bool}:
  Observation.TensorObservation{+left_model,+left_equation} = left_observation
  Observation.TensorObservation{+right_model,+right_equation} = right_observation
  %right_equation : {Contract.tensor_ready(~F32,_) == True{} : Bool}
  normalized_owners(left_model,right_model,shape,axis,
    %Equal.sym(Tensors.Tensor<F32>,Observation.tensor_owner_pack(~F32,right_model),right,right_equation) : {Contract.tensor_ready(~F32,normalized((Observation.tensor_owner_pack(~F32,left_model),_),shape,axis)) == True{} : Bool}
    %Equal.sym(Tensors.Tensor<F32>,Observation.tensor_owner_pack(~F32,left_model),left,left_equation) : {Contract.tensor_ready(~F32,normalized((_,right),shape,axis)) == True{} : Bool}
    ready)

def normalized_ready(pair: Tensors.Tensor<F32> & Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32,
  ready: {Contract.tensor_ready(~F32,normalized(pair,shape,axis)) == True{} : Bool}) -> {CopyContract.pair_ready(~F32,pair) == True{} : Bool}:
  (left,right) = pair
  normalized_observed(left,right,shape,axis,Observation.observe_tensor(~F32,left),Observation.observe_tensor(~F32,right),ready)

def exponential_owners(+left: Observation.TensorOwnerModel<F32>,+right: Observation.TensorOwnerModel<F32>,+shape: List<&2,U32>,+axis: U32,
  ready: {Contract.tensor_ready(~F32,normalized(F.copy(exponential_call(Observation.tensor_owner_pack(~F32,left),Observation.tensor_owner_pack(~F32,right),shape,axis)),shape,axis)) == True{} : Bool}) -> {Contract.tensor_ready(~F32,exponential_call(Observation.tensor_owner_pack(~F32,left),Observation.tensor_owner_pack(~F32,right),shape,axis)) == True{} : Bool}:
  ReadyFlow.copied(~F32,exponential_call(Observation.tensor_owner_pack(~F32,left),Observation.tensor_owner_pack(~F32,right),shape,axis),normalized_ready(F.copy(exponential_call(Observation.tensor_owner_pack(~F32,left),Observation.tensor_owner_pack(~F32,right),shape,axis)),shape,axis,ready))

def exponential_observed(-left: Tensors.Tensor<F32>,-right: Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32,
  left_observation: Observation.TensorObservation<F32,left>,right_observation: Observation.TensorObservation<F32,right>,
  ready: {Contract.tensor_ready(~F32,normalized(F.copy(exponential_call(left,right,shape,axis)),shape,axis)) == True{} : Bool}) -> {Contract.tensor_ready(~F32,exponential_call(left,right,shape,axis)) == True{} : Bool}:
  Observation.TensorObservation{+left_model,+left_equation} = left_observation
  Observation.TensorObservation{+right_model,+right_equation} = right_observation
  %left_equation : {Contract.tensor_ready(~F32,exponential_call(_,right,shape,axis)) == True{} : Bool}
  %right_equation : {Contract.tensor_ready(~F32,exponential_call(Observation.tensor_owner_pack(~F32,left_model),_,shape,axis)) == True{} : Bool}
  exponential_owners(left_model,right_model,shape,axis,
    %Equal.sym(Tensors.Tensor<F32>,Observation.tensor_owner_pack(~F32,right_model),right,right_equation) : {Contract.tensor_ready(~F32,normalized(F.copy(exponential_call(Observation.tensor_owner_pack(~F32,left_model),_,shape,axis)),shape,axis)) == True{} : Bool}
    %Equal.sym(Tensors.Tensor<F32>,Observation.tensor_owner_pack(~F32,left_model),left,left_equation) : {Contract.tensor_ready(~F32,normalized(F.copy(exponential_call(_,right,shape,axis)),shape,axis)) == True{} : Bool}
    ready)

def exponential_ready(left: Tensors.Tensor<F32>,right: Tensors.Tensor<F32>,+shape: List<&2,U32>,+axis: U32,
  ready: {Contract.tensor_ready(~F32,normalized(F.copy(exponential_call(left,right,shape,axis)),shape,axis)) == True{} : Bool}) -> {Contract.tensor_ready(~F32,exponential_call(left,right,shape,axis)) == True{} : Bool}:
  exponential_observed(left,right,shape,axis,Observation.observe_tensor(~F32,left),Observation.observe_tensor(~F32,right),ready)

# left operation reshape(reduce_axes(fold, initial, right, [axis])): the reduction
# keeps its axis and broadcasts back along it.
def Folded(~Accumulator: Data,~Context: Data,~reader: Context -> U32 -> F32,~operation: F32 -> F32 -> F32,~fold: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,+context: Context,+initial: Accumulator,+shape: List<&2,U32>,+axis: U32,-tensor: Tensors.Tensor<F32>) -> Data:
  CombineBroadcast.Combined(~F32,~Context,~Fold(~Accumulator,~Context),~operation,~reader,~fold_value(~Accumulator,~Context,~reader,~fold,~finish),
    Broadcast.Expanded{context,Broadcast.expansion(shape,shape)},Broadcast.Expanded{fold_context(~Accumulator,~Context,initial,context,shape,axis),Broadcast.expansion(unit(shape,axis),shape)},tensor)

def Extents(-tensor: Tensors.Tensor<F32>,+shape: List<&2,U32>) -> Data:
  {Contract.extents(~F32,tensor) == shape : List<&2,U32>}

def reshaped(~Accumulator: Data,~Context: Data,~reader: Context -> U32 -> F32,~operation: F32 -> F32 -> F32,~fold: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,+context: Context,+name: String,+initial: Accumulator,
  +left: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,+axis: U32,-right: Tensors.Tensor<F32>,
  left_contract: Global.Contract(~F32,~Context,~reader,context,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers)),
  evidence: Global.Shaped(~F32,~Fold(~Accumulator,~Context),~fold_value(~Accumulator,~Context,~reader,~fold,~finish),fold_context(~Accumulator,~Context,initial,context,shape,axis),unit(shape,axis),right)) ->
  Shape.Both<Folded(~Accumulator,~Context,~reader,~operation,~fold,~finish,context,initial,shape,axis,Tensors.combine(~F32,~operation,name,1.0,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),right)),
    Extents(Tensors.combine(~F32,~operation,name,1.0,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),right),shape)>:
  Shape.Both{Shape.Both{+well_formed,Shape.Both{+counted,+coherent}},+values} = left_contract
  Shape.Both{usable,right_contract} = evidence
  Global.Usable{+right_bands,+right_geometry,+right_fill,+right_workers,+equation} = usable
  %equation :
    Shape.Both<Folded(~Accumulator,~Context,~reader,~operation,~fold,~finish,context,initial,shape,axis,Tensors.combine(~F32,~operation,name,1.0,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),_)),
      Extents(Tensors.combine(~F32,~operation,name,1.0,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),_),shape)>
  Shape.Both{
    CombineBroadcast.broadcast(~F32,~Context,~Fold(~Accumulator,~Context),~operation,~Operations.combine_kernel(~F32,~operation),~OperationLaws.scalar_kernel(~F32,~operation),
      ~reader,~fold_value(~Accumulator,~Context,~reader,~fold,~finish),context,fold_context(~Accumulator,~Context,initial,context,shape,axis),name,1.0,
      shape,unit(shape,axis),shape,geometry,right_geometry,fill,right_fill,workers,right_workers,
      Model.pack(~F32,left),Model.pack(~F32,right_bands),UnitBroadcast.broadcast(shape,U32.to_nat(axis)),counted,
      Shape.Both{Shape.Both{well_formed,Shape.Both{counted,coherent}},values},
      %Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,right_bands),right_geometry,unit(shape,axis),right_fill,Tensors.Usable{},right_workers},right,equation) :
        Global.Contract(~F32,~Fold(~Accumulator,~Context),~fold_value(~Accumulator,~Context,~reader,~fold,~finish),fold_context(~Accumulator,~Context,initial,context,shape,axis),_)
      right_contract),
    ReadyFlow.combined_extents(~F32,~Operations.combine_kernel(~F32,~operation),name,1.0,Model.pack(~F32,left),geometry,shape,fill,Tensors.Usable{},workers,
      Model.pack(~F32,right_bands),right_geometry,unit(shape,axis),right_fill,Tensors.Usable{},right_workers,
      UnitBroadcast.broadcast(shape,U32.to_nat(axis)))}

# The reduction reshaped to keep its axis as a unit extent.
def reduced_shaped(~Accumulator: Data,~Context: Data,~reader: Context -> U32 -> F32,~fold: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,+context: Context,+initial: Accumulator,+shape: List<&2,U32>,+axis: U32,tensor: Tensors.Tensor<F32>,
  usable: Global.Usable<F32,Contract.extents(~F32,tensor),tensor>,
  contract: Global.Contract(~F32,~Fold(~Accumulator,~Context),~fold_value(~Accumulator,~Context,~reader,~fold,~finish),fold_context(~Accumulator,~Context,initial,context,shape,axis),tensor),
  accepted: {Contract.tensor_ready(~F32,F.reshape(tensor,unit(shape,axis))) == True{} : Bool}) ->
  Global.Shaped(~F32,~Fold(~Accumulator,~Context),~fold_value(~Accumulator,~Context,~reader,~fold,~finish),fold_context(~Accumulator,~Context,initial,context,shape,axis),unit(shape,axis),F.reshape(tensor,unit(shape,axis))):
  Global.Usable{+bands,+reduced_geometry,+reduced_fill,+reduced_workers,+equation} = usable
  +back = Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),reduced_geometry,Contract.extents(~F32,tensor),reduced_fill,Tensors.Usable{},reduced_workers},tensor,equation)
  %equation : Global.Shaped(~F32,~Fold(~Accumulator,~Context),~fold_value(~Accumulator,~Context,~reader,~fold,~finish),fold_context(~Accumulator,~Context,initial,context,shape,axis),unit(shape,axis),F.reshape(_,unit(shape,axis)))
  ReshapeValues.reshape(~F32,~Fold(~Accumulator,~Context),~fold_value(~Accumulator,~Context,~reader,~fold,~finish),fold_context(~Accumulator,~Context,initial,context,shape,axis),unit(shape,axis),
    bands,reduced_geometry,Contract.extents(~F32,tensor),reduced_fill,reduced_workers,
    %back : Global.Contract(~F32,~Fold(~Accumulator,~Context),~fold_value(~Accumulator,~Context,~reader,~fold,~finish),fold_context(~Accumulator,~Context,initial,context,shape,axis),_) contract,
    %back : {Contract.tensor_ready(~F32,F.reshape(_,unit(shape,axis))) == True{} : Bool} accepted)

def folded(~Accumulator: Data,~Context: Data,~reader: Context -> U32 -> F32,~operation: F32 -> F32 -> F32,~fold: Accumulator -> F32 -> Accumulator,~finish: Accumulator -> F32,+context: Context,+name: String,+reduction: String,+operations: F32,+initial: Accumulator,
  +left: Model.Model<F32>,+right: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,+axis: U32,
  left_contract: Global.Contract(~F32,~Context,~reader,context,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers)),
  right_contract: Global.Contract(~F32,~Context,~reader,context,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers)),
  ready: {Contract.tensor_ready(~F32,Tensors.combine(~F32,~operation,name,1.0,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),
    reduced(~Accumulator,~fold,~finish,reduction,operations,initial,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),shape,axis))) == True{} : Bool}) ->
  Shape.Both<Folded(~Accumulator,~Context,~reader,~operation,~fold,~finish,context,initial,shape,axis,Tensors.combine(~F32,~operation,name,1.0,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),
      reduced(~Accumulator,~fold,~finish,reduction,operations,initial,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),shape,axis))),
    Extents(Tensors.combine(~F32,~operation,name,1.0,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),
      reduced(~Accumulator,~fold,~finish,reduction,operations,initial,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),shape,axis)),shape)>:
  +folded_ready = ReadyFlow.combine_right(~F32,~operation,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),
    reduced(~Accumulator,~fold,~finish,reduction,operations,initial,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),shape,axis),name,1.0,ready)
  +reduction_ready = ReadyFlow.reshaped(~F32,Layout.reduce_axes(~F32,~Accumulator,~fold,~finish,reduction,operations,initial,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),[axis]),unit(shape,axis),folded_ready)
  reshaped(~Accumulator,~Context,~reader,~operation,~fold,~finish,context,name,initial,left,geometry,shape,fill,workers,axis,
    reduced(~Accumulator,~fold,~finish,reduction,operations,initial,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),shape,axis),left_contract,
    reduced_shaped(~Accumulator,~Context,~reader,~fold,~finish,context,initial,shape,axis,Layout.reduce_axes(~F32,~Accumulator,~fold,~finish,reduction,operations,initial,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),[axis]),
      Global.usable(~F32,Layout.reduce_axes(~F32,~Accumulator,~fold,~finish,reduction,operations,initial,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),[axis]),reduction_ready),
      Reduce.relabel(~F32,~Fold(~Accumulator,~Context),~Fold(~Accumulator,~Context),
        ~Reduce.value(~F32,~Accumulator,~Reduce.Ordered<Context>,~Reduce.ordered_value(~F32,~Context,~reader),~fold,~finish),~fold_value(~Accumulator,~Context,~reader,~fold,~finish),
        ~same_fold(~Accumulator,~Context),~same_fold_agrees(~Accumulator,~Context,~reader,~fold,~finish),
        fold_context(~Accumulator,~Context,initial,context,shape,axis),fold_context(~Accumulator,~Context,initial,context,shape,axis),{==},Layout.reduce_axes(~F32,~Accumulator,~fold,~finish,reduction,operations,initial,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),[axis]),
        Reduce.reduce_axes(~F32,~Accumulator,~Context,~reader,~fold,~finish,context,reduction,operations,initial,[axis],right,geometry,shape,fill,workers,right_contract,reduction_ready)),
      folded_ready))

def CenteredContract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,-tensor: Tensors.Tensor<F32>) -> Data:
  Folded(~F32,~Logical,~original,~F32.sub,~F32.max,~Tensors.same(~F32),logical,F32.div(F32.neg(1.0),0.0),shape,axis,tensor)

def ExponentialContract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,-tensor: Tensors.Tensor<F32>) -> Data:
  Global.Contract(~F32,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32))))),exponential_context(~Logical,Softmax{logical,shape,axis}),tensor)

def NormalizedContract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,-tensor: Tensors.Tensor<F32>) -> Data:
  Folded(~Sum.Lanes,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32))))),~F32.div,~Sum.step,~Sum.total,exponential_context(~Logical,Softmax{logical,shape,axis}),Sum.zero(),shape,axis,tensor)

def SoftmaxContract(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,-tensor: Tensors.Tensor<F32>) -> Data:
  Global.Contract(~F32,~Softmax<Logical>,~value(~Logical,~original),Softmax{logical,shape,axis},tensor)

def Normalizing(~Logical: Data,before: Normalized(~Logical),after: Softmax<Logical>) -> Data:
  {normalized_context(~Logical,after) == before : Normalized(~Logical)}

def normalizing_agrees(~Logical: Data,~original: Logical -> U32 -> F32,+before: Normalized(~Logical),+after: Softmax<Logical>,+related: Normalizing(~Logical,before,after),+index: U32) ->
  {Leaf.value(~F32,~Broadcast.Expanded<Exponential(~Logical)>,~Broadcast.Expanded<Fold(~Sum.Lanes,~Exponential(~Logical))>,~F32.div,~Broadcast.value(~F32,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32)))))),~Broadcast.value(~F32,~Fold(~Sum.Lanes,~Exponential(~Logical)),~fold_value(~Sum.Lanes,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32))))),~Sum.step,~Sum.total)),before,index) == value(~Logical,~original,after,index) : F32}:
  %related : {Leaf.value(~F32,~Broadcast.Expanded<Exponential(~Logical)>,~Broadcast.Expanded<Fold(~Sum.Lanes,~Exponential(~Logical))>,~F32.div,~Broadcast.value(~F32,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32)))))),~Broadcast.value(~F32,~Fold(~Sum.Lanes,~Exponential(~Logical)),~fold_value(~Sum.Lanes,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32))))),~Sum.step,~Sum.total)),_,index) == value(~Logical,~original,after,index) : F32}
  {==}

# The normalized values are the softmax values at its context.
def relabeled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,-tensor: Tensors.Tensor<F32>,
  usable: Global.Usable<F32,shape,tensor>,contract: NormalizedContract(~Logical,~original,logical,shape,axis,tensor)) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,tensor):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  %equation : SoftmaxContract(~Logical,~original,logical,shape,axis,_)
  Global.relabeled(~F32,~Normalized(~Logical),~Softmax<Logical>,~Leaf.value(~F32,~Broadcast.Expanded<Exponential(~Logical)>,~Broadcast.Expanded<Fold(~Sum.Lanes,~Exponential(~Logical))>,~F32.div,~Broadcast.value(~F32,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32)))))),~Broadcast.value(~F32,~Fold(~Sum.Lanes,~Exponential(~Logical)),~fold_value(~Sum.Lanes,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32))))),~Sum.step,~Sum.total))),~value(~Logical,~original),~Normalizing(~Logical),~normalizing_agrees(~Logical,~original),
    normalized_context(~Logical,Softmax{logical,shape,axis}),Softmax{logical,shape,axis},{==},Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},
    %Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},tensor,equation) :
      NormalizedContract(~Logical,~original,logical,shape,axis,_) contract)

def normalized_parts(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,-tensor: Tensors.Tensor<F32>,
  usable: Global.Usable<F32,Contract.extents(~F32,tensor),tensor>,parts: Shape.Both<NormalizedContract(~Logical,~original,logical,shape,axis,tensor),Extents(tensor,shape)>) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,tensor):
  Shape.Both{contract,+extents} = parts
  relabeled(~Logical,~original,logical,shape,axis,tensor,%extents : Global.Usable<F32,_,tensor> usable,contract)

def normalized_models(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,
  +left: Model.Model<F32>,+right: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  left_contract: ExponentialContract(~Logical,~original,logical,shape,axis,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers)),
  right_contract: ExponentialContract(~Logical,~original,logical,shape,axis,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers)),
  +ready: {Contract.tensor_ready(~F32,normalized((CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),CombinePublic.tensor(~F32,right,geometry,shape,fill,workers)),shape,axis)) == True{} : Bool}) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,normalized((CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),CombinePublic.tensor(~F32,right,geometry,shape,fill,workers)),shape,axis)):
  normalized_parts(~Logical,~original,logical,shape,axis,normalized((CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),CombinePublic.tensor(~F32,right,geometry,shape,fill,workers)),shape,axis),
    Global.usable(~F32,normalized((CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),CombinePublic.tensor(~F32,right,geometry,shape,fill,workers)),shape,axis),ready),
    folded(~Sum.Lanes,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32))))),~F32.div,~Sum.step,~Sum.total,exponential_context(~Logical,Softmax{logical,shape,axis}),"div","softmax",6.0,Sum.zero(),left,right,geometry,shape,fill,workers,axis,
      left_contract,right_contract,ready))

def copies_models(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,+model: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  evidence: CopyContract.Copied<F32,model>,+contract: ExponentialContract(~Logical,~original,logical,shape,axis,CombinePublic.tensor(~F32,model,geometry,shape,fill,workers)),
  ready: {Contract.tensor_ready(~F32,normalized(F.copy(CombinePublic.tensor(~F32,model,geometry,shape,fill,workers)),shape,axis)) == True{} : Bool}) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,normalized(F.copy(CombinePublic.tensor(~F32,model,geometry,shape,fill,workers)),shape,axis)):
  CopyContract.Copied{+left,+right,+equation,+left_content,+right_content} = evidence
  %equation : SoftmaxContract(~Logical,~original,logical,shape,axis,normalized(Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),_,
    first => second => (Tensors.Tensor{first,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{second,geometry,shape,fill,Tensors.Usable{},workers})),shape,axis))
  normalized_models(~Logical,~original,logical,shape,axis,left,right,geometry,fill,workers,
    CopyContract.kept(~F32,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32))))),exponential_context(~Logical,Softmax{logical,shape,axis}),model,left,geometry,shape,fill,workers,left_content,contract),
    CopyContract.kept(~F32,~Exponential(~Logical),~BandMap.value(~F32,~Unit,~Centered(~Logical),~Access.apply_element(~F32,~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32))))),exponential_context(~Logical,Softmax{logical,shape,axis}),model,right,geometry,shape,fill,workers,right_content,contract),
    %Equal.sym(Bands.Bands(~F32) & Bands.Bands(~F32),(Model.pack(~F32,left),Model.pack(~F32,right)),Bands.copy(~F32,Model.pack(~F32,model)),equation) :
      {Contract.tensor_ready(~F32,normalized(Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),_,
        first => second => (Tensors.Tensor{first,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{second,geometry,shape,fill,Tensors.Usable{},workers})),shape,axis)) == True{} : Bool}
      ready)

def copies_usable(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,-tensor: Tensors.Tensor<F32>,
  usable: Global.Usable<F32,shape,tensor>,contract: ExponentialContract(~Logical,~original,logical,shape,axis,tensor),
  ready: {Contract.tensor_ready(~F32,normalized(F.copy(tensor),shape,axis)) == True{} : Bool}) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,normalized(F.copy(tensor),shape,axis)):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  %equation : SoftmaxContract(~Logical,~original,logical,shape,axis,normalized(F.copy(_),shape,axis))
  copies_models(~Logical,~original,logical,shape,axis,bands,geometry,fill,workers,CopyContract.copied(~F32,bands),
    %Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},tensor,equation) : ExponentialContract(~Logical,~original,logical,shape,axis,_) contract,
    %Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},tensor,equation) : {Contract.tensor_ready(~F32,normalized(F.copy(_),shape,axis)) == True{} : Bool} ready)

def exponential_models(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,+centered: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  contract: CenteredContract(~Logical,~original,logical,shape,axis,CombinePublic.tensor(~F32,centered,geometry,shape,fill,workers)),
  +ready: {Contract.tensor_ready(~F32,normalized(F.copy(F.exp(CombinePublic.tensor(~F32,centered,geometry,shape,fill,workers))),shape,axis)) == True{} : Bool}) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,normalized(F.copy(F.exp(CombinePublic.tensor(~F32,centered,geometry,shape,fill,workers))),shape,axis)):
  copies_usable(~Logical,~original,logical,shape,axis,F.exp(CombinePublic.tensor(~F32,centered,geometry,shape,fill,workers)),
    Global.usable(~F32,F.exp(CombinePublic.tensor(~F32,centered,geometry,shape,fill,workers)),
      ReadyFlow.copied(~F32,F.exp(CombinePublic.tensor(~F32,centered,geometry,shape,fill,workers)),
        normalized_ready(F.copy(F.exp(CombinePublic.tensor(~F32,centered,geometry,shape,fill,workers))),shape,axis,ready))),
    NormalUnary.apply(~F32,~Centered(~Logical),~Arithmetic.exp,~Operations.mapped_lanes(~Arithmetic.exp),~OperationLaws.mapped_lanes(~Arithmetic.exp),~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),~Broadcast.value(~F32,~Fold(~F32,~Logical),~fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32)))),
      centered_context(~Logical,Softmax{logical,shape,axis}),4.5,centered,geometry,shape,fill,workers,contract),
    ready)

def exponential_usable(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,-tensor: Tensors.Tensor<F32>,
  usable: Global.Usable<F32,shape,tensor>,contract: CenteredContract(~Logical,~original,logical,shape,axis,tensor),
  ready: {Contract.tensor_ready(~F32,normalized(F.copy(F.exp(tensor)),shape,axis)) == True{} : Bool}) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,normalized(F.copy(F.exp(tensor)),shape,axis)):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  %equation : SoftmaxContract(~Logical,~original,logical,shape,axis,normalized(F.copy(F.exp(_)),shape,axis))
  exponential_models(~Logical,~original,logical,shape,axis,bands,geometry,fill,workers,
    %Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},tensor,equation) : CenteredContract(~Logical,~original,logical,shape,axis,_) contract,
    %Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},tensor,equation) : {Contract.tensor_ready(~F32,normalized(F.copy(F.exp(_)),shape,axis)) == True{} : Bool} ready)

def exponential_parts(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,-tensor: Tensors.Tensor<F32>,
  usable: Global.Usable<F32,Contract.extents(~F32,tensor),tensor>,parts: Shape.Both<CenteredContract(~Logical,~original,logical,shape,axis,tensor),Extents(tensor,shape)>,
  ready: {Contract.tensor_ready(~F32,normalized(F.copy(F.exp(tensor)),shape,axis)) == True{} : Bool}) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,normalized(F.copy(F.exp(tensor)),shape,axis)):
  Shape.Both{contract,+extents} = parts
  exponential_usable(~Logical,~original,logical,shape,axis,tensor,%extents : Global.Usable<F32,_,tensor> usable,contract,ready)

def exponential(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+axis: U32,
  +left: Model.Model<F32>,+right: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  left_contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,left,geometry,shape,fill,workers)),
  right_contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,right,geometry,shape,fill,workers)),
  +ready: {Contract.tensor_ready(~F32,normalized(F.copy(exponential_call(CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),shape,axis)),shape,axis)) == True{} : Bool}) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,normalized(F.copy(exponential_call(CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),shape,axis)),shape,axis)):
  +centered_ready = ReadyFlow.applied(~F32,~Operations.mapped_lanes(~Arithmetic.exp),centered_call(CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),shape,axis),4.5,
    exponential_ready(CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),shape,axis,ready))
  exponential_parts(~Logical,~original,logical,shape,axis,centered_call(CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),shape,axis),
    Global.usable(~F32,centered_call(CombinePublic.tensor(~F32,left,geometry,shape,fill,workers),CombinePublic.tensor(~F32,right,geometry,shape,fill,workers),shape,axis),centered_ready),
    folded(~F32,~Logical,~original,~F32.sub,~F32.max,~Tensors.same(~F32),logical,"sub","softmax",1.0,F32.div(F32.neg(1.0),0.0),left,right,geometry,shape,fill,workers,axis,left_contract,right_contract,centered_ready),
    ready)

def modeled(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+model: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,+axis: U32,
  evidence: CopyContract.Copied<F32,model>,+contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,model,geometry,shape,fill,workers)),
  ready: {Contract.tensor_ready(~F32,F.softmax(CombinePublic.tensor(~F32,model,geometry,shape,fill,workers),axis)) == True{} : Bool}) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,F.softmax(CombinePublic.tensor(~F32,model,geometry,shape,fill,workers),axis)):
  CopyContract.Copied{+left,+right,+equation,+left_content,+right_content} = evidence
  +call = after_first_call(F.copy(CombinePublic.tensor(~F32,model,geometry,shape,fill,workers)),shape,axis)
  %call : SoftmaxContract(~Logical,~original,logical,shape,axis,_)
  %equation : SoftmaxContract(~Logical,~original,logical,shape,axis,after_first(Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),_,
    first => second => (Tensors.Tensor{first,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{second,geometry,shape,fill,Tensors.Usable{},workers})),shape,axis))
  exponential(~Logical,~original,logical,shape,axis,left,right,geometry,fill,workers,
    CopyContract.kept(~F32,~Logical,~original,logical,model,left,geometry,shape,fill,workers,left_content,contract),
    CopyContract.kept(~F32,~Logical,~original,logical,model,right,geometry,shape,fill,workers,right_content,contract),
    %Equal.sym(Bands.Bands(~F32) & Bands.Bands(~F32),(Model.pack(~F32,left),Model.pack(~F32,right)),Bands.copy(~F32,Model.pack(~F32,model)),equation) :
      {Contract.tensor_ready(~F32,after_first(Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),_,
        first => second => (Tensors.Tensor{first,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{second,geometry,shape,fill,Tensors.Usable{},workers})),shape,axis)) == True{} : Bool}
    %Equal.sym(Tensors.Tensor<F32>,after_first(F.copy(CombinePublic.tensor(~F32,model,geometry,shape,fill,workers)),shape,axis),
        after_first_public(F.copy(CombinePublic.tensor(~F32,model,geometry,shape,fill,workers)),shape,axis),call) :
      {Contract.tensor_ready(~F32,_) == True{} : Bool}
    ready)

def observed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,-bands: Bands.Bands(~F32),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,+axis: U32,
  observation: Model.Observation<F32,bands>,contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  ready: {Contract.tensor_ready(~F32,F.softmax(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},axis)) == True{} : Bool}) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,F.softmax(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},axis)):
  Model.Observation{+model,+equation} = observation
  %equation : SoftmaxContract(~Logical,~original,logical,shape,axis,F.softmax(Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers},axis))
  modeled(~Logical,~original,logical,model,geometry,shape,fill,workers,axis,CopyContract.copied(~F32,model),
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,model),bands,equation) : Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers}) contract,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,model),bands,equation) : {Contract.tensor_ready(~F32,F.softmax(Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers},axis)) == True{} : Bool} ready)

# The premises are the contract of the input bands and a Ready result.
def softmax(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,bands: Bands.Bands(~F32),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,+axis: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  ready: {Contract.tensor_ready(~F32,F.softmax(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},axis)) == True{} : Bool}) ->
  SoftmaxContract(~Logical,~original,logical,shape,axis,F.softmax(Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers},axis)):
  observed(~Logical,~original,logical,bands,geometry,shape,fill,workers,axis,Model.observe(~F32,bands),contract,ready)
