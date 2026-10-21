# Source-level FP32 cross entropy. The rows of the logits pass the centered
# and folded steps of softmax: the maximum of the row is subtracted, then the
# logarithm of the sum of the exponentials of the row. The loss adds the
# products with the target and divides by -N; the gradient divides the
# exponentials minus the target by N.
import Base
import ../tensor_f32.bend as F
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_loss.bend as Loss
import ../tensor_shape.bend as ShapeOps
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../tensor_band.bend as Bands
import ../math_fp32.bend as Arithmetic
import ../math_sum.bend as Sum
import ../storage_buffer.bend as Storage
import ./tensor_contract.bend as Contract
import ./tensor_copy_contract.bend as Copy
import ./tensor_reduce_values.bend as Reduce
import ./tensor_shape_refinement.bend as Shapes
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_ready_flow.bend as Flow
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_model.bend as Model
import ./tensor_band_map.bend as BandMap
import ./tensor_band_normal_unary.bend as NormalUnary
import ./tensor_band_combine_leaf.bend as Leaf
import ./tensor_broadcast_public.bend as Broadcast
import ./tensor_combine_public.bend as CombinePublic
import ./tensor_unit_broadcast.bend as UnitBroadcast
import ./tensor_softmax_values.bend as Softmax
import ./tensor_band_concat_execution.bend as Observation
import ./tensor_loss_values.bend as LossValues
import ./storage_shape.bend as Shape
import ./word_encoding.bend as Words

def Tensor() -> Type:
  Tensors.Tensor<F32>

# ---- What the loss, the gradient and the target hold ----

# The products of the target and the log probabilities, folded in row-major
# order in the lanes of math_sum.bend.
def Products(~Target: Data,~P: Data) -> Data:
  Reduce.Lines<Sum.Lanes,Reduce.Ordered<Leaf.Context<Target,P>>>

def products(~Target: Data,~P: Data,+target: Target,+log: P,+shape: List<&2,U32>) -> Products(~Target,~P):
  Reduce.Lines{Sum.zero(),Reduce.Ordered{Leaf.Context{target,log},None{}},Shapes.elements(shape)}

# The rank-0 loss: the sum of target * p divided by -N.
def LossContract(~Target: Data,~P: Data,~target_value: Target -> U32 -> F32,~p_value: P -> U32 -> F32,+target: Target,+log: P,+images: U32,+shape: List<&2,U32>,
  -tensor: Tensor()) -> Data:
  Global.Contract(~F32,~BandMap.Context<F32,Products(~Target,~P)>,
    ~BandMap.value(~F32,~F32,~Products(~Target,~P),~Bands.scalar_value(~F32,~F32.div),
      ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Leaf.Context<Target,P>>,~Reduce.ordered_value(~F32,~Leaf.Context<Target,P>,~Leaf.value(~F32,~Target,~P,~F32.mul,~target_value,~p_value)),~Sum.step,~Sum.total)),
    BandMap.Context{F32.neg(U32.to_f32(images)),products(~Target,~P,target,log,shape)},tensor)

# The gradient at i: (exp(p[i]) - target[i]) / N.
def GradientContract(~Target: Data,~P: Data,~target_value: Target -> U32 -> F32,~p_value: P -> U32 -> F32,+target: Target,+log: P,+images: U32,-tensor: Tensor()) -> Data:
  Global.Contract(~F32,~BandMap.Context<F32,Leaf.Context<BandMap.Context<Unit,P>,Target>>,
    ~BandMap.value(~F32,~F32,~Leaf.Context<BandMap.Context<Unit,P>,Target>,~Bands.scalar_value(~F32,~F32.div),
      ~Leaf.value(~F32,~BandMap.Context<Unit,P>,~Target,~F32.sub,~BandMap.value(~F32,~Unit,~P,~Access.apply_element(~F32,~Arithmetic.exp),~p_value),~target_value)),
    BandMap.Context{U32.to_f32(images),Leaf.Context{BandMap.Context{Unit{},log},target}},tensor)

# The loss, the gradient, and the target as it was given.
def Results(~Target: Data,~P: Data,~target_value: Target -> U32 -> F32,~p_value: P -> U32 -> F32,+target: Target,+log: P,+images: U32,+shape: List<&2,U32>,
  result: Tensor() & (Tensor() & Tensor())) -> Data:
  (loss,(gradient,returned)) = result
  Shape.Both<LossContract(~Target,~P,~target_value,~p_value,target,log,images,shape,loss),
    Shape.Both<GradientContract(~Target,~P,~target_value,~p_value,target,log,images,gradient),Global.Contract(~F32,~Target,~target_value,target,returned)>>

# ---- The loss and the gradient of the log probabilities ----

# The products of the target and the log probabilities, added and divided by -N.
def loss_part(~Target: Data,~P: Data,~target_value: Target -> U32 -> F32,~p_value: P -> U32 -> F32,+target: Target,+log: P,+images: U32,+shape: List<&2,U32>,
  +product: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,+first: Model.Model<F32>,+pg: Bands.Geometry,+pf: F32,+pw: U32,
  product_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,product,tg,shape,tf,tw)),
  first_contract: Global.Contract(~F32,~P,~p_value,log,CombinePublic.tensor(~F32,first,pg,shape,pf,pw)),
  ready: {Contract.tensor_ready(~F32,F.div_scalar(F.sum(F.mul(CombinePublic.tensor(~F32,product,tg,shape,tf,tw),CombinePublic.tensor(~F32,first,pg,shape,pf,pw))),
    F32.neg(U32.to_f32(images)))) == True{} : Bool}) ->
  LossContract(~Target,~P,~target_value,~p_value,target,log,images,shape,
    F.div_scalar(F.sum(F.mul(CombinePublic.tensor(~F32,product,tg,shape,tf,tw),CombinePublic.tensor(~F32,first,pg,shape,pf,pw))),F32.neg(U32.to_f32(images)))):
  +sum_ready = NormalUnary.scalar_status(~F32,~F32.div,1.0,F.sum(F.mul(CombinePublic.tensor(~F32,product,tg,shape,tf,tw),CombinePublic.tensor(~F32,first,pg,shape,pf,pw))),
    F32.neg(U32.to_f32(images)),ready)
  +product_ready = Flow.reduced_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,F.mul(CombinePublic.tensor(~F32,product,tg,shape,tf,tw),CombinePublic.tensor(~F32,first,pg,shape,pf,pw)),
    "sum",6.0,Sum.zero(),sum_ready)
  +extents = Flow.combined_extents(~F32,~Operations.combine_kernel(~F32,~F32.mul),"mul",1.0,Model.pack(~F32,product),tg,shape,tf,Tensors.Usable{},tw,
    Model.pack(~F32,first),pg,shape,pf,Tensors.Usable{},pw,UnitBroadcast.self_broadcast(shape))
  LossValues.normalized(~Products(~Target,~P),
    ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Leaf.Context<Target,P>>,~Reduce.ordered_value(~F32,~Leaf.Context<Target,P>,~Leaf.value(~F32,~Target,~P,~F32.mul,~target_value,~p_value)),~Sum.step,~Sum.total),
    Loss.Mean{},F32.neg(U32.to_f32(images)),products(~Target,~P,target,log,shape),
    F.sum(F.mul(CombinePublic.tensor(~F32,product,tg,shape,tf,tw),CombinePublic.tensor(~F32,first,pg,shape,pf,pw))),
    Global.usable(~F32,F.sum(F.mul(CombinePublic.tensor(~F32,product,tg,shape,tf,tw),CombinePublic.tensor(~F32,first,pg,shape,pf,pw))),sum_ready),
    LossValues.summed(~Leaf.Context<Target,P>,~Leaf.value(~F32,~Target,~P,~F32.mul,~target_value,~p_value),"sum",Leaf.Context{target,log},shape,
      F.mul(CombinePublic.tensor(~F32,product,tg,shape,tf,tw),CombinePublic.tensor(~F32,first,pg,shape,pf,pw)),
      %extents : Global.Usable<F32,_,F.mul(CombinePublic.tensor(~F32,product,tg,shape,tf,tw),CombinePublic.tensor(~F32,first,pg,shape,pf,pw))>
        Global.usable(~F32,F.mul(CombinePublic.tensor(~F32,product,tg,shape,tf,tw),CombinePublic.tensor(~F32,first,pg,shape,pf,pw)),product_ready),
      CombinePublic.combine(~F32,~Target,~P,~F32.mul,~Operations.combine_kernel(~F32,~F32.mul),~OperationsLaws.scalar_kernel(~F32,~F32.mul),~target_value,~p_value,
        target,log,"mul",1.0,shape,tg,pg,tf,pf,tw,pw,Model.pack(~F32,product),Model.pack(~F32,first),product_contract,first_contract),
      sum_ready))

# The exponentials of the log probabilities as bands, minus the target, divided by N.
def differences(~Target: Data,~P: Data,~target_value: Target -> U32 -> F32,~p_value: P -> U32 -> F32,+target: Target,+log: P,+images: U32,+shape: List<&2,U32>,
  +exponential: Model.Model<F32>,+eg: Bands.Geometry,+ef: F32,+ew: U32,+difference: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,
  contract: Global.Contract(~F32,~BandMap.Context<Unit,P>,~BandMap.value(~F32,~Unit,~P,~Access.apply_element(~F32,~Arithmetic.exp),~p_value),BandMap.Context{Unit{},log},
    CombinePublic.tensor(~F32,exponential,eg,shape,ef,ew)),
  difference_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),
  ready: {Contract.tensor_ready(~F32,F.div_scalar(F.sub(CombinePublic.tensor(~F32,exponential,eg,shape,ef,ew),CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),
    U32.to_f32(images))) == True{} : Bool}) ->
  GradientContract(~Target,~P,~target_value,~p_value,target,log,images,
    F.div_scalar(F.sub(CombinePublic.tensor(~F32,exponential,eg,shape,ef,ew),CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),U32.to_f32(images))):
  +subtracted_ready = NormalUnary.scalar_status(~F32,~F32.div,1.0,
    F.sub(CombinePublic.tensor(~F32,exponential,eg,shape,ef,ew),CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),U32.to_f32(images),ready)
  LossValues.normalized(~Leaf.Context<BandMap.Context<Unit,P>,Target>,
    ~Leaf.value(~F32,~BandMap.Context<Unit,P>,~Target,~F32.sub,~BandMap.value(~F32,~Unit,~P,~Access.apply_element(~F32,~Arithmetic.exp),~p_value),~target_value),
    Loss.Mean{},U32.to_f32(images),Leaf.Context{BandMap.Context{Unit{},log},target},
    F.sub(CombinePublic.tensor(~F32,exponential,eg,shape,ef,ew),CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),
    Global.usable(~F32,F.sub(CombinePublic.tensor(~F32,exponential,eg,shape,ef,ew),CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),subtracted_ready),
    CombinePublic.combine(~F32,~BandMap.Context<Unit,P>,~Target,~F32.sub,~Operations.combine_kernel(~F32,~F32.sub),~OperationsLaws.scalar_kernel(~F32,~F32.sub),
      ~BandMap.value(~F32,~Unit,~P,~Access.apply_element(~F32,~Arithmetic.exp),~p_value),~target_value,
      BandMap.Context{Unit{},log},target,"sub",1.0,shape,eg,tg,ef,tf,ew,tw,Model.pack(~F32,exponential),Model.pack(~F32,difference),contract,difference_contract))

def exponentials(~Target: Data,~P: Data,~target_value: Target -> U32 -> F32,~p_value: P -> U32 -> F32,+target: Target,+log: P,+images: U32,+shape: List<&2,U32>,
  -exponential: Tensor(),usable: Global.Usable<F32,shape,exponential>,
  contract: Global.Contract(~F32,~BandMap.Context<Unit,P>,~BandMap.value(~F32,~Unit,~P,~Access.apply_element(~F32,~Arithmetic.exp),~p_value),BandMap.Context{Unit{},log},exponential),
  +difference: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,
  difference_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),
  ready: {Contract.tensor_ready(~F32,F.div_scalar(F.sub(exponential,CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),U32.to_f32(images))) == True{} : Bool}) ->
  GradientContract(~Target,~P,~target_value,~p_value,target,log,images,F.div_scalar(F.sub(exponential,CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),U32.to_f32(images))):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensor(),CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),exponential,equation)
  %equation : GradientContract(~Target,~P,~target_value,~p_value,target,log,images,F.div_scalar(F.sub(_,CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),U32.to_f32(images)))
  differences(~Target,~P,~target_value,~p_value,target,log,images,shape,bands,geometry,fill,workers,difference,tg,tf,tw,
    %back : Global.Contract(~F32,~BandMap.Context<Unit,P>,~BandMap.value(~F32,~Unit,~P,~Access.apply_element(~F32,~Arithmetic.exp),~p_value),BandMap.Context{Unit{},log},_) contract,
    difference_contract,
    %back : {Contract.tensor_ready(~F32,F.div_scalar(F.sub(_,CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),U32.to_f32(images))) == True{} : Bool} ready)

# The exponentials of the second copy of the log probabilities, minus the second copy of the target, divided by N.
def gradient_part(~Target: Data,~P: Data,~target_value: Target -> U32 -> F32,~p_value: P -> U32 -> F32,+target: Target,+log: P,+images: U32,+shape: List<&2,U32>,
  +second: Model.Model<F32>,+pg: Bands.Geometry,+pf: F32,+pw: U32,+difference: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,
  second_contract: Global.Contract(~F32,~P,~p_value,log,CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),
  difference_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),
  ready: {Contract.tensor_ready(~F32,F.div_scalar(F.sub(F.exp(CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),
    U32.to_f32(images))) == True{} : Bool}) ->
  GradientContract(~Target,~P,~target_value,~p_value,target,log,images,
    F.div_scalar(F.sub(F.exp(CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),U32.to_f32(images))):
  exponentials(~Target,~P,~target_value,~p_value,target,log,images,shape,F.exp(CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),
    Global.usable(~F32,F.exp(CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),{==}),
    NormalUnary.apply(~F32,~P,~Arithmetic.exp,~Operations.mapped_lanes(~Arithmetic.exp),~OperationsLaws.mapped_lanes(~Arithmetic.exp),~p_value,log,4.5,second,pg,shape,pf,pw,second_contract),
    difference,tg,tf,tw,difference_contract,ready)

# Both copies of the log probabilities and of the target as bands of models.
def gradient_models(~Target: Data,~P: Data,~target_value: Target -> U32 -> F32,~p_value: P -> U32 -> F32,+target: Target,+log: P,+images: U32,+shape: List<&2,U32>,
  +first: Model.Model<F32>,+second: Model.Model<F32>,+pg: Bands.Geometry,+pf: F32,+pw: U32,+product: Model.Model<F32>,+difference: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,
  -returned: Tensor(),
  first_contract: Global.Contract(~F32,~P,~p_value,log,CombinePublic.tensor(~F32,first,pg,shape,pf,pw)),
  second_contract: Global.Contract(~F32,~P,~p_value,log,CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),
  product_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,product,tg,shape,tf,tw)),
  difference_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),
  returned_contract: Global.Contract(~F32,~Target,~target_value,target,returned),
  ready: LossValues.Ready(Loss.entropy_gradient(images,(CombinePublic.tensor(~F32,first,pg,shape,pf,pw),CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),
    (CombinePublic.tensor(~F32,product,tg,shape,tf,tw),CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),returned))) ->
  Results(~Target,~P,~target_value,~p_value,target,log,images,shape,
    Loss.entropy_gradient(images,(CombinePublic.tensor(~F32,first,pg,shape,pf,pw),CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),
      (CombinePublic.tensor(~F32,product,tg,shape,tf,tw),CombinePublic.tensor(~F32,difference,tg,shape,tf,tw)),returned)):
  Shape.Both{loss_ok,gradient_ok} = ready
  Shape.Both{
    loss_part(~Target,~P,~target_value,~p_value,target,log,images,shape,product,tg,tf,tw,first,pg,pf,pw,product_contract,first_contract,loss_ok),
    Shape.Both{
      gradient_part(~Target,~P,~target_value,~p_value,target,log,images,shape,second,pg,pf,pw,difference,tg,tf,tw,second_contract,difference_contract,gradient_ok),
      returned_contract}}

# ---- The log probabilities of the centered rows ----

# The centered rows minus the logarithm of the sum of their exponentials,
# broadcast back along each row.
def LogProbabilities(~C: Data) -> Data:
  Leaf.Context<Broadcast.Expanded<C>,Broadcast.Expanded<Softmax.Fold(~Sum.Lanes,~C)>>

def log_probabilities(~C: Data,+centered: C,+shape: List<&2,U32>) -> LogProbabilities(~C):
  Leaf.Context{Broadcast.Expanded{centered,Broadcast.expansion(shape,shape)},
    Broadcast.Expanded{Softmax.fold_context(~Sum.Lanes,~C,Sum.zero(),centered,shape,1),Broadcast.expansion(Softmax.unit(shape,1),shape)}}

# What the results hold when the rows enter centered.
def Logs(~C: Data,~c_value: C -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+centered: C,+target: Target,+images: U32,+shape: List<&2,U32>,
  result: Tensor() & (Tensor() & Tensor())) -> Data:
  Results(~Target,~LogProbabilities(~C),~target_value,
    ~Leaf.value(~F32,~Broadcast.Expanded<C>,~Broadcast.Expanded<Softmax.Fold(~Sum.Lanes,~C)>,~F32.sub,~Broadcast.value(~F32,~C,~c_value),
      ~Broadcast.value(~F32,~Softmax.Fold(~Sum.Lanes,~C),~Softmax.fold_value(~Sum.Lanes,~C,~c_value,~Loss.exponential_step,~Loss.logarithm_total))),
    target,log_probabilities(~C,centered,shape),images,shape,result)

# A Ready gradient had Ready exponentials, so the log probabilities were Ready.
def log_ready_owners(+images: U32,-first: Tensor(),+second: Observation.TensorOwnerModel<F32>,-product: Tensor(),+difference: Observation.TensorOwnerModel<F32>,-returned: Tensor(),
  ready: LossValues.Ready(Loss.entropy_gradient(images,(first,Observation.tensor_owner_pack(~F32,second)),(product,Observation.tensor_owner_pack(~F32,difference)),returned))) ->
  {Contract.tensor_ready(~F32,Observation.tensor_owner_pack(~F32,second)) == True{} : Bool}:
  Shape.Both{loss_ok,gradient_ok} = ready
  +subtracted_ready = NormalUnary.scalar_status(~F32,~F32.div,1.0,
    F.sub(F.exp(Observation.tensor_owner_pack(~F32,second)),Observation.tensor_owner_pack(~F32,difference)),U32.to_f32(images),gradient_ok)
  Flow.applied(~F32,~Operations.mapped_lanes(~Arithmetic.exp),Observation.tensor_owner_pack(~F32,second),4.5,
    Shape.first({Contract.tensor_ready(~F32,F.exp(Observation.tensor_owner_pack(~F32,second))) == True{} : Bool},
      {Contract.tensor_ready(~F32,Observation.tensor_owner_pack(~F32,difference)) == True{} : Bool},
      Flow.combine_status(~F32,~Operations.combine_kernel(~F32,~F32.sub),"sub",1.0,F.exp(Observation.tensor_owner_pack(~F32,second)),
        Observation.tensor_owner_pack(~F32,difference),subtracted_ready)))

def log_ready_observed(+images: U32,-first: Tensor(),-second: Tensor(),-product: Tensor(),-difference: Tensor(),-returned: Tensor(),
  second_observation: Observation.TensorObservation<F32,second>,difference_observation: Observation.TensorObservation<F32,difference>,
  ready: LossValues.Ready(Loss.entropy_gradient(images,(first,second),(product,difference),returned))) -> {Contract.tensor_ready(~F32,second) == True{} : Bool}:
  Observation.TensorObservation{+second_model,+second_equation} = second_observation
  Observation.TensorObservation{+difference_model,+difference_equation} = difference_observation
  %second_equation : {Contract.tensor_ready(~F32,_) == True{} : Bool}
  log_ready_owners(images,first,second_model,product,difference_model,returned,
    %Equal.sym(Tensor(),Observation.tensor_owner_pack(~F32,difference_model),difference,difference_equation) :
      LossValues.Ready(Loss.entropy_gradient(images,(first,Observation.tensor_owner_pack(~F32,second_model)),(product,_),returned))
    %Equal.sym(Tensor(),Observation.tensor_owner_pack(~F32,second_model),second,second_equation) :
      LossValues.Ready(Loss.entropy_gradient(images,(first,_),(product,difference),returned))
      ready)

def log_ready_of(+images: U32,-first: Tensor(),second: Tensor(),-product: Tensor(),difference: Tensor(),-returned: Tensor(),
  ready: LossValues.Ready(Loss.entropy_gradient(images,(first,second),(product,difference),returned))) -> {Contract.tensor_ready(~F32,second) == True{} : Bool}:
  log_ready_observed(images,first,second,product,difference,returned,Observation.observe_tensor(~F32,second),Observation.observe_tensor(~F32,difference),ready)

def log_ready(+images: U32,pair: Tensor() & Tensor(),targets: Tensor() & Tensor(),-returned: Tensor(),
  ready: LossValues.Ready(Loss.entropy_gradient(images,pair,targets,returned))) -> {Copy.pair_ready(~F32,pair) == True{} : Bool}:
  (first,second) = pair
  (product,difference) = targets
  log_ready_of(images,first,second,product,difference,returned,ready)

# The two copies of the target as bands of models.
def targets_copied(~C: Data,~c_value: C -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+centered: C,+target: Target,+images: U32,+shape: List<&2,U32>,
  +first: Model.Model<F32>,+second: Model.Model<F32>,+pg: Bands.Geometry,+pf: F32,+pw: U32,+kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,-returned: Tensor(),
  first_contract: Global.Contract(~F32,~LogProbabilities(~C),
    ~Leaf.value(~F32,~Broadcast.Expanded<C>,~Broadcast.Expanded<Softmax.Fold(~Sum.Lanes,~C)>,~F32.sub,~Broadcast.value(~F32,~C,~c_value),
      ~Broadcast.value(~F32,~Softmax.Fold(~Sum.Lanes,~C),~Softmax.fold_value(~Sum.Lanes,~C,~c_value,~Loss.exponential_step,~Loss.logarithm_total))),
    log_probabilities(~C,centered,shape),CombinePublic.tensor(~F32,first,pg,shape,pf,pw)),
  second_contract: Global.Contract(~F32,~LogProbabilities(~C),
    ~Leaf.value(~F32,~Broadcast.Expanded<C>,~Broadcast.Expanded<Softmax.Fold(~Sum.Lanes,~C)>,~F32.sub,~Broadcast.value(~F32,~C,~c_value),
      ~Broadcast.value(~F32,~Softmax.Fold(~Sum.Lanes,~C),~Softmax.fold_value(~Sum.Lanes,~C,~c_value,~Loss.exponential_step,~Loss.logarithm_total))),
    log_probabilities(~C,centered,shape),CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),
  evidence: Copy.Copied<F32,kept>,+kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),
  returned_contract: Global.Contract(~F32,~Target,~target_value,target,returned),
  ready: LossValues.Ready(Loss.entropy_gradient(images,(CombinePublic.tensor(~F32,first,pg,shape,pf,pw),CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),
    F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned))) ->
  Logs(~C,~c_value,~Target,~target_value,centered,target,images,shape,
    Loss.entropy_gradient(images,(CombinePublic.tensor(~F32,first,pg,shape,pf,pw),CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),
      F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned)):
  Copy.Copied{+product,+difference,+equation,+product_content,+difference_content} = evidence
  %equation : Logs(~C,~c_value,~Target,~target_value,centered,target,images,shape,
    Loss.entropy_gradient(images,(CombinePublic.tensor(~F32,first,pg,shape,pf,pw),CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),
      Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensor() & Tensor()),_,
        left => right => (Tensors.Tensor{left,tg,shape,tf,Tensors.Usable{},tw},Tensors.Tensor{right,tg,shape,tf,Tensors.Usable{},tw})),returned))
  gradient_models(~Target,~LogProbabilities(~C),~target_value,
    ~Leaf.value(~F32,~Broadcast.Expanded<C>,~Broadcast.Expanded<Softmax.Fold(~Sum.Lanes,~C)>,~F32.sub,~Broadcast.value(~F32,~C,~c_value),
      ~Broadcast.value(~F32,~Softmax.Fold(~Sum.Lanes,~C),~Softmax.fold_value(~Sum.Lanes,~C,~c_value,~Loss.exponential_step,~Loss.logarithm_total))),
    target,log_probabilities(~C,centered,shape),images,shape,first,second,pg,pf,pw,product,difference,tg,tf,tw,returned,first_contract,second_contract,
    Copy.kept(~F32,~Target,~target_value,target,kept,product,tg,shape,tf,tw,product_content,kept_contract),
    Copy.kept(~F32,~Target,~target_value,target,kept,difference,tg,shape,tf,tw,difference_content,kept_contract),
    returned_contract,
    %Equal.sym(Bands.Bands(~F32) & Bands.Bands(~F32),(Model.pack(~F32,product),Model.pack(~F32,difference)),Bands.copy(~F32,Model.pack(~F32,kept)),equation) :
      LossValues.Ready(Loss.entropy_gradient(images,(CombinePublic.tensor(~F32,first,pg,shape,pf,pw),CombinePublic.tensor(~F32,second,pg,shape,pf,pw)),
        Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensor() & Tensor()),_,
          left => right => (Tensors.Tensor{left,tg,shape,tf,Tensors.Usable{},tw},Tensors.Tensor{right,tg,shape,tf,Tensors.Usable{},tw})),returned))
    ready)

# The two copies of the log probabilities as bands of models.
def logs_copied(~C: Data,~c_value: C -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+centered: C,+target: Target,+images: U32,+shape: List<&2,U32>,
  +logs: Model.Model<F32>,+pg: Bands.Geometry,+pf: F32,+pw: U32,evidence: Copy.Copied<F32,logs>,
  +contract: Global.Contract(~F32,~LogProbabilities(~C),
    ~Leaf.value(~F32,~Broadcast.Expanded<C>,~Broadcast.Expanded<Softmax.Fold(~Sum.Lanes,~C)>,~F32.sub,~Broadcast.value(~F32,~C,~c_value),
      ~Broadcast.value(~F32,~Softmax.Fold(~Sum.Lanes,~C),~Softmax.fold_value(~Sum.Lanes,~C,~c_value,~Loss.exponential_step,~Loss.logarithm_total))),
    log_probabilities(~C,centered,shape),CombinePublic.tensor(~F32,logs,pg,shape,pf,pw)),
  +kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,-returned: Tensor(),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),
  returned_contract: Global.Contract(~F32,~Target,~target_value,target,returned),
  ready: LossValues.Ready(Loss.entropy_gradient(images,F.copy(CombinePublic.tensor(~F32,logs,pg,shape,pf,pw)),F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned))) ->
  Logs(~C,~c_value,~Target,~target_value,centered,target,images,shape,
    Loss.entropy_gradient(images,F.copy(CombinePublic.tensor(~F32,logs,pg,shape,pf,pw)),F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned)):
  Copy.Copied{+first,+second,+equation,+first_content,+second_content} = evidence
  %equation : Logs(~C,~c_value,~Target,~target_value,centered,target,images,shape,
    Loss.entropy_gradient(images,Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensor() & Tensor()),_,
        left => right => (Tensors.Tensor{left,pg,shape,pf,Tensors.Usable{},pw},Tensors.Tensor{right,pg,shape,pf,Tensors.Usable{},pw})),
      F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned))
  targets_copied(~C,~c_value,~Target,~target_value,centered,target,images,shape,first,second,pg,pf,pw,kept,tg,tf,tw,returned,
    Copy.kept(~F32,~LogProbabilities(~C),
      ~Leaf.value(~F32,~Broadcast.Expanded<C>,~Broadcast.Expanded<Softmax.Fold(~Sum.Lanes,~C)>,~F32.sub,~Broadcast.value(~F32,~C,~c_value),
        ~Broadcast.value(~F32,~Softmax.Fold(~Sum.Lanes,~C),~Softmax.fold_value(~Sum.Lanes,~C,~c_value,~Loss.exponential_step,~Loss.logarithm_total))),
      log_probabilities(~C,centered,shape),logs,first,pg,shape,pf,pw,first_content,contract),
    Copy.kept(~F32,~LogProbabilities(~C),
      ~Leaf.value(~F32,~Broadcast.Expanded<C>,~Broadcast.Expanded<Softmax.Fold(~Sum.Lanes,~C)>,~F32.sub,~Broadcast.value(~F32,~C,~c_value),
        ~Broadcast.value(~F32,~Softmax.Fold(~Sum.Lanes,~C),~Softmax.fold_value(~Sum.Lanes,~C,~c_value,~Loss.exponential_step,~Loss.logarithm_total))),
      log_probabilities(~C,centered,shape),logs,second,pg,shape,pf,pw,second_content,contract),
    Copy.copied(~F32,kept),kept_contract,returned_contract,
    %Equal.sym(Bands.Bands(~F32) & Bands.Bands(~F32),(Model.pack(~F32,first),Model.pack(~F32,second)),Bands.copy(~F32,Model.pack(~F32,logs)),equation) :
      LossValues.Ready(Loss.entropy_gradient(images,Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensor() & Tensor()),_,
          left => right => (Tensors.Tensor{left,pg,shape,pf,Tensors.Usable{},pw},Tensors.Tensor{right,pg,shape,pf,Tensors.Usable{},pw})),
        F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned))
    ready)

# Ready log probabilities of the shape, as bands of a model.
def logs_usable(~C: Data,~c_value: C -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+centered: C,+target: Target,+images: U32,+shape: List<&2,U32>,
  -logs: Tensor(),usable: Global.Usable<F32,shape,logs>,
  contract: Global.Contract(~F32,~LogProbabilities(~C),
    ~Leaf.value(~F32,~Broadcast.Expanded<C>,~Broadcast.Expanded<Softmax.Fold(~Sum.Lanes,~C)>,~F32.sub,~Broadcast.value(~F32,~C,~c_value),
      ~Broadcast.value(~F32,~Softmax.Fold(~Sum.Lanes,~C),~Softmax.fold_value(~Sum.Lanes,~C,~c_value,~Loss.exponential_step,~Loss.logarithm_total))),
    log_probabilities(~C,centered,shape),logs),
  +kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,-returned: Tensor(),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),
  returned_contract: Global.Contract(~F32,~Target,~target_value,target,returned),
  ready: LossValues.Ready(Loss.entropy_gradient(images,F.copy(logs),F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned))) ->
  Logs(~C,~c_value,~Target,~target_value,centered,target,images,shape,
    Loss.entropy_gradient(images,F.copy(logs),F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned)):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensor(),CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),logs,equation)
  %equation : Logs(~C,~c_value,~Target,~target_value,centered,target,images,shape,
    Loss.entropy_gradient(images,F.copy(_),F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned))
  logs_copied(~C,~c_value,~Target,~target_value,centered,target,images,shape,bands,geometry,fill,workers,Copy.copied(~F32,bands),
    %back : Global.Contract(~F32,~LogProbabilities(~C),
      ~Leaf.value(~F32,~Broadcast.Expanded<C>,~Broadcast.Expanded<Softmax.Fold(~Sum.Lanes,~C)>,~F32.sub,~Broadcast.value(~F32,~C,~c_value),
        ~Broadcast.value(~F32,~Softmax.Fold(~Sum.Lanes,~C),~Softmax.fold_value(~Sum.Lanes,~C,~c_value,~Loss.exponential_step,~Loss.logarithm_total))),
      log_probabilities(~C,centered,shape),_) contract,
    kept,tg,tf,tw,returned,kept_contract,returned_contract,
    %back : LossValues.Ready(Loss.entropy_gradient(images,F.copy(_),F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned)) ready)

def logs_folded(~C: Data,~c_value: C -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+centered: C,+target: Target,+images: U32,+shape: List<&2,U32>,
  logs: Tensor(),+ready_logs: {Contract.tensor_ready(~F32,logs) == True{} : Bool},
  folded: Shape.Both<Global.Contract(~F32,~LogProbabilities(~C),
    ~Leaf.value(~F32,~Broadcast.Expanded<C>,~Broadcast.Expanded<Softmax.Fold(~Sum.Lanes,~C)>,~F32.sub,~Broadcast.value(~F32,~C,~c_value),
      ~Broadcast.value(~F32,~Softmax.Fold(~Sum.Lanes,~C),~Softmax.fold_value(~Sum.Lanes,~C,~c_value,~Loss.exponential_step,~Loss.logarithm_total))),
    log_probabilities(~C,centered,shape),logs),Softmax.Extents(logs,shape)>,
  +kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,-returned: Tensor(),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),
  returned_contract: Global.Contract(~F32,~Target,~target_value,target,returned),
  ready: LossValues.Ready(Loss.entropy_gradient(images,F.copy(logs),F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned))) ->
  Logs(~C,~c_value,~Target,~target_value,centered,target,images,shape,
    Loss.entropy_gradient(images,F.copy(logs),F.copy(CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),returned)):
  Shape.Both{contract,+extents} = folded
  logs_usable(~C,~c_value,~Target,~target_value,centered,target,images,shape,logs,
    %extents : Global.Usable<F32,_,logs> Global.usable(~F32,logs,ready_logs),contract,kept,tg,tf,tw,returned,kept_contract,returned_contract,ready)

# The log probabilities of two copies of the centered rows of [N,K].
def shifted_models(~C: Data,~c_value: C -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+centered: C,+target: Target,+images: U32,+classes: U32,
  +left: Model.Model<F32>,+right: Model.Model<F32>,+cg: Bands.Geometry,+cf: F32,+cw: U32,+kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,-returned: Tensor(),
  left_contract: Global.Contract(~F32,~C,~c_value,centered,CombinePublic.tensor(~F32,left,cg,[images,classes],cf,cw)),
  right_contract: Global.Contract(~F32,~C,~c_value,centered,CombinePublic.tensor(~F32,right,cg,[images,classes],cf,cw)),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)),
  returned_contract: Global.Contract(~F32,~Target,~target_value,target,returned),
  +ready: LossValues.Ready(Loss.entropy_shifted(images,(CombinePublic.tensor(~F32,left,cg,[images,classes],cf,cw),CombinePublic.tensor(~F32,right,cg,[images,classes],cf,cw)),
    (CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw),returned)))) ->
  Logs(~C,~c_value,~Target,~target_value,centered,target,images,[images,classes],
    Loss.entropy_shifted(images,(CombinePublic.tensor(~F32,left,cg,[images,classes],cf,cw),CombinePublic.tensor(~F32,right,cg,[images,classes],cf,cw)),
      (CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw),returned))):
  +logs_ready = Flow.copied(~F32,
    F.sub(CombinePublic.tensor(~F32,left,cg,[images,classes],cf,cw),
      F.reshape(Layout.reduce_axes(~F32,~Sum.Lanes,~Loss.exponential_step,~Loss.logarithm_total,"cross_entropy",10.5,Sum.zero(),CombinePublic.tensor(~F32,right,cg,[images,classes],cf,cw),[1]),[images,1])),
    log_ready(images,F.copy(F.sub(CombinePublic.tensor(~F32,left,cg,[images,classes],cf,cw),
      F.reshape(Layout.reduce_axes(~F32,~Sum.Lanes,~Loss.exponential_step,~Loss.logarithm_total,"cross_entropy",10.5,Sum.zero(),CombinePublic.tensor(~F32,right,cg,[images,classes],cf,cw),[1]),[images,1]))),
      F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)),returned,ready))
  logs_folded(~C,~c_value,~Target,~target_value,centered,target,images,[images,classes],
    F.sub(CombinePublic.tensor(~F32,left,cg,[images,classes],cf,cw),
      F.reshape(Layout.reduce_axes(~F32,~Sum.Lanes,~Loss.exponential_step,~Loss.logarithm_total,"cross_entropy",10.5,Sum.zero(),CombinePublic.tensor(~F32,right,cg,[images,classes],cf,cw),[1]),[images,1])),
    logs_ready,
    Softmax.folded(~Sum.Lanes,~C,~c_value,~F32.sub,~Loss.exponential_step,~Loss.logarithm_total,centered,"sub","cross_entropy",10.5,Sum.zero(),
      left,right,cg,[images,classes],cf,cw,1,left_contract,right_contract,logs_ready),
    kept,tg,tf,tw,returned,kept_contract,returned_contract,ready)

# ---- The centered rows of the logits ----

# The number of rows of [N,K]; zero for another rank.
def rows(shape: List<&2,U32>) -> U32:
  match shape:
    case [images,classes]: images
    case shape: 0

# What cross entropy holds: the rows of the logits enter centered by their
# maximum, as in softmax.
def Entropy(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+logical: Logical,+target: Target,+shape: List<&2,U32>,
  result: Tensor() & (Tensor() & Tensor())) -> Data:
  Logs(~Softmax.Centered(~Logical),
    ~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Softmax.Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),
      ~Broadcast.value(~F32,~Softmax.Fold(~F32,~Logical),~Softmax.fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32)))),
    ~Target,~target_value,Softmax.centered_context(~Logical,Softmax.Softmax{logical,shape,1}),target,rows(shape),shape,result)

# Ready log probabilities had a Ready reduction of the second copy of the centered rows.
def shifted_ready_owners(+images: U32,+values: Observation.TensorOwnerModel<F32>,+for_sum: Observation.TensorOwnerModel<F32>,target: Tensor(),-kept: Tensor(),
  ready: LossValues.Ready(Loss.entropy_shifted(images,(Observation.tensor_owner_pack(~F32,values),Observation.tensor_owner_pack(~F32,for_sum)),(target,kept)))) -> {Contract.tensor_ready(~F32,Observation.tensor_owner_pack(~F32,for_sum)) == True{} : Bool}:
  +logs_ready = Flow.copied(~F32,F.sub(Observation.tensor_owner_pack(~F32,values),F.reshape(Layout.reduce_axes(~F32,~Sum.Lanes,~Loss.exponential_step,~Loss.logarithm_total,"cross_entropy",10.5,Sum.zero(),Observation.tensor_owner_pack(~F32,for_sum),[1]),[images,1])),
    log_ready(images,F.copy(F.sub(Observation.tensor_owner_pack(~F32,values),F.reshape(Layout.reduce_axes(~F32,~Sum.Lanes,~Loss.exponential_step,~Loss.logarithm_total,"cross_entropy",10.5,Sum.zero(),Observation.tensor_owner_pack(~F32,for_sum),[1]),[images,1]))),F.copy(target),kept,ready))
  Flow.reduced(~F32,~Sum.Lanes,~Loss.exponential_step,~Loss.logarithm_total,Observation.tensor_owner_pack(~F32,for_sum),"cross_entropy",10.5,Sum.zero(),[1],
    Flow.reshaped(~F32,Layout.reduce_axes(~F32,~Sum.Lanes,~Loss.exponential_step,~Loss.logarithm_total,"cross_entropy",10.5,Sum.zero(),Observation.tensor_owner_pack(~F32,for_sum),[1]),[images,1],
      Flow.combine_right(~F32,~F32.sub,Observation.tensor_owner_pack(~F32,values),F.reshape(Layout.reduce_axes(~F32,~Sum.Lanes,~Loss.exponential_step,~Loss.logarithm_total,"cross_entropy",10.5,Sum.zero(),Observation.tensor_owner_pack(~F32,for_sum),[1]),[images,1]),"sub",1.0,logs_ready)))

def shifted_ready_observed(+images: U32,-values: Tensor(),-for_sum: Tensor(),target: Tensor(),-kept: Tensor(),
  values_observation: Observation.TensorObservation<F32,values>,for_sum_observation: Observation.TensorObservation<F32,for_sum>,
  ready: LossValues.Ready(Loss.entropy_shifted(images,(values,for_sum),(target,kept)))) -> {Contract.tensor_ready(~F32,for_sum) == True{} : Bool}:
  Observation.TensorObservation{+values_model,+values_equation} = values_observation
  Observation.TensorObservation{+for_sum_model,+for_sum_equation} = for_sum_observation
  %for_sum_equation : {Contract.tensor_ready(~F32,_) == True{} : Bool}
  shifted_ready_owners(images,values_model,for_sum_model,target,kept,
    %Equal.sym(Tensor(),Observation.tensor_owner_pack(~F32,for_sum_model),for_sum,for_sum_equation) :
      LossValues.Ready(Loss.entropy_shifted(images,(Observation.tensor_owner_pack(~F32,values_model),_),(target,kept)))
    %Equal.sym(Tensor(),Observation.tensor_owner_pack(~F32,values_model),values,values_equation) :
      LossValues.Ready(Loss.entropy_shifted(images,(_,for_sum),(target,kept)))
      ready)

def shifted_ready(+images: U32,pair: Tensor() & Tensor(),targets: Tensor() & Tensor(),
  ready: LossValues.Ready(Loss.entropy_shifted(images,pair,targets))) -> {Copy.pair_ready(~F32,pair) == True{} : Bool}:
  (values,for_sum) = pair
  (target,kept) = targets
  shifted_ready_observed(images,values,for_sum,target,kept,Observation.observe_tensor(~F32,values),Observation.observe_tensor(~F32,for_sum),ready)

# The two copies of the target as bands of models.
def rows_targets(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+logical: Logical,+target: Target,+images: U32,+classes: U32,
  +left: Model.Model<F32>,+right: Model.Model<F32>,+cg: Bands.Geometry,+cf: F32,+cw: U32,
  left_contract: Softmax.CenteredContract(~Logical,~original,logical,[images,classes],1,CombinePublic.tensor(~F32,left,cg,[images,classes],cf,cw)),
  right_contract: Softmax.CenteredContract(~Logical,~original,logical,[images,classes],1,CombinePublic.tensor(~F32,right,cg,[images,classes],cf,cw)),
  +kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,evidence: Copy.Copied<F32,kept>,
  +kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)),
  ready: LossValues.Ready(Loss.entropy_shifted(images,(CombinePublic.tensor(~F32,left,cg,[images,classes],cf,cw),CombinePublic.tensor(~F32,right,cg,[images,classes],cf,cw)),
    F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw))))) ->
  Entropy(~Logical,~original,~Target,~target_value,logical,target,[images,classes],
    Loss.entropy_shifted(images,(CombinePublic.tensor(~F32,left,cg,[images,classes],cf,cw),CombinePublic.tensor(~F32,right,cg,[images,classes],cf,cw)),
      F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)))):
  Copy.Copied{+first,+second,+equation,+first_content,+second_content} = evidence
  %equation : Entropy(~Logical,~original,~Target,~target_value,logical,target,[images,classes],
    Loss.entropy_shifted(images,(CombinePublic.tensor(~F32,left,cg,[images,classes],cf,cw),CombinePublic.tensor(~F32,right,cg,[images,classes],cf,cw)),
      Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensor() & Tensor()),_,
        one => two => (Tensors.Tensor{one,tg,[images,classes],tf,Tensors.Usable{},tw},Tensors.Tensor{two,tg,[images,classes],tf,Tensors.Usable{},tw}))))
  shifted_models(~Softmax.Centered(~Logical),
    ~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Softmax.Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),
      ~Broadcast.value(~F32,~Softmax.Fold(~F32,~Logical),~Softmax.fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32)))),
    ~Target,~target_value,Softmax.centered_context(~Logical,Softmax.Softmax{logical,[images,classes],1}),target,images,classes,
    left,right,cg,cf,cw,first,tg,tf,tw,CombinePublic.tensor(~F32,second,tg,[images,classes],tf,tw),left_contract,right_contract,
    Copy.kept(~F32,~Target,~target_value,target,kept,first,tg,[images,classes],tf,tw,first_content,kept_contract),
    Copy.kept(~F32,~Target,~target_value,target,kept,second,tg,[images,classes],tf,tw,second_content,kept_contract),
    %Equal.sym(Bands.Bands(~F32) & Bands.Bands(~F32),(Model.pack(~F32,first),Model.pack(~F32,second)),Bands.copy(~F32,Model.pack(~F32,kept)),equation) :
      LossValues.Ready(Loss.entropy_shifted(images,(CombinePublic.tensor(~F32,left,cg,[images,classes],cf,cw),CombinePublic.tensor(~F32,right,cg,[images,classes],cf,cw)),
        Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensor() & Tensor()),_,
          one => two => (Tensors.Tensor{one,tg,[images,classes],tf,Tensors.Usable{},tw},Tensors.Tensor{two,tg,[images,classes],tf,Tensors.Usable{},tw}))))
    ready)

# The two copies of the centered rows as bands of models.
def centered_copied(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+logical: Logical,+target: Target,+images: U32,+classes: U32,
  +centered: Model.Model<F32>,+cg: Bands.Geometry,+cf: F32,+cw: U32,evidence: Copy.Copied<F32,centered>,
  +contract: Softmax.CenteredContract(~Logical,~original,logical,[images,classes],1,CombinePublic.tensor(~F32,centered,cg,[images,classes],cf,cw)),
  +kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)),
  ready: LossValues.Ready(Loss.entropy_shifted(images,F.copy(CombinePublic.tensor(~F32,centered,cg,[images,classes],cf,cw)),F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw))))) ->
  Entropy(~Logical,~original,~Target,~target_value,logical,target,[images,classes],
    Loss.entropy_shifted(images,F.copy(CombinePublic.tensor(~F32,centered,cg,[images,classes],cf,cw)),F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)))):
  Copy.Copied{+left,+right,+equation,+left_content,+right_content} = evidence
  %equation : Entropy(~Logical,~original,~Target,~target_value,logical,target,[images,classes],
    Loss.entropy_shifted(images,Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensor() & Tensor()),_,
        one => two => (Tensors.Tensor{one,cg,[images,classes],cf,Tensors.Usable{},cw},Tensors.Tensor{two,cg,[images,classes],cf,Tensors.Usable{},cw})),
      F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw))))
  rows_targets(~Logical,~original,~Target,~target_value,logical,target,images,classes,left,right,cg,cf,cw,
    Copy.kept(~F32,~Softmax.Centered(~Logical),
      ~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Softmax.Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),
        ~Broadcast.value(~F32,~Softmax.Fold(~F32,~Logical),~Softmax.fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32)))),
      Softmax.centered_context(~Logical,Softmax.Softmax{logical,[images,classes],1}),centered,left,cg,[images,classes],cf,cw,left_content,contract),
    Copy.kept(~F32,~Softmax.Centered(~Logical),
      ~Leaf.value(~F32,~Broadcast.Expanded<Logical>,~Broadcast.Expanded<Softmax.Fold(~F32,~Logical)>,~F32.sub,~Broadcast.value(~F32,~Logical,~original),
        ~Broadcast.value(~F32,~Softmax.Fold(~F32,~Logical),~Softmax.fold_value(~F32,~Logical,~original,~F32.max,~Tensors.same(~F32)))),
      Softmax.centered_context(~Logical,Softmax.Softmax{logical,[images,classes],1}),centered,right,cg,[images,classes],cf,cw,right_content,contract),
    kept,tg,tf,tw,Copy.copied(~F32,kept),kept_contract,
    %Equal.sym(Bands.Bands(~F32) & Bands.Bands(~F32),(Model.pack(~F32,left),Model.pack(~F32,right)),Bands.copy(~F32,Model.pack(~F32,centered)),equation) :
      LossValues.Ready(Loss.entropy_shifted(images,Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensor() & Tensor()),_,
          one => two => (Tensors.Tensor{one,cg,[images,classes],cf,Tensors.Usable{},cw},Tensors.Tensor{two,cg,[images,classes],cf,Tensors.Usable{},cw})),
        F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw))))
    ready)

# Ready centered rows of [N,K], as bands of a model.
def centered_usable(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+logical: Logical,+target: Target,+images: U32,+classes: U32,
  -centered: Tensor(),usable: Global.Usable<F32,[images,classes],centered>,
  contract: Softmax.CenteredContract(~Logical,~original,logical,[images,classes],1,centered),
  +kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)),
  ready: LossValues.Ready(Loss.entropy_shifted(images,F.copy(centered),F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw))))) ->
  Entropy(~Logical,~original,~Target,~target_value,logical,target,[images,classes],
    Loss.entropy_shifted(images,F.copy(centered),F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)))):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensor(),CombinePublic.tensor(~F32,bands,geometry,[images,classes],fill,workers),centered,equation)
  %equation : Entropy(~Logical,~original,~Target,~target_value,logical,target,[images,classes],
    Loss.entropy_shifted(images,F.copy(_),F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw))))
  centered_copied(~Logical,~original,~Target,~target_value,logical,target,images,classes,bands,geometry,fill,workers,Copy.copied(~F32,bands),
    %back : Softmax.CenteredContract(~Logical,~original,logical,[images,classes],1,_) contract,
    kept,tg,tf,tw,kept_contract,
    %back : LossValues.Ready(Loss.entropy_shifted(images,F.copy(_),F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)))) ready)

def centered_folded(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+logical: Logical,+target: Target,+images: U32,+classes: U32,
  centered: Tensor(),+centered_ready: {Contract.tensor_ready(~F32,centered) == True{} : Bool},
  folded: Shape.Both<Softmax.CenteredContract(~Logical,~original,logical,[images,classes],1,centered),Softmax.Extents(centered,[images,classes])>,
  +kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)),
  ready: LossValues.Ready(Loss.entropy_shifted(images,F.copy(centered),F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw))))) ->
  Entropy(~Logical,~original,~Target,~target_value,logical,target,[images,classes],
    Loss.entropy_shifted(images,F.copy(centered),F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)))):
  Shape.Both{contract,+extents} = folded
  centered_usable(~Logical,~original,~Target,~target_value,logical,target,images,classes,centered,
    %extents : Global.Usable<F32,_,centered> Global.usable(~F32,centered,centered_ready),contract,kept,tg,tf,tw,kept_contract,ready)

# The rows of two copies of the logits of [N,K], centered by their maximum.
def rows_models(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+logical: Logical,+target: Target,+images: U32,+classes: U32,
  +first: Model.Model<F32>,+second: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,+kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,
  first_contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,first,geometry,[images,classes],fill,workers)),
  second_contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,second,geometry,[images,classes],fill,workers)),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)),
  +ready: LossValues.Ready(Loss.entropy_rows(images,(CombinePublic.tensor(~F32,first,geometry,[images,classes],fill,workers),CombinePublic.tensor(~F32,second,geometry,[images,classes],fill,workers)),
    CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)))) ->
  Entropy(~Logical,~original,~Target,~target_value,logical,target,[images,classes],
    Loss.entropy_rows(images,(CombinePublic.tensor(~F32,first,geometry,[images,classes],fill,workers),CombinePublic.tensor(~F32,second,geometry,[images,classes],fill,workers)),
      CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw))):
  +centered_ready = Flow.copied(~F32,
    F.sub(CombinePublic.tensor(~F32,first,geometry,[images,classes],fill,workers),
      F.reshape(Layout.reduce_axes(~F32,~F32,~F32.max,~Tensors.same(~F32),"cross_entropy",1.0,F32.div(F32.neg(1.0),0.0),CombinePublic.tensor(~F32,second,geometry,[images,classes],fill,workers),[1]),[images,1])),
    shifted_ready(images,F.copy(F.sub(CombinePublic.tensor(~F32,first,geometry,[images,classes],fill,workers),
      F.reshape(Layout.reduce_axes(~F32,~F32,~F32.max,~Tensors.same(~F32),"cross_entropy",1.0,F32.div(F32.neg(1.0),0.0),CombinePublic.tensor(~F32,second,geometry,[images,classes],fill,workers),[1]),[images,1]))),
      F.copy(CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)),ready))
  centered_folded(~Logical,~original,~Target,~target_value,logical,target,images,classes,
    F.sub(CombinePublic.tensor(~F32,first,geometry,[images,classes],fill,workers),
      F.reshape(Layout.reduce_axes(~F32,~F32,~F32.max,~Tensors.same(~F32),"cross_entropy",1.0,F32.div(F32.neg(1.0),0.0),CombinePublic.tensor(~F32,second,geometry,[images,classes],fill,workers),[1]),[images,1])),
    centered_ready,
    Softmax.folded(~F32,~Logical,~original,~F32.sub,~F32.max,~Tensors.same(~F32),logical,"sub","cross_entropy",1.0,F32.div(F32.neg(1.0),0.0),
      first,second,geometry,[images,classes],fill,workers,1,first_contract,second_contract,centered_ready),
    kept,tg,tf,tw,kept_contract,ready)

# ---- The shapes ----

# The two copies of the logits as bands of models.
def logits_copied(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+logical: Logical,+target: Target,+images: U32,+classes: U32,
  +logits: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,evidence: Copy.Copied<F32,logits>,
  +contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,logits,geometry,[images,classes],fill,workers)),
  +kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)),
  ready: LossValues.Ready(Loss.entropy_rows(images,F.copy(CombinePublic.tensor(~F32,logits,geometry,[images,classes],fill,workers)),CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)))) ->
  Entropy(~Logical,~original,~Target,~target_value,logical,target,[images,classes],
    Loss.entropy_rows(images,F.copy(CombinePublic.tensor(~F32,logits,geometry,[images,classes],fill,workers)),CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw))):
  Copy.Copied{+first,+second,+equation,+first_content,+second_content} = evidence
  %equation : Entropy(~Logical,~original,~Target,~target_value,logical,target,[images,classes],
    Loss.entropy_rows(images,Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensor() & Tensor()),_,
        one => two => (Tensors.Tensor{one,geometry,[images,classes],fill,Tensors.Usable{},workers},Tensors.Tensor{two,geometry,[images,classes],fill,Tensors.Usable{},workers})),
      CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)))
  rows_models(~Logical,~original,~Target,~target_value,logical,target,images,classes,first,second,geometry,fill,workers,kept,tg,tf,tw,
    Copy.kept(~F32,~Logical,~original,logical,logits,first,geometry,[images,classes],fill,workers,first_content,contract),
    Copy.kept(~F32,~Logical,~original,logical,logits,second,geometry,[images,classes],fill,workers,second_content,contract),
    kept_contract,
    %Equal.sym(Bands.Bands(~F32) & Bands.Bands(~F32),(Model.pack(~F32,first),Model.pack(~F32,second)),Bands.copy(~F32,Model.pack(~F32,logits)),equation) :
      LossValues.Ready(Loss.entropy_rows(images,Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensor() & Tensor()),_,
          one => two => (Tensors.Tensor{one,geometry,[images,classes],fill,Tensors.Usable{},workers},Tensors.Tensor{two,geometry,[images,classes],fill,Tensors.Usable{},workers})),
        CombinePublic.tensor(~F32,kept,tg,[images,classes],tf,tw)))
    ready)

# A failed loss is never Ready.
def failed_loss(-Goal: Data,shape: List<&2,U32>,-logits: Tensor(),-target: Tensor(),ready: LossValues.Ready(Loss.entropy_failed(shape,logits,target))) -> Goal:
  Shape.Both{loss_ok,gradient_ok} = ready
  Empty.absurd(Goal,Words.false_true(loss_ok))

# Logits of rank two, of the shape of the target.
def ranked(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+logical: Logical,+target: Target,+shape: List<&2,U32>,
  +logits: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,+kept: Model.Model<F32>,+tg: Bands.Geometry,+tf: F32,+tw: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers)),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)),
  ready: LossValues.Ready(Loss.entropy_ranked(shape,CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers),CombinePublic.tensor(~F32,kept,tg,shape,tf,tw)))) ->
  Entropy(~Logical,~original,~Target,~target_value,logical,target,shape,
    Loss.entropy_ranked(shape,CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers),CombinePublic.tensor(~F32,kept,tg,shape,tf,tw))):
  match shape:
    case []:
      failed_loss(Entropy(~Logical,~original,~Target,~target_value,logical,target,[],
        Loss.entropy_ranked([],CombinePublic.tensor(~F32,logits,geometry,[],fill,workers),CombinePublic.tensor(~F32,kept,tg,[],tf,tw))),
        [],CombinePublic.tensor(~F32,logits,geometry,[],fill,workers),CombinePublic.tensor(~F32,kept,tg,[],tf,tw),ready)
    case +images <> rest:
      match rest:
        case []:
          failed_loss(Entropy(~Logical,~original,~Target,~target_value,logical,target,[images],
            Loss.entropy_ranked([images],CombinePublic.tensor(~F32,logits,geometry,[images],fill,workers),CombinePublic.tensor(~F32,kept,tg,[images],tf,tw))),
            [images],CombinePublic.tensor(~F32,logits,geometry,[images],fill,workers),CombinePublic.tensor(~F32,kept,tg,[images],tf,tw),ready)
        case +classes <> more:
          match more:
            case []:
              logits_copied(~Logical,~original,~Target,~target_value,logical,target,images,classes,logits,geometry,fill,workers,Copy.copied(~F32,logits),contract,
                kept,tg,tf,tw,kept_contract,ready)
            case +third <> others:
              failed_loss(Entropy(~Logical,~original,~Target,~target_value,logical,target,images <> (classes <> (third <> others)),
                Loss.entropy_ranked(images <> (classes <> (third <> others)),CombinePublic.tensor(~F32,logits,geometry,images <> (classes <> (third <> others)),fill,workers),
                  CombinePublic.tensor(~F32,kept,tg,images <> (classes <> (third <> others)),tf,tw))),
                images <> (classes <> (third <> others)),CombinePublic.tensor(~F32,logits,geometry,images <> (classes <> (third <> others)),fill,workers),
                CombinePublic.tensor(~F32,kept,tg,images <> (classes <> (third <> others)),tf,tw),ready)

# A target of the shape of the logits.
def selected(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,same: Bool,+logical: Logical,+target: Target,+shape: List<&2,U32>,
  +logits: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,+kept: Model.Model<F32>,+tg: Bands.Geometry,+target_shape: List<&2,U32>,+tf: F32,+tw: U32,
  +classified: {ShapeOps.same_extents(shape,target_shape) == same : Bool},
  contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers)),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,target_shape,tf,tw)),
  ready: LossValues.Ready(Loss.entropy_shaped(same,CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers),shape,CombinePublic.tensor(~F32,kept,tg,target_shape,tf,tw)))) ->
  Entropy(~Logical,~original,~Target,~target_value,logical,target,shape,
    Loss.entropy_shaped(same,CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers),shape,CombinePublic.tensor(~F32,kept,tg,target_shape,tf,tw))):
  match same:
    case True{}:
      +equal = Shapes.same_extents(shape,target_shape,classified)
      %equal : Entropy(~Logical,~original,~Target,~target_value,logical,target,shape,
        Loss.entropy_shaped(True{},CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers),shape,CombinePublic.tensor(~F32,kept,tg,_,tf,tw)))
      ranked(~Logical,~original,~Target,~target_value,logical,target,shape,logits,geometry,fill,workers,kept,tg,tf,tw,contract,
        %Equal.sym(List<&2,U32>,shape,target_shape,equal) : Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,_,tf,tw)) kept_contract,
        %Equal.sym(List<&2,U32>,shape,target_shape,equal) :
          LossValues.Ready(Loss.entropy_shaped(True{},CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers),shape,CombinePublic.tensor(~F32,kept,tg,_,tf,tw)))
          ready)
    case False{}:
      failed_loss(Entropy(~Logical,~original,~Target,~target_value,logical,target,shape,
        Loss.entropy_shaped(False{},CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers),shape,CombinePublic.tensor(~F32,kept,tg,target_shape,tf,tw))),
        shape,CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers),CombinePublic.tensor(~F32,kept,tg,target_shape,tf,tw),ready)

def modeled(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+logical: Logical,+target: Target,
  +logits: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,+kept: Model.Model<F32>,+tg: Bands.Geometry,+target_shape: List<&2,U32>,+tf: F32,+tw: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers)),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,CombinePublic.tensor(~F32,kept,tg,target_shape,tf,tw)),
  ready: LossValues.Ready(Loss.cross_entropy(CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers),CombinePublic.tensor(~F32,kept,tg,target_shape,tf,tw)))) ->
  Entropy(~Logical,~original,~Target,~target_value,logical,target,shape,
    Loss.cross_entropy(CombinePublic.tensor(~F32,logits,geometry,shape,fill,workers),CombinePublic.tensor(~F32,kept,tg,target_shape,tf,tw))):
  selected(~Logical,~original,~Target,~target_value,ShapeOps.same_extents(shape,target_shape),logical,target,shape,logits,geometry,fill,workers,kept,tg,target_shape,tf,tw,{==},
    contract,kept_contract,ready)

def observed(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+logical: Logical,+target: Target,
  -logits: Bands.Bands(~F32),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  -kept: Bands.Bands(~F32),+tg: Bands.Geometry,+target_shape: List<&2,U32>,+tf: F32,+tw: U32,
  logits_observation: Model.Observation<F32,logits>,kept_observation: Model.Observation<F32,kept>,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{logits,geometry,shape,fill,Tensors.Usable{},workers}),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,Tensors.Tensor{kept,tg,target_shape,tf,Tensors.Usable{},tw}),
  ready: LossValues.Ready(Loss.cross_entropy(Tensors.Tensor{logits,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{kept,tg,target_shape,tf,Tensors.Usable{},tw}))) ->
  Entropy(~Logical,~original,~Target,~target_value,logical,target,shape,
    Loss.cross_entropy(Tensors.Tensor{logits,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{kept,tg,target_shape,tf,Tensors.Usable{},tw})):
  Model.Observation{+logits_model,+logits_equation} = logits_observation
  Model.Observation{+kept_model,+kept_equation} = kept_observation
  %logits_equation : Entropy(~Logical,~original,~Target,~target_value,logical,target,shape,
    Loss.cross_entropy(Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{kept,tg,target_shape,tf,Tensors.Usable{},tw}))
  %kept_equation : Entropy(~Logical,~original,~Target,~target_value,logical,target,shape,
    Loss.cross_entropy(CombinePublic.tensor(~F32,logits_model,geometry,shape,fill,workers),Tensors.Tensor{_,tg,target_shape,tf,Tensors.Usable{},tw}))
  modeled(~Logical,~original,~Target,~target_value,logical,target,logits_model,geometry,shape,fill,workers,kept_model,tg,target_shape,tf,tw,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,logits_model),logits,logits_equation) :
      Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers}) contract,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,kept_model),kept,kept_equation) :
      Global.Contract(~F32,~Target,~target_value,target,Tensors.Tensor{_,tg,target_shape,tf,Tensors.Usable{},tw}) kept_contract,
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,kept_model),kept,kept_equation) :
      LossValues.Ready(Loss.cross_entropy(CombinePublic.tensor(~F32,logits_model,geometry,shape,fill,workers),Tensors.Tensor{_,tg,target_shape,tf,Tensors.Usable{},tw}))
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,logits_model),logits,logits_equation) :
      LossValues.Ready(Loss.cross_entropy(Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{kept,tg,target_shape,tf,Tensors.Usable{},tw}))
      ready)

# Logits [N,K] and a target of the same shape, each of its contract in any
# layout: a Ready loss and gradient hold the values of Entropy, and the target
# returns as it was given.
def cross_entropy(~Logical: Data,~original: Logical -> U32 -> F32,~Target: Data,~target_value: Target -> U32 -> F32,+logical: Logical,+target: Target,
  logits: Bands.Bands(~F32),+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  kept: Bands.Bands(~F32),+tg: Bands.Geometry,+target_shape: List<&2,U32>,+tf: F32,+tw: U32,
  contract: Global.Contract(~F32,~Logical,~original,logical,Tensors.Tensor{logits,geometry,shape,fill,Tensors.Usable{},workers}),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,Tensors.Tensor{kept,tg,target_shape,tf,Tensors.Usable{},tw}),
  ready: LossValues.Ready(Loss.cross_entropy(Tensors.Tensor{logits,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{kept,tg,target_shape,tf,Tensors.Usable{},tw}))) ->
  Entropy(~Logical,~original,~Target,~target_value,logical,target,shape,
    Loss.cross_entropy(Tensors.Tensor{logits,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{kept,tg,target_shape,tf,Tensors.Usable{},tw})):
  observed(~Logical,~original,~Target,~target_value,logical,target,logits,geometry,shape,fill,workers,kept,tg,target_shape,tf,tw,
    Model.observe(~F32,logits),Model.observe(~F32,kept),contract,kept_contract,ready)
