# Mean squared error composes copy, subtraction, square, reduction and scaling.
import Base
import ../tensor.bend as Tensors
import ../tensor_loss.bend as Loss
import ../tensor_shape.bend as ShapeOps
import ../tensor_layout.bend as Layout
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../tensor_band.bend as Bands
import ../storage_buffer.bend as Storage
import ./tensor_contract.bend as Contract
import ./tensor_copy_contract.bend as Copy
import ./tensor_reduce_values.bend as Reduce
import ../math_sum.bend as Sum
import ./tensor_ready_flow.bend as Flow
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_shape_refinement.bend as Shapes
import ./storage_shape.bend as Shape
import ./word_encoding.bend as Words
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_model.bend as Model
import ./tensor_band_map.bend as BandMap
import ./tensor_band_normal_unary.bend as NormalUnary
import ./tensor_band_combine_leaf.bend as Leaf
import ./tensor_combine_public.bend as CombinePublic
import ./tensor_unit_broadcast.bend as UnitBroadcast

# The squares of the difference, and the fold of all of them in place.
def Squares(~Prediction: Data,~Target: Data) -> Data:
  BandMap.Context<Unit,Leaf.Context<Prediction,Target>>

def Total(~Prediction: Data,~Target: Data) -> Data:
  Reduce.Lines<Sum.Lanes,Reduce.Ordered<Squares(~Prediction,~Target)>>

def total_context(~Prediction: Data,~Target: Data,+prediction: Prediction,+target: Target,+shape: List<&2,U32>) -> Total(~Prediction,~Target):
  Reduce.Lines{Sum.zero(),Reduce.Ordered{BandMap.Context{Unit{},Leaf.Context{prediction,target}},None{}},Shapes.elements(shape)}

# The two hypotheses concern only the returned loss and gradient.
def Ready(result: Loss.Tensor() & (Loss.Tensor() & Loss.Tensor())) -> Data:
  (loss,(gradient,target)) = result
  Shape.Both<{Contract.tensor_ready(~F32,loss) == True{} : Bool},{Contract.tensor_ready(~F32,gradient) == True{} : Bool}>

# A loss or gradient divided by the count for Mean, as it is for Sum.
def Normalized(~Logical: Data,~original: Logical -> U32 -> F32,reduction: Loss.Reduction,+count: F32,+logical: Logical,-tensor: Loss.Tensor()) -> Data:
  match reduction:
    case Loss.Mean{}: Global.Contract(~F32,~BandMap.Context<F32,Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~F32.div),~original),BandMap.Context{count,logical},tensor)
    case Loss.Sum{}: Global.Contract(~F32,~Logical,~original,logical,tensor)

# The rank-0 loss folds the squares of the difference of prediction and
# target, the gradient holds the difference times two, and the target returns
# as it was given.
def Contract(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+shape: List<&2,U32>,
  +reduction: Loss.Reduction,result: Loss.Tensor() & (Loss.Tensor() & Loss.Tensor())) -> Data:
  (loss,(gradient,returned)) = result
  Shape.Both<
    Normalized(~Total(~Prediction,~Target),
      ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Squares(~Prediction,~Target)>,~Reduce.ordered_value(~F32,~Squares(~Prediction,~Target),
        ~BandMap.value(~F32,~Unit,~Leaf.Context<Prediction,Target>,~Access.apply_element(~F32,~Loss.square),~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value))),~Sum.step,~Sum.total),
      reduction,U32.to_f32(Tensors.count_or_zero(shape)),total_context(~Prediction,~Target,prediction,target,shape),loss),
    Shape.Both<
      Normalized(~BandMap.Context<F32,Leaf.Context<Prediction,Target>>,
        ~BandMap.value(~F32,~F32,~Leaf.Context<Prediction,Target>,~Bands.scalar_value(~F32,~F32.mul),~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value)),
        reduction,U32.to_f32(Tensors.count_or_zero(shape)),BandMap.Context{2.0,Leaf.Context{prediction,target}},gradient),
      Global.Contract(~F32,~Target,~target_value,target,returned)>>

def normalize_ready(reduction: Loss.Reduction,+count: F32,tensor: Loss.Tensor(),ready: {Contract.tensor_ready(~F32,Loss.reduced(reduction,count,tensor)) == True{} : Bool}) ->
  {Contract.tensor_ready(~F32,tensor) == True{} : Bool}:
  match reduction:
    case Loss.Mean{}: NormalUnary.scalar_status(~F32,~F32.div,1.0,tensor,count,ready)
    case Loss.Sum{}: ready

# A Ready loss comes from a Ready difference.
def squared_ready(tensor: Loss.Tensor(),+reduction: Loss.Reduction,+count: F32,-target: Loss.Tensor(),ready: Ready(Loss.mse_squares(reduction,count,target,Layout.copy(~F32,tensor)))) ->
  {Contract.tensor_ready(~F32,tensor) == True{} : Bool}:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,status,workers}:
      match status:
        case Tensors.Usable{}: {==}
        case Tensors.Failed{problem}:
          match reduction:
            case Loss.Mean{}:
              Shape.Both{loss_ready,gradient_ready} = ready
              Empty.absurd({Contract.tensor_ready(~F32,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}) == True{} : Bool},Words.false_true(loss_ready))
            case Loss.Sum{}:
              Shape.Both{loss_ready,gradient_ready} = ready
              Empty.absurd({Contract.tensor_ready(~F32,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}) == True{} : Bool},Words.false_true(loss_ready))

def counted_ready(+reduction: Loss.Reduction,-target: Loss.Tensor(),shaped: Loss.Tensor() & List<&2,U32>,+ready: Ready(Loss.mse_counted(reduction,target,shaped))) ->
  {Contract.tensor_ready(~F32,Pair.fst(Loss.Tensor(),List<&2,U32>,shaped)) == True{} : Bool}:
  (difference,+shape) = shaped
  squared_ready(difference,reduction,U32.to_f32(Tensors.count_or_zero(shape)),target,ready)

# A loss or gradient of the contract, divided by the count for Mean.
def normalized(~Logical: Data,~original: Logical -> U32 -> F32,reduction: Loss.Reduction,+count: F32,+logical: Logical,tensor: Loss.Tensor(),
  usable: Global.Usable<F32,Contract.extents(~F32,tensor),tensor>,contract: Global.Contract(~F32,~Logical,~original,logical,tensor)) ->
  Normalized(~Logical,~original,reduction,count,logical,Loss.reduced(reduction,count,tensor)):
  match reduction:
    case Loss.Mean{}:
      Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
      %equation : Normalized(~Logical,~original,Loss.Mean{},count,logical,Tensors.combine_scalar(~F32,~F32.div,1.0,_,count))
      NormalUnary.combine_scalar(~F32,~Logical,~F32.div,~original,logical,1.0,count,bands,geometry,Contract.extents(~F32,tensor),fill,workers,
        %Equal.sym(Loss.Tensor(),Tensors.Tensor{Model.pack(~F32,bands),geometry,Contract.extents(~F32,tensor),fill,Tensors.Usable{},workers},tensor,equation) :
          Global.Contract(~F32,~Logical,~original,logical,_)
        contract)
    case Loss.Sum{}: contract

# The sum of every value of a tensor of the shape, by an operation of the name.
def summed(~Logical: Data,~original: Logical -> U32 -> F32,+name: String,+logical: Logical,+shape: List<&2,U32>,-squares: Loss.Tensor(),usable: Global.Usable<F32,shape,squares>,
  contract: Global.Contract(~F32,~Logical,~original,logical,squares),
  ready: {Contract.tensor_ready(~F32,Layout.reduce_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,name,6.0,Sum.zero(),squares)) == True{} : Bool}) ->
  Reduce.Whole(~F32,~Sum.Lanes,~Logical,~original,~Sum.step,~Sum.total,logical,Sum.zero(),shape,Layout.reduce_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,name,6.0,Sum.zero(),squares)):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Loss.Tensor(),Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},squares,equation)
  %equation : Reduce.Whole(~F32,~Sum.Lanes,~Logical,~original,~Sum.step,~Sum.total,logical,Sum.zero(),shape,Layout.reduce_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,name,6.0,Sum.zero(),_))
  Reduce.reduce_all(~F32,~Sum.Lanes,~Logical,~original,~Sum.step,~Sum.total,logical,name,6.0,Sum.zero(),bands,geometry,shape,fill,workers,
    %back : Global.Contract(~F32,~Logical,~original,logical,_) contract,
    %back : {Contract.tensor_ready(~F32,Layout.reduce_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,name,6.0,Sum.zero(),_)) == True{} : Bool} ready)

# The loss and gradient of the two copies of a difference of the shape.
def copied(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+shape: List<&2,U32>,
  +reduction: Loss.Reduction,-returned: Loss.Tensor(),+first: Model.Model<F32>,+second: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  first_contract: Global.Contract(~F32,~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},
    Tensors.Tensor{Model.pack(~F32,first),geometry,shape,fill,Tensors.Usable{},workers}),
  second_contract: Global.Contract(~F32,~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},
    Tensors.Tensor{Model.pack(~F32,second),geometry,shape,fill,Tensors.Usable{},workers}),
  returned_contract: Global.Contract(~F32,~Target,~target_value,target,returned),
  ready: Ready(Loss.mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(shape)),returned,
    (Tensors.Tensor{Model.pack(~F32,first),geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{Model.pack(~F32,second),geometry,shape,fill,Tensors.Usable{},workers})))) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,
    Loss.mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(shape)),returned,
      (Tensors.Tensor{Model.pack(~F32,first),geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{Model.pack(~F32,second),geometry,shape,fill,Tensors.Usable{},workers}))):
  Shape.Both{+loss_ready,gradient_ready} = ready
  +count = U32.to_f32(Tensors.count_or_zero(shape))
  +total_ready = normalize_ready(reduction,count,Layout.reduce_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"mse_loss",6.0,Sum.zero(),
    Tensors.apply(~F32,~Operations.mapped(~F32,~Loss.square),1.0,Tensors.Tensor{Model.pack(~F32,second),geometry,shape,fill,Tensors.Usable{},workers})),loss_ready)
  Shape.Both{
    normalized(~Total(~Prediction,~Target),
      ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Squares(~Prediction,~Target)>,~Reduce.ordered_value(~F32,~Squares(~Prediction,~Target),
        ~BandMap.value(~F32,~Unit,~Leaf.Context<Prediction,Target>,~Access.apply_element(~F32,~Loss.square),~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value))),~Sum.step,~Sum.total),
      reduction,count,total_context(~Prediction,~Target,prediction,target,shape),
      Layout.reduce_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"mse_loss",6.0,Sum.zero(),
        Tensors.apply(~F32,~Operations.mapped(~F32,~Loss.square),1.0,Tensors.Tensor{Model.pack(~F32,second),geometry,shape,fill,Tensors.Usable{},workers})),
      Global.usable(~F32,Layout.reduce_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"mse_loss",6.0,Sum.zero(),
        Tensors.apply(~F32,~Operations.mapped(~F32,~Loss.square),1.0,Tensors.Tensor{Model.pack(~F32,second),geometry,shape,fill,Tensors.Usable{},workers})),total_ready),
      summed(~BandMap.Context<Unit,Leaf.Context<Prediction,Target>>,
        ~BandMap.value(~F32,~Unit,~Leaf.Context<Prediction,Target>,~Access.apply_element(~F32,~Loss.square),~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value)),
        "mse_loss",BandMap.Context{Unit{},Leaf.Context{prediction,target}},shape,Tensors.apply(~F32,~Operations.mapped(~F32,~Loss.square),1.0,Tensors.Tensor{Model.pack(~F32,second),geometry,shape,fill,Tensors.Usable{},workers}),Global.usable(~F32,Tensors.apply(~F32,~Operations.mapped(~F32,~Loss.square),1.0,Tensors.Tensor{Model.pack(~F32,second),geometry,shape,fill,Tensors.Usable{},workers}),{==}),
        NormalUnary.apply(~F32,~Leaf.Context<Prediction,Target>,~Loss.square,~Operations.mapped(~F32,~Loss.square),~OperationsLaws.mapped(~F32,~Loss.square),
          ~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},1.0,second,geometry,shape,fill,workers,second_contract),
        total_ready)),
    Shape.Both{
      normalized(~BandMap.Context<F32,Leaf.Context<Prediction,Target>>,
        ~BandMap.value(~F32,~F32,~Leaf.Context<Prediction,Target>,~Bands.scalar_value(~F32,~F32.mul),~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value)),
        reduction,count,BandMap.Context{2.0,Leaf.Context{prediction,target}},
        Tensors.combine_scalar(~F32,~F32.mul,1.0,Tensors.Tensor{Model.pack(~F32,first),geometry,shape,fill,Tensors.Usable{},workers},2.0),
        Global.usable(~F32,Tensors.combine_scalar(~F32,~F32.mul,1.0,Tensors.Tensor{Model.pack(~F32,first),geometry,shape,fill,Tensors.Usable{},workers},2.0),{==}),
        NormalUnary.combine_scalar(~F32,~Leaf.Context<Prediction,Target>,~F32.mul,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),
          Leaf.Context{prediction,target},1.0,2.0,first,geometry,shape,fill,workers,first_contract)),
      returned_contract}}

# The difference of the shape is copied before the loss and the gradient.
def squared(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+shape: List<&2,U32>,
  +reduction: Loss.Reduction,-returned: Loss.Tensor(),+difference: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,
  evidence: Copy.Copied<F32,difference>,
  +contract: Global.Contract(~F32,~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},
    Tensors.Tensor{Model.pack(~F32,difference),geometry,shape,fill,Tensors.Usable{},workers}),
  returned_contract: Global.Contract(~F32,~Target,~target_value,target,returned),
  ready: Ready(Loss.mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(shape)),returned,Layout.copy(~F32,Tensors.Tensor{Model.pack(~F32,difference),geometry,shape,fill,Tensors.Usable{},workers})))) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,
    Loss.mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(shape)),returned,Layout.copy(~F32,Tensors.Tensor{Model.pack(~F32,difference),geometry,shape,fill,Tensors.Usable{},workers}))):
  Copy.Copied{+first,+second,+equation,+first_content,+second_content} = evidence
  %equation : Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,
    Loss.mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(shape)),returned,Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Loss.Tensor() & Loss.Tensor()),_,
      left => right => (Tensors.Tensor{left,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{right,geometry,shape,fill,Tensors.Usable{},workers}))))
  copied(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,returned,first,second,geometry,fill,workers,
    Copy.kept(~F32,~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},
      difference,first,geometry,shape,fill,workers,first_content,contract),
    Copy.kept(~F32,~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},
      difference,second,geometry,shape,fill,workers,second_content,contract),
    returned_contract,
    %Equal.sym(Bands.Bands(~F32) & Bands.Bands(~F32),(Model.pack(~F32,first),Model.pack(~F32,second)),Bands.copy(~F32,Model.pack(~F32,difference)),equation) :
      Ready(Loss.mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(shape)),returned,Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Loss.Tensor() & Loss.Tensor()),_,
        left => right => (Tensors.Tensor{left,geometry,shape,fill,Tensors.Usable{},workers},Tensors.Tensor{right,geometry,shape,fill,Tensors.Usable{},workers}))))
    ready)

# A Ready difference is a tensor of the shape of the operands.
def measured(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+shape: List<&2,U32>,
  +reduction: Loss.Reduction,-returned: Loss.Tensor(),difference: Loss.Tensor(),usable: Global.Usable<F32,Contract.extents(~F32,difference),difference>,
  +extents: {Contract.extents(~F32,difference) == shape : List<&2,U32>},
  contract: Global.Contract(~F32,~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},difference),
  returned_contract: Global.Contract(~F32,~Target,~target_value,target,returned),
  ready: Ready(Loss.mse_counted(reduction,returned,Tensors.shape(~F32,difference)))) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_counted(reduction,returned,Tensors.shape(~F32,difference))):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Loss.Tensor(),Tensors.Tensor{Model.pack(~F32,bands),geometry,Contract.extents(~F32,difference),fill,Tensors.Usable{},workers},difference,equation)
  %equation : Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,Loss.mse_counted(reduction,returned,Tensors.shape(~F32,_)))
  %Equal.sym(List<&2,U32>,Contract.extents(~F32,difference),shape,extents) :
    Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,
      Loss.mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(_)),returned,Layout.copy(~F32,Tensors.Tensor{Model.pack(~F32,bands),geometry,_,fill,Tensors.Usable{},workers})))
  squared(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,returned,bands,geometry,fill,workers,Copy.copied(~F32,bands),
    %extents : Global.Contract(~F32,~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},
      Tensors.Tensor{Model.pack(~F32,bands),geometry,_,fill,Tensors.Usable{},workers})
    %back : Global.Contract(~F32,~Leaf.Context<Prediction,Target>,~Leaf.value(~F32,~Prediction,~Target,~F32.sub,~prediction_value,~target_value),Leaf.Context{prediction,target},_)
    contract,
    returned_contract,
    %extents : Ready(Loss.mse_squares(reduction,U32.to_f32(Tensors.count_or_zero(_)),returned,Layout.copy(~F32,Tensors.Tensor{Model.pack(~F32,bands),geometry,_,fill,Tensors.Usable{},workers})))
    %back : Ready(Loss.mse_counted(reduction,returned,Tensors.shape(~F32,_)))
    ready)

# Operands of the same shape: the difference of the prediction and the kept
# copy of the target.
def kept(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+shape: List<&2,U32>,
  +reduction: Loss.Reduction,-returned: Loss.Tensor(),+prediction_bands: Model.Model<F32>,+prediction_geometry: Bands.Geometry,+prediction_fill: F32,+prediction_workers: U32,
  +kept_bands: Model.Model<F32>,+kept_geometry: Bands.Geometry,+kept_fill: F32,+kept_workers: U32,
  prediction_contract: Global.Contract(~F32,~Prediction,~prediction_value,prediction,Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers}),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,shape,kept_fill,Tensors.Usable{},kept_workers}),
  returned_contract: Global.Contract(~F32,~Target,~target_value,target,returned),
  +ready: Ready(Loss.mse_counted(reduction,returned,Tensors.shape(~F32,Tensors.combine(~F32,~F32.sub,"mse_loss",1.0,
    Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
    Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,shape,kept_fill,Tensors.Usable{},kept_workers}))))) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,
    Loss.mse_counted(reduction,returned,Tensors.shape(~F32,Tensors.combine(~F32,~F32.sub,"mse_loss",1.0,
      Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
      Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,shape,kept_fill,Tensors.Usable{},kept_workers})))):
  +difference_ready = {
    %Flow.measured_source(~F32,Tensors.combine(~F32,~F32.sub,"mse_loss",1.0,
        Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
        Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,shape,kept_fill,Tensors.Usable{},kept_workers})) :
      {Contract.tensor_ready(~F32,_) == True{} : Bool}
    counted_ready(reduction,returned,Tensors.shape(~F32,Tensors.combine(~F32,~F32.sub,"mse_loss",1.0,
        Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
        Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,shape,kept_fill,Tensors.Usable{},kept_workers})),ready)
    : {Contract.tensor_ready(~F32,Tensors.combine(~F32,~F32.sub,"mse_loss",1.0,
        Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
        Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,shape,kept_fill,Tensors.Usable{},kept_workers})) == True{} : Bool}}
  measured(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,returned,
    Tensors.combine(~F32,~F32.sub,"mse_loss",1.0,
      Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
      Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,shape,kept_fill,Tensors.Usable{},kept_workers}),
    Global.usable(~F32,Tensors.combine(~F32,~F32.sub,"mse_loss",1.0,
      Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
      Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,shape,kept_fill,Tensors.Usable{},kept_workers}),difference_ready),
    Flow.combined_extents(~F32,~Operations.combine_kernel(~F32,~F32.sub),"mse_loss",1.0,
      Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers,
      Model.pack(~F32,kept_bands),kept_geometry,shape,kept_fill,Tensors.Usable{},kept_workers,UnitBroadcast.self_broadcast(shape)),
    CombinePublic.combine(~F32,~Prediction,~Target,~F32.sub,~Operations.combine_kernel(~F32,~F32.sub),~OperationsLaws.scalar_kernel(~F32,~F32.sub),~prediction_value,~target_value,
      prediction,target,"mse_loss",1.0,shape,prediction_geometry,kept_geometry,prediction_fill,kept_fill,prediction_workers,kept_workers,
      Model.pack(~F32,prediction_bands),Model.pack(~F32,kept_bands),prediction_contract,kept_contract),
    returned_contract,ready)

# Operands of other shapes are rejected.
def selected(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,same: Bool,+prediction: Prediction,+target: Target,
  +shape: List<&2,U32>,+target_shape: List<&2,U32>,+reduction: Loss.Reduction,-returned: Loss.Tensor(),
  +prediction_bands: Model.Model<F32>,+prediction_geometry: Bands.Geometry,+prediction_fill: F32,+prediction_workers: U32,
  +kept_bands: Model.Model<F32>,+kept_geometry: Bands.Geometry,+kept_fill: F32,+kept_workers: U32,+classified: {ShapeOps.same_extents(shape,target_shape) == same : Bool},
  prediction_contract: Global.Contract(~F32,~Prediction,~prediction_value,prediction,Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers}),
  kept_contract: Global.Contract(~F32,~Target,~target_value,target,Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,target_shape,kept_fill,Tensors.Usable{},kept_workers}),
  returned_contract: Global.Contract(~F32,~Target,~target_value,target,returned),
  ready: Ready(Loss.mse_kept(reduction,same,Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
    (returned,Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,target_shape,kept_fill,Tensors.Usable{},kept_workers})))) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,
    Loss.mse_kept(reduction,same,Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
      (returned,Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,target_shape,kept_fill,Tensors.Usable{},kept_workers}))):
  match same:
    case True{}:
      +equal = Shapes.same_extents(shape,target_shape,classified)
      %equal :
        Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,
          Loss.mse_kept(reduction,True{},Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
            (returned,Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,_,kept_fill,Tensors.Usable{},kept_workers})))
      kept(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,returned,prediction_bands,prediction_geometry,prediction_fill,prediction_workers,
        kept_bands,kept_geometry,kept_fill,kept_workers,prediction_contract,
        %Equal.sym(List<&2,U32>,shape,target_shape,equal) :
          Global.Contract(~F32,~Target,~target_value,target,Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,_,kept_fill,Tensors.Usable{},kept_workers})
        kept_contract,returned_contract,
        %Equal.sym(List<&2,U32>,shape,target_shape,equal) : Ready(Loss.mse_kept(reduction,True{},Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
          (returned,Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,_,kept_fill,Tensors.Usable{},kept_workers})))
        ready)
    case False{}:
      Empty.absurd(Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,
          Loss.mse_kept(reduction,False{},Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
            (returned,Tensors.Tensor{Model.pack(~F32,kept_bands),kept_geometry,target_shape,kept_fill,Tensors.Usable{},kept_workers}))),
        Words.false_true(
          %Flow.rejected(~F32,Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},"mse_loss") : {_ == True{} : Bool}
          %Flow.measured_source(~F32,Tensors.rejected(~F32,"mse_loss",Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers})) : {Contract.tensor_ready(~F32,_) == True{} : Bool}
          counted_ready(reduction,returned,Tensors.shape(~F32,Tensors.rejected(~F32,"mse_loss",Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers})),ready)))

# The target is copied: one copy returns, the other meets the prediction.
def copies(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+reduction: Loss.Reduction,
  +prediction_bands: Model.Model<F32>,+prediction_geometry: Bands.Geometry,+shape: List<&2,U32>,+prediction_fill: F32,+prediction_workers: U32,
  +target_bands: Model.Model<F32>,+target_geometry: Bands.Geometry,+target_shape: List<&2,U32>,+target_fill: F32,+target_workers: U32,evidence: Copy.Copied<F32,target_bands>,
  prediction_contract: Global.Contract(~F32,~Prediction,~prediction_value,prediction,Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers}),
  +target_contract: Global.Contract(~F32,~Target,~target_value,target,Tensors.Tensor{Model.pack(~F32,target_bands),target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers}),
  ready: Ready(Loss.mse(Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
    Tensors.Tensor{Model.pack(~F32,target_bands),target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers},reduction))) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,
    Loss.mse(Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
      Tensors.Tensor{Model.pack(~F32,target_bands),target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers},reduction)):
  Copy.Copied{+returned,+kept,+equation,+returned_content,+kept_content} = evidence
  +same = ShapeOps.same_extents(shape,target_shape)
  %equation : Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,
    Loss.mse_kept(reduction,same,Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
      Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Loss.Tensor() & Loss.Tensor()),_,
        left => right => (Tensors.Tensor{left,target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers},Tensors.Tensor{right,target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers}))))
  selected(~Prediction,~Target,~prediction_value,~target_value,same,prediction,target,shape,target_shape,reduction,
    Tensors.Tensor{Model.pack(~F32,returned),target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers},
    prediction_bands,prediction_geometry,prediction_fill,prediction_workers,kept,target_geometry,target_fill,target_workers,{==},prediction_contract,
    Copy.kept(~F32,~Target,~target_value,target,target_bands,kept,target_geometry,target_shape,target_fill,target_workers,kept_content,target_contract),
    Copy.kept(~F32,~Target,~target_value,target,target_bands,returned,target_geometry,target_shape,target_fill,target_workers,returned_content,target_contract),
    %Equal.sym(Bands.Bands(~F32) & Bands.Bands(~F32),(Model.pack(~F32,returned),Model.pack(~F32,kept)),Bands.copy(~F32,Model.pack(~F32,target_bands)),equation) :
      Ready(Loss.mse_kept(reduction,same,Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
        Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Loss.Tensor() & Loss.Tensor()),_,
          left => right => (Tensors.Tensor{left,target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers},Tensors.Tensor{right,target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers}))))
    ready)

# The loss and gradient of a prediction against a target of the same shape,
# both of the contract in any layout.
def mse(~Prediction: Data,~Target: Data,~prediction_value: Prediction -> U32 -> F32,~target_value: Target -> U32 -> F32,+prediction: Prediction,+target: Target,+reduction: Loss.Reduction,
  +prediction_bands: Model.Model<F32>,+prediction_geometry: Bands.Geometry,+shape: List<&2,U32>,+prediction_fill: F32,+prediction_workers: U32,
  +target_bands: Model.Model<F32>,+target_geometry: Bands.Geometry,+target_shape: List<&2,U32>,+target_fill: F32,+target_workers: U32,
  prediction_contract: Global.Contract(~F32,~Prediction,~prediction_value,prediction,Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers}),
  +target_contract: Global.Contract(~F32,~Target,~target_value,target,Tensors.Tensor{Model.pack(~F32,target_bands),target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers}),
  ready: Ready(Loss.mse(Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
    Tensors.Tensor{Model.pack(~F32,target_bands),target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers},reduction))) ->
  Contract(~Prediction,~Target,~prediction_value,~target_value,prediction,target,shape,reduction,
    Loss.mse(Tensors.Tensor{Model.pack(~F32,prediction_bands),prediction_geometry,shape,prediction_fill,Tensors.Usable{},prediction_workers},
      Tensors.Tensor{Model.pack(~F32,target_bands),target_geometry,target_shape,target_fill,Tensors.Usable{},target_workers},reduction)):
  copies(~Prediction,~Target,~prediction_value,~target_value,prediction,target,reduction,prediction_bands,prediction_geometry,shape,prediction_fill,prediction_workers,
    target_bands,target_geometry,target_shape,target_fill,target_workers,Copy.copied(~F32,target_bands),prediction_contract,target_contract,ready)
