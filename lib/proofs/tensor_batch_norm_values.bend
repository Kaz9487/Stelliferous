# Batch normalization composes the channel sums of sum_axes, broadcast
# combinations, operations with a scalar, sqrt, a tensor of ones and copies.
# Each step is stated for any tensor: once its result is Ready, the result is
# a tensor of its shape that holds the values of the step's reader.
import Base
import ../tensor.bend as Tensors
import ../tensor_f32.bend as F
import ../tensor_batch_norm.bend as Norm
import ../tensor_shape.bend as Shapes
import ../tensor_layout.bend as Layout
import ../tensor_operations.bend as Operations
import ../tensor_access.bend as Access
import ../tensor_band.bend as Bands
import ../math_fp32.bend as Arithmetic
import ../math_sum.bend as Sum
import ../storage_buffer.bend as Storage
import ./storage_shape.bend as Shape
import ./tensor_contract.bend as Contract
import ./tensor_copy_contract.bend as Copy
import ./tensor_reduce_values.bend as Reduce
import ./tensor_reshape_values.bend as ReshapeValues
import ./tensor_full_values.bend as FullValues
import ./tensor_ready_flow.bend as Flow
import ./tensor_operations_refinement.bend as OperationLaws
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_band_model.bend as Model
import ./tensor_band_map.bend as BandMap
import ./tensor_band_normal_unary.bend as NormalUnary
import ./tensor_band_combine_leaf.bend as Leaf
import ./tensor_broadcast_public.bend as Broadcast
import ./tensor_combine_broadcast.bend as CombineBroadcast
import ./tensor_unit_broadcast.bend as UnitBroadcast
import ./tensor_band_shape_refinement.bend as BandShape
import ./word_encoding.bend as Words
import ./tensor_band_concat_execution.bend as Observation
import ../traversal_partition.bend as Partition

# ---- One step ----

# Once Ready, a tensor of the shape that holds the reader's values.
def Holds(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensors.Tensor<F32>) -> Type:
  @ready: {Contract.tensor_ready(~F32,tensor) == True{} : Bool} -> Global.Shaped(~F32,~Logical,~original,logical,shape,tensor)

def holds(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensors.Tensor<F32>,
  +shaped: Global.Shaped(~F32,~Logical,~original,logical,shape,tensor)) -> Holds(~Logical,~original,logical,shape,tensor):
  ready => shaped

# The same tensor read through another reader that agrees at every index.
def restated(~Before: Data,~After: Data,~before_value: Before -> U32 -> F32,~after_value: After -> U32 -> F32,+before: Before,+after: After,
  agrees: @+index: U32 -> {before_value(before,index) == after_value(after,index) : F32},+shape: List<&2,U32>,-tensor: Tensors.Tensor<F32>,
  holds: Holds(~Before,~before_value,before,shape,tensor)) -> Holds(~After,~after_value,after,shape,tensor):
  ready => Global.restated(~F32,~Before,~After,~before_value,~after_value,before,after,agrees,shape,tensor,holds(ready))

# The reader of a combination whose left operand has the common extents.
def Combined(~Left: Data,~Right: Data) -> Data:
  Leaf.Context<Broadcast.Expanded<Left>,Broadcast.Expanded<Right>>

def combined_context(~Left: Data,~Right: Data,+left: Left,+right: Right,+shape: List<&2,U32>,+right_shape: List<&2,U32>) -> Combined(~Left,~Right):
  Leaf.Context{Broadcast.Expanded{left,Broadcast.expansion(shape,shape)},Broadcast.Expanded{right,Broadcast.expansion(right_shape,shape)}}

def combined_models(~Left: Data,~Right: Data,~operation: F32 -> F32 -> F32,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  +left_context: Left,+right_context: Right,+name: String,+shape: List<&2,U32>,+right_shape: List<&2,U32>,
  +left: Model.Model<F32>,+lg: Bands.Geometry,+lf: F32,+lw: U32,+right: Model.Model<F32>,+rg: Bands.Geometry,+rf: F32,+rw: U32,
  left_contract: Global.Contract(~F32,~Left,~left_value,left_context,Tensors.Tensor{Model.pack(~F32,left),lg,shape,lf,Tensors.Usable{},lw}),
  right_contract: Global.Contract(~F32,~Right,~right_value,right_context,Tensors.Tensor{Model.pack(~F32,right),rg,right_shape,rf,Tensors.Usable{},rw}),
  +common: {Shapes.broadcast(shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>},
  ready: {Contract.tensor_ready(~F32,Tensors.combine(~F32,~operation,name,1.0,Tensors.Tensor{Model.pack(~F32,left),lg,shape,lf,Tensors.Usable{},lw},
    Tensors.Tensor{Model.pack(~F32,right),rg,right_shape,rf,Tensors.Usable{},rw})) == True{} : Bool}) ->
  Global.Shaped(~F32,~Combined(~Left,~Right),~Leaf.value(~F32,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~Broadcast.value(~F32,~Left,~left_value),~Broadcast.value(~F32,~Right,~right_value)),
    combined_context(~Left,~Right,left_context,right_context,shape,right_shape),shape,
    Tensors.combine(~F32,~operation,name,1.0,Tensors.Tensor{Model.pack(~F32,left),lg,shape,lf,Tensors.Usable{},lw},
      Tensors.Tensor{Model.pack(~F32,right),rg,right_shape,rf,Tensors.Usable{},rw})):
  Shape.Both{Shape.Both{+well_formed,Shape.Both{+counted,+coherent}},+values} = left_contract
  Global.shaped(~F32,~Combined(~Left,~Right),~Leaf.value(~F32,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~Broadcast.value(~F32,~Left,~left_value),~Broadcast.value(~F32,~Right,~right_value)),
    combined_context(~Left,~Right,left_context,right_context,shape,right_shape),shape,
    Tensors.combine(~F32,~operation,name,1.0,Tensors.Tensor{Model.pack(~F32,left),lg,shape,lf,Tensors.Usable{},lw},
      Tensors.Tensor{Model.pack(~F32,right),rg,right_shape,rf,Tensors.Usable{},rw}),
    ready,
    Flow.combined_extents(~F32,~Operations.combine_kernel(~F32,~operation),name,1.0,Model.pack(~F32,left),lg,shape,lf,Tensors.Usable{},lw,
      Model.pack(~F32,right),rg,right_shape,rf,Tensors.Usable{},rw,common),
    CombineBroadcast.broadcast(~F32,~Left,~Right,~operation,~Operations.combine_kernel(~F32,~operation),~OperationLaws.scalar_kernel(~F32,~operation),
      ~left_value,~right_value,left_context,right_context,name,1.0,shape,right_shape,shape,lg,rg,lf,rf,lw,rw,Model.pack(~F32,left),Model.pack(~F32,right),
      common,counted,Shape.Both{Shape.Both{well_formed,Shape.Both{counted,coherent}},values},right_contract))

def combined_shaped(~Left: Data,~Right: Data,~operation: F32 -> F32 -> F32,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  +left_context: Left,+right_context: Right,+name: String,+shape: List<&2,U32>,+right_shape: List<&2,U32>,-left: Tensors.Tensor<F32>,-right: Tensors.Tensor<F32>,
  left_shaped: Global.Shaped(~F32,~Left,~left_value,left_context,shape,left),right_shaped: Global.Shaped(~F32,~Right,~right_value,right_context,right_shape,right),
  +common: {Shapes.broadcast(shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>},
  ready: {Contract.tensor_ready(~F32,Tensors.combine(~F32,~operation,name,1.0,left,right)) == True{} : Bool}) ->
  Global.Shaped(~F32,~Combined(~Left,~Right),~Leaf.value(~F32,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~Broadcast.value(~F32,~Left,~left_value),~Broadcast.value(~F32,~Right,~right_value)),
    combined_context(~Left,~Right,left_context,right_context,shape,right_shape),shape,Tensors.combine(~F32,~operation,name,1.0,left,right)):
  Shape.Both{left_usable,left_contract} = left_shaped
  Global.Usable{+lb,+lg,+lf,+lw,+left_equation} = left_usable
  Shape.Both{right_usable,right_contract} = right_shaped
  Global.Usable{+rb,+rg,+rf,+rw,+right_equation} = right_usable
  +left_back = Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,lb),lg,shape,lf,Tensors.Usable{},lw},left,left_equation)
  +right_back = Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,rb),rg,right_shape,rf,Tensors.Usable{},rw},right,right_equation)
  %left_equation : Global.Shaped(~F32,~Combined(~Left,~Right),~Leaf.value(~F32,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~Broadcast.value(~F32,~Left,~left_value),~Broadcast.value(~F32,~Right,~right_value)),
    combined_context(~Left,~Right,left_context,right_context,shape,right_shape),shape,Tensors.combine(~F32,~operation,name,1.0,_,right))
  %right_equation : Global.Shaped(~F32,~Combined(~Left,~Right),~Leaf.value(~F32,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~Broadcast.value(~F32,~Left,~left_value),~Broadcast.value(~F32,~Right,~right_value)),
    combined_context(~Left,~Right,left_context,right_context,shape,right_shape),shape,
    Tensors.combine(~F32,~operation,name,1.0,Tensors.Tensor{Model.pack(~F32,lb),lg,shape,lf,Tensors.Usable{},lw},_))
  combined_models(~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,name,shape,right_shape,lb,lg,lf,lw,rb,rg,rf,rw,
    %left_back : Global.Contract(~F32,~Left,~left_value,left_context,_) left_contract,
    %right_back : Global.Contract(~F32,~Right,~right_value,right_context,_) right_contract,
    common,
    %left_back : {Contract.tensor_ready(~F32,Tensors.combine(~F32,~operation,name,1.0,_,Tensors.Tensor{Model.pack(~F32,rb),rg,right_shape,rf,Tensors.Usable{},rw})) == True{} : Bool}
    %right_back : {Contract.tensor_ready(~F32,Tensors.combine(~F32,~operation,name,1.0,left,_)) == True{} : Bool}
    ready)

def combined_ready(~Left: Data,~Right: Data,~operation: F32 -> F32 -> F32,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  +left_context: Left,+right_context: Right,+name: String,+shape: List<&2,U32>,+right_shape: List<&2,U32>,-left: Tensors.Tensor<F32>,-right: Tensors.Tensor<F32>,
  left_holds: Holds(~Left,~left_value,left_context,shape,left),right_holds: Holds(~Right,~right_value,right_context,right_shape,right),
  +common: {Shapes.broadcast(shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>},
  operands: Shape.Both<{Contract.tensor_ready(~F32,left) == True{} : Bool},{Contract.tensor_ready(~F32,right) == True{} : Bool}>,
  ready: {Contract.tensor_ready(~F32,Tensors.combine(~F32,~operation,name,1.0,left,right)) == True{} : Bool}) ->
  Global.Shaped(~F32,~Combined(~Left,~Right),~Leaf.value(~F32,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~Broadcast.value(~F32,~Left,~left_value),~Broadcast.value(~F32,~Right,~right_value)),
    combined_context(~Left,~Right,left_context,right_context,shape,right_shape),shape,Tensors.combine(~F32,~operation,name,1.0,left,right)):
  Shape.Both{left_ready,right_ready} = operands
  combined_shaped(~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,name,shape,right_shape,left,right,
    left_holds(left_ready),right_holds(right_ready),common,ready)

# A combination of a left operand of the common extents with a right one that
# broadcasts to them.
def combined(~Left: Data,~Right: Data,~operation: F32 -> F32 -> F32,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  +left_context: Left,+right_context: Right,+name: String,+shape: List<&2,U32>,+right_shape: List<&2,U32>,left: Tensors.Tensor<F32>,right: Tensors.Tensor<F32>,
  left_holds: Holds(~Left,~left_value,left_context,shape,left),right_holds: Holds(~Right,~right_value,right_context,right_shape,right),
  +common: {Shapes.broadcast(shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>}) ->
  Holds(~Combined(~Left,~Right),~Leaf.value(~F32,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~Broadcast.value(~F32,~Left,~left_value),~Broadcast.value(~F32,~Right,~right_value)),
    combined_context(~Left,~Right,left_context,right_context,shape,right_shape),shape,Tensors.combine(~F32,~operation,name,1.0,left,right)):
  +ready => combined_ready(~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,name,shape,right_shape,left,right,left_holds,right_holds,common,
    Flow.combine_status(~F32,~Operations.combine_kernel(~F32,~operation),name,1.0,left,right,ready),ready)

# An operation of every value with a scalar.
def scaled_shaped(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32 -> F32,+logical: Logical,+shape: List<&2,U32>,+scalar: F32,
  -tensor: Tensors.Tensor<F32>,shaped: Global.Shaped(~F32,~Logical,~original,logical,shape,tensor)) ->
  Global.Shaped(~F32,~BandMap.Context<F32,Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~operation),~original),BandMap.Context{scalar,logical},shape,
    Tensors.combine_scalar(~F32,~operation,1.0,tensor,scalar)):
  Shape.Both{usable,contract} = shaped
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},tensor,equation)
  %equation : Global.Shaped(~F32,~BandMap.Context<F32,Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~operation),~original),BandMap.Context{scalar,logical},
    shape,Tensors.combine_scalar(~F32,~operation,1.0,_,scalar))
  Global.shaped(~F32,~BandMap.Context<F32,Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~operation),~original),BandMap.Context{scalar,logical},shape,
    Tensors.combine_scalar(~F32,~operation,1.0,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},scalar),{==},{==},
    NormalUnary.combine_scalar(~F32,~Logical,~operation,~original,logical,1.0,scalar,bands,geometry,shape,fill,workers,
      %back : Global.Contract(~F32,~Logical,~original,logical,_) contract))

def scaled(~Logical: Data,~original: Logical -> U32 -> F32,~operation: F32 -> F32 -> F32,+logical: Logical,+shape: List<&2,U32>,+scalar: F32,
  tensor: Tensors.Tensor<F32>,holds: Holds(~Logical,~original,logical,shape,tensor)) ->
  Holds(~BandMap.Context<F32,Logical>,~BandMap.value(~F32,~F32,~Logical,~Bands.scalar_value(~F32,~operation),~original),BandMap.Context{scalar,logical},shape,
    Tensors.combine_scalar(~F32,~operation,1.0,tensor,scalar)):
  ready => scaled_shaped(~Logical,~original,~operation,logical,shape,scalar,tensor,holds(NormalUnary.scalar_status(~F32,~operation,1.0,tensor,scalar,ready)))

# The square root of every value.
def root_shaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensors.Tensor<F32>,
  shaped: Global.Shaped(~F32,~Logical,~original,logical,shape,tensor)) ->
  Global.Shaped(~F32,~BandMap.Context<Unit,Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Arithmetic.sqrt),~original),BandMap.Context{Unit{},logical},
    shape,F.sqrt(tensor)):
  Shape.Both{usable,contract} = shaped
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},tensor,equation)
  %equation : Global.Shaped(~F32,~BandMap.Context<Unit,Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Arithmetic.sqrt),~original),
    BandMap.Context{Unit{},logical},shape,F.sqrt(_))
  Global.shaped(~F32,~BandMap.Context<Unit,Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Arithmetic.sqrt),~original),BandMap.Context{Unit{},logical},
    shape,F.sqrt(Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers}),{==},{==},
    NormalUnary.apply(~F32,~Logical,~Arithmetic.sqrt,~Operations.mapped_lanes(~Arithmetic.sqrt),~OperationLaws.mapped_lanes(~Arithmetic.sqrt),~original,logical,0.125,
      bands,geometry,shape,fill,workers,%back : Global.Contract(~F32,~Logical,~original,logical,_) contract))

def root(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,tensor: Tensors.Tensor<F32>,holds: Holds(~Logical,~original,logical,shape,tensor)) ->
  Holds(~BandMap.Context<Unit,Logical>,~BandMap.value(~F32,~Unit,~Logical,~Access.apply_element(~F32,~Arithmetic.sqrt),~original),BandMap.Context{Unit{},logical},
    shape,F.sqrt(tensor)):
  ready => root_shaped(~Logical,~original,logical,shape,tensor,holds(Flow.applied(~F32,~Operations.mapped_lanes(~Arithmetic.sqrt),tensor,0.125,ready)))

# The same values under other extents of as many elements.
def reshaped_shaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+target: List<&2,U32>,-tensor: Tensors.Tensor<F32>,
  shaped: Global.Shaped(~F32,~Logical,~original,logical,shape,tensor),accepted: {Contract.tensor_ready(~F32,F.reshape(tensor,target)) == True{} : Bool}) ->
  Global.Shaped(~F32,~Logical,~original,logical,target,F.reshape(tensor,target)):
  Shape.Both{usable,contract} = shaped
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,shape,fill,Tensors.Usable{},workers},tensor,equation)
  %equation : Global.Shaped(~F32,~Logical,~original,logical,target,F.reshape(_,target))
  ReshapeValues.reshape(~F32,~Logical,~original,logical,target,bands,geometry,shape,fill,workers,
    %back : Global.Contract(~F32,~Logical,~original,logical,_) contract,
    %back : {Contract.tensor_ready(~F32,F.reshape(_,target)) == True{} : Bool} accepted)

def reshaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+shape: List<&2,U32>,+target: List<&2,U32>,tensor: Tensors.Tensor<F32>,
  holds: Holds(~Logical,~original,logical,shape,tensor)) -> Holds(~Logical,~original,logical,target,F.reshape(tensor,target)):
  +ready => reshaped_shaped(~Logical,~original,logical,shape,target,tensor,holds(Flow.reshaped(~F32,tensor,target,ready)),ready)

# The sums of each channel of [N,C,H,W] over its images and positions, in the
# row-major order of (n,h,w).
def Lines(~Logical: Data) -> Data:
  Reduce.Lines<Sum.Lanes,Reduce.Ordered<Logical>>

def channel_lines(~Logical: Data,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32) -> Lines(~Logical):
  Reduce.Lines{Sum.zero(),Reduce.source(~Logical,logical,[images,channels,height,width],[1n],[0n,2n,3n]),ShapeLaws.elements([images,height,width])}

def summed_shaped(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32,-tensor: Tensors.Tensor<F32>,
  shaped: Global.Shaped(~F32,~Logical,~original,logical,[images,channels,height,width],tensor),
  accepted: {Contract.tensor_ready(~F32,F.sum_axes(tensor,[0,2,3])) == True{} : Bool}) ->
  Global.Shaped(~F32,~Lines(~Logical),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Logical>,~Reduce.ordered_value(~F32,~Logical,~original),~Sum.step,~Sum.total),
    channel_lines(~Logical,logical,images,channels,height,width),[channels],F.sum_axes(tensor,[0,2,3])):
  Shape.Both{usable,contract} = shaped
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},tensor,equation)
  +ready = {%back : {Contract.tensor_ready(~F32,F.sum_axes(_,[0,2,3])) == True{} : Bool} accepted :
    {Contract.tensor_ready(~F32,F.sum_axes(Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},[0,2,3])) == True{} : Bool}}
  %equation : Global.Shaped(~F32,~Lines(~Logical),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Logical>,~Reduce.ordered_value(~F32,~Logical,~original),~Sum.step,~Sum.total),
    channel_lines(~Logical,logical,images,channels,height,width),[channels],F.sum_axes(_,[0,2,3]))
  Global.shaped(~F32,~Lines(~Logical),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Logical>,~Reduce.ordered_value(~F32,~Logical,~original),~Sum.step,~Sum.total),
    channel_lines(~Logical,logical,images,channels,height,width),[channels],
    F.sum_axes(Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},[0,2,3]),ready,
    Reduce.counted_extents(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"sum_axes",6.0,Tensors.count_of([channels]),Tensors.count_of([images,height,width]),Sum.zero(),
      [images,channels,height,width],[1n],[0n,2n,3n],fill,workers,Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers}),
    Reduce.sum_axes(~Logical,~original,logical,[0,2,3],bands,geometry,[images,channels,height,width],fill,workers,
      %back : Global.Contract(~F32,~Logical,~original,logical,_) contract,ready))

def summed(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+images: U32,+channels: U32,+height: U32,+width: U32,tensor: Tensors.Tensor<F32>,
  holds: Holds(~Logical,~original,logical,[images,channels,height,width],tensor)) ->
  Holds(~Lines(~Logical),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Logical>,~Reduce.ordered_value(~F32,~Logical,~original),~Sum.step,~Sum.total),
    channel_lines(~Logical,logical,images,channels,height,width),[channels],F.sum_axes(tensor,[0,2,3])):
  +ready => summed_shaped(~Logical,~original,logical,images,channels,height,width,tensor,
    holds(Flow.reduced(~F32,~Sum.Lanes,~Sum.step,~Sum.total,tensor,"sum_axes",6.0,Sum.zero(),[0,2,3],ready)),ready)

# A tensor of ones.
def full_extents(+shape: List<&2,U32>,+value: F32,count: Maybe<&2,U32>) -> {Contract.extents(~F32,Tensors.full_count(~F32,shape,count,value)) == shape : List<&2,U32>}:
  match count:
    case Some{count}: {==}
    case None{}: {==}

def ones(+shape: List<&2,U32>) -> Holds(~F32,~FullValues.constant(~F32),1.0,shape,F.ones(shape)):
  +ready => Global.shaped(~F32,~F32,~FullValues.constant(~F32),1.0,shape,F.ones(shape),ready,full_extents(shape,1.0,Tensors.count_of(shape)),FullValues.full(~F32,shape,1.0,ready))

# A tensor described by Data, its bands as a model. Steps name a tensor as
# often as they need through its owner.
def Owner() -> Data:
  Observation.TensorOwnerModel<F32>

def seen(+owner: Owner()) -> Tensors.Tensor<F32>:
  Observation.tensor_owner_pack(~F32,owner)

# What a step does with the owners of the two copies of a tensor, each of
# which holds the tensor's values once Ready.
def Copies(~Logical: Data,~original: Logical -> U32 -> F32,~Goal: Tensors.Tensor<F32> & Tensors.Tensor<F32> -> Type,+logical: Logical,+shape: List<&2,U32>) -> Type:
  @+first: Owner() -> @+second: Owner() -> @first_holds: Holds(~Logical,~original,logical,shape,seen(first)) ->
    @second_holds: Holds(~Logical,~original,logical,shape,seen(second)) -> Goal((seen(first),seen(second)))

def copied_model(~Logical: Data,~original: Logical -> U32 -> F32,~Goal: Tensors.Tensor<F32> & Tensors.Tensor<F32> -> Type,+logical: Logical,+shape: List<&2,U32>,
  +model: Model.Model<F32>,+geometry: Bands.Geometry,+extents: List<&2,U32>,+fill: F32,+workers: U32,evidence: Copy.Copied<F32,model>,
  shaped: Global.Shaped(~F32,~Logical,~original,logical,shape,Tensors.Tensor{Model.pack(~F32,model),geometry,extents,fill,Tensors.Usable{},workers}),
  body: Copies(~Logical,~original,~Goal,logical,shape)) ->
  Goal(F.copy(Tensors.Tensor{Model.pack(~F32,model),geometry,extents,fill,Tensors.Usable{},workers})):
  Copy.Copied{+left,+right,+equation,+left_content,+right_content} = evidence
  Shape.Both{usable,+contract} = shaped
  Global.Usable{+bands,+usable_geometry,+usable_fill,+usable_workers,+usable_equation} = usable
  +measured = Equal.sym(List<&2,U32>,shape,extents,Equal.cong(Tensors.Tensor<F32>,List<&2,U32>,tensor => Contract.extents(~F32,tensor),
    Tensors.Tensor{Model.pack(~F32,bands),usable_geometry,shape,usable_fill,Tensors.Usable{},usable_workers},
    Tensors.Tensor{Model.pack(~F32,model),geometry,extents,fill,Tensors.Usable{},workers},usable_equation))
  %equation : Goal(Storage.bind_pair(~Bands.Bands(~F32),~Bands.Bands(~F32),~(Tensors.Tensor<F32> & Tensors.Tensor<F32>),_,first => second =>
    (Tensors.Tensor{first,geometry,extents,fill,Tensors.Usable{},workers},Tensors.Tensor{second,geometry,extents,fill,Tensors.Usable{},workers})))
  body(Observation.TensorOwner{left,geometry,extents,fill,Tensors.Usable{},workers},Observation.TensorOwner{right,geometry,extents,fill,Tensors.Usable{},workers},
    +ready => Global.shaped(~F32,~Logical,~original,logical,shape,Tensors.Tensor{Model.pack(~F32,left),geometry,extents,fill,Tensors.Usable{},workers},ready,measured,
      Copy.kept(~F32,~Logical,~original,logical,model,left,geometry,extents,fill,workers,left_content,contract)),
    +ready => Global.shaped(~F32,~Logical,~original,logical,shape,Tensors.Tensor{Model.pack(~F32,right),geometry,extents,fill,Tensors.Usable{},workers},ready,measured,
      Copy.kept(~F32,~Logical,~original,logical,model,right,geometry,extents,fill,workers,right_content,contract)))

def copied_usable(~Logical: Data,~original: Logical -> U32 -> F32,~Goal: Tensors.Tensor<F32> & Tensors.Tensor<F32> -> Type,+logical: Logical,+shape: List<&2,U32>,
  -bands: Bands.Bands(~F32),+geometry: Bands.Geometry,+extents: List<&2,U32>,+fill: F32,+workers: U32,observation: Model.Observation<F32,bands>,
  shaped: Global.Shaped(~F32,~Logical,~original,logical,shape,Tensors.Tensor{bands,geometry,extents,fill,Tensors.Usable{},workers}),
  body: Copies(~Logical,~original,~Goal,logical,shape)) ->
  Goal(F.copy(Tensors.Tensor{bands,geometry,extents,fill,Tensors.Usable{},workers})):
  Model.Observation{+model,+equation} = observation
  %equation : Goal(F.copy(Tensors.Tensor{_,geometry,extents,fill,Tensors.Usable{},workers}))
  copied_model(~Logical,~original,~Goal,logical,shape,model,geometry,extents,fill,workers,Copy.copied(~F32,model),
    %Equal.sym(Bands.Bands(~F32),Model.pack(~F32,model),bands,equation) :
      Global.Shaped(~F32,~Logical,~original,logical,shape,Tensors.Tensor{_,geometry,extents,fill,Tensors.Usable{},workers})
    shaped,body)

# Both copies of a failed tensor are failed.
def copied_failed(~Logical: Data,~original: Logical -> U32 -> F32,~Goal: Tensors.Tensor<F32> & Tensors.Tensor<F32> -> Type,+logical: Logical,+shape: List<&2,U32>,
  -bands: Bands.Bands(~F32),+geometry: Bands.Geometry,+extents: List<&2,U32>,+fill: F32,+problem: Tensors.Problem,+workers: U32,observation: Model.Observation<F32,bands>,
  whole: Model.Observation<F32,Partition.Leaf{Tensors.buffer(~F32,fill,0),0,1,Unit{}}>,body: Copies(~Logical,~original,~Goal,logical,shape)) ->
  Goal(F.copy(Tensors.Tensor{bands,geometry,extents,fill,Tensors.Failed{problem},workers})):
  Model.Observation{+model,+equation} = observation
  Model.Observation{+whole_model,+whole_equation} = whole
  +first = {Observation.TensorOwner{model,geometry,extents,fill,Tensors.Failed{problem},workers} : Owner()}
  +second = {Observation.TensorOwner{whole_model,Bands.Geometry{1,1,1,Tensors.count_or_zero(extents)},extents,fill,Tensors.Failed{problem},workers} : Owner()}
  %equation : Goal((Tensors.Tensor{_,geometry,extents,fill,Tensors.Failed{problem},workers},
    Tensors.whole(~F32,Tensors.buffer(~F32,fill,0),extents,fill,Tensors.Failed{problem},workers)))
  %whole_equation : Goal((Tensors.Tensor{Model.pack(~F32,model),geometry,extents,fill,Tensors.Failed{problem},workers},
    Tensors.Tensor{_,Bands.Geometry{1,1,1,Tensors.count_or_zero(extents)},extents,fill,Tensors.Failed{problem},workers}))
  body(first,second,
    ready => Empty.absurd(Global.Shaped(~F32,~Logical,~original,logical,shape,seen(first)),Words.false_true(ready)),
    ready => Empty.absurd(Global.Shaped(~F32,~Logical,~original,logical,shape,seen(second)),Words.false_true(ready)))

# The two copies of a tensor each hold its values once Ready.
def copied(~Logical: Data,~original: Logical -> U32 -> F32,~Goal: Tensors.Tensor<F32> & Tensors.Tensor<F32> -> Type,+logical: Logical,+shape: List<&2,U32>,
  tensor: Tensors.Tensor<F32>,holds: Holds(~Logical,~original,logical,shape,tensor),body: Copies(~Logical,~original,~Goal,logical,shape)) ->
  Goal(F.copy(tensor)):
  match tensor:
    case Tensors.Tensor{bands,+geometry,+extents,+fill,status,+workers}:
      match status:
        case Tensors.Usable{}:
          copied_usable(~Logical,~original,~Goal,logical,shape,bands,geometry,extents,fill,workers,Model.observe(~F32,bands),holds({==}),body)
        case Tensors.Failed{+problem}:
          copied_failed(~Logical,~original,~Goal,logical,shape,bands,geometry,extents,fill,problem,workers,Model.observe(~F32,bands),
            Model.observe(~F32,Partition.Leaf{Tensors.buffer(~F32,fill,0),0,1,Unit{}}),body)

# [C] seen as [1,C,1,1] broadcasts to [N,C,H,W].
def spread_broadcast(+images: U32,+channels: U32,+height: U32,+width: U32) ->
  {Shapes.broadcast([images,channels,height,width],[1,channels,1,1]) == Some{[images,channels,height,width]} : Maybe<&2,List<&2,U32>>}:
  %Equal.sym(Bool,U32.is_eq(channels,channels),True{},BandShape.equal_words(channels,channels,{==})) :
    {Shapes.prepend_extent(Shapes.broadcast_extent(U32.is_eq(images,1),U32.is_eq(images,1),True{},images,1),
      Shapes.prepend_extent(Shapes.broadcast_extent(_,U32.is_eq(channels,1),U32.is_eq(channels,1),channels,channels),
        Shapes.prepend_extent(Shapes.broadcast_extent(U32.is_eq(height,1),U32.is_eq(height,1),True{},height,1),
          Shapes.prepend_extent(Shapes.broadcast_extent(U32.is_eq(width,1),U32.is_eq(width,1),True{},width,1),Some{[]})))) ==
      Some{[images,channels,height,width]} : Maybe<&2,List<&2,U32>>}
  %Equal.sym(Maybe<&2,U32>,Shapes.broadcast_extent(U32.is_eq(images,1),U32.is_eq(images,1),True{},images,1),Some{images},
      UnitBroadcast.singleton(images,U32.is_eq(images,1),U32.is_eq(images,1),{==},{==})) :
    {Shapes.prepend_extent(_,Shapes.prepend_extent(Some{channels},
        Shapes.prepend_extent(Shapes.broadcast_extent(U32.is_eq(height,1),U32.is_eq(height,1),True{},height,1),
          Shapes.prepend_extent(Shapes.broadcast_extent(U32.is_eq(width,1),U32.is_eq(width,1),True{},width,1),Some{[]})))) ==
      Some{[images,channels,height,width]} : Maybe<&2,List<&2,U32>>}
  %Equal.sym(Maybe<&2,U32>,Shapes.broadcast_extent(U32.is_eq(height,1),U32.is_eq(height,1),True{},height,1),Some{height},
      UnitBroadcast.singleton(height,U32.is_eq(height,1),U32.is_eq(height,1),{==},{==})) :
    {Shapes.prepend_extent(Some{images},Shapes.prepend_extent(Some{channels},Shapes.prepend_extent(_,
          Shapes.prepend_extent(Shapes.broadcast_extent(U32.is_eq(width,1),U32.is_eq(width,1),True{},width,1),Some{[]})))) ==
      Some{[images,channels,height,width]} : Maybe<&2,List<&2,U32>>}
  %Equal.sym(Maybe<&2,U32>,Shapes.broadcast_extent(U32.is_eq(width,1),U32.is_eq(width,1),True{},width,1),Some{width},
      UnitBroadcast.singleton(width,U32.is_eq(width,1),U32.is_eq(width,1),{==},{==})) :
    {Shapes.prepend_extent(Some{images},Shapes.prepend_extent(Some{channels},Shapes.prepend_extent(Some{height},Shapes.prepend_extent(_,Some{[]})))) ==
      Some{[images,channels,height,width]} : Maybe<&2,List<&2,U32>>}
  {==}

# ---- The values ----

# x - m, with m [C] read at the channel of each index of x [N,C,H,W].
def Centered(~X: Data,~M: Data) -> Data:
  Combined(~X,~M)

def centered_value(~X: Data,~x_value: X -> U32 -> F32,~M: Data,~m_value: M -> U32 -> F32,context: Centered(~X,~M),index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<X>,~Broadcast.Expanded<M>,~F32.sub,~Broadcast.value(~F32,~X,~x_value),~Broadcast.value(~F32,~M,~m_value),context,index)

def centered_context(~X: Data,~M: Data,+x: X,+m: M,+images: U32,+channels: U32,+height: U32,+width: U32) -> Centered(~X,~M):
  combined_context(~X,~M,x,m,[images,channels,height,width],[1,channels,1,1])

# 1 / sqrt(v + epsilon) of v [C].
def Root(~V: Data) -> Data:
  BandMap.Context<Unit,BandMap.Context<F32,V>>

def root_value(~V: Data,~v_value: V -> U32 -> F32,context: Root(~V),index: U32) -> F32:
  BandMap.value(~F32,~Unit,~BandMap.Context<F32,V>,~Access.apply_element(~F32,~Arithmetic.sqrt),
    ~BandMap.value(~F32,~F32,~V,~Bands.scalar_value(~F32,~F32.add),~v_value),context,index)

def Inverse(~V: Data) -> Data:
  Combined(~F32,~Root(~V))

def inverse_value(~V: Data,~v_value: V -> U32 -> F32,context: Inverse(~V),index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<F32>,~Broadcast.Expanded<Root(~V)>,~F32.div,~Broadcast.value(~F32,~F32,~FullValues.constant(~F32)),
    ~Broadcast.value(~F32,~Root(~V),~root_value(~V,~v_value)),context,index)

def inverse_context(~V: Data,+v: V,+channels: U32,+epsilon: F32) -> Inverse(~V):
  combined_context(~F32,~Root(~V),1.0,BandMap.Context{Unit{},BandMap.Context{epsilon,v}},[channels],[channels])

# (x - m) * (1 / sqrt(v + epsilon)), the inverse read at the channel of each
# index.
def Normalized(~X: Data,~M: Data,~V: Data) -> Data:
  Combined(~Centered(~X,~M),~Inverse(~V))

def normalized_value(~X: Data,~x_value: X -> U32 -> F32,~M: Data,~m_value: M -> U32 -> F32,~V: Data,~v_value: V -> U32 -> F32,context: Normalized(~X,~M,~V),
  index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<Centered(~X,~M)>,~Broadcast.Expanded<Inverse(~V)>,~F32.mul,~Broadcast.value(~F32,~Centered(~X,~M),~centered_value(~X,~x_value,~M,~m_value)),
    ~Broadcast.value(~F32,~Inverse(~V),~inverse_value(~V,~v_value)),context,index)

def normalized_context(~X: Data,~M: Data,~V: Data,+x: X,+m: M,+v: V,+images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32) -> Normalized(~X,~M,~V):
  combined_context(~Centered(~X,~M),~Inverse(~V),centered_context(~X,~M,x,m,images,channels,height,width),inverse_context(~V,v,channels,epsilon),
    [images,channels,height,width],[1,channels,1,1])

# n * gamma + beta, gamma and beta read at the channel of each index.
def Affine(~N: Data,~G: Data,~B: Data) -> Data:
  Combined(~Combined(~N,~G),~B)

def affine_value(~N: Data,~n_value: N -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,context: Affine(~N,~G,~B),index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<Combined(~N,~G)>,~Broadcast.Expanded<B>,~F32.add,
    ~Broadcast.value(~F32,~Combined(~N,~G),~Leaf.value(~F32,~Broadcast.Expanded<N>,~Broadcast.Expanded<G>,~F32.mul,~Broadcast.value(~F32,~N,~n_value),
      ~Broadcast.value(~F32,~G,~g_value))),
    ~Broadcast.value(~F32,~B,~b_value),context,index)

def affine_context(~N: Data,~G: Data,~B: Data,+n: N,+gamma: G,+beta: B,+images: U32,+channels: U32,+height: U32,+width: U32) -> Affine(~N,~G,~B):
  combined_context(~Combined(~N,~G),~B,combined_context(~N,~G,n,gamma,[images,channels,height,width],[1,channels,1,1]),beta,[images,channels,height,width],[1,channels,1,1])

# running * (1 - momentum) + statistic * momentum.
def Moved(~R: Data,~S: Data) -> Data:
  Combined(~BandMap.Context<F32,R>,~BandMap.Context<F32,S>)

def moved_value(~R: Data,~r_value: R -> U32 -> F32,~S: Data,~s_value: S -> U32 -> F32,context: Moved(~R,~S),index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<BandMap.Context<F32,R>>,~Broadcast.Expanded<BandMap.Context<F32,S>>,~F32.add,
    ~Broadcast.value(~F32,~BandMap.Context<F32,R>,~BandMap.value(~F32,~F32,~R,~Bands.scalar_value(~F32,~F32.mul),~r_value)),
    ~Broadcast.value(~F32,~BandMap.Context<F32,S>,~BandMap.value(~F32,~F32,~S,~Bands.scalar_value(~F32,~F32.mul),~s_value)),context,index)

def moved_context(~R: Data,~S: Data,+running: R,+statistic: S,+channels: U32,+momentum: F32) -> Moved(~R,~S):
  combined_context(~BandMap.Context<F32,R>,~BandMap.Context<F32,S>,BandMap.Context{(1.0 - momentum : F32),running},BandMap.Context{momentum,statistic},[channels],[channels])

# The statistics of a batch: mean = sum_axes(x,[0,2,3]) / M with M = N * H * W.
def Mean(~X: Data) -> Data:
  BandMap.Context<F32,Lines(~X)>

def mean_value(~X: Data,~x_value: X -> U32 -> F32,context: Mean(~X),index: U32) -> F32:
  BandMap.value(~F32,~F32,~Lines(~X),~Bands.scalar_value(~F32,~F32.div),
    ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<X>,~Reduce.ordered_value(~F32,~X,~x_value),~Sum.step,~Sum.total),context,index)

def mean_context(~X: Data,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32) -> Mean(~X):
  BandMap.Context{U32.to_f32(Norm.count([images,channels,height,width])),channel_lines(~X,x,images,channels,height,width)}

# d * d of d = x - mean.
def Square(~X: Data) -> Data:
  Combined(~Centered(~X,~Mean(~X)),~Centered(~X,~Mean(~X)))

def square_value(~X: Data,~x_value: X -> U32 -> F32,context: Square(~X),index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<Centered(~X,~Mean(~X))>,~Broadcast.Expanded<Centered(~X,~Mean(~X))>,~F32.mul,
    ~Broadcast.value(~F32,~Centered(~X,~Mean(~X)),~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value))),
    ~Broadcast.value(~F32,~Centered(~X,~Mean(~X)),~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value))),context,index)

def square_context(~X: Data,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32) -> Square(~X):
  +centered = centered_context(~X,~Mean(~X),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width)
  combined_context(~Centered(~X,~Mean(~X)),~Centered(~X,~Mean(~X)),centered,centered,[images,channels,height,width],[images,channels,height,width])

# The channel sums of d * d divided by a count: M for the variance that
# normalizes, M - 1 for the running variance.
def Variance(~X: Data) -> Data:
  BandMap.Context<F32,Lines(~Square(~X))>

def variance_value(~X: Data,~x_value: X -> U32 -> F32,context: Variance(~X),index: U32) -> F32:
  BandMap.value(~F32,~F32,~Lines(~Square(~X)),~Bands.scalar_value(~F32,~F32.div),
    ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Square(~X)>,~Reduce.ordered_value(~F32,~Square(~X),~square_value(~X,~x_value)),~Sum.step,~Sum.total),context,index)

def variance_context(~X: Data,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,+count: U32) -> Variance(~X):
  BandMap.Context{U32.to_f32(count),channel_lines(~Square(~X),square_context(~X,x,images,channels,height,width),images,channels,height,width)}

# What forward returns, once each tensor is Ready: the output, the
# normalized input, the inverse deviation, gamma and beta as they were given,
# and the moved running statistics.
def ForwardHolds(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,result: Norm.Forward()) -> Type:
  (y,((xhat,deviation),((gammas,betas),(means,variances)))) = result
  +count = Norm.count([images,channels,height,width])
  +statistic = mean_context(~X,x,images,channels,height,width)
  +biased = variance_context(~X,x,images,channels,height,width,count)
  +normalized = normalized_context(~X,~Mean(~X),~Variance(~X),x,statistic,biased,images,channels,height,width,epsilon)
  Holds(~Affine(~Normalized(~X,~Mean(~X),~Variance(~X)),~G,~B),
    ~affine_value(~Normalized(~X,~Mean(~X),~Variance(~X)),~normalized_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value),~Variance(~X),~variance_value(~X,~x_value)),
      ~G,~g_value,~B,~b_value),
    affine_context(~Normalized(~X,~Mean(~X),~Variance(~X)),~G,~B,normalized,gamma,beta,images,channels,height,width),[images,channels,height,width],y) &
  (Holds(~Normalized(~X,~Mean(~X),~Variance(~X)),~normalized_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value),~Variance(~X),~variance_value(~X,~x_value)),normalized,
    [images,channels,height,width],xhat) &
  (Holds(~Inverse(~Variance(~X)),~inverse_value(~Variance(~X),~variance_value(~X,~x_value)),inverse_context(~Variance(~X),biased,channels,epsilon),[channels],deviation) &
  (Holds(~G,~g_value,gamma,[channels],gammas) &
  (Holds(~B,~b_value,beta,[channels],betas) &
  (Holds(~Moved(~RM,~Mean(~X)),~moved_value(~RM,~rm_value,~Mean(~X),~mean_value(~X,~x_value)),moved_context(~RM,~Mean(~X),mean,statistic,channels,momentum),[channels],
    means) &
  Holds(~Moved(~RV,~Variance(~X)),~moved_value(~RV,~rv_value,~Variance(~X),~variance_value(~X,~x_value)),
    moved_context(~RV,~Variance(~X),variance,variance_context(~X,x,images,channels,height,width,(count - 1 : U32)),channels,momentum),[channels],variances))))))

# A combination read through the reader of its stage.
def combined_as(~Left: Data,~Right: Data,~After: Data,~operation: F32 -> F32 -> F32,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  ~after_value: After -> U32 -> F32,+left_context: Left,+right_context: Right,+after: After,+name: String,+shape: List<&2,U32>,+right_shape: List<&2,U32>,
  agrees: @+index: U32 -> {Leaf.value(~F32,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~Broadcast.value(~F32,~Left,~left_value),
    ~Broadcast.value(~F32,~Right,~right_value),combined_context(~Left,~Right,left_context,right_context,shape,right_shape),index) == after_value(after,index) : F32},
  left: Tensors.Tensor<F32>,right: Tensors.Tensor<F32>,
  left_holds: Holds(~Left,~left_value,left_context,shape,left),right_holds: Holds(~Right,~right_value,right_context,right_shape,right),
  +common: {Shapes.broadcast(shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>}) ->
  Holds(~After,~after_value,after,shape,Tensors.combine(~F32,~operation,name,1.0,left,right)):
  restated(~Combined(~Left,~Right),~After,
    ~Leaf.value(~F32,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~Broadcast.value(~F32,~Left,~left_value),~Broadcast.value(~F32,~Right,~right_value)),
    ~after_value,combined_context(~Left,~Right,left_context,right_context,shape,right_shape),after,agrees,shape,Tensors.combine(~F32,~operation,name,1.0,left,right),
    combined(~Left,~Right,~operation,~left_value,~right_value,left_context,right_context,name,shape,right_shape,left,right,left_holds,right_holds,common))

# What a step does with the owner of a tensor, which holds the tensor's
# values once Ready.
def Seen(~Logical: Data,~original: Logical -> U32 -> F32,~Goal: Tensors.Tensor<F32> -> Type,+logical: Logical,+shape: List<&2,U32>) -> Type:
  @+owner: Owner() -> @holds: Holds(~Logical,~original,logical,shape,seen(owner)) -> Goal(seen(owner))

def observed_owner(~Logical: Data,~original: Logical -> U32 -> F32,~Goal: Tensors.Tensor<F32> -> Type,+logical: Logical,+shape: List<&2,U32>,-tensor: Tensors.Tensor<F32>,
  observation: Observation.TensorObservation<F32,tensor>,holds: Holds(~Logical,~original,logical,shape,tensor),body: Seen(~Logical,~original,~Goal,logical,shape)) ->
  Goal(tensor):
  Observation.TensorObservation{+owner,+equation} = observation
  %equation : Goal(_)
  body(owner,%Equal.sym(Tensors.Tensor<F32>,Observation.tensor_owner_pack(~F32,owner),tensor,equation) : Holds(~Logical,~original,logical,shape,_) holds)

# A tensor, described by its owner.
def observed(~Logical: Data,~original: Logical -> U32 -> F32,~Goal: Tensors.Tensor<F32> -> Type,+logical: Logical,+shape: List<&2,U32>,tensor: Tensors.Tensor<F32>,
  holds: Holds(~Logical,~original,logical,shape,tensor),body: Seen(~Logical,~original,~Goal,logical,shape)) -> Goal(tensor):
  observed_owner(~Logical,~original,~Goal,logical,shape,tensor,Observation.observe_tensor(~F32,tensor),holds,body)

# ---- The shared steps ----

# x - spread(m).
def centered_step(~X: Data,~x_value: X -> U32 -> F32,~M: Data,~m_value: M -> U32 -> F32,+x: X,+m: M,+images: U32,+channels: U32,+height: U32,+width: U32,
  +inputs: Owner(),+statistic: Owner(),input_holds: Holds(~X,~x_value,x,[images,channels,height,width],seen(inputs)),
  statistic_holds: Holds(~M,~m_value,m,[channels],seen(statistic))) ->
  Holds(~Centered(~X,~M),~centered_value(~X,~x_value,~M,~m_value),centered_context(~X,~M,x,m,images,channels,height,width),[images,channels,height,width],
    F.sub(seen(inputs),Norm.spread(seen(statistic),channels))):
  combined_as(~X,~M,~Centered(~X,~M),~F32.sub,~x_value,~m_value,~centered_value(~X,~x_value,~M,~m_value),x,m,centered_context(~X,~M,x,m,images,channels,height,width),"sub",
    [images,channels,height,width],[1,channels,1,1],index => {==},seen(inputs),Norm.spread(seen(statistic),channels),input_holds,
    reshaped(~M,~m_value,m,[channels],[1,channels,1,1],seen(statistic),statistic_holds),spread_broadcast(images,channels,height,width))

# 1 / sqrt(v + epsilon).
def inverse(~V: Data,~v_value: V -> U32 -> F32,+v: V,+channels: U32,+epsilon: F32,+variance: Owner(),variance_holds: Holds(~V,~v_value,v,[channels],seen(variance))) ->
  Holds(~Inverse(~V),~inverse_value(~V,~v_value),inverse_context(~V,v,channels,epsilon),[channels],Norm.inverse_deviation(seen(variance),channels,epsilon)):
  +root_context = {BandMap.Context{Unit{},BandMap.Context{epsilon,v}} : Root(~V)}
  combined_as(~F32,~Root(~V),~Inverse(~V),~F32.div,~FullValues.constant(~F32),~root_value(~V,~v_value),~inverse_value(~V,~v_value),1.0,root_context,
    inverse_context(~V,v,channels,epsilon),"div",[channels],[channels],index => {==},F.ones([channels]),F.sqrt(F.add_scalar(seen(variance),epsilon)),ones([channels]),
    restated(~BandMap.Context<Unit,BandMap.Context<F32,V>>,~Root(~V),
      ~BandMap.value(~F32,~Unit,~BandMap.Context<F32,V>,~Access.apply_element(~F32,~Arithmetic.sqrt),~BandMap.value(~F32,~F32,~V,~Bands.scalar_value(~F32,~F32.add),~v_value)),
      ~root_value(~V,~v_value),root_context,root_context,index => {==},[channels],F.sqrt(F.add_scalar(seen(variance),epsilon)),
      root(~BandMap.Context<F32,V>,~BandMap.value(~F32,~F32,~V,~Bands.scalar_value(~F32,~F32.add),~v_value),BandMap.Context{epsilon,v},[channels],
        F.add_scalar(seen(variance),epsilon),scaled(~V,~v_value,~F32.add,v,[channels],epsilon,seen(variance),variance_holds))),
    UnitBroadcast.self_broadcast([channels]))

# The same of a tensor, described by its owner.
def inverse_of(~V: Data,~v_value: V -> U32 -> F32,+v: V,+channels: U32,+epsilon: F32,variance: Tensors.Tensor<F32>,variance_holds: Holds(~V,~v_value,v,[channels],variance)) ->
  Holds(~Inverse(~V),~inverse_value(~V,~v_value),inverse_context(~V,v,channels,epsilon),[channels],Norm.inverse_deviation(variance,channels,epsilon)):
  observed(~V,~v_value,~(tensor => Holds(~Inverse(~V),~inverse_value(~V,~v_value),inverse_context(~V,v,channels,epsilon),[channels],
      Norm.inverse_deviation(tensor,channels,epsilon))),
    v,[channels],variance,variance_holds,owner => holds => inverse(~V,~v_value,v,channels,epsilon,owner,holds))

# centered * spread(inverse).
def normalized_step(~X: Data,~x_value: X -> U32 -> F32,~M: Data,~m_value: M -> U32 -> F32,~V: Data,~v_value: V -> U32 -> F32,+x: X,+m: M,+v: V,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+centered: Owner(),+deviation: Owner(),
  centered_holds: Holds(~Centered(~X,~M),~centered_value(~X,~x_value,~M,~m_value),centered_context(~X,~M,x,m,images,channels,height,width),[images,channels,height,width],
    seen(centered)),
  deviation_holds: Holds(~Inverse(~V),~inverse_value(~V,~v_value),inverse_context(~V,v,channels,epsilon),[channels],seen(deviation))) ->
  Holds(~Normalized(~X,~M,~V),~normalized_value(~X,~x_value,~M,~m_value,~V,~v_value),normalized_context(~X,~M,~V,x,m,v,images,channels,height,width,epsilon),
    [images,channels,height,width],F.mul(seen(centered),Norm.spread(seen(deviation),channels))):
  combined_as(~Centered(~X,~M),~Inverse(~V),~Normalized(~X,~M,~V),~F32.mul,~centered_value(~X,~x_value,~M,~m_value),~inverse_value(~V,~v_value),
    ~normalized_value(~X,~x_value,~M,~m_value,~V,~v_value),centered_context(~X,~M,x,m,images,channels,height,width),inverse_context(~V,v,channels,epsilon),
    normalized_context(~X,~M,~V,x,m,v,images,channels,height,width,epsilon),"mul",[images,channels,height,width],[1,channels,1,1],index => {==},
    seen(centered),Norm.spread(seen(deviation),channels),centered_holds,
    reshaped(~Inverse(~V),~inverse_value(~V,~v_value),inverse_context(~V,v,channels,epsilon),[channels],[1,channels,1,1],seen(deviation),deviation_holds),
    spread_broadcast(images,channels,height,width))

# n * gamma + beta.
def affine_step(~N: Data,~n_value: N -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,+n: N,+gamma: G,+beta: B,
  +images: U32,+channels: U32,+height: U32,+width: U32,+normalized: Owner(),+gammas: Owner(),+betas: Owner(),
  normalized_holds: Holds(~N,~n_value,n,[images,channels,height,width],seen(normalized)),gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),
  beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas))) ->
  Holds(~Affine(~N,~G,~B),~affine_value(~N,~n_value,~G,~g_value,~B,~b_value),affine_context(~N,~G,~B,n,gamma,beta,images,channels,height,width),[images,channels,height,width],
    F.add(F.mul(seen(normalized),Norm.spread(seen(gammas),channels)),Norm.spread(seen(betas),channels))):
  combined_as(~Combined(~N,~G),~B,~Affine(~N,~G,~B),~F32.add,
    ~Leaf.value(~F32,~Broadcast.Expanded<N>,~Broadcast.Expanded<G>,~F32.mul,~Broadcast.value(~F32,~N,~n_value),~Broadcast.value(~F32,~G,~g_value)),~b_value,
    ~affine_value(~N,~n_value,~G,~g_value,~B,~b_value),combined_context(~N,~G,n,gamma,[images,channels,height,width],[1,channels,1,1]),beta,
    affine_context(~N,~G,~B,n,gamma,beta,images,channels,height,width),"add",[images,channels,height,width],[1,channels,1,1],index => {==},
    F.mul(seen(normalized),Norm.spread(seen(gammas),channels)),Norm.spread(seen(betas),channels),
    combined(~N,~G,~F32.mul,~n_value,~g_value,n,gamma,"mul",[images,channels,height,width],[1,channels,1,1],seen(normalized),Norm.spread(seen(gammas),channels),
      normalized_holds,reshaped(~G,~g_value,gamma,[channels],[1,channels,1,1],seen(gammas),gamma_holds),spread_broadcast(images,channels,height,width)),
    reshaped(~B,~b_value,beta,[channels],[1,channels,1,1],seen(betas),beta_holds),spread_broadcast(images,channels,height,width))

# running * (1 - momentum) + statistic * momentum.
def moved(~R: Data,~r_value: R -> U32 -> F32,~S: Data,~s_value: S -> U32 -> F32,+running: R,+statistic: S,+channels: U32,+momentum: F32,
  +runnings: Owner(),+statistics: Owner(),running_holds: Holds(~R,~r_value,running,[channels],seen(runnings)),
  statistic_holds: Holds(~S,~s_value,statistic,[channels],seen(statistics))) ->
  Holds(~Moved(~R,~S),~moved_value(~R,~r_value,~S,~s_value),moved_context(~R,~S,running,statistic,channels,momentum),[channels],
    Norm.moved(seen(runnings),seen(statistics),momentum)):
  combined_as(~BandMap.Context<F32,R>,~BandMap.Context<F32,S>,~Moved(~R,~S),~F32.add,~BandMap.value(~F32,~F32,~R,~Bands.scalar_value(~F32,~F32.mul),~r_value),
    ~BandMap.value(~F32,~F32,~S,~Bands.scalar_value(~F32,~F32.mul),~s_value),~moved_value(~R,~r_value,~S,~s_value),BandMap.Context{(1.0 - momentum : F32),running},
    BandMap.Context{momentum,statistic},moved_context(~R,~S,running,statistic,channels,momentum),"add",[channels],[channels],index => {==},
    F.mul_scalar(seen(runnings),(1.0 - momentum : F32)),F.mul_scalar(seen(statistics),momentum),
    scaled(~R,~r_value,~F32.mul,running,[channels],(1.0 - momentum : F32),seen(runnings),running_holds),
    scaled(~S,~s_value,~F32.mul,statistic,[channels],momentum,seen(statistics),statistic_holds),UnitBroadcast.self_broadcast([channels]))

# ---- Forward ----

# gamma and beta are copied: one copy scales and shifts xhat, the other comes
# back with the normalized input, the inverse deviation and the moved
# running statistics.
def normalized_forward(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,
  +statistic: Owner(),+unbiased: Owner(),+deviation: Owner(),+xhat: Owner(),+xhat_kept: Owner(),+gammas: Owner(),+betas: Owner(),+means: Owner(),+variances: Owner(),
  statistic_holds: Holds(~Mean(~X),~mean_value(~X,~x_value),mean_context(~X,x,images,channels,height,width),[channels],seen(statistic)),
  unbiased_holds: Holds(~Variance(~X),~variance_value(~X,~x_value),
    variance_context(~X,x,images,channels,height,width,(Norm.count([images,channels,height,width]) - 1 : U32)),[channels],seen(unbiased)),
  deviation_holds: Holds(~Inverse(~Variance(~X)),~inverse_value(~Variance(~X),~variance_value(~X,~x_value)),
    inverse_context(~Variance(~X),variance_context(~X,x,images,channels,height,width,Norm.count([images,channels,height,width])),channels,epsilon),[channels],
    seen(deviation)),
  xhat_holds: Holds(~Normalized(~X,~Mean(~X),~Variance(~X)),
    ~normalized_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value),~Variance(~X),~variance_value(~X,~x_value)),
    normalized_context(~X,~Mean(~X),~Variance(~X),x,mean_context(~X,x,images,channels,height,width),
      variance_context(~X,x,images,channels,height,width,Norm.count([images,channels,height,width])),images,channels,height,width,epsilon),
    [images,channels,height,width],seen(xhat)),
  kept_holds: Holds(~Normalized(~X,~Mean(~X),~Variance(~X)),
    ~normalized_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value),~Variance(~X),~variance_value(~X,~x_value)),
    normalized_context(~X,~Mean(~X),~Variance(~X),x,mean_context(~X,x,images,channels,height,width),
      variance_context(~X,x,images,channels,height,width,Norm.count([images,channels,height,width])),images,channels,height,width,epsilon),
    [images,channels,height,width],seen(xhat_kept)),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas)),
  mean_holds: Holds(~RM,~rm_value,mean,[channels],seen(means)),variance_holds: Holds(~RV,~rv_value,variance,[channels],seen(variances))) ->
  ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    Norm.normalized(channels,momentum,seen(statistic),seen(unbiased),seen(deviation),(seen(xhat),seen(xhat_kept)),(seen(gammas),seen(betas)),
      (seen(means),seen(variances)))):
  copied(~G,~g_value,~(pair => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
      momentum,Norm.finished(seen(xhat_kept),seen(deviation),Norm.moved(seen(means),seen(statistic),momentum),Norm.moved(seen(variances),seen(unbiased),momentum),
        Norm.affine(seen(xhat),channels,pair,F.copy(seen(betas)))))),
    gamma,[channels],seen(gammas),gamma_holds,
    gamma_first => gamma_second => gamma_first_holds => gamma_second_holds =>
      copied(~B,~b_value,~(pair => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,
          epsilon,momentum,Norm.finished(seen(xhat_kept),seen(deviation),Norm.moved(seen(means),seen(statistic),momentum),Norm.moved(seen(variances),seen(unbiased),momentum),
            Norm.affine(seen(xhat),channels,(seen(gamma_first),seen(gamma_second)),pair)))),
        beta,[channels],seen(betas),beta_holds,
        beta_first => beta_second => beta_first_holds => beta_second_holds =>
          (affine_step(~Normalized(~X,~Mean(~X),~Variance(~X)),
              ~normalized_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value),~Variance(~X),~variance_value(~X,~x_value)),~G,~g_value,~B,~b_value,
              normalized_context(~X,~Mean(~X),~Variance(~X),x,mean_context(~X,x,images,channels,height,width),
                variance_context(~X,x,images,channels,height,width,Norm.count([images,channels,height,width])),images,channels,height,width,epsilon),
              gamma,beta,images,channels,height,width,xhat,gamma_first,beta_first,xhat_holds,gamma_first_holds,beta_first_holds),
            (kept_holds,(deviation_holds,(gamma_second_holds,(beta_second_holds,
              (moved(~RM,~rm_value,~Mean(~X),~mean_value(~X,~x_value),mean,mean_context(~X,x,images,channels,height,width),channels,momentum,means,statistic,mean_holds,
                  statistic_holds),
                moved(~RV,~rv_value,~Variance(~X),~variance_value(~X,~x_value),variance,
                  variance_context(~X,x,images,channels,height,width,(Norm.count([images,channels,height,width]) - 1 : U32)),channels,momentum,variances,unbiased,
                  variance_holds,unbiased_holds)))))))))

# The inverse deviation is copied: one copy scales d, the other comes back.
def scaled_forward(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,
  +statistic: Owner(),+centered: Owner(),+unbiased: Owner(),+deviation: Owner(),+deviation_kept: Owner(),+gammas: Owner(),+betas: Owner(),+means: Owner(),+variances: Owner(),
  statistic_holds: Holds(~Mean(~X),~mean_value(~X,~x_value),mean_context(~X,x,images,channels,height,width),[channels],seen(statistic)),
  centered_holds: Holds(~Centered(~X,~Mean(~X)),~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value)),
    centered_context(~X,~Mean(~X),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width),[images,channels,height,width],seen(centered)),
  unbiased_holds: Holds(~Variance(~X),~variance_value(~X,~x_value),
    variance_context(~X,x,images,channels,height,width,(Norm.count([images,channels,height,width]) - 1 : U32)),[channels],seen(unbiased)),
  deviation_holds: Holds(~Inverse(~Variance(~X)),~inverse_value(~Variance(~X),~variance_value(~X,~x_value)),
    inverse_context(~Variance(~X),variance_context(~X,x,images,channels,height,width,Norm.count([images,channels,height,width])),channels,epsilon),[channels],
    seen(deviation)),
  kept_holds: Holds(~Inverse(~Variance(~X)),~inverse_value(~Variance(~X),~variance_value(~X,~x_value)),
    inverse_context(~Variance(~X),variance_context(~X,x,images,channels,height,width,Norm.count([images,channels,height,width])),channels,epsilon),[channels],
    seen(deviation_kept)),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas)),
  mean_holds: Holds(~RM,~rm_value,mean,[channels],seen(means)),variance_holds: Holds(~RV,~rv_value,variance,[channels],seen(variances))) ->
  ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    Norm.scaled(channels,momentum,seen(statistic),seen(centered),seen(unbiased),(seen(deviation),seen(deviation_kept)),(seen(gammas),seen(betas)),
      (seen(means),seen(variances)))):
  copied(~Normalized(~X,~Mean(~X),~Variance(~X)),~normalized_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value),~Variance(~X),~variance_value(~X,~x_value)),
    ~(pair => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
      Norm.normalized(channels,momentum,seen(statistic),seen(unbiased),seen(deviation_kept),pair,(seen(gammas),seen(betas)),(seen(means),seen(variances))))),
    normalized_context(~X,~Mean(~X),~Variance(~X),x,mean_context(~X,x,images,channels,height,width),
      variance_context(~X,x,images,channels,height,width,Norm.count([images,channels,height,width])),images,channels,height,width,epsilon),
    [images,channels,height,width],F.mul(seen(centered),Norm.spread(seen(deviation),channels)),
    normalized_step(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value),~Variance(~X),~variance_value(~X,~x_value),x,mean_context(~X,x,images,channels,height,width),
      variance_context(~X,x,images,channels,height,width,Norm.count([images,channels,height,width])),images,channels,height,width,epsilon,centered,deviation,
      centered_holds,deviation_holds),
    xhat => xhat_kept => xhat_holds => xhat_kept_holds =>
      normalized_forward(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
        statistic,unbiased,deviation_kept,xhat,xhat_kept,gammas,betas,means,variances,statistic_holds,unbiased_holds,kept_holds,xhat_holds,xhat_kept_holds,gamma_holds,
        beta_holds,mean_holds,variance_holds))

# The channel sums of d * d divided by a count, the variance before the
# running variance moves towards it.
def divided(~X: Data,~x_value: X -> U32 -> F32,+x: X,+images: U32,+channels: U32,+height: U32,+width: U32,+count: U32,+squares: Owner(),
  squares_holds: Holds(~Lines(~Square(~X)),
    ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Square(~X)>,~Reduce.ordered_value(~F32,~Square(~X),~square_value(~X,~x_value)),~Sum.step,~Sum.total),
    channel_lines(~Square(~X),square_context(~X,x,images,channels,height,width),images,channels,height,width),[channels],seen(squares))) ->
  Holds(~Variance(~X),~variance_value(~X,~x_value),variance_context(~X,x,images,channels,height,width,count),[channels],F.div_scalar(seen(squares),U32.to_f32(count))):
  restated(~BandMap.Context<F32,Lines(~Square(~X))>,~Variance(~X),
    ~BandMap.value(~F32,~F32,~Lines(~Square(~X)),~Bands.scalar_value(~F32,~F32.div),
      ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Square(~X)>,~Reduce.ordered_value(~F32,~Square(~X),~square_value(~X,~x_value)),~Sum.step,~Sum.total)),
    ~variance_value(~X,~x_value),variance_context(~X,x,images,channels,height,width,count),variance_context(~X,x,images,channels,height,width,count),index => {==},
    [channels],F.div_scalar(seen(squares),U32.to_f32(count)),
    scaled(~Lines(~Square(~X)),
      ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Square(~X)>,~Reduce.ordered_value(~F32,~Square(~X),~square_value(~X,~x_value)),~Sum.step,~Sum.total),
      ~F32.div,channel_lines(~Square(~X),square_context(~X,x,images,channels,height,width),images,channels,height,width),[channels],U32.to_f32(count),seen(squares),
      squares_holds))

# The channel sums s of d * d are copied: one copy gives the inverse
# deviation, the other the running variance.
def deviated_forward(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,
  +statistic: Owner(),+centered: Owner(),+squares: Owner(),+squares_kept: Owner(),+gammas: Owner(),+betas: Owner(),+means: Owner(),+variances: Owner(),
  statistic_holds: Holds(~Mean(~X),~mean_value(~X,~x_value),mean_context(~X,x,images,channels,height,width),[channels],seen(statistic)),
  centered_holds: Holds(~Centered(~X,~Mean(~X)),~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value)),
    centered_context(~X,~Mean(~X),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width),[images,channels,height,width],seen(centered)),
  squares_holds: Holds(~Lines(~Square(~X)),
    ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Square(~X)>,~Reduce.ordered_value(~F32,~Square(~X),~square_value(~X,~x_value)),~Sum.step,~Sum.total),
    channel_lines(~Square(~X),square_context(~X,x,images,channels,height,width),images,channels,height,width),[channels],seen(squares)),
  kept_holds: Holds(~Lines(~Square(~X)),
    ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Square(~X)>,~Reduce.ordered_value(~F32,~Square(~X),~square_value(~X,~x_value)),~Sum.step,~Sum.total),
    channel_lines(~Square(~X),square_context(~X,x,images,channels,height,width),images,channels,height,width),[channels],seen(squares_kept)),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas)),
  mean_holds: Holds(~RM,~rm_value,mean,[channels],seen(means)),variance_holds: Holds(~RV,~rv_value,variance,[channels],seen(variances))) ->
  ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    Norm.deviated(Norm.Settings{epsilon,momentum},channels,Norm.count([images,channels,height,width]),seen(statistic),seen(centered),(seen(squares),seen(squares_kept)),
      (seen(gammas),seen(betas)),(seen(means),seen(variances)))):
  +count = Norm.count([images,channels,height,width])
  +biased = variance_context(~X,x,images,channels,height,width,count)
  copied(~Inverse(~Variance(~X)),~inverse_value(~Variance(~X),~variance_value(~X,~x_value)),
    ~(pair => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
      Norm.scaled(channels,momentum,seen(statistic),seen(centered),F.div_scalar(seen(squares_kept),U32.to_f32((count - 1 : U32))),pair,(seen(gammas),seen(betas)),
        (seen(means),seen(variances))))),
    inverse_context(~Variance(~X),biased,channels,epsilon),[channels],Norm.inverse_deviation(F.div_scalar(seen(squares),U32.to_f32(count)),channels,epsilon),
    inverse_of(~Variance(~X),~variance_value(~X,~x_value),biased,channels,epsilon,F.div_scalar(seen(squares),U32.to_f32(count)),
      divided(~X,~x_value,x,images,channels,height,width,count,squares,squares_holds)),
    deviation => deviation_kept => deviation_holds => deviation_kept_holds =>
      observed(~Variance(~X),~variance_value(~X,~x_value),
        ~(unbiased => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
          Norm.scaled(channels,momentum,seen(statistic),seen(centered),unbiased,(seen(deviation),seen(deviation_kept)),(seen(gammas),seen(betas)),
            (seen(means),seen(variances))))),
        variance_context(~X,x,images,channels,height,width,(count - 1 : U32)),[channels],F.div_scalar(seen(squares_kept),U32.to_f32((count - 1 : U32))),
        divided(~X,~x_value,x,images,channels,height,width,(count - 1 : U32),squares_kept,kept_holds),
        unbiased => unbiased_holds =>
          scaled_forward(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
            statistic,centered,unbiased,deviation,deviation_kept,gammas,betas,means,variances,statistic_holds,centered_holds,unbiased_holds,deviation_holds,
            deviation_kept_holds,gamma_holds,beta_holds,mean_holds,variance_holds)))

# The channel sums of d * d of two copies of d.
def squared_forward(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,
  +statistic: Owner(),+centered: Owner(),+left: Owner(),+right: Owner(),+gammas: Owner(),+betas: Owner(),+means: Owner(),+variances: Owner(),
  statistic_holds: Holds(~Mean(~X),~mean_value(~X,~x_value),mean_context(~X,x,images,channels,height,width),[channels],seen(statistic)),
  centered_holds: Holds(~Centered(~X,~Mean(~X)),~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value)),
    centered_context(~X,~Mean(~X),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width),[images,channels,height,width],seen(centered)),
  left_holds: Holds(~Centered(~X,~Mean(~X)),~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value)),
    centered_context(~X,~Mean(~X),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width),[images,channels,height,width],seen(left)),
  right_holds: Holds(~Centered(~X,~Mean(~X)),~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value)),
    centered_context(~X,~Mean(~X),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width),[images,channels,height,width],seen(right)),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas)),
  mean_holds: Holds(~RM,~rm_value,mean,[channels],seen(means)),variance_holds: Holds(~RV,~rv_value,variance,[channels],seen(variances))) ->
  ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    Norm.squared(Norm.Settings{epsilon,momentum},channels,Norm.count([images,channels,height,width]),seen(statistic),seen(centered),(seen(left),seen(right)),
      (seen(gammas),seen(betas)),(seen(means),seen(variances)))):
  +centered_reading = centered_context(~X,~Mean(~X),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width)
  copied(~Lines(~Square(~X)),
    ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Square(~X)>,~Reduce.ordered_value(~F32,~Square(~X),~square_value(~X,~x_value)),~Sum.step,~Sum.total),
    ~(pair => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
      Norm.deviated(Norm.Settings{epsilon,momentum},channels,Norm.count([images,channels,height,width]),seen(statistic),seen(centered),pair,(seen(gammas),seen(betas)),
        (seen(means),seen(variances))))),
    channel_lines(~Square(~X),square_context(~X,x,images,channels,height,width),images,channels,height,width),[channels],F.sum_axes(F.mul(seen(left),seen(right)),[0,2,3]),
    summed(~Square(~X),~square_value(~X,~x_value),square_context(~X,x,images,channels,height,width),images,channels,height,width,F.mul(seen(left),seen(right)),
      combined_as(~Centered(~X,~Mean(~X)),~Centered(~X,~Mean(~X)),~Square(~X),~F32.mul,~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value)),
        ~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value)),~square_value(~X,~x_value),centered_reading,centered_reading,
        square_context(~X,x,images,channels,height,width),"mul",[images,channels,height,width],[images,channels,height,width],index => {==},seen(left),seen(right),
        left_holds,right_holds,UnitBroadcast.self_broadcast([images,channels,height,width]))),
    squares => squares_kept => squares_holds => squares_kept_holds =>
      deviated_forward(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
        statistic,centered,squares,squares_kept,gammas,betas,means,variances,statistic_holds,centered_holds,squares_holds,squares_kept_holds,gamma_holds,beta_holds,
        mean_holds,variance_holds))

# d is copied once more: two copies are squared, the third is normalized.
def differenced_forward(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,
  +statistic: Owner(),+centered: Owner(),+centered_kept: Owner(),+gammas: Owner(),+betas: Owner(),+means: Owner(),+variances: Owner(),
  statistic_holds: Holds(~Mean(~X),~mean_value(~X,~x_value),mean_context(~X,x,images,channels,height,width),[channels],seen(statistic)),
  centered_holds: Holds(~Centered(~X,~Mean(~X)),~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value)),
    centered_context(~X,~Mean(~X),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width),[images,channels,height,width],seen(centered)),
  kept_holds: Holds(~Centered(~X,~Mean(~X)),~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value)),
    centered_context(~X,~Mean(~X),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width),[images,channels,height,width],seen(centered_kept)),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas)),
  mean_holds: Holds(~RM,~rm_value,mean,[channels],seen(means)),variance_holds: Holds(~RV,~rv_value,variance,[channels],seen(variances))) ->
  ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    Norm.differenced(Norm.Settings{epsilon,momentum},channels,Norm.count([images,channels,height,width]),seen(statistic),(seen(centered),seen(centered_kept)),
      (seen(gammas),seen(betas)),(seen(means),seen(variances)))):
  copied(~Centered(~X,~Mean(~X)),~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value)),
    ~(pair => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
      Norm.squared(Norm.Settings{epsilon,momentum},channels,Norm.count([images,channels,height,width]),seen(statistic),seen(centered_kept),pair,(seen(gammas),seen(betas)),
        (seen(means),seen(variances))))),
    centered_context(~X,~Mean(~X),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width),[images,channels,height,width],seen(centered),
    centered_holds,
    left => right => left_holds => right_holds =>
      squared_forward(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
        statistic,centered_kept,left,right,gammas,betas,means,variances,statistic_holds,kept_holds,left_holds,right_holds,gamma_holds,beta_holds,mean_holds,
        variance_holds))

# d = x - mean, copied.
def centered_forward(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,
  +statistic: Owner(),+statistic_kept: Owner(),+inputs: Owner(),+gammas: Owner(),+betas: Owner(),+means: Owner(),+variances: Owner(),
  statistic_holds: Holds(~Mean(~X),~mean_value(~X,~x_value),mean_context(~X,x,images,channels,height,width),[channels],seen(statistic)),
  kept_holds: Holds(~Mean(~X),~mean_value(~X,~x_value),mean_context(~X,x,images,channels,height,width),[channels],seen(statistic_kept)),
  input_holds: Holds(~X,~x_value,x,[images,channels,height,width],seen(inputs)),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas)),
  mean_holds: Holds(~RM,~rm_value,mean,[channels],seen(means)),variance_holds: Holds(~RV,~rv_value,variance,[channels],seen(variances))) ->
  ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    Norm.centered(Norm.Settings{epsilon,momentum},channels,Norm.count([images,channels,height,width]),(seen(statistic),seen(statistic_kept)),seen(inputs),
      (seen(gammas),seen(betas)),(seen(means),seen(variances)))):
  copied(~Centered(~X,~Mean(~X)),~centered_value(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value)),
    ~(pair => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
      Norm.differenced(Norm.Settings{epsilon,momentum},channels,Norm.count([images,channels,height,width]),seen(statistic_kept),pair,(seen(gammas),seen(betas)),
        (seen(means),seen(variances))))),
    centered_context(~X,~Mean(~X),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width),[images,channels,height,width],
    F.sub(seen(inputs),Norm.spread(seen(statistic),channels)),
    centered_step(~X,~x_value,~Mean(~X),~mean_value(~X,~x_value),x,mean_context(~X,x,images,channels,height,width),images,channels,height,width,inputs,statistic,
      input_holds,statistic_holds),
    centered => centered_kept => centered_holds => centered_kept_holds =>
      differenced_forward(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
        statistic_kept,centered,centered_kept,gammas,betas,means,variances,kept_holds,centered_holds,centered_kept_holds,gamma_holds,beta_holds,mean_holds,
        variance_holds))

# mean = sum_axes(x,[0,2,3]) / M of one copy of x, copied.
def split_forward(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,
  +inputs: Owner(),+inputs_kept: Owner(),+gammas: Owner(),+betas: Owner(),+means: Owner(),+variances: Owner(),
  input_holds: Holds(~X,~x_value,x,[images,channels,height,width],seen(inputs)),kept_holds: Holds(~X,~x_value,x,[images,channels,height,width],seen(inputs_kept)),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas)),
  mean_holds: Holds(~RM,~rm_value,mean,[channels],seen(means)),variance_holds: Holds(~RV,~rv_value,variance,[channels],seen(variances))) ->
  ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    Norm.split(Norm.Settings{epsilon,momentum},channels,Norm.count([images,channels,height,width]),(seen(inputs),seen(inputs_kept)),(seen(gammas),seen(betas)),
      (seen(means),seen(variances)))):
  +count = Norm.count([images,channels,height,width])
  copied(~Mean(~X),~mean_value(~X,~x_value),
    ~(pair => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
      Norm.centered(Norm.Settings{epsilon,momentum},channels,count,pair,seen(inputs_kept),(seen(gammas),seen(betas)),(seen(means),seen(variances))))),
    mean_context(~X,x,images,channels,height,width),[channels],F.div_scalar(F.sum_axes(seen(inputs),[0,2,3]),U32.to_f32(count)),
    restated(~BandMap.Context<F32,Lines(~X)>,~Mean(~X),
      ~BandMap.value(~F32,~F32,~Lines(~X),~Bands.scalar_value(~F32,~F32.div),
        ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<X>,~Reduce.ordered_value(~F32,~X,~x_value),~Sum.step,~Sum.total)),
      ~mean_value(~X,~x_value),mean_context(~X,x,images,channels,height,width),mean_context(~X,x,images,channels,height,width),index => {==},[channels],
      F.div_scalar(F.sum_axes(seen(inputs),[0,2,3]),U32.to_f32(count)),
      scaled(~Lines(~X),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<X>,~Reduce.ordered_value(~F32,~X,~x_value),~Sum.step,~Sum.total),~F32.div,
        channel_lines(~X,x,images,channels,height,width),[channels],U32.to_f32(count),F.sum_axes(seen(inputs),[0,2,3]),
        summed(~X,~x_value,x,images,channels,height,width,seen(inputs),input_holds))),
    statistic => statistic_kept => statistic_holds => statistic_kept_holds =>
      centered_forward(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
        statistic,statistic_kept,inputs_kept,gammas,betas,means,variances,statistic_holds,statistic_kept_holds,kept_holds,gamma_holds,beta_holds,mean_holds,
        variance_holds))

# x, the bands of a model in [N,C,H,W], is copied: one copy gives the mean,
# the other is centered.
def forward_model(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,
  +bands: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,+gammas: Owner(),+betas: Owner(),+means: Owner(),+variances: Owner(),
  +input_shaped: Global.Shaped(~F32,~X,~x_value,x,[images,channels,height,width],
    Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers}),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas)),
  mean_holds: Holds(~RM,~rm_value,mean,[channels],seen(means)),variance_holds: Holds(~RV,~rv_value,variance,[channels],seen(variances))) ->
  ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    Norm.forward(Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},(seen(gammas),seen(betas)),
      (seen(means),seen(variances)),Norm.Settings{epsilon,momentum})):
  copied(~X,~x_value,
    ~(pair => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
      Norm.split(Norm.Settings{epsilon,momentum},channels,Norm.count([images,channels,height,width]),pair,(seen(gammas),seen(betas)),(seen(means),seen(variances))))),
    x,[images,channels,height,width],Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},
    holds(~X,~x_value,x,[images,channels,height,width],Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},
      input_shaped),
    inputs => inputs_kept => input_holds => kept_holds =>
      split_forward(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
        inputs,inputs_kept,gammas,betas,means,variances,input_holds,kept_holds,gamma_holds,beta_holds,mean_holds,variance_holds))

def forward_usable(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,-inputs: Tensors.Tensor<F32>,
  input_shaped: Global.Shaped(~F32,~X,~x_value,x,[images,channels,height,width],inputs),+gammas: Owner(),+betas: Owner(),+means: Owner(),+variances: Owner(),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas)),
  mean_holds: Holds(~RM,~rm_value,mean,[channels],seen(means)),variance_holds: Holds(~RV,~rv_value,variance,[channels],seen(variances))) ->
  ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    Norm.forward(inputs,(seen(gammas),seen(betas)),(seen(means),seen(variances)),Norm.Settings{epsilon,momentum})):
  Shape.Both{usable,contract} = input_shaped
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},inputs,equation)
  %equation : ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    Norm.forward(_,(seen(gammas),seen(betas)),(seen(means),seen(variances)),Norm.Settings{epsilon,momentum}))
  forward_model(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    bands,geometry,fill,workers,gammas,betas,means,variances,
    Shape.Both{Global.Usable{bands,geometry,fill,workers,{==}},%back : Global.Contract(~F32,~X,~x_value,x,_) contract},
    gamma_holds,beta_holds,mean_holds,variance_holds)

# Forward of x [N,C,H,W] and of gamma, beta, the running mean and the running
# variance [C], each a tensor of its reader in any layout. Each tensor
# forward returns holds, once Ready, the values of its reader in
# ForwardHolds: the output y = xhat * gamma + beta, xhat = (x - mean) *
# deviation with mean = sum_axes(x,[0,2,3]) / M and deviation =
# 1 / sqrt(s / M + epsilon) of s = sum_axes(d * d,[0,2,3]), gamma and beta
# as given, and the running statistics moved towards mean and s / (M - 1).
def forward(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,
  inputs: Tensors.Tensor<F32>,gammas: Tensors.Tensor<F32>,betas: Tensors.Tensor<F32>,means: Tensors.Tensor<F32>,variances: Tensors.Tensor<F32>,
  input_shaped: Global.Shaped(~F32,~X,~x_value,x,[images,channels,height,width],inputs),
  +gamma_shaped: Global.Shaped(~F32,~G,~g_value,gamma,[channels],gammas),+beta_shaped: Global.Shaped(~F32,~B,~b_value,beta,[channels],betas),
  +mean_shaped: Global.Shaped(~F32,~RM,~rm_value,mean,[channels],means),+variance_shaped: Global.Shaped(~F32,~RV,~rv_value,variance,[channels],variances)) ->
  ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
    Norm.forward(inputs,(gammas,betas),(means,variances),Norm.Settings{epsilon,momentum})):
  observed(~G,~g_value,
    ~(tensor => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
      Norm.forward(inputs,(tensor,betas),(means,variances),Norm.Settings{epsilon,momentum}))),
    gamma,[channels],gammas,holds(~G,~g_value,gamma,[channels],gammas,gamma_shaped),
    gamma_owner => gamma_holds =>
      observed(~B,~b_value,
        ~(tensor => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,momentum,
          Norm.forward(inputs,(seen(gamma_owner),tensor),(means,variances),Norm.Settings{epsilon,momentum}))),
        beta,[channels],betas,holds(~B,~b_value,beta,[channels],betas,beta_shaped),
        beta_owner => beta_holds =>
          observed(~RM,~rm_value,
            ~(tensor => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
              momentum,Norm.forward(inputs,(seen(gamma_owner),seen(beta_owner)),(tensor,variances),Norm.Settings{epsilon,momentum}))),
            mean,[channels],means,holds(~RM,~rm_value,mean,[channels],means,mean_shaped),
            mean_owner => mean_holds =>
              observed(~RV,~rv_value,
                ~(tensor => ForwardHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
                  momentum,Norm.forward(inputs,(seen(gamma_owner),seen(beta_owner)),(seen(mean_owner),tensor),Norm.Settings{epsilon,momentum}))),
                variance,[channels],variances,holds(~RV,~rv_value,variance,[channels],variances,variance_shaped),
                variance_owner => variance_holds =>
                  forward_usable(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
                    momentum,inputs,input_shaped,gamma_owner,beta_owner,mean_owner,variance_owner,gamma_holds,beta_holds,mean_holds,variance_holds)))))

# ---- Evaluate ----

# What evaluate returns, once each tensor is Ready: the output of the running
# statistics, then gamma, beta and the running statistics as they were given.
def EvaluateHolds(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,result: Norm.Evaluated()) -> Type:
  (y,((gammas,betas),(means,variances))) = result
  Holds(~Affine(~Normalized(~X,~RM,~RV),~G,~B),~affine_value(~Normalized(~X,~RM,~RV),~normalized_value(~X,~x_value,~RM,~rm_value,~RV,~rv_value),~G,~g_value,~B,~b_value),
    affine_context(~Normalized(~X,~RM,~RV),~G,~B,normalized_context(~X,~RM,~RV,x,mean,variance,images,channels,height,width,epsilon),gamma,beta,images,channels,height,width),
    [images,channels,height,width],y) &
  (Holds(~G,~g_value,gamma,[channels],gammas) &
  (Holds(~B,~b_value,beta,[channels],betas) &
  (Holds(~RM,~rm_value,mean,[channels],means) &
  Holds(~RV,~rv_value,variance,[channels],variances))))

# The normalized input of the running statistics, scaled and shifted by the
# first copies of gamma and beta.
def affined_evaluate(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,
  +inputs: Owner(),+gamma_first: Owner(),+gamma_second: Owner(),+beta_first: Owner(),+beta_second: Owner(),+mean_first: Owner(),+mean_second: Owner(),
  +variance_first: Owner(),+variance_second: Owner(),input_holds: Holds(~X,~x_value,x,[images,channels,height,width],seen(inputs)),
  gamma_first_holds: Holds(~G,~g_value,gamma,[channels],seen(gamma_first)),gamma_second_holds: Holds(~G,~g_value,gamma,[channels],seen(gamma_second)),
  beta_first_holds: Holds(~B,~b_value,beta,[channels],seen(beta_first)),beta_second_holds: Holds(~B,~b_value,beta,[channels],seen(beta_second)),
  mean_first_holds: Holds(~RM,~rm_value,mean,[channels],seen(mean_first)),mean_second_holds: Holds(~RM,~rm_value,mean,[channels],seen(mean_second)),
  variance_first_holds: Holds(~RV,~rv_value,variance,[channels],seen(variance_first)),
  variance_second_holds: Holds(~RV,~rv_value,variance,[channels],seen(variance_second))) ->
  EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
    Norm.evaluation(seen(mean_second),seen(variance_second),
      Norm.affine(F.mul(F.sub(seen(inputs),Norm.spread(seen(mean_first),channels)),Norm.spread(Norm.inverse_deviation(seen(variance_first),channels,epsilon),channels)),
        channels,(seen(gamma_first),seen(gamma_second)),(seen(beta_first),seen(beta_second))))):
  +centered_context = centered_context(~X,~RM,x,mean,images,channels,height,width)
  +normalized = normalized_context(~X,~RM,~RV,x,mean,variance,images,channels,height,width,epsilon)
  observed(~Centered(~X,~RM),~centered_value(~X,~x_value,~RM,~rm_value),
    ~(tensor => EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
      Norm.evaluation(seen(mean_second),seen(variance_second),
        Norm.affine(F.mul(tensor,Norm.spread(Norm.inverse_deviation(seen(variance_first),channels,epsilon),channels)),channels,(seen(gamma_first),seen(gamma_second)),
          (seen(beta_first),seen(beta_second)))))),
    centered_context,[images,channels,height,width],F.sub(seen(inputs),Norm.spread(seen(mean_first),channels)),
    centered_step(~X,~x_value,~RM,~rm_value,x,mean,images,channels,height,width,inputs,mean_first,input_holds,mean_first_holds),
    centered => centered_holds =>
      observed(~Inverse(~RV),~inverse_value(~RV,~rv_value),
        ~(tensor => EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
          Norm.evaluation(seen(mean_second),seen(variance_second),
            Norm.affine(F.mul(seen(centered),Norm.spread(tensor,channels)),channels,(seen(gamma_first),seen(gamma_second)),(seen(beta_first),seen(beta_second)))))),
        inverse_context(~RV,variance,channels,epsilon),[channels],Norm.inverse_deviation(seen(variance_first),channels,epsilon),
        inverse(~RV,~rv_value,variance,channels,epsilon,variance_first,variance_first_holds),
        deviation => deviation_holds =>
          observed(~Normalized(~X,~RM,~RV),~normalized_value(~X,~x_value,~RM,~rm_value,~RV,~rv_value),
            ~(tensor => EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
              Norm.evaluation(seen(mean_second),seen(variance_second),
                Norm.affine(tensor,channels,(seen(gamma_first),seen(gamma_second)),(seen(beta_first),seen(beta_second)))))),
            normalized,[images,channels,height,width],F.mul(seen(centered),Norm.spread(seen(deviation),channels)),
            normalized_step(~X,~x_value,~RM,~rm_value,~RV,~rv_value,x,mean,variance,images,channels,height,width,epsilon,centered,deviation,centered_holds,deviation_holds),
            xhat => xhat_holds =>
              (affine_step(~Normalized(~X,~RM,~RV),~normalized_value(~X,~x_value,~RM,~rm_value,~RV,~rv_value),~G,~g_value,~B,~b_value,normalized,gamma,beta,
                  images,channels,height,width,xhat,gamma_first,beta_first,xhat_holds,gamma_first_holds,beta_first_holds),
                (gamma_second_holds,(beta_second_holds,(mean_second_holds,variance_second_holds)))))))

# The running statistics, gamma and beta are copied: the first copies make
# the output, the second come back.
def evaluated_holds(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+inputs: Owner(),+gammas: Owner(),+betas: Owner(),+means: Owner(),+variances: Owner(),
  input_holds: Holds(~X,~x_value,x,[images,channels,height,width],seen(inputs)),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas)),
  mean_holds: Holds(~RM,~rm_value,mean,[channels],seen(means)),variance_holds: Holds(~RV,~rv_value,variance,[channels],seen(variances))) ->
  EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
    Norm.evaluated(channels,epsilon,seen(inputs),(seen(gammas),seen(betas)),F.copy(seen(means)),F.copy(seen(variances)))):
  copied(~RM,~rm_value,
    ~(pair => EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
      Norm.evaluated(channels,epsilon,seen(inputs),(seen(gammas),seen(betas)),pair,F.copy(seen(variances))))),
    mean,[channels],seen(means),mean_holds,
    mean_first => mean_second => mean_first_holds => mean_second_holds =>
      copied(~RV,~rv_value,
        ~(pair => EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
          Norm.evaluated(channels,epsilon,seen(inputs),(seen(gammas),seen(betas)),(seen(mean_first),seen(mean_second)),pair))),
        variance,[channels],seen(variances),variance_holds,
        variance_first => variance_second => variance_first_holds => variance_second_holds =>
          copied(~G,~g_value,
            ~(pair => EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
              Norm.evaluation(seen(mean_second),seen(variance_second),
                Norm.affine(F.mul(F.sub(seen(inputs),Norm.spread(seen(mean_first),channels)),
                  Norm.spread(Norm.inverse_deviation(seen(variance_first),channels,epsilon),channels)),channels,pair,F.copy(seen(betas)))))),
            gamma,[channels],seen(gammas),gamma_holds,
            gamma_first => gamma_second => gamma_first_holds => gamma_second_holds =>
              copied(~B,~b_value,
                ~(pair => EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
                  Norm.evaluation(seen(mean_second),seen(variance_second),
                    Norm.affine(F.mul(F.sub(seen(inputs),Norm.spread(seen(mean_first),channels)),
                      Norm.spread(Norm.inverse_deviation(seen(variance_first),channels,epsilon),channels)),channels,(seen(gamma_first),seen(gamma_second)),pair)))),
                beta,[channels],seen(betas),beta_holds,
                beta_first => beta_second => beta_first_holds => beta_second_holds =>
                  affined_evaluate(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
                    inputs,gamma_first,gamma_second,beta_first,beta_second,mean_first,mean_second,variance_first,variance_second,input_holds,gamma_first_holds,
                    gamma_second_holds,beta_first_holds,beta_second_holds,mean_first_holds,mean_second_holds,variance_first_holds,variance_second_holds)))))

def evaluate_usable(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,-inputs: Tensors.Tensor<F32>,
  input_shaped: Global.Shaped(~F32,~X,~x_value,x,[images,channels,height,width],inputs),+gammas: Owner(),+betas: Owner(),+means: Owner(),+variances: Owner(),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),beta_holds: Holds(~B,~b_value,beta,[channels],seen(betas)),
  mean_holds: Holds(~RM,~rm_value,mean,[channels],seen(means)),variance_holds: Holds(~RV,~rv_value,variance,[channels],seen(variances))) ->
  EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
    Norm.evaluate(inputs,(seen(gammas),seen(betas)),(seen(means),seen(variances)),Norm.Settings{epsilon,momentum})):
  Shape.Both{usable,contract} = input_shaped
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},inputs,equation)
  +owner = {Observation.TensorOwner{bands,geometry,[images,channels,height,width],fill,Tensors.Usable{},workers} : Owner()}
  %equation : EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
    Norm.evaluate(_,(seen(gammas),seen(betas)),(seen(means),seen(variances)),Norm.Settings{epsilon,momentum}))
  evaluated_holds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
    owner,gammas,betas,means,variances,
    holds(~X,~x_value,x,[images,channels,height,width],seen(owner),
      Shape.Both{Global.Usable{bands,geometry,fill,workers,{==}},%back : Global.Contract(~F32,~X,~x_value,x,_) contract}),
    gamma_holds,beta_holds,mean_holds,variance_holds)

# Evaluate of x [N,C,H,W] and of gamma, beta, the running mean and the
# running variance [C], each a tensor of its reader in any layout. Each
# tensor evaluate returns holds, once Ready, the values of its reader in
# EvaluateHolds: the output (x - running mean) * (1 / sqrt(running variance +
# epsilon)) * gamma + beta, then the other tensors as they were given.
def evaluate(~X: Data,~x_value: X -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~B: Data,~b_value: B -> U32 -> F32,
  ~RM: Data,~rm_value: RM -> U32 -> F32,~RV: Data,~rv_value: RV -> U32 -> F32,+x: X,+gamma: G,+beta: B,+mean: RM,+variance: RV,
  +images: U32,+channels: U32,+height: U32,+width: U32,+epsilon: F32,+momentum: F32,
  inputs: Tensors.Tensor<F32>,gammas: Tensors.Tensor<F32>,betas: Tensors.Tensor<F32>,means: Tensors.Tensor<F32>,variances: Tensors.Tensor<F32>,
  input_shaped: Global.Shaped(~F32,~X,~x_value,x,[images,channels,height,width],inputs),
  +gamma_shaped: Global.Shaped(~F32,~G,~g_value,gamma,[channels],gammas),+beta_shaped: Global.Shaped(~F32,~B,~b_value,beta,[channels],betas),
  +mean_shaped: Global.Shaped(~F32,~RM,~rm_value,mean,[channels],means),+variance_shaped: Global.Shaped(~F32,~RV,~rv_value,variance,[channels],variances)) ->
  EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
    Norm.evaluate(inputs,(gammas,betas),(means,variances),Norm.Settings{epsilon,momentum})):
  observed(~G,~g_value,
    ~(tensor => EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
      Norm.evaluate(inputs,(tensor,betas),(means,variances),Norm.Settings{epsilon,momentum}))),
    gamma,[channels],gammas,holds(~G,~g_value,gamma,[channels],gammas,gamma_shaped),
    gamma_owner => gamma_holds =>
      observed(~B,~b_value,
        ~(tensor => EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
          Norm.evaluate(inputs,(seen(gamma_owner),tensor),(means,variances),Norm.Settings{epsilon,momentum}))),
        beta,[channels],betas,holds(~B,~b_value,beta,[channels],betas,beta_shaped),
        beta_owner => beta_holds =>
          observed(~RM,~rm_value,
            ~(tensor => EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
              Norm.evaluate(inputs,(seen(gamma_owner),seen(beta_owner)),(tensor,variances),Norm.Settings{epsilon,momentum}))),
            mean,[channels],means,holds(~RM,~rm_value,mean,[channels],means,mean_shaped),
            mean_owner => mean_holds =>
              observed(~RV,~rv_value,
                ~(tensor => EvaluateHolds(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,
                  epsilon,Norm.evaluate(inputs,(seen(gamma_owner),seen(beta_owner)),(seen(mean_owner),tensor),Norm.Settings{epsilon,momentum}))),
                variance,[channels],variances,holds(~RV,~rv_value,variance,[channels],variances,variance_shaped),
                variance_owner => variance_holds =>
                  evaluate_usable(~X,~x_value,~G,~g_value,~B,~b_value,~RM,~rm_value,~RV,~rv_value,x,gamma,beta,mean,variance,images,channels,height,width,epsilon,
                    momentum,inputs,input_shaped,gamma_owner,beta_owner,mean_owner,variance_owner,gamma_holds,beta_holds,mean_holds,variance_holds)))))

# ---- Backward ----

# dy * xhat.
def Product(~D: Data,~XH: Data) -> Data:
  Combined(~D,~XH)

def product_value(~D: Data,~dy_value: D -> U32 -> F32,~XH: Data,~xhat_value: XH -> U32 -> F32,context: Product(~D,~XH),index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<D>,~Broadcast.Expanded<XH>,~F32.mul,~Broadcast.value(~F32,~D,~dy_value),~Broadcast.value(~F32,~XH,~xhat_value),context,index)

def product_context(~D: Data,~XH: Data,+dy: D,+xhat: XH,+images: U32,+channels: U32,+height: U32,+width: U32) -> Product(~D,~XH):
  combined_context(~D,~XH,dy,xhat,[images,channels,height,width],[images,channels,height,width])

# gamma * deviation, divided by M.
def Weight(~G: Data,~DV: Data) -> Data:
  Combined(~G,~DV)

def weight_value(~G: Data,~g_value: G -> U32 -> F32,~DV: Data,~deviation_value: DV -> U32 -> F32,context: Weight(~G,~DV),index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<G>,~Broadcast.Expanded<DV>,~F32.mul,~Broadcast.value(~F32,~G,~g_value),~Broadcast.value(~F32,~DV,~deviation_value),context,index)

def weight_context(~G: Data,~DV: Data,+gamma: G,+deviation: DV,+channels: U32) -> Weight(~G,~DV):
  combined_context(~G,~DV,gamma,deviation,[channels],[channels])

def Scale(~G: Data,~DV: Data) -> Data:
  BandMap.Context<F32,Weight(~G,~DV)>

def scale_value(~G: Data,~g_value: G -> U32 -> F32,~DV: Data,~deviation_value: DV -> U32 -> F32,context: Scale(~G,~DV),index: U32) -> F32:
  BandMap.value(~F32,~F32,~Weight(~G,~DV),~Bands.scalar_value(~F32,~F32.div),~weight_value(~G,~g_value,~DV,~deviation_value),context,index)

def scale_context(~G: Data,~DV: Data,+gamma: G,+deviation: DV,+channels: U32,+count: U32) -> Scale(~G,~DV):
  BandMap.Context{U32.to_f32(count),weight_context(~G,~DV,gamma,deviation,channels)}

# dy * M - dbeta, dbeta read at the channel of each index.
def Shifted(~D: Data) -> Data:
  Combined(~BandMap.Context<F32,D>,~Lines(~D))

def shifted_value(~D: Data,~dy_value: D -> U32 -> F32,context: Shifted(~D),index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<BandMap.Context<F32,D>>,~Broadcast.Expanded<Lines(~D)>,~F32.sub,
    ~Broadcast.value(~F32,~BandMap.Context<F32,D>,~BandMap.value(~F32,~F32,~D,~Bands.scalar_value(~F32,~F32.mul),~dy_value)),
    ~Broadcast.value(~F32,~Lines(~D),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<D>,~Reduce.ordered_value(~F32,~D,~dy_value),~Sum.step,~Sum.total)),context,index)

def shifted_context(~D: Data,+dy: D,+images: U32,+channels: U32,+height: U32,+width: U32,+count: U32) -> Shifted(~D):
  combined_context(~BandMap.Context<F32,D>,~Lines(~D),BandMap.Context{U32.to_f32(count),dy},channel_lines(~D,dy,images,channels,height,width),
    [images,channels,height,width],[1,channels,1,1])

# xhat * dgamma, dgamma read at the channel of each index.
def Projected(~D: Data,~XH: Data) -> Data:
  Combined(~XH,~Lines(~Product(~D,~XH)))

def projected_value(~D: Data,~dy_value: D -> U32 -> F32,~XH: Data,~xhat_value: XH -> U32 -> F32,context: Projected(~D,~XH),index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<XH>,~Broadcast.Expanded<Lines(~Product(~D,~XH))>,~F32.mul,~Broadcast.value(~F32,~XH,~xhat_value),
    ~Broadcast.value(~F32,~Lines(~Product(~D,~XH)),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Product(~D,~XH)>,
      ~Reduce.ordered_value(~F32,~Product(~D,~XH),~product_value(~D,~dy_value,~XH,~xhat_value)),~Sum.step,~Sum.total)),context,index)

def projected_context(~D: Data,~XH: Data,+dy: D,+xhat: XH,+images: U32,+channels: U32,+height: U32,+width: U32) -> Projected(~D,~XH):
  combined_context(~XH,~Lines(~Product(~D,~XH)),xhat,channel_lines(~Product(~D,~XH),product_context(~D,~XH,dy,xhat,images,channels,height,width),images,channels,height,width),
    [images,channels,height,width],[1,channels,1,1])

# dx = ((dy * M - dbeta) - xhat * dgamma) * ((gamma * deviation) / M).
def Inner(~D: Data,~XH: Data) -> Data:
  Combined(~Shifted(~D),~Projected(~D,~XH))

def inner_value(~D: Data,~dy_value: D -> U32 -> F32,~XH: Data,~xhat_value: XH -> U32 -> F32,context: Inner(~D,~XH),index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<Shifted(~D)>,~Broadcast.Expanded<Projected(~D,~XH)>,~F32.sub,~Broadcast.value(~F32,~Shifted(~D),~shifted_value(~D,~dy_value)),
    ~Broadcast.value(~F32,~Projected(~D,~XH),~projected_value(~D,~dy_value,~XH,~xhat_value)),context,index)

def inner_context(~D: Data,~XH: Data,+dy: D,+xhat: XH,+images: U32,+channels: U32,+height: U32,+width: U32,+count: U32) -> Inner(~D,~XH):
  combined_context(~Shifted(~D),~Projected(~D,~XH),shifted_context(~D,dy,images,channels,height,width,count),projected_context(~D,~XH,dy,xhat,images,channels,height,width),
    [images,channels,height,width],[images,channels,height,width])

def InputGradient(~D: Data,~XH: Data,~G: Data,~DV: Data) -> Data:
  Combined(~Inner(~D,~XH),~Scale(~G,~DV))

def input_gradient_value(~D: Data,~dy_value: D -> U32 -> F32,~XH: Data,~xhat_value: XH -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~DV: Data,
  ~deviation_value: DV -> U32 -> F32,context: InputGradient(~D,~XH,~G,~DV),index: U32) -> F32:
  Leaf.value(~F32,~Broadcast.Expanded<Inner(~D,~XH)>,~Broadcast.Expanded<Scale(~G,~DV)>,~F32.mul,~Broadcast.value(~F32,~Inner(~D,~XH),~inner_value(~D,~dy_value,~XH,~xhat_value)),
    ~Broadcast.value(~F32,~Scale(~G,~DV),~scale_value(~G,~g_value,~DV,~deviation_value)),context,index)

def input_gradient_context(~D: Data,~XH: Data,~G: Data,~DV: Data,+dy: D,+xhat: XH,+gamma: G,+deviation: DV,+images: U32,+channels: U32,+height: U32,+width: U32,
  +count: U32) -> InputGradient(~D,~XH,~G,~DV):
  combined_context(~Inner(~D,~XH),~Scale(~G,~DV),inner_context(~D,~XH,dy,xhat,images,channels,height,width,count),scale_context(~G,~DV,gamma,deviation,channels,count),
    [images,channels,height,width],[1,channels,1,1])

# What backward returns, once each tensor is Ready: dx, dgamma = sum_axes(dy *
# xhat,[0,2,3]), dbeta = sum_axes(dy,[0,2,3]) and gamma as it was given.
def BackwardHolds(~D: Data,~dy_value: D -> U32 -> F32,~XH: Data,~xhat_value: XH -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~DV: Data,
  ~deviation_value: DV -> U32 -> F32,+dy: D,+xhat: XH,+gamma: G,+deviation: DV,+images: U32,+channels: U32,+height: U32,+width: U32,result: Norm.Backward()) -> Type:
  (dx,(dgamma,(dbeta,gammas))) = result
  Holds(~InputGradient(~D,~XH,~G,~DV),~input_gradient_value(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value),
    input_gradient_context(~D,~XH,~G,~DV,dy,xhat,gamma,deviation,images,channels,height,width,Norm.count([images,channels,height,width])),[images,channels,height,width],dx) &
  (Holds(~Lines(~Product(~D,~XH)),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Product(~D,~XH)>,
      ~Reduce.ordered_value(~F32,~Product(~D,~XH),~product_value(~D,~dy_value,~XH,~xhat_value)),~Sum.step,~Sum.total),
    channel_lines(~Product(~D,~XH),product_context(~D,~XH,dy,xhat,images,channels,height,width),images,channels,height,width),[channels],dgamma) &
  (Holds(~Lines(~D),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<D>,~Reduce.ordered_value(~F32,~D,~dy_value),~Sum.step,~Sum.total),
    channel_lines(~D,dy,images,channels,height,width),[channels],dbeta) &
  Holds(~G,~g_value,gamma,[channels],gammas)))

# dx from the second copies of dbeta and dgamma; the first copies come back
# with gamma.
def propagated_backward(~D: Data,~dy_value: D -> U32 -> F32,~XH: Data,~xhat_value: XH -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~DV: Data,
  ~deviation_value: DV -> U32 -> F32,+dy: D,+xhat: XH,+gamma: G,+deviation: DV,+images: U32,+channels: U32,+height: U32,+width: U32,
  +gradients: Owner(),+normalized: Owner(),+scale: Owner(),+dbeta: Owner(),+dbeta_kept: Owner(),+dgamma: Owner(),+dgamma_kept: Owner(),+gammas: Owner(),
  gradient_holds: Holds(~D,~dy_value,dy,[images,channels,height,width],seen(gradients)),
  normalized_holds: Holds(~XH,~xhat_value,xhat,[images,channels,height,width],seen(normalized)),
  scale_holds: Holds(~Weight(~G,~DV),~weight_value(~G,~g_value,~DV,~deviation_value),weight_context(~G,~DV,gamma,deviation,channels),[channels],seen(scale)),
  dbeta_holds: Holds(~Lines(~D),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<D>,~Reduce.ordered_value(~F32,~D,~dy_value),~Sum.step,~Sum.total),
    channel_lines(~D,dy,images,channels,height,width),[channels],seen(dbeta)),
  dbeta_kept_holds: Holds(~Lines(~D),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<D>,~Reduce.ordered_value(~F32,~D,~dy_value),~Sum.step,~Sum.total),
    channel_lines(~D,dy,images,channels,height,width),[channels],seen(dbeta_kept)),
  dgamma_holds: Holds(~Lines(~Product(~D,~XH)),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Product(~D,~XH)>,
      ~Reduce.ordered_value(~F32,~Product(~D,~XH),~product_value(~D,~dy_value,~XH,~xhat_value)),~Sum.step,~Sum.total),
    channel_lines(~Product(~D,~XH),product_context(~D,~XH,dy,xhat,images,channels,height,width),images,channels,height,width),[channels],seen(dgamma)),
  dgamma_kept_holds: Holds(~Lines(~Product(~D,~XH)),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Product(~D,~XH)>,
      ~Reduce.ordered_value(~F32,~Product(~D,~XH),~product_value(~D,~dy_value,~XH,~xhat_value)),~Sum.step,~Sum.total),
    channel_lines(~Product(~D,~XH),product_context(~D,~XH,dy,xhat,images,channels,height,width),images,channels,height,width),[channels],seen(dgamma_kept)),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas))) ->
  BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
    Norm.propagated(channels,Norm.count([images,channels,height,width]),seen(gradients),seen(normalized),seen(scale),(seen(dbeta),seen(dbeta_kept)),
      (seen(dgamma),seen(dgamma_kept)),seen(gammas))):
  +count = Norm.count([images,channels,height,width])
  +total = U32.to_f32(count)
  +dbeta_lines = channel_lines(~D,dy,images,channels,height,width)
  +dgamma_lines = channel_lines(~Product(~D,~XH),product_context(~D,~XH,dy,xhat,images,channels,height,width),images,channels,height,width)
  +shifted = shifted_context(~D,dy,images,channels,height,width,count)
  +projected = projected_context(~D,~XH,dy,xhat,images,channels,height,width)
  (combined_as(~Inner(~D,~XH),~Scale(~G,~DV),~InputGradient(~D,~XH,~G,~DV),~F32.mul,~inner_value(~D,~dy_value,~XH,~xhat_value),
      ~scale_value(~G,~g_value,~DV,~deviation_value),~input_gradient_value(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value),
      inner_context(~D,~XH,dy,xhat,images,channels,height,width,count),scale_context(~G,~DV,gamma,deviation,channels,count),
      input_gradient_context(~D,~XH,~G,~DV,dy,xhat,gamma,deviation,images,channels,height,width,count),"mul",[images,channels,height,width],[1,channels,1,1],index => {==},
      F.sub(F.sub(F.mul_scalar(seen(gradients),total),Norm.spread(seen(dbeta),channels)),F.mul(seen(normalized),Norm.spread(seen(dgamma),channels))),
      Norm.spread(F.div_scalar(seen(scale),total),channels),
      combined_as(~Shifted(~D),~Projected(~D,~XH),~Inner(~D,~XH),~F32.sub,~shifted_value(~D,~dy_value),~projected_value(~D,~dy_value,~XH,~xhat_value),
        ~inner_value(~D,~dy_value,~XH,~xhat_value),shifted,projected,inner_context(~D,~XH,dy,xhat,images,channels,height,width,count),"sub",[images,channels,height,width],
        [images,channels,height,width],index => {==},F.sub(F.mul_scalar(seen(gradients),total),Norm.spread(seen(dbeta),channels)),
        F.mul(seen(normalized),Norm.spread(seen(dgamma),channels)),
        combined_as(~BandMap.Context<F32,D>,~Lines(~D),~Shifted(~D),~F32.sub,~BandMap.value(~F32,~F32,~D,~Bands.scalar_value(~F32,~F32.mul),~dy_value),
          ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<D>,~Reduce.ordered_value(~F32,~D,~dy_value),~Sum.step,~Sum.total),~shifted_value(~D,~dy_value),
          BandMap.Context{total,dy},dbeta_lines,shifted,"sub",[images,channels,height,width],[1,channels,1,1],index => {==},
          F.mul_scalar(seen(gradients),total),Norm.spread(seen(dbeta),channels),
          scaled(~D,~dy_value,~F32.mul,dy,[images,channels,height,width],total,seen(gradients),gradient_holds),
          reshaped(~Lines(~D),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<D>,~Reduce.ordered_value(~F32,~D,~dy_value),~Sum.step,~Sum.total),dbeta_lines,[channels],
            [1,channels,1,1],seen(dbeta),dbeta_holds),
          spread_broadcast(images,channels,height,width)),
        combined_as(~XH,~Lines(~Product(~D,~XH)),~Projected(~D,~XH),~F32.mul,~xhat_value,
          ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Product(~D,~XH)>,~Reduce.ordered_value(~F32,~Product(~D,~XH),~product_value(~D,~dy_value,~XH,~xhat_value)),
            ~Sum.step,~Sum.total),
          ~projected_value(~D,~dy_value,~XH,~xhat_value),xhat,dgamma_lines,projected,"mul",[images,channels,height,width],[1,channels,1,1],index => {==},
          seen(normalized),Norm.spread(seen(dgamma),channels),normalized_holds,
          reshaped(~Lines(~Product(~D,~XH)),
            ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Product(~D,~XH)>,~Reduce.ordered_value(~F32,~Product(~D,~XH),~product_value(~D,~dy_value,~XH,~xhat_value)),
              ~Sum.step,~Sum.total),
            dgamma_lines,[channels],[1,channels,1,1],seen(dgamma),dgamma_holds),
          spread_broadcast(images,channels,height,width)),
        UnitBroadcast.self_broadcast([images,channels,height,width])),
      reshaped(~Scale(~G,~DV),~scale_value(~G,~g_value,~DV,~deviation_value),scale_context(~G,~DV,gamma,deviation,channels,count),[channels],[1,channels,1,1],
        F.div_scalar(seen(scale),total),
        restated(~BandMap.Context<F32,Weight(~G,~DV)>,~Scale(~G,~DV),
          ~BandMap.value(~F32,~F32,~Weight(~G,~DV),~Bands.scalar_value(~F32,~F32.div),~weight_value(~G,~g_value,~DV,~deviation_value)),
          ~scale_value(~G,~g_value,~DV,~deviation_value),scale_context(~G,~DV,gamma,deviation,channels,count),scale_context(~G,~DV,gamma,deviation,channels,count),
          index => {==},[channels],F.div_scalar(seen(scale),total),
          scaled(~Weight(~G,~DV),~weight_value(~G,~g_value,~DV,~deviation_value),~F32.div,weight_context(~G,~DV,gamma,deviation,channels),[channels],total,seen(scale),
            scale_holds))),
      spread_broadcast(images,channels,height,width)),
    (dgamma_kept_holds,(dbeta_kept_holds,gamma_holds)))

# dbeta = sum_axes(dy,[0,2,3]) and dgamma = sum_axes(dy * xhat,[0,2,3]) of
# the two copies of dy, each copied.
def sums_backward(~D: Data,~dy_value: D -> U32 -> F32,~XH: Data,~xhat_value: XH -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~DV: Data,
  ~deviation_value: DV -> U32 -> F32,+dy: D,+xhat: XH,+gamma: G,+deviation: DV,+images: U32,+channels: U32,+height: U32,+width: U32,
  +for_beta: Owner(),+for_gamma: Owner(),+gradients: Owner(),+normalized: Owner(),+normalized_kept: Owner(),+scale: Owner(),+gammas: Owner(),
  beta_holds: Holds(~D,~dy_value,dy,[images,channels,height,width],seen(for_beta)),gamma_input_holds: Holds(~D,~dy_value,dy,[images,channels,height,width],seen(for_gamma)),
  gradient_holds: Holds(~D,~dy_value,dy,[images,channels,height,width],seen(gradients)),
  normalized_holds: Holds(~XH,~xhat_value,xhat,[images,channels,height,width],seen(normalized)),
  kept_holds: Holds(~XH,~xhat_value,xhat,[images,channels,height,width],seen(normalized_kept)),
  scale_holds: Holds(~Weight(~G,~DV),~weight_value(~G,~g_value,~DV,~deviation_value),weight_context(~G,~DV,gamma,deviation,channels),[channels],seen(scale)),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas))) ->
  BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
    Norm.sums(channels,Norm.count([images,channels,height,width]),(seen(for_beta),seen(for_gamma)),seen(gradients),seen(normalized),seen(normalized_kept),seen(scale),
      seen(gammas))):
  +count = Norm.count([images,channels,height,width])
  +product = product_context(~D,~XH,dy,xhat,images,channels,height,width)
  copied(~Lines(~D),~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<D>,~Reduce.ordered_value(~F32,~D,~dy_value),~Sum.step,~Sum.total),
    ~(pair => BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
      Norm.propagated(channels,count,seen(gradients),seen(normalized_kept),seen(scale),pair,F.copy(F.sum_axes(F.mul(seen(for_gamma),seen(normalized)),[0,2,3])),
        seen(gammas)))),
    channel_lines(~D,dy,images,channels,height,width),[channels],F.sum_axes(seen(for_beta),[0,2,3]),
    summed(~D,~dy_value,dy,images,channels,height,width,seen(for_beta),beta_holds),
    dbeta => dbeta_kept => dbeta_holds => dbeta_kept_holds =>
      copied(~Lines(~Product(~D,~XH)),
        ~Reduce.value(~F32,~Sum.Lanes,~Reduce.Ordered<Product(~D,~XH)>,~Reduce.ordered_value(~F32,~Product(~D,~XH),~product_value(~D,~dy_value,~XH,~xhat_value)),
          ~Sum.step,~Sum.total),
        ~(pair => BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
          Norm.propagated(channels,count,seen(gradients),seen(normalized_kept),seen(scale),(seen(dbeta),seen(dbeta_kept)),pair,seen(gammas)))),
        channel_lines(~Product(~D,~XH),product,images,channels,height,width),[channels],F.sum_axes(F.mul(seen(for_gamma),seen(normalized)),[0,2,3]),
        summed(~Product(~D,~XH),~product_value(~D,~dy_value,~XH,~xhat_value),product,images,channels,height,width,F.mul(seen(for_gamma),seen(normalized)),
          combined_as(~D,~XH,~Product(~D,~XH),~F32.mul,~dy_value,~xhat_value,~product_value(~D,~dy_value,~XH,~xhat_value),dy,xhat,product,"mul",
            [images,channels,height,width],[images,channels,height,width],index => {==},seen(for_gamma),seen(normalized),gamma_input_holds,normalized_holds,
            UnitBroadcast.self_broadcast([images,channels,height,width]))),
        dgamma => dgamma_kept => dgamma_holds => dgamma_kept_holds =>
          propagated_backward(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
            gradients,normalized_kept,scale,dbeta,dbeta_kept,dgamma,dgamma_kept,gammas,gradient_holds,kept_holds,scale_holds,dbeta_holds,dbeta_kept_holds,dgamma_holds,
            dgamma_kept_holds,gamma_holds)))

# The first copy of dy is copied once more, and gamma * deviation is formed
# from the first copy of gamma.
def copied_backward(~D: Data,~dy_value: D -> U32 -> F32,~XH: Data,~xhat_value: XH -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~DV: Data,
  ~deviation_value: DV -> U32 -> F32,+dy: D,+xhat: XH,+gamma: G,+deviation: DV,+images: U32,+channels: U32,+height: U32,+width: U32,
  +gradients: Owner(),+gradients_kept: Owner(),+normalized: Owner(),+normalized_kept: Owner(),+deviations: Owner(),+gammas: Owner(),+gammas_kept: Owner(),
  gradient_holds: Holds(~D,~dy_value,dy,[images,channels,height,width],seen(gradients)),
  gradient_kept_holds: Holds(~D,~dy_value,dy,[images,channels,height,width],seen(gradients_kept)),
  normalized_holds: Holds(~XH,~xhat_value,xhat,[images,channels,height,width],seen(normalized)),
  normalized_kept_holds: Holds(~XH,~xhat_value,xhat,[images,channels,height,width],seen(normalized_kept)),
  deviation_holds: Holds(~DV,~deviation_value,deviation,[channels],seen(deviations)),
  gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas)),gamma_kept_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas_kept))) ->
  BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
    Norm.copied(channels,Norm.count([images,channels,height,width]),(seen(gradients),seen(gradients_kept)),(seen(normalized),seen(normalized_kept)),seen(deviations),
      (seen(gammas),seen(gammas_kept)))):
  +count = Norm.count([images,channels,height,width])
  copied(~D,~dy_value,
    ~(pair => BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
      Norm.sums(channels,count,pair,seen(gradients_kept),seen(normalized),seen(normalized_kept),F.mul(seen(gammas),seen(deviations)),seen(gammas_kept)))),
    dy,[images,channels,height,width],seen(gradients),gradient_holds,
    for_beta => for_gamma => beta_holds => gamma_input_holds =>
      observed(~Weight(~G,~DV),~weight_value(~G,~g_value,~DV,~deviation_value),
        ~(tensor => BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
          Norm.sums(channels,count,(seen(for_beta),seen(for_gamma)),seen(gradients_kept),seen(normalized),seen(normalized_kept),tensor,seen(gammas_kept)))),
        weight_context(~G,~DV,gamma,deviation,channels),[channels],F.mul(seen(gammas),seen(deviations)),
        combined_as(~G,~DV,~Weight(~G,~DV),~F32.mul,~g_value,~deviation_value,~weight_value(~G,~g_value,~DV,~deviation_value),gamma,deviation,
          weight_context(~G,~DV,gamma,deviation,channels),"mul",[channels],[channels],index => {==},seen(gammas),seen(deviations),gamma_holds,deviation_holds,
          UnitBroadcast.self_broadcast([channels])),
        scale => scale_holds =>
          sums_backward(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
            for_beta,for_gamma,gradients_kept,normalized,normalized_kept,scale,gammas_kept,beta_holds,gamma_input_holds,gradient_kept_holds,normalized_holds,
            normalized_kept_holds,scale_holds,gamma_kept_holds)))

# dy, the bands of a model in [N,C,H,W], xhat and gamma are copied.
def backward_model(~D: Data,~dy_value: D -> U32 -> F32,~XH: Data,~xhat_value: XH -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~DV: Data,
  ~deviation_value: DV -> U32 -> F32,+dy: D,+xhat: XH,+gamma: G,+deviation: DV,+images: U32,+channels: U32,+height: U32,+width: U32,
  +bands: Model.Model<F32>,+geometry: Bands.Geometry,+fill: F32,+workers: U32,+normalized: Owner(),+deviations: Owner(),+gammas: Owner(),
  +gradient_shaped: Global.Shaped(~F32,~D,~dy_value,dy,[images,channels,height,width],
    Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers}),
  normalized_holds: Holds(~XH,~xhat_value,xhat,[images,channels,height,width],seen(normalized)),
  deviation_holds: Holds(~DV,~deviation_value,deviation,[channels],seen(deviations)),gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas))) ->
  BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
    Norm.backward(Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},(seen(normalized),seen(deviations)),
      seen(gammas))):
  +count = Norm.count([images,channels,height,width])
  copied(~D,~dy_value,
    ~(pair => BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
      Norm.copied(channels,count,pair,F.copy(seen(normalized)),seen(deviations),F.copy(seen(gammas))))),
    dy,[images,channels,height,width],Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},
    holds(~D,~dy_value,dy,[images,channels,height,width],Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},
      gradient_shaped),
    gradients => gradients_kept => gradient_holds => gradient_kept_holds =>
      copied(~XH,~xhat_value,
        ~(pair => BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
          Norm.copied(channels,count,(seen(gradients),seen(gradients_kept)),pair,seen(deviations),F.copy(seen(gammas))))),
        xhat,[images,channels,height,width],seen(normalized),normalized_holds,
        normalized_first => normalized_second => normalized_first_holds => normalized_second_holds =>
          copied(~G,~g_value,
            ~(pair => BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
              Norm.copied(channels,count,(seen(gradients),seen(gradients_kept)),(seen(normalized_first),seen(normalized_second)),seen(deviations),pair))),
            gamma,[channels],seen(gammas),gamma_holds,
            gamma_first => gamma_second => gamma_first_holds => gamma_second_holds =>
              copied_backward(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
                gradients,gradients_kept,normalized_first,normalized_second,deviations,gamma_first,gamma_second,gradient_holds,gradient_kept_holds,
                normalized_first_holds,normalized_second_holds,deviation_holds,gamma_first_holds,gamma_second_holds))))

def backward_usable(~D: Data,~dy_value: D -> U32 -> F32,~XH: Data,~xhat_value: XH -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~DV: Data,
  ~deviation_value: DV -> U32 -> F32,+dy: D,+xhat: XH,+gamma: G,+deviation: DV,+images: U32,+channels: U32,+height: U32,+width: U32,-gradients: Tensors.Tensor<F32>,
  gradient_shaped: Global.Shaped(~F32,~D,~dy_value,dy,[images,channels,height,width],gradients),+normalized: Owner(),+deviations: Owner(),+gammas: Owner(),
  normalized_holds: Holds(~XH,~xhat_value,xhat,[images,channels,height,width],seen(normalized)),
  deviation_holds: Holds(~DV,~deviation_value,deviation,[channels],seen(deviations)),gamma_holds: Holds(~G,~g_value,gamma,[channels],seen(gammas))) ->
  BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
    Norm.backward(gradients,(seen(normalized),seen(deviations)),seen(gammas))):
  Shape.Both{usable,contract} = gradient_shaped
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  +back = Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,[images,channels,height,width],fill,Tensors.Usable{},workers},gradients,equation)
  %equation : BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
    Norm.backward(_,(seen(normalized),seen(deviations)),seen(gammas)))
  backward_model(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
    bands,geometry,fill,workers,normalized,deviations,gammas,
    Shape.Both{Global.Usable{bands,geometry,fill,workers,{==}},%back : Global.Contract(~F32,~D,~dy_value,dy,_) contract},
    normalized_holds,deviation_holds,gamma_holds)

# Backward of dy and xhat [N,C,H,W] and of the inverse deviation and gamma
# [C], each a tensor of its reader in any layout. Each tensor backward
# returns holds, once Ready, the values of its reader in BackwardHolds: with
# M = N * H * W, dbeta = sum_axes(dy,[0,2,3]) and dgamma = sum_axes(dy *
# xhat,[0,2,3]), dx = ((dy * M - dbeta) - xhat * dgamma) * ((gamma *
# deviation) / M), then dgamma, dbeta and gamma as it was given.
def backward(~D: Data,~dy_value: D -> U32 -> F32,~XH: Data,~xhat_value: XH -> U32 -> F32,~G: Data,~g_value: G -> U32 -> F32,~DV: Data,
  ~deviation_value: DV -> U32 -> F32,+dy: D,+xhat: XH,+gamma: G,+deviation: DV,+images: U32,+channels: U32,+height: U32,+width: U32,
  gradients: Tensors.Tensor<F32>,normalized: Tensors.Tensor<F32>,deviations: Tensors.Tensor<F32>,gammas: Tensors.Tensor<F32>,
  gradient_shaped: Global.Shaped(~F32,~D,~dy_value,dy,[images,channels,height,width],gradients),
  +normalized_shaped: Global.Shaped(~F32,~XH,~xhat_value,xhat,[images,channels,height,width],normalized),
  +deviation_shaped: Global.Shaped(~F32,~DV,~deviation_value,deviation,[channels],deviations),+gamma_shaped: Global.Shaped(~F32,~G,~g_value,gamma,[channels],gammas)) ->
  BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
    Norm.backward(gradients,(normalized,deviations),gammas)):
  observed(~XH,~xhat_value,
    ~(tensor => BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
      Norm.backward(gradients,(tensor,deviations),gammas))),
    xhat,[images,channels,height,width],normalized,holds(~XH,~xhat_value,xhat,[images,channels,height,width],normalized,normalized_shaped),
    normalized_owner => normalized_holds =>
      observed(~DV,~deviation_value,
        ~(tensor => BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
          Norm.backward(gradients,(seen(normalized_owner),tensor),gammas))),
        deviation,[channels],deviations,holds(~DV,~deviation_value,deviation,[channels],deviations,deviation_shaped),
        deviation_owner => deviation_holds =>
          observed(~G,~g_value,
            ~(tensor => BackwardHolds(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
              Norm.backward(gradients,(seen(normalized_owner),seen(deviation_owner)),tensor))),
            gamma,[channels],gammas,holds(~G,~g_value,gamma,[channels],gammas,gamma_shaped),
            gamma_owner => gamma_holds =>
              backward_usable(~D,~dy_value,~XH,~xhat_value,~G,~g_value,~DV,~deviation_value,dy,xhat,gamma,deviation,images,channels,height,width,
                gradients,gradient_shaped,normalized_owner,deviation_owner,gamma_owner,normalized_holds,deviation_holds,gamma_holds))))
