# The six real public comparisons inherit the binary value theorems. Their
# kernel law is discharged here, not left as a premise of the public result.
import Base
import ../tensor_f32.bend as F32Tensor
import ../tensor.bend as Tensors
import ../tensor_lanes.bend as Lanes
import ../math_compare.bend as Compare
import ../storage_buffer.bend as Storage
import ../tensor_shape.bend as Shapes
import ../tensor_band.bend as Bands
import ./tensor_lanes_refinement.bend as LanesLaws
import ./tensor_combine_public.bend as CombinePublic
import ./tensor_combine_broadcast.bend as CombineBroadcast
import ./tensor_broadcast_public.bend as Broadcast
import ./tensor_band_combine_leaf.bend as Leaf
import ./tensor_global_shape_count.bend as Global

def BandLaw(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  ~operation: F32 -> F32 -> F32,~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32> -> Tensors.Tensor<F32>) -> Type:
  @+left: Left -> @+right: Right -> @+left_geometry: Bands.Geometry -> @+right_geometry: Bands.Geometry -> @+shape: List<&2,U32> ->
  @+left_fill: F32 -> @+right_fill: F32 -> @+left_workers: U32 -> @+right_workers: U32 -> @left_bands: Bands.Bands(~F32) -> @right_bands: Bands.Bands(~F32) ->
  @left_contract: Global.Contract(~F32,~Left,~left_value,left,Tensors.Tensor{left_bands,left_geometry,shape,left_fill,Tensors.Usable{},left_workers}) ->
  @right_contract: Global.Contract(~F32,~Right,~right_value,right,Tensors.Tensor{right_bands,right_geometry,shape,right_fill,Tensors.Usable{},right_workers}) ->
  Global.Contract(~F32,~Leaf.Context<Left,Right>,~Leaf.value(~F32,~Left,~Right,~operation,~left_value,~right_value),Leaf.Context{left,right},
    call(Tensors.Tensor{left_bands,left_geometry,shape,left_fill,Tensors.Usable{},left_workers},Tensors.Tensor{right_bands,right_geometry,shape,right_fill,Tensors.Usable{},right_workers}))

def banded(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,~name: String,~cost: F32) ->
  BandLaw(~Left,~Right,~left_value,~right_value,~operation,~Tensors.combine_with(~F32,~Lanes.combine(~operation),name,cost)):
  left => right => left_geometry => right_geometry => shape => left_fill => right_fill => left_workers => right_workers => left_bands => right_bands => left_contract => right_contract =>
    CombinePublic.combine(~F32,~Left,~Right,~operation,~Lanes.combine(~operation),~LanesLaws.combine(~operation),~left_value,~right_value,
      left,right,name,cost,shape,left_geometry,right_geometry,left_fill,right_fill,left_workers,right_workers,left_bands,right_bands,left_contract,right_contract)

def BroadcastLaw(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  ~operation: F32 -> F32 -> F32,~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32> -> Tensors.Tensor<F32>) -> Type:
  @+left: Left -> @+right: Right -> @+left_shape: List<&2,U32> -> @+right_shape: List<&2,U32> -> @+shape: List<&2,U32> ->
  @+left_geometry: Bands.Geometry -> @+right_geometry: Bands.Geometry -> @+left_fill: F32 -> @+right_fill: F32 -> @+left_workers: U32 -> @+right_workers: U32 ->
  @left_bands: Bands.Bands(~F32) -> @right_bands: Bands.Bands(~F32) ->
  @+common: {Shapes.broadcast(left_shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>} -> @+counted: Global.Counted<shape> ->
  @left_contract: Global.Contract(~F32,~Left,~left_value,left,Tensors.Tensor{left_bands,left_geometry,left_shape,left_fill,Tensors.Usable{},left_workers}) ->
  @right_contract: Global.Contract(~F32,~Right,~right_value,right,Tensors.Tensor{right_bands,right_geometry,right_shape,right_fill,Tensors.Usable{},right_workers}) ->
  CombineBroadcast.Combined(~F32,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left,Broadcast.expansion(left_shape,shape)},
    Broadcast.Expanded{right,Broadcast.expansion(right_shape,shape)},
    call(Tensors.Tensor{left_bands,left_geometry,left_shape,left_fill,Tensors.Usable{},left_workers},Tensors.Tensor{right_bands,right_geometry,right_shape,right_fill,Tensors.Usable{},right_workers}))

def broadcast(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,~name: String,~cost: F32) ->
  BroadcastLaw(~Left,~Right,~left_value,~right_value,~operation,~Tensors.combine_with(~F32,~Lanes.combine(~operation),name,cost)):
  left => right => left_shape => right_shape => shape => left_geometry => right_geometry => left_fill => right_fill => left_workers => right_workers => left_bands => right_bands =>
    common => counted => left_contract => right_contract =>
    CombineBroadcast.broadcast(~F32,~Left,~Right,~operation,~Lanes.combine(~operation),~LanesLaws.combine(~operation),~left_value,~right_value,
      left,right,name,cost,left_shape,right_shape,shape,left_geometry,right_geometry,left_fill,right_fill,left_workers,right_workers,left_bands,right_bands,
      common,counted,left_contract,right_contract)

# One rule for operands of one shape, one for operands that broadcast.
type Rules<-Same: Type,-Broadcasting: Type> is Type:
  Rules{same: Same,broadcasting: Broadcasting}

def PublicLaw(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,
  ~operation: F32 -> F32 -> F32,~call: Tensors.Tensor<F32> -> Tensors.Tensor<F32> -> Tensors.Tensor<F32>) -> Type:
  Rules<BandLaw(~Left,~Right,~left_value,~right_value,~operation,~call),BroadcastLaw(~Left,~Right,~left_value,~right_value,~operation,~call)>

def public(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32,~operation: F32 -> F32 -> F32,~name: String,~cost: F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~operation,~Tensors.combine_with(~F32,~Lanes.combine(~operation),name,cost)):
  Rules{banded(~Left,~Right,~left_value,~right_value,~operation,~name,~cost),broadcast(~Left,~Right,~left_value,~right_value,~operation,~name,~cost)}

def less(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.less_elements,~F32Tensor.less):
  public(~Left,~Right,~left_value,~right_value,~Compare.less_elements,~"less",~2.625)

def less_equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.less_equal_elements,~F32Tensor.less_equal):
  public(~Left,~Right,~left_value,~right_value,~Compare.less_equal_elements,~"less_equal",~2.625)

def greater(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.greater_elements,~F32Tensor.greater):
  public(~Left,~Right,~left_value,~right_value,~Compare.greater_elements,~"greater",~2.625)

def greater_equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.greater_equal_elements,~F32Tensor.greater_equal):
  public(~Left,~Right,~left_value,~right_value,~Compare.greater_equal_elements,~"greater_equal",~2.625)

def equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.equal_elements,~F32Tensor.equal):
  public(~Left,~Right,~left_value,~right_value,~Compare.equal_elements,~"equal",~1.375)

def not_equal(~Left: Data,~Right: Data,~left_value: Left -> U32 -> F32,~right_value: Right -> U32 -> F32) ->
  PublicLaw(~Left,~Right,~left_value,~right_value,~Compare.not_equal_elements,~F32Tensor.not_equal):
  public(~Left,~Right,~left_value,~right_value,~Compare.not_equal_elements,~"not_equal",~1.375)
