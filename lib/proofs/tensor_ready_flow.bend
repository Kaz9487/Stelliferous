# Readiness flows backwards through the public operations used by softmax.
import Base
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_shape.bend as Shapes
import ./tensor_copy_contract.bend as CopyContract
import ../tensor_operations.bend as Operations
import ../tensor_band.bend as Bands
import ../tensor_view.bend as Views
import ../storage_buffer.bend as Storage
import ./tensor_contract.bend as Contract
import ./storage_shape.bend as Shape
import ./word_encoding.bend as WordEncoding
import ./tensor_shape_refinement.bend as ShapeLaws
import ../tensor_band_schedule.bend as BandSchedule

def first_left(+left: Tensors.Status,+right: Tensors.Status,ready: {Tensors.usable(Tensors.first_failure(left,right)) == True{} : Bool}) -> {Tensors.usable(left) == True{} : Bool}:
  match left:
    case Tensors.Usable{}: {==}
    case Tensors.Failed{problem}: ready

def first_right(+left: Tensors.Status,+right: Tensors.Status,ready: {Tensors.usable(Tensors.first_failure(left,right)) == True{} : Bool}) -> {Tensors.usable(right) == True{} : Bool}:
  match left:
    case Tensors.Usable{}: ready
    case Tensors.Failed{problem}: Empty.absurd({Tensors.usable(right) == True{} : Bool},WordEncoding.false_true(ready))

# A usable combination of other extents had usable operands.
def broadcast_status(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+name: String,+operations: F32,
  +previous: Tensors.Status,common: Maybe<&2,List<&2,U32>>,+workers: U32,left: Bands.Bands(~Element),+geometry: Bands.Geometry,+ls: List<&2,U32>,+fill: Element,
  right: Bands.Bands(~Element),+right_geometry: Bands.Geometry,+rs: List<&2,U32>,rf: Element,
  ready: {Contract.tensor_ready(~Element,Tensors.combine_broadcast_with(~Element,~kernel,name,operations,previous,common,workers,left,geometry,ls,fill,right,right_geometry,rs,rf)) == True{} : Bool}) ->
  {Tensors.usable(previous) == True{} : Bool}:
  match previous:
    case Tensors.Usable{}: {==}
    case Tensors.Failed{problem}: ready

# A usable combination of equal extents had usable operands.
def equal_status(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+operations: F32,+previous: Tensors.Status,+workers: U32,+shape: List<&2,U32>,
  left: Bands.Bands(~Element),+geometry: Bands.Geometry,+fill: Element,right: Bands.Bands(~Element),+right_geometry: Bands.Geometry,+right_fill: Element,
  ready: {Contract.tensor_ready(~Element,Tensors.combine_equal_with(~Element,~kernel,operations,previous,workers,shape,left,geometry,fill,right,right_geometry,right_fill)) == True{} : Bool}) ->
  {Tensors.usable(previous) == True{} : Bool}:
  match previous:
    case Tensors.Usable{}: {==}
    case Tensors.Failed{problem}: ready

# Equal extents combine as they are; other extents broadcast.
def shapes(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+name: String,+operations: F32,same: Bool,
  left: Bands.Bands(~Element),+geometry: Bands.Geometry,+ls: List<&2,U32>,+fill: Element,+lt: Tensors.Status,+lw: U32,
  right: Bands.Bands(~Element),+right_geometry: Bands.Geometry,+rs: List<&2,U32>,+rf: Element,+rt: Tensors.Status,+rw: U32,
  ready: {Contract.tensor_ready(~Element,Tensors.combine_shapes(~Element,~kernel,name,operations,same,left,geometry,ls,fill,lt,lw,right,right_geometry,rs,rf,rt,rw)) == True{} : Bool}) ->
  {Tensors.usable(Tensors.first_failure(lt,rt)) == True{} : Bool}:
  match same:
    case True{}:
      equal_status(~Element,~kernel,operations,Tensors.first_failure(lt,rt),U32.max(lw,rw),ls,left,geometry,fill,right,right_geometry,rf,ready)
    case False{}:
      broadcast_status(~Element,~kernel,name,operations,Tensors.first_failure(lt,rt),Shapes.broadcast(ls,rs),U32.max(lw,rw),left,geometry,ls,fill,right,right_geometry,rs,rf,ready)

def combine_status(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+name: String,+operations: F32,left: Tensors.Tensor<Element>,right: Tensors.Tensor<Element>,
  ready: {Contract.tensor_ready(~Element,Tensors.combine_with(~Element,~kernel,name,operations,left,right)) == True{} : Bool}) -> Shape.Both<{Contract.tensor_ready(~Element,left) == True{} : Bool},{Contract.tensor_ready(~Element,right) == True{} : Bool}>:
  match left right:
    case Tensors.Tensor{left,+geometry,+ls,+fill,+lt,+lw} Tensors.Tensor{right,+right_geometry,+rs,+rf,+rt,+rw}:
      +accepted = shapes(~Element,~kernel,name,operations,Shapes.same_extents(ls,rs),left,geometry,ls,fill,lt,lw,right,right_geometry,rs,rf,rt,rw,ready)
      Shape.Both{first_left(lt,rt,accepted),first_right(lt,rt,accepted)}

def right_ready(-left: Bool,-right: Bool,evidence: Shape.Both<{left == True{} : Bool},{right == True{} : Bool}>) -> {right == True{} : Bool}:
  Shape.Both{other,result} = evidence
  result

# A combination whose left operand has the common extents has those extents.
def met_extents(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+operations: F32,target: BandSchedule.Target,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  left: Tensors.Tensor<Element>,right: Tensors.Tensor<Element>) ->
  {Contract.extents(~Element,Tensors.combine_met_with(~Element,~kernel,operations,target,shape,fill,workers,left,right)) == shape : List<&2,U32>}:
  match target:
    case BandSchedule.Target{geometry,layout,split}: {==}

def right_offered(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+operations: F32,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  left: Tensors.Tensor<Element>,+left_target: BandSchedule.Target,offered: Tensors.Tensor<Element> & BandSchedule.Target) ->
  {Contract.extents(~Element,Tensors.then(~Tensors.Tensor<Element>,~BandSchedule.Target,~Tensors.Tensor<Element>,offered,right => right_target =>
      Tensors.combine_met_with(~Element,~kernel,operations,BandSchedule.meet(left_target,right_target),shape,fill,workers,left,right))) == shape : List<&2,U32>}:
  (right,right_target) = offered
  met_extents(~Element,~kernel,operations,BandSchedule.meet(left_target,right_target),shape,fill,workers,left,right)

def left_offered(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+operations: F32,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  offered: Tensors.Tensor<Element> & BandSchedule.Target,right: Tensors.Tensor<Element>) ->
  {Contract.extents(~Element,Tensors.then(~Tensors.Tensor<Element>,~BandSchedule.Target,~Tensors.Tensor<Element>,offered,left => left_target =>
      Tensors.then(~Tensors.Tensor<Element>,~BandSchedule.Target,~Tensors.Tensor<Element>,Tensors.offer(~Element,right),right => right_target =>
        Tensors.combine_met_with(~Element,~kernel,operations,BandSchedule.meet(left_target,right_target),shape,fill,workers,left,right)))) == shape : List<&2,U32>}:
  (left,+left_target) = offered
  right_offered(~Element,~kernel,operations,shape,fill,workers,left,left_target,Tensors.offer(~Element,right))

def offered_extents(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+operations: F32,+workers: U32,+shape: List<&2,U32>,+fill: Element,
  left: Tensors.Tensor<Element>,right: Tensors.Tensor<Element>) ->
  {Contract.extents(~Element,Tensors.combine_offered_with(~Element,~kernel,operations,workers,shape,fill,left,right)) == shape : List<&2,U32>}:
  left_offered(~Element,~kernel,operations,shape,fill,workers,Tensors.offer(~Element,left),right)

def spread_extents(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+operations: F32,boxed: Bool,+shape: List<&2,U32>,+workers: U32,+geometry: Bands.Geometry,+layout: Bands.Plan,
  left: Bands.Bands(~Element),+fill: Element,source: Storage.Buffer<Element>,+right_shape: List<&2,U32>,right_fill: Element) ->
  {Contract.extents(~Element,Tensors.combine_spread_when(~Element,~kernel,operations,boxed,shape,workers,geometry,layout,left,fill,source,right_shape,right_fill)) == shape : List<&2,U32>}:
  match boxed:
    case True{}: {==}
    case False{}:
      offered_extents(~Element,~kernel,operations,workers,shape,fill,Tensors.Tensor{left,geometry,shape,fill,Tensors.Usable{},workers},
        Tensors.broadcast_to(~Element,Tensors.whole(~Element,source,right_shape,right_fill,Tensors.Usable{},workers),shape))

def viewed_extents(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+operations: F32,+shape: List<&2,U32>,+workers: U32,+geometry: Bands.Geometry,+fill: Element,
  planned: Bands.Bands(~Element) & Bands.Plan,source: Storage.Buffer<Element>,+right_shape: List<&2,U32>,right_fill: Element) ->
  {Contract.extents(~Element,Tensors.combine_viewed_with(~Element,~kernel,operations,shape,workers,geometry,fill,planned,source,right_shape,right_fill)) == shape : List<&2,U32>}:
  (left,+layout) = planned
  spread_extents(~Element,~kernel,operations,Bands.spread_boxed(Bands.Spread{geometry,shape,right_shape},layout) && Bool.not(Shapes.has_zero(shape)),
    shape,workers,geometry,layout,left,fill,source,right_shape,right_fill)

def stretched_extents(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+operations: F32,+shape: List<&2,U32>,+workers: U32,left: Tensors.Tensor<Element>,right: Tensors.Tensor<Element>) ->
  {Contract.extents(~Element,Tensors.combine_stretched_with(~Element,~kernel,operations,shape,workers,left,right)) == shape : List<&2,U32>}:
  Tensors.Tensor{bands,+geometry,left_shape,+fill,status,left_workers} = left
  Tensors.Tensor{right_bands,+right_geometry,+right_shape,+right_fill,right_status,right_workers} = right
  +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(right_shape)} : Bands.Geometry}
  viewed_extents(~Element,~kernel,operations,shape,workers,geometry,fill,Bands.plan(~Element,bands),
    Tensors.whole_storage(~Element,whole,right_fill,Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.Tensor{right_bands,right_geometry,right_shape,right_fill,right_status,right_workers})),
    right_shape,right_fill)

def common_extents(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+name: String,+operations: F32,status: Tensors.Status,+workers: U32,
  left: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,
  right: Bands.Bands(~Element),+right_geometry: Bands.Geometry,+right_shape: List<&2,U32>,+right_fill: Element) ->
  {Contract.extents(~Element,Tensors.combine_broadcast_with(~Element,~kernel,name,operations,status,Some{shape},workers,left,geometry,shape,left_fill,right,right_geometry,right_shape,right_fill)) == shape : List<&2,U32>}:
  match status:
    case Tensors.Usable{}:
      %Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},ShapeLaws.same_extents_reflexive(shape)) :
        {Contract.extents(~Element,Tensors.combine_common_with(~Element,~kernel,name,operations,_,Tensors.count_of(shape),shape,workers,left_fill,
          Tensors.Tensor{left,geometry,shape,left_fill,Tensors.Usable{},workers},Tensors.Tensor{right,right_geometry,right_shape,right_fill,Tensors.Usable{},workers})) == shape : List<&2,U32>}
      stretched_extents(~Element,~kernel,operations,shape,workers,Tensors.Tensor{left,geometry,shape,left_fill,Tensors.Usable{},workers},
        Tensors.Tensor{right,right_geometry,right_shape,right_fill,Tensors.Usable{},workers})
    case Tensors.Failed{problem}: {==}

def equal_extents(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+operations: F32,status: Tensors.Status,+workers: U32,+shape: List<&2,U32>,
  left: Bands.Bands(~Element),+geometry: Bands.Geometry,+left_fill: Element,right: Bands.Bands(~Element),+right_geometry: Bands.Geometry,right_fill: Element) ->
  {Contract.extents(~Element,Tensors.combine_equal_with(~Element,~kernel,operations,status,workers,shape,left,geometry,left_fill,right,right_geometry,right_fill)) == shape : List<&2,U32>}:
  match status:
    case Tensors.Usable{}:
      offered_extents(~Element,~kernel,operations,workers,shape,left_fill,Tensors.Tensor{left,geometry,shape,left_fill,Tensors.Usable{},workers},
        Tensors.Tensor{right,right_geometry,shape,right_fill,Tensors.Usable{},workers})
    case Tensors.Failed{problem}: {==}

def shapes_extents(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+name: String,+operations: F32,same: Bool,
  left: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+lt: Tensors.Status,+lw: U32,
  right: Bands.Bands(~Element),+right_geometry: Bands.Geometry,+right_shape: List<&2,U32>,+right_fill: Element,+rt: Tensors.Status,+rw: U32,
  +common: {Shapes.broadcast(shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>}) ->
  {Contract.extents(~Element,Tensors.combine_shapes(~Element,~kernel,name,operations,same,left,geometry,shape,left_fill,lt,lw,right,right_geometry,right_shape,right_fill,rt,rw)) == shape : List<&2,U32>}:
  match same:
    case True{}:
      equal_extents(~Element,~kernel,operations,Tensors.first_failure(lt,rt),U32.max(lw,rw),shape,left,geometry,left_fill,right,right_geometry,right_fill)
    case False{}:
      %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(shape,right_shape),Some{shape},common) :
        {Contract.extents(~Element,Tensors.combine_broadcast_with(~Element,~kernel,name,operations,Tensors.first_failure(lt,rt),_,U32.max(lw,rw),
          left,geometry,shape,left_fill,right,right_geometry,right_shape,right_fill)) == shape : List<&2,U32>}
      common_extents(~Element,~kernel,name,operations,Tensors.first_failure(lt,rt),U32.max(lw,rw),left,geometry,shape,left_fill,right,right_geometry,right_shape,right_fill)

def combined_extents(~Element: Data,~kernel: @+count: Nat -> @+right_view: Views.View -> @+offset: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element> -> Storage.Buffer<Element> & Storage.Buffer<Element>,+name: String,+operations: F32,
  left: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+left_fill: Element,+lt: Tensors.Status,+lw: U32,
  right: Bands.Bands(~Element),+right_geometry: Bands.Geometry,+right_shape: List<&2,U32>,+right_fill: Element,+rt: Tensors.Status,+rw: U32,
  +common: {Shapes.broadcast(shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>}) ->
  {Contract.extents(~Element,Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{left,geometry,shape,left_fill,lt,lw},
    Tensors.Tensor{right,right_geometry,right_shape,right_fill,rt,rw})) == shape : List<&2,U32>}:
  shapes_extents(~Element,~kernel,name,operations,Shapes.same_extents(shape,right_shape),left,geometry,shape,left_fill,lt,lw,right,right_geometry,right_shape,right_fill,rt,rw,common)

def combine_right(~Element: Data,~operation: Element -> Element -> Element,left: Tensors.Tensor<Element>,right: Tensors.Tensor<Element>,+name: String,+operations: F32,
  ready: {Contract.tensor_ready(~Element,Tensors.combine(~Element,~operation,name,operations,left,right)) == True{} : Bool}) -> {Contract.tensor_ready(~Element,right) == True{} : Bool}:
  right_ready(Contract.tensor_ready(~Element,left),Contract.tensor_ready(~Element,right),combine_status(~Element,~Operations.combine_kernel(~Element,~operation),name,operations,left,right,ready))

# Unary/layout failures cannot be repaired by these public calls.
def applied(~Element: Data,~mapper: @+count: Nat -> @+position: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element>,tensor: Tensors.Tensor<Element>,+operations: F32,
  ready: {Contract.tensor_ready(~Element,Tensors.apply(~Element,~mapper,operations,tensor)) == True{} : Bool}) -> {Contract.tensor_ready(~Element,tensor) == True{} : Bool}:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}: {==}
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}: ready

def reshaped(~Element: Data,tensor: Tensors.Tensor<Element>,+target: List<&2,U32>,
  ready: {Contract.tensor_ready(~Element,Layout.reshape(~Element,tensor,target)) == True{} : Bool}) -> {Contract.tensor_ready(~Element,tensor) == True{} : Bool}:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}: {==}
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}: ready

def reduced(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,tensor: Tensors.Tensor<Element>,+name: String,+operations: F32,+initial: Accumulator,+axes: List<&2,U32>,
  ready: {Contract.tensor_ready(~Element,Layout.reduce_axes(~Element,~Accumulator,~operation,~finish,name,operations,initial,tensor,axes)) == True{} : Bool}) -> {Contract.tensor_ready(~Element,tensor) == True{} : Bool}:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}: {==}
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}: ready

def reduced_all(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,tensor: Tensors.Tensor<Element>,+name: String,+operations: F32,+initial: Accumulator,
  ready: {Contract.tensor_ready(~Element,Layout.reduce_all(~Element,~Accumulator,~operation,~finish,name,operations,initial,tensor)) == True{} : Bool}) -> {Contract.tensor_ready(~Element,tensor) == True{} : Bool}:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}: {==}
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}: ready

def copied(~Element: Data,tensor: Tensors.Tensor<Element>,
  ready: {CopyContract.pair_ready(~Element,Layout.copy(~Element,tensor)) == True{} : Bool}) -> {Contract.tensor_ready(~Element,tensor) == True{} : Bool}:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}: {==}
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}: ready

def measured_source(~Element: Data,tensor: Tensors.Tensor<Element>) -> {Pair.fst(Tensors.Tensor<Element>,List<&2,U32>,Tensors.shape(~Element,tensor)) == tensor : Tensors.Tensor<Element>}:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,status,workers}: {==}

def rejected(~Element: Data,tensor: Tensors.Tensor<Element>,+name: String) -> {Contract.tensor_ready(~Element,Tensors.rejected(~Element,name,tensor)) == False{} : Bool}:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}: {==}
    case Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}: {==}
