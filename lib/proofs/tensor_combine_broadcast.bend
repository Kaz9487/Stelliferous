# Two Normal tensors whose extents broadcast to common extents combine into a
# Normal tensor of those extents. At every flat index it holds the operation
# of the two operands, each read as it is or through the view its extents
# stretch to. A left operand of the common extents keeps its bands, which
# read the parts of the right source their views reach; another left operand
# is first repeated to the common extents.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_shape.bend as Shapes
import ../tensor_view.bend as Views
import ../storage_buffer.bend as Storage
import ../tensor_band_schedule.bend as Schedule
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ../traversal_partition.bend as Partition
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_band_values.bend as BandValues
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_count.bend as BandCount
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_combine_leaf.bend as Leaf
import ./tensor_combine_public.bend as CombinePublic
import ./tensor_combine_kernel_values.bend as CombineKernel
import ./tensor_broadcast_public.bend as Broadcast
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_band_spread.bend as Spread
import ./tensor_gather_values.bend as GatherValues
import ./tensor_copy_values.bend as CopyValues
import ./tensor_contract.bend as Contract
import ./tensor_whole_storage.bend as WholeStorage
import ./tensor_whole_storage_values.bend as WholeStorageValues
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_refinement.bend as Count
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_unit_broadcast.bend as UnitBroadcast
import ./machine_limit_bounds.bend as Limits
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding

# The contract of a combination of two operands read at the common extents.
def Combined(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Broadcast.Expanded<Left>,+right_context: Broadcast.Expanded<Right>,-tensor: Tensors.Tensor<Element>) -> Data:
  Global.Contract(~Element,~Leaf.Context<Broadcast.Expanded<Left>,Broadcast.Expanded<Right>>,
    ~Leaf.value(~Element,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~Broadcast.value(~Element,~Left,~left_value),~Broadcast.value(~Element,~Right,~right_value)),
    Leaf.Context{left_context,right_context},tensor)

# A band combined with its part reads the right operand through its view.
def spread_context(~Left: Data,~Right: Data,context: CombineKernel.Combine<Left,Right>) -> Leaf.Context<Left,Broadcast.Expanded<Right>>:
  match context:
    case CombineKernel.Combine{left,right,source}: Leaf.Context{left,Broadcast.Expanded{right,Some{source}}}

def Spreading(~Left: Data,~Right: Data,before: CombineKernel.Combine<Left,Right>,after: Leaf.Context<Left,Broadcast.Expanded<Right>>) -> Data:
  {spread_context(~Left,~Right,before) == after : Leaf.Context<Left,Broadcast.Expanded<Right>>}

def spread_agrees(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  +before: CombineKernel.Combine<Left,Right>,+after: Leaf.Context<Left,Broadcast.Expanded<Right>>,+related: Spreading(~Left,~Right,before,after),+index: U32) ->
  {CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation,before,index) ==
    Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value),after,index) : Element}:
  match before:
    case CombineKernel.Combine{+left,+right,+source}:
      %related : {operation(left_value(left,index),right_value(right,CopyValues.address(source,U32.to_nat(index)))) ==
        Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value),_,index) : Element}
      {==}

def spread_read(~Element: Data,~Left: Data,~Right: Data,~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,~operation: Element -> Element -> Element,
  +left_context: Left,+right_context: Right,+view: Views.View,+geometry: Bands.Geometry,+content: Model.Content<Element>,
  evidence: Shape.Both<BandValues.Values(~Element,~CombineKernel.Combine<Left,Right>,~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
      CombineKernel.Combine{left_context,right_context,view},geometry,content),BandShape.WellFormed<Element,geometry,content>>) ->
  Shape.Both<BandValues.Values(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,
      ~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),
      Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{view}}},geometry,content),BandShape.WellFormed<Element,geometry,content>>:
  Shape.Both{values,formed} = evidence
  Shape.Both{
    BandValues.values(~Element,~CombineKernel.Combine<Left,Right>,~Leaf.Context<Left,Broadcast.Expanded<Right>>,
      ~CombineKernel.value(~Element,~Left,~Right,~left_value,~right_value,~operation),
      ~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),
      ~Spreading(~Left,~Right),~spread_agrees(~Element,~Left,~Right,~left_value,~right_value,~operation),
      CombineKernel.Combine{left_context,right_context,view},Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{view}}},{==},geometry,content,values),
    formed}

# Positive elements of a geometry give positive extents and room for its
# height below the capacity.
type Room<-height: U32,-width: U32,-limits: Machine.Limits> is Data:
  Room{width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},height_positive: {Nat.is_lt(0n,U32.to_nat(height)) == True{} : Bool},
    height_bound: {Nat.is_le(Nat.double(U32.to_nat(height)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}

def room(+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,
  +positive: {Nat.is_lt(0n,BandGather.elements(Bands.Geometry{images,channels,height,width})) == True{} : Bool},
  pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Room<height,width,limits>:
  +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +frame = Nat.mul(U32.to_nat(channels),spatial)
  +total = Nat.mul(U32.to_nat(images),frame)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +images_positive = RowMajor.left_factor_positive(U32.to_nat(images),frame,positive)
  +frame_positive = DivisionOrder.right_factor_positive(U32.to_nat(images),frame,positive)
  +channels_positive = RowMajor.left_factor_positive(U32.to_nat(channels),spatial,frame_positive)
  +spatial_positive = DivisionOrder.right_factor_positive(U32.to_nat(channels),spatial,frame_positive)
  +width_positive = DivisionOrder.right_factor_positive(U32.to_nat(height),U32.to_nat(width),spatial_positive)
  +frame_room = Order.transitive(frame,total,capacity,
    %RowMajor.multiply_commutative(frame,U32.to_nat(images)) : {Nat.is_le(frame,_) == True{} : Bool}
    RowMajor.product_covers(frame,U32.to_nat(images),images_positive),pixels)
  +spatial_room = Order.transitive(spatial,frame,capacity,
    %RowMajor.multiply_commutative(spatial,U32.to_nat(channels)) : {Nat.is_le(spatial,_) == True{} : Bool}
    RowMajor.product_covers(spatial,U32.to_nat(channels),channels_positive),frame_room)
  Room{width_positive,RowMajor.left_factor_positive(U32.to_nat(height),U32.to_nat(width),spatial_positive),
    Limits.double_bound(limits,U32.to_nat(height),
      Order.transitive(U32.to_nat(height),spatial,capacity,RowMajor.product_covers(U32.to_nat(height),U32.to_nat(width),width_positive),spatial_room))}

# Counted elements of a geometry fit the capacity of the machine.
def pixels(+geometry: Bands.Geometry,+count: U32,+limits: Machine.Limits,+capacity: U32,+capacity_limits: {capacity == Machine.capacity(limits) : U32},
  +capacity_default: {capacity == Machine.capacity(Default.limits()) : U32},evidence: Count.Count<BandGather.elements(geometry),count>) ->
  {Nat.is_le(BandGather.elements(geometry),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Count.Count{exact,bounded} = evidence
  %capacity_limits : {Nat.is_le(BandGather.elements(geometry),U32.to_nat(_)) == True{} : Bool}
  BandCount.natural_room(BandGather.elements(geometry),count,capacity,exact,
    %Equal.sym(U32,capacity,Machine.capacity(Default.limits()),capacity_default) : {U32.is_le(count,_) == True{} : Bool} bounded)

# The left bands of a boxed plan combined with the parts of the right source.
def spread_boxed(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+images: U32,+channels: U32,+height: U32,+width: U32,
  +fill: Element,+workers: U32,+source: Buffer.Model<Element>,+source_shape: List<&2,U32>,+source_fill: Element,+source_count: U32,+full: List<&2,Views.Axis>,+limits: Machine.Limits,
  +count: U32,+accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},+source_accepted: {Tensors.count_of(source_shape) == Some{source_count} : Maybe<&2,U32>},
  +equation: {Views.stretch_axes(Spread.source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +boxed: {Bands.spread_boxed(Bands.Spread{Bands.Geometry{images,channels,height,width},shape,source_shape},Model.layout(~Element,left)) == True{} : Bool},
  +pixels: {Nat.is_le(BandGather.elements(Bands.Geometry{images,channels,height,width}),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +left_formed: BandShape.WellFormed<Element,Bands.Geometry{images,channels,height,width},Model.content(~Element,left)>,
  +left_coherent: Global.Coherent(shape,Bands.Geometry{images,channels,height,width}),
  +left_values: BandValues.Values(~Element,~Left,~left_value,left_context,Bands.Geometry{images,channels,height,width},Model.content(~Element,left)),
  +source_formed: Contract.WellFormed<Element,source_shape,Buffer.content(~Element,source)>,+source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  room: Room<height,width,limits>) ->
  Global.Contract(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),
    Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{Views.View{full,0}}}},
    Tensors.combine_spread_when(~Element,~kernel,operations,True{},shape,workers,Bands.Geometry{images,channels,height,width},Model.layout(~Element,left),Model.pack(~Element,left),fill,
      Buffer.pack(~Element,source),source_shape,source_fill)):
  Room{+width_positive,+height_positive,+height_bound} = room
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +context = {Bands.Spread{geometry,shape,source_shape} : Bands.Spread}
  +parallel = Schedule.parallel(geometry,workers,12.0,operations,Model.layout(~Element,left))
  Global.contract(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),
    Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{Views.View{full,0}}}},geometry,shape,fill,workers,
    Bands.combine_spread_with(~Element,~kernel,context,parallel,Model.pack(~Element,left),Bands.spread(~Element,context,source_fill,Model.layout(~Element,left),Buffer.pack(~Element,source))),
    Global.Counted{count,accepted},left_coherent,
    spread_read(~Element,~Left,~Right,~left_value,~right_value,~operation,left_context,right_context,Views.View{full,0},geometry,
      Model.contents(~Element,Bands.combine_spread_with(~Element,~kernel,context,parallel,Model.pack(~Element,left),Bands.spread(~Element,context,source_fill,Model.layout(~Element,left),Buffer.pack(~Element,source)))),
      Spread.spread_combined(~Element,~Left,~Right,~left_value,~right_value,~operation,~kernel,~kernel_law,parallel,left,left_context,right_context,images,channels,height,width,
        shape,source_shape,full,source_count,limits,source_fill,source,WholeStorageValues.model_ready(~Element,source_shape,source,source_formed),source_values,
        Broadcast.whole_length(~Element,source_shape,source_count,source,source_formed,Global.whole_counted(source_shape,source_count,source_accepted)),
        Global.whole_counted(source_shape,source_count,source_accepted),equation,nonempty,left_coherent,width_positive,height_positive,height_bound,pixels,left_formed,left_values,boxed)))

# Bands that cannot read their parts combine with the right operand repeated
# to the common extents.
def offered_broadcast(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+geometry: Bands.Geometry,+fill: Element,+workers: U32,+full: List<&2,Views.Axis>,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,CombinePublic.tensor(~Element,left,geometry,shape,fill,workers)),-right: Tensors.Tensor<Element>,
  shaped: Global.Shaped(~Element,~Broadcast.Expanded<Right>,~Broadcast.value(~Element,~Right,~right_value),Broadcast.Expanded{right_context,Some{Views.View{full,0}}},shape,right)) ->
  Global.Contract(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),
    Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{Views.View{full,0}}}},
    Tensors.combine_offered_with(~Element,~kernel,operations,workers,shape,fill,CombinePublic.tensor(~Element,left,geometry,shape,fill,workers),right)):
  Shape.Both{usable,right_contract} = shaped
  Global.Usable{+right_bands,+right_geometry,+right_fill,+right_workers,+equation} = usable
  +view = {Views.View{full,0} : Views.View}
  %equation :
    Global.Contract(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),
      Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{view}}},
      Tensors.combine_offered_with(~Element,~kernel,operations,workers,shape,fill,CombinePublic.tensor(~Element,left,geometry,shape,fill,workers),_))
  CombinePublic.offered(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~kernel,~kernel_law,~left_value,~Broadcast.value(~Element,~Right,~right_value),
    left_context,Broadcast.Expanded{right_context,Some{view}},operations,shape,left,right_bands,geometry,right_geometry,fill,right_fill,workers,right_workers,fill,workers,
    left_contract,
    %Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{Model.pack(~Element,right_bands),right_geometry,shape,right_fill,Tensors.Usable{},right_workers},right,equation) :
      Global.Contract(~Element,~Broadcast.Expanded<Right>,~Broadcast.value(~Element,~Right,~right_value),Broadcast.Expanded{right_context,Some{view}},_)
    right_contract)

# The left bands combine with the parts of the right source their views reach
# when every band has a view; otherwise with the right repeated.
def spread_when(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  boxed: Bool,+left_context: Left,+right_context: Right,+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+geometry: Bands.Geometry,+fill: Element,+workers: U32,
  +source: Buffer.Model<Element>,+source_shape: List<&2,U32>,+source_fill: Element,+source_count: U32,+full: List<&2,Views.Axis>,
  +classified: {Bands.spread_boxed(Bands.Spread{geometry,shape,source_shape},Model.layout(~Element,left)) && Bool.not(Shapes.has_zero(shape)) == boxed : Bool},
  +counted: Global.Counted<shape>,+source_accepted: {Tensors.count_of(source_shape) == Some{source_count} : Maybe<&2,U32>},
  +equation: {Views.stretch_axes(Spread.source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +left_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,+left_coherent: Global.Coherent(shape,geometry),
  +left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),
  +source_formed: Contract.WellFormed<Element,source_shape,Buffer.content(~Element,source)>,+source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.content(~Element,source)),
  machine: WholeStorageValues.MachineObservation) ->
  Global.Contract(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),
    Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{Views.View{full,0}}}},
    Tensors.combine_spread_when(~Element,~kernel,operations,boxed,shape,workers,geometry,Model.layout(~Element,left),Model.pack(~Element,left),fill,Buffer.pack(~Element,source),source_shape,source_fill)):
  match boxed:
    case True{}:
      Bands.Geometry{+images,+channels,+height,+width} = geometry
      Global.Counted{+count,+accepted} = counted
      WholeStorageValues.MachineObservation{+limits,+capacity,+capacity_limits,+capacity_default} = machine
      +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
      +flag = Bands.spread_boxed(Bands.Spread{geometry,shape,source_shape},Model.layout(~Element,left))
      +nonempty = Boolean.not_true(Shapes.has_zero(shape),Boolean.right(flag,Bool.not(Shapes.has_zero(shape)),classified))
      +room_pixels = pixels(geometry,count,limits,capacity,capacity_limits,capacity_default,Global.geometry_evidence(shape,geometry,Global.Counted{count,accepted},left_coherent))
      spread_boxed(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,operations,shape,left,images,channels,height,width,
        fill,workers,source,source_shape,source_fill,source_count,full,limits,count,accepted,source_accepted,equation,nonempty,
        Boolean.left(flag,Bool.not(Shapes.has_zero(shape)),classified),room_pixels,left_formed,left_coherent,left_values,source_formed,source_values,
        room(images,channels,height,width,limits,
          %left_coherent : {Nat.is_lt(0n,_) == True{} : Bool}
          CountCompletion.positive_elements(shape,nonempty),room_pixels))
    case False{}:
      +source_exact = Global.whole_counted(source_shape,source_count,source_accepted)
      offered_broadcast(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,operations,shape,left,geometry,fill,workers,full,
        Shape.Both{Shape.Both{
            %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,left)),Model.content(~Element,left),Model.packed_contents(~Element,left)) :
              BandShape.WellFormed<Element,geometry,_>
            left_formed,
            Shape.Both{counted,left_coherent}},
          %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,left)),Model.content(~Element,left),Model.packed_contents(~Element,left)) :
            BandValues.Values(~Element,~Left,~left_value,left_context,geometry,_)
          left_values},
        Tensors.broadcast_to(~Element,Tensors.whole(~Element,Buffer.pack(~Element,source),source_shape,source_fill,Tensors.Usable{},workers),shape),
        Broadcast.broadcast_to(~Element,~Right,~right_value,right_context,source_shape,shape,Views.View{full,0},source_fill,workers,
          Model.Leaf{source,0,1},Bands.Geometry{1,1,1,Tensors.count_or_zero(source_shape)},
          WholeLeaf.held(~Element,~Right,~right_value,right_context,source,source_shape,source_fill,workers,Shape.Both{source_formed,source_values}),
          counted,Broadcast.stretched_view(source_shape,source_count,shape,full,source_exact,equation)))

# The whole right source, observed as a model.
def viewed_route(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+geometry: Bands.Geometry,+fill: Element,+workers: U32,
  +source_shape: List<&2,U32>,+source_fill: Element,+source_count: U32,+full: List<&2,Views.Axis>,-storage: Storage.Buffer<Element>,observation: Buffer.Observation<Element,storage>,
  +counted: Global.Counted<shape>,+source_accepted: {Tensors.count_of(source_shape) == Some{source_count} : Maybe<&2,U32>},
  +equation: {Views.stretch_axes(Spread.source_axes(source_shape,source_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +left_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,left)>,+left_coherent: Global.Coherent(shape,geometry),
  +left_values: BandValues.Values(~Element,~Left,~left_value,left_context,geometry,Model.content(~Element,left)),
  +source_formed: Contract.WellFormed<Element,source_shape,Buffer.contents(~Element,storage)>,+source_values: Contract.Values(~Element,~Right,~right_value,right_context,Buffer.contents(~Element,storage))) ->
  Global.Contract(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),
    Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{Views.View{full,0}}}},
    Tensors.combine_viewed_with(~Element,~kernel,operations,shape,workers,geometry,fill,(Model.pack(~Element,left),Model.layout(~Element,left)),storage,source_shape,source_fill)):
  Buffer.Observation{+source,+stored} = observation
  %stored : Global.Contract(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,
    ~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),
    Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{Views.View{full,0}}}},
    Tensors.combine_viewed_with(~Element,~kernel,operations,shape,workers,geometry,fill,(Model.pack(~Element,left),Model.layout(~Element,left)),_,source_shape,source_fill))
  spread_when(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,
    Bands.spread_boxed(Bands.Spread{geometry,shape,source_shape},Model.layout(~Element,left)) && Bool.not(Shapes.has_zero(shape)),
    left_context,right_context,operations,shape,left,geometry,fill,workers,source,source_shape,source_fill,source_count,full,{==},counted,source_accepted,equation,
    left_formed,left_coherent,left_values,
    %Buffer.packed_contents(~Element,source) : Contract.WellFormed<Element,source_shape,_>
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,source),storage,stored) : Contract.WellFormed<Element,source_shape,Buffer.contents(~Element,_)>
    source_formed,
    %Buffer.packed_contents(~Element,source) : Contract.Values(~Element,~Right,~right_value,right_context,_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,source),storage,stored) : Contract.Values(~Element,~Right,~right_value,right_context,Buffer.contents(~Element,_))
    source_values,
    WholeStorageValues.observe_machine())

# A left operand of the common extents keeps its bands; the right operand is
# read through its view stretched to them.
def stretched_step(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+lg: Bands.Geometry,+lf: Element,+workers: U32,
  +rs: List<&2,U32>,+rf: Element,+right_count: U32,+right_accepted: {Tensors.count_of(rs) == Some{right_count} : Maybe<&2,U32>},
  +full: List<&2,Views.Axis>,+equation: {Views.stretch_axes(Spread.source_axes(rs,right_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  +counted: Global.Counted<shape>,+left_formed: BandShape.WellFormed<Element,lg,Model.content(~Element,left)>,+left_coherent: Global.Coherent(shape,lg),
  +left_values: BandValues.Values(~Element,~Left,~left_value,left_context,lg,Model.content(~Element,left)),
  +model: Buffer.Model<Element>,facts: GatherValues.Facts(~Element,~Right,~right_value,right_context,rs,Buffer.content(~Element,model))) ->
  Global.Contract(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{Views.View{full,0}}}},
    Tensors.combine_viewed_with(~Element,~kernel,operations,shape,workers,lg,lf,Bands.plan(~Element,Model.pack(~Element,left)),Buffer.pack(~Element,model),rs,rf)):
  Shape.Both{+formed,+values} = facts
  +packed = Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model))
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,left)),(Model.pack(~Element,left),Model.layout(~Element,left)),Model.plan(~Element,left)) :
    Global.Contract(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{Views.View{full,0}}}},
      Tensors.combine_viewed_with(~Element,~kernel,operations,shape,workers,lg,lf,_,Buffer.pack(~Element,model),rs,rf))
  viewed_route(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,operations,shape,left,lg,lf,workers,rs,rf,right_count,full,
    Buffer.pack(~Element,model),Buffer.observe(~Element,Buffer.pack(~Element,model)),counted,right_accepted,equation,left_formed,left_coherent,left_values,
    %packed : Contract.WellFormed<Element,rs,_> formed,
    %packed : Contract.Values(~Element,~Right,~right_value,right_context,_) values)

# The right operand enters as the storage of its one band.
def stretched_route(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+lg: Bands.Geometry,+lf: Element,+lw: U32,+left_shape: List<&2,U32>,+workers: U32,
  +right: Model.Model<Element>,+rg: Bands.Geometry,+rs: List<&2,U32>,+rf: Element,+rw: U32,+right_count: U32,+right_accepted: {Tensors.count_of(rs) == Some{right_count} : Maybe<&2,U32>},
  +full: List<&2,Views.Axis>,+equation: {Views.stretch_axes(Spread.source_axes(rs,right_count,List.length(&2,U32,shape)),shape) == Some{full} : Maybe<&2,List<&2,Views.Axis>>},
  same_left: {Shapes.same_extents(left_shape,shape) == True{} : Bool},+counted: Global.Counted<shape>,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,CombinePublic.tensor(~Element,left,lg,left_shape,lf,lw)),
  +right_formed: BandShape.WellFormed<Element,rg,Model.contents(~Element,Model.pack(~Element,right))>,+right_coherent: Global.Coherent(rs,rg),
  +right_values: BandValues.Values(~Element,~Right,~right_value,right_context,rg,Model.contents(~Element,Model.pack(~Element,right)))) ->
  Global.Contract(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{Views.View{full,0}}}},
    Tensors.combine_stretched_with(~Element,~kernel,operations,shape,workers,CombinePublic.tensor(~Element,left,lg,left_shape,lf,lw),CombinePublic.tensor(~Element,right,rg,rs,rf,rw))):
  Shape.Both{Shape.Both{+left_formed,Shape.Both{+left_counted,+left_coherent}},+left_values} = left_contract
  +equal = ShapeLaws.same_extents(left_shape,shape,same_left)
  +left_shaped = {
    %Model.packed_contents(~Element,left) : BandShape.WellFormed<Element,lg,_>
    left_formed : BandShape.WellFormed<Element,lg,Model.content(~Element,left)>}
  +left_read = {
    %Model.packed_contents(~Element,left) : BandValues.Values(~Element,~Left,~left_value,left_context,lg,_)
    left_values : BandValues.Values(~Element,~Left,~left_value,left_context,lg,Model.content(~Element,left))}
  +left_spanned = {%equal : Global.Coherent(_,lg) left_coherent : Global.Coherent(shape,lg)}
  WholeLeaf.with_whole(~Element,~Right,~right_value,~Tensors.Tensor<Element>,
    ~(result => Global.Contract(~Element,~Leaf.Context<Left,Broadcast.Expanded<Right>>,~Leaf.value(~Element,~Left,~Broadcast.Expanded<Right>,~operation,~left_value,~Broadcast.value(~Element,~Right,~right_value)),Leaf.Context{left_context,Broadcast.Expanded{right_context,Some{Views.View{full,0}}}},result)),
    ~(source => right_shape => right_fill => right_status => right_workers =>
      Tensors.combine_viewed_with(~Element,~kernel,operations,shape,workers,lg,lf,Bands.plan(~Element,Model.pack(~Element,left)),source,right_shape,right_fill)),
    right_context,right,rg,rs,rf,rw,
    Shape.Both{Shape.Both{right_formed,Shape.Both{Global.Counted{right_count,right_accepted},right_coherent}},right_values},
    model => facts => stretched_step(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,operations,shape,left,lg,lf,workers,
      rs,rf,right_count,right_accepted,full,equation,counted,left_shaped,left_spanned,left_read,model,facts))

# Some of one shape is Some of another only when the two are one.
def found(+shape: List<&2,U32>,+target: List<&2,U32>,equation: {Some{shape} == Some{target} : Maybe<&2,List<&2,U32>>}) -> {shape == target : List<&2,U32>}:
  Equal.cong(Maybe<&2,List<&2,U32>>,List<&2,U32>,result => Maybe.default(&2,List<&2,U32>,result,shape),Some{shape},Some{target},equation)

# Operands of one shape are read as they are.
def equal_route(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+shape: List<&2,U32>,+left: Model.Model<Element>,+right: Model.Model<Element>,
  +lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,CombinePublic.tensor(~Element,left,lg,shape,lf,lw)),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,CombinePublic.tensor(~Element,right,rg,shape,rf,rw))) ->
  Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion(shape,shape)},Broadcast.Expanded{right_context,Broadcast.expansion(shape,shape)},
    Tensors.combine_shapes(~Element,~kernel,name,operations,True{},Model.pack(~Element,left),lg,shape,lf,Tensors.Usable{},lw,Model.pack(~Element,right),rg,shape,rf,Tensors.Usable{},rw)):
  +workers = U32.max(lw,rw)
  %Equal.sym(Bool,Shapes.same_extents(shape,shape),True{},ShapeLaws.same_extents_reflexive(shape)) :
    Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,
      Broadcast.Expanded{left_context,Broadcast.expansion_when(_,shape,shape)},Broadcast.Expanded{right_context,Broadcast.expansion_when(_,shape,shape)},
      Tensors.combine_shapes(~Element,~kernel,name,operations,True{},Model.pack(~Element,left),lg,shape,lf,Tensors.Usable{},lw,Model.pack(~Element,right),rg,shape,rf,Tensors.Usable{},rw))
  CombinePublic.offered(~Element,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~kernel,~kernel_law,~Broadcast.value(~Element,~Left,~left_value),~Broadcast.value(~Element,~Right,~right_value),
    Broadcast.Expanded{left_context,None{}},Broadcast.Expanded{right_context,None{}},operations,shape,left,right,lg,rg,lf,rf,workers,workers,lf,workers,
    Broadcast.plain(~Element,~Left,~left_value,left_context,Model.pack(~Element,left),lg,shape,lf,workers,left_contract),
    Broadcast.plain(~Element,~Right,~right_value,right_context,Model.pack(~Element,right),rg,shape,rf,workers,right_contract))

# Operands of equal extents have those extents in common.
def matched(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+shape: List<&2,U32>,
  +left: Model.Model<Element>,+right: Model.Model<Element>,+lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32,
  +equal: {left_shape == right_shape : List<&2,U32>},common: {Shapes.broadcast(left_shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>},
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,CombinePublic.tensor(~Element,left,lg,left_shape,lf,lw)),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,CombinePublic.tensor(~Element,right,rg,right_shape,rf,rw))) ->
  Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion(left_shape,shape)},Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
    Tensors.combine_shapes(~Element,~kernel,name,operations,True{},Model.pack(~Element,left),lg,left_shape,lf,Tensors.Usable{},lw,Model.pack(~Element,right),rg,right_shape,rf,Tensors.Usable{},rw)):
  +same = found(left_shape,shape,Equal.trans(Maybe<&2,List<&2,U32>>,Some{left_shape},Shapes.broadcast(left_shape,left_shape),Some{shape},
    Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(left_shape,left_shape),Some{left_shape},UnitBroadcast.self_broadcast(left_shape)),
    %Equal.sym(List<&2,U32>,left_shape,right_shape,equal) : {Shapes.broadcast(left_shape,_) == Some{shape} : Maybe<&2,List<&2,U32>>} common))
  %equal : Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,
    Broadcast.Expanded{left_context,Broadcast.expansion(left_shape,shape)},Broadcast.Expanded{right_context,Broadcast.expansion(_,shape)},
    Tensors.combine_shapes(~Element,~kernel,name,operations,True{},Model.pack(~Element,left),lg,left_shape,lf,Tensors.Usable{},lw,Model.pack(~Element,right),rg,_,rf,Tensors.Usable{},rw))
  %same : Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,
    Broadcast.Expanded{left_context,Broadcast.expansion(left_shape,_)},Broadcast.Expanded{right_context,Broadcast.expansion(left_shape,_)},
    Tensors.combine_shapes(~Element,~kernel,name,operations,True{},Model.pack(~Element,left),lg,left_shape,lf,Tensors.Usable{},lw,Model.pack(~Element,right),rg,left_shape,rf,Tensors.Usable{},rw))
  equal_route(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,name,operations,left_shape,left,right,lg,rg,lf,rf,lw,rw,
    left_contract,
    %Equal.sym(List<&2,U32>,left_shape,right_shape,equal) : Global.Contract(~Element,~Right,~right_value,right_context,CombinePublic.tensor(~Element,right,rg,_,rf,rw)) right_contract)

# The right extents differ from the common ones when the left extents are
# them and the operands differ.
def differs(right_same: Bool,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+shape: List<&2,U32>,
  +classified: {Shapes.same_extents(right_shape,shape) == right_same : Bool},+left_same: {left_shape == shape : List<&2,U32>},
  differ: {Shapes.same_extents(left_shape,right_shape) == False{} : Bool}) -> {right_same == False{} : Bool}:
  match right_same:
    case True{}:
      +right_equal = ShapeLaws.same_extents(right_shape,shape,classified)
      Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Shapes.same_extents(left_shape,right_shape),True{},
        Equal.sym(Bool,Shapes.same_extents(left_shape,right_shape),False{},differ),
        %Equal.sym(List<&2,U32>,left_shape,shape,left_same) : {Shapes.same_extents(_,right_shape) == True{} : Bool}
        %Equal.sym(List<&2,U32>,right_shape,shape,right_equal) : {Shapes.same_extents(shape,_) == True{} : Bool}
        ShapeLaws.same_extents_reflexive(shape))))
    case False{}: {==}

# A left operand of the common extents keeps its bands.
def kept(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operations: F32,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+shape: List<&2,U32>,
  +left: Model.Model<Element>,+right: Model.Model<Element>,+lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+workers: U32,+left_count: U32,+right_count: U32,
  +same_left: {Shapes.same_extents(left_shape,shape) == True{} : Bool},differ: {Shapes.same_extents(left_shape,right_shape) == False{} : Bool},+counted: Global.Counted<shape>,
  stretched: Shape.Both<Broadcast.Stretched<left_shape,left_count,shape>,Broadcast.Stretched<right_shape,right_count,shape>>,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,CombinePublic.tensor(~Element,left,lg,left_shape,lf,workers)),
  +right_formed: BandShape.WellFormed<Element,rg,Model.contents(~Element,Model.pack(~Element,right))>,+right_coherent: Global.Coherent(right_shape,rg),
  +right_values: BandValues.Values(~Element,~Right,~right_value,right_context,rg,Model.contents(~Element,Model.pack(~Element,right))),
  +right_accepted: {Tensors.count_of(right_shape) == Some{right_count} : Maybe<&2,U32>}) ->
  Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion_when(True{},left_shape,shape)},
    Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
    Tensors.combine_stretched_with(~Element,~kernel,operations,shape,workers,CombinePublic.tensor(~Element,left,lg,left_shape,lf,workers),CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers))):
  Shape.Both{left_stretched,right_stretched} = stretched
  Broadcast.Stretched{+full,+equation} = right_stretched
  +right_differs = differs(Shapes.same_extents(right_shape,shape),left_shape,right_shape,shape,{==},ShapeLaws.same_extents(left_shape,shape,same_left),differ)
  %Equal.sym(Bool,Shapes.same_extents(right_shape,shape),False{},right_differs) :
    Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,None{}},Broadcast.Expanded{right_context,Broadcast.expansion_when(_,right_shape,shape)},
      Tensors.combine_stretched_with(~Element,~kernel,operations,shape,workers,CombinePublic.tensor(~Element,left,lg,left_shape,lf,workers),CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers)))
  %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(right_shape),shape),Some{Views.View{full,0}},
      Broadcast.stretched_view(right_shape,right_count,shape,full,Global.whole_counted(right_shape,right_count,right_accepted),equation)) :
    Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,None{}},Broadcast.Expanded{right_context,_},
      Tensors.combine_stretched_with(~Element,~kernel,operations,shape,workers,CombinePublic.tensor(~Element,left,lg,left_shape,lf,workers),CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers)))
  stretched_route(~Element,~Broadcast.Expanded<Left>,~Right,~operation,~kernel,~kernel_law,~Broadcast.value(~Element,~Left,~left_value),~right_value,
    Broadcast.Expanded{left_context,None{}},right_context,operations,shape,left,lg,lf,workers,left_shape,workers,
    right,rg,right_shape,rf,workers,right_count,right_accepted,full,equation,same_left,counted,
    Broadcast.plain(~Element,~Left,~left_value,left_context,Model.pack(~Element,left),lg,left_shape,lf,workers,left_contract),
    right_formed,right_coherent,right_values)

# A repeated left operand meets a right operand of the common extents as it
# is, and a right operand of other extents through its view.
def repeated(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  right_same: Bool,+left_context: Left,+right_context: Right,+operations: F32,+right_shape: List<&2,U32>,+shape: List<&2,U32>,+repeated_bands: Model.Model<Element>,+repeated_geometry: Bands.Geometry,
  +right: Model.Model<Element>,+rg: Bands.Geometry,+lf: Element,+repeated_fill: Element,+repeated_workers: U32,+rf: Element,+workers: U32,+left_full: List<&2,Views.Axis>,
  +right_count: U32,+classified: {Shapes.same_extents(right_shape,shape) == right_same : Bool},+counted: Global.Counted<shape>,
  +right_accepted: {Tensors.count_of(right_shape) == Some{right_count} : Maybe<&2,U32>},right_stretched: Broadcast.Stretched<right_shape,right_count,shape>,
  left_contract: Global.Contract(~Element,~Broadcast.Expanded<Left>,~Broadcast.value(~Element,~Left,~left_value),Broadcast.Expanded{left_context,Some{Views.View{left_full,0}}},
    CombinePublic.tensor(~Element,repeated_bands,repeated_geometry,shape,repeated_fill,repeated_workers)),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers))) ->
  Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Some{Views.View{left_full,0}}},
    Broadcast.Expanded{right_context,Broadcast.expansion_when(right_same,right_shape,shape)},
    Tensors.combine_repeated_when(~Element,~kernel,operations,right_same,shape,workers,lf,
      CombinePublic.tensor(~Element,repeated_bands,repeated_geometry,shape,repeated_fill,repeated_workers),
      CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers))):
  match right_same:
    case True{}:
      +equal = ShapeLaws.same_extents(right_shape,shape,classified)
      %Equal.sym(List<&2,U32>,right_shape,shape,equal) :
        Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Some{Views.View{left_full,0}}},Broadcast.Expanded{right_context,None{}},
          Tensors.combine_repeated_when(~Element,~kernel,operations,True{},shape,workers,lf,CombinePublic.tensor(~Element,repeated_bands,repeated_geometry,shape,repeated_fill,repeated_workers),
            CombinePublic.tensor(~Element,right,rg,_,rf,workers)))
      CombinePublic.offered(~Element,~Broadcast.Expanded<Left>,~Broadcast.Expanded<Right>,~operation,~kernel,~kernel_law,~Broadcast.value(~Element,~Left,~left_value),~Broadcast.value(~Element,~Right,~right_value),
        Broadcast.Expanded{left_context,Some{Views.View{left_full,0}}},Broadcast.Expanded{right_context,None{}},operations,shape,repeated_bands,right,repeated_geometry,rg,
        repeated_fill,rf,repeated_workers,workers,lf,workers,left_contract,
        Broadcast.plain(~Element,~Right,~right_value,right_context,Model.pack(~Element,right),rg,shape,rf,workers,
          %equal : Global.Contract(~Element,~Right,~right_value,right_context,CombinePublic.tensor(~Element,right,rg,_,rf,workers)) right_contract))
    case False{}:
      Broadcast.Stretched{+full,+equation} = right_stretched
      Shape.Both{Shape.Both{+right_formed,Shape.Both{right_counted,+right_coherent}},right_values} = right_contract
      %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(right_shape),shape),Some{Views.View{full,0}},
          Broadcast.stretched_view(right_shape,right_count,shape,full,Global.whole_counted(right_shape,right_count,right_accepted),equation)) :
        Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Some{Views.View{left_full,0}}},Broadcast.Expanded{right_context,_},
          Tensors.combine_repeated_when(~Element,~kernel,operations,False{},shape,workers,lf,
            CombinePublic.tensor(~Element,repeated_bands,repeated_geometry,shape,repeated_fill,repeated_workers),
            CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers)))
      stretched_route(~Element,~Broadcast.Expanded<Left>,~Right,~operation,~kernel,~kernel_law,~Broadcast.value(~Element,~Left,~left_value),~right_value,
        Broadcast.Expanded{left_context,Some{Views.View{left_full,0}}},right_context,operations,shape,repeated_bands,repeated_geometry,
        repeated_fill,repeated_workers,shape,workers,right,rg,right_shape,rf,workers,right_count,right_accepted,full,equation,ShapeLaws.same_extents_reflexive(shape),counted,
        left_contract,right_formed,right_coherent,right_values)

# The repeated left operand reads the left operand through its view.
def repeated_whole(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operations: F32,+right_shape: List<&2,U32>,+shape: List<&2,U32>,+right: Model.Model<Element>,+rg: Bands.Geometry,
  +lf: Element,+rf: Element,+workers: U32,+left_full: List<&2,Views.Axis>,+right_count: U32,+counted: Global.Counted<shape>,
  +right_accepted: {Tensors.count_of(right_shape) == Some{right_count} : Maybe<&2,U32>},-left: Tensors.Tensor<Element>,
  shaped: Global.Shaped(~Element,~Broadcast.Expanded<Left>,~Broadcast.value(~Element,~Left,~left_value),Broadcast.Expanded{left_context,Some{Views.View{left_full,0}}},shape,left),
  right_stretched: Broadcast.Stretched<right_shape,right_count,shape>,
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers))) ->
  Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Some{Views.View{left_full,0}}},
    Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
    Tensors.combine_repeated_with(~Element,~kernel,operations,shape,workers,lf,left,CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers))):
  Shape.Both{usable,left_contract} = shaped
  Global.Usable{+repeated_bands,+repeated_geometry,+repeated_fill,+repeated_workers,+equation} = usable
  %equation :
    Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Some{Views.View{left_full,0}}},Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
      Tensors.combine_repeated_with(~Element,~kernel,operations,shape,workers,lf,_,CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers)))
  repeated(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,Shapes.same_extents(right_shape,shape),left_context,right_context,operations,right_shape,shape,
    repeated_bands,repeated_geometry,right,rg,lf,repeated_fill,repeated_workers,rf,workers,left_full,right_count,{==},counted,right_accepted,right_stretched,
    %Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{Model.pack(~Element,repeated_bands),repeated_geometry,shape,repeated_fill,Tensors.Usable{},repeated_workers},left,equation) :
      Global.Contract(~Element,~Broadcast.Expanded<Left>,~Broadcast.value(~Element,~Left,~left_value),Broadcast.Expanded{left_context,Some{Views.View{left_full,0}}},_)
    left_contract,
    right_contract)

# A left operand of other extents is first repeated to the common extents.
def left_repeated(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+operations: F32,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+shape: List<&2,U32>,
  +left: Model.Model<Element>,+right: Model.Model<Element>,+lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+workers: U32,+left_count: U32,+right_count: U32,
  +counted: Global.Counted<shape>,+left_accepted: {Tensors.count_of(left_shape) == Some{left_count} : Maybe<&2,U32>},
  +right_accepted: {Tensors.count_of(right_shape) == Some{right_count} : Maybe<&2,U32>},
  stretched: Shape.Both<Broadcast.Stretched<left_shape,left_count,shape>,Broadcast.Stretched<right_shape,right_count,shape>>,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,CombinePublic.tensor(~Element,left,lg,left_shape,lf,workers)),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers))) ->
  Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion_when(False{},left_shape,shape)},
    Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
    Tensors.combine_repeated_with(~Element,~kernel,operations,shape,workers,lf,Tensors.broadcast_to(~Element,CombinePublic.tensor(~Element,left,lg,left_shape,lf,workers),shape),
      CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers))):
  Shape.Both{left_stretched,right_stretched} = stretched
  Broadcast.Stretched{+left_full,+left_equation} = left_stretched
  +view = {Views.View{left_full,0} : Views.View}
  +left_view = Broadcast.stretched_view(left_shape,left_count,shape,left_full,Global.whole_counted(left_shape,left_count,left_accepted),left_equation)
  %Equal.sym(Maybe<&2,Views.View>,Tensors.stretched(Tensors.view_of(left_shape),shape),Some{view},left_view) :
    Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,_},Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
      Tensors.combine_repeated_with(~Element,~kernel,operations,shape,workers,lf,Tensors.broadcast_to(~Element,CombinePublic.tensor(~Element,left,lg,left_shape,lf,workers),shape),
        CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers)))
  repeated_whole(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,operations,right_shape,shape,right,rg,lf,rf,workers,
    left_full,right_count,counted,right_accepted,Tensors.broadcast_to(~Element,CombinePublic.tensor(~Element,left,lg,left_shape,lf,workers),shape),
    Broadcast.broadcast_to(~Element,~Left,~left_value,left_context,left_shape,shape,view,lf,workers,left,lg,left_contract,counted,left_view),
    right_stretched,right_contract)

# Operands of other extents meet at their common extents: a left operand of
# those extents keeps its bands, another is repeated to them.
def common_route(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  same: Bool,+left_context: Left,+right_context: Right,+name: String,+operations: F32,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+shape: List<&2,U32>,
  +left: Model.Model<Element>,+right: Model.Model<Element>,+lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+workers: U32,
  +classified: {Shapes.same_extents(left_shape,shape) == same : Bool},differ: {Shapes.same_extents(left_shape,right_shape) == False{} : Bool},
  +common: {Shapes.broadcast(left_shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>},+counted: Global.Counted<shape>,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,CombinePublic.tensor(~Element,left,lg,left_shape,lf,workers)),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers))) ->
  Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion_when(same,left_shape,shape)},
    Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
    Tensors.combine_common_with(~Element,~kernel,name,operations,same,Tensors.count_of(shape),shape,workers,lf,
      CombinePublic.tensor(~Element,left,lg,left_shape,lf,workers),CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers))):
  match same:
    case True{}:
      Shape.Both{Shape.Both{+left_formed,Shape.Both{left_counted,+left_coherent}},left_values} = left_contract
      Global.Counted{+left_count,+left_accepted} = left_counted
      Shape.Both{Shape.Both{+right_formed,Shape.Both{right_counted,+right_coherent}},right_values} = right_contract
      Global.Counted{+right_count,+right_accepted} = right_counted
      kept(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,operations,left_shape,right_shape,shape,
        left,right,lg,rg,lf,rf,workers,left_count,right_count,classified,differ,counted,
        Broadcast.stretched(left_shape,right_shape,shape,left_count,right_count,Global.whole_counted(left_shape,left_count,left_accepted),
          Global.whole_counted(right_shape,right_count,right_accepted),common),
        Shape.Both{Shape.Both{left_formed,Shape.Both{Global.Counted{left_count,left_accepted},left_coherent}},left_values},
        right_formed,right_coherent,right_values,right_accepted)
    case False{}:
      Global.Counted{+count,+accepted} = counted
      Shape.Both{Shape.Both{+left_formed,Shape.Both{left_counted,+left_coherent}},left_values} = left_contract
      Global.Counted{+left_count,+left_accepted} = left_counted
      Shape.Both{Shape.Both{+right_formed,Shape.Both{right_counted,+right_coherent}},right_values} = right_contract
      Global.Counted{+right_count,+right_accepted} = right_counted
      %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},accepted) :
        Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion_when(False{},left_shape,shape)},
          Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
          Tensors.combine_common_with(~Element,~kernel,name,operations,False{},_,shape,workers,lf,
            CombinePublic.tensor(~Element,left,lg,left_shape,lf,workers),CombinePublic.tensor(~Element,right,rg,right_shape,rf,workers)))
      left_repeated(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,operations,left_shape,right_shape,shape,
        left,right,lg,rg,lf,rf,workers,left_count,right_count,Global.Counted{count,accepted},left_accepted,right_accepted,
        Broadcast.stretched(left_shape,right_shape,shape,left_count,right_count,Global.whole_counted(left_shape,left_count,left_accepted),
          Global.whole_counted(right_shape,right_count,right_accepted),common),
        Shape.Both{Shape.Both{left_formed,Shape.Both{Global.Counted{left_count,left_accepted},left_coherent}},left_values},
        Shape.Both{Shape.Both{right_formed,Shape.Both{Global.Counted{right_count,right_accepted},right_coherent}},right_values})

# Operands of equal extents combine as they are; operands of other extents
# meet at their common extents.
def shapes(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  same: Bool,+left_context: Left,+right_context: Right,+name: String,+operations: F32,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+shape: List<&2,U32>,
  +left: Model.Model<Element>,+right: Model.Model<Element>,+lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32,
  +classified: {Shapes.same_extents(left_shape,right_shape) == same : Bool},+common: {Shapes.broadcast(left_shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>},
  +counted: Global.Counted<shape>,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,CombinePublic.tensor(~Element,left,lg,left_shape,lf,lw)),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,CombinePublic.tensor(~Element,right,rg,right_shape,rf,rw))) ->
  Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion(left_shape,shape)},
    Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
    Tensors.combine_shapes(~Element,~kernel,name,operations,same,Model.pack(~Element,left),lg,left_shape,lf,Tensors.Usable{},lw,Model.pack(~Element,right),rg,right_shape,rf,Tensors.Usable{},rw)):
  match same:
    case True{}:
      matched(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,name,operations,left_shape,right_shape,shape,
        left,right,lg,rg,lf,rf,lw,rw,ShapeLaws.same_extents(left_shape,right_shape,classified),common,left_contract,right_contract)
    case False{}:
      +workers = U32.max(lw,rw)
      %Equal.sym(Maybe<&2,List<&2,U32>>,Shapes.broadcast(left_shape,right_shape),Some{shape},common) :
        Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion(left_shape,shape)},
          Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
          Tensors.combine_broadcast_with(~Element,~kernel,name,operations,Tensors.Usable{},_,workers,Model.pack(~Element,left),lg,left_shape,lf,Model.pack(~Element,right),rg,right_shape,rf))
      common_route(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,Shapes.same_extents(left_shape,shape),left_context,right_context,name,operations,
        left_shape,right_shape,shape,left,right,lg,rg,lf,rf,workers,{==},classified,common,counted,left_contract,right_contract)

def observed(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+shape: List<&2,U32>,
  +lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32,-left: Bands.Bands(~Element),-right: Bands.Bands(~Element),
  left_observed: Model.Observation<Element,left>,right_observed: Model.Observation<Element,right>,
  +common: {Shapes.broadcast(left_shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>},+counted: Global.Counted<shape>,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,Tensors.Tensor{left,lg,left_shape,lf,Tensors.Usable{},lw}),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,Tensors.Tensor{right,rg,right_shape,rf,Tensors.Usable{},rw})) ->
  Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion(left_shape,shape)},
    Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{left,lg,left_shape,lf,Tensors.Usable{},lw},Tensors.Tensor{right,rg,right_shape,rf,Tensors.Usable{},rw})):
  Model.Observation{+l,+le} = left_observed
  Model.Observation{+r,+re} = right_observed
  %le : Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion(left_shape,shape)},
    Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{_,lg,left_shape,lf,Tensors.Usable{},lw},Tensors.Tensor{right,rg,right_shape,rf,Tensors.Usable{},rw}))
  %re : Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion(left_shape,shape)},
    Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
    Tensors.combine_with(~Element,~kernel,name,operations,CombinePublic.tensor(~Element,l,lg,left_shape,lf,lw),Tensors.Tensor{_,rg,right_shape,rf,Tensors.Usable{},rw}))
  shapes(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,Shapes.same_extents(left_shape,right_shape),left_context,right_context,name,operations,
    left_shape,right_shape,shape,l,r,lg,rg,lf,rf,lw,rw,{==},common,counted,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,l),left,le) : Global.Contract(~Element,~Left,~left_value,left_context,Tensors.Tensor{_,lg,left_shape,lf,Tensors.Usable{},lw}) left_contract,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,r),right,re) : Global.Contract(~Element,~Right,~right_value,right_context,Tensors.Tensor{_,rg,right_shape,rf,Tensors.Usable{},rw}) right_contract)

# Two Normal tensors whose extents broadcast to counted common extents
# combine into a Normal tensor that holds, at every flat index, the operation
# of the two operands read at the common extents.
def broadcast(~Element: Data,~Left: Data,~Right: Data,~operation: Element -> Element -> Element,~kernel: OperationsLaws.Kernel(~Element),~kernel_law: OperationsLaws.KernelLaw(~Element,~operation,~kernel),~left_value: Left -> U32 -> Element,~right_value: Right -> U32 -> Element,
  +left_context: Left,+right_context: Right,+name: String,+operations: F32,+left_shape: List<&2,U32>,+right_shape: List<&2,U32>,+shape: List<&2,U32>,
  +lg: Bands.Geometry,+rg: Bands.Geometry,+lf: Element,+rf: Element,+lw: U32,+rw: U32,left: Bands.Bands(~Element),right: Bands.Bands(~Element),
  +common: {Shapes.broadcast(left_shape,right_shape) == Some{shape} : Maybe<&2,List<&2,U32>>},+counted: Global.Counted<shape>,
  left_contract: Global.Contract(~Element,~Left,~left_value,left_context,Tensors.Tensor{left,lg,left_shape,lf,Tensors.Usable{},lw}),
  right_contract: Global.Contract(~Element,~Right,~right_value,right_context,Tensors.Tensor{right,rg,right_shape,rf,Tensors.Usable{},rw})) ->
  Combined(~Element,~Left,~Right,~operation,~left_value,~right_value,Broadcast.Expanded{left_context,Broadcast.expansion(left_shape,shape)},
    Broadcast.Expanded{right_context,Broadcast.expansion(right_shape,shape)},
    Tensors.combine_with(~Element,~kernel,name,operations,Tensors.Tensor{left,lg,left_shape,lf,Tensors.Usable{},lw},Tensors.Tensor{right,rg,right_shape,rf,Tensors.Usable{},rw})):
  observed(~Element,~Left,~Right,~operation,~kernel,~kernel_law,~left_value,~right_value,left_context,right_context,name,operations,left_shape,right_shape,shape,
    lg,rg,lf,rf,lw,rw,left,right,Model.observe(~Element,left),Model.observe(~Element,right),common,counted,left_contract,right_contract)
