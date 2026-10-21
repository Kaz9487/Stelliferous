# Every gathered flat value follows the existing finite band relation. The
# coordinate receipt is produced from the bounded flat position, and each
# sample reuses the official gather refinement rather than an output oracle.
import Base
import ../tensor_band.bend as Bands
import ../tensor.bend as Tensors
import ../tensor_window.bend as Window
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./tensor_band_model.bend as Model
import ./tensor_band_values.bend as BandValues
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_from_whole.bend as FromWhole
import ./tensor_band_inspection.bend as Inspection
import ./tensor_contract.bend as Contract
import ./tensor_whole_storage.bend as WholeStorage
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_shape_count.bend as BandShapeCount
import ./tensor_band_count.bend as BandCount
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_count_refinement.bend as Count
import ./tensor_shape_refinement.bend as ShapeLaws
import ./machine_limit_bounds.bend as Limits
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./row_major_coordinates.bend as RowCoordinates
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_order.bend as Order
import ./write_program.bend as Writes

type Coordinates<-geometry: Bands.Geometry,-position: Nat> is Data:
  Coordinates{image: Nat,channel: Nat,row: Nat,column: Nat,
    inside: BandGather.Inside(geometry,image,channel,row,column),
    exact: {position == BandGather.point(geometry,image,channel,row,column) : Nat}}

def FlatCoordinates(geometry: Bands.Geometry,-position: Nat) -> Data:
  Bands.Geometry{images,channels,height,width} = geometry
  RowCoordinates.Coordinates<Nat.mul(U32.to_nat(images),U32.to_nat(channels)),U32.to_nat(height),U32.to_nat(width),position>

def split_coordinates(+geometry: Bands.Geometry,+position: Nat,
  coordinates: FlatCoordinates(geometry,position),
  +channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool}) -> Coordinates<geometry,position>:
  Bands.Geometry{+image_count,+channel_count,+row_count,+column_count} = geometry
  RowCoordinates.Coordinates{+combined,+row,+column,+combined_inside,row_inside,column_inside,+exact} = coordinates
  +image = Nat.div(combined,U32.to_nat(channel_count))
  +channel = Nat.mod(combined,U32.to_nat(channel_count))
  +split = DivisionOrder.quotient_exact(combined,U32.to_nat(channel_count),channels_positive)
  Coordinates{image,channel,row,column,
    BandGather.PointInside{RowMajor.row_inside(image,U32.to_nat(image_count),U32.to_nat(channel_count),channel,
        %split : {Nat.is_lt(_,Nat.mul(U32.to_nat(image_count),U32.to_nat(channel_count))) == True{} : Bool} combined_inside),
      DivisionOrder.remainder_small(combined,U32.to_nat(channel_count),channels_positive),row_inside,column_inside},
    Equal.trans(Nat,position,RowMajor.index(combined,U32.to_nat(row_count),U32.to_nat(column_count),row,column),BandGather.point(geometry,image,channel,row,column),exact,
      Equal.cong(Nat,Nat,value => RowMajor.index(value,U32.to_nat(row_count),U32.to_nat(column_count),row,column),combined,Nat.add(Nat.mul(image,U32.to_nat(channel_count)),channel),split))}

def coordinates(+geometry: Bands.Geometry,+position: Nat,+channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},
  +width_positive: {BandGather.width_positive(geometry) == True{} : Bool},inside: {Nat.is_lt(position,BandGather.elements(geometry)) == True{} : Bool}) -> Coordinates<geometry,position>:
  Bands.Geometry{+image_count,+channel_count,+row_count,+column_count} = geometry
  split_coordinates(geometry,position,
    RowCoordinates.locate(Nat.mul(U32.to_nat(image_count),U32.to_nat(channel_count)),U32.to_nat(row_count),U32.to_nat(column_count),position,width_positive,
      %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(image_count),U32.to_nat(channel_count)),Nat.mul(U32.to_nat(row_count),U32.to_nat(column_count))),BandGather.elements(geometry),
        RowMajor.assoc(U32.to_nat(image_count),U32.to_nat(channel_count),Nat.mul(U32.to_nat(row_count),U32.to_nat(column_count)))) : {Nat.is_lt(position,_) == True{} : Bool} inside),channels_positive)

def gathered_length(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,result: Buffer.Model<Element>) ->
  {Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))) == Buffer.length(~Element,result) : Nat}:
  Buffer.Model{+tree,+length,+certificate} = result
  Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),
    Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,Buffer.Model{tree,length,certificate}))),
    Buffer.Content{Writes.pure(Element,BandGather.refinement_writes(~Element,geometry,bands),tree),length},BandGather.gathered_writes(~Element,geometry,bands,Buffer.Model{tree,length,certificate}))

def gathered_point(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,
  +bands: Model.Model<Element>,+result: Buffer.Model<Element>,+domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>,
  +checked: {BandGather.complete(~Element,geometry,bands) == True{} : Bool},+values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),
  +position: Nat,location: Coordinates<geometry,position>,depth_ok: {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}) ->
  Contract.At(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))),position):
  Coordinates{+image,+channel,+row,+column,inside,exact} = location
  %Equal.sym(Nat,position,BandGather.point(geometry,image,channel,row,column),exact) :
    {Buffer.value(~Element,U32.from_nat(_),Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))) == original(logical,U32.from_nat(_)) : Element}
  BandGather.gathered_value(~Element,~Logical,~original,logical,geometry,limits,bands,result,domain,checked,values,image,channel,row,column,inside,depth_ok)

def gathered_range(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,
  +bands: Model.Model<Element>,+result: Buffer.Model<Element>,+domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>,
  +checked: {BandGather.complete(~Element,geometry,bands) == True{} : Bool},+values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),
  remaining: Nat,+position: Nat,+room: {Nat.is_le(Nat.add(position,remaining),BandGather.elements(geometry)) == True{} : Bool},
  +depth_ok: {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}) ->
  Members.All(Nat,{index => Contract.At(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))),index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{gathered_point(~Element,~Logical,~original,logical,geometry,limits,bands,result,domain,checked,values,position,
          coordinates(geometry,position,BandGather.domain_channels_positive(geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result)),domain),BandGather.gather_width_positive(geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result)),domain),Order.room_index(position,rest,BandGather.elements(geometry),room)),depth_ok),
        gathered_range(~Element,~Logical,~original,logical,geometry,limits,bands,result,domain,checked,values,rest,1n+position,Order.advance_room(position,rest,BandGather.elements(geometry),room),depth_ok)}

def gathered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,
  +bands: Model.Model<Element>,+result: Buffer.Model<Element>,+domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>,
  +checked: {BandGather.complete(~Element,geometry,bands) == True{} : Bool},+values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),
  +length_exact: {Buffer.length(~Element,result) == BandGather.elements(geometry) : Nat},+depth_ok: {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}) ->
  Contract.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))):
  %Equal.sym(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))),Buffer.length(~Element,result),gathered_length(~Element,geometry,bands,result)) :
    Members.All(Nat,{index => Contract.At(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))),index) : Nat -> Data},Ranges.range(_,0n))
  %Equal.sym(Nat,Buffer.length(~Element,result),BandGather.elements(geometry),length_exact) :
    Members.All(Nat,{index => Contract.At(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))),index) : Nat -> Data},Ranges.range(_,0n))
  gathered_range(~Element,~Logical,~original,logical,geometry,limits,bands,result,domain,checked,values,BandGather.elements(geometry),0n,Order.reflexive(BandGather.elements(geometry)),depth_ok)

def fresh_model(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,+result: Buffer.Model<Element>,
  allocated: {Buffer.pack(~Element,result) == Tensors.buffer(~Element,fill,Bands.elements(geometry)) : Storage.Buffer<Element>},
  coherent: Global.Coherent(shape,geometry)) -> Contract.WellFormed<Element,shape,Buffer.content(~Element,result)>:
  %Buffer.packed_contents(~Element,result) : Contract.WellFormed<Element,shape,_>
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,result)),Buffer.contents(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry))),
    Equal.cong(Storage.Buffer<Element>,Buffer.Content<Element>,Buffer.contents(~Element),Buffer.pack(~Element,result),Tensors.buffer(~Element,fill,Bands.elements(geometry)),allocated)) : Contract.WellFormed<Element,shape,_>
  %Equal.sym(U32,Bands.elements(geometry),Global.count(shape,counted),BandShapeCount.global_element_count_exact(shape,geometry,counted,coherent)) : Contract.WellFormed<Element,shape,Buffer.contents(~Element,Tensors.buffer(~Element,fill,_))>
  WholeStorage.fresh(~Element,shape,fill,counted)

def geometry_length(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,result: Buffer.Model<Element>,
  well_formed: Contract.WellFormed<Element,shape,Buffer.content(~Element,result)>,coherent: Global.Coherent(shape,geometry)) ->
  {Buffer.length(~Element,result) == BandGather.elements(geometry) : Nat}:
  Buffer.Model{tree,+length,certificate} = result
  Contract.WellFormed{length_exact,ready} = well_formed
  Equal.trans(Nat,length,ShapeLaws.elements(shape),BandGather.elements(geometry),length_exact,coherent)

def model_ready(~Element: Data,+shape: List<&2,U32>,+result: Buffer.Model<Element>,well_formed: Contract.WellFormed<Element,shape,Buffer.content(~Element,result)>) -> Buffer.Sized<Element,result>:
  Contract.WellFormed{length_exact,ready} = well_formed
  Buffer.ready_content(~Element,result,ready)

def model_depth(~Element: Data,+result: Buffer.Model<Element>,ready: Buffer.Sized<Element,result>) -> {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}:
  Buffer.Sized{depth_ok,room} = ready
  depth_ok

def output_domain(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+counted: Global.Counted<shape>,+result: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  +capacity_limits: {capacity == Machine.capacity(limits) : U32},+capacity_default: {capacity == Machine.capacity(Default.limits()) : U32},
  evidence: Count.Count<BandGather.elements(geometry),Global.count(shape,counted)>,ready: Buffer.Sized<Element,result>,
  length_exact: {Buffer.length(~Element,result) == BandGather.elements(geometry) : Nat},
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool}) ->
  BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>:
  Count.Count{exact,bounded} = evidence
  Buffer.Sized{+depth_ok,room} = ready
  BandGather.Domain{Buffer.capacity_bound(~Element,result,depth_ok),channels_positive,height_positive,width_positive,
    %capacity_limits : {Nat.is_le(BandGather.elements(geometry),U32.to_nat(_)) == True{} : Bool}
    BandCount.natural_room(BandGather.elements(geometry),Global.count(shape,counted),capacity,exact,
      %Equal.sym(U32,capacity,Machine.capacity(Default.limits()),capacity_default) : {U32.is_le(Global.count(shape,counted),_) == True{} : Bool} bounded),
    %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,result)))) == True{} : Bool} room}

def empty_gathered(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+result: Buffer.Model<Element>,
  length_exact: {Buffer.length(~Element,result) == BandGather.elements(geometry) : Nat},natural_zero: {BandGather.elements(geometry) == 0n : Nat}) ->
  Contract.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))):
  %Equal.sym(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))),0n,
    Equal.trans(Nat,Buffer.content_length(~Element,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))),Buffer.length(~Element,result),0n,
      gathered_length(~Element,geometry,bands,result),Equal.trans(Nat,Buffer.length(~Element,result),BandGather.elements(geometry),0n,length_exact,natural_zero))) :
    Members.All(Nat,{index => Contract.At(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result))),index) : Nat -> Data},Ranges.range(_,0n))
  Unit{}

def banded_selected_limits(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,+result: Buffer.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+capacity_limits: {capacity == Machine.capacity(limits) : U32},+capacity_default: {capacity == Machine.capacity(Default.limits()) : U32},
  +allocated: {Buffer.pack(~Element,result) == Tensors.buffer(~Element,fill,Bands.elements(geometry)) : Storage.Buffer<Element>},
  +coherent: Global.Coherent(shape,geometry),+metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),classification: BandShapeCount.GeometryCase<geometry>) ->
  Contract.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))):
  match classification:
    case BandShapeCount.ZeroGeometry{native_zero,natural_zero}:
      empty_gathered(~Element,~Logical,~original,logical,geometry,bands,result,
        geometry_length(~Element,geometry,shape,result,fresh_model(~Element,geometry,shape,fill,counted,result,allocated,coherent),coherent),natural_zero)
    case BandShapeCount.PositiveGeometry{+channels_positive,+height_positive,+width_positive}:
      +well_formed = fresh_model(~Element,geometry,shape,fill,counted,result,allocated,coherent)
      +ready = model_ready(~Element,shape,result,well_formed)
      +length_exact = geometry_length(~Element,geometry,shape,result,well_formed,coherent)
      +domain = output_domain(~Element,geometry,shape,counted,result,limits,capacity,capacity_limits,capacity_default,Global.geometry_evidence(shape,geometry,counted,coherent),ready,length_exact,channels_positive,height_positive,width_positive)
      gathered(~Element,~Logical,~original,logical,geometry,limits,bands,result,domain,
        BandShape.well_formed_checked(~Element,geometry,bands,capacity,
          %Equal.sym(U32,capacity,Machine.capacity(limits),capacity_limits) : {U32.is_le(_,2147483648) == True{} : Bool} Limits.capacity_fits(limits),
          %Equal.sym(U32,capacity,Machine.capacity(limits),capacity_limits) : {Nat.is_le(Inspection.total_rows(geometry),U32.to_nat(_)) == True{} : Bool}
          BandGather.rows_room(geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result)),domain),metadata),
        values,length_exact,model_depth(~Element,result,ready))

type MachineObservation is Data:
  MachineObservation{limits: Machine.Limits,capacity: U32,
    capacity_limits: {capacity == Machine.capacity(limits) : U32},
    capacity_default: {capacity == Machine.capacity(Default.limits()) : U32}}

def observe_machine() -> MachineObservation:
  MachineObservation{Default.limits(),Machine.capacity(Default.limits()),{==},{==}}

def banded_selected_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,+result: Buffer.Model<Element>,
  +allocated: {Buffer.pack(~Element,result) == Tensors.buffer(~Element,fill,Bands.elements(geometry)) : Storage.Buffer<Element>},
  +coherent: Global.Coherent(shape,geometry),+metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),+classification: BandShapeCount.GeometryCase<geometry>,observation: MachineObservation) ->
  Contract.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))):
  MachineObservation{+limits,+capacity,+capacity_limits,+capacity_default} = observation
  banded_selected_limits(~Element,~Logical,~original,logical,geometry,bands,shape,fill,counted,result,limits,capacity,capacity_limits,capacity_default,allocated,coherent,metadata,values,classification)

def banded_selected(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,+result: Buffer.Model<Element>,
  +allocated: {Buffer.pack(~Element,result) == Tensors.buffer(~Element,fill,Bands.elements(geometry)) : Storage.Buffer<Element>},
  +coherent: Global.Coherent(shape,geometry),+metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),classification: BandShapeCount.GeometryCase<geometry>) ->
  Contract.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Buffer.pack(~Element,result)))):
  banded_selected_observed(~Element,~Logical,~original,logical,geometry,bands,shape,fill,counted,result,allocated,coherent,metadata,values,classification,observe_machine())

def banded_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,
  +coherent: Global.Coherent(shape,geometry),+metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands)),allocation: Buffer.Observation<Element,Tensors.buffer(~Element,fill,Bands.elements(geometry))>) ->
  Contract.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))))):
  Buffer.Observation{+result,+allocated} = allocation
  %allocated : Contract.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),_)))
  banded_selected(~Element,~Logical,~original,logical,geometry,bands,shape,fill,counted,result,allocated,coherent,metadata,values,BandShapeCount.classify(geometry))

def banded(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,
  coherent: Global.Coherent(shape,geometry),metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands))) ->
  Contract.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Bands.gather(~Element,geometry,Model.pack(~Element,bands),Tensors.buffer(~Element,fill,Bands.elements(geometry))))):
  banded_observed(~Element,~Logical,~original,logical,geometry,bands,shape,fill,counted,coherent,metadata,values,Buffer.observe(~Element,Tensors.buffer(~Element,fill,Bands.elements(geometry))))

# The one leaf of a whole image holds each value at its own row-major index.
def whole_point(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +tree: Tree.Tree<Element>,+length: Nat,+values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},0,height,tree,length),
  +position: Nat,location: RowCoordinates.Coordinates<U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),position>) ->
  Contract.At(~Element,~Logical,~original,logical,Buffer.Content{tree,length},position):
  RowCoordinates.Coordinates{+channel,+row,+column,channel_inside,+row_inside,column_inside,exact} = location
  %Equal.sym(Nat,position,RowMajor.index(channel,U32.to_nat(height),U32.to_nat(width),row,column),exact) :
    {Tree.read(Element,tree,U32.from_nat(_)) == original(logical,U32.from_nat(_)) : Element}
  %BandValues.located_pixel(Bands.Geometry{batch,channels,height,width},0n,channel,row,column,row,row_inside,{==}) :
    {Tree.read(Element,tree,U32.from_nat(RowMajor.index(channel,U32.to_nat(height),U32.to_nat(width),row,column))) == original(logical,U32.from_nat(_)) : Element}
  BandValues.leaf_read(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},0,height,tree,length,channel,row,column,values,channel_inside,row_inside,column_inside)

def whole_range(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +tree: Tree.Tree<Element>,+length: Nat,+values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},0,height,tree,length),
  remaining: Nat,+position: Nat,+room: {Nat.is_le(Nat.add(position,remaining),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))) == True{} : Bool}) ->
  Members.All(Nat,{index => Contract.At(~Element,~Logical,~original,logical,Buffer.Content{tree,length},index) : Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      +cells = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))
      +inside = Order.room_index(position,rest,cells,room)
      +plane = DivisionOrder.right_factor_positive(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)),Order.le_lt_trans(0n,position,cells,Order.zero_le(position),inside))
      +width_positive = DivisionOrder.right_factor_positive(U32.to_nat(height),U32.to_nat(width),plane)
      Shape.Both{whole_point(~Element,~Logical,~original,logical,batch,channels,height,width,tree,length,values,position,
          RowCoordinates.locate(U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),position,width_positive,inside)),
        whole_range(~Element,~Logical,~original,logical,batch,channels,height,width,tree,length,values,rest,1n+position,Order.advance_room(position,rest,cells,room))}

def whole_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +tree: Tree.Tree<Element>,+length: Nat,+values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},0,height,tree,length),
  exact: {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width))) : Nat}) ->
  Contract.Values(~Element,~Logical,~original,logical,Buffer.Content{tree,length}):
  +cells = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))
  %Equal.sym(Nat,length,cells,exact) : Members.All(Nat,{index => Contract.At(~Element,~Logical,~original,logical,Buffer.Content{tree,length},index) : Nat -> Data},Ranges.range(_,0n))
  whole_range(~Element,~Logical,~original,logical,batch,channels,height,width,tree,length,values,cells,0n,Order.reflexive(cells))

# The leaf that holds the whole image, read through its own buffer.
def whole_buffer(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +buffer: Buffer.Model<Element>,+first: U32,+rows: U32,+first_zero: {first == 0 : U32},+rows_height: {rows == height : U32},
  fields: BandShape.Leaf(Bands.Geometry{batch,channels,height,width},first,rows,Buffer.content_length(~Element,Buffer.content(~Element,buffer))),
  values: BandValues.Values(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},Model.content(~Element,Model.Leaf{buffer,first,rows}))) ->
  Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,buffer)):
  Buffer.Model{+tree,+length,certificate} = buffer
  BandShape.Fields{local,rows_fit,length_exact} = fields
  +started = {%first_zero : BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},_,rows,tree,length) values :
    BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},0,rows,tree,length)}
  +image = {%rows_height : BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},0,_,tree,length) started :
    BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},0,height,tree,length)}
  whole_values(~Element,~Logical,~original,logical,batch,channels,height,width,tree,length,image,
    %rows_height : {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(_),U32.to_nat(width))) : Nat} length_exact)

# The values of the whole storage of one leaf.
def leaf_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+whole: Bool,+geometry: Bands.Geometry,+buffer: Buffer.Model<Element>,+first: U32,+rows: U32,
  +shape: List<&2,U32>,+fill: Element,+counted: Global.Counted<shape>,+held: {Bands.whole_image(geometry,first,rows) == whole : Bool},coherent: Global.Coherent(shape,geometry),
  metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,Model.Leaf{buffer,first,rows})>,
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,Model.Leaf{buffer,first,rows}))) ->
  Contract.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Tensors.leaf_whole_storage(~Element,whole,geometry,fill,Buffer.pack(~Element,buffer),first,rows))):
  match whole:
    case False{}: banded(~Element,~Logical,~original,logical,geometry,Model.Leaf{buffer,first,rows},shape,fill,counted,coherent,metadata,values)
    case True{}:
      Bands.Geometry{+batch,+channels,+height,+width} = geometry
      BandShape.WellFormed{fields,ready,coverage} = metadata
      %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,buffer)),Buffer.content(~Element,buffer),Buffer.packed_contents(~Element,buffer)) : Contract.Values(~Element,~Logical,~original,logical,_)
      whole_buffer(~Element,~Logical,~original,logical,batch,channels,height,width,buffer,first,rows,
        FromWhole.held_first(batch,first,rows,height,held),FromWhole.held_rows(batch,first,rows,height,held),fields,values)

def values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,+shape: List<&2,U32>,+fill: Element,
  +counted: Global.Counted<shape>,coherent: Global.Coherent(shape,geometry),metadata: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands))) ->
  Contract.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,Tensors.whole_storage(~Element,geometry,fill,Model.pack(~Element,bands)))):
  match bands:
    case Model.Fork{+left,+right}: banded(~Element,~Logical,~original,logical,geometry,Model.Fork{left,right},shape,fill,counted,coherent,metadata,values)
    case Model.Leaf{+buffer,+first,+rows}:
      leaf_values(~Element,~Logical,~original,logical,Bands.whole_image(geometry,first,rows),geometry,buffer,first,rows,shape,fill,counted,{==},coherent,metadata,values)

