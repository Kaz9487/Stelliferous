# A whole tensor is the one leaf of the whole geometry: one band of one row
# of one channel, which holds the elements in storage order. Storage that
# holds a reader's values at every index makes such a tensor of the reader's
# contract, and with_whole gives its body that leaf of any tensor of the
# contract, arrived in the whole geometry.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../traversal_partition.bend as Partition
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./tensor_contract.bend as Contract
import ./tensor_gather_values.bend as GatherValues
import ./tensor_global_shape_count.bend as Global
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_whole_storage.bend as WholeStorage
import ./tensor_whole_storage_values.bend as WholeStorageValues
import ./tensor_storage_count.bend as StorageCount
import ./tensor_band_target.bend as Target
import ./tensor_band_arrive.bend as Arrive
import ./tensor_band_count.bend as BandCount
import ../storage_buffer.bend as Storage
import ../tensor_operations.bend as Operations
import ./tensor_copy_values.bend as CopyValues
import ./tensor_operations_refinement.bend as OperationsLaws
import ./word_encoding.bend as WordEncoding

# What the one leaf supplies to every cell.
type CellScope<-Element: Data,-Context: Data,-reader: Context -> U32 -> Element> is Data:
  CellScope{context: Context,count: U32,tree: Tree.Tree<Element>,length: Nat,
    length_exact: {length == U32.to_nat(count) : Nat},values: Contract.Values(~Element,~Context,~reader,context,Buffer.Content{tree,length})}

def cell_channels(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+scope: CellScope<Element,Context,reader>) -> Nat:
  CellScope{context,count,tree,length,length_exact,values} = scope
  BandValues.channels(Bands.Geometry{1,1,1,count})

def cell_rows(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+scope: CellScope<Element,Context,reader>) -> Nat:
  U32.to_nat(1)

def cell_width(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+scope: CellScope<Element,Context,reader>) -> Nat:
  CellScope{context,count,tree,length,length_exact,values} = scope
  BandValues.width(Bands.Geometry{1,1,1,count})

def Cell(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+scope: CellScope<Element,Context,reader>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  CellScope{context,count,tree,length,length_exact,values} = scope
  BandValues.At(~Element,~Context,~reader,context,Bands.Geometry{1,1,1,count},0,1,tree,length,channel,row,column)

# The cell's position in the leaf is its row-major index in the image.
def leaf_cell(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+scope: CellScope<Element,Context,reader>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,cell_channels(~Element,~Context,~reader,scope)) == True{} : Bool},+row_inside: {Nat.is_lt(row,cell_rows(~Element,~Context,~reader,scope)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,cell_width(~Element,~Context,~reader,scope)) == True{} : Bool}) ->
  Cell(~Element,~Context,~reader,scope,channel,row,column):
  CellScope{+context,+count,+tree,+length,+length_exact,+values} = scope
  +geometry = {Bands.Geometry{1,1,1,count} : Bands.Geometry}
  +position = BandValues.position(geometry,1,channel,row,column)
  +inside = BandValues.position_within(geometry,1,length,channel,row,column,
    Equal.trans(Nat,length,U32.to_nat(count),BandValues.cell_count(geometry,1),length_exact,Equal.sym(Nat,BandValues.cell_count(geometry,1),U32.to_nat(count),Global.row_cells(count))),
    channel_inside,row_inside,column_inside)
  Shape.Both{inside,
    %Equal.sym(Nat,BandValues.pixel(geometry,channel,row,column),BandValues.point(geometry,0n,channel,row,column),BandValues.located_pixel(geometry,0n,channel,row,column,row,row_inside,{==})) :
      {Tree.read(Element,tree,U32.from_nat(position)) == reader(context,U32.from_nat(_)) : Element}
    Contract.read(~Element,~Context,~reader,context,Buffer.Content{tree,length},position,values,inside)}

def values(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+model: Buffer.Model<Element>,+shape: List<&2,U32>,+count: U32,
  accepted: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model))) ->
  BandValues.Values(~Element,~Context,~reader,context,Bands.Geometry{1,1,1,count},Model.content(~Element,Model.Leaf{model,0,1})):
  Buffer.Model{+tree,+length,certificate} = model
  Shape.Both{Contract.WellFormed{length_exact,ready},held_values} = facts
  BandValues.table(~CellScope<Element,Context,reader>,~Cell(~Element,~Context,~reader),~cell_channels(~Element,~Context,~reader),~cell_rows(~Element,~Context,~reader),~cell_width(~Element,~Context,~reader),~leaf_cell(~Element,~Context,~reader),
    CellScope{context,count,tree,length,Equal.trans(Nat,length,ShapeLaws.elements(shape),U32.to_nat(count),length_exact,Global.whole_counted(shape,count,accepted)),held_values})

# A whole tensor whose storage holds the reader's values at every index holds
# the contract of the reader.
def held_counted(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+model: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  counted: Global.Counted<shape>,+formed: Contract.WellFormed<Element,shape,Buffer.content(~Element,model)>,stored: Contract.Values(~Element,~Context,~reader,context,Buffer.content(~Element,model))) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.whole(~Element,Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers)):
  Global.Counted{+count,+accepted} = counted
  %Equal.sym(U32,Tensors.count_or_zero(shape),count,Global.whole_width(shape,count,accepted)) :
    Global.Contract(~Element,~Context,~reader,context,Tensors.Tensor{Partition.Leaf{Buffer.pack(~Element,model),0,1,Unit{}},Bands.Geometry{1,1,1,_},shape,fill,Tensors.Usable{},workers})
  Shape.Both{Shape.Both{
      Global.whole_well_formed(~Element,Buffer.pack(~Element,model),shape,count,accepted,
        %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) :
          Contract.WellFormed<Element,shape,_>
        formed),
      Shape.Both{Global.Counted{count,accepted},Global.whole_coherent(shape,count,accepted)}},
    %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,Model.Leaf{model,0,1})),Model.content(~Element,Model.Leaf{model,0,1}),
        Model.packed_contents(~Element,Model.Leaf{model,0,1})) :
      BandValues.Values(~Element,~Context,~reader,context,Bands.Geometry{1,1,1,count},_)
    values(~Element,~Context,~reader,context,model,shape,count,accepted,Shape.Both{formed,stored})}

def held(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+model: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model))) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.whole(~Element,Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers)):
  Shape.Both{+formed,stored} = facts
  held_counted(~Element,~Context,~reader,context,model,shape,fill,workers,StorageCount.counted(~Element,shape,Buffer.content(~Element,model),formed),formed,stored)

# Storage that holds the reader's values, as a whole tensor, holds the
# contract of the reader.
def stored(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,-storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  observation: Buffer.Observation<Element,storage>,facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.contents(~Element,storage))) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.whole(~Element,storage,shape,fill,Tensors.Usable{},workers)):
  Buffer.Observation{+model,+equation} = observation
  %equation : Global.Contract(~Element,~Context,~reader,context,Tensors.whole(~Element,_,shape,fill,Tensors.Usable{},workers))
  held(~Element,~Context,~reader,context,model,shape,fill,workers,
    %Buffer.packed_contents(~Element,model) : GatherValues.Facts(~Element,~Context,~reader,context,shape,_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),storage,equation) : GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.contents(~Element,_))
    facts)

# Such a whole tensor is a Ready tensor of its shape.
def held_shaped(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,+model: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model))) ->
  Global.Shaped(~Element,~Context,~reader,context,shape,Tensors.whole(~Element,Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers)):
  Global.shaped(~Element,~Context,~reader,context,shape,Tensors.whole(~Element,Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers),{==},{==},
    held(~Element,~Context,~reader,context,model,shape,fill,workers,facts))

def stored_shaped(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,-storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  observation: Buffer.Observation<Element,storage>,facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.contents(~Element,storage))) ->
  Global.Shaped(~Element,~Context,~reader,context,shape,Tensors.whole(~Element,storage,shape,fill,Tensors.Usable{},workers)):
  Buffer.Observation{+model,+equation} = observation
  %equation : Global.Shaped(~Element,~Context,~reader,context,shape,Tensors.whole(~Element,_,shape,fill,Tensors.Usable{},workers))
  held_shaped(~Element,~Context,~reader,context,model,shape,fill,workers,
    %Buffer.packed_contents(~Element,model) : GatherValues.Facts(~Element,~Context,~reader,context,shape,_)
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),storage,equation) : GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.contents(~Element,_))
    facts)

# A checked copy that returns Ready leaves its output, which holds the
# reader's values, as the whole tensor of the target shape.
def copied(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+context: Context,+name: String,+source_shape: List<&2,U32>,
  +target_shape: List<&2,U32>,+fill: Element,+workers: U32,
  facts: GatherValues.Facts(~Element,~Context,~reader,context,target_shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome))),
  accepted: {Contract.tensor_ready(~Element,Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,target_shape,fill,workers,outcome))) == True{} : Bool}) ->
  Global.Contract(~Element,~Context,~reader,context,Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,target_shape,fill,workers,outcome))):
  match outcome:
    case Operations.Accepted{(input,Storage.Buffer{storage,length,certificate})}:
      stored(~Element,~Context,~reader,context,Storage.Buffer{storage,length,certificate},target_shape,fill,workers,Buffer.observe(~Element,Storage.Buffer{storage,length,certificate}),facts)
    case Operations.Rejected{(input,output)}:
      Empty.absurd(Global.Contract(~Element,~Context,~reader,context,Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,target_shape,fill,workers,Operations.Rejected{(input,output)}))),
        WordEncoding.false_true(accepted))

# A result that, when Ready, holds the contract of the reader.
def ReadyContract(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,+context: Context,-tensor: Tensors.Tensor<Element>) -> Type:
  {Contract.tensor_ready(~Element,tensor) == True{} : Bool} -> Global.Contract(~Element,~Context,~reader,context,tensor)

# A tensor of a counted shape arrives in the one band of the whole geometry.
def whole_arrival(+shape: List<&2,U32>,+counted: Global.Counted<shape>) ->
  Target.Arrival<shape,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},Bands.unsplit(Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)})>:
  Global.Counted{+count,+accepted} = counted
  %Equal.sym(U32,Tensors.count_or_zero(shape),count,Global.whole_width(shape,count,accepted)) :
    Target.Arrival<shape,Bands.Geometry{1,1,1,_},Bands.unsplit(Bands.Geometry{1,1,1,_})>
  Target.Arrival{Global.Counted{count,accepted},Global.whole_coherent(shape,count,accepted),
    Shape.Both{BandShape.RowSpan{{==},{==}},BandPlan.LeafCoverage{{==},{==}}}}

# The body runs on storage that holds the reader's values at every index.
def stepped(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,~Result: Type,~Holds: Result -> Type,
  ~body: Storage.Buffer<Element> -> List<&2,U32> -> Element -> Tensors.Status -> U32 -> Result,
  +context: Context,-storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+workers: U32,observation: Buffer.Observation<Element,storage>,
  facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.contents(~Element,storage)),
  step: @+model: Buffer.Model<Element> -> @facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model)) ->
    Holds(body(Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers))) ->
  Holds(body(storage,shape,fill,Tensors.Usable{},workers)):
  Buffer.Observation{+input,+same} = observation
  %same : Holds(body(_,shape,fill,Tensors.Usable{},workers))
  step(input,
    %Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,input),Buffer.contents(~Element,storage),Buffer.buffer_contents(~Element,input,storage,same)) :
      GatherValues.Facts(~Element,~Context,~reader,context,shape,_)
    facts)

# The storage of arrived bands of the whole geometry holds the reader's
# values at every index.
def entered(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,~Result: Type,~Holds: Result -> Type,
  ~body: Storage.Buffer<Element> -> List<&2,U32> -> Element -> Tensors.Status -> U32 -> Result,
  +context: Context,-arrived: Bands.Bands(~Element),+shape: List<&2,U32>,+fill: Element,+workers: U32,observation: Model.Observation<Element,arrived>,
  +counted: Global.Counted<shape>,
  evidence: Arrive.Arrived(~Element,~Context,~reader,context,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},Bands.unsplit(Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)}),arrived),
  step: @+model: Buffer.Model<Element> -> @facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model)) ->
    Holds(body(Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers))) ->
  Holds(body(Tensors.whole_storage(~Element,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},fill,arrived),shape,fill,Tensors.Usable{},workers)):
  Model.Observation{+model,+equation} = observation
  Shape.Both{layout,Shape.Both{well_formed,stored}} = evidence
  +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)} : Bands.Geometry}
  +backward = Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),arrived,equation)
  +packed = Model.packed_contents(~Element,model)
  +formed = {
    %packed : BandShape.WellFormed<Element,whole,_>
    %backward : BandShape.WellFormed<Element,whole,Model.contents(~Element,_)> well_formed : BandShape.WellFormed<Element,whole,Model.content(~Element,model)>}
  +coherent = Target.arrival_coherent(shape,whole,Bands.unsplit(whole),whole_arrival(shape,counted))
  %equation : Holds(body(Tensors.whole_storage(~Element,whole,fill,_),shape,fill,Tensors.Usable{},workers))
  stepped(~Element,~Context,~reader,~Result,~Holds,~body,context,Tensors.whole_storage(~Element,whole,fill,Model.pack(~Element,model)),shape,fill,workers,
    Buffer.observe(~Element,Tensors.whole_storage(~Element,whole,fill,Model.pack(~Element,model))),
    Shape.Both{WholeStorage.well_formed(~Element,whole,model,shape,fill,counted,coherent,formed),
      WholeStorageValues.values(~Element,~Context,~reader,context,whole,model,shape,fill,counted,coherent,formed,
        %packed : BandValues.Values(~Element,~Context,~reader,context,whole,_)
        %backward : BandValues.Values(~Element,~Context,~reader,context,whole,Model.contents(~Element,_)) stored)},
    step)

# A whole tensor arrives in the whole geometry as its own storage.
def arrived_whole(~Element: Data,-storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32) ->
  {Tensors.whole_storage(~Element,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},fill,Tensors.arrive(~Element,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)},
    Bands.unsplit(Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)}),Tensors.whole(~Element,storage,shape,fill,status,workers))) == storage : Storage.Buffer<Element>}:
  +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)} : Bands.Geometry}
  %Equal.sym(Bool,Bands.same_geometry(whole,whole),True{},BandCount.same_geometry_self(whole)) :
    {Tensors.whole_storage(~Element,whole,fill,Tensors.arrive_when(~Element,_,whole,fill,Bands.unsplit(whole),whole,Partition.Leaf{storage,0,1,Unit{}})) == storage : Storage.Buffer<Element>}
  {==}

# with_whole of a whole tensor runs its body on the tensor's storage.
def with_whole_tensor(~Element: Data,~Result: Type,~body: Storage.Buffer<Element> -> List<&2,U32> -> Element -> Tensors.Status -> U32 -> Result,
  -storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,+status: Tensors.Status,+workers: U32) ->
  {Tensors.with_whole(~Element,~Result,Tensors.whole(~Element,storage,shape,fill,status,workers),body) == body(storage,shape,fill,status,workers) : Result}:
  +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)} : Bands.Geometry}
  Equal.cong(Storage.Buffer<Element>,Result,owner => body(owner,shape,fill,status,workers),
    Tensors.whole_storage(~Element,whole,fill,Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.whole(~Element,storage,shape,fill,status,workers))),storage,
    arrived_whole(~Element,storage,shape,fill,status,workers))

# with_whole gives its body the one leaf of the tensor arrived in the whole
# geometry, which holds the reader's values at every index.
def with_whole(~Element: Data,~Context: Data,~reader: Context -> U32 -> Element,~Result: Type,~Holds: Result -> Type,
  ~body: Storage.Buffer<Element> -> List<&2,U32> -> Element -> Tensors.Status -> U32 -> Result,
  +context: Context,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Context,~reader,context,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  step: @+model: Buffer.Model<Element> -> @facts: GatherValues.Facts(~Element,~Context,~reader,context,shape,Buffer.content(~Element,model)) ->
    Holds(body(Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},workers))) ->
  Holds(Tensors.with_whole(~Element,~Result,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},body)):
  Shape.Both{Shape.Both{well_formed,Shape.Both{+counted,coherent}},values} = contract
  +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)} : Bands.Geometry}
  entered(~Element,~Context,~reader,~Result,~Holds,~body,context,Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),shape,fill,workers,
    Model.observe(~Element,Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})),counted,
    Arrive.arrived(~Element,~Context,~reader,context,shape,whole,Bands.unsplit(whole),geometry,fill,Tensors.Usable{},workers,bands,whole_arrival(shape,counted),
      %Model.packed_contents(~Element,bands) : BandShape.WellFormed<Element,geometry,_> well_formed,coherent,
      %Model.packed_contents(~Element,bands) : BandValues.Values(~Element,~Context,~reader,context,geometry,_) values),
    step)

