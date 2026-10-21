# Band map: the in-place traversal of every leaf, the values it leaves, and the
# public band invariant that unary calls preserve.
import Base
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_model.bend as Model
import ../tensor_band.bend as Bands
import ../tensor_access.bend as Access
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ./tensor_operations_refinement.bend as OperationLaws
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./storage_tree_model.bend as Tree
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./traversal_map_readback.bend as MapReadback
import ../tensor_band_schedule.bend as BandSchedule
import ../machine_limits.bend as Machine
import ./storage_certificate.bend as Certificate
import ./storage_refinement.bend as StorageLaws
import ./tensor_band_each.bend as Each
import ./tensor_band_indexed.bend as BandIndexed
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_plan.bend as BandPlan
import ./traversal_array_refinement.bend as ArrayLaws
import ../tensor.bend as Tensors
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop

# Every band-local map has the common ordered in-place traversal semantics.
# Fork selection changes ownership scheduling, not the content model.
def output(~Element: Data,~Context: Data,state: Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>) -> Tree.Tree<Element>:
  MapReadback.output(~Element,~Context,state)

def refinement_values(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,count: Nat,context: Context,input: Tree.Tree<Element>) -> Tree.Tree<Element>:
  MapReadback.run(~Element,~Context,~operation,count,0,context,input)

def finished(~Element: Data,~Context: Data,+state: Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>) ->
  {Traversal.mapped_output(~Element,~Context,ArrayLaws.represent_map(~Element,~Context,state)) == Tree.pack(Element,output(~Element,~Context,state)) : Array<Element>}:
  Traversal.State{Unit{},output,position,context} = state
  {==}

def raw(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+count: Nat,+context: Context,+input: Tree.Tree<Element>) ->
  {Traversal.map_in_place(~Element,~Context,~operation,count,0,context,Tree.pack(Element,input)) == Tree.pack(Element,refinement_values(~Element,~Context,~operation,count,context,input)) : Array<Element>}:
  %Equal.sym(Traversal.State<&2,&1,Unit,Array<Element>,Context>,
    Loop.run(~Traversal.State<&2,&1,Unit,Array<Element>,Context>,~Traversal.map_step(~Element,~Context,~operation),count,Traversal.State{Unit{},Tree.pack(Element,input),0,context}),
    ArrayLaws.represent_map(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>,~ArrayLaws.model_map_step(~Element,~Context,~ArrayLaws.unindexed(~Element,~Context,~operation)),count,Traversal.State{Unit{},input,0,context})),
    ArrayLaws.map_loop_refines_model(~Element,~Context,~operation,count,Traversal.State{Unit{},input,0,context})) :
    {Traversal.mapped_output(~Element,~Context,_) == Tree.pack(Element,refinement_values(~Element,~Context,~operation,count,context,input)) : Array<Element>}
  finished(~Element,~Context,Loop.run(~Traversal.State<&2,&2,Unit,Tree.Tree<Element>,Context>,~ArrayLaws.model_map_step(~Element,~Context,~ArrayLaws.unindexed(~Element,~Context,~operation)),count,Traversal.State{Unit{},input,0,context}))

def leaf_model(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,context: Context,input: Buffer.Model<Element>) -> Buffer.Content<Element>:
  Buffer.Model{tree,+length,certificate} = input
  Buffer.Content{refinement_values(~Element,~Context,~operation,length,context,tree),length}

def refinement_leaf(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,+input: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Bands.map_leaf(~Element,~Context,~operation,context,Buffer.pack(~Element,input))) == leaf_model(~Element,~Context,~operation,context,input) : Buffer.Content<Element>}:
  Buffer.Model{+tree,+length,certificate} = input
  Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,length},
    Tree.reflect(Element,Traversal.map_in_place(~Element,~Context,~operation,length,0,context,Tree.pack(Element,tree))),refinement_values(~Element,~Context,~operation,length,context,tree),
    Equal.trans(Tree.Tree<Element>,Tree.reflect(Element,Traversal.map_in_place(~Element,~Context,~operation,length,0,context,Tree.pack(Element,tree))),
      Tree.reflect(Element,Tree.pack(Element,refinement_values(~Element,~Context,~operation,length,context,tree))),refinement_values(~Element,~Context,~operation,length,context,tree),
      Equal.cong(Array<Element>,Tree.Tree<Element>,storage => Tree.reflect(Element,storage),Traversal.map_in_place(~Element,~Context,~operation,length,0,context,Tree.pack(Element,tree)),
        Tree.pack(Element,refinement_values(~Element,~Context,~operation,length,context,tree)),raw(~Element,~Context,~operation,length,context,tree)),
      StorageLaws.reflect_packed(Element,refinement_values(~Element,~Context,~operation,length,context,tree))))

def band_model(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,context: Context,first: U32,rows: U32,input: Buffer.Model<Element>) -> Model.Content<Element>:
  Model.RowBand{leaf_model(~Element,~Context,~operation,context,input),first,rows}

def model(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,bands: Model.Model<Element>) -> Model.Content<Element>:
  Each.content_map(~Element,~Context,~band_model(~Element,~Context,~operation),context,bands)

# What a band leaf proves: its contents are the ordered map of the band.
def LeafRule(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,~leaf: @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>) -> Type:
  @+context: Context -> @+input: Buffer.Model<Element> ->
  {Buffer.contents(~Element,leaf(context,Buffer.pack(~Element,input))) == leaf_model(~Element,~Context,~operation,context,input) : Buffer.Content<Element>}

def owner_content(~Element: Data,owner: OperationLaws.Owner<&1,Array<Element>>) -> Buffer.Content<Element>:
  OperationLaws.Owner{storage,length} = owner
  Buffer.Content{Tree.reflect(Element,storage),length}

def contents_of_owner(~Element: Data,buffer: Storage.Buffer<Element>) ->
  {Buffer.contents(~Element,buffer) == owner_content(~Element,OperationLaws.observe_owner(~Element,buffer)) : Buffer.Content<Element>}:
  Storage.Buffer{storage,length,certificate} = buffer
  {==}

# A lawful mapper over a whole band is a band leaf.
def whole(~Element: Data,~Operation: Data,~operation: Operation -> Element -> Element,~mapper: @+count: Nat -> @+position: U32 -> @+context: Operation -> Storage.Buffer<Element> -> Storage.Buffer<Element>,~lawful: OperationLaws.MapLaw_context(~Element,~Operation,~operation,~mapper),+context: Operation,+input: Buffer.Model<Element>) ->
  {Buffer.contents(~Element,Bands.map_whole_context(~Element,~Operation,~mapper,context,Buffer.pack(~Element,input))) ==
    leaf_model(~Element,~Operation,~operation,context,input) : Buffer.Content<Element>}:
  Buffer.Model{+tree,+length,+certificate} = input
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,mapper(length,0,context,Storage.Buffer{Tree.pack(Element,tree),length,certificate})),
    owner_content(~Element,OperationLaws.observe_owner(~Element,mapper(length,0,context,Storage.Buffer{Tree.pack(Element,tree),length,certificate}))),
    contents_of_owner(~Element,mapper(length,0,context,Storage.Buffer{Tree.pack(Element,tree),length,certificate}))) :
    {_ == leaf_model(~Element,~Operation,~operation,context,Buffer.Model{tree,length,certificate}) : Buffer.Content<Element>}
  %Equal.sym(OperationLaws.Owner<&1,Array<Element>>,OperationLaws.observe_owner(~Element,mapper(length,0,context,Storage.Buffer{Tree.pack(Element,tree),length,certificate})),
    OperationLaws.Owner{Traversal.map_in_place(~Element,~Operation,~operation,length,0,context,Tree.pack(Element,tree)),length},
    lawful(length,0,context,tree,length,certificate)) :
    {owner_content(~Element,_) == leaf_model(~Element,~Operation,~operation,context,Buffer.Model{tree,length,certificate}) : Buffer.Content<Element>}
  refinement_leaf(~Element,~Operation,~operation,context,Buffer.Model{tree,length,certificate})

def band(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,~leaf: @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,~rule: LeafRule(~Element,~Context,~operation,~leaf),
  +context: Context,+first: U32,+rows: U32,+input: Buffer.Model<Element>) ->
  {Model.contents(~Element,Bands.map_band(~Element,~Context,~leaf,context,first,rows,Buffer.pack(~Element,input),Unit{})) ==
    band_model(~Element,~Context,~operation,context,first,rows,input) : Model.Content<Element>}:
  Equal.cong(Buffer.Content<Element>,Model.Content<Element>,input => Model.RowBand{input,first,rows},
    Buffer.contents(~Element,leaf(context,Buffer.pack(~Element,input))),leaf_model(~Element,~Context,~operation,context,input),
    rule(context,input))

# The band map on the plan of its own bands, run in parallel or not.
def map(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,~leaf: @+context: Context -> Storage.Buffer<Element> -> Storage.Buffer<Element>,~rule: LeafRule(~Element,~Context,~operation,~leaf),
  +parallel: Bool,+context: Context,+bands: Model.Model<Element>) ->
  {Model.contents(~Element,Bands.map_with(~Element,~Context,~leaf,Model.layout(~Element,bands),parallel,context,Model.pack(~Element,bands))) == model(~Element,~Context,~operation,context,bands) : Model.Content<Element>}:
  Each.mapped(~Element,~Context,~Bands.map_band(~Element,~Context,~leaf),~band_model(~Element,~Context,~operation),~band(~Element,~Context,~operation,~leaf,~rule),
    bands,parallel,context)

def refinement_scheduled_map(~Element: Data,~Operation: Data,~operation: Operation -> Element -> Element,~mapper: @+count: Nat -> @+position: U32 -> @+context: Operation -> Storage.Buffer<Element> -> Storage.Buffer<Element>,~lawful: OperationLaws.MapLaw_context(~Element,~Operation,~operation,~mapper),+scope: Operation,+operations: F32,+geometry: Bands.Geometry,+workers: U32,+bands: Model.Model<Element>) ->
  {Model.contents(~Element,BandSchedule.map_context(~Element,~Operation,~mapper,scope,geometry,workers,operations,Model.pack(~Element,bands))) ==
    model(~Element,~Operation,~operation,scope,bands) : Model.Content<Element>}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,bands)),(Model.pack(~Element,bands),Model.layout(~Element,bands)),Model.plan(~Element,bands)) :
    {Model.contents(~Element,BandSchedule.mapped_context(~Element,~Operation,~mapper,scope,geometry,workers,operations,_)) == model(~Element,~Operation,~operation,scope,bands) : Model.Content<Element>}
  map(~Element,~Operation,~operation,~Bands.map_whole_context(~Element,~Operation,~mapper),~whole(~Element,~Operation,~operation,~mapper,~lawful),
    BandSchedule.parallel(geometry,workers,8.0,operations,Model.layout(~Element,bands)),scope,bands)

def refinement_scheduled_scalar(~Element: Data,~operation: Element -> Element -> Element,+operations: F32,+geometry: Bands.Geometry,+workers: U32,+value: Element,+bands: Model.Model<Element>) ->
  {Model.contents(~Element,BandSchedule.scalar(~Element,~operation,geometry,workers,operations,value,Model.pack(~Element,bands))) ==
    model(~Element,~Element,~Bands.scalar_value(~Element,~operation),value,bands) : Model.Content<Element>}:
  %Equal.sym(Bands.Bands(~Element) & Bands.Plan,Bands.plan(~Element,Model.pack(~Element,bands)),(Model.pack(~Element,bands),Model.layout(~Element,bands)),Model.plan(~Element,bands)) :
    {Model.contents(~Element,BandSchedule.scalar_planned(~Element,~operation,geometry,workers,operations,value,_)) == model(~Element,~Element,~Bands.scalar_value(~Element,~operation),value,bands) : Model.Content<Element>}
  map(~Element,~Element,~Bands.scalar_value(~Element,~operation),~Bands.map_leaf(~Element,~Element,~Bands.scalar_value(~Element,~operation)),~refinement_leaf(~Element,~Element,~Bands.scalar_value(~Element,~operation)),BandSchedule.parallel(geometry,workers,8.0,operations,Model.layout(~Element,bands)),value,bands)

# Band maps preserve the common logical content relation. The static operation
# may be arbitrary: no floating-point algebra or reassociation is used.
type Context<-Operation: Data,-Logical: Data> is Data:
  Context{operation: Operation,logical: Logical}

def value(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,
  context: Context<Operation,Logical>,index: U32) -> Element:
  Context{operation_context,logical_context} = context
  operation(operation_context,original(logical_context,index))

def leaf_content_ready(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,+input: Buffer.Model<Element>,ready: Buffer.Sized<Element,input>) ->
  Buffer.ContentReady<Element,leaf_model(~Element,~Context,~operation,context,input)>:
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = input
  Buffer.Sized{depth_ok,room} = ready
  Buffer.ContentReady{depth,MapReadback.balanced(~Element,~Context,~operation,length,0,context,tree,depth,Buffer.balanced(~Element,input)),depth_ok,
    %Equal.sym(U32,Tree.capacity(Element,refinement_values(~Element,~Context,~operation,length,context,tree)),Tree.capacity(Element,tree),MapReadback.capacity(~Element,~Context,~operation,length,0,context,tree)) :
      {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool} room}

def band_ready(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+ready: Buffer.Sized<Element,input>) ->
  Buffer.ContentReady<Element,leaf_model(~Element,~Context,~operation,context,input)>:
  leaf_content_ready(~Element,~Context,~operation,context,input,ready)

def modeled_ready(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,bands: Model.Model<Element>,ready: Model.Ready(~Element,bands)) ->
  Model.ContentReady(~Element,model(~Element,~Context,~operation,context,bands)):
  Each.map_ready_zero(~Element,~Context,~band_model(~Element,~Context,~operation),~Context,~Each.own(~Context),~Model.always(~Element,~Context,~Model.ready_band(~Element)),
    ~band_ready(~Element,~Context,~operation),context,bands,ready)

# What the rules of one band need beside the context of the operation.
type Placed<-Context: Data> is Data:
  Placed{context: Context,geometry: Bands.Geometry}

def placed_context(~Context: Data,+scope: Placed<Context>) -> Context:
  Placed{context,geometry} = scope
  context

def placed_fields(~Element: Data,~Context: Data,+scope: Placed<Context>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Placed{context,geometry} = scope
  BandShape.band_fields(~Element,geometry)

def band_shape(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+scope: Placed<Context>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +metadata: placed_fields(~Element,~Context,scope,first,rows,Buffer.content(~Element,input))) ->
  placed_fields(~Element,~Context,scope,first,rows,leaf_model(~Element,~Context,~operation,placed_context(~Context,scope),input)):
  Placed{+context,+geometry} = scope
  Buffer.Model{tree,length,certificate} = input
  metadata

def modeled_shape(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,+geometry: Bands.Geometry,+bands: Model.Model<Element>,
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands))) -> BandShape.Contents(~Element,geometry,model(~Element,~Context,~operation,context,bands)):
  Each.map_one(~Element,~Context,~band_model(~Element,~Context,~operation),~Placed<Context>,~placed_context(~Context),~placed_fields(~Element,~Context),~placed_fields(~Element,~Context),
    ~band_shape(~Element,~Context,~operation),Placed{context,geometry},bands,metadata)

def band_layout(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,+first: U32,+rows: U32,+input: Buffer.Model<Element>) ->
  {Model.content_layout(~Element,band_model(~Element,~Context,~operation,context,first,rows,input)) == Bands.Rows{first,rows} : Bands.Plan}:
  {==}

def modeled_layout(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,bands: Model.Model<Element>) ->
  {Model.content_layout(~Element,model(~Element,~Context,~operation,context,bands)) == Model.layout(~Element,bands) : Bands.Plan}:
  Each.mapped_layout(~Element,~Context,~band_model(~Element,~Context,~operation),~band_layout(~Element,~Context,~operation),context,bands)

def modeled_well_formed(~Element: Data,~Context: Data,~operation: Context -> Element -> Element,+context: Context,+geometry: Bands.Geometry,+bands: Model.Model<Element>,
  evidence: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>) -> BandShape.WellFormed<Element,geometry,model(~Element,~Context,~operation,context,bands)>:
  BandShape.WellFormed{metadata,ready,coverage} = evidence
  BandShape.WellFormed{modeled_shape(~Element,~Context,~operation,context,geometry,bands,metadata),modeled_ready(~Element,~Context,~operation,context,bands,Model.ready(~Element,bands,ready)),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,model(~Element,~Context,~operation,context,bands)),Model.layout(~Element,bands),modeled_layout(~Element,~Context,~operation,context,bands)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
    %Equal.sym(Bands.Plan,Model.layout(~Element,bands),Model.content_layout(~Element,Model.content(~Element,bands)),Model.layout_matches_content(~Element,bands)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry)) coverage}

def scheduled_map_well_formed_context(~Element: Data,~Operation: Data,~operation: Operation -> Element -> Element,~mapper: @+count: Nat -> @+position: U32 -> @+context: Operation -> Storage.Buffer<Element> -> Storage.Buffer<Element>,~lawful: OperationLaws.MapLaw_context(~Element,~Operation,~operation,~mapper),+scope: Operation,+operations: F32,+geometry: Bands.Geometry,+workers: U32,+bands: Model.Model<Element>,
  evidence: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,BandSchedule.map_context(~Element,~Operation,~mapper,scope,geometry,workers,operations,Model.pack(~Element,bands)))>:
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandSchedule.map_context(~Element,~Operation,~mapper,scope,geometry,workers,operations,Model.pack(~Element,bands))),
    model(~Element,~Operation,~operation,scope,bands),refinement_scheduled_map(~Element,~Operation,~operation,~mapper,~lawful,scope,operations,geometry,workers,bands)) : BandShape.WellFormed<Element,geometry,_>
  modeled_well_formed(~Element,~Operation,~operation,scope,geometry,bands,evidence)

def scheduled_scalar_well_formed(~Element: Data,~operation: Element -> Element -> Element,+operations: F32,+scalar: Element,+geometry: Bands.Geometry,+workers: U32,+bands: Model.Model<Element>,
  evidence: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,BandSchedule.scalar(~Element,~operation,geometry,workers,operations,scalar,Model.pack(~Element,bands)))>:
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandSchedule.scalar(~Element,~operation,geometry,workers,operations,scalar,Model.pack(~Element,bands))),
    model(~Element,~Element,~Bands.scalar_value(~Element,~operation),scalar,bands),refinement_scheduled_scalar(~Element,~operation,operations,geometry,workers,scalar,bands)) : BandShape.WellFormed<Element,geometry,_>
  modeled_well_formed(~Element,~Element,~Bands.scalar_value(~Element,~operation),scalar,geometry,bands,evidence)

# A mapped cell stays at its position and holds the operation applied to the
# value of the same pixel.
def mapped_cell(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,
  +operation_context: Operation,+logical_context: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+channel: Nat,+row: Nat,+column: Nat,
  ready: Buffer.Sized<Element,input>,cell: BandValues.At(~Element,~Logical,~original,logical_context,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input),channel,row,column)) ->
  BandValues.At(~Element,~Context<Operation,Logical>,~value(~Element,~Operation,~Logical,~operation,~original),Context{operation_context,logical_context},geometry,first,rows,
    Buffer.content_tree(~Element,leaf_model(~Element,~Operation,~operation,operation_context,input)),Buffer.content_length(~Element,leaf_model(~Element,~Operation,~operation,operation_context,input)),channel,row,column):
  Buffer.Model{+tree,+length,+certificate} = input
  Buffer.Sized{+depth_ok,room} = ready
  Shape.Both{+inside,equal} = cell
  +position = BandValues.position(geometry,rows,channel,row,column)
  +pixel = U32.from_nat(BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(first),row),column))
  Shape.Both{inside,
    Equal.trans(Element,Tree.read(Element,refinement_values(~Element,~Operation,~operation,Buffer.length(~Element,input),operation_context,Buffer.tree(~Element,input)),U32.from_nat(position)),
      operation(operation_context,Tree.read(Element,Buffer.tree(~Element,input),U32.from_nat(position))),operation(operation_context,original(logical_context,pixel)),
      MapReadback.inside(~Element,~Operation,~operation,Buffer.length(~Element,input),0n,position,operation_context,Buffer.tree(~Element,input),Buffer.depth(~Element,input),Tree.capacity(Element,Buffer.tree(~Element,input)),
        {==},Buffer.balanced(~Element,input),depth_ok,Buffer.capacity_bound(~Element,input,depth_ok),room,inside),
      Equal.cong(Element,Element,element => operation(operation_context,element),Tree.read(Element,Buffer.tree(~Element,input),U32.from_nat(position)),original(logical_context,pixel),equal))}

# What one leaf supplies to every cell of its output.
type CellScope<-Element: Data,-Operation: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  CellScope{operation_context: Operation,logical_context: Logical,geometry: Bands.Geometry,first: U32,rows: U32,input: Buffer.Model<Element>,ready: Buffer.Sized<Element,input>,values: BandValues.Leaf(~Element,~Logical,~original,logical_context,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input))}

def cell_channels(~Element: Data,~Operation: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Operation,Logical,original>) -> Nat:
  CellScope{operation_context,logical_context,geometry,first,rows,input,ready,values} = scope
  BandValues.channels(geometry)

def cell_rows(~Element: Data,~Operation: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Operation,Logical,original>) -> Nat:
  CellScope{operation_context,logical_context,geometry,first,rows,input,ready,values} = scope
  U32.to_nat(rows)

def cell_width(~Element: Data,~Operation: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Operation,Logical,original>) -> Nat:
  CellScope{operation_context,logical_context,geometry,first,rows,input,ready,values} = scope
  BandValues.width(geometry)

def Cell(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Operation,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  CellScope{operation_context,logical_context,geometry,first,rows,input,ready,values} = scope
  BandValues.At(~Element,~Context<Operation,Logical>,~value(~Element,~Operation,~Logical,~operation,~original),Context{operation_context,logical_context},geometry,first,rows,
    Buffer.content_tree(~Element,leaf_model(~Element,~Operation,~operation,operation_context,input)),Buffer.content_length(~Element,leaf_model(~Element,~Operation,~operation,operation_context,input)),channel,row,column)

def leaf_cell(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Operation,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,cell_channels(~Element,~Operation,~Logical,~original,scope)) == True{} : Bool},+row_inside: {Nat.is_lt(row,cell_rows(~Element,~Operation,~Logical,~original,scope)) == True{} : Bool},+column_inside: {Nat.is_lt(column,cell_width(~Element,~Operation,~Logical,~original,scope)) == True{} : Bool}) ->
  Cell(~Element,~Operation,~Logical,~operation,~original,scope,channel,row,column):
  CellScope{+operation_context,+logical_context,+geometry,+first,+rows,+input,+ready,+values} = scope
  mapped_cell(~Element,~Operation,~Logical,~operation,~original,operation_context,logical_context,geometry,first,rows,input,channel,row,column,ready,
    BandValues.leaf_cell(~Element,~Logical,~original,logical_context,geometry,first,rows,Buffer.tree(~Element,input),Buffer.length(~Element,input),channel,row,column,values,channel_inside,row_inside,column_inside))

def semantics_leaf(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,
  +operation_context: Operation,+logical_context: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,input: Buffer.Model<Element>,
  ready: Buffer.Sized<Element,input>,values: BandValues.Values(~Element,~Logical,~original,logical_context,geometry,Model.RowBand{Buffer.content(~Element,input),first,rows})) ->
  BandValues.Values(~Element,~Context<Operation,Logical>,~value(~Element,~Operation,~Logical,~operation,~original),Context{operation_context,logical_context},geometry,
    Model.RowBand{leaf_model(~Element,~Operation,~operation,operation_context,input),first,rows}):
  Buffer.Model{+tree,+length,+certificate} = input
  BandValues.table(~CellScope<Element,Operation,Logical,original>,~Cell(~Element,~Operation,~Logical,~operation,~original),~cell_channels(~Element,~Operation,~Logical,~original),~cell_rows(~Element,~Operation,~Logical,~original),~cell_width(~Element,~Operation,~Logical,~original),~leaf_cell(~Element,~Operation,~Logical,~operation,~original),CellScope{operation_context,logical_context,geometry,first,rows,Buffer.Model{tree,length,certificate},ready,values})

type Valued<-Operation: Data,-Logical: Data> is Data:
  Valued{operation_context: Operation,logical_context: Logical,geometry: Bands.Geometry}

def valued_context(~Operation: Data,~Logical: Data,+scope: Valued<Operation,Logical>) -> Operation:
  Valued{operation_context,logical_context,geometry} = scope
  operation_context

def valued_before(~Element: Data,~Operation: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Valued<Operation,Logical>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Valued{operation_context,logical_context,geometry} = scope
  BandValues.band_values(~Element,~Logical,~original,logical_context,geometry)

def valued_after(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,+scope: Valued<Operation,Logical>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Valued{operation_context,logical_context,geometry} = scope
  BandValues.band_values(~Element,~Context<Operation,Logical>,~value(~Element,~Operation,~Logical,~operation,~original),Context{operation_context,logical_context},geometry)

def band_semantics(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,+scope: Valued<Operation,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+ready: Buffer.Sized<Element,input>,
  +values: valued_before(~Element,~Operation,~Logical,~original,scope,first,rows,Buffer.content(~Element,input))) ->
  valued_after(~Element,~Operation,~Logical,~operation,~original,scope,first,rows,leaf_model(~Element,~Operation,~operation,valued_context(~Operation,~Logical,scope),input)):
  Valued{+operation_context,+logical_context,+geometry} = scope
  semantics_leaf(~Element,~Operation,~Logical,~operation,~original,operation_context,logical_context,geometry,first,rows,input,ready,values)

def modeled(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~original: Logical -> U32 -> Element,
  +operation_context: Operation,+logical_context: Logical,+geometry: Bands.Geometry,+bands: Model.Model<Element>,ready: Model.Ready(~Element,bands),
  values: BandValues.Values(~Element,~Logical,~original,logical_context,geometry,Model.content(~Element,bands))) ->
  BandValues.Values(~Element,~Context<Operation,Logical>,~value(~Element,~Operation,~Logical,~operation,~original),Context{operation_context,logical_context},geometry,
    model(~Element,~Operation,~operation,operation_context,bands)):
  Each.map_ready_one(~Element,~Operation,~band_model(~Element,~Operation,~operation),~Valued<Operation,Logical>,~valued_context(~Operation,~Logical),
    ~valued_before(~Element,~Operation,~Logical,~original),~valued_after(~Element,~Operation,~Logical,~operation,~original),
    ~band_semantics(~Element,~Operation,~Logical,~operation,~original),Valued{operation_context,logical_context,geometry},bands,ready,values)

def semantics_scheduled_map(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~mapper: @+count: Nat -> @+position: U32 -> @+context: Operation -> Storage.Buffer<Element> -> Storage.Buffer<Element>,~lawful: OperationLaws.MapLaw_context(~Element,~Operation,~operation,~mapper),~original: Logical -> U32 -> Element,+scope: Operation,+operations: F32,
  +logical_context: Logical,+geometry: Bands.Geometry,+workers: U32,+bands: Model.Model<Element>,ready: Model.Ready(~Element,bands),
  values: BandValues.Values(~Element,~Logical,~original,logical_context,geometry,Model.content(~Element,bands))) ->
  BandValues.Values(~Element,~Context<Operation,Logical>,~value(~Element,~Operation,~Logical,~operation,~original),Context{scope,logical_context},geometry,
    Model.contents(~Element,BandSchedule.map_context(~Element,~Operation,~mapper,scope,geometry,workers,operations,Model.pack(~Element,bands)))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandSchedule.map_context(~Element,~Operation,~mapper,scope,geometry,workers,operations,Model.pack(~Element,bands))),
    model(~Element,~Operation,~operation,scope,bands),refinement_scheduled_map(~Element,~Operation,~operation,~mapper,~lawful,scope,operations,geometry,workers,bands)) :
    BandValues.Values(~Element,~Context<Operation,Logical>,~value(~Element,~Operation,~Logical,~operation,~original),Context{scope,logical_context},geometry,_)
  modeled(~Element,~Operation,~Logical,~operation,~original,scope,logical_context,geometry,bands,ready,values)

def semantics_scheduled_scalar(~Element: Data,~Logical: Data,~operation: Element -> Element -> Element,~original: Logical -> U32 -> Element,+operations: F32,
  +scalar: Element,+logical_context: Logical,+geometry: Bands.Geometry,+workers: U32,+bands: Model.Model<Element>,ready: Model.Ready(~Element,bands),
  values: BandValues.Values(~Element,~Logical,~original,logical_context,geometry,Model.content(~Element,bands))) ->
  BandValues.Values(~Element,~Context<Element,Logical>,~value(~Element,~Element,~Logical,~Bands.scalar_value(~Element,~operation),~original),Context{scalar,logical_context},geometry,
    Model.contents(~Element,BandSchedule.scalar(~Element,~operation,geometry,workers,operations,scalar,Model.pack(~Element,bands)))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,BandSchedule.scalar(~Element,~operation,geometry,workers,operations,scalar,Model.pack(~Element,bands))),
    model(~Element,~Element,~Bands.scalar_value(~Element,~operation),scalar,bands),refinement_scheduled_scalar(~Element,~operation,operations,geometry,workers,scalar,bands)) :
    BandValues.Values(~Element,~Context<Element,Logical>,~value(~Element,~Element,~Logical,~Bands.scalar_value(~Element,~operation),~original),Context{scalar,logical_context},geometry,_)
  modeled(~Element,~Element,~Logical,~Bands.scalar_value(~Element,~operation),~original,scalar,logical_context,geometry,bands,ready,values)

def gathered_scalar(~Element: Data,~Logical: Data,~operation: Element -> Element -> Element,~original: Logical -> U32 -> Element,+operations: F32,
  +scalar: Element,+logical_context: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+workers: U32,+bands: Model.Model<Element>,+result: Buffer.Model<Element>,
  domain: BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,result))>,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,values: BandValues.Values(~Element,~Logical,~original,logical_context,geometry,Model.content(~Element,bands)),
  +image: Nat,+channel: Nat,+row: Nat,+column: Nat,inside: BandGather.Inside(geometry,image,channel,row,column),
  depth_ok: {Nat.is_le(Buffer.depth(~Element,result),30n) == True{} : Bool}) ->
  {Buffer.value(~Element,U32.from_nat(BandGather.point(geometry,image,channel,row,column)),
    Buffer.contents(~Element,Bands.gather(~Element,geometry,BandSchedule.scalar(~Element,~operation,geometry,workers,operations,scalar,Model.pack(~Element,bands)),Buffer.pack(~Element,result)))) ==
    operation(original(logical_context,U32.from_nat(BandGather.point(geometry,image,channel,row,column))),scalar) : Element}:
  BandShape.WellFormed{metadata,ready,coverage} = well_formed
  BandGather.observed_value(~Element,~Context<Element,Logical>,~value(~Element,~Element,~Logical,~Bands.scalar_value(~Element,~operation),~original),Context{scalar,logical_context},geometry,limits,
    BandSchedule.scalar(~Element,~operation,geometry,workers,operations,scalar,Model.pack(~Element,bands)),Model.observe(~Element,BandSchedule.scalar(~Element,~operation,geometry,workers,operations,scalar,Model.pack(~Element,bands))),result,domain,
    scheduled_scalar_well_formed(~Element,~operation,operations,scalar,geometry,workers,bands,well_formed),
    semantics_scheduled_scalar(~Element,~Logical,~operation,~original,operations,scalar,logical_context,geometry,workers,bands,Model.ready(~Element,bands,ready),values),image,channel,row,column,inside,depth_ok)

# Public unary calls preserve the band invariant and the shared logical
# values. Failed inputs have their own API contracts.
# The positive-width receipt selects the unchanged band indexed schedule; the
# indexed operation also needs the gather domain, which keeps the native index
# of every pixel within the machine word.
def indexed_width_guard(geometry: Bands.Geometry) ->
  {Tensors.indexed_width_positive(geometry) == BandShape.width_positive(geometry) : Bool}:
  match geometry:
    case Bands.Geometry{batch,channels,height,width}: {==}

def indexed_selected(~Element: Data,~operation: U32 -> Element -> Element,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,bands: Bands.Bands(~Element),
  width_positive: {BandShape.width_positive(geometry) == True{} : Bool}) ->
  {Tensors.apply_indexed(~Element,~operation,operations,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}) ==
    Tensors.apply_indexed_positive(~Element,~operation,True{},operations,bands,geometry,shape,fill,workers) : Tensors.Tensor<Element>}:
  %Equal.sym(Bool,Tensors.indexed_width_positive(geometry),True{},
    Equal.trans(Bool,Tensors.indexed_width_positive(geometry),BandShape.width_positive(geometry),True{},indexed_width_guard(geometry),width_positive)) :
    {Tensors.apply_indexed_positive(~Element,~operation,_,operations,bands,geometry,shape,fill,workers) ==
      Tensors.apply_indexed_positive(~Element,~operation,True{},operations,bands,geometry,shape,fill,workers) : Tensors.Tensor<Element>}
  {==}

def WellFormed(~Element: Data,tensor: Tensors.Tensor<Element>) -> Data:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,status,workers}: BandShape.WellFormed<Element,geometry,Model.contents(~Element,bands)>

def LogicalValues(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,tensor: Tensors.Tensor<Element>) -> Data:
  match tensor:
    case Tensors.Tensor{bands,geometry,shape,fill,status,workers}: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.contents(~Element,bands))

def apply_ready(~Element: Data,~Operation: Data,~Logical: Data,~operation: Operation -> Element -> Element,~mapper: @+count: Nat -> @+position: U32 -> @+context: Operation -> Storage.Buffer<Element> -> Storage.Buffer<Element>,~lawful: OperationLaws.MapLaw_context(~Element,~Operation,~operation,~mapper),~original: Logical -> U32 -> Element,+scope: Operation,+operations: F32,
  +logical: Logical,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands))) ->
  Shape.Both<WellFormed(~Element,Tensors.apply_context(~Element,~Operation,~mapper,scope,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})),
    LogicalValues(~Element,~Context<Operation,Logical>,~value(~Element,~Operation,~Logical,~operation,~original),Context{scope,logical},
      Tensors.apply_context(~Element,~Operation,~mapper,scope,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}))>:
  BandShape.WellFormed{metadata,ready,coverage} = well_formed
  Shape.Both{scheduled_map_well_formed_context(~Element,~Operation,~operation,~mapper,~lawful,scope,operations,geometry,workers,bands,well_formed),
    semantics_scheduled_map(~Element,~Operation,~Logical,~operation,~mapper,~lawful,~original,scope,operations,logical,geometry,workers,bands,Model.ready(~Element,bands,ready),values)}

def scalar_ready(~Element: Data,~Logical: Data,~operation: Element -> Element -> Element,~original: Logical -> U32 -> Element,+operations: F32,
  +logical: Logical,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+bands: Model.Model<Element>,+scalar: Element,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands))) ->
  Shape.Both<WellFormed(~Element,Tensors.combine_scalar(~Element,~operation,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scalar)),
    LogicalValues(~Element,~Context<Element,Logical>,~value(~Element,~Element,~Logical,~Bands.scalar_value(~Element,~operation),~original),Context{scalar,logical},
      Tensors.combine_scalar(~Element,~operation,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},scalar))>:
  BandShape.WellFormed{metadata,ready,coverage} = well_formed
  Shape.Both{scheduled_scalar_well_formed(~Element,~operation,operations,scalar,geometry,workers,bands,well_formed),
    semantics_scheduled_scalar(~Element,~Logical,~operation,~original,operations,scalar,logical,geometry,workers,bands,Model.ready(~Element,bands,ready),values)}

def indexed_ready(~Element: Data,~Logical: Data,~operation: U32 -> Element -> Element,~original: Logical -> U32 -> Element,+operations: F32,
  +logical: Logical,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+shape: List<&2,U32>,+fill: Element,+workers: U32,+bands: Model.Model<Element>,
  domain: BandGather.Domain<geometry,limits,capacity>,
  +width_positive: {BandShape.width_positive(geometry) == True{} : Bool},+well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,bands)>,
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,bands))) ->
  Shape.Both<WellFormed(~Element,Tensors.apply_indexed(~Element,~operation,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})),
    LogicalValues(~Element,~Logical,~BandIndexed.value(~Element,~Logical,~operation,~original),logical,
      Tensors.apply_indexed(~Element,~operation,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}))>:
  %Equal.sym(Tensors.Tensor<Element>,Tensors.apply_indexed(~Element,~operation,operations,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
    Tensors.apply_indexed_positive(~Element,~operation,True{},operations,Model.pack(~Element,bands),geometry,shape,fill,workers),
    indexed_selected(~Element,~operation,operations,geometry,shape,fill,workers,Model.pack(~Element,bands),width_positive)) :
    Shape.Both<WellFormed(~Element,_),LogicalValues(~Element,~Logical,~BandIndexed.value(~Element,~Logical,~operation,~original),logical,_)>
  Shape.Both{BandIndexed.scheduled_well_formed(~Element,~operation,operations,geometry,workers,fill,bands,width_positive,well_formed),
    BandIndexed.scheduled_values(~Element,~Logical,~operation,~original,operations,logical,geometry,limits,capacity,workers,fill,bands,domain,width_positive,well_formed,values)}

# Sticky failure does not apply an operation or discard the band layout.
def apply_failed(~Element: Data,~mapper: @+count: Nat -> @+position: U32 -> Storage.Buffer<Element> -> Storage.Buffer<Element>,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+problem: Tensors.Problem,bands: Bands.Bands(~Element)) ->
  {Tensors.apply(~Element,~mapper,operations,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}) ==
    Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers} : Tensors.Tensor<Element>}:
  {==}

def indexed_failed(~Element: Data,~operation: U32 -> Element -> Element,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+problem: Tensors.Problem,bands: Bands.Bands(~Element)) ->
  {Tensors.apply_indexed(~Element,~operation,operations,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}) ==
    Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers} : Tensors.Tensor<Element>}:
  {==}

def scalar_failed(~Element: Data,~operation: Element -> Element -> Element,+operations: F32,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,+problem: Tensors.Problem,+scalar: Element,bands: Bands.Bands(~Element)) ->
  {Tensors.combine_scalar(~Element,~operation,operations,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers},scalar) ==
    Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers} : Tensors.Tensor<Element>}:
  {==}

