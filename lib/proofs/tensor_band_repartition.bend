# Repartition into requested row bands through the shared range copies.
import Base
import ./tensor_band_range.bend as BandRange
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_model.bend as Model
import ./tensor_band_each.bend as Each
import ../tensor_band.bend as Bands
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./nat_order.bend as Order
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_shape.bend as Shape
import ./tensor_band_count.bend as BandCount
import ./tensor_band_from_whole.bend as FromWhole
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./write_program.bend as Writes
import ../storage_buffer.bend as Storage
import ../traversal_partition.bend as Partition

# Repartition allocates each requested row band and uses the actual range
# reader. Source content is preserved between leaves; certificates may vary.
type Content<-Element: Data> is Data:
  Content{source: Model.Content<Element>,result: Model.Content<Element>}

def contents(~Element: Data,owners: Bands.Bands(~Element) & Bands.Bands(~Element)) -> Content<Element>:
  (source,result) = owners
  Content{Model.contents(~Element,source),Model.contents(~Element,result)}

def result(~Element: Data,content: Content<Element>) -> Model.Content<Element>:
  Content{source,result} = content
  result

def leaf_buffer(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,source: Model.Content<Element>) -> Buffer.Content<Element>:
  +count = Bands.local_count(geometry,rows)
  BandRange.result(~Element,BandRange.written(~Element,Bands.Range{geometry,first,rows,0},source,Buffer.Content{Tree.fresh(Element,Storage.depth_for(count),fill),U32.to_nat(count)}))

def leaf_model(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,source: Model.Content<Element>) -> Model.Content<Element>:
  Model.RowBand{leaf_buffer(~Element,geometry,fill,first,rows,source),first,rows}

# The contents of one requested band.
def band(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+source: Model.Content<Element>,+first: U32,+rows: U32) -> Buffer.Content<Element>:
  leaf_buffer(~Element,geometry,fill,first,rows,source)

def model(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Content<Element>) -> Model.Content<Element>:
  Model.build(~Element,~band(~Element,geometry,fill,source),layout)

def ranged(~Element: Data,first: U32,rows: U32,content: BandRange.Content<Element>) -> Content<Element>:
  BandRange.Content{source,output} = content
  Content{source,Model.RowBand{output,first,rows}}

def range_result(~Element: Data,first: U32,rows: U32,owners: Bands.Bands(~Element) & Storage.Buffer<Element>) -> Bands.Bands(~Element) & Bands.Bands(~Element):
  Storage.bind_pair(~Bands.Bands(~Element),~Storage.Buffer<Element>,~(Bands.Bands(~Element) & Bands.Bands(~Element)),owners,
    source => output => (source,Partition.Leaf{output,first,rows,Unit{}}))

def range_contents(~Element: Data,+first: U32,+rows: U32,owners: Bands.Bands(~Element) & Storage.Buffer<Element>) ->
  {contents(~Element,range_result(~Element,first,rows,owners)) == ranged(~Element,first,rows,BandRange.contents(~Element,owners)) : Content<Element>}:
  (source,output) = owners
  {==}

def leaf(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+source: Model.Model<Element>) ->
  {contents(~Element,Bands.repartition(~Element,geometry,fill,Bands.Rows{first,rows},Model.pack(~Element,source))) ==
    Content{Model.content(~Element,source),leaf_model(~Element,geometry,fill,first,rows,Model.content(~Element,source))} : Content<Element>}:
  +request = {Bands.Range{geometry,first,rows,0} : Bands.Range}
  +count = Bands.local_count(geometry,rows)
  Equal.trans(Content<Element>,contents(~Element,Bands.repartition(~Element,geometry,fill,Bands.Rows{first,rows},Model.pack(~Element,source))),
    ranged(~Element,first,rows,BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count)))),
    Content{Model.content(~Element,source),leaf_model(~Element,geometry,fill,first,rows,Model.content(~Element,source))},
    range_contents(~Element,first,rows,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count))),
    Equal.trans(Content<Element>,ranged(~Element,first,rows,BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count)))),
      ranged(~Element,first,rows,BandRange.written(~Element,request,Model.content(~Element,source),Buffer.contents(~Element,Bands.allocate(~Element,fill,count)))),
      Content{Model.content(~Element,source),leaf_model(~Element,geometry,fill,first,rows,Model.content(~Element,source))},
      Equal.cong(BandRange.Content<Element>,Content<Element>,content => ranged(~Element,first,rows,content),
        BandRange.contents(~Element,BandRange.run(~Element,request,Model.pack(~Element,source),Bands.allocate(~Element,fill,count))),BandRange.written(~Element,request,Model.content(~Element,source),Buffer.contents(~Element,Bands.allocate(~Element,fill,count))),
        BandRange.native_range(~Element,request,source,Bands.allocate(~Element,fill,count))),
      Equal.cong(Buffer.Content<Element>,Content<Element>,output => ranged(~Element,first,rows,BandRange.written(~Element,request,Model.content(~Element,source),output)),
        Buffer.contents(~Element,Bands.allocate(~Element,fill,count)),Buffer.Content{Tree.fresh(Element,Storage.depth_for(count),fill),U32.to_nat(count)},AllocationLaws.contents(~Element,Storage.depth_for(count),fill,U32.to_nat(count)))))

# The source and the bands beside it, as the shared build reads them.
def pair(~Element: Data,content: Content<Element>) -> Model.Content<Element> & Model.Content<Element>:
  Content{source,result} = content
  (source,result)

def record(~Element: Data,made: Model.Content<Element> & Model.Content<Element>) -> Content<Element>:
  (source,result) = made
  Content{source,result}

def seen_contents(~Element: Data,owners: Bands.Bands(~Element) & Bands.Bands(~Element)) -> {Each.seen(~Element,~Bands.Bands(~Element),~Model.Content<Element>,~Model.contents(~Element),owners) == pair(~Element,contents(~Element,owners)) : Model.Content<Element> & Model.Content<Element>}:
  (source,result) = owners
  {==}

def contents_seen(~Element: Data,owners: Bands.Bands(~Element) & Bands.Bands(~Element)) -> {contents(~Element,owners) == record(~Element,Each.seen(~Element,~Bands.Bands(~Element),~Model.Content<Element>,~Model.contents(~Element),owners)) : Content<Element>}:
  (source,result) = owners
  {==}

def filling_band(~Element: Data,context: Bands.Filling<Element>,source: Model.Content<Element>) -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>:
  Bands.Filling{geometry,fill} = context
  band(~Element,geometry,fill,source)

def band_rule(~Element: Data,+context: Bands.Filling<Element>,+first: U32,+rows: U32,+source: Model.Model<Element>) ->
  {Each.seen(~Element,~Bands.Bands(~Element),~Model.Content<Element>,~Model.contents(~Element),Bands.repartition_band(~Element,context,first,rows,Model.pack(~Element,source))) ==
    (Model.contents(~Element,Model.pack(~Element,source)),Model.RowBand{filling_band(~Element,context,Model.contents(~Element,Model.pack(~Element,source)),first,rows),first,rows}) : Model.Content<Element> & Model.Content<Element>}:
  Bands.Filling{+geometry,+fill} = context
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Model.pack(~Element,source)),Model.content(~Element,source),Model.packed_contents(~Element,source)) :
    {Each.seen(~Element,~Bands.Bands(~Element),~Model.Content<Element>,~Model.contents(~Element),Bands.repartition(~Element,geometry,fill,Bands.Rows{first,rows},Model.pack(~Element,source))) == (_,Model.RowBand{band(~Element,geometry,fill,_,first,rows),first,rows}) : Model.Content<Element> & Model.Content<Element>}
  Equal.trans(Model.Content<Element> & Model.Content<Element>,Each.seen(~Element,~Bands.Bands(~Element),~Model.Content<Element>,~Model.contents(~Element),Bands.repartition(~Element,geometry,fill,Bands.Rows{first,rows},Model.pack(~Element,source))),pair(~Element,contents(~Element,Bands.repartition(~Element,geometry,fill,Bands.Rows{first,rows},Model.pack(~Element,source)))),
    (Model.content(~Element,source),Model.RowBand{band(~Element,geometry,fill,Model.content(~Element,source),first,rows),first,rows}),
    seen_contents(~Element,Bands.repartition(~Element,geometry,fill,Bands.Rows{first,rows},Model.pack(~Element,source))),
    Equal.cong(Content<Element>,Model.Content<Element> & Model.Content<Element>,content => pair(~Element,content),contents(~Element,Bands.repartition(~Element,geometry,fill,Bands.Rows{first,rows},Model.pack(~Element,source))),
      Content{Model.content(~Element,source),leaf_model(~Element,geometry,fill,first,rows,Model.content(~Element,source))},leaf(~Element,geometry,fill,first,rows,source)))

def repartition(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>) ->
  {contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))) ==
    Content{Model.contents(~Element,Model.pack(~Element,source)),model(~Element,geometry,fill,layout,Model.contents(~Element,Model.pack(~Element,source)))} : Content<Element>}:
  Equal.trans(Content<Element>,contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))),record(~Element,Each.seen(~Element,~Bands.Bands(~Element),~Model.Content<Element>,~Model.contents(~Element),Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),
    Content{Model.contents(~Element,Model.pack(~Element,source)),model(~Element,geometry,fill,layout,Model.contents(~Element,Model.pack(~Element,source)))},
    contents_seen(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))),
    Equal.cong(Model.Content<Element> & Model.Content<Element>,Content<Element>,made => record(~Element,made),Each.seen(~Element,~Bands.Bands(~Element),~Model.Content<Element>,~Model.contents(~Element),Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))),
      (Model.contents(~Element,Model.pack(~Element,source)),model(~Element,geometry,fill,layout,Model.contents(~Element,Model.pack(~Element,source)))),
      Each.built(~Element,~Bands.Bands(~Element),~Bands.Filling<Element>,~Model.Model<Element>,~Model.pack(~Element),~Each.bands_held(~Element),~Model.Content<Element>,~Model.contents(~Element),
    ~Bands.repartition_band(~Element),~filling_band(~Element),~band_rule(~Element),layout,Bands.Filling{geometry,fill},source)))

def result_contents(~Element: Data,owners: Bands.Bands(~Element) & Bands.Bands(~Element)) ->
  {Model.contents(~Element,Bands.repartitioned(~Element,owners)) == result(~Element,contents(~Element,owners)) : Model.Content<Element>}:
  (source,result) = owners
  {==}

def repartitioned(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>) ->
  {Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))) ==
    model(~Element,geometry,fill,layout,Model.content(~Element,source)) : Model.Content<Element>}:
  Equal.trans(Model.Content<Element>,Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),
    result(~Element,contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),model(~Element,geometry,fill,layout,Model.content(~Element,source)),
    result_contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))),
    %Model.packed_contents(~Element,source) : {result(~Element,contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))) == model(~Element,geometry,fill,layout,_) : Model.Content<Element>}
    Equal.cong(Content<Element>,Model.Content<Element>,owners => result(~Element,owners),contents(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))),
      Content{Model.contents(~Element,Model.pack(~Element,source)),model(~Element,geometry,fill,layout,Model.contents(~Element,Model.pack(~Element,source)))},repartition(~Element,geometry,fill,layout,source)))

type ModelLayoutScope<-Element: Data> is Data:
  ModelLayoutScope{geometry: Bands.Geometry,fill: Element,source: Model.Content<Element>}

def model_layout_band(~Element: Data,+scope: ModelLayoutScope<Element>) -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>:
  ModelLayoutScope{geometry,fill,source} = scope
  band(~Element,geometry,fill,source)

def model_layout(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Content<Element>) ->
  {Model.content_layout(~Element,model(~Element,geometry,fill,layout,source)) == layout : Bands.Plan}:
  Model.build_layout(~Element,~ModelLayoutScope<Element>,~model_layout_band(~Element),
    ModelLayoutScope{geometry,fill,source},layout)

# Repartition outputs retain the planner's shape and allocation invariants.
# Target Fits is shared with whole conversion; range performs the real copies.
def leaf_ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+source: Model.Content<Element>,+capacity: U32,+division_capacity: U32,
  facts: FromWhole.LeafFacts<geometry,first,rows,capacity,division_capacity>) -> Model.ContentReady(~Element,leaf_model(~Element,geometry,fill,first,rows,source)):
  FromWhole.LeafFacts{allocation,contents,metadata} = facts
  +count = Bands.local_count(geometry,rows)
  Writes.content_ready(~Element,BandRange.content_writes(~Element,Bands.Range{geometry,first,rows,0},source),Buffer.tree(~Element,BandCount.model(~Element,count,fill)),U32.to_nat(count),
    Buffer.content_ready(~Element,BandCount.model(~Element,count,fill),BandCount.bounded_model(~Element,count,fill,allocation)))

type ModeledReadyScope<-Element: Data> is Data:
  ModeledReadyScope{geometry: Bands.Geometry,fill: Element,source: Model.Content<Element>,capacity: U32,division_capacity: U32}

def modeled_ready_band_2(~Element: Data,+scope: ModeledReadyScope<Element>) -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>:
  ModeledReadyScope{geometry,fill,source,capacity,division_capacity} = scope
  band(~Element,geometry,fill,source)

def modeled_ready_room(~Element: Data,+scope: ModeledReadyScope<Element>) -> FromWhole.Room:
  ModeledReadyScope{geometry,fill,source,capacity,division_capacity} = scope
  FromWhole.Room{geometry,capacity,division_capacity}

def modeled_ready_band(~Element: Data,+scope: ModeledReadyScope<Element>,+first: U32,+rows: U32,
  facts: FromWhole.fits_band(modeled_ready_room(~Element,scope),first,rows)) ->
  Model.ready_band(~Element,first,rows,modeled_ready_band_2(~Element,scope,first,rows)):
  ModeledReadyScope{+geometry,+fill,+source,+capacity,+division_capacity} = scope
  leaf_ready(~Element,geometry,fill,first,rows,source,capacity,division_capacity,facts)

def modeled_ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Content<Element>,+capacity: U32,+division_capacity: U32,
  facts: FromWhole.Fits(geometry,layout,capacity,division_capacity)) -> Model.ContentReady(~Element,model(~Element,geometry,fill,layout,source)):
  Model.build_all(~Element,~ModeledReadyScope<Element>,~modeled_ready_band_2(~Element),~FromWhole.Room,~FromWhole.fits_band,~modeled_ready_room(~Element),~Model.always(~Element,~ModeledReadyScope<Element>,~Model.ready_band(~Element)),~modeled_ready_band(~Element),
    ModeledReadyScope{geometry,fill,source,capacity,division_capacity},layout,facts)

def modeled_ready_band_fields(~Element: Data,+scope: ModeledReadyScope<Element>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  ModeledReadyScope{geometry,fill,source,capacity,division_capacity} = scope
  BandShape.band_fields(~Element,geometry)

def modeled_shape_band(~Element: Data,+scope: ModeledReadyScope<Element>,+first: U32,+rows: U32,
  facts: FromWhole.fits_band(modeled_ready_room(~Element,scope),first,rows)) ->
  modeled_ready_band_fields(~Element,scope,first,rows,modeled_ready_band_2(~Element,scope,first,rows)):
  ModeledReadyScope{+geometry,+fill,+source,+capacity,+division_capacity} = scope
  FromWhole.LeafFacts{allocation,contents,metadata} = facts
  metadata

def modeled_shape(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Content<Element>,+capacity: U32,+division_capacity: U32,
  facts: FromWhole.Fits(geometry,layout,capacity,division_capacity)) -> BandShape.Contents(~Element,geometry,model(~Element,geometry,fill,layout,source)):
  Model.build_all(~Element,~ModeledReadyScope<Element>,~modeled_ready_band_2(~Element),~FromWhole.Room,~FromWhole.fits_band,~modeled_ready_room(~Element),~modeled_ready_band_fields(~Element),~modeled_shape_band(~Element),
    ModeledReadyScope{geometry,fill,source,capacity,division_capacity},layout,facts)

def modeled_well_formed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Content<Element>,+capacity: U32,+division_capacity: U32,
  +fits: FromWhole.Fits(geometry,layout,capacity,division_capacity),coverage: BandPlan.Coverage(layout,0n,Inspection.total_rows(geometry))) -> BandShape.WellFormed<Element,geometry,model(~Element,geometry,fill,layout,source)>:
  BandShape.WellFormed{modeled_shape(~Element,geometry,fill,layout,source,capacity,division_capacity,fits),modeled_ready(~Element,geometry,fill,layout,source,capacity,division_capacity,fits),
    %Equal.sym(Bands.Plan,Model.content_layout(~Element,model(~Element,geometry,fill,layout,source)),layout,model_layout(~Element,geometry,fill,layout,source)) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry)) coverage}

def repartitioned_well_formed(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>,+capacity: U32,+division_capacity: U32,
  fits: FromWhole.Fits(geometry,layout,capacity,division_capacity),coverage: BandPlan.Coverage(layout,0n,Inspection.total_rows(geometry))) ->
  BandShape.WellFormed<Element,geometry,Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))))>:
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),
    model(~Element,geometry,fill,layout,Model.content(~Element,source)),repartitioned(~Element,geometry,fill,layout,source)) : BandShape.WellFormed<Element,geometry,_>
  modeled_well_formed(~Element,geometry,fill,layout,Model.content(~Element,source),capacity,division_capacity,fits,coverage)

def target_bounds(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+output: Buffer.Model<Element>,+limits: Machine.Limits,+capacity: U32,
  +domain: BandGather.Domain<geometry,limits,capacity>,metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,output)),ready: Buffer.Sized<Element,output>) ->
  BandRange.Target(Bands.Range{geometry,first,rows,0},Machine.capacity(limits),Tree.capacity(Element,Buffer.tree(~Element,output))):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  Buffer.Sized{depth_ok,room} = ready
  BandRange.TargetFields{Limits.capacity_fits(limits),Buffer.capacity_bound(~Element,output,depth_ok),channels_positive,width_positive,
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry),U32.to_nat(Machine.capacity(limits)),rows_fit,BandGather.rows_room(geometry,limits,capacity,domain)),
    %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool} room}

# A cell of a requested band is the cell the range copy wrote at its position.
def copied_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,source_values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  target: FromWhole.LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},+row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool}) ->
  BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,
    Buffer.content_tree(~Element,leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),Buffer.content_length(~Element,leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),channel,row,column):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  FromWhole.LeafFacts{allocation,contents,+metadata} = target
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  +count = Bands.local_count(Bands.Geometry{images,channels,height,width},rows)
  +output = BandCount.model(~Element,count,fill)
  +ready = BandCount.bounded_model(~Element,count,fill,allocation)
  Shape.Both{BandValues.position_within(Bands.Geometry{images,channels,height,width},rows,U32.to_nat(count),channel,row,column,length_exact,channel_inside,row_inside,column_inside),
    BandRange.written_point(~Element,~Logical,~original,logical,Bands.Range{Bands.Geometry{images,channels,height,width},first,rows,0},source,output,limits,U32.to_nat(count),capacity,domain,
      target_bounds(~Element,Bands.Geometry{images,channels,height,width},first,rows,output,limits,capacity,domain,metadata,ready),source_shape,metadata,ready,source_values,
      channel,row,column,channel_inside,row_inside,column_inside)}

# What one requested band supplies to every cell.
type CellScope<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  CellScope{logical: Logical,geometry: Bands.Geometry,fill: Element,first: U32,rows: U32,source: Model.Model<Element>,limits: Machine.Limits,capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>,source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,source_values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),target: FromWhole.LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>}

def cell_channels(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Logical,original>) -> Nat:
  CellScope{logical,geometry,fill,first,rows,source,limits,capacity,domain,source_shape,source_values,target} = scope
  BandValues.channels(geometry)

def cell_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Logical,original>) -> Nat:
  CellScope{logical,geometry,fill,first,rows,source,limits,capacity,domain,source_shape,source_values,target} = scope
  U32.to_nat(rows)

def cell_width(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Logical,original>) -> Nat:
  CellScope{logical,geometry,fill,first,rows,source,limits,capacity,domain,source_shape,source_values,target} = scope
  BandValues.width(geometry)

def Cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  CellScope{logical,geometry,fill,first,rows,source,limits,capacity,domain,source_shape,source_values,target} = scope
  BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,
    Buffer.content_tree(~Element,leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),Buffer.content_length(~Element,leaf_buffer(~Element,geometry,fill,first,rows,Model.content(~Element,source))),channel,row,column)

def leaf_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CellScope<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,cell_channels(~Element,~Logical,~original,scope)) == True{} : Bool},+row_inside: {Nat.is_lt(row,cell_rows(~Element,~Logical,~original,scope)) == True{} : Bool},+column_inside: {Nat.is_lt(column,cell_width(~Element,~Logical,~original,scope)) == True{} : Bool}) ->
  Cell(~Element,~Logical,~original,scope,channel,row,column):
  CellScope{+logical,+geometry,+fill,+first,+rows,+source,+limits,+capacity,+domain,+source_shape,+source_values,+target} = scope
  copied_cell(~Element,~Logical,~original,logical,geometry,fill,first,rows,source,limits,capacity,domain,source_shape,source_values,target,channel,row,column,channel_inside,row_inside,column_inside)

type ModeledValuesScope<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  ModeledValuesScope{logical: Logical,geometry: Bands.Geometry,fill: Element,source: Model.Model<Element>,limits: Machine.Limits,capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>,source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,source_values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source))}

def modeled_values_band_2(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: ModeledValuesScope<Element,Logical,original>) -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>:
  ModeledValuesScope{logical,geometry,fill,source,limits,capacity,domain,source_shape,source_values} = scope
  band(~Element,geometry,fill,Model.content(~Element,source))

def modeled_values_room(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: ModeledValuesScope<Element,Logical,original>) -> FromWhole.Room:
  ModeledValuesScope{logical,geometry,fill,source,limits,capacity,domain,source_shape,source_values} = scope
  FromWhole.Room{geometry,capacity,Machine.half_word(limits)}

def modeled_values_band_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: ModeledValuesScope<Element,Logical,original>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  ModeledValuesScope{logical,geometry,fill,source,limits,capacity,domain,source_shape,source_values} = scope
  BandValues.band_values(~Element,~Logical,~original,logical,geometry)

def modeled_values_band(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: ModeledValuesScope<Element,Logical,original>,+first: U32,+rows: U32,
  target: FromWhole.fits_band(modeled_values_room(~Element,~Logical,~original,scope),first,rows)) ->
  modeled_values_band_values(~Element,~Logical,~original,scope,first,rows,modeled_values_band_2(~Element,~Logical,~original,scope,first,rows)):
  ModeledValuesScope{+logical,+geometry,+fill,+source,+limits,+capacity,+domain,+source_shape,+source_values} = scope
  BandValues.table(~CellScope<Element,Logical,original>,~Cell(~Element,~Logical,~original),~cell_channels(~Element,~Logical,~original),~cell_rows(~Element,~Logical,~original),~cell_width(~Element,~Logical,~original),~leaf_cell(~Element,~Logical,~original),CellScope{logical,geometry,fill,first,rows,source,limits,capacity,domain,source_shape,source_values,target})

def modeled_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>,
  +source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,+source_values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  target: FromWhole.Fits(geometry,layout,capacity,Machine.half_word(limits))) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,model(~Element,geometry,fill,layout,Model.content(~Element,source))):
  Model.build_all(~Element,~ModeledValuesScope<Element,Logical,original>,~modeled_values_band_2(~Element,~Logical,~original),~FromWhole.Room,~FromWhole.fits_band,~modeled_values_room(~Element,~Logical,~original),~modeled_values_band_values(~Element,~Logical,~original),~modeled_values_band(~Element,~Logical,~original),
    ModeledValuesScope{logical,geometry,fill,source,limits,capacity,domain,source_shape,source_values},layout,target)

def repartitioned_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+source: Model.Model<Element>,
  +limits: Machine.Limits,+capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>,
  source_shape: BandShape.WellFormed<Element,geometry,Model.content(~Element,source)>,source_values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,source)),
  target: FromWhole.Fits(geometry,layout,capacity,Machine.half_word(limits))) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source))))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.repartitioned(~Element,Bands.repartition(~Element,geometry,fill,layout,Model.pack(~Element,source)))),
    model(~Element,geometry,fill,layout,Model.content(~Element,source)),repartitioned(~Element,geometry,fill,layout,source)) : BandValues.Values(~Element,~Logical,~original,logical,geometry,_)
  modeled_values(~Element,~Logical,~original,logical,geometry,fill,layout,source,limits,capacity,domain,source_shape,source_values,target)
