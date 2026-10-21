# A tensor arrives in the layout an operation runs in with the plan of that
# layout, well formed bands and the values it held. Bands of the same geometry
# are aligned. Bands of another geometry are read in row-major order and cut:
# every cell of a cut reads the whole buffer at its pixel, which lies below
# the buffer's length. Whether a geometry is empty is decided here once, and
# the bands of an empty geometry hold no element.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../storage_buffer.bend as Storage
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_contract.bend as Contract
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_model.bend as Model
import ./tensor_band_values.bend as BandValues
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_shape_count.bend as BandShapeCount
import ./tensor_band_count.bend as BandCount
import ./tensor_count_refinement.bend as Count
import ./tensor_band_from_whole.bend as FromWhole
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./boolean_logic.bend as Boolean
import ../tensor_shape.bend as Shapes
import ./tensor_band_combine_leaf.bend as Leaf
import ./tensor_band_target.bend as Target
import ./tensor_whole_storage.bend as WholeStorage
import ./tensor_whole_storage_values.bend as WholeStorageValues

def counts_band(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  {Bands.local_count(geometry,rows) == 0 : U32}

def Counts(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>) -> Data:
  Model.ContentAll(~Element,~counts_band(~Element,geometry),Model.content(~Element,bands))

type WidthCountsScope is Data:
  WidthCountsScope{images: U32,channels: U32,height: U32}

def width_counts_counts_band(~Element: Data,+scope: WidthCountsScope) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  WidthCountsScope{images,channels,height} = scope
  counts_band(~Element,Bands.Geometry{images,channels,height,0})

def width_counts_band(~Element: Data,+scope: WidthCountsScope,+first: U32,+rows: U32,+buffer: Buffer.Model<Element>) ->
  width_counts_counts_band(~Element,scope,first,rows,Buffer.content(~Element,buffer)):
  WidthCountsScope{+images,+channels,+height} = scope
  BandCount.width_native(images,channels,height,rows)

def width_counts(~Element: Data,+images: U32,+channels: U32,+height: U32,bands: Model.Model<Element>) ->
  Counts(~Element,Bands.Geometry{images,channels,height,0},bands):
  Model.model_zero(~Element,~WidthCountsScope,~width_counts_counts_band(~Element),~width_counts_band(~Element),
    WidthCountsScope{images,channels,height},bands)

type ChannelCountsScope is Data:
  ChannelCountsScope{images: U32,height: U32,width: U32}

def channel_counts_counts_band(~Element: Data,+scope: ChannelCountsScope) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  ChannelCountsScope{images,height,width} = scope
  counts_band(~Element,Bands.Geometry{images,0,height,width})

def channel_counts_band(~Element: Data,+scope: ChannelCountsScope,+first: U32,+rows: U32,+buffer: Buffer.Model<Element>) ->
  channel_counts_counts_band(~Element,scope,first,rows,Buffer.content(~Element,buffer)):
  ChannelCountsScope{+images,+height,+width} = scope
  BandCount.channels_native(images,height,width,rows)

def channel_counts(~Element: Data,+images: U32,+height: U32,+width: U32,bands: Model.Model<Element>) ->
  Counts(~Element,Bands.Geometry{images,0,height,width},bands):
  Model.model_zero(~Element,~ChannelCountsScope,~channel_counts_counts_band(~Element),~channel_counts_band(~Element),
    ChannelCountsScope{images,height,width},bands)

def no_rows_leaf(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,
  total_zero: {Inspection.total_rows(geometry) == 0n : Nat},metadata: BandShape.Leaf(geometry,first,rows,length)) -> {Bands.local_count(geometry,rows) == 0 : U32}:
  Bands.Geometry{images,channels,height,width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  FromWhole.empty_count(geometry,rows,
    FromWhole.no_rows_natural(U32.to_nat(first),U32.to_nat(rows),%total_zero : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool} rows_fit))

type NoRowsCountsScope is Data:
  NoRowsCountsScope{geometry: Bands.Geometry,total_zero: {Inspection.total_rows(geometry) == 0n : Nat}}

def no_rows_counts_band_fields(~Element: Data,+scope: NoRowsCountsScope) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  NoRowsCountsScope{geometry,total_zero} = scope
  BandShape.band_fields(~Element,geometry)

def no_rows_counts_counts_band(~Element: Data,+scope: NoRowsCountsScope) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  NoRowsCountsScope{geometry,total_zero} = scope
  counts_band(~Element,geometry)

def no_rows_counts_band(~Element: Data,+scope: NoRowsCountsScope,+first: U32,+rows: U32,+buffer: Buffer.Model<Element>,
  +metadata: no_rows_counts_band_fields(~Element,scope,first,rows,Buffer.content(~Element,buffer))) ->
  no_rows_counts_counts_band(~Element,scope,first,rows,Buffer.content(~Element,buffer)):
  NoRowsCountsScope{+geometry,+total_zero} = scope
  Buffer.Model{tree,+length,certificate} = buffer
  no_rows_leaf(geometry,first,rows,length,total_zero,metadata)

def no_rows_counts(~Element: Data,+geometry: Bands.Geometry,bands: Model.Model<Element>,+total_zero: {Inspection.total_rows(geometry) == 0n : Nat},
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands))) -> Counts(~Element,geometry,bands):
  Model.model_one(~Element,~NoRowsCountsScope,~no_rows_counts_band_fields(~Element),~no_rows_counts_counts_band(~Element),~no_rows_counts_band(~Element),
    NoRowsCountsScope{geometry,total_zero},bands,metadata)

# Source metadata and Ready discharge empty local lengths. The actual empty
# binary preserves content, so its complete WF and pointwise values follow.
def zero_length(+geometry: Bands.Geometry,+rows: U32,+length: Nat,
  native_zero: {Bands.local_count(geometry,rows) == 0 : U32},counted: BandShape.Count<geometry,rows,length>) -> {length == 0n : Nat}:
  Equal.trans(Nat,length,U32.to_nat(Bands.local_count(geometry,rows)),0n,
    Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),length,BandShapeCount.count_refinement_exact(geometry,rows,length,counted)),
    Equal.cong(U32,Nat,U32.to_nat,Bands.local_count(geometry,rows),0,native_zero))

type ValuesScope<-Context: Data> is Data:
  ValuesScope{context: Context,geometry: Bands.Geometry}

def values_counts_band(~Element: Data,~Context: Data,+scope: ValuesScope<Context>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  ValuesScope{context,geometry} = scope
  counts_band(~Element,geometry)

def values_band_fields(~Element: Data,~Context: Data,+scope: ValuesScope<Context>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  ValuesScope{context,geometry} = scope
  BandShape.band_fields(~Element,geometry)

def values_band_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+scope: ValuesScope<Context>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  ValuesScope{context,geometry} = scope
  BandValues.band_values(~Element,~Context,~value,context,geometry)

def values_band(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+scope: ValuesScope<Context>,+first: U32,+rows: U32,+buffer: Buffer.Model<Element>,+ready: Buffer.Sized<Element,buffer>,
  +counts: values_counts_band(~Element,~Context,scope,first,rows,Buffer.content(~Element,buffer)),+metadata: values_band_fields(~Element,~Context,scope,first,rows,Buffer.content(~Element,buffer))) ->
  values_band_values(~Element,~Context,~value,scope,first,rows,Buffer.content(~Element,buffer)):
  ValuesScope{+context,+geometry} = scope
  Buffer.Model{+tree,+length,+certificate} = buffer
  +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
  +metadata = metadata
  BandValues.empty_leaf(~Element,~Context,~value,context,geometry,first,rows,tree,length,
    BandValues.empty_count(geometry,first,rows,length,metadata,zero_length(geometry,rows,length,counts,BandShapeCount.counted(~Element,geometry,first,rows,input,metadata,ready))))

# Bands that each hold no element hold any values.
def empty_band_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,bands: Model.Model<Element>,counts: Counts(~Element,geometry,bands),
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands)) ->
  BandValues.Values(~Element,~Context,~value,context,geometry,Model.content(~Element,bands)):
  Model.ready_two(~Element,~ValuesScope<Context>,~values_counts_band(~Element,~Context),~values_band_fields(~Element,~Context),~values_band_values(~Element,~Context,~value),~values_band(~Element,~Context,~value),
    ValuesScope{context,geometry},bands,ready,counts,metadata)

# A plan whose bands lie in the batch fits a geometry without elements.
def empty_fits(+geometry: Bands.Geometry,+layout: Bands.Plan,+capacity: U32,+division_capacity: U32,empty: BandShapeCount.EmptyGeometry<geometry>,
  spans: BandShape.Spans(geometry,layout)) -> FromWhole.Fits(geometry,layout,capacity,division_capacity):
  match geometry empty:
    case Bands.Geometry{+images,+channels,+height,+width} BandShapeCount.Width{+zero}:
      %Equal.sym(U32,width,0,zero) : FromWhole.Fits(Bands.Geometry{images,channels,height,_},layout,capacity,division_capacity)
      FromWhole.width(images,channels,height,layout,capacity,division_capacity,spans)
    case Bands.Geometry{+images,+channels,+height,+width} BandShapeCount.Channels{+zero}:
      %Equal.sym(U32,channels,0,zero) : FromWhole.Fits(Bands.Geometry{images,_,height,width},layout,capacity,division_capacity)
      FromWhole.channels(images,height,width,layout,capacity,division_capacity,spans)
    case Bands.Geometry{images,channels,height,width} BandShapeCount.Rows{zero}: FromWhole.no_rows(geometry,layout,capacity,division_capacity,zero,spans)

# Every band of an empty geometry holds no element.
def empty_counts(~Element: Data,+geometry: Bands.Geometry,+bands: Model.Model<Element>,empty: BandShapeCount.EmptyGeometry<geometry>,
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands))) -> Counts(~Element,geometry,bands):
  match geometry empty:
    case Bands.Geometry{+images,+channels,+height,+width} BandShapeCount.Width{zero}:
      %Equal.sym(U32,width,0,zero) : Counts(~Element,Bands.Geometry{images,channels,height,_},bands)
      width_counts(~Element,images,channels,height,bands)
    case Bands.Geometry{+images,+channels,+height,+width} BandShapeCount.Channels{zero}:
      %Equal.sym(U32,channels,0,zero) : Counts(~Element,Bands.Geometry{images,_,height,width},bands)
      channel_counts(~Element,images,height,width,bands)
    case Bands.Geometry{images,channels,height,width} BandShapeCount.Rows{zero}: no_rows_counts(~Element,geometry,bands,zero,metadata)

type Dimensions<-geometry: Bands.Geometry> is Data:
  Dimensions{channels: {BandGather.channels_positive(geometry) == True{} : Bool},height: {BandGather.height_positive(geometry) == True{} : Bool},width: {BandGather.width_positive(geometry) == True{} : Bool}}

def nonempty_dimensions(+geometry: Bands.Geometry,nonempty: {Shapes.has_zero(BandCount.extents(geometry)) == False{} : Bool}) -> Dimensions<geometry>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  +remaining = Boolean.or_false_right(U32.is_zero(images),U32.is_zero(channels) || (U32.is_zero(height) || (U32.is_zero(width) || False{})),nonempty)
  +channels_nonzero = Boolean.or_false_left(U32.is_zero(channels),U32.is_zero(height) || (U32.is_zero(width) || False{}),remaining)
  +remaining = Boolean.or_false_right(U32.is_zero(channels),U32.is_zero(height) || (U32.is_zero(width) || False{}),remaining)
  +height_nonzero = Boolean.or_false_left(U32.is_zero(height),U32.is_zero(width) || False{},remaining)
  +remaining = Boolean.or_false_right(U32.is_zero(height),U32.is_zero(width) || False{},remaining)
  Dimensions{Arithmetic.word_nonzero_positive(channels,channels_nonzero),Arithmetic.word_nonzero_positive(height,height_nonzero),
    Arithmetic.word_nonzero_positive(width,Boolean.or_false_left(U32.is_zero(width),False{},remaining))}

def DimensionChoice(+geometry: Bands.Geometry) -> Data:
  Either<&2,&2,{Shapes.has_zero(BandCount.extents(geometry)) == True{} : Bool},Dimensions<geometry>>

def dimensions_selected(empty: Bool,+geometry: Bands.Geometry,classified: {Shapes.has_zero(BandCount.extents(geometry)) == empty : Bool}) -> DimensionChoice(geometry):
  match empty:
    case True{}: Inl{classified}
    case False{}: Inr{nonempty_dimensions(geometry,classified)}

def dimension_choice(+geometry: Bands.Geometry) -> DimensionChoice(geometry):
  dimensions_selected(Shapes.has_zero(BandCount.extents(geometry)),geometry,{==})

def metadata_width_positive(geometry: Bands.Geometry,evidence: {BandGather.width_positive(geometry) == True{} : Bool}) -> {BandShape.width_positive(geometry) == True{} : Bool}:
  Bands.Geometry{images,channels,height,width} = geometry
  evidence

def Arrived(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+layout: Bands.Plan,-bands: Bands.Bands(~Element)) -> Data:
  Shape.Both<{Model.content_layout(~Element,Model.contents(~Element,bands)) == layout : Bands.Plan},
    Shape.Both<BandShape.WellFormed<Element,geometry,Model.contents(~Element,bands)>,BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.contents(~Element,bands))>>

# A row of the batch makes the height positive.
def row_height_positive(+geometry: Bands.Geometry,+row: Nat,row_inside: {Nat.is_lt(row,Inspection.total_rows(geometry)) == True{} : Bool}) ->
  {BandGather.height_positive(geometry) == True{} : Bool}:
  Bands.Geometry{+images,channels,+height,width} = geometry
  DivisionOrder.right_factor_positive(U32.to_nat(images),U32.to_nat(height),
    Order.le_lt_trans(0n,row,Nat.mul(U32.to_nat(images),U32.to_nat(height)),Order.zero_le(row),row_inside))

# A whole owner that holds the counted elements bounds every element of the
# geometry by its own room.
def count_domain(~Element: Data,+geometry: Bands.Geometry,+input: Buffer.Model<Element>,+count: U32,
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  length_exact: {U32.to_nat(count) == Buffer.length(~Element,input) : Nat},ready: Buffer.Sized<Element,input>,
  counted: Count.Count<ShapeLaws.elements(BandCount.extents(geometry)),count>) ->
  BandGather.Domain<geometry,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,input))>:
  match ready counted:
    case Buffer.Sized{+depth_ok,+room} Count.Count{+exact,+within}:
      +expected_length = Equal.trans(Nat,BandGather.elements(geometry),ShapeLaws.elements(BandCount.extents(geometry)),Buffer.length(~Element,input),
        Equal.sym(Nat,ShapeLaws.elements(BandCount.extents(geometry)),BandGather.elements(geometry),BandCount.domain_elements(geometry)),
        Equal.trans(Nat,ShapeLaws.elements(BandCount.extents(geometry)),U32.to_nat(count),Buffer.length(~Element,input),exact,length_exact))
      FromWhole.owned(~Element,geometry,Default.limits(),Machine.capacity(Default.limits()),input,expected_length,Buffer.Sized{depth_ok,room},
        BandCount.evidence(geometry,count,channels_positive,height_positive,width_positive,Count.Count{exact,within}))

def whole_length(~Element: Data,+shape: List<&2,U32>,input: Buffer.Model<Element>,
  exact: {Buffer.content_length(~Element,Buffer.content(~Element,input)) == ShapeLaws.elements(shape) : Nat}) ->
  {Buffer.length(~Element,input) == ShapeLaws.elements(shape) : Nat}:
  Buffer.Model{tree,length,certificate} = input
  exact

# The same bound for a whole owner of a shape coherent with the geometry.
def whole_domain(~Element: Data,+geometry: Bands.Geometry,+shape: List<&2,U32>,+input: Buffer.Model<Element>,+counted: Global.Counted<shape>,
  coherent: Global.Coherent(shape,geometry),
  channels_positive: {BandGather.channels_positive(geometry) == True{} : Bool},height_positive: {BandGather.height_positive(geometry) == True{} : Bool},width_positive: {BandGather.width_positive(geometry) == True{} : Bool},
  length_exact: {Buffer.length(~Element,input) == ShapeLaws.elements(shape) : Nat},ready: Buffer.Sized<Element,input>) ->
  BandGather.Domain<geometry,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,input))>:
  +evidence = Global.evidence(shape,counted)
  +geometry_evidence = Global.geometry_evidence(shape,geometry,counted,coherent)
  +expected_length = Equal.trans(Nat,U32.to_nat(Global.count(shape,counted)),ShapeLaws.elements(shape),Buffer.length(~Element,input),
    Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(Global.count(shape,counted)),Count.count_exact(ShapeLaws.elements(shape),Global.count(shape,counted),evidence)),
    Equal.sym(Nat,Buffer.length(~Element,input),ShapeLaws.elements(shape),length_exact))
  count_domain(~Element,geometry,input,Global.count(shape,counted),channels_positive,height_positive,width_positive,expected_length,ready,
    %Equal.sym(Nat,ShapeLaws.elements(BandCount.extents(geometry)),BandGather.elements(geometry),BandCount.domain_elements(geometry)) : Count.Count<_,Global.count(shape,counted)>
    geometry_evidence)

# One leaf cut from a whole buffer: what each cell needs to read the buffer's
# values.
type CutScope<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  CutScope{logical: Logical,geometry: Bands.Geometry,first: U32,rows: U32,input: Tree.Tree<Element>,length: Nat,whole: Buffer.Model<Element>,
    metadata: BandShape.Leaf(geometry,first,rows,length),
    cut: BandValues.Leaf(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),Buffer.tree(~Element,whole),geometry,first,rows,input,length),
    values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,whole)),
    expected_length: {BandGather.elements(geometry) == Buffer.length(~Element,whole) : Nat}}

def cut_channels(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CutScope<Element,Logical,original>) -> Nat:
  CutScope{logical,geometry,first,rows,input,length,whole,metadata,cut,values,expected_length} = scope
  BandValues.channels(geometry)

def cut_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CutScope<Element,Logical,original>) -> Nat:
  CutScope{logical,geometry,first,rows,input,length,whole,metadata,cut,values,expected_length} = scope
  U32.to_nat(rows)

def cut_width(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CutScope<Element,Logical,original>) -> Nat:
  CutScope{logical,geometry,first,rows,input,length,whole,metadata,cut,values,expected_length} = scope
  BandValues.width(geometry)

def CutCell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CutScope<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  CutScope{logical,geometry,first,rows,input,length,whole,metadata,cut,values,expected_length} = scope
  BandValues.At(~Element,~Logical,~original,logical,geometry,first,rows,input,length,channel,row,column)

def cut_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: CutScope<Element,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,cut_channels(~Element,~Logical,~original,scope)) == True{} : Bool},+row_inside: {Nat.is_lt(row,cut_rows(~Element,~Logical,~original,scope)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,cut_width(~Element,~Logical,~original,scope)) == True{} : Bool}) ->
  CutCell(~Element,~Logical,~original,scope,channel,row,column):
  CutScope{+logical,+geometry,+first,+rows,+input,+length,Buffer.Model{+tree,+whole_length,+certificate},+metadata,+cut,+values,+expected_length} = scope
  +global = Nat.add(U32.to_nat(first),row)
  +pixel = BandValues.pixel(geometry,channel,global,column)
  +position = BandValues.position(geometry,rows,channel,row,column)
  +global_inside = BandGather.cell_row_inside(geometry,first,rows,length,row,metadata,row_inside)
  +below = BandGather.pixel_below(geometry,channel,global,column,channel_inside,global_inside,column_inside,row_height_positive(geometry,global,global_inside))
  +cell = BandValues.leaf_cell(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),tree,geometry,first,rows,input,length,channel,row,column,cut,channel_inside,row_inside,column_inside)
  +stored = Shape.second({Nat.is_lt(position,length) == True{} : Bool},{Tree.read(Element,input,U32.from_nat(position)) == BandValues.stored_value(~Element,tree,U32.from_nat(pixel)) : Element},cell)
  Shape.Both{Shape.first({Nat.is_lt(position,length) == True{} : Bool},{Tree.read(Element,input,U32.from_nat(position)) == BandValues.stored_value(~Element,tree,U32.from_nat(pixel)) : Element},cell),
    Equal.trans(Element,Tree.read(Element,input,U32.from_nat(position)),Tree.read(Element,tree,U32.from_nat(pixel)),original(logical,U32.from_nat(pixel)),stored,
      Contract.read(~Element,~Logical,~original,logical,Buffer.Content{tree,whole_length},pixel,values,%expected_length : {Nat.is_lt(pixel,_) == True{} : Bool} below))}

def cut_leaf(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,
  +whole: Buffer.Model<Element>,metadata: BandShape.Leaf(geometry,first,rows,length),
  cut: BandValues.Leaf(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),Buffer.tree(~Element,whole),geometry,first,rows,input,length),
  values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,whole)),expected_length: {BandGather.elements(geometry) == Buffer.length(~Element,whole) : Nat}) ->
  BandValues.Leaf(~Element,~Logical,~original,logical,geometry,first,rows,input,length):
  BandValues.table(~CutScope<Element,Logical,original>,~CutCell(~Element,~Logical,~original),~cut_channels(~Element,~Logical,~original),~cut_rows(~Element,~Logical,~original),~cut_width(~Element,~Logical,~original),~cut_cell(~Element,~Logical,~original),
    CutScope{logical,geometry,first,rows,input,length,whole,metadata,cut,values,expected_length})

# Bands cut from a whole buffer that holds the logical values below its length
# hold them.
def cut_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,+whole: Buffer.Model<Element>,
  +expected_length: {BandGather.elements(geometry) == Buffer.length(~Element,whole) : Nat},+values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,whole)),
  bands: Model.Content<Element>,metadata: BandShape.Contents(~Element,geometry,bands),
  cut: BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),Buffer.tree(~Element,whole),geometry,bands)) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,bands):
  match bands:
    case Model.RowBand{Buffer.Content{+input,+length},+first,+rows}:
      cut_leaf(~Element,~Logical,~original,logical,geometry,first,rows,input,length,whole,metadata,cut,values,expected_length)
    case Model.Joined{left,right}:
      Shape.Both{left_metadata,right_metadata} = metadata
      Shape.Both{left_cut,right_cut} = cut
      Shape.Both{cut_values(~Element,~Logical,~original,logical,geometry,whole,expected_length,values,left,left_metadata,left_cut),
        cut_values(~Element,~Logical,~original,logical,geometry,whole,expected_length,values,right,right_metadata,right_cut)}

# Bands of an empty geometry hold any values.
def empty_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,-bands: Bands.Bands(~Element),
  observation: Model.Observation<Element,bands>,empty: BandShapeCount.EmptyGeometry<geometry>,shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,bands)>) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.contents(~Element,bands)):
  Model.Observation{+model,+equation} = observation
  BandShape.WellFormed{metadata,ready,coverage} = shape
  +contents = Model.observed_contents(~Element,model,bands,equation)
  +meta = {%contents : BandShape.Contents(~Element,geometry,_) metadata : BandShape.Contents(~Element,geometry,Model.content(~Element,model))}
  +prepared = {%contents : Model.ContentReady(~Element,_) ready : Model.ContentReady(~Element,Model.content(~Element,model))}
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,bands),Model.content(~Element,model),contents) : BandValues.Values(~Element,~Logical,~original,logical,geometry,_)
  empty_band_values(~Element,~Logical,~original,logical,geometry,model,empty_counts(~Element,geometry,model,empty,meta),meta,Model.ready(~Element,model,prepared))

def empty_values(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: Bands.Geometry,bands: Bands.Bands(~Element),
  empty: BandShapeCount.EmptyGeometry<geometry>,shape: BandShape.WellFormed<Element,geometry,Model.contents(~Element,bands)>) ->
  BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.contents(~Element,bands)):
  empty_observed(~Element,~Logical,~original,logical,geometry,bands,Model.observe(~Element,bands),empty,shape)

# Bands of the geometry are aligned to the plan.
def aligned(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,+fill: Element,
  +model: Model.Model<Element>,+target: Target.Arrival<shape,geometry,layout>,+well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),choice: DimensionChoice(geometry)) ->
  Arrived(~Element,~Logical,~original,logical,geometry,layout,Bands.align(~Element,geometry,fill,layout,Model.pack(~Element,model))):
  match choice:
    case Inr{dimensions}:
      Dimensions{channels_positive,height_positive,width_positive} = dimensions
      +domain = Global.domain(shape,geometry,Target.arrival_counted(shape,geometry,layout,target),Target.arrival_coherent(shape,geometry,layout,target),
        channels_positive,height_positive,width_positive)
      +fits = FromWhole.planned(geometry,Default.limits(),Machine.capacity(Default.limits()),layout,Target.arrival_spans(shape,geometry,layout,target),domain)
      Shape.Both{Leaf.aligned_layout(~Element,geometry,fill,layout,model),Shape.Both{
        Leaf.aligned_well_formed(~Element,geometry,fill,layout,model,Machine.capacity(Default.limits()),Machine.half_word(Default.limits()),well_formed,fits,Target.arrival_covered(shape,geometry,layout,target)),
        Leaf.aligned_values(~Element,~Logical,~original,logical,geometry,fill,layout,model,Default.limits(),Machine.capacity(Default.limits()),domain,well_formed,values,fits)}}
    case Inl{empty}:
      +classified = BandShapeCount.empty_geometry_classified(geometry,empty)
      +formed = Leaf.aligned_well_formed(~Element,geometry,fill,layout,model,Machine.capacity(Default.limits()),Machine.half_word(Default.limits()),well_formed,
        empty_fits(geometry,layout,Machine.capacity(Default.limits()),Machine.half_word(Default.limits()),classified,Target.arrival_spans(shape,geometry,layout,target)),
        Target.arrival_covered(shape,geometry,layout,target))
      Shape.Both{Leaf.aligned_layout(~Element,geometry,fill,layout,model),Shape.Both{formed,
        empty_values(~Element,~Logical,~original,logical,geometry,Bands.align(~Element,geometry,fill,layout,Model.pack(~Element,model)),classified,formed)}}

# The plan of the operand that offered the layout fits the whole buffer.
def cut_fits(~Element: Data,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,+input: Buffer.Model<Element>,+target: Target.Arrival<shape,geometry,layout>,
  length_exact: {Buffer.length(~Element,input) == ShapeLaws.elements(shape) : Nat},ready: Buffer.Sized<Element,input>,choice: DimensionChoice(geometry)) ->
  FromWhole.Fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),Machine.half_word(Default.limits())):
  match choice:
    case Inr{dimensions}:
      Dimensions{channels_positive,height_positive,width_positive} = dimensions
      FromWhole.planned(geometry,Default.limits(),Tree.capacity(Element,Buffer.tree(~Element,input)),layout,Target.arrival_spans(shape,geometry,layout,target),
        whole_domain(~Element,geometry,shape,input,Target.arrival_counted(shape,geometry,layout,target),Target.arrival_coherent(shape,geometry,layout,target),
          channels_positive,height_positive,width_positive,length_exact,ready))
    case Inl{empty}:
      empty_fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),Machine.half_word(Default.limits()),BandShapeCount.empty_geometry_classified(geometry,empty),
        Target.arrival_spans(shape,geometry,layout,target))

# A whole buffer of the shape cut by the plan.
def cut_buffer(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,+fill: Element,
  +input: Buffer.Model<Element>,+target: Target.Arrival<shape,geometry,layout>,+length_exact: {Buffer.length(~Element,input) == ShapeLaws.elements(shape) : Nat},+ready: Buffer.Sized<Element,input>,
  values: Contract.Values(~Element,~Logical,~original,logical,Buffer.content(~Element,input)),choice: DimensionChoice(geometry)) ->
  Arrived(~Element,~Logical,~original,logical,geometry,layout,Bands.from_whole(~Element,geometry,fill,layout,Buffer.pack(~Element,input))):
  +expected_length = Equal.trans(Nat,BandGather.elements(geometry),ShapeLaws.elements(shape),Buffer.length(~Element,input),
    Equal.sym(Nat,ShapeLaws.elements(shape),BandGather.elements(geometry),Target.arrival_coherent(shape,geometry,layout,target)),
    Equal.sym(Nat,Buffer.length(~Element,input),ShapeLaws.elements(shape),length_exact))
  +owned = {FromWhole.Owned{expected_length,ready} : FromWhole.Owned<Element,geometry,input>}
  +fits = cut_fits(~Element,shape,geometry,layout,input,target,length_exact,ready,choice)
  +bands = Model.contents(~Element,Bands.from_whole(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))
  +layout_exact = FromWhole.from_whole_layout(~Element,geometry,fill,layout,input)
  +metadata = FromWhole.from_whole_shape(~Element,geometry,fill,layout,input,Machine.half_word(Default.limits()),owned,fits)
  Shape.Both{layout_exact,Shape.Both{
    BandShape.WellFormed{metadata,FromWhole.from_whole_ready(~Element,geometry,fill,layout,input,Machine.half_word(Default.limits()),owned,fits),
      %Equal.sym(Bands.Plan,Model.content_layout(~Element,bands),layout,layout_exact) : BandPlan.Coverage(_,0n,Inspection.total_rows(geometry))
      Target.arrival_covered(shape,geometry,layout,target)},
    cut_values(~Element,~Logical,~original,logical,geometry,input,expected_length,values,bands,metadata,
      FromWhole.semantics_from_whole(~Element,geometry,fill,layout,input,Machine.half_word(Default.limits()),owned,fits))}}

def cut_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,+fill: Element,
  -buffer: Storage.Buffer<Element>,observation: Buffer.Observation<Element,buffer>,+target: Target.Arrival<shape,geometry,layout>,
  formed: Contract.WellFormed<Element,shape,Buffer.contents(~Element,buffer)>,values: Contract.Values(~Element,~Logical,~original,logical,Buffer.contents(~Element,buffer)),
  choice: DimensionChoice(geometry)) ->
  Arrived(~Element,~Logical,~original,logical,geometry,layout,Bands.from_whole(~Element,geometry,fill,layout,buffer)):
  Buffer.Observation{+input,+equation} = observation
  Contract.WellFormed{length_exact,content_ready} = formed
  +same_contents = Equal.sym(Buffer.Content<Element>,Buffer.content(~Element,input),Buffer.contents(~Element,buffer),Buffer.buffer_contents(~Element,input,buffer,equation))
  %equation : Arrived(~Element,~Logical,~original,logical,geometry,layout,Bands.from_whole(~Element,geometry,fill,layout,_))
  cut_buffer(~Element,~Logical,~original,logical,shape,geometry,layout,fill,input,target,
    whole_length(~Element,shape,input,%same_contents : {Buffer.content_length(~Element,_) == ShapeLaws.elements(shape) : Nat} length_exact),
    Buffer.ready_content(~Element,input,%same_contents : Buffer.ContentReady<Element,_> content_ready),
    %same_contents : Contract.Values(~Element,~Logical,~original,logical,_) values,choice)

# Bands of another geometry are read in row-major order and cut by the plan.
def cut(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,+other: Bands.Geometry,+fill: Element,
  +model: Model.Model<Element>,+target: Target.Arrival<shape,geometry,layout>,+well_formed: BandShape.WellFormed<Element,other,Model.content(~Element,model)>,+coherent: Global.Coherent(shape,other),
  values: BandValues.Values(~Element,~Logical,~original,logical,other,Model.content(~Element,model)),choice: DimensionChoice(geometry)) ->
  Arrived(~Element,~Logical,~original,logical,geometry,layout,Bands.from_whole(~Element,geometry,fill,layout,Tensors.whole_storage(~Element,other,fill,Model.pack(~Element,model)))):
  +counted = Target.arrival_counted(shape,geometry,layout,target)
  cut_observed(~Element,~Logical,~original,logical,shape,geometry,layout,fill,Tensors.whole_storage(~Element,other,fill,Model.pack(~Element,model)),
    Buffer.observe(~Element,Tensors.whole_storage(~Element,other,fill,Model.pack(~Element,model))),target,
    WholeStorage.well_formed(~Element,other,model,shape,fill,counted,coherent,well_formed),
    WholeStorageValues.values(~Element,~Logical,~original,logical,other,model,shape,fill,counted,coherent,well_formed,values),choice)

def arrived_when(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,same: Bool,+logical: Logical,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,
  +other: Bands.Geometry,+fill: Element,+model: Model.Model<Element>,compared: {Bands.same_geometry(geometry,other) == same : Bool},+target: Target.Arrival<shape,geometry,layout>,
  well_formed: BandShape.WellFormed<Element,other,Model.content(~Element,model)>,coherent: Global.Coherent(shape,other),
  values: BandValues.Values(~Element,~Logical,~original,logical,other,Model.content(~Element,model))) ->
  Arrived(~Element,~Logical,~original,logical,geometry,layout,Tensors.arrive_when(~Element,same,geometry,fill,layout,other,Model.pack(~Element,model))):
  match same:
    case True{}:
      +equal = Equal.sym(Bands.Geometry,geometry,other,BandCount.same_geometry_equal(geometry,other,compared))
      aligned(~Element,~Logical,~original,logical,shape,geometry,layout,fill,model,target,
        %equal : BandShape.WellFormed<Element,_,Model.content(~Element,model)> well_formed,
        %equal : BandValues.Values(~Element,~Logical,~original,logical,_,Model.content(~Element,model)) values,dimension_choice(geometry))
    case False{}: cut(~Element,~Logical,~original,logical,shape,geometry,layout,other,fill,model,target,well_formed,coherent,values,dimension_choice(geometry))

# A tensor of the shape arrives in a layout whose bands lie in the images of
# the geometry and cover its rows.
def arrived(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+shape: List<&2,U32>,+geometry: Bands.Geometry,+layout: Bands.Plan,
  +other: Bands.Geometry,+fill: Element,-status: Tensors.Status,-workers: U32,+model: Model.Model<Element>,+target: Target.Arrival<shape,geometry,layout>,
  well_formed: BandShape.WellFormed<Element,other,Model.content(~Element,model)>,coherent: Global.Coherent(shape,other),
  values: BandValues.Values(~Element,~Logical,~original,logical,other,Model.content(~Element,model))) ->
  Arrived(~Element,~Logical,~original,logical,geometry,layout,Tensors.arrive(~Element,geometry,layout,Tensors.Tensor{Model.pack(~Element,model),other,shape,fill,status,workers})):
  arrived_when(~Element,~Logical,~original,Bands.same_geometry(geometry,other),logical,shape,geometry,layout,other,fill,model,{==},target,well_formed,coherent,values)
