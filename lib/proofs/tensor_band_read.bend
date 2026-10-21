# One reader for every band tensor. An index of the geometry names an image,
# a channel, a row and a column in NCHW order, the row of the batch picks the
# band that covers it, and that band stores the cell at its position. Bands
# that hold the values of any reader hold them where this reader reads.
import Base
import ../tensor_band.bend as Bands
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_band_values.bend as BandValues
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_shape_refinement.bend as BandShape
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./natural_addition.bend as Addition
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic

# The first row of the bands.
def start(~Element: Data,content: Model.Content<Element>) -> Nat:
  match content:
    case Model.RowBand{buffer,first,rows}: U32.to_nat(first)
    case Model.Joined{left,right}: start(~Element,left)

# The cell of the bands at a channel, a row of the batch and a column: the
# band whose rows hold the row, read at its position there.
def cell(~Element: Data,content: Model.Content<Element>,+geometry: Bands.Geometry,+channel: Nat,+row: Nat,+column: Nat) -> Element:
  match content:
    case Model.RowBand{Buffer.Content{tree,length},first,rows}:
      Tree.read(Element,tree,U32.from_nat(BandValues.position(geometry,rows,channel,Nat.sub(row,U32.to_nat(first)),column)))
    case Model.Joined{left,+right}:
      Bool.pick(Element,Nat.is_lt(row,start(~Element,right)),cell(~Element,left,geometry,channel,row,column),cell(~Element,right,geometry,channel,row,column))

# The element of the bands at an index of the geometry.
def read(~Element: Data,content: Model.Content<Element>,geometry: Bands.Geometry,index: U32) -> Element:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  +position = U32.to_nat(index)
  +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +plane = Nat.div(position,spatial)
  +offset = Nat.mod(position,spatial)
  cell(~Element,content,Bands.Geometry{images,channels,height,width},Nat.mod(plane,U32.to_nat(channels)),
    Nat.add(Nat.mul(Nat.div(plane,U32.to_nat(channels)),U32.to_nat(height)),Nat.div(offset,U32.to_nat(width))),Nat.mod(offset,U32.to_nat(width)))

# Bands that cover a run of rows start at its first row.
def start_covered(~Element: Data,content: Model.Content<Element>,+first: Nat,+rows: Nat,
  coverage: BandPlan.Coverage(Model.content_layout(~Element,content),first,rows)) -> {start(~Element,content) == first : Nat}:
  match content:
    case Model.RowBand{buffer,native_first,native_rows}:
      BandPlan.LeafCoverage{first_equal,rows_equal} = coverage
      first_equal
    case Model.Joined{left,right}:
      BandPlan.ForkCoverage{left_rows,right_rows,rows_equal,left_coverage,right_coverage} = coverage
      start_covered(~Element,left,first,left_rows,left_coverage)

def Read(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+content: Model.Content<Element>,
  +channel: Nat,+row: Nat,+column: Nat) -> Data:
  {cell(~Element,content,geometry,channel,row,column) == value(context,U32.from_nat(BandValues.pixel(geometry,channel,row,column))) : Element}

# A row of two joined bands lies in the left one before the first row of the
# right one, and in the right one from there.
def covered_branch(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,
  +left: Model.Content<Element>,+right: Model.Content<Element>,+first: Nat,+left_rows: Nat,+right_rows: Nat,+position: Nat,+channel: Nat,+column: Nat,choose_left: Bool,
  compared: {Nat.is_lt(Nat.add(first,position),start(~Element,right)) == choose_left : Bool},+choice: {Nat.is_lt(position,left_rows) == choose_left : Bool},
  inside: {Nat.is_lt(position,Nat.add(left_rows,right_rows)) == True{} : Bool},
  left_case: {Nat.is_lt(position,left_rows) == True{} : Bool} -> Read(~Element,~Context,~value,context,geometry,left,channel,Nat.add(first,position),column),
  right_case: {Nat.is_lt(Nat.sub(position,left_rows),right_rows) == True{} : Bool} ->
    Read(~Element,~Context,~value,context,geometry,right,channel,Nat.add(Nat.add(first,left_rows),Nat.sub(position,left_rows)),column)) ->
  Read(~Element,~Context,~value,context,geometry,Model.Joined{left,right},channel,Nat.add(first,position),column):
  match choose_left:
    case True{}:
      %Equal.sym(Bool,Nat.is_lt(Nat.add(first,position),start(~Element,right)),True{},compared) :
        {Bool.pick(Element,_,cell(~Element,left,geometry,channel,Nat.add(first,position),column),cell(~Element,right,geometry,channel,Nat.add(first,position),column)) ==
          value(context,U32.from_nat(BandValues.pixel(geometry,channel,Nat.add(first,position),column))) : Element}
      left_case(choice)
    case False{}:
      +above = Order.ge_true(position,left_rows,Order.comparison_not_less(Nat.cmp(position,left_rows),choice))
      +restored = Equal.trans(Nat,Nat.add(Nat.add(first,left_rows),Nat.sub(position,left_rows)),Nat.add(first,Nat.add(left_rows,Nat.sub(position,left_rows))),Nat.add(first,position),
        Addition.associative(first,left_rows,Nat.sub(position,left_rows)),Equal.cong(Nat,Nat,extent => Nat.add(first,extent),Nat.add(left_rows,Nat.sub(position,left_rows)),position,Order.restore_shift(left_rows,position,above)))
      %Equal.sym(Bool,Nat.is_lt(Nat.add(first,position),start(~Element,right)),False{},compared) :
        {Bool.pick(Element,_,cell(~Element,left,geometry,channel,Nat.add(first,position),column),cell(~Element,right,geometry,channel,Nat.add(first,position),column)) ==
          value(context,U32.from_nat(BandValues.pixel(geometry,channel,Nat.add(first,position),column))) : Element}
      %restored : Read(~Element,~Context,~value,context,geometry,right,channel,_,column)
      right_case(Order.sub_lt(position,left_rows,right_rows,above,inside))

# A row the bands cover lies in one of their bands, and the cell there holds
# the reader's value at its pixel.
def covered_cell(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,content: Model.Content<Element>,
  +first: Nat,+rows: Nat,+position: Nat,+channel: Nat,+column: Nat,coverage: BandPlan.Coverage(Model.content_layout(~Element,content),first,rows),
  values: BandValues.Values(~Element,~Context,~value,context,geometry,content),inside: {Nat.is_lt(position,rows) == True{} : Bool},
  +channel_inside: {Nat.is_lt(channel,BandValues.channels(geometry)) == True{} : Bool},+column_inside: {Nat.is_lt(column,BandValues.width(geometry)) == True{} : Bool}) ->
  Read(~Element,~Context,~value,context,geometry,content,channel,Nat.add(first,position),column):
  match content:
    case Model.RowBand{Buffer.Content{+tree,+length},+native_first,+native_rows}:
      BandPlan.LeafCoverage{first_equal,rows_equal} = coverage
      %first_equal : Read(~Element,~Context,~value,context,geometry,Model.RowBand{Buffer.Content{tree,length},native_first,native_rows},channel,Nat.add(_,position),column)
      %Equal.sym(Nat,Nat.sub(Nat.add(U32.to_nat(native_first),position),U32.to_nat(native_first)),position,Order.subtract_shift(U32.to_nat(native_first),position)) :
        {Tree.read(Element,tree,U32.from_nat(BandValues.position(geometry,native_rows,channel,_,column))) ==
          value(context,U32.from_nat(BandValues.pixel(geometry,channel,Nat.add(U32.to_nat(native_first),position),column))) : Element}
      BandValues.leaf_read(~Element,~Context,~value,context,geometry,native_first,native_rows,tree,length,channel,position,column,values,channel_inside,
        %Equal.sym(Nat,U32.to_nat(native_rows),rows,rows_equal) : {Nat.is_lt(position,_) == True{} : Bool} inside,column_inside)
    case Model.Joined{+left,+right}:
      BandPlan.ForkCoverage{+left_rows,+right_rows,rows_equal,left_coverage,+right_coverage} = coverage
      Shape.Both{left_values,right_values} = values
      +start_exact = start_covered(~Element,right,Nat.add(first,left_rows),right_rows,right_coverage)
      +compared = Equal.trans(Bool,Nat.is_lt(Nat.add(first,position),start(~Element,right)),Nat.is_lt(Nat.add(first,position),Nat.add(first,left_rows)),Nat.is_lt(position,left_rows),
        Equal.cong(Nat,Bool,bound => Nat.is_lt(Nat.add(first,position),bound),start(~Element,right),Nat.add(first,left_rows),start_exact),Order.lt_offset(first,position,left_rows))
      covered_branch(~Element,~Context,~value,context,geometry,left,right,first,left_rows,right_rows,position,channel,column,Nat.is_lt(position,left_rows),compared,{==},
        %rows_equal : {Nat.is_lt(position,_) == True{} : Bool} inside,
        in_left => covered_cell(~Element,~Context,~value,context,geometry,left,first,left_rows,position,channel,column,left_coverage,left_values,in_left,channel_inside,column_inside),
        in_right => covered_cell(~Element,~Context,~value,context,geometry,right,Nat.add(first,left_rows),right_rows,Nat.sub(position,left_rows),channel,column,right_coverage,right_values,
          in_right,channel_inside,column_inside))

# Bands that cover the rows of the batch and hold a reader's values hold, at
# every index of the geometry, the reader's value there.
def read_values(~Element: Data,~Context: Data,~value: Context -> U32 -> Element,+context: Context,+geometry: Bands.Geometry,+content: Model.Content<Element>,
  coverage: BandPlan.Coverage(Model.content_layout(~Element,content),0n,Inspection.total_rows(geometry)),
  values: BandValues.Values(~Element,~Context,~value,context,geometry,content),
  +index: U32,+inside: {Nat.is_lt(U32.to_nat(index),BandGather.elements(geometry)) == True{} : Bool}) ->
  {read(~Element,content,geometry,index) == value(context,index) : Element}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +images_nat = U32.to_nat(images)
  +channels_nat = U32.to_nat(channels)
  +height_nat = U32.to_nat(height)
  +width_nat = U32.to_nat(width)
  +position = U32.to_nat(index)
  +spatial = Nat.mul(height_nat,width_nat)
  +plane = Nat.div(position,spatial)
  +offset = Nat.mod(position,spatial)
  +image = Nat.div(plane,channels_nat)
  +channel = Nat.mod(plane,channels_nat)
  +row = Nat.div(offset,width_nat)
  +column = Nat.mod(offset,width_nat)
  +global = Nat.add(Nat.mul(image,height_nat),row)
  +positive = Order.le_lt_trans(0n,position,Nat.mul(images_nat,Nat.mul(channels_nat,spatial)),Order.zero_le(position),inside)
  +frame_positive = DivisionOrder.right_factor_positive(images_nat,Nat.mul(channels_nat,spatial),positive)
  +spatial_positive = DivisionOrder.right_factor_positive(channels_nat,spatial,frame_positive)
  +channels_positive = RowMajor.left_factor_positive(channels_nat,spatial,frame_positive)
  +width_positive = DivisionOrder.right_factor_positive(height_nat,width_nat,spatial_positive)
  +plane_inside = DivisionOrder.quotient_below(position,Nat.mul(images_nat,channels_nat),spatial,spatial_positive,
    %Equal.sym(Nat,Nat.mul(Nat.mul(images_nat,channels_nat),spatial),Nat.mul(images_nat,Nat.mul(channels_nat,spatial)),RowMajor.assoc(images_nat,channels_nat,spatial)) :
      {Nat.is_lt(position,_) == True{} : Bool} inside)
  +image_inside = DivisionOrder.quotient_below(plane,images_nat,channels_nat,channels_positive,plane_inside)
  +row_inside = DivisionOrder.quotient_below(offset,height_nat,width_nat,width_positive,DivisionOrder.remainder_small(position,spatial,spatial_positive))
  +held = covered_cell(~Element,~Context,~value,context,geometry,content,0n,Nat.mul(images_nat,height_nat),global,channel,column,coverage,values,
    RowMajor.radix(image,images_nat,row,height_nat,image_inside,row_inside),DivisionOrder.remainder_small(plane,channels_nat,channels_positive),
    DivisionOrder.remainder_small(offset,width_nat,width_positive))
  +point_exact = {
    %DivisionOrder.quotient_exact(plane,channels_nat,channels_positive) : {Nat.add(Nat.mul(_,spatial),Nat.add(Nat.mul(row,width_nat),column)) == position : Nat}
    %DivisionOrder.quotient_exact(offset,width_nat,width_positive) : {Nat.add(Nat.mul(plane,spatial),_) == position : Nat}
    Equal.sym(Nat,position,Nat.add(Nat.mul(plane,spatial),offset),DivisionOrder.quotient_exact(position,spatial,spatial_positive))
    : {BandValues.point(geometry,image,channel,row,column) == position : Nat}}
  +pixel_exact = Equal.trans(Nat,BandValues.pixel(geometry,channel,global,column),BandValues.point(geometry,image,channel,row,column),position,
    BandValues.located_pixel(geometry,image,channel,row,column,global,row_inside,{==}),point_exact)
  Equal.trans(Element,read(~Element,content,geometry,index),value(context,U32.from_nat(BandValues.pixel(geometry,channel,global,column))),value(context,index),held,
    Equal.cong(U32,Element,word => value(context,word),U32.from_nat(BandValues.pixel(geometry,channel,global,column)),index,
      Equal.trans(U32,U32.from_nat(BandValues.pixel(geometry,channel,global,column)),U32.from_nat(position),index,
        Equal.cong(Nat,U32,U32.from_nat,BandValues.pixel(geometry,channel,global,column),position,pixel_exact),Arithmetic.native_roundtrip(index))))

# ---- Another reader that agrees on the indices of the geometry ----

# Two readers that agree at every index of a geometry where their contexts
# are related.
def Agrees(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,
  ~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data) -> Type:
  @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> @+related: Related(before,after,geometry) -> @+index: U32 ->
    @+inside: {Nat.is_lt(U32.to_nat(index),BandGather.elements(geometry)) == True{} : Bool} -> {before_value(before,index) == after_value(after,index) : Element}

# One band of well formed bands that hold the first reader's values.
type CellScope<-Element: Data,-Before: Data,-After: Data,-before_value: Before -> U32 -> Element,-Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data> is Data:
  CellScope{before: Before,after: After,geometry: Bands.Geometry,related: Related(before,after,geometry),first: U32,rows: U32,tree: Tree.Tree<Element>,length: Nat,
    metadata: BandShape.Leaf(geometry,first,rows,length),leaf: BandValues.Leaf(~Element,~Before,~before_value,before,geometry,first,rows,tree,length),
    capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},room: {Nat.is_le(BandGather.elements(geometry),U32.to_nat(capacity)) == True{} : Bool}}

def cell_channels(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data,
  +scope: CellScope<Element,Before,After,before_value,Related>) -> Nat:
  CellScope{before,after,geometry,related,first,rows,tree,length,metadata,leaf,capacity,capacity_ok,room} = scope
  BandValues.channels(geometry)

def cell_rows(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data,
  +scope: CellScope<Element,Before,After,before_value,Related>) -> Nat:
  CellScope{before,after,geometry,related,first,rows,tree,length,metadata,leaf,capacity,capacity_ok,room} = scope
  U32.to_nat(rows)

def cell_width(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data,
  +scope: CellScope<Element,Before,After,before_value,Related>) -> Nat:
  CellScope{before,after,geometry,related,first,rows,tree,length,metadata,leaf,capacity,capacity_ok,room} = scope
  BandValues.width(geometry)

def AfterCell(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,
  ~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data,+scope: CellScope<Element,Before,After,before_value,Related>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  CellScope{before,after,+geometry,related,first,+rows,tree,length,metadata,leaf,capacity,capacity_ok,room} = scope
  BandValues.At(~Element,~After,~after_value,after,geometry,first,rows,tree,length,channel,row,column)

# The cell holds the first reader's value at its pixel, a pixel of the
# geometry, where the second reader agrees.
def after_cell(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,
  ~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data,~agrees: Agrees(~Element,~Before,~After,~before_value,~after_value,~Related),
  +scope: CellScope<Element,Before,After,before_value,Related>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,cell_channels(~Element,~Before,~After,~before_value,~Related,scope)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,cell_rows(~Element,~Before,~After,~before_value,~Related,scope)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,cell_width(~Element,~Before,~After,~before_value,~Related,scope)) == True{} : Bool}) ->
  AfterCell(~Element,~Before,~After,~before_value,~after_value,~Related,scope,channel,row,column):
  CellScope{+before,+after,+geometry,+related,+first,+rows,+tree,+length,+metadata,+leaf,+capacity,+capacity_ok,+room} = scope
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +global = Nat.add(U32.to_nat(first),row)
  +pixel = BandValues.pixel(geometry,channel,global,column)
  +stored = BandValues.leaf_read(~Element,~Before,~before_value,before,geometry,first,rows,tree,length,channel,row,column,leaf,channel_inside,row_inside,column_inside)
  +global_inside = BandGather.cell_row_inside(geometry,first,rows,length,row,metadata,row_inside)
  +height_positive = DivisionOrder.right_factor_positive(U32.to_nat(images),U32.to_nat(height),
    Order.le_lt_trans(0n,global,Nat.mul(U32.to_nat(images),U32.to_nat(height)),Order.zero_le(global),global_inside))
  +pixel_inside = BandGather.pixel_below(geometry,channel,global,column,channel_inside,global_inside,column_inside,height_positive)
  +roundtrip = Bounds.roundtrip(pixel,capacity,capacity_ok,
    Order.transitive(pixel,BandGather.elements(geometry),U32.to_nat(capacity),Bounds.strict_le(pixel,BandGather.elements(geometry),pixel_inside),room))
  +agreed = agrees(before,after,geometry,related,U32.from_nat(pixel),
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(pixel)),pixel,roundtrip) : {Nat.is_lt(_,BandGather.elements(geometry)) == True{} : Bool} pixel_inside)
  Shape.Both{BandValues.position_within(geometry,rows,length,channel,row,column,BandValues.leaf_length(geometry,first,rows,length,metadata),channel_inside,row_inside,column_inside),
    Equal.trans(Element,Tree.read(Element,tree,U32.from_nat(BandValues.position(geometry,rows,channel,row,column))),before_value(before,U32.from_nat(pixel)),
      after_value(after,U32.from_nat(pixel)),stored,agreed)}

# The bands, the two contexts and their relation, as the context of every
# band.
type WholeScope<-Before: Data,-After: Data,-Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data> is Data:
  WholeScope{before: Before,after: After,geometry: Bands.Geometry,related: Related(before,after,geometry),
    capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},room: {Nat.is_le(BandGather.elements(geometry),U32.to_nat(capacity)) == True{} : Bool}}

def whole_fields(~Element: Data,~Before: Data,~After: Data,~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data,
  +scope: WholeScope<Before,After,Related>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  WholeScope{before,after,geometry,related,capacity,capacity_ok,room} = scope
  BandShape.band_fields(~Element,geometry)

def whole_before(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data,
  +scope: WholeScope<Before,After,Related>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  WholeScope{before,after,geometry,related,capacity,capacity_ok,room} = scope
  BandValues.band_values(~Element,~Before,~before_value,before,geometry)

def whole_after(~Element: Data,~Before: Data,~After: Data,~after_value: After -> U32 -> Element,~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data,
  +scope: WholeScope<Before,After,Related>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  WholeScope{before,after,geometry,related,capacity,capacity_ok,room} = scope
  BandValues.band_values(~Element,~After,~after_value,after,geometry)

def after_band(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,
  ~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data,~agrees: Agrees(~Element,~Before,~After,~before_value,~after_value,~Related),
  +scope: WholeScope<Before,After,Related>,+first: U32,+rows: U32,+buffer: Buffer.Content<Element>,
  +facts: Model.Both(~Element,~WholeScope<Before,After,Related>,~whole_fields(~Element,~Before,~After,~Related),~whole_before(~Element,~Before,~After,~before_value,~Related),scope,first,rows,buffer)) ->
  whole_after(~Element,~Before,~After,~after_value,~Related,scope,first,rows,buffer):
  WholeScope{+before,+after,+geometry,+related,+capacity,+capacity_ok,+room} = scope
  Buffer.Content{+tree,+length} = buffer
  Shape.Both{metadata,leaf} = facts
  BandValues.table(~CellScope<Element,Before,After,before_value,Related>,~AfterCell(~Element,~Before,~After,~before_value,~after_value,~Related),
    ~cell_channels(~Element,~Before,~After,~before_value,~Related),~cell_rows(~Element,~Before,~After,~before_value,~Related),~cell_width(~Element,~Before,~After,~before_value,~Related),
    ~after_cell(~Element,~Before,~After,~before_value,~after_value,~Related,~agrees),CellScope{before,after,geometry,related,first,rows,tree,length,metadata,leaf,capacity,capacity_ok,room})

# Well formed bands that hold a reader's values hold those of any reader that
# agrees with it on the indices of the geometry.
def agreed(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,
  ~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data,~agrees: Agrees(~Element,~Before,~After,~before_value,~after_value,~Related),
  +before: Before,+after: After,+geometry: Bands.Geometry,+related: Related(before,after,geometry),+content: Model.Content<Element>,
  metadata: BandShape.Contents(~Element,geometry,content),values: BandValues.Values(~Element,~Before,~before_value,before,geometry,content),
  +capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(BandGather.elements(geometry),U32.to_nat(capacity)) == True{} : Bool}) ->
  BandValues.Values(~Element,~After,~after_value,after,geometry,content):
  +scope = {WholeScope{before,after,geometry,related,capacity,capacity_ok,room} : WholeScope<Before,After,Related>}
  Model.content_one(~Element,~WholeScope<Before,After,Related>,
    ~Model.Both(~Element,~WholeScope<Before,After,Related>,~whole_fields(~Element,~Before,~After,~Related),~whole_before(~Element,~Before,~After,~before_value,~Related)),
    ~whole_after(~Element,~Before,~After,~after_value,~Related),~after_band(~Element,~Before,~After,~before_value,~after_value,~Related,~agrees),scope,content,
    Model.content_both(~Element,~WholeScope<Before,After,Related>,~whole_fields(~Element,~Before,~After,~Related),~whole_before(~Element,~Before,~After,~before_value,~Related),
      scope,content,metadata,values))
