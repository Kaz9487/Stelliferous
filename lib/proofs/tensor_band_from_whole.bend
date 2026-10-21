# Whole to band conversion: copy premises, allocation, the channel copies and
# the values they establish, including empty and existing-owner cases.
import Base
import ./tensor_band_gather.bend as BandGather
import ./storage_buffer_model.bend as Buffer
import ../tensor_band.bend as Bands
import ./storage_tree_model.bend as Tree
import ./tensor_band_model.bend as Model
import ./tensor_band_each.bend as Each
import ./storage_shape.bend as Shape
import ../machine_limits.bend as Machine
import ./row_major_index.bend as RowMajor
import ./tensor_band_copy.bend as BandCopy
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_values.bend as BandValues
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./boolean_logic.bend as Boolean
import ./encoded_coordinates.bend as Encoded
import ./machine_limit_bounds.bend as Limits
import ./nat_division_order.bend as DivisionOrder
import ./tensor_band_count.bend as BandCount
import ./tensor_band_inspection.bend as Inspection
import ./u32_comparison.bend as Comparison
import ./word_encoding.bend as WordEncoding
import ./write_program.bend as Writes
import ../storage_buffer.bend as Storage
import ./natural_addition.bend as Addition
import ./storage_allocation_refinement.bend as AllocationLaws
import ./storage_copy_refinement.bend as StorageCopyLaws
import ./tensor_allocation_depth.bend as Depth
import ./u32_division.bend as U32Division
import ./word_multiplication.bend as Multiplication
import ./storage_clone_refinement.bend as CloneLaws

# The inverse channel copy used by whole-to-band conversion establishes Values.
# Its source addresses reuse the gather coordinate proof in the other direction.
def write_rows(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> Tree.Tree<Element>:
  Bands.Geometry{images,+channels,+height,+width} = geometry
  Writes.pure(Element,BandCopy.writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),
    ((first / height * channels * height + first % height) * width : U32),0,input),output)

def written(~Element: Data,+call: BandGather.Call,input: Tree.Tree<Element>,output: Tree.Tree<Element>) -> Tree.Tree<Element>:
  write_rows(~Element,BandGather.values_geometry(call),BandGather.values_first(call),BandGather.values_rows(call),input,output)

def copied_cell(~Element: Data,+call: BandGather.Call,+input: Tree.Tree<Element>,+output: Buffer.Model<Element>,+channel: Nat,+row: Nat,+column: Nat,
  facts: BandGather.Facts<call>,ready: Buffer.Sized<Element,output>,length_equal: {Buffer.length(~Element,output) == BandGather.length(call) : Nat},
  +channel_inside: {Nat.is_lt(channel,BandValues.channels(BandGather.values_geometry(call))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(BandGather.values_rows(call))) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,BandValues.width(BandGather.values_geometry(call))) == True{} : Bool}) ->
  BandValues.At(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),input,BandGather.values_geometry(call),BandGather.values_first(call),BandGather.values_rows(call),
    written(~Element,call,input,Buffer.tree(~Element,output)),BandGather.length(call),channel,row,column):
  BandGather.Call{Bands.Geometry{+images,+channels,+height,+width},+first,+rows,+length,+capacity,+division_capacity} = call
  BandGather.Facts{+valid,+stride_exact,+length_exact,+length_bound,+division_ok,+plane_bound,+height_positive,+width_positive,+count_exact,+height_bound,+row_span} = facts
  Buffer.Sized{+depth_ok,room} = ready
  +plane = (rows * width : U32)
  +index = Nat.add(Nat.mul(row,U32.to_nat(width)),column)
  +position = BandValues.position(Bands.Geometry{images,channels,height,width},rows,channel,row,column)
  +pixel = U32.from_nat(BandValues.pixel(Bands.Geometry{images,channels,height,width},channel,Nat.add(U32.to_nat(first),row),column))
  +index_inside = {
    %Equal.sym(Nat,U32.to_nat(plane),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),count_exact) : {Nat.is_lt(index,_) == True{} : Bool}
    RowMajor.radix(row,U32.to_nat(rows),column,U32.to_nat(width),row_inside,column_inside) : {Nat.is_lt(index,U32.to_nat(plane)) == True{} : Bool}}
  +target_region = BandCopy.whole_region(U32.to_nat(channels),plane,Tree.capacity(Element,Buffer.tree(~Element,output)))
  +target_valid = BandCopy.whole_valid(U32.to_nat(channels),plane,Tree.capacity(Element,Buffer.tree(~Element,output)),Buffer.capacity_bound(~Element,output,depth_ok),
    Order.le_lt_trans(0n,index,U32.to_nat(plane),Order.zero_le(index),index_inside),
    %length_exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool}
    %length_equal : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,output)))) == True{} : Bool} room)
  +query = BandCopy.readback_query(target_region,channel,index)
  +encoded = Encoded.encoding(query,position,
    %count_exact : {U32.to_nat(query) == Nat.add(Nat.mul(channel,_),index) : Nat}
    BandCopy.readback_query_exact(target_region,channel,index,target_valid,channel_inside,index_inside))
  +bound = {
    %Equal.sym(Nat,length,Nat.mul(U32.to_nat(channels),U32.to_nat(plane)),length_exact) : {Nat.is_lt(position,_) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(plane),Nat.mul(U32.to_nat(rows),U32.to_nat(width)),count_exact) : {Nat.is_lt(position,Nat.mul(U32.to_nat(channels),_)) == True{} : Bool}
    BandValues.position_inside(Bands.Geometry{images,channels,height,width},rows,channel,row,column,channel_inside,row_inside,column_inside) : {Nat.is_lt(position,length) == True{} : Bool}}
  Shape.Both{bound,
    %encoded : {Tree.read(Element,written(~Element,call,input,Buffer.tree(~Element,output)),_) == Tree.read(Element,input,pixel) : Element}
    %BandGather.leaf_address(call,Nat.div(Nat.add(U32.to_nat(first),row),U32.to_nat(height)),channel,Nat.mod(Nat.add(U32.to_nat(first),row),U32.to_nat(height)),column,row,
      BandGather.Facts{valid,stride_exact,length_exact,length_bound,division_ok,plane_bound,height_positive,width_positive,count_exact,height_bound,row_span},
      channel_inside,DivisionOrder.remainder_small(Nat.add(U32.to_nat(first),row),U32.to_nat(height),height_positive),column_inside,row_inside,
      Equal.sym(Nat,Nat.add(U32.to_nat(first),row),Nat.add(Nat.mul(Nat.div(Nat.add(U32.to_nat(first),row),U32.to_nat(height)),U32.to_nat(height)),Nat.mod(Nat.add(U32.to_nat(first),row),U32.to_nat(height))),DivisionOrder.quotient_exact(Nat.add(U32.to_nat(first),row),U32.to_nat(height),height_positive))) :
      {Tree.read(Element,written(~Element,call,input,Buffer.tree(~Element,output)),query) == Tree.read(Element,input,_) : Element}
    BandCopy.read_inside(~Element,U32.to_nat(channels),0n,channel,index,target_region,(height * width : U32),BandCopy.destination(BandGather.values_region(call)),input,Buffer.tree(~Element,output),
      Buffer.depth(~Element,output),Buffer.balanced(~Element,output),depth_ok,{==},target_valid,Order.reflexive(U32.to_nat(channels)),channel_inside,index_inside)}

# What one copied leaf supplies to every cell.
type CellScope<-Element: Data> is Data:
  CellScope{call: BandGather.Call,input: Tree.Tree<Element>,output: Buffer.Model<Element>,facts: BandGather.Facts<call>,ready: Buffer.Sized<Element,output>,length_equal: {Buffer.length(~Element,output) == BandGather.length(call) : Nat}}

def cell_channels(~Element: Data,+scope: CellScope<Element>) -> Nat:
  CellScope{call,input,output,facts,ready,length_equal} = scope
  BandValues.channels(BandGather.values_geometry(call))

def cell_rows(~Element: Data,+scope: CellScope<Element>) -> Nat:
  CellScope{call,input,output,facts,ready,length_equal} = scope
  U32.to_nat(BandGather.values_rows(call))

def cell_width(~Element: Data,+scope: CellScope<Element>) -> Nat:
  CellScope{call,input,output,facts,ready,length_equal} = scope
  BandValues.width(BandGather.values_geometry(call))

def Cell(~Element: Data,+scope: CellScope<Element>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  CellScope{call,input,output,facts,ready,length_equal} = scope
  BandValues.At(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),input,BandGather.values_geometry(call),BandGather.values_first(call),BandGather.values_rows(call),
    written(~Element,call,input,Buffer.tree(~Element,output)),BandGather.length(call),channel,row,column)

def leaf_cell(~Element: Data,+scope: CellScope<Element>,+channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,cell_channels(~Element,scope)) == True{} : Bool},+row_inside: {Nat.is_lt(row,cell_rows(~Element,scope)) == True{} : Bool},+column_inside: {Nat.is_lt(column,cell_width(~Element,scope)) == True{} : Bool}) ->
  Cell(~Element,scope,channel,row,column):
  CellScope{+call,+input,+output,+facts,+ready,+length_equal} = scope
  copied_cell(~Element,call,input,output,channel,row,column,facts,ready,length_equal,channel_inside,row_inside,column_inside)

def values(~Element: Data,+call: BandGather.Call,+input: Tree.Tree<Element>,+output: Buffer.Model<Element>,
  facts: BandGather.Facts<call>,ready: Buffer.Sized<Element,output>,length_equal: {Buffer.length(~Element,output) == BandGather.length(call) : Nat}) ->
  BandValues.Leaf(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),input,BandGather.values_geometry(call),BandGather.values_first(call),BandGather.values_rows(call),
    written(~Element,call,input,Buffer.tree(~Element,output)),BandGather.length(call)):
  BandValues.table(~CellScope<Element>,~Cell(~Element),~cell_channels(~Element),~cell_rows(~Element),~cell_width(~Element),~leaf_cell(~Element),CellScope{call,input,output,facts,ready,length_equal})

# Actual whole-to-band channel copies, including allocation and source owner
# content preservation.
type Content<-Element: Data> is Data:
  Content{source: Buffer.Content<Element>,bands: Model.Content<Element>}

def contents(~Element: Data,owners: Storage.Buffer<Element> & Bands.Bands(~Element)) -> Content<Element>:
  (source,bands) = owners
  Content{Buffer.contents(~Element,source),Model.contents(~Element,bands)}

def bands(~Element: Data,content: Content<Element>) -> Model.Content<Element>:
  Content{source,bands} = content
  bands

def result(~Element: Data,first: U32,rows: U32,owners: StorageCopyLaws.Owners<Element>) -> Content<Element>:
  StorageCopyLaws.Owners{input,input_length,output,output_length} = owners
  Content{Buffer.Content{input,input_length},Model.RowBand{Buffer.Content{output,output_length},first,rows}}

def finished(~Element: Data,+first: U32,+rows: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {contents(~Element,Bands.whole_result(~Element,first,rows,owners)) == result(~Element,first,rows,StorageCopyLaws.observe(~Element,owners)) : Content<Element>}:
  (Storage.Buffer{input,input_length,input_certificate},Storage.Buffer{output,output_length,output_certificate}) = owners
  {==}

def run(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Storage.Buffer<Element> & Bands.Bands(~Element):
  Bands.Geometry{images,+channels,+height,+width} = geometry
  Bands.whole_result(~Element,first,rows,Bands.planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),
    ((first / height * channels * height + first % height) * width : U32),0,(input,output)))

def copy_model(~Element: Data,geometry: Bands.Geometry,+first: U32,+rows: U32,input: Buffer.Content<Element>,output: Buffer.Content<Element>) -> Content<Element>:
  Buffer.Content{+input,+input_length} = input
  Buffer.Content{output,output_length} = output
  Content{Buffer.Content{input,input_length},Model.RowBand{Buffer.Content{write_rows(~Element,geometry,first,rows,input,output),output_length},first,rows}}

def copy(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>) ->
  {contents(~Element,run(~Element,geometry,first,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output))) ==
    copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),Buffer.content(~Element,output)) : Content<Element>}:
  Bands.Geometry{images,+channels,+height,+width} = geometry
  Buffer.Model{+input,+input_length,+input_certificate} = input
  Buffer.Model{+output,+output_length,+output_certificate} = output
  +start = ((first / height * channels * height + first % height) * width : U32)
  %Equal.sym(Content<Element>,contents(~Element,Bands.whole_result(~Element,first,rows,Bands.planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),start,0,
      (Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate}))))),
    result(~Element,first,rows,StorageCopyLaws.observe(~Element,Bands.planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),start,0,
      (Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate}))))),
    finished(~Element,first,rows,Bands.planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),start,0,
      (Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate}))))) :
    {_ == copy_model(~Element,geometry,first,rows,Buffer.Content{input,input_length},Buffer.Content{output,output_length}) : Content<Element>}
  Equal.cong(StorageCopyLaws.Owners<Element>,Content<Element>,owners => result(~Element,first,rows,owners),
    StorageCopyLaws.observe(~Element,Bands.planes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),start,0,
      (Buffer.pack(~Element,Buffer.Model{input,input_length,input_certificate}),Buffer.pack(~Element,Buffer.Model{output,output_length,output_certificate})))),
    StorageCopyLaws.Owners{input,input_length,write_rows(~Element,geometry,first,rows,input,output),output_length},
    BandCopy.packed_writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),start,0,
      Buffer.Model{input,input_length,input_certificate},Buffer.Model{output,output_length,output_certificate}))

type Context<-Element: Data> is Data:
  Context{geometry: Bands.Geometry,first: U32,rows: U32,input: Buffer.Model<Element>}

def proposition(~Element: Data,context: Context<Element>,-output: Storage.Buffer<Element>) -> Type:
  Context{+geometry,+first,+rows,+input} = context
  {contents(~Element,run(~Element,geometry,first,rows,Buffer.pack(~Element,input),output)) ==
    copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),Buffer.contents(~Element,output)) : Content<Element>}

def copy_native(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,output: Storage.Buffer<Element>) ->
  {contents(~Element,run(~Element,geometry,first,rows,Buffer.pack(~Element,input),output)) ==
    copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),Buffer.contents(~Element,output)) : Content<Element>}:
  Buffer.lift(~Element,~Context<Element>,~proposition(~Element),Context{geometry,first,rows,input},output,Buffer.observe(~Element,output),output =>
    +output = output
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)) :
      {contents(~Element,run(~Element,geometry,first,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output))) == copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),_) : Content<Element>}
    copy(~Element,geometry,first,rows,input,output))

def leaf_model(~Element: Data,+geometry: Bands.Geometry,fill: Element,+first: U32,+rows: U32,input: Buffer.Content<Element>) -> Content<Element>:
  +count = Bands.local_count(geometry,rows)
  copy_model(~Element,geometry,first,rows,input,Buffer.Content{Tree.fresh(Element,Storage.depth_for(count),fill),U32.to_nat(count)})

def refinement_leaf(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>) ->
  {contents(~Element,Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))) == leaf_model(~Element,geometry,fill,first,rows,Buffer.content(~Element,input)) : Content<Element>}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  +count = Bands.local_count(geometry,rows)
  Equal.trans(Content<Element>,contents(~Element,Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))),
    copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),Buffer.contents(~Element,Bands.allocate(~Element,fill,count))),leaf_model(~Element,geometry,fill,first,rows,Buffer.content(~Element,input)),
    copy_native(~Element,geometry,first,rows,input,Bands.allocate(~Element,fill,count)),
    Equal.cong(Buffer.Content<Element>,Content<Element>,output => copy_model(~Element,geometry,first,rows,Buffer.content(~Element,input),output),
      Buffer.contents(~Element,Bands.allocate(~Element,fill,count)),Buffer.Content{Tree.fresh(Element,Storage.depth_for(count),fill),U32.to_nat(count)},AllocationLaws.contents(~Element,Storage.depth_for(count),fill,U32.to_nat(count))))

# The contents of one band made from a whole buffer.
def band(~Element: Data,+geometry: Bands.Geometry,fill: Element,input: Buffer.Content<Element>,+first: U32,+rows: U32) -> Buffer.Content<Element>:
  Buffer.Content{input,length} = input
  +count = Bands.local_count(geometry,rows)
  Buffer.Content{write_rows(~Element,geometry,first,rows,input,Tree.fresh(Element,Storage.depth_for(count),fill)),U32.to_nat(count)}

def model(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+input: Buffer.Content<Element>) -> Model.Content<Element>:
  Model.build(~Element,~band(~Element,geometry,fill,input),layout)

def leaf_source(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Content<Element>) ->
  {leaf_model(~Element,geometry,fill,first,rows,input) == Content{input,model(~Element,geometry,fill,Bands.Rows{first,rows},input)} : Content<Element>}:
  Buffer.Content{tree,length} = input
  {==}

# The source and the band of a leaf, as the two buffers whole_rows gives.
def unpaired(~Element: Data,content: Content<Element>) -> Buffer.Content<Element> & Buffer.Content<Element>:
  Content{+source,bands} = content
  match bands:
    case Model.RowBand{buffer,first,rows}: (source,buffer)
    case Model.Joined{left,right}: (source,source)

def rows_paired(~Element: Data,+first: U32,+rows: U32,owners: Storage.Buffer<Element> & Storage.Buffer<Element>) ->
  {unpaired(~Element,contents(~Element,Bands.whole_result(~Element,first,rows,owners))) == CloneLaws.contents(~Element,owners) : Buffer.Content<Element> & Buffer.Content<Element>}:
  (source,band) = owners
  {==}

# Rows [first, first + rows) of every plane copied out of a whole buffer,
# beside the source.
def rows_contents(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>) ->
  {CloneLaws.contents(~Element,Bands.whole_rows(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))) ==
    (Buffer.content(~Element,input),band(~Element,geometry,fill,Buffer.content(~Element,input),first,rows)) : Buffer.Content<Element> & Buffer.Content<Element>}:
  Equal.trans(Buffer.Content<Element> & Buffer.Content<Element>,CloneLaws.contents(~Element,Bands.whole_rows(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))),
    unpaired(~Element,contents(~Element,Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input)))),
    (Buffer.content(~Element,input),band(~Element,geometry,fill,Buffer.content(~Element,input),first,rows)),
    Equal.sym(Buffer.Content<Element> & Buffer.Content<Element>,unpaired(~Element,contents(~Element,Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input)))),
      CloneLaws.contents(~Element,Bands.whole_rows(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))),
      rows_paired(~Element,first,rows,Bands.whole_rows(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input)))),
    Equal.trans(Buffer.Content<Element> & Buffer.Content<Element>,unpaired(~Element,contents(~Element,Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input)))),
      unpaired(~Element,leaf_model(~Element,geometry,fill,first,rows,Buffer.content(~Element,input))),
      (Buffer.content(~Element,input),band(~Element,geometry,fill,Buffer.content(~Element,input),first,rows)),
      Equal.cong(Content<Element>,Buffer.Content<Element> & Buffer.Content<Element>,content => unpaired(~Element,content),
        contents(~Element,Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))),leaf_model(~Element,geometry,fill,first,rows,Buffer.content(~Element,input)),
        refinement_leaf(~Element,geometry,fill,first,rows,input)),
      Equal.cong(Content<Element>,Buffer.Content<Element> & Buffer.Content<Element>,content => unpaired(~Element,content),
        leaf_model(~Element,geometry,fill,first,rows,Buffer.content(~Element,input)),Content{Buffer.content(~Element,input),model(~Element,geometry,fill,Bands.Rows{first,rows},Buffer.content(~Element,input))},
        leaf_source(~Element,geometry,fill,first,rows,Buffer.content(~Element,input)))))

def call(+geometry: Bands.Geometry,first: U32,+rows: U32,capacity: U32,division_capacity: U32) -> BandGather.Call:
  BandGather.Call{geometry,first,rows,U32.to_nat(Bands.local_count(geometry,rows)),capacity,division_capacity}

type LeafFacts<-geometry: Bands.Geometry,-first: U32,-rows: U32,-capacity: U32,-division_capacity: U32> is Data:
  LeafFacts{allocation: {U32.is_le(Bands.local_count(geometry,rows),1073741824) == True{} : Bool},
    contents: Either<&2,&2,{U32.to_nat(Bands.local_count(geometry,rows)) == 0n : Nat},BandGather.Facts<call(geometry,first,rows,capacity,division_capacity)>>,
    metadata: BandShape.Leaf(geometry,first,rows,U32.to_nat(Bands.local_count(geometry,rows)))}

type Room is Data:
  Room{geometry: Bands.Geometry,capacity: U32,division_capacity: U32}

def fits_band(room: Room,+first: U32,+rows: U32) -> Data:
  Room{geometry,capacity,division_capacity} = room
  LeafFacts<geometry,first,rows,capacity,division_capacity>

def Fits(+geometry: Bands.Geometry,layout: Bands.Plan,+capacity: U32,+division_capacity: U32) -> Data:
  Model.PlanAll(~Room,~fits_band,Room{geometry,capacity,division_capacity},layout)

# What the conversion of one band needs, and the facts of one band read from it.
type Conversion<-Element: Data> is Data:
  Conversion{geometry: Bands.Geometry,fill: Element,input: Buffer.Content<Element>,capacity: U32,division_capacity: U32}

def conversion_band(~Element: Data,+conversion: Conversion<Element>) -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>:
  Conversion{geometry,fill,input,capacity,division_capacity} = conversion
  band(~Element,geometry,fill,input)

def conversion_room(~Element: Data,+conversion: Conversion<Element>) -> Room:
  Conversion{geometry,fill,input,capacity,division_capacity} = conversion
  Room{geometry,capacity,division_capacity}

def conversion_values(~Element: Data,+conversion: Conversion<Element>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Conversion{geometry,fill,input,capacity,division_capacity} = conversion
  BandValues.band_values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),Buffer.content_tree(~Element,input),geometry)

def conversion_fields(~Element: Data,+conversion: Conversion<Element>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Conversion{geometry,fill,input,capacity,division_capacity} = conversion
  BandShape.band_fields(~Element,geometry)

# The source and the bands beside it, as the shared build reads them.
def pair(~Element: Data,content: Content<Element>) -> Buffer.Content<Element> & Model.Content<Element>:
  Content{source,bands} = content
  (source,bands)

def record(~Element: Data,made: Buffer.Content<Element> & Model.Content<Element>) -> Content<Element>:
  (source,bands) = made
  Content{source,bands}

def seen_contents(~Element: Data,owners: Storage.Buffer<Element> & Bands.Bands(~Element)) ->
  {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),owners) == pair(~Element,contents(~Element,owners)) : Buffer.Content<Element> & Model.Content<Element>}:
  (source,bands) = owners
  {==}

def contents_seen(~Element: Data,owners: Storage.Buffer<Element> & Bands.Bands(~Element)) ->
  {contents(~Element,owners) == record(~Element,Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),owners)) : Content<Element>}:
  (source,bands) = owners
  {==}

def filling_band(~Element: Data,context: Bands.Filling<Element>,input: Buffer.Content<Element>) -> @+first: U32 -> @+rows: U32 -> Buffer.Content<Element>:
  Bands.Filling{geometry,fill} = context
  band(~Element,geometry,fill,input)

def whole_rule(~Element: Data,+context: Bands.Filling<Element>,+first: U32,+rows: U32,+input: Buffer.Model<Element>) ->
  {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.whole_band(~Element,context,first,rows,Buffer.pack(~Element,input))) ==
    (Buffer.contents(~Element,Buffer.pack(~Element,input)),Model.RowBand{filling_band(~Element,context,Buffer.contents(~Element,Buffer.pack(~Element,input)),first,rows),first,rows}) : Buffer.Content<Element> & Model.Content<Element>}:
  Bands.Filling{+geometry,+fill} = context
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.content(~Element,input),Buffer.packed_contents(~Element,input)) :
    {Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))) == (_,Model.RowBand{band(~Element,geometry,fill,_,first,rows),first,rows}) : Buffer.Content<Element> & Model.Content<Element>}
  Equal.trans(Buffer.Content<Element> & Model.Content<Element>,Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))),
    pair(~Element,contents(~Element,Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input)))),
    (Buffer.content(~Element,input),Model.RowBand{band(~Element,geometry,fill,Buffer.content(~Element,input),first,rows),first,rows}),
    seen_contents(~Element,Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))),
    Equal.cong(Content<Element>,Buffer.Content<Element> & Model.Content<Element>,content => pair(~Element,content),
      contents(~Element,Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))),
      Content{Buffer.content(~Element,input),model(~Element,geometry,fill,Bands.Rows{first,rows},Buffer.content(~Element,input))},
      Equal.trans(Content<Element>,contents(~Element,Bands.whole_leaf(~Element,geometry,fill,first,rows,Buffer.pack(~Element,input))),leaf_model(~Element,geometry,fill,first,rows,Buffer.content(~Element,input)),
        Content{Buffer.content(~Element,input),model(~Element,geometry,fill,Bands.Rows{first,rows},Buffer.content(~Element,input))},
        refinement_leaf(~Element,geometry,fill,first,rows,input),leaf_source(~Element,geometry,fill,first,rows,Buffer.content(~Element,input)))))

def ranges(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>) ->
  {contents(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))) ==
    Content{Buffer.contents(~Element,Buffer.pack(~Element,input)),model(~Element,geometry,fill,layout,Buffer.contents(~Element,Buffer.pack(~Element,input)))} : Content<Element>}:
  Equal.trans(Content<Element>,contents(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),
    record(~Element,Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))),
    Content{Buffer.contents(~Element,Buffer.pack(~Element,input)),model(~Element,geometry,fill,layout,Buffer.contents(~Element,Buffer.pack(~Element,input)))},
    contents_seen(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),
    Equal.cong(Buffer.Content<Element> & Model.Content<Element>,Content<Element>,made => record(~Element,made),
      Each.seen(~Element,~Storage.Buffer<Element>,~Buffer.Content<Element>,~Buffer.contents(~Element),Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),
      (Buffer.contents(~Element,Buffer.pack(~Element,input)),model(~Element,geometry,fill,layout,Buffer.contents(~Element,Buffer.pack(~Element,input)))),
      Each.built(~Element,~Storage.Buffer<Element>,~Bands.Filling<Element>,~Buffer.Model<Element>,~Buffer.pack(~Element),~Each.buffer_held(~Element),~Buffer.Content<Element>,~Buffer.contents(~Element),
    ~Bands.whole_band(~Element),~filling_band(~Element),~whole_rule(~Element),layout,Bands.Filling{geometry,fill},input)))

def result_contents(~Element: Data,owners: Storage.Buffer<Element> & Bands.Bands(~Element)) ->
  {Model.contents(~Element,Bands.result(~Element,owners)) == bands(~Element,contents(~Element,owners)) : Model.Content<Element>}:
  (source,result) = owners
  {==}

def refinement_copied(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>) ->
  {Model.contents(~Element,Bands.result(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))) == model(~Element,geometry,fill,layout,Buffer.content(~Element,input)) : Model.Content<Element>}:
  Equal.trans(Model.Content<Element>,Model.contents(~Element,Bands.result(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))),
    bands(~Element,contents(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))),model(~Element,geometry,fill,layout,Buffer.content(~Element,input)),
    result_contents(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),
    %Buffer.packed_contents(~Element,input) :
      {bands(~Element,contents(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))) == model(~Element,geometry,fill,layout,_) : Model.Content<Element>}
    Equal.cong(Content<Element>,Model.Content<Element>,result => bands(~Element,result),contents(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))),
      Content{Buffer.contents(~Element,Buffer.pack(~Element,input)),model(~Element,geometry,fill,layout,Buffer.contents(~Element,Buffer.pack(~Element,input)))},ranges(~Element,geometry,fill,layout,input)))

def model_layout(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+input: Buffer.Content<Element>) ->
  {Model.content_layout(~Element,model(~Element,geometry,fill,layout,input)) == layout : Bands.Plan}:
  Model.build_layout(~Element,~Conversion<Element>,~conversion_band(~Element),Conversion{geometry,fill,input,0,0},layout)

def copied_layout(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>) ->
  {Model.content_layout(~Element,Model.contents(~Element,Bands.result(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))))) == layout : Bands.Plan}:
  Equal.trans(Bands.Plan,Model.content_layout(~Element,Model.contents(~Element,Bands.result(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))))),
    Model.content_layout(~Element,model(~Element,geometry,fill,layout,Buffer.content(~Element,input))),layout,
    Equal.cong(Model.Content<Element>,Bands.Plan,result => Model.content_layout(~Element,result),
      Model.contents(~Element,Bands.result(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))),model(~Element,geometry,fill,layout,Buffer.content(~Element,input)),refinement_copied(~Element,geometry,fill,layout,input)),
    model_layout(~Element,geometry,fill,layout,Buffer.content(~Element,input)))

# Whole conversion establishes the shared Values relation through its actual
# allocation and channel-copy traversal. Plan geometry and machine bounds are
# explicit obligations; no readback or successful output equation is assumed.
def leaf_empty(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,
  empty: {U32.to_nat(Bands.local_count(geometry,rows)) == 0n : Nat},metadata: BandShape.Leaf(geometry,first,rows,U32.to_nat(Bands.local_count(geometry,rows)))) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),input,geometry,
    model(~Element,geometry,fill,Bands.Rows{first,rows},Buffer.Content{input,length})):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  +count = Bands.local_count(Bands.Geometry{images,channels,height,width},rows)
  BandValues.empty_leaf(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),input,Bands.Geometry{images,channels,height,width},first,rows,
    write_rows(~Element,Bands.Geometry{images,channels,height,width},first,rows,input,Buffer.tree(~Element,BandCount.model(~Element,count,fill))),U32.to_nat(count),
    Equal.trans(Nat,BandValues.cell_count(Bands.Geometry{images,channels,height,width},rows),U32.to_nat(count),0n,Equal.sym(Nat,U32.to_nat(count),BandValues.cell_count(Bands.Geometry{images,channels,height,width},rows),length_exact),empty))

def leaf_copied(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+capacity: U32,+division_capacity: U32,
  allocation: {U32.is_le(Bands.local_count(geometry,rows),1073741824) == True{} : Bool},facts: BandGather.Facts<call(geometry,first,rows,capacity,division_capacity)>) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),input,geometry,
    model(~Element,geometry,fill,Bands.Rows{first,rows},Buffer.Content{input,length})):
  +count = Bands.local_count(geometry,rows)
  values(~Element,call(geometry,first,rows,capacity,division_capacity),input,BandCount.model(~Element,count,fill),facts,BandCount.bounded_model(~Element,count,fill,allocation),{==})

def semantics_leaf(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Tree.Tree<Element>,+length: Nat,+capacity: U32,+division_capacity: U32,
  facts: LeafFacts<geometry,first,rows,capacity,division_capacity>) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),input,geometry,
    model(~Element,geometry,fill,Bands.Rows{first,rows},Buffer.Content{input,length})):
  LeafFacts{allocation,contents,metadata} = facts
  match contents:
    case Inl{empty}: leaf_empty(~Element,geometry,fill,first,rows,input,length,empty,metadata)
    case Inr{facts}: leaf_copied(~Element,geometry,fill,first,rows,input,length,capacity,division_capacity,allocation,facts)

def conversion_values_band(~Element: Data,+conversion: Conversion<Element>,+first: U32,+rows: U32,facts: fits_band(conversion_room(~Element,conversion),first,rows)) ->
  conversion_values(~Element,conversion,first,rows,conversion_band(~Element,conversion,first,rows)):
  Conversion{+geometry,+fill,Buffer.Content{+input,+length},+capacity,+division_capacity} = conversion
  semantics_leaf(~Element,geometry,fill,first,rows,input,length,capacity,division_capacity,facts)

def modeled(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+input: Tree.Tree<Element>,+length: Nat,+capacity: U32,+division_capacity: U32,
  fits: Fits(geometry,layout,capacity,division_capacity)) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),input,geometry,model(~Element,geometry,fill,layout,Buffer.Content{input,length})):
  Model.build_all(~Element,~Conversion<Element>,~conversion_band(~Element),~Room,~fits_band,~conversion_room(~Element),~conversion_values(~Element),~conversion_values_band(~Element),
    Conversion{geometry,fill,Buffer.Content{input,length},capacity,division_capacity},layout,fits)

def copied_values(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>,+division_capacity: U32,
  fits: Fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),Buffer.tree(~Element,input),geometry,
    Model.contents(~Element,Bands.result(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))))):
  Buffer.Model{+tree,+length,+certificate} = input
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.result(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))),
    model(~Element,geometry,fill,layout,Buffer.content(~Element,input)),refinement_copied(~Element,geometry,fill,layout,input)) :
    BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),tree,geometry,_)
  modeled(~Element,geometry,fill,layout,tree,length,Tree.capacity(Element,tree),division_capacity,fits)

def leaf_ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,input: Buffer.Content<Element>,+capacity: U32,+division_capacity: U32,
  fits: LeafFacts<geometry,first,rows,capacity,division_capacity>) -> Buffer.ContentReady<Element,band(~Element,geometry,fill,input,first,rows)>:
  Bands.Geometry{images,+channels,+height,+width} = geometry
  Buffer.Content{+input,length} = input
  LeafFacts{allocation,contents,metadata} = fits
  +count = Bands.local_count(geometry,rows)
  Writes.content_ready(~Element,BandCopy.writes(~Element,U32.to_nat(channels),0,(rows * width : U32),(height * width : U32),(rows * width : U32),
      ((first / height * channels * height + first % height) * width : U32),0,input),
    Buffer.tree(~Element,BandCount.model(~Element,count,fill)),U32.to_nat(count),
    Buffer.content_ready(~Element,BandCount.model(~Element,count,fill),BandCount.bounded_model(~Element,count,fill,allocation)))

def conversion_ready_band(~Element: Data,+conversion: Conversion<Element>,+first: U32,+rows: U32,facts: fits_band(conversion_room(~Element,conversion),first,rows)) ->
  Buffer.ContentReady<Element,conversion_band(~Element,conversion,first,rows)>:
  Conversion{+geometry,+fill,+input,+capacity,+division_capacity} = conversion
  leaf_ready(~Element,geometry,fill,first,rows,input,capacity,division_capacity,facts)

def modeled_ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+input: Buffer.Content<Element>,+capacity: U32,+division_capacity: U32,
  fits: Fits(geometry,layout,capacity,division_capacity)) -> Model.ContentReady(~Element,model(~Element,geometry,fill,layout,input)):
  Model.build_all(~Element,~Conversion<Element>,~conversion_band(~Element),~Room,~fits_band,~conversion_room(~Element),~Model.always(~Element,~Conversion<Element>,~Model.ready_band(~Element)),~conversion_ready_band(~Element),
    Conversion{geometry,fill,input,capacity,division_capacity},layout,fits)

def copied_ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>,+division_capacity: U32,
  fits: Fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  Model.ContentReady(~Element,Model.contents(~Element,Bands.result(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.result(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))),
    model(~Element,geometry,fill,layout,Buffer.content(~Element,input)),refinement_copied(~Element,geometry,fill,layout,input)) : Model.ContentReady(~Element,_)
  modeled_ready(~Element,geometry,fill,layout,Buffer.content(~Element,input),Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity,fits)

def conversion_fields_band(~Element: Data,+conversion: Conversion<Element>,+first: U32,+rows: U32,facts: fits_band(conversion_room(~Element,conversion),first,rows)) ->
  conversion_fields(~Element,conversion,first,rows,conversion_band(~Element,conversion,first,rows)):
  Conversion{geometry,fill,Buffer.Content{tree,length},capacity,division_capacity} = conversion
  LeafFacts{allocation,contents,metadata} = facts
  metadata

def modeled_shape(~Element: Data,+geometry: Bands.Geometry,+fill: Element,layout: Bands.Plan,+input: Buffer.Content<Element>,+capacity: U32,+division_capacity: U32,
  fits: Fits(geometry,layout,capacity,division_capacity)) -> BandShape.Contents(~Element,geometry,model(~Element,geometry,fill,layout,input)):
  Model.build_all(~Element,~Conversion<Element>,~conversion_band(~Element),~Room,~fits_band,~conversion_room(~Element),~conversion_fields(~Element),~conversion_fields_band(~Element),
    Conversion{geometry,fill,input,capacity,division_capacity},layout,fits)

def copied_shape(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>,+division_capacity: U32,
  fits: Fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  BandShape.Contents(~Element,geometry,Model.contents(~Element,Bands.result(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input))))):
  %Equal.sym(Model.Content<Element>,Model.contents(~Element,Bands.result(~Element,Bands.whole_ranges(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))),
    model(~Element,geometry,fill,layout,Buffer.content(~Element,input)),refinement_copied(~Element,geometry,fill,layout,input)) : BandShape.Contents(~Element,geometry,_)
  modeled_shape(~Element,geometry,fill,layout,Buffer.content(~Element,input),Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity,fits)

# Build whole conversion's copy premises from the actual image-local plan and
# the checked full tensor shape. Empty leaves do not need a source address.
def empty_count(+geometry: Bands.Geometry,+rows: U32,empty: {U32.to_nat(rows) == 0n : Nat}) -> {Bands.local_count(geometry,rows) == 0 : U32}:
  Bands.Geometry{images,+channels,height,+width} = geometry
  +prefix_zero = Equal.trans(U32,(channels * rows : U32),U32.from_nat(Nat.mul(U32.to_nat(channels),U32.to_nat(rows))),0,
    Multiplication.native_multiply(channels,rows),
    Equal.cong(Nat,U32,U32.from_nat,Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),0n,
      %Equal.sym(Nat,U32.to_nat(rows),0n,empty) : {Nat.mul(U32.to_nat(channels),_) == 0n : Nat}
      RowMajor.multiply_zero_right(U32.to_nat(channels))))
  %Equal.sym(U32,(channels * rows : U32),0,prefix_zero) : {(_ * width : U32) == 0 : U32}
  Arithmetic.multiply_zero_left(width)

def empty_leaf(+geometry: Bands.Geometry,+first: U32,+rows: U32,+capacity: U32,+division_capacity: U32,+empty: {U32.to_nat(rows) == 0n : Nat},
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry)) == True{} : Bool}) ->
  LeafFacts<geometry,first,rows,capacity,division_capacity>:
  Bands.Geometry{images,+channels,+height,+width} = geometry
  +count_zero = empty_count(geometry,rows,empty)
  +native_zero = Arithmetic.to_nat_injective(rows,0,empty)
  +length_zero = {
    %Equal.sym(Nat,U32.to_nat(rows),0n,empty) : {Nat.mul(U32.to_nat(channels),Nat.mul(_,U32.to_nat(width))) == 0n : Nat}
    RowMajor.multiply_zero_right(U32.to_nat(channels)) : {Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) == 0n : Nat}}
  LeafFacts{
    %Equal.sym(U32,Bands.local_count(geometry,rows),0,count_zero) : {U32.is_le(_,1073741824) == True{} : Bool} {==},
    Inl{Equal.cong(U32,Nat,U32.to_nat,Bands.local_count(geometry,rows),0,count_zero)},
    BandShape.Fields{
      %Equal.sym(U32,rows,0,native_zero) : {U32.is_le(_,(height - first % height : U32)) == True{} : Bool}
      %Equal.sym(Bool,U32.is_le(0,(height - first % height : U32)),Nat.is_le(0n,U32.to_nat((height - first % height : U32))),Comparison.le(0,(height - first % height : U32))) : {_ == True{} : Bool}
      Order.zero_le(U32.to_nat((height - first % height : U32))),rows_fit,
      Equal.trans(Nat,U32.to_nat(Bands.local_count(geometry,rows)),0n,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),
        Equal.cong(U32,Nat,U32.to_nat,Bands.local_count(geometry,rows),0,count_zero),
        Equal.sym(Nat,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),0n,length_zero))}}

def positive_leaf(+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+first: U32,+rows: U32,+capacity: U32,+image: Nat,
  +inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},+rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +span: BandPlan.LeafWithin<first,rows,Nat.mul(image,U32.to_nat(height)),Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat(height))>,
  domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>) ->
  LeafFacts<Bands.Geometry{images,channels,height,width},first,rows,capacity,Machine.half_word(limits)>:
  BandGather.Domain{+capacity_ok,+channels_positive,+height_positive,+width_positive,+machine_room,+allocation_room} = domain
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +spatial = Nat.mul(U32.to_nat(height),U32.to_nat(width))
  +frame = Nat.mul(U32.to_nat(channels),spatial)
  +total = Nat.mul(U32.to_nat(images),frame)
  +machine_capacity = Machine.capacity(limits)
  +images_positive = Order.le_lt_trans(0n,image,U32.to_nat(images),Order.zero_le(image),inside)
  +frame_room = Order.transitive(frame,total,U32.to_nat(machine_capacity),
    %RowMajor.multiply_commutative(frame,U32.to_nat(images)) : {Nat.is_le(frame,_) == True{} : Bool}
    RowMajor.product_covers(frame,U32.to_nat(images),images_positive),machine_room)
  +spatial_room = Order.transitive(spatial,frame,U32.to_nat(machine_capacity),
    %RowMajor.multiply_commutative(spatial,U32.to_nat(channels)) : {Nat.is_le(spatial,_) == True{} : Bool}
    RowMajor.product_covers(spatial,U32.to_nat(channels),channels_positive),frame_room)
  +height_room = Order.transitive(U32.to_nat(height),spatial,U32.to_nat(machine_capacity),RowMajor.product_covers(U32.to_nat(height),U32.to_nat(width),width_positive),spatial_room)
  +rows_bound = Order.drop_left_le(Nat.sub(U32.to_nat(first),Nat.mul(image,U32.to_nat(height))),U32.to_nat(rows),U32.to_nat(height),BandPlan.offset_span(first,rows,image,U32.to_nat(height),span))
  +plane = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +length = Nat.mul(U32.to_nat(channels),plane)
  +plane_spatial = RowMajor.mul_le(U32.to_nat(rows),U32.to_nat(height),U32.to_nat(width),rows_bound)
  +length_room = Order.transitive(length,frame,U32.to_nat(machine_capacity),
    %RowMajor.multiply_commutative(plane,U32.to_nat(channels)) : {Nat.is_le(_,frame) == True{} : Bool}
    %RowMajor.multiply_commutative(spatial,U32.to_nat(channels)) : {Nat.is_le(Nat.mul(plane,U32.to_nat(channels)),_) == True{} : Bool}
    RowMajor.mul_le(plane,spatial,U32.to_nat(channels),plane_spatial),frame_room)
  +count = Bands.local_count(geometry,rows)
  +count_exact = Equal.trans(Nat,U32.to_nat(count),Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),U32.to_nat(width)),length,
    Encoded.multiply_three(channels,rows,width,machine_capacity,Limits.capacity_fits(limits),width_positive,
      %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),U32.to_nat(width)),length,RowMajor.assoc(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width))) :
        {Nat.is_le(_,U32.to_nat(machine_capacity)) == True{} : Bool} length_room),RowMajor.assoc(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width)))
  +allocatable = Arithmetic.count_within(count,length,machine_capacity,1073741824,count_exact,length_room,Limits.capacity_allocatable(limits))
  +local_guard = Inspection.local_guard(height,first,rows,Machine.half_word(limits),Limits.half_word_fits(limits),height_positive,Limits.double_bound(limits,U32.to_nat(height),height_room),
    BandPlan.local_span(first,rows,image,U32.to_nat(height),height_positive,rows_positive,span))
  +rows_fit = BandPlan.total_rows(first,rows,image,U32.to_nat(images),U32.to_nat(height),inside,span)
  LeafFacts{allocatable,
    Inr{BandGather.from_leaf_facts(limits,images,channels,height,width,first,rows,U32.to_nat(count),Storage.depth_for(count),capacity,capacity_ok,channels_positive,height_positive,width_positive,rows_positive,machine_room,allocation_room,
      Inspection.LeafFacts{local_guard,rows_fit,count_exact,Depth.allocation_room(count,Depth.bounded(count,allocatable))})},
    BandShape.Fields{local_guard,rows_fit,count_exact}}

def selected_leaf(empty: Bool,+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+first: U32,+rows: U32,+capacity: U32,+image: Nat,
  compared: {U32.is_zero(rows) == empty : Bool},inside: {Nat.is_lt(image,U32.to_nat(images)) == True{} : Bool},
  +span: BandPlan.LeafWithin<first,rows,Nat.mul(image,U32.to_nat(height)),Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat(height))>,
  domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>) ->
  LeafFacts<Bands.Geometry{images,channels,height,width},first,rows,capacity,Machine.half_word(limits)>:
  match empty:
    case True{}: empty_leaf(Bands.Geometry{images,channels,height,width},first,rows,capacity,Machine.half_word(limits),Equal.cong(U32,Nat,U32.to_nat,rows,0,Arithmetic.equal_of_is_eq(rows,0,compared)),BandPlan.total_rows(first,rows,image,U32.to_nat(images),U32.to_nat(height),inside,span))
    case False{}: positive_leaf(images,channels,height,width,limits,first,rows,capacity,image,inside,Arithmetic.word_nonzero_positive(rows,compared),span,domain)

def bounds_leaf(+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+first: U32,+rows: U32,+capacity: U32,
  local: BandPlan.ImageSpan<first,rows,U32.to_nat(images),U32.to_nat(height)>,within: BandPlan.LeafWithin<first,rows,0n,Nat.mul(U32.to_nat(images),U32.to_nat(height))>,domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>) ->
  LeafFacts<Bands.Geometry{images,channels,height,width},first,rows,capacity,Machine.half_word(limits)>:
  match local:
    case BandPlan.EmptySpan{empty}:
      BandPlan.LeafWithin{lower,upper} = within
      empty_leaf(Bands.Geometry{images,channels,height,width},first,rows,capacity,Machine.half_word(limits),empty,upper)
    case BandPlan.ImageSpan{image,inside,span}: selected_leaf(U32.is_zero(rows),images,channels,height,width,limits,first,rows,capacity,image,{==},inside,span,domain)

# The bounds of every leaf of an image-local plan give the copy premises of every leaf.
type Bounds is Data:
  Bounds{images: U32,channels: U32,height: U32,width: U32,limits: Machine.Limits,capacity: U32,domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>}

def fits(layout: Bands.Plan,+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+capacity: U32,
  local: BandPlan.ImageLocal(layout,U32.to_nat(images),U32.to_nat(height)),within: BandPlan.Within(layout,0n,Nat.mul(U32.to_nat(images),U32.to_nat(height))),+domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>) ->
  Fits(Bands.Geometry{images,channels,height,width},layout,capacity,Machine.half_word(limits)):
  match layout:
    case Bands.Rows{first,rows}: bounds_leaf(images,channels,height,width,limits,first,rows,capacity,local,within,domain)
    case Bands.Split{+left,+right}:
      Shape.Both{left_local,right_local} = local
      Shape.Both{left_within,right_within} = within
      Shape.Both{fits(left,images,channels,height,width,limits,capacity,left_local,left_within,domain),fits(right,images,channels,height,width,limits,capacity,right_local,right_within,domain)}

def split_plan(geometry: Bands.Geometry,depth: Nat) -> Bands.Plan:
  Bands.Geometry{images,channels,height,width} = geometry
  Bands.image_plan(U32.to_nat(images),0,depth,height)

def image_plan(+geometry: Bands.Geometry,+limits: Machine.Limits,+depth: Nat,+capacity: U32,+domain: BandGather.Domain<geometry,limits,capacity>) ->
  Fits(geometry,split_plan(geometry,depth),capacity,Machine.half_word(limits)):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandGather.Domain{capacity_ok,channels_positive,+height_positive,width_positive,machine_room,allocation_room} = domain
  +row_room = BandGather.rows_room(geometry,limits,capacity,domain)
  +images_room = Order.transitive(U32.to_nat(images),Nat.mul(U32.to_nat(images),U32.to_nat(height)),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(images),U32.to_nat(height),height_positive),row_room)
  fits(Bands.image_plan(U32.to_nat(images),0,depth,height),images,channels,height,width,limits,capacity,
    BandPlan.bounds_image_plan(U32.to_nat(images),0,depth,height,U32.to_nat(images),Machine.capacity(limits),Limits.capacity_fits(limits),Order.reflexive(U32.to_nat(images)),
      images_room,row_room),BandPlan.image_within(images,height,depth,Machine.capacity(limits),Limits.capacity_fits(limits),images_room,row_room),domain)

# A leaf that holds the whole image starts at row 0, holds the height, and its
# batch is one.
def Held(batch: U32,first: U32,rows: U32,height: U32) -> Data:
  {U32.is_eq(batch,1) && U32.is_zero(first) && U32.is_eq(rows,height) == True{} : Bool}

def held_batch(+batch: U32,+first: U32,+rows: U32,+height: U32,held: Held(batch,first,rows,height)) -> {batch == 1 : U32}:
  +front = Boolean.left((U32.is_eq(batch,1) && U32.is_zero(first) : Bool),U32.is_eq(rows,height),held)
  Arithmetic.equal_of_is_eq(batch,1,Boolean.left(U32.is_eq(batch,1),U32.is_zero(first),front))

def held_first(+batch: U32,+first: U32,+rows: U32,+height: U32,held: Held(batch,first,rows,height)) -> {first == 0 : U32}:
  +front = Boolean.left((U32.is_eq(batch,1) && U32.is_zero(first) : Bool),U32.is_eq(rows,height),held)
  Arithmetic.word_zero(first,Boolean.right(U32.is_eq(batch,1),U32.is_zero(first),front))

def held_rows(+batch: U32,+first: U32,+rows: U32,+height: U32,held: Held(batch,first,rows,height)) -> {rows == height : U32}:
  Arithmetic.equal_of_is_eq(rows,height,Boolean.right((U32.is_eq(batch,1) && U32.is_zero(first) : Bool),U32.is_eq(rows,height),held))

# What a cut reads from its whole owner: the whole length and a sized buffer.
type Owned<-Element: Data,-geometry: Bands.Geometry,-input: Buffer.Model<Element>> is Data:
  Owned{expected_length: {BandGather.elements(geometry) == Buffer.length(~Element,input) : Nat},ready: Buffer.Sized<Element,input>}

def single_length(~Element: Data,+images: U32,+channels: U32,+height: U32,+width: U32,+input: Buffer.Model<Element>,one: {images == 1 : U32},
  expected_length: {BandGather.elements(Bands.Geometry{images,channels,height,width}) == Buffer.length(~Element,input) : Nat}) ->
  {Buffer.length(~Element,input) == BandValues.cell_count(Bands.Geometry{images,channels,height,width},height) : Nat}:
  +whole = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))
  Equal.trans(Nat,Buffer.length(~Element,input),Nat.add(whole,0n),whole,
    Equal.sym(Nat,Nat.add(whole,0n),Buffer.length(~Element,input),
      %one : {Nat.mul(U32.to_nat(_),whole) == Buffer.length(~Element,input) : Nat}
      expected_length),
    Addition.zero_right(whole))

# A leaf over the whole image is the whole buffer relabelled: its content,
# readiness and metadata are the buffer's.
def whole_values(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +held: {Bands.whole_image(geometry,first,rows) == True{} : Bool},owned: Owned<Element,geometry,input>) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),Buffer.tree(~Element,input),geometry,
    Model.RowBand{Buffer.contents(~Element,Buffer.pack(~Element,input)),first,rows}):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Buffer.Model{+tree,+length,+certificate} = input
  Owned{expected_length,ready} = owned
  %Equal.sym(U32,first,0,held_first(images,first,rows,height,held)) :
    BandValues.band_values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),tree,Bands.Geometry{images,channels,height,width},_,rows,Buffer.contents(~Element,Buffer.pack(~Element,input)))
  %Equal.sym(U32,rows,height,held_rows(images,first,rows,height,held)) :
    BandValues.band_values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),tree,Bands.Geometry{images,channels,height,width},0,_,Buffer.contents(~Element,Buffer.pack(~Element,input)))
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.content(~Element,input),Buffer.packed_contents(~Element,input)) :
    BandValues.band_values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),tree,Bands.Geometry{images,channels,height,width},0,height,_)
  BandValues.whole_leaf(~Element,images,channels,height,width,tree,length,
    single_length(~Element,images,channels,height,width,input,held_batch(images,first,rows,height,held),expected_length))

def whole_shape(~Element: Data,+geometry: Bands.Geometry,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+division_capacity: U32,
  +held: {Bands.whole_image(geometry,first,rows) == True{} : Bool},owned: Owned<Element,geometry,input>,
  fits: Fits(geometry,Bands.Rows{first,rows},Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  BandShape.Contents(~Element,geometry,Model.RowBand{Buffer.contents(~Element,Buffer.pack(~Element,input)),first,rows}):
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  Buffer.Model{+tree,+length,+certificate} = input
  Owned{expected_length,ready} = owned
  LeafFacts{allocation,contents,metadata} = fits
  BandShape.Fields{local,rows_fit,count_exact} = metadata
  +length_exact = single_length(~Element,images,channels,height,width,input,held_batch(images,first,rows,height,held),expected_length)
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.content(~Element,input),Buffer.packed_contents(~Element,input)) :
    BandShape.band_fields(~Element,Bands.Geometry{images,channels,height,width},first,rows,_)
  BandShape.Fields{local,rows_fit,
    %Equal.sym(U32,rows,height,held_rows(images,first,rows,height,held)) : {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(_),U32.to_nat(width))) : Nat}
    length_exact}

def whole_ready(~Element: Data,+geometry: Bands.Geometry,+input: Buffer.Model<Element>,owned: Owned<Element,geometry,input>,+first: U32,+rows: U32) ->
  Model.ContentReady(~Element,Model.RowBand{Buffer.contents(~Element,Buffer.pack(~Element,input)),first,rows}):
  Owned{expected_length,ready} = owned
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,input)),Buffer.content(~Element,input),Buffer.packed_contents(~Element,input)) : Buffer.ContentReady<Element,_>
  Buffer.content_ready(~Element,input,ready)

# One leaf cut from a whole buffer: the buffer relabelled when the leaf holds
# the whole image, otherwise a copy.
def rows_layout(~Element: Data,whole: Bool,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>) ->
  {Model.content_layout(~Element,Model.contents(~Element,Bands.from_rows(~Element,whole,geometry,fill,first,rows,Buffer.pack(~Element,input)))) == Bands.Rows{first,rows} : Bands.Plan}:
  match whole:
    case True{}: {==}
    case False{}: copied_layout(~Element,geometry,fill,Bands.Rows{first,rows},input)

def rows_values(~Element: Data,whole: Bool,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+division_capacity: U32,
  held: {Bands.whole_image(geometry,first,rows) == whole : Bool},owned: Owned<Element,geometry,input>,
  fits: Fits(geometry,Bands.Rows{first,rows},Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),Buffer.tree(~Element,input),geometry,
    Model.contents(~Element,Bands.from_rows(~Element,whole,geometry,fill,first,rows,Buffer.pack(~Element,input)))):
  match whole:
    case True{}: whole_values(~Element,geometry,first,rows,input,held,owned)
    case False{}: copied_values(~Element,geometry,fill,Bands.Rows{first,rows},input,division_capacity,fits)

def rows_ready(~Element: Data,whole: Bool,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+division_capacity: U32,
  owned: Owned<Element,geometry,input>,fits: Fits(geometry,Bands.Rows{first,rows},Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  Model.ContentReady(~Element,Model.contents(~Element,Bands.from_rows(~Element,whole,geometry,fill,first,rows,Buffer.pack(~Element,input)))):
  match whole:
    case True{}: whole_ready(~Element,geometry,input,owned,first,rows)
    case False{}: copied_ready(~Element,geometry,fill,Bands.Rows{first,rows},input,division_capacity,fits)

def rows_shape(~Element: Data,whole: Bool,+geometry: Bands.Geometry,+fill: Element,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+division_capacity: U32,
  held: {Bands.whole_image(geometry,first,rows) == whole : Bool},owned: Owned<Element,geometry,input>,
  fits: Fits(geometry,Bands.Rows{first,rows},Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  BandShape.Contents(~Element,geometry,Model.contents(~Element,Bands.from_rows(~Element,whole,geometry,fill,first,rows,Buffer.pack(~Element,input)))):
  match whole:
    case True{}: whole_shape(~Element,geometry,first,rows,input,division_capacity,held,owned,fits)
    case False{}: copied_shape(~Element,geometry,fill,Bands.Rows{first,rows},input,division_capacity,fits)

# The bands of a whole buffer cut by a plan.
def from_whole_layout(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>) ->
  {Model.content_layout(~Element,Model.contents(~Element,Bands.from_whole(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))) == layout : Bands.Plan}:
  match layout:
    case Bands.Rows{+first,+rows}: rows_layout(~Element,Bands.whole_image(geometry,first,rows),geometry,fill,first,rows,input)
    case Bands.Split{+left,+right}: copied_layout(~Element,geometry,fill,Bands.Split{left,right},input)

def semantics_from_whole(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>,+division_capacity: U32,
  owned: Owned<Element,geometry,input>,fits: Fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  BandValues.Values(~Element,~Tree.Tree<Element>,~BandValues.stored_value(~Element),Buffer.tree(~Element,input),geometry,
    Model.contents(~Element,Bands.from_whole(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))):
  match layout:
    case Bands.Rows{+first,+rows}: rows_values(~Element,Bands.whole_image(geometry,first,rows),geometry,fill,first,rows,input,division_capacity,{==},owned,fits)
    case Bands.Split{+left,+right}: copied_values(~Element,geometry,fill,Bands.Split{left,right},input,division_capacity,fits)

def from_whole_ready(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>,+division_capacity: U32,
  owned: Owned<Element,geometry,input>,fits: Fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  Model.ContentReady(~Element,Model.contents(~Element,Bands.from_whole(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))):
  match layout:
    case Bands.Rows{+first,+rows}: rows_ready(~Element,Bands.whole_image(geometry,first,rows),geometry,fill,first,rows,input,division_capacity,owned,fits)
    case Bands.Split{+left,+right}: copied_ready(~Element,geometry,fill,Bands.Split{left,right},input,division_capacity,fits)

def from_whole_shape(~Element: Data,+geometry: Bands.Geometry,+fill: Element,+layout: Bands.Plan,+input: Buffer.Model<Element>,+division_capacity: U32,
  owned: Owned<Element,geometry,input>,fits: Fits(geometry,layout,Tree.capacity(Element,Buffer.tree(~Element,input)),division_capacity)) ->
  BandShape.Contents(~Element,geometry,Model.contents(~Element,Bands.from_whole(~Element,geometry,fill,layout,Buffer.pack(~Element,input)))):
  match layout:
    case Bands.Rows{+first,+rows}: rows_shape(~Element,Bands.whole_image(geometry,first,rows),geometry,fill,first,rows,input,division_capacity,{==},owned,fits)
    case Bands.Split{+left,+right}: copied_shape(~Element,geometry,fill,Bands.Split{left,right},input,division_capacity,fits)

# Replace a whole-tensor allocation envelope with the actual whole owner's
# capacity while retaining the independently proved machine envelope.
def owned(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,+input: Buffer.Model<Element>,
  expected_length: {BandGather.elements(geometry) == Buffer.length(~Element,input) : Nat},ready: Buffer.Sized<Element,input>,whole: BandGather.Domain<geometry,limits,capacity>) ->
  BandGather.Domain<geometry,limits,Tree.capacity(Element,Buffer.tree(~Element,input))>:
  match ready whole:
    case Buffer.Sized{+depth_ok,room} BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room}:
      BandGather.Domain{Buffer.capacity_bound(~Element,input,depth_ok),channels_positive,height_positive,width_positive,machine_room,
        %Equal.sym(Nat,BandGather.elements(geometry),Buffer.length(~Element,input),expected_length) :
          {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,input)))) == True{} : Bool}
        room}

# A checked existing owner supplies the target allocation and copy premises.
# Empty row leaves require no address; positive leaves use actual metadata.
def positive(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+capacity: U32,
  +domain: BandGather.Domain<geometry,limits,capacity>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),+ready: Buffer.Sized<Element,input>,
  rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},counted: BandShape.Count<geometry,rows,Buffer.length(~Element,input)>) ->
  LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>:
  BandShape.Count{+exact,allocatable} = counted
  LeafFacts{allocatable,
    Inr{
      %Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),Buffer.length(~Element,input),exact) :
        BandGather.Facts<BandGather.Call{geometry,first,rows,_,capacity,Machine.half_word(limits)}>
      BandGather.from_metadata(~Element,geometry,limits,first,rows,input,capacity,domain,metadata,rows_positive,ready)},
    %Equal.sym(Nat,U32.to_nat(Bands.local_count(geometry,rows)),Buffer.length(~Element,input),exact) : BandShape.Leaf(geometry,first,rows,_)
    metadata}

def selected(~Element: Data,empty: Bool,+geometry: Bands.Geometry,+limits: Machine.Limits,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+capacity: U32,
  compared: {U32.is_zero(rows) == empty : Bool},+domain: BandGather.Domain<geometry,limits,capacity>,+metadata: BandShape.Leaf(geometry,first,rows,Buffer.length(~Element,input)),
  +ready: Buffer.Sized<Element,input>) -> LeafFacts<geometry,first,rows,capacity,Machine.half_word(limits)>:
  match empty geometry:
    case True{} Bands.Geometry{images,channels,height,width}:
      BandShape.Fields{local,rows_fit,length_exact} = metadata
      empty_leaf(geometry,first,rows,capacity,Machine.half_word(limits),Equal.cong(U32,Nat,U32.to_nat,rows,0,Arithmetic.equal_of_is_eq(rows,0,compared)),rows_fit)
    case False{} Bands.Geometry{+images,+channels,+height,+width}:
      BandGather.Domain{capacity_ok,channels_positive,height_positive,+width_positive,machine_room,allocation_room} = domain
      +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
      positive(~Element,geometry,limits,first,rows,input,capacity,domain,metadata,ready,Arithmetic.word_nonzero_positive(rows,compared),
        BandShape.count(~Element,geometry,first,rows,input,width_positive,metadata,ready))

type Owner is Data:
  Owner{geometry: Bands.Geometry,limits: Machine.Limits,capacity: U32,domain: BandGather.Domain<geometry,limits,capacity>}

def owner_fields(~Element: Data,+owner: Owner) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Owner{geometry,limits,capacity,domain} = owner
  BandShape.band_fields(~Element,geometry)

def owner_room(+owner: Owner) -> Room:
  Owner{geometry,limits,capacity,domain} = owner
  Room{geometry,capacity,Machine.half_word(limits)}

def target_band(~Element: Data,+owner: Owner,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+ready: Buffer.Sized<Element,input>,
  +metadata: owner_fields(~Element,owner,first,rows,Buffer.content(~Element,input))) -> fits_band(owner_room(owner),first,rows):
  Owner{+geometry,+limits,+capacity,+domain} = owner
  Buffer.Model{+tree,+length,+certificate} = input
  +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
  selected(~Element,U32.is_zero(rows),geometry,limits,first,rows,input,capacity,{==},domain,metadata,ready)

def target(~Element: Data,+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,bands: Model.Model<Element>,+domain: BandGather.Domain<geometry,limits,capacity>,
  metadata: BandShape.Contents(~Element,geometry,Model.content(~Element,bands)),ready: Model.Ready(~Element,bands)) ->
  Fits(geometry,Model.layout(~Element,bands),capacity,Machine.half_word(limits)):
  Model.ready_layout_one(~Element,~Owner,~owner_fields(~Element),~Room,~fits_band,~owner_room,~target_band(~Element),
    Owner{geometry,limits,capacity,domain},bands,ready,metadata)

# ---- A plan of bands in one image fits every geometry of those rows ----

def owner_rows(+owner: Owner) -> BandShape.RowRoom:
  Owner{geometry,limits,capacity,domain} = owner
  BandShape.row_room(geometry)

# A band whose rows lie in one image, as the bounds of that image.
def image_span(empty: Bool,+images: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,+first: U32,+rows: U32,+capacity: U32,
  compared: {U32.is_zero(rows) == empty : Bool},+span: BandShape.RowSpan<images,height,first,rows>,
  +domain: BandGather.Domain<Bands.Geometry{images,channels,height,width},limits,capacity>) ->
  BandPlan.ImageSpan<first,rows,U32.to_nat(images),U32.to_nat(height)>:
  match empty:
    case True{}: BandPlan.EmptySpan{Equal.cong(U32,Nat,U32.to_nat,rows,0,Arithmetic.equal_of_is_eq(rows,0,compared))}
    case False{}:
      BandShape.RowSpan{+local,+rows_fit} = span
      BandGather.Domain{capacity_ok,+channels_positive,+height_positive,+width_positive,+machine_room,allocation_room} = domain
      +first_nat = U32.to_nat(first)
      +rows_nat = U32.to_nat(rows)
      +height_nat = U32.to_nat(height)
      +first_inside = Bounds.nat_trans(first_nat,Nat.add(first_nat,rows_nat),Nat.mul(U32.to_nat(images),height_nat),
        %Addition.zero_right(first_nat) : {Nat.is_lt(_,Nat.add(first_nat,rows_nat)) == True{} : Bool}
        Order.offset_lt(first_nat,0n,rows_nat,Arithmetic.word_nonzero_positive(rows,compared)),rows_fit)
      +image = Nat.div(first_nat,height_nat)
      +image_inside = DivisionOrder.quotient_below(first_nat,U32.to_nat(images),height_nat,height_positive,first_inside)
      +spatial = Nat.mul(height_nat,U32.to_nat(width))
      +frame = Nat.mul(U32.to_nat(channels),spatial)
      +frame_room = Order.transitive(frame,Nat.mul(U32.to_nat(images),frame),U32.to_nat(Machine.capacity(limits)),
        %RowMajor.multiply_commutative(frame,U32.to_nat(images)) : {Nat.is_le(frame,_) == True{} : Bool}
        RowMajor.product_covers(frame,U32.to_nat(images),Order.le_lt_trans(0n,image,U32.to_nat(images),Order.zero_le(image),image_inside)),machine_room)
      +spatial_room = Order.transitive(spatial,frame,U32.to_nat(Machine.capacity(limits)),
        %RowMajor.multiply_commutative(spatial,U32.to_nat(channels)) : {Nat.is_le(spatial,_) == True{} : Bool}
        RowMajor.product_covers(spatial,U32.to_nat(channels),channels_positive),frame_room)
      +height_room = Order.transitive(height_nat,spatial,U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(height_nat,U32.to_nat(width),width_positive),spatial_room)
      +offset_span = Inspection.local_span(height,first,rows,Machine.half_word(limits),Limits.half_word_fits(limits),height_positive,
        Limits.double_bound(limits,height_nat,height_room),local)
      +upper = {
        %Equal.sym(Nat,first_nat,Nat.add(Nat.mul(image,height_nat),Nat.mod(first_nat,height_nat)),DivisionOrder.quotient_exact(first_nat,height_nat,height_positive)) :
          {Nat.is_le(Nat.add(_,rows_nat),Nat.add(Nat.mul(image,height_nat),height_nat)) == True{} : Bool}
        %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(image,height_nat),Nat.mod(first_nat,height_nat)),rows_nat),Nat.add(Nat.mul(image,height_nat),Nat.add(Nat.mod(first_nat,height_nat),rows_nat)),
          Addition.associative(Nat.mul(image,height_nat),Nat.mod(first_nat,height_nat),rows_nat)) :
          {Nat.is_le(_,Nat.add(Nat.mul(image,height_nat),height_nat)) == True{} : Bool}
        Order.offset_le(Nat.mul(image,height_nat),Nat.add(Nat.mod(first_nat,height_nat),rows_nat),height_nat,offset_span)
        : {Nat.is_le(Nat.add(first_nat,rows_nat),Nat.add(Nat.mul(image,height_nat),height_nat)) == True{} : Bool}}
      BandPlan.ImageSpan{image,image_inside,BandPlan.LeafWithin{DivisionOrder.product_below(first_nat,height_nat,height_positive),upper}}

def planned_band(+owner: Owner,+first: U32,+rows: U32,+span: BandShape.row_span(owner_rows(owner),first,rows)) -> fits_band(owner_room(owner),first,rows):
  Owner{+geometry,+limits,+capacity,+domain} = owner
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandShape.RowSpan{local,+rows_fit} = span
  bounds_leaf(images,channels,height,width,limits,first,rows,capacity,
    image_span(U32.is_zero(rows),images,channels,height,width,limits,first,rows,capacity,{==},BandShape.RowSpan{local,rows_fit},domain),
    BandPlan.LeafWithin{Order.zero_le(U32.to_nat(first)),rows_fit},domain)

# A plan whose bands lie in one image fits every geometry of those images
# and rows whose cells fit the domain.
def planned(+geometry: Bands.Geometry,+limits: Machine.Limits,+capacity: U32,layout: Bands.Plan,spans: BandShape.Spans(geometry,layout),
  +domain: BandGather.Domain<geometry,limits,capacity>) -> Fits(geometry,layout,capacity,Machine.half_word(limits)):
  Model.plan_one(~Owner,~BandShape.RowRoom,~BandShape.row_span,~owner_rows,~Room,~fits_band,~owner_room,~planned_band,Owner{geometry,limits,capacity,domain},layout,spans)

def same_room(+room: Room) -> Room:
  room

def room_rows(+room: Room) -> BandShape.RowRoom:
  Room{geometry,capacity,division_capacity} = room
  BandShape.row_room(geometry)

def fitted_band(+room: Room,+first: U32,+rows: U32,+fact: fits_band(same_room(room),first,rows)) -> BandShape.row_span(room_rows(room),first,rows):
  Room{+geometry,capacity,division_capacity} = room
  Bands.Geometry{images,channels,height,width} = geometry
  LeafFacts{allocation,contents,metadata} = fact
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  BandShape.RowSpan{local,rows_fit}

# A plan that fits the bands of a geometry lies in its images.
def fitted_spans(+geometry: Bands.Geometry,layout: Bands.Plan,+capacity: U32,+division_capacity: U32,fits: Fits(geometry,layout,capacity,division_capacity)) ->
  BandShape.Spans(geometry,layout):
  Model.plan_one(~Room,~Room,~fits_band,~same_room,~BandShape.RowRoom,~BandShape.row_span,~room_rows,~fitted_band,Room{geometry,capacity,division_capacity},layout,fits)

# Empty band conversions need no address envelope: the rows of a band and a
# zero count of its cells give its facts.
def empty_count_leaf(+images: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+capacity: U32,+division_capacity: U32,
  +native_zero: {Bands.local_count(Bands.Geometry{images,channels,height,width},rows) == 0 : U32},
  length_zero: {Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) == 0n : Nat},span: BandShape.RowSpan<images,height,first,rows>) ->
  LeafFacts<Bands.Geometry{images,channels,height,width},first,rows,capacity,division_capacity>:
  BandShape.RowSpan{local,rows_fit} = span
  +geometry = {Bands.Geometry{images,channels,height,width} : Bands.Geometry}
  +count_zero = Equal.cong(U32,Nat,U32.to_nat,Bands.local_count(geometry,rows),0,native_zero)
  LeafFacts{
    %Equal.sym(U32,Bands.local_count(geometry,rows),0,native_zero) : {U32.is_le(_,1073741824) == True{} : Bool} {==},
    Inl{count_zero},
    BandShape.Fields{local,rows_fit,Equal.trans(Nat,U32.to_nat(Bands.local_count(geometry,rows)),0n,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),
      count_zero,Equal.sym(Nat,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),0n,length_zero))}}

def no_rows_natural(+first: Nat,+rows: Nat,bound: {Nat.is_le(Nat.add(first,rows),0n) == True{} : Bool}) -> {rows == 0n : Nat}:
  match first rows:
    case 0n 0n: {==}
    case 0n 1n+rows: Empty.absurd({1n+rows == 0n : Nat},WordEncoding.false_true(bound))
    case 1n+first rows: Empty.absurd({rows == 0n : Nat},WordEncoding.false_true(bound))

def no_rows_leaf(+geometry: Bands.Geometry,+first: U32,+rows: U32,+capacity: U32,+division_capacity: U32,
  total_zero: {Inspection.total_rows(geometry) == 0n : Nat},span: BandShape.row_span(BandShape.row_room(geometry),first,rows)) ->
  LeafFacts<geometry,first,rows,capacity,division_capacity>:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandShape.RowSpan{local,+rows_fit} = span
  empty_leaf(Bands.Geometry{images,channels,height,width},first,rows,capacity,division_capacity,
    no_rows_natural(U32.to_nat(first),U32.to_nat(rows),%total_zero : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool} rows_fit),rows_fit)

# A tensor without elements: no width, no channels or no rows.
type Vacant is Data:
  NoWidth{images: U32,channels: U32,height: U32,capacity: U32,division_capacity: U32}
  NoChannels{images: U32,height: U32,width: U32,capacity: U32,division_capacity: U32}
  NoRows{geometry: Bands.Geometry,capacity: U32,division_capacity: U32,total_zero: {Inspection.total_rows(geometry) == 0n : Nat}}

def vacant_room(+vacant: Vacant) -> Room:
  match vacant:
    case NoWidth{images,channels,height,capacity,division_capacity}: Room{Bands.Geometry{images,channels,height,0},capacity,division_capacity}
    case NoChannels{images,height,width,capacity,division_capacity}: Room{Bands.Geometry{images,0,height,width},capacity,division_capacity}
    case NoRows{geometry,capacity,division_capacity,total_zero}: Room{geometry,capacity,division_capacity}

def vacant_rows(+vacant: Vacant) -> BandShape.RowRoom:
  match vacant:
    case NoWidth{images,channels,height,capacity,division_capacity}: BandShape.RowRoom{images,height}
    case NoChannels{images,height,width,capacity,division_capacity}: BandShape.RowRoom{images,height}
    case NoRows{geometry,capacity,division_capacity,total_zero}: BandShape.row_room(geometry)

def vacant_span(+vacant: Vacant,+first: U32,+rows: U32,+span: BandShape.row_span(vacant_rows(vacant),first,rows)) -> fits_band(vacant_room(vacant),first,rows):
  match vacant:
    case NoWidth{+images,+channels,+height,+capacity,+division_capacity}:
      empty_count_leaf(images,channels,height,0,first,rows,capacity,division_capacity,BandCount.width_native(images,channels,height,rows),
        %Equal.sym(Nat,Nat.mul(U32.to_nat(rows),0n),0n,RowMajor.multiply_zero_right(U32.to_nat(rows))) : {Nat.mul(U32.to_nat(channels),_) == 0n : Nat}
        RowMajor.multiply_zero_right(U32.to_nat(channels)),span)
    case NoChannels{+images,+height,+width,+capacity,+division_capacity}:
      empty_count_leaf(images,0,height,width,first,rows,capacity,division_capacity,BandCount.channels_native(images,height,width,rows),{==},span)
    case NoRows{+geometry,+capacity,+division_capacity,+total_zero}: no_rows_leaf(geometry,first,rows,capacity,division_capacity,total_zero,span)

# A plan whose bands lie in the batch fits a geometry without elements.
def vacant(+vacant: Vacant,layout: Bands.Plan,spans: Model.PlanAll(~BandShape.RowRoom,~BandShape.row_span,vacant_rows(vacant),layout)) ->
  Model.PlanAll(~Room,~fits_band,vacant_room(vacant),layout):
  Model.plan_one(~Vacant,~BandShape.RowRoom,~BandShape.row_span,~vacant_rows,~Room,~fits_band,~vacant_room,~vacant_span,vacant,layout,spans)

def width(+images: U32,+channels: U32,+height: U32,layout: Bands.Plan,+capacity: U32,+division_capacity: U32,
  spans: BandShape.Spans(Bands.Geometry{images,channels,height,0},layout)) -> Fits(Bands.Geometry{images,channels,height,0},layout,capacity,division_capacity):
  vacant(NoWidth{images,channels,height,capacity,division_capacity},layout,spans)

def channels(+images: U32,+height: U32,+width: U32,layout: Bands.Plan,+capacity: U32,+division_capacity: U32,
  spans: BandShape.Spans(Bands.Geometry{images,0,height,width},layout)) -> Fits(Bands.Geometry{images,0,height,width},layout,capacity,division_capacity):
  vacant(NoChannels{images,height,width,capacity,division_capacity},layout,spans)

def no_rows(+geometry: Bands.Geometry,layout: Bands.Plan,+capacity: U32,+division_capacity: U32,+total_zero: {Inspection.total_rows(geometry) == 0n : Nat},
  spans: BandShape.Spans(geometry,layout)) -> Fits(geometry,layout,capacity,division_capacity):
  vacant(NoRows{geometry,capacity,division_capacity,total_zero},layout,spans)

# The actual row planner stops before creating empty children. Nonempty
# images therefore provide the positive-core premise of halo preparation.
type SplitFacts<-count: U32,-capacity: U32> is Data:
  SplitFacts{left_positive: {Nat.is_lt(0n,U32.to_nat((count / 2 : U32))) == True{} : Bool},
    right_positive: {Nat.is_lt(0n,U32.to_nat((count - count / 2 : U32))) == True{} : Bool},
    left_room: {Nat.is_le(U32.to_nat((count / 2 : U32)),U32.to_nat(capacity)) == True{} : Bool},
    right_room: {Nat.is_le(U32.to_nat((count - count / 2 : U32)),U32.to_nat(capacity)) == True{} : Bool}}

def plan_nonempty_split(+count: U32,+capacity: U32,capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},+room: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool},
  enabled: {U32.is_lt(1,count) == True{} : Bool}) -> SplitFacts<count,capacity>:
  +half = (count / 2 : U32)
  +quotient = Nat.div(U32.to_nat(count),2n)
  +decoded = U32Division.divide(count,2,2147483648,{==},{==},{==})
  +at_least_two = Order.strict_follow(1n,U32.to_nat(count),{%Comparison.lt(1,count) : {_ == True{} : Bool} enabled : {Nat.is_lt(1n,U32.to_nat(count)) == True{} : Bool}})
  +quotient_positive = Bounds.predecessor_strict(0n,quotient,DivisionOrder.quotient_le(2n,U32.to_nat(count),2n,{==},at_least_two))
  +half_inside = BandPlan.half_bound(count)
  +quotient_less = Bounds.nat_trans(quotient,Nat.mul(quotient,2n),U32.to_nat(count),
    %Equal.sym(Nat,Nat.mul(quotient,2n),Nat.mul(2n,quotient),RowMajor.multiply_commutative(quotient,2n)) : {Nat.is_lt(quotient,_) == True{} : Bool}
    %Addition.zero_right(quotient) : {Nat.is_lt(_,Nat.mul(2n,quotient)) == True{} : Bool}
    RowMajor.multiply_lt(1n,2n,quotient,{==},quotient_positive),DivisionOrder.product_below(U32.to_nat(count),2n,{==}))
  +remainder_room = Order.transitive(Nat.sub(U32.to_nat(count),U32.to_nat(half)),U32.to_nat(count),U32.to_nat(capacity),Order.sub_le(U32.to_nat(count),U32.to_nat(half)),room)
  +remainder_exact = Arithmetic.subtract(count,half,capacity,capacity_ok,half_inside,remainder_room)
  SplitFacts{
    %Equal.sym(Nat,U32.to_nat(half),quotient,decoded) : {Nat.is_lt(0n,_) == True{} : Bool} quotient_positive,
    %Equal.sym(Nat,U32.to_nat((count - half : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(half)),remainder_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.positive_difference(U32.to_nat(count),U32.to_nat(half),{%Equal.sym(Nat,U32.to_nat(half),quotient,decoded) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} quotient_less : {Nat.is_lt(U32.to_nat(half),U32.to_nat(count)) == True{} : Bool}}),
    Order.transitive(U32.to_nat(half),U32.to_nat(count),U32.to_nat(capacity),half_inside,room),
    %Equal.sym(Nat,U32.to_nat((count - half : U32)),Nat.sub(U32.to_nat(count),U32.to_nat(half)),remainder_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} remainder_room}

def children(+depth: Nat,+first: U32,+count: U32,+capacity: U32,facts: SplitFacts<count,capacity>,
  left: {Nat.is_lt(0n,U32.to_nat((count / 2 : U32))) == True{} : Bool} -> {Nat.is_le(U32.to_nat((count / 2 : U32)),U32.to_nat(capacity)) == True{} : Bool} -> BandPlan.Nonempty(Bands.row_plan(depth,U32.is_lt(1,(count / 2 : U32)),first,(count / 2 : U32))),
  right: {Nat.is_lt(0n,U32.to_nat((count - count / 2 : U32))) == True{} : Bool} -> {Nat.is_le(U32.to_nat((count - count / 2 : U32)),U32.to_nat(capacity)) == True{} : Bool} -> BandPlan.Nonempty(Bands.row_plan(depth,U32.is_lt(1,(count - count / 2 : U32)),(first + count / 2 : U32),(count - count / 2 : U32)))) ->
  BandPlan.Nonempty(Bands.row_plan(1n+depth,True{},first,count)):
  SplitFacts{left_positive,right_positive,left_room,right_room} = facts
  Shape.Both{left(left_positive,left_room),right(right_positive,right_room)}

def rows(depth: Nat,+enabled: Bool,+first: U32,+count: U32,+capacity: U32,+capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  room: {Nat.is_le(U32.to_nat(count),U32.to_nat(capacity)) == True{} : Bool},positive: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  comparison: {U32.is_lt(1,count) == enabled : Bool}) -> BandPlan.Nonempty(Bands.row_plan(depth,enabled,first,count)):
  match depth enabled:
    case 0n enabled: positive
    case 1n+rest False{}: positive
    case 1n+ +rest True{}:
      children(rest,first,count,capacity,plan_nonempty_split(count,capacity,capacity_ok,room,comparison),
        positive => room => rows(rest,U32.is_lt(1,(count / 2 : U32)),first,(count / 2 : U32),capacity,capacity_ok,room,positive,{==}),
        positive => room => rows(rest,U32.is_lt(1,(count - count / 2 : U32)),(first + count / 2 : U32),(count - count / 2 : U32),capacity,capacity_ok,room,positive,{==}))
