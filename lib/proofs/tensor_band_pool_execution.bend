# One band of a band pool. The band assembles the rows its outputs read into
# one window and runs the checked leaf pool on it. The leaf pool theorem
# gives every output cell as the fold of its window samples. Two facts of the
# band carry it to the whole input: every window cell holds the input at its
# pixel, and every row a tap reads is valid exactly when the image row is,
# and then it is that row.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_band_pool.bend as BandPool
import ../tensor_band_window.bend as BandWindow
import ../tensor_access.bend as Access
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ../machine_limits.bend as Machine
import ../default_machine_limits.bend as Default
import ./storage_buffer_model.bend as Buffer
import ./storage_shape.bend as Shape
import ./tensor_band_model.bend as Model
import ./tensor_band_each.bend as Each
import ./tensor_band_halo.bend as BandHalo
import ./tensor_band_window.bend as TensorBandWindow
import ./tensor_band_values.bend as BandValues
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_count.bend as BandCount
import ./tensor_sweep_values.bend as Sweeps
import ./tensor_pool_leaf_values.bend as PoolLeaf
import ./tensor_contract.bend as Contract
import ./tensor_gather_values.bend as GatherValues
import ./tensor_copy_values.bend as CopyValues
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_allocation_depth.bend as Depth
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./encoded_coordinates.bend as Encoded
import ./machine_limit_bounds.bend as Limits
import ./word_encoding.bend as WordEncoding
import ./storage_tree_model.bend as Tree
import ./ordered_fold_refinement.bend as Fold
import ./nat_to_u32_bounds.bend as Bounds
import ./boolean_logic.bend as Boolean
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges

# ---- The window of a band ----

# The output rows a window writes, as a word count of cells.
def window_count(geometry: BandWindow.Geometry,window: BandWindow.RowWindow) -> U32:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{first,rows},above,below,window} = window
  (channels * rows * output_width : U32)

# An active output band fits in one image; a positive batch includes that
# image.
def rows_elements(+channels: Nat,+rows: Nat,+height: Nat,+width: Nat,
  within: {Nat.is_le(rows,height) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(channels,Nat.mul(rows,width)),Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(channels,Nat.mul(rows,width)),Nat.mul(Nat.mul(rows,width),channels),RowMajor.multiply_commutative(channels,Nat.mul(rows,width))) :
    {Nat.is_le(_,Nat.mul(channels,Nat.mul(height,width))) == True{} : Bool}
  %Equal.sym(Nat,Nat.mul(channels,Nat.mul(height,width)),Nat.mul(Nat.mul(height,width),channels),RowMajor.multiply_commutative(channels,Nat.mul(height,width))) :
    {Nat.is_le(Nat.mul(Nat.mul(rows,width),channels),_) == True{} : Bool}
  RowMajor.mul_le(Nat.mul(rows,width),Nat.mul(height,width),channels,RowMajor.mul_le(rows,height,width,within))

def local_elements(+images: Nat,+channels: Nat,+height: Nat,+width: Nat,+rows: Nat,
  positive: {Nat.is_lt(0n,images) == True{} : Bool},within: {Nat.is_le(rows,height) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(channels,Nat.mul(rows,width)),Nat.mul(images,Nat.mul(channels,Nat.mul(height,width)))) == True{} : Bool}:
  Order.transitive(Nat.mul(channels,Nat.mul(rows,width)),Nat.mul(channels,Nat.mul(height,width)),Nat.mul(images,Nat.mul(channels,Nat.mul(height,width))),
    rows_elements(channels,rows,height,width,within),DivisionOrder.le_multiple(images,Nat.mul(channels,Nat.mul(height,width)),positive))

# The output geometry whose count bounds every window's fresh destination.
def allocation_guards_output_geometry(geometry: BandWindow.Geometry) -> Bands.Geometry:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  Bands.Geometry{images,channels,output_height,output_width}

def count_bounded(+images: U32,+channels: U32,+height: U32,+width: U32,+rows: U32,+limits: Machine.Limits,
  positive_batch: {Nat.is_lt(0n,U32.to_nat(images)) == True{} : Bool},width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  rows_inside: {Nat.is_le(U32.to_nat(rows),U32.to_nat(height)) == True{} : Bool},
  global_room: {Nat.is_le(Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.is_le((channels * rows * width : U32),1073741824) == True{} : Bool}:
  +elements = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))
  +room = Order.transitive(elements,Nat.mul(U32.to_nat(images),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits)),
    local_elements(U32.to_nat(images),U32.to_nat(channels),U32.to_nat(height),U32.to_nat(width),U32.to_nat(rows),positive_batch,rows_inside),global_room)
  +exact = Equal.trans(Nat,U32.to_nat((channels * rows * width : U32)),Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),U32.to_nat(width)),elements,
    Encoded.multiply_three(channels,rows,width,Machine.capacity(limits),Limits.capacity_fits(limits),width_positive,
      %Equal.sym(Nat,Nat.mul(Nat.mul(U32.to_nat(channels),U32.to_nat(rows)),U32.to_nat(width)),elements,RowMajor.assoc(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width))) :
        {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      room),RowMajor.assoc(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width)))
  Arithmetic.count_within((channels * rows * width : U32),elements,Machine.capacity(limits),1073741824,exact,room,Limits.capacity_allocatable(limits))

def images(geometry: BandWindow.Geometry) -> U32:
  BandWindow.Geometry{Bands.Geometry{images,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  images

def from_window(+geometry: BandWindow.Geometry,+window: BandWindow.RowWindow,+limits: Machine.Limits,
  domain: BandGather.Domain<allocation_guards_output_geometry(geometry),limits,Machine.capacity(limits)>,
  positive_batch: {Nat.is_lt(0n,U32.to_nat(images(geometry))) == True{} : Bool},
  rows_inside: {Nat.is_le(U32.to_nat(TensorBandWindow.output_rows(window)),U32.to_nat(TensorBandWindow.output_height(geometry))) == True{} : Bool}) ->
  Depth.Sized<window_count(geometry,window)>:
  match geometry window:
    case BandWindow.Geometry{Bands.Geometry{+images,+channels,height,width},kernel,stride,padding,+output_height,+output_width}
      BandWindow.RowWindow{BandWindow.Rows{out_first,+out_rows},above,below,held}:
      BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
      Depth.bounded((channels * out_rows * output_width : U32),count_bounded(images,channels,output_height,output_width,out_rows,limits,positive_batch,width_positive,rows_inside,machine_room))

def active(+geometry: BandWindow.Geometry,+first: U32,+rows: U32,+limits: Machine.Limits,
  domain: BandGather.Domain<allocation_guards_output_geometry(geometry),limits,Machine.capacity(limits)>,positive_batch: {Nat.is_lt(0n,U32.to_nat(images(geometry))) == True{} : Bool}) ->
  Depth.Sized<window_count(geometry,BandWindow.window(geometry,first,rows))>:
  from_window(geometry,BandWindow.window(geometry,first,rows),limits,domain,positive_batch,TensorBandWindow.bounded(geometry,first,rows))

# The input read through the two sweeps of a whole pool.
def global_context(~Element: Data,~Logical: Data,logical: Logical,geometry: BandWindow.Geometry,+pad: Element) -> Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  PoolLeaf.context(~Element,~Logical,logical,Access.Pool{height,width,kernel,stride,padding,output_height,output_width},0,pad)

def values_output_geometry(geometry: BandWindow.Geometry) -> Bands.Geometry:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  Bands.Geometry{batch,channels,output_height,output_width}

# The column-swept planes of the whole input.
def planes_geometry(geometry: BandWindow.Geometry) -> Bands.Geometry:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  Bands.Geometry{batch,channels,height,output_width}

# The row of the assembled window that the first tap of the first output row
# of a band reads, before padding.
def window_origin(geometry: BandWindow.Geometry,+first: U32,window: BandWindow.RowWindow) -> U32:
  BandWindow.Geometry{Bands.Geometry{batch,channels,+height,width},kernel,+stride,padding,+output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+output_first,output_rows},above,below,BandWindow.Rows{+window_first,window_rows}} = window
  ((output_first - first / height * output_height) * stride + first / height * height - window_first : U32)

def assembled_input(~Element: Data,+geometry: BandWindow.Geometry,fill: Element,first: U32,rows: U32,window: BandWindow.RowWindow,
  core: Buffer.Content<Element>,above: Buffer.Content<Element>,below: Buffer.Content<Element>) -> Buffer.Content<Element>:
  BandWindow.RowWindow{output,above_range,below_range,window_range} = window
  TensorBandWindow.assembly_content(~Element,BandWindow.input_geometry(geometry),fill,window_range,BandWindow.Rows{first,rows},above_range,below_range,core,above,below)

# The planes a window sweeps along the columns: one row per assembled row.
def middle_count(geometry: BandWindow.Geometry,window: BandWindow.RowWindow) -> U32:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{output,above,below,BandWindow.Rows{first,rows}} = window
  (channels * rows * output_width : U32)

def middle_cells(geometry: BandWindow.Geometry,window: BandWindow.RowWindow) -> Nat:
  BandWindow.RowWindow{output,above,below,BandWindow.Rows{first,+rows}} = window
  BandValues.cell_count(planes_geometry(geometry),rows)

# The number of cells of the output band of a window.
def window_cells(geometry: BandWindow.Geometry,window: BandWindow.RowWindow) -> Nat:
  BandWindow.RowWindow{BandWindow.Rows{first,+rows},above,below,input} = window
  BandValues.cell_count(values_output_geometry(geometry),rows)

# ---- The rows a window reads ----

# One tap of an output row: the window row it reads is valid exactly when the
# row of the image is, and then it is that row of the image.
def RowTap(+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+height: U32,+image: Nat,+local: U32,+global: U32,+tap: U32) -> Data:
  Shape.Both<{U32.is_lt((local * stride + origin + tap - padding : U32),window_rows) == U32.is_lt((global * stride + 0 + tap - padding : U32),height) : Bool},
    Boolean.When(U32.is_lt((global * stride + 0 + tap - padding : U32),height),
      {Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat((global * stride + 0 + tap - padding : U32))) == Nat.add(U32.to_nat(window_first),U32.to_nat((local * stride + origin + tap - padding : U32))) : Nat})>

def RowTaps(+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+height: U32,+image: Nat,+local: U32,+global: U32,remaining: Nat,+tap: U32) -> Data:
  match remaining:
    case 0n: Unit
    case 1n+rest: Shape.Both<RowTap(stride,padding,origin,window_first,window_rows,height,image,local,global,tap),RowTaps(stride,padding,origin,window_first,window_rows,height,image,local,global,rest,(tap + 1 : U32))>

# The taps of one output row of a band: the row inside the band and the row
# inside its image.
def RowTapsAt(+kernel: U32,+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+height: U32,+out_height: U32,+output_first: U32,+row: Nat) -> Data:
  RowTaps(stride,padding,origin,window_first,window_rows,height,Nat.div(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height)),U32.from_nat(row),
    U32.from_nat(Nat.mod(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height))),U32.to_nat(kernel),0)

def RowFacts(+kernel: U32,+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+height: U32,+out_height: U32,+output_first: U32,+output_rows: U32) -> Data:
  Members.All(Nat,{row => RowTapsAt(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,row) : Nat -> Data},Ranges.range(U32.to_nat(output_rows),0n))

# The bounds of a pooled window: its output rows fit in the batch, and the
# input, the column-swept planes and the output fit the machine capacity, for
# the whole batch and for the rows the window holds.
type Rooms<-batch: U32,-channels: U32,-height: U32,-width: U32,-out_height: U32,-out_width: U32,-window_rows: U32,-output_first: U32,-output_rows: U32,-limits: Machine.Limits> is Data:
  Rooms{output_fit: {Nat.is_le(Nat.add(U32.to_nat(output_first),U32.to_nat(output_rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(out_height))) == True{} : Bool},
    out_height_positive: {Nat.is_lt(0n,U32.to_nat(out_height)) == True{} : Bool},
    input_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    planes_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(height),U32.to_nat(out_width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    output_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(out_height),U32.to_nat(out_width)))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    local_input_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    local_planes_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(window_rows),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    local_output_room: {Nat.is_le(Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(output_rows),U32.to_nat(out_width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

def WindowRooms(geometry: BandWindow.Geometry,window: BandWindow.RowWindow,limits: Machine.Limits) -> Data:
  BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{output_first,output_rows},above,below,BandWindow.Rows{window_first,window_rows}} = window
  Rooms<batch,channels,height,width,output_height,output_width,window_rows,output_first,output_rows,limits>

# For every output row of the band and every tap: the assembled row it reads
# is valid exactly when the row of the image is, and then it is that row.
def WindowRows(+geometry: BandWindow.Geometry,+first: U32,+window: BandWindow.RowWindow) -> Data:
  BandWindow.Geometry{Bands.Geometry{batch,channels,+height,width},+kernel,+stride,+padding,+output_height,output_width} = geometry
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},above,below,BandWindow.Rows{+window_first,+window_rows}} = window
  RowFacts(kernel,stride,padding,((output_first - first / height * output_height) * stride + first / height * height - window_first : U32),window_first,window_rows,height,output_height,output_first,output_rows)

def grouped(+batch: Nat,+channels: Nat,+rest: Nat,+limit: Nat,room: {Nat.is_le(Nat.mul(batch,Nat.mul(channels,rest)),limit) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(Nat.mul(batch,channels),rest),limit) == True{} : Bool}:
  %Equal.sym(Nat,Nat.mul(Nat.mul(batch,channels),rest),Nat.mul(batch,Nat.mul(channels,rest)),RowMajor.assoc(batch,channels,rest)) : {Nat.is_le(_,limit) == True{} : Bool}
  room

# ---- The window pool is the pool of the input ----

# A valid sample of a run of the window reads the window cell of its row and
# column, which holds the input at the pixel of the image row it stands for.
def window_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +window_first: U32,+window_rows: U32,+tree: Tree.Tree<Element>,+length: Nat,+channel: Nat,+image: Nat,+near: Nat,+far: Nat,valid: Bool,+column: U32,
  +exact: {U32.is_lt(column,width) == valid : Bool},
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},window_first,window_rows,tree,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+near_inside: {Nat.is_lt(near,U32.to_nat(window_rows)) == True{} : Bool},
  +far_inside: {Nat.is_lt(far,U32.to_nat(height)) == True{} : Bool},
  +located: {Nat.add(Nat.mul(image,U32.to_nat(height)),far) == Nat.add(U32.to_nat(window_first),near) : Nat}) ->
  Boolean.When(valid,{Contract.stored(~Element,Buffer.Content{tree,length},U32.from_nat(RowMajor.index(channel,U32.to_nat(window_rows),U32.to_nat(width),near,U32.to_nat(column)))) ==
    original(logical,U32.from_nat(RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),U32.to_nat(height),U32.to_nat(width),far,U32.to_nat(column)))) : Element}):
  match valid:
    case False{}: Unit{}
    case True{}:
      +source = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
      %BandValues.located_pixel(source,image,channel,far,U32.to_nat(column),Nat.add(U32.to_nat(window_first),near),far_inside,located) :
        {Tree.read(Element,tree,U32.from_nat(BandValues.position(source,window_rows,channel,near,U32.to_nat(column)))) == original(logical,U32.from_nat(_)) : Element}
      BandValues.leaf_read(~Element,~Logical,~original,logical,source,window_first,window_rows,tree,length,channel,near,U32.to_nat(column),values,channel_inside,near_inside,
        Bounds.as_nat_lt(column,width,exact))

# One column tap of a window row and of the image row it stands for.
def window_column(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+pad: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,+padding: U32,
  +window_first: U32,+window_rows: U32,+tree: Tree.Tree<Element>,+length: Nat,+first: U32,+channel: Nat,+image: Nat,+near: Nat,+far: Nat,+tap: U32,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},window_first,window_rows,tree,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+near_inside: {Nat.is_lt(near,U32.to_nat(window_rows)) == True{} : Bool},
  +far_inside: {Nat.is_lt(far,U32.to_nat(height)) == True{} : Bool},
  +located: {Nat.add(Nat.mul(image,U32.to_nat(height)),far) == Nat.add(U32.to_nat(window_first),near) : Nat}) ->
  {PoolLeaf.run_sample(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),Buffer.Content{tree,length},PoolLeaf.Run{channel,U32.to_nat(window_rows),width,near,first,padding,pad},tap) ==
    PoolLeaf.run_sample(~Element,~Logical,~original,logical,PoolLeaf.Run{Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),U32.to_nat(height),width,far,first,padding,pad},tap) : Element}:
  +column = (first + tap - padding : U32)
  Sweeps.moved(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~Logical,~original,Buffer.Content{tree,length},logical,pad,U32.is_lt(column,width),
    U32.from_nat(RowMajor.index(channel,U32.to_nat(window_rows),U32.to_nat(width),near,U32.to_nat(column))),
    U32.from_nat(RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),U32.to_nat(height),U32.to_nat(width),far,U32.to_nat(column))),
    window_read(~Element,~Logical,~original,logical,batch,channels,height,width,window_first,window_rows,tree,length,channel,image,near,far,U32.is_lt(column,width),column,{==},
      values,channel_inside,near_inside,far_inside,located))

def window_columns(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,remaining: Nat,+tap: U32,+logical: Logical,+pad: Element,+batch: U32,+channels: U32,+height: U32,+width: U32,+padding: U32,
  +window_first: U32,+window_rows: U32,+tree: Tree.Tree<Element>,+length: Nat,+first: U32,+channel: Nat,+image: Nat,+near: Nat,+far: Nat,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},window_first,window_rows,tree,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},+near_inside: {Nat.is_lt(near,U32.to_nat(window_rows)) == True{} : Bool},
  +far_inside: {Nat.is_lt(far,U32.to_nat(height)) == True{} : Bool},
  +located: {Nat.add(Nat.mul(image,U32.to_nat(height)),far) == Nat.add(U32.to_nat(window_first),near) : Nat}) ->
  Fold.Agree(~Element,~Buffer.Content<Element>,~PoolLeaf.Run<Element>,~PoolLeaf.run_sample(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
    ~Logical,~PoolLeaf.Run<Element>,~PoolLeaf.run_sample(~Element,~Logical,~original),
    Buffer.Content{tree,length},PoolLeaf.Run{channel,U32.to_nat(window_rows),width,near,first,padding,pad},
    logical,PoolLeaf.Run{Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),U32.to_nat(height),width,far,first,padding,pad},remaining,tap):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{window_column(~Element,~Logical,~original,logical,pad,batch,channels,height,width,padding,window_first,window_rows,tree,length,first,channel,image,near,far,tap,
          values,channel_inside,near_inside,far_inside,located),
        window_columns(~Element,~Logical,~original,rest,(tap + 1 : U32),logical,pad,batch,channels,height,width,padding,window_first,window_rows,tree,length,first,channel,image,near,far,
          values,channel_inside,near_inside,far_inside,located)}

# A window row read by a row tap and the image row it stands for: both valid
# or both pad, and a valid pair holds the same run.
def window_line(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+padding: U32,+window_first: U32,+window_rows: U32,+tree: Tree.Tree<Element>,+length: Nat,
  +first: U32,+channel: Nat,+image: Nat,valid: Bool,+near: U32,+far: U32,
  +exact: {U32.is_lt(far,height) == valid : Bool},
  +same: {U32.is_lt(near,window_rows) == U32.is_lt(far,height) : Bool},
  located: Boolean.When(valid,{Nat.add(Nat.mul(image,U32.to_nat(height)),U32.to_nat(far)) == Nat.add(U32.to_nat(window_first),U32.to_nat(near)) : Nat}),
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},window_first,window_rows,tree,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool}) ->
  {PoolLeaf.line_when(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation,Buffer.Content{tree,length},channel,window_rows,width,kernel,padding,first,pad,valid,near) ==
    PoolLeaf.line_when(~Element,~Logical,~original,~operation,logical,Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),height,width,kernel,padding,first,pad,valid,far) : Element}:
  match valid:
    case False{}: {==}
    case True{}:
      +far_inside = Bounds.as_nat_lt(far,height,exact)
      +near_inside = Bounds.as_nat_lt(near,window_rows,Equal.trans(Bool,U32.is_lt(near,window_rows),U32.is_lt(far,height),True{},same,exact))
      Fold.agreed(~Element,~Element,~Buffer.Content<Element>,~PoolLeaf.Run<Element>,~PoolLeaf.run_sample(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
        ~Logical,~PoolLeaf.Run<Element>,~PoolLeaf.run_sample(~Element,~Logical,~original),~operation,U32.to_nat(kernel),0,
        Buffer.Content{tree,length},PoolLeaf.Run{channel,U32.to_nat(window_rows),width,U32.to_nat(near),first,padding,pad},
        logical,PoolLeaf.Run{Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),U32.to_nat(height),width,U32.to_nat(far),first,padding,pad},pad,
        window_columns(~Element,~Logical,~original,U32.to_nat(kernel),0,logical,pad,batch,channels,height,width,padding,window_first,window_rows,tree,length,first,channel,image,
          U32.to_nat(near),U32.to_nat(far),values,channel_inside,near_inside,far_inside,located))

# One row tap of an output cell of the window and of its pixel.
def window_tap(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+tree: Tree.Tree<Element>,+length: Nat,
  +channel: Nat,+image: Nat,+row: Nat,+inner: Nat,+column: Nat,+tap: U32,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},window_first,window_rows,tree,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},
  fact: RowTap(stride,padding,origin,window_first,window_rows,height,image,U32.from_nat(row),U32.from_nat(inner),tap)) ->
  {PoolLeaf.line_sample(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation,Buffer.Content{tree,length},
      PoolLeaf.Cell{channel,window_rows,width,kernel,padding,(U32.from_nat(row) * stride + origin : U32),(U32.from_nat(column) * stride + 0 : U32),pad},tap) ==
    PoolLeaf.line_sample(~Element,~Logical,~original,~operation,logical,
      PoolLeaf.Cell{Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),height,width,kernel,padding,(U32.from_nat(inner) * stride + 0 : U32),(U32.from_nat(column) * stride + 0 : U32),pad},tap) : Element}:
  Shape.Both{+same,located} = fact
  +near = (U32.from_nat(row) * stride + origin + tap - padding : U32)
  +far = (U32.from_nat(inner) * stride + 0 + tap - padding : U32)
  +first = (U32.from_nat(column) * stride + 0 : U32)
  %Equal.sym(Bool,U32.is_lt(near,window_rows),U32.is_lt(far,height),same) :
    {PoolLeaf.line_when(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation,Buffer.Content{tree,length},channel,window_rows,width,kernel,padding,first,pad,_,near) ==
      PoolLeaf.line_when(~Element,~Logical,~original,~operation,logical,Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),height,width,kernel,padding,first,pad,U32.is_lt(far,height),far) : Element}
  window_line(~Element,~Logical,~original,~operation,logical,pad,batch,channels,height,width,kernel,padding,window_first,window_rows,tree,length,first,channel,image,
    U32.is_lt(far,height),near,far,{==},same,located,values,channel_inside)

def window_taps(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,remaining: Nat,+tap: U32,+logical: Logical,+pad: Element,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+origin: U32,+window_first: U32,+window_rows: U32,+tree: Tree.Tree<Element>,+length: Nat,
  +channel: Nat,+image: Nat,+row: Nat,+inner: Nat,+column: Nat,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},window_first,window_rows,tree,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},
  facts: RowTaps(stride,padding,origin,window_first,window_rows,height,image,U32.from_nat(row),U32.from_nat(inner),remaining,tap)) ->
  Fold.Agree(~Element,~Buffer.Content<Element>,~PoolLeaf.Cell<Element>,~PoolLeaf.line_sample(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation),
    ~Logical,~PoolLeaf.Cell<Element>,~PoolLeaf.line_sample(~Element,~Logical,~original,~operation),
    Buffer.Content{tree,length},PoolLeaf.Cell{channel,window_rows,width,kernel,padding,(U32.from_nat(row) * stride + origin : U32),(U32.from_nat(column) * stride + 0 : U32),pad},
    logical,PoolLeaf.Cell{Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),height,width,kernel,padding,(U32.from_nat(inner) * stride + 0 : U32),(U32.from_nat(column) * stride + 0 : U32),pad},
    remaining,tap):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{head,tail} = facts
      Shape.Both{window_tap(~Element,~Logical,~original,~operation,logical,pad,batch,channels,height,width,kernel,stride,padding,origin,window_first,window_rows,tree,length,
          channel,image,row,inner,column,tap,values,channel_inside,head),
        window_taps(~Element,~Logical,~original,~operation,rest,(tap + 1 : U32),logical,pad,batch,channels,height,width,kernel,stride,padding,origin,window_first,window_rows,tree,length,
          channel,image,row,inner,column,values,channel_inside,tail)}

# A cell of the window pool is the cell of the input pool at the pixel of its
# output row: the rows read range and the window content agree tap by tap.
def window_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+out_width: U32,+origin: U32,
  +window_first: U32,+window_rows: U32,+output_first: U32,+output_rows: U32,+tree: Tree.Tree<Element>,+length: Nat,+channel: Nat,+row: Nat,+column: Nat,
  +values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},window_first,window_rows,tree,length),
  +channel_inside: {Nat.is_lt(channel,U32.to_nat(channels)) == True{} : Bool},
  facts: RowTapsAt(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,row)) ->
  {PoolLeaf.at(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation,Buffer.Content{tree,length},Access.Pool{window_rows,width,kernel,stride,padding,output_rows,out_width},origin,pad,
      channel,row,column) ==
    PoolLeaf.at(~Element,~Logical,~original,~operation,logical,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},0,pad,
      Nat.add(Nat.mul(Nat.div(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height)),U32.to_nat(channels)),channel),Nat.mod(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height)),column) : Element}:
  +image = Nat.div(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height))
  +inner = Nat.mod(Nat.add(U32.to_nat(output_first),row),U32.to_nat(out_height))
  Fold.agreed(~Element,~Element,~Buffer.Content<Element>,~PoolLeaf.Cell<Element>,~PoolLeaf.line_sample(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation),
    ~Logical,~PoolLeaf.Cell<Element>,~PoolLeaf.line_sample(~Element,~Logical,~original,~operation),~operation,U32.to_nat(kernel),0,
    Buffer.Content{tree,length},PoolLeaf.Cell{channel,window_rows,width,kernel,padding,(U32.from_nat(row) * stride + origin : U32),(U32.from_nat(column) * stride + 0 : U32),pad},
    logical,PoolLeaf.Cell{Nat.add(Nat.mul(image,U32.to_nat(channels)),channel),height,width,kernel,padding,(U32.from_nat(inner) * stride + 0 : U32),(U32.from_nat(column) * stride + 0 : U32),pad},pad,
    window_taps(~Element,~Logical,~original,~operation,U32.to_nat(kernel),0,logical,pad,batch,channels,height,width,kernel,stride,padding,origin,window_first,window_rows,tree,length,
      channel,image,row,inner,column,values,channel_inside,facts))

# What a window with output rows supplies: both fresh destinations can be
# allocated and hold their cell counts, the assembled rows are ready and hold
# the logical input, the bounds, and the rows read by every tap.
def WindowFacts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+first: U32,+rows: U32,
  +window: BandWindow.RowWindow,+core: Buffer.Content<Element>,+above: Buffer.Content<Element>,+below: Buffer.Content<Element>,+limits: Machine.Limits) -> Data:
  BandWindow.RowWindow{BandWindow.Rows{+output_first,+output_rows},above_range,below_range,BandWindow.Rows{+window_first,+window_rows}} = window
  +input = assembled_input(~Element,geometry,fill,first,rows,window,core,above,below)
  Shape.Both<Depth.Sized<middle_count(geometry,window)>,Shape.Both<Depth.Sized<window_count(geometry,window)>,
    Shape.Both<{U32.to_nat(middle_count(geometry,window)) == middle_cells(geometry,window) : Nat},
      Shape.Both<{U32.to_nat(window_count(geometry,window)) == window_cells(geometry,window) : Nat},
        Shape.Both<BandHalo.RangeValues(~Element,~Logical,~original,logical,BandWindow.input_geometry(geometry),BandWindow.Rows{window_first,window_rows},input),
          Shape.Both<WindowRooms(geometry,window,limits),WindowRows(geometry,first,window)>>>>>>

# What a band and its halo supply: the band has no output row, or its window
# supplies the facts above.
def BandFacts(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+pad: Element,+fill: Element,+limits: Machine.Limits,
  +first: U32,+rows: U32,+core: Buffer.Content<Element>,+halo: BandHalo.Seen<Element>) -> Data:
  BandHalo.Seen{+above,+below,+window} = halo
  Either<&2,&2,{TensorBandWindow.output_rows(window) == 0 : U32},WindowFacts(~Element,~Logical,~original,logical,geometry,pad,fill,first,rows,window,core,above,below,limits)>

# ---- The pool of one band ----

# What a pooled band holds: it is ready, and every cell holds the pool of its
# pixel over the whole logical input.
def Pooled(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+pad: Element,
  +first: U32,+rows: U32,+buffer: Buffer.Content<Element>) -> Data:
  Shape.Both<Buffer.ContentReady<Element,buffer>,
    BandValues.band_values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),
      global_context(~Element,~Logical,logical,geometry,pad),values_output_geometry(geometry),first,rows,buffer)>

# The output rows of one band, with what every cell needs: the leaf facts of
# the window pool, the window content, the bounds and the rows read range.
type BandScope<-Element: Data,-Logical: Data,-original: Logical -> U32 -> Element,-operation: Element -> Element -> Element> is Data:
  BandScope{logical: Logical,pad: Element,batch: U32,channels: U32,height: U32,width: U32,kernel: U32,stride: U32,padding: U32,out_height: U32,out_width: U32,
    window_first: U32,window_rows: U32,output_first: U32,output_rows: U32,origin: U32,limits: Machine.Limits,
    window: Tree.Tree<Element>,window_length: Nat,tree: Tree.Tree<Element>,length: Nat,
    window_values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},window_first,window_rows,window,window_length),
    values: Contract.Values(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Buffer.Content<Element>>>,
      ~Sweeps.value(~Element,~Sweeps.Swept<Element,Buffer.Content<Element>>,~Sweeps.value(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation),~operation),
      PoolLeaf.context(~Element,~Buffer.Content<Element>,Buffer.Content{window,window_length},Access.Pool{window_rows,width,kernel,stride,padding,output_rows,out_width},origin,pad),
      Buffer.Content{tree,length}),
    length_exact: {length == BandValues.cell_count(Bands.Geometry{batch,channels,out_height,out_width},output_rows) : Nat},
    rooms: Rooms<batch,channels,height,width,out_height,out_width,window_rows,output_first,output_rows,limits>,
    facts: RowFacts(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,output_rows)}

def band_channels(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+scope: BandScope<Element,Logical,original,operation>) -> Nat:
  BandScope{logical,pad,batch,channels,height,width,kernel,stride,padding,out_height,out_width,window_first,window_rows,output_first,output_rows,origin,limits,
    window,window_length,tree,length,window_values,values,length_exact,rooms,facts} = scope
  U32.to_nat(channels)

def band_rows(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+scope: BandScope<Element,Logical,original,operation>) -> Nat:
  BandScope{logical,pad,batch,channels,height,width,kernel,stride,padding,out_height,out_width,window_first,window_rows,output_first,output_rows,origin,limits,
    window,window_length,tree,length,window_values,values,length_exact,rooms,facts} = scope
  U32.to_nat(output_rows)

def band_width(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+scope: BandScope<Element,Logical,original,operation>) -> Nat:
  BandScope{logical,pad,batch,channels,height,width,kernel,stride,padding,out_height,out_width,window_first,window_rows,output_first,output_rows,origin,limits,
    window,window_length,tree,length,window_values,values,length_exact,rooms,facts} = scope
  U32.to_nat(out_width)

def BandCell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+scope: BandScope<Element,Logical,original,operation>,
  +channel: Nat,+row: Nat,+column: Nat) -> Data:
  BandScope{logical,pad,batch,channels,height,width,kernel,stride,padding,out_height,out_width,window_first,window_rows,output_first,output_rows,origin,limits,
    window,window_length,tree,length,window_values,values,length_exact,rooms,facts} = scope
  BandValues.At(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),
    PoolLeaf.context(~Element,~Logical,logical,Access.Pool{height,width,kernel,stride,padding,out_height,out_width},0,pad),
    Bands.Geometry{batch,channels,out_height,out_width},output_first,output_rows,tree,length,channel,row,column)

# One output cell of a band: the leaf facts give the window pool at its
# position, the leaf coordinates read it as the pool of its window cell, the
# window cell is the input cell of its pixel, and the leaf coordinates read
# that back as the input pool at the pixel.
def band_cell(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+scope: BandScope<Element,Logical,original,operation>,
  +channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,band_channels(~Element,~Logical,~original,~operation,scope)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,band_rows(~Element,~Logical,~original,~operation,scope)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,band_width(~Element,~Logical,~original,~operation,scope)) == True{} : Bool}) ->
  BandCell(~Element,~Logical,~original,~operation,scope,channel,row,column):
  BandScope{+logical,+pad,+batch,+channels,+height,+width,+kernel,+stride,+padding,+out_height,+out_width,+window_first,+window_rows,+output_first,+output_rows,+origin,+limits,
    +window,+window_length,+tree,+length,+window_values,+values,+length_exact,+rooms,+facts} = scope
  Rooms{+output_fit,+out_height_positive,+input_room,+planes_room,+output_room,+local_input_room,+local_planes_room,+local_output_room} = rooms
  +output = {Bands.Geometry{batch,channels,out_height,out_width} : Bands.Geometry}
  +local = {Access.Pool{window_rows,width,kernel,stride,padding,output_rows,out_width} : Access.Pool}
  +global = {Access.Pool{height,width,kernel,stride,padding,out_height,out_width} : Access.Pool}
  +content = {Buffer.Content{window,window_length} : Buffer.Content<Element>}
  +images = U32.to_nat(batch)
  +count = U32.to_nat(channels)
  +tall = U32.to_nat(out_height)
  +limit = U32.to_nat(Machine.capacity(limits))
  +position = BandValues.position(output,output_rows,channel,row,column)
  +whole = Nat.add(U32.to_nat(output_first),row)
  +image = Nat.div(whole,tall)
  +inner = Nat.mod(whole,tall)
  +plane = Nat.add(Nat.mul(image,count),channel)
  +window_pool = Sweeps.value(~Element,~Sweeps.Swept<Element,Buffer.Content<Element>>,~Sweeps.value(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation),~operation,
    PoolLeaf.context(~Element,~Buffer.Content<Element>,content,local,origin,pad),U32.from_nat(position))
  +input_pool = Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation,
    PoolLeaf.context(~Element,~Logical,logical,global,0,pad),U32.from_nat(BandValues.pixel(output,channel,whole,column)))
  +window_cell_pool = PoolLeaf.at(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation,content,local,origin,pad,channel,row,column)
  +input_cell_pool = PoolLeaf.at(~Element,~Logical,~original,~operation,logical,global,0,pad,plane,inner,column)
  +inside = BandValues.position_within(output,output_rows,length,channel,row,column,length_exact,channel_inside,row_inside,column_inside)
  +image_ok = BandValues.image_below(U32.to_nat(output_first),U32.to_nat(output_rows),row,images,tall,out_height_positive,row_inside,output_fit)
  +stored = Contract.read(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Buffer.Content<Element>>>,
    ~Sweeps.value(~Element,~Sweeps.Swept<Element,Buffer.Content<Element>>,~Sweeps.value(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation),~operation),
    PoolLeaf.context(~Element,~Buffer.Content<Element>,content,local,origin,pad),Buffer.Content{tree,length},position,values,inside)
  +window_coordinates = PoolLeaf.cell(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation,content,pad,window_rows,width,kernel,stride,padding,output_rows,out_width,origin,
    channel,row,column,count,limits,channel_inside,row_inside,column_inside,local_input_room,local_planes_room,local_output_room)
  +read_range = window_cell(~Element,~Logical,~original,~operation,logical,pad,batch,channels,height,width,kernel,stride,padding,out_height,out_width,origin,
    window_first,window_rows,output_first,output_rows,window,window_length,channel,row,column,window_values,channel_inside,
    Members.all_member(Nat,{row => RowTapsAt(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,row) : Nat -> Data},row,
      Ranges.range(U32.to_nat(output_rows),0n),facts,Members.range_member(U32.to_nat(output_rows),0n,row,row_inside)))
  +input_coordinates = PoolLeaf.cell(~Element,~Logical,~original,~operation,logical,pad,height,width,kernel,stride,padding,out_height,out_width,0,plane,inner,column,Nat.mul(images,count),limits,
    RowMajor.radix(image,images,channel,count,image_ok,channel_inside),DivisionOrder.remainder_small(whole,tall,out_height_positive),column_inside,
    grouped(images,count,Nat.mul(U32.to_nat(height),U32.to_nat(width)),limit,input_room),
    grouped(images,count,Nat.mul(U32.to_nat(height),U32.to_nat(out_width)),limit,planes_room),
    grouped(images,count,Nat.mul(tall,U32.to_nat(out_width)),limit,output_room))
  Shape.Both{inside,
    Equal.trans(Element,Tree.read(Element,tree,U32.from_nat(position)),window_pool,input_pool,stored,
      Equal.trans(Element,window_pool,window_cell_pool,input_pool,window_coordinates,
        Equal.trans(Element,window_cell_pool,input_cell_pool,input_pool,read_range,Equal.sym(Element,input_pool,input_cell_pool,input_coordinates))))}

# A buffer of the output rows of a band that holds, at every index, the pool
# of its window read through the window, and one index for every cell, holds
# the pool of the input at every cell.
def pooled_band(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+out_width: U32,
  +window_first: U32,+window_rows: U32,+output_first: U32,+output_rows: U32,+origin: U32,+limits: Machine.Limits,+window: Tree.Tree<Element>,+window_length: Nat,+content: Buffer.Content<Element>,
  pooled: GatherValues.Facts(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Buffer.Content<Element>>>,
    ~Sweeps.value(~Element,~Sweeps.Swept<Element,Buffer.Content<Element>>,~Sweeps.value(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation),~operation),
    PoolLeaf.context(~Element,~Buffer.Content<Element>,Buffer.Content{window,window_length},Access.Pool{window_rows,width,kernel,stride,padding,output_rows,out_width},origin,pad),
    [channels,output_rows,out_width],content),
  +window_values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},window_first,window_rows,window,window_length),
  +rooms: Rooms<batch,channels,height,width,out_height,out_width,window_rows,output_first,output_rows,limits>,
  +facts: RowFacts(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,output_rows)) ->
  Pooled(~Element,~Logical,~original,~operation,logical,BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,out_height,out_width},pad,output_first,output_rows,content):
  match content:
    case Buffer.Content{+tree,+length}:
      Shape.Both{Contract.WellFormed{+length_exact,ready},values} = pooled
      Shape.Both{ready,
        BandValues.table(~BandScope<Element,Logical,original,operation>,~BandCell(~Element,~Logical,~original,~operation),~band_channels(~Element,~Logical,~original,~operation),
          ~band_rows(~Element,~Logical,~original,~operation),~band_width(~Element,~Logical,~original,~operation),~band_cell(~Element,~Logical,~original,~operation),
          BandScope{logical,pad,batch,channels,height,width,kernel,stride,padding,out_height,out_width,window_first,window_rows,output_first,output_rows,origin,limits,
            window,window_length,tree,length,window_values,values,
            Equal.trans(Nat,length,ShapeLaws.elements([channels,output_rows,out_width]),BandValues.cell_count(Bands.Geometry{batch,channels,out_height,out_width},output_rows),
              length_exact,ShapeLaws.elements_three(channels,output_rows,out_width)),
            rooms,facts})}

# The checked leaf pool of one window, with its two fresh destinations
# observed as models.
def window_observed(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,+fill: Element,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+out_width: U32,
  +window_first: U32,+window_rows: U32,+output_first: U32,+output_rows: U32,+origin: U32,+limits: Machine.Limits,+actual: {limits == Default.limits() : Machine.Limits},
  +input: Buffer.Model<Element>,input_ready: Buffer.ContentReady<Element,Buffer.content(~Element,input)>,
  middle_observation: Buffer.Observation<Element,Tensors.buffer(~Element,fill,(channels * window_rows * out_width : U32))>,
  observation: Buffer.Observation<Element,Tensors.buffer(~Element,fill,(channels * output_rows * out_width : U32))>,
  middle_allocation: Depth.Sized<(channels * window_rows * out_width : U32)>,allocation: Depth.Sized<(channels * output_rows * out_width : U32)>,
  +middle_exact: {U32.to_nat((channels * window_rows * out_width : U32)) == BandValues.cell_count(Bands.Geometry{batch,channels,height,out_width},window_rows) : Nat},
  +count_exact: {U32.to_nat((channels * output_rows * out_width : U32)) == BandValues.cell_count(Bands.Geometry{batch,channels,out_height,out_width},output_rows) : Nat},
  +values: BandValues.Values(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},Model.RowBand{Buffer.content(~Element,input),window_first,window_rows}),
  +rooms: Rooms<batch,channels,height,width,out_height,out_width,window_rows,output_first,output_rows,limits>,
  +facts: RowFacts(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,output_rows),
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),
    Access.Pool{window_rows,width,kernel,stride,padding,output_rows,out_width},origin,channels,
    Tensors.buffer(~Element,fill,(channels * window_rows * out_width : U32)),Tensors.buffer(~Element,fill,(channels * output_rows * out_width : U32))))) == True{} : Bool}) ->
  Pooled(~Element,~Logical,~original,~operation,logical,BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,out_height,out_width},pad,output_first,output_rows,
    CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),
      Access.Pool{window_rows,width,kernel,stride,padding,output_rows,out_width},origin,channels,
      Tensors.buffer(~Element,fill,(channels * window_rows * out_width : U32)),Tensors.buffer(~Element,fill,(channels * output_rows * out_width : U32)))))):
  Buffer.Model{+input_tree,+input_length,+input_certificate} = input
  Buffer.Observation{+middle_model,+middle_equation} = middle_observation
  Buffer.Observation{+output,+equation} = observation
  +input = {Buffer.Model{input_tree,input_length,input_certificate} : Buffer.Model<Element>}
  +window = {Buffer.Content{input_tree,input_length} : Buffer.Content<Element>}
  +pool = {Access.Pool{window_rows,width,kernel,stride,padding,output_rows,out_width} : Access.Pool}
  +middle = (channels * window_rows * out_width : U32)
  +count = (channels * output_rows * out_width : U32)
  +middle_length = PoolLeaf.fresh_length(~Element,fill,middle,middle_model,middle_equation)
  +length = PoolLeaf.fresh_length(~Element,fill,count,output,equation)
  +middle_cells = Equal.trans(Nat,PoolLeaf.swept_cells(pool,channels),U32.to_nat(middle),Buffer.length(~Element,middle_model),
    Equal.sym(Nat,U32.to_nat(middle),BandValues.cell_count(Bands.Geometry{batch,channels,height,out_width},window_rows),middle_exact),
    Equal.sym(Nat,Buffer.length(~Element,middle_model),U32.to_nat(middle),middle_length))
  +middle_shaped = Equal.trans(Nat,Buffer.length(~Element,middle_model),U32.to_nat(middle),ShapeLaws.elements([channels,window_rows,out_width]),middle_length,
    Equal.trans(Nat,U32.to_nat(middle),BandValues.cell_count(Bands.Geometry{batch,channels,height,out_width},window_rows),ShapeLaws.elements([channels,window_rows,out_width]),middle_exact,
      Equal.sym(Nat,ShapeLaws.elements([channels,window_rows,out_width]),BandValues.cell_count(Bands.Geometry{batch,channels,height,out_width},window_rows),ShapeLaws.elements_three(channels,window_rows,out_width))))
  +cells = Equal.trans(Nat,PoolLeaf.pooled_cells(pool,channels),U32.to_nat(count),Buffer.length(~Element,output),
    Equal.sym(Nat,U32.to_nat(count),BandValues.cell_count(Bands.Geometry{batch,channels,out_height,out_width},output_rows),count_exact),
    Equal.sym(Nat,Buffer.length(~Element,output),U32.to_nat(count),length))
  +shaped = Equal.trans(Nat,Buffer.length(~Element,output),U32.to_nat(count),ShapeLaws.elements([channels,output_rows,out_width]),length,
    Equal.trans(Nat,U32.to_nat(count),BandValues.cell_count(Bands.Geometry{batch,channels,out_height,out_width},output_rows),ShapeLaws.elements([channels,output_rows,out_width]),count_exact,
      Equal.sym(Nat,ShapeLaws.elements([channels,output_rows,out_width]),BandValues.cell_count(Bands.Geometry{batch,channels,out_height,out_width},output_rows),ShapeLaws.elements_three(channels,output_rows,out_width))))
  +accepted_models = {
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,middle_model),Tensors.buffer(~Element,fill,middle),middle_equation) :
      {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,origin,channels,_,Buffer.pack(~Element,output)))) == True{} : Bool}
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,output),Tensors.buffer(~Element,fill,count),equation) :
      {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,origin,channels,Tensors.buffer(~Element,fill,middle),_))) == True{} : Bool}
    accepted : {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,origin,channels,Buffer.pack(~Element,middle_model),Buffer.pack(~Element,output)))) == True{} : Bool}}
  +pooled = PoolLeaf.pooled_facts(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),~operation,window,pad,pool,origin,channels,input,middle_model,output,
    [channels,window_rows,out_width],[channels,output_rows,out_width],limits,actual,
    Buffer.ready_content(~Element,input,input_ready),
    Buffer.ready_observed(~Element,Tensors.buffer(~Element,fill,middle),middle_model,middle_equation,BandCount.allocated_ready(~Element,middle,fill,middle_allocation)),
    Buffer.ready_observed(~Element,Tensors.buffer(~Element,fill,count),output,equation,BandCount.allocated_ready(~Element,count,fill,allocation)),
    Contract.stored_values(~Element,window),middle_cells,middle_shaped,cells,shaped,accepted_models)
  +content = CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,origin,channels,
    Buffer.pack(~Element,middle_model),Buffer.pack(~Element,output))))
  %middle_equation : Pooled(~Element,~Logical,~original,~operation,logical,BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,out_height,out_width},pad,output_first,output_rows,
    CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,origin,channels,_,Tensors.buffer(~Element,fill,count)))))
  %equation : Pooled(~Element,~Logical,~original,~operation,logical,BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,out_height,out_width},pad,output_first,output_rows,
    CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),pool,origin,channels,Buffer.pack(~Element,middle_model),_))))
  pooled_band(~Element,~Logical,~original,~operation,logical,pad,batch,channels,height,width,kernel,stride,padding,out_height,out_width,
    window_first,window_rows,output_first,output_rows,origin,limits,input_tree,input_length,content,pooled,values,rooms,facts)

def window_output(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+pad: Element,+fill: Element,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+kernel: U32,+stride: U32,+padding: U32,+out_height: U32,+out_width: U32,
  +window_first: U32,+window_rows: U32,+output_first: U32,+output_rows: U32,+origin: U32,+limits: Machine.Limits,+actual: {limits == Default.limits() : Machine.Limits},
  +input: Buffer.Model<Element>,
  range_values: BandHalo.RangeValues(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},BandWindow.Rows{window_first,window_rows},Buffer.content(~Element,input)),
  middle_allocation: Depth.Sized<(channels * window_rows * out_width : U32)>,allocation: Depth.Sized<(channels * output_rows * out_width : U32)>,
  +middle_exact: {U32.to_nat((channels * window_rows * out_width : U32)) == BandValues.cell_count(Bands.Geometry{batch,channels,height,out_width},window_rows) : Nat},
  +count_exact: {U32.to_nat((channels * output_rows * out_width : U32)) == BandValues.cell_count(Bands.Geometry{batch,channels,out_height,out_width},output_rows) : Nat},
  +rooms: Rooms<batch,channels,height,width,out_height,out_width,window_rows,output_first,output_rows,limits>,
  +facts: RowFacts(kernel,stride,padding,origin,window_first,window_rows,height,out_height,output_first,output_rows),
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),
    Access.Pool{window_rows,width,kernel,stride,padding,output_rows,out_width},origin,channels,
    Tensors.buffer(~Element,fill,(channels * window_rows * out_width : U32)),Tensors.buffer(~Element,fill,(channels * output_rows * out_width : U32))))) == True{} : Bool}) ->
  Pooled(~Element,~Logical,~original,~operation,logical,BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,out_height,out_width},pad,output_first,output_rows,
    CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.pool(~Element,~operation,pad,Buffer.pack(~Element,input),
      Access.Pool{window_rows,width,kernel,stride,padding,output_rows,out_width},origin,channels,
      Tensors.buffer(~Element,fill,(channels * window_rows * out_width : U32)),Tensors.buffer(~Element,fill,(channels * output_rows * out_width : U32)))))):
  Buffer.Model{+tree,+length,+certificate} = input
  Shape.Both{input_ready,Shape.Both{metadata,+values}} = range_values
  +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
  window_observed(~Element,~Logical,~original,~operation,logical,pad,fill,batch,channels,height,width,kernel,stride,padding,out_height,out_width,
    window_first,window_rows,output_first,output_rows,origin,limits,actual,input,input_ready,
    Buffer.observe(~Element,Tensors.buffer(~Element,fill,(channels * window_rows * out_width : U32))),Buffer.observe(~Element,Tensors.buffer(~Element,fill,(channels * output_rows * out_width : U32))),
    middle_allocation,allocation,middle_exact,count_exact,values,rooms,facts,accepted)

# A band without output rows holds no cell.
def empty_band(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Element -> Element -> Element,+logical: Logical,+geometry: BandWindow.Geometry,+pad: Element,
  +first: U32,+content: Buffer.Content<Element>,ready: Buffer.ContentReady<Element,content>) ->
  Pooled(~Element,~Logical,~original,~operation,logical,geometry,pad,first,0,content):
  match geometry content:
    case BandWindow.Geometry{Bands.Geometry{+batch,+channels,+height,+width},+kernel,+stride,+padding,+output_height,+output_width} Buffer.Content{+tree,+length}:
      Shape.Both{ready,
        BandValues.empty_leaf(~Element,~Sweeps.Swept<Element,Sweeps.Swept<Element,Logical>>,~Sweeps.value(~Element,~Sweeps.Swept<Element,Logical>,~Sweeps.value(~Element,~Logical,~original,~operation),~operation),
          global_context(~Element,~Logical,logical,BandWindow.Geometry{Bands.Geometry{batch,channels,height,width},kernel,stride,padding,output_height,output_width},pad),
          Bands.Geometry{batch,channels,output_height,output_width},first,0,tree,length,RowMajor.multiply_zero_right(U32.to_nat(channels)))}

# A band of the pool through its checked call: when the call accepted, what
# holds of its output holds of the band.
def checked_outcome(~Element: Data,~Scope: Data,~After: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,+scope: Scope,+first: U32,+rows: U32,
  outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,
  accepted: {Each.usable(~Element,Each.checked(~Element,BandPool.checked(~Element,first,rows,outcome))) == True{} : Bool},
  rule: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) == True{} : Bool} ->
    After(scope,first,rows,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)))) ->
  Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,BandPool.checked(~Element,first,rows,outcome))))):
  match outcome:
    case Operations.Accepted{(input,Storage.Buffer{storage,length,certificate})}: rule({==})
    case Operations.Rejected{(input,output)}:
      Empty.absurd(Model.ContentAll(~Element,~After(scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,BandPool.checked(~Element,first,rows,Operations.Rejected{(input,output)}))))),
        WordEncoding.false_true(accepted))
