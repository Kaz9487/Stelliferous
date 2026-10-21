# The band concat holds the concat of its parts at every cell: every band of
# the output is the leaf concat of the bands of the parts there along the
# middle axis of [outer, extent, inner], and the leaf concat theorem gives
# its values.
import Base
import ../tensor.bend as Tensors
import ../tensor_layout.bend as Layout
import ../tensor_band.bend as Bands
import ../traversal_partition.bend as Partition
import ../tensor_band_concat.bend as BandConcat
import ../tensor_shape.bend as Shapes
import ../tensor_window.bend as Window
import ../machine_limits.bend as Machine
import ./tensor_band_concat_execution.bend as BandConcatExecution
import ./tensor_band_concat_caller.bend as BandConcatCaller
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_inspection.bend as Inspection
import ./storage_shape.bend as Shape
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_band_values.bend as BandValues
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_encoding.bend as WordEncoding
import ./boolean_logic.bend as Boolean
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_division_order.bend as DivisionOrder
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_count.bend as BandCount
import ../default_machine_limits.bend as Default
import ./binary_radix.bend as Radix
import ./tensor_concat_leaf_values.bend as ConcatLeaf
import ./tensor_set_slice_values.bend as Slices
import ./tensor_contract.bend as Contract
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_gather_values.bend as GatherValues
import ./tensor_full_values.bend as Full
import ./tensor_count_refinement.bend as Count
import ./u32_division.bend as U32Division
import ./tensor_band_each.bend as Each
import ../tensor_band_schedule.bend as BandSchedule
import ./tensor_band_combine_leaf.bend as BandCombineLeaf

# ---- The concat of readers in [outer, extent, inner] ----

# The parts in order, each with its extent along the axis.
type Extents<-Part: Data> is Data:
  Last{part: Part,extent: U32}
  Next{part: Part,extent: U32,rest: Extents<Part>}

def extent_total(~Part: Data,parts: Extents<Part>) -> Nat:
  match parts:
    case Last{part,extent}: U32.to_nat(extent)
    case Next{part,extent,rest}: Nat.add(U32.to_nat(extent),extent_total(~Part,rest))

# The row-major index of an outer index, a place along the axis and an offset
# inside it.
def framed(+extent: Nat,+inner: Nat,+outer: Nat,+middle: Nat,+offset: Nat) -> Nat:
  Nat.add(Nat.mul(outer,Nat.mul(extent,inner)),Nat.add(Nat.mul(middle,inner),offset))

# The part that covers a place along the axis, read there.
def sample(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,parts: Extents<Part>,+inner: Nat,+outer: Nat,+middle: Nat,+offset: Nat) -> Element:
  match parts:
    case Last{part,+extent}: part_value(part,U32.from_nat(framed(U32.to_nat(extent),inner,outer,middle,offset)))
    case Next{part,+extent,rest}:
      Bool.pick(Element,Nat.is_lt(middle,U32.to_nat(extent)),part_value(part,U32.from_nat(framed(U32.to_nat(extent),inner,outer,middle,offset))),
        sample(~Element,~Part,~part_value,rest,inner,outer,Nat.sub(middle,U32.to_nat(extent)),offset))

# The parts and the cells inside the axis.
type Concatenated<-Part: Data> is Data:
  Concatenated{parts: Extents<Part>,inner: Nat}

# The concat at an index: the outer index, the place along the axis and the
# offset it names.
def value(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,context: Concatenated<Part>,index: U32) -> Element:
  Concatenated{+parts,+inner} = context
  +extent = extent_total(~Part,parts)
  +position = U32.to_nat(index)
  sample(~Element,~Part,~part_value,parts,inner,Nat.div(position,Nat.mul(extent,inner)),Nat.mod(Nat.div(position,inner),extent),Nat.mod(position,inner))

# An index that names a place below the extent and an offset below the inner
# cells holds the part there.
def framed_value(~Element: Data,~Part: Data,~part_value: Part -> U32 -> Element,+parts: Extents<Part>,+inner: Nat,+outer: Nat,+middle: Nat,+offset: Nat,+index: U32,
  +encoded: {U32.to_nat(index) == framed(extent_total(~Part,parts),inner,outer,middle,offset) : Nat},
  +middle_inside: {Nat.is_lt(middle,extent_total(~Part,parts)) == True{} : Bool},+offset_inside: {Nat.is_lt(offset,inner) == True{} : Bool}) ->
  {value(~Element,~Part,~part_value,Concatenated{parts,inner},index) == sample(~Element,~Part,~part_value,parts,inner,outer,middle,offset) : Element}:
  +extent = extent_total(~Part,parts)
  +frame = Nat.mul(extent,inner)
  +local = Nat.add(Nat.mul(middle,inner),offset)
  +line = Nat.add(Nat.mul(outer,extent),middle)
  +position = U32.to_nat(index)
  +inner_positive = Order.le_lt_trans(0n,offset,inner,Order.zero_le(offset),offset_inside)
  +extent_positive = Order.le_lt_trans(0n,middle,extent,Order.zero_le(middle),middle_inside)
  +frame_positive = RowMajor.multiply_lt(0n,extent,inner,extent_positive,inner_positive)
  +local_inside = RowMajor.radix(middle,extent,offset,inner,middle_inside,offset_inside)
  +lined = {
    %RowMajor.assoc(outer,extent,inner) : {Nat.add(_,local) == Nat.add(Nat.mul(line,inner),offset) : Nat}
    %Equal.sym(Nat,Nat.mul(line,inner),Nat.add(Nat.mul(Nat.mul(outer,extent),inner),Nat.mul(middle,inner)),RowMajor.distrib(Nat.mul(outer,extent),middle,inner)) :
      {Nat.add(Nat.mul(Nat.mul(outer,extent),inner),local) == Nat.add(_,offset) : Nat}
    Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(Nat.mul(outer,extent),inner),Nat.mul(middle,inner)),offset),Nat.add(Nat.mul(Nat.mul(outer,extent),inner),local),
      Addition.associative(Nat.mul(Nat.mul(outer,extent),inner),Nat.mul(middle,inner),offset))
    : {framed(extent,inner,outer,middle,offset) == Nat.add(Nat.mul(line,inner),offset) : Nat}}
  +outer_exact = {
    %Equal.sym(Nat,position,framed(extent,inner,outer,middle,offset),encoded) : {Nat.div(_,frame) == outer : Nat}
    Radix.quotient(outer,local,frame,frame_positive,local_inside)
    : {Nat.div(position,frame) == outer : Nat}}
  +line_exact = {
    %Equal.sym(Nat,position,framed(extent,inner,outer,middle,offset),encoded) : {Nat.div(_,inner) == line : Nat}
    %Equal.sym(Nat,framed(extent,inner,outer,middle,offset),Nat.add(Nat.mul(line,inner),offset),lined) : {Nat.div(_,inner) == line : Nat}
    Radix.quotient(line,offset,inner,inner_positive,offset_inside)
    : {Nat.div(position,inner) == line : Nat}}
  +middle_exact = Equal.trans(Nat,Nat.mod(Nat.div(position,inner),extent),Nat.mod(line,extent),middle,
    Equal.cong(Nat,Nat,value => Nat.mod(value,extent),Nat.div(position,inner),line,line_exact),
    Radix.remainder(outer,middle,extent,extent_positive,middle_inside))
  +offset_exact = {
    %Equal.sym(Nat,position,framed(extent,inner,outer,middle,offset),encoded) : {Nat.mod(_,inner) == offset : Nat}
    %Equal.sym(Nat,framed(extent,inner,outer,middle,offset),Nat.add(Nat.mul(line,inner),offset),lined) : {Nat.mod(_,inner) == offset : Nat}
    Radix.remainder(line,offset,inner,inner_positive,offset_inside)
    : {Nat.mod(position,inner) == offset : Nat}}
  %Equal.sym(Nat,Nat.div(position,frame),outer,outer_exact) :
    {sample(~Element,~Part,~part_value,parts,inner,_,Nat.mod(Nat.div(position,inner),extent),Nat.mod(position,inner)) == sample(~Element,~Part,~part_value,parts,inner,outer,middle,offset) : Element}
  %Equal.sym(Nat,Nat.mod(Nat.div(position,inner),extent),middle,middle_exact) :
    {sample(~Element,~Part,~part_value,parts,inner,outer,_,Nat.mod(position,inner)) == sample(~Element,~Part,~part_value,parts,inner,outer,middle,offset) : Element}
  %Equal.sym(Nat,Nat.mod(position,inner),offset,offset_exact) :
    {sample(~Element,~Part,~part_value,parts,inner,outer,middle,_) == sample(~Element,~Part,~part_value,parts,inner,outer,middle,offset) : Element}
  {==}

# ---- A tensor of a joining as [outer, extent, inner] ----

# The cells inside the axis.
def joining_inner(joining: BandConcat.Joining) -> Nat:
  match joining:
    case BandConcat.Channels{axis,batch,height,width}: Nat.mul(U32.to_nat(height),U32.to_nat(width))
    case BandConcat.Lines{axis,lines,inner}: U32.to_nat(inner)

# The outer index of the cell of a band at a line and an offset of the band,
# and its offset inside the axis.
def outer_at(joining: BandConcat.Joining,+first: U32,+line: Nat,+offset: Nat) -> Nat:
  match joining:
    case BandConcat.Channels{axis,batch,+height,+width}: Nat.div(Nat.add(U32.to_nat(first),Nat.div(offset,U32.to_nat(width))),U32.to_nat(height))
    case BandConcat.Lines{axis,lines,inner}: Nat.add(U32.to_nat(first),line)

def inner_at(joining: BandConcat.Joining,+first: U32,+line: Nat,+offset: Nat) -> Nat:
  match joining:
    case BandConcat.Channels{axis,batch,+height,+width}:
      Nat.add(Nat.mul(Nat.mod(Nat.add(U32.to_nat(first),Nat.div(offset,U32.to_nat(width))),U32.to_nat(height)),U32.to_nat(width)),Nat.mod(offset,U32.to_nat(width)))
    case BandConcat.Lines{axis,lines,inner}: offset

# The word products of the bands fit the capacity: the rows of the batch times
# the width, or the extent times the inner cells.
def Roomy(joining: BandConcat.Joining,+extent: U32,+capacity: U32) -> Data:
  match joining:
    case BandConcat.Channels{axis,+batch,+height,+width}:
      {Nat.is_le(Nat.mul(Nat.mul(U32.to_nat(batch),U32.to_nat(height)),U32.to_nat(width)),U32.to_nat(capacity)) == True{} : Bool}
    case BandConcat.Lines{axis,lines,+inner}: {Nat.is_le(Nat.mul(U32.to_nat(extent),U32.to_nat(inner)),U32.to_nat(capacity)) == True{} : Bool}

type Fitted<-joining: BandConcat.Joining,-extent: U32,-capacity: U32> is Data:
  Fitted{capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},roomy: Roomy(joining,extent,capacity)}

# The rows of a band of a joining fit its batch whatever the extent.
def rows_moved(+joining: BandConcat.Joining,+extent: U32,+other: U32,+first: U32,+rows: U32,
  fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(BandConcat.geometry(joining,extent))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(BandConcat.geometry(joining,other))) == True{} : Bool}:
  match joining:
    case BandConcat.Channels{axis,batch,height,width}: fit
    case BandConcat.Lines{axis,lines,inner}: fit

# The rows of a band fit the batch, as its metadata states.
def fitted_rows(+geometry: Bands.Geometry,+first: U32,+rows: U32,+length: Nat,metadata: BandShape.Leaf(geometry,first,rows,length)) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry)) == True{} : Bool}:
  Bands.Geometry{+images,+channels,+height,+width} = geometry
  BandShape.Fields{local,rows_fit,length_exact} = metadata
  rows_fit

# The cells of a channel of an image: the image, the channel, then the cell
# inside the channel.
def image_frame(+image: Nat,+channels: Nat,+channel: Nat,+height: Nat,+width: Nat,+row: Nat,+column: Nat) ->
  {RowMajor.index(Nat.add(Nat.mul(image,channels),channel),height,width,row,column) == Nat.add(Nat.mul(image,Nat.mul(channels,Nat.mul(height,width))),RowMajor.index(channel,height,width,row,column)) : Nat}:
  +spatial = Nat.mul(height,width)
  +pixel = Nat.add(Nat.mul(row,width),column)
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(image,channels),channel),spatial),Nat.add(Nat.mul(Nat.mul(image,channels),spatial),Nat.mul(channel,spatial)),RowMajor.distrib(Nat.mul(image,channels),channel,spatial)) :
    {Nat.add(_,pixel) == Nat.add(Nat.mul(image,Nat.mul(channels,spatial)),RowMajor.index(channel,height,width,row,column)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(image,channels),spatial),Nat.mul(image,Nat.mul(channels,spatial)),RowMajor.assoc(image,channels,spatial)) :
    {Nat.add(Nat.add(_,Nat.mul(channel,spatial)),pixel) == Nat.add(Nat.mul(image,Nat.mul(channels,spatial)),RowMajor.index(channel,height,width,row,column)) : Nat}
  Addition.associative(Nat.mul(image,Nat.mul(channels,spatial)),Nat.mul(channel,spatial),pixel)

# A cell of a band of a joining at a channel, a row and a column names a
# line, a place along the axis and an offset of the band as [outer, extent,
# inner], and its pixel is that cell of the tensor.
type Box<-joining: BandConcat.Joining,-extent: U32,-first: U32,-rows: U32,-position: Nat,-pixel: Nat> is Data:
  Box{line: Nat,middle: Nat,offset: Nat,
    stored: {position == ConcatLeaf.cell_at(extent,BandConcat.band_inner(joining,rows),line,middle,offset) : Nat},
    placed: {pixel == framed(U32.to_nat(extent),joining_inner(joining),outer_at(joining,first,line,offset),middle,inner_at(joining,first,line,offset)) : Nat},
    line_inside: {Nat.is_lt(line,U32.to_nat(BandConcat.band_outer(joining,rows))) == True{} : Bool},
    middle_inside: {Nat.is_lt(middle,U32.to_nat(extent)) == True{} : Bool},
    offset_inside: {Nat.is_lt(offset,U32.to_nat(BandConcat.band_inner(joining,rows))) == True{} : Bool},
    inner_inside: {Nat.is_lt(inner_at(joining,first,line,offset),joining_inner(joining)) == True{} : Bool}}

def boxed(+joining: BandConcat.Joining,+extent: U32,+first: U32,+rows: U32,+capacity: U32,+fitted: Fitted<joining,extent,capacity>,
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(BandConcat.geometry(joining,extent))) == True{} : Bool},
  +channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,BandValues.channels(BandConcat.geometry(joining,extent))) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,BandValues.width(BandConcat.geometry(joining,extent))) == True{} : Bool}) ->
  Box<joining,extent,first,rows,BandValues.position(BandConcat.geometry(joining,extent),rows,channel,row,column),
    BandValues.pixel(BandConcat.geometry(joining,extent),channel,Nat.add(U32.to_nat(first),row),column)>:
  match joining:
    case BandConcat.Channels{+axis,+batch,+height,+width}:
      Fitted{+capacity_ok,+roomy} = fitted
      +wide = U32.to_nat(width)
      +tall = U32.to_nat(height)
      +band = U32.to_nat(rows)
      +whole = Nat.add(U32.to_nat(first),row)
      +offset = Nat.add(Nat.mul(row,wide),column)
      +total = Nat.mul(U32.to_nat(batch),tall)
      +width_positive = Order.le_lt_trans(0n,column,wide,Order.zero_le(column),column_inside)
      +whole_inside = Bounds.nat_trans(whole,Nat.add(U32.to_nat(first),band),total,Order.offset_lt(U32.to_nat(first),row,band,row_inside),rows_fit)
      +height_positive = DivisionOrder.right_factor_positive(U32.to_nat(batch),tall,Order.le_lt_trans(0n,whole,total,Order.zero_le(whole),whole_inside))
      +cells = Arithmetic.multiply(rows,width,capacity,capacity_ok,
        Order.transitive(Nat.mul(band,wide),Nat.mul(total,wide),U32.to_nat(capacity),
          RowMajor.mul_le(band,total,wide,Order.drop_left_le(U32.to_nat(first),band,total,rows_fit)),roomy))
      +row_exact = Radix.quotient(row,column,wide,width_positive,column_inside)
      +column_exact = Radix.remainder(row,column,wide,width_positive,column_inside)
      +image = Nat.div(whole,tall)
      +inner_row = Nat.mod(whole,tall)
      Box{0n,channel,offset,
        %Equal.sym(Nat,U32.to_nat(U32.mul(rows,width)),Nat.mul(band,wide),cells) :
          {Nat.add(Nat.mul(channel,Nat.mul(band,wide)),offset) == Nat.add(Nat.mul(channel,_),offset) : Nat}
        {==},
        %Equal.sym(Nat,Nat.div(offset,wide),row,row_exact) :
          {RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(extent)),channel),tall,wide,inner_row,column) ==
            framed(U32.to_nat(extent),Nat.mul(tall,wide),Nat.div(Nat.add(U32.to_nat(first),_),tall),channel,
              Nat.add(Nat.mul(Nat.mod(Nat.add(U32.to_nat(first),_),tall),wide),Nat.mod(offset,wide))) : Nat}
        %Equal.sym(Nat,Nat.mod(offset,wide),column,column_exact) :
          {RowMajor.index(Nat.add(Nat.mul(image,U32.to_nat(extent)),channel),tall,wide,inner_row,column) ==
            framed(U32.to_nat(extent),Nat.mul(tall,wide),image,channel,Nat.add(Nat.mul(inner_row,wide),_)) : Nat}
        image_frame(image,U32.to_nat(extent),channel,tall,wide,inner_row,column),
        {==},channel_inside,
        %Equal.sym(Nat,U32.to_nat(U32.mul(rows,width)),Nat.mul(band,wide),cells) : {Nat.is_lt(offset,_) == True{} : Bool}
        RowMajor.radix(row,band,column,wide,row_inside,column_inside),
        %Equal.sym(Nat,Nat.div(offset,wide),row,row_exact) :
          {Nat.is_lt(Nat.add(Nat.mul(Nat.mod(Nat.add(U32.to_nat(first),_),tall),wide),Nat.mod(offset,wide)),Nat.mul(tall,wide)) == True{} : Bool}
        %Equal.sym(Nat,Nat.mod(offset,wide),column,column_exact) : {Nat.is_lt(Nat.add(Nat.mul(inner_row,wide),_),Nat.mul(tall,wide)) == True{} : Bool}
        RowMajor.radix(inner_row,tall,column,wide,DivisionOrder.remainder_small(whole,tall,height_positive),column_inside)}
    case BandConcat.Lines{+axis,+lines,+inner}:
      Fitted{+capacity_ok,+roomy} = fitted
      match channel:
        case 1n+ +rest:
          Empty.absurd(Box<BandConcat.Lines{axis,lines,inner},extent,first,rows,BandValues.position(BandConcat.geometry(BandConcat.Lines{axis,lines,inner},extent),rows,1n+rest,row,column),
            BandValues.pixel(BandConcat.geometry(BandConcat.Lines{axis,lines,inner},extent),1n+rest,Nat.add(U32.to_nat(first),row),column)>,Order.below_zero(rest,channel_inside))
        case 0n:
          +across = U32.to_nat(inner)
          +span = Nat.mul(U32.to_nat(extent),across)
          +wide = Arithmetic.multiply(extent,inner,capacity,capacity_ok,roomy)
          +column_cells = {%wide : {Nat.is_lt(column,_) == True{} : Bool} column_inside : {Nat.is_lt(column,span) == True{} : Bool}}
          +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),across,Order.le_lt_trans(0n,column,span,Order.zero_le(column),column_cells))
          +middle = Nat.div(column,across)
          +offset = Nat.mod(column,across)
          +split = DivisionOrder.quotient_exact(column,across,inner_positive)
          +whole = Nat.add(U32.to_nat(first),row)
          +line_total = {%Addition.zero_right(U32.to_nat(lines)) : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool} rows_fit :
            {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(lines)) == True{} : Bool}}
          +whole_inside = Bounds.nat_trans(whole,Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(lines),Order.offset_lt(U32.to_nat(first),row,U32.to_nat(rows),row_inside),line_total)
          +lines_positive = Order.le_lt_trans(0n,whole,U32.to_nat(lines),Order.zero_le(whole),whole_inside)
          +image_zero = Radix.quotient(0n,whole,U32.to_nat(lines),lines_positive,whole_inside)
          +row_same = Radix.remainder(0n,whole,U32.to_nat(lines),lines_positive,whole_inside)
          Box{row,middle,offset,
            %split : {Nat.add(Nat.mul(row,U32.to_nat(U32.mul(extent,inner))),column) == Nat.add(Nat.mul(row,span),_) : Nat}
            %Equal.sym(Nat,U32.to_nat(U32.mul(extent,inner)),span,wide) : {Nat.add(Nat.mul(row,_),column) == Nat.add(Nat.mul(row,span),column) : Nat}
            {==},
            %Equal.sym(Nat,Nat.div(whole,U32.to_nat(lines)),0n,image_zero) :
              {RowMajor.index(Nat.add(Nat.mul(_,1n),0n),U32.to_nat(lines),U32.to_nat(U32.mul(extent,inner)),Nat.mod(whole,U32.to_nat(lines)),column) ==
                framed(U32.to_nat(extent),across,whole,middle,offset) : Nat}
            %Equal.sym(Nat,Nat.mod(whole,U32.to_nat(lines)),whole,row_same) :
              {RowMajor.index(0n,U32.to_nat(lines),U32.to_nat(U32.mul(extent,inner)),_,column) == framed(U32.to_nat(extent),across,whole,middle,offset) : Nat}
            %split : {Nat.add(Nat.mul(whole,U32.to_nat(U32.mul(extent,inner))),column) == Nat.add(Nat.mul(whole,span),_) : Nat}
            %Equal.sym(Nat,U32.to_nat(U32.mul(extent,inner)),span,wide) : {Nat.add(Nat.mul(whole,_),column) == Nat.add(Nat.mul(whole,span),column) : Nat}
            {==},
            row_inside,DivisionOrder.quotient_below(column,U32.to_nat(extent),across,inner_positive,column_cells),
            DivisionOrder.remainder_small(column,across,inner_positive),DivisionOrder.remainder_small(column,across,inner_positive)}

# A line, a place along the axis and an offset of a band of a joining name a
# cell of the band, whose pixel is that cell of the tensor.
type Cell<-joining: BandConcat.Joining,-extent: U32,-first: U32,-rows: U32,-line: Nat,-middle: Nat,-offset: Nat> is Data:
  Cell{channel: Nat,row: Nat,column: Nat,
    channel_inside: {Nat.is_lt(channel,BandValues.channels(BandConcat.geometry(joining,extent))) == True{} : Bool},
    row_inside: {Nat.is_lt(row,U32.to_nat(rows)) == True{} : Bool},
    column_inside: {Nat.is_lt(column,BandValues.width(BandConcat.geometry(joining,extent))) == True{} : Bool},
    stored: {BandValues.position(BandConcat.geometry(joining,extent),rows,channel,row,column) == ConcatLeaf.cell_at(extent,BandConcat.band_inner(joining,rows),line,middle,offset) : Nat},
    placed: {BandValues.pixel(BandConcat.geometry(joining,extent),channel,Nat.add(U32.to_nat(first),row),column) ==
      framed(U32.to_nat(extent),joining_inner(joining),outer_at(joining,first,line,offset),middle,inner_at(joining,first,line,offset)) : Nat}}

def unboxed(+joining: BandConcat.Joining,+extent: U32,+first: U32,+rows: U32,+capacity: U32,+fitted: Fitted<joining,extent,capacity>,
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(BandConcat.geometry(joining,extent))) == True{} : Bool},
  +line: Nat,+middle: Nat,+offset: Nat,
  +line_inside: {Nat.is_lt(line,U32.to_nat(BandConcat.band_outer(joining,rows))) == True{} : Bool},
  +middle_inside: {Nat.is_lt(middle,U32.to_nat(extent)) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(BandConcat.band_inner(joining,rows))) == True{} : Bool}) ->
  Cell<joining,extent,first,rows,line,middle,offset>:
  match joining:
    case BandConcat.Channels{+axis,+batch,+height,+width}:
      Fitted{+capacity_ok,+roomy} = fitted
      match line:
        case 1n+ +rest: Empty.absurd(Cell<BandConcat.Channels{axis,batch,height,width},extent,first,rows,1n+rest,middle,offset>,Order.below_zero(rest,line_inside))
        case 0n:
          +wide = U32.to_nat(width)
          +tall = U32.to_nat(height)
          +band = U32.to_nat(rows)
          +total = Nat.mul(U32.to_nat(batch),tall)
          +cells = Arithmetic.multiply(rows,width,capacity,capacity_ok,
            Order.transitive(Nat.mul(band,wide),Nat.mul(total,wide),U32.to_nat(capacity),
              RowMajor.mul_le(band,total,wide,Order.drop_left_le(U32.to_nat(first),band,total,rows_fit)),roomy))
          +offset_cells = {%cells : {Nat.is_lt(offset,_) == True{} : Bool} offset_inside : {Nat.is_lt(offset,Nat.mul(band,wide)) == True{} : Bool}}
          +width_positive = DivisionOrder.right_factor_positive(band,wide,Order.le_lt_trans(0n,offset,Nat.mul(band,wide),Order.zero_le(offset),offset_cells))
          +row = Nat.div(offset,wide)
          +column = Nat.mod(offset,wide)
          +split = DivisionOrder.quotient_exact(offset,wide,width_positive)
          +whole = Nat.add(U32.to_nat(first),row)
          Cell{middle,row,column,middle_inside,DivisionOrder.quotient_below(offset,band,wide,width_positive,offset_cells),
            DivisionOrder.remainder_small(offset,wide,width_positive),
            %Equal.sym(Nat,U32.to_nat(U32.mul(rows,width)),Nat.mul(band,wide),cells) :
              {Nat.add(Nat.mul(middle,Nat.mul(band,wide)),Nat.add(Nat.mul(row,wide),column)) == Nat.add(Nat.mul(middle,_),offset) : Nat}
            %Equal.sym(Nat,offset,Nat.add(Nat.mul(row,wide),column),split) :
              {Nat.add(Nat.mul(middle,Nat.mul(band,wide)),Nat.add(Nat.mul(row,wide),column)) == Nat.add(Nat.mul(middle,Nat.mul(band,wide)),_) : Nat}
            {==},
            image_frame(Nat.div(whole,tall),U32.to_nat(extent),middle,tall,wide,Nat.mod(whole,tall),column)}
    case BandConcat.Lines{+axis,+lines,+inner}:
      Fitted{+capacity_ok,+roomy} = fitted
      +across = U32.to_nat(inner)
      +span = Nat.mul(U32.to_nat(extent),across)
      +wide = Arithmetic.multiply(extent,inner,capacity,capacity_ok,roomy)
      +column = Nat.add(Nat.mul(middle,across),offset)
      +whole = Nat.add(U32.to_nat(first),line)
      +line_total = {%Addition.zero_right(U32.to_nat(lines)) : {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),_) == True{} : Bool} rows_fit :
        {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(lines)) == True{} : Bool}}
      +whole_inside = Bounds.nat_trans(whole,Nat.add(U32.to_nat(first),U32.to_nat(rows)),U32.to_nat(lines),Order.offset_lt(U32.to_nat(first),line,U32.to_nat(rows),line_inside),line_total)
      +lines_positive = Order.le_lt_trans(0n,whole,U32.to_nat(lines),Order.zero_le(whole),whole_inside)
      +image_zero = Radix.quotient(0n,whole,U32.to_nat(lines),lines_positive,whole_inside)
      +row_same = Radix.remainder(0n,whole,U32.to_nat(lines),lines_positive,whole_inside)
      Cell{0n,line,column,{==},line_inside,
        %Equal.sym(Nat,U32.to_nat(U32.mul(extent,inner)),span,wide) : {Nat.is_lt(column,_) == True{} : Bool}
        RowMajor.radix(middle,U32.to_nat(extent),offset,across,middle_inside,offset_inside),
        %Equal.sym(Nat,U32.to_nat(U32.mul(extent,inner)),span,wide) : {Nat.add(Nat.mul(line,_),column) == Nat.add(Nat.mul(line,span),column) : Nat}
        {==},
        %Equal.sym(Nat,Nat.div(whole,U32.to_nat(lines)),0n,image_zero) :
          {RowMajor.index(Nat.add(Nat.mul(_,1n),0n),U32.to_nat(lines),U32.to_nat(U32.mul(extent,inner)),Nat.mod(whole,U32.to_nat(lines)),column) ==
            framed(U32.to_nat(extent),across,whole,middle,offset) : Nat}
        %Equal.sym(Nat,Nat.mod(whole,U32.to_nat(lines)),whole,row_same) :
          {RowMajor.index(0n,U32.to_nat(lines),U32.to_nat(U32.mul(extent,inner)),_,column) == framed(U32.to_nat(extent),across,whole,middle,offset) : Nat}
        %Equal.sym(Nat,U32.to_nat(U32.mul(extent,inner)),span,wide) : {Nat.add(Nat.mul(whole,_),column) == Nat.add(Nat.mul(whole,span),column) : Nat}
        {==}}

# The cells a band of a joining stores are the cells of its band shape.
def band_cells(+joining: BandConcat.Joining,+extent: U32,+first: U32,+rows: U32,+capacity: U32,+fitted: Fitted<joining,extent,capacity>,
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(BandConcat.geometry(joining,extent))) == True{} : Bool}) ->
  {BandValues.cell_count(BandConcat.geometry(joining,extent),rows) == ShapeLaws.elements(BandConcat.band_shape(joining,extent,rows)) : Nat}:
  match joining:
    case BandConcat.Channels{+axis,+batch,+height,+width}:
      Fitted{+capacity_ok,+roomy} = fitted
      +wide = U32.to_nat(width)
      +band = U32.to_nat(rows)
      +total = Nat.mul(U32.to_nat(batch),U32.to_nat(height))
      +cells = Arithmetic.multiply(rows,width,capacity,capacity_ok,
        Order.transitive(Nat.mul(band,wide),Nat.mul(total,wide),U32.to_nat(capacity),
          RowMajor.mul_le(band,total,wide,Order.drop_left_le(U32.to_nat(first),band,total,rows_fit)),roomy))
      +stored = Nat.mul(U32.to_nat(extent),U32.to_nat(U32.mul(rows,width)))
      %Equal.sym(Nat,ShapeLaws.elements([1,extent,U32.mul(rows,width)]),Nat.mul(U32.to_nat(1),stored),ShapeLaws.elements_three(1,extent,U32.mul(rows,width))) :
        {Nat.mul(U32.to_nat(extent),Nat.mul(band,wide)) == _ : Nat}
      %Equal.sym(Nat,Nat.add(stored,0n),stored,Addition.zero_right(stored)) : {Nat.mul(U32.to_nat(extent),Nat.mul(band,wide)) == _ : Nat}
      Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(extent),value),Nat.mul(band,wide),U32.to_nat(U32.mul(rows,width)),Equal.sym(Nat,U32.to_nat(U32.mul(rows,width)),Nat.mul(band,wide),cells))
    case BandConcat.Lines{+axis,+lines,+inner}:
      Fitted{+capacity_ok,+roomy} = fitted
      +band = U32.to_nat(rows)
      +span = Nat.mul(U32.to_nat(extent),U32.to_nat(inner))
      +wide = Arithmetic.multiply(extent,inner,capacity,capacity_ok,roomy)
      +stored = Nat.mul(band,U32.to_nat(U32.mul(extent,inner)))
      %Equal.sym(Nat,ShapeLaws.elements([rows,extent,inner]),Nat.mul(band,span),ShapeLaws.elements_three(rows,extent,inner)) : {Nat.mul(U32.to_nat(1),stored) == _ : Nat}
      %Equal.sym(Nat,Nat.add(stored,0n),stored,Addition.zero_right(stored)) : {_ == Nat.mul(band,span) : Nat}
      Equal.cong(Nat,Nat,value => Nat.mul(band,value),U32.to_nat(U32.mul(extent,inner)),span,wide)

# ---- The concat the band concat reads ----

def tensor_original(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,input: BandConcatExecution.TensorContent<Element>,index: U32) -> Element:
  BandConcatExecution.TensorContent{bands,geometry,shape,fill,status,workers} = input
  band_value(bands,geometry,index)

def part_extent(~Element: Data,+joining: BandConcat.Joining,source: BandConcatExecution.TensorContent<Element>) -> U32:
  BandConcat.extent(joining,BandConcatExecution.tensor_shape(~Element,source))

def part_geometry(~Element: Data,+joining: BandConcat.Joining,source: BandConcatExecution.TensorContent<Element>) -> Bands.Geometry:
  BandConcat.geometry(joining,part_extent(~Element,joining,source))

# The parts with their extents along the axis of the joining.
def extents_of(~Element: Data,+joining: BandConcat.Joining,parts: BandConcatExecution.PartContents<Element>) -> Extents<BandConcatExecution.TensorContent<Element>>:
  match parts:
    case BandConcatExecution.SingleContent{+input}: Last{input,part_extent(~Element,joining,input)}
    case BandConcatExecution.ConsContent{+input,rest}: Next{input,part_extent(~Element,joining,input),extents_of(~Element,joining,rest)}

def parts_total(~Element: Data,+joining: BandConcat.Joining,parts: BandConcatExecution.PartContents<Element>) -> Nat:
  extent_total(~BandConcatExecution.TensorContent<Element>,extents_of(~Element,joining,parts))

# The concat of the parts as the band concat reads them.
def logical(~Element: Data,+joining: BandConcat.Joining,parts: BandConcatExecution.PartContents<Element>) -> Concatenated<BandConcatExecution.TensorContent<Element>>:
  Concatenated{extents_of(~Element,joining,parts),joining_inner(joining)}

def concat_value(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,context: Concatenated<BandConcatExecution.TensorContent<Element>>,index: U32) -> Element:
  value(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),context,index)

# A part holds the reader's values in the geometry of the joining, and its
# products of words fit the capacity.
def SourceContract(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,input: BandConcatExecution.TensorContent<Element>) -> Data:
  BandConcatExecution.TensorContent{+bands,+geometry,+shape,+fill,+status,+workers} = input
  Shape.Both<BandShape.WellFormed<Element,BandConcat.geometry(joining,BandConcat.extent(joining,shape)),bands>,
    Shape.Both<BandValues.Values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),BandConcatExecution.TensorContent{bands,geometry,shape,fill,status,workers},
        BandConcat.geometry(joining,BandConcat.extent(joining,shape)),bands),
      Fitted<joining,BandConcat.extent(joining,shape),capacity>>>

def SourceContracts(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,parts: BandConcatExecution.PartContents<Element>) -> Data:
  match parts:
    case BandConcatExecution.SingleContent{input}: SourceContract(~Element,~band_value,joining,capacity,input)
    case BandConcatExecution.ConsContent{input,rest}:
      Shape.Both<SourceContract(~Element,~band_value,joining,capacity,input),SourceContracts(~Element,~band_value,joining,capacity,rest)>

def first_shape(~Element: Data,parts: BandConcatExecution.PartContents<Element>) -> List<&2,U32>:
  match parts:
    case BandConcatExecution.SingleContent{input}: BandConcatExecution.tensor_shape(~Element,input)
    case BandConcatExecution.ConsContent{input,rest}: BandConcatExecution.tensor_shape(~Element,input)

def first_fill(~Element: Data,parts: BandConcatExecution.PartContents<Element>) -> Element:
  match parts:
    case BandConcatExecution.SingleContent{input}: BandConcatCaller.tensor_fill(~Element,input)
    case BandConcatExecution.ConsContent{input,rest}: BandConcatCaller.tensor_fill(~Element,input)

# ---- The bands of the parts at every band of the output ----

# What a part holds at one of its bands: the metadata of the band, ready, the
# values of its cells, and its products of words in range.
def SourceBand(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,
  +source: BandConcatExecution.TensorContent<Element>,+first: U32,+rows: U32,+content: Buffer.Content<Element>) -> Data:
  Shape.Both<BandShape.band_fields(~Element,part_geometry(~Element,joining,source),first,rows,content),
    Shape.Both<Buffer.ContentReady<Element,content>,
      Shape.Both<BandValues.band_values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),source,part_geometry(~Element,joining,source),first,rows,content),
        Fitted<joining,part_extent(~Element,joining,source),capacity>>>>

# A part at one band of the output: its band as the one leaf of a whole
# tensor of the band shape of the part.
def PartAt(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  owner: BandConcatExecution.TensorOwnerModel<Element>,+source: BandConcatExecution.TensorContent<Element>) -> Data:
  match owner:
    case BandConcatExecution.TensorOwner{Model.Leaf{+buffer,leaf_first,leaf_rows},geometry,shape,+fill,status,workers}:
      Shape.Both<{BandConcatExecution.TensorOwner{Model.Leaf{buffer,leaf_first,leaf_rows},geometry,shape,fill,status,workers} ==
          BandConcatExecution.band_owner(~Element,joining,part_extent(~Element,joining,source),fill,rows,buffer) : BandConcatExecution.TensorOwnerModel<Element>},
        SourceBand(~Element,~band_value,joining,capacity,source,first,rows,Buffer.content(~Element,buffer))>
    case BandConcatExecution.TensorOwner{Model.Fork{left,right},geometry,shape,fill,status,workers}: Empty

def PartsAt(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  parts: BandConcatExecution.PartsOwnerModel<Element>,sources: BandConcatExecution.PartContents<Element>) -> Data:
  match parts sources:
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.SingleContent{+source}: PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source)
    case BandConcatExecution.ConsOwner{owner,rest} BandConcatExecution.ConsContent{+source,others}:
      Shape.Both<PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source),PartsAt(~Element,~band_value,joining,capacity,first,rows,rest,others)>
    case parts sources: Empty

# The parts in the joining of the output.
type Sources<-Element: Data> is Data:
  Sources{joining: BandConcat.Joining,capacity: U32,parts: BandConcatExecution.PartContents<Element>}

def PartsIn(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+context: Sources<Element>,+first: U32,+rows: U32,
  parts: BandConcatExecution.PartsOwnerModel<Element>) -> Data:
  Sources{+joining,+capacity,sources} = context
  PartsAt(~Element,~band_value,joining,capacity,first,rows,parts,sources)

def ValidAt(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,valid: Bool,+context: Sources<Element>,+first: U32,+rows: U32,
  parts: BandConcatExecution.PartsOwnerModel<Element>) -> Data:
  match valid:
    case True{}: PartsIn(~Element,~band_value,context,first,rows,parts)
    case False{}: Unit

# A gathered band holds the band of every part when the gather found it valid.
def JoinedAt(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+context: Sources<Element>,+joined: BandConcatExecution.Joined<Element>,
  +first: U32,+rows: U32) -> Data:
  BandConcatExecution.Joined{parts,fill,valid} = joined
  ValidAt(~Element,~band_value,valid,context,first,rows,parts)

def AllAt(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+context: Sources<Element>,gathered: BandConcatExecution.Gathered(~Element)) -> Data:
  Model.ShapedAll(~BandConcatExecution.Joined<Element>,~Sources<Element>,~JoinedAt(~Element,~band_value),context,gathered)

def invalid_at(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+context: Sources<Element>,gathered: BandConcatExecution.Gathered(~Element)) ->
  AllAt(~Element,~band_value,context,BandConcatExecution.invalid(~Element,gathered)):
  match gathered:
    case Model.Piece{joined,first,rows}:
      match joined:
        case BandConcatExecution.Joined{parts,fill,valid}: Unit{}
    case Model.Pieces{left,right}: Shape.Both{invalid_at(~Element,~band_value,context,left),invalid_at(~Element,~band_value,context,right)}

# The band of a part is the band owner of the part's shape.
def band_owned(~Element: Data,+source: BandConcatExecution.TensorContent<Element>,+joining: BandConcat.Joining,+shape: List<&2,U32>,+fill: Element,+rows: U32,+buffer: Buffer.Model<Element>,
  same_shape: {BandConcatExecution.tensor_shape(~Element,source) == shape : List<&2,U32>}) ->
  {BandConcatExecution.band_owner(~Element,joining,BandConcat.extent(joining,shape),fill,rows,buffer) == BandConcatExecution.band_owner(~Element,joining,part_extent(~Element,joining,source),fill,rows,buffer) : BandConcatExecution.TensorOwnerModel<Element>}:
  Equal.cong(List<&2,U32>,BandConcatExecution.TensorOwnerModel<Element>,extents => BandConcatExecution.band_owner(~Element,joining,BandConcat.extent(joining,extents),fill,rows,buffer),shape,BandConcatExecution.tensor_shape(~Element,source),
    Equal.sym(List<&2,U32>,BandConcatExecution.tensor_shape(~Element,source),shape,same_shape))

# The bands of the last part, each the only band of its piece.
def last_bands_at(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,
  +source: BandConcatExecution.TensorContent<Element>,+shape: List<&2,U32>,+fill: Element,bands: Model.Model<Element>,
  +same_shape: {BandConcatExecution.tensor_shape(~Element,source) == shape : List<&2,U32>},
  fields: BandShape.Contents(~Element,part_geometry(~Element,joining,source),Model.content(~Element,bands)),
  ready: Model.ContentReady(~Element,Model.content(~Element,bands)),
  values: BandValues.Values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),source,part_geometry(~Element,joining,source),Model.content(~Element,bands)),
  +fitted: Fitted<joining,part_extent(~Element,joining,source),capacity>) ->
  AllAt(~Element,~band_value,Sources{joining,capacity,BandConcatExecution.SingleContent{source}},BandConcatExecution.last_bands(~Element,joining,BandConcat.extent(joining,shape),fill,bands)):
  match bands:
    case Model.Leaf{+buffer,+first,+rows}:
      Shape.Both{band_owned(~Element,source,joining,shape,fill,rows,buffer,same_shape),Shape.Both{fields,Shape.Both{ready,Shape.Both{values,fitted}}}}
    case Model.Fork{left,right}:
      Shape.Both{left_fields,right_fields} = fields
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_values,right_values} = values
      Shape.Both{last_bands_at(~Element,~band_value,joining,capacity,source,shape,fill,left,same_shape,left_fields,left_ready,left_values,fitted),
        last_bands_at(~Element,~band_value,joining,capacity,source,shape,fill,right,same_shape,right_fields,right_ready,right_values,fitted)}

# A band joined in front of a valid gathered band of the same rows keeps it
# valid, and the part of the band holds there.
def joined_valid(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,chosen: Bool,+valid: Bool,+joining: BandConcat.Joining,+capacity: U32,
  +source: BandConcatExecution.TensorContent<Element>,+others: BandConcatExecution.PartContents<Element>,+first: U32,+rows: U32,+other_first: U32,+other_rows: U32,
  +owner: BandConcatExecution.TensorOwnerModel<Element>,+parts: BandConcatExecution.PartsOwnerModel<Element>,
  +classified: {valid && U32.is_eq(first,other_first) && U32.is_eq(rows,other_rows) == chosen : Bool},
  head: PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source),
  rest: ValidAt(~Element,~band_value,valid,Sources{joining,capacity,others},other_first,other_rows,parts)) ->
  ValidAt(~Element,~band_value,chosen,Sources{joining,capacity,BandConcatExecution.ConsContent{source,others}},first,rows,BandConcatExecution.ConsOwner{owner,parts}):
  match chosen:
    case False{}: Unit{}
    case True{}:
      +both = Boolean.left(valid && U32.is_eq(first,other_first),U32.is_eq(rows,other_rows),classified)
      +valid_true = Boolean.left(valid,U32.is_eq(first,other_first),both)
      +first_same = Arithmetic.equal_of_is_eq(first,other_first,Boolean.right(valid,U32.is_eq(first,other_first),both))
      +rows_same = Arithmetic.equal_of_is_eq(rows,other_rows,Boolean.right(valid && U32.is_eq(first,other_first),U32.is_eq(rows,other_rows),classified))
      Shape.Both{head,
        %Equal.sym(U32,first,other_first,first_same) : PartsAt(~Element,~band_value,joining,capacity,_,rows,parts,others)
        %Equal.sym(U32,rows,other_rows,rows_same) : PartsAt(~Element,~band_value,joining,capacity,other_first,_,parts,others)
        %valid_true : ValidAt(~Element,~band_value,_,Sources{joining,capacity,others},other_first,other_rows,parts)
        rest}

# The bands of a part joined in front of the gathered bands of the parts
# after it.
def joined_bands_at(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,
  +source: BandConcatExecution.TensorContent<Element>,+others: BandConcatExecution.PartContents<Element>,+shape: List<&2,U32>,+fill: Element,
  bands: Model.Model<Element>,gathered: BandConcatExecution.Gathered(~Element),
  +same_shape: {BandConcatExecution.tensor_shape(~Element,source) == shape : List<&2,U32>},
  fields: BandShape.Contents(~Element,part_geometry(~Element,joining,source),Model.content(~Element,bands)),
  ready: Model.ContentReady(~Element,Model.content(~Element,bands)),
  values: BandValues.Values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),source,part_geometry(~Element,joining,source),Model.content(~Element,bands)),
  +fitted: Fitted<joining,part_extent(~Element,joining,source),capacity>,
  rest: AllAt(~Element,~band_value,Sources{joining,capacity,others},gathered)) ->
  AllAt(~Element,~band_value,Sources{joining,capacity,BandConcatExecution.ConsContent{source,others}},BandConcatExecution.joined_bands(~Element,joining,BandConcat.extent(joining,shape),fill,bands,gathered)):
  match bands gathered:
    case Model.Leaf{+buffer,+first,+rows} Model.Piece{other,+other_first,+other_rows}:
      match other:
        case BandConcatExecution.Joined{+parts,other_fill,+valid}:
          joined_valid(~Element,~band_value,valid && U32.is_eq(first,other_first) && U32.is_eq(rows,other_rows),valid,joining,capacity,source,others,first,rows,other_first,other_rows,
            BandConcatExecution.band_owner(~Element,joining,BandConcat.extent(joining,shape),fill,rows,buffer),parts,{==},
            Shape.Both{band_owned(~Element,source,joining,shape,fill,rows,buffer,same_shape),Shape.Both{fields,Shape.Both{ready,Shape.Both{values,fitted}}}},rest)
    case Model.Fork{left,right} Model.Pieces{other_left,other_right}:
      Shape.Both{left_fields,right_fields} = fields
      Shape.Both{left_ready,right_ready} = ready
      Shape.Both{left_values,right_values} = values
      Shape.Both{left_rest,right_rest} = rest
      Shape.Both{joined_bands_at(~Element,~band_value,joining,capacity,source,others,shape,fill,left,other_left,same_shape,left_fields,left_ready,left_values,fitted,left_rest),
        joined_bands_at(~Element,~band_value,joining,capacity,source,others,shape,fill,right,other_right,same_shape,right_fields,right_ready,right_values,fitted,right_rest)}
    case Model.Leaf{buffer,first,rows} Model.Pieces{+other_left,+other_right}:
      invalid_at(~Element,~band_value,Sources{joining,capacity,BandConcatExecution.ConsContent{source,others}},Model.Pieces{other_left,other_right})
    case Model.Fork{left,right} Model.Piece{+other,+other_first,+other_rows}:
      invalid_at(~Element,~band_value,Sources{joining,capacity,BandConcatExecution.ConsContent{source,others}},Model.Piece{other,other_first,other_rows})

def last_part_at(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,
  owner: BandConcatExecution.TensorOwnerModel<Element>,
  contract: SourceContract(~Element,~band_value,joining,capacity,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)))) ->
  AllAt(~Element,~band_value,Sources{joining,capacity,BandConcatExecution.SingleContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))}},
    BandConcatExecution.last_part(~Element,joining,owner)):
  BandConcatExecution.TensorOwner{+bands,+geometry,+shape,+fill,+status,+workers} = owner
  Shape.Both{BandShape.WellFormed{metadata,ready,coverage},Shape.Both{values,fitted}} = contract
  +source = {BandConcatExecution.TensorContent{Model.contents(~Element,Model.pack(~Element,bands)),geometry,shape,fill,status,workers} : BandConcatExecution.TensorContent<Element>}
  last_bands_at(~Element,~band_value,joining,capacity,source,shape,fill,bands,{==},
    %Model.packed_contents(~Element,bands) : BandShape.Contents(~Element,BandConcat.geometry(joining,BandConcat.extent(joining,shape)),_) metadata,
    %Model.packed_contents(~Element,bands) : Model.ContentReady(~Element,_) ready,
    %Model.packed_contents(~Element,bands) :
      BandValues.Values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),source,BandConcat.geometry(joining,BandConcat.extent(joining,shape)),_) values,
    fitted)

def joined_part_at(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,
  owner: BandConcatExecution.TensorOwnerModel<Element>,+others: BandConcatExecution.PartContents<Element>,gathered: BandConcatExecution.Gathered(~Element),
  contract: SourceContract(~Element,~band_value,joining,capacity,BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner))),
  rest: AllAt(~Element,~band_value,Sources{joining,capacity,others},gathered)) ->
  AllAt(~Element,~band_value,Sources{joining,capacity,BandConcatExecution.ConsContent{BandConcatExecution.tensor_contents(~Element,BandConcatExecution.tensor_owner_pack(~Element,owner)),others}},
    BandConcatExecution.joined_part(~Element,joining,owner,gathered)):
  BandConcatExecution.TensorOwner{+bands,+geometry,+shape,+fill,+status,+workers} = owner
  Shape.Both{BandShape.WellFormed{metadata,ready,coverage},Shape.Both{values,fitted}} = contract
  +source = {BandConcatExecution.TensorContent{Model.contents(~Element,Model.pack(~Element,bands)),geometry,shape,fill,status,workers} : BandConcatExecution.TensorContent<Element>}
  joined_bands_at(~Element,~band_value,joining,capacity,source,others,shape,fill,bands,gathered,{==},
    %Model.packed_contents(~Element,bands) : BandShape.Contents(~Element,BandConcat.geometry(joining,BandConcat.extent(joining,shape)),_) metadata,
    %Model.packed_contents(~Element,bands) : Model.ContentReady(~Element,_) ready,
    %Model.packed_contents(~Element,bands) :
      BandValues.Values(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),source,BandConcat.geometry(joining,BandConcat.extent(joining,shape)),_) values,
    fitted,rest)

# Every piece the gather makes holds the band of every part where it is valid.
def gather_at(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,
  sources: BandConcatExecution.PartsOwnerModel<Element>,
  contracts: SourceContracts(~Element,~band_value,joining,capacity,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)))) ->
  AllAt(~Element,~band_value,Sources{joining,capacity,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))},
    BandConcatExecution.gather(~Element,joining,sources)):
  match sources:
    case BandConcatExecution.SingleOwner{input}: last_part_at(~Element,~band_value,joining,capacity,input,contracts)
    case BandConcatExecution.ConsOwner{input,+rest}:
      Shape.Both{head,tail} = contracts
      joined_part_at(~Element,~band_value,joining,capacity,input,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,rest)),
        BandConcatExecution.gather(~Element,joining,rest),head,gather_at(~Element,~band_value,joining,capacity,rest,tail))

# ---- The gathered pieces follow the plan the concat runs ----

def owner_layout(~Element: Data,owner: BandConcatExecution.TensorOwnerModel<Element>) -> Bands.Plan:
  BandConcatExecution.TensorOwner{bands,geometry,shape,fill,status,workers} = owner
  Model.layout(~Element,bands)

# Every part has the plan the concat runs.
def Layouts(~Element: Data,sources: BandConcatExecution.PartsOwnerModel<Element>,+layout: Bands.Plan) -> Data:
  match sources:
    case BandConcatExecution.SingleOwner{input}: {owner_layout(~Element,input) == layout : Bands.Plan}
    case BandConcatExecution.ConsOwner{input,rest}: Shape.Both<{owner_layout(~Element,input) == layout : Bands.Plan},Layouts(~Element,rest,layout)>

# When every part has the band plan the concat runs, the gathered pieces have
# that plan.
def gathered_layout(~Element: Data,+joining: BandConcat.Joining,sources: BandConcatExecution.PartsOwnerModel<Element>,+layout: Bands.Plan,layouts: Layouts(~Element,sources,layout)) ->
  {BandConcatExecution.gathered_plan(~Element,BandConcatExecution.gather(~Element,joining,sources)) == layout : Bands.Plan}:
  match sources:
    case BandConcatExecution.SingleOwner{BandConcatExecution.TensorOwner{+bands,geometry,+shape,+fill,status,workers}}:
      Equal.trans(Bands.Plan,BandConcatExecution.gathered_plan(~Element,BandConcatExecution.last_bands(~Element,joining,BandConcat.extent(joining,shape),fill,bands)),Model.layout(~Element,bands),layout,
        BandConcatExecution.last_bands_plan(~Element,joining,BandConcat.extent(joining,shape),fill,bands),layouts)
    case BandConcatExecution.ConsOwner{BandConcatExecution.TensorOwner{+bands,geometry,+shape,+fill,status,workers},+rest}:
      Shape.Both{+head,tail} = layouts
      +rest_plan = BandConcatExecution.gathered_plan(~Element,BandConcatExecution.gather(~Element,joining,rest))
      +rest_layout = gathered_layout(~Element,joining,rest,layout,tail)
      +same = {
        %Equal.sym(Bands.Plan,Model.layout(~Element,bands),layout,head) : {Bands.same_plan(_,rest_plan) == True{} : Bool}
        %Equal.sym(Bands.Plan,rest_plan,layout,rest_layout) : {Bands.same_plan(layout,_) == True{} : Bool}
        BandCombineLeaf.same_plan_self(layout)
        : {Bands.same_plan(Model.layout(~Element,bands),rest_plan) == True{} : Bool}}
      Equal.trans(Bands.Plan,BandConcatExecution.gathered_plan(~Element,BandConcatExecution.joined_bands(~Element,joining,BandConcat.extent(joining,shape),fill,bands,BandConcatExecution.gather(~Element,joining,rest))),
        Model.layout(~Element,bands),layout,
        BandConcatExecution.joined_bands_plan(~Element,joining,BandConcat.extent(joining,shape),fill,bands,BandConcatExecution.gather(~Element,joining,rest),same),head)

# ---- One band of the output: the leaf concat of the bands of the parts ----

def head_extent(~Element: Data,+joining: BandConcat.Joining,sources: BandConcatExecution.PartContents<Element>) -> Nat:
  match sources:
    case BandConcatExecution.SingleContent{source}: U32.to_nat(part_extent(~Element,joining,source))
    case BandConcatExecution.ConsContent{source,others}: U32.to_nat(part_extent(~Element,joining,source))

# The cells an owner holds as a whole tensor.
def owner_content(~Element: Data,owner: BandConcatExecution.TensorOwnerModel<Element>) -> Buffer.Content<Element>:
  match owner:
    case BandConcatExecution.TensorOwner{Model.Leaf{buffer,first,rows},geometry,shape,fill,status,workers}: Buffer.content(~Element,buffer)
    case BandConcatExecution.TensorOwner{Model.Fork{left,right},geometry,shape,+fill,status,workers}: Buffer.Content{Tree.fresh(Element,0n,fill),0n}

def layer_count(~Element: Data,+joining: BandConcat.Joining,+source: BandConcatExecution.TensorContent<Element>,+rows: U32) -> U32:
  U32.from_nat(ShapeLaws.elements(BandConcat.band_shape(joining,part_extent(~Element,joining,source),rows)))

# The bands of the parts as layers along the axis of a band.
def layers_of(~Element: Data,+joining: BandConcat.Joining,parts: BandConcatExecution.PartsOwnerModel<Element>,sources: BandConcatExecution.PartContents<Element>,+rows: U32) ->
  ConcatLeaf.Layers<Buffer.Content<Element>>:
  match parts sources:
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.SingleContent{+source}:
      ConcatLeaf.Alone{owner_content(~Element,owner),part_extent(~Element,joining,source),layer_count(~Element,joining,source,rows)}
    case BandConcatExecution.ConsOwner{owner,rest} BandConcatExecution.ConsContent{+source,others}:
      ConcatLeaf.Before{owner_content(~Element,owner),part_extent(~Element,joining,source),layer_count(~Element,joining,source,rows),layers_of(~Element,joining,rest,others,rows)}
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.ConsContent{+source,others}:
      ConcatLeaf.Alone{owner_content(~Element,owner),part_extent(~Element,joining,source),layer_count(~Element,joining,source,rows)}
    case BandConcatExecution.ConsOwner{owner,rest} BandConcatExecution.SingleContent{+source}:
      ConcatLeaf.Alone{owner_content(~Element,owner),part_extent(~Element,joining,source),layer_count(~Element,joining,source,rows)}

# The whole-tensor facts of the band of one part: its cells, ready, read as stored.
def piece_facts(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  +source: BandConcatExecution.TensorContent<Element>,+buffer: Buffer.Model<Element>,
  band: SourceBand(~Element,~band_value,joining,capacity,source,first,rows,Buffer.content(~Element,buffer))) ->
  GatherValues.Facts(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),Buffer.content(~Element,buffer),BandConcat.band_shape(joining,part_extent(~Element,joining,source),rows),Buffer.content(~Element,buffer)):
  Shape.Both{+fields,Shape.Both{ready,Shape.Both{values,+fitted}}} = band
  +geometry = part_geometry(~Element,joining,source)
  +length = Buffer.content_length(~Element,Buffer.content(~Element,buffer))
  Shape.Both{Contract.WellFormed{
      Equal.trans(Nat,length,BandValues.cell_count(geometry,rows),ShapeLaws.elements(BandConcat.band_shape(joining,part_extent(~Element,joining,source),rows)),
        BandValues.leaf_length(geometry,first,rows,length,fields),
        band_cells(joining,part_extent(~Element,joining,source),first,rows,capacity,fitted,fitted_rows(geometry,first,rows,length,fields))),
      ready},
    Contract.stored_values(~Element,Buffer.content(~Element,buffer))}

def Pieces(~Element: Data) -> Data:
  ConcatLeaf.Pieces<Element,Buffer.Content<Element>,Contract.stored(~Element)>

def last_piece(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  owner: BandConcatExecution.TensorOwnerModel<Element>,+source: BandConcatExecution.TensorContent<Element>,
  fact: PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source)) -> Pieces(~Element):
  match owner:
    case BandConcatExecution.TensorOwner{Model.Leaf{+buffer,leaf_first,leaf_rows},geometry,shape,+fill,status,workers}:
      Shape.Both{same_owner,band} = fact
      +band_shape = BandConcat.band_shape(joining,part_extent(~Element,joining,source),rows)
      ConcatLeaf.Last{Buffer.content(~Element,buffer),band_shape,Model.Leaf{buffer,0,1},Bands.Geometry{1,1,1,Tensors.count_or_zero(band_shape)},fill,1,
        WholeLeaf.held(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),Buffer.content(~Element,buffer),buffer,band_shape,fill,1,
          piece_facts(~Element,~band_value,joining,capacity,first,rows,source,buffer,band))}
    case BandConcatExecution.TensorOwner{Model.Fork{left,right},geometry,shape,fill,status,workers}: Empty.absurd(Pieces(~Element),fact)

def next_piece(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  owner: BandConcatExecution.TensorOwnerModel<Element>,+source: BandConcatExecution.TensorContent<Element>,
  fact: PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source),rest: Pieces(~Element)) -> Pieces(~Element):
  match owner:
    case BandConcatExecution.TensorOwner{Model.Leaf{+buffer,leaf_first,leaf_rows},geometry,shape,+fill,status,workers}:
      Shape.Both{same_owner,band} = fact
      +band_shape = BandConcat.band_shape(joining,part_extent(~Element,joining,source),rows)
      ConcatLeaf.Next{Buffer.content(~Element,buffer),band_shape,Model.Leaf{buffer,0,1},Bands.Geometry{1,1,1,Tensors.count_or_zero(band_shape)},fill,1,
        WholeLeaf.held(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),Buffer.content(~Element,buffer),buffer,band_shape,fill,1,
          piece_facts(~Element,~band_value,joining,capacity,first,rows,source,buffer,band)),rest}
    case BandConcatExecution.TensorOwner{Model.Fork{left,right},geometry,shape,fill,status,workers}: Empty.absurd(Pieces(~Element),fact)

# The bands of the parts at one band, as the pieces of the leaf concat.
def pieces_of(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  parts: BandConcatExecution.PartsOwnerModel<Element>,sources: BandConcatExecution.PartContents<Element>,
  +facts: PartsAt(~Element,~band_value,joining,capacity,first,rows,parts,sources)) -> Pieces(~Element):
  match parts sources:
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.SingleContent{+source}:
      last_piece(~Element,~band_value,joining,capacity,first,rows,owner,source,facts)
    case BandConcatExecution.ConsOwner{+owner,+rest} BandConcatExecution.ConsContent{+source,+others}:
      next_piece(~Element,~band_value,joining,capacity,first,rows,owner,source,
        Shape.first(PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source),PartsAt(~Element,~band_value,joining,capacity,first,rows,rest,others),facts),
        pieces_of(~Element,~band_value,joining,capacity,first,rows,rest,others,
          Shape.second(PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source),PartsAt(~Element,~band_value,joining,capacity,first,rows,rest,others),facts)))
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.ConsContent{source,others}: Empty.absurd(Pieces(~Element),facts)
    case BandConcatExecution.ConsOwner{owner,rest} BandConcatExecution.SingleContent{source}: Empty.absurd(Pieces(~Element),facts)

def PiecesPacked(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  +parts: BandConcatExecution.PartsOwnerModel<Element>,+sources: BandConcatExecution.PartContents<Element>,
  +facts: PartsAt(~Element,~band_value,joining,capacity,first,rows,parts,sources)) -> Data:
  {ConcatLeaf.packed(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),pieces_of(~Element,~band_value,joining,capacity,first,rows,parts,sources,facts)) ==
    BandConcatExecution.parts_owner_pack(~Element,parts) : Tensors.Parts<Element>}

# The band of one part packs to its band owner.
def single_packed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  owner: BandConcatExecution.TensorOwnerModel<Element>,+source: BandConcatExecution.TensorContent<Element>,+fact: PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source)) ->
  PiecesPacked(~Element,~band_value,joining,capacity,first,rows,BandConcatExecution.SingleOwner{owner},BandConcatExecution.SingleContent{source},fact):
  match owner:
    case BandConcatExecution.TensorOwner{Model.Leaf{+buffer,+leaf_first,+leaf_rows},+geometry,+shape,+fill,+status,+workers}:
      Shape.Both{+same_owner,band} = fact
      %Equal.sym(BandConcatExecution.TensorOwnerModel<Element>,BandConcatExecution.TensorOwner{Model.Leaf{buffer,leaf_first,leaf_rows},geometry,shape,fill,status,workers},
        BandConcatExecution.band_owner(~Element,joining,part_extent(~Element,joining,source),fill,rows,buffer),same_owner) :
        {Tensors.Single{Tensors.whole(~Element,Buffer.pack(~Element,buffer),BandConcat.band_shape(joining,part_extent(~Element,joining,source),rows),fill,Tensors.Usable{},1)} ==
          Tensors.Single{BandConcatExecution.tensor_owner_pack(~Element,_)} : Tensors.Parts<Element>}
      {==}
    case BandConcatExecution.TensorOwner{Model.Fork{+left,+right},+geometry,+shape,+fill,+status,+workers}:
      Empty.absurd(PiecesPacked(~Element,~band_value,joining,capacity,first,rows,
        BandConcatExecution.SingleOwner{BandConcatExecution.TensorOwner{Model.Fork{left,right},geometry,shape,fill,status,workers}},BandConcatExecution.SingleContent{source},fact),fact)

# The band of one part in front of the others packs to its band owner in
# front of theirs.
def cons_packed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  owner: BandConcatExecution.TensorOwnerModel<Element>,+rest: BandConcatExecution.PartsOwnerModel<Element>,+source: BandConcatExecution.TensorContent<Element>,
  +others: BandConcatExecution.PartContents<Element>,
  +facts: Shape.Both<PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source),PartsAt(~Element,~band_value,joining,capacity,first,rows,rest,others)>,
  recurse: @+tail: PartsAt(~Element,~band_value,joining,capacity,first,rows,rest,others) -> PiecesPacked(~Element,~band_value,joining,capacity,first,rows,rest,others,tail)) ->
  PiecesPacked(~Element,~band_value,joining,capacity,first,rows,BandConcatExecution.ConsOwner{owner,rest},BandConcatExecution.ConsContent{source,others},facts):
  match owner:
    case BandConcatExecution.TensorOwner{Model.Leaf{+buffer,+leaf_first,+leaf_rows},+geometry,+shape,+fill,+status,+workers}:
      Shape.Both{Shape.Both{+same_owner,band},+tail} = facts
      +rest_pieces = pieces_of(~Element,~band_value,joining,capacity,first,rows,rest,others,tail)
      %Equal.sym(Tensors.Parts<Element>,ConcatLeaf.packed(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),rest_pieces),BandConcatExecution.parts_owner_pack(~Element,rest),recurse(tail)) :
        {Tensors.Cons{Tensors.whole(~Element,Buffer.pack(~Element,buffer),BandConcat.band_shape(joining,part_extent(~Element,joining,source),rows),fill,Tensors.Usable{},1),_} ==
          Tensors.Cons{BandConcatExecution.tensor_owner_pack(~Element,BandConcatExecution.TensorOwner{Model.Leaf{buffer,leaf_first,leaf_rows},geometry,shape,fill,status,workers}),BandConcatExecution.parts_owner_pack(~Element,rest)} : Tensors.Parts<Element>}
      %Equal.sym(BandConcatExecution.TensorOwnerModel<Element>,BandConcatExecution.TensorOwner{Model.Leaf{buffer,leaf_first,leaf_rows},geometry,shape,fill,status,workers},
        BandConcatExecution.band_owner(~Element,joining,part_extent(~Element,joining,source),fill,rows,buffer),same_owner) :
        {Tensors.Cons{Tensors.whole(~Element,Buffer.pack(~Element,buffer),BandConcat.band_shape(joining,part_extent(~Element,joining,source),rows),fill,Tensors.Usable{},1),BandConcatExecution.parts_owner_pack(~Element,rest)} ==
          Tensors.Cons{BandConcatExecution.tensor_owner_pack(~Element,_),BandConcatExecution.parts_owner_pack(~Element,rest)} : Tensors.Parts<Element>}
      {==}
    case BandConcatExecution.TensorOwner{Model.Fork{+left,+right},+geometry,+shape,+fill,+status,+workers}:
      +banded = {BandConcatExecution.TensorOwner{Model.Fork{left,right},geometry,shape,fill,status,workers} : BandConcatExecution.TensorOwnerModel<Element>}
      Empty.absurd(PiecesPacked(~Element,~band_value,joining,capacity,first,rows,BandConcatExecution.ConsOwner{banded,rest},BandConcatExecution.ConsContent{source,others},facts),
        Shape.first(PartAt(~Element,~band_value,joining,capacity,first,rows,banded,source),PartsAt(~Element,~band_value,joining,capacity,first,rows,rest,others),facts))

# The pieces pack to the band owners the gather holds.
def pieces_packed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  +parts: BandConcatExecution.PartsOwnerModel<Element>,+sources: BandConcatExecution.PartContents<Element>,
  +facts: PartsAt(~Element,~band_value,joining,capacity,first,rows,parts,sources)) ->
  PiecesPacked(~Element,~band_value,joining,capacity,first,rows,parts,sources,facts):
  match parts sources:
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.SingleContent{+source}:
      single_packed(~Element,~band_value,joining,capacity,first,rows,owner,source,facts)
    case BandConcatExecution.ConsOwner{owner,+rest} BandConcatExecution.ConsContent{+source,+others}:
      cons_packed(~Element,~band_value,joining,capacity,first,rows,owner,rest,source,others,facts,
        tail => pieces_packed(~Element,~band_value,joining,capacity,first,rows,rest,others,tail))
    case BandConcatExecution.SingleOwner{+owner} BandConcatExecution.ConsContent{+source,+others}:
      Empty.absurd(PiecesPacked(~Element,~band_value,joining,capacity,first,rows,BandConcatExecution.SingleOwner{owner},BandConcatExecution.ConsContent{source,others},facts),facts)
    case BandConcatExecution.ConsOwner{+owner,+rest} BandConcatExecution.SingleContent{+source}:
      Empty.absurd(PiecesPacked(~Element,~band_value,joining,capacity,first,rows,BandConcatExecution.ConsOwner{owner,rest},BandConcatExecution.SingleContent{source},facts),facts)

def PiecesLined(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  +parts: BandConcatExecution.PartsOwnerModel<Element>,+sources: BandConcatExecution.PartContents<Element>,
  +facts: PartsAt(~Element,~band_value,joining,capacity,first,rows,parts,sources)) -> Data:
  {ConcatLeaf.sources_of(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),pieces_of(~Element,~band_value,joining,capacity,first,rows,parts,sources,facts)) ==
    ConcatLeaf.lined(~Buffer.Content<Element>,layers_of(~Element,joining,parts,sources,rows),BandConcat.band_outer(joining,rows),BandConcat.band_inner(joining,rows)) : ConcatLeaf.Sources<Buffer.Content<Element>>}

# The band of one part is one layer.
def single_lined(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  owner: BandConcatExecution.TensorOwnerModel<Element>,+source: BandConcatExecution.TensorContent<Element>,+fact: PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source)) ->
  PiecesLined(~Element,~band_value,joining,capacity,first,rows,BandConcatExecution.SingleOwner{owner},BandConcatExecution.SingleContent{source},fact):
  match owner:
    case BandConcatExecution.TensorOwner{Model.Leaf{buffer,leaf_first,leaf_rows},geometry,shape,fill,status,workers}:
      Shape.Both{same_owner,band} = fact
      {==}
    case BandConcatExecution.TensorOwner{Model.Fork{+left,+right},+geometry,+shape,+fill,+status,+workers}:
      Empty.absurd(PiecesLined(~Element,~band_value,joining,capacity,first,rows,
        BandConcatExecution.SingleOwner{BandConcatExecution.TensorOwner{Model.Fork{left,right},geometry,shape,fill,status,workers}},BandConcatExecution.SingleContent{source},fact),fact)

# The band of one part is the layer in front of the layers of the others.
def cons_lined(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  owner: BandConcatExecution.TensorOwnerModel<Element>,+rest: BandConcatExecution.PartsOwnerModel<Element>,+source: BandConcatExecution.TensorContent<Element>,
  +others: BandConcatExecution.PartContents<Element>,
  +facts: Shape.Both<PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source),PartsAt(~Element,~band_value,joining,capacity,first,rows,rest,others)>,
  recurse: @+tail: PartsAt(~Element,~band_value,joining,capacity,first,rows,rest,others) -> PiecesLined(~Element,~band_value,joining,capacity,first,rows,rest,others,tail)) ->
  PiecesLined(~Element,~band_value,joining,capacity,first,rows,BandConcatExecution.ConsOwner{owner,rest},BandConcatExecution.ConsContent{source,others},facts):
  match owner:
    case BandConcatExecution.TensorOwner{Model.Leaf{+buffer,leaf_first,leaf_rows},geometry,shape,fill,status,workers}:
      Shape.Both{Shape.Both{same_owner,band},+tail} = facts
      Equal.cong(ConcatLeaf.Sources<Buffer.Content<Element>>,ConcatLeaf.Sources<Buffer.Content<Element>>,
        lined => ConcatLeaf.Then{Buffer.content(~Element,buffer),BandConcat.band_shape(joining,part_extent(~Element,joining,source),rows),lined},
        ConcatLeaf.sources_of(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),pieces_of(~Element,~band_value,joining,capacity,first,rows,rest,others,tail)),
        ConcatLeaf.lined(~Buffer.Content<Element>,layers_of(~Element,joining,rest,others,rows),BandConcat.band_outer(joining,rows),BandConcat.band_inner(joining,rows)),recurse(tail))
    case BandConcatExecution.TensorOwner{Model.Fork{+left,+right},+geometry,+shape,+fill,+status,+workers}:
      +banded = {BandConcatExecution.TensorOwner{Model.Fork{left,right},geometry,shape,fill,status,workers} : BandConcatExecution.TensorOwnerModel<Element>}
      Empty.absurd(PiecesLined(~Element,~band_value,joining,capacity,first,rows,BandConcatExecution.ConsOwner{banded,rest},BandConcatExecution.ConsContent{source,others},facts),
        Shape.first(PartAt(~Element,~band_value,joining,capacity,first,rows,banded,source),PartsAt(~Element,~band_value,joining,capacity,first,rows,rest,others),facts))

# The pieces line up the layers of the parts.
def pieces_lined(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  +parts: BandConcatExecution.PartsOwnerModel<Element>,+sources: BandConcatExecution.PartContents<Element>,
  +facts: PartsAt(~Element,~band_value,joining,capacity,first,rows,parts,sources)) ->
  PiecesLined(~Element,~band_value,joining,capacity,first,rows,parts,sources,facts):
  match parts sources:
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.SingleContent{+source}:
      single_lined(~Element,~band_value,joining,capacity,first,rows,owner,source,facts)
    case BandConcatExecution.ConsOwner{owner,+rest} BandConcatExecution.ConsContent{+source,+others}:
      cons_lined(~Element,~band_value,joining,capacity,first,rows,owner,rest,source,others,facts,
        tail => pieces_lined(~Element,~band_value,joining,capacity,first,rows,rest,others,tail))
    case BandConcatExecution.SingleOwner{+owner} BandConcatExecution.ConsContent{+source,+others}:
      Empty.absurd(PiecesLined(~Element,~band_value,joining,capacity,first,rows,BandConcatExecution.SingleOwner{owner},BandConcatExecution.ConsContent{source,others},facts),facts)
    case BandConcatExecution.ConsOwner{+owner,+rest} BandConcatExecution.SingleContent{+source}:
      Empty.absurd(PiecesLined(~Element,~band_value,joining,capacity,first,rows,BandConcatExecution.ConsOwner{owner,rest},BandConcatExecution.SingleContent{source},facts),facts)

# The layers have the extents of the parts.
def layers_total(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  +parts: BandConcatExecution.PartsOwnerModel<Element>,+sources: BandConcatExecution.PartContents<Element>,
  +facts: PartsAt(~Element,~band_value,joining,capacity,first,rows,parts,sources)) ->
  {ConcatLeaf.layers_total(~Buffer.Content<Element>,layers_of(~Element,joining,parts,sources,rows)) == parts_total(~Element,joining,sources) : Nat}:
  match parts sources:
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.SingleContent{source}: {==}
    case BandConcatExecution.ConsOwner{owner,+rest} BandConcatExecution.ConsContent{+source,+others}:
      match facts:
        case Shape.Both{head_fact,+tail}:
          Equal.cong(Nat,Nat,total => Nat.add(U32.to_nat(part_extent(~Element,joining,source)),total),
            ConcatLeaf.layers_total(~Buffer.Content<Element>,layers_of(~Element,joining,rest,others,rows)),parts_total(~Element,joining,others),
            layers_total(~Element,~band_value,joining,capacity,first,rows,rest,others,tail))
    case BandConcatExecution.SingleOwner{+owner} BandConcatExecution.ConsContent{+source,+others}:
      Empty.absurd({ConcatLeaf.layers_total(~Buffer.Content<Element>,layers_of(~Element,joining,BandConcatExecution.SingleOwner{owner},BandConcatExecution.ConsContent{source,others},rows)) ==
        parts_total(~Element,joining,BandConcatExecution.ConsContent{source,others}) : Nat},facts)
    case BandConcatExecution.ConsOwner{+owner,+rest} BandConcatExecution.SingleContent{+source}:
      Empty.absurd({ConcatLeaf.layers_total(~Buffer.Content<Element>,layers_of(~Element,joining,BandConcatExecution.ConsOwner{owner,rest},BandConcatExecution.SingleContent{source},rows)) ==
        parts_total(~Element,joining,BandConcatExecution.SingleContent{source}) : Nat},facts)

def right_le(+a: Nat,+b: Nat) -> {Nat.is_le(b,Nat.add(a,b)) == True{} : Bool}:
  %Order.commutative(b,a) : {Nat.is_le(b,_) == True{} : Bool}
  Order.le_add(b,a)

# The cells of a band shape grow with the extent.
def band_elements_le(+joining: BandConcat.Joining,+extent: U32,+total: U32,+rows: U32,+room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(total)) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(BandConcat.band_shape(joining,extent,rows)),ShapeLaws.elements(BandConcat.band_shape(joining,total,rows))) == True{} : Bool}:
  +outer = U32.to_nat(BandConcat.band_outer(joining,rows))
  +inner = U32.to_nat(BandConcat.band_inner(joining,rows))
  +small = Nat.mul(U32.to_nat(extent),inner)
  +large = Nat.mul(U32.to_nat(total),inner)
  %Equal.sym(Nat,ShapeLaws.elements(BandConcat.band_shape(joining,extent,rows)),Nat.mul(outer,small),ShapeLaws.elements_three(BandConcat.band_outer(joining,rows),extent,BandConcat.band_inner(joining,rows))) :
    {Nat.is_le(_,ShapeLaws.elements(BandConcat.band_shape(joining,total,rows))) == True{} : Bool}
  %Equal.sym(Nat,ShapeLaws.elements(BandConcat.band_shape(joining,total,rows)),Nat.mul(outer,large),ShapeLaws.elements_three(BandConcat.band_outer(joining,rows),total,BandConcat.band_inner(joining,rows))) :
    {Nat.is_le(Nat.mul(outer,small),_) == True{} : Bool}
  %RowMajor.multiply_commutative(small,outer) : {Nat.is_le(_,Nat.mul(outer,large)) == True{} : Bool}
  %RowMajor.multiply_commutative(large,outer) : {Nat.is_le(Nat.mul(small,outer),_) == True{} : Bool}
  RowMajor.mul_le(small,large,outer,RowMajor.mul_le(U32.to_nat(extent),U32.to_nat(total),inner,room))

# The count of one layer: its cells, which fit below the capacity with the
# cells of the band.
def layer_counted(~Element: Data,+joining: BandConcat.Joining,+source: BandConcatExecution.TensorContent<Element>,+rows: U32,+extent: U32,+capacity: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(part_extent(~Element,joining,source)),U32.to_nat(extent)) == True{} : Bool},
  +cells: {Nat.is_le(ShapeLaws.elements(BandConcat.band_shape(joining,extent,rows)),U32.to_nat(capacity)) == True{} : Bool}) ->
  {Nat.mul(U32.to_nat(BandConcat.band_outer(joining,rows)),Slices.cells(part_extent(~Element,joining,source),BandConcat.band_inner(joining,rows))) == U32.to_nat(layer_count(~Element,joining,source,rows)) : Nat}:
  +count = ShapeLaws.elements(BandConcat.band_shape(joining,part_extent(~Element,joining,source),rows))
  Equal.trans(Nat,Nat.mul(U32.to_nat(BandConcat.band_outer(joining,rows)),Slices.cells(part_extent(~Element,joining,source),BandConcat.band_inner(joining,rows))),count,U32.to_nat(U32.from_nat(count)),
    Equal.sym(Nat,count,Nat.mul(U32.to_nat(BandConcat.band_outer(joining,rows)),Slices.cells(part_extent(~Element,joining,source),BandConcat.band_inner(joining,rows))),
      ShapeLaws.elements_three(BandConcat.band_outer(joining,rows),part_extent(~Element,joining,source),BandConcat.band_inner(joining,rows))),
    Equal.sym(Nat,U32.to_nat(U32.from_nat(count)),count,
      Bounds.roundtrip(count,capacity,capacity_ok,
        Order.transitive(count,ShapeLaws.elements(BandConcat.band_shape(joining,extent,rows)),U32.to_nat(capacity),
          band_elements_le(joining,part_extent(~Element,joining,source),extent,rows,room),cells))))

def layers_counted(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  +parts: BandConcatExecution.PartsOwnerModel<Element>,+sources: BandConcatExecution.PartContents<Element>,
  +facts: PartsAt(~Element,~band_value,joining,capacity,first,rows,parts,sources),+extent: U32,
  +capacity_ok: {U32.is_le(capacity,2147483648) == True{} : Bool},
  +room: {Nat.is_le(parts_total(~Element,joining,sources),U32.to_nat(extent)) == True{} : Bool},
  +cells: {Nat.is_le(ShapeLaws.elements(BandConcat.band_shape(joining,extent,rows)),U32.to_nat(capacity)) == True{} : Bool}) ->
  ConcatLeaf.Counted(~Buffer.Content<Element>,layers_of(~Element,joining,parts,sources,rows),BandConcat.band_outer(joining,rows),BandConcat.band_inner(joining,rows)):
  match parts sources:
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.SingleContent{+source}:
      layer_counted(~Element,joining,source,rows,extent,capacity,capacity_ok,room,cells)
    case BandConcatExecution.ConsOwner{owner,+rest} BandConcatExecution.ConsContent{+source,+others}:
      match facts:
        case Shape.Both{head_fact,+tail}:
          +head = U32.to_nat(part_extent(~Element,joining,source))
          +others_total = parts_total(~Element,joining,others)
          Shape.Both{layer_counted(~Element,joining,source,rows,extent,capacity,capacity_ok,
              Order.transitive(head,Nat.add(head,others_total),U32.to_nat(extent),Order.le_add(head,others_total),room),cells),
            layers_counted(~Element,~band_value,joining,capacity,first,rows,rest,others,tail,extent,capacity_ok,
              Order.transitive(others_total,Nat.add(head,others_total),U32.to_nat(extent),right_le(head,others_total),room),cells)}
    case BandConcatExecution.SingleOwner{+owner} BandConcatExecution.ConsContent{+source,+others}:
      Empty.absurd(ConcatLeaf.Counted(~Buffer.Content<Element>,layers_of(~Element,joining,BandConcatExecution.SingleOwner{owner},BandConcatExecution.ConsContent{source,others},rows),
        BandConcat.band_outer(joining,rows),BandConcat.band_inner(joining,rows)),facts)
    case BandConcatExecution.ConsOwner{+owner,+rest} BandConcatExecution.SingleContent{+source}:
      Empty.absurd(ConcatLeaf.Counted(~Buffer.Content<Element>,layers_of(~Element,joining,BandConcatExecution.ConsOwner{owner,rest},BandConcatExecution.SingleContent{source},rows),
        BandConcat.band_outer(joining,rows),BandConcat.band_inner(joining,rows)),facts)

def Sampled(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+first: U32,+rows: U32,
  +content: Buffer.Content<Element>,+source: BandConcatExecution.TensorContent<Element>,+line: Nat,+middle: Nat,+offset: Nat) -> Data:
  {Contract.stored(~Element,content,U32.from_nat(ConcatLeaf.cell_at(part_extent(~Element,joining,source),BandConcat.band_inner(joining,rows),line,middle,offset))) ==
    tensor_original(~Element,~band_value,source,U32.from_nat(framed(U32.to_nat(part_extent(~Element,joining,source)),joining_inner(joining),outer_at(joining,first,line,offset),middle,inner_at(joining,first,line,offset)))) : Element}

# The cell of a leaf a line, a place and an offset name holds the part there.
def along_cell(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+first: U32,+rows: U32,
  +source: BandConcatExecution.TensorContent<Element>,+tree: Tree.Tree<Element>,+length: Nat,+line: Nat,+middle: Nat,+offset: Nat,
  values: BandValues.Leaf(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),source,part_geometry(~Element,joining,source),first,rows,tree,length),
  cell: Cell<joining,part_extent(~Element,joining,source),first,rows,line,middle,offset>) ->
  {Tree.read(Element,tree,U32.from_nat(ConcatLeaf.cell_at(part_extent(~Element,joining,source),BandConcat.band_inner(joining,rows),line,middle,offset))) ==
    tensor_original(~Element,~band_value,source,U32.from_nat(framed(U32.to_nat(part_extent(~Element,joining,source)),joining_inner(joining),outer_at(joining,first,line,offset),middle,inner_at(joining,first,line,offset)))) : Element}:
  Cell{+channel,+row,+column,+channel_inside,+row_inside,+column_inside,+stored,+placed} = cell
  +extent = part_extent(~Element,joining,source)
  +band = part_geometry(~Element,joining,source)
  %stored : {Tree.read(Element,tree,U32.from_nat(_)) ==
    tensor_original(~Element,~band_value,source,U32.from_nat(framed(U32.to_nat(extent),joining_inner(joining),outer_at(joining,first,line,offset),middle,inner_at(joining,first,line,offset)))) : Element}
  %placed : {Tree.read(Element,tree,U32.from_nat(BandValues.position(band,rows,channel,row,column))) == tensor_original(~Element,~band_value,source,U32.from_nat(_)) : Element}
  BandValues.leaf_read(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),source,band,first,rows,tree,length,channel,row,column,values,
    channel_inside,row_inside,column_inside)

# A cell of the band of a part holds the part there.
def along_part(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  owner: BandConcatExecution.TensorOwnerModel<Element>,+source: BandConcatExecution.TensorContent<Element>,
  fact: PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source),+line: Nat,+middle: Nat,+offset: Nat,
  +line_inside: {Nat.is_lt(line,U32.to_nat(BandConcat.band_outer(joining,rows))) == True{} : Bool},
  +middle_inside: {Nat.is_lt(middle,U32.to_nat(part_extent(~Element,joining,source))) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(BandConcat.band_inner(joining,rows))) == True{} : Bool}) ->
  Sampled(~Element,~band_value,joining,first,rows,owner_content(~Element,owner),source,line,middle,offset):
  match owner:
    case BandConcatExecution.TensorOwner{Model.Leaf{buffer,+leaf_first,+leaf_rows},+geometry,+shape,+fill,+status,+workers}:
      match buffer:
        case Buffer.Model{+tree,+length,+certificate}:
          Shape.Both{same_owner,Shape.Both{+fields,Shape.Both{ready,Shape.Both{values,+fitted}}}} = fact
          along_cell(~Element,~band_value,joining,first,rows,source,tree,length,line,middle,offset,values,
            unboxed(joining,part_extent(~Element,joining,source),first,rows,capacity,fitted,
              fitted_rows(part_geometry(~Element,joining,source),first,rows,length,fields),line,middle,offset,line_inside,middle_inside,offset_inside))
    case BandConcatExecution.TensorOwner{Model.Fork{+left,+right},+geometry,+shape,+fill,+status,+workers}:
      Empty.absurd(Sampled(~Element,~band_value,joining,first,rows,owner_content(~Element,BandConcatExecution.TensorOwner{Model.Fork{left,right},geometry,shape,fill,status,workers}),source,line,middle,offset),fact)

def Along(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+first: U32,+rows: U32,
  +parts: BandConcatExecution.PartsOwnerModel<Element>,+sources: BandConcatExecution.PartContents<Element>,+line: Nat,+middle: Nat,+offset: Nat) -> Data:
  {ConcatLeaf.along(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),layers_of(~Element,joining,parts,sources,rows),BandConcat.band_inner(joining,rows),line,middle,offset) ==
    sample(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),extents_of(~Element,joining,sources),joining_inner(joining),
      outer_at(joining,first,line,offset),middle,inner_at(joining,first,line,offset)) : Element}

# The cell of the layers at a place along the axis is the cell of the part
# that covers it: both pick the part by the same comparison.
def along_sources(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  parts: BandConcatExecution.PartsOwnerModel<Element>,sources: BandConcatExecution.PartContents<Element>,
  chosen: Bool,facts: PartsAt(~Element,~band_value,joining,capacity,first,rows,parts,sources),+line: Nat,+middle: Nat,+offset: Nat,
  +classified: {Nat.is_lt(middle,head_extent(~Element,joining,sources)) == chosen : Bool},
  +middle_inside: {Nat.is_lt(middle,parts_total(~Element,joining,sources)) == True{} : Bool},
  +line_inside: {Nat.is_lt(line,U32.to_nat(BandConcat.band_outer(joining,rows))) == True{} : Bool},
  +offset_inside: {Nat.is_lt(offset,U32.to_nat(BandConcat.band_inner(joining,rows))) == True{} : Bool}) ->
  Along(~Element,~band_value,joining,first,rows,parts,sources,line,middle,offset):
  match parts sources chosen:
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.SingleContent{+source} chosen:
      along_part(~Element,~band_value,joining,capacity,first,rows,owner,source,facts,line,middle,offset,line_inside,middle_inside,offset_inside)
    case BandConcatExecution.ConsOwner{+owner,+rest} BandConcatExecution.ConsContent{+source,+others} True{}:
      match facts:
        case Shape.Both{head_fact,tail}:
          +extent = U32.to_nat(part_extent(~Element,joining,source))
          +inner = BandConcat.band_inner(joining,rows)
          +outer = outer_at(joining,first,line,offset)
          +place = inner_at(joining,first,line,offset)
          +cell = Contract.stored(~Element,owner_content(~Element,owner),U32.from_nat(ConcatLeaf.cell_at(part_extent(~Element,joining,source),inner,line,middle,offset)))
          +read = tensor_original(~Element,~band_value,source,U32.from_nat(framed(extent,joining_inner(joining),outer,middle,place)))
          +later = ConcatLeaf.along(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),layers_of(~Element,joining,rest,others,rows),inner,line,Nat.sub(middle,extent),offset)
          +ordered = sample(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),extents_of(~Element,joining,others),joining_inner(joining),outer,Nat.sub(middle,extent),place)
          %Equal.sym(Bool,Nat.is_lt(middle,extent),True{},classified) : {Bool.pick(Element,_,cell,later) == Bool.pick(Element,Nat.is_lt(middle,extent),read,ordered) : Element}
          %Equal.sym(Bool,Nat.is_lt(middle,extent),True{},classified) : {cell == Bool.pick(Element,_,read,ordered) : Element}
          along_part(~Element,~band_value,joining,capacity,first,rows,owner,source,head_fact,line,middle,offset,line_inside,classified,offset_inside)
    case BandConcatExecution.ConsOwner{+owner,+rest} BandConcatExecution.ConsContent{+source,+others} False{}:
      match facts:
        case Shape.Both{head_fact,tail}:
          +extent = U32.to_nat(part_extent(~Element,joining,source))
          +inner = BandConcat.band_inner(joining,rows)
          +outer = outer_at(joining,first,line,offset)
          +place = inner_at(joining,first,line,offset)
          +cell = Contract.stored(~Element,owner_content(~Element,owner),U32.from_nat(ConcatLeaf.cell_at(part_extent(~Element,joining,source),inner,line,middle,offset)))
          +read = tensor_original(~Element,~band_value,source,U32.from_nat(framed(extent,joining_inner(joining),outer,middle,place)))
          +later = ConcatLeaf.along(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),layers_of(~Element,joining,rest,others,rows),inner,line,Nat.sub(middle,extent),offset)
          +ordered = sample(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),extents_of(~Element,joining,others),joining_inner(joining),outer,Nat.sub(middle,extent),place)
          %Equal.sym(Bool,Nat.is_lt(middle,extent),False{},classified) : {Bool.pick(Element,_,cell,later) == Bool.pick(Element,Nat.is_lt(middle,extent),read,ordered) : Element}
          %Equal.sym(Bool,Nat.is_lt(middle,extent),False{},classified) : {later == Bool.pick(Element,_,read,ordered) : Element}
          along_sources(~Element,~band_value,joining,capacity,first,rows,rest,others,Nat.is_lt(Nat.sub(middle,extent),head_extent(~Element,joining,others)),tail,
            line,Nat.sub(middle,extent),offset,{==},
            Order.sub_lt(middle,extent,parts_total(~Element,joining,others),Order.not_less(middle,extent,classified),middle_inside),line_inside,offset_inside)
    case BandConcatExecution.SingleOwner{+owner} BandConcatExecution.ConsContent{+source,+others} chosen:
      Empty.absurd(Along(~Element,~band_value,joining,first,rows,BandConcatExecution.SingleOwner{owner},BandConcatExecution.ConsContent{source,others},line,middle,offset),facts)
    case BandConcatExecution.ConsOwner{+owner,+rest} BandConcatExecution.SingleContent{+source} chosen:
      Empty.absurd(Along(~Element,~band_value,joining,first,rows,BandConcatExecution.ConsOwner{owner,rest},BandConcatExecution.SingleContent{source},line,middle,offset),facts)

# The band rows fit in the batch, as every part states of its band.
def part_rows_fit(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  owner: BandConcatExecution.TensorOwnerModel<Element>,+source: BandConcatExecution.TensorContent<Element>,
  fact: PartAt(~Element,~band_value,joining,capacity,first,rows,owner,source),+extent: U32) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(BandConcat.geometry(joining,extent))) == True{} : Bool}:
  match owner:
    case BandConcatExecution.TensorOwner{Model.Leaf{+buffer,leaf_first,leaf_rows},geometry,shape,fill,status,workers}:
      Shape.Both{same_owner,Shape.Both{+fields,rest}} = fact
      rows_moved(joining,part_extent(~Element,joining,source),extent,first,rows,
        fitted_rows(part_geometry(~Element,joining,source),first,rows,Buffer.content_length(~Element,Buffer.content(~Element,buffer)),fields))
    case BandConcatExecution.TensorOwner{Model.Fork{left,right},geometry,shape,fill,status,workers}:
      Empty.absurd({Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(BandConcat.geometry(joining,extent))) == True{} : Bool},fact)

def parts_rows_fit(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+capacity: U32,+first: U32,+rows: U32,
  parts: BandConcatExecution.PartsOwnerModel<Element>,sources: BandConcatExecution.PartContents<Element>,
  facts: PartsAt(~Element,~band_value,joining,capacity,first,rows,parts,sources),+extent: U32) ->
  {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(BandConcat.geometry(joining,extent))) == True{} : Bool}:
  match parts sources:
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.SingleContent{+source}:
      part_rows_fit(~Element,~band_value,joining,capacity,first,rows,owner,source,facts,extent)
    case BandConcatExecution.ConsOwner{owner,rest} BandConcatExecution.ConsContent{+source,others}:
      Shape.Both{head_fact,tail} = facts
      part_rows_fit(~Element,~band_value,joining,capacity,first,rows,owner,source,head_fact,extent)
    case BandConcatExecution.SingleOwner{owner} BandConcatExecution.ConsContent{source,others}:
      Empty.absurd({Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(BandConcat.geometry(joining,extent))) == True{} : Bool},facts)
    case BandConcatExecution.ConsOwner{owner,rest} BandConcatExecution.SingleContent{source}:
      Empty.absurd({Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(BandConcat.geometry(joining,extent))) == True{} : Bool},facts)

# ---- The cells of one band of the output ----

# The output of a band concat: its joining and extent, its parts, and the
# room of its cells.
type Scope<-Element: Data> is Data:
  Scope{joining: BandConcat.Joining,extent: U32,parts: BandConcatExecution.PartContents<Element>,
    total: {U32.to_nat(extent) == parts_total(~Element,joining,parts) : Nat},
    limits: Machine.Limits,actual: {limits == Default.limits() : Machine.Limits},
    domain: BandGather.Domain<BandConcat.geometry(joining,extent),limits,Machine.capacity(limits)>,
    fitted: Fitted<joining,extent,Machine.capacity(limits)>}

def scope_context(~Element: Data,+scope: Scope<Element>) -> BandConcat.Context:
  Scope{+joining,+extent,parts,total,limits,actual,domain,fitted} = scope
  BandConcat.Context{joining,extent}

def scope_sources(~Element: Data,+scope: Scope<Element>) -> Sources<Element>:
  Scope{+joining,extent,+parts,total,+limits,actual,domain,fitted} = scope
  Sources{joining,Machine.capacity(limits),parts}

# Every cell of an output band holds the concat of the parts.
def OutputValues(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: Scope<Element>) ->
  @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Scope{+joining,+extent,+parts,total,limits,actual,domain,fitted} = scope
  BandValues.band_values(~Element,~Concatenated<BandConcatExecution.TensorContent<Element>>,~concat_value(~Element,~band_value),logical(~Element,joining,parts),
    BandConcat.geometry(joining,extent))

# The leaf concat of a band: the layers along its middle axis over the fill
# value.
def Lined(~Element: Data,+fill: Element,+layers: ConcatLeaf.Layers<Buffer.Content<Element>>,+joining: BandConcat.Joining,+extent: U32,+rows: U32) -> ConcatLeaf.Appended<Element,Buffer.Content<Element>>:
  ConcatLeaf.Appended{fill,ConcatLeaf.lined(~Buffer.Content<Element>,layers,BandConcat.band_outer(joining,rows),BandConcat.band_inner(joining,rows)),1,BandConcat.band_shape(joining,extent,rows),0}

# One band of the output with what every cell needs: the parts there, the
# leaf concat of their bands and its count.
type CellScope<-Element: Data,-band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element> is Data:
  CellScope{joining: BandConcat.Joining,extent: U32,sources: BandConcatExecution.PartContents<Element>,first: U32,rows: U32,count: U32,
    owners: BandConcatExecution.PartsOwnerModel<Element>,limits: Machine.Limits,
    facts: PartsAt(~Element,~band_value,joining,Machine.capacity(limits),first,rows,owners,sources),fill: Element,
    tree: Tree.Tree<Element>,length: Nat,
    total: {U32.to_nat(extent) == parts_total(~Element,joining,sources) : Nat},
    actual: {limits == Default.limits() : Machine.Limits},
    domain: BandGather.Domain<BandConcat.geometry(joining,extent),limits,Machine.capacity(limits)>,
    fitted: Fitted<joining,extent,Machine.capacity(limits)>,
    counted: Count.Count<ShapeLaws.elements(BandConcat.band_shape(joining,extent,rows)),count>,
    length_exact: {length == ShapeLaws.elements(BandConcat.band_shape(joining,extent,rows)) : Nat},
    values: Contract.Values(~Element,~ConcatLeaf.Appended<Element,Buffer.Content<Element>>,~ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element)),
      Lined(~Element,fill,layers_of(~Element,joining,owners,sources,rows),joining,extent,rows),Buffer.Content{tree,length})}

def cell_channels(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: CellScope<Element,band_value>) -> Nat:
  CellScope{+joining,+extent,sources,first,rows,count,owners,limits,facts,fill,tree,length,total,actual,domain,fitted,counted,length_exact,values} = scope
  BandValues.channels(BandConcat.geometry(joining,extent))

def cell_rows(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: CellScope<Element,band_value>) -> Nat:
  CellScope{joining,extent,sources,first,+rows,count,owners,limits,facts,fill,tree,length,total,actual,domain,fitted,counted,length_exact,values} = scope
  U32.to_nat(rows)

def cell_width(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: CellScope<Element,band_value>) -> Nat:
  CellScope{+joining,+extent,sources,first,rows,count,owners,limits,facts,fill,tree,length,total,actual,domain,fitted,counted,length_exact,values} = scope
  BandValues.width(BandConcat.geometry(joining,extent))

def ConcatCell(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: CellScope<Element,band_value>,
  +channel: Nat,+row: Nat,+column: Nat) -> Data:
  CellScope{+joining,+extent,+sources,+first,+rows,count,owners,limits,facts,fill,+tree,+length,total,actual,domain,fitted,counted,length_exact,values} = scope
  BandValues.At(~Element,~Concatenated<BandConcatExecution.TensorContent<Element>>,~concat_value(~Element,~band_value),logical(~Element,joining,sources),
    BandConcat.geometry(joining,extent),first,rows,tree,length,channel,row,column)

# The cell of an output band at a channel, a row and a column as its line,
# place along the axis and offset.
def CellBox(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: CellScope<Element,band_value>,
  +channel: Nat,+row: Nat,+column: Nat) -> Data:
  CellScope{+joining,+extent,sources,+first,+rows,count,owners,limits,facts,fill,tree,length,total,actual,domain,fitted,counted,length_exact,values} = scope
  Box<joining,extent,first,rows,BandValues.position(BandConcat.geometry(joining,extent),rows,channel,row,column),
    BandValues.pixel(BandConcat.geometry(joining,extent),channel,Nat.add(U32.to_nat(first),row),column)>

def concat_boxed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: CellScope<Element,band_value>,
  +channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,cell_channels(~Element,~band_value,scope)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,cell_rows(~Element,~band_value,scope)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,cell_width(~Element,~band_value,scope)) == True{} : Bool},
  box: CellBox(~Element,~band_value,scope,channel,row,column)) ->
  ConcatCell(~Element,~band_value,scope,channel,row,column):
  CellScope{+joining,+extent,+sources,+first,+rows,+count,+owners,+limits,+facts,+fill,+tree,+length,+total,+actual,+domain,+fitted,+counted,+length_exact,+values} = scope
  BandGather.Domain{+capacity_ok,channels_positive,+height_positive,width_positive,+machine_room,allocation_room} = domain
  Count.Count{+exact,+bounded} = counted
  Box{+line,+middle,+offset,+stored_at,+placed,+line_inside,+middle_inside,+offset_inside,+inner_inside} = box
  +geometry = BandConcat.geometry(joining,extent)
  +capacity = Machine.capacity(limits)
  +limit = U32.to_nat(capacity)
  +outer = BandConcat.band_outer(joining,rows)
  +inner = BandConcat.band_inner(joining,rows)
  +shape = BandConcat.band_shape(joining,extent,rows)
  +position = BandValues.position(geometry,rows,channel,row,column)
  +whole = Nat.add(U32.to_nat(first),row)
  +pixel = BandValues.pixel(geometry,channel,whole,column)
  +layers = layers_of(~Element,joining,owners,sources,rows)
  +context = Lined(~Element,fill,layers,joining,extent,rows)
  +logical = logical(~Element,joining,sources)
  +rows_fit = parts_rows_fit(~Element,~band_value,joining,capacity,first,rows,owners,sources,facts,extent)
  +outer_index = outer_at(joining,first,line,offset)
  +place = inner_at(joining,first,line,offset)
  +length_cells = Equal.trans(Nat,length,ShapeLaws.elements(shape),BandValues.cell_count(geometry,rows),length_exact,
    Equal.sym(Nat,BandValues.cell_count(geometry,rows),ShapeLaws.elements(shape),band_cells(joining,extent,first,rows,capacity,fitted,rows_fit)))
  +inside = BandValues.position_within(geometry,rows,length,channel,row,column,length_cells,channel_inside,row_inside,column_inside)
  +count_cells = Equal.trans(Nat,Nat.mul(U32.to_nat(outer),Slices.cells(extent,inner)),ShapeLaws.elements(shape),U32.to_nat(count),
    Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(U32.to_nat(outer),Slices.cells(extent,inner)),ShapeLaws.elements_three(outer,extent,inner)),exact)
  +count_room = Bounds.as_nat_le(count,capacity,
    %Equal.sym(Machine.Limits,limits,Default.limits(),actual) : {U32.is_le(count,Machine.capacity(_)) == True{} : Bool} bounded)
  +positive = {
    %exact : {Nat.is_lt(0n,_) == True{} : Bool}
    %length_exact : {Nat.is_lt(0n,_) == True{} : Bool}
    Order.le_lt_trans(0n,position,length,Order.zero_le(position),inside)
    : {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}}
  +target = {ConcatLeaf.Target{count_cells,U32Division.nonzero(count,positive),capacity_ok,count_room} : ConcatLeaf.Target<outer,extent,inner,count,capacity>}
  +layer_extents = layers_total(~Element,~band_value,joining,capacity,first,rows,owners,sources,facts)
  +parts_room = {
    %total : {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool}
    Order.reflexive(U32.to_nat(extent))
    : {Nat.is_le(parts_total(~Element,joining,sources),U32.to_nat(extent)) == True{} : Bool}}
  +room = {
    %Equal.sym(Nat,ConcatLeaf.layers_total(~Buffer.Content<Element>,layers),parts_total(~Element,joining,sources),layer_extents) : {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool}
    parts_room
    : {Nat.is_le(Nat.add(U32.to_nat(0),ConcatLeaf.layers_total(~Buffer.Content<Element>,layers)),U32.to_nat(extent)) == True{} : Bool}}
  +parts_inside = {
    %total : {Nat.is_lt(middle,_) == True{} : Bool}
    middle_inside
    : {Nat.is_lt(middle,parts_total(~Element,joining,sources)) == True{} : Bool}}
  +layers_inside = {
    %Equal.sym(Nat,ConcatLeaf.layers_total(~Buffer.Content<Element>,layers),parts_total(~Element,joining,sources),layer_extents) : {Nat.is_lt(middle,_) == True{} : Bool}
    parts_inside
    : {Nat.is_lt(middle,ConcatLeaf.layers_total(~Buffer.Content<Element>,layers)) == True{} : Bool}}
  +cells_room = {
    %Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),exact) : {Nat.is_le(_,limit) == True{} : Bool}
    count_room
    : {Nat.is_le(ShapeLaws.elements(shape),limit) == True{} : Bool}}
  +layers_counted = layers_counted(~Element,~band_value,joining,capacity,first,rows,owners,sources,facts,extent,capacity_ok,parts_room,cells_room)
  +stored = Contract.read(~Element,~ConcatLeaf.Appended<Element,Buffer.Content<Element>>,~ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element)),context,Buffer.Content{tree,length},position,values,inside)
  +along = ConcatLeaf.along(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),layers,inner,line,middle,offset)
  +lined = {
    %Equal.sym(Nat,position,ConcatLeaf.cell_at(extent,inner,line,middle,offset),stored_at) : {ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element),context,U32.from_nat(_)) == along : Element}
    ConcatLeaf.appended_middle(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),layers,Nat.is_lt(middle,U32.to_nat(ConcatLeaf.head_extent(~Buffer.Content<Element>,layers))),
      outer,extent,inner,count,capacity,target,0,fill,line,middle,offset,layers_counted,room,line_inside,layers_inside,offset_inside,{==})
    : {ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element),context,U32.from_nat(position)) == along : Element}}
  +picked = along_sources(~Element,~band_value,joining,capacity,first,rows,owners,sources,Nat.is_lt(middle,head_extent(~Element,joining,sources)),facts,line,middle,offset,{==},
    parts_inside,line_inside,offset_inside)
  +pixel_inside = BandGather.pixel_below(geometry,channel,whole,column,channel_inside,
    Bounds.nat_trans(whole,Nat.add(U32.to_nat(first),U32.to_nat(rows)),Inspection.total_rows(geometry),Order.offset_lt(U32.to_nat(first),row,U32.to_nat(rows),row_inside),rows_fit),
    column_inside,height_positive)
  +encoded = {
    %total : {U32.to_nat(U32.from_nat(pixel)) == framed(_,joining_inner(joining),outer_index,middle,place) : Nat}
    Equal.trans(Nat,U32.to_nat(U32.from_nat(pixel)),pixel,framed(U32.to_nat(extent),joining_inner(joining),outer_index,middle,place),
      Bounds.roundtrip(pixel,capacity,capacity_ok,Bounds.strict_le(pixel,limit,Bounds.nat_trans(pixel,BandGather.elements(geometry),limit,pixel_inside,machine_room))),placed)
    : {U32.to_nat(U32.from_nat(pixel)) == framed(parts_total(~Element,joining,sources),joining_inner(joining),outer_index,middle,place) : Nat}}
  +sampled = sample(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),extents_of(~Element,joining,sources),joining_inner(joining),outer_index,middle,place)
  +concat = concat_value(~Element,~band_value,logical,U32.from_nat(pixel))
  +global = framed_value(~Element,~BandConcatExecution.TensorContent<Element>,~tensor_original(~Element,~band_value),extents_of(~Element,joining,sources),joining_inner(joining),
    outer_index,middle,place,U32.from_nat(pixel),encoded,parts_inside,inner_inside)
  Shape.Both{inside,
    Equal.trans(Element,Tree.read(Element,tree,U32.from_nat(position)),ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element),context,U32.from_nat(position)),concat,stored,
      Equal.trans(Element,ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element),context,U32.from_nat(position)),along,concat,lined,
        Equal.trans(Element,along,sampled,concat,picked,Equal.sym(Element,concat,sampled,global))))}

# One cell of an output band: the leaf concat stores at its position the
# layers at its line, place and offset, which is the band of the part that
# covers its place, which holds that part there, which is the concat at its
# pixel.
def concat_cell(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: CellScope<Element,band_value>,
  +channel: Nat,+row: Nat,+column: Nat,
  +channel_inside: {Nat.is_lt(channel,cell_channels(~Element,~band_value,scope)) == True{} : Bool},
  +row_inside: {Nat.is_lt(row,cell_rows(~Element,~band_value,scope)) == True{} : Bool},
  +column_inside: {Nat.is_lt(column,cell_width(~Element,~band_value,scope)) == True{} : Bool}) ->
  ConcatCell(~Element,~band_value,scope,channel,row,column):
  CellScope{+joining,+extent,+sources,+first,+rows,count,+owners,+limits,+facts,fill,tree,length,total,actual,domain,+fitted,counted,length_exact,values} = scope
  concat_boxed(~Element,~band_value,CellScope{joining,extent,sources,first,rows,count,owners,limits,facts,fill,tree,length,total,actual,domain,fitted,counted,length_exact,values},
    channel,row,column,channel_inside,row_inside,column_inside,
    boxed(joining,extent,first,rows,Machine.capacity(limits),fitted,parts_rows_fit(~Element,~band_value,joining,Machine.capacity(limits),first,rows,owners,sources,facts,extent),
      channel,row,column,channel_inside,row_inside,column_inside))

def scope_joining(~Element: Data,+scope: Scope<Element>) -> BandConcat.Joining:
  Scope{+joining,extent,parts,total,limits,actual,domain,fitted} = scope
  joining

def scope_extent(~Element: Data,+scope: Scope<Element>) -> U32:
  Scope{joining,+extent,parts,total,limits,actual,domain,fitted} = scope
  extent

def scope_parts(~Element: Data,+scope: Scope<Element>) -> BandConcatExecution.PartContents<Element>:
  Scope{joining,extent,+parts,total,limits,actual,domain,fitted} = scope
  parts

# A leaf concat of the bands of the parts, read as a band of the output.
def band_content(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: Scope<Element>,+first: U32,+rows: U32,+count: U32,
  +owners: BandConcatExecution.PartsOwnerModel<Element>,+facts: PartsIn(~Element,~band_value,scope_sources(~Element,scope),first,rows,owners),+fill: Element,
  +counted: Count.Count<ShapeLaws.elements(BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows)),count>,model: Buffer.Model<Element>,
  stored: GatherValues.Facts(~Element,~ConcatLeaf.Appended<Element,Buffer.Content<Element>>,~ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element)),
    Lined(~Element,fill,layers_of(~Element,scope_joining(~Element,scope),owners,scope_parts(~Element,scope),rows),scope_joining(~Element,scope),scope_extent(~Element,scope),rows),
    BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows),Buffer.content(~Element,model))) ->
  Shape.Both<Buffer.ContentReady<Element,Buffer.content(~Element,model)>,
    BandValues.band_values(~Element,~Concatenated<BandConcatExecution.TensorContent<Element>>,~concat_value(~Element,~band_value),logical(~Element,scope_joining(~Element,scope),scope_parts(~Element,scope)),
      BandConcat.geometry(scope_joining(~Element,scope),scope_extent(~Element,scope)),first,rows,Buffer.content(~Element,model))>:
  Scope{+joining,+extent,+sources,+total,+limits,+actual,+domain,+fitted} = scope
  match model:
    case Buffer.Model{+tree,+length,certificate}:
      Shape.Both{Contract.WellFormed{+length_exact,ready},+values} = stored
      Shape.Both{ready,
        BandValues.table(~CellScope<Element,band_value>,~ConcatCell(~Element,~band_value),~cell_channels(~Element,~band_value),~cell_rows(~Element,~band_value),~cell_width(~Element,~band_value),
          ~concat_cell(~Element,~band_value),
          CellScope{joining,extent,sources,first,rows,count,owners,limits,facts,fill,tree,length,total,actual,domain,fitted,counted,length_exact,values})}

# What a checked band gives the output when its tensor is the band.
def Banded(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: Scope<Element>,+first: U32,+rows: U32,tensor: Tensors.Tensor<Element>) -> Data:
  Model.ContentAll(~Element,~Global.Made(~Element,~Scope<Element>,~OutputValues(~Element,~band_value),scope),
    Model.contents(~Element,Each.made(~Element,Each.checked(~Element,BandConcat.checked(~Element,True{},first,rows,tensor)))))

# The pieces of a band of the output and the leaf concat they make.
def band_pieces(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: Scope<Element>,+first: U32,+rows: U32,
  +owners: BandConcatExecution.PartsOwnerModel<Element>,+facts: PartsIn(~Element,~band_value,scope_sources(~Element,scope),first,rows,owners)) -> Pieces(~Element):
  Scope{+joining,extent,+parts,total,+limits,actual,domain,fitted} = scope
  pieces_of(~Element,~band_value,joining,Machine.capacity(limits),first,rows,owners,parts,facts)

def Joined(~Element: Data,+pieces: Pieces(~Element),+shape: List<&2,U32>,+fill: Element,count: Maybe<&2,U32>) -> Tensors.Tensor<Element>:
  Layout.append_parts(~Element,ConcatLeaf.packed(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),pieces),1,
    (Tensors.with_workers(~Element,Tensors.full_count(~Element,shape,count,fill),1),0))

def joined_context(~Element: Data,+pieces: Pieces(~Element),+shape: List<&2,U32>,+fill: Element) -> ConcatLeaf.Appended<Element,Buffer.Content<Element>>:
  ConcatLeaf.Appended{fill,ConcatLeaf.sources_of(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),pieces),1,shape,0}

# The checked band of storage that holds the leaf concat.
def band_stored(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: Scope<Element>,+first: U32,+rows: U32,+count: U32,
  +owners: BandConcatExecution.PartsOwnerModel<Element>,+facts: PartsIn(~Element,~band_value,scope_sources(~Element,scope),first,rows,owners),+fill: Element,
  +counts: {Tensors.count_of(BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows)) == Some{count} : Maybe<&2,U32>},
  +model: Buffer.Model<Element>,
  stored: GatherValues.Facts(~Element,~ConcatLeaf.Appended<Element,Buffer.Content<Element>>,~ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element)),
    joined_context(~Element,band_pieces(~Element,~band_value,scope,first,rows,owners,facts),BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows),fill),
    BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows),Buffer.content(~Element,model))) ->
  Model.ContentAll(~Element,~Global.Made(~Element,~Scope<Element>,~OutputValues(~Element,~band_value),scope),
    Model.contents(~Element,Each.made(~Element,Each.checked(~Element,Partition.Leaf{Bands.Checked{Buffer.pack(~Element,model),True{}},first,rows,Unit{}})))):
  Scope{+joining,+extent,+sources,+total,+limits,+actual,+domain,+fitted} = scope
  +shape = BandConcat.band_shape(joining,extent,rows)
  %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,model)),Buffer.content(~Element,model),Buffer.packed_contents(~Element,model)) :
    Shape.Both<Buffer.ContentReady<Element,_>,
      BandValues.band_values(~Element,~Concatenated<BandConcatExecution.TensorContent<Element>>,~concat_value(~Element,~band_value),logical(~Element,joining,sources),BandConcat.geometry(joining,extent),first,rows,_)>
  band_content(~Element,~band_value,Scope{joining,extent,sources,total,limits,actual,domain,fitted},first,rows,count,owners,facts,fill,Count.count_of(shape,count,counts),model,
    %pieces_lined(~Element,~band_value,joining,Machine.capacity(limits),first,rows,owners,sources,facts) :
      GatherValues.Facts(~Element,~ConcatLeaf.Appended<Element,Buffer.Content<Element>>,~ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element)),ConcatLeaf.Appended{fill,_,1,shape,0},shape,Buffer.content(~Element,model))
    stored)

# The leaf concat as a tensor of the band shape: its checked band reads the
# storage of its one band.
def band_shaped(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: Scope<Element>,+first: U32,+rows: U32,+count: U32,
  +owners: BandConcatExecution.PartsOwnerModel<Element>,+facts: PartsIn(~Element,~band_value,scope_sources(~Element,scope),first,rows,owners),+fill: Element,
  +counts: {Tensors.count_of(BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows)) == Some{count} : Maybe<&2,U32>},-tensor: Tensors.Tensor<Element>,
  shaped: Global.Shaped(~Element,~ConcatLeaf.Appended<Element,Buffer.Content<Element>>,~ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element)),
    joined_context(~Element,band_pieces(~Element,~band_value,scope,first,rows,owners,facts),BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows),fill),
    BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows),tensor)) ->
  Banded(~Element,~band_value,scope,first,rows,tensor):
  Shape.Both{usable,contract} = shaped
  Global.Usable{+bands,+geometry,+made_fill,+workers,+equation} = usable
  +shape = BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows)
  +context = joined_context(~Element,band_pieces(~Element,~band_value,scope,first,rows,owners,facts),shape,fill)
  %equation : Banded(~Element,~band_value,scope,first,rows,_)
  WholeLeaf.with_whole(~Element,~ConcatLeaf.Appended<Element,Buffer.Content<Element>>,~ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element)),~Bands.Transformed(~Element),
    ~(result => Model.ContentAll(~Element,~Global.Made(~Element,~Scope<Element>,~OutputValues(~Element,~band_value),scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,result))))),
    ~(storage => shape => value => status => workers => Partition.Leaf{Bands.Checked{storage,True{} && Tensors.usable(status)},first,rows,Unit{}}),
    context,bands,geometry,shape,made_fill,workers,
    %Equal.sym(Tensors.Tensor<Element>,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,made_fill,Tensors.Usable{},workers},tensor,equation) :
      Global.Contract(~Element,~ConcatLeaf.Appended<Element,Buffer.Content<Element>>,~ConcatLeaf.appended_value(~Element,~Element,~Buffer.Content<Element>,~Full.constant(~Element),~Contract.stored(~Element)),context,_)
    contract,
    model => stored => band_stored(~Element,~band_value,scope,first,rows,count,owners,facts,fill,counts,model,stored))

# The leaf concat of the bands has the count of the band shape; a shape
# without one leaves it failed.
def band_counted(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: Scope<Element>,+first: U32,+rows: U32,
  +owners: BandConcatExecution.PartsOwnerModel<Element>,+facts: PartsIn(~Element,~band_value,scope_sources(~Element,scope),first,rows,owners),+fill: Element,
  found: Maybe<&2,U32>,+counts: {Tensors.count_of(BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows)) == found : Maybe<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,Joined(~Element,band_pieces(~Element,~band_value,scope,first,rows,owners,facts),
    BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows),fill,found)) == True{} : Bool}) ->
  Banded(~Element,~band_value,scope,first,rows,
    Joined(~Element,band_pieces(~Element,~band_value,scope,first,rows,owners,facts),BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows),fill,found)):
  match found:
    case Some{+count}:
      +shape = BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows)
      +pieces = band_pieces(~Element,~band_value,scope,first,rows,owners,facts)
      band_shaped(~Element,~band_value,scope,first,rows,count,owners,facts,fill,counts,Joined(~Element,pieces,shape,fill,Some{count}),
        ConcatLeaf.counted_shaped(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),Some{count},pieces,1,shape,fill,1,counts,accepted))
    case None{}:
      +shape = BandConcat.band_shape(scope_joining(~Element,scope),scope_extent(~Element,scope),rows)
      +pieces = band_pieces(~Element,~band_value,scope,first,rows,owners,facts)
      Empty.absurd(Banded(~Element,~band_value,scope,first,rows,Joined(~Element,pieces,shape,fill,None{})),
        ConcatLeaf.uncounted(~Element,ConcatLeaf.packed(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),pieces),1,shape,fill,1,accepted))

# A usable checked band came from a valid gather and a ready tensor.
def checked_ready(~Element: Data,valid: Bool,+first: U32,+rows: U32,tensor: Tensors.Tensor<Element>,
  accepted: {Each.usable(~Element,Each.checked(~Element,BandConcat.checked(~Element,valid,first,rows,tensor))) == True{} : Bool}) ->
  Shape.Both<{valid == True{} : Bool},{Contract.tensor_ready(~Element,tensor) == True{} : Bool}>:
  match valid tensor:
    case True{} Tensors.Tensor{bands,geometry,shape,fill,status,workers}:
      match status:
        case Tensors.Usable{}: Shape.Both{{==},{==}}
        case Tensors.Failed{+problem}:
          Empty.absurd(Shape.Both<{True{} == True{} : Bool},{Contract.tensor_ready(~Element,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}) == True{} : Bool}>,
            WordEncoding.false_true(accepted))
    case False{} Tensors.Tensor{bands,+geometry,+shape,+fill,+status,+workers}:
      Empty.absurd(Shape.Both<{False{} == True{} : Bool},{Contract.tensor_ready(~Element,Tensors.Tensor{bands,geometry,shape,fill,status,workers}) == True{} : Bool}>,
        WordEncoding.false_true(accepted))

# The leaf concat of the bands of the parts at one band.
def joined_band(~Element: Data,+owners: BandConcatExecution.PartsOwnerModel<Element>,+shape: List<&2,U32>,+fill: Element) -> Tensors.Tensor<Element>:
  Layout.band_joined(~Element,BandConcatExecution.parts_owner_pack(~Element,owners),shape,fill)

# One band of the output: the gather found the band of every part there, and
# the leaf concat of those bands holds the concat at every cell.
def band_rule(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+scope: Scope<Element>,+first: U32,+rows: U32,
  +joined: BandConcatExecution.Joined<Element>,
  fact: Model.viewed(~BandConcatExecution.Joined<Element>,~Sources<Element>,~JoinedAt(~Element,~band_value),~Scope<Element>,~scope_sources(~Element),scope,joined,first,rows),
  accepted: {Each.usable(~Element,Each.checked(~Element,BandConcat.band(~Element,~Layout.band_joined(~Element),scope_context(~Element,scope),first,rows,
    BandConcatExecution.joined_pack(~Element,joined),Unit{}))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~Global.Made(~Element,~Scope<Element>,~OutputValues(~Element,~band_value),scope),
    Model.contents(~Element,Each.made(~Element,Each.checked(~Element,BandConcat.band(~Element,~Layout.band_joined(~Element),scope_context(~Element,scope),first,rows,
      BandConcatExecution.joined_pack(~Element,joined),Unit{}))))):
  Scope{+joining,+extent,+sources,+total,+limits,+actual,+domain,+fitted} = scope
  BandConcatExecution.Joined{+owners,+fill,+valid} = joined
  +whole = {Scope{joining,extent,sources,total,limits,actual,domain,fitted} : Scope<Element>}
  +shape = BandConcat.band_shape(joining,extent,rows)
  +ready = checked_ready(~Element,valid,first,rows,joined_band(~Element,owners,shape,fill),accepted)
  +valid_true = Shape.first({valid == True{} : Bool},{Contract.tensor_ready(~Element,joined_band(~Element,owners,shape,fill)) == True{} : Bool},ready)
  +facts = {
    %valid_true : ValidAt(~Element,~band_value,_,Sources{joining,Machine.capacity(limits),sources},first,rows,owners)
    fact
    : PartsIn(~Element,~band_value,scope_sources(~Element,whole),first,rows,owners)}
  +pieces = band_pieces(~Element,~band_value,whole,first,rows,owners,facts)
  +packed = pieces_packed(~Element,~band_value,joining,Machine.capacity(limits),first,rows,owners,sources,facts)
  %Equal.sym(Bool,valid,True{},valid_true) :
    Model.ContentAll(~Element,~Global.Made(~Element,~Scope<Element>,~OutputValues(~Element,~band_value),whole),
      Model.contents(~Element,Each.made(~Element,Each.checked(~Element,BandConcat.checked(~Element,_,first,rows,joined_band(~Element,owners,shape,fill))))))
  %packed : Banded(~Element,~band_value,whole,first,rows,Layout.band_joined(~Element,_,shape,fill))
  band_counted(~Element,~band_value,whole,first,rows,owners,facts,fill,Tensors.count_of(shape),{==},
    %Equal.sym(Tensors.Parts<Element>,ConcatLeaf.packed(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),pieces),BandConcatExecution.parts_owner_pack(~Element,owners),packed) :
      {Contract.tensor_ready(~Element,Layout.band_joined(~Element,_,shape,fill)) == True{} : Bool}
    Shape.second({valid == True{} : Bool},{Contract.tensor_ready(~Element,joined_band(~Element,owners,shape,fill)) == True{} : Bool},ready))

# ---- Every band of the output ----

# Every piece of the gather through its band call.
def band_leaves(~Element: Data,+context: BandConcat.Context,+gathered: BandConcatExecution.Gathered(~Element)) -> Bands.Transformed(~Element):
  Each.shaped_leaf_map(~BandConcatExecution.Joined<Element>,~BandConcat.Gathered<Element>,~BandConcatExecution.joined_pack(~Element),~Bands.Checked<Element>,~BandConcat.Context,
    ~BandConcat.band(~Element,~Layout.band_joined(~Element)),context,gathered)

# Every band of the band concat holds the concat at its cells when every
# band call was usable: the gather follows the plan of the parts and holds the
# band of every part where it is valid.
def executed(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,+joining: BandConcat.Joining,+extent: U32,
  +sources: BandConcatExecution.PartsOwnerModel<Element>,
  +total: {U32.to_nat(extent) == parts_total(~Element,joining,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))) : Nat},
  +limits: Machine.Limits,+actual: {limits == Default.limits() : Machine.Limits},+domain: BandGather.Domain<BandConcat.geometry(joining,extent),limits,Machine.capacity(limits)>,
  +fitted: Fitted<joining,extent,Machine.capacity(limits)>,+layout: Bands.Plan,+workers: U32,
  contracts: SourceContracts(~Element,~band_value,joining,Machine.capacity(limits),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),
  layouts: Layouts(~Element,sources,layout),
  accepted: {Each.usable(~Element,BandConcat.execute(~Element,~Layout.band_joined(~Element),layout,joining,extent,workers,
    BandConcatExecution.parts_owner_pack(~Element,sources))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~Global.Made(~Element,~Scope<Element>,~OutputValues(~Element,~band_value),
      Scope{joining,extent,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources)),total,limits,actual,domain,fitted}),
    Model.contents(~Element,Each.made(~Element,BandConcat.execute(~Element,~Layout.band_joined(~Element),layout,joining,extent,workers,
      BandConcatExecution.parts_owner_pack(~Element,sources))))):
  +geometry = BandConcat.geometry(joining,extent)
  +parts = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))
  +scope = {Scope{joining,extent,parts,total,limits,actual,domain,fitted} : Scope<Element>}
  +context = {BandConcat.Context{joining,extent} : BandConcat.Context}
  +gathered = BandConcatExecution.gather(~Element,joining,sources)
  +parallel = BandSchedule.parallel(geometry,workers,8.0,0.0,layout)
  +plan = BandConcatExecution.gathered_plan(~Element,gathered)
  +planned = gathered_layout(~Element,joining,sources,layout,layouts)
  +packed = BandConcatExecution.gathered(~Element,joining,sources)
  +walked = Each.shaped_planned(~BandConcatExecution.Joined<Element>,~BandConcat.Gathered<Element>,~BandConcatExecution.joined_pack(~Element),~Bands.Checked<Element>,~BandConcat.Context,
    ~BandConcat.band(~Element,~Layout.band_joined(~Element)),~BandConcat.unjoined(~Element),gathered,parallel,context)
  %Equal.sym(BandConcat.Tree(~Element),BandConcat.gather(~Element,joining,BandConcatExecution.parts_owner_pack(~Element,sources)),BandConcatExecution.gathered_pack(~Element,gathered),packed) :
    Model.ContentAll(~Element,~Global.Made(~Element,~Scope<Element>,~OutputValues(~Element,~band_value),scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,
      Bands.each(~BandConcat.Gathered<Element>,~Bands.Checked<Element>,~Unit,~BandConcat.Context,~Bands.unshared,~BandConcat.band(~Element,~Layout.band_joined(~Element)),~BandConcat.unjoined(~Element),
        layout,parallel,context,_,Unit{})))))
  %planned :
    Model.ContentAll(~Element,~Global.Made(~Element,~Scope<Element>,~OutputValues(~Element,~band_value),scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,
      Bands.each(~BandConcat.Gathered<Element>,~Bands.Checked<Element>,~Unit,~BandConcat.Context,~Bands.unshared,~BandConcat.band(~Element,~Layout.band_joined(~Element)),~BandConcat.unjoined(~Element),
        _,parallel,context,BandConcatExecution.gathered_pack(~Element,gathered),Unit{})))))
  %Equal.sym(Bands.Transformed(~Element),
    Bands.each(~BandConcat.Gathered<Element>,~Bands.Checked<Element>,~Unit,~BandConcat.Context,~Bands.unshared,~BandConcat.band(~Element,~Layout.band_joined(~Element)),~BandConcat.unjoined(~Element),
      plan,parallel,context,BandConcatExecution.gathered_pack(~Element,gathered),Unit{}),band_leaves(~Element,context,gathered),walked) :
    Model.ContentAll(~Element,~Global.Made(~Element,~Scope<Element>,~OutputValues(~Element,~band_value),scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,_))))
  Each.shaped_checked_all(~Element,~BandConcatExecution.Joined<Element>,~BandConcat.Gathered<Element>,~BandConcatExecution.joined_pack(~Element),~BandConcat.Context,
    ~BandConcat.band(~Element,~Layout.band_joined(~Element)),~Scope<Element>,~scope_context(~Element),
    ~Model.viewed(~BandConcatExecution.Joined<Element>,~Sources<Element>,~JoinedAt(~Element,~band_value),~Scope<Element>,~scope_sources(~Element)),
    ~Global.Made(~Element,~Scope<Element>,~OutputValues(~Element,~band_value)),~band_rule(~Element,~band_value),scope,gathered,
    Model.shaped_all_view(~BandConcatExecution.Joined<Element>,~Sources<Element>,~JoinedAt(~Element,~band_value),~Scope<Element>,~scope_sources(~Element),scope,gathered,
      gather_at(~Element,~band_value,joining,Machine.capacity(limits),sources,contracts)),
    %walked : {Each.usable(~Element,Each.checked(~Element,_)) == True{} : Bool}
    %Equal.sym(Bands.Plan,plan,layout,planned) :
      {Each.usable(~Element,Each.checked(~Element,
        Bands.each(~BandConcat.Gathered<Element>,~Bands.Checked<Element>,~Unit,~BandConcat.Context,~Bands.unshared,~BandConcat.band(~Element,~Layout.band_joined(~Element)),~BandConcat.unjoined(~Element),
          _,parallel,context,BandConcatExecution.gathered_pack(~Element,gathered),Unit{}))) == True{} : Bool}
    %packed :
      {Each.usable(~Element,Each.checked(~Element,
        Bands.each(~BandConcat.Gathered<Element>,~Bands.Checked<Element>,~Unit,~BandConcat.Context,~Bands.unshared,~BandConcat.band(~Element,~Layout.band_joined(~Element)),~BandConcat.unjoined(~Element),
          layout,parallel,context,_,Unit{}))) == True{} : Bool}
    accepted)

# What the caller states of the result, whatever the scope.
def scoped(~Element: Data,~Result: @-tensor: Tensors.Tensor<Element> -> Data,+scope: Scope<Element>,-tensor: Tensors.Tensor<Element>) -> Data:
  Result(tensor)

# The leading part gives the concat its first shape and fill.
def concat_lead(~Element: Data,+axis: U32,parts: Tensors.Parts<Element>) ->
  {Layout.concat(~Element,parts,axis) ==
    Layout.concat_first(~Element,axis,parts,first_shape(~Element,BandConcatExecution.parts_contents(~Element,parts)),first_fill(~Element,BandConcatExecution.parts_contents(~Element,parts))) : Tensors.Tensor<Element>}:
  match parts:
    case Tensors.Single{Tensors.Tensor{bands,geometry,shape,fill,status,workers}}: {==}
    case Tensors.Cons{Tensors.Tensor{bands,geometry,shape,fill,status,workers},rest}: {==}

# Parts in one plan, each holding the reader's values in the geometry of the
# joining, run the band concat: a Ready result has bands that are Normal and
# hold the concat of the parts at every cell, and the caller states what
# follows of them.
def joined(~Element: Data,~band_value: Model.Content<Element> -> Bands.Geometry -> U32 -> Element,~Result: @-tensor: Tensors.Tensor<Element> -> Data,
  +sources: BandConcatExecution.PartsOwnerModel<Element>,+joining: BandConcat.Joining,+extent: U32,+layout: Bands.Plan,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +total: {U32.to_nat(extent) == parts_total(~Element,joining,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))) : Nat},
  +domain: BandGather.Domain<BandConcat.geometry(joining,extent),Default.limits(),Machine.capacity(Default.limits())>,
  +fitted: Fitted<joining,extent,Machine.capacity(Default.limits())>,
  counted: Global.Counted<shape>,coherent: Global.Coherent(shape,BandConcat.geometry(joining,extent)),
  contracts: SourceContracts(~Element,~band_value,joining,Machine.capacity(Default.limits()),BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),
  layouts: Layouts(~Element,sources,layout),
  banded: @+model: Model.Model<Element> -> @normal: Global.Normal(~Element,Tensors.Tensor{Model.pack(~Element,model),BandConcat.geometry(joining,extent),shape,fill,Tensors.Usable{},workers}) ->
    @values: BandValues.Values(~Element,~Concatenated<BandConcatExecution.TensorContent<Element>>,~concat_value(~Element,~band_value),
      logical(~Element,joining,BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))),BandConcat.geometry(joining,extent),
      Model.contents(~Element,Model.pack(~Element,model))) ->
    Result(Tensors.Tensor{Model.pack(~Element,model),BandConcat.geometry(joining,extent),shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,extent,
    BandConcatExecution.parts_owner_pack(~Element,sources),shape,fill,workers)) == True{} : Bool}) ->
  Result(BandConcat.join(~Element,~Layout.band_joined(~Element),layout,joining,extent,BandConcatExecution.parts_owner_pack(~Element,sources),shape,fill,workers)):
  +geometry = BandConcat.geometry(joining,extent)
  +parts = BandConcatExecution.parts_contents(~Element,BandConcatExecution.parts_owner_pack(~Element,sources))
  +scope = {Scope{joining,extent,parts,total,Default.limits(),{==},domain,fitted} : Scope<Element>}
  Global.transformed(~Element,~Scope<Element>,~OutputValues(~Element,~band_value),~scoped(~Element,~Result),scope,"concat",Tensors.InvalidArgument{shape},geometry,shape,fill,workers,
    Default.limits(),BandGather.rows_room(geometry,Default.limits(),Machine.capacity(Default.limits()),domain),
    BandConcat.execute(~Element,~Layout.band_joined(~Element),layout,joining,extent,workers,BandConcatExecution.parts_owner_pack(~Element,sources)),
    usable => executed(~Element,~band_value,joining,extent,sources,total,Default.limits(),{==},domain,fitted,layout,workers,contracts,layouts,usable),
    counted,coherent,model => normal => facts => banded(model,normal,facts),accepted)
