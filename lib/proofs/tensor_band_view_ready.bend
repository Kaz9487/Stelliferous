# The gather of a view in its banded form never fails: every band's local
# view fits its storage, so every leaf gather is accepted, and the output
# bands cover the output geometry.
import Base
import ../tensor.bend as Tensors
import ../tensor_view.bend as Views
import ../tensor_shape.bend as Shapes
import ../tensor_band.bend as Bands
import ../tensor_band_view.bend as BandViews
import ../machine_limits.bend as Machine
import ../tensor_band_schedule.bend as BandSchedule
import ./tensor_band_view_values.bend as Values
import ./tensor_band_view_address.bend as Address
import ./tensor_row_major_coordinates.bend as Coordinates
import ./tensor_view_lanes.bend as Lanes
import ./tensor_shape_refinement.bend as ShapeLaws
import ./storage_shape.bend as Shape
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./u32_full_range.bend as FullRange
import ./boolean_logic.bend as Boolean
import ../storage_buffer.bend as Storage
import ./storage_buffer_model.bend as Buffer
import ./storage_certificate.bend as Certificate
import ./tensor_band_model.bend as Model
import ./tensor_band_plan.bend as BandPlan
import ./tensor_band_each.bend as Each
import ./tensor_band_inspection.bend as Inspection
import ./tensor_band_shape_refinement.bend as BandShape
import ../tensor_operations.bend as Operations
import ./storage_tree_model.bend as Tree
import ./storage_allocation_refinement.bend as AllocationLaws
import ./tensor_contract.bend as Contract
import ./tensor_gather_values.bend as GatherValues
import ./tensor_copy_values.bend as CopyValues
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_operation_bounds.bend as OperationBounds
import ./tensor_canonical_view.bend as Canonical
import ./tensor_band_spread.bend as Spread
import ./tensor_band_count.bend as BandCount
import ./tensor_allocation_depth.bend as Depth
import ./tensor_count_refinement.bend as Count
import ./tensor_count_completion.bend as CountCompletion
import ./nat_division_order.bend as DivisionOrder
import ./binary_radix.bend as Radix
import ./word_encoding.bend as WordEncoding
import ../default_machine_limits.bend as Default
import ./machine_limit_bounds.bend as Limits
import ./tensor_band_gather.bend as BandGather
import ./u32_division.bend as U32Division
import ./checked_word_completion.bend as Completion
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_band_values.bend as BandValues
import ./tensor_global_shape_count.bend as Global
import ./tensor_band_arrive.bend as Arrive
import ./tensor_band_target.bend as Target
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_band_view_classified.bend as Classified
import ./tensor_line_values.bend as LineValues

# ---- The last address of a local view ----

# The weight of the last coordinate of every axis.
def last_weight(+view_axes: List<&2,Views.Axis>) -> Nat:
  Coordinates.weight(Lanes.points(view_axes,Lanes.lasts(view_axes)))

def last_weight_appended(+left: List<&2,Views.Axis>,+right: List<&2,Views.Axis>) ->
  {last_weight(List.append(&2,Views.Axis,left,right)) == Nat.add(last_weight(left),last_weight(right)) : Nat}:
  match left:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      +own = Nat.mul(Nat.sub(U32.to_nat(extent),1n),U32.to_nat(stride))
      %Equal.sym(Nat,last_weight(List.append(&2,Views.Axis,tail,right)),Nat.add(last_weight(tail),last_weight(right)),last_weight_appended(tail,right)) :
        {Nat.add(own,_) == Nat.add(Nat.add(own,last_weight(tail)),last_weight(right)) : Nat}
      Equal.sym(Nat,Nat.add(Nat.add(own,last_weight(tail)),last_weight(right)),Nat.add(own,Nat.add(last_weight(tail),last_weight(right))),
        Addition.associative(own,last_weight(tail),last_weight(right)))

def head_positive(+extent: U32,+tail: List<&2,U32>,nonempty: {Shapes.has_zero(extent <> tail) == False{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}:
  Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(tail),nonempty))

def last_step(+extent: U32,+positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}) ->
  {U32.to_nat((extent - 1 : U32)) == Nat.sub(U32.to_nat(extent),1n) : Nat}:
  FullRange.subtract(extent,1,extent,Order.strict_follow(0n,U32.to_nat(extent),positive),Order.sub_le(U32.to_nat(extent),1n))

# The last coordinates of local axes weigh their channels times the band
# words and their columns.
def last_weight_local(+plane: U32,+band: U32,+capacity: U32,+channels: Nat,+columns: Nat,+view_axes: List<&2,Views.Axis>,
  +nonempty: {Shapes.has_zero(Views.axis_extents(view_axes)) == False{} : Bool},
  +channel_span: {Nat.is_le(Address.span(plane,False{},view_axes),channels) == True{} : Bool},
  +column_span: {Nat.is_le(Address.span(plane,True{},view_axes),columns) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Nat.mul(channels,U32.to_nat(band)),columns),U32.to_nat(capacity)) == True{} : Bool}) ->
  {last_weight(BandViews.localized_axes(plane,band,view_axes)) ==
    Nat.add(Nat.mul(U32.to_nat(band),Address.span(plane,False{},view_axes)),Address.span(plane,True{},view_axes)) : Nat}:
  match view_axes:
    case []: Equal.sym(Nat,Nat.add(Nat.mul(U32.to_nat(band),0n),0n),0n,Address.zero_sum(U32.to_nat(band)))
    case Views.Axis{+extent,+stride} <> +tail:
      +positive = head_positive(extent,Views.axis_extents(tail),nonempty)
      +tail_nonempty = Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty)
      +digit = Nat.sub(U32.to_nat(extent),1n)
      +steps = U32.to_nat((extent - 1 : U32))
      +quotient = U32.to_nat(BandViews.component(plane,False{},stride))
      +leftover = U32.to_nat(BandViews.component(plane,True{},stride))
      +factor = U32.to_nat(band)
      +others = Address.span(plane,False{},tail)
      +rest = Address.span(plane,True{},tail)
      +local_tail = BandViews.localized_axes(plane,band,tail)
      %Equal.sym(Nat,steps,digit,last_step(extent,positive)) :
        {Nat.add(Nat.mul(digit,U32.to_nat(BandViews.localized(plane,band,stride))),last_weight(local_tail)) ==
          Nat.add(Nat.mul(factor,Nat.add(Nat.mul(_,quotient),others)),Nat.add(Nat.mul(_,leftover),rest)) : Nat}
      %Equal.sym(Nat,Nat.mul(digit,U32.to_nat(BandViews.localized(plane,band,stride))),Nat.mul(digit,Nat.add(Nat.mul(quotient,factor),leftover)),
          Address.local_word(plane,band,capacity,extent,stride,digit,channels,columns,Order.last_below(U32.to_nat(extent),positive),
            Address.left_part(Nat.mul(steps,quotient),others,channels,channel_span),Address.left_part(Nat.mul(steps,leftover),rest,columns,column_span),room)) :
        {Nat.add(_,last_weight(local_tail)) == Nat.add(Nat.mul(factor,Nat.add(Nat.mul(digit,quotient),others)),Nat.add(Nat.mul(digit,leftover),rest)) : Nat}
      %Equal.sym(Nat,last_weight(local_tail),Nat.add(Nat.mul(factor,others),rest),
          last_weight_local(plane,band,capacity,channels,columns,tail,tail_nonempty,
            Address.right_part(Nat.mul(steps,quotient),others,channels,channel_span),Address.right_part(Nat.mul(steps,leftover),rest,columns,column_span),room)) :
        {Nat.add(Nat.mul(digit,Nat.add(Nat.mul(quotient,factor),leftover)),_) ==
          Nat.add(Nat.mul(factor,Nat.add(Nat.mul(digit,quotient),others)),Nat.add(Nat.mul(digit,leftover),rest)) : Nat}
      Address.spread(factor,digit,quotient,leftover,others,rest)

def last_weight_repeated(+repeats: List<&2,U32>) -> {last_weight(BandViews.repeated(repeats)) == 0n : Nat}:
  match repeats:
    case []: {==}
    case +extent <> +tail:
      %Equal.sym(Nat,last_weight(BandViews.repeated(tail)),0n,last_weight_repeated(tail)) : {Nat.add(Nat.mul(Nat.sub(U32.to_nat(extent),1n),0n),_) == 0n : Nat}
      Address.zero_sum(Nat.sub(U32.to_nat(extent),1n))

def last_weight_rows(+rows: U32,+width: U32,+repeats: List<&2,U32>) ->
  {last_weight(BandViews.rows_axes(True{},rows,width,repeats)) == Nat.mul(Nat.sub(U32.to_nat(rows),1n),U32.to_nat(width)) : Nat}:
  +own = Nat.mul(Nat.sub(U32.to_nat(rows),1n),U32.to_nat(width))
  %Equal.sym(Nat,last_weight(BandViews.repeated(repeats)),0n,last_weight_repeated(repeats)) : {Nat.add(own,_) == own : Nat}
  Addition.zero_right(own)

# The last channel times the band words, the last row and the last column
# lie inside the band's storage.
def last_inside(+channels: Nat,+rows: Nat,+width: Nat,+factor: Nat,+channel: Nat,+column: Nat,
  +rows_positive: {Nat.is_lt(0n,rows) == True{} : Bool},+channel_below: {Nat.is_lt(channel,channels) == True{} : Bool},
  +column_below: {Nat.is_lt(column,width) == True{} : Bool},+factor_exact: {factor == Nat.mul(rows,width) : Nat}) ->
  {Nat.is_lt(Nat.add(Nat.mul(factor,channel),Nat.add(Nat.mul(Nat.sub(rows,1n),width),column)),Nat.mul(channels,Nat.mul(rows,width))) == True{} : Bool}:
  +inner = RowMajor.radix(Nat.sub(rows,1n),rows,column,width,Order.last_below(rows,rows_positive),column_below)
  %RowMajor.multiply_commutative(channel,factor) : {Nat.is_lt(Nat.add(_,Nat.add(Nat.mul(Nat.sub(rows,1n),width),column)),Nat.mul(channels,Nat.mul(rows,width))) == True{} : Bool}
  %Equal.sym(Nat,factor,Nat.mul(rows,width),factor_exact) : {Nat.is_lt(Nat.add(Nat.mul(channel,_),Nat.add(Nat.mul(Nat.sub(rows,1n),width),column)),Nat.mul(channels,Nat.mul(rows,width))) == True{} : Bool}
  RowMajor.radix(channel,channels,Nat.add(Nat.mul(Nat.sub(rows,1n),width),column),Nat.mul(rows,width),channel_below,inner)

# The local view of a band of rows rows reads inside the band's storage.
def local_fits(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+offset: U32,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+rows: U32,+capacity: U32,+size: Nat,
  +fitted: Values.Fitted<before,repeats,after,offset,batch,channels,height,width>,
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +before_nonempty: {Shapes.has_zero(Views.axis_extents(before)) == False{} : Bool},+after_nonempty: {Shapes.has_zero(Views.axis_extents(after)) == False{} : Bool},
  +nonempty: {Shapes.has_zero(Views.axis_extents(BandViews.axes(BandViews.Banded{before,repeats,after,offset},Bands.Geometry{batch,channels,height,width},rows))) == False{} : Bool},
  +band_exact: {U32.to_nat((rows * width : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(width)) : Nat},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(channels),U32.to_nat((rows * width : U32))),U32.to_nat(capacity)) == True{} : Bool},
  +size_exact: {size == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
  +size_room: {Nat.is_le(size,U32.to_nat(capacity)) == True{} : Bool}) ->
  {Views.fits(BandViews.local(BandViews.Banded{before,repeats,after,offset},Bands.Geometry{batch,channels,height,width},rows),U32.from_nat(size)) == True{} : Bool}:
  Values.Fitted{batch_positive,height_positive,+channel_reach,+column_reach,early,late,repeated,rows_exact} = fitted
  +plane = (height * width : U32)
  +band = (rows * width : U32)
  +factor = U32.to_nat(band)
  +local_before = BandViews.localized_axes(plane,band,before)
  +local_after = BandViews.localized_axes(plane,band,after)
  +middle = BandViews.rows_axes(True{},rows,width,repeats)
  +local_offset = BandViews.localized(plane,band,offset)
  +channel_total = Address.reach_total(plane,False{},before,after,offset)
  +column_total = Address.reach_total(plane,True{},before,after,offset)
  +start_channel = U32.to_nat(BandViews.component(plane,False{},offset))
  +start_column = U32.to_nat(BandViews.component(plane,True{},offset))
  +early_channels = Address.span(plane,False{},before)
  +late_channels = Address.span(plane,False{},after)
  +early_columns = Address.span(plane,True{},before)
  +late_columns = Address.span(plane,True{},after)
  +channel_spans = Address.right_part(start_channel,Nat.add(early_channels,late_channels),channel_total,Order.reflexive(channel_total))
  +column_spans = Address.right_part(start_column,Nat.add(early_columns,late_columns),column_total,Order.reflexive(column_total))
  +local_room = Address.band_room(channel_total,U32.to_nat(channels),column_total,U32.to_nat(width),U32.to_nat(rows),factor,U32.to_nat(capacity),
    channel_reach,column_reach,rows_positive,band_exact,room)
  +offset_exact = Address.local_exact(plane,band,capacity,offset,channel_total,column_total,
    Address.left_part(start_channel,Nat.add(early_channels,late_channels),channel_total,Order.reflexive(channel_total)),
    Address.left_part(start_column,Nat.add(early_columns,late_columns),column_total,Order.reflexive(column_total)),local_room)
  +early_weight = last_weight_local(plane,band,capacity,channel_total,column_total,before,before_nonempty,
    Address.left_part(early_channels,late_channels,channel_total,channel_spans),Address.left_part(early_columns,late_columns,column_total,column_spans),local_room)
  +late_weight = last_weight_local(plane,band,capacity,channel_total,column_total,after,after_nonempty,
    Address.right_part(early_channels,late_channels,channel_total,channel_spans),Address.right_part(early_columns,late_columns,column_total,column_spans),local_room)
  +row_words = Nat.mul(Nat.sub(U32.to_nat(rows),1n),U32.to_nat(width))
  +axes = List.append(&2,Views.Axis,local_before,List.append(&2,Views.Axis,middle,local_after))
  +weight_exact = {
    %last_weight_appended(middle,local_after) : {last_weight(axes) == Nat.add(last_weight(local_before),_) : Nat}
    last_weight_appended(local_before,List.append(&2,Views.Axis,middle,local_after)) :
    {last_weight(axes) == Nat.add(last_weight(local_before),Nat.add(last_weight(middle),last_weight(local_after))) : Nat}}
  +bound = last_inside(U32.to_nat(channels),U32.to_nat(rows),U32.to_nat(width),factor,channel_total,column_total,rows_positive,channel_reach,column_reach,band_exact)
  Lanes.fits_bound(axes,local_offset,U32.from_nat(size),nonempty,
    %Equal.sym(Nat,Coordinates.address(Lanes.points(axes,Lanes.lasts(axes)),U32.to_nat(local_offset),False{}),Nat.add(last_weight(axes),U32.to_nat(local_offset)),
        Coordinates.address_weight(Lanes.points(axes,Lanes.lasts(axes)),U32.to_nat(local_offset))) : {Nat.is_lt(_,U32.to_nat(U32.from_nat(size))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(size)),size,FullRange.roundtrip(size,capacity,size_room)) : {Nat.is_lt(Nat.add(last_weight(axes),U32.to_nat(local_offset)),_) == True{} : Bool}
    %Equal.sym(Nat,size,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),size_exact) : {Nat.is_lt(Nat.add(last_weight(axes),U32.to_nat(local_offset)),_) == True{} : Bool}
    %Equal.sym(Nat,last_weight(axes),Nat.add(last_weight(local_before),Nat.add(last_weight(middle),last_weight(local_after))),weight_exact) :
      {Nat.is_lt(Nat.add(_,U32.to_nat(local_offset)),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))) == True{} : Bool}
    %Equal.sym(Nat,last_weight(local_before),Nat.add(Nat.mul(factor,early_channels),early_columns),early_weight) :
      {Nat.is_lt(Nat.add(Nat.add(_,Nat.add(last_weight(middle),last_weight(local_after))),U32.to_nat(local_offset)),
        Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))) == True{} : Bool}
    %Equal.sym(Nat,last_weight(middle),row_words,last_weight_rows(rows,width,repeats)) :
      {Nat.is_lt(Nat.add(Nat.add(Nat.add(Nat.mul(factor,early_channels),early_columns),Nat.add(_,last_weight(local_after))),U32.to_nat(local_offset)),
        Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))) == True{} : Bool}
    %Equal.sym(Nat,last_weight(local_after),Nat.add(Nat.mul(factor,late_channels),late_columns),late_weight) :
      {Nat.is_lt(Nat.add(Nat.add(Nat.add(Nat.mul(factor,early_channels),early_columns),Nat.add(row_words,_)),U32.to_nat(local_offset)),
        Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(local_offset),Nat.add(Nat.mul(start_channel,factor),start_column),offset_exact) :
      {Nat.is_lt(Nat.add(Nat.add(Nat.add(Nat.mul(factor,early_channels),early_columns),Nat.add(row_words,Nat.add(Nat.mul(factor,late_channels),late_columns))),_),
        Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))) == True{} : Bool}
    %Equal.sym(Nat,
        Nat.add(Nat.add(0n,Nat.add(Nat.add(Nat.mul(factor,early_channels),early_columns),Nat.add(row_words,Nat.add(Nat.mul(factor,late_channels),late_columns)))),
          Nat.add(Nat.mul(start_channel,factor),start_column)),
        Nat.add(0n,Nat.add(Nat.mul(factor,channel_total),Nat.add(row_words,column_total))),
        Address.assembled(factor,0n,early_channels,early_columns,row_words,late_channels,late_columns,start_channel,start_column)) :
      {Nat.is_lt(_,Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))) == True{} : Bool}
    bound)

# ---- Rows the checked bands cover ----

def inspected(~Element: Data,pair: Bands.Bands(~Element) & Bands.Inspection) -> Bands.Inspection:
  (bands,inspection) = pair
  inspection

# The inspection of the bands a checked transform made.
def inspection(~Element: Data,+geometry: Bands.Geometry,result: Bands.Bands(~Element) & Bool) -> Bands.Inspection:
  (bands,usable) = result
  inspected(~Element,Bands.inspect(~Element,geometry,bands))

# Valid leaves from a start over a count of rows.
type Covered<-inspection: Bands.Inspection,-start: Nat,-rows: Nat> is Data:
  Covered{valid_ok: {Inspection.valid(inspection) == True{} : Bool},first_exact: {U32.to_nat(Inspection.first(inspection)) == start : Nat},
    end_exact: {U32.to_nat(Inspection.end(inspection)) == Nat.add(start,rows) : Nat}}

# A checked transform every call of which was usable, whose bands cover rows
# from a start.
def Spans(~Element: Data,+geometry: Bands.Geometry,-result: Bands.Bands(~Element) & Bool,+start: Nat,+rows: Nat) -> Data:
  Shape.Both<{Each.usable(~Element,result) == True{} : Bool},Covered<inspection(~Element,geometry,result),start,rows>>

def covered_joined(+left: Bands.Inspection,+right: Bands.Inspection,+start: Nat,+left_rows: Nat,+right_rows: Nat,
  left_covered: Covered<left,start,left_rows>,right_covered: Covered<right,Nat.add(start,left_rows),right_rows>) ->
  Covered<Bands.join_inspection(left,right),start,Nat.add(left_rows,right_rows)>:
  Bands.Inspection{+left_valid,+left_first,+left_end,left_leaves} = left
  Bands.Inspection{+right_valid,+right_first,+right_end,right_leaves} = right
  Covered{left_ok,left_starts,left_ends} = left_covered
  Covered{right_ok,right_starts,right_ends} = right_covered
  +adjacent = Arithmetic.to_nat_injective(left_end,right_first,
    Equal.trans(Nat,U32.to_nat(left_end),Nat.add(start,left_rows),U32.to_nat(right_first),left_ends,Equal.sym(Nat,U32.to_nat(right_first),Nat.add(start,left_rows),right_starts)))
  Covered{
    %Equal.sym(Bool,left_valid,True{},left_ok) : {_ && right_valid && U32.is_eq(left_end,right_first) == True{} : Bool}
    %Equal.sym(Bool,right_valid,True{},right_ok) : {True{} && _ && U32.is_eq(left_end,right_first) == True{} : Bool}
    %adjacent : {True{} && True{} && U32.is_eq(left_end,_) == True{} : Bool}
    Arithmetic.equal_reflexive(left_end),
    left_starts,
    Equal.trans(Nat,U32.to_nat(right_end),Nat.add(Nat.add(start,left_rows),right_rows),Nat.add(start,Nat.add(left_rows,right_rows)),right_ends,
      Addition.associative(start,left_rows,right_rows))}

def inspections_joined(~Element: Data,left: Bands.Bands(~Element) & Bands.Inspection,right: Bands.Bands(~Element) & Bands.Inspection,
  +start: Nat,+left_rows: Nat,+right_rows: Nat,
  left_covered: Covered<inspected(~Element,left),start,left_rows>,right_covered: Covered<inspected(~Element,right),Nat.add(start,left_rows),right_rows>) ->
  Covered<inspected(~Element,Bands.reduced(~Storage.Buffer<Element>,~Bands.Inspection,~Bands.join_inspection,left,right)),start,Nat.add(left_rows,right_rows)>:
  (left_bands,+left_inspection) = left
  (right_bands,+right_inspection) = right
  covered_joined(left_inspection,right_inspection,start,left_rows,right_rows,left_covered,right_covered)

# Two halves of adjacent rows, joined.
def spans_joined(~Element: Data,+geometry: Bands.Geometry,left: Bands.Bands(~Element) & Bool,right: Bands.Bands(~Element) & Bool,
  +start: Nat,+left_rows: Nat,+right_rows: Nat,
  left_spans: Spans(~Element,geometry,left,start,left_rows),right_spans: Spans(~Element,geometry,right,Nat.add(start,left_rows),right_rows)) ->
  Spans(~Element,geometry,Bands.reduced(~Storage.Buffer<Element>,~Bool,~Bool.and,left,right),start,Nat.add(left_rows,right_rows)):
  (left_bands,+left_usable) = left
  (right_bands,+right_usable) = right
  Shape.Both{left_ok,left_covered} = left_spans
  Shape.Both{right_ok,right_covered} = right_spans
  Shape.Both{
    %Equal.sym(Bool,left_usable,True{},left_ok) : {_ && right_usable == True{} : Bool} right_ok,
    inspections_joined(~Element,Bands.inspect(~Element,geometry,left_bands),Bands.inspect(~Element,geometry,right_bands),start,left_rows,right_rows,left_covered,right_covered)}

# Every leaf of the model through a call whose result covers the rows of the
# leaf scaled by a factor: the whole result covers the rows of the model,
# scaled.
def spans_tree(~Element: Data,~Context: Data,~leaf: Context -> U32 -> U32 -> Storage.Buffer<Element> -> Unit -> Bands.Transformed(~Element),~Scope: Data,
  ~context: @+scope: Scope -> Context,~First: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data,
  ~output: @+scope: Scope -> Bands.Geometry,~factor: @+scope: Scope -> Nat,
  ~rule: @+scope: Scope -> @+first: U32 -> @+rows: U32 -> @+input: Buffer.Model<Element> -> @+first_fact: First(scope,first,rows,Buffer.content(~Element,input)) ->
    Spans(~Element,output(scope),Each.checked(~Element,leaf(context(scope),first,rows,Buffer.pack(~Element,input),Unit{})),
      Nat.mul(U32.to_nat(first),factor(scope)),Nat.mul(U32.to_nat(rows),factor(scope))),
  +scope: Scope,bands: Model.Model<Element>,+start: Nat,+total: Nat,coverage: BandPlan.Coverage(Model.layout(~Element,bands),start,total),
  first_fact: Model.ContentAll(~Element,~First(scope),Model.content(~Element,bands))) ->
  Spans(~Element,output(scope),Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),bands)),Nat.mul(start,factor(scope)),Nat.mul(total,factor(scope))):
  match bands:
    case Model.Leaf{+input,+first,+rows}:
      BandPlan.LeafCoverage{first_equal,rows_equal} = coverage
      %first_equal : Spans(~Element,output(scope),Each.checked(~Element,leaf(context(scope),first,rows,Buffer.pack(~Element,input),Unit{})),Nat.mul(_,factor(scope)),Nat.mul(total,factor(scope)))
      %rows_equal : Spans(~Element,output(scope),Each.checked(~Element,leaf(context(scope),first,rows,Buffer.pack(~Element,input),Unit{})),Nat.mul(U32.to_nat(first),factor(scope)),Nat.mul(_,factor(scope)))
      rule(scope,first,rows,input,first_fact)
    case Model.Fork{+left,+right}:
      BandPlan.ForkCoverage{+left_rows,+right_rows,rows_equal,left_coverage,right_coverage} = coverage
      Shape.Both{left_fact,right_fact} = first_fact
      %Equal.sym(Nat,total,Nat.add(left_rows,right_rows),rows_equal) : Spans(~Element,output(scope),Bands.reduced(~Storage.Buffer<Element>,~Bool,~Bool.and,Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),left)),Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),right))),Nat.mul(start,factor(scope)),Nat.mul(_,factor(scope)))
      %Equal.sym(Nat,Nat.mul(Nat.add(left_rows,right_rows),factor(scope)),Nat.add(Nat.mul(left_rows,factor(scope)),Nat.mul(right_rows,factor(scope))),RowMajor.distrib(left_rows,right_rows,factor(scope))) :
        Spans(~Element,output(scope),Bands.reduced(~Storage.Buffer<Element>,~Bool,~Bool.and,Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),left)),Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),right))),Nat.mul(start,factor(scope)),_)
      spans_joined(~Element,output(scope),Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),left)),
        Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),right)),Nat.mul(start,factor(scope)),Nat.mul(left_rows,factor(scope)),Nat.mul(right_rows,factor(scope)),
        spans_tree(~Element,~Context,~leaf,~Scope,~context,~First,~output,~factor,~rule,scope,left,start,left_rows,left_coverage,left_fact),
        %RowMajor.distrib(start,left_rows,factor(scope)) : Spans(~Element,output(scope),Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Context,~leaf,context(scope),right)),_,Nat.mul(right_rows,factor(scope)))
        spans_tree(~Element,~Context,~leaf,~Scope,~context,~First,~output,~factor,~rule,scope,right,Nat.add(start,left_rows),right_rows,right_coverage,right_fact))

# ---- One band ----

# A leaf whose fields and room hold inspects valid.
def leaf_valid(+batch: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+length: Nat,+depth: Nat,
  local: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool},
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) == True{} : Bool},
  length_exact: {length == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
  room: {Nat.is_le(length,U32.to_nat(U32.shln(1,depth))) == True{} : Bool}) ->
  {Inspection.valid(Bands.leaf_inspection(Bands.Geometry{batch,channels,height,width},length,depth,first,rows)) == True{} : Bool}:
  +counted = Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width)))
  +fit = Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height)))
  +allocated = Nat.is_le(length,U32.to_nat(U32.shln(1,depth)))
  %Equal.sym(Bool,U32.is_le(rows,(height - first % height : U32)),True{},local) : {_ && fit && Nat.is_eq(length,counted) && allocated == True{} : Bool}
  %Equal.sym(Bool,fit,True{},rows_fit) : {True{} && _ && Nat.is_eq(length,counted) && allocated == True{} : Bool}
  %Equal.sym(Bool,allocated,True{},room) : {True{} && True{} && Nat.is_eq(length,counted) && _ == True{} : Bool}
  %Equal.sym(Nat,length,counted,length_exact) : {True{} && True{} && Nat.is_eq(_,counted) && True{} == True{} : Bool}
  %Equal.sym(Bool,Nat.is_eq(counted,counted),True{},Order.equal_reflexive(counted)) : {True{} && True{} && _ && True{} == True{} : Bool}
  {==}

# A usable whole result of one band, held in a model of the fields of the
# band, covers the rows of the band.
def model_spans(~Element: Data,+batch: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+model: Buffer.Model<Element>,+shape: List<&2,U32>,+fill: Element,
  fields: BandShape.Leaf(Bands.Geometry{batch,channels,height,width},first,rows,Buffer.length(~Element,model)),sized: Buffer.Sized<Element,model>,
  end_exact: {U32.to_nat((first + rows : U32)) == Nat.add(U32.to_nat(first),U32.to_nat(rows)) : Nat}) ->
  Spans(~Element,Bands.Geometry{batch,channels,height,width},Each.checked(~Element,Tensors.checked_band(~Element,first,rows,
    Tensors.whole(~Element,Buffer.pack(~Element,model),shape,fill,Tensors.Usable{},1))),U32.to_nat(first),U32.to_nat(rows)):
  Buffer.Model{+tree,+length,Certificate.Certificate{+depth,+equation}} = model
  BandShape.Fields{local,rows_fit,length_exact} = fields
  Buffer.Sized{depth_ok,room} = sized
  +capacity_exact = Shape.capacity(Element,depth,tree,Buffer.balanced(~Element,Buffer.Model{tree,length,Certificate.Certificate{depth,equation}}))
  Shape.Both{{==},Covered{leaf_valid(batch,channels,height,width,first,rows,length,depth,local,rows_fit,length_exact,
      %capacity_exact : {Nat.is_le(length,U32.to_nat(_)) == True{} : Bool} room),{==},end_exact}}

# ---- Words of one band ----

def multiply_swap(+a: Nat,+b: Nat,+c: Nat) -> {Nat.mul(a,Nat.mul(b,c)) == Nat.mul(b,Nat.mul(a,c)) : Nat}:
  Equal.trans(Nat,Nat.mul(a,Nat.mul(b,c)),Nat.mul(Nat.mul(b,a),c),Nat.mul(b,Nat.mul(a,c)),
    %RowMajor.multiply_commutative(a,b) : {Nat.mul(a,Nat.mul(b,c)) == Nat.mul(_,c) : Nat}
    Equal.sym(Nat,Nat.mul(Nat.mul(a,b),c),Nat.mul(a,Nat.mul(b,c)),RowMajor.assoc(a,b,c)),
    RowMajor.assoc(b,a,c))

# A remainder of a value scaled by a factor, by a divisor scaled by it.
def scaled_remainder(+value: Nat,+divisor: Nat,+factor: Nat,+positive: {Nat.is_lt(0n,divisor) == True{} : Bool},
  +factor_positive: {Nat.is_lt(0n,factor) == True{} : Bool}) ->
  {Nat.mod(Nat.mul(value,factor),Nat.mul(divisor,factor)) == Nat.mul(Nat.mod(value,divisor),factor) : Nat}:
  +quotient = Nat.div(value,divisor)
  +remainder = Nat.mod(value,divisor)
  +scaled = Nat.mul(divisor,factor)
  +own = Nat.mul(remainder,factor)
  %Equal.sym(Nat,value,Nat.add(Nat.mul(quotient,divisor),remainder),DivisionOrder.quotient_exact(value,divisor,positive)) :
    {Nat.mod(Nat.mul(_,factor),scaled) == own : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(quotient,divisor),remainder),factor),Nat.add(Nat.mul(Nat.mul(quotient,divisor),factor),own),
      RowMajor.distrib(Nat.mul(quotient,divisor),remainder,factor)) : {Nat.mod(_,scaled) == own : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(quotient,divisor),factor),Nat.mul(quotient,scaled),RowMajor.assoc(quotient,divisor,factor)) :
    {Nat.mod(Nat.add(_,own),scaled) == own : Nat}
  Radix.remainder(quotient,own,scaled,Order.product_positive(divisor,factor,positive,factor_positive),
    RowMajor.multiply_lt(remainder,divisor,factor,DivisionOrder.remainder_small(value,divisor,positive),factor_positive))

# The axes of the local view of a band have the extents of the band.
def local_extents(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+rows: U32) ->
  {Views.axis_extents(BandViews.axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows)) ==
    BandViews.extents(BandViews.Banded{before,repeats,after,start},rows) : List<&2,U32>}:
  +plane = (height * width : U32)
  +band = (rows * width : U32)
  +local_before = BandViews.localized_axes(plane,band,before)
  +local_after = BandViews.localized_axes(plane,band,after)
  +middle = BandViews.rows_axes(True{},rows,width,repeats)
  +tail = List.append(&2,Views.Axis,middle,local_after)
  +extents = BandViews.extents(BandViews.Banded{before,repeats,after,start},rows)
  %Equal.sym(List<&2,U32>,Views.axis_extents(List.append(&2,Views.Axis,local_before,tail)),
      List.append(&2,U32,Views.axis_extents(local_before),Views.axis_extents(tail)),LineValues.extents_appended(local_before,tail)) : {_ == extents : List<&2,U32>}
  %Equal.sym(List<&2,U32>,Views.axis_extents(local_before),Views.axis_extents(before),Address.extents_localized(plane,band,before)) :
    {List.append(&2,U32,_,Views.axis_extents(tail)) == extents : List<&2,U32>}
  %Equal.sym(List<&2,U32>,Views.axis_extents(tail),List.append(&2,U32,Views.axis_extents(middle),Views.axis_extents(local_after)),LineValues.extents_appended(middle,local_after)) :
    {List.append(&2,U32,Views.axis_extents(before),_) == extents : List<&2,U32>}
  %Equal.sym(List<&2,U32>,Views.axis_extents(local_after),Views.axis_extents(after),Address.extents_localized(plane,band,after)) :
    {List.append(&2,U32,Views.axis_extents(before),List.append(&2,U32,Views.axis_extents(middle),_)) == extents : List<&2,U32>}
  %Equal.sym(List<&2,U32>,Views.axis_extents(BandViews.repeated(repeats)),repeats,Address.extents_repeated(repeats)) :
    {List.append(&2,U32,Views.axis_extents(before),List.append(&2,U32,rows <> _,Views.axis_extents(after))) == extents : List<&2,U32>}
  {==}

# The extents of a band count its channels, its repeated rows and its
# columns.
def extents_count(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+rows: U32,
  +early: {U32.to_nat(BandViews.product_or_zero(Views.axis_extents(before))) == ShapeLaws.elements(Views.axis_extents(before)) : Nat},
  +late: {U32.to_nat(BandViews.product_or_zero(Views.axis_extents(after))) == ShapeLaws.elements(Views.axis_extents(after)) : Nat},
  +repeated: {U32.to_nat(BandViews.product_or_zero(repeats)) == ShapeLaws.elements(repeats) : Nat},
  +rows_word: {U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(BandViews.product_or_zero(repeats))) : Nat}) ->
  {ShapeLaws.elements(BandViews.extents(BandViews.Banded{before,repeats,after,start},rows)) ==
    Nat.mul(U32.to_nat(BandViews.product_or_zero(Views.axis_extents(before))),
      Nat.mul(U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32)),U32.to_nat(BandViews.product_or_zero(Views.axis_extents(after))))) : Nat}:
  +early_list = Views.axis_extents(before)
  +late_list = Views.axis_extents(after)
  +middle = {rows <> repeats : List<&2,U32>}
  +tail = List.append(&2,U32,middle,late_list)
  +eb = ShapeLaws.elements(early_list)
  +ea = ShapeLaws.elements(late_list)
  +es = ShapeLaws.elements(repeats)
  +scale = BandViews.product_or_zero(repeats)
  +channels = BandViews.product_or_zero(early_list)
  +width = BandViews.product_or_zero(late_list)
  +out_rows = (rows * scale : U32)
  %Equal.sym(Nat,ShapeLaws.elements(List.append(&2,U32,early_list,tail)),Nat.mul(eb,ShapeLaws.elements(tail)),ShapeLaws.elements_appended(early_list,tail)) :
    {_ == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(out_rows),U32.to_nat(width))) : Nat}
  %Equal.sym(Nat,ShapeLaws.elements(tail),Nat.mul(ShapeLaws.elements(middle),ea),ShapeLaws.elements_appended(middle,late_list)) :
    {Nat.mul(eb,_) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(out_rows),U32.to_nat(width))) : Nat}
  %Equal.sym(Nat,U32.to_nat(channels),eb,early) :
    {Nat.mul(eb,Nat.mul(Nat.mul(U32.to_nat(rows),es),ea)) == Nat.mul(_,Nat.mul(U32.to_nat(out_rows),U32.to_nat(width))) : Nat}
  %Equal.sym(Nat,U32.to_nat(width),ea,late) :
    {Nat.mul(eb,Nat.mul(Nat.mul(U32.to_nat(rows),es),ea)) == Nat.mul(eb,Nat.mul(U32.to_nat(out_rows),_)) : Nat}
  %Equal.sym(Nat,U32.to_nat(out_rows),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),rows_word) :
    {Nat.mul(eb,Nat.mul(Nat.mul(U32.to_nat(rows),es),ea)) == Nat.mul(eb,Nat.mul(_,ea)) : Nat}
  %Equal.sym(Nat,U32.to_nat(scale),es,repeated) :
    {Nat.mul(eb,Nat.mul(Nat.mul(U32.to_nat(rows),es),ea)) == Nat.mul(eb,Nat.mul(Nat.mul(U32.to_nat(rows),_),ea)) : Nat}
  {==}

# A band of no rows has extents of no elements and an empty buffer.
def empty_formed(~Element: Data,+fill: Element,+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+rows: U32,
  +zero: {U32.to_nat(rows) == 0n : Nat}) ->
  Contract.WellFormed<Element,BandViews.extents(BandViews.Banded{before,repeats,after,start},rows),Buffer.contents(~Element,Tensors.buffer(~Element,fill,0))>:
  +early_list = Views.axis_extents(before)
  +late_list = Views.axis_extents(after)
  +middle = {rows <> repeats : List<&2,U32>}
  +tail = List.append(&2,U32,middle,late_list)
  +eb = ShapeLaws.elements(early_list)
  +shape = BandViews.extents(BandViews.Banded{before,repeats,after,start},rows)
  +nothing = Equal.trans(Nat,ShapeLaws.elements(shape),Nat.mul(eb,ShapeLaws.elements(tail)),0n,ShapeLaws.elements_appended(early_list,tail),
    %Equal.sym(Nat,ShapeLaws.elements(tail),Nat.mul(ShapeLaws.elements(middle),ShapeLaws.elements(late_list)),ShapeLaws.elements_appended(middle,late_list)) :
      {Nat.mul(eb,_) == 0n : Nat}
    %Equal.sym(Nat,U32.to_nat(rows),0n,zero) : {Nat.mul(eb,Nat.mul(Nat.mul(_,ShapeLaws.elements(repeats)),ShapeLaws.elements(late_list))) == 0n : Nat}
    RowMajor.multiply_zero_right(eb))
  Contract.WellFormed{
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n},Contract.empty_contents(~Element,fill)) :
      {Buffer.content_length(~Element,_) == ShapeLaws.elements(shape) : Nat}
    Equal.sym(Nat,ShapeLaws.elements(shape),0n,nothing),
    Contract.empty_ready(~Element,fill)}

# A usable whole result of one band in any storage of its extents.
def buffer_spans(~Element: Data,+batch: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,-storage: Storage.Buffer<Element>,+shape: List<&2,U32>,+fill: Element,
  observation: Buffer.Observation<Element,storage>,formed: Contract.WellFormed<Element,shape,Buffer.contents(~Element,storage)>,
  local: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool},
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) == True{} : Bool},
  +shape_exact: {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
  end_exact: {U32.to_nat((first + rows : U32)) == Nat.add(U32.to_nat(first),U32.to_nat(rows)) : Nat}) ->
  Spans(~Element,Bands.Geometry{batch,channels,height,width},Each.checked(~Element,Tensors.checked_band(~Element,first,rows,
    Tensors.whole(~Element,storage,shape,fill,Tensors.Usable{},1))),U32.to_nat(first),U32.to_nat(rows)):
  Buffer.Observation{+model,+equation} = observation
  Contract.WellFormed{length_exact,ready} = formed
  +length_model = Equal.trans(Nat,Buffer.length(~Element,model),Buffer.content_length(~Element,Buffer.content(~Element,model)),
    Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,model)),Buffer.length(~Element,model),Buffer.content_length_of(~Element,model)),
    Equal.trans(Nat,Buffer.content_length(~Element,Buffer.content(~Element,model)),ShapeLaws.elements(shape),Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))),
      %Buffer.packed_contents(~Element,model) : {Buffer.content_length(~Element,_) == ShapeLaws.elements(shape) : Nat}
      %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),storage,equation) : {Buffer.content_length(~Element,Buffer.contents(~Element,_)) == ShapeLaws.elements(shape) : Nat}
      length_exact,shape_exact))
  +sized = Buffer.ready_content(~Element,model,
    %Buffer.packed_contents(~Element,model) : Buffer.ContentReady<Element,_>
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),storage,equation) : Buffer.ContentReady<Element,Buffer.contents(~Element,_)>
    ready)
  %equation : Spans(~Element,Bands.Geometry{batch,channels,height,width},Each.checked(~Element,Tensors.checked_band(~Element,first,rows,
    Tensors.whole(~Element,_,shape,fill,Tensors.Usable{},1))),U32.to_nat(first),U32.to_nat(rows))
  model_spans(~Element,batch,channels,height,width,first,rows,model,shape,fill,BandShape.Fields{local,rows_fit,length_model},sized,end_exact)

# An accepted copy leaves the band of its output.
def copied_spans(~Element: Data,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+first: U32,+rows: U32,+name: String,+source_shape: List<&2,U32>,+shape: List<&2,U32>,+fill: Element,
  +source: Views.View,+reversed: List<&2,Bool>,+input: Buffer.Content<Element>,
  facts: GatherValues.Facts(~Element,~GatherValues.Directed<Buffer.Content<Element>>,~GatherValues.directed(~Element,~Buffer.Content<Element>,~Contract.stored(~Element)),
    GatherValues.Directed{input,source,reversed},shape,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome))),
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) == True{} : Bool},
  local: {U32.is_le(rows,(height - first % height : U32)) == True{} : Bool},
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(height))) == True{} : Bool},
  +shape_exact: {ShapeLaws.elements(shape) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
  end_exact: {U32.to_nat((first + rows : U32)) == Nat.add(U32.to_nat(first),U32.to_nat(rows)) : Nat}) ->
  Spans(~Element,Bands.Geometry{batch,channels,height,width},Each.checked(~Element,Tensors.checked_band(~Element,first,rows,
    Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,shape,fill,1,outcome)))),U32.to_nat(first),U32.to_nat(rows)):
  match outcome:
    case Operations.Accepted{(kept,Storage.Buffer{storage,length,certificate})}:
      Shape.Both{formed,values} = facts
      buffer_spans(~Element,batch,channels,height,width,first,rows,Storage.Buffer{storage,length,certificate},shape,fill,
        Buffer.observe(~Element,Storage.Buffer{storage,length,certificate}),formed,local,rows_fit,shape_exact,end_exact)
    case Operations.Rejected{(kept,output)}:
      Empty.absurd(Spans(~Element,Bands.Geometry{batch,channels,height,width},Each.checked(~Element,Tensors.checked_band(~Element,first,rows,
        Tensors.result_of(~Element,Tensors.copied(~Element,name,source_shape,shape,fill,1,Operations.Rejected{(kept,output)})))),U32.to_nat(first),U32.to_nat(rows)),
        WordEncoding.false_true(accepted))

# ---- A copy into fresh storage ----

# The fresh storage of count elements, observed as a model.
def fresh_length(~Element: Data,+fill: Element,+count: U32,+output: Buffer.Model<Element>,
  allocated: {Buffer.pack(~Element,output) == Tensors.buffer(~Element,fill,count) : Storage.Buffer<Element>}) ->
  {U32.to_nat(count) == Buffer.length(~Element,output) : Nat}:
  +fresh = {Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)} : Buffer.Content<Element>}
  +content = Equal.trans(Buffer.Content<Element>,Buffer.content(~Element,output),Buffer.contents(~Element,Buffer.pack(~Element,output)),fresh,
    Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Buffer.pack(~Element,output)),Buffer.content(~Element,output),Buffer.packed_contents(~Element,output)),
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,output),Tensors.buffer(~Element,fill,count),allocated) : {Buffer.contents(~Element,_) == fresh : Buffer.Content<Element>}
    AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count)))
  Equal.sym(Nat,Buffer.length(~Element,output),U32.to_nat(count),
    Equal.trans(Nat,Buffer.length(~Element,output),Buffer.content_length(~Element,Buffer.content(~Element,output)),U32.to_nat(count),
      Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,output)),Buffer.length(~Element,output),Buffer.content_length_of(~Element,output)),
      Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.content(~Element,output),fresh,content)))

def fresh_sized(~Element: Data,+fill: Element,+count: U32,+output: Buffer.Model<Element>,
  allocated: {Buffer.pack(~Element,output) == Tensors.buffer(~Element,fill,count) : Storage.Buffer<Element>},sized: Depth.Sized<count>) -> Buffer.Sized<Element,output>:
  Buffer.ready_observed(~Element,Tensors.buffer(~Element,fill,count),output,allocated,BandCount.allocated_ready(~Element,count,fill,sized))

def allocation_sized(+expected: Nat,+count: U32,evidence: Count.Count<expected,count>) -> Depth.Sized<count>:
  Count.Count{exact,bounded} = evidence
  Depth.bounded(count,bounded)

# A copy whose guard passes is accepted.
def denoted_accepted(~Element: Data,+input: Buffer.Model<Element>,+source: Views.View,+reversed: List<&2,Bool>,+output: Buffer.Model<Element>,+target: Views.View,+count: U32,
  counted: {Views.numel(target) == Some{count} : Maybe<&2,U32>},
  guard: {OperationBounds.copy_guard(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),target) == True{} : Bool}) ->
  {GatherValues.accepted_owners(~Element,OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),target)) == True{} : Bool}:
  Buffer.Model{+tree,+length,+certificate} = input
  Buffer.Model{+other_tree,+other_length,+other_certificate} = output
  Views.View{+target_axes,+target_offset} = target
  +target = {Views.View{target_axes,target_offset} : Views.View}
  %Equal.sym(Bool,Shapes.same(Views.shape(source),Views.shape(target)) && Views.is_contiguous(target) &&
      Operations.valid_buffer(length,Storage.capacity(~Element,Tree.pack(Element,tree),certificate),source) &&
      Operations.valid_buffer(other_length,Storage.capacity(~Element,Tree.pack(Element,other_tree),other_certificate),target),True{},guard) :
    {GatherValues.accepted_owners(~Element,OperationsLaws.copy_model(~Element,_,Views.numel(target),Views.mirrored(source,reversed),target,
      Tree.reflect(Element,Tree.pack(Element,tree)),length,Tree.reflect(Element,Tree.pack(Element,other_tree)),other_length)) == True{} : Bool}
  %Equal.sym(Maybe<&2,U32>,Views.numel(target),Some{count},counted) :
    {GatherValues.accepted_owners(~Element,OperationsLaws.copy_model(~Element,True{},_,Views.mirrored(source,reversed),target,
      Tree.reflect(Element,Tree.pack(Element,tree)),length,Tree.reflect(Element,Tree.pack(Element,other_tree)),other_length)) == True{} : Bool}
  {==}

def copy_accepted(~Element: Data,+input: Buffer.Model<Element>,+source: Views.View,+reversed: List<&2,Bool>,+output: Buffer.Model<Element>,+target: Views.View,+count: U32,
  counted: {Views.numel(target) == Some{count} : Maybe<&2,U32>},
  guard: {OperationBounds.copy_guard(~Element,Buffer.pack(~Element,input),source,Buffer.pack(~Element,output),target) == True{} : Bool}) ->
  {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),target))) == True{} : Bool}:
  %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,
      OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),target)),
      OperationsLaws.copy_denotation(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),target),
      OperationsLaws.copy(~Element,Buffer.pack(~Element,input),source,reversed,Buffer.pack(~Element,output),target)) :
    {GatherValues.accepted_owners(~Element,_) == True{} : Bool}
  denoted_accepted(~Element,input,source,reversed,output,target,count,counted,guard)

# A copy of a view that fits its input into storage of as many elements
# laid out as the canonical view of its extents passes its guard.
def fresh_guard(~Element: Data,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,+axes: List<&2,Views.Axis>,+offset: U32,+shape: List<&2,U32>,+count: U32,
  input_ready: Buffer.Sized<Element,input>,output_ready: Buffer.Sized<Element,output>,
  +extents: {Views.axis_extents(axes) == shape : List<&2,U32>},
  +fits: {Views.fits(Views.View{axes,offset},U32.from_nat(Buffer.length(~Element,input))) == True{} : Bool},
  +output_length: {U32.to_nat(count) == Buffer.length(~Element,output) : Nat},
  +exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat}) ->
  {OperationBounds.copy_guard(~Element,Buffer.pack(~Element,input),Views.View{axes,offset},Buffer.pack(~Element,output),Canonical.canonical_view(shape,count)) == True{} : Bool}:
  +source = {Views.View{axes,offset} : Views.View}
  +canon = Canonical.canonical_view(shape,count)
  +same = {
    %Equal.sym(List<&2,U32>,Views.axis_extents(axes),shape,extents) : {Shapes.same_extents(_,Views.axis_extents_of(canon)) == True{} : Bool}
    %Equal.sym(List<&2,U32>,Views.axis_extents_of(canon),shape,Canonical.canonical_shape(shape,count,exact)) : {Shapes.same_extents(shape,_) == True{} : Bool}
    ShapeLaws.same_extents_reflexive(shape) : {Shapes.same(Views.shape(source),Views.shape(canon)) == True{} : Bool}}
  +input_valid = Spread.valid_view(~Element,input,source,input_ready,fits)
  +output_valid = Spread.valid_view(~Element,output,canon,output_ready,
    %Equal.sym(U32,U32.from_nat(Buffer.length(~Element,output)),count,Arithmetic.from_nat_exact(count,Buffer.length(~Element,output),output_length)) :
      {Views.fits(canon,_) == True{} : Bool}
    Spread.canonical_fits(shape,count,exact))
  %Equal.sym(Bool,Shapes.same(Views.shape(source),Views.shape(canon)),True{},same) :
    {_ && Views.is_contiguous(canon) && OperationBounds.valid(~Element,Buffer.pack(~Element,input),source) && OperationBounds.valid(~Element,Buffer.pack(~Element,output),canon) == True{} : Bool}
  %Equal.sym(Bool,Views.is_contiguous(canon),True{},Canonical.canonical_contiguous(shape,count,exact)) :
    {True{} && _ && OperationBounds.valid(~Element,Buffer.pack(~Element,input),source) && OperationBounds.valid(~Element,Buffer.pack(~Element,output),canon) == True{} : Bool}
  %Equal.sym(Bool,OperationBounds.valid(~Element,Buffer.pack(~Element,input),source),True{},input_valid) :
    {True{} && True{} && _ && OperationBounds.valid(~Element,Buffer.pack(~Element,output),canon) == True{} : Bool}
  %Equal.sym(Bool,OperationBounds.valid(~Element,Buffer.pack(~Element,output),canon),True{},output_valid) : {True{} && True{} && True{} && _ == True{} : Bool}
  {==}

# ---- One band of a gather ----

def fitted_channels(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+offset: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  fitted: Values.Fitted<before,repeats,after,offset,batch,channels,height,width>) -> {Nat.is_lt(0n,U32.to_nat(channels)) == True{} : Bool}:
  Values.Fitted{batch_positive,height_positive,channel_reach,column_reach,early,late,repeated,rows_exact} = fitted
  Order.positive_bound(Address.reach_total((height * width : U32),False{},before,after,offset),U32.to_nat(channels),channel_reach)

def sized_room(~Element: Data,+model: Buffer.Model<Element>,sized: Buffer.Sized<Element,model>) ->
  {Nat.is_le(Buffer.length(~Element,model),U32.to_nat(Tree.capacity(Element,Buffer.tree(~Element,model)))) == True{} : Bool}:
  Buffer.Sized{depth_ok,room} = sized
  room

# A band of rows gathers through its local view into fresh storage of its
# extents; the copy accepts it.
def fresh_band(~Element: Data,+name: String,+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+reversed: List<&2,Bool>,+fill: Element,+fitted: Values.Fitted<before,repeats,after,start,batch,channels,height,width>,
  +first: U32,+rows: U32,+input: Buffer.Model<Element>,+input_ready: Buffer.Sized<Element,input>,
  +length_exact: {Buffer.length(~Element,input) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
  +rows_positive: {Nat.is_lt(0n,U32.to_nat(rows)) == True{} : Bool},
  +early_positive: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(before))) == True{} : Bool},
  +late_positive: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(after))) == True{} : Bool},
  +count: U32,+exact: {ShapeLaws.elements(BandViews.extents(BandViews.Banded{before,repeats,after,start},rows)) == U32.to_nat(count) : Nat},
  +counts: {Tensors.count_of(BandViews.extents(BandViews.Banded{before,repeats,after,start},rows)) == Some{count} : Maybe<&2,U32>},
  +positive_count: {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
  observation: Buffer.Observation<Element,Tensors.buffer(~Element,fill,count)>,
  local: {U32.is_le((rows * BandViews.product_or_zero(repeats) : U32),((height * BandViews.product_or_zero(repeats) : U32) -
    (first * BandViews.product_or_zero(repeats) : U32) % (height * BandViews.product_or_zero(repeats) : U32) : U32)) == True{} : Bool},
  rows_fit: {Nat.is_le(Nat.add(U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32))),
    Nat.mul(U32.to_nat(batch),U32.to_nat((height * BandViews.product_or_zero(repeats) : U32)))) == True{} : Bool},
  +shape_exact: {ShapeLaws.elements(BandViews.extents(BandViews.Banded{before,repeats,after,start},rows)) ==
    Nat.mul(U32.to_nat(BandViews.product_or_zero(Views.axis_extents(before))),
      Nat.mul(U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32)),U32.to_nat(BandViews.product_or_zero(Views.axis_extents(after))))) : Nat},
  end_exact: {U32.to_nat(((first * BandViews.product_or_zero(repeats) : U32) + (rows * BandViews.product_or_zero(repeats) : U32) : U32)) ==
    Nat.add(U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32))) : Nat}) ->
  Spans(~Element,Bands.Geometry{batch,BandViews.product_or_zero(Views.axis_extents(before)),(height * BandViews.product_or_zero(repeats) : U32),
      BandViews.product_or_zero(Views.axis_extents(after))},
    Each.checked(~Element,Tensors.checked_band(~Element,(first * BandViews.product_or_zero(repeats) : U32),(rows * BandViews.product_or_zero(repeats) : U32),
      Tensors.result_of(~Element,Tensors.copied(~Element,name,[channels,rows,width],BandViews.extents(BandViews.Banded{before,repeats,after,start},rows),fill,1,
        Operations.copy(~Element,Buffer.pack(~Element,input),BandViews.local(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows),reversed,
          Tensors.buffer(~Element,fill,count),Canonical.canonical_view(BandViews.extents(BandViews.Banded{before,repeats,after,start},rows),count)))))),
    U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32))):
  Buffer.Observation{+output,+allocation} = observation
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +scale = BandViews.product_or_zero(repeats)
  +shape = BandViews.extents(banded,rows)
  +plane = (height * width : U32)
  +band = (rows * width : U32)
  +local_axes = BandViews.axes(banded,geometry,rows)
  +local_offset = BandViews.localized(plane,band,start)
  +source = BandViews.local(banded,geometry,rows)
  +canon = Canonical.canonical_view(shape,count)
  +capacity = Tree.capacity(Element,Buffer.tree(~Element,input))
  +size = Buffer.length(~Element,input)
  +size_room = sized_room(~Element,input,input_ready)
  +channels_positive = fitted_channels(before,repeats,after,start,batch,channels,height,width,fitted)
  +band_words = Nat.mul(U32.to_nat(rows),U32.to_nat(width))
  +band_le = {%Equal.sym(Nat,size,Nat.mul(U32.to_nat(channels),band_words),length_exact) : {Nat.is_le(band_words,_) == True{} : Bool}
    DivisionOrder.le_multiple(U32.to_nat(channels),band_words,channels_positive) : {Nat.is_le(band_words,size) == True{} : Bool}}
  +band_exact = FullRange.multiply(rows,width,capacity,Order.transitive(band_words,size,U32.to_nat(capacity),band_le,size_room))
  +room = {%Equal.sym(Nat,U32.to_nat(band),band_words,band_exact) : {Nat.is_le(Nat.mul(U32.to_nat(channels),_),U32.to_nat(capacity)) == True{} : Bool}
    %length_exact : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    size_room : {Nat.is_le(Nat.mul(U32.to_nat(channels),U32.to_nat(band)),U32.to_nat(capacity)) == True{} : Bool}}
  +before_nonempty = CountCompletion.positive_not_zero(Views.axis_extents(before),early_positive)
  +after_nonempty = CountCompletion.positive_not_zero(Views.axis_extents(after),late_positive)
  +shape_nonempty = CountCompletion.positive_not_zero(shape,%Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),exact) : {Nat.is_lt(0n,_) == True{} : Bool} positive_count)
  +extents_exact = local_extents(before,repeats,after,start,batch,channels,height,width,rows)
  +nonempty = {%Equal.sym(List<&2,U32>,Views.axis_extents(local_axes),shape,extents_exact) : {Shapes.has_zero(_) == False{} : Bool}
    shape_nonempty : {Shapes.has_zero(Views.axis_extents(local_axes)) == False{} : Bool}}
  +fits = local_fits(before,repeats,after,start,batch,channels,height,width,rows,capacity,size,fitted,rows_positive,before_nonempty,after_nonempty,nonempty,
    band_exact,room,length_exact,size_room)
  +output_ready = fresh_sized(~Element,fill,count,output,allocation,allocation_sized(ShapeLaws.elements(shape),count,Count.count_of(shape,count,counts)))
  +output_length = fresh_length(~Element,fill,count,output,allocation)
  +guard = fresh_guard(~Element,input,output,local_axes,local_offset,shape,count,input_ready,output_ready,extents_exact,fits,output_length,exact)
  +numel = {%Equal.sym(List<&2,U32>,Views.axis_extents_of(canon),shape,Canonical.canonical_shape(shape,count,exact)) : {Shapes.numel(Shapes.Shape{_}) == Some{count} : Maybe<&2,U32>}
    CountCompletion.numel(shape,count,exact) : {Views.numel(canon) == Some{count} : Maybe<&2,U32>}}
  +accepted = {%allocation : {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,_,canon))) == True{} : Bool}
    copy_accepted(~Element,input,source,reversed,output,canon,count,numel,guard) :
    {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Tensors.buffer(~Element,fill,count),canon))) == True{} : Bool}}
  +facts = GatherValues.targeted_facts(~Element,~Buffer.Content<Element>,~Contract.stored(~Element),canon,Buffer.content(~Element,input),source,reversed,input,shape,shape,fill,count,
    input_ready,Contract.stored_values(~Element,Buffer.content(~Element,input)),Canonical.view(shape,count,exact),counts,count => counted => {==},accepted)
  copied_spans(~Element,Operations.copy(~Element,Buffer.pack(~Element,input),source,reversed,Tensors.buffer(~Element,fill,count),canon),
    batch,BandViews.product_or_zero(Views.axis_extents(before)),(height * scale : U32),BandViews.product_or_zero(Views.axis_extents(after)),
    (first * scale : U32),(rows * scale : U32),name,[channels,rows,width],shape,fill,source,reversed,Buffer.content(~Element,input),facts,accepted,local,rows_fit,shape_exact,end_exact)

# A band of rows gathers through its local view; a band of none is empty.
def rows_spans(~Element: Data,empty: Bool,+name: String,+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+reversed: List<&2,Bool>,+fill: Element,+fitted: Values.Fitted<before,repeats,after,start,batch,channels,height,width>,
  +first: U32,+rows: U32,+input: Buffer.Model<Element>,+input_ready: Buffer.Sized<Element,input>,
  +length_exact: {Buffer.length(~Element,input) == Nat.mul(U32.to_nat(channels),Nat.mul(U32.to_nat(rows),U32.to_nat(width))) : Nat},
  +compared: {U32.is_zero(rows) == empty : Bool},+limits: Machine.Limits,+default: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +channels_positive: {Nat.is_lt(0n,U32.to_nat(BandViews.product_or_zero(Views.axis_extents(before)))) == True{} : Bool},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(BandViews.product_or_zero(Views.axis_extents(after)))) == True{} : Bool},
  +scale_positive: {Nat.is_lt(0n,U32.to_nat(BandViews.product_or_zero(repeats))) == True{} : Bool},
  +early: {U32.to_nat(BandViews.product_or_zero(Views.axis_extents(before))) == ShapeLaws.elements(Views.axis_extents(before)) : Nat},
  +late: {U32.to_nat(BandViews.product_or_zero(Views.axis_extents(after))) == ShapeLaws.elements(Views.axis_extents(after)) : Nat},
  +rows_word: {U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32)) == Nat.mul(U32.to_nat(rows),U32.to_nat(BandViews.product_or_zero(repeats))) : Nat},
  +count_room: {Nat.is_le(Nat.mul(U32.to_nat(BandViews.product_or_zero(Views.axis_extents(before))),
      Nat.mul(U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32)),U32.to_nat(BandViews.product_or_zero(Views.axis_extents(after))))),
    U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +local: {U32.is_le((rows * BandViews.product_or_zero(repeats) : U32),((height * BandViews.product_or_zero(repeats) : U32) -
    (first * BandViews.product_or_zero(repeats) : U32) % (height * BandViews.product_or_zero(repeats) : U32) : U32)) == True{} : Bool},
  +rows_fit: {Nat.is_le(Nat.add(U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32))),
    Nat.mul(U32.to_nat(batch),U32.to_nat((height * BandViews.product_or_zero(repeats) : U32)))) == True{} : Bool},
  +shape_exact: {ShapeLaws.elements(BandViews.extents(BandViews.Banded{before,repeats,after,start},rows)) ==
    Nat.mul(U32.to_nat(BandViews.product_or_zero(Views.axis_extents(before))),
      Nat.mul(U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32)),U32.to_nat(BandViews.product_or_zero(Views.axis_extents(after))))) : Nat},
  +end_exact: {U32.to_nat(((first * BandViews.product_or_zero(repeats) : U32) + (rows * BandViews.product_or_zero(repeats) : U32) : U32)) ==
    Nat.add(U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32))) : Nat}) ->
  Spans(~Element,BandViews.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),
    Each.checked(~Element,Tensors.gathered_rows(~Element,empty,name,BandViews.Banded{before,repeats,after,start},batch,channels,height,width,reversed,fill,first,rows,Buffer.pack(~Element,input))),
    U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),U32.to_nat((rows * BandViews.product_or_zero(repeats) : U32))):
  match empty:
    case True{}:
      +scale = BandViews.product_or_zero(repeats)
      buffer_spans(~Element,batch,BandViews.product_or_zero(Views.axis_extents(before)),(height * scale : U32),BandViews.product_or_zero(Views.axis_extents(after)),
        (first * scale : U32),(rows * scale : U32),Tensors.buffer(~Element,fill,0),BandViews.extents(BandViews.Banded{before,repeats,after,start},rows),fill,
        Buffer.observe(~Element,Tensors.buffer(~Element,fill,0)),
        empty_formed(~Element,fill,before,repeats,after,start,rows,Equal.cong(U32,Nat,U32.to_nat,rows,0,Arithmetic.word_zero(rows,compared))),
        local,rows_fit,shape_exact,end_exact)
    case False{}:
      +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
      +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
      +scale = BandViews.product_or_zero(repeats)
      +shape = BandViews.extents(banded,rows)
      +source = BandViews.local(banded,geometry,rows)
      +out_channels = BandViews.product_or_zero(Views.axis_extents(before))
      +out_width = BandViews.product_or_zero(Views.axis_extents(after))
      +out_rows = (rows * scale : U32)
      +capacity = Machine.capacity(limits)
      +band_count = Nat.mul(U32.to_nat(out_channels),Nat.mul(U32.to_nat(out_rows),U32.to_nat(out_width)))
      +count = U32.from_nat(band_count)
      +count_exact = FullRange.roundtrip(band_count,capacity,count_room)
      +exact = Equal.trans(Nat,ShapeLaws.elements(shape),band_count,U32.to_nat(count),shape_exact,Equal.sym(Nat,U32.to_nat(count),band_count,count_exact))
      +counts = CountCompletion.count_of(shape,count,exact,%default : {U32.is_le(count,_) == True{} : Bool}
        U32Division.nat_le_native(count,capacity,%Equal.sym(Nat,U32.to_nat(count),band_count,count_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool} count_room))
      +canon = Canonical.canonical_view(shape,count)
      +rows_positive = Arithmetic.word_nonzero_positive(rows,compared)
      +out_rows_positive = {%Equal.sym(Nat,U32.to_nat(out_rows),Nat.mul(U32.to_nat(rows),U32.to_nat(scale)),rows_word) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.product_positive(U32.to_nat(rows),U32.to_nat(scale),rows_positive,scale_positive) : {Nat.is_lt(0n,U32.to_nat(out_rows)) == True{} : Bool}}
      +positive_count = {%Equal.sym(Nat,U32.to_nat(count),band_count,count_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
        Order.product_positive(U32.to_nat(out_channels),Nat.mul(U32.to_nat(out_rows),U32.to_nat(out_width)),channels_positive,
          Order.product_positive(U32.to_nat(out_rows),U32.to_nat(out_width),out_rows_positive,width_positive)) : {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}}
      %Equal.sym(Maybe<&2,U32>,Tensors.count_of(shape),Some{count},counts) :
        Spans(~Element,BandViews.output(banded,geometry),Each.checked(~Element,Tensors.checked_band(~Element,(first * scale : U32),out_rows,
          Tensors.result_of(~Element,Tensors.gather_views(~Element,name,Some{source},reversed,Tensors.view_of(shape),_,Buffer.pack(~Element,input),[channels,rows,width],shape,fill,1)))),
          U32.to_nat((first * scale : U32)),U32.to_nat(out_rows))
      %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(shape),Some{canon},Canonical.view(shape,count,exact)) :
        Spans(~Element,BandViews.output(banded,geometry),Each.checked(~Element,Tensors.checked_band(~Element,(first * scale : U32),out_rows,
          Tensors.result_of(~Element,Tensors.gather_views(~Element,name,Some{source},reversed,_,Some{count},Buffer.pack(~Element,input),[channels,rows,width],shape,fill,1)))),
          U32.to_nat((first * scale : U32)),U32.to_nat(out_rows))
      fresh_band(~Element,name,before,repeats,after,start,batch,channels,height,width,reversed,fill,fitted,first,rows,input,input_ready,length_exact,rows_positive,
        %early : {Nat.is_lt(0n,_) == True{} : Bool} channels_positive,%late : {Nat.is_lt(0n,_) == True{} : Bool} width_positive,
        count,exact,counts,positive_count,Buffer.observe(~Element,Tensors.buffer(~Element,fill,count)),local,rows_fit,shape_exact,end_exact)

# What every band of one gather shares: the call, the banded form of its
# view over the input geometry, the flags of its local view, and what the
# classification and the machine establish of that form.
type Scope<-Element: Data> is Data:
  Scope{name: String,banded: BandViews.Banded,geometry: Bands.Geometry,local: List<&2,Bool>,fill: Element,words: Values.Words(banded,geometry),limits: Machine.Limits,
    default: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},room: BandGather.Domain<BandViews.output(banded,geometry),limits,Machine.capacity(limits)>}

def scope_gathered(~Element: Data,+scope: Scope<Element>) -> Tensors.Gathered<Element>:
  Scope{name,banded,geometry,local,fill,words,limits,default,room} = scope
  Tensors.Gathered{name,banded,geometry,local,fill}

def scope_output(~Element: Data,+scope: Scope<Element>) -> Bands.Geometry:
  Scope{name,+banded,+geometry,local,fill,words,limits,default,room} = scope
  BandViews.output(banded,geometry)

def scope_scale(~Element: Data,+scope: Scope<Element>) -> Nat:
  Scope{name,+banded,geometry,local,fill,words,limits,default,room} = scope
  U32.to_nat(BandViews.scale(banded))

def scope_fields(~Element: Data,+scope: Scope<Element>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Scope{name,banded,+geometry,local,fill,words,limits,default,room} = scope
  BandShape.band_fields(~Element,geometry)

# The fields of an input band and its ready storage.
def Input(~Element: Data,+scope: Scope<Element>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Model.Both(~Element,~Scope<Element>,~scope_fields(~Element),~Model.always(~Element,~Scope<Element>,~Model.ready_band(~Element)),scope)

# Every band of a gather is usable and covers its rows, repeated.
def leaf_spans(~Element: Data,+scope: Scope<Element>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+first_fact: Input(~Element,scope,first,rows,Buffer.content(~Element,input))) ->
  Spans(~Element,scope_output(~Element,scope),Each.checked(~Element,Bands.transform_band(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element),
    scope_gathered(~Element,scope),first,rows,Buffer.pack(~Element,input),Unit{})),Nat.mul(U32.to_nat(first),scope_scale(~Element,scope)),Nat.mul(U32.to_nat(rows),scope_scale(~Element,scope))):
  Scope{+name,+banded,+geometry,+flags,+fill,+words,+limits,+default,+room} = scope
  BandViews.Banded{+before,+repeats,+after,+start} = banded
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Shape.Both{+fitted,+plane} = words
  Values.Fitted{+batch_positive,+height_positive,+channel_reach,+column_reach,+early,+late,+repeated,+rows_exact} = fitted
  Values.Plane{+divisor,+plane_exact,+image_exact} = plane
  Address.Divisor{+cap,+cap_ok,+bound,+positive} = divisor
  BandGather.Domain{+capacity_ok,+channels_positive,+out_height_positive,+width_positive,+machine_room,+allocation_room} = room
  Buffer.Model{+tree,+length,+certificate} = input
  Shape.Both{+fields,+content_ready} = first_fact
  BandShape.Fields{+local,+rows_fit,+length_exact} = fields
  +fitted = {Values.Fitted{batch_positive,height_positive,channel_reach,column_reach,early,late,repeated,rows_exact} : Values.Fitted<before,repeats,after,start,batch,channels,height,width>}
  +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
  +scale = BandViews.product_or_zero(repeats)
  +scale_value = U32.to_nat(scale)
  +out_channels = BandViews.product_or_zero(Views.axis_extents(before))
  +out_height = (height * scale : U32)
  +out_width = BandViews.product_or_zero(Views.axis_extents(after))
  +output = {Bands.Geometry{batch,out_channels,out_height,out_width} : Bands.Geometry}
  +capacity = Machine.capacity(limits)
  +out_first = (first * scale : U32)
  +out_rows = (rows * scale : U32)
  +h = U32.to_nat(height)
  +w = U32.to_nat(width)
  +f = U32.to_nat(first)
  +r = U32.to_nat(rows)
  +domain = {BandGather.Domain{capacity_ok,channels_positive,out_height_positive,width_positive,machine_room,allocation_room} : BandGather.Domain<output,limits,capacity>}
  +total_room = BandGather.rows_room(output,limits,capacity,domain)
  +fit = Values.scaled_rows_fit(batch,height,scale,first,rows,rows_fit,rows_exact)
  +fit_room = Order.transitive(Nat.mul(Nat.add(f,r),scale_value),Nat.mul(U32.to_nat(batch),U32.to_nat(out_height)),U32.to_nat(capacity),fit,total_room)
  +first_word = Values.scaled_first_exact(first,rows,scale,capacity,fit_room)
  +rows_word = Values.scaled_rows_exact(first,rows,scale,capacity,fit_room)
  +sum_exact = {
    %Equal.sym(Nat,U32.to_nat(out_first),Nat.mul(f,scale_value),first_word) : {Nat.add(_,U32.to_nat(out_rows)) == Nat.mul(Nat.add(f,r),scale_value) : Nat}
    %Equal.sym(Nat,U32.to_nat(out_rows),Nat.mul(r,scale_value),rows_word) : {Nat.add(Nat.mul(f,scale_value),_) == Nat.mul(Nat.add(f,r),scale_value) : Nat}
    Equal.sym(Nat,Nat.mul(Nat.add(f,r),scale_value),Nat.add(Nat.mul(f,scale_value),Nat.mul(r,scale_value)),RowMajor.distrib(f,r,scale_value)) :
    {Nat.add(U32.to_nat(out_first),U32.to_nat(out_rows)) == Nat.mul(Nat.add(f,r),scale_value) : Nat}}
  +rows_fit_out = {%Equal.sym(Nat,Nat.add(U32.to_nat(out_first),U32.to_nat(out_rows)),Nat.mul(Nat.add(f,r),scale_value),sum_exact) :
      {Nat.is_le(_,Nat.mul(U32.to_nat(batch),U32.to_nat(out_height))) == True{} : Bool}
    fit : {Nat.is_le(Nat.add(U32.to_nat(out_first),U32.to_nat(out_rows)),Nat.mul(U32.to_nat(batch),U32.to_nat(out_height))) == True{} : Bool}}
  +end_exact = FullRange.add(out_first,out_rows,capacity,
    %Equal.sym(Nat,Nat.add(U32.to_nat(out_first),U32.to_nat(out_rows)),Nat.mul(Nat.add(f,r),scale_value),sum_exact) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    fit_room)
  +plane_positive = {%plane_exact : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(h,w)) == True{} : Bool}}
  +height_double = Order.transitive(Nat.double(h),Nat.double(Nat.mul(h,w)),U32.to_nat(cap),
    Order.double_le(h,Nat.mul(h,w),RowMajor.product_covers(h,w,DivisionOrder.right_factor_positive(h,w,plane_positive))),
    %plane_exact : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool} bound)
  +input_span = Inspection.local_span(height,first,rows,cap,cap_ok,height_positive,height_double,local)
  +scale_positive = DivisionOrder.right_factor_positive(h,scale_value,%rows_exact : {Nat.is_lt(0n,_) == True{} : Bool} out_height_positive)
  +span_out = {
    %Equal.sym(Nat,U32.to_nat(out_first),Nat.mul(f,scale_value),first_word) :
      {Nat.is_le(Nat.add(Nat.mod(_,U32.to_nat(out_height)),U32.to_nat(out_rows)),U32.to_nat(out_height)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(out_height),Nat.mul(h,scale_value),rows_exact) :
      {Nat.is_le(Nat.add(Nat.mod(Nat.mul(f,scale_value),_),U32.to_nat(out_rows)),_) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(out_rows),Nat.mul(r,scale_value),rows_word) :
      {Nat.is_le(Nat.add(Nat.mod(Nat.mul(f,scale_value),Nat.mul(h,scale_value)),_),Nat.mul(h,scale_value)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mod(Nat.mul(f,scale_value),Nat.mul(h,scale_value)),Nat.mul(Nat.mod(f,h),scale_value),scaled_remainder(f,h,scale_value,height_positive,scale_positive)) :
      {Nat.is_le(Nat.add(_,Nat.mul(r,scale_value)),Nat.mul(h,scale_value)) == True{} : Bool}
    %RowMajor.distrib(Nat.mod(f,h),r,scale_value) : {Nat.is_le(_,Nat.mul(h,scale_value)) == True{} : Bool}
    RowMajor.mul_le(Nat.add(Nat.mod(f,h),r),h,scale_value,input_span) :
    {Nat.is_le(Nat.add(Nat.mod(U32.to_nat(out_first),U32.to_nat(out_height)),U32.to_nat(out_rows)),U32.to_nat(out_height)) == True{} : Bool}}
  +height_room = Order.transitive(U32.to_nat(out_height),Nat.mul(U32.to_nat(batch),U32.to_nat(out_height)),U32.to_nat(capacity),
    DivisionOrder.le_multiple(U32.to_nat(batch),U32.to_nat(out_height),batch_positive),total_room)
  +local_out = Inspection.local_guard(out_height,out_first,out_rows,Machine.half_word(limits),Limits.half_word_fits(limits),out_height_positive,
    Limits.double_bound(limits,U32.to_nat(out_height),height_room),span_out)
  +oc = U32.to_nat(out_channels)
  +ow = U32.to_nat(out_width)
  +oh = U32.to_nat(out_height)
  +b = U32.to_nat(batch)
  +rows_le = Address.right_part(U32.to_nat(out_first),U32.to_nat(out_rows),Nat.mul(b,oh),rows_fit_out)
  +count_room = Order.transitive(Nat.mul(oc,Nat.mul(U32.to_nat(out_rows),ow)),Nat.mul(oc,Nat.mul(Nat.mul(b,oh),ow)),U32.to_nat(capacity),
    DivisionOrder.multiply_left_le(oc,Nat.mul(U32.to_nat(out_rows),ow),Nat.mul(Nat.mul(b,oh),ow),RowMajor.mul_le(U32.to_nat(out_rows),Nat.mul(b,oh),ow,rows_le)),
    %Equal.sym(Nat,Nat.mul(Nat.mul(b,oh),ow),Nat.mul(b,Nat.mul(oh,ow)),RowMajor.assoc(b,oh,ow)) : {Nat.is_le(Nat.mul(oc,_),U32.to_nat(capacity)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(oc,Nat.mul(b,Nat.mul(oh,ow))),Nat.mul(b,Nat.mul(oc,Nat.mul(oh,ow))),multiply_swap(oc,b,Nat.mul(oh,ow))) : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
    machine_room)
  %first_word : Spans(~Element,output,Each.checked(~Element,Tensors.gathered_rows(~Element,U32.is_zero(rows),name,BandViews.Banded{before,repeats,after,start},
    batch,channels,height,width,flags,fill,first,rows,Buffer.pack(~Element,input))),_,Nat.mul(r,scale_value))
  %rows_word : Spans(~Element,output,Each.checked(~Element,Tensors.gathered_rows(~Element,U32.is_zero(rows),name,BandViews.Banded{before,repeats,after,start},
    batch,channels,height,width,flags,fill,first,rows,Buffer.pack(~Element,input))),U32.to_nat(out_first),_)
  rows_spans(~Element,U32.is_zero(rows),name,before,repeats,after,start,batch,channels,height,width,flags,fill,fitted,first,rows,input,
    Buffer.ready_content(~Element,input,content_ready),length_exact,{==},limits,default,channels_positive,width_positive,scale_positive,early,late,rows_word,count_room,
    local_out,rows_fit_out,extents_count(before,repeats,after,start,rows,early,late,repeated,rows_word),end_exact)

# ---- Every band of a gather ----

# Valid leaves over every row of a geometry cover it.
def covers_true(+geometry: Bands.Geometry,+inspection: Bands.Inspection,covered: Covered<inspection,0n,Inspection.total_rows(geometry)>) ->
  {Bands.covers(geometry,inspection) == True{} : Bool}:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Bands.Inspection{+valid,+first,+end,+leaves} = inspection
  Covered{valid_ok,first_exact,end_exact} = covered
  +total = Nat.mul(U32.to_nat(batch),U32.to_nat(height))
  %Equal.sym(Bool,valid,True{},valid_ok) : {_ && U32.is_zero(first) && Nat.is_eq(U32.to_nat(end),total) == True{} : Bool}
  %Equal.sym(U32,first,0,Arithmetic.to_nat_injective(first,0,first_exact)) : {True{} && U32.is_zero(_) && Nat.is_eq(U32.to_nat(end),total) == True{} : Bool}
  %Equal.sym(Nat,U32.to_nat(end),total,end_exact) : {True{} && True{} && Nat.is_eq(_,total) == True{} : Bool}
  Order.equal_reflexive(total)

def accepted_ready(~Element: Data,ok: Bool,+operation: String,+reason: Tensors.Reason,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,
  +fill: Element,+workers: U32,ok_true: {ok == True{} : Bool}) ->
  {Contract.tensor_ready(~Element,Tensors.transformed_accepted(~Element,ok,operation,reason,bands,geometry,shape,fill,workers)) == True{} : Bool}:
  match ok:
    case True{}: {==}
    case False{}:
      Empty.absurd({Contract.tensor_ready(~Element,Tensors.transformed_accepted(~Element,False{},operation,reason,bands,geometry,shape,fill,workers)) == True{} : Bool},
        WordEncoding.false_true(ok_true))

def inspected_ready(~Element: Data,+usable: Bool,+operation: String,+reason: Tensors.Reason,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  inspected_pair: Bands.Bands(~Element) & Bands.Inspection,usable_ok: {usable == True{} : Bool},
  covered: Covered<inspected(~Element,inspected_pair),0n,Inspection.total_rows(geometry)>) ->
  {Contract.tensor_ready(~Element,Tensors.transformed_inspected(~Element,usable,operation,reason,geometry,shape,fill,workers,inspected_pair)) == True{} : Bool}:
  (bands,+inspection) = inspected_pair
  accepted_ready(~Element,usable && Bands.covers(geometry,inspection),operation,reason,bands,geometry,shape,fill,workers,
    %Equal.sym(Bool,usable,True{},usable_ok) : {_ && Bands.covers(geometry,inspection) == True{} : Bool} covers_true(geometry,inspection,covered))

# A checked transform whose calls were all usable and whose bands cover the
# geometry gives a Ready tensor.
def transformed_ready(~Element: Data,+operation: String,+reason: Tensors.Reason,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  result: Bands.Bands(~Element) & Bool,spans: Spans(~Element,geometry,result,0n,Inspection.total_rows(geometry))) ->
  {Contract.tensor_ready(~Element,Tensors.transformed(~Element,operation,reason,geometry,shape,fill,workers,result)) == True{} : Bool}:
  (bands,+usable) = result
  Shape.Both{usable_ok,covered} = spans
  inspected_ready(~Element,usable,operation,reason,geometry,shape,fill,workers,Bands.inspect(~Element,geometry,bands),usable_ok,covered)

def words_rows(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+offset: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  words: Values.Words(BandViews.Banded{before,repeats,after,offset},Bands.Geometry{batch,channels,height,width})) ->
  {U32.to_nat((height * BandViews.product_or_zero(repeats) : U32)) == Nat.mul(U32.to_nat(height),U32.to_nat(BandViews.product_or_zero(repeats))) : Nat}:
  Shape.Both{fitted,plane} = words
  Values.Fitted{batch_positive,height_positive,channel_reach,column_reach,early,late,repeated,rows_exact} = fitted
  rows_exact

# Bands whose images and rows the view keeps in place gather their own cells
# without failing.
def banded_ready(~Element: Data,+name: String,+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,
  +batch: U32,+channels: U32,+height: U32,+width: U32,+local: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,+model: Model.Model<Element>,
  +words: Values.Words(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),
  +limits: Machine.Limits,+default: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +room: BandGather.Domain<BandViews.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),limits,Machine.capacity(limits)>,
  well_formed: BandShape.WellFormed<Element,Bands.Geometry{batch,channels,height,width},Model.content(~Element,model)>) ->
  {Contract.tensor_ready(~Element,Tensors.gather_banded(~Element,name,BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},local,
    shape,target,fill,workers,Model.pack(~Element,model))) == True{} : Bool}:
  BandShape.WellFormed{metadata,ready,coverage} = well_formed
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +scope = {Scope{name,banded,geometry,local,fill,words,limits,default,room} : Scope<Element>}
  +context = {Tensors.Gathered{name,banded,geometry,local,fill} : Tensors.Gathered<Element>}
  +scale = BandViews.product_or_zero(repeats)
  +out_height = (height * scale : U32)
  +output = {Bands.Geometry{batch,BandViews.product_or_zero(Views.axis_extents(before)),out_height,BandViews.product_or_zero(Views.axis_extents(after))} : Bands.Geometry}
  +b = U32.to_nat(batch)
  +h = U32.to_nat(height)
  +s = U32.to_nat(scale)
  +schedule = Each.scheduled(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element),context,geometry,workers,12.0,0.0,model)
  +first_fact = Model.content_both(~Element,~Scope<Element>,~scope_fields(~Element),~Model.always(~Element,~Scope<Element>,~Model.ready_band(~Element)),scope,
    Model.content(~Element,model),metadata,ready)
  +spans = spans_tree(~Element,~Tensors.Gathered<Element>,~Bands.transform_band(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element)),~Scope<Element>,
    ~scope_gathered(~Element),~Input(~Element),~scope_output(~Element),~scope_scale(~Element),~leaf_spans(~Element),scope,model,0n,Nat.mul(b,h),
    %Equal.sym(Bands.Plan,Model.layout(~Element,model),Model.content_layout(~Element,Model.content(~Element,model)),Model.layout_matches_content(~Element,model)) :
      BandPlan.Coverage(_,0n,Nat.mul(b,h))
    coverage,first_fact)
  %Equal.sym(Bands.Bands(~Element) & Bool,BandSchedule.transform(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element),context,geometry,workers,12.0,0.0,Model.pack(~Element,model)),
    Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Tensors.Gathered<Element>,~Bands.transform_band(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element)),context,model)),schedule) :
    {Contract.tensor_ready(~Element,Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},output,target,fill,workers,_)) == True{} : Bool}
  transformed_ready(~Element,name,Tensors.InvalidArgument{shape},output,target,fill,workers,
    Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Tensors.Gathered<Element>,~Bands.transform_band(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element)),context,model)),
    %Equal.sym(Nat,U32.to_nat(out_height),Nat.mul(h,s),words_rows(before,repeats,after,start,batch,channels,height,width,words)) :
      Spans(~Element,output,Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Tensors.Gathered<Element>,~Bands.transform_band(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element)),context,model)),0n,Nat.mul(b,_))
    %RowMajor.assoc(b,h,s) : Spans(~Element,output,Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~Tensors.Gathered<Element>,~Bands.transform_band(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element)),context,model)),0n,_)
    spans)

# ---- A view of one image of one row ----

# The last coordinates of axes weigh their channels times the plane and
# their columns.
def last_weight_whole(+plane: U32,+divisor: Address.Divisor<plane>,+view_axes: List<&2,Views.Axis>,
  +nonempty: {Shapes.has_zero(Views.axis_extents(view_axes)) == False{} : Bool}) ->
  {last_weight(view_axes) == Nat.add(Nat.mul(U32.to_nat(plane),Address.span(plane,False{},view_axes)),Address.span(plane,True{},view_axes)) : Nat}:
  match view_axes:
    case []: Equal.sym(Nat,Nat.add(Nat.mul(U32.to_nat(plane),0n),0n),0n,Address.zero_sum(U32.to_nat(plane)))
    case Views.Axis{+extent,+stride} <> +tail:
      +positive = head_positive(extent,Views.axis_extents(tail),nonempty)
      +tail_nonempty = Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(tail)),nonempty)
      +digit = Nat.sub(U32.to_nat(extent),1n)
      +steps = U32.to_nat((extent - 1 : U32))
      +quotient = U32.to_nat(BandViews.component(plane,False{},stride))
      +leftover = U32.to_nat(BandViews.component(plane,True{},stride))
      +factor = U32.to_nat(plane)
      +others = Address.span(plane,False{},tail)
      +rest = Address.span(plane,True{},tail)
      %Equal.sym(Nat,steps,digit,last_step(extent,positive)) :
        {Nat.add(Nat.mul(digit,U32.to_nat(stride)),last_weight(tail)) ==
          Nat.add(Nat.mul(factor,Nat.add(Nat.mul(_,quotient),others)),Nat.add(Nat.mul(_,leftover),rest)) : Nat}
      %Equal.sym(Nat,U32.to_nat(stride),Nat.add(Nat.mul(quotient,factor),leftover),Address.split_word(plane,stride,divisor)) :
        {Nat.add(Nat.mul(digit,_),last_weight(tail)) == Nat.add(Nat.mul(factor,Nat.add(Nat.mul(digit,quotient),others)),Nat.add(Nat.mul(digit,leftover),rest)) : Nat}
      %Equal.sym(Nat,last_weight(tail),Nat.add(Nat.mul(factor,others),rest),last_weight_whole(plane,divisor,tail,tail_nonempty)) :
        {Nat.add(Nat.mul(digit,Nat.add(Nat.mul(quotient,factor),leftover)),_) ==
          Nat.add(Nat.mul(factor,Nat.add(Nat.mul(digit,quotient),others)),Nat.add(Nat.mul(digit,leftover),rest)) : Nat}
      Address.spread(factor,digit,quotient,leftover,others,rest)

# A reach of axes from a start that stays inside the words is accepted.
type Reach<-plane: U32,-column: Bool,-view_axes: List<&2,Views.Axis>,-start: U32> is Data:
  Reach{total: U32,found: {BandViews.reach(plane,column,view_axes,Some{start}) == Some{total} : Maybe<&2,U32>},
    exact: {U32.to_nat(total) == Nat.add(U32.to_nat(start),Address.span(plane,column,view_axes)) : Nat}}

def reach_step(+plane: U32,+column: Bool,+extent: U32,+stride: U32,+tail: List<&2,Views.Axis>,+start: U32,+product: U32,
  +multiplied: {Shapes.checked_multiply((extent - 1 : U32),BandViews.component(plane,column,stride)) == Some{product} : Maybe<&2,U32>},
  +valid: {U32.is_le(start,(4294967295 - product : U32)) == True{} : Bool},
  +product_exact: {U32.to_nat(product) == Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(BandViews.component(plane,column,stride))) : Nat},
  +sum_exact: {U32.to_nat((product + start : U32)) == Nat.add(U32.to_nat(product),U32.to_nat(start)) : Nat},
  next: Reach<plane,column,tail,(product + start : U32)>) -> Reach<plane,column,Views.Axis{extent,stride} <> tail,start>:
  Reach{+total,found,+exact} = next
  +own = Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(BandViews.component(plane,column,stride)))
  +rest = Address.span(plane,column,tail)
  +first = U32.to_nat(start)
  Reach{total,
    %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply((extent - 1 : U32),BandViews.component(plane,column,stride)),Some{product},multiplied) :
      {BandViews.reach(plane,column,tail,Shapes.checked_add_to(_,start)) == Some{total} : Maybe<&2,U32>}
    %Equal.sym(Bool,U32.is_le(start,(4294967295 - product : U32)),True{},valid) :
      {BandViews.reach(plane,column,tail,Shapes.sum_when(_,(product + start : U32))) == Some{total} : Maybe<&2,U32>}
    found,
    Equal.trans(Nat,U32.to_nat(total),Nat.add(Nat.add(own,first),rest),Nat.add(first,Nat.add(own,rest)),
      %product_exact : {U32.to_nat(total) == Nat.add(Nat.add(_,first),rest) : Nat}
      %sum_exact : {U32.to_nat(total) == Nat.add(_,rest) : Nat}
      exact,
      Equal.trans(Nat,Nat.add(Nat.add(own,first),rest),Nat.add(own,Nat.add(first,rest)),Nat.add(first,Nat.add(own,rest)),
        Addition.associative(own,first,rest),RowMajor.add_swap(own,first,rest)))}

def reach_accepted(+plane: U32,+column: Bool,+view_axes: List<&2,Views.Axis>,+start: U32,+largest: U32,+named: {largest == 4294967295 : U32},
  +room: {Nat.is_le(Nat.add(U32.to_nat(start),Address.span(plane,column,view_axes)),U32.to_nat(largest)) == True{} : Bool}) ->
  Reach<plane,column,view_axes,start>:
  match view_axes:
    case []: Reach{start,{==},Equal.sym(Nat,Nat.add(U32.to_nat(start),0n),U32.to_nat(start),Addition.zero_right(U32.to_nat(start)))}
    case Views.Axis{+extent,+stride} <> +tail:
      +word = BandViews.component(plane,column,stride)
      +own = Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(word))
      +rest = Address.span(plane,column,tail)
      +first = U32.to_nat(start)
      +bound = U32.to_nat(largest)
      +product = ((extent - 1 : U32) * word : U32)
      +turned = Equal.trans(Nat,Nat.add(Nat.add(own,first),rest),Nat.add(own,Nat.add(first,rest)),Nat.add(first,Nat.add(own,rest)),
        Addition.associative(own,first,rest),RowMajor.add_swap(own,first,rest))
      +grouped = {%Equal.sym(Nat,Nat.add(Nat.add(own,first),rest),Nat.add(first,Nat.add(own,rest)),turned) : {Nat.is_le(_,bound) == True{} : Bool} room : {Nat.is_le(Nat.add(Nat.add(own,first),rest),bound) == True{} : Bool}}
      +sum_room = Address.left_part(Nat.add(own,first),rest,bound,grouped)
      +own_room = Address.left_part(own,first,bound,sum_room)
      +product_exact = FullRange.multiply((extent - 1 : U32),word,largest,own_room)
      +multiplied = Completion.actual((extent - 1 : U32),word,Machine.word_max(),product,Equal.sym(Nat,U32.to_nat(product),own,product_exact),FullRange.maximum(product),{==})
      +sum_bound = {%Equal.sym(Nat,U32.to_nat(product),own,product_exact) : {Nat.is_le(Nat.add(_,first),bound) == True{} : Bool} sum_room :
        {Nat.is_le(Nat.add(U32.to_nat(product),first),bound) == True{} : Bool}}
      +sum_exact = FullRange.add(product,start,largest,sum_bound)
      +product_le = Address.left_part(U32.to_nat(product),first,bound,sum_bound)
      +difference = FullRange.subtract(largest,product,largest,product_le,Order.sub_le(bound,U32.to_nat(product)))
      +valid = {%named : {U32.is_le(start,(_ - product : U32)) == True{} : Bool}
        U32Division.nat_le_native(start,(largest - product : U32),
          %Equal.sym(Nat,U32.to_nat((largest - product : U32)),Nat.sub(bound,U32.to_nat(product)),difference) : {Nat.is_le(first,_) == True{} : Bool}
          Order.cancel_offset_le(U32.to_nat(product),first,Nat.sub(bound,U32.to_nat(product)),
            %Equal.sym(Nat,Nat.add(U32.to_nat(product),Nat.sub(bound,U32.to_nat(product))),bound,Order.restore_shift(U32.to_nat(product),bound,product_le)) :
              {Nat.is_le(Nat.add(U32.to_nat(product),first),_) == True{} : Bool}
            sum_bound)) :
        {U32.is_le(start,(4294967295 - product : U32)) == True{} : Bool}}
      +next_room = {%Equal.sym(Nat,U32.to_nat((product + start : U32)),Nat.add(U32.to_nat(product),first),sum_exact) : {Nat.is_le(Nat.add(_,rest),bound) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat(product),own,product_exact) : {Nat.is_le(Nat.add(Nat.add(_,first),rest),bound) == True{} : Bool}
        grouped : {Nat.is_le(Nat.add(U32.to_nat((product + start : U32)),rest),bound) == True{} : Bool}}
      reach_step(plane,column,extent,stride,tail,start,product,multiplied,valid,product_exact,sum_exact,
        reach_accepted(plane,column,tail,(product + start : U32),largest,named,next_room))

def divisor_positive(+plane: U32,divisor: Address.Divisor<plane>) -> {Nat.is_lt(0n,U32.to_nat(plane)) == True{} : Bool}:
  Address.Divisor{cap,cap_ok,bound,positive} = divisor
  positive

# An accepted reach below a limit passes the check of a classification.
def reach_below(+plane: U32,+column: Bool,+view_axes: List<&2,Views.Axis>,+start: U32,+limit: U32,reached: Reach<plane,column,view_axes,start>,
  below: {Nat.is_lt(Nat.add(U32.to_nat(start),Address.span(plane,column,view_axes)),U32.to_nat(limit)) == True{} : Bool}) ->
  {BandViews.below(BandViews.reach(plane,column,view_axes,Some{start}),limit) == True{} : Bool}:
  Reach{+total,found,exact} = reached
  %Equal.sym(Maybe<&2,U32>,BandViews.reach(plane,column,view_axes,Some{start}),Some{total},found) : {BandViews.below(_,limit) == True{} : Bool}
  U32Division.nat_lt_native(total,limit,
    %Equal.sym(Nat,U32.to_nat(total),Nat.add(U32.to_nat(start),Address.span(plane,column,view_axes)),exact) :
      {Nat.is_lt(_,U32.to_nat(limit)) == True{} : Bool}
    below)

# The address of the last coordinates, split into channels times the plane
# and columns.
def last_split(+factor: Nat,+channels: Nat,+columns: Nat,+quotient: Nat,+remainder: Nat) ->
  {Nat.add(Nat.add(Nat.mul(factor,channels),columns),Nat.add(Nat.mul(quotient,factor),remainder)) ==
    Nat.add(Nat.mul(factor,Nat.add(channels,quotient)),Nat.add(columns,remainder)) : Nat}:
  %RowMajor.multiply_commutative(factor,quotient) :
    {Nat.add(Nat.add(Nat.mul(factor,channels),columns),Nat.add(_,remainder)) == Nat.add(Nat.mul(factor,Nat.add(channels,quotient)),Nat.add(columns,remainder)) : Nat}
  Address.collect(factor,channels,columns,quotient,remainder)

# A view whose last address lies inside a geometry of one image of one row
# keeps it whole.
def whole_fits(+view_axes: List<&2,Views.Axis>,+offset: U32,+count: U32,+largest: U32,+named: {largest == 4294967295 : U32},+divisor: Address.Divisor<(1 * count : U32)>,
  +plane_exact: {U32.to_nat((1 * count : U32)) == U32.to_nat(count) : Nat},
  +nonempty: {Shapes.has_zero(Views.axis_extents(view_axes)) == False{} : Bool},
  +counted: {BandViews.counted(Views.axis_extents(view_axes)) == True{} : Bool},
  +bound: {Nat.is_lt(Coordinates.address(Lanes.points(view_axes,Lanes.lasts(view_axes)),U32.to_nat(offset),False{}),U32.to_nat(count)) == True{} : Bool}) ->
  {BandViews.fits(BandViews.Banded{[],[],view_axes,offset},Bands.Geometry{1,1,1,count}) == True{} : Bool}:
  +plane = (1 * count : U32)
  +factor = U32.to_nat(plane)
  +size = U32.to_nat(count)
  +channels = Address.span(plane,False{},view_axes)
  +columns = Address.span(plane,True{},view_axes)
  +quotient_word = BandViews.component(plane,False{},offset)
  +remainder_word = BandViews.component(plane,True{},offset)
  +quotient = U32.to_nat(quotient_word)
  +remainder = U32.to_nat(remainder_word)
  +reached = Equal.trans(Nat,Coordinates.address(Lanes.points(view_axes,Lanes.lasts(view_axes)),U32.to_nat(offset),False{}),
    Nat.add(last_weight(view_axes),U32.to_nat(offset)),Nat.add(Nat.mul(factor,Nat.add(channels,quotient)),Nat.add(columns,remainder)),
    Coordinates.address_weight(Lanes.points(view_axes,Lanes.lasts(view_axes)),U32.to_nat(offset)),
    %Equal.sym(Nat,last_weight(view_axes),Nat.add(Nat.mul(factor,channels),columns),last_weight_whole(plane,divisor,view_axes,nonempty)) :
      {Nat.add(_,U32.to_nat(offset)) == Nat.add(Nat.mul(factor,Nat.add(channels,quotient)),Nat.add(columns,remainder)) : Nat}
    %Equal.sym(Nat,U32.to_nat(offset),Nat.add(Nat.mul(quotient,factor),remainder),Address.split_word(plane,offset,divisor)) :
      {Nat.add(Nat.add(Nat.mul(factor,channels),columns),_) == Nat.add(Nat.mul(factor,Nat.add(channels,quotient)),Nat.add(columns,remainder)) : Nat}
    last_split(factor,channels,columns,quotient,remainder))
  +inside = {%Equal.sym(Nat,factor,size,plane_exact) : {Nat.is_lt(Nat.add(Nat.mul(factor,Nat.add(channels,quotient)),Nat.add(columns,remainder)),_) == True{} : Bool}
    %reached : {Nat.is_lt(_,size) == True{} : Bool} bound :
    {Nat.is_lt(Nat.add(Nat.mul(factor,Nat.add(channels,quotient)),Nat.add(columns,remainder)),factor) == True{} : Bool}}
  +scaled_below = Order.le_lt_trans(Nat.mul(factor,Nat.add(channels,quotient)),Nat.add(Nat.mul(factor,Nat.add(channels,quotient)),Nat.add(columns,remainder)),factor,
    Order.le_add(Nat.mul(factor,Nat.add(channels,quotient)),Nat.add(columns,remainder)),inside)
  +channel_zero = RowMajor.cancel_multiply_lt(Nat.add(channels,quotient),1n,factor,
    %Equal.sym(Nat,Nat.add(factor,0n),factor,Addition.zero_right(factor)) : {Nat.is_lt(Nat.mul(Nat.add(channels,quotient),factor),_) == True{} : Bool}
    %RowMajor.multiply_commutative(factor,Nat.add(channels,quotient)) : {Nat.is_lt(_,factor) == True{} : Bool} scaled_below)
  +column_inside = Order.le_lt_trans(Nat.add(columns,remainder),Nat.add(Nat.mul(factor,Nat.add(channels,quotient)),Nat.add(columns,remainder)),factor,
    %Order.commutative(Nat.add(columns,remainder),Nat.mul(factor,Nat.add(channels,quotient))) : {Nat.is_le(Nat.add(columns,remainder),_) == True{} : Bool}
    Order.le_add(Nat.add(columns,remainder),Nat.mul(factor,Nat.add(channels,quotient))),inside)
  +count_room = Bounds.as_nat_le(count,largest,%Equal.sym(U32,largest,4294967295,named) : {U32.is_le(count,_) == True{} : Bool} FullRange.maximum(count))
  +plane_room = {%Equal.sym(Nat,factor,size,plane_exact) : {Nat.is_le(_,U32.to_nat(largest)) == True{} : Bool} count_room : {Nat.is_le(factor,U32.to_nat(largest)) == True{} : Bool}}
  +channel_low = {%Order.commutative(channels,quotient) : {Nat.is_lt(_,1n) == True{} : Bool} channel_zero : {Nat.is_lt(Nat.add(quotient,channels),1n) == True{} : Bool}}
  +column_low = {%Order.commutative(columns,remainder) : {Nat.is_lt(_,factor) == True{} : Bool} column_inside : {Nat.is_lt(Nat.add(remainder,columns),factor) == True{} : Bool}}
  +channel_room = Order.transitive(Nat.add(quotient,channels),factor,U32.to_nat(largest),
    Order.transitive(Nat.add(quotient,channels),1n,factor,Bounds.strict_le(Nat.add(quotient,channels),1n,channel_low),Order.positive_one(factor,divisor_positive(plane,divisor))),
    plane_room)
  +column_room = Order.transitive(Nat.add(remainder,columns),factor,U32.to_nat(largest),Bounds.strict_le(Nat.add(remainder,columns),factor,column_low),plane_room)
  +early = BandViews.below(BandViews.reach(plane,False{},view_axes,Some{quotient_word}),1)
  +late = BandViews.below(BandViews.reach(plane,True{},view_axes,Some{remainder_word}),count)
  +nothing = {[] : List<&2,U32>}
  +unit = BandViews.product_or_zero(nothing)
  +early_ok = reach_below(plane,False{},view_axes,quotient_word,1,reach_accepted(plane,False{},view_axes,quotient_word,largest,named,channel_room),channel_low)
  +late_ok = reach_below(plane,True{},view_axes,remainder_word,count,reach_accepted(plane,True{},view_axes,remainder_word,largest,named,column_room),
    %plane_exact : {Nat.is_lt(Nat.add(remainder,columns),_) == True{} : Bool} column_low)
  +unit_product = Completion.actual(1,1,Machine.word_max(),1,{==},FullRange.maximum(1),{==})
  %Equal.sym(Bool,early,True{},early_ok) :
    {U32.is_lt(0,1) && U32.is_lt(0,1) && _ && late && BandViews.counted(nothing) && BandViews.counted(Views.axis_extents(view_axes)) && BandViews.counted(nothing) &&
      Views.has_count(Shapes.checked_multiply(1,unit)) == True{} : Bool}
  %Equal.sym(Bool,late,True{},late_ok) :
    {U32.is_lt(0,1) && U32.is_lt(0,1) && True{} && _ && BandViews.counted(nothing) && BandViews.counted(Views.axis_extents(view_axes)) && BandViews.counted(nothing) &&
      Views.has_count(Shapes.checked_multiply(1,unit)) == True{} : Bool}
  %Equal.sym(Bool,BandViews.counted(Views.axis_extents(view_axes)),True{},counted) :
    {U32.is_lt(0,1) && U32.is_lt(0,1) && True{} && True{} && BandViews.counted(nothing) && _ && BandViews.counted(nothing) &&
      Views.has_count(Shapes.checked_multiply(1,unit)) == True{} : Bool}
  %Equal.sym(Maybe<&2,U32>,Shapes.checked_multiply(1,unit),Some{1},unit_product) :
    {U32.is_lt(0,1) && U32.is_lt(0,1) && True{} && True{} && BandViews.counted(nothing) && True{} && BandViews.counted(nothing) &&
      Views.has_count(_) == True{} : Bool}
  {==}

# Such a view classifies whole.
def whole_classified(+view_axes: List<&2,Views.Axis>,+offset: U32,+count: U32,
  fits: {BandViews.fits(BandViews.Banded{[],[],view_axes,offset},Bands.Geometry{1,1,1,count}) == True{} : Bool}) ->
  {BandViews.classify(view_axes,offset,Bands.Geometry{1,1,1,count}) == Some{BandViews.Banded{[],[],view_axes,offset}} : Maybe<&2,BandViews.Banded>}:
  %Equal.sym(Bool,BandViews.fits(BandViews.Banded{[],[],view_axes,offset},Bands.Geometry{1,1,1,count}),True{},fits) :
    {BandViews.kept_when(_,BandViews.Banded{[],[],view_axes,offset}) == Some{BandViews.Banded{[],[],view_axes,offset}} : Maybe<&2,BandViews.Banded>}
  {==}

# A view whose last address lies inside the elements of a whole geometry
# classifies whole.
def whole_found(+view_axes: List<&2,Views.Axis>,+offset: U32,+count: U32,+plane: Values.Plane<1,1,1,count>,
  +nonempty: {Shapes.has_zero(Views.axis_extents(view_axes)) == False{} : Bool},
  +counted: {BandViews.counted(Views.axis_extents(view_axes)) == True{} : Bool},
  +bound: {Nat.is_lt(Coordinates.address(Lanes.points(view_axes,Lanes.lasts(view_axes)),U32.to_nat(offset),False{}),U32.to_nat(count)) == True{} : Bool}) ->
  {BandViews.classify(view_axes,offset,Bands.Geometry{1,1,1,count}) == Some{BandViews.Banded{[],[],view_axes,offset}} : Maybe<&2,BandViews.Banded>}:
  Values.Plane{+divisor,+plane_exact,+image_exact} = plane
  whole_classified(view_axes,offset,count,
    whole_fits(view_axes,offset,count,4294967295,{==},divisor,
      Equal.trans(Nat,U32.to_nat((1 * count : U32)),Nat.mul(U32.to_nat(1),U32.to_nat(count)),U32.to_nat(count),plane_exact,Addition.zero_right(U32.to_nat(count))),
      nonempty,counted,bound))

# Extents of a count with no zero have a product.
def axes_counted(+extents: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(extents) == U32.to_nat(count) : Nat},
  +nonempty: {Shapes.has_zero(extents) == False{} : Bool}) -> {BandViews.counted(extents) == True{} : Bool}:
  Equal.cong(Maybe<&2,U32>,Bool,Views.has_count,Shapes.product(extents,Some{1}),Some{count},
    %nonempty : {Shapes.count_nonempty(_,extents) == Some{count} : Maybe<&2,U32>} CountCompletion.numel(extents,count,exact))

# ---- The public gather ----

# A view classified on a geometry gathers band by band without failing.
def classified_ready(~Element: Data,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,+banded: BandViews.Banded,+geometry: Bands.Geometry,
  +local: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,+model: Model.Model<Element>,
  +classified: Classified.Classified<view_axes,offset,geometry,banded>,
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,counted: Global.Counted<shape>,coherent: Global.Coherent(shape,geometry),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat}) ->
  {Contract.tensor_ready(~Element,Tensors.gather_banded(~Element,name,banded,geometry,local,shape,target,fill,workers,Model.pack(~Element,model))) == True{} : Bool}:
  BandViews.Banded{+before,+repeats,+after,+start} = banded
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Classified.Classified{+axes_exact,+offset_exact,+fits} = classified
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +fitted = Values.fitted(before,repeats,after,start,batch,channels,height,width,fits)
  +whole = Classified.whole_axes(banded,geometry)
  +output_elements = BandGather.elements(BandViews.output(banded,geometry))
  +coherent_output = Equal.trans(Nat,ShapeLaws.elements(target),ShapeLaws.elements(Views.axis_extents(view_axes)),output_elements,same_elements,
    %Equal.sym(List<&2,Views.Axis>,view_axes,whole,axes_exact) : {ShapeLaws.elements(Views.axis_extents(_)) == output_elements : Nat}
    Values.whole_elements(before,repeats,after,start,batch,channels,height,width,fitted))
  banded_ready(~Element,name,before,repeats,after,start,batch,channels,height,width,local,shape,target,fill,workers,model,
    Values.words(before,repeats,after,start,batch,channels,height,width,shape,fitted,counted,coherent),Default.limits(),{==},
    Values.output_domain(target,count,counts,nonzero,batch,BandViews.product_or_zero(Views.axis_extents(before)),(height * BandViews.product_or_zero(repeats) : U32),
      BandViews.product_or_zero(Views.axis_extents(after)),coherent_output),well_formed)

# Arrived bands are read through their model.
def arrived_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +banded: BandViews.Banded,+whole: Bands.Geometry,+local: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,-bands: Bands.Bands(~Element),
  observation: Model.Observation<Element,bands>,arrived: Arrive.Arrived(~Element,~Logical,~original,logical,whole,Bands.unsplit(whole),bands),
  +classified: Classified.Classified<view_axes,offset,whole,banded>,counted: Global.Counted<shape>,coherent: Global.Coherent(shape,whole),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat}) ->
  {Contract.tensor_ready(~Element,Tensors.gather_banded(~Element,name,banded,whole,local,shape,target,fill,workers,bands)) == True{} : Bool}:
  Model.Observation{+source,+equation} = observation
  Shape.Both{same,Shape.Both{well_formed,values}} = arrived
  +contents = Model.observed_contents(~Element,source,bands,equation)
  %equation : {Contract.tensor_ready(~Element,Tensors.gather_banded(~Element,name,banded,whole,local,shape,target,fill,workers,_)) == True{} : Bool}
  classified_ready(~Element,name,view_axes,offset,banded,whole,local,shape,target,fill,workers,source,classified,
    %contents : BandShape.WellFormed<Element,whole,_> well_formed,counted,coherent,count,counts,nonzero,same_elements)

# Other bands arrive as the one band of the whole geometry, which classifies
# the view with no axis before its rows, so the flags of its local view are
# found.
def arrival_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,geometry),
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  +whole_found: {BandViews.classify(view_axes,offset,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)}) == Some{BandViews.Banded{[],[],view_axes,offset}} : Maybe<&2,BandViews.Banded>}) ->
  {Contract.tensor_ready(~Element,Tensors.gather_arrived(~Element,name,reversed,view_axes,offset,Model.pack(~Element,model),geometry,shape,target,fill,workers)) == True{} : Bool}:
  +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)} : Bands.Geometry}
  +banded = {BandViews.Banded{[],[],view_axes,offset} : BandViews.Banded}
  +arrival = WholeLeaf.whole_arrival(shape,counted)
  %Equal.sym(Maybe<&2,BandViews.Banded>,BandViews.classify(view_axes,offset,whole),Some{banded},whole_found) :
    {Contract.tensor_ready(~Element,Tensors.gather_whole(~Element,name,_,reversed,whole,shape,target,fill,workers,
      Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})) == True{} : Bool}
  arrived_ready(~Element,~Logical,~original,logical,name,view_axes,offset,banded,whole,BandViews.local_directions([],[],view_axes,reversed,reversed),shape,target,fill,workers,
    Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers}),
    Model.observe(~Element,Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers})),
    Arrive.arrived(~Element,~Logical,~original,logical,shape,whole,Bands.unsplit(whole),geometry,fill,Tensors.Usable{},workers,model,arrival,well_formed,coherent,values),
    %whole_found : Classified.Classification(view_axes,offset,whole,_) Classified.classify(view_axes,offset,whole),
    counted,Target.arrival_coherent(shape,whole,Bands.unsplit(whole),arrival),count,counts,nonzero,same_elements)

# A classified view gathers band by band when its local view has flags.
def local_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,found: Maybe<&2,List<&2,Bool>>,+logical: Logical,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+banded: BandViews.Banded,+model: Model.Model<Element>,+geometry: Bands.Geometry,
  +shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  +classified: Classified.Classified<view_axes,offset,geometry,banded>,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,geometry),
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  +whole_found: {BandViews.classify(view_axes,offset,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)}) == Some{BandViews.Banded{[],[],view_axes,offset}} : Maybe<&2,BandViews.Banded>}) ->
  {Contract.tensor_ready(~Element,Tensors.gather_local(~Element,name,banded,found,reversed,view_axes,offset,Model.pack(~Element,model),geometry,shape,target,fill,workers)) == True{} : Bool}:
  match found:
    case Some{+local}:
      classified_ready(~Element,name,view_axes,offset,banded,geometry,local,shape,target,fill,workers,model,classified,well_formed,counted,coherent,count,counts,nonzero,same_elements)
    case None{}:
      arrival_ready(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
        well_formed,counted,coherent,values,count,counts,nonzero,same_elements,whole_found)

# The view is classified by the geometry of the input first.
def found_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,found: Maybe<&2,BandViews.Banded>,+logical: Logical,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,
  +classification: Classified.Classification(view_axes,offset,geometry,found),
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,geometry),
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  +whole_found: {BandViews.classify(view_axes,offset,Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)}) == Some{BandViews.Banded{[],[],view_axes,offset}} : Maybe<&2,BandViews.Banded>}) ->
  {Contract.tensor_ready(~Element,Tensors.gather_found(~Element,name,found,reversed,view_axes,offset,Model.pack(~Element,model),geometry,shape,target,fill,workers)) == True{} : Bool}:
  match found:
    case Some{+banded}:
      local_ready(~Element,~Logical,~original,BandViews.local_flags(banded,geometry,reversed),logical,name,view_axes,offset,reversed,banded,model,geometry,shape,target,fill,workers,
        classification,well_formed,counted,coherent,values,count,counts,nonzero,same_elements,whole_found)
    case None{}:
      arrival_ready(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
        well_formed,counted,coherent,values,count,counts,nonzero,same_elements,whole_found)

# A target with no elements is an empty whole result; otherwise the view
# classifies whole.
def counted_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,empty: Bool,+logical: Logical,+name: String,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<shape>,+coherent: Global.Coherent(shape,geometry),
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(target) == Some{count} : Maybe<&2,U32>},+compared: {U32.is_zero(count) == empty : Bool},
  +same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  bound: {Shapes.has_zero(target) == False{} : Bool} ->
    {Nat.is_lt(Coordinates.address(Lanes.points(view_axes,Lanes.lasts(view_axes)),U32.to_nat(offset),False{}),ShapeLaws.elements(shape)) == True{} : Bool}) ->
  {Contract.tensor_ready(~Element,Tensors.gather_counted(~Element,name,empty,view_axes,offset,reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers)) == True{} : Bool}:
  match empty:
    case True{}: {==}
    case False{}:
      Global.Counted{+input_count,+input_counts} = counted
      +counted = {Global.Counted{input_count,input_counts} : Global.Counted<shape>}
      +width = Tensors.count_or_zero(shape)
      +address = Coordinates.address(Lanes.points(view_axes,Lanes.lasts(view_axes)),U32.to_nat(offset),False{})
      +width_exact = Equal.trans(Nat,ShapeLaws.elements(shape),U32.to_nat(input_count),U32.to_nat(width),Global.whole_counted(shape,input_count,input_counts),
        Equal.cong(U32,Nat,U32.to_nat,input_count,width,Equal.sym(U32,width,input_count,Global.whole_width(shape,input_count,input_counts))))
      +target_exact = Values.count_elements(target,count,Count.count_of(target,count,counts))
      +target_positive = {%Equal.sym(Nat,ShapeLaws.elements(target),U32.to_nat(count),target_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
        Arithmetic.word_nonzero_positive(count,compared) : {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool}}
      +inside = {%width_exact : {Nat.is_lt(address,_) == True{} : Bool} bound(CountCompletion.positive_not_zero(target,target_positive)) :
        {Nat.is_lt(address,U32.to_nat(width)) == True{} : Bool}}
      +whole = {Bands.Geometry{1,1,1,width} : Bands.Geometry}
      +arrival = WholeLeaf.whole_arrival(shape,counted)
      +domain = Global.domain(shape,whole,counted,Target.arrival_coherent(shape,whole,Bands.unsplit(whole),arrival),{==},{==},
        Order.le_lt_trans(0n,address,U32.to_nat(width),Order.zero_le(address),inside))
      +axes_exact = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(view_axes)),ShapeLaws.elements(target),U32.to_nat(count),
        Equal.sym(Nat,ShapeLaws.elements(target),ShapeLaws.elements(Views.axis_extents(view_axes)),same_elements),target_exact)
      +nonempty = CountCompletion.positive_not_zero(Views.axis_extents(view_axes),
        %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(view_axes)),U32.to_nat(count),axes_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
        Arithmetic.word_nonzero_positive(count,compared))
      found_ready(~Element,~Logical,~original,BandViews.classify(view_axes,offset,geometry),logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
        Classified.classify(view_axes,offset,geometry),well_formed,counted,coherent,values,count,counts,compared,same_elements,
        whole_found(view_axes,offset,width,Values.plane_words(1,1,1,width,Default.limits(),{==},domain),nonempty,
          axes_counted(Views.axis_extents(view_axes),count,axes_exact,nonempty),inside))

def model_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers}),
  counted: Global.Counted<target>,+same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  bound: {Shapes.has_zero(target) == False{} : Bool} ->
    {Nat.is_lt(Coordinates.address(Lanes.points(view_axes,Lanes.lasts(view_axes)),U32.to_nat(offset),False{}),ShapeLaws.elements(shape)) == True{} : Bool}) ->
  {Contract.tensor_ready(~Element,Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,Model.pack(~Element,model),geometry,shape,target,
    fill,workers)) == True{} : Bool}:
  Shape.Both{Shape.Both{well_formed,Shape.Both{+input_counted,+coherent}},values} = contract
  Global.Counted{+count,+counts} = counted
  +contents = Model.packed_contents(~Element,model)
  %Equal.sym(Maybe<&2,U32>,Tensors.count_of(target),Some{count},counts) :
    {Contract.tensor_ready(~Element,Tensors.gather_axes(~Element,name,_,view_axes,offset,reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers)) == True{} : Bool}
  counted_ready(~Element,~Logical,~original,U32.is_zero(count),logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
    %contents : BandShape.WellFormed<Element,geometry,_> well_formed,input_counted,coherent,
    %contents : BandValues.Values(~Element,~Logical,~original,logical,geometry,_) values,count,counts,{==},same_elements,bound)

def observed_ready(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,-bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  observation: Model.Observation<Element,bands>,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  counted: Global.Counted<target>,+same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  bound: {Shapes.has_zero(target) == False{} : Bool} ->
    {Nat.is_lt(Coordinates.address(Lanes.points(view_axes,Lanes.lasts(view_axes)),U32.to_nat(offset),False{}),ShapeLaws.elements(shape)) == True{} : Bool}) ->
  {Contract.tensor_ready(~Element,Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,bands,geometry,shape,target,fill,workers)) == True{} : Bool}:
  Model.Observation{+model,+equation} = observation
  %equation : {Contract.tensor_ready(~Element,Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,_,geometry,shape,target,fill,workers)) == True{} : Bool}
  model_ready(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,model,geometry,shape,target,fill,workers,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) :
      Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{_,geometry,shape,fill,Tensors.Usable{},workers}) contract,
    counted,same_elements,bound)

# The public gather of a Ready tensor of the contract through a view whose
# elements are those of a counted target and whose last address lies inside
# the input, read with any flags, never fails.
def gather(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers}),
  counted: Global.Counted<target>,+same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  bound: {Shapes.has_zero(target) == False{} : Bool} ->
    {Nat.is_lt(Coordinates.address(Lanes.points(view_axes,Lanes.lasts(view_axes)),U32.to_nat(offset),False{}),ShapeLaws.elements(shape)) == True{} : Bool}) ->
  {Contract.tensor_ready(~Element,Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,bands,geometry,shape,target,fill,workers)) == True{} : Bool}:
  observed_ready(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,bands,geometry,shape,target,fill,workers,Model.observe(~Element,bands),contract,counted,
    same_elements,bound)

# ---- The extents of a gather ----

def accepted_extents(~Element: Data,ok: Bool,+operation: String,+reason: Tensors.Reason,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,
  +fill: Element,+workers: U32) ->
  {Contract.extents(~Element,Tensors.transformed_accepted(~Element,ok,operation,reason,bands,geometry,shape,fill,workers)) == shape : List<&2,U32>}:
  match ok:
    case True{}: {==}
    case False{}: {==}

def inspected_extents(~Element: Data,+usable: Bool,+operation: String,+reason: Tensors.Reason,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  inspected_pair: Bands.Bands(~Element) & Bands.Inspection) ->
  {Contract.extents(~Element,Tensors.transformed_inspected(~Element,usable,operation,reason,geometry,shape,fill,workers,inspected_pair)) == shape : List<&2,U32>}:
  (bands,+inspection) = inspected_pair
  accepted_extents(~Element,usable && Bands.covers(geometry,inspection),operation,reason,bands,geometry,shape,fill,workers)

# A checked transform is labelled with its shape, whether or not it was
# accepted.
def transformed_extents(~Element: Data,+operation: String,+reason: Tensors.Reason,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  result: Bands.Bands(~Element) & Bool) ->
  {Contract.extents(~Element,Tensors.transformed(~Element,operation,reason,geometry,shape,fill,workers,result)) == shape : List<&2,U32>}:
  (bands,+usable) = result
  inspected_extents(~Element,usable,operation,reason,geometry,shape,fill,workers,Bands.inspect(~Element,geometry,bands))

def banded_extents(~Element: Data,+name: String,+banded: BandViews.Banded,+geometry: Bands.Geometry,+local: List<&2,Bool>,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,bands: Bands.Bands(~Element)) ->
  {Contract.extents(~Element,Tensors.gather_banded(~Element,name,banded,geometry,local,shape,target,fill,workers,bands)) == target : List<&2,U32>}:
  transformed_extents(~Element,name,Tensors.InvalidArgument{shape},BandViews.output(banded,geometry),target,fill,workers,
    BandSchedule.transform(~Element,~Tensors.Gathered<Element>,~Tensors.gathered_band(~Element),Tensors.Gathered{name,banded,geometry,local,fill},geometry,workers,12.0,0.0,bands))

def whole_local_extents(~Element: Data,+name: String,+banded: BandViews.Banded,found: Maybe<&2,List<&2,Bool>>,+whole: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,tensor: Tensors.Tensor<Element>) ->
  {Contract.extents(~Element,Tensors.gather_whole_local(~Element,name,banded,found,whole,shape,target,fill,workers,tensor)) == target : List<&2,U32>}:
  match found:
    case Some{+local}: banded_extents(~Element,name,banded,whole,local,shape,target,fill,workers,Tensors.arrive(~Element,whole,Bands.unsplit(whole),tensor))
    case None{}: {==}

def whole_extents(~Element: Data,+name: String,found: Maybe<&2,BandViews.Banded>,+reversed: List<&2,Bool>,+whole: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,
  +fill: Element,+workers: U32,tensor: Tensors.Tensor<Element>) ->
  {Contract.extents(~Element,Tensors.gather_whole(~Element,name,found,reversed,whole,shape,target,fill,workers,tensor)) == target : List<&2,U32>}:
  match found:
    case Some{+banded}: whole_local_extents(~Element,name,banded,BandViews.local_flags(banded,whole,reversed),whole,shape,target,fill,workers,tensor)
    case None{}: {==}

def arrival_extents(~Element: Data,+name: String,+reversed: List<&2,Bool>,+view_axes: List<&2,Views.Axis>,+offset: U32,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Contract.extents(~Element,Tensors.gather_arrived(~Element,name,reversed,view_axes,offset,bands,geometry,shape,target,fill,workers)) == target : List<&2,U32>}:
  +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(shape)} : Bands.Geometry}
  whole_extents(~Element,name,BandViews.classify(view_axes,offset,whole),reversed,whole,shape,target,fill,workers,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Usable{},workers})

def found_local_extents(~Element: Data,+name: String,+banded: BandViews.Banded,found: Maybe<&2,List<&2,Bool>>,+reversed: List<&2,Bool>,+view_axes: List<&2,Views.Axis>,
  +offset: U32,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Contract.extents(~Element,Tensors.gather_local(~Element,name,banded,found,reversed,view_axes,offset,bands,geometry,shape,target,fill,workers)) == target : List<&2,U32>}:
  match found:
    case Some{+local}: banded_extents(~Element,name,banded,geometry,local,shape,target,fill,workers,bands)
    case None{}: arrival_extents(~Element,name,reversed,view_axes,offset,bands,geometry,shape,target,fill,workers)

def found_extents(~Element: Data,+name: String,found: Maybe<&2,BandViews.Banded>,+reversed: List<&2,Bool>,+view_axes: List<&2,Views.Axis>,+offset: U32,
  bands: Bands.Bands(~Element),+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Contract.extents(~Element,Tensors.gather_found(~Element,name,found,reversed,view_axes,offset,bands,geometry,shape,target,fill,workers)) == target : List<&2,U32>}:
  match found:
    case Some{+banded}:
      found_local_extents(~Element,name,banded,BandViews.local_flags(banded,geometry,reversed),reversed,view_axes,offset,bands,geometry,shape,target,fill,workers)
    case None{}: arrival_extents(~Element,name,reversed,view_axes,offset,bands,geometry,shape,target,fill,workers)

def counted_extents(~Element: Data,+name: String,empty: Bool,+view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Contract.extents(~Element,Tensors.gather_counted(~Element,name,empty,view_axes,offset,reversed,bands,geometry,shape,target,fill,workers)) == target : List<&2,U32>}:
  match empty:
    case True{}: {==}
    case False{}: found_extents(~Element,name,BandViews.classify(view_axes,offset,geometry),reversed,view_axes,offset,bands,geometry,shape,target,fill,workers)

def axes_extents(~Element: Data,+name: String,count: Maybe<&2,U32>,+view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Contract.extents(~Element,Tensors.gather_axes(~Element,name,count,view_axes,offset,reversed,bands,geometry,shape,target,fill,workers)) == target : List<&2,U32>}:
  match count:
    case Some{+count}: counted_extents(~Element,name,U32.is_zero(count),view_axes,offset,reversed,bands,geometry,shape,target,fill,workers)
    case None{}: {==}

# A gather is labelled with its target, whether or not it was accepted.
def gather_extents(~Element: Data,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,+reversed: List<&2,Bool>,bands: Bands.Bands(~Element),+geometry: Bands.Geometry,
  +shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Contract.extents(~Element,Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,bands,geometry,shape,target,fill,workers)) == target : List<&2,U32>}:
  axes_extents(~Element,name,Tensors.count_of(target),view_axes,offset,reversed,bands,geometry,shape,target,fill,workers)

# The public gather of a tensor of the contract through a view whose elements
# are those of a counted target and whose last address, when the target has
# elements, lies inside the input: a Ready tensor of the target that holds at
# every index the input at the address the view, read with its flags, gives
# to that index.
def shaped(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+name: String,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +reversed: List<&2,Bool>,+model: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+target: List<&2,U32>,+fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,model),geometry,shape,fill,Tensors.Usable{},workers}),
  counted: Global.Counted<target>,+same_elements: {ShapeLaws.elements(target) == ShapeLaws.elements(Views.axis_extents(view_axes)) : Nat},
  bound: {Shapes.has_zero(target) == False{} : Bool} ->
    {Nat.is_lt(Coordinates.address(Lanes.points(view_axes,Lanes.lasts(view_axes)),U32.to_nat(offset),False{}),ShapeLaws.elements(shape)) == True{} : Bool}) ->
  Global.Shaped(~Element,~GatherValues.Directed<Logical>,~GatherValues.directed(~Element,~Logical,~original),GatherValues.Directed{logical,Views.View{view_axes,offset},reversed},target,
    Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers)):
  +accepted = gather(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers,contract,counted,
    same_elements,bound)
  Global.shaped(~Element,~GatherValues.Directed<Logical>,~GatherValues.directed(~Element,~Logical,~original),GatherValues.Directed{logical,Views.View{view_axes,offset},reversed},target,
    Tensors.gather(~Element,name,Tensors.Usable{},Some{Views.View{view_axes,offset}},reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers),accepted,
    gather_extents(~Element,name,view_axes,offset,reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers),
    Values.gather(~Element,~Logical,~original,logical,name,view_axes,offset,reversed,Model.pack(~Element,model),geometry,shape,target,fill,workers,contract,same_elements,accepted))
