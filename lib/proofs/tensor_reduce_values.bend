# The values of a reduction. Output k folds, from the initial value and in
# increasing position, the length values of its source from k * length on
# and holds finish of the fold. A reduction over listed axes reads the input
# through the view that puts its kept axes first and its reduced axes last;
# each band of the input folds the lines of its own rows through its local
# view, or the input arrives as one band first.
import Base
import ../tensor.bend as Tensors
import ../tensor_band.bend as Bands
import ../tensor_band_view.bend as BandViews
import ../tensor_band_schedule.bend as BandSchedule
import ../traversal_array.bend as Traversal
import ../traversal_loop.bend as Loop
import ../traversal_partition.bend as Partition
import ../tensor_access.bend as Access
import ../tensor_shape.bend as Shapes
import ./storage_buffer_model.bend as Buffer
import ./storage_tree_model.bend as Tree
import ./storage_shape.bend as Shape
import ./tensor_copy_values.bend as CopyValues
import ./tensor_line_values.bend as LineValues
import ./tensor_access_refinement.bend as AccessLaws
import ./tensor_cursor_refinement.bend as CursorLaws
import ./traversal_visit_refinement.bend as Visit
import ./consecutive_writes.bend as Consecutive
import ./write_program.bend as Writes
import ./sequence_membership.bend as Members
import ./ordered_ranges.bend as Ranges
import ./tensor_band_values.bend as BandValues
import ./natural_addition.bend as Addition
import ./nat_order.bend as Order
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_full_range.bend as FullRange
import ./row_major_index.bend as RowMajor
import ../tensor_operations.bend as Operations
import ../storage_buffer.bend as Storage
import ../tensor_band_reduce.bend as BandReduce
import ./storage_certificate.bend as Certificate
import ./storage_refinement.bend as StorageLaws
import ./tensor_gather_values.bend as GatherValues
import ./tensor_operations_refinement.bend as OperationsLaws
import ./tensor_band_model.bend as Model
import ./tensor_band_shape_refinement.bend as BandShape
import ./tensor_global_shape_count.bend as Global
import ./word_encoding.bend as WordEncoding
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_band_each.bend as Each
import ./tensor_band_inspection.bend as Inspection
import ./tensor_contract.bend as Contract
import ./tensor_allocation_depth.bend as Depth
import ./tensor_band_count.bend as BandCount
import ./storage_allocation_refinement.bend as AllocationLaws
import ./u32_division.bend as NativeDivision
import ./encoded_coordinates.bend as Encoding
import ../default_machine_limits.bend as Default
import ./tensor_band_gather.bend as BandGather
import ./tensor_band_arrive.bend as Arrive
import ./tensor_band_target.bend as Target
import ./tensor_whole_leaf.bend as WholeLeaf
import ./tensor_full_values.bend as Full
import ./tensor_count_refinement.bend as Count
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_shape_refinement.bend as ShapeLaws
import ../tensor_view.bend as Views
import ../tensor_layout.bend as Layout
import ./tensor_view_metadata_refinement.bend as Metadata
import ./tensor_band_view_values.bend as ViewValues
import ./tensor_band_view_classified.bend as Classified
import ./tensor_band_view_ready.bend as ViewReady
import ./tensor_band_read.bend as Read
import ./tensor_canonical_view.bend as Canonical
import ./tensor_band_view_address.bend as Address
import ./list_storage_readback.bend as ListReadback
import ./boolean_logic.bend as Boolean
import ../tensor_f32.bend as F
import ../math_sum.bend as Sum
import ./tensor_band_map.bend as BandMap
import ./tensor_band_normal_unary.bend as NormalUnary
import ./tensor_combine_public.bend as CombinePublic

# ---- What a reduction holds ----

type Lines<-Accumulator: Data,-Context: Data> is Data:
  Lines{initial: Accumulator,context: Context,length: Nat}

# The fold of the length values of the reader from index * length on.
def folded(~Element: Data,~Accumulator: Data,~Context: Data,~reader: @+context: Context -> @+index: Nat -> Element,~operation: Accumulator -> Element -> Accumulator,
  lines: Lines<Accumulator,Context>,index: U32) -> Accumulator:
  Lines{initial,+context,+length} = lines
  LineValues.line_fold(~Context,~Element,~Accumulator,~reader,~operation,length,context,Nat.mul(U32.to_nat(index),length),initial)

def value(~Element: Data,~Accumulator: Data,~Context: Data,~reader: @+context: Context -> @+index: Nat -> Element,~operation: Accumulator -> Element -> Accumulator,
  ~finish: Accumulator -> Element,lines: Lines<Accumulator,Context>,index: U32) -> Element:
  finish(folded(~Element,~Accumulator,~Context,~reader,~operation,lines,index))

# The input in an order of its axes: through a view, or in place.
type Ordered<-Logical: Data> is Data:
  Ordered{logical: Logical,view: Maybe<&2,Views.View>}

def through(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,view: Maybe<&2,Views.View>,+index: Nat) -> Element:
  match view:
    case Some{view}: original(logical,CopyValues.address(view,index))
    case None{}: original(logical,U32.from_nat(index))

def ordered_value(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+context: Ordered<Logical>,+index: Nat) -> Element:
  Ordered{+logical,view} = context
  through(~Element,~Logical,~original,logical,view,index)

# ---- The kept axes and the line of a banded view ----

def sub_after(+left: Nat,+right: Nat,+removed: Nat,+inside: {Nat.is_le(removed,right) == True{} : Bool}) ->
  {Nat.sub(Nat.add(left,right),removed) == Nat.add(left,Nat.sub(right,removed)) : Nat}:
  +rest = Nat.sub(right,removed)
  %Order.restore_shift(removed,right,inside) : {Nat.sub(Nat.add(left,_),removed) == Nat.add(left,rest) : Nat}
  %Addition.associative(left,removed,rest) :
    {Nat.sub(_,removed) == Nat.add(left,rest) : Nat}
  %Order.commutative(removed,left) : {Nat.sub(Nat.add(_,rest),removed) == Nat.add(left,rest) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(removed,left),rest),Nat.add(removed,Nat.add(left,rest)),Addition.associative(removed,left,rest)) : {Nat.sub(_,removed) == Nat.add(left,rest) : Nat}
  Order.subtract_shift(removed,Nat.add(left,rest))

def take_after(~Element: Data,+left: List<&2,Element>,+right: List<&2,Element>,+count: Nat) ->
  {List.take(&2,Element,List.append(&2,Element,left,right),Nat.add(List.length(&2,Element,left),count)) == List.append(&2,Element,left,List.take(&2,Element,right,count)) : List<&2,Element>}:
  match left:
    case []: {==}
    case +head <> +tail:
      Equal.cong(List<&2,Element>,List<&2,Element>,others => head <> others,
        List.take(&2,Element,List.append(&2,Element,tail,right),Nat.add(List.length(&2,Element,tail),count)),List.append(&2,Element,tail,List.take(&2,Element,right,count)),
        take_after(~Element,tail,right,count))

def drop_after(~Element: Data,+left: List<&2,Element>,+right: List<&2,Element>,+count: Nat) ->
  {List.drop(&2,Element,List.append(&2,Element,left,right),Nat.add(List.length(&2,Element,left),count)) == List.drop(&2,Element,right,count) : List<&2,Element>}:
  match left:
    case []: {==}
    case +head <> +tail: drop_after(~Element,tail,right,count)

def extents_taken(+axes: List<&2,Views.Axis>,+count: Nat) -> {Views.axis_extents(List.take(&2,Views.Axis,axes,count)) == List.take(&2,U32,Views.axis_extents(axes),count) : List<&2,U32>}:
  match axes count:
    case [] count: {==}
    case Views.Axis{extent,stride} <> tail 0n: {==}
    case Views.Axis{+extent,+stride} <> +tail 1n+ +previous:
      Equal.cong(List<&2,U32>,List<&2,U32>,others => extent <> others,Views.axis_extents(List.take(&2,Views.Axis,tail,previous)),List.take(&2,U32,Views.axis_extents(tail),previous),
        extents_taken(tail,previous))

def extents_dropped(+axes: List<&2,Views.Axis>,+count: Nat) -> {Views.axis_extents(List.drop(&2,Views.Axis,axes,count)) == List.drop(&2,U32,Views.axis_extents(axes),count) : List<&2,U32>}:
  match axes count:
    case [] count: {==}
    case Views.Axis{extent,stride} <> tail 0n: {==}
    case Views.Axis{extent,stride} <> +tail 1n+ +previous: extents_dropped(tail,previous)

# The axes before the line, a part of the axes after the rows, and the line.
def kept_part(+after: List<&2,Views.Axis>,+reduced: Nat) -> List<&2,Views.Axis>:
  List.take(&2,Views.Axis,after,Nat.sub(List.length(&2,Views.Axis,after),reduced))

def line_part(+after: List<&2,Views.Axis>,+reduced: Nat) -> List<&2,Views.Axis>:
  List.drop(&2,Views.Axis,after,Nat.sub(List.length(&2,Views.Axis,after),reduced))

def kept_elements(+after: List<&2,Views.Axis>,+reduced: Nat) -> Nat:
  ShapeLaws.elements(Views.axis_extents(kept_part(after,reduced)))

def line_elements(+after: List<&2,Views.Axis>,+reduced: Nat) -> Nat:
  ShapeLaws.elements(Views.axis_extents(line_part(after,reduced)))

# The axes after the rows hold the kept part times the line.
def after_split(+after: List<&2,Views.Axis>,+reduced: Nat) ->
  {ShapeLaws.elements(Views.axis_extents(after)) == Nat.mul(kept_elements(after,reduced),line_elements(after,reduced)) : Nat}:
  +taken = Nat.sub(List.length(&2,Views.Axis,after),reduced)
  %Equal.sym(List<&2,U32>,Views.axis_extents(List.take(&2,Views.Axis,after,taken)),List.take(&2,U32,Views.axis_extents(after),taken),extents_taken(after,taken)) :
    {ShapeLaws.elements(Views.axis_extents(after)) == Nat.mul(ShapeLaws.elements(_),line_elements(after,reduced)) : Nat}
  %Equal.sym(List<&2,U32>,Views.axis_extents(List.drop(&2,Views.Axis,after,taken)),List.drop(&2,U32,Views.axis_extents(after),taken),extents_dropped(after,taken)) :
    {ShapeLaws.elements(Views.axis_extents(after)) == Nat.mul(ShapeLaws.elements(List.take(&2,U32,Views.axis_extents(after),taken)),ShapeLaws.elements(_)) : Nat}
  CountCompletion.elements_split(Views.axis_extents(after),taken)

# The local axes of a band: the localized axes before the rows, the rows and
# their repeats, then the localized axes after the rows.
def front(+banded: BandViews.Banded,+geometry: Bands.Geometry,+rows: U32) -> List<&2,Views.Axis>:
  BandViews.Banded{before,+repeats,after,offset} = banded
  Bands.Geometry{batch,channels,+height,+width} = geometry
  List.append(&2,Views.Axis,BandViews.localized_axes((height * width : U32),(rows * width : U32),before),BandViews.rows_axes(True{},rows,width,repeats))

def back(+banded: BandViews.Banded,+geometry: Bands.Geometry,+rows: U32) -> List<&2,Views.Axis>:
  BandViews.Banded{before,repeats,+after,offset} = banded
  Bands.Geometry{batch,channels,+height,+width} = geometry
  BandViews.localized_axes((height * width : U32),(rows * width : U32),after)

def local_axes(+banded: BandViews.Banded,+geometry: Bands.Geometry,+rows: U32) -> List<&2,Views.Axis>:
  BandViews.axes(banded,geometry,rows)

def front_back(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+geometry: Bands.Geometry,+rows: U32) ->
  {local_axes(BandViews.Banded{before,repeats,after,start},geometry,rows) ==
    List.append(&2,Views.Axis,front(BandViews.Banded{before,repeats,after,start},geometry,rows),back(BandViews.Banded{before,repeats,after,start},geometry,rows)) : List<&2,Views.Axis>}:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Equal.sym(List<&2,Views.Axis>,
    List.append(&2,Views.Axis,front(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows),
      back(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows)),
    local_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows),
    LineValues.append_associative(~Views.Axis,BandViews.localized_axes((height * width : U32),(rows * width : U32),before),BandViews.rows_axes(True{},rows,width,repeats),
      BandViews.localized_axes((height * width : U32),(rows * width : U32),after)))

# The local count of axes kept is the front and the kept part of the back.
def local_kept_count(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+geometry: Bands.Geometry,+rows: U32,+reduced: Nat,
  +within: {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == True{} : Bool}) ->
  {Nat.sub(List.length(&2,Views.Axis,local_axes(BandViews.Banded{before,repeats,after,start},geometry,rows)),reduced) ==
    Nat.add(List.length(&2,Views.Axis,front(BandViews.Banded{before,repeats,after,start},geometry,rows)),Nat.sub(List.length(&2,Views.Axis,after),reduced)) : Nat}:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +head = front(banded,geometry,rows)
  +tail = back(banded,geometry,rows)
  +tail_length = Address.localized_length((height * width : U32),(rows * width : U32),after)
  %Equal.sym(List<&2,Views.Axis>,local_axes(banded,geometry,rows),List.append(&2,Views.Axis,head,tail),front_back(before,repeats,after,start,geometry,rows)) :
    {Nat.sub(List.length(&2,Views.Axis,_),reduced) == Nat.add(List.length(&2,Views.Axis,head),Nat.sub(List.length(&2,Views.Axis,after),reduced)) : Nat}
  %Equal.sym(Nat,List.length(&2,Views.Axis,List.append(&2,Views.Axis,head,tail)),Nat.add(List.length(&2,Views.Axis,head),List.length(&2,Views.Axis,tail)),
      ListReadback.append_length(~Views.Axis,head,tail)) :
    {Nat.sub(_,reduced) == Nat.add(List.length(&2,Views.Axis,head),Nat.sub(List.length(&2,Views.Axis,after),reduced)) : Nat}
  %Equal.sym(Nat,List.length(&2,Views.Axis,tail),List.length(&2,Views.Axis,after),tail_length) : {Nat.sub(Nat.add(List.length(&2,Views.Axis,head),_),reduced) == Nat.add(List.length(&2,Views.Axis,head),Nat.sub(List.length(&2,Views.Axis,after),reduced)) : Nat}
  sub_after(List.length(&2,Views.Axis,head),List.length(&2,Views.Axis,after),reduced,within)

def local_kept(+banded: BandViews.Banded,+geometry: Bands.Geometry,+rows: U32,+reduced: Nat) -> List<&2,Views.Axis>:
  BandReduce.kept_axes(local_axes(banded,geometry,rows),reduced)

def local_line(+banded: BandViews.Banded,+geometry: Bands.Geometry,+rows: U32,+reduced: Nat) -> List<&2,Views.Axis>:
  OperationsLaws.line_axes(local_axes(banded,geometry,rows),reduced)

# The local line is the line of the localized axes after the rows: the
# extents of the line.
def local_line_elements(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+geometry: Bands.Geometry,+rows: U32,+reduced: Nat,
  +within: {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == True{} : Bool}) ->
  {ShapeLaws.elements(Views.axis_extents(local_line(BandViews.Banded{before,repeats,after,start},geometry,rows,reduced))) == line_elements(after,reduced) : Nat}:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +head = front(banded,geometry,rows)
  +tail = back(banded,geometry,rows)
  +taken = Nat.sub(List.length(&2,Views.Axis,after),reduced)
  %Equal.sym(Nat,Nat.sub(List.length(&2,Views.Axis,local_axes(banded,geometry,rows)),reduced),Nat.add(List.length(&2,Views.Axis,head),taken),
      local_kept_count(before,repeats,after,start,geometry,rows,reduced,within)) :
    {ShapeLaws.elements(Views.axis_extents(List.drop(&2,Views.Axis,local_axes(banded,geometry,rows),_))) == line_elements(after,reduced) : Nat}
  %Equal.sym(List<&2,Views.Axis>,local_axes(banded,geometry,rows),List.append(&2,Views.Axis,head,tail),front_back(before,repeats,after,start,geometry,rows)) :
    {ShapeLaws.elements(Views.axis_extents(List.drop(&2,Views.Axis,_,Nat.add(List.length(&2,Views.Axis,head),taken)))) == line_elements(after,reduced) : Nat}
  %Equal.sym(List<&2,Views.Axis>,List.drop(&2,Views.Axis,List.append(&2,Views.Axis,head,tail),Nat.add(List.length(&2,Views.Axis,head),taken)),List.drop(&2,Views.Axis,tail,taken),
      drop_after(~Views.Axis,head,tail,taken)) :
    {ShapeLaws.elements(Views.axis_extents(_)) == line_elements(after,reduced) : Nat}
  %Equal.sym(List<&2,U32>,Views.axis_extents(List.drop(&2,Views.Axis,tail,taken)),List.drop(&2,U32,Views.axis_extents(tail),taken),extents_dropped(tail,taken)) :
    {ShapeLaws.elements(_) == line_elements(after,reduced) : Nat}
  %Equal.sym(List<&2,U32>,Views.axis_extents(tail),Views.axis_extents(after),Address.extents_localized((height * width : U32),(rows * width : U32),after)) :
    {ShapeLaws.elements(List.drop(&2,U32,_,taken)) == line_elements(after,reduced) : Nat}
  %Equal.sym(List<&2,U32>,Views.axis_extents(List.drop(&2,Views.Axis,after,taken)),List.drop(&2,U32,Views.axis_extents(after),taken),extents_dropped(after,taken)) : {ShapeLaws.elements(List.drop(&2,U32,Views.axis_extents(after),taken)) == ShapeLaws.elements(_) : Nat}
  {==}

# The elements of the rows axis and the repeats of a band.
def rows_elements(+rows: U32,+width: U32,+repeats: List<&2,U32>) ->
  {ShapeLaws.elements(Views.axis_extents(BandViews.rows_axes(True{},rows,width,repeats))) == Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats)) : Nat}:
  Equal.cong(List<&2,U32>,Nat,extents => Nat.mul(U32.to_nat(rows),ShapeLaws.elements(extents)),Views.axis_extents(BandViews.repeated(repeats)),repeats,Address.extents_repeated(repeats))

# The local kept axes hold the axes before the rows, the rows with their
# repeats and the kept part after them.
def local_kept_elements(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+geometry: Bands.Geometry,+rows: U32,+reduced: Nat,
  +within: {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == True{} : Bool}) ->
  {ShapeLaws.elements(Views.axis_extents(local_kept(BandViews.Banded{before,repeats,after,start},geometry,rows,reduced))) ==
    Nat.mul(Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))),kept_elements(after,reduced)) : Nat}:
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +plane = (height * width : U32)
  +band = (rows * width : U32)
  +early = BandViews.localized_axes(plane,band,before)
  +middle = BandViews.rows_axes(True{},rows,width,repeats)
  +head = front(banded,geometry,rows)
  +tail = back(banded,geometry,rows)
  +taken = Nat.sub(List.length(&2,Views.Axis,after),reduced)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +rs = Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))
  +ka = kept_elements(after,reduced)
  %Equal.sym(Nat,Nat.sub(List.length(&2,Views.Axis,local_axes(banded,geometry,rows)),reduced),Nat.add(List.length(&2,Views.Axis,head),taken),
      local_kept_count(before,repeats,after,start,geometry,rows,reduced,within)) :
    {ShapeLaws.elements(Views.axis_extents(List.take(&2,Views.Axis,local_axes(banded,geometry,rows),_))) == Nat.mul(Nat.mul(eb,rs),ka) : Nat}
  %Equal.sym(List<&2,Views.Axis>,local_axes(banded,geometry,rows),List.append(&2,Views.Axis,head,tail),front_back(before,repeats,after,start,geometry,rows)) :
    {ShapeLaws.elements(Views.axis_extents(List.take(&2,Views.Axis,_,Nat.add(List.length(&2,Views.Axis,head),taken)))) == Nat.mul(Nat.mul(eb,rs),ka) : Nat}
  %Equal.sym(List<&2,Views.Axis>,List.take(&2,Views.Axis,List.append(&2,Views.Axis,head,tail),Nat.add(List.length(&2,Views.Axis,head),taken)),
      List.append(&2,Views.Axis,head,List.take(&2,Views.Axis,tail,taken)),take_after(~Views.Axis,head,tail,taken)) :
    {ShapeLaws.elements(Views.axis_extents(_)) == Nat.mul(Nat.mul(eb,rs),ka) : Nat}
  %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,head,List.take(&2,Views.Axis,tail,taken)))),
      Nat.mul(ShapeLaws.elements(Views.axis_extents(head)),ShapeLaws.elements(Views.axis_extents(List.take(&2,Views.Axis,tail,taken)))),
      LineValues.axes_elements(head,List.take(&2,Views.Axis,tail,taken))) :
    {_ == Nat.mul(Nat.mul(eb,rs),ka) : Nat}
  %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(head)),Nat.mul(ShapeLaws.elements(Views.axis_extents(early)),ShapeLaws.elements(Views.axis_extents(middle))),
      LineValues.axes_elements(early,middle)) :
    {Nat.mul(_,ShapeLaws.elements(Views.axis_extents(List.take(&2,Views.Axis,tail,taken)))) == Nat.mul(Nat.mul(eb,rs),ka) : Nat}
  %Equal.sym(List<&2,U32>,Views.axis_extents(early),Views.axis_extents(before),Address.extents_localized(plane,band,before)) :
    {Nat.mul(Nat.mul(ShapeLaws.elements(_),ShapeLaws.elements(Views.axis_extents(middle))),ShapeLaws.elements(Views.axis_extents(List.take(&2,Views.Axis,tail,taken)))) ==
      Nat.mul(Nat.mul(eb,rs),ka) : Nat}
  %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(middle)),rs,rows_elements(rows,width,repeats)) :
    {Nat.mul(Nat.mul(eb,_),ShapeLaws.elements(Views.axis_extents(List.take(&2,Views.Axis,tail,taken)))) == Nat.mul(Nat.mul(eb,rs),ka) : Nat}
  %Equal.sym(List<&2,U32>,Views.axis_extents(List.take(&2,Views.Axis,tail,taken)),List.take(&2,U32,Views.axis_extents(tail),taken),extents_taken(tail,taken)) :
    {Nat.mul(Nat.mul(eb,rs),ShapeLaws.elements(_)) == Nat.mul(Nat.mul(eb,rs),ka) : Nat}
  %Equal.sym(List<&2,U32>,Views.axis_extents(tail),Views.axis_extents(after),Address.extents_localized(plane,band,after)) :
    {Nat.mul(Nat.mul(eb,rs),ShapeLaws.elements(List.take(&2,U32,_,taken))) == Nat.mul(Nat.mul(eb,rs),ka) : Nat}
  %Equal.sym(List<&2,U32>,Views.axis_extents(List.take(&2,Views.Axis,after,taken)),List.take(&2,U32,Views.axis_extents(after),taken),extents_taken(after,taken)) :
    {Nat.mul(Nat.mul(eb,rs),ShapeLaws.elements(List.take(&2,U32,Views.axis_extents(after),taken))) == Nat.mul(Nat.mul(eb,rs),ShapeLaws.elements(_)) : Nat}
  {==}

# ---- Row-major positions over a line ----

# A position of a table whose columns are lines of length values is the
# position of the line times the length plus the position inside the line.
def index_scaled(+outer: Nat,+height: Nat,+narrow: Nat,+row: Nat,+column: Nat,+length: Nat,+element: Nat) ->
  {RowMajor.index(outer,height,Nat.mul(narrow,length),row,Nat.add(Nat.mul(column,length),element)) ==
    Nat.add(Nat.mul(RowMajor.index(outer,height,narrow,row,column),length),element) : Nat}:
  +wide = Nat.mul(narrow,length)
  +x = Nat.mul(outer,Nat.mul(height,narrow))
  +y = Nat.add(Nat.mul(row,narrow),column)
  +scaled_x = Equal.trans(Nat,Nat.mul(x,length),Nat.mul(outer,Nat.mul(Nat.mul(height,narrow),length)),Nat.mul(outer,Nat.mul(height,wide)),
    RowMajor.assoc(outer,Nat.mul(height,narrow),length),
    Equal.cong(Nat,Nat,value => Nat.mul(outer,value),Nat.mul(Nat.mul(height,narrow),length),Nat.mul(height,wide),RowMajor.assoc(height,narrow,length)))
  +scaled_y = Equal.trans(Nat,Nat.mul(y,length),Nat.add(Nat.mul(Nat.mul(row,narrow),length),Nat.mul(column,length)),Nat.add(Nat.mul(row,wide),Nat.mul(column,length)),
    RowMajor.distrib(Nat.mul(row,narrow),column,length),
    Equal.cong(Nat,Nat,value => Nat.add(value,Nat.mul(column,length)),Nat.mul(Nat.mul(row,narrow),length),Nat.mul(row,wide),RowMajor.assoc(row,narrow,length)))
  %Equal.sym(Nat,Nat.mul(Nat.add(x,y),length),Nat.add(Nat.mul(x,length),Nat.mul(y,length)),RowMajor.distrib(x,y,length)) :
    {RowMajor.index(outer,height,wide,row,Nat.add(Nat.mul(column,length),element)) == Nat.add(_,element) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(x,length),Nat.mul(y,length)),element),Nat.add(Nat.mul(x,length),Nat.add(Nat.mul(y,length),element)),
      Addition.associative(Nat.mul(x,length),Nat.mul(y,length),element)) :
    {RowMajor.index(outer,height,wide,row,Nat.add(Nat.mul(column,length),element)) == _ : Nat}
  %Equal.sym(Nat,Nat.mul(x,length),Nat.mul(outer,Nat.mul(height,wide)),scaled_x) :
    {RowMajor.index(outer,height,wide,row,Nat.add(Nat.mul(column,length),element)) == Nat.add(_,Nat.add(Nat.mul(y,length),element)) : Nat}
  %Equal.sym(Nat,Nat.mul(y,length),Nat.add(Nat.mul(row,wide),Nat.mul(column,length)),scaled_y) :
    {RowMajor.index(outer,height,wide,row,Nat.add(Nat.mul(column,length),element)) == Nat.add(Nat.mul(outer,Nat.mul(height,wide)),Nat.add(_,element)) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(row,wide),Nat.mul(column,length)),element),Nat.add(Nat.mul(row,wide),Nat.add(Nat.mul(column,length),element)),
      Addition.associative(Nat.mul(row,wide),Nat.mul(column,length),element)) :
    {RowMajor.index(outer,height,wide,row,Nat.add(Nat.mul(column,length),element)) == Nat.add(Nat.mul(outer,Nat.mul(height,wide)),_) : Nat}
  {==}

# The same for the pixel of a geometry whose columns are lines.
def pixel_scaled(+images: U32,+channels: U32,+height: U32,+narrow: U32,+wide: U32,+length: Nat,+channel: Nat,+row: Nat,+column: Nat,+element: Nat,
  +wide_exact: {U32.to_nat(wide) == Nat.mul(U32.to_nat(narrow),length) : Nat}) ->
  {BandValues.pixel(Bands.Geometry{images,channels,height,wide},channel,row,Nat.add(Nat.mul(column,length),element)) ==
    Nat.add(Nat.mul(BandValues.pixel(Bands.Geometry{images,channels,height,narrow},channel,row,column),length),element) : Nat}:
  +h = U32.to_nat(height)
  +outer = Nat.add(Nat.mul(Nat.div(row,h),U32.to_nat(channels)),channel)
  %Equal.sym(Nat,U32.to_nat(wide),Nat.mul(U32.to_nat(narrow),length),wide_exact) :
    {RowMajor.index(outer,h,_,Nat.mod(row,h),Nat.add(Nat.mul(column,length),element)) ==
      Nat.add(Nat.mul(RowMajor.index(outer,h,U32.to_nat(narrow),Nat.mod(row,h),column),length),element) : Nat}
  index_scaled(outer,h,U32.to_nat(narrow),Nat.mod(row,h),column,length,element)

# ---- One band ----

# What every band of one reduction shares: the logical input, the view and
# its banded form over the input geometry, the number of reduced axes, the
# initial accumulator, the fill of fresh storage, and what the classification
# and the output geometry established.
type Scope<-Element: Data,-Accumulator: Data,-Logical: Data> is Data:
  Scope{logical: Logical,view_axes: List<&2,Views.Axis>,offset: U32,before: List<&2,Views.Axis>,repeats: List<&2,U32>,after: List<&2,Views.Axis>,start: U32,
    batch: U32,channels: U32,height: U32,width: U32,reduced: Nat,initial: Accumulator,fill: Element,limits: Machine.Limits,
    words: ViewValues.Words(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),
    classified: Classified.Classified<view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.Banded{before,repeats,after,start}>,
    room: BandGather.Domain<BandReduce.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},reduced),limits,Machine.capacity(limits)>,
    within: {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == True{} : Bool}}

def scope_context(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>) -> BandReduce.Context<Accumulator,Element>:
  Scope{logical,view_axes,offset,+before,+repeats,+after,+start,+batch,+channels,+height,+width,+reduced,+initial,+fill,limits,words,classified,room,within} = scope
  BandReduce.Context{BandViews.Banded{before,repeats,after,start},reduced,Bands.Geometry{batch,channels,height,width},initial,fill}

def scope_lines(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>) -> Lines<Accumulator,Ordered<Logical>>:
  Scope{+logical,+view_axes,+offset,before,repeats,+after,start,batch,channels,height,width,+reduced,+initial,fill,limits,words,classified,room,within} = scope
  Lines{initial,Ordered{logical,Some{Views.View{view_axes,offset}}},line_elements(after,reduced)}

def input_geometry(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>) -> Bands.Geometry:
  Scope{logical,view_axes,offset,before,repeats,after,start,+batch,+channels,+height,+width,reduced,initial,fill,limits,words,classified,room,within} = scope
  Bands.Geometry{batch,channels,height,width}

def output_geometry(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>) -> Bands.Geometry:
  Scope{logical,view_axes,offset,+before,+repeats,+after,+start,+batch,+channels,+height,+width,+reduced,initial,fill,limits,words,classified,room,within} = scope
  BandReduce.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},reduced)

def scaled_word(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>,+value: U32) -> U32:
  Scope{logical,view_axes,offset,before,+repeats,after,start,batch,channels,height,width,reduced,initial,fill,limits,words,classified,room,within} = scope
  (value * BandViews.product_or_zero(repeats) : U32)

# The offset of the local view of a band, its kept axes and its line.
def scope_offset(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>,+rows: U32) -> U32:
  Scope{logical,view_axes,offset,before,repeats,after,+start,batch,channels,+height,+width,reduced,initial,fill,limits,words,classified,room,within} = scope
  BandViews.localized((height * width : U32),(rows * width : U32),start)

def scope_kept(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>,+rows: U32) -> List<&2,Views.Axis>:
  Scope{logical,view_axes,offset,+before,+repeats,+after,+start,+batch,+channels,+height,+width,+reduced,initial,fill,limits,words,classified,room,within} = scope
  local_kept(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows,reduced)

def scope_line(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>,+rows: U32) -> List<&2,Views.Axis>:
  Scope{logical,view_axes,offset,+before,+repeats,+after,+start,+batch,+channels,+height,+width,+reduced,initial,fill,limits,words,classified,room,within} = scope
  local_line(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows,reduced)

def scope_initial(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>) -> Accumulator:
  Scope{logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,reduced,+initial,fill,limits,words,classified,room,within} = scope
  initial

# The facts of one input band: its fields, its buffer is ready, and its
# values.
def input_fields(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  BandShape.band_fields(~Element,input_geometry(~Element,~Accumulator,~Logical,scope))

def input_values(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Scope<Element,Accumulator,Logical>) ->
  @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Scope{+logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,reduced,initial,fill,limits,words,classified,room,within} = scope
  BandValues.band_values(~Element,~Logical,~original,logical,input_geometry(~Element,~Accumulator,~Logical,scope))

def Input(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,+scope: Scope<Element,Accumulator,Logical>) ->
  @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Model.Both(~Element,~Scope<Element,Accumulator,Logical>,
    ~Model.Both(~Element,~Scope<Element,Accumulator,Logical>,~input_fields(~Element,~Accumulator,~Logical),~Model.always(~Element,~Scope<Element,Accumulator,Logical>,~Model.ready_band(~Element))),
    ~input_values(~Element,~Accumulator,~Logical,~original),scope)

# An output band holds at each cell the fold of the line of the view at the
# pixel of the cell.
def OutputValues(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  BandValues.band_values(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
    scope_lines(~Element,~Accumulator,~Logical,scope),output_geometry(~Element,~Accumulator,~Logical,scope))

def Output(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>) -> @+first: U32 -> @+rows: U32 -> @+buffer: Buffer.Content<Element> -> Data:
  Global.Made(~Element,~Scope<Element,Accumulator,Logical>,~OutputValues(~Element,~Accumulator,~Logical,~original,~operation,~finish),scope)

# The outputs a band writes: the lines of its local view into the output
# tree.
def reduced_tree(~Element: Data,~Accumulator: Data,~Logical: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>,+rows: U32,+count: U32,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>) -> Tree.Tree<Element>:
  CopyValues.output_tree(~Element,OperationsLaws.runs_model(~Element,~Accumulator,~operation,~finish,U32.to_nat(count),scope_kept(~Element,~Accumulator,~Logical,scope,rows),
    scope_offset(~Element,~Accumulator,~Logical,scope,rows),Operations.line_runs(scope_line(~Element,~Accumulator,~Logical,scope,rows)),scope_initial(~Element,~Accumulator,~Logical,scope),
    input,output))

# The input band of a reduced band and the output tree it wrote into.
type Cells<-Element: Data,-Accumulator: Data,-Logical: Data,-original: Logical -> U32 -> Element> is Data:
  Cells{scope: Scope<Element,Accumulator,Logical>,first: U32,rows: U32,source_tree: Tree.Tree<Element>,source_length: Nat,
    input_fact: Input(~Element,~Accumulator,~Logical,~original,scope,first,rows,Buffer.Content{source_tree,source_length}),
    output: Tree.Tree<Element>,count: U32,
    ready: Buffer.ContentReady<Element,Buffer.Content{output,U32.to_nat(count)}>,
    capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
    room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
    counted: {U32.to_nat(count) == ShapeLaws.elements(Views.axis_extents(scope_kept(~Element,~Accumulator,~Logical,scope,rows))) : Nat}}

def cells_geometry(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: Cells<Element,Accumulator,Logical,original>) -> Bands.Geometry:
  Cells{+scope,first,rows,source_tree,source_length,input_fact,output,count,ready,capacity_ok,room,counted} = evidence
  output_geometry(~Element,~Accumulator,~Logical,scope)

def cells_rows(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,evidence: Cells<Element,Accumulator,Logical,original>) -> U32:
  Cells{+scope,first,+rows,source_tree,source_length,input_fact,output,count,ready,capacity_ok,room,counted} = evidence
  scaled_word(~Element,~Accumulator,~Logical,scope,rows)

def CellAt(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +evidence: Cells<Element,Accumulator,Logical,original>,+channel: Nat,+row: Nat,+column: Nat) -> Data:
  Cells{+scope,+first,+rows,+source_tree,source_length,input_fact,+output,+count,ready,capacity_ok,room,counted} = evidence
  BandValues.At(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
    scope_lines(~Element,~Accumulator,~Logical,scope),output_geometry(~Element,~Accumulator,~Logical,scope),scaled_word(~Element,~Accumulator,~Logical,scope,first),
    scaled_word(~Element,~Accumulator,~Logical,scope,rows),reduced_tree(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,count,source_tree,output),U32.to_nat(count),
    channel,row,column)

# A word known to be positive is the count of its extents.
def positive_exact(result: Maybe<&2,U32>,+extents: List<&2,U32>,count: ShapeLaws.Count(result,ShapeLaws.elements(extents)),
  positive: {Nat.is_lt(0n,U32.to_nat(BandViews.known(result))) == True{} : Bool}) -> {U32.to_nat(BandViews.known(result)) == ShapeLaws.elements(extents) : Nat}:
  match result:
    case Some{+value}: Equal.sym(Nat,ShapeLaws.elements(extents),U32.to_nat(value),count)
    case None{}: Empty.absurd({U32.to_nat(BandViews.known(None{})) == ShapeLaws.elements(extents) : Nat},WordEncoding.false_true(positive))

# The output geometry has the kept part as its columns.
def kept_exact(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +reduced: Nat,+limits: Machine.Limits,
  domain: BandGather.Domain<BandReduce.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},reduced),limits,Machine.capacity(limits)>) ->
  {U32.to_nat(BandViews.product_or_zero(Views.axis_extents(kept_part(after,reduced)))) == kept_elements(after,reduced) : Nat}:
  BandGather.Domain{capacity_ok,channels_positive,height_positive,width_positive,machine_room,allocation_room} = domain
  +extents = Views.axis_extents(kept_part(after,reduced))
  positive_exact(Shapes.product(extents,Some{1}),extents,ShapeLaws.nonempty_count(extents),width_positive)

# Element element of the line of output cell channel, row, column of a band
# is the input at the address the view gives to that element of the line of
# the pixel of the cell.
def element_read(~Element: Data,~Logical: Data,~original: Logical -> U32 -> Element,+logical: Logical,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,+limits: Machine.Limits,
  +words: ViewValues.Words(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),
  +classified: Classified.Classified<view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.Banded{before,repeats,after,start}>,
  +rows_room: {Nat.is_le(Nat.mul(U32.to_nat(batch),U32.to_nat((height * BandViews.product_or_zero(repeats) : U32))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +first: U32,+rows: U32,+source_tree: Tree.Tree<Element>,+source_length: Nat,
  +fields: BandShape.Leaf(Bands.Geometry{batch,channels,height,width},first,rows,source_length),
  +source_ready: Buffer.ContentReady<Element,Buffer.Content{source_tree,source_length}>,
  +source_values: BandValues.Leaf(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},first,rows,source_tree,source_length),
  +reduced: Nat,+kept_exact: {U32.to_nat(BandViews.product_or_zero(Views.axis_extents(kept_part(after,reduced)))) == kept_elements(after,reduced) : Nat},
  +channel: Nat,+row: Nat,+column: Nat,
  +channel_below: {Nat.is_lt(channel,ShapeLaws.elements(Views.axis_extents(before))) == True{} : Bool},
  +row_below: {Nat.is_lt(row,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))) == True{} : Bool},
  +column_below: {Nat.is_lt(column,kept_elements(after,reduced)) == True{} : Bool},
  +element: Nat,+inside: {Nat.is_lt(element,line_elements(after,reduced)) == True{} : Bool}) ->
  {CopyValues.sampled(~Element,CopyValues.Sampled{BandViews.local(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows),source_tree},
      Nat.add(Nat.mul(RowMajor.index(channel,Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats)),kept_elements(after,reduced),row,column),line_elements(after,reduced)),element)) ==
    ordered_value(~Element,~Logical,~original,Ordered{logical,Some{Views.View{view_axes,offset}}},
      Nat.add(Nat.mul(BandValues.pixel(BandReduce.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},reduced),channel,
        Nat.add(U32.to_nat((first * BandViews.product_or_zero(repeats) : U32)),row),column),line_elements(after,reduced)),element)) : Element}:
  Shape.Both{ViewValues.Fitted{batch_positive,height_positive,channel_reach,column_reach,early,+late,repeated,rows_exact},plane} = words
  +scale = BandViews.product_or_zero(repeats)
  +before_count = BandViews.product_or_zero(Views.axis_extents(before))
  +after_count = BandViews.product_or_zero(Views.axis_extents(after))
  +kept_count = BandViews.product_or_zero(Views.axis_extents(kept_part(after,reduced)))
  +ka = kept_elements(after,reduced)
  +length = line_elements(after,reduced)
  +ea = ShapeLaws.elements(Views.axis_extents(after))
  +rs = Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))
  +global = Nat.add(U32.to_nat((first * scale : U32)),row)
  +wide = Nat.add(Nat.mul(column,length),element)
  +split = after_split(after,reduced)
  +wide_below = {%Equal.sym(Nat,ea,Nat.mul(ka,length),split) : {Nat.is_lt(wide,_) == True{} : Bool}
    RowMajor.radix(column,ka,element,length,column_below,inside) : {Nat.is_lt(wide,ea) == True{} : Bool}}
  +wide_exact = Equal.trans(Nat,U32.to_nat(after_count),ea,Nat.mul(U32.to_nat(kept_count),length),late,
    Equal.trans(Nat,ea,Nat.mul(ka,length),Nat.mul(U32.to_nat(kept_count),length),split,
      Equal.cong(Nat,Nat,value => Nat.mul(value,length),ka,U32.to_nat(kept_count),Equal.sym(Nat,U32.to_nat(kept_count),ka,kept_exact))))
  %index_scaled(channel,rs,ka,row,column,length,element) :
    {Tree.read(Element,source_tree,CopyValues.address(BandViews.local(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows),_)) ==
      ordered_value(~Element,~Logical,~original,Ordered{logical,Some{Views.View{view_axes,offset}}},
        Nat.add(Nat.mul(BandValues.pixel(Bands.Geometry{batch,before_count,(height * scale : U32),kept_count},channel,global,column),length),element)) : Element}
  %split :
    {Tree.read(Element,source_tree,CopyValues.address(BandViews.local(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows),
        RowMajor.index(channel,rs,_,row,wide))) ==
      ordered_value(~Element,~Logical,~original,Ordered{logical,Some{Views.View{view_axes,offset}}},
        Nat.add(Nat.mul(BandValues.pixel(Bands.Geometry{batch,before_count,(height * scale : U32),kept_count},channel,global,column),length),element)) : Element}
  %pixel_scaled(batch,before_count,(height * scale : U32),kept_count,after_count,length,channel,global,column,element,wide_exact) :
    {Tree.read(Element,source_tree,CopyValues.address(BandViews.local(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows),
        RowMajor.index(channel,rs,ea,row,wide))) ==
      ordered_value(~Element,~Logical,~original,Ordered{logical,Some{Views.View{view_axes,offset}}},_) : Element}
  ViewValues.local_read(~Element,~Logical,~original,logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,limits,words,classified,rows_room,
    first,rows,source_tree,source_length,fields,source_ready,source_values,channel,row,wide,channel_below,row_below,wide_below)

# One cell of an output band: the band folds, at the position of the cell,
# the line of its local view, which reads the line of the whole view at the
# pixel of the cell.
def cell(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +evidence: Cells<Element,Accumulator,Logical,original>,+channel: Nat,+row: Nat,+column: Nat,
  channel_inside: {Nat.is_lt(channel,BandValues.channels(cells_geometry(~Element,~Accumulator,~Logical,~original,evidence))) == True{} : Bool},
  row_inside: {Nat.is_lt(row,U32.to_nat(cells_rows(~Element,~Accumulator,~Logical,~original,evidence))) == True{} : Bool},
  column_inside: {Nat.is_lt(column,BandValues.width(cells_geometry(~Element,~Accumulator,~Logical,~original,evidence))) == True{} : Bool}) ->
  CellAt(~Element,~Accumulator,~Logical,~original,~operation,~finish,evidence,channel,row,column):
  Cells{+scope,+first,+rows,+source_tree,+source_length,+input_fact,+output,+count,+ready,+capacity_ok,+room,+counted} = evidence
  Scope{+logical,+view_axes,+offset,+before,+repeats,+after,+start,+batch,+channels,+height,+width,+reduced,+initial,fill,+limits,+words,+classified,+domain,+within} = scope
  Shape.Both{ViewValues.Fitted{batch_positive,height_positive,channel_reach,column_reach,+early,late,+repeated,+rows_exact},plane} = words
  Shape.Both{Shape.Both{+fields,+source_ready},+source_values} = input_fact
  +channel_inside = channel_inside
  +row_inside = row_inside
  +column_inside = column_inside
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +scale = BandViews.product_or_zero(repeats)
  +before_count = BandViews.product_or_zero(Views.axis_extents(before))
  +kept_count = BandViews.product_or_zero(Views.axis_extents(kept_part(after,reduced)))
  +reduced_geometry = {Bands.Geometry{batch,before_count,(height * scale : U32),kept_count} : Bands.Geometry}
  +capacity = Machine.capacity(limits)
  +es = ShapeLaws.elements(repeats)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +ka = kept_elements(after,reduced)
  +length = line_elements(after,reduced)
  +first_n = U32.to_nat(first)
  +rows_n = U32.to_nat(rows)
  +rs = Nat.mul(rows_n,es)
  +rows_room = BandGather.rows_room(reduced_geometry,limits,capacity,domain)
  +rows_fit = ViewValues.band_rows_fit(geometry,first,rows,source_length,fields)
  +fit = ViewValues.scaled_rows_fit(batch,height,scale,first,rows,rows_fit,rows_exact)
  +fit_room = Order.transitive(Nat.mul(Nat.add(first_n,rows_n),U32.to_nat(scale)),Nat.mul(U32.to_nat(batch),U32.to_nat((height * scale : U32))),U32.to_nat(capacity),fit,rows_room)
  +first_exact = ViewValues.scaled_first_exact(first,rows,scale,capacity,fit_room)
  +band_rows_exact = ViewValues.scaled_rows_exact(first,rows,scale,capacity,fit_room)
  +rows_es = {%repeated : {U32.to_nat((rows * scale : U32)) == Nat.mul(rows_n,_) : Nat} band_rows_exact : {U32.to_nat((rows * scale : U32)) == rs : Nat}}
  +row_below = {%rows_es : {Nat.is_lt(row,_) == True{} : Bool} row_inside : {Nat.is_lt(row,rs) == True{} : Bool}}
  +channel_below = {%early : {Nat.is_lt(channel,_) == True{} : Bool} channel_inside : {Nat.is_lt(channel,eb) == True{} : Bool}}
  +columns = kept_exact(before,repeats,after,start,batch,channels,height,width,reduced,limits,domain)
  +column_below = {%columns : {Nat.is_lt(column,_) == True{} : Bool} column_inside : {Nat.is_lt(column,ka) == True{} : Bool}}
  +k = RowMajor.index(channel,rs,ka,row,column)
  +position = BandValues.position(reduced_geometry,(rows * scale : U32),channel,row,column)
  +position_exact = {
    %Equal.sym(Nat,U32.to_nat((rows * scale : U32)),rs,rows_es) : {RowMajor.index(channel,_,U32.to_nat(kept_count),row,column) == k : Nat}
    %Equal.sym(Nat,U32.to_nat(kept_count),ka,columns) : {RowMajor.index(channel,rs,_,row,column) == k : Nat}
    {==} : {position == k : Nat}}
  +count_exact = Equal.trans(Nat,U32.to_nat(count),ShapeLaws.elements(Views.axis_extents(local_kept(banded,geometry,rows,reduced))),Nat.mul(Nat.mul(eb,rs),ka),counted,
    local_kept_elements(before,repeats,after,start,geometry,rows,reduced,within))
  +k_below = {
    %Equal.sym(Nat,U32.to_nat(count),Nat.mul(Nat.mul(eb,rs),ka),count_exact) : {Nat.is_lt(k,_) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(Nat.mul(eb,rs),ka),Nat.mul(eb,Nat.mul(rs,ka)),RowMajor.assoc(eb,rs,ka)) : {Nat.is_lt(k,_) == True{} : Bool}
    RowMajor.in_bounds(channel,eb,rs,ka,row,column,channel_below,row_below,column_below) : {Nat.is_lt(k,U32.to_nat(count)) == True{} : Bool}}
  +bound = {%Equal.sym(Nat,position,k,position_exact) : {Nat.is_lt(_,U32.to_nat(count)) == True{} : Bool} k_below : {Nat.is_lt(position,U32.to_nat(count)) == True{} : Bool}}
  +global = Nat.add(U32.to_nat((first * scale : U32)),row)
  +pixel = BandValues.pixel(reduced_geometry,channel,global,column)
  +global_inside = ViewValues.scaled_row_inside(batch,height,scale,first,rows,row,row_inside,first_exact,band_rows_exact,fit)
  +pixel_word = FullRange.roundtrip(pixel,capacity,
    Order.transitive(pixel,BandGather.elements(reduced_geometry),U32.to_nat(capacity),
      Bounds.strict_le(pixel,BandGather.elements(reduced_geometry),BandGather.pixel_below(reduced_geometry,channel,global,column,channel_inside,global_inside,column_inside,
        ViewValues.domain_height(reduced_geometry,limits,capacity,domain))),
      ViewValues.domain_elements(reduced_geometry,limits,capacity,domain)))
  +local = BandViews.local(banded,geometry,rows)
  +loff = BandViews.localized((height * width : U32),(rows * width : U32),start)
  +axes = local_axes(banded,geometry,rows)
  +kept = local_kept(banded,geometry,rows,reduced)
  +line = local_line(banded,geometry,rows,reduced)
  +count_n = U32.to_nat(count)
  +context = {Ordered{logical,Some{Views.View{view_axes,offset}}} : Ordered<Logical>}
  +tree = reduced_tree(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,count,source_tree,output)
  +folded_line = LineValues.line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,ShapeLaws.elements(Views.axis_extents(line)),
    CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,kept,line),loff},source_tree},Nat.mul(k,ShapeLaws.elements(Views.axis_extents(line))),initial)
  +value_line = LineValues.line_fold(~Ordered<Logical>,~Element,~Accumulator,~ordered_value(~Element,~Logical,~original),~operation,length,context,Nat.mul(pixel,length),initial)
  +lines_read = LineValues.lines_value(~Element,~Accumulator,~operation,~finish,count_n,kept,line,loff,initial,source_tree,output,count_n,k,ready,capacity_ok,room,k_below,counted)
  +agreed = {
    %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(line)),length,local_line_elements(before,repeats,after,start,geometry,rows,reduced,within)) :
      {LineValues.line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,_,
          CopyValues.Sampled{Views.View{List.append(&2,Views.Axis,kept,line),loff},source_tree},Nat.mul(k,_),initial) == value_line : Accumulator}
    %Equal.sym(List<&2,Views.Axis>,List.append(&2,Views.Axis,kept,line),axes,LineValues.take_drop(~Views.Axis,axes,Nat.sub(List.length(&2,Views.Axis,axes),reduced))) :
      {LineValues.line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,length,
          CopyValues.Sampled{Views.View{_,loff},source_tree},Nat.mul(k,length),initial) == value_line : Accumulator}
    %Addition.zero_right(Nat.mul(k,length)) :
      {LineValues.line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,length,CopyValues.Sampled{local,source_tree},_,initial) ==
        value_line : Accumulator}
    %Addition.zero_right(Nat.mul(pixel,length)) :
      {LineValues.line_fold(~CopyValues.Sampled<Element>,~Element,~Accumulator,~CopyValues.sampled(~Element),~operation,length,CopyValues.Sampled{local,source_tree},
          Nat.add(Nat.mul(k,length),0n),initial) ==
        LineValues.line_fold(~Ordered<Logical>,~Element,~Accumulator,~ordered_value(~Element,~Logical,~original),~operation,length,context,_,initial) : Accumulator}
    LineValues.line_agreed(~Element,~Accumulator,~CopyValues.Sampled<Element>,~CopyValues.sampled(~Element),~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,
      ~CopyValues.Sampled{local,source_tree},~Nat.mul(k,length),~context,~Nat.mul(pixel,length),~length,
      ~(element => inside => element_read(~Element,~Logical,~original,logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,limits,words,classified,rows_room,
        first,rows,source_tree,source_length,fields,source_ready,source_values,reduced,columns,channel,row,column,channel_below,row_below,column_below,element,inside)),
      length,0n,initial,Order.reflexive(length)) :
    {folded_line == value_line : Accumulator}}
  +read = {
    %Equal.sym(Nat,position,k,position_exact) :
      {Tree.read(Element,tree,U32.from_nat(_)) ==
        finish(LineValues.line_fold(~Ordered<Logical>,~Element,~Accumulator,~ordered_value(~Element,~Logical,~original),~operation,length,context,
          Nat.mul(U32.to_nat(U32.from_nat(pixel)),length),initial)) : Element}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(pixel)),pixel,pixel_word) :
      {Tree.read(Element,tree,U32.from_nat(k)) ==
        finish(LineValues.line_fold(~Ordered<Logical>,~Element,~Accumulator,~ordered_value(~Element,~Logical,~original),~operation,length,context,Nat.mul(_,length),initial)) : Element}
    Equal.trans(Element,Tree.read(Element,tree,U32.from_nat(k)),finish(folded_line),finish(value_line),lines_read,Equal.cong(Accumulator,Element,finish,folded_line,value_line,agreed)) :
    {Tree.read(Element,tree,U32.from_nat(position)) ==
      value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish,scope_lines(~Element,~Accumulator,~Logical,scope),U32.from_nat(pixel)) :
      Element}}
  Shape.Both{bound,read}

# Every cell of an output band.
def values_output(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>,+first: U32,+rows: U32,+source_tree: Tree.Tree<Element>,+source_length: Nat,
  +input_fact: Input(~Element,~Accumulator,~Logical,~original,scope,first,rows,Buffer.Content{source_tree,source_length}),
  +output: Tree.Tree<Element>,+count: U32,+ready: Buffer.ContentReady<Element,Buffer.Content{output,U32.to_nat(count)}>,
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +counted: {U32.to_nat(count) == ShapeLaws.elements(Views.axis_extents(scope_kept(~Element,~Accumulator,~Logical,scope,rows))) : Nat}) ->
  OutputValues(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),scaled_word(~Element,~Accumulator,~Logical,scope,rows),
    Buffer.Content{reduced_tree(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,count,source_tree,output),U32.to_nat(count)}):
  BandValues.cells(~Cells<Element,Accumulator,Logical,original>,~CellAt(~Element,~Accumulator,~Logical,~original,~operation,~finish),~cells_geometry(~Element,~Accumulator,~Logical,~original),
    ~cells_rows(~Element,~Accumulator,~Logical,~original),~cell(~Element,~Accumulator,~Logical,~original,~operation,~finish),
    Cells{scope,first,rows,source_tree,source_length,input_fact,output,count,ready,capacity_ok,room,counted})

# The lines leave a ready output tree.
def runs_ready(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +count: Nat,+kept: List<&2,Views.Axis>,+offset: U32,runs: List<&2,Views.Axis> & Views.Axis,+initial: Accumulator,+input: Tree.Tree<Element>,+output: Tree.Tree<Element>,
  +output_length: Nat,ready: Buffer.ContentReady<Element,Buffer.Content{output,output_length}>) ->
  Buffer.ContentReady<Element,Buffer.Content{CopyValues.output_tree(~Element,OperationsLaws.runs_model(~Element,~Accumulator,~operation,~finish,count,kept,offset,runs,initial,input,output)),
    output_length}>:
  (+outer,Views.Axis{+extent,+stride}) = runs
  +view = {Views.View{List.append(&2,Views.Axis,kept,outer),offset} : Views.View}
  +start = {AccessLaws.LineCoordinates{CursorLaws.initial(view),Operations.elements_of(outer),extent,stride,initial} : AccessLaws.LineCoordinates<Accumulator>}
  +visit = CopyValues.visit_writes(~Element,~AccessLaws.LineCoordinates<Accumulator>,~AccessLaws.sample_line(~Element,~Accumulator,~operation,~finish),
    ~AccessLaws.next_line(~Accumulator),count,0,start,input)
  %Equal.sym(Tree.Tree<Element>,
      CopyValues.output_tree(~Element,AccessLaws.lines_model(~Element,~Accumulator,~operation,~finish,count,view,Operations.elements_of(outer),extent,stride,initial,input,output)),
      Writes.pure(Element,visit,output),
      CopyValues.visit_program(~Element,~AccessLaws.LineCoordinates<Accumulator>,~AccessLaws.sample_line(~Element,~Accumulator,~operation,~finish),
        ~AccessLaws.next_line(~Accumulator),count,0,start,input,output)) :
    Buffer.ContentReady<Element,Buffer.Content{_,output_length}>
  Writes.content_ready(~Element,visit,output,output_length,ready)

# The band of the lines of a band: ready, holding the fold at every cell.
def band_output(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>,+first: U32,+rows: U32,+source_tree: Tree.Tree<Element>,+source_length: Nat,
  +input_fact: Input(~Element,~Accumulator,~Logical,~original,scope,first,rows,Buffer.Content{source_tree,source_length}),
  +output: Tree.Tree<Element>,+count: U32,+ready: Buffer.ContentReady<Element,Buffer.Content{output,U32.to_nat(count)}>,
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output),2147483648) == True{} : Bool},
  +room: {Nat.is_le(U32.to_nat(count),U32.to_nat(Tree.capacity(Element,output))) == True{} : Bool},
  +counted: {U32.to_nat(count) == ShapeLaws.elements(Views.axis_extents(scope_kept(~Element,~Accumulator,~Logical,scope,rows))) : Nat}) ->
  Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),scaled_word(~Element,~Accumulator,~Logical,scope,rows),
    Buffer.Content{reduced_tree(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,count,source_tree,output),U32.to_nat(count)}):
  Shape.Both{
    runs_ready(~Element,~Accumulator,~operation,~finish,U32.to_nat(count),scope_kept(~Element,~Accumulator,~Logical,scope,rows),scope_offset(~Element,~Accumulator,~Logical,scope,rows),
      Operations.line_runs(scope_line(~Element,~Accumulator,~Logical,scope,rows)),scope_initial(~Element,~Accumulator,~Logical,scope),source_tree,output,U32.to_nat(count),ready),
    values_output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,first,rows,source_tree,source_length,input_fact,output,count,ready,capacity_ok,room,counted)}

# ---- One band through its checked call ----

def scope_banded(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>) -> BandViews.Banded:
  Scope{logical,view_axes,offset,+before,+repeats,+after,+start,batch,channels,height,width,reduced,initial,fill,limits,words,classified,room,within} = scope
  BandViews.Banded{before,repeats,after,start}

def scope_reduced(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>) -> Nat:
  Scope{logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,+reduced,initial,fill,limits,words,classified,room,within} = scope
  reduced

def scope_fill(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>) -> Element:
  Scope{logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,reduced,initial,+fill,limits,words,classified,room,within} = scope
  fill

# The call of a band: the lines of its local view into the output storage.
def reduced_call(~Element: Data,~Accumulator: Data,~Logical: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>,+rows: U32,input: Storage.Buffer<Element>,output: Storage.Buffer<Element>) -> Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>:
  Scope{logical,view_axes,offset,+before,+repeats,+after,+start,+batch,+channels,+height,+width,+reduced,+initial,fill,limits,words,classified,room,within} = scope
  Operations.reduce_lines(~Element,~Accumulator,~operation,~finish,input,local_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows),reduced,
    BandViews.localized((height * width : U32),(rows * width : U32),start),initial,output)

def reduced_count(~Element: Data,~Accumulator: Data,~Logical: Data,+scope: Scope<Element,Accumulator,Logical>,+rows: U32) -> U32:
  BandViews.known(Views.numel(Views.View{scope_kept(~Element,~Accumulator,~Logical,scope,rows),0}))

# The kept axes of a band hold no more elements than the machine.
def kept_room(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +reduced: Nat,+limits: Machine.Limits,+words: ViewValues.Words(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),
  +domain: BandGather.Domain<BandReduce.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},reduced),limits,Machine.capacity(limits)>,
  +first: U32,+rows: U32,+source_length: Nat,+fields: BandShape.Leaf(Bands.Geometry{batch,channels,height,width},first,rows,source_length)) ->
  {Nat.is_le(Nat.mul(Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))),kept_elements(after,reduced)),
    U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  Shape.Both{ViewValues.Fitted{batch_positive,height_positive,channel_reach,column_reach,+early,late,+repeated,+rows_exact},plane} = words
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +scale = BandViews.product_or_zero(repeats)
  +s = U32.to_nat(scale)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +es = ShapeLaws.elements(repeats)
  +ka = kept_elements(after,reduced)
  +rows_n = U32.to_nat(rows)
  +first_n = U32.to_nat(first)
  +b = U32.to_nat(batch)
  +h = U32.to_nat((height * scale : U32))
  +total = Nat.mul(b,h)
  +cap = U32.to_nat(Machine.capacity(limits))
  +rs = Nat.mul(rows_n,es)
  +y = Nat.mul(h,ka)
  +fit = ViewValues.scaled_rows_fit(batch,height,scale,first,rows,ViewValues.band_rows_fit(geometry,first,rows,source_length,fields),rows_exact)
  +rows_scaled = Order.transitive(Nat.mul(rows_n,s),Nat.mul(Nat.add(first_n,rows_n),s),total,
    RowMajor.mul_le(rows_n,Nat.add(first_n,rows_n),s,
      %Order.commutative(rows_n,first_n) : {Nat.is_le(rows_n,_) == True{} : Bool}
      Order.le_add(rows_n,first_n)),fit)
  +rows_total = {%repeated : {Nat.is_le(Nat.mul(rows_n,_),total) == True{} : Bool} rows_scaled : {Nat.is_le(rs,total) == True{} : Bool}}
  +left_le = {
    %RowMajor.multiply_commutative(rs,eb) : {Nat.is_le(_,Nat.mul(eb,total)) == True{} : Bool}
    %RowMajor.multiply_commutative(total,eb) : {Nat.is_le(Nat.mul(rs,eb),_) == True{} : Bool}
    RowMajor.mul_le(rs,total,eb,rows_total) : {Nat.is_le(Nat.mul(eb,rs),Nat.mul(eb,total)) == True{} : Bool}}
  +whole_le = RowMajor.mul_le(Nat.mul(eb,rs),Nat.mul(eb,total),ka,left_le)
  +rearranged = Equal.trans(Nat,Nat.mul(Nat.mul(eb,total),ka),Nat.mul(eb,Nat.mul(total,ka)),Nat.mul(b,Nat.mul(eb,y)),
    RowMajor.assoc(eb,total,ka),
    Equal.trans(Nat,Nat.mul(eb,Nat.mul(total,ka)),Nat.mul(eb,Nat.mul(b,y)),Nat.mul(b,Nat.mul(eb,y)),
      Equal.cong(Nat,Nat,value => Nat.mul(eb,value),Nat.mul(total,ka),Nat.mul(b,y),RowMajor.assoc(b,h,ka)),
      Equal.trans(Nat,Nat.mul(eb,Nat.mul(b,y)),Nat.mul(Nat.mul(eb,b),y),Nat.mul(b,Nat.mul(eb,y)),
        Equal.sym(Nat,Nat.mul(Nat.mul(eb,b),y),Nat.mul(eb,Nat.mul(b,y)),RowMajor.assoc(eb,b,y)),
        Equal.trans(Nat,Nat.mul(Nat.mul(eb,b),y),Nat.mul(Nat.mul(b,eb),y),Nat.mul(b,Nat.mul(eb,y)),
          Equal.cong(Nat,Nat,value => Nat.mul(value,y),Nat.mul(eb,b),Nat.mul(b,eb),RowMajor.multiply_commutative(eb,b)),
          RowMajor.assoc(b,eb,y)))))
  +reduced_geometry = BandReduce.output(BandViews.Banded{before,repeats,after,start},geometry,reduced)
  +room_output = {
    %kept_exact(before,repeats,after,start,batch,channels,height,width,reduced,limits,domain) : {Nat.is_le(Nat.mul(b,Nat.mul(eb,Nat.mul(h,_))),cap) == True{} : Bool}
    %early : {Nat.is_le(Nat.mul(b,Nat.mul(_,Nat.mul(h,U32.to_nat(BandViews.product_or_zero(Views.axis_extents(kept_part(after,reduced))))))),cap) == True{} : Bool}
    ViewValues.domain_elements(reduced_geometry,limits,Machine.capacity(limits),domain) : {Nat.is_le(Nat.mul(b,Nat.mul(eb,y)),cap) == True{} : Bool}}
  Order.transitive(Nat.mul(Nat.mul(eb,rs),ka),Nat.mul(Nat.mul(eb,total),ka),cap,whole_le,
    %Equal.sym(Nat,Nat.mul(Nat.mul(eb,total),ka),Nat.mul(b,Nat.mul(eb,y)),rearranged) : {Nat.is_le(_,cap) == True{} : Bool} room_output)

# The output storage of a band can be allocated.
def known_bound(result: Maybe<&2,U32>,+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +reduced: Nat,+limits: Machine.Limits,+words: ViewValues.Words(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),
  +domain: BandGather.Domain<BandReduce.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},reduced),limits,Machine.capacity(limits)>,
  +within: {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == True{} : Bool},
  +first: U32,+rows: U32,+source_length: Nat,+fields: BandShape.Leaf(Bands.Geometry{batch,channels,height,width},first,rows,source_length),
  count: ShapeLaws.Count(result,ShapeLaws.elements(Views.axis_extents(local_kept(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows,reduced))))) ->
  {U32.is_le(BandViews.known(result),1073741824) == True{} : Bool}:
  match result:
    case Some{+value}:
      +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
      +kept = local_kept(BandViews.Banded{before,repeats,after,start},geometry,rows,reduced)
      +product = Nat.mul(Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(rows),ShapeLaws.elements(repeats))),kept_elements(after,reduced))
      +cap = U32.to_nat(Machine.capacity(limits))
      +below = {
        %count : {Nat.is_le(_,cap) == True{} : Bool}
        %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(kept)),product,local_kept_elements(before,repeats,after,start,geometry,rows,reduced,within)) : {Nat.is_le(_,cap) == True{} : Bool}
        kept_room(before,repeats,after,start,batch,channels,height,width,reduced,limits,words,domain,first,rows,source_length,fields) :
        {Nat.is_le(U32.to_nat(value),cap) == True{} : Bool}}
      Encoding.native_le_transitive(value,Machine.capacity(limits),1073741824,NativeDivision.nat_le_native(value,Machine.capacity(limits),below),Limits.capacity_allocatable(limits))
    case None{}: {==}

# The output content of an accepted call of the lines model.
def modeled_right(~Element: Data,+input_length: Nat,+output_length: Nat,result: AccessLaws.ModelResult<Tree.Tree<Element>,Element>) ->
  {CopyValues.right_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,result)}) ==
    Buffer.Content{CopyValues.output_tree(~Element,result),output_length} : Buffer.Content<Element>}:
  match result:
    case AccessLaws.ModelResult{+reader,+output}:
      Equal.cong(Tree.Tree<Element>,Buffer.Content<Element>,tree => Buffer.Content{tree,output_length},Tree.reflect(Element,Tree.pack(Element,output)),output,
        StorageLaws.reflect_packed(Element,output))

# A call whose check holds writes the lines of its kept axes; a call whose
# check fails is not accepted.
def modeled_output(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  valid: Bool,result: Maybe<&2,U32>,+scope: Scope<Element,Accumulator,Logical>,+first: U32,+rows: U32,
  +input_tree: Tree.Tree<Element>,+input_length: Nat,+output_tree: Tree.Tree<Element>,+output_length: Nat,
  -outcome: Operations.Outcome<OperationsLaws.PairOwners(Element)>,
  +observed: {outcome == OperationsLaws.lines_model(~Element,~Accumulator,~operation,~finish,valid,result,scope_kept(~Element,~Accumulator,~Logical,scope,rows),
    scope_line(~Element,~Accumulator,~Logical,scope,rows),scope_offset(~Element,~Accumulator,~Logical,scope,rows),scope_initial(~Element,~Accumulator,~Logical,scope),
    input_tree,input_length,output_tree,output_length) : Operations.Outcome<OperationsLaws.PairOwners(Element)>},
  +input_fact: Input(~Element,~Accumulator,~Logical,~original,scope,first,rows,Buffer.Content{input_tree,input_length}),
  +ready: Buffer.ContentReady<Element,Buffer.Content{output_tree,output_length}>,
  +capacity_ok: {U32.is_le(Tree.capacity(Element,output_tree),2147483648) == True{} : Bool},
  +room: {Nat.is_le(output_length,U32.to_nat(Tree.capacity(Element,output_tree))) == True{} : Bool},
  +exact: {output_length == U32.to_nat(BandViews.known(result)) : Nat},
  +count: ShapeLaws.Count(result,ShapeLaws.elements(Views.axis_extents(scope_kept(~Element,~Accumulator,~Logical,scope,rows)))),
  accepted: {GatherValues.accepted_owners(~Element,outcome) == True{} : Bool}) ->
  Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),scaled_word(~Element,~Accumulator,~Logical,scope,rows),
    CopyValues.right_content(~Element,outcome)):
  match valid result:
    case True{} Some{+value}:
      +model = OperationsLaws.runs_model(~Element,~Accumulator,~operation,~finish,U32.to_nat(value),scope_kept(~Element,~Accumulator,~Logical,scope,rows),
        scope_offset(~Element,~Accumulator,~Logical,scope,rows),Operations.line_runs(scope_line(~Element,~Accumulator,~Logical,scope,rows)),scope_initial(~Element,~Accumulator,~Logical,scope),
        input_tree,output_tree)
      %Equal.sym(Operations.Outcome<OperationsLaws.PairOwners(Element)>,outcome,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,model)},observed) :
        Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),scaled_word(~Element,~Accumulator,~Logical,scope,rows),
          CopyValues.right_content(~Element,_))
      %Equal.sym(Buffer.Content<Element>,CopyValues.right_content(~Element,Operations.Accepted{OperationsLaws.modeled_pair(~Element,input_length,output_length,model)}),
        Buffer.Content{CopyValues.output_tree(~Element,model),output_length},modeled_right(~Element,input_length,output_length,model)) :
        Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),scaled_word(~Element,~Accumulator,~Logical,scope,rows),_)
      %Equal.sym(Nat,output_length,U32.to_nat(value),exact) :
        Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),scaled_word(~Element,~Accumulator,~Logical,scope,rows),
          Buffer.Content{CopyValues.output_tree(~Element,model),_})
      band_output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,first,rows,input_tree,input_length,input_fact,output_tree,value,
        %exact : Buffer.ContentReady<Element,Buffer.Content{output_tree,_}> ready,
        capacity_ok,
        %exact : {Nat.is_le(_,U32.to_nat(Tree.capacity(Element,output_tree))) == True{} : Bool} room,
        Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(scope_kept(~Element,~Accumulator,~Logical,scope,rows))),U32.to_nat(value),count))
    case True{} None{}:
      Empty.absurd(Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),
        scaled_word(~Element,~Accumulator,~Logical,scope,rows),CopyValues.right_content(~Element,outcome)),
        WordEncoding.false_true(%observed : {GatherValues.accepted_owners(~Element,_) == True{} : Bool} accepted))
    case False{} Some{value}:
      Empty.absurd(Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),
        scaled_word(~Element,~Accumulator,~Logical,scope,rows),CopyValues.right_content(~Element,outcome)),
        WordEncoding.false_true(%observed : {GatherValues.accepted_owners(~Element,_) == True{} : Bool} accepted))
    case False{} None{}:
      Empty.absurd(Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),
        scaled_word(~Element,~Accumulator,~Logical,scope,rows),CopyValues.right_content(~Element,outcome)),
        WordEncoding.false_true(%observed : {GatherValues.accepted_owners(~Element,_) == True{} : Bool} accepted))

# The call on two storage models: its outcome is the one of the lines model.
def packed_output(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+output: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Accumulator,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),
  +output_ready: Buffer.Sized<Element,output>,
  +exact: {Buffer.length(~Element,output) == U32.to_nat(reduced_count(~Element,~Accumulator,~Logical,scope,rows)) : Nat},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,
    reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output)))) == True{} : Bool}) ->
  Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),scaled_word(~Element,~Accumulator,~Logical,scope,rows),
    CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,
      reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output))))):
  Scope{+logical,+view_axes,+offset,+before,+repeats,+after,+start,+batch,+channels,+height,+width,+reduced,+initial,+fill,+limits,+words,+classified,+domain,+within} = scope
  Buffer.Model{+input_tree,+input_length,Certificate.Certificate{+input_depth,+input_equation}} = input
  Buffer.Model{+output_tree,+output_length,Certificate.Certificate{+output_depth,+output_equation}} = output
  Buffer.Sized{+depth_ok,+output_room} = output_ready
  +scope = {Scope{logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,reduced,initial,fill,limits,words,classified,domain,within} :
    Scope<Element,Accumulator,Logical>}
  +input = {Buffer.Model{input_tree,input_length,Certificate.Certificate{input_depth,input_equation}} : Buffer.Model<Element>}
  +output = {Buffer.Model{output_tree,output_length,Certificate.Certificate{output_depth,output_equation}} : Buffer.Model<Element>}
  +axes = local_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},rows)
  +kept = scope_kept(~Element,~Accumulator,~Logical,scope,rows)
  +line = scope_line(~Element,~Accumulator,~Logical,scope,rows)
  +local_offset = scope_offset(~Element,~Accumulator,~Logical,scope,rows)
  +numel = Views.numel(Views.View{kept,0})
  +valid = OperationsLaws.lines_fit(axes,reduced,local_offset,input_length,Storage.capacity(~Element,Tree.pack(Element,input_tree),Certificate.Certificate{input_depth,input_equation}),
    output_length,Storage.capacity(~Element,Tree.pack(Element,output_tree),Certificate.Certificate{output_depth,output_equation}))
  -calls = reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),Buffer.pack(~Element,output))
  -stored = OperationsLaws.lines_model(~Element,~Accumulator,~operation,~finish,valid,numel,kept,line,local_offset,initial,
    Tree.reflect(Element,Tree.pack(Element,input_tree)),input_length,Tree.reflect(Element,Tree.pack(Element,output_tree)),output_length)
  -inputs = OperationsLaws.lines_model(~Element,~Accumulator,~operation,~finish,valid,numel,kept,line,local_offset,initial,input_tree,input_length,
    Tree.reflect(Element,Tree.pack(Element,output_tree)),output_length)
  -trees = OperationsLaws.lines_model(~Element,~Accumulator,~operation,~finish,valid,numel,kept,line,local_offset,initial,input_tree,input_length,output_tree,output_length)
  +packed = Equal.trans(Operations.Outcome<OperationsLaws.PairOwners(Element)>,stored,inputs,trees,
    Equal.cong(Tree.Tree<Element>,Operations.Outcome<OperationsLaws.PairOwners(Element)>,tree => OperationsLaws.lines_model(~Element,~Accumulator,~operation,~finish,valid,numel,kept,line,
      local_offset,initial,tree,input_length,Tree.reflect(Element,Tree.pack(Element,output_tree)),output_length),Tree.reflect(Element,Tree.pack(Element,input_tree)),input_tree,
      StorageLaws.reflect_packed(Element,input_tree)),
    Equal.cong(Tree.Tree<Element>,Operations.Outcome<OperationsLaws.PairOwners(Element)>,tree => OperationsLaws.lines_model(~Element,~Accumulator,~operation,~finish,valid,numel,kept,line,
      local_offset,initial,input_tree,input_length,tree,output_length),Tree.reflect(Element,Tree.pack(Element,output_tree)),output_tree,StorageLaws.reflect_packed(Element,output_tree)))
  +observed = Equal.trans(Operations.Outcome<OperationsLaws.PairOwners(Element)>,OperationsLaws.observe_pair_outcome(~Element,calls),stored,trees,
    OperationsLaws.lines(~Element,~Accumulator,~operation,~finish,Buffer.pack(~Element,input),axes,reduced,local_offset,initial,Buffer.pack(~Element,output)),packed)
  modeled_output(~Element,~Accumulator,~Logical,~original,~operation,~finish,valid,numel,scope,first,rows,input_tree,input_length,output_tree,output_length,
    OperationsLaws.observe_pair_outcome(~Element,calls),observed,input_fact,
    Buffer.content_ready(~Element,output,Buffer.Sized{depth_ok,output_room}),Buffer.capacity_bound(~Element,output,depth_ok),output_room,exact,
    ShapeLaws.numel(Views.axis_extents(kept)),accepted)

# The output buffer of the call, named by its storage model.
def buffered_output(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Accumulator,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),
  -output: Storage.Buffer<Element>,observation: Buffer.Observation<Element,output>,
  ready: Buffer.ContentReady<Element,Buffer.contents(~Element,output)>,
  exact: {Buffer.content_length(~Element,Buffer.contents(~Element,output)) == U32.to_nat(reduced_count(~Element,~Accumulator,~Logical,scope,rows)) : Nat},
  accepted: {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,
    reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),output))) == True{} : Bool}) ->
  Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),scaled_word(~Element,~Accumulator,~Logical,scope,rows),
    CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,
      reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),output)))):
  Buffer.Observation{+model,+equation} = observation
  +count = reduced_count(~Element,~Accumulator,~Logical,scope,rows)
  %equation : Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,scaled_word(~Element,~Accumulator,~Logical,scope,first),
    scaled_word(~Element,~Accumulator,~Logical,scope,rows),
    CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,
      reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),_))))
  +model_ready = {
    %Buffer.packed_contents(~Element,model) : Buffer.ContentReady<Element,_>
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),output,equation) : Buffer.ContentReady<Element,Buffer.contents(~Element,_)>
    ready : Buffer.ContentReady<Element,Buffer.content(~Element,model)>}
  +model_exact = Equal.trans(Nat,Buffer.length(~Element,model),Buffer.content_length(~Element,Buffer.content(~Element,model)),U32.to_nat(count),
    Equal.sym(Nat,Buffer.content_length(~Element,Buffer.content(~Element,model)),Buffer.length(~Element,model),Buffer.content_length_of(~Element,model)),
    %Buffer.packed_contents(~Element,model) : {Buffer.content_length(~Element,_) == U32.to_nat(count) : Nat}
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),output,equation) : {Buffer.content_length(~Element,Buffer.contents(~Element,_)) == U32.to_nat(count) : Nat}
    exact)
  packed_output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,first,rows,input,model,input_fact,
    Buffer.ready_content(~Element,model,model_ready),model_exact,
    %Equal.sym(Storage.Buffer<Element>,Buffer.pack(~Element,model),output,equation) :
      {GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,
        reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),_))) == True{} : Bool}
    accepted)

# The checked band of a call is usable exactly when the call was accepted.
def usable_accepted(~Element: Data,outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+first: U32,+rows: U32) ->
  {Each.usable(~Element,Each.checked(~Element,BandReduce.checked(~Element,first,rows,outcome))) ==
    GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome)) : Bool}:
  match outcome:
    case Operations.Accepted{(input,output)}: {==}
    case Operations.Rejected{(input,output)}: {==}

# An accepted call leaves the band of its output.
def made_output(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  outcome: Operations.Outcome<(Storage.Buffer<Element> & Storage.Buffer<Element>)>,+scope: Scope<Element,Accumulator,Logical>,+first: U32,+rows: U32,
  facts: Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,first,rows,CopyValues.right_content(~Element,OperationsLaws.observe_pair_outcome(~Element,outcome))),
  accepted: {Each.usable(~Element,Each.checked(~Element,BandReduce.checked(~Element,first,rows,outcome))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope),
    Model.contents(~Element,Each.made(~Element,Each.checked(~Element,BandReduce.checked(~Element,first,rows,outcome))))):
  match outcome:
    case Operations.Accepted{(input,Storage.Buffer{storage,length,certificate})}: facts
    case Operations.Rejected{(input,output)}:
      Empty.absurd(Model.ContentAll(~Element,~Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope),
        Model.contents(~Element,Each.made(~Element,Each.checked(~Element,BandReduce.checked(~Element,first,rows,Operations.Rejected{(input,output)}))))),
        WordEncoding.false_true(accepted))

# A band of rows folds the lines of its local view into fresh storage of its
# kept axes.
def folded_output(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Accumulator,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),
  accepted: {Each.usable(~Element,Each.checked(~Element,BandReduce.folded_rows(~Element,~Accumulator,~operation,~finish,False{},scope_banded(~Element,~Accumulator,~Logical,scope),
    scope_reduced(~Element,~Accumulator,~Logical,scope),input_geometry(~Element,~Accumulator,~Logical,scope),scope_initial(~Element,~Accumulator,~Logical,scope),
    scope_fill(~Element,~Accumulator,~Logical,scope),first,rows,Buffer.pack(~Element,input)))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,
    BandReduce.folded_rows(~Element,~Accumulator,~operation,~finish,False{},scope_banded(~Element,~Accumulator,~Logical,scope),
      scope_reduced(~Element,~Accumulator,~Logical,scope),input_geometry(~Element,~Accumulator,~Logical,scope),scope_initial(~Element,~Accumulator,~Logical,scope),
      scope_fill(~Element,~Accumulator,~Logical,scope),first,rows,Buffer.pack(~Element,input)))))):
  Scope{+logical,+view_axes,+offset,+before,+repeats,+after,+start,+batch,+channels,+height,+width,+reduced,+initial,+fill,+limits,+words,+classified,+domain,+within} = scope
  Buffer.Model{+tree,+length,+certificate} = input
  Shape.Both{Shape.Both{+fields,+ready},+values} = input_fact
  +scope = {Scope{logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,reduced,initial,fill,limits,words,classified,domain,within} :
    Scope<Element,Accumulator,Logical>}
  +input = {Buffer.Model{tree,length,certificate} : Buffer.Model<Element>}
  +accepted = accepted
  +scale = BandViews.product_or_zero(repeats)
  +first_scaled = (first * scale : U32)
  +rows_scaled = (rows * scale : U32)
  +kept = scope_kept(~Element,~Accumulator,~Logical,scope,rows)
  +count = reduced_count(~Element,~Accumulator,~Logical,scope,rows)
  +bound = known_bound(Views.numel(Views.View{kept,0}),before,repeats,after,start,batch,channels,height,width,reduced,limits,words,domain,within,first,rows,length,fields,
    ShapeLaws.numel(Views.axis_extents(kept)))
  made_output(~Element,~Accumulator,~Logical,~original,~operation,~finish,reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),Tensors.buffer(~Element,fill,count)),scope,first_scaled,rows_scaled,
    buffered_output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,first,rows,input,Shape.Both{Shape.Both{fields,ready},values},
      Tensors.buffer(~Element,fill,count),Buffer.observe(~Element,Tensors.buffer(~Element,fill,count)),
      Buffer.native_content_ready(~Element,Tensors.buffer(~Element,fill,count),BandCount.allocated_ready(~Element,count,fill,Depth.bounded(count,bound))),
      Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Tensors.buffer(~Element,fill,count)),
        Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),fill),U32.to_nat(count)},AllocationLaws.contents(~Element,Tensors.depth_for(count),fill,U32.to_nat(count))),
      Equal.trans(Bool,GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),Tensors.buffer(~Element,fill,count)))),
        Each.usable(~Element,Each.checked(~Element,BandReduce.checked(~Element,first_scaled,rows_scaled,reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),Tensors.buffer(~Element,fill,count))))),True{},
        Equal.sym(Bool,Each.usable(~Element,Each.checked(~Element,BandReduce.checked(~Element,first_scaled,rows_scaled,reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),Tensors.buffer(~Element,fill,count))))),
          GatherValues.accepted_owners(~Element,OperationsLaws.observe_pair_outcome(~Element,reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),Tensors.buffer(~Element,fill,count)))),usable_accepted(~Element,reduced_call(~Element,~Accumulator,~Logical,~operation,~finish,scope,rows,Buffer.pack(~Element,input),Tensors.buffer(~Element,fill,count)),first_scaled,rows_scaled)),accepted)),
    accepted)

# A band of no rows leaves an empty output band.
def empty_output(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,+empty: {U32.is_zero(rows) == True{} : Bool}) ->
  Model.ContentAll(~Element,~Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,
    BandReduce.folded_rows(~Element,~Accumulator,~operation,~finish,True{},scope_banded(~Element,~Accumulator,~Logical,scope),
      scope_reduced(~Element,~Accumulator,~Logical,scope),input_geometry(~Element,~Accumulator,~Logical,scope),scope_initial(~Element,~Accumulator,~Logical,scope),
      scope_fill(~Element,~Accumulator,~Logical,scope),first,rows,Buffer.pack(~Element,input)))))):
  Scope{+logical,+view_axes,+offset,+before,+repeats,+after,+start,+batch,+channels,+height,+width,+reduced,+initial,+fill,limits,words,classified,domain,within} = scope
  +output = BandReduce.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},reduced)
  +scale = BandViews.product_or_zero(repeats)
  +lines = {Lines{initial,Ordered{logical,Some{Views.View{view_axes,offset}}},line_elements(after,reduced)} : Lines<Accumulator,Ordered<Logical>>}
  +fresh = {Buffer.Content{Tree.fresh(Element,Tensors.depth_for(0),fill),0n} : Buffer.Content<Element>}
  +no_cells = {
    %Equal.sym(U32,rows,0,Arithmetic.equal_of_is_eq(rows,0,empty)) : {BandValues.cell_count(output,(_ * scale : U32)) == 0n : Nat}
    %Equal.sym(U32,(0 * scale : U32),0,Arithmetic.multiply_zero_left(scale)) : {BandValues.cell_count(output,_) == 0n : Nat}
    RowMajor.multiply_zero_right(BandValues.channels(output)) : {BandValues.cell_count(output,(rows * scale : U32)) == 0n : Nat}}
  Shape.Both{Contract.empty_ready(~Element,fill),
    %Equal.sym(Buffer.Content<Element>,Buffer.contents(~Element,Tensors.buffer(~Element,fill,0)),fresh,Contract.empty_contents(~Element,fill)) :
      BandValues.band_values(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
        lines,output,(first * scale : U32),(rows * scale : U32),_)
    BandValues.empty_leaf(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
      lines,output,(first * scale : U32),(rows * scale : U32),Tree.fresh(Element,Tensors.depth_for(0),fill),0n,no_cells)}

# A band of rows folds its lines; a band of none is empty.
def rows_output(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  empty: Bool,+scope: Scope<Element,Accumulator,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Accumulator,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),
  +compared: {U32.is_zero(rows) == empty : Bool},
  accepted: {Each.usable(~Element,Each.checked(~Element,BandReduce.folded_rows(~Element,~Accumulator,~operation,~finish,empty,scope_banded(~Element,~Accumulator,~Logical,scope),
    scope_reduced(~Element,~Accumulator,~Logical,scope),input_geometry(~Element,~Accumulator,~Logical,scope),scope_initial(~Element,~Accumulator,~Logical,scope),
    scope_fill(~Element,~Accumulator,~Logical,scope),first,rows,Buffer.pack(~Element,input)))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,
    BandReduce.folded_rows(~Element,~Accumulator,~operation,~finish,empty,scope_banded(~Element,~Accumulator,~Logical,scope),
      scope_reduced(~Element,~Accumulator,~Logical,scope),input_geometry(~Element,~Accumulator,~Logical,scope),scope_initial(~Element,~Accumulator,~Logical,scope),
      scope_fill(~Element,~Accumulator,~Logical,scope),first,rows,Buffer.pack(~Element,input)))))):
  match empty:
    case True{}: empty_output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,first,rows,input,compared)
    case False{}: folded_output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,first,rows,input,input_fact,accepted)

# One band through its checked call: a usable call leaves the output band.
def leaf_output(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>,+first: U32,+rows: U32,+input: Buffer.Model<Element>,
  +input_fact: Input(~Element,~Accumulator,~Logical,~original,scope,first,rows,Buffer.content(~Element,input)),
  accepted: {Each.usable(~Element,Each.checked(~Element,Bands.transform_band(~Element,~BandReduce.Context<Accumulator,Element>,~BandReduce.leaf(~Element,~Accumulator,~operation,~finish),
    scope_context(~Element,~Accumulator,~Logical,scope),first,rows,Buffer.pack(~Element,input),Unit{}))) == True{} : Bool}) ->
  Model.ContentAll(~Element,~Output(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope),Model.contents(~Element,Each.made(~Element,Each.checked(~Element,
    Bands.transform_band(~Element,~BandReduce.Context<Accumulator,Element>,~BandReduce.leaf(~Element,~Accumulator,~operation,~finish),
      scope_context(~Element,~Accumulator,~Logical,scope),first,rows,Buffer.pack(~Element,input),Unit{}))))):
  Scope{+logical,+view_axes,+offset,+before,+repeats,+after,+start,+batch,+channels,+height,+width,+reduced,+initial,+fill,+limits,+words,+classified,+domain,+within} = scope
  +scope = {Scope{logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,reduced,initial,fill,limits,words,classified,domain,within} :
    Scope<Element,Accumulator,Logical>}
  rows_output(~Element,~Accumulator,~Logical,~original,~operation,~finish,U32.is_zero(rows),scope,first,rows,input,input_fact,{==},accepted)

# ---- Every band ----

def Reduced(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +scope: Scope<Element,Accumulator,Logical>,-tensor: Tensors.Tensor<Element>) -> Data:
  Global.Contract(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
    scope_lines(~Element,~Accumulator,~Logical,scope),tensor)

# A Normal result whose output k holds finish of the fold of line k of the
# view, from the initial value.
def Lined(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+view: Views.View,+initial: Accumulator,+length: Nat,-tensor: Tensors.Tensor<Element>) -> Data:
  Global.Contract(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
    Lines{initial,Ordered{logical,Some{view}},length},tensor)

# Every band through its checked call, on the plan of its own bands.
def scheduled_lines(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+name: String,+operations: F32,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +reduced: Nat,+initial: Accumulator,+fill: Element,+shape: List<&2,U32>,+workers: U32,+limits: Machine.Limits,
  +words: ViewValues.Words(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),
  +classified: Classified.Classified<view_axes,offset,Bands.Geometry{batch,channels,height,width},BandViews.Banded{before,repeats,after,start}>,
  +domain: BandGather.Domain<BandReduce.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},reduced),limits,Machine.capacity(limits)>,
  +within: {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == True{} : Bool},
  +model: Model.Model<Element>,metadata: BandShape.Contents(~Element,Bands.Geometry{batch,channels,height,width},Model.content(~Element,model)),
  ready: Model.ContentReady(~Element,Model.content(~Element,model)),
  values: BandValues.Values(~Element,~Logical,~original,logical,Bands.Geometry{batch,channels,height,width},Model.content(~Element,model)),
  counted: Global.Counted<shape>,coherent: Global.Coherent(shape,BandReduce.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},reduced)),
  accepted: {Contract.tensor_ready(~Element,BandReduce.banded_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,BandViews.Banded{before,repeats,after,start},
    Bands.Geometry{batch,channels,height,width},shape,fill,workers,Model.pack(~Element,model))) == True{} : Bool}) ->
  Reduced(~Element,~Accumulator,~Logical,~original,~operation,~finish,
    Scope{logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,reduced,initial,fill,limits,words,classified,domain,within},
    BandReduce.banded_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,BandViews.Banded{before,repeats,after,start},
      Bands.Geometry{batch,channels,height,width},shape,fill,workers,Model.pack(~Element,model))):
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +scope = {Scope{logical,view_axes,offset,before,repeats,after,start,batch,channels,height,width,reduced,initial,fill,limits,words,classified,domain,within} :
    Scope<Element,Accumulator,Logical>}
  +context = {BandReduce.Context{banded,reduced,geometry,initial,fill} : BandReduce.Context<Accumulator,Element>}
  +output = BandReduce.output(banded,geometry,reduced)
  +schedule = Each.scheduled(~Element,~BandReduce.Context<Accumulator,Element>,~BandReduce.leaf(~Element,~Accumulator,~operation,~finish),context,geometry,workers,4.0,operations,model)
  +first_fact = Model.content_both(~Element,~Scope<Element,Accumulator,Logical>,
    ~Model.Both(~Element,~Scope<Element,Accumulator,Logical>,~input_fields(~Element,~Accumulator,~Logical),
      ~Model.always(~Element,~Scope<Element,Accumulator,Logical>,~Model.ready_band(~Element))),
    ~input_values(~Element,~Accumulator,~Logical,~original),scope,Model.content(~Element,model),
    Model.content_both(~Element,~Scope<Element,Accumulator,Logical>,~input_fields(~Element,~Accumulator,~Logical),
      ~Model.always(~Element,~Scope<Element,Accumulator,Logical>,~Model.ready_band(~Element)),scope,Model.content(~Element,model),metadata,ready),values)
  %Equal.sym(Bands.Bands(~Element) & Bool,
    BandSchedule.transform(~Element,~BandReduce.Context<Accumulator,Element>,~BandReduce.leaf(~Element,~Accumulator,~operation,~finish),context,geometry,workers,4.0,operations,
      Model.pack(~Element,model)),
    Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~BandReduce.Context<Accumulator,Element>,
      ~Bands.transform_band(~Element,~BandReduce.Context<Accumulator,Element>,~BandReduce.leaf(~Element,~Accumulator,~operation,~finish)),context,model)),schedule) :
    Reduced(~Element,~Accumulator,~Logical,~original,~operation,~finish,scope,Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},output,shape,fill,workers,_))
  Global.transformed(~Element,~Scope<Element,Accumulator,Logical>,~OutputValues(~Element,~Accumulator,~Logical,~original,~operation,~finish),
    ~Reduced(~Element,~Accumulator,~Logical,~original,~operation,~finish),scope,
    name,Tensors.InvalidArgument{shape},output,shape,fill,workers,limits,BandGather.rows_room(output,limits,Machine.capacity(limits),domain),
    Each.checked(~Element,Each.leaf_map(~Element,~Bands.Checked<Element>,~BandReduce.Context<Accumulator,Element>,
      ~Bands.transform_band(~Element,~BandReduce.Context<Accumulator,Element>,~BandReduce.leaf(~Element,~Accumulator,~operation,~finish)),context,model)),
    usable => Each.checked_all(~Element,~BandReduce.Context<Accumulator,Element>,
      ~Bands.transform_band(~Element,~BandReduce.Context<Accumulator,Element>,~BandReduce.leaf(~Element,~Accumulator,~operation,~finish)),
      ~Scope<Element,Accumulator,Logical>,~scope_context(~Element,~Accumulator,~Logical),~Input(~Element,~Accumulator,~Logical,~original),
      ~Output(~Element,~Accumulator,~Logical,~original,~operation,~finish),~leaf_output(~Element,~Accumulator,~Logical,~original,~operation,~finish),scope,model,first_fact,usable),
    counted,coherent,bands => normal => facts => Shape.Both{normal,facts},
    %schedule : {Contract.tensor_ready(~Element,Tensors.transformed(~Element,name,Tensors.InvalidArgument{shape},output,shape,fill,workers,_)) == True{} : Bool} accepted)

# ---- The kept axes and the line of the whole view ----

# The axes of a banded view up to its rows.
def whole_head(+banded: BandViews.Banded,+geometry: Bands.Geometry) -> List<&2,Views.Axis>:
  BandViews.Banded{+before,+repeats,after,offset} = banded
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  List.append(&2,Views.Axis,Classified.images_axes(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32)),
    List.append(&2,Views.Axis,before,BandViews.rows_axes(U32.is_lt(1,height),height,width,repeats)))

def whole_split(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32) ->
  {Classified.whole_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}) ==
    List.append(&2,Views.Axis,whole_head(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),after) : List<&2,Views.Axis>}:
  +images = Classified.images_axes(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32))
  +rowed = BandViews.rows_axes(U32.is_lt(1,height),height,width,repeats)
  +early = List.append(&2,Views.Axis,before,rowed)
  Equal.trans(List<&2,Views.Axis>,
    List.append(&2,Views.Axis,images,List.append(&2,Views.Axis,before,List.append(&2,Views.Axis,rowed,after))),
    List.append(&2,Views.Axis,images,List.append(&2,Views.Axis,early,after)),
    List.append(&2,Views.Axis,List.append(&2,Views.Axis,images,early),after),
    Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,rest => List.append(&2,Views.Axis,images,rest),
      List.append(&2,Views.Axis,before,List.append(&2,Views.Axis,rowed,after)),List.append(&2,Views.Axis,early,after),
      Equal.sym(List<&2,Views.Axis>,List.append(&2,Views.Axis,early,after),List.append(&2,Views.Axis,before,List.append(&2,Views.Axis,rowed,after)),
        LineValues.append_associative(~Views.Axis,before,rowed,after))),
    Equal.sym(List<&2,Views.Axis>,List.append(&2,Views.Axis,List.append(&2,Views.Axis,images,early),after),
      List.append(&2,Views.Axis,images,List.append(&2,Views.Axis,early,after)),LineValues.append_associative(~Views.Axis,images,early,after)))

# The whole count of kept axes is the head and the kept part after the rows.
def whole_count(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +reduced: Nat,+within: {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == True{} : Bool}) ->
  {Nat.sub(List.length(&2,Views.Axis,Classified.whole_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width})),reduced) ==
    Nat.add(List.length(&2,Views.Axis,whole_head(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width})),
      Nat.sub(List.length(&2,Views.Axis,after),reduced)) : Nat}:
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +whole = Classified.whole_axes(banded,geometry)
  +head = whole_head(banded,geometry)
  +taken = Nat.sub(List.length(&2,Views.Axis,after),reduced)
  %Equal.sym(List<&2,Views.Axis>,whole,List.append(&2,Views.Axis,head,after),whole_split(before,repeats,after,start,batch,channels,height,width)) :
    {Nat.sub(List.length(&2,Views.Axis,_),reduced) == Nat.add(List.length(&2,Views.Axis,head),taken) : Nat}
  %Equal.sym(Nat,List.length(&2,Views.Axis,List.append(&2,Views.Axis,head,after)),Nat.add(List.length(&2,Views.Axis,head),List.length(&2,Views.Axis,after)),
      ListReadback.append_length(~Views.Axis,head,after)) :
    {Nat.sub(_,reduced) == Nat.add(List.length(&2,Views.Axis,head),taken) : Nat}
  sub_after(List.length(&2,Views.Axis,head),List.length(&2,Views.Axis,after),reduced,within)

# The line of the whole view is the line of the axes after the rows.
def whole_line(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +reduced: Nat,+within: {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == True{} : Bool}) ->
  {OperationsLaws.line_axes(Classified.whole_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),reduced) == line_part(after,reduced) :
    List<&2,Views.Axis>}:
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +whole = Classified.whole_axes(banded,geometry)
  +head = whole_head(banded,geometry)
  +taken = Nat.sub(List.length(&2,Views.Axis,after),reduced)
  %Equal.sym(Nat,Nat.sub(List.length(&2,Views.Axis,whole),reduced),Nat.add(List.length(&2,Views.Axis,head),taken),
      whole_count(before,repeats,after,start,batch,channels,height,width,reduced,within)) :
    {List.drop(&2,Views.Axis,whole,_) == line_part(after,reduced) : List<&2,Views.Axis>}
  %Equal.sym(List<&2,Views.Axis>,whole,List.append(&2,Views.Axis,head,after),whole_split(before,repeats,after,start,batch,channels,height,width)) :
    {List.drop(&2,Views.Axis,_,Nat.add(List.length(&2,Views.Axis,head),taken)) == line_part(after,reduced) : List<&2,Views.Axis>}
  drop_after(~Views.Axis,head,after,taken)

# The kept axes of the whole view are the head and the kept part after the
# rows.
def whole_kept(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +reduced: Nat,+within: {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == True{} : Bool}) ->
  {OperationsLaws.kept_axes(Classified.whole_axes(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),reduced) ==
    List.append(&2,Views.Axis,whole_head(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}),kept_part(after,reduced)) :
    List<&2,Views.Axis>}:
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +whole = Classified.whole_axes(banded,geometry)
  +head = whole_head(banded,geometry)
  +taken = Nat.sub(List.length(&2,Views.Axis,after),reduced)
  %Equal.sym(Nat,Nat.sub(List.length(&2,Views.Axis,whole),reduced),Nat.add(List.length(&2,Views.Axis,head),taken),
      whole_count(before,repeats,after,start,batch,channels,height,width,reduced,within)) :
    {List.take(&2,Views.Axis,whole,_) == List.append(&2,Views.Axis,head,kept_part(after,reduced)) : List<&2,Views.Axis>}
  %Equal.sym(List<&2,Views.Axis>,whole,List.append(&2,Views.Axis,head,after),whole_split(before,repeats,after,start,batch,channels,height,width)) :
    {List.take(&2,Views.Axis,_,Nat.add(List.length(&2,Views.Axis,head),taken)) == List.append(&2,Views.Axis,head,kept_part(after,reduced)) : List<&2,Views.Axis>}
  take_after(~Views.Axis,head,after,taken)

# The head holds the images, the axes before the rows and the repeated rows.
def head_elements(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +fitted: ViewValues.Fitted<before,repeats,after,start,batch,channels,height,width>) ->
  {ShapeLaws.elements(Views.axis_extents(whole_head(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width}))) ==
    Nat.mul(U32.to_nat(batch),Nat.mul(ShapeLaws.elements(Views.axis_extents(before)),Nat.mul(U32.to_nat(height),ShapeLaws.elements(repeats)))) : Nat}:
  ViewValues.Fitted{+batch_positive,+height_positive,channel_reach,column_reach,early,late,repeated,rows_exact} = fitted
  +images = Classified.images_axes(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32))
  +rowed = BandViews.rows_axes(U32.is_lt(1,height),height,width,repeats)
  +early = List.append(&2,Views.Axis,before,rowed)
  +b = U32.to_nat(batch)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +hes = Nat.mul(U32.to_nat(height),ShapeLaws.elements(repeats))
  +target = Nat.mul(b,Nat.mul(eb,hes))
  Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,images,early))),
    Nat.mul(ShapeLaws.elements(Views.axis_extents(images)),ShapeLaws.elements(Views.axis_extents(early))),target,
    LineValues.axes_elements(images,early),
    %Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(images)),b,
        ViewValues.images_elements(U32.is_lt(1,batch),batch,(channels * (height * width : U32) : U32),{==},batch_positive)) :
      {Nat.mul(_,ShapeLaws.elements(Views.axis_extents(early))) == target : Nat}
    Equal.cong(Nat,Nat,value => Nat.mul(b,value),ShapeLaws.elements(Views.axis_extents(early)),Nat.mul(eb,hes),
      Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(early)),Nat.mul(eb,ShapeLaws.elements(Views.axis_extents(rowed))),Nat.mul(eb,hes),
        LineValues.axes_elements(before,rowed),
        Equal.cong(Nat,Nat,value => Nat.mul(eb,value),ShapeLaws.elements(Views.axis_extents(rowed)),hes,
          ViewValues.rows_elements(U32.is_lt(1,height),height,width,repeats,{==},height_positive)))))

# Extents whose elements fit a word are counted by their product.
def kept_word(+after: List<&2,Views.Axis>,+reduced: Nat,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  +room: {Nat.is_le(kept_elements(after,reduced),U32.to_nat(limit)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,kept_elements(after,reduced)) == True{} : Bool}) ->
  {U32.to_nat(BandViews.product_or_zero(Views.axis_extents(kept_part(after,reduced)))) == kept_elements(after,reduced) : Nat}:
  +extents = Views.axis_extents(kept_part(after,reduced))
  +ka = kept_elements(after,reduced)
  +complete = CountCompletion.product_complete(extents,1,limit,maximum,CountCompletion.positive_not_zero(extents,positive),
    %Equal.sym(Nat,Nat.add(ka,0n),ka,Addition.zero_right(ka)) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool} room)
  ViewValues.counted_exact(extents,
    %Equal.sym(Maybe<&2,U32>,Shapes.product(extents,Some{1}),Some{U32.from_nat(Nat.mul(U32.to_nat(1),ka))},complete) : {Views.has_count(_) == True{} : Bool} {==})

# The kept axes of the whole view hold the outputs, when they are a positive
# count.
def kept_whole(+before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,+batch: U32,+channels: U32,+height: U32,+width: U32,
  +reduced: Nat,+within: {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == True{} : Bool},
  +fitted: ViewValues.Fitted<before,repeats,after,start,batch,channels,height,width>,+count: U32,+limit: U32,+maximum: {limit == Machine.word_max() : U32},
  +exact: {ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(Classified.whole_axes(BandViews.Banded{before,repeats,after,start},
    Bands.Geometry{batch,channels,height,width}),reduced))) == U32.to_nat(count) : Nat},
  +nonzero: {U32.is_zero(count) == False{} : Bool}) ->
  {ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(Classified.whole_axes(BandViews.Banded{before,repeats,after,start},
    Bands.Geometry{batch,channels,height,width}),reduced))) ==
    BandGather.elements(BandReduce.output(BandViews.Banded{before,repeats,after,start},Bands.Geometry{batch,channels,height,width},reduced)) : Nat}:
  ViewValues.Fitted{batch_positive,height_positive,channel_reach,column_reach,+early,late,+repeated,+rows_exact} = fitted
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +head = whole_head(banded,geometry)
  +kept = OperationsLaws.kept_axes(Classified.whole_axes(banded,geometry),reduced)
  +scale = BandViews.product_or_zero(repeats)
  +b = U32.to_nat(batch)
  +h = U32.to_nat(height)
  +eb = ShapeLaws.elements(Views.axis_extents(before))
  +es = ShapeLaws.elements(repeats)
  +hes = Nat.mul(h,es)
  +ka = kept_elements(after,reduced)
  +outer = Nat.mul(b,Nat.mul(eb,hes))
  +product = Nat.mul(outer,ka)
  +target = Nat.mul(b,Nat.mul(eb,Nat.mul(hes,ka)))
  +split = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(kept)),ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,head,kept_part(after,reduced)))),product,
    Equal.cong(List<&2,Views.Axis>,Nat,axes => ShapeLaws.elements(Views.axis_extents(axes)),kept,List.append(&2,Views.Axis,head,kept_part(after,reduced)),
      whole_kept(before,repeats,after,start,batch,channels,height,width,reduced,within)),
    %head_elements(before,repeats,after,start,batch,channels,height,width,fitted) :
      {ShapeLaws.elements(Views.axis_extents(List.append(&2,Views.Axis,head,kept_part(after,reduced)))) == Nat.mul(_,ka) : Nat}
    LineValues.axes_elements(head,kept_part(after,reduced)))
  +product_count = Equal.trans(Nat,product,ShapeLaws.elements(Views.axis_extents(kept)),U32.to_nat(count),
    Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents(kept)),product,split),exact)
  +positive = {%Equal.sym(Nat,product,U32.to_nat(count),product_count) : {Nat.is_lt(0n,_) == True{} : Bool}
    Arithmetic.word_nonzero_positive(count,nonzero) : {Nat.is_lt(0n,product) == True{} : Bool}}
  +kept_positive = DivisionOrder.right_factor_positive(outer,ka,positive)
  +kept_below = Order.transitive(ka,product,U32.to_nat(limit),
    %RowMajor.multiply_commutative(ka,outer) : {Nat.is_le(ka,_) == True{} : Bool}
    RowMajor.product_covers(ka,outer,RowMajor.left_factor_positive(outer,ka,positive)),
    %Equal.sym(Nat,product,U32.to_nat(count),product_count) : {Nat.is_le(_,U32.to_nat(limit)) == True{} : Bool}
    Bounds.as_nat_le(count,limit,%Equal.sym(U32,limit,Machine.word_max(),maximum) : {U32.is_le(count,_) == True{} : Bool} FullRange.maximum(count)))
  +height_es = {%repeated : {U32.to_nat((height * scale : U32)) == Nat.mul(h,_) : Nat} rows_exact : {U32.to_nat((height * scale : U32)) == hes : Nat}}
  +rearranged = Equal.trans(Nat,product,Nat.mul(b,Nat.mul(Nat.mul(eb,hes),ka)),target,
    RowMajor.assoc(b,Nat.mul(eb,hes),ka),
    Equal.cong(Nat,Nat,value => Nat.mul(b,value),Nat.mul(Nat.mul(eb,hes),ka),Nat.mul(eb,Nat.mul(hes,ka)),RowMajor.assoc(eb,hes,ka)))
  %Equal.sym(Nat,U32.to_nat(BandViews.product_or_zero(Views.axis_extents(before))),eb,early) :
    {ShapeLaws.elements(Views.axis_extents(kept)) ==
      Nat.mul(b,Nat.mul(_,Nat.mul(U32.to_nat((height * scale : U32)),U32.to_nat(BandViews.product_or_zero(Views.axis_extents(kept_part(after,reduced))))))) : Nat}
  %Equal.sym(Nat,U32.to_nat((height * scale : U32)),hes,height_es) :
    {ShapeLaws.elements(Views.axis_extents(kept)) == Nat.mul(b,Nat.mul(eb,Nat.mul(_,U32.to_nat(BandViews.product_or_zero(Views.axis_extents(kept_part(after,reduced))))))) : Nat}
  %Equal.sym(Nat,U32.to_nat(BandViews.product_or_zero(Views.axis_extents(kept_part(after,reduced)))),ka,kept_word(after,reduced,limit,maximum,kept_below,kept_positive)) :
    {ShapeLaws.elements(Views.axis_extents(kept)) == Nat.mul(b,Nat.mul(eb,Nat.mul(hes,_))) : Nat}
  Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(kept)),product,target,split,rearranged)

# ---- The public lines ----

# A banded view whose reduced axes lie after the rows.
def InBand(view_axes: List<&2,Views.Axis>,offset: U32,geometry: Bands.Geometry,reduced: Nat,banded: BandViews.Banded) -> Data:
  BandViews.Banded{before,repeats,after,start} = banded
  Shape.Both<Classified.Classified<view_axes,offset,geometry,BandViews.Banded{before,repeats,after,start}>,
    {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == True{} : Bool}>

def InBands(view_axes: List<&2,Views.Axis>,offset: U32,geometry: Bands.Geometry,reduced: Nat,found: Maybe<&2,BandViews.Banded>) -> Data:
  match found:
    case Some{banded}: InBand(view_axes,offset,geometry,reduced,banded)
    case None{}: Unit

def kept_band(valid: Bool,+view_axes: List<&2,Views.Axis>,+offset: U32,+geometry: Bands.Geometry,+reduced: Nat,
  +before: List<&2,Views.Axis>,+repeats: List<&2,U32>,+after: List<&2,Views.Axis>,+start: U32,
  +compared: {Nat.is_le(reduced,List.length(&2,Views.Axis,after)) == valid : Bool},
  classified: Classified.Classified<view_axes,offset,geometry,BandViews.Banded{before,repeats,after,start}>) ->
  InBands(view_axes,offset,geometry,reduced,BandViews.kept_when(valid,BandViews.Banded{before,repeats,after,start})):
  match valid:
    case True{}: Shape.Both{classified,compared}
    case False{}: Unit{}

# A classification kept for its reduced axes.
def in_band(found: Maybe<&2,BandViews.Banded>,+view_axes: List<&2,Views.Axis>,+offset: U32,+geometry: Bands.Geometry,+reduced: Nat,
  classification: Classified.Classification(view_axes,offset,geometry,found)) -> InBands(view_axes,offset,geometry,reduced,BandReduce.in_band(found,reduced)):
  match found:
    case Some{BandViews.Banded{+before,+repeats,+after,+start}}:
      kept_band(Nat.is_le(reduced,List.length(&2,Views.Axis,after)),view_axes,offset,geometry,reduced,before,repeats,after,start,{==},classification)
    case None{}: Unit{}

# Bands whose images and rows a view keeps in place fold their own lines.
def banded_lines(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+name: String,+operations: F32,+initial: Accumulator,+reduced: Nat,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +banded: BandViews.Banded,+geometry: Bands.Geometry,+source_shape: List<&2,U32>,+shape: List<&2,U32>,+fill: Element,+workers: U32,+length: U32,+model: Model.Model<Element>,
  +inband: InBand(view_axes,offset,geometry,reduced,banded),
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,counted: Global.Counted<source_shape>,coherent: Global.Coherent(source_shape,geometry),
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +kept_same: {ShapeLaws.elements(shape) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(view_axes,reduced))) : Nat},
  +line_same: {U32.to_nat(length) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.line_axes(view_axes,reduced))) : Nat},
  accepted: {Contract.tensor_ready(~Element,BandReduce.banded_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,banded,geometry,shape,fill,workers,
    Model.pack(~Element,model))) == True{} : Bool}) ->
  Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,U32.to_nat(length),
    BandReduce.banded_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,banded,geometry,shape,fill,workers,Model.pack(~Element,model))):
  BandViews.Banded{+before,+repeats,+after,+start} = banded
  Bands.Geometry{+batch,+channels,+height,+width} = geometry
  Shape.Both{Classified.Classified{+axes_exact,+offset_exact,+fits},+within} = inband
  BandShape.WellFormed{metadata,ready,coverage} = well_formed
  +banded = {BandViews.Banded{before,repeats,after,start} : BandViews.Banded}
  +geometry = {Bands.Geometry{batch,channels,height,width} : Bands.Geometry}
  +classified = {Classified.Classified{axes_exact,offset_exact,fits} : Classified.Classified<view_axes,offset,geometry,banded>}
  +fitted = ViewValues.fitted(before,repeats,after,start,batch,channels,height,width,fits)
  +whole = Classified.whole_axes(banded,geometry)
  +output = BandReduce.output(banded,geometry,reduced)
  +exact = ViewValues.count_elements(shape,count,Count.count_of(shape,count,counts))
  +kept_view = {%axes_exact : {ShapeLaws.elements(shape) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(_,reduced))) : Nat} kept_same :
    {ShapeLaws.elements(shape) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(whole,reduced))) : Nat}}
  +kept_count = Equal.trans(Nat,ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(whole,reduced))),ShapeLaws.elements(shape),U32.to_nat(count),
    Equal.sym(Nat,ShapeLaws.elements(shape),ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(whole,reduced))),kept_view),exact)
  +coherent_output = Equal.trans(Nat,ShapeLaws.elements(shape),ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(whole,reduced))),BandGather.elements(output),
    kept_view,kept_whole(before,repeats,after,start,batch,channels,height,width,reduced,within,fitted,count,Machine.word_max(),{==},kept_count,nonzero))
  +line_exact = Equal.trans(Nat,line_elements(after,reduced),ShapeLaws.elements(Views.axis_extents(OperationsLaws.line_axes(whole,reduced))),U32.to_nat(length),
    Equal.cong(List<&2,Views.Axis>,Nat,axes => ShapeLaws.elements(Views.axis_extents(axes)),line_part(after,reduced),OperationsLaws.line_axes(whole,reduced),
      Equal.sym(List<&2,Views.Axis>,OperationsLaws.line_axes(whole,reduced),line_part(after,reduced),whole_line(before,repeats,after,start,batch,channels,height,width,reduced,within))),
    Equal.sym(Nat,U32.to_nat(length),ShapeLaws.elements(Views.axis_extents(OperationsLaws.line_axes(whole,reduced))),
      %axes_exact : {U32.to_nat(length) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.line_axes(_,reduced))) : Nat} line_same))
  %line_exact : Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,_,
    BandReduce.banded_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,banded,geometry,shape,fill,workers,Model.pack(~Element,model)))
  scheduled_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,name,operations,view_axes,offset,before,repeats,after,start,batch,channels,height,width,
    reduced,initial,fill,shape,workers,Default.limits(),
    ViewValues.words(before,repeats,after,start,batch,channels,height,width,source_shape,fitted,counted,coherent),classified,
    ViewValues.output_domain(shape,count,counts,nonzero,batch,BandViews.product_or_zero(Views.axis_extents(before)),(height * BandViews.product_or_zero(repeats) : U32),
      BandViews.product_or_zero(Views.axis_extents(kept_part(after,reduced))),coherent_output),
    within,model,metadata,ready,values,Global.Counted{count,counts},coherent_output,accepted)

# Arrived bands are read through their model.
def arrived_lines(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+name: String,+operations: F32,+initial: Accumulator,+reduced: Nat,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +banded: BandViews.Banded,+whole: Bands.Geometry,+source_shape: List<&2,U32>,+shape: List<&2,U32>,+fill: Element,+workers: U32,+length: U32,
  -bands: Bands.Bands(~Element),observation: Model.Observation<Element,bands>,arrived: Arrive.Arrived(~Element,~Logical,~original,logical,whole,Bands.unsplit(whole),bands),
  +inband: InBand(view_axes,offset,whole,reduced,banded),counted: Global.Counted<source_shape>,coherent: Global.Coherent(source_shape,whole),
  +count: U32,+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +kept_same: {ShapeLaws.elements(shape) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(view_axes,reduced))) : Nat},
  +line_same: {U32.to_nat(length) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.line_axes(view_axes,reduced))) : Nat},
  accepted: {Contract.tensor_ready(~Element,BandReduce.banded_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,banded,whole,shape,fill,workers,bands)) ==
    True{} : Bool}) ->
  Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,U32.to_nat(length),
    BandReduce.banded_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,banded,whole,shape,fill,workers,bands)):
  Model.Observation{+source,+equation} = observation
  Shape.Both{same,Shape.Both{well_formed,values}} = arrived
  +contents = Model.observed_contents(~Element,source,bands,equation)
  %equation : Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,U32.to_nat(length),
    BandReduce.banded_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,banded,whole,shape,fill,workers,_))
  banded_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,name,operations,initial,reduced,view_axes,offset,banded,whole,source_shape,shape,fill,workers,length,
    source,inband,%contents : BandShape.WellFormed<Element,whole,_> well_formed,counted,coherent,
    %contents : BandValues.Values(~Element,~Logical,~original,logical,whole,_) values,count,counts,nonzero,kept_same,line_same,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,source),bands,equation) :
      {Contract.tensor_ready(~Element,BandReduce.banded_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,banded,whole,shape,fill,workers,_)) == True{} : Bool}
      accepted)

# Other bands first arrive as the one band of the whole geometry, whose view
# is classified again.
def whole_lines(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  found: Maybe<&2,BandViews.Banded>,+logical: Logical,+name: String,+operations: F32,+initial: Accumulator,+reduced: Nat,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +model: Model.Model<Element>,+geometry: Bands.Geometry,+source_shape: List<&2,U32>,+source_fill: Element,+source_workers: U32,
  +shape: List<&2,U32>,+fill: Element,+workers: U32,+length: U32,
  +inbands: InBands(view_axes,offset,Bands.Geometry{1,1,1,Tensors.count_or_zero(source_shape)},reduced,found),
  well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<source_shape>,coherent: Global.Coherent(source_shape,geometry),
  values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +kept_same: {ShapeLaws.elements(shape) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(view_axes,reduced))) : Nat},
  +line_same: {U32.to_nat(length) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.line_axes(view_axes,reduced))) : Nat},
  accepted: {Contract.tensor_ready(~Element,BandReduce.whole_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,found,
    Bands.Geometry{1,1,1,Tensors.count_or_zero(source_shape)},shape,fill,workers,
    Tensors.Tensor{Model.pack(~Element,model),geometry,source_shape,source_fill,Tensors.Usable{},source_workers})) == True{} : Bool}) ->
  Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,U32.to_nat(length),
    BandReduce.whole_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,found,Bands.Geometry{1,1,1,Tensors.count_or_zero(source_shape)},shape,fill,workers,
      Tensors.Tensor{Model.pack(~Element,model),geometry,source_shape,source_fill,Tensors.Usable{},source_workers})):
  match found:
    case Some{+banded}:
      +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(source_shape)} : Bands.Geometry}
      +arrival = WholeLeaf.whole_arrival(source_shape,counted)
      arrived_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,name,operations,initial,reduced,view_axes,offset,banded,whole,source_shape,shape,fill,workers,length,
        Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.Tensor{Model.pack(~Element,model),geometry,source_shape,source_fill,Tensors.Usable{},source_workers}),Model.observe(~Element,Tensors.arrive(~Element,whole,Bands.unsplit(whole),Tensors.Tensor{Model.pack(~Element,model),geometry,source_shape,source_fill,Tensors.Usable{},source_workers})),
        Arrive.arrived(~Element,~Logical,~original,logical,source_shape,whole,Bands.unsplit(whole),geometry,source_fill,Tensors.Usable{},source_workers,model,arrival,
          well_formed,coherent,values),
        inbands,counted,Target.arrival_coherent(source_shape,whole,Bands.unsplit(whole),arrival),count,counts,nonzero,kept_same,line_same,accepted)
    case None{}:
      Empty.absurd(Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,U32.to_nat(length),
        BandReduce.whole_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,None{},Bands.Geometry{1,1,1,Tensors.count_or_zero(source_shape)},shape,fill,workers,
          Tensors.Tensor{Model.pack(~Element,model),geometry,source_shape,source_fill,Tensors.Usable{},source_workers})),WordEncoding.false_true(accepted))

# The view is classified by the geometry of the input first.
def found_lines(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  found: Maybe<&2,BandViews.Banded>,+logical: Logical,+name: String,+operations: F32,+initial: Accumulator,+reduced: Nat,+view_axes: List<&2,Views.Axis>,+offset: U32,
  +model: Model.Model<Element>,+geometry: Bands.Geometry,+source_shape: List<&2,U32>,+source_fill: Element,+source_workers: U32,
  +shape: List<&2,U32>,+fill: Element,+workers: U32,+length: U32,
  +inbands: InBands(view_axes,offset,geometry,reduced,found),
  +well_formed: BandShape.WellFormed<Element,geometry,Model.content(~Element,model)>,+counted: Global.Counted<source_shape>,+coherent: Global.Coherent(source_shape,geometry),
  +values: BandValues.Values(~Element,~Logical,~original,logical,geometry,Model.content(~Element,model)),
  +count: U32,+counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},+nonzero: {U32.is_zero(count) == False{} : Bool},
  +kept_same: {ShapeLaws.elements(shape) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(view_axes,reduced))) : Nat},
  +line_same: {U32.to_nat(length) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.line_axes(view_axes,reduced))) : Nat},
  accepted: {Contract.tensor_ready(~Element,BandReduce.found_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,found,view_axes,offset,
    Model.pack(~Element,model),geometry,source_shape,source_fill,source_workers,shape,fill,workers)) == True{} : Bool}) ->
  Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,U32.to_nat(length),
    BandReduce.found_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,found,view_axes,offset,
      Model.pack(~Element,model),geometry,source_shape,source_fill,source_workers,shape,fill,workers)):
  match found:
    case Some{+banded}:
      banded_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,name,operations,initial,reduced,view_axes,offset,banded,geometry,source_shape,shape,fill,workers,
        length,model,inbands,well_formed,counted,coherent,values,count,counts,nonzero,kept_same,line_same,accepted)
    case None{}:
      +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(source_shape)} : Bands.Geometry}
      whole_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,BandReduce.in_band(BandViews.classify(view_axes,offset,whole),reduced),logical,name,operations,initial,
        reduced,view_axes,offset,model,geometry,source_shape,source_fill,source_workers,shape,fill,workers,length,
        in_band(BandViews.classify(view_axes,offset,whole),view_axes,offset,whole,reduced,Classified.classify(view_axes,offset,whole)),
        well_formed,counted,coherent,values,count,counts,nonzero,kept_same,line_same,accepted)

# ---- Lines of no value, or no output ----

def empty_cells(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  remaining: Nat,+position: Nat,+logical: Logical,+view: Views.View,+initial: Accumulator,+content: Buffer.Content<Element>,
  cells: Members.All(Nat,{index => Contract.At(~Element,~Element,~Full.constant(~Element),finish(initial),content,index) : Nat -> Data},Ranges.range(remaining,position))) ->
  Members.All(Nat,{index => Contract.At(~Element,~Lines<Accumulator,Ordered<Logical>>,
    ~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),Lines{initial,Ordered{logical,Some{view}},0n},content,index) :
    Nat -> Data},Ranges.range(remaining,position)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      Shape.Both{here,later} = cells
      Shape.Both{here,empty_cells(~Element,~Accumulator,~Logical,~original,~operation,~finish,rest,1n+position,logical,view,initial,content,later)}

# Outputs of no value hold finish of the initial value; no output holds none.
def empty_facts(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  count_zero: Bool,+logical: Logical,+view: Views.View,+count: U32,+length: U32,+initial: Accumulator,+shape: List<&2,U32>,+content: Buffer.Content<Element>,
  +count_classified: {U32.is_zero(count) == count_zero : Bool},
  +empty: {count_zero || U32.is_zero(length) == True{} : Bool},
  exact: {Buffer.content_length(~Element,content) == U32.to_nat(count) : Nat},
  facts: GatherValues.Facts(~Element,~Element,~Full.constant(~Element),finish(initial),shape,content)) ->
  GatherValues.Facts(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
    Lines{initial,Ordered{logical,Some{view}},U32.to_nat(length)},shape,content):
  match count_zero:
    case True{}:
      Shape.Both{formed,cells} = facts
      +none = {%Arithmetic.word_zero(count,count_classified) : {Buffer.content_length(~Element,content) == U32.to_nat(_) : Nat} exact :
        {Buffer.content_length(~Element,content) == 0n : Nat}}
      Shape.Both{formed,
        %Equal.sym(Nat,Buffer.content_length(~Element,content),0n,none) :
          Members.All(Nat,{index => Contract.At(~Element,~Lines<Accumulator,Ordered<Logical>>,
            ~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
            Lines{initial,Ordered{logical,Some{view}},U32.to_nat(length)},content,index) : Nat -> Data},Ranges.range(_,0n))
        Unit{}}
    case False{}:
      Shape.Both{formed,cells} = facts
      %Equal.sym(U32,length,0,Arithmetic.word_zero(length,empty)) :
        GatherValues.Facts(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
          Lines{initial,Ordered{logical,Some{view}},U32.to_nat(_)},shape,content)
      Shape.Both{formed,empty_cells(~Element,~Accumulator,~Logical,~original,~operation,~finish,Buffer.content_length(~Element,content),0n,logical,view,initial,content,cells)}

def empty_lines(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+view: Views.View,+count: U32,+length: U32,+initial: Accumulator,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},+empty: {U32.is_zero(count) || U32.is_zero(length) == True{} : Bool}) ->
  Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,view,initial,U32.to_nat(length),
    Tensors.whole(~Element,Tensors.buffer(~Element,finish(initial),count),shape,fill,Tensors.Usable{},workers)):
  WholeLeaf.stored(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
    Lines{initial,Ordered{logical,Some{view}},U32.to_nat(length)},
    Tensors.buffer(~Element,finish(initial),count),shape,fill,workers,Buffer.observe(~Element,Tensors.buffer(~Element,finish(initial),count)),
    empty_facts(~Element,~Accumulator,~Logical,~original,~operation,~finish,U32.is_zero(count),logical,view,count,length,initial,shape,
      Buffer.contents(~Element,Tensors.buffer(~Element,finish(initial),count)),{==},empty,
      Equal.cong(Buffer.Content<Element>,Nat,Buffer.content_length(~Element),Buffer.contents(~Element,Tensors.buffer(~Element,finish(initial),count)),
        Buffer.Content{Tree.fresh(Element,Tensors.depth_for(count),finish(initial)),U32.to_nat(count)},
        AllocationLaws.contents(~Element,Tensors.depth_for(count),finish(initial),U32.to_nat(count))),
      Full.initial_counted(~Element,shape,finish(initial),count,Count.count_of(shape,count,counts))))

# ---- The call ----

# No output or lines of no value give every output finish of the initial
# value; otherwise the view of a usable tensor is classified.
def chosen_lines(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  count_zero: Bool,length_zero: Bool,+logical: Logical,+name: String,+operations: F32,+count: U32,+length: U32,+initial: Accumulator,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reduced: Nat,
  +model: Model.Model<Element>,+geometry: Bands.Geometry,+source_shape: List<&2,U32>,+source_fill: Element,+source_workers: U32,
  +count_classified: {U32.is_zero(count) == count_zero : Bool},+length_classified: {U32.is_zero(length) == length_zero : Bool},
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,model),geometry,source_shape,source_fill,Tensors.Usable{},source_workers}),
  +counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +kept_same: {ShapeLaws.elements(shape) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(view_axes,reduced))) : Nat},
  +line_same: {U32.to_nat(length) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.line_axes(view_axes,reduced))) : Nat},
  accepted: {Contract.tensor_ready(~Element,BandReduce.lines_when(~Element,~Accumulator,~operation,~finish,name,count_zero || length_zero,operations,count,initial,reduced,view_axes,offset,
    Model.pack(~Element,model),geometry,source_shape,source_fill,source_workers,shape,fill,workers)) == True{} : Bool}) ->
  Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,U32.to_nat(length),
    BandReduce.lines_when(~Element,~Accumulator,~operation,~finish,name,count_zero || length_zero,operations,count,initial,reduced,view_axes,offset,
      Model.pack(~Element,model),geometry,source_shape,source_fill,source_workers,shape,fill,workers)):
  match count_zero length_zero:
    case True{} length_zero:
      empty_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},count,length,initial,shape,fill,workers,counts,
        %Equal.sym(Bool,U32.is_zero(count),True{},count_classified) : {_ || U32.is_zero(length) == True{} : Bool} {==})
    case False{} True{}:
      empty_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},count,length,initial,shape,fill,workers,counts,
        %Equal.sym(Bool,U32.is_zero(count),False{},count_classified) : {_ || U32.is_zero(length) == True{} : Bool}
        %Equal.sym(Bool,U32.is_zero(length),True{},length_classified) : {False{} || _ == True{} : Bool} {==})
    case False{} False{}:
      Shape.Both{Shape.Both{well_formed,Shape.Both{+counted,+coherent}},values} = contract
      +contents = Model.packed_contents(~Element,model)
      found_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,BandReduce.in_band(BandViews.classify(view_axes,offset,geometry),reduced),logical,name,operations,initial,
        reduced,view_axes,offset,model,geometry,source_shape,source_fill,source_workers,shape,fill,workers,length,
        in_band(BandViews.classify(view_axes,offset,geometry),view_axes,offset,geometry,reduced,Classified.classify(view_axes,offset,geometry)),
        %contents : BandShape.WellFormed<Element,geometry,_> well_formed,counted,coherent,
        %contents : BandValues.Values(~Element,~Logical,~original,logical,geometry,_) values,count,counts,count_classified,kept_same,line_same,accepted)

# A usable tensor, read through its model.
def usable_lines(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+name: String,+operations: F32,+count: U32,+length: U32,+initial: Accumulator,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reduced: Nat,
  -bands: Bands.Bands(~Element),observation: Model.Observation<Element,bands>,+geometry: Bands.Geometry,+source_shape: List<&2,U32>,+source_fill: Element,+source_workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{bands,geometry,source_shape,source_fill,Tensors.Usable{},source_workers}),
  +counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +kept_same: {ShapeLaws.elements(shape) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(view_axes,reduced))) : Nat},
  +line_same: {U32.to_nat(length) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.line_axes(view_axes,reduced))) : Nat},
  accepted: {Contract.tensor_ready(~Element,BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,shape,fill,workers,
    Tensors.Tensor{bands,geometry,source_shape,source_fill,Tensors.Usable{},source_workers},Some{Views.View{view_axes,offset}},reduced)) == True{} : Bool}) ->
  Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,U32.to_nat(length),
    BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,shape,fill,workers,
      Tensors.Tensor{bands,geometry,source_shape,source_fill,Tensors.Usable{},source_workers},Some{Views.View{view_axes,offset}},reduced)):
  Model.Observation{+model,+equation} = observation
  %equation : Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,U32.to_nat(length),
    BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,shape,fill,workers,
      Tensors.Tensor{_,geometry,source_shape,source_fill,Tensors.Usable{},source_workers},Some{Views.View{view_axes,offset}},reduced))
  chosen_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,U32.is_zero(count),U32.is_zero(length),logical,name,operations,count,length,initial,shape,fill,workers,
    view_axes,offset,reduced,model,geometry,source_shape,source_fill,source_workers,{==},{==},
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) :
      Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{_,geometry,source_shape,source_fill,Tensors.Usable{},source_workers})
    contract,
    counts,kept_same,line_same,
    %Equal.sym(Bands.Bands(~Element),Model.pack(~Element,model),bands,equation) :
      {Contract.tensor_ready(~Element,BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,shape,fill,workers,
        Tensors.Tensor{_,geometry,source_shape,source_fill,Tensors.Usable{},source_workers},Some{Views.View{view_axes,offset}},reduced)) == True{} : Bool}
    accepted)

# The lines of a view of a tensor of the contract: count outputs of shape,
# whose elements are the kept axes of the view, and lines of length values,
# the elements of its last reduced axes. When the call returns Ready, output
# k holds finish of the fold of line k of the view from the initial value.
def lines(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+name: String,+operations: F32,+count: U32,+length: U32,+initial: Accumulator,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  tensor: Tensors.Tensor<Element>,
  contract: @ready: {Contract.tensor_ready(~Element,tensor) == True{} : Bool} -> Global.Contract(~Element,~Logical,~original,logical,tensor),
  +view_axes: List<&2,Views.Axis>,+offset: U32,+reduced: Nat,
  +counts: {Tensors.count_of(shape) == Some{count} : Maybe<&2,U32>},
  +kept_same: {ShapeLaws.elements(shape) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.kept_axes(view_axes,reduced))) : Nat},
  +line_same: {U32.to_nat(length) == ShapeLaws.elements(Views.axis_extents(OperationsLaws.line_axes(view_axes,reduced))) : Nat},
  accepted: {Contract.tensor_ready(~Element,BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,shape,fill,workers,tensor,
    Some{Views.View{view_axes,offset}},reduced)) == True{} : Bool}) ->
  Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,U32.to_nat(length),
    BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,shape,fill,workers,tensor,Some{Views.View{view_axes,offset}},reduced)):
  match tensor:
    case Tensors.Tensor{bands,+geometry,+source_shape,+source_fill,status,+source_workers}:
      match status:
        case Tensors.Usable{}:
          usable_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,name,operations,count,length,initial,shape,fill,workers,view_axes,offset,reduced,
            bands,Model.observe(~Element,bands),geometry,source_shape,source_fill,source_workers,contract({==}),counts,kept_same,line_same,accepted)
        case Tensors.Failed{+problem}:
          Empty.absurd(Lined(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,Views.View{view_axes,offset},initial,U32.to_nat(length),
            BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,shape,fill,workers,
              Tensors.Tensor{bands,geometry,source_shape,source_fill,Tensors.Failed{problem},source_workers},Some{Views.View{view_axes,offset}},reduced)),
            WordEncoding.false_true(accepted))

# ---- The shape of the lines ----

def whole_extents_of(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  found: Maybe<&2,BandViews.Banded>,+name: String,+operations: F32,+initial: Accumulator,+reduced: Nat,+whole: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  tensor: Tensors.Tensor<Element>) ->
  {Contract.extents(~Element,BandReduce.whole_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,found,whole,shape,fill,workers,tensor)) == shape :
    List<&2,U32>}:
  match found:
    case Some{+banded}:
      ViewReady.transformed_extents(~Element,name,Tensors.InvalidArgument{shape},BandReduce.output(banded,whole,reduced),shape,fill,workers,
        BandSchedule.transform(~Element,~BandReduce.Context<Accumulator,Element>,~BandReduce.leaf(~Element,~Accumulator,~operation,~finish),
          BandReduce.Context{banded,reduced,whole,initial,fill},whole,workers,4.0,operations,Tensors.arrive(~Element,whole,Bands.unsplit(whole),tensor)))
    case None{}: {==}

def found_extents_of(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  found: Maybe<&2,BandViews.Banded>,+name: String,+operations: F32,+initial: Accumulator,+reduced: Nat,+axes: List<&2,Views.Axis>,+offset: U32,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,+source_shape: List<&2,U32>,+source_fill: Element,+source_workers: U32,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Contract.extents(~Element,BandReduce.found_lines(~Element,~Accumulator,~operation,~finish,name,operations,initial,reduced,found,axes,offset,bands,geometry,source_shape,source_fill,
    source_workers,shape,fill,workers)) == shape : List<&2,U32>}:
  match found:
    case Some{+banded}:
      ViewReady.transformed_extents(~Element,name,Tensors.InvalidArgument{shape},BandReduce.output(banded,geometry,reduced),shape,fill,workers,
        BandSchedule.transform(~Element,~BandReduce.Context<Accumulator,Element>,~BandReduce.leaf(~Element,~Accumulator,~operation,~finish),
          BandReduce.Context{banded,reduced,geometry,initial,fill},geometry,workers,4.0,operations,bands))
    case None{}:
      +whole = {Bands.Geometry{1,1,1,Tensors.count_or_zero(source_shape)} : Bands.Geometry}
      whole_extents_of(~Element,~Accumulator,~operation,~finish,BandReduce.in_band(BandViews.classify(axes,offset,whole),reduced),name,operations,initial,reduced,whole,shape,fill,
        workers,Tensors.Tensor{bands,geometry,source_shape,source_fill,Tensors.Usable{},source_workers})

def chosen_extents_of(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  empty: Bool,+name: String,+operations: F32,+count: U32,+initial: Accumulator,+reduced: Nat,+axes: List<&2,Views.Axis>,+offset: U32,bands: Bands.Bands(~Element),
  +geometry: Bands.Geometry,+source_shape: List<&2,U32>,+source_fill: Element,+source_workers: U32,+shape: List<&2,U32>,+fill: Element,+workers: U32) ->
  {Contract.extents(~Element,BandReduce.lines_when(~Element,~Accumulator,~operation,~finish,name,empty,operations,count,initial,reduced,axes,offset,bands,geometry,source_shape,
    source_fill,source_workers,shape,fill,workers)) == shape : List<&2,U32>}:
  match empty:
    case True{}: {==}
    case False{}:
      found_extents_of(~Element,~Accumulator,~operation,~finish,BandReduce.in_band(BandViews.classify(axes,offset,geometry),reduced),name,operations,initial,reduced,axes,offset,
        bands,geometry,source_shape,source_fill,source_workers,shape,fill,workers)

# The lines have the shape of their outputs.
def lines_extents(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+name: String,
  +operations: F32,+count: U32,+length: U32,+initial: Accumulator,+shape: List<&2,U32>,+fill: Element,+workers: U32,tensor: Tensors.Tensor<Element>,
  view: Maybe<&2,Views.View>,+reduced: Nat) ->
  {Contract.extents(~Element,BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,shape,fill,workers,tensor,view,reduced)) == shape :
    List<&2,U32>}:
  match tensor:
    case Tensors.Tensor{bands,+geometry,+source_shape,+source_fill,status,+source_workers}:
      match status view:
        case Tensors.Usable{} Some{Views.View{+axes,+offset}}:
          chosen_extents_of(~Element,~Accumulator,~operation,~finish,U32.is_zero(count) || U32.is_zero(length),name,operations,count,initial,reduced,axes,offset,bands,
            geometry,source_shape,source_fill,source_workers,shape,fill,workers)
        case Tensors.Usable{} None{}: {==}
        case Tensors.Failed{problem} view: {==}

def counted_extents(~Element: Data,~Accumulator: Data,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,+name: String,
  +operations: F32,count: Maybe<&2,U32>,length: Maybe<&2,U32>,+initial: Accumulator,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,+fill: Element,
  +workers: U32,tensor: Tensors.Tensor<Element>) ->
  {Contract.extents(~Element,Layout.reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,shape,kept,reduced,fill,workers,tensor)) ==
    Layout.extents_at(shape,kept) : List<&2,U32>}:
  match count length:
    case Some{+count} Some{+length}:
      lines_extents(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,Layout.extents_at(shape,kept),fill,workers,tensor,
        Layout.permuted(Tensors.view_of(shape),List.append(&2,Nat,kept,reduced)),List.length(&2,Nat,reduced))
    case Some{count} None{}: {==}
    case None{} Some{length}: {==}
    case None{} None{}: {==}

# ---- Another reader that agrees on the outputs ----

def count_room(+expected: Nat,+count: U32,evidence: Count.Count<expected,count>) -> {U32.is_le(count,Machine.capacity(Default.limits())) == True{} : Bool}:
  Count.Count{exact,bounded} = evidence
  bounded

# A tensor of the contract with the shape target holds the values of another
# reader that agrees with its reader on the indices of its geometry.
def agreed_contract(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,
  ~Related: @+before: Before -> @+after: After -> @+geometry: Bands.Geometry -> Data,~agrees: Read.Agrees(~Element,~Before,~After,~before_value,~after_value,~Related),
  +before: Before,+after: After,+target: List<&2,U32>,
  related: @+geometry: Bands.Geometry -> @coherent: Global.Coherent(target,geometry) -> Related(before,after,geometry),
  tensor: Tensors.Tensor<Element>,+shaped: {Contract.extents(~Element,tensor) == target : List<&2,U32>},
  contract: Global.Contract(~Element,~Before,~before_value,before,tensor),accepted: {Contract.tensor_ready(~Element,tensor) == True{} : Bool}) ->
  Global.Contract(~Element,~After,~after_value,after,tensor):
  match tensor:
    case Tensors.Tensor{bands,+geometry,+shape,fill,status,workers}:
      match status:
        case Tensors.Usable{}:
          Shape.Both{Shape.Both{BandShape.WellFormed{+metadata,ready,coverage},Shape.Both{counted,+coherent}},values} = contract
          Global.Counted{+count,+counts} = counted
          +capacity = Machine.capacity(Default.limits())
          +room = {
            %coherent : {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
            %Equal.sym(Nat,ShapeLaws.elements(shape),U32.to_nat(count),ViewValues.count_elements(shape,count,Count.count_of(shape,count,counts))) :
              {Nat.is_le(_,U32.to_nat(capacity)) == True{} : Bool}
            Bounds.as_nat_le(count,capacity,count_room(ShapeLaws.elements(shape),count,Count.count_of(shape,count,counts))) :
            {Nat.is_le(BandGather.elements(geometry),U32.to_nat(capacity)) == True{} : Bool}}
          +target_coherent = {%shaped : Global.Coherent(_,geometry) coherent : Global.Coherent(target,geometry)}
          Shape.Both{Shape.Both{BandShape.WellFormed{metadata,ready,coverage},Shape.Both{Global.Counted{count,counts},coherent}},
            Read.agreed(~Element,~Before,~After,~before_value,~after_value,~Related,~agrees,before,after,geometry,related(geometry,target_coherent),
              Model.contents(~Element,bands),metadata,values,capacity,Limits.capacity_fits(Default.limits()),room)}
        case Tensors.Failed{problem}:
          Empty.absurd(Global.Contract(~Element,~After,~after_value,after,Tensors.Tensor{bands,geometry,shape,fill,Tensors.Failed{problem},workers}),
            WordEncoding.false_true(accepted))

# ---- Reductions over listed axes ----

# The input read with its kept axes first and its reduced axes last, through
# the permuted view of its shape.
def source(~Logical: Data,+logical: Logical,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>) -> Ordered<Logical>:
  Ordered{logical,Layout.permuted(Tensors.view_of(shape),List.append(&2,Nat,kept,reduced))}

# A reduction over the reduced axes holds, at the row-major index k of the
# kept axes, finish of the fold of the values of the reduced axes at k in
# row-major order.
def Listed(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+initial: Accumulator,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,-tensor: Tensors.Tensor<Element>) -> Data:
  Global.Contract(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
    Lines{initial,source(~Logical,logical,shape,kept,reduced),ShapeLaws.elements(Layout.extents_at(shape,reduced))},tensor)

# A tensor of the contract holds the values of another reader at a related
# context when the two readers agree there.
def relabel(~Element: Data,~Before: Data,~After: Data,~before_value: Before -> U32 -> Element,~after_value: After -> U32 -> Element,~Related: Before -> After -> Data,
  ~agreement: BandValues.Agreement(~Element,~Before,~After,~before_value,~after_value,~Related),+before: Before,+after: After,+related: Related(before,after),
  tensor: Tensors.Tensor<Element>,contract: Global.Contract(~Element,~Before,~before_value,before,tensor)) ->
  Global.Contract(~Element,~After,~after_value,after,tensor):
  match tensor:
    case Tensors.Tensor{bands,+geometry,shape,fill,status,workers}:
      Shape.Both{normal,values} = contract
      Shape.Both{normal,BandValues.values(~Element,~Before,~After,~before_value,~after_value,~Related,~agreement,before,after,related,geometry,Model.contents(~Element,bands),values)}

# ---- The extents of an order of axes ----

def appended_extents(+shape: List<&2,U32>,left: List<&2,Nat>,+right: List<&2,Nat>) ->
  {Layout.extents_at(shape,List.append(&2,Nat,left,right)) == List.append(&2,U32,Layout.extents_at(shape,left),Layout.extents_at(shape,right)) : List<&2,U32>}:
  match left:
    case []: {==}
    case +index <> +rest:
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => Layout.extent_at(shape,index) <> tail,Layout.extents_at(shape,List.append(&2,Nat,rest,right)),
        List.append(&2,U32,Layout.extents_at(shape,rest),Layout.extents_at(shape,right)),appended_extents(shape,rest,right))

def extents_length(+shape: List<&2,U32>,indices: List<&2,Nat>) -> {List.length(&2,U32,Layout.extents_at(shape,indices)) == List.length(&2,Nat,indices) : Nat}:
  match indices:
    case []: {==}
    case index <> +rest:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,Layout.extents_at(shape,rest)),List.length(&2,Nat,rest),extents_length(shape,rest))

def take_prefix(~Element: Data,left: List<&2,Element>,+right: List<&2,Element>) ->
  {List.take(&2,Element,List.append(&2,Element,left,right),List.length(&2,Element,left)) == left : List<&2,Element>}:
  match left:
    case []:
      match right:
        case []: {==}
        case head <> tail: {==}
    case +head <> +tail:
      Equal.cong(List<&2,Element>,List<&2,Element>,others => head <> others,
        List.take(&2,Element,List.append(&2,Element,tail,right),List.length(&2,Element,tail)),tail,take_prefix(~Element,tail,right))

def dropped_none(~Element: Data,items: List<&2,Element>) -> {List.drop(&2,Element,items,0n) == items : List<&2,Element>}:
  match items:
    case []: {==}
    case head <> tail: {==}

def dropped_all(~Element: Data,items: List<&2,Element>) -> {List.drop(&2,Element,items,List.length(&2,Element,items)) == [] : List<&2,Element>}:
  match items:
    case []: {==}
    case head <> +tail: dropped_all(~Element,tail)

def drop_prefix(~Element: Data,left: List<&2,Element>,+right: List<&2,Element>) ->
  {List.drop(&2,Element,List.append(&2,Element,left,right),List.length(&2,Element,left)) == right : List<&2,Element>}:
  match left:
    case []: dropped_none(~Element,right)
    case head <> +tail: drop_prefix(~Element,tail,right)

# A view whose extents are the kept extents and then the reduced ones keeps
# as many axes as there are kept extents.
def split_count(+axes: List<&2,Views.Axis>,+kept: List<&2,U32>,+line: List<&2,U32>,+reduced: Nat,
  +exact: {Views.axis_extents(axes) == List.append(&2,U32,kept,line) : List<&2,U32>},+lined: {List.length(&2,U32,line) == reduced : Nat}) ->
  {Nat.sub(List.length(&2,Views.Axis,axes),reduced) == List.length(&2,U32,kept) : Nat}:
  +lk = List.length(&2,U32,kept)
  +lr = List.length(&2,U32,line)
  +lengths = Equal.trans(Nat,List.length(&2,Views.Axis,axes),List.length(&2,U32,Views.axis_extents(axes)),Nat.add(lk,lr),
    Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(axes)),List.length(&2,Views.Axis,axes),Metadata.extents_length(axes)),
    Equal.trans(Nat,List.length(&2,U32,Views.axis_extents(axes)),List.length(&2,U32,List.append(&2,U32,kept,line)),Nat.add(lk,lr),
      Equal.cong(List<&2,U32>,Nat,items => List.length(&2,U32,items),Views.axis_extents(axes),List.append(&2,U32,kept,line),exact),
      ListReadback.append_length(~U32,kept,line)))
  %Equal.sym(Nat,List.length(&2,Views.Axis,axes),Nat.add(lk,lr),lengths) : {Nat.sub(_,reduced) == lk : Nat}
  %lined : {Nat.sub(Nat.add(lk,lr),_) == lk : Nat}
  %Equal.sym(Nat,Nat.add(lk,lr),Nat.add(lr,lk),Order.commutative(lk,lr)) : {Nat.sub(_,lr) == lk : Nat}
  Order.subtract_shift(lr,lk)

def split_kept(+axes: List<&2,Views.Axis>,+kept: List<&2,U32>,+line: List<&2,U32>,+reduced: Nat,
  +exact: {Views.axis_extents(axes) == List.append(&2,U32,kept,line) : List<&2,U32>},+lined: {List.length(&2,U32,line) == reduced : Nat}) ->
  {Views.axis_extents(OperationsLaws.kept_axes(axes,reduced)) == kept : List<&2,U32>}:
  +count = Nat.sub(List.length(&2,Views.Axis,axes),reduced)
  %Equal.sym(List<&2,U32>,Views.axis_extents(List.take(&2,Views.Axis,axes,count)),List.take(&2,U32,Views.axis_extents(axes),count),extents_taken(axes,count)) :
    {_ == kept : List<&2,U32>}
  %Equal.sym(List<&2,U32>,Views.axis_extents(axes),List.append(&2,U32,kept,line),exact) : {List.take(&2,U32,_,count) == kept : List<&2,U32>}
  %Equal.sym(Nat,count,List.length(&2,U32,kept),split_count(axes,kept,line,reduced,exact,lined)) :
    {List.take(&2,U32,List.append(&2,U32,kept,line),_) == kept : List<&2,U32>}
  take_prefix(~U32,kept,line)

def split_line(+axes: List<&2,Views.Axis>,+kept: List<&2,U32>,+line: List<&2,U32>,+reduced: Nat,
  +exact: {Views.axis_extents(axes) == List.append(&2,U32,kept,line) : List<&2,U32>},+lined: {List.length(&2,U32,line) == reduced : Nat}) ->
  {Views.axis_extents(OperationsLaws.line_axes(axes,reduced)) == line : List<&2,U32>}:
  +count = Nat.sub(List.length(&2,Views.Axis,axes),reduced)
  %Equal.sym(List<&2,U32>,Views.axis_extents(List.drop(&2,Views.Axis,axes,count)),List.drop(&2,U32,Views.axis_extents(axes),count),extents_dropped(axes,count)) :
    {_ == line : List<&2,U32>}
  %Equal.sym(List<&2,U32>,Views.axis_extents(axes),List.append(&2,U32,kept,line),exact) : {List.drop(&2,U32,_,count) == line : List<&2,U32>}
  %Equal.sym(Nat,count,List.length(&2,U32,kept),split_count(axes,kept,line,reduced,exact,lined)) :
    {List.drop(&2,U32,List.append(&2,U32,kept,line),_) == line : List<&2,U32>}
  drop_prefix(~U32,kept,line)

# The extent of an axis found at an index.
def axis_extent(found: Maybe<&2,Views.Axis>) -> U32:
  match found:
    case Some{Views.Axis{extent,stride}}: extent
    case None{}: 0

def got_extent(axes: List<&2,Views.Axis>,index: Nat) -> {Layout.extent_at(Views.axis_extents(axes),index) == axis_extent(List.get(&2,Views.Axis,axes,index)) : U32}:
  match axes index:
    case [] index: {==}
    case Views.Axis{extent,stride} <> tail 0n: {==}
    case Views.Axis{extent,stride} <> tail 1n+previous: got_extent(tail,previous)

def prepended(found: Maybe<&2,Views.Axis>,rest: Maybe<&2,Views.View>,+head: U32,+tail: List<&2,U32>,
  head_exact: {axis_extent(found) == head : U32},tail_exact: {Metadata.observed_extents(rest,tail) == tail : List<&2,U32>}) ->
  {Metadata.observed_extents(Views.prepend_axis(found,rest),head <> tail) == head <> tail : List<&2,U32>}:
  match found rest:
    case Some{Views.Axis{+extent,stride}} Some{Views.View{+axes,offset}}:
      %Equal.sym(U32,extent,head,head_exact) : {_ <> Views.axis_extents(axes) == head <> tail : List<&2,U32>}
      %Equal.sym(List<&2,U32>,Views.axis_extents(axes),tail,tail_exact) : {head <> _ == head <> tail : List<&2,U32>}
      {==}
    case Some{axis} None{}: {==}
    case None{} Some{view}: {==}
    case None{} None{}: {==}

def reordered_extents(order: List<&2,Nat>,+axes: List<&2,Views.Axis>,+offset: U32) ->
  {Metadata.observed_extents(Views.reorder(order,axes,offset),Layout.extents_at(Views.axis_extents(axes),order)) == Layout.extents_at(Views.axis_extents(axes),order) : List<&2,U32>}:
  match order:
    case []: {==}
    case +index <> +rest:
      prepended(List.get(&2,Views.Axis,axes,index),Views.reorder(rest,axes,offset),Layout.extent_at(Views.axis_extents(axes),index),
        Layout.extents_at(Views.axis_extents(axes),rest),
        Equal.sym(U32,Layout.extent_at(Views.axis_extents(axes),index),axis_extent(List.get(&2,Views.Axis,axes,index)),got_extent(axes,index)),
        reordered_extents(rest,axes,offset))

def checked_extents(valid: Bool,+axes: List<&2,Views.Axis>,+offset: U32,+order: List<&2,Nat>,+shape: List<&2,U32>,
  +exact: {Views.axis_extents(axes) == shape : List<&2,U32>}) ->
  {Metadata.observed_extents(Views.permute_when(valid,Views.View{axes,offset},order),Layout.extents_at(shape,order)) == Layout.extents_at(shape,order) : List<&2,U32>}:
  match valid:
    case True{}:
      %exact : {Metadata.observed_extents(Views.reorder(order,axes,offset),Layout.extents_at(_,order)) == Layout.extents_at(_,order) : List<&2,U32>}
      reordered_extents(order,axes,offset)
    case False{}: {==}

# The permuted view of a shape has the extents of the shape at its order.
def permuted_extents(view: Maybe<&2,Views.View>,+order: List<&2,Nat>,+shape: List<&2,U32>,
  exact: {Metadata.observed_extents(view,shape) == shape : List<&2,U32>}) ->
  {Metadata.observed_extents(Layout.permuted(view,order),Layout.extents_at(shape,order)) == Layout.extents_at(shape,order) : List<&2,U32>}:
  match view:
    case Some{Views.View{+axes,+offset}}:
      checked_extents(Nat.is_eq(Views.rank(Views.View{axes,offset}),List.length(&2,Nat,order)) && Views.unique_axes(order),axes,offset,order,shape,exact)
    case None{}: {==}

# ---- Reductions over listed axes ----

# The permuted view of the input: its kept extents are the outputs and its
# reduced extents a line.
def viewed_lines(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  view: Maybe<&2,Views.View>,+logical: Logical,+name: String,+operations: F32,+initial: Accumulator,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,
  +count: U32,+length: U32,+bands: Model.Model<Element>,+geometry: Bands.Geometry,+fill: Element,+workers: U32,
  +found: {Layout.permuted(Tensors.view_of(shape),List.append(&2,Nat,kept,reduced)) == view : Maybe<&2,Views.View>},
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  +counts: {Tensors.count_of(Layout.extents_at(shape,kept)) == Some{count} : Maybe<&2,U32>},
  +lengths: {Tensors.count_of(Layout.extents_at(shape,reduced)) == Some{length} : Maybe<&2,U32>},
  accepted: {Contract.tensor_ready(~Element,BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,Layout.extents_at(shape,kept),fill,workers,
    Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},view,List.length(&2,Nat,reduced))) == True{} : Bool}) ->
  Global.Contract(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
    Lines{initial,Ordered{logical,view},U32.to_nat(length)},
    BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,Layout.extents_at(shape,kept),fill,workers,
      Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},view,List.length(&2,Nat,reduced))):
  match view:
    case Some{Views.View{+axes,+offset}}:
      +order = List.append(&2,Nat,kept,reduced)
      +outer = Layout.extents_at(shape,kept)
      +inner = Layout.extents_at(shape,reduced)
      +rank = List.length(&2,Nat,reduced)
      +viewed_extents = {%found : {Metadata.observed_extents(_,Layout.extents_at(shape,order)) == Layout.extents_at(shape,order) : List<&2,U32>}
        permuted_extents(Tensors.view_of(shape),order,shape,Metadata.contiguous_shape(shape,0)) :
        {Views.axis_extents(axes) == Layout.extents_at(shape,order) : List<&2,U32>}}
      +exact = Equal.trans(List<&2,U32>,Views.axis_extents(axes),Layout.extents_at(shape,order),List.append(&2,U32,outer,inner),viewed_extents,appended_extents(shape,kept,reduced))
      +lined = extents_length(shape,reduced)
      +kept_same = Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),outer,Views.axis_extents(OperationsLaws.kept_axes(axes,rank)),
        Equal.sym(List<&2,U32>,Views.axis_extents(OperationsLaws.kept_axes(axes,rank)),outer,split_kept(axes,outer,inner,rank,exact,lined)))
      +line_same = Equal.trans(Nat,U32.to_nat(length),ShapeLaws.elements(inner),ShapeLaws.elements(Views.axis_extents(OperationsLaws.line_axes(axes,rank))),
        Equal.sym(Nat,ShapeLaws.elements(inner),U32.to_nat(length),ViewValues.count_elements(inner,length,Count.count_of(inner,length,lengths))),
        Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),inner,Views.axis_extents(OperationsLaws.line_axes(axes,rank)),
          Equal.sym(List<&2,U32>,Views.axis_extents(OperationsLaws.line_axes(axes,rank)),inner,split_line(axes,outer,inner,rank,exact,lined))))
      lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,name,operations,count,length,initial,outer,fill,workers,
        Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},ready => contract,axes,offset,rank,counts,kept_same,line_same,accepted)
    case None{}:
      Empty.absurd(Global.Contract(~Element,~Lines<Accumulator,Ordered<Logical>>,
          ~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),Lines{initial,Ordered{logical,None{}},U32.to_nat(length)},
          BandReduce.lines(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,Layout.extents_at(shape,kept),fill,workers,
            Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},None{},List.length(&2,Nat,reduced))),
        WordEncoding.false_true(accepted))

def counted_lines(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  count: Maybe<&2,U32>,length: Maybe<&2,U32>,+logical: Logical,+name: String,+operations: F32,+initial: Accumulator,+shape: List<&2,U32>,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+fill: Element,+workers: U32,
  +counts: {Tensors.count_of(Layout.extents_at(shape,kept)) == count : Maybe<&2,U32>},
  +lengths: {Tensors.count_of(Layout.extents_at(shape,reduced)) == length : Maybe<&2,U32>},
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,shape,kept,reduced,fill,workers,
    Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) == True{} : Bool}) ->
  Listed(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,initial,shape,kept,reduced,
    Layout.reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,count,length,initial,shape,kept,reduced,fill,workers,
      Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})):
  match count length:
    case Some{+count} Some{+length}:
      %Equal.sym(Nat,ShapeLaws.elements(Layout.extents_at(shape,reduced)),U32.to_nat(length),
        ViewValues.count_elements(Layout.extents_at(shape,reduced),length,Count.count_of(Layout.extents_at(shape,reduced),length,lengths))) :
        Global.Contract(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
          Lines{initial,source(~Logical,logical,shape,kept,reduced),_},
          Layout.reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,Some{count},Some{length},initial,shape,kept,reduced,fill,workers,
            Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}))
      viewed_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,Layout.permuted(Tensors.view_of(shape),List.append(&2,Nat,kept,reduced)),logical,name,operations,
        initial,shape,kept,reduced,count,length,bands,geometry,fill,workers,{==},contract,counts,lengths,accepted)
    case Some{count} None{}:
      Empty.absurd(Listed(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,initial,shape,kept,reduced,
        Layout.reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,Some{count},None{},initial,shape,kept,reduced,fill,workers,
          Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})),WordEncoding.false_true(accepted))
    case None{} Some{length}:
      Empty.absurd(Listed(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,initial,shape,kept,reduced,
        Layout.reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,None{},Some{length},initial,shape,kept,reduced,fill,workers,
          Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})),WordEncoding.false_true(accepted))
    case None{} None{}:
      Empty.absurd(Listed(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,initial,shape,kept,reduced,
        Layout.reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,None{},None{},initial,shape,kept,reduced,fill,workers,
          Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})),WordEncoding.false_true(accepted))

# The reduction of the reduced axes of a tensor of the contract, the other
# axes kept.
def listed(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  valid: Bool,+logical: Logical,+name: String,+operations: F32,+initial: Accumulator,+kept: List<&2,Nat>,+reduced: List<&2,Nat>,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.reduce_listed(~Element,~Accumulator,~operation,~finish,name,operations,initial,
    Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},kept,reduced,valid)) == True{} : Bool}) ->
  Listed(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,initial,shape,kept,reduced,
    Layout.reduce_listed(~Element,~Accumulator,~operation,~finish,name,operations,initial,
      Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},kept,reduced,valid)):
  match valid:
    case True{}:
      counted_lines(~Element,~Accumulator,~Logical,~original,~operation,~finish,Tensors.count_of(Layout.extents_at(shape,kept)),Tensors.count_of(Layout.extents_at(shape,reduced)),
        logical,name,operations,initial,shape,kept,reduced,bands,geometry,fill,workers,{==},{==},contract,accepted)
    case False{}:
      Empty.absurd(Listed(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,initial,shape,kept,reduced,
        Layout.reduce_listed(~Element,~Accumulator,~operation,~finish,name,operations,initial,
          Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},kept,reduced,False{})),WordEncoding.false_true(accepted))

def kept_axes(+shape: List<&2,U32>,+axes: List<&2,U32>) -> List<&2,Nat>:
  Layout.axes_where(List.length(&2,U32,shape),0n,Layout.naturals(axes),False{})

def reduced_axes(+shape: List<&2,U32>,+axes: List<&2,U32>) -> List<&2,Nat>:
  Layout.axes_where(List.length(&2,U32,shape),0n,Layout.naturals(axes),True{})

# reduce_axes of a tensor of the contract, whatever its layout: when the call
# returns Ready, the result has the extents of the kept axes and holds at the
# row-major index k of them finish of the fold of the values of the listed
# axes at k in row-major order.
def reduce_axes(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+name: String,+operations: F32,+initial: Accumulator,+axes: List<&2,U32>,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.reduce_axes(~Element,~Accumulator,~operation,~finish,name,operations,initial,
    Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},axes)) == True{} : Bool}) ->
  Listed(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,initial,shape,kept_axes(shape,axes),reduced_axes(shape,axes),
    Layout.reduce_axes(~Element,~Accumulator,~operation,~finish,name,operations,initial,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers},axes)):
  +rank = List.length(&2,U32,shape)
  listed(~Element,~Accumulator,~Logical,~original,~operation,~finish,Views.unique_axes(Layout.naturals(axes)) && Layout.all_below(axes,rank),logical,name,operations,initial,
    kept_axes(shape,axes),reduced_axes(shape,axes),bands,geometry,shape,fill,workers,contract,accepted)

# ---- Every element ----

def every_length(remaining: Nat,+index: Nat) -> {List.length(&2,Nat,Layout.every_index(remaining,index)) == remaining : Nat}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,Layout.every_index(rest,1n+index)),rest,every_length(rest,1n+index))

# The indices from start on hold no axis below start.
def every_after(remaining: Nat,+start: Nat,+axis: Nat,+below: {Nat.is_lt(axis,start) == True{} : Bool}) ->
  {Views.contains_index(Layout.every_index(remaining,start),axis) == False{} : Bool}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      +different = Equal.trans(Bool,Nat.is_eq(start,axis),Nat.is_eq(axis,start),False{},Order.equal_symmetric(start,axis),Order.less_not_equal(axis,start,below))
      %Equal.sym(Bool,Nat.is_eq(start,axis),False{},different) : {_ || Views.contains_index(Layout.every_index(rest,1n+start),axis) == False{} : Bool}
      every_after(rest,1n+start,axis,Order.lt_succ(axis,start,Bounds.strict_le(axis,start,below)))

def every_unique(remaining: Nat,+start: Nat) -> {Views.unique_axes(Layout.every_index(remaining,start)) == True{} : Bool}:
  match remaining:
    case 0n: {==}
    case 1n+ +rest:
      %Equal.sym(Bool,Views.contains_index(Layout.every_index(rest,1n+start),start),False{},
          every_after(rest,1n+start,start,Order.lt_succ(start,start,Order.reflexive(start)))) :
        {Bool.not(_) && Views.unique_axes(Layout.every_index(rest,1n+start)) == True{} : Bool}
      every_unique(rest,1n+start)

# The axis at start leads the axes from start on.
def picked(axes: List<&2,Views.Axis>,start: Nat,+offset: U32,inside: {Nat.is_lt(start,List.length(&2,Views.Axis,axes)) == True{} : Bool}) ->
  {Views.prepend_axis(List.get(&2,Views.Axis,axes,start),Some{Views.View{List.drop(&2,Views.Axis,axes,1n+start),offset}}) ==
    Some{Views.View{List.drop(&2,Views.Axis,axes,start),offset}} : Maybe<&2,Views.View>}:
  match axes start:
    case [] start:
      Empty.absurd({Views.prepend_axis(List.get(&2,Views.Axis,[],start),Some{Views.View{List.drop(&2,Views.Axis,[],1n+start),offset}}) ==
        Some{Views.View{List.drop(&2,Views.Axis,[],start),offset}} : Maybe<&2,Views.View>},Order.below_zero(start,inside))
    case +head <> +tail 0n:
      Equal.cong(List<&2,Views.Axis>,Maybe<&2,Views.View>,rest => Some{Views.View{head <> rest,offset}},List.drop(&2,Views.Axis,tail,0n),tail,dropped_none(~Views.Axis,tail))
    case head <> +tail 1n+ +previous: picked(tail,previous,offset,inside)

# The indices from start on pick the axes from start on.
def every_reorder(remaining: Nat,+start: Nat,+axes: List<&2,Views.Axis>,+offset: U32,
  +sized: {Nat.add(start,remaining) == List.length(&2,Views.Axis,axes) : Nat}) ->
  {Views.reorder(Layout.every_index(remaining,start),axes,offset) == Some{Views.View{List.drop(&2,Views.Axis,axes,start),offset}} : Maybe<&2,Views.View>}:
  match remaining:
    case 0n:
      +at_end = Equal.trans(Nat,start,Nat.add(start,0n),List.length(&2,Views.Axis,axes),Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)),sized)
      %Equal.sym(Nat,start,List.length(&2,Views.Axis,axes),at_end) :
        {Some{Views.View{[],offset}} == Some{Views.View{List.drop(&2,Views.Axis,axes,_),offset}} : Maybe<&2,Views.View>}
      Equal.cong(List<&2,Views.Axis>,Maybe<&2,Views.View>,rest => Some{Views.View{rest,offset}},[],List.drop(&2,Views.Axis,axes,List.length(&2,Views.Axis,axes)),
        Equal.sym(List<&2,Views.Axis>,List.drop(&2,Views.Axis,axes,List.length(&2,Views.Axis,axes)),[],dropped_all(~Views.Axis,axes)))
    case 1n+ +rest:
      +room = {%Addition.successor_right(start,rest) : {_ == List.length(&2,Views.Axis,axes) : Nat} sized :
        {Nat.add(1n+start,rest) == List.length(&2,Views.Axis,axes) : Nat}}
      %Equal.sym(Maybe<&2,Views.View>,Views.reorder(Layout.every_index(rest,1n+start),axes,offset),Some{Views.View{List.drop(&2,Views.Axis,axes,1n+start),offset}},
          every_reorder(rest,1n+start,axes,offset,room)) :
        {Views.prepend_axis(List.get(&2,Views.Axis,axes,start),_) == Some{Views.View{List.drop(&2,Views.Axis,axes,start),offset}} : Maybe<&2,Views.View>}
      picked(axes,start,offset,Order.below_complement(start,rest,List.length(&2,Views.Axis,axes),sized))

# Every axis in order permutes the row-major view of a shape into itself.
def every_view(+shape: List<&2,U32>,+count: U32,+exact: {ShapeLaws.elements(shape) == U32.to_nat(count) : Nat}) ->
  {Layout.permuted(Tensors.view_of(shape),Layout.every_index(List.length(&2,U32,shape),0n)) == Some{Canonical.canonical_view(shape,count)} : Maybe<&2,Views.View>}:
  +rank = List.length(&2,U32,shape)
  +every = Layout.every_index(rank,0n)
  +canonical = Canonical.canonical_view(shape,count)
  +axes = Canonical.axes(canonical)
  +shaped = Canonical.canonical_shape(shape,count,exact)
  +ranked = Equal.cong(List<&2,U32>,Nat,items => List.length(&2,U32,items),Views.axis_extents_of(canonical),shape,shaped)
  +sized = Equal.trans(Nat,rank,List.length(&2,U32,Views.axis_extents(axes)),List.length(&2,Views.Axis,axes),
    Equal.sym(Nat,List.length(&2,U32,Views.axis_extents(axes)),rank,ranked),Metadata.extents_length(axes))
  +valid = {
    %Equal.sym(Nat,Views.rank(canonical),rank,ranked) : {Nat.is_eq(_,List.length(&2,Nat,every)) && Views.unique_axes(every) == True{} : Bool}
    %Equal.sym(Nat,List.length(&2,Nat,every),rank,every_length(rank,0n)) : {Nat.is_eq(rank,_) && Views.unique_axes(every) == True{} : Bool}
    %Equal.sym(Bool,Nat.is_eq(rank,rank),True{},Order.equal_reflexive(rank)) : {_ && Views.unique_axes(every) == True{} : Bool}
    every_unique(rank,0n) : {Nat.is_eq(Views.rank(canonical),List.length(&2,Nat,every)) && Views.unique_axes(every) == True{} : Bool}}
  %Equal.sym(Maybe<&2,Views.View>,Tensors.view_of(shape),Some{canonical},Canonical.view(shape,count,exact)) :
    {Layout.permuted(_,every) == Some{canonical} : Maybe<&2,Views.View>}
  %Equal.sym(Bool,Nat.is_eq(Views.rank(canonical),List.length(&2,Nat,every)) && Views.unique_axes(every),True{},valid) :
    {Views.permute_when(_,canonical,every) == Some{canonical} : Maybe<&2,Views.View>}
  %dropped_none(~Views.Axis,axes) : {Views.reorder(every,axes,0) == Some{Views.View{_,0}} : Maybe<&2,Views.View>}
  every_reorder(rank,0n,axes,0,sized)

def drop_next(xs: List<&2,U32>,index: Nat,inside: {Nat.is_lt(index,List.length(&2,U32,xs)) == True{} : Bool}) ->
  {List.drop(&2,U32,xs,index) == Layout.extent_at(xs,index) <> List.drop(&2,U32,xs,1n+index) : List<&2,U32>}:
  match xs index:
    case [] index: Empty.absurd({List.drop(&2,U32,[],index) == Layout.extent_at([],index) <> List.drop(&2,U32,[],1n+index) : List<&2,U32>},Order.below_zero(index,inside))
    case +head <> +tail 0n:
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,tail,0n),tail,dropped_none(~U32,tail)) : {head <> tail == head <> _ : List<&2,U32>}
      {==}
    case head <> +tail 1n+ +previous: drop_next(tail,previous,inside)

# Every index from start reads the extents of a shape from start.
def every_extents(remaining: Nat,+start: Nat,+shape: List<&2,U32>,+sized: {Nat.add(start,remaining) == List.length(&2,U32,shape) : Nat}) ->
  {Layout.extents_at(shape,Layout.every_index(remaining,start)) == List.drop(&2,U32,shape,start) : List<&2,U32>}:
  match remaining:
    case 0n:
      +at_end = Equal.trans(Nat,start,Nat.add(start,0n),List.length(&2,U32,shape),Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start)),sized)
      %Equal.sym(Nat,start,List.length(&2,U32,shape),at_end) : {[] == List.drop(&2,U32,shape,_) : List<&2,U32>}
      Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,List.length(&2,U32,shape)),[],dropped_all(~U32,shape))
    case 1n+ +rest:
      +room = {%Addition.successor_right(start,rest) : {_ == List.length(&2,U32,shape) : Nat} sized :
        {Nat.add(1n+start,rest) == List.length(&2,U32,shape) : Nat}}
      %Equal.sym(List<&2,U32>,Layout.extents_at(shape,Layout.every_index(rest,1n+start)),List.drop(&2,U32,shape,1n+start),every_extents(rest,1n+start,shape,room)) :
        {Layout.extent_at(shape,start) <> _ == List.drop(&2,U32,shape,start) : List<&2,U32>}
      Equal.sym(List<&2,U32>,List.drop(&2,U32,shape,start),Layout.extent_at(shape,start) <> List.drop(&2,U32,shape,1n+start),
        drop_next(shape,start,Order.below_complement(start,rest,List.length(&2,U32,shape),sized)))

# reduce_all of a tensor of the contract: when the call returns Ready, its one
# element holds finish of the fold of every element in row-major order.
def Whole(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+initial: Accumulator,+shape: List<&2,U32>,-tensor: Tensors.Tensor<Element>) -> Data:
  Global.Contract(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
    Lines{initial,Ordered{logical,None{}},ShapeLaws.elements(shape)},tensor)

# The one output folds the line of the row-major view, which reads every
# element at its own index.
def EveryRelated(~Accumulator: Data,~Logical: Data,+logical: Logical,+view: Views.View,+initial: Accumulator,+length: Nat,
  +before: Lines<Accumulator,Ordered<Logical>>,+after: Lines<Accumulator,Ordered<Logical>>,+geometry: Bands.Geometry) -> Data:
  Shape.Both<{before == Lines{initial,Ordered{logical,Some{view}},length} : Lines<Accumulator,Ordered<Logical>>},
    Shape.Both<{after == Lines{initial,Ordered{logical,None{}},length} : Lines<Accumulator,Ordered<Logical>>},
      {Nat.is_le(BandGather.elements(geometry),1n) == True{} : Bool}>>

def every_agrees(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+view: Views.View,+initial: Accumulator,+length: Nat,+zero: {Access.zero_contiguous(view) == True{} : Bool},
  +sized: {length == ShapeLaws.elements(Views.axis_extents_of(view)) : Nat},
  +before: Lines<Accumulator,Ordered<Logical>>,+after: Lines<Accumulator,Ordered<Logical>>,+geometry: Bands.Geometry,
  +related: EveryRelated(~Accumulator,~Logical,logical,view,initial,length,before,after,geometry),+index: U32,
  +inside: {Nat.is_lt(U32.to_nat(index),BandGather.elements(geometry)) == True{} : Bool}) ->
  {value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish,before,index) ==
    value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish,after,index) : Element}:
  Shape.Both{+before_exact,Shape.Both{+after_exact,+single}} = related
  +viewed = {Ordered{logical,Some{view}} : Ordered<Logical>}
  +direct = {Ordered{logical,None{}} : Ordered<Logical>}
  +first = Order.zero_below(U32.to_nat(index),
    Order.transitive(1n+U32.to_nat(index),BandGather.elements(geometry),1n,Order.strict_follow(U32.to_nat(index),BandGather.elements(geometry),inside),single))
  %Equal.sym(Lines<Accumulator,Ordered<Logical>>,before,Lines{initial,viewed,length},before_exact) :
    {value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish,_,index) ==
      value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish,after,index) : Element}
  %Equal.sym(Lines<Accumulator,Ordered<Logical>>,after,Lines{initial,direct,length},after_exact) :
    {value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish,Lines{initial,viewed,length},index) ==
      value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish,_,index) : Element}
  %Equal.sym(Nat,U32.to_nat(index),0n,first) :
    {finish(LineValues.line_fold(~Ordered<Logical>,~Element,~Accumulator,~ordered_value(~Element,~Logical,~original),~operation,length,viewed,Nat.mul(_,length),initial)) ==
      finish(LineValues.line_fold(~Ordered<Logical>,~Element,~Accumulator,~ordered_value(~Element,~Logical,~original),~operation,length,direct,Nat.mul(_,length),initial)) : Element}
  Equal.cong(Accumulator,Element,finish,
    LineValues.line_fold(~Ordered<Logical>,~Element,~Accumulator,~ordered_value(~Element,~Logical,~original),~operation,length,viewed,0n,initial),
    LineValues.line_fold(~Ordered<Logical>,~Element,~Accumulator,~ordered_value(~Element,~Logical,~original),~operation,length,direct,0n,initial),
    LineValues.line_agreed(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,
      ~viewed,~0n,~direct,~0n,~length,
      ~(offset => inside => Equal.cong(U32,Element,word => original(logical,word),CopyValues.address(view,offset),U32.from_nat(offset),
        CopyValues.zero_contiguous_address(view,offset,zero,%sized : {Nat.is_lt(offset,_) == True{} : Bool} inside))),
      length,0n,initial,Order.reflexive(length)))

# Every element folds through the row-major view, which reads it at its own
# index.
def counted_all(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  length: Maybe<&2,U32>,+logical: Logical,+name: String,+operations: F32,+initial: Accumulator,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +lengths: {Tensors.count_of(Layout.extents_at(shape,Layout.every_index(List.length(&2,U32,shape),0n))) == length : Maybe<&2,U32>},
  contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.reduce_all(~Element,~Accumulator,~operation,~finish,name,operations,initial,
    Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) == True{} : Bool}) ->
  Whole(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,initial,shape,
    Layout.reduce_all(~Element,~Accumulator,~operation,~finish,name,operations,initial,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})):
  match length:
    case Some{+length}:
      +rank = List.length(&2,U32,shape)
      +every = Layout.every_index(rank,0n)
      +accepted = accepted
      +extents = Equal.trans(List<&2,U32>,Layout.extents_at(shape,every),List.drop(&2,U32,shape,0n),shape,every_extents(rank,0n,shape,{==}),dropped_none(~U32,shape))
      +exact = {%extents : {ShapeLaws.elements(_) == U32.to_nat(length) : Nat}
        ViewValues.count_elements(Layout.extents_at(shape,every),length,Count.count_of(Layout.extents_at(shape,every),length,lengths)) :
        {ShapeLaws.elements(shape) == U32.to_nat(length) : Nat}}
      +canonical = Canonical.canonical_view(shape,length)
      +count = ShapeLaws.elements(shape)
      +lines = {Lines{initial,Ordered{logical,Some{canonical}},count} : Lines<Accumulator,Ordered<Logical>>}
      +direct = {Lines{initial,Ordered{logical,None{}},count} : Lines<Accumulator,Ordered<Logical>>}
      +viewed = {
        %every_view(shape,length,exact) :
          Global.Contract(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
            Lines{initial,Ordered{logical,_},count},
            Layout.reduce_all(~Element,~Accumulator,~operation,~finish,name,operations,initial,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}))
        %extents :
          Global.Contract(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
            Lines{initial,Ordered{logical,Layout.permuted(Tensors.view_of(shape),every)},ShapeLaws.elements(_)},
            Layout.reduce_all(~Element,~Accumulator,~operation,~finish,name,operations,initial,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}))
        listed(~Element,~Accumulator,~Logical,~original,~operation,~finish,True{},logical,name,operations,initial,[],every,bands,geometry,shape,fill,workers,contract,accepted) :
        Global.Contract(~Element,~Lines<Accumulator,Ordered<Logical>>,~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
          lines,
          Layout.reduce_all(~Element,~Accumulator,~operation,~finish,name,operations,initial,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}))}
      +sized = Equal.sym(Nat,ShapeLaws.elements(Views.axis_extents_of(canonical)),count,
        Equal.cong(List<&2,U32>,Nat,items => ShapeLaws.elements(items),Views.axis_extents_of(canonical),shape,Canonical.canonical_shape(shape,length,exact)))
      agreed_contract(~Element,~Lines<Accumulator,Ordered<Logical>>,~Lines<Accumulator,Ordered<Logical>>,
        ~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
        ~value(~Element,~Accumulator,~Ordered<Logical>,~ordered_value(~Element,~Logical,~original),~operation,~finish),
        ~EveryRelated(~Accumulator,~Logical,logical,canonical,initial,count),
        ~every_agrees(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,canonical,initial,count,
          Canonical.canonical_contiguous(shape,length,exact),sized),
        lines,direct,[],
        geometry => coherent => Shape.Both{{==},Shape.Both{{==},%coherent : {Nat.is_le(_,1n) == True{} : Bool} {==}}},
        Layout.reduce_all(~Element,~Accumulator,~operation,~finish,name,operations,initial,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
        counted_extents(~Element,~Accumulator,~operation,~finish,name,operations,Tensors.count_of(Layout.extents_at(shape,[])),Tensors.count_of(Layout.extents_at(shape,every)),
          initial,shape,[],every,fill,workers,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
        viewed,accepted)
    case None{}:
      +every = Layout.every_index(List.length(&2,U32,shape),0n)
      Empty.absurd(Whole(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,initial,shape,
          Layout.reduce_all(~Element,~Accumulator,~operation,~finish,name,operations,initial,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})),
        WordEncoding.false_true(
          %lengths : {Contract.tensor_ready(~Element,Layout.reduce_counted(~Element,~Accumulator,~operation,~finish,name,operations,Tensors.count_of(Layout.extents_at(shape,[])),_,
            initial,shape,[],every,fill,workers,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) == True{} : Bool}
          accepted))

def reduce_all(~Element: Data,~Accumulator: Data,~Logical: Data,~original: Logical -> U32 -> Element,~operation: Accumulator -> Element -> Accumulator,~finish: Accumulator -> Element,
  +logical: Logical,+name: String,+operations: F32,+initial: Accumulator,
  +bands: Model.Model<Element>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: Element,+workers: U32,
  +contract: Global.Contract(~Element,~Logical,~original,logical,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers}),
  accepted: {Contract.tensor_ready(~Element,Layout.reduce_all(~Element,~Accumulator,~operation,~finish,name,operations,initial,
    Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})) == True{} : Bool}) ->
  Whole(~Element,~Accumulator,~Logical,~original,~operation,~finish,logical,initial,shape,
    Layout.reduce_all(~Element,~Accumulator,~operation,~finish,name,operations,initial,Tensors.Tensor{Model.pack(~Element,bands),geometry,shape,fill,Tensors.Usable{},workers})):
  counted_all(~Element,~Accumulator,~Logical,~original,~operation,~finish,
    Tensors.count_of(Layout.extents_at(shape,Layout.every_index(List.length(&2,U32,shape),0n))),logical,name,operations,initial,bands,geometry,shape,fill,workers,{==},
    contract,accepted)

# ---- The public FP32 reductions ----

# sum, max and min hold the fold of every element in row-major order: the sum
# of math_sum.bend from Sum.zero, the maximum from -inf, the minimum from +inf.
def sum(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),
  accepted: {Contract.tensor_ready(~F32,F.sum(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers))) == True{} : Bool}) ->
  Whole(~F32,~Sum.Lanes,~Logical,~original,~Sum.step,~Sum.total,logical,Sum.zero(),shape,F.sum(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers))):
  reduce_all(~F32,~Sum.Lanes,~Logical,~original,~Sum.step,~Sum.total,logical,"sum",6.0,Sum.zero(),bands,geometry,shape,fill,workers,contract,accepted)

def max(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),
  accepted: {Contract.tensor_ready(~F32,F.max(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers))) == True{} : Bool}) ->
  Whole(~F32,~F32,~Logical,~original,~F32.max,~Tensors.same(~F32),logical,F32.div(F32.neg(1.0),0.0),shape,F.max(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers))):
  reduce_all(~F32,~F32,~Logical,~original,~F32.max,~Tensors.same(~F32),logical,"max",1.0,F32.div(F32.neg(1.0),0.0),bands,geometry,shape,fill,workers,contract,accepted)

def min(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),
  accepted: {Contract.tensor_ready(~F32,F.min(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers))) == True{} : Bool}) ->
  Whole(~F32,~F32,~Logical,~original,~F32.min,~Tensors.same(~F32),logical,F32.div(1.0,0.0),shape,F.min(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers))):
  reduce_all(~F32,~F32,~Logical,~original,~F32.min,~Tensors.same(~F32),logical,"min",1.0,F32.div(1.0,0.0),bands,geometry,shape,fill,workers,contract,accepted)

# sum_axes, max_axes and min_axes hold the same folds over the listed axes.
def sum_axes(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+axes: List<&2,U32>,+bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),
  accepted: {Contract.tensor_ready(~F32,F.sum_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),axes)) == True{} : Bool}) ->
  Listed(~F32,~Sum.Lanes,~Logical,~original,~Sum.step,~Sum.total,logical,Sum.zero(),shape,kept_axes(shape,axes),reduced_axes(shape,axes),
    F.sum_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),axes)):
  reduce_axes(~F32,~Sum.Lanes,~Logical,~original,~Sum.step,~Sum.total,logical,"sum_axes",6.0,Sum.zero(),axes,bands,geometry,shape,fill,workers,contract,accepted)

def max_axes(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+axes: List<&2,U32>,+bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),
  accepted: {Contract.tensor_ready(~F32,F.max_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),axes)) == True{} : Bool}) ->
  Listed(~F32,~F32,~Logical,~original,~F32.max,~Tensors.same(~F32),logical,F32.div(F32.neg(1.0),0.0),shape,kept_axes(shape,axes),reduced_axes(shape,axes),
    F.max_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),axes)):
  reduce_axes(~F32,~F32,~Logical,~original,~F32.max,~Tensors.same(~F32),logical,"max_axes",1.0,F32.div(F32.neg(1.0),0.0),axes,bands,geometry,shape,fill,workers,contract,accepted)

def min_axes(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+axes: List<&2,U32>,+bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),
  accepted: {Contract.tensor_ready(~F32,F.min_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),axes)) == True{} : Bool}) ->
  Listed(~F32,~F32,~Logical,~original,~F32.min,~Tensors.same(~F32),logical,F32.div(1.0,0.0),shape,kept_axes(shape,axes),reduced_axes(shape,axes),
    F.min_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),axes)):
  reduce_axes(~F32,~F32,~Logical,~original,~F32.min,~Tensors.same(~F32),logical,"min_axes",1.0,F32.div(1.0,0.0),axes,bands,geometry,shape,fill,workers,contract,accepted)

# A sum of the lines divided by count holds each sum divided by count.
def Divided(~Logical: Data,~original: Logical -> U32 -> F32,+context: Lines<Sum.Lanes,Ordered<Logical>>,+count: F32,-tensor: Tensors.Tensor<F32>) -> Data:
  Global.Contract(~F32,~BandMap.Context<F32,Lines<Sum.Lanes,Ordered<Logical>>>,
    ~BandMap.value(~F32,~F32,~Lines<Sum.Lanes,Ordered<Logical>>,~Bands.scalar_value(~F32,~F32.div),
      ~value(~F32,~Sum.Lanes,~Ordered<Logical>,~ordered_value(~F32,~Logical,~original),~Sum.step,~Sum.total)),
    BandMap.Context{count,context},tensor)

def divided(~Logical: Data,~original: Logical -> U32 -> F32,+context: Lines<Sum.Lanes,Ordered<Logical>>,+count: F32,total: Tensors.Tensor<F32>,
  usable: Global.Usable<F32,Contract.extents(~F32,total),total>,
  contract: Global.Contract(~F32,~Lines<Sum.Lanes,Ordered<Logical>>,~value(~F32,~Sum.Lanes,~Ordered<Logical>,~ordered_value(~F32,~Logical,~original),~Sum.step,~Sum.total),context,total)) ->
  Divided(~Logical,~original,context,count,Tensors.combine_scalar(~F32,~F32.div,1.0,total,count)):
  Global.Usable{+bands,+geometry,+fill,+workers,+equation} = usable
  %equation : Divided(~Logical,~original,context,count,Tensors.combine_scalar(~F32,~F32.div,1.0,_,count))
  NormalUnary.combine_scalar(~F32,~Lines<Sum.Lanes,Ordered<Logical>>,~F32.div,~value(~F32,~Sum.Lanes,~Ordered<Logical>,~ordered_value(~F32,~Logical,~original),~Sum.step,~Sum.total),
    context,1.0,count,bands,geometry,Contract.extents(~F32,total),fill,workers,
    %Equal.sym(Tensors.Tensor<F32>,Tensors.Tensor{Model.pack(~F32,bands),geometry,Contract.extents(~F32,total),fill,Tensors.Usable{},workers},total,equation) :
      Global.Contract(~F32,~Lines<Sum.Lanes,Ordered<Logical>>,~value(~F32,~Sum.Lanes,~Ordered<Logical>,~ordered_value(~F32,~Logical,~original),~Sum.step,~Sum.total),context,_)
    contract)

# mean holds the sum divided by the number of elements.
def mean(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),
  accepted: {Contract.tensor_ready(~F32,F.mean(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers))) == True{} : Bool}) ->
  Divided(~Logical,~original,Lines{Sum.zero(),Ordered{logical,None{}},ShapeLaws.elements(shape)},U32.to_f32(Tensors.count_or_zero(shape)),
    F.mean(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers))):
  +total_ready = NormalUnary.scalar_status(~F32,~F32.div,1.0,
    Layout.reduce_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"mean",6.0,Sum.zero(),CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),
    U32.to_f32(Tensors.count_or_zero(shape)),accepted)
  divided(~Logical,~original,Lines{Sum.zero(),Ordered{logical,None{}},ShapeLaws.elements(shape)},U32.to_f32(Tensors.count_or_zero(shape)),
    Layout.reduce_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"mean",6.0,Sum.zero(),CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),
    Global.usable(~F32,Layout.reduce_all(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"mean",6.0,Sum.zero(),CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),total_ready),
    reduce_all(~F32,~Sum.Lanes,~Logical,~original,~Sum.step,~Sum.total,logical,"mean",6.0,Sum.zero(),bands,geometry,shape,fill,workers,contract,total_ready))

# mean_axes holds each sum of sum_axes divided by the number of values it adds.
def mean_axes(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+axes: List<&2,U32>,+bands: Model.Model<F32>,+geometry: Bands.Geometry,+shape: List<&2,U32>,+fill: F32,+workers: U32,
  +contract: Global.Contract(~F32,~Logical,~original,logical,CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers)),
  accepted: {Contract.tensor_ready(~F32,F.mean_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),axes)) == True{} : Bool}) ->
  Divided(~Logical,~original,
    Lines{Sum.zero(),source(~Logical,logical,shape,kept_axes(shape,axes),reduced_axes(shape,axes)),ShapeLaws.elements(Layout.extents_at(shape,reduced_axes(shape,axes)))},
    U32.to_f32(Tensors.count_or_zero(Layout.extents_at(shape,reduced_axes(shape,axes)))),F.mean_axes(CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),axes)):
  +total_ready = NormalUnary.scalar_status(~F32,~F32.div,1.0,
    Layout.reduce_axes(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"mean_axes",6.0,Sum.zero(),CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),axes),
    U32.to_f32(Tensors.count_or_zero(Layout.extents_at(shape,reduced_axes(shape,axes)))),accepted)
  divided(~Logical,~original,
    Lines{Sum.zero(),source(~Logical,logical,shape,kept_axes(shape,axes),reduced_axes(shape,axes)),ShapeLaws.elements(Layout.extents_at(shape,reduced_axes(shape,axes)))},
    U32.to_f32(Tensors.count_or_zero(Layout.extents_at(shape,reduced_axes(shape,axes)))),
    Layout.reduce_axes(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"mean_axes",6.0,Sum.zero(),CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),axes),
    Global.usable(~F32,Layout.reduce_axes(~F32,~Sum.Lanes,~Sum.step,~Sum.total,"mean_axes",6.0,Sum.zero(),CombinePublic.tensor(~F32,bands,geometry,shape,fill,workers),axes),total_ready),
    reduce_axes(~F32,~Sum.Lanes,~Logical,~original,~Sum.step,~Sum.total,logical,"mean_axes",6.0,Sum.zero(),axes,bands,geometry,shape,fill,workers,contract,total_ready))
