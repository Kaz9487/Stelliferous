# A view read over the box of a target piece. The part of a piece steps the
# view strides over the box extents, so box index j reads the view offset of
# the first cell of the piece plus the strides times the box coordinates of j.
import Base
import ../tensor_view.bend as Views
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_move.bend as Move
import ../tensor.bend as Tensors
import ./tensor_move_readback.bend as Held
import ./tensor_move_part.bend as Part
import ./tensor_grid_box.bend as Box
import ./tensor_grid_pieces.bend as GridLaws
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_public.bend as TiledPublic
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_extrema.bend as Extrema
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./binary_radix.bend as Radix
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds

# ---- Views ----

def view_offset(+view: Views.View) -> U32:
  Views.View{axes,+offset} = view
  offset

def view_axes_of(+view: Views.View) -> List<&2,Views.Axis>:
  Views.View{+axes,offset} = view
  axes

def view_eta(+view: Views.View) -> {Views.View{view_axes_of(view),view_offset(view)} == view : Views.View}:
  match view:
    case Views.View{+axes,+offset}: {==}

def view_steps(axes: List<&2,Views.Axis>) -> List<&2,U32>:
  match axes:
    case Nil{}: []
    case Con{Views.Axis{extent,+stride},rest}: stride <> view_steps(rest)

def prepended_offset(+axis: Views.Axis,+view: Views.View) -> {view_offset(Grid.prepended(axis,view)) == view_offset(view) : U32}:
  match view:
    case Views.View{+axes,+offset}: {==}

def prepended_axes(+axis: Views.Axis,+view: Views.View) -> {view_axes_of(Grid.prepended(axis,view)) == Con{axis,view_axes_of(view)} : List<&2,Views.Axis>}:
  match view:
    case Views.View{+axes,+offset}: {==}

# Axes and extents of one length.
def SameLength(axes: List<&2,Views.Axis>,extents: List<&2,U32>) -> Data:
  match axes extents:
    case Nil{} Nil{}: Unit
    case Con{axis,rest} Con{extent,later}: SameLength(rest,later)
    case axes extents: Empty

def mirrored_extents(+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>) -> {Views.axis_extents(Views.mirrored_axes(axes,flags)) == Views.axis_extents(axes) : List<&2,U32>}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}:
      Equal.cong(List<&2,U32>,List<&2,U32>,later => Con{extent,later},Views.axis_extents(Views.mirrored_axes(rest,Views.later_flags(flags))),Views.axis_extents(rest),
        mirrored_extents(rest,Views.later_flags(flags)))

# The view offset at the coordinates of index is the offset plus the strides
# times the row-major coordinates of index.
def restricted_offset(+axes: List<&2,Views.Axis>,+boxes: List<&2,U32>,+index: U32,+offset: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(ShapeLaws.elements(Views.axis_extents(axes))),U32.to_nat(cap)) == True{} : Bool},
  +inside: {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},+same: SameLength(axes,boxes)) ->
  {view_offset(Grid.restricted(axes,Grid.coordinates(Views.axis_extents(axes),index),boxes,offset)) ==
    (offset + Box.stepped_sum(Views.axis_extents(axes),view_steps(axes),U32.to_nat(index)) : U32) : U32}:
  match axes boxes:
    case Nil{} Nil{}:
      Equal.sym(U32,(offset + 0 : U32),offset,Arithmetic.add_zero(offset))
    case Nil{} Con{+box,+boxes}:
      Empty.absurd({view_offset(Grid.restricted(Nil{},Grid.coordinates(Nil{},index),Con{box,boxes},offset)) ==
        (offset + Box.stepped_sum(Nil{},Nil{},U32.to_nat(index)) : U32) : U32},same)
    case Con{+axis,+rest} Nil{}:
      Empty.absurd({view_offset(Grid.restricted(Con{axis,rest},Grid.coordinates(Views.axis_extents(Con{axis,rest}),index),Nil{},offset)) ==
        (offset + Box.stepped_sum(Views.axis_extents(Con{axis,rest}),view_steps(Con{axis,rest}),U32.to_nat(index)) : U32) : U32},same)
    case Con{Views.Axis{+extent,+stride},+rest} Con{+box,+boxes}:
      +shape = Views.axis_extents(rest)
      +cells = ShapeLaws.elements(shape)
      +value = U32.to_nat(index)
      +whole = Nat.mul(U32.to_nat(extent),cells)
      +whole_positive = Order.positive_bound(value,whole,inside)
      +positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),cells,whole_positive)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),cells,whole_positive)
      +whole_room = Order.transitive(whole,Nat.double(whole),U32.to_nat(cap),Order.le_double(whole),room)
      +cells_le = DivisionOrder.le_multiple(U32.to_nat(extent),cells,extent_positive)
      +extent_le = RowMajor.product_covers(U32.to_nat(extent),cells,positive)
      +inner_room = Order.transitive(Nat.double(cells),Nat.double(whole),U32.to_nat(cap),Order.double_le(cells,whole,cells_le),room)
      +extent_room = Order.transitive(Nat.double(U32.to_nat(extent)),Nat.double(whole),U32.to_nat(cap),Order.double_le(U32.to_nat(extent),whole,extent_le),room)
      +width = U32.max(Grid.product(shape),1)
      +width_exact = Extrema.max_one_exact(Grid.product(shape),cells,TiledPublic.product_exact(cap,cap_ok,shape,Order.transitive(cells,whole,U32.to_nat(cap),cells_le,whole_room)),positive)
      +quotient = Nat.div(value,cells)
      +remainder = Nat.mod(value,cells)
      +exact = DivisionOrder.quotient_exact(value,cells,positive)
      +remainder_in = DivisionOrder.remainder_small(value,cells,positive)
      +quotient_in = DivisionOrder.quotient_below(value,U32.to_nat(extent),cells,positive,inside)
      +quotient_exact = Radix.quotient_word(index,width,quotient,remainder,cells,cap,cap_ok,exact,width_exact,positive,remainder_in,inner_room)
      +remainder_exact = Radix.remainder_word(index,width,quotient,remainder,cells,cap,cap_ok,exact,width_exact,positive,remainder_in,inner_room)
      +extent_width = Extrema.max_one_exact(extent,U32.to_nat(extent),{==},extent_positive)
      +digit_exact = Radix.remainder_word((index / width : U32),U32.max(extent,1),0n,quotient,U32.to_nat(extent),cap,cap_ok,quotient_exact,extent_width,extent_positive,quotient_in,extent_room)
      +quotient_room = Order.transitive(quotient,U32.to_nat(extent),U32.to_nat(cap),Bounds.strict_le(quotient,U32.to_nat(extent),quotient_in),
        Order.transitive(U32.to_nat(extent),Nat.double(U32.to_nat(extent)),U32.to_nat(cap),Order.le_double(U32.to_nat(extent)),extent_room))
      +remainder_room = Order.transitive(remainder,cells,U32.to_nat(cap),Bounds.strict_le(remainder,cells,remainder_in),
        Order.transitive(cells,Nat.double(cells),U32.to_nat(cap),Order.le_double(cells),inner_room))
      +remainder_back = Bounds.roundtrip(remainder,cap,cap_ok,remainder_room)
      +digit = Arithmetic.to_nat_injective(U32.from_nat(quotient),((index / width : U32) % U32.max(extent,1) : U32),
        Equal.trans(Nat,U32.to_nat(U32.from_nat(quotient)),quotient,U32.to_nat(((index / width : U32) % U32.max(extent,1) : U32)),
          Bounds.roundtrip(quotient,cap,cap_ok,quotient_room),Equal.sym(Nat,U32.to_nat(((index / width : U32) % U32.max(extent,1) : U32)),quotient,digit_exact)))
      +later = Arithmetic.to_nat_injective(U32.from_nat(remainder),(index % width : U32),
        Equal.trans(Nat,U32.to_nat(U32.from_nat(remainder)),remainder,U32.to_nat((index % width : U32)),
          remainder_back,Equal.sym(Nat,U32.to_nat((index % width : U32)),remainder,remainder_exact)))
      +next = (offset + U32.from_nat(quotient) * stride : U32)
      +tail = Box.stepped_sum(shape,view_steps(rest),remainder)
      +shifted = restricted_offset(rest,boxes,U32.from_nat(remainder),next,cap,cap_ok,inner_room,
        {%Equal.sym(Nat,U32.to_nat(U32.from_nat(remainder)),remainder,remainder_back) : {Nat.is_lt(_,cells) == True{} : Bool} remainder_in :
          {Nat.is_lt(U32.to_nat(U32.from_nat(remainder)),cells) == True{} : Bool}},same)
      +inner_view = Grid.restricted(rest,Grid.coordinates(shape,(index % width : U32)),boxes,(offset + ((index / width : U32) % U32.max(extent,1) : U32) * stride : U32))
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,shape}),Con{Grid.product(shape),Grid.strides(shape)},GridLaws.strides_con(extent,shape)) :
        {view_offset(Grid.restricted(Con{Views.Axis{extent,stride},rest},Grid.coordinates_by(Con{extent,shape},_,index),Con{box,boxes},offset)) ==
          (offset + Box.stepped_sum(Con{extent,shape},view_steps(Con{Views.Axis{extent,stride},rest}),value) : U32) : U32}
      %Equal.sym(U32,view_offset(Grid.prepended(Views.Axis{box,stride},inner_view)),view_offset(inner_view),prepended_offset(Views.Axis{box,stride},inner_view)) :
        {_ == (offset + (U32.from_nat(quotient) * stride + tail : U32) : U32) : U32}
      %digit : {view_offset(Grid.restricted(rest,Grid.coordinates(shape,(index % width : U32)),boxes,(offset + _ * stride : U32))) ==
        (offset + (U32.from_nat(quotient) * stride + tail : U32) : U32) : U32}
      %later : {view_offset(Grid.restricted(rest,Grid.coordinates(shape,_),boxes,next)) == (offset + (U32.from_nat(quotient) * stride + tail : U32) : U32) : U32}
      Equal.trans(U32,view_offset(Grid.restricted(rest,Grid.coordinates(shape,U32.from_nat(remainder)),boxes,next)),
        (next + Box.stepped_sum(shape,view_steps(rest),U32.to_nat(U32.from_nat(remainder))) : U32),(offset + (U32.from_nat(quotient) * stride + tail : U32) : U32),shifted,
        Equal.trans(U32,(next + Box.stepped_sum(shape,view_steps(rest),U32.to_nat(U32.from_nat(remainder))) : U32),(next + tail : U32),
          (offset + (U32.from_nat(quotient) * stride + tail : U32) : U32),
          Equal.cong(Nat,U32,count => (next + Box.stepped_sum(shape,view_steps(rest),count) : U32),U32.to_nat(U32.from_nat(remainder)),remainder,remainder_back),
          Arithmetic.add_associative(offset,(U32.from_nat(quotient) * stride : U32),tail)))

# The axes of a view read over a box: the box extents with the view strides.
def boxed_axes(axes: List<&2,Views.Axis>,boxes: List<&2,U32>) -> List<&2,Views.Axis>:
  match axes boxes:
    case Con{Views.Axis{extent,+stride},rest} Con{+box,later}: Views.Axis{box,stride} <> boxed_axes(rest,later)
    case axes boxes: []

def restricted_axes(+axes: List<&2,Views.Axis>,+boxes: List<&2,U32>,+index: U32,+offset: U32) ->
  {view_axes_of(Grid.restricted(axes,Grid.coordinates(Views.axis_extents(axes),index),boxes,offset)) == boxed_axes(axes,boxes) : List<&2,Views.Axis>}:
  match axes boxes:
    case Nil{} +boxes: {==}
    case Con{Views.Axis{+extent,+stride},+rest} Nil{}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,Views.axis_extents(rest)}),Con{Grid.product(Views.axis_extents(rest)),Grid.strides(Views.axis_extents(rest))},GridLaws.strides_con(extent,Views.axis_extents(rest))) :
        {view_axes_of(Grid.restricted(Con{Views.Axis{extent,stride},rest},Grid.coordinates_by(Con{extent,Views.axis_extents(rest)},_,index),Nil{},offset)) == boxed_axes(Con{Views.Axis{extent,stride},rest},Nil{}) : List<&2,Views.Axis>}
      {==}
    case Con{Views.Axis{+extent,+stride},+rest} Con{+box,+later}:
      +inner = U32.max(Grid.product(Views.axis_extents(rest)),1)
      +inner_view = Grid.restricted(rest,Grid.coordinates(Views.axis_extents(rest),(index % inner : U32)),later,(offset + ((index / inner : U32) % U32.max(extent,1) : U32) * stride : U32))
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,Views.axis_extents(rest)}),Con{Grid.product(Views.axis_extents(rest)),Grid.strides(Views.axis_extents(rest))},GridLaws.strides_con(extent,Views.axis_extents(rest))) :
        {view_axes_of(Grid.restricted(Con{Views.Axis{extent,stride},rest},Grid.coordinates_by(Con{extent,Views.axis_extents(rest)},_,index),Con{box,later},offset)) == boxed_axes(Con{Views.Axis{extent,stride},rest},Con{box,later}) : List<&2,Views.Axis>}
      Equal.trans(List<&2,Views.Axis>,view_axes_of(Grid.prepended(Views.Axis{box,stride},inner_view)),Con{Views.Axis{box,stride},view_axes_of(inner_view)},
        Con{Views.Axis{box,stride},boxed_axes(rest,later)},prepended_axes(Views.Axis{box,stride},inner_view),
        Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,tail => Con{Views.Axis{box,stride},tail},view_axes_of(inner_view),boxed_axes(rest,later),
          restricted_axes(rest,later,(index % inner : U32),(offset + ((index / inner : U32) % U32.max(extent,1) : U32) * stride : U32))))

# ---- The part of a box ----

def move_extents(axes: List<&2,Move.Axis>) -> List<&2,U32>:
  match axes:
    case Nil{}: []
    case Con{Move.Axis{+extent,stride,back,target},rest}: extent <> move_extents(rest)

def move_steps(axes: List<&2,Move.Axis>) -> List<&2,U32>:
  match axes:
    case Nil{}: []
    case Con{Move.Axis{extent,+stride,+back,target},rest}: Move.directed(back,stride) <> move_steps(rest)

def directed_twice(+back: Bool,+stride: U32) -> {Move.directed(back,Move.directed(back,stride)) == stride : U32}:
  match back:
    case True{}: Arithmetic.negate_twice(stride)
    case False{}: {==}

def total_elements(+axes: List<&2,Move.Axis>) -> {Part.total(axes) == ShapeLaws.elements(move_extents(axes)) : Nat}:
  match axes:
    case Nil{}: {==}
    case Con{Move.Axis{+extent,+stride,+back,+target},+rest}:
      Equal.cong(Nat,Nat,cells => Nat.mul(U32.to_nat(extent),cells),Part.total(rest),ShapeLaws.elements(move_extents(rest)),total_elements(rest))

# The move axes of a box read with flags and dense target strides.
def box_axes(+axes: List<&2,Views.Axis>,+boxes: List<&2,U32>,+flags: List<&2,Bool>) -> List<&2,Move.Axis>:
  Tensors.view_axes(boxed_axes(axes,boxes),flags,Tasks.strides_of(boxes))

# The box axes of a first axis and box step the target by the cells of the
# later boxes.
def box_axes_con(+extent: U32,+stride: U32,+rest: List<&2,Views.Axis>,+box: U32,+later: List<&2,U32>,+flags: List<&2,Bool>) ->
  {box_axes(Con{Views.Axis{extent,stride},rest},Con{box,later},flags) == Tensors.view_axes(Con{Views.Axis{box,stride},boxed_axes(rest,later)},flags,Con{Grid.product(later),Grid.strides(later)}) : List<&2,Move.Axis>}:
  %Equal.sym(List<&2,U32>,Grid.strides(Con{box,later}),Con{Grid.product(later),Grid.strides(later)},GridLaws.strides_con(box,later)) :
    {Tensors.view_axes(Con{Views.Axis{box,stride},boxed_axes(rest,later)},flags,_) == Tensors.view_axes(Con{Views.Axis{box,stride},boxed_axes(rest,later)},flags,Con{Grid.product(later),Grid.strides(later)}) : List<&2,Move.Axis>}
  {==}

def box_extents(+axes: List<&2,Views.Axis>,+boxes: List<&2,U32>,+flags: List<&2,Bool>,+same: SameLength(axes,boxes)) ->
  {move_extents(box_axes(axes,boxes,flags)) == boxes : List<&2,U32>}:
  match axes boxes:
    case Nil{} Nil{}: {==}
    case Nil{} Con{+box,+later}: Empty.absurd({move_extents(box_axes(Nil{},Con{box,later},flags)) == Con{box,later} : List<&2,U32>},same)
    case Con{+axis,+rest} Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest} Con{+box,+later}:
      %Equal.sym(List<&2,Move.Axis>,box_axes(Con{Views.Axis{extent,stride},rest},Con{box,later},flags),Tensors.view_axes(Con{Views.Axis{box,stride},boxed_axes(rest,later)},flags,Con{Grid.product(later),Grid.strides(later)}),box_axes_con(extent,stride,rest,box,later,flags)) :
        {move_extents(_) == Con{box,later} : List<&2,U32>}
      Equal.cong(List<&2,U32>,List<&2,U32>,tail => Con{box,tail},move_extents(box_axes(rest,later,Views.later_flags(flags))),later,
        box_extents(rest,later,Views.later_flags(flags),same))

def box_steps(+axes: List<&2,Views.Axis>,+boxes: List<&2,U32>,+flags: List<&2,Bool>,+same: SameLength(axes,boxes)) ->
  {move_steps(box_axes(axes,boxes,flags)) == view_steps(axes) : List<&2,U32>}:
  match axes boxes:
    case Nil{} Nil{}: {==}
    case Nil{} Con{+box,+later}: {==}
    case Con{+axis,+rest} Nil{}: Empty.absurd({move_steps(box_axes(Con{axis,rest},Nil{},flags)) == view_steps(Con{axis,rest}) : List<&2,U32>},same)
    case Con{Views.Axis{+extent,+stride},+rest} Con{+box,+later}:
      %Equal.sym(List<&2,Move.Axis>,box_axes(Con{Views.Axis{extent,stride},rest},Con{box,later},flags),Tensors.view_axes(Con{Views.Axis{box,stride},boxed_axes(rest,later)},flags,Con{Grid.product(later),Grid.strides(later)}),box_axes_con(extent,stride,rest,box,later,flags)) :
        {move_steps(_) == view_steps(Con{Views.Axis{extent,stride},rest}) : List<&2,U32>}
      +back = Views.flag(flags)
      Equal.trans(List<&2,U32>,Con{Move.directed(back,Move.directed(back,stride)),move_steps(box_axes(rest,later,Views.later_flags(flags)))},
        Con{stride,move_steps(box_axes(rest,later,Views.later_flags(flags)))},Con{stride,view_steps(rest)},
        Equal.cong(U32,List<&2,U32>,head => Con{head,move_steps(box_axes(rest,later,Views.later_flags(flags)))},Move.directed(back,Move.directed(back,stride)),stride,
          directed_twice(back,stride)),
        Equal.cong(List<&2,U32>,List<&2,U32>,tail => Con{stride,tail},move_steps(box_axes(rest,later,Views.later_flags(flags))),view_steps(rest),
          box_steps(rest,later,Views.later_flags(flags),same)))

def box_rows(+axes: List<&2,Views.Axis>,+boxes: List<&2,U32>,+flags: List<&2,Bool>,+same: SameLength(axes,boxes)) ->
  {Move.rows_of(box_axes(axes,boxes,flags)) == Grid.product(boxes) : U32}:
  match axes boxes:
    case Nil{} Nil{}: {==}
    case Nil{} Con{+box,+later}: Empty.absurd({Move.rows_of(box_axes(Nil{},Con{box,later},flags)) == Grid.product(Con{box,later}) : U32},same)
    case Con{+axis,+rest} Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest} Con{+box,+later}:
      %Equal.sym(List<&2,Move.Axis>,box_axes(Con{Views.Axis{extent,stride},rest},Con{box,later},flags),Tensors.view_axes(Con{Views.Axis{box,stride},boxed_axes(rest,later)},flags,Con{Grid.product(later),Grid.strides(later)}),box_axes_con(extent,stride,rest,box,later,flags)) :
        {Move.rows_of(_) == Grid.product(Con{box,later}) : U32}
      Equal.cong(U32,U32,cells => (box * cells : U32),Move.rows_of(box_axes(rest,later,Views.later_flags(flags))),Grid.product(later),
        box_rows(rest,later,Views.later_flags(flags),same))

def box_dense(+axes: List<&2,Views.Axis>,+boxes: List<&2,U32>,+flags: List<&2,Bool>,+same: SameLength(axes,boxes)) -> Part.DenseSteps(box_axes(axes,boxes,flags)):
  match axes boxes:
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{+box,+later}: Empty.absurd(Part.DenseSteps(box_axes(Nil{},Con{box,later},flags)),same)
    case Con{+axis,+rest} Nil{}: Unit{}
    case Con{Views.Axis{+extent,+stride},+rest} Con{+box,+later}:
      %Equal.sym(List<&2,Move.Axis>,box_axes(Con{Views.Axis{extent,stride},rest},Con{box,later},flags),Tensors.view_axes(Con{Views.Axis{box,stride},boxed_axes(rest,later)},flags,Con{Grid.product(later),Grid.strides(later)}),box_axes_con(extent,stride,rest,box,later,flags)) :
        Part.DenseSteps(_)
      Held.Both{Equal.sym(U32,Move.rows_of(box_axes(rest,later,Views.later_flags(flags))),Grid.product(later),box_rows(rest,later,Views.later_flags(flags),same)),
        box_dense(rest,later,Views.later_flags(flags),same)}

def move_targets(axes: List<&2,Move.Axis>) -> List<&2,U32>:
  match axes:
    case Nil{}: []
    case Con{Move.Axis{extent,stride,back,+target},rest}: target <> move_targets(rest)

# A box index has the source and target offsets plus the steps times its
# coordinates.
def cell_offsets(+axes: List<&2,Move.Axis>,+index: Nat,+source: U32,+target: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Part.total(axes)),U32.to_nat(cap)) == True{} : Bool},+inside: {Nat.is_lt(index,Part.total(axes)) == True{} : Bool}) ->
  {Move.row_offsets(axes,U32.from_nat(index),source,target) == Move.Offsets{(source + Box.stepped_sum(move_extents(axes),move_steps(axes),index) : U32),
    (target + Box.stepped_sum(move_extents(axes),move_targets(axes),index) : U32)} : Move.Offsets}:
  match axes:
    case Nil{}:
      %Equal.sym(U32,(source + 0 : U32),source,Arithmetic.add_zero(source)) : {Move.Offsets{source,target} == Move.Offsets{_,(target + 0 : U32)} : Move.Offsets}
      %Equal.sym(U32,(target + 0 : U32),target,Arithmetic.add_zero(target)) : {Move.Offsets{source,target} == Move.Offsets{source,_} : Move.Offsets}
      {==}
    case Con{Move.Axis{+extent,+stride,+back,+step},+rest}:
      +cells = Part.total(rest)
      +whole = Nat.mul(U32.to_nat(extent),cells)
      +whole_positive = Order.positive_bound(index,whole,inside)
      +positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),cells,whole_positive)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),cells,whole_positive)
      +whole_room = Order.transitive(whole,Nat.double(whole),U32.to_nat(cap),Order.le_double(whole),room)
      +cells_le = DivisionOrder.le_multiple(U32.to_nat(extent),cells,extent_positive)
      +extent_le = RowMajor.product_covers(U32.to_nat(extent),cells,positive)
      +inner_room = Order.transitive(Nat.double(cells),Nat.double(whole),U32.to_nat(cap),Order.double_le(cells,whole,cells_le),room)
      +extent_room = Order.transitive(Nat.double(U32.to_nat(extent)),Nat.double(whole),U32.to_nat(cap),Order.double_le(U32.to_nat(extent),whole,extent_le),room)
      +rows = Part.rows_exact(rest,cap,cap_ok,positive,Order.transitive(cells,whole,U32.to_nat(cap),cells_le,whole_room))
      +quotient = Nat.div(index,cells)
      +remainder = Nat.mod(index,cells)
      +remainder_in = DivisionOrder.remainder_small(index,cells,positive)
      +quotient_in = DivisionOrder.quotient_below(index,U32.to_nat(extent),cells,positive,inside)
      +index_back = Bounds.roundtrip(index,cap,cap_ok,Order.transitive(index,whole,U32.to_nat(cap),Bounds.strict_le(index,whole,inside),whole_room))
      +exact = Equal.trans(Nat,U32.to_nat(U32.from_nat(index)),index,Nat.add(Nat.mul(quotient,cells),remainder),index_back,DivisionOrder.quotient_exact(index,cells,positive))
      +peeled = Part.peel(extent,stride,back,step,rest,U32.from_nat(index),source,target,quotient,remainder,cap,cap_ok,cells,rows,exact,remainder_in,quotient_in,inner_room,extent_room)
      +moved = (U32.from_nat(quotient) * Move.directed(back,stride) : U32)
      +moved_target = (U32.from_nat(quotient) * step : U32)
      +tail = Box.stepped_sum(move_extents(rest),move_steps(rest),remainder)
      +tail_target = Box.stepped_sum(move_extents(rest),move_targets(rest),remainder)
      %total_elements(rest) : {Move.row_offsets(Con{Move.Axis{extent,stride,back,step},rest},U32.from_nat(index),source,target) ==
        Move.Offsets{(source + (U32.from_nat(Nat.div(index,_)) * Move.directed(back,stride) + Box.stepped_sum(move_extents(rest),move_steps(rest),Nat.mod(index,_)) : U32) : U32),
          (target + (U32.from_nat(Nat.div(index,_)) * step + Box.stepped_sum(move_extents(rest),move_targets(rest),Nat.mod(index,_)) : U32) : U32)} : Move.Offsets}
      Equal.trans(Move.Offsets,Move.row_offsets(Con{Move.Axis{extent,stride,back,step},rest},U32.from_nat(index),source,target),
        Move.row_offsets(rest,U32.from_nat(remainder),(source + moved : U32),(target + moved_target : U32)),
        Move.Offsets{(source + (moved + tail : U32) : U32),(target + (moved_target + tail_target : U32) : U32)},peeled,
        Equal.trans(Move.Offsets,Move.row_offsets(rest,U32.from_nat(remainder),(source + moved : U32),(target + moved_target : U32)),
          Move.Offsets{((source + moved : U32) + tail : U32),((target + moved_target : U32) + tail_target : U32)},
          Move.Offsets{(source + (moved + tail : U32) : U32),(target + (moved_target + tail_target : U32) : U32)},
          cell_offsets(rest,remainder,(source + moved : U32),(target + moved_target : U32),cap,cap_ok,inner_room,remainder_in),
          {%Equal.sym(U32,((source + moved : U32) + tail : U32),(source + (moved + tail : U32) : U32),Arithmetic.add_associative(source,moved,tail)) :
              {Move.Offsets{_,((target + moved_target : U32) + tail_target : U32)} == Move.Offsets{(source + (moved + tail : U32) : U32),(target + (moved_target + tail_target : U32) : U32)} : Move.Offsets}
            %Equal.sym(U32,((target + moved_target : U32) + tail_target : U32),(target + (moved_target + tail_target : U32) : U32),Arithmetic.add_associative(target,moved_target,tail_target)) :
              {Move.Offsets{(source + (moved + tail : U32) : U32),_} == Move.Offsets{(source + (moved + tail : U32) : U32),(target + (moved_target + tail_target : U32) : U32)} : Move.Offsets}
            {==} :
            {Move.Offsets{((source + moved : U32) + tail : U32),((target + moved_target : U32) + tail_target : U32)} ==
              Move.Offsets{(source + (moved + tail : U32) : U32),(target + (moved_target + tail_target : U32) : U32)} : Move.Offsets}}))

# A box index reads the source offset plus the steps times its coordinates.
def cell_sum(+axes: List<&2,Move.Axis>,+index: Nat,+source: U32,+target: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Part.total(axes)),U32.to_nat(cap)) == True{} : Bool},+inside: {Nat.is_lt(index,Part.total(axes)) == True{} : Bool}) ->
  {Part.cell_source(axes,index,source,target) == (source + Box.stepped_sum(move_extents(axes),move_steps(axes),index) : U32) : U32}:
  Equal.cong(Move.Offsets,U32,Part.offsets_source,Move.row_offsets(axes,U32.from_nat(index),source,target),Move.Offsets{(source + Box.stepped_sum(move_extents(axes),move_steps(axes),index) : U32),
    (target + Box.stepped_sum(move_extents(axes),move_targets(axes),index) : U32)},cell_offsets(axes,index,source,target,cap,cap_ok,room,inside))

# A box index writes the target offset plus the target steps times its
# coordinates.
def cell_target_sum(+axes: List<&2,Move.Axis>,+index: Nat,+source: U32,+target: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.double(Part.total(axes)),U32.to_nat(cap)) == True{} : Bool},+inside: {Nat.is_lt(index,Part.total(axes)) == True{} : Bool}) ->
  {Part.cell_target(axes,index,source,target) == (target + Box.stepped_sum(move_extents(axes),move_targets(axes),index) : U32) : U32}:
  Equal.cong(Move.Offsets,U32,Part.offsets_target,Move.row_offsets(axes,U32.from_nat(index),source,target),Move.Offsets{(source + Box.stepped_sum(move_extents(axes),move_steps(axes),index) : U32),
    (target + Box.stepped_sum(move_extents(axes),move_targets(axes),index) : U32)},cell_offsets(axes,index,source,target,cap,cap_ok,room,inside))

def same_grid_box(+axes: List<&2,Views.Axis>,+axis: U32,+prefix: U32,+units: U32) -> SameLength(axes,Grid.box(Views.axis_extents(axes),axis,prefix,units)):
  match axes:
    case Nil{}: Unit{}
    case Con{Views.Axis{+extent,+stride},+rest}: same_grid_box(rest,(axis + 1 : U32),prefix,units)

def same_box_of(+axes: List<&2,Views.Axis>,+grid: Grid.Grid,+count: U32) -> SameLength(axes,Grid.box_of(grid,Views.axis_extents(axes),count)):
  match grid:
    case Grid.Grid{+total,+prefix,+outer,+extent,+unit_cells,+piece_units,+per_outer,+pieces}:
      same_grid_box(axes,0,prefix,(count / U32.max(unit_cells,1) : U32))

# ---- A view read over a target piece ----

# Box index index of the part of a piece reads the view offset plus the view
# strides times the coordinates of cell start + index.
def view_piece_with(+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+grid: Grid.Grid,+piece: U32,+index: Nat,
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +boxed: Box.PieceBox(Views.axis_extents(axes),view_steps(Views.mirrored_axes(axes,flags)),Grid.box_of(grid,Views.axis_extents(axes),Grid.span(grid,piece)),
    U32.to_nat(Grid.start(grid,piece)),U32.to_nat(Grid.span(grid,piece)),index)) ->
  {Part.cell_source(box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(grid,Views.axis_extents(axes),Grid.span(grid,piece)),flags),index,
      view_offset(Grid.piece_view(grid,Views.axis_extents(axes),Grid.start(grid,piece),Grid.span(grid,piece),Views.mirrored(Views.View{axes,offset},flags))),0) ==
    (Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),view_steps(Views.mirrored_axes(axes,flags)),
      Nat.add(U32.to_nat(Grid.start(grid,piece)),index)) : U32) : U32}:
  Held.Both{+shift,+more} = boxed
  Held.Both{+index_box,+rest} = more
  Held.Both{+box_le,+box_end} = rest
  Held.Both{+ending,+box_cells} = box_end
  +extents = Views.axis_extents(axes)
  +mirrored = Views.mirrored_axes(axes,flags)
  +moved_offset = Views.mirrored_offset(axes,flags,offset)
  +steps = view_steps(mirrored)
  +start = Grid.start(grid,piece)
  +first = U32.to_nat(start)
  +span = Grid.span(grid,piece)
  +boxes = Grid.box_of(grid,extents,span)
  +cap = Machine.half_word(limits)
  +cap_ok = LimitBounds.half_word_fits(limits)
  +same_extents = mirrored_extents(axes,flags)
  +back = Equal.sym(List<&2,U32>,Views.axis_extents(mirrored),extents,same_extents)
  +same = {%same_extents : SameLength(mirrored,Grid.box_of(grid,_,span)) same_box_of(mirrored,grid,span) : SameLength(mirrored,boxes)}
  +start_in = Order.le_lt_trans(first,Nat.add(first,index),ShapeLaws.elements(extents),Order.le_add(first,index),ending)
  +whole_room = LimitBounds.double_bound(limits,ShapeLaws.elements(extents),room)
  +read_offset = view_offset(Grid.restricted(mirrored,Grid.coordinates(extents,start),boxes,moved_offset))
  +before = Box.stepped_sum(extents,steps,first)
  +offset_exact = {%same_extents :
      {view_offset(Grid.restricted(mirrored,Grid.coordinates(_,start),boxes,moved_offset)) == (moved_offset + Box.stepped_sum(_,steps,first) : U32) : U32}
    restricted_offset(mirrored,boxes,start,moved_offset,cap,cap_ok,
      {%back : {Nat.is_le(Nat.double(ShapeLaws.elements(_)),U32.to_nat(cap)) == True{} : Bool} whole_room :
        {Nat.is_le(Nat.double(ShapeLaws.elements(Views.axis_extents(mirrored))),U32.to_nat(cap)) == True{} : Bool}},
      {%back : {Nat.is_lt(first,ShapeLaws.elements(_)) == True{} : Bool} start_in : {Nat.is_lt(first,ShapeLaws.elements(Views.axis_extents(mirrored))) == True{} : Bool}},
      same) :
    {read_offset == (moved_offset + before : U32) : U32}}
  +moves = box_axes(mirrored,boxes,flags)
  +box_total = Equal.trans(Nat,Part.total(moves),ShapeLaws.elements(move_extents(moves)),ShapeLaws.elements(boxes),total_elements(moves),
    Equal.cong(List<&2,U32>,Nat,value => ShapeLaws.elements(value),move_extents(moves),boxes,box_extents(mirrored,boxes,flags,same)))
  +index_moves = {%Equal.sym(Nat,Part.total(moves),ShapeLaws.elements(boxes),box_total) : {Nat.is_lt(index,_) == True{} : Bool} index_box :
    {Nat.is_lt(index,Part.total(moves)) == True{} : Bool}}
  +moves_room = {%Equal.sym(Nat,Part.total(moves),ShapeLaws.elements(boxes),box_total) : {Nat.is_le(Nat.double(_),U32.to_nat(cap)) == True{} : Bool}
    Order.transitive(Nat.double(ShapeLaws.elements(boxes)),Nat.double(ShapeLaws.elements(extents)),U32.to_nat(cap),
      Order.double_le(ShapeLaws.elements(boxes),ShapeLaws.elements(extents),box_le),whole_room) :
    {Nat.is_le(Nat.double(Part.total(moves)),U32.to_nat(cap)) == True{} : Bool}}
  +inside = Box.stepped_sum(boxes,steps,index)
  +read_sum = (read_offset + Box.stepped_sum(move_extents(moves),move_steps(moves),index) : U32)
  Equal.trans(U32,Part.cell_source(moves,index,read_offset,0),read_sum,(moved_offset + Box.stepped_sum(extents,steps,Nat.add(first,index)) : U32),
    cell_sum(moves,index,read_offset,0,cap,cap_ok,moves_room,index_moves),
    Equal.trans(U32,read_sum,(read_offset + inside : U32),(moved_offset + Box.stepped_sum(extents,steps,Nat.add(first,index)) : U32),
      {%box_extents(mirrored,boxes,flags,same) : {read_sum == (read_offset + Box.stepped_sum(_,steps,index) : U32) : U32}
        %box_steps(mirrored,boxes,flags,same) : {read_sum == (read_offset + Box.stepped_sum(move_extents(moves),_,index) : U32) : U32}
        {==} : {read_sum == (read_offset + inside : U32) : U32}},
      Equal.trans(U32,(read_offset + inside : U32),((moved_offset + before : U32) + inside : U32),(moved_offset + Box.stepped_sum(extents,steps,Nat.add(first,index)) : U32),
        Equal.cong(U32,U32,value => (value + inside : U32),read_offset,(moved_offset + before : U32),offset_exact),
        Equal.trans(U32,((moved_offset + before : U32) + inside : U32),(moved_offset + (before + inside : U32) : U32),
          (moved_offset + Box.stepped_sum(extents,steps,Nat.add(first,index)) : U32),Arithmetic.add_associative(moved_offset,before,inside),
          Equal.cong(U32,U32,value => (moved_offset + value : U32),(before + inside : U32),Box.stepped_sum(extents,steps,Nat.add(first,index)),
            Equal.trans(U32,(before + inside : U32),(inside + before : U32),Box.stepped_sum(extents,steps,Nat.add(first,index)),
              Arithmetic.add_commutative(before,inside),shift))))))

def view_piece_source(+limits: Machine.Limits,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+target: U32,
  +room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Views.axis_extents(axes))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece: U32,+inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Grid.grid(Views.axis_extents(axes),target)))) == True{} : Bool},+index: Nat,
  +index_in: {Nat.is_lt(index,U32.to_nat(Grid.span(Grid.grid(Views.axis_extents(axes),target),piece))) == True{} : Bool}) ->
  {Part.cell_source(box_axes(Views.mirrored_axes(axes,flags),Grid.box_of(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),
        Grid.span(Grid.grid(Views.axis_extents(axes),target),piece)),flags),index,
      view_offset(Grid.piece_view(Grid.grid(Views.axis_extents(axes),target),Views.axis_extents(axes),Grid.start(Grid.grid(Views.axis_extents(axes),target),piece),
        Grid.span(Grid.grid(Views.axis_extents(axes),target),piece),Views.mirrored(Views.View{axes,offset},flags))),0) ==
    (Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(Views.axis_extents(axes),view_steps(Views.mirrored_axes(axes,flags)),
      Nat.add(U32.to_nat(Grid.start(Grid.grid(Views.axis_extents(axes),target),piece)),index)) : U32) : U32}:
  view_piece_with(limits,axes,flags,offset,Grid.grid(Views.axis_extents(axes),target),piece,index,room,
    Box.grid_piece_box(limits,Views.axis_extents(axes),view_steps(Views.mirrored_axes(axes,flags)),target,room,nonempty,rank_room,piece,inside,index,index_in))
