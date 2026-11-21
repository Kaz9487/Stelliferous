# The source cells a split reads lie in the box its component takes: every
# target cell axis paired with a source axis reads it at its coordinate, the
# anchors read the source groups their lanes cover, and every other axis
# reads nothing.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tasks.bend as Tasks
import ./tensor_task_reads.bend as TaskReads
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./word_multiplication.bend as Multiplication
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_placing_law.bend as PlacingLaw
import ./tensor_placing_facts.bend as PlacingFacts
import ./tensor_move_readback.bend as Held
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_grid_pieces.bend as GridPieces
import ./tensor_box_digits.bend as BoxDigits
import ./tensor_relane_geometry.bend as RelaneGeometry
import ../kernel_gemm.bend as KernelGemm
import ./tensor_components.bend as Components
import ./tensor_target_cells.bend as Targets
import ./boolean_logic.bend as Logic
import ./list_storage_readback.bend as ListReadback
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./machine_limit_bounds.bend as Limits
import ../machine_limits.bend as Machine
import ./tensor_moved.bend as Moved
import ./tensor_view_reads.bend as ViewReads
import ./tensor_box_cells.bend as BoxCells
import ../tensor.bend as Tensors
import ./nat_division.bend as Division
import ./tensor_split_value.bend as SplitValue
import ./tensor_split_cell.bend as SplitCell
import ./tensor_split_gather.bend as SplitGather
import ./tensor_split_laws.bend as SplitLaws
import ./nat_ceiling.bend as Ceiling
import ./tensor_tiled_values.bend as TiledValues
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./word_encoding.bend as WordEncoding

# ---- What the links of a split read ----

# The first digit of a list, zero past its end, and the digits after it.
def digit_head(digits: List<&2,Nat>) -> Nat:
  match digits:
    case Nil{}: 0n
    case Con{+digit,rest}: digit

def digit_rest(digits: List<&2,Nat>) -> List<&2,Nat>:
  match digits:
    case Nil{}: Nil{}
    case Con{digit,+rest}: rest

# The extents of the target cell axes of links: every axis but the new
# lanes.
def link_extents(links: List<&2,Tiled.Link>,shape: List<&2,U32>) -> List<&2,U32>:
  match links shape:
    case Con{Tiled.Lane{},+links} Con{extent,+shape}: link_extents(links,shape)
    case Con{link,+links} Con{+extent,+shape}: extent <> link_extents(links,shape)
    case links shape: []

# The value every reading of the links reads at the digits of a target cell:
# the anchors read the source group cell, every other axis its digit.
def link_values(links: List<&2,Tiled.Link>,shape: List<&2,U32>,+digits: List<&2,Nat>,+cell: Nat) -> List<&2,Nat>:
  match links shape:
    case Con{Tiled.Lane{},+links} Con{extent,+shape}: link_values(links,shape,digits,cell)
    case Con{Tiled.Anchor{},+links} Con{extent,+shape}: cell <> link_values(links,shape,digit_rest(digits),cell)
    case Con{link,+links} Con{extent,+shape}: digit_head(digits) <> link_values(links,shape,digit_rest(digits),cell)
    case links shape: []

# The stride of a source cell axis as a number.
def weight(+cells: List<&2,U32>,+axis: U32) -> Nat:
  U32.to_nat(Tasks.nth(Tasks.strides_of(cells),axis))

# The source cell number the readings of the links reach at the values of a
# target cell.
def link_sum(links: List<&2,Tiled.Link>,shape: List<&2,U32>,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,
  +cell: Nat) -> Nat:
  match links shape:
    case Con{Tiled.Lane{},+links} Con{extent,+shape}: link_sum(links,shape,digits,cells,lane,position,groups_axis,cell)
    case Con{Tiled.Linked{+axis},+links} Con{extent,+shape}:
      Nat.add(Nat.mul(digit_head(digits),weight(cells,Tiled.source_axis(lane,True{},position,axis))),
        link_sum(links,shape,digit_rest(digits),cells,lane,position,groups_axis,cell))
    case Con{Tiled.Anchor{},+links} Con{extent,+shape}:
      Nat.add(Nat.mul(cell,weight(cells,groups_axis)),link_sum(links,shape,digit_rest(digits),cells,lane,position,groups_axis,cell))
    case Con{Tiled.Alone{},+links} Con{extent,+shape}: link_sum(links,shape,digit_rest(digits),cells,lane,position,groups_axis,cell)
    case links shape: 0n

# A term of a read and the words after it.
def read_term(+value: Nat,+stride: U32,+sum: Nat,+tail: U32) ->
  {((U32.from_nat(value) * stride : U32) + (U32.from_nat(sum) + tail : U32) : U32) ==
    (U32.from_nat(Nat.add(Nat.mul(value,U32.to_nat(stride)),sum)) + tail : U32) : U32}:
  %WordArithmetic.native_add(Nat.mul(value,U32.to_nat(stride)),sum) :
    {((U32.from_nat(value) * stride : U32) + (U32.from_nat(sum) + tail : U32) : U32) == (_ + tail : U32) : U32}
  %Multiplication.native_encoded_multiply(value,U32.to_nat(stride)) :
    {((U32.from_nat(value) * stride : U32) + (U32.from_nat(sum) + tail : U32) : U32) == ((_ + U32.from_nat(sum) : U32) + tail : U32) : U32}
  %Equal.sym(U32,U32.from_nat(U32.to_nat(stride)),stride,Arithmetic.native_roundtrip(stride)) :
    {((U32.from_nat(value) * stride : U32) + (U32.from_nat(sum) + tail : U32) : U32) == (((U32.from_nat(value) * _ : U32) + U32.from_nat(sum) : U32) + tail : U32) : U32}
  Equal.sym(U32,(((U32.from_nat(value) * stride : U32) + U32.from_nat(sum) : U32) + tail : U32),((U32.from_nat(value) * stride : U32) + (U32.from_nat(sum) + tail : U32) : U32),
    Arithmetic.add_associative((U32.from_nat(value) * stride : U32),U32.from_nat(sum),tail))

# The readings of the links, before any readings after them, reach at the
# values of a target cell their source cell number before what those after
# them reach.
def links_word(links: List<&2,Tiled.Link>,shape: List<&2,U32>,+digits: List<&2,Nat>,+cells: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,
  +lane: Nat,+position: Grid.Position,+width: U32,+source_groups: U32,+groups_axis: U32,+index: U32,+cell: Nat,+tail: List<&2,Tasks.Reading>,+tail_values: List<&2,Nat>) ->
  {TaskReads.placed_word(List.append(&2,Tasks.Reading,Tiled.link_readings(links,shape,Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},index),tail),
      List.append(&2,Nat,link_values(links,shape,digits,cell),tail_values),Tasks.strides_of(cells)) ==
    (U32.from_nat(link_sum(links,shape,digits,cells,lane,position,groups_axis,cell)) + TaskReads.placed_word(tail,tail_values,Tasks.strides_of(cells)) : U32) : U32}:
  match links shape:
    case Con{Tiled.Lane{},+links} Con{+extent,+shape}:
      links_word(links,shape,digits,cells,source_grid,target_grid,lane,position,width,source_groups,groups_axis,index,cell,tail,tail_values)
    case Con{Tiled.Linked{+axis},+links} Con{+extent,+shape}:
      +strides = Tasks.strides_of(cells)
      +sides = {Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis} : Tiled.Sides}
      +stride = Tasks.nth(strides,Tiled.source_axis(lane,True{},position,axis))
      +sum = link_sum(links,shape,digit_rest(digits),cells,lane,position,groups_axis,cell)
      +rest = TaskReads.placed_word(tail,tail_values,strides)
      %Equal.sym(U32,TaskReads.placed_word(List.append(&2,Tasks.Reading,Tiled.link_readings(links,shape,sides,(index + 1 : U32)),tail),
          List.append(&2,Nat,link_values(links,shape,digit_rest(digits),cell),tail_values),strides),(U32.from_nat(sum) + rest : U32),
          links_word(links,shape,digit_rest(digits),cells,source_grid,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32),cell,tail,tail_values)) :
        {((U32.from_nat(digit_head(digits)) * stride : U32) + _ : U32) == (U32.from_nat(Nat.add(Nat.mul(digit_head(digits),U32.to_nat(stride)),sum)) + rest : U32) : U32}
      read_term(digit_head(digits),stride,sum,rest)
    case Con{Tiled.Anchor{},+links} Con{+extent,+shape}:
      +strides = Tasks.strides_of(cells)
      +sides = {Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis} : Tiled.Sides}
      +stride = Tasks.nth(strides,groups_axis)
      +sum = link_sum(links,shape,digit_rest(digits),cells,lane,position,groups_axis,cell)
      +rest = TaskReads.placed_word(tail,tail_values,strides)
      %Equal.sym(U32,TaskReads.placed_word(List.append(&2,Tasks.Reading,Tiled.link_readings(links,shape,sides,(index + 1 : U32)),tail),
          List.append(&2,Nat,link_values(links,shape,digit_rest(digits),cell),tail_values),strides),(U32.from_nat(sum) + rest : U32),
          links_word(links,shape,digit_rest(digits),cells,source_grid,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32),cell,tail,tail_values)) :
        {((U32.from_nat(cell) * stride : U32) + _ : U32) == (U32.from_nat(Nat.add(Nat.mul(cell,U32.to_nat(stride)),sum)) + rest : U32) : U32}
      read_term(cell,stride,sum,rest)
    case Con{Tiled.Alone{},+links} Con{+extent,+shape}:
      links_word(links,shape,digit_rest(digits),cells,source_grid,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32),cell,tail,tail_values)
    case Nil{} +shape:
      +strides = Tasks.strides_of(cells)
      Equal.sym(U32,(0 + TaskReads.placed_word(tail,tail_values,strides) : U32),TaskReads.placed_word(tail,tail_values,strides),
        Arithmetic.add_zero_left(TaskReads.placed_word(tail,tail_values,strides)))
    case Con{+link,+links} Nil{}:
      +strides = Tasks.strides_of(cells)
      Equal.sym(U32,(0 + TaskReads.placed_word(tail,tail_values,strides) : U32),TaskReads.placed_word(tail,tail_values,strides),
        Arithmetic.add_zero_left(TaskReads.placed_word(tail,tail_values,strides)))

# ---- The stride of a source cell axis ----

# The stride of an axis inside the cells is the cells after it.
def stride_weight(cells: List<&2,U32>,place: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(cap)) == True{} : Bool},+inside: {Nat.is_lt(place,List.length(&2,U32,cells)) == True{} : Bool}) ->
  {weight(cells,U32.from_nat(place)) == ShapeLaws.elements(PlacingLaw.dropped(~U32,1n+place,cells)) : Nat}:
  match cells place:
    case Con{+extent,+rest} 0n:
      +inner = ShapeLaws.elements(rest)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive)
      +rest_room = Order.transitive(inner,Nat.mul(U32.to_nat(extent),inner),U32.to_nat(cap),DivisionOrder.le_multiple(U32.to_nat(extent),inner,extent_positive),room)
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridPieces.strides_con(extent,rest)) :
        {U32.to_nat(Tasks.nth(_,U32.from_nat(0n))) == inner : Nat}
      %BoxDigits.product_from(rest) : {U32.to_nat(_) == inner : Nat}
      Bounds.roundtrip(inner,cap,cap_ok,rest_room)
    case Con{+extent,+rest} 1n+ +more:
      +inner = ShapeLaws.elements(rest)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive)
      +rest_room = Order.transitive(inner,Nat.mul(U32.to_nat(extent),inner),U32.to_nat(cap),DivisionOrder.le_multiple(U32.to_nat(extent),inner,extent_positive),room)
      +length = List.length(&2,U32,rest)
      +rest_rank = Order.transitive(length,1n+length,U32.to_nat(cap),Bounds.strict_le(length,1n+length,Order.lt_succ(length,length,Order.reflexive(length))),rank_room)
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridPieces.strides_con(extent,rest)) :
        {U32.to_nat(Tasks.nth(_,U32.from_nat(1n+more))) == ShapeLaws.elements(PlacingLaw.dropped(~U32,1n+more,rest)) : Nat}
      %Equal.sym(U32,Tasks.nth(Con{Grid.product(rest),Grid.strides(rest)},U32.from_nat(1n+more)),Tasks.nth(Grid.strides(rest),U32.from_nat(more)),
          RelaneGeometry.nth_next(Grid.product(rest),Grid.strides(rest),more,cap,cap_ok,Order.transitive(1n+more,length,U32.to_nat(cap),Order.strict_follow(more,length,inside),
            rest_rank))) :
        {U32.to_nat(_) == ShapeLaws.elements(PlacingLaw.dropped(~U32,1n+more,rest)) : Nat}
      stride_weight(rest,more,cap,cap_ok,rest_room,inner_positive,rest_rank,inside)
    case Nil{} +place:
      Empty.absurd({weight(Nil{},U32.from_nat(place)) == ShapeLaws.elements(PlacingLaw.dropped(~U32,1n+place,Nil{})) : Nat},Order.below_zero(place,inside))

# ---- The source cell axes of paired axes ----

# The source cell axis of a logical source axis of a split: one earlier past
# the lanes, one later with the groups outermost.
def cell_place(+lane: Nat,+position: Grid.Position,+axis: Nat) -> Nat:
  +inner = Bool.pick(Nat,True{} && Nat.is_lt(lane,axis),Nat.sub(axis,1n),axis)
  Bool.pick(Nat,True{} && Grid.is_outermost(position),1n+inner,inner)

def place_of(+after: Bool,+outer: Bool,+axis: Nat) -> Nat:
  Bool.pick(Nat,outer,1n+Bool.pick(Nat,after,Nat.sub(axis,1n),axis),Bool.pick(Nat,after,Nat.sub(axis,1n),axis))

def place_step(+after: Bool,+outer: Bool,+axis: Nat,+above: {Nat.is_le(Bool.pick(Nat,after,1n,0n),axis) == True{} : Bool}) ->
  {place_of(after,outer,Nat.add(axis,1n)) == 1n+place_of(after,outer,axis) : Nat}:
  match after outer:
    case False{} False{}: Addition.successor_width(axis)
    case False{} True{}: Equal.cong(Nat,Nat,value => 1n+value,Nat.add(axis,1n),1n+axis,Addition.successor_width(axis))
    case True{} False{}:
      Equal.trans(Nat,Nat.sub(Nat.add(axis,1n),1n),axis,1n+Nat.sub(axis,1n),DivisionOrder.subtract_sum(axis,1n),
        Equal.sym(Nat,1n+Nat.sub(axis,1n),axis,Order.restore_shift(1n,axis,above)))
    case True{} True{}:
      Equal.cong(Nat,Nat,value => 1n+value,Nat.sub(Nat.add(axis,1n),1n),1n+Nat.sub(axis,1n),
        Equal.trans(Nat,Nat.sub(Nat.add(axis,1n),1n),axis,1n+Nat.sub(axis,1n),DivisionOrder.subtract_sum(axis,1n),
          Equal.sym(Nat,1n+Nat.sub(axis,1n),axis,Order.restore_shift(1n,axis,above))))

# Which side of the lanes the source axes left lie on: before, all of them,
# or after.
def Side(+side: Bool,+lane: Nat,+axis: Nat,+count: Nat) -> Data:
  match side:
    case False{}: {Nat.is_le(Nat.add(axis,count),lane) == True{} : Bool}
    case True{}: {Nat.is_lt(lane,axis) == True{} : Bool}

def side_here(+side: Bool,+lane: Nat,+axis: Nat,+count: Nat,+side_ok: Side(side,lane,axis,1n+count)) -> {Nat.is_lt(lane,axis) == side : Bool}:
  match side:
    case False{}: Order.not_below(axis,lane,Order.transitive(axis,Nat.add(axis,1n+count),lane,Order.le_add(axis,1n+count),side_ok))
    case True{}: side_ok

def side_next(+side: Bool,+lane: Nat,+axis: Nat,+count: Nat,+side_ok: Side(side,lane,axis,1n+count)) -> {Nat.is_lt(lane,Nat.add(axis,1n)) == side : Bool}:
  match side:
    case False{}:
      Order.not_below(Nat.add(axis,1n),lane,Order.transitive(Nat.add(axis,1n),Nat.add(axis,1n+count),lane,Order.offset_le(axis,1n,1n+count,Order.zero_le(count)),side_ok))
    case True{}: Order.lt_le(lane,axis,Nat.add(axis,1n),side_ok,Order.le_add(axis,1n))

def side_after(+side: Bool,+lane: Nat,+axis: Nat,+count: Nat,+side_ok: Side(side,lane,axis,1n+count)) -> Side(side,lane,Nat.add(axis,1n),count):
  match side:
    case False{}:
      {%Equal.sym(Nat,Nat.add(Nat.add(axis,1n),count),Nat.add(axis,Nat.add(1n,count)),Addition.associative(axis,1n,count)) : {Nat.is_le(_,lane) == True{} : Bool}
        side_ok : {Nat.is_le(Nat.add(Nat.add(axis,1n),count),lane) == True{} : Bool}}
    case True{}: Order.lt_le(lane,axis,Nat.add(axis,1n),side_ok,Order.le_add(axis,1n))

def side_above(+side: Bool,+lane: Nat,+axis: Nat,+count: Nat,+side_ok: Side(side,lane,axis,1n+count)) -> {Nat.is_le(Bool.pick(Nat,side,1n,0n),axis) == True{} : Bool}:
  match side:
    case False{}: Order.zero_le(axis)
    case True{}: Order.transitive(1n,1n+lane,axis,Order.zero_le(lane),Order.strict_follow(lane,axis,side_ok))

# The source cell axis after a paired source axis on the same side of the
# lanes is the next.
def place_next(+side: Bool,+lane: Nat,+position: Grid.Position,+axis: Nat,+count: Nat,+side_ok: Side(side,lane,axis,1n+count)) ->
  {cell_place(lane,position,Nat.add(axis,1n)) == 1n+cell_place(lane,position,axis) : Nat}:
  +here = side_here(side,lane,axis,count,side_ok)
  +next = side_next(side,lane,axis,count,side_ok)
  %Equal.sym(Bool,Nat.is_lt(lane,Nat.add(axis,1n)),side,next) :
    {place_of(True{} && _,True{} && Grid.is_outermost(position),Nat.add(axis,1n)) == 1n+cell_place(lane,position,axis) : Nat}
  %Equal.sym(Bool,Nat.is_lt(lane,axis),side,here) :
    {place_of(True{} && side,True{} && Grid.is_outermost(position),Nat.add(axis,1n)) == 1n+place_of(True{} && _,True{} && Grid.is_outermost(position),axis) : Nat}
  place_step(True{} && side,True{} && Grid.is_outermost(position),axis,side_above(side,lane,axis,count,side_ok))

# ---- Paired axes ----

# Whether the anchors of links read the source group cell at the digits of
# their target axes: it lies among the groups of the anchor lanes.
def anchor_fits(links: List<&2,Tiled.Link>,shape: List<&2,U32>,+digits: List<&2,Nat>,+width: U32,+cell: Nat) -> Bool:
  match links shape:
    case Con{Tiled.Lane{},+links} Con{extent,+shape}: anchor_fits(links,shape,digits,width,cell)
    case Con{Tiled.Anchor{},+links} Con{extent,+shape}:
      +first = Nat.add(U32.to_nat(0),Nat.mul(U32.to_nat(width),digit_head(digits)))
      +eights = U32.to_nat(U32.max(8,1))
      Nat.is_le(Nat.div(first,eights),Nat.add(U32.to_nat(0),cell)) && (Nat.is_le(Nat.add(U32.to_nat(0),cell),Nat.div(Nat.sub(Nat.add(first,U32.to_nat(width)),1n),eights)) &&
        anchor_fits(links,shape,digit_rest(digits),width,cell))
    case Con{link,+links} Con{extent,+shape}: anchor_fits(links,shape,digit_rest(digits),width,cell)
    case links shape: True{}

# Paired links at the digits of their target axes, before more links, reach
# the number of those digits in units of the source cells after the paired
# source axes. The target axes have the size of the source axes and the
# links hold no anchors.
def PairedSum(found: Maybe<&2,List<&2,Tiled.Link>>,+target: List<&2,U32>,+size: Nat,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,+width: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+more_digits: List<&2,Nat>) -> Data:
  match found:
    case Some{+links}:
      KernelGemm.Both<{ShapeLaws.elements(target) == size : Nat},KernelGemm.Both<
        {link_sum(List.append(&2,Tiled.Link,links,more_links),List.append(&2,U32,target,more_shape),List.append(&2,Nat,digits,more_digits),cells,lane,position,groups_axis,cell) ==
          Nat.add(Nat.mul(ShapeLaws.digits_value(target,digits),ShapeLaws.elements(rest)),link_sum(more_links,more_shape,more_digits,cells,lane,position,groups_axis,cell)) : Nat},
        {anchor_fits(List.append(&2,Tiled.Link,links,more_links),List.append(&2,U32,target,more_shape),List.append(&2,Nat,digits,more_digits),width,cell) ==
          anchor_fits(more_links,more_shape,more_digits,width,cell) : Bool}>>
    case None{}: Unit

# The only digit below one is zero.
def digit_zero(digit: Nat,+less: {Nat.is_lt(digit,1n) == True{} : Bool}) -> {digit == 0n : Nat}:
  match digit:
    case 0n: {==}
    case 1n+ +more: Empty.absurd({1n+more == 0n : Nat},Order.below_zero(more,less))

# An axis of one index leaves the cells of the axes after it.
def unit_elements(+extent: U32,+rest: List<&2,U32>,+unit: {extent == 1 : U32}) -> {ShapeLaws.elements(rest) == ShapeLaws.elements(Con{extent,rest}) : Nat}:
  %Equal.sym(U32,extent,1,unit) : {ShapeLaws.elements(rest) == Nat.mul(U32.to_nat(_),ShapeLaws.elements(rest)) : Nat}
  Equal.sym(Nat,Nat.mul(1n,ShapeLaws.elements(rest)),ShapeLaws.elements(rest),TaskReads.one_times(ShapeLaws.elements(rest)))

# Both ends reached: no links, no digits.
def ended_sum(+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,+width: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+more_digits: List<&2,Nat>,+below: Coordinates.DigitsBelow(Nil{},digits)) ->
  PairedSum(Some{[]},Nil{},ShapeLaws.elements(Nil{}),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits):
  match digits:
    case Nil{}: KernelGemm.Both{{==},KernelGemm.Both{{==},{==}}}
    case Con{+digit,+digits}: Empty.absurd(PairedSum(Some{[]},Nil{},ShapeLaws.elements(Nil{}),Con{digit,digits},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits),below)

# A target axis of one index reads nothing at its digit zero.
def alone_sum(found: Maybe<&2,List<&2,Tiled.Link>>,+extent: U32,+target: List<&2,U32>,+size: Nat,+digit: Nat,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,+width: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+more_digits: List<&2,Nat>,
  +unit: {extent == 1 : U32},+zero: {digit == 0n : Nat},+held: PairedSum(found,target,size,digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)) ->
  PairedSum(Tiled.prepended(Tiled.Alone{},found),Con{extent,target},size,Con{digit,digits},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits):
  match found:
    case Some{+links}:
      KernelGemm.Both{+same,KernelGemm.Both{+sum,+fits}} = held
      +inner = ShapeLaws.digits_value(target,digits)
      +outer = ShapeLaws.elements(rest)
      +later = link_sum(more_links,more_shape,more_digits,cells,lane,position,groups_axis,cell)
      +sizes = Equal.trans(Nat,ShapeLaws.elements(Con{extent,target}),ShapeLaws.elements(target),size,
        Equal.sym(Nat,ShapeLaws.elements(target),ShapeLaws.elements(Con{extent,target}),unit_elements(extent,target,unit)),same)
      +sums = {%Equal.sym(Nat,digit,0n,zero) :
          {link_sum(List.append(&2,Tiled.Link,links,more_links),List.append(&2,U32,target,more_shape),List.append(&2,Nat,digits,more_digits),cells,lane,position,groups_axis,cell) ==
            Nat.add(Nat.mul(Nat.add(Nat.mul(_,ShapeLaws.elements(target)),inner),outer),later) : Nat}
        sum : {link_sum(List.append(&2,Tiled.Link,links,more_links),List.append(&2,U32,target,more_shape),List.append(&2,Nat,digits,more_digits),cells,lane,position,groups_axis,cell) ==
            Nat.add(Nat.mul(Nat.add(Nat.mul(digit,ShapeLaws.elements(target)),inner),outer),later) : Nat}}
      KernelGemm.Both{sizes,KernelGemm.Both{sums,fits}}
    case None{}: Unit{}

# Two axes of one extent: the source cell axis weighs the cells after the
# paired source axis.
def linked_sum(found: Maybe<&2,List<&2,Tiled.Link>>,+index: Nat,+extent: U32,+target: List<&2,U32>,+other: U32,+source: List<&2,U32>,+digit: Nat,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,+width: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+more_digits: List<&2,Nat>,
  +same_extent: {extent == other : U32},+weight_is: {weight(cells,Tiled.source_axis(lane,True{},position,index)) == Nat.mul(ShapeLaws.elements(source),ShapeLaws.elements(rest)) : Nat},
  +held: PairedSum(found,target,ShapeLaws.elements(source),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)) ->
  PairedSum(Tiled.prepended(Tiled.Linked{index},found),Con{extent,target},ShapeLaws.elements(Con{other,source}),Con{digit,digits},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits):
  match found:
    case Some{+links}:
      KernelGemm.Both{+same,KernelGemm.Both{+sum,+fits}} = held
      +target_size = ShapeLaws.elements(target)
      +source_size = ShapeLaws.elements(source)
      +inner = ShapeLaws.digits_value(target,digits)
      +outer = ShapeLaws.elements(rest)
      +later = link_sum(more_links,more_shape,more_digits,cells,lane,position,groups_axis,cell)
      +links_sum = link_sum(List.append(&2,Tiled.Link,links,more_links),List.append(&2,U32,target,more_shape),List.append(&2,Nat,digits,more_digits),cells,lane,position,groups_axis,cell)
      +sizes = {%Equal.sym(U32,extent,other,same_extent) : {Nat.mul(U32.to_nat(_),target_size) == Nat.mul(U32.to_nat(other),source_size) : Nat}
        Equal.cong(Nat,Nat,value => Nat.mul(U32.to_nat(other),value),target_size,source_size,same) :
        {Nat.mul(U32.to_nat(extent),target_size) == Nat.mul(U32.to_nat(other),source_size) : Nat}}
      +grouped = Equal.sym(Nat,Nat.add(Nat.add(Nat.mul(digit,Nat.mul(target_size,outer)),Nat.mul(inner,outer)),later),
        Nat.add(Nat.mul(digit,Nat.mul(target_size,outer)),Nat.add(Nat.mul(inner,outer),later)),Addition.associative(Nat.mul(digit,Nat.mul(target_size,outer)),Nat.mul(inner,outer),later))
      +spread = {%Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(digit,target_size),inner),outer),Nat.add(Nat.mul(Nat.mul(digit,target_size),outer),Nat.mul(inner,outer)),
            RowMajor.distrib(Nat.mul(digit,target_size),inner,outer)) :
          {Nat.add(Nat.mul(digit,Nat.mul(target_size,outer)),Nat.add(Nat.mul(inner,outer),later)) == Nat.add(_,later) : Nat}
        %Equal.sym(Nat,Nat.mul(Nat.mul(digit,target_size),outer),Nat.mul(digit,Nat.mul(target_size,outer)),RowMajor.assoc(digit,target_size,outer)) :
          {Nat.add(Nat.mul(digit,Nat.mul(target_size,outer)),Nat.add(Nat.mul(inner,outer),later)) == Nat.add(Nat.add(_,Nat.mul(inner,outer)),later) : Nat}
        grouped :
        {Nat.add(Nat.mul(digit,Nat.mul(target_size,outer)),Nat.add(Nat.mul(inner,outer),later)) == Nat.add(Nat.mul(Nat.add(Nat.mul(digit,target_size),inner),outer),later) : Nat}}
      +sums = {%Equal.sym(Nat,weight(cells,Tiled.source_axis(lane,True{},position,index)),Nat.mul(source_size,outer),weight_is) :
          {Nat.add(Nat.mul(digit,_),links_sum) == Nat.add(Nat.mul(Nat.add(Nat.mul(digit,target_size),inner),outer),later) : Nat}
        %Equal.sym(Nat,links_sum,Nat.add(Nat.mul(inner,outer),later),sum) :
          {Nat.add(Nat.mul(digit,Nat.mul(source_size,outer)),_) == Nat.add(Nat.mul(Nat.add(Nat.mul(digit,target_size),inner),outer),later) : Nat}
        %same : {Nat.add(Nat.mul(digit,Nat.mul(_,outer)),Nat.add(Nat.mul(inner,outer),later)) == Nat.add(Nat.mul(Nat.add(Nat.mul(digit,target_size),inner),outer),later) : Nat}
        spread : {Nat.add(Nat.mul(digit,weight(cells,Tiled.source_axis(lane,True{},position,index))),links_sum) ==
          Nat.add(Nat.mul(Nat.add(Nat.mul(digit,target_size),inner),outer),later) : Nat}}
      KernelGemm.Both{sizes,KernelGemm.Both{sums,fits}}
    case None{}: Unit{}

# A pairing out of fuel ends without links.
def zero_target(unit: Bool,+target: List<&2,U32>,+source: List<&2,U32>,+index: Nat,+size: Nat,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,+width: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+more_digits: List<&2,Nat>) ->
  PairedSum(Tiled.paired(0n,Bool.pick(Tiled.Step,unit,Tiled.TargetUnit{},Tiled.Unpaired{}),target,source,index),target,size,digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits):
  match unit:
    case True{}: Unit{}
    case False{}: Unit{}

def zero_source(unit: Bool,+target: List<&2,U32>,+source: List<&2,U32>,+index: Nat,+size: Nat,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,+width: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+more_digits: List<&2,Nat>) ->
  PairedSum(Tiled.paired(0n,Bool.pick(Tiled.Step,unit,Tiled.SourceUnit{},Tiled.Unpaired{}),target,source,index),target,size,digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits):
  match unit:
    case True{}: Unit{}
    case False{}: Unit{}

def zero_both(unit: Bool,other_unit: Bool,same: Bool,+target: List<&2,U32>,+source: List<&2,U32>,+index: Nat,+size: Nat,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,+width: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+more_digits: List<&2,Nat>) ->
  PairedSum(Tiled.paired(0n,Bool.pick(Tiled.Step,unit,Tiled.TargetUnit{},Bool.pick(Tiled.Step,other_unit,Tiled.SourceUnit{},Bool.pick(Tiled.Step,same,Tiled.SameExtent{},Tiled.Unpaired{}))),
    target,source,index),target,size,digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits):
  match unit other_unit same:
    case True{} other_unit same: Unit{}
    case False{} True{} same: Unit{}
    case False{} False{} True{}: Unit{}
    case False{} False{} False{}: Unit{}

# A step at a target axis that may have one index.
def target_only(unit: Bool,+left: Nat,+extent: U32,+target: List<&2,U32>,+source: List<&2,U32>,+index: Nat,+digit: Nat,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,+width: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+more_digits: List<&2,Nat>,
  +unit_is: {U32.is_eq(extent,1) == unit : Bool},+here: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},
  +held: PairedSum(Tiled.paired(left,Tiled.step_of(target,source),target,source,index),target,ShapeLaws.elements(source),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)) ->
  PairedSum(Tiled.paired(1n+left,Bool.pick(Tiled.Step,unit,Tiled.TargetUnit{},Tiled.Unpaired{}),Con{extent,target},source,index),Con{extent,target},ShapeLaws.elements(source),
    Con{digit,digits},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits):
  match unit:
    case True{}:
      +unit = Arithmetic.equal_of_is_eq(extent,1,unit_is)
      +below = {%unit : {Nat.is_lt(digit,U32.to_nat(_)) == True{} : Bool} here : {Nat.is_lt(digit,1n) == True{} : Bool}}
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,target,0n),target,ShapeLaws.drop_zero(target)) :
        PairedSum(Tiled.prepended(Tiled.Alone{},Tiled.paired(left,Tiled.step_of(_,source),_,source,index)),Con{extent,target},ShapeLaws.elements(source),Con{digit,digits},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)
      alone_sum(Tiled.paired(left,Tiled.step_of(target,source),target,source,index),extent,target,ShapeLaws.elements(source),digit,digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,unit,digit_zero(digit,below),held)
    case False{}: Unit{}

# A step at a source axis that may have one index.
def source_only(unit: Bool,+left: Nat,+target: List<&2,U32>,+other: U32,+source: List<&2,U32>,+index: Nat,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,+width: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+more_digits: List<&2,Nat>,
  +unit_is: {U32.is_eq(other,1) == unit : Bool},
  +held: PairedSum(Tiled.paired(left,Tiled.step_of(target,source),target,source,Nat.add(index,1n)),target,ShapeLaws.elements(source),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)) ->
  PairedSum(Tiled.paired(1n+left,Bool.pick(Tiled.Step,unit,Tiled.SourceUnit{},Tiled.Unpaired{}),target,Con{other,source},index),target,ShapeLaws.elements(Con{other,source}),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits):
  match unit:
    case True{}:
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,source,0n),source,ShapeLaws.drop_zero(source)) :
        PairedSum(Tiled.paired(left,Tiled.step_of(target,_),target,_,Nat.add(index,1n)),target,ShapeLaws.elements(Con{other,source}),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)
      %unit_elements(other,source,Arithmetic.equal_of_is_eq(other,1,unit_is)) :
        PairedSum(Tiled.paired(left,Tiled.step_of(target,source),target,source,Nat.add(index,1n)),target,_,digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)
      held
    case False{}: Unit{}

# A step at two axes.
def paired_step(unit: Bool,other_unit: Bool,same: Bool,+left: Nat,+extent: U32,+target: List<&2,U32>,+other: U32,+source: List<&2,U32>,+index: Nat,+digit: Nat,
  +digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,+width: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+more_digits: List<&2,Nat>,+unit_is: {U32.is_eq(extent,1) == unit : Bool},+other_is: {U32.is_eq(other,1) == other_unit : Bool},+same_is: {U32.is_eq(extent,other) == same : Bool},
  +here: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},
  +weight_is: {weight(cells,Tiled.source_axis(lane,True{},position,index)) == Nat.mul(ShapeLaws.elements(source),ShapeLaws.elements(rest)) : Nat},
  +target_held: PairedSum(Tiled.paired(left,Tiled.step_of(target,Con{other,source}),target,Con{other,source},index),target,ShapeLaws.elements(Con{other,source}),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits),
  +source_held: PairedSum(Tiled.paired(left,Tiled.step_of(Con{extent,target},source),Con{extent,target},source,Nat.add(index,1n)),Con{extent,target},ShapeLaws.elements(source),
    Con{digit,digits},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits),
  +same_held: PairedSum(Tiled.paired(left,Tiled.step_of(target,source),target,source,Nat.add(index,1n)),target,ShapeLaws.elements(source),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)) ->
  PairedSum(Tiled.paired(1n+left,Bool.pick(Tiled.Step,unit,Tiled.TargetUnit{},Bool.pick(Tiled.Step,other_unit,Tiled.SourceUnit{},Bool.pick(Tiled.Step,same,Tiled.SameExtent{},Tiled.Unpaired{}))),
    Con{extent,target},Con{other,source},index),Con{extent,target},ShapeLaws.elements(Con{other,source}),Con{digit,digits},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits):
  match unit other_unit same:
    case True{} other_unit same:
      target_only(True{},left,extent,target,Con{other,source},index,digit,digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,unit_is,here,target_held)
    case False{} True{} same:
      source_only(True{},left,Con{extent,target},other,source,index,Con{digit,digits},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,other_is,source_held)
    case False{} False{} True{}:
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,target,0n),target,ShapeLaws.drop_zero(target)) :
        PairedSum(Tiled.prepended(Tiled.Linked{index},Tiled.paired(left,Tiled.step_of(_,List.drop(&2,U32,source,0n)),_,List.drop(&2,U32,source,0n),Nat.add(index,1n))),
          Con{extent,target},ShapeLaws.elements(Con{other,source}),Con{digit,digits},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,source,0n),source,ShapeLaws.drop_zero(source)) :
        PairedSum(Tiled.prepended(Tiled.Linked{index},Tiled.paired(left,Tiled.step_of(target,_),target,_,Nat.add(index,1n))),
          Con{extent,target},ShapeLaws.elements(Con{other,source}),Con{digit,digits},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)
      linked_sum(Tiled.paired(left,Tiled.step_of(target,source),target,source,Nat.add(index,1n)),index,extent,target,other,source,digit,digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,
        Arithmetic.equal_of_is_eq(extent,other,same_is),weight_is,same_held)
    case False{} False{} False{}: Unit{}

# The source cells after the next paired source axis.
def next_from(+side: Bool,+index: Nat,+other: U32,+source: List<&2,U32>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+rest: List<&2,U32>,
  +cells_from: {PlacingLaw.dropped(~U32,cell_place(lane,position,index),cells) == List.append(&2,U32,Con{other,source},rest) : List<&2,U32>},
  +side_ok: Side(side,lane,index,List.length(&2,U32,Con{other,source}))) ->
  {PlacingLaw.dropped(~U32,cell_place(lane,position,Nat.add(index,1n)),cells) == List.append(&2,U32,source,rest) : List<&2,U32>}:
  +place = cell_place(lane,position,index)
  %Equal.sym(Nat,cell_place(lane,position,Nat.add(index,1n)),1n+place,place_next(side,lane,position,index,List.length(&2,U32,source),side_ok)) :
    {PlacingLaw.dropped(~U32,_,cells) == List.append(&2,U32,source,rest) : List<&2,U32>}
  PlacingLaw.dropped_next(~U32,place,cells,other,List.append(&2,U32,source,rest),cells_from)

# The weight of a paired source axis is the cells after it.
def paired_weight(+index: Nat,+other: U32,+source: List<&2,U32>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+rest: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(cap)) == True{} : Bool},
  +cells_from: {PlacingLaw.dropped(~U32,cell_place(lane,position,index),cells) == List.append(&2,U32,Con{other,source},rest) : List<&2,U32>}) ->
  {weight(cells,Tiled.source_axis(lane,True{},position,index)) == Nat.mul(ShapeLaws.elements(source),ShapeLaws.elements(rest)) : Nat}:
  +place = cell_place(lane,position,index)
  +after = List.append(&2,U32,source,rest)
  +inside = PlacingFacts.dropped_inside(cells,place,other,after,cells_from)
  Equal.trans(Nat,weight(cells,U32.from_nat(place)),ShapeLaws.elements(after),Nat.mul(ShapeLaws.elements(source),ShapeLaws.elements(rest)),
    Equal.trans(Nat,weight(cells,U32.from_nat(place)),ShapeLaws.elements(PlacingLaw.dropped(~U32,1n+place,cells)),ShapeLaws.elements(after),
      stride_weight(cells,place,cap,cap_ok,room,positive,rank_room,inside),
      Equal.cong(List<&2,U32>,Nat,values => ShapeLaws.elements(values),PlacingLaw.dropped(~U32,1n+place,cells),after,PlacingLaw.dropped_next(~U32,place,cells,other,after,cells_from))),
    ShapeLaws.elements_appended(source,rest))

# Paired target and source axes: the links reach the number of the target
# digits in units of the source cells after them.
def paired_sum(target: List<&2,U32>,source: List<&2,U32>,fuel: Nat,+index: Nat,+digits: List<&2,Nat>,+side: Bool,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,+width: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+more_digits: List<&2,Nat>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(cap)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(cap)) == True{} : Bool},
  +cells_from: {PlacingLaw.dropped(~U32,cell_place(lane,position,index),cells) == List.append(&2,U32,source,rest) : List<&2,U32>},
  +side_ok: Side(side,lane,index,List.length(&2,U32,source)),+below: Coordinates.DigitsBelow(target,digits)) ->
  PairedSum(Tiled.paired(fuel,Tiled.step_of(target,source),target,source,index),target,ShapeLaws.elements(source),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits):
  match target source fuel:
    case Nil{} Nil{} 0n: ended_sum(digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,below)
    case Nil{} Nil{} 1n+ +left: ended_sum(digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,below)
    case Nil{} Con{+other,+more} 0n: zero_source(U32.is_eq(other,1),Nil{},Con{other,more},index,ShapeLaws.elements(Con{other,more}),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)
    case Nil{} Con{+other,+more} 1n+ +left:
      +held = paired_sum(Nil{},more,left,Nat.add(index,1n),digits,side,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,cap,cap_ok,room,positive,rank_room,next_from(side,index,other,more,cells,lane,position,rest,cells_from,side_ok),
        side_after(side,lane,index,List.length(&2,U32,more),side_ok),below)
      source_only(U32.is_eq(other,1),left,Nil{},other,more,index,digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,{==},held)
    case Con{+extent,+rest_target} Nil{} 0n: zero_target(U32.is_eq(extent,1),Con{extent,rest_target},Nil{},index,ShapeLaws.elements(Nil{}),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)
    case Con{+extent,+rest_target} Nil{} 1n+ +left:
      match digits:
        case Con{+digit,+rest_digits}:
          KernelGemm.Both{+here,+later} = below
          +held = paired_sum(rest_target,Nil{},left,index,rest_digits,side,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,cap,cap_ok,room,positive,rank_room,cells_from,side_ok,later)
          target_only(U32.is_eq(extent,1),left,extent,rest_target,Nil{},index,digit,rest_digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,{==},here,held)
        case Nil{}:
          Empty.absurd(PairedSum(Tiled.paired(1n+left,Tiled.step_of(Con{extent,rest_target},Nil{}),Con{extent,rest_target},Nil{},index),Con{extent,rest_target},
            ShapeLaws.elements(Nil{}),Nil{},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits),below)
    case Con{+extent,+rest_target} Con{+other,+more} 0n:
      zero_both(U32.is_eq(extent,1),U32.is_eq(other,1),U32.is_eq(extent,other),Con{extent,rest_target},Con{other,more},index,ShapeLaws.elements(Con{other,more}),digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits)
    case Con{+extent,+rest_target} Con{+other,+more} 1n+ +left:
      match digits:
        case Con{+digit,+rest_digits}:
          KernelGemm.Both{+here,+later} = below
          +next = next_from(side,index,other,more,cells,lane,position,rest,cells_from,side_ok)
          +next_side = side_after(side,lane,index,List.length(&2,U32,more),side_ok)
          +target_held = paired_sum(rest_target,Con{other,more},left,index,rest_digits,side,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,cap,cap_ok,room,positive,rank_room,cells_from,side_ok,later)
          +source_held = paired_sum(Con{extent,rest_target},more,left,Nat.add(index,1n),Con{digit,rest_digits},side,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,cap,cap_ok,room,positive,rank_room,next,next_side,KernelGemm.Both{here,later})
          +same_held = paired_sum(rest_target,more,left,Nat.add(index,1n),rest_digits,side,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,cap,cap_ok,room,positive,rank_room,next,next_side,later)
          paired_step(U32.is_eq(extent,1),U32.is_eq(other,1),U32.is_eq(extent,other),left,extent,rest_target,other,more,index,digit,rest_digits,cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits,{==},{==},{==},here,
            paired_weight(index,other,more,cells,lane,position,rest,cap,cap_ok,room,positive,rank_room,cells_from),target_held,source_held,same_held)
        case Nil{}:
          Empty.absurd(PairedSum(Tiled.paired(1n+left,Tiled.step_of(Con{extent,rest_target},Con{other,more}),Con{extent,rest_target},Con{other,more},index),
            Con{extent,rest_target},ShapeLaws.elements(Con{other,more}),Nil{},cells,lane,position,groups_axis,cell,width,rest,more_links,more_shape,more_digits),below)

# ---- The source cell axes paired links read ----

# Every paired link reads a source cell axis of the extent of its target
# axis, and every anchor has the extent of the anchors.
def LinksIn(links: List<&2,Tiled.Link>,shape: List<&2,U32>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+anchors: U32) -> Data:
  match links shape:
    case Con{Tiled.Lane{},links} Con{extent,shape}: LinksIn(links,shape,cells,lane,position,anchors)
    case Con{Tiled.Linked{+axis},links} Con{+extent,shape}:
      KernelGemm.Both<{Grid.lanes_at(cells,cell_place(lane,position,axis)) == extent : U32},
        KernelGemm.Both<{Nat.is_lt(cell_place(lane,position,axis),List.length(&2,U32,cells)) == True{} : Bool},LinksIn(links,shape,cells,lane,position,anchors)>>
    case Con{Tiled.Anchor{},links} Con{+extent,shape}: KernelGemm.Both<{extent == anchors : U32},LinksIn(links,shape,cells,lane,position,anchors)>
    case Con{Tiled.Alone{},links} Con{extent,shape}: LinksIn(links,shape,cells,lane,position,anchors)
    case links shape: Unit

# Paired links before more links read their source cell axes and every
# target axis but the new lanes.
def PairedIn(found: Maybe<&2,List<&2,Tiled.Link>>,+target: List<&2,U32>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+anchors: U32,
  +more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>) -> Data:
  match found:
    case Some{+links}:
      KernelGemm.Both<LinksIn(List.append(&2,Tiled.Link,links,more_links),List.append(&2,U32,target,more_shape),cells,lane,position,anchors),
        {link_extents(List.append(&2,Tiled.Link,links,more_links),List.append(&2,U32,target,more_shape)) ==
          List.append(&2,U32,target,link_extents(more_links,more_shape)) : List<&2,U32>}>
    case None{}: Unit

def ended_in(+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+anchors: U32,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,
  +more_in: LinksIn(more_links,more_shape,cells,lane,position,anchors)) -> PairedIn(Some{[]},Nil{},cells,lane,position,anchors,more_links,more_shape):
  KernelGemm.Both{more_in,{==}}

# A target axis of one index reads nothing.
def alone_in(found: Maybe<&2,List<&2,Tiled.Link>>,+extent: U32,+target: List<&2,U32>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+anchors: U32,
  +more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+held: PairedIn(found,target,cells,lane,position,anchors,more_links,more_shape)) ->
  PairedIn(Tiled.prepended(Tiled.Alone{},found),Con{extent,target},cells,lane,position,anchors,more_links,more_shape):
  match found:
    case Some{+links}:
      KernelGemm.Both{+links_in,+extents} = held
      KernelGemm.Both{links_in,Equal.cong(List<&2,U32>,List<&2,U32>,values => extent <> values,
        link_extents(List.append(&2,Tiled.Link,links,more_links),List.append(&2,U32,target,more_shape)),List.append(&2,U32,target,link_extents(more_links,more_shape)),extents)}
    case None{}: Unit{}

# Two axes of one extent: the target axis reads the source cell axis of the
# paired source axis, which has its extent.
def linked_in(found: Maybe<&2,List<&2,Tiled.Link>>,+index: Nat,+extent: U32,+target: List<&2,U32>,+other: U32,+source: List<&2,U32>,+cells: List<&2,U32>,+lane: Nat,
  +position: Grid.Position,+anchors: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+same_extent: {extent == other : U32},
  +cells_from: {PlacingLaw.dropped(~U32,cell_place(lane,position,index),cells) == List.append(&2,U32,Con{other,source},rest) : List<&2,U32>},
  +held: PairedIn(found,target,cells,lane,position,anchors,more_links,more_shape)) ->
  PairedIn(Tiled.prepended(Tiled.Linked{index},found),Con{extent,target},cells,lane,position,anchors,more_links,more_shape):
  match found:
    case Some{+links}:
      KernelGemm.Both{+links_in,+extents} = held
      +place = cell_place(lane,position,index)
      +after = List.append(&2,U32,source,rest)
      KernelGemm.Both{KernelGemm.Both{Equal.trans(U32,Grid.lanes_at(cells,place),other,extent,PlacingFacts.lanes_dropped(cells,place,other,after,cells_from),
            Equal.sym(U32,extent,other,same_extent)),KernelGemm.Both{PlacingFacts.dropped_inside(cells,place,other,after,cells_from),links_in}},
        Equal.cong(List<&2,U32>,List<&2,U32>,values => extent <> values,
          link_extents(List.append(&2,Tiled.Link,links,more_links),List.append(&2,U32,target,more_shape)),List.append(&2,U32,target,link_extents(more_links,more_shape)),extents)}
    case None{}: Unit{}

# A pairing out of fuel ends without links.
def zero_target_in(unit: Bool,+target: List<&2,U32>,+source: List<&2,U32>,+index: Nat,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+anchors: U32,
  +more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>) ->
  PairedIn(Tiled.paired(0n,Bool.pick(Tiled.Step,unit,Tiled.TargetUnit{},Tiled.Unpaired{}),target,source,index),target,cells,lane,position,anchors,more_links,more_shape):
  match unit:
    case True{}: Unit{}
    case False{}: Unit{}

def zero_source_in(unit: Bool,+target: List<&2,U32>,+source: List<&2,U32>,+index: Nat,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+anchors: U32,
  +more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>) ->
  PairedIn(Tiled.paired(0n,Bool.pick(Tiled.Step,unit,Tiled.SourceUnit{},Tiled.Unpaired{}),target,source,index),target,cells,lane,position,anchors,more_links,more_shape):
  match unit:
    case True{}: Unit{}
    case False{}: Unit{}

def zero_both_in(unit: Bool,other_unit: Bool,same: Bool,+target: List<&2,U32>,+source: List<&2,U32>,+index: Nat,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,
  +anchors: U32,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>) ->
  PairedIn(Tiled.paired(0n,Bool.pick(Tiled.Step,unit,Tiled.TargetUnit{},Bool.pick(Tiled.Step,other_unit,Tiled.SourceUnit{},Bool.pick(Tiled.Step,same,Tiled.SameExtent{},Tiled.Unpaired{}))),
    target,source,index),target,cells,lane,position,anchors,more_links,more_shape):
  match unit other_unit same:
    case True{} other_unit same: Unit{}
    case False{} True{} same: Unit{}
    case False{} False{} True{}: Unit{}
    case False{} False{} False{}: Unit{}

# A step at a target axis that may have one index.
def target_only_in(unit: Bool,+left: Nat,+extent: U32,+target: List<&2,U32>,+source: List<&2,U32>,+index: Nat,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,
  +anchors: U32,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,
  +held: PairedIn(Tiled.paired(left,Tiled.step_of(target,source),target,source,index),target,cells,lane,position,anchors,more_links,more_shape)) ->
  PairedIn(Tiled.paired(1n+left,Bool.pick(Tiled.Step,unit,Tiled.TargetUnit{},Tiled.Unpaired{}),Con{extent,target},source,index),Con{extent,target},cells,lane,position,anchors,
    more_links,more_shape):
  match unit:
    case True{}:
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,target,0n),target,ShapeLaws.drop_zero(target)) :
        PairedIn(Tiled.prepended(Tiled.Alone{},Tiled.paired(left,Tiled.step_of(_,source),_,source,index)),Con{extent,target},cells,lane,position,anchors,more_links,more_shape)
      alone_in(Tiled.paired(left,Tiled.step_of(target,source),target,source,index),extent,target,cells,lane,position,anchors,more_links,more_shape,held)
    case False{}: Unit{}

# A step at a source axis that may have one index.
def source_only_in(unit: Bool,+left: Nat,+target: List<&2,U32>,+other: U32,+source: List<&2,U32>,+index: Nat,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,
  +anchors: U32,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,
  +held: PairedIn(Tiled.paired(left,Tiled.step_of(target,source),target,source,Nat.add(index,1n)),target,cells,lane,position,anchors,more_links,more_shape)) ->
  PairedIn(Tiled.paired(1n+left,Bool.pick(Tiled.Step,unit,Tiled.SourceUnit{},Tiled.Unpaired{}),target,Con{other,source},index),target,cells,lane,position,anchors,more_links,more_shape):
  match unit:
    case True{}:
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,source,0n),source,ShapeLaws.drop_zero(source)) :
        PairedIn(Tiled.paired(left,Tiled.step_of(target,_),target,_,Nat.add(index,1n)),target,cells,lane,position,anchors,more_links,more_shape)
      held
    case False{}: Unit{}

# A step at two axes.
def paired_step_in(unit: Bool,other_unit: Bool,same: Bool,+left: Nat,+extent: U32,+target: List<&2,U32>,+other: U32,+source: List<&2,U32>,+index: Nat,+cells: List<&2,U32>,
  +lane: Nat,+position: Grid.Position,+anchors: U32,+rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,+same_is: {U32.is_eq(extent,other) == same : Bool},
  +cells_from: {PlacingLaw.dropped(~U32,cell_place(lane,position,index),cells) == List.append(&2,U32,Con{other,source},rest) : List<&2,U32>},
  +target_held: PairedIn(Tiled.paired(left,Tiled.step_of(target,Con{other,source}),target,Con{other,source},index),target,cells,lane,position,anchors,more_links,more_shape),
  +source_held: PairedIn(Tiled.paired(left,Tiled.step_of(Con{extent,target},source),Con{extent,target},source,Nat.add(index,1n)),Con{extent,target},cells,lane,position,anchors,
    more_links,more_shape),
  +same_held: PairedIn(Tiled.paired(left,Tiled.step_of(target,source),target,source,Nat.add(index,1n)),target,cells,lane,position,anchors,more_links,more_shape)) ->
  PairedIn(Tiled.paired(1n+left,Bool.pick(Tiled.Step,unit,Tiled.TargetUnit{},Bool.pick(Tiled.Step,other_unit,Tiled.SourceUnit{},Bool.pick(Tiled.Step,same,Tiled.SameExtent{},Tiled.Unpaired{}))),
    Con{extent,target},Con{other,source},index),Con{extent,target},cells,lane,position,anchors,more_links,more_shape):
  match unit other_unit same:
    case True{} other_unit same:
      target_only_in(True{},left,extent,target,Con{other,source},index,cells,lane,position,anchors,more_links,more_shape,target_held)
    case False{} True{} same:
      source_only_in(True{},left,Con{extent,target},other,source,index,cells,lane,position,anchors,more_links,more_shape,source_held)
    case False{} False{} True{}:
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,target,0n),target,ShapeLaws.drop_zero(target)) :
        PairedIn(Tiled.prepended(Tiled.Linked{index},Tiled.paired(left,Tiled.step_of(_,List.drop(&2,U32,source,0n)),_,List.drop(&2,U32,source,0n),Nat.add(index,1n))),
          Con{extent,target},cells,lane,position,anchors,more_links,more_shape)
      %Equal.sym(List<&2,U32>,List.drop(&2,U32,source,0n),source,ShapeLaws.drop_zero(source)) :
        PairedIn(Tiled.prepended(Tiled.Linked{index},Tiled.paired(left,Tiled.step_of(target,_),target,_,Nat.add(index,1n))),Con{extent,target},cells,lane,position,anchors,
          more_links,more_shape)
      linked_in(Tiled.paired(left,Tiled.step_of(target,source),target,source,Nat.add(index,1n)),index,extent,target,other,source,cells,lane,position,anchors,rest,more_links,
        more_shape,Arithmetic.equal_of_is_eq(extent,other,same_is),cells_from,same_held)
    case False{} False{} False{}: Unit{}

# Paired target and source axes before more links: the paired links read
# their source cell axes and every target axis is read.
def paired_in(target: List<&2,U32>,source: List<&2,U32>,fuel: Nat,+index: Nat,+side: Bool,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+anchors: U32,
  +rest: List<&2,U32>,+more_links: List<&2,Tiled.Link>,+more_shape: List<&2,U32>,
  +cells_from: {PlacingLaw.dropped(~U32,cell_place(lane,position,index),cells) == List.append(&2,U32,source,rest) : List<&2,U32>},
  +side_ok: Side(side,lane,index,List.length(&2,U32,source)),+more_in: LinksIn(more_links,more_shape,cells,lane,position,anchors)) ->
  PairedIn(Tiled.paired(fuel,Tiled.step_of(target,source),target,source,index),target,cells,lane,position,anchors,more_links,more_shape):
  match target source fuel:
    case Nil{} Nil{} 0n: ended_in(cells,lane,position,anchors,more_links,more_shape,more_in)
    case Nil{} Nil{} 1n+ +left: ended_in(cells,lane,position,anchors,more_links,more_shape,more_in)
    case Nil{} Con{+other,+more} 0n: zero_source_in(U32.is_eq(other,1),Nil{},Con{other,more},index,cells,lane,position,anchors,more_links,more_shape)
    case Nil{} Con{+other,+more} 1n+ +left:
      +held = paired_in(Nil{},more,left,Nat.add(index,1n),side,cells,lane,position,anchors,rest,more_links,more_shape,next_from(side,index,other,more,cells,lane,position,rest,cells_from,side_ok),
        side_after(side,lane,index,List.length(&2,U32,more),side_ok),more_in)
      source_only_in(U32.is_eq(other,1),left,Nil{},other,more,index,cells,lane,position,anchors,more_links,more_shape,held)
    case Con{+extent,+rest_target} Nil{} 0n: zero_target_in(U32.is_eq(extent,1),Con{extent,rest_target},Nil{},index,cells,lane,position,anchors,more_links,more_shape)
    case Con{+extent,+rest_target} Nil{} 1n+ +left:
      +held = paired_in(rest_target,Nil{},left,index,side,cells,lane,position,anchors,rest,more_links,more_shape,cells_from,side_ok,more_in)
      target_only_in(U32.is_eq(extent,1),left,extent,rest_target,Nil{},index,cells,lane,position,anchors,more_links,more_shape,held)
    case Con{+extent,+rest_target} Con{+other,+more} 0n:
      zero_both_in(U32.is_eq(extent,1),U32.is_eq(other,1),U32.is_eq(extent,other),Con{extent,rest_target},Con{other,more},index,cells,lane,position,anchors,more_links,more_shape)
    case Con{+extent,+rest_target} Con{+other,+more} 1n+ +left:
      +next = next_from(side,index,other,more,cells,lane,position,rest,cells_from,side_ok)
      +next_side = side_after(side,lane,index,List.length(&2,U32,more),side_ok)
      +target_held = paired_in(rest_target,Con{other,more},left,index,side,cells,lane,position,anchors,rest,more_links,more_shape,cells_from,side_ok,more_in)
      +source_held = paired_in(Con{extent,rest_target},more,left,Nat.add(index,1n),side,cells,lane,position,anchors,rest,more_links,more_shape,next,next_side,more_in)
      +same_held = paired_in(rest_target,more,left,Nat.add(index,1n),side,cells,lane,position,anchors,rest,more_links,more_shape,next,next_side,more_in)
      paired_step_in(U32.is_eq(extent,1),U32.is_eq(other,1),U32.is_eq(extent,other),left,extent,rest_target,other,more,index,cells,lane,position,anchors,rest,more_links,more_shape,
        {==},cells_from,target_held,source_held,same_held)

# The paired links after the lanes, before no more links.
def post_in(+links: List<&2,Tiled.Link>,+post: List<&2,U32>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+anchors: U32,
  +held: PairedIn(Some{links},post,cells,lane,position,anchors,[],[])) -> LinksIn(links,post,cells,lane,position,anchors):
  KernelGemm.Both{+links_in,+extents} = held
  +more_links = List.append(&2,Tiled.Link,links,[])
  {%ListReadback.append_empty(~Tiled.Link,links) : LinksIn(_,post,cells,lane,position,anchors)
    %ListReadback.append_empty(~U32,post) : LinksIn(more_links,_,cells,lane,position,anchors)
    links_in : LinksIn(links,post,cells,lane,position,anchors)}

def post_extents(+links: List<&2,Tiled.Link>,+post: List<&2,U32>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+anchors: U32,
  +held: PairedIn(Some{links},post,cells,lane,position,anchors,[],[])) -> {link_extents(links,post) == post : List<&2,U32>}:
  KernelGemm.Both{+links_in,+extents} = held
  +more_links = List.append(&2,Tiled.Link,links,[])
  +more_post = List.append(&2,U32,post,[])
  {%ListReadback.append_empty(~Tiled.Link,links) : {link_extents(_,post) == post : List<&2,U32>}
    %ListReadback.append_empty(~U32,post) : {link_extents(more_links,_) == post : List<&2,U32>}
    Equal.trans(List<&2,U32>,link_extents(more_links,more_post),more_post,post,extents,ListReadback.append_empty(~U32,post)) :
    {link_extents(links,post) == post : List<&2,U32>}}

# The links a pairing finds, none when it finds none.
def links_of(found: Maybe<&2,List<&2,Tiled.Link>>) -> List<&2,Tiled.Link>:
  match found:
    case Some{+links}: links
    case None{}: []

def post_links_in(found: Maybe<&2,List<&2,Tiled.Link>>,+post: List<&2,U32>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+anchors: U32,
  +held: PairedIn(found,post,cells,lane,position,anchors,[],[])) -> LinksIn(links_of(found),post,cells,lane,position,anchors):
  match found:
    case Some{+links}: post_in(links,post,cells,lane,position,anchors,held)
    case None{}: Unit{}

# What the links of a split read: the source cell axes of paired links, and
# every target axis but the new lanes.
def SplitIn(found: Maybe<&2,List<&2,Tiled.Link>>,+shape: List<&2,U32>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+anchors: U32,+extents: List<&2,U32>) -> Data:
  match found:
    case Some{+links}: KernelGemm.Both<LinksIn(links,shape,cells,lane,position,anchors),{link_extents(links,shape) == extents : List<&2,U32>}>
    case None{}: Unit

def joined_in(+pre_links: List<&2,Tiled.Link>,+post_links: List<&2,Tiled.Link>,+pre: List<&2,U32>,+post: List<&2,U32>,+anchors: U32,+width: U32,+cells: List<&2,U32>,
  +lane: Nat,+position: Grid.Position,
  +pre_held: PairedIn(Some{pre_links},pre,cells,lane,position,anchors,Tiled.Anchor{} <> Tiled.Lane{} <> post_links,anchors <> width <> post),
  +extents_after: {link_extents(post_links,post) == post : List<&2,U32>}) ->
  SplitIn(Some{List.append(&2,Tiled.Link,pre_links,Tiled.Anchor{} <> Tiled.Lane{} <> post_links)},List.append(&2,U32,pre,anchors <> width <> post),cells,lane,position,anchors,
    List.append(&2,U32,pre,anchors <> post)):
  KernelGemm.Both{+links_in,+extents} = pre_held
  +links = List.append(&2,Tiled.Link,pre_links,Tiled.Anchor{} <> Tiled.Lane{} <> post_links)
  +shape = List.append(&2,U32,pre,anchors <> width <> post)
  KernelGemm.Both{links_in,Equal.trans(List<&2,U32>,link_extents(links,shape),List.append(&2,U32,pre,anchors <> link_extents(post_links,post)),List.append(&2,U32,pre,anchors <> post),
    extents,Equal.cong(List<&2,U32>,List<&2,U32>,values => List.append(&2,U32,pre,anchors <> values),link_extents(post_links,post),post,extents_after))}

def joined_split(found_pre: Maybe<&2,List<&2,Tiled.Link>>,found_post: Maybe<&2,List<&2,Tiled.Link>>,+pre: List<&2,U32>,+post: List<&2,U32>,+anchors: U32,+width: U32,
  +cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,
  +pre_held: PairedIn(found_pre,pre,cells,lane,position,anchors,Tiled.Anchor{} <> Tiled.Lane{} <> links_of(found_post),anchors <> width <> post),
  +post_held: PairedIn(found_post,post,cells,lane,position,anchors,[],[])) ->
  SplitIn(Tiled.joined_links(found_pre,found_post),List.append(&2,U32,pre,anchors <> width <> post),cells,lane,position,anchors,List.append(&2,U32,pre,anchors <> post)):
  match found_pre found_post:
    case Some{+pre_links} Some{+post_links}:
      joined_in(pre_links,post_links,pre,post,anchors,width,cells,lane,position,pre_held,post_extents(post_links,post,cells,lane,position,anchors,post_held))
    case Some{+pre_links} None{}: Unit{}
    case None{} found_post: Unit{}

# ---- The links read inside the classes of a target cell ----

# Readings after the links that read nothing, one for each extent and digit.
def UnreadTail(readings: List<&2,Tasks.Reading>,extents: List<&2,U32>,digits: List<&2,Nat>) -> Data:
  match readings extents digits:
    case Con{Tasks.Reading{extent,block,+source},+readings} Con{other,+extents} Con{digit,+digits}:
      KernelGemm.Both<{source == Tasks.Unread{} : Tasks.Source},UnreadTail(readings,extents,digits)>
    case Nil{} Nil{} Nil{}: Unit
    case readings extents digits: Empty

# Readings that read nothing place any digits inside any classes.
def unread_placed(readings: List<&2,Tasks.Reading>,extents: List<&2,U32>,digits: List<&2,Nat>,classes: List<&2,Tasks.Class>,+grid: Grid.Grid,+axis: U32,+cap: U32,
  +tail: UnreadTail(readings,extents,digits),+inside: Targets.ClassesIn(classes,grid,extents,digits,axis)) -> TaskReads.PlacedRow(readings,classes,digits,cap):
  match readings extents digits classes:
    case Con{Tasks.Reading{+extent,+block,+source},+readings} Con{+other,+extents} Con{+digit,+digits} Con{+class,+classes}:
      KernelGemm.Both{+unread,+later} = tail
      Held.Both{+here,+rest} = inside
      +placed = {%Equal.sym(Tasks.Source,source,Tasks.Unread{},unread) : TaskReads.PlacedIn(Tasks.Reading{extent,block,_},class,digit,cap)
        Unit{} : TaskReads.PlacedIn(Tasks.Reading{extent,block,source},class,digit,cap)}
      Held.Both{placed,unread_placed(readings,extents,digits,classes,grid,(axis + 1 : U32),cap,later,rest)}
    case Con{+reading,+readings} Con{+other,+extents} Con{+digit,+digits} Nil{}:
      Empty.absurd(TaskReads.PlacedRow(Con{reading,readings},Nil{},Con{digit,digits},cap),inside)
    case Nil{} Nil{} Nil{} Nil{}: Unit{}
    case Nil{} Nil{} Nil{} Con{+class,+classes}:
      Empty.absurd(TaskReads.PlacedRow(Nil{},Con{class,classes},Nil{},cap),inside)
    case Nil{} Nil{} Con{+digit,+digits} classes: Empty.absurd(TaskReads.PlacedRow(Nil{},classes,Con{digit,digits},cap),tail)
    case Nil{} Con{+other,+extents} digits classes: Empty.absurd(TaskReads.PlacedRow(Nil{},classes,digits,cap),tail)
    case Con{+reading,+readings} Nil{} digits classes: Empty.absurd(TaskReads.PlacedRow(Con{reading,readings},classes,digits,cap),tail)
    case Con{+reading,+readings} Con{+other,+extents} Nil{} classes: Empty.absurd(TaskReads.PlacedRow(Con{reading,readings},classes,Nil{},cap),tail)

# The links read their values inside the classes of the target digits, the
# readings after them their digits.
def links_placed(links: List<&2,Tiled.Link>,shape: List<&2,U32>,digits: List<&2,Nat>,classes: List<&2,Tasks.Class>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,
  +lane: Nat,+position: Grid.Position,+width: U32,+source_groups: U32,+groups_axis: U32,+index: U32,+cell: Nat,+tail: List<&2,Tasks.Reading>,+tail_extents: List<&2,U32>,
  +tail_digits: List<&2,Nat>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +rooms: Components.ReadingsRoom(List.append(&2,Tasks.Reading,Tiled.link_readings(links,shape,Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},index),tail),
    cap),
  +below: Coordinates.DigitsBelow(link_extents(links,shape),digits),+fits: {anchor_fits(links,shape,digits,width,cell) == True{} : Bool},
  +cell_below: {Nat.is_lt(cell,U32.to_nat(source_groups)) == True{} : Bool},+tail_ok: UnreadTail(tail,tail_extents,tail_digits),
  +inside: Targets.ClassesIn(classes,target_grid,List.append(&2,U32,link_extents(links,shape),tail_extents),List.append(&2,Nat,digits,tail_digits),index)) ->
  TaskReads.PlacedRow(List.append(&2,Tasks.Reading,Tiled.link_readings(links,shape,Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},index),tail),
    classes,List.append(&2,Nat,link_values(links,shape,digits,cell),tail_digits),cap):
  match links shape:
    case Con{Tiled.Lane{},+links} Con{+extent,+shape}:
      links_placed(links,shape,digits,classes,source_grid,target_grid,lane,position,width,source_groups,groups_axis,index,cell,tail,tail_extents,tail_digits,cap,cap_ok,
        rooms,below,fits,cell_below,tail_ok,inside)
    case Con{Tiled.Linked{+axis},+links} Con{+extent,+shape}:
      match digits classes:
        case Con{+digit,+digits} Con{Tasks.Class{+first,+count,+source_first,+source_count},+classes}:
          Held.Both{+room,+rooms_rest} = rooms
          KernelGemm.Both{+here,+later} = below
          Held.Both{+class_in,+inside_rest} = inside
          +read_axis = Tiled.source_axis(lane,True{},position,axis)
          Held.Both{TaskReads.shared_placed(extent,Grid.block(target_grid,index,extent),read_axis,Grid.block(source_grid,read_axis,extent),extent,first,count,source_first,source_count,
              digit,cap,cap_ok,room,class_in,here),
            links_placed(links,shape,digits,classes,source_grid,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32),cell,tail,tail_extents,tail_digits,cap,
              cap_ok,rooms_rest,later,fits,cell_below,tail_ok,inside_rest)}
        case Nil{} classes:
          Empty.absurd(TaskReads.PlacedRow(List.append(&2,Tasks.Reading,Tiled.link_readings(Con{Tiled.Linked{axis},links},Con{extent,shape},
              Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},index),tail),classes,
            List.append(&2,Nat,link_values(Con{Tiled.Linked{axis},links},Con{extent,shape},Nil{},cell),tail_digits),cap),below)
        case Con{+digit,+digits} Nil{}:
          Empty.absurd(TaskReads.PlacedRow(List.append(&2,Tasks.Reading,Tiled.link_readings(Con{Tiled.Linked{axis},links},Con{extent,shape},
              Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},index),tail),Nil{},
            List.append(&2,Nat,link_values(Con{Tiled.Linked{axis},links},Con{extent,shape},Con{digit,digits},cell),tail_digits),cap),inside)
    case Con{Tiled.Anchor{},+links} Con{+extent,+shape}:
      match digits classes:
        case Con{+digit,+digits} Con{Tasks.Class{+first,+count,+source_first,+source_count},+classes}:
          Held.Both{+room,+rooms_rest} = rooms
          KernelGemm.Both{+here,+later} = below
          Held.Both{+class_in,+inside_rest} = inside
          +first_lane = Nat.add(U32.to_nat(0),Nat.mul(U32.to_nat(width),digit))
          +eights = U32.to_nat(U32.max(8,1))
          +above_ok = Nat.is_le(Nat.div(first_lane,eights),Nat.add(U32.to_nat(0),cell))
          +under_ok = Nat.is_le(Nat.add(U32.to_nat(0),cell),Nat.div(Nat.sub(Nat.add(first_lane,U32.to_nat(width)),1n),eights))
          +rest_fits = anchor_fits(links,shape,digits,width,cell)
          +inner = Logic.right(above_ok,under_ok && rest_fits,fits)
          Held.Both{TaskReads.read_placed(extent,Grid.block(target_grid,index,extent),groups_axis,Grid.block(source_grid,groups_axis,source_groups),width,0,width,8,0,source_groups,
              first,count,source_first,source_count,digit,cell,cap,cap_ok,room,class_in,Logic.left(above_ok,under_ok && rest_fits,fits),Logic.left(under_ok,rest_fits,inner),
              cell_below),
            links_placed(links,shape,digits,classes,source_grid,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32),cell,tail,tail_extents,tail_digits,cap,
              cap_ok,rooms_rest,later,Logic.right(under_ok,rest_fits,inner),cell_below,tail_ok,inside_rest)}
        case Nil{} classes:
          Empty.absurd(TaskReads.PlacedRow(List.append(&2,Tasks.Reading,Tiled.link_readings(Con{Tiled.Anchor{},links},Con{extent,shape},
              Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},index),tail),classes,
            List.append(&2,Nat,link_values(Con{Tiled.Anchor{},links},Con{extent,shape},Nil{},cell),tail_digits),cap),below)
        case Con{+digit,+digits} Nil{}:
          Empty.absurd(TaskReads.PlacedRow(List.append(&2,Tasks.Reading,Tiled.link_readings(Con{Tiled.Anchor{},links},Con{extent,shape},
              Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},index),tail),Nil{},
            List.append(&2,Nat,link_values(Con{Tiled.Anchor{},links},Con{extent,shape},Con{digit,digits},cell),tail_digits),cap),inside)
    case Con{Tiled.Alone{},+links} Con{+extent,+shape}:
      match digits classes:
        case Con{+digit,+digits} Con{Tasks.Class{+first,+count,+source_first,+source_count},+classes}:
          Held.Both{+room,+rooms_rest} = rooms
          KernelGemm.Both{+here,+later} = below
          Held.Both{+class_in,+inside_rest} = inside
          Held.Both{Unit{},
            links_placed(links,shape,digits,classes,source_grid,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32),cell,tail,tail_extents,tail_digits,cap,
              cap_ok,rooms_rest,later,fits,cell_below,tail_ok,inside_rest)}
        case Nil{} classes:
          Empty.absurd(TaskReads.PlacedRow(List.append(&2,Tasks.Reading,Tiled.link_readings(Con{Tiled.Alone{},links},Con{extent,shape},
              Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},index),tail),classes,
            List.append(&2,Nat,link_values(Con{Tiled.Alone{},links},Con{extent,shape},Nil{},cell),tail_digits),cap),below)
        case Con{+digit,+digits} Nil{}:
          Empty.absurd(TaskReads.PlacedRow(List.append(&2,Tasks.Reading,Tiled.link_readings(Con{Tiled.Alone{},links},Con{extent,shape},
              Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},index),tail),Nil{},
            List.append(&2,Nat,link_values(Con{Tiled.Alone{},links},Con{extent,shape},Con{digit,digits},cell),tail_digits),cap),inside)
    case Nil{} +shape:
      match digits:
        case Nil{}: unread_placed(tail,tail_extents,tail_digits,classes,target_grid,index,cap,tail_ok,inside)
        case Con{+digit,+digits}:
          Empty.absurd(TaskReads.PlacedRow(tail,classes,List.append(&2,Nat,link_values(Nil{},shape,Con{digit,digits},cell),tail_digits),cap),below)
    case Con{+link,+links} Nil{}:
      match digits:
        case Nil{}: unread_placed(tail,tail_extents,tail_digits,classes,target_grid,index,cap,tail_ok,inside)
        case Con{+digit,+digits}:
          Empty.absurd(TaskReads.PlacedRow(tail,classes,List.append(&2,Nat,link_values(Con{link,links},Nil{},Con{digit,digits},cell),tail_digits),cap),below)

# ---- The links of a split ----

# Paired axes before no more links.
def post_facts(+links: List<&2,Tiled.Link>,+post: List<&2,U32>,+size: Nat,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,
  +cell: Nat,+width: U32,+rest: List<&2,U32>,+held: PairedSum(Some{links},post,size,digits,cells,lane,position,groups_axis,cell,width,rest,[],[],[])) ->
  KernelGemm.Both<{link_sum(links,post,digits,cells,lane,position,groups_axis,cell) == Nat.mul(ShapeLaws.digits_value(post,digits),ShapeLaws.elements(rest)) : Nat},
    {anchor_fits(links,post,digits,width,cell) == True{} : Bool}>:
  KernelGemm.Both{+same,KernelGemm.Both{+sum,+fits}} = held
  +value = Nat.mul(ShapeLaws.digits_value(post,digits),ShapeLaws.elements(rest))
  +more_links = List.append(&2,Tiled.Link,links,[])
  +more_post = List.append(&2,U32,post,[])
  +more_digits = List.append(&2,Nat,digits,[])
  +sum_clean = {%ListReadback.append_empty(~Tiled.Link,links) : {link_sum(_,post,digits,cells,lane,position,groups_axis,cell) == value : Nat}
    %ListReadback.append_empty(~U32,post) : {link_sum(more_links,_,digits,cells,lane,position,groups_axis,cell) == value : Nat}
    %ListReadback.append_empty(~Nat,digits) : {link_sum(more_links,more_post,_,cells,lane,position,groups_axis,cell) == value : Nat}
    Equal.trans(Nat,link_sum(more_links,more_post,more_digits,cells,lane,position,groups_axis,cell),Nat.add(value,0n),value,sum,Addition.zero_right(value)) :
    {link_sum(links,post,digits,cells,lane,position,groups_axis,cell) == value : Nat}}
  +fits_clean = {%ListReadback.append_empty(~Tiled.Link,links) : {anchor_fits(_,post,digits,width,cell) == True{} : Bool}
    %ListReadback.append_empty(~U32,post) : {anchor_fits(more_links,_,digits,width,cell) == True{} : Bool}
    %ListReadback.append_empty(~Nat,digits) : {anchor_fits(more_links,more_post,_,width,cell) == True{} : Bool}
    fits : {anchor_fits(links,post,digits,width,cell) == True{} : Bool}}
  KernelGemm.Both{sum_clean,fits_clean}

# Whether the anchor reads the source group cell at an anchor digit.
def anchor_holds(+width: U32,+anchor: Nat,+cell: Nat) -> Bool:
  +first = Nat.add(U32.to_nat(0),Nat.mul(U32.to_nat(width),anchor))
  +eights = U32.to_nat(U32.max(8,1))
  Nat.is_le(Nat.div(first,eights),Nat.add(U32.to_nat(0),cell)) && (Nat.is_le(Nat.add(U32.to_nat(0),cell),Nat.div(Nat.sub(Nat.add(first,U32.to_nat(width)),1n),eights)) && True{})

# The links of a split, with the facts of the paired axes after the lanes.
def joined_with(+pre_links: List<&2,Tiled.Link>,+post_links: List<&2,Tiled.Link>,+pre: List<&2,U32>,+post: List<&2,U32>,+anchors: U32,+width: U32,+pre_size: Nat,
  +pre_digits: List<&2,Nat>,+anchor: Nat,+post_digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,
  +pre_rest: List<&2,U32>,+post_rest: List<&2,U32>,
  +pre_held: PairedSum(Some{pre_links},pre,pre_size,pre_digits,cells,lane,position,groups_axis,cell,width,pre_rest,Tiled.Anchor{} <> Tiled.Lane{} <> post_links,anchors <> width <> post,
    anchor <> post_digits),
  +post_known: KernelGemm.Both<{link_sum(post_links,post,post_digits,cells,lane,position,groups_axis,cell) == Nat.mul(ShapeLaws.digits_value(post,post_digits),ShapeLaws.elements(post_rest)) : Nat},
    {anchor_fits(post_links,post,post_digits,width,cell) == True{} : Bool}>) ->
  KernelGemm.Both<{link_sum(List.append(&2,Tiled.Link,pre_links,Tiled.Anchor{} <> Tiled.Lane{} <> post_links),List.append(&2,U32,pre,anchors <> width <> post),
        List.append(&2,Nat,pre_digits,anchor <> post_digits),cells,lane,position,groups_axis,cell) ==
      Nat.add(Nat.mul(ShapeLaws.digits_value(pre,pre_digits),ShapeLaws.elements(pre_rest)),
        Nat.add(Nat.mul(cell,weight(cells,groups_axis)),Nat.mul(ShapeLaws.digits_value(post,post_digits),ShapeLaws.elements(post_rest)))) : Nat},
    {anchor_fits(List.append(&2,Tiled.Link,pre_links,Tiled.Anchor{} <> Tiled.Lane{} <> post_links),List.append(&2,U32,pre,anchors <> width <> post),
      List.append(&2,Nat,pre_digits,anchor <> post_digits),width,cell) == anchor_holds(width,anchor,cell) : Bool}>:
  KernelGemm.Both{+pre_same,KernelGemm.Both{+pre_sum,+pre_fits}} = pre_held
  KernelGemm.Both{+post_sum,+post_fits} = post_known
  +links = List.append(&2,Tiled.Link,pre_links,Tiled.Anchor{} <> Tiled.Lane{} <> post_links)
  +shape = List.append(&2,U32,pre,anchors <> width <> post)
  +digits = List.append(&2,Nat,pre_digits,anchor <> post_digits)
  +before = Nat.mul(ShapeLaws.digits_value(pre,pre_digits),ShapeLaws.elements(pre_rest))
  +read = Nat.mul(cell,weight(cells,groups_axis))
  +first = Nat.add(U32.to_nat(0),Nat.mul(U32.to_nat(width),anchor))
  +eights = U32.to_nat(U32.max(8,1))
  +above = Nat.is_le(Nat.div(first,eights),Nat.add(U32.to_nat(0),cell))
  +under = Nat.is_le(Nat.add(U32.to_nat(0),cell),Nat.div(Nat.sub(Nat.add(first,U32.to_nat(width)),1n),eights))
  +sums = Equal.trans(Nat,link_sum(links,shape,digits,cells,lane,position,groups_axis,cell),
    Nat.add(before,Nat.add(read,link_sum(post_links,post,post_digits,cells,lane,position,groups_axis,cell))),
    Nat.add(before,Nat.add(read,Nat.mul(ShapeLaws.digits_value(post,post_digits),ShapeLaws.elements(post_rest)))),pre_sum,
    Equal.cong(Nat,Nat,value => Nat.add(before,Nat.add(read,value)),link_sum(post_links,post,post_digits,cells,lane,position,groups_axis,cell),
      Nat.mul(ShapeLaws.digits_value(post,post_digits),ShapeLaws.elements(post_rest)),post_sum))
  +fits = Equal.trans(Bool,anchor_fits(links,shape,digits,width,cell),above && (under && anchor_fits(post_links,post,post_digits,width,cell)),anchor_holds(width,anchor,cell),
    pre_fits,Equal.cong(Bool,Bool,value => above && (under && value),anchor_fits(post_links,post,post_digits,width,cell),True{},post_fits))
  KernelGemm.Both{sums,fits}


# The links of a split: paired axes before the lanes, the anchors, the new
# lanes and paired axes after them. They reach the source cell number of
# both pairings and the anchor read, and the anchor reads the source group
# cell when it holds.
def joined_facts(+pre_links: List<&2,Tiled.Link>,+post_links: List<&2,Tiled.Link>,+pre: List<&2,U32>,+post: List<&2,U32>,+anchors: U32,+width: U32,+pre_size: Nat,
  +post_size: Nat,+pre_digits: List<&2,Nat>,+anchor: Nat,+post_digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,
  +pre_rest: List<&2,U32>,+post_rest: List<&2,U32>,
  +pre_held: PairedSum(Some{pre_links},pre,pre_size,pre_digits,cells,lane,position,groups_axis,cell,width,pre_rest,Tiled.Anchor{} <> Tiled.Lane{} <> post_links,anchors <> width <> post,
    anchor <> post_digits),
  +post_held: PairedSum(Some{post_links},post,post_size,post_digits,cells,lane,position,groups_axis,cell,width,post_rest,[],[],[])) ->
  KernelGemm.Both<{link_sum(List.append(&2,Tiled.Link,pre_links,Tiled.Anchor{} <> Tiled.Lane{} <> post_links),List.append(&2,U32,pre,anchors <> width <> post),
        List.append(&2,Nat,pre_digits,anchor <> post_digits),cells,lane,position,groups_axis,cell) ==
      Nat.add(Nat.mul(ShapeLaws.digits_value(pre,pre_digits),ShapeLaws.elements(pre_rest)),
        Nat.add(Nat.mul(cell,weight(cells,groups_axis)),Nat.mul(ShapeLaws.digits_value(post,post_digits),ShapeLaws.elements(post_rest)))) : Nat},
    {anchor_fits(List.append(&2,Tiled.Link,pre_links,Tiled.Anchor{} <> Tiled.Lane{} <> post_links),List.append(&2,U32,pre,anchors <> width <> post),
      List.append(&2,Nat,pre_digits,anchor <> post_digits),width,cell) == anchor_holds(width,anchor,cell) : Bool}>:
  joined_with(pre_links,post_links,pre,post,anchors,width,pre_size,pre_digits,anchor,post_digits,cells,lane,position,groups_axis,cell,pre_rest,post_rest,pre_held,
    post_facts(post_links,post,post_size,post_digits,cells,lane,position,groups_axis,cell,width,post_rest,post_held))
# ---- A shape around an axis ----

# The extents before an axis, from a shape that has them first.
def taken_front(front: List<&2,U32>,back: List<&2,U32>) -> {List.take(&2,U32,List.append(&2,U32,front,back),List.length(&2,U32,front)) == front : List<&2,U32>}:
  match front back:
    case Nil{} Nil{}: {==}
    case Nil{} Con{+extent,+rest}: {==}
    case Con{+extent,+rest} back:
      Equal.cong(List<&2,U32>,List<&2,U32>,values => extent <> values,List.take(&2,U32,List.append(&2,U32,rest,back),List.length(&2,U32,rest)),rest,taken_front(rest,back))

# The extents count past the extents before an axis.
def dropped_front(front: List<&2,U32>,+back: List<&2,U32>,+count: Nat) ->
  {List.drop(&2,U32,List.append(&2,U32,front,back),Nat.add(List.length(&2,U32,front),count)) == List.drop(&2,U32,back,count) : List<&2,U32>}:
  match front:
    case Nil{}: {==}
    case Con{+extent,+rest}: dropped_front(rest,back,count)

# A shape without the axis after the extents before it.
def removed_front(front: List<&2,U32>,+extent: U32,+back: List<&2,U32>) ->
  {Grid.removed(List.append(&2,U32,front,extent <> back),List.length(&2,U32,front)) == List.append(&2,U32,front,back) : List<&2,U32>}:
  match front:
    case Nil{}: {==}
    case Con{+first,+rest}:
      Equal.cong(List<&2,U32>,List<&2,U32>,values => first <> values,Grid.removed(List.append(&2,U32,rest,extent <> back),List.length(&2,U32,rest)),List.append(&2,U32,rest,back),
        removed_front(rest,extent,back))

# A shape without the second of two axes after the extents before them.
def removed_next(front: List<&2,U32>,+extent: U32,+next: U32,+back: List<&2,U32>) ->
  {Grid.removed(List.append(&2,U32,front,extent <> next <> back),Nat.add(List.length(&2,U32,front),1n)) == List.append(&2,U32,front,extent <> back) : List<&2,U32>}:
  match front:
    case Nil{}: {==}
    case Con{+first,+rest}:
      Equal.cong(List<&2,U32>,List<&2,U32>,values => first <> values,Grid.removed(List.append(&2,U32,rest,extent <> next <> back),Nat.add(List.length(&2,U32,rest),1n)),
        List.append(&2,U32,rest,extent <> back),removed_next(rest,extent,next,back))

# The extent of the axis after the extents before it.
def lanes_front(front: List<&2,U32>,+extent: U32,+back: List<&2,U32>) -> {Grid.lanes_at(List.append(&2,U32,front,extent <> back),List.length(&2,U32,front)) == extent : U32}:
  match front:
    case Nil{}: {==}
    case Con{+first,+rest}: lanes_front(rest,extent,back)

def lanes_next(front: List<&2,U32>,+extent: U32,+next: U32,+back: List<&2,U32>) ->
  {Grid.lanes_at(List.append(&2,U32,front,extent <> next <> back),Nat.add(List.length(&2,U32,front),1n)) == next : U32}:
  match front:
    case Nil{}: {==}
    case Con{+first,+rest}: lanes_next(rest,extent,next,back)

# The axis after the extents before it lies inside the shape.
def inside_front(front: List<&2,U32>,+extent: U32,+back: List<&2,U32>) ->
  {Nat.is_lt(List.length(&2,U32,front),List.length(&2,U32,List.append(&2,U32,front,extent <> back))) == True{} : Bool}:
  match front:
    case Nil{}: {==}
    case Con{+first,+rest}: inside_front(rest,extent,back)

def inside_next(front: List<&2,U32>,+extent: U32,+next: U32,+back: List<&2,U32>) ->
  {Nat.is_lt(Nat.add(List.length(&2,U32,front),1n),List.length(&2,U32,List.append(&2,U32,front,extent <> next <> back))) == True{} : Bool}:
  match front:
    case Nil{}: {==}
    case Con{+first,+rest}: inside_next(rest,extent,next,back)

# The cells of a source with its lanes after the extents before them.
def source_cells_of(+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+position: Grid.Position) ->
  {Tiled.cell_shape(List.append(&2,U32,front,lanes <> back),List.length(&2,U32,front),position) ==
    Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)) : List<&2,U32>}:
  +shape = List.append(&2,U32,front,lanes <> back)
  +axis = List.length(&2,U32,front)
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shape)),True{},inside_front(front,lanes,back)) :
    {Grid.placed(_,position,Grid.removed(shape,axis),Grid.groups(Grid.lanes_at(shape,axis))) == Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)) : List<&2,U32>}
  %Equal.sym(List<&2,U32>,Grid.removed(shape,axis),List.append(&2,U32,front,back),removed_front(front,lanes,back)) :
    {Grid.placed(True{},position,_,Grid.groups(Grid.lanes_at(shape,axis))) == Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)) : List<&2,U32>}
  %Equal.sym(U32,Grid.lanes_at(shape,axis),lanes,lanes_front(front,lanes,back)) :
    {Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(_)) == Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)) : List<&2,U32>}
  {==}

# The cells of a target that splits an axis in anchors and width, its lanes
# on the second.
def target_cells_of(+front: List<&2,U32>,+anchors: U32,+width: U32,+back: List<&2,U32>,+position: Grid.Position) ->
  {Tiled.cell_shape(List.append(&2,U32,front,anchors <> width <> back),Nat.add(List.length(&2,U32,front),1n),position) ==
    Grid.placed(True{},position,List.append(&2,U32,front,anchors <> back),Grid.groups(width)) : List<&2,U32>}:
  +shape = List.append(&2,U32,front,anchors <> width <> back)
  +axis = Nat.add(List.length(&2,U32,front),1n)
  +others = List.append(&2,U32,front,anchors <> back)
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shape)),True{},inside_next(front,anchors,width,back)) :
    {Grid.placed(_,position,Grid.removed(shape,axis),Grid.groups(Grid.lanes_at(shape,axis))) == Grid.placed(True{},position,others,Grid.groups(width)) : List<&2,U32>}
  %Equal.sym(List<&2,U32>,Grid.removed(shape,axis),others,removed_next(front,anchors,width,back)) :
    {Grid.placed(True{},position,_,Grid.groups(Grid.lanes_at(shape,axis))) == Grid.placed(True{},position,others,Grid.groups(width)) : List<&2,U32>}
  %Equal.sym(U32,Grid.lanes_at(shape,axis),width,lanes_next(front,anchors,width,back)) :
    {Grid.placed(True{},position,others,Grid.groups(_)) == Grid.placed(True{},position,others,Grid.groups(width)) : List<&2,U32>}
  {==}

# ---- The numbers of a target cell and the source cell it reads ----

# The digits of a target cell: those of the other axes and the group, with
# the groups innermost or outermost.
def target_digits(position: Grid.Position,+digits: List<&2,Nat>,+group: Nat) -> List<&2,Nat>:
  match position:
    case Grid.Innermost{}: List.append(&2,Nat,digits,[group])
    case Grid.Outermost{}: group <> digits

def target_below(position: Grid.Position,+others: List<&2,U32>,+groups: U32,+digits: List<&2,Nat>,+group: Nat,+below: Coordinates.DigitsBelow(others,digits),
  +group_below: {Nat.is_lt(group,U32.to_nat(groups)) == True{} : Bool}) -> Coordinates.DigitsBelow(Grid.placed(True{},position,others,groups),target_digits(position,digits,group)):
  match position:
    case Grid.Innermost{}: Coordinates.digits_joined(others,digits,[groups],[group],below,KernelGemm.Both{group_below,Unit{}})
    case Grid.Outermost{}: KernelGemm.Both{group_below,below}

def target_number(position: Grid.Position,+others: List<&2,U32>,+groups: U32,+digits: List<&2,Nat>,+group: Nat,+below: Coordinates.DigitsBelow(others,digits)) ->
  {ShapeLaws.digits_value(Grid.placed(True{},position,others,groups),target_digits(position,digits,group)) ==
    Representation.index_at(position,ShapeLaws.elements(others),U32.to_nat(groups),ShapeLaws.digits_value(others,digits),group) : Nat}:
  match position:
    case Grid.Innermost{}:
      +number = ShapeLaws.digits_value(others,digits)
      +count = U32.to_nat(groups)
      +last = Nat.add(Nat.mul(group,1n),0n)
      +spread = Nat.add(Nat.mul(number,Nat.mul(count,1n)),last)
      Equal.trans(Nat,ShapeLaws.digits_value(List.append(&2,U32,others,[groups]),List.append(&2,Nat,digits,[group])),spread,Nat.add(Nat.mul(number,count),group),
        Coordinates.digits_appended(others,digits,[groups],[group],below),
        {%Equal.sym(Nat,Nat.mul(count,1n),count,Order.times_one(count)) : {Nat.add(Nat.mul(number,_),last) == Nat.add(Nat.mul(number,count),group) : Nat}
          %Equal.sym(Nat,last,group,Equal.trans(Nat,last,Nat.mul(group,1n),group,Addition.zero_right(Nat.mul(group,1n)),Order.times_one(group))) :
            {Nat.add(Nat.mul(number,count),_) == Nat.add(Nat.mul(number,count),group) : Nat}
          {==} : {spread == Nat.add(Nat.mul(number,count),group) : Nat}})
    case Grid.Outermost{}: {==}

# The digits of the axes other than the groups: the coordinates of an outer
# index before the anchors, the anchor and the coordinates of an inner index.
def others_digits(+front: List<&2,U32>,+outer: Nat,+anchor: Nat,+back: List<&2,U32>,+inner: Nat) -> List<&2,Nat>:
  List.append(&2,Nat,Targets.cell_coordinates(front,outer),anchor <> Targets.cell_coordinates(back,inner))

def others_below(+front: List<&2,U32>,+anchors: U32,+back: List<&2,U32>,+outer: Nat,+anchor: Nat,+inner: Nat,
  +outer_below: {Nat.is_lt(outer,ShapeLaws.elements(front)) == True{} : Bool},+anchor_below: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},
  +inner_below: {Nat.is_lt(inner,ShapeLaws.elements(back)) == True{} : Bool}) ->
  Coordinates.DigitsBelow(List.append(&2,U32,front,anchors <> back),others_digits(front,outer,anchor,back,inner)):
  Coordinates.digits_joined(front,Targets.cell_coordinates(front,outer),anchors <> back,anchor <> Targets.cell_coordinates(back,inner),
    BoxDigits.coordinates_below(front,outer,outer_below),KernelGemm.Both{anchor_below,BoxDigits.coordinates_below(back,inner,inner_below)})

def others_number(+front: List<&2,U32>,+anchors: U32,+back: List<&2,U32>,+outer: Nat,+anchor: Nat,+inner: Nat,
  +outer_below: {Nat.is_lt(outer,ShapeLaws.elements(front)) == True{} : Bool},+inner_below: {Nat.is_lt(inner,ShapeLaws.elements(back)) == True{} : Bool}) ->
  {ShapeLaws.digits_value(List.append(&2,U32,front,anchors <> back),others_digits(front,outer,anchor,back,inner)) ==
    Nat.add(Nat.mul(Nat.add(Nat.mul(outer,U32.to_nat(anchors)),anchor),ShapeLaws.elements(back)),inner) : Nat}:
  +count = U32.to_nat(anchors)
  +size = ShapeLaws.elements(back)
  +front_digits = Targets.cell_coordinates(front,outer)
  +back_digits = Targets.cell_coordinates(back,inner)
  +result = Nat.add(Nat.mul(Nat.add(Nat.mul(outer,count),anchor),size),inner)
  +spread = Nat.add(Nat.mul(ShapeLaws.digits_value(front,front_digits),Nat.mul(count,size)),Nat.add(Nat.mul(anchor,size),ShapeLaws.digits_value(back,back_digits)))
  +outer_part = Nat.mul(Nat.mul(outer,count),size)
  +anchor_part = Nat.mul(anchor,size)
  Equal.trans(Nat,ShapeLaws.digits_value(List.append(&2,U32,front,anchors <> back),others_digits(front,outer,anchor,back,inner)),spread,result,
    Coordinates.digits_appended(front,front_digits,anchors <> back,anchor <> back_digits,BoxDigits.coordinates_below(front,outer,outer_below)),
    {%Equal.sym(Nat,ShapeLaws.digits_value(front,front_digits),outer,BoxDigits.value_coordinates(front,outer,outer_below)) :
        {Nat.add(Nat.mul(_,Nat.mul(count,size)),Nat.add(anchor_part,ShapeLaws.digits_value(back,back_digits))) == result : Nat}
      %Equal.sym(Nat,ShapeLaws.digits_value(back,back_digits),inner,BoxDigits.value_coordinates(back,inner,inner_below)) :
        {Nat.add(Nat.mul(outer,Nat.mul(count,size)),Nat.add(anchor_part,_)) == result : Nat}
      %RowMajor.assoc(outer,count,size) : {Nat.add(_,Nat.add(anchor_part,inner)) == result : Nat}
      %Addition.associative(outer_part,anchor_part,inner) : {_ == result : Nat}
      %RowMajor.distrib(Nat.mul(outer,count),anchor,size) : {Nat.add(_,inner) == result : Nat}
      {==} : {spread == result : Nat}})

# ---- The source cells a split reads ----

# The source cells after the paired axes before the lanes, and after those
# after them.
def pre_rest(position: Grid.Position,+back: List<&2,U32>,+groups: U32) -> List<&2,U32>:
  match position:
    case Grid.Innermost{}: List.append(&2,U32,back,[groups])
    case Grid.Outermost{}: back

def post_rest(position: Grid.Position,+groups: U32) -> List<&2,U32>:
  match position:
    case Grid.Innermost{}: [groups]
    case Grid.Outermost{}: []

# The source groups axis of the source cells, and the source cells one of
# its groups spans.
def groups_axis_of(+position: Grid.Position,+cells: List<&2,U32>) -> U32:
  Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,cells)) - 1 : U32))

def groups_stride(position: Grid.Position,+others: List<&2,U32>) -> Nat:
  match position:
    case Grid.Innermost{}: 1n
    case Grid.Outermost{}: ShapeLaws.elements(others)

# A list after its own length.
def dropped_length(front: List<&2,U32>,+back: List<&2,U32>) -> {PlacingLaw.dropped(~U32,List.length(&2,U32,front),List.append(&2,U32,front,back)) == back : List<&2,U32>}:
  match front:
    case Nil{}: {==}
    case Con{+extent,+rest}: dropped_length(rest,back)

# The paired axes before the lanes start the source cells.
def pre_from(position: Grid.Position,+front: List<&2,U32>,+back: List<&2,U32>,+groups: U32) ->
  {PlacingLaw.dropped(~U32,cell_place(List.length(&2,U32,front),position,0n),Grid.placed(True{},position,List.append(&2,U32,front,back),groups)) ==
    List.append(&2,U32,front,pre_rest(position,back,groups)) : List<&2,U32>}:
  match position:
    case Grid.Innermost{}:
      +lane = List.length(&2,U32,front)
      +cells = List.append(&2,U32,List.append(&2,U32,front,back),[groups])
      %Equal.sym(Bool,Nat.is_lt(lane,0n),False{},Order.not_below(0n,lane,Order.zero_le(lane))) :
        {PlacingLaw.dropped(~U32,place_of(True{} && _,True{} && Grid.is_outermost(Grid.Innermost{}),0n),cells) == List.append(&2,U32,front,List.append(&2,U32,back,[groups])) : List<&2,U32>}
      ListReadback.append_associative(~U32,front,back,[groups])
    case Grid.Outermost{}:
      +lane = List.length(&2,U32,front)
      +cells = {groups <> List.append(&2,U32,front,back) : List<&2,U32>}
      %Equal.sym(Bool,Nat.is_lt(lane,0n),False{},Order.not_below(0n,lane,Order.zero_le(lane))) :
        {PlacingLaw.dropped(~U32,place_of(True{} && _,True{} && Grid.is_outermost(Grid.Outermost{}),0n),cells) == List.append(&2,U32,front,back) : List<&2,U32>}
      {==}

# The lane axis lies before the axis after it.
def lane_before(+lane: Nat) -> {Nat.is_lt(lane,Nat.add(lane,1n)) == True{} : Bool}:
  %Equal.sym(Nat,Nat.add(lane,1n),1n+lane,Addition.successor_width(lane)) : {Nat.is_lt(lane,_) == True{} : Bool}
  Order.lt_succ(lane,lane,Order.reflexive(lane))

# The paired axes after the lanes start the source cells after the lanes.
def post_from(position: Grid.Position,+front: List<&2,U32>,+back: List<&2,U32>,+groups: U32) ->
  {PlacingLaw.dropped(~U32,cell_place(List.length(&2,U32,front),position,Nat.add(List.length(&2,U32,front),1n)),Grid.placed(True{},position,List.append(&2,U32,front,back),groups)) ==
    List.append(&2,U32,back,post_rest(position,groups)) : List<&2,U32>}:
  match position:
    case Grid.Innermost{}:
      +lane = List.length(&2,U32,front)
      +cells = List.append(&2,U32,List.append(&2,U32,front,back),[groups])
      +after = List.append(&2,U32,back,[groups])
      %Equal.sym(Bool,Nat.is_lt(lane,Nat.add(lane,1n)),True{},lane_before(lane)) :
        {PlacingLaw.dropped(~U32,place_of(True{} && _,True{} && Grid.is_outermost(Grid.Innermost{}),Nat.add(lane,1n)),cells) == after : List<&2,U32>}
      %Equal.sym(Nat,Nat.sub(Nat.add(lane,1n),1n),lane,DivisionOrder.subtract_sum(lane,1n)) : {PlacingLaw.dropped(~U32,_,cells) == after : List<&2,U32>}
      %Equal.sym(List<&2,U32>,cells,List.append(&2,U32,front,after),ListReadback.append_associative(~U32,front,back,[groups])) :
        {PlacingLaw.dropped(~U32,lane,_) == after : List<&2,U32>}
      dropped_length(front,after)
    case Grid.Outermost{}:
      +lane = List.length(&2,U32,front)
      +cells = {groups <> List.append(&2,U32,front,back) : List<&2,U32>}
      %Equal.sym(Bool,Nat.is_lt(lane,Nat.add(lane,1n)),True{},lane_before(lane)) :
        {PlacingLaw.dropped(~U32,place_of(True{} && _,True{} && Grid.is_outermost(Grid.Outermost{}),Nat.add(lane,1n)),cells) == List.append(&2,U32,back,[]) : List<&2,U32>}
      %Equal.sym(Nat,Nat.sub(Nat.add(lane,1n),1n),lane,DivisionOrder.subtract_sum(lane,1n)) : {PlacingLaw.dropped(~U32,1n+_,cells) == List.append(&2,U32,back,[]) : List<&2,U32>}
      %Equal.sym(List<&2,U32>,List.append(&2,U32,back,[]),back,ListReadback.append_empty(~U32,back)) :
        {PlacingLaw.dropped(~U32,lane,List.append(&2,U32,front,back)) == _ : List<&2,U32>}
      dropped_length(front,back)

# The stride of the source groups axis.
def groups_weight(position: Grid.Position,+others: List<&2,U32>,+groups: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(cap)) == True{} : Bool}) ->
  {weight(Grid.placed(True{},position,others,groups),groups_axis_of(position,Grid.placed(True{},position,others,groups))) == groups_stride(position,others) : Nat}:
  match position:
    case Grid.Innermost{}:
      +cells = List.append(&2,U32,others,[groups])
      +count = List.length(&2,U32,others)
      %Equal.sym(Nat,List.length(&2,U32,cells),1n+count,Public.appended_length(others,groups)) : {weight(cells,(U32.from_nat(_) - 1 : U32)) == 1n : Nat}
      %Equal.sym(U32,(U32.from_nat(1n+count) - 1 : U32),U32.from_nat(count),RelaneGeometry.less_one(count)) : {weight(cells,_) == 1n : Nat}
      Equal.trans(Nat,weight(cells,U32.from_nat(count)),ShapeLaws.elements(PlacingLaw.dropped(~U32,1n+count,cells)),1n,
        stride_weight(cells,count,cap,cap_ok,room,positive,rank_room,inside_front(others,groups,[])),
        Equal.cong(List<&2,U32>,Nat,values => ShapeLaws.elements(values),PlacingLaw.dropped(~U32,1n+count,cells),[],
          PlacingLaw.dropped_next(~U32,count,cells,groups,[],dropped_length(others,[groups]))))
    case Grid.Outermost{}:
      stride_weight(groups <> others,0n,cap,cap_ok,room,positive,rank_room,{==})

# The source cell number of the links of a split at the outer index before
# the lanes, the group cell and the inner index after them.
def source_number(position: Grid.Position,+front: List<&2,U32>,+back: List<&2,U32>,+groups: U32,+outer: Nat,+cell: Nat,+inner: Nat) ->
  {Nat.add(Nat.mul(outer,ShapeLaws.elements(pre_rest(position,back,groups))),Nat.add(Nat.mul(cell,groups_stride(position,List.append(&2,U32,front,back))),
      Nat.mul(inner,ShapeLaws.elements(post_rest(position,groups))))) ==
    Representation.index_at(position,ShapeLaws.elements(List.append(&2,U32,front,back)),U32.to_nat(groups),Nat.add(Nat.mul(outer,ShapeLaws.elements(back)),inner),cell) : Nat}:
  match position:
    case Grid.Innermost{}:
      +size = ShapeLaws.elements(back)
      +count = U32.to_nat(groups)
      +unit = Nat.mul(count,1n)
      +before = Nat.mul(Nat.mul(outer,size),count)
      +after = Nat.mul(inner,count)
      +result = Nat.add(Nat.mul(Nat.add(Nat.mul(outer,size),inner),count),cell)
      %Equal.sym(Nat,ShapeLaws.elements(List.append(&2,U32,back,[groups])),Nat.mul(size,unit),ShapeLaws.elements_appended(back,[groups])) :
        {Nat.add(Nat.mul(outer,_),Nat.add(Nat.mul(cell,1n),Nat.mul(inner,unit))) == result : Nat}
      %Equal.sym(Nat,Nat.mul(cell,1n),cell,Order.times_one(cell)) : {Nat.add(Nat.mul(outer,Nat.mul(size,unit)),Nat.add(_,Nat.mul(inner,unit))) == result : Nat}
      %Equal.sym(Nat,unit,count,Order.times_one(count)) : {Nat.add(Nat.mul(outer,Nat.mul(size,_)),Nat.add(cell,Nat.mul(inner,_))) == result : Nat}
      %RowMajor.assoc(outer,size,count) : {Nat.add(_,Nat.add(cell,after)) == result : Nat}
      %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(outer,size),inner),count),Nat.add(before,after),RowMajor.distrib(Nat.mul(outer,size),inner,count)) :
        {Nat.add(before,Nat.add(cell,after)) == Nat.add(_,cell) : Nat}
      %Order.commutative(after,cell) : {Nat.add(before,_) == Nat.add(Nat.add(before,after),cell) : Nat}
      Equal.sym(Nat,Nat.add(Nat.add(before,after),cell),Nat.add(before,Nat.add(after,cell)),Addition.associative(before,after,cell))
    case Grid.Outermost{}:
      +front_part = Nat.mul(outer,ShapeLaws.elements(back))
      +read = Nat.mul(cell,ShapeLaws.elements(List.append(&2,U32,front,back)))
      %Equal.sym(Nat,Nat.mul(inner,1n),inner,Order.times_one(inner)) : {Nat.add(front_part,Nat.add(read,_)) == Nat.add(read,Nat.add(front_part,inner)) : Nat}
      Equal.trans(Nat,Nat.add(front_part,Nat.add(read,inner)),Nat.add(Nat.add(front_part,read),inner),Nat.add(read,Nat.add(front_part,inner)),
        Equal.sym(Nat,Nat.add(Nat.add(front_part,read),inner),Nat.add(front_part,Nat.add(read,inner)),Addition.associative(front_part,read,inner)),
        Equal.trans(Nat,Nat.add(Nat.add(front_part,read),inner),Nat.add(Nat.add(read,front_part),inner),Nat.add(read,Nat.add(front_part,inner)),
          Equal.cong(Nat,Nat,value => Nat.add(value,inner),Nat.add(front_part,read),Nat.add(read,front_part),Order.commutative(front_part,read)),
          Addition.associative(read,front_part,inner)))

# ---- The links of a split at a target cell ----

# What the links of a split give at the digits of a target cell: the source
# cell number they reach, whether the anchor reads the group cell, and the
# target axes they read.
def LinksFacts(found: Maybe<&2,List<&2,Tiled.Link>>,+shape: List<&2,U32>,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,
  +cell: Nat,+width: U32,+number: Nat,+fits: Bool,+extents: List<&2,U32>) -> Data:
  match found:
    case Some{+links}:
      KernelGemm.Both<{link_sum(links,shape,digits,cells,lane,position,groups_axis,cell) == number : Nat},
        KernelGemm.Both<{anchor_fits(links,shape,digits,width,cell) == fits : Bool},{link_extents(links,shape) == extents : List<&2,U32>}>>
    case None{}: Unit

# The facts of links whose sum is a number.
def facts_at(+links: List<&2,Tiled.Link>,+shape: List<&2,U32>,+digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,+cell: Nat,
  +width: U32,+sum: Nat,+number: Nat,+fits: Bool,+extents: List<&2,U32>,
  +joined: KernelGemm.Both<{link_sum(links,shape,digits,cells,lane,position,groups_axis,cell) == sum : Nat},
    {anchor_fits(links,shape,digits,width,cell) == fits : Bool}>,
  +extents_exact: {link_extents(links,shape) == extents : List<&2,U32>},+exact: {sum == number : Nat}) -> LinksFacts(Some{links},shape,digits,cells,lane,position,groups_axis,cell,width,number,fits,extents):
  KernelGemm.Both{+summed,+fitted} = joined
  KernelGemm.Both{Equal.trans(Nat,link_sum(links,shape,digits,cells,lane,position,groups_axis,cell),sum,number,summed,exact),KernelGemm.Both{fitted,extents_exact}}

# The links of two pairings, each with the facts of its axes.
def links_facts(found_pre: Maybe<&2,List<&2,Tiled.Link>>,found_post: Maybe<&2,List<&2,Tiled.Link>>,+pre: List<&2,U32>,+post: List<&2,U32>,+anchors: U32,+width: U32,
  +pre_size: Nat,+post_size: Nat,+pre_digits: List<&2,Nat>,+anchor: Nat,+post_digits: List<&2,Nat>,+cells: List<&2,U32>,+lane: Nat,+position: Grid.Position,+groups_axis: U32,
  +cell: Nat,+pre_rest: List<&2,U32>,+post_rest: List<&2,U32>,+number: Nat,
  +number_exact: {Nat.add(Nat.mul(ShapeLaws.digits_value(pre,pre_digits),ShapeLaws.elements(pre_rest)),
      Nat.add(Nat.mul(cell,weight(cells,groups_axis)),Nat.mul(ShapeLaws.digits_value(post,post_digits),ShapeLaws.elements(post_rest)))) == number : Nat},
  +pre_held: PairedSum(found_pre,pre,pre_size,pre_digits,cells,lane,position,groups_axis,cell,width,pre_rest,Tiled.Anchor{} <> Tiled.Lane{} <> links_of(found_post),
    anchors <> width <> post,anchor <> post_digits),
  +post_held: PairedSum(found_post,post,post_size,post_digits,cells,lane,position,groups_axis,cell,width,post_rest,[],[],[]),
  +known: SplitIn(Tiled.joined_links(found_pre,found_post),List.append(&2,U32,pre,anchors <> width <> post),cells,lane,position,anchors,List.append(&2,U32,pre,anchors <> post))) ->
  LinksFacts(Tiled.joined_links(found_pre,found_post),List.append(&2,U32,pre,anchors <> width <> post),List.append(&2,Nat,pre_digits,anchor <> post_digits),cells,lane,position,
    groups_axis,cell,width,number,anchor_holds(width,anchor,cell),List.append(&2,U32,pre,anchors <> post)):
  match found_pre found_post:
    case Some{+pre_links} Some{+post_links}:
      KernelGemm.Both{+links_in,+extents_exact} = known
      +joined = joined_facts(pre_links,post_links,pre,post,anchors,width,pre_size,post_size,pre_digits,anchor,post_digits,cells,lane,position,groups_axis,cell,pre_rest,post_rest,
        pre_held,post_held)
      facts_at(List.append(&2,Tiled.Link,pre_links,Tiled.Anchor{} <> Tiled.Lane{} <> post_links),List.append(&2,U32,pre,anchors <> width <> post),
        List.append(&2,Nat,pre_digits,anchor <> post_digits),cells,lane,position,groups_axis,cell,width,
        Nat.add(Nat.mul(ShapeLaws.digits_value(pre,pre_digits),ShapeLaws.elements(pre_rest)),
          Nat.add(Nat.mul(cell,weight(cells,groups_axis)),Nat.mul(ShapeLaws.digits_value(post,post_digits),ShapeLaws.elements(post_rest)))),
        number,anchor_holds(width,anchor,cell),List.append(&2,U32,pre,anchors <> post),joined,extents_exact,number_exact)
    case Some{+pre_links} None{}: Unit{}
    case None{} found_post: Unit{}

# The links of a split read the source cell axes of paired links and every
# target axis but the new lanes.
def split_links_in(+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+position: Grid.Position) ->
  SplitIn(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),
    Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),List.length(&2,U32,front),position,anchors,List.append(&2,U32,pre,anchors <> post)):
  +lane = List.length(&2,U32,front)
  +groups = Grid.groups(lanes)
  +cells = Grid.placed(True{},position,List.append(&2,U32,front,back),groups)
  +found_post = Tiled.pairing(post,back,Nat.add(lane,1n))
  +post_held = paired_in(post,back,Nat.add(List.length(&2,U32,post),List.length(&2,U32,back)),Nat.add(lane,1n),True{},cells,lane,position,anchors,post_rest(position,groups),[],[],
    post_from(position,front,back,groups),lane_before(lane),Unit{})
  +pre_held = paired_in(pre,front,Nat.add(List.length(&2,U32,pre),lane),0n,False{},cells,lane,position,anchors,pre_rest(position,back,groups),
    Tiled.Anchor{} <> Tiled.Lane{} <> links_of(found_post),anchors <> width <> post,pre_from(position,front,back,groups),Order.reflexive(lane),
    KernelGemm.Both{{==},post_links_in(found_post,post,cells,lane,position,anchors,post_held)})
  joined_split(Tiled.pairing(pre,front,0n),found_post,pre,post,anchors,width,cells,lane,position,pre_held,post_held)

# The links of a split at the digits of a target cell: they reach the source
# cell of its outer and inner index at a group cell, the anchor reads that
# cell when it holds, and they read every target axis but the new lanes.
def cell_links(+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,+position: Grid.Position,
  +outer: Nat,+anchor: Nat,+inner: Nat,+cell: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(cap)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(cap)) == True{} : Bool},
  +outer_below: {Nat.is_lt(outer,ShapeLaws.elements(pre)) == True{} : Bool},+inner_below: {Nat.is_lt(inner,ShapeLaws.elements(post)) == True{} : Bool}) ->
  LinksFacts(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),
    others_digits(pre,outer,anchor,post,inner),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),List.length(&2,U32,front),position,
    groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),cell,width,
    Representation.index_at(position,ShapeLaws.elements(List.append(&2,U32,front,back)),U32.to_nat(Grid.groups(lanes)),Nat.add(Nat.mul(outer,ShapeLaws.elements(back)),inner),cell),
    anchor_holds(width,anchor,cell),List.append(&2,U32,pre,anchors <> post)):
  +lane = List.length(&2,U32,front)
  +groups = Grid.groups(lanes)
  +cells = Grid.placed(True{},position,List.append(&2,U32,front,back),groups)
  +groups_axis = groups_axis_of(position,cells)
  +pre_digits = Targets.cell_coordinates(pre,outer)
  +post_digits = Targets.cell_coordinates(post,inner)
  +found_post = Tiled.pairing(post,back,Nat.add(lane,1n))
  +before = pre_rest(position,back,groups)
  +after = post_rest(position,groups)
  +number = Representation.index_at(position,ShapeLaws.elements(List.append(&2,U32,front,back)),U32.to_nat(groups),Nat.add(Nat.mul(outer,ShapeLaws.elements(back)),inner),cell)
  +pre_held = paired_sum(pre,front,Nat.add(List.length(&2,U32,pre),lane),0n,pre_digits,False{},cells,lane,position,groups_axis,cell,width,before,
    Tiled.Anchor{} <> Tiled.Lane{} <> links_of(found_post),anchors <> width <> post,anchor <> post_digits,cap,cap_ok,room,positive,rank_room,pre_from(position,front,back,groups),
    Order.reflexive(lane),BoxDigits.coordinates_below(pre,outer,outer_below))
  +post_held = paired_sum(post,back,Nat.add(List.length(&2,U32,post),List.length(&2,U32,back)),Nat.add(lane,1n),post_digits,True{},cells,lane,position,groups_axis,cell,width,
    after,[],[],[],cap,cap_ok,room,positive,rank_room,post_from(position,front,back,groups),lane_before(lane),BoxDigits.coordinates_below(post,inner,inner_below))
  +exact = {%Equal.sym(Nat,ShapeLaws.digits_value(pre,pre_digits),outer,BoxDigits.value_coordinates(pre,outer,outer_below)) :
      {Nat.add(Nat.mul(_,ShapeLaws.elements(before)),Nat.add(Nat.mul(cell,weight(cells,groups_axis)),Nat.mul(ShapeLaws.digits_value(post,post_digits),ShapeLaws.elements(after)))) ==
        number : Nat}
    %Equal.sym(Nat,ShapeLaws.digits_value(post,post_digits),inner,BoxDigits.value_coordinates(post,inner,inner_below)) :
      {Nat.add(Nat.mul(outer,ShapeLaws.elements(before)),Nat.add(Nat.mul(cell,weight(cells,groups_axis)),Nat.mul(_,ShapeLaws.elements(after)))) == number : Nat}
    %Equal.sym(Nat,weight(cells,groups_axis),groups_stride(position,List.append(&2,U32,front,back)),
        groups_weight(position,List.append(&2,U32,front,back),groups,cap,cap_ok,room,positive,rank_room)) :
      {Nat.add(Nat.mul(outer,ShapeLaws.elements(before)),Nat.add(Nat.mul(cell,_),Nat.mul(inner,ShapeLaws.elements(after)))) == number : Nat}
    source_number(position,front,back,groups,outer,cell,inner) :
    {Nat.add(Nat.mul(ShapeLaws.digits_value(pre,pre_digits),ShapeLaws.elements(before)),
        Nat.add(Nat.mul(cell,weight(cells,groups_axis)),Nat.mul(ShapeLaws.digits_value(post,post_digits),ShapeLaws.elements(after)))) == number : Nat}}
  links_facts(Tiled.pairing(pre,front,0n),found_post,pre,post,anchors,width,ShapeLaws.elements(front),ShapeLaws.elements(back),pre_digits,anchor,post_digits,cells,lane,position,
    groups_axis,cell,before,after,number,exact,pre_held,post_held,split_links_in(front,lanes,back,pre,anchors,width,post,position))

# ---- The readings of a split at a target cell ----

# The values the readings of a split read at the digits of a target cell:
# the links, and the group for the groups reading.
def split_values(position: Grid.Position,+links: List<&2,Tiled.Link>,+shape: List<&2,U32>,+digits: List<&2,Nat>,+cell: Nat,+group: Nat) -> List<&2,Nat>:
  match position:
    case Grid.Innermost{}: List.append(&2,Nat,link_values(links,shape,digits,cell),[group])
    case Grid.Outermost{}: group <> link_values(links,shape,digits,cell)

# The readings of a split read their values inside the classes of the
# target digits.
def cell_row_placed(position: Grid.Position,+links: List<&2,Tiled.Link>,+shape: List<&2,U32>,+digits: List<&2,Nat>,classes: List<&2,Tasks.Class>,+source_grid: Grid.Grid,
  +target_grid: Grid.Grid,+lane: Nat,+width: U32,+source_groups: U32,+groups_axis: U32,+cell: Nat,+group: Nat,+others: List<&2,U32>,+groups: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +rooms: Components.ReadingsRoom(Tiled.split_readings(Some{links},shape,Grid.placed(True{},position,others,groups),
    Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),groups),cap),
  +below: Coordinates.DigitsBelow(others,digits),+fits: {anchor_fits(links,shape,digits,width,cell) == True{} : Bool},
  +cell_below: {Nat.is_lt(cell,U32.to_nat(source_groups)) == True{} : Bool},+extents: {link_extents(links,shape) == others : List<&2,U32>},
  +inside: Targets.ClassesIn(classes,target_grid,Grid.placed(True{},position,others,groups),target_digits(position,digits,group),0)) ->
  TaskReads.PlacedRow(Tiled.split_readings(Some{links},shape,Grid.placed(True{},position,others,groups),
    Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),groups),classes,
    split_values(position,links,shape,digits,cell,group),cap):
  match position classes:
    case Grid.Innermost{} classes:
      +sides = {Tiled.Sides{source_grid,target_grid,lane,Grid.Innermost{},width,source_groups,groups_axis} : Tiled.Sides}
      +cells = List.append(&2,U32,others,[groups])
      +tail = {[Tiled.groups_reading(sides,groups,(U32.from_nat(List.length(&2,U32,cells)) - 1 : U32))] : List<&2,Tasks.Reading>}
      +link_below = {%Equal.sym(List<&2,U32>,link_extents(links,shape),others,extents) : Coordinates.DigitsBelow(_,digits)
        below : Coordinates.DigitsBelow(link_extents(links,shape),digits)}
      +classes_in = {%Equal.sym(List<&2,U32>,link_extents(links,shape),others,extents) :
          Targets.ClassesIn(classes,target_grid,List.append(&2,U32,_,[groups]),List.append(&2,Nat,digits,[group]),0)
        inside : Targets.ClassesIn(classes,target_grid,List.append(&2,U32,link_extents(links,shape),[groups]),List.append(&2,Nat,digits,[group]),0)}
      links_placed(links,shape,digits,classes,source_grid,target_grid,lane,Grid.Innermost{},width,source_groups,groups_axis,0,cell,tail,[groups],[group],cap,cap_ok,rooms,
        link_below,fits,cell_below,KernelGemm.Both{{==},Unit{}},classes_in)
    case Grid.Outermost{} Con{+class,+rest}:
      Held.Both{+head_room,+rooms_rest} = rooms
      Held.Both{+class_in,+inside_rest} = inside
      +sides = {Tiled.Sides{source_grid,target_grid,lane,Grid.Outermost{},width,source_groups,groups_axis} : Tiled.Sides}
      +readings = Tiled.link_readings(links,shape,sides,1)
      +values = link_values(links,shape,digits,cell)
      +read = link_extents(links,shape)
      +link_below = {%Equal.sym(List<&2,U32>,read,others,extents) : Coordinates.DigitsBelow(_,digits) below : Coordinates.DigitsBelow(read,digits)}
      +classes_in = {%Equal.sym(List<&2,U32>,List.append(&2,U32,read,[]),others,Equal.trans(List<&2,U32>,List.append(&2,U32,read,[]),read,others,ListReadback.append_empty(~U32,read),extents)) :
          Targets.ClassesIn(rest,target_grid,_,List.append(&2,Nat,digits,[]),1)
        %Equal.sym(List<&2,Nat>,List.append(&2,Nat,digits,[]),digits,ListReadback.append_empty(~Nat,digits)) : Targets.ClassesIn(rest,target_grid,others,_,1)
        inside_rest : Targets.ClassesIn(rest,target_grid,List.append(&2,U32,read,[]),List.append(&2,Nat,digits,[]),1)}
      +rooms_links = {%Equal.sym(List<&2,Tasks.Reading>,List.append(&2,Tasks.Reading,readings,[]),readings,ListReadback.append_empty(~Tasks.Reading,readings)) :
          Components.ReadingsRoom(_,cap)
        rooms_rest : Components.ReadingsRoom(List.append(&2,Tasks.Reading,readings,[]),cap)}
      +placed = links_placed(links,shape,digits,rest,source_grid,target_grid,lane,Grid.Outermost{},width,source_groups,groups_axis,1,cell,[],[],[],cap,cap_ok,rooms_links,link_below,
        fits,cell_below,Unit{},classes_in)
      +placed_clean = {%ListReadback.append_empty(~Tasks.Reading,readings) : TaskReads.PlacedRow(_,rest,values,cap)
        %ListReadback.append_empty(~Nat,values) : TaskReads.PlacedRow(List.append(&2,Tasks.Reading,readings,[]),rest,_,cap)
        placed : TaskReads.PlacedRow(readings,rest,values,cap)}
      Held.Both{Unit{},placed_clean}
    case Grid.Outermost{} Nil{}:
      Empty.absurd(TaskReads.PlacedRow(Tiled.split_readings(Some{links},shape,Grid.placed(True{},Grid.Outermost{},others,groups),
        Tiled.Sides{source_grid,target_grid,lane,Grid.Outermost{},width,source_groups,groups_axis},True{},groups),Nil{},
        split_values(Grid.Outermost{},links,shape,digits,cell,group),cap),inside)

# The readings of a split reach at their values the source cell number of
# the links.
def cell_row_word(position: Grid.Position,+links: List<&2,Tiled.Link>,+shape: List<&2,U32>,+digits: List<&2,Nat>,+cells: List<&2,U32>,+source_grid: Grid.Grid,
  +target_grid: Grid.Grid,+lane: Nat,+width: U32,+source_groups: U32,+groups_axis: U32,+cell: Nat,+group: Nat,+target_cells: List<&2,U32>,+groups: U32) ->
  {TaskReads.placed_word(Tiled.split_readings(Some{links},shape,target_cells,Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},
      Grid.is_outermost(position),groups),split_values(position,links,shape,digits,cell,group),Tasks.strides_of(cells)) ==
    U32.from_nat(link_sum(links,shape,digits,cells,lane,position,groups_axis,cell)) : U32}:
  match position:
    case Grid.Innermost{}:
      +sides = {Tiled.Sides{source_grid,target_grid,lane,Grid.Innermost{},width,source_groups,groups_axis} : Tiled.Sides}
      +tail = {[Tiled.groups_reading(sides,groups,(U32.from_nat(List.length(&2,U32,target_cells)) - 1 : U32))] : List<&2,Tasks.Reading>}
      +sum = U32.from_nat(link_sum(links,shape,digits,cells,lane,Grid.Innermost{},groups_axis,cell))
      Equal.trans(U32,TaskReads.placed_word(List.append(&2,Tasks.Reading,Tiled.link_readings(links,shape,sides,0),tail),
          List.append(&2,Nat,link_values(links,shape,digits,cell),[group]),Tasks.strides_of(cells)),(sum + 0 : U32),sum,
        links_word(links,shape,digits,cells,source_grid,target_grid,lane,Grid.Innermost{},width,source_groups,groups_axis,0,cell,tail,[group]),Arithmetic.add_zero(sum))
    case Grid.Outermost{}:
      +sides = {Tiled.Sides{source_grid,target_grid,lane,Grid.Outermost{},width,source_groups,groups_axis} : Tiled.Sides}
      +readings = Tiled.link_readings(links,shape,sides,1)
      +values = link_values(links,shape,digits,cell)
      +strides = Tasks.strides_of(cells)
      +sum = U32.from_nat(link_sum(links,shape,digits,cells,lane,Grid.Outermost{},groups_axis,cell))
      %ListReadback.append_empty(~Tasks.Reading,readings) : {TaskReads.placed_word(_,values,strides) == sum : U32}
      %ListReadback.append_empty(~Nat,values) : {TaskReads.placed_word(List.append(&2,Tasks.Reading,readings,[]),_,strides) == sum : U32}
      Equal.trans(U32,TaskReads.placed_word(List.append(&2,Tasks.Reading,readings,[]),List.append(&2,Nat,values,[]),strides),(sum + 0 : U32),sum,
        links_word(links,shape,digits,cells,source_grid,target_grid,lane,Grid.Outermost{},width,source_groups,groups_axis,1,cell,[],[]),Arithmetic.add_zero(sum))

# ---- A target cell of a separable split reads inside its box ----

# A target cell of a piece of a separable component, at the digits of its
# outer index, anchor, inner index and group, reads through the links a
# source cell the component takes.
def separate_links(+limits: Machine.Limits,+links: List<&2,Tiled.Link>,+shape: List<&2,U32>,+digits: List<&2,Nat>,+cells: List<&2,U32>,+others: List<&2,U32>,+groups: U32,
  +lane: Nat,+position: Grid.Position,+width: U32,+source_groups: U32,+groups_axis: U32,+cell: Nat,+group: Nat,+component: U32,+piece: U32,+local: Nat,+number: Nat,+fits: Bool,
  +word: U32,+below: Coordinates.DigitsBelow(others,digits),+cell_below: {Nat.is_lt(cell,U32.to_nat(source_groups)) == True{} : Bool},
  +coordinates: {Targets.cell_coordinates(Grid.placed(True{},position,others,groups),
      Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(Grid.placed(True{},position,others,groups)),piece)),local)) == target_digits(position,digits,group) : List<&2,Nat>},
  +word_exact: {U32.to_nat(word) == number : Nat},+fits_true: {fits == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +separate: {Tasks.separable([Tiled.split_readings(Some{links},shape,Grid.placed(True{},position,others,groups),
      Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
      groups)]) == True{} : Bool},
  +facts: Held.Both<Components.RowOk(limits,Tiled.split_readings(Some{links},shape,Grid.placed(True{},position,others,groups),
        Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
        groups),Tensors.grid_of(cells),cells),
    Held.Both<Components.RowEnds(Tiled.split_readings(Some{links},shape,Grid.placed(True{},position,others,groups),
        Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
        groups),
      Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Some{links},shape,Grid.placed(True{},position,others,groups),
          Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
          groups)]),[Tiled.split_readings(Some{links},shape,Grid.placed(True{},position,others,groups),
          Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
          groups)]),component)),
      Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Some{links},shape,Grid.placed(True{},position,others,groups),
          Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
          groups)]),[Tiled.split_readings(Some{links},shape,Grid.placed(True{},position,others,groups),
          Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
          groups)]),component),Tensors.grid_of(Grid.placed(True{},position,others,groups)),Grid.placed(True{},position,others,groups),Machine.half_word(limits))>>,
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(cells),cells,0,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of([Tiled.split_readings(Some{links},shape,Grid.placed(True{},position,others,groups),
      Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
      groups)]),Tensors.grid_of(Grid.placed(True{},position,others,groups)),Grid.placed(True{},position,others,groups),component),piece),
  +local_in: {Nat.is_lt(local,U32.to_nat(Grid.span(Tensors.grid_of(Grid.placed(True{},position,others,groups)),piece))) == True{} : Bool},
  +links_known: LinksFacts(Some{links},shape,digits,cells,lane,position,groups_axis,cell,width,number,fits,others)) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(cells),cells,Tasks.source_box(Tasks.components_of([Tiled.split_readings(Some{links},shape,
      Grid.placed(True{},position,others,groups),Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,
      groups_axis},Grid.is_outermost(position),groups)]),0,Tensors.grid_of(cells),cells,0,component)),Tensors.grid_of(cells)),U32.to_nat(word)):
  Held.Both{Held.Both{+rooms,Held.Both{+axes_ok,+axes_below}},Held.Both{+ends,+classes_ok}} = facts
  KernelGemm.Both{+summed,KernelGemm.Both{+fitted,+extents}} = links_known
  +target = Grid.placed(True{},position,others,groups)
  +source_grid = Tensors.grid_of(cells)
  +target_grid = Tensors.grid_of(target)
  +sides = {Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis} : Tiled.Sides}
  +row = Tiled.split_readings(Some{links},shape,target,sides,Grid.is_outermost(position),groups)
  +classes = Tasks.chosen(Tasks.class_lists(Tasks.target_rank([row]),[row]),component)
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +capacity = Machine.capacity(limits)
  +cap_half = Bounds.as_nat_le(capacity,cap,Limits.capacity_within_half(limits))
  +fits_links = Equal.trans(Bool,anchor_fits(links,shape,digits,width,cell),fits,True{},fitted,fits_true)
  +classes_found = Targets.target_classes(limits,target,Grid.piece_target(),target_room,target_nonempty,target_rank,[row],component,separate,classes_ok,piece,member,local,local_in)
  +classes_in = {%coordinates : Targets.ClassesIn(classes,target_grid,target,_,0)
    classes_found : Targets.ClassesIn(classes,target_grid,target,target_digits(position,digits,group),0)}
  +row_ok = cell_row_placed(position,links,shape,digits,classes,source_grid,target_grid,lane,width,source_groups,groups_axis,cell,group,others,groups,cap,cap_ok,rooms,below,
    fits_links,cell_below,extents,classes_in)
  +values = split_values(position,links,shape,digits,cell,group)
  +fixed = Grid.coordinates(cells,0)
  +source_double = Limits.double_bound(limits,ShapeLaws.elements(cells),room)
  +found = TaskReads.digits_placed(row,classes,values,source_grid,cells,fixed,0,cap,cap_ok,row_ok,axes_ok,blocks,ViewReads.coordinates_in(cells,0,cap,cap_ok,source_double,nonempty))
  +placed = TaskReads.placed_digits(row,values,cells,fixed,0)
  +apart = Logic.left(Tasks.distinct(Tasks.read_axes(row)),Bool.not(Tasks.crossing(row)),
    Logic.left(Tasks.distinct(Tasks.read_axes(row)) && Bool.not(Tasks.crossing(row)),Tasks.separable([]),separate))
  +number_eq = TaskReads.placed_number(row,values,cells,fixed,cap,cap_ok,TaskReads.zero_offset(cells,cap,cap_ok,source_double,nonempty),apart,axes_below,
    TaskReads.row_paired(row,classes,values,cap,row_ok),BoxDigits.coordinates_count(cells,0),Order.transitive(List.length(&2,U32,cells),U32.to_nat(capacity),U32.to_nat(cap),rank_room,cap_half))
  +word_eq = cell_row_word(position,links,shape,digits,cells,source_grid,target_grid,lane,width,source_groups,groups_axis,cell,group,target,groups)
  +sum = link_sum(links,shape,digits,cells,lane,position,groups_axis,cell)
  +read_word = TaskReads.placed_word(row,values,Tasks.strides_of(cells))
  +value = ShapeLaws.digits_value(cells,placed)
  +exact = Equal.trans(U32,word,U32.from_nat(U32.to_nat(word)),U32.from_nat(value),Equal.sym(U32,U32.from_nat(U32.to_nat(word)),word,Arithmetic.native_roundtrip(word)),
    Equal.trans(U32,U32.from_nat(U32.to_nat(word)),U32.from_nat(sum),U32.from_nat(value),
      Equal.cong(Nat,U32,count => U32.from_nat(count),U32.to_nat(word),sum,Equal.trans(Nat,U32.to_nat(word),number,sum,word_exact,Equal.sym(Nat,sum,number,summed))),
      Equal.trans(U32,U32.from_nat(sum),read_word,U32.from_nat(value),Equal.sym(U32,read_word,U32.from_nat(sum),word_eq),Equal.sym(U32,U32.from_nat(value),read_word,number_eq))))
  +box = Tasks.boxed(row,classes,source_grid,cells,fixed,0)
  %Equal.sym(Bool,Tasks.separable([row]),True{},separate) :
    Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(source_grid,cells,Tasks.box_when(_,row,classes,source_grid,cells,0)),source_grid),U32.to_nat(word))
  TaskReads.held_of(limits,cells,box,placed,word,room,nonempty,rank_room,BoxCells.boxed_coordinates_in(row,classes,source_grid,cells,0,cap,cap_ok,rooms,ends,axes_ok),
    ViewReads.digits_box(box,cells,placed,found),exact)

# ---- The anchor read holds at the group cells a target cell reads ----

def holds_of(+first: Bool,+second: Bool,+first_true: {first == True{} : Bool},+second_true: {second == True{} : Bool}) -> {first && (second && True{}) == True{} : Bool}:
  %Equal.sym(Bool,first,True{},first_true) : {_ && (second && True{}) == True{} : Bool}
  %Equal.sym(Bool,second,True{},second_true) : {True{} && (_ && True{}) == True{} : Bool}
  {==}

# A value below a bound is at most the bound less one.
def below_last(+value: Nat,+bound: Nat,+below: {Nat.is_lt(value,bound) == True{} : Bool}) -> {Nat.is_le(value,Nat.sub(bound,1n)) == True{} : Bool}:
  +one_le = Order.transitive(1n,1n+value,bound,Order.le_add(1n,value),Order.strict_follow(value,bound,below))
  Division.lt_succ_le(value,Nat.sub(bound,1n),
    {%Equal.sym(Nat,Nat.add(1n,Nat.sub(bound,1n)),bound,Order.restore_shift(1n,bound,one_le)) : {Nat.is_lt(value,_) == True{} : Bool}
      below : {Nat.is_lt(value,1n+Nat.sub(bound,1n)) == True{} : Bool}})

# The source cell of the first lane of a group inside the width lies among
# the groups of the anchor lanes.
def first_holds(+width: U32,+anchor: Nat,+group: Nat,+inside: {Nat.is_lt(Nat.mul(group,8n),U32.to_nat(width)) == True{} : Bool}) ->
  {anchor_holds(width,anchor,Nat.div(SplitValue.first_lane(anchor,width,group),8n)) == True{} : Bool}:
  +size = U32.to_nat(width)
  +start = Nat.mul(size,anchor)
  +first = SplitValue.first_lane(anchor,width,group)
  +turned = RowMajor.multiply_commutative(size,anchor)
  +low_start = {%Equal.sym(Nat,start,Nat.mul(anchor,size),turned) : {Nat.is_le(_,first) == True{} : Bool}
    Order.le_add(Nat.mul(anchor,size),Nat.mul(group,8n)) : {Nat.is_le(start,first) == True{} : Bool}}
  +ends = {%Equal.sym(Nat,start,Nat.mul(anchor,size),turned) : {Nat.is_lt(first,Nat.add(_,size)) == True{} : Bool}
    Order.offset_lt(Nat.mul(anchor,size),Nat.mul(group,8n),size,inside) : {Nat.is_lt(first,Nat.add(start,size)) == True{} : Bool}}
  +last = Nat.sub(Nat.add(start,size),1n)
  holds_of(Nat.is_le(Nat.div(start,8n),Nat.div(first,8n)),Nat.is_le(Nat.div(first,8n),Nat.div(last,8n)),DivisionOrder.quotient_le(start,first,8n,{==},low_start),
    DivisionOrder.quotient_le(first,last,8n,{==},below_last(first,Nat.add(start,size),ends)))

# The source cell after it, when a lane of the group inside the width needs
# it, lies among them too.
def second_holds(+width: U32,+anchor: Nat,+group: Nat,
  +need: {SplitCell.needed(U32.to_nat(width),group,Nat.mod(SplitValue.first_lane(anchor,width,group),8n)) == True{} : Bool}) ->
  {anchor_holds(width,anchor,1n+Nat.div(SplitValue.first_lane(anchor,width,group),8n)) == True{} : Bool}:
  +size = U32.to_nat(width)
  +start = Nat.mul(size,anchor)
  +lead = Nat.mul(anchor,size)
  +first = SplitValue.first_lane(anchor,width,group)
  +cell = Nat.div(first,8n)
  +product = Nat.mul(cell,8n)
  +shift = Nat.mod(first,8n)
  +rest = Nat.sub(8n,shift)
  +more = Nat.add(Nat.mul(group,8n),rest)
  +turned = RowMajor.multiply_commutative(size,anchor)
  +fits = Logic.right(Nat.is_lt(0n,shift),Nat.is_lt(more,size),need)
  +shift_le = Bounds.strict_le(shift,8n,DivisionOrder.remainder_small(first,8n,{==}))
  +whole = Equal.trans(Nat,Nat.add(8n,product),Nat.add(product,8n),Nat.add(lead,more),Order.commutative(8n,product),
    Equal.trans(Nat,Nat.add(product,8n),Nat.add(product,Nat.add(shift,rest)),Nat.add(lead,more),
      Equal.cong(Nat,Nat,value => Nat.add(product,value),8n,Nat.add(shift,rest),Equal.sym(Nat,Nat.add(shift,rest),8n,Order.restore_shift(shift,8n,shift_le))),
      Equal.trans(Nat,Nat.add(product,Nat.add(shift,rest)),Nat.add(Nat.add(product,shift),rest),Nat.add(lead,more),
        Equal.sym(Nat,Nat.add(Nat.add(product,shift),rest),Nat.add(product,Nat.add(shift,rest)),Addition.associative(product,shift,rest)),
        Equal.trans(Nat,Nat.add(Nat.add(product,shift),rest),Nat.add(first,rest),Nat.add(lead,more),
          Equal.cong(Nat,Nat,value => Nat.add(value,rest),Nat.add(product,shift),first,Equal.sym(Nat,first,Nat.add(product,shift),DivisionOrder.quotient_exact(first,8n,{==}))),
          Addition.associative(lead,Nat.mul(group,8n),rest)))))
  +ends = {%Equal.sym(Nat,Nat.add(8n,product),Nat.add(lead,more),whole) :
      {Nat.is_lt(_,Nat.add(start,size)) == True{} : Bool}
    %Equal.sym(Nat,start,lead,turned) : {Nat.is_lt(Nat.add(lead,more),Nat.add(_,size)) == True{} : Bool}
    Order.offset_lt(lead,more,size,fits) : {Nat.is_lt(Nat.mul(1n+cell,8n),Nat.add(start,size)) == True{} : Bool}}
  +last = Nat.sub(Nat.add(start,size),1n)
  +under = {%Division.exact_quotient(1n+cell,8n,{==}) : {Nat.is_le(_,Nat.div(last,8n)) == True{} : Bool}
    DivisionOrder.quotient_le(Nat.mul(1n+cell,8n),last,8n,{==},below_last(Nat.mul(1n+cell,8n),Nat.add(start,size),ends)) :
    {Nat.is_le(1n+cell,Nat.div(last,8n)) == True{} : Bool}}
  +low_start = {%Equal.sym(Nat,start,lead,turned) : {Nat.is_le(_,first) == True{} : Bool}
    Order.le_add(lead,Nat.mul(group,8n)) : {Nat.is_le(start,first) == True{} : Bool}}
  +above = Order.transitive(Nat.div(start,8n),cell,1n+cell,DivisionOrder.quotient_le(start,first,8n,{==},low_start),
    Bounds.strict_le(cell,1n+cell,Order.lt_succ(cell,cell,Order.reflexive(cell))))
  holds_of(Nat.is_le(Nat.div(start,8n),1n+cell),Nat.is_le(1n+cell,Nat.div(last,8n)),above,under)

# The readings of no links cross every target cell axis.
def placed_crossing(position: Grid.Position,others: List<&2,U32>,+groups: U32) ->
  {Tasks.crossing(Tiled.crossing_readings(Tensors.grid_of(Grid.placed(True{},position,others,groups)),Grid.placed(True{},position,others,groups),0)) == True{} : Bool}:
  match position others:
    case Grid.Innermost{} Nil{}: {==}
    case Grid.Innermost{} Con{+extent,+rest}: {==}
    case Grid.Outermost{} others: {==}

# A target cell of a piece of a component reads at the digits of its outer
# index, anchor, inner index and group a source cell the component takes:
# through the links when the component is separable, the whole source
# otherwise.
def cell_held_at(separate: Bool,found: Maybe<&2,List<&2,Tiled.Link>>,+limits: Machine.Limits,+shape: List<&2,U32>,+digits: List<&2,Nat>,+cells: List<&2,U32>,
  +others: List<&2,U32>,+groups: U32,+lane: Nat,+position: Grid.Position,+width: U32,+source_groups: U32,+groups_axis: U32,+cell: Nat,+group: Nat,+component: U32,+piece: U32,
  +local: Nat,+number: Nat,+fits: Bool,+word: U32,+below: Coordinates.DigitsBelow(others,digits),+cell_below: {Nat.is_lt(cell,U32.to_nat(source_groups)) == True{} : Bool},
  +coordinates: {Targets.cell_coordinates(Grid.placed(True{},position,others,groups),
      Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(Grid.placed(True{},position,others,groups)),piece)),local)) == target_digits(position,digits,group) : List<&2,Nat>},
  +word_exact: {U32.to_nat(word) == number : Nat},+word_below: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(cells)) == True{} : Bool},+fits_true: {fits == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},
  +target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +separate_value: {Tasks.separable([Tiled.split_readings(found,shape,Grid.placed(True{},position,others,groups),
      Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
      groups)]) == separate : Bool},
  +facts: Logic.When(separate,Held.Both<Components.RowOk(limits,Tiled.split_readings(found,shape,Grid.placed(True{},position,others,groups),
        Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
        groups),Tensors.grid_of(cells),cells),
    Held.Both<Components.RowEnds(Tiled.split_readings(found,shape,Grid.placed(True{},position,others,groups),
        Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
        groups),
      Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(found,shape,Grid.placed(True{},position,others,groups),
          Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
          groups)]),[Tiled.split_readings(found,shape,Grid.placed(True{},position,others,groups),
          Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
          groups)]),component)),
      Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(found,shape,Grid.placed(True{},position,others,groups),
          Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
          groups)]),[Tiled.split_readings(found,shape,Grid.placed(True{},position,others,groups),
          Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
          groups)]),component),Tensors.grid_of(Grid.placed(True{},position,others,groups)),Grid.placed(True{},position,others,groups),Machine.half_word(limits))>>),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(cells),cells,0,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(Tasks.components_of([Tiled.split_readings(found,shape,Grid.placed(True{},position,others,groups),
      Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),
      groups)]),Tensors.grid_of(Grid.placed(True{},position,others,groups)),Grid.placed(True{},position,others,groups),component),piece),
  +local_in: {Nat.is_lt(local,U32.to_nat(Grid.span(Tensors.grid_of(Grid.placed(True{},position,others,groups)),piece))) == True{} : Bool},
  +links_known: LinksFacts(found,shape,digits,cells,lane,position,groups_axis,cell,width,number,fits,others)) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(cells),cells,Tasks.source_box(Tasks.components_of([Tiled.split_readings(found,shape,
      Grid.placed(True{},position,others,groups),Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,
      groups_axis},Grid.is_outermost(position),groups)]),0,Tensors.grid_of(cells),cells,0,component)),Tensors.grid_of(cells)),U32.to_nat(word)):
  match separate found:
    case True{} Some{+links}:
      separate_links(limits,links,shape,digits,cells,others,groups,lane,position,width,source_groups,groups_axis,cell,group,component,piece,local,number,fits,word,below,cell_below,
        coordinates,word_exact,fits_true,room,nonempty,rank_room,target_room,target_nonempty,target_rank,separate_value,facts,blocks,member,local_in,links_known)
    case True{} None{}:
      +target = Grid.placed(True{},position,others,groups)
      +row = Tiled.crossing_readings(Tensors.grid_of(target),target,0)
      +not_crossing = Logic.right(Tasks.distinct(Tasks.read_axes(row)),Bool.not(Tasks.crossing(row)),
        Logic.left(Tasks.distinct(Tasks.read_axes(row)) && Bool.not(Tasks.crossing(row)),Tasks.separable([]),separate_value))
      Empty.absurd(Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(cells),cells,Tasks.source_box(Tasks.components_of([row]),0,Tensors.grid_of(cells),cells,0,
          component)),Tensors.grid_of(cells)),U32.to_nat(word)),
        WordEncoding.false_true({%placed_crossing(position,others,groups) : {Bool.not(_) == True{} : Bool} not_crossing : {Bool.not(True{}) == True{} : Bool}}))
    case False{} found:
      +target = Grid.placed(True{},position,others,groups)
      +source_grid = Tensors.grid_of(cells)
      +row = Tiled.split_readings(found,shape,target,Tiled.Sides{source_grid,Tensors.grid_of(target),lane,position,width,source_groups,groups_axis},Grid.is_outermost(position),groups)
      +classes = Tasks.chosen(Tasks.class_lists(Tasks.target_rank([row]),[row]),component)
      %Equal.sym(Bool,Tasks.separable([row]),False{},separate_value) :
        Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(source_grid,cells,Tasks.box_when(_,row,classes,source_grid,cells,0)),source_grid),U32.to_nat(word))
      BoxCells.cell_held(limits,cells,Grid.piece_target(),room,nonempty,rank_room,Tasks.whole_spans(cells),U32.to_nat(word),Components.whole_in(cells),
        ViewReads.whole_in(cells,U32.to_nat(word),word_below),word_below)

# ---- A word a target cell reads ----

# A target cell of a piece of a component, at its outer index, anchor,
# inner index and group, reads a source group cell: the word of that source
# cell lies in a piece the component takes.
def cell_word_held(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +position: Grid.Position,+outer: U32,+after: U32,+source_outer: U32,+component: U32,+piece: U32,+local: Nat,+above: Nat,+anchor: Nat,+inner: Nat,+group: Nat,+cell: Nat,
  +word: U32,+pre_size: {ShapeLaws.elements(pre) == U32.to_nat(outer) : Nat},+post_size: {ShapeLaws.elements(post) == U32.to_nat(after) : Nat},
  +front_size: {ShapeLaws.elements(front) == U32.to_nat(outer) : Nat},+back_size: {ShapeLaws.elements(back) == U32.to_nat(after) : Nat},
  +source_outer_exact: {U32.to_nat(source_outer) == Nat.mul(U32.to_nat(outer),U32.to_nat(after)) : Nat},
  +above_inside: {Nat.is_lt(above,U32.to_nat(outer)) == True{} : Bool},+anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},
  +inner_inside: {Nat.is_lt(inner,U32.to_nat(after)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(width))) == True{} : Bool},
  +cell_below: {Nat.is_lt(cell,U32.to_nat(Grid.groups(lanes))) == True{} : Bool},
  +index_exact: {Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece)),local) ==
    Representation.index_at(position,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(anchors)),U32.to_nat(after)),U32.to_nat(Grid.groups(width)),
      Nat.add(Nat.mul(Nat.add(Nat.mul(above,U32.to_nat(anchors)),anchor),U32.to_nat(after)),inner),group) : Nat},
  +fits_true: {anchor_holds(width,anchor,cell) == True{} : Bool},
  +word_exact: {U32.to_nat(word) ==
    Representation.index_at(position,U32.to_nat(source_outer),U32.to_nat(Grid.groups(lanes)),Nat.add(Nat.mul(above,U32.to_nat(after)),inner),cell) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+facts: Logic.When(Tasks.separable([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Held.Both<Components.RowOk(limits,Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Held.Both<Components.RowEnds(Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Machine.half_word(limits))>>),+blocks: ViewReads.BlocksIn(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,Machine.half_word(limits)),+member: Moved.Member(Tasks.target_pieces(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),component),piece),+local_in: {Nat.is_lt(local,U32.to_nat(Grid.span(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece))) == True{} : Bool}) ->
  Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Tasks.source_box(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),0,Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,component)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))),U32.to_nat(word)):
  +source_cells = Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))
  +others = List.append(&2,U32,pre,anchors <> post)
  +target = Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))
  +count = U32.to_nat(anchors)
  +outers = U32.to_nat(outer)
  +afters = U32.to_nat(after)
  +groups_count = U32.to_nat(Grid.groups(width))
  +sources = U32.to_nat(Grid.groups(lanes))
  +pair = List.append(&2,U32,front,back)
  +outer_below = {%Equal.sym(Nat,ShapeLaws.elements(pre),outers,pre_size) : {Nat.is_lt(above,_) == True{} : Bool}
    above_inside : {Nat.is_lt(above,ShapeLaws.elements(pre)) == True{} : Bool}}
  +inner_below = {%Equal.sym(Nat,ShapeLaws.elements(post),afters,post_size) : {Nat.is_lt(inner,_) == True{} : Bool}
    inner_inside : {Nat.is_lt(inner,ShapeLaws.elements(post)) == True{} : Bool}}
  +digits = others_digits(pre,above,anchor,post,inner)
  +digits_below = others_below(pre,anchors,post,above,anchor,inner,outer_below,anchor_inside,inner_below)
  +links = cell_links(front,lanes,back,pre,anchors,width,post,position,above,anchor,inner,cell,Machine.capacity(limits),Limits.capacity_fits(limits),room,nonempty,rank_room,
    outer_below,inner_below)
  +pair_size = Equal.trans(Nat,ShapeLaws.elements(pair),Nat.mul(ShapeLaws.elements(front),ShapeLaws.elements(back)),Nat.mul(outers,afters),ShapeLaws.elements_appended(front,back),
    Equal.trans(Nat,Nat.mul(ShapeLaws.elements(front),ShapeLaws.elements(back)),Nat.mul(outers,ShapeLaws.elements(back)),Nat.mul(outers,afters),
      Equal.cong(Nat,Nat,value => Nat.mul(value,ShapeLaws.elements(back)),ShapeLaws.elements(front),outers,front_size),
      Equal.cong(Nat,Nat,value => Nat.mul(outers,value),ShapeLaws.elements(back),afters,back_size)))
  +source_size = Equal.trans(Nat,U32.to_nat(source_outer),Nat.mul(outers,afters),ShapeLaws.elements(pair),source_outer_exact,
    Equal.sym(Nat,ShapeLaws.elements(pair),Nat.mul(outers,afters),pair_size))
  +number = Representation.index_at(position,ShapeLaws.elements(pair),sources,Nat.add(Nat.mul(above,ShapeLaws.elements(back)),inner),cell)
  +word_number = {%source_size : {U32.to_nat(word) == Representation.index_at(position,_,sources,Nat.add(Nat.mul(above,ShapeLaws.elements(back)),inner),cell) : Nat}
    %Equal.sym(Nat,ShapeLaws.elements(back),afters,back_size) :
      {U32.to_nat(word) == Representation.index_at(position,U32.to_nat(source_outer),sources,Nat.add(Nat.mul(above,_),inner),cell) : Nat}
    word_exact : {U32.to_nat(word) == number : Nat}}
  +target_digits_value = target_digits(position,digits,group)
  +others_size = Equal.trans(Nat,ShapeLaws.elements(others),Nat.mul(ShapeLaws.elements(pre),Nat.mul(count,ShapeLaws.elements(post))),Nat.mul(Nat.mul(outers,count),afters),
    ShapeLaws.elements_appended(pre,anchors <> post),
    Equal.trans(Nat,Nat.mul(ShapeLaws.elements(pre),Nat.mul(count,ShapeLaws.elements(post))),Nat.mul(outers,Nat.mul(count,afters)),Nat.mul(Nat.mul(outers,count),afters),
      Equal.trans(Nat,Nat.mul(ShapeLaws.elements(pre),Nat.mul(count,ShapeLaws.elements(post))),Nat.mul(outers,Nat.mul(count,ShapeLaws.elements(post))),
        Nat.mul(outers,Nat.mul(count,afters)),
        Equal.cong(Nat,Nat,value => Nat.mul(value,Nat.mul(count,ShapeLaws.elements(post))),ShapeLaws.elements(pre),outers,pre_size),
        Equal.cong(Nat,Nat,value => Nat.mul(outers,Nat.mul(count,value)),ShapeLaws.elements(post),afters,post_size)),
      Equal.sym(Nat,Nat.mul(Nat.mul(outers,count),afters),Nat.mul(outers,Nat.mul(count,afters)),RowMajor.assoc(outers,count,afters))))
  +start_local = Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(target),piece)),local)
  +placed_value = {%Equal.sym(Nat,start_local,Representation.index_at(position,Nat.mul(Nat.mul(outers,count),afters),groups_count,
        Nat.add(Nat.mul(Nat.add(Nat.mul(above,count),anchor),afters),inner),group),index_exact) :
      {ShapeLaws.digits_value(target,target_digits_value) == _ : Nat}
    %others_size : {ShapeLaws.digits_value(target,target_digits_value) ==
      Representation.index_at(position,_,groups_count,Nat.add(Nat.mul(Nat.add(Nat.mul(above,count),anchor),afters),inner),group) : Nat}
    %post_size : {ShapeLaws.digits_value(target,target_digits_value) ==
      Representation.index_at(position,ShapeLaws.elements(others),groups_count,Nat.add(Nat.mul(Nat.add(Nat.mul(above,count),anchor),_),inner),group) : Nat}
    Equal.trans(Nat,ShapeLaws.digits_value(target,target_digits_value),
      Representation.index_at(position,ShapeLaws.elements(others),groups_count,ShapeLaws.digits_value(others,digits),group),
      Representation.index_at(position,ShapeLaws.elements(others),groups_count,Nat.add(Nat.mul(Nat.add(Nat.mul(above,count),anchor),ShapeLaws.elements(post)),inner),group),
      target_number(position,others,Grid.groups(width),digits,group,digits_below),
      Equal.cong(Nat,Nat,value => Representation.index_at(position,ShapeLaws.elements(others),groups_count,value,group),ShapeLaws.digits_value(others,digits),
        Nat.add(Nat.mul(Nat.add(Nat.mul(above,count),anchor),ShapeLaws.elements(post)),inner),others_number(pre,anchors,post,above,anchor,inner,outer_below,inner_below))) :
    {ShapeLaws.digits_value(target,target_digits_value) == start_local : Nat}}
  +coordinates = {%placed_value : {Targets.cell_coordinates(target,_) == target_digits_value : List<&2,Nat>}
    BoxDigits.coordinates_value(target,target_digits_value,target_below(position,others,Grid.groups(width),digits,group,digits_below,group_inside)) :
    {Targets.cell_coordinates(target,start_local) == target_digits_value : List<&2,Nat>}}
  +outer_index = Nat.add(Nat.mul(above,afters),inner)
  +outer_index_below = {%Equal.sym(Nat,U32.to_nat(source_outer),Nat.mul(outers,afters),source_outer_exact) : {Nat.is_lt(outer_index,_) == True{} : Bool}
    RowMajor.radix(above,outers,inner,afters,above_inside,inner_inside) : {Nat.is_lt(outer_index,U32.to_nat(source_outer)) == True{} : Bool}}
  +cells_size = Equal.trans(Nat,ShapeLaws.elements(source_cells),Nat.mul(sources,ShapeLaws.elements(pair)),Nat.mul(sources,U32.to_nat(source_outer)),
    Public.placed_cells(position,pair,Grid.groups(lanes)),
    Equal.cong(Nat,Nat,value => Nat.mul(sources,value),ShapeLaws.elements(pair),U32.to_nat(source_outer),Equal.sym(Nat,U32.to_nat(source_outer),ShapeLaws.elements(pair),source_size)))
  +read = Representation.index_at(position,U32.to_nat(source_outer),sources,outer_index,cell)
  +word_below = {%Equal.sym(Nat,U32.to_nat(word),read,word_exact) : {Nat.is_lt(_,ShapeLaws.elements(source_cells)) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements(source_cells),Nat.mul(sources,U32.to_nat(source_outer)),cells_size) : {Nat.is_lt(read,_) == True{} : Bool}
    Representation.index_inside(position,U32.to_nat(source_outer),sources,outer_index,cell,outer_index_below,cell_below) :
    {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(source_cells)) == True{} : Bool}}
  cell_held_at(Tasks.separable([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),limits,List.append(&2,U32,pre,anchors <> width <> post),digits,source_cells,others,Grid.groups(width),List.length(&2,U32,front),position,width,Grid.groups(lanes),
    groups_axis_of(position,source_cells),cell,group,component,piece,local,number,anchor_holds(width,anchor,cell),word,digits_below,cell_below,coordinates,word_number,word_below,
    fits_true,room,nonempty,rank_room,target_room,target_nonempty,target_rank,{==},facts,blocks,member,local_in,links)

# ---- The source cells every target cell of a piece reads ----

# The first lane of a group inside the groups of the width lies inside it.
def group_lanes(+width: U32,+group: Nat,+exact: {U32.to_nat(Grid.groups(width)) == Ceiling.ceiling(U32.to_nat(width),8n) : Nat},
  +inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(width))) == True{} : Bool}) -> {Nat.is_lt(Nat.mul(group,8n),U32.to_nat(width)) == True{} : Bool}:
  Equal.trans(Bool,Nat.is_lt(Nat.mul(group,8n),U32.to_nat(width)),Nat.is_lt(group,Ceiling.ceiling(U32.to_nat(width),8n)),True{},
    Equal.sym(Bool,Nat.is_lt(group,Ceiling.ceiling(U32.to_nat(width),8n)),Nat.is_lt(Nat.mul(group,8n),U32.to_nat(width)),Ceiling.upper_comparison(U32.to_nat(width),8n,group,{==})),
    {%exact : {Nat.is_lt(group,_) == True{} : Bool} inside : {Nat.is_lt(group,Ceiling.ceiling(U32.to_nat(width),8n)) == True{} : Bool}})

# The next source cell a target cell needs lies in a piece its component
# takes.
def high_held(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +position: Grid.Position,+outer: U32,+after: U32,+source_outer: U32,+component: U32,+piece: U32,+local: Nat,+above: Nat,+anchor: Nat,+inner: Nat,+group: Nat,+spot: Tiled.Spot,
  +pre_size: {ShapeLaws.elements(pre) == U32.to_nat(outer) : Nat},+post_size: {ShapeLaws.elements(post) == U32.to_nat(after) : Nat},
  +front_size: {ShapeLaws.elements(front) == U32.to_nat(outer) : Nat},+back_size: {ShapeLaws.elements(back) == U32.to_nat(after) : Nat},
  +source_outer_exact: {U32.to_nat(source_outer) == Nat.mul(U32.to_nat(outer),U32.to_nat(after)) : Nat},
  +above_inside: {Nat.is_lt(above,U32.to_nat(outer)) == True{} : Bool},+anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},
  +inner_inside: {Nat.is_lt(inner,U32.to_nat(after)) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(width))) == True{} : Bool},
  +index_exact: {Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece)),local) ==
    Representation.index_at(position,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(anchors)),U32.to_nat(after)),U32.to_nat(Grid.groups(width)),
      Nat.add(Nat.mul(Nat.add(Nat.mul(above,U32.to_nat(anchors)),anchor),U32.to_nat(after)),inner),group) : Nat},
  +fits_true: {anchor_holds(width,anchor,1n+Nat.div(SplitValue.first_lane(anchor,width,group),8n)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+facts: Logic.When(Tasks.separable([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Held.Both<Components.RowOk(limits,Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Held.Both<Components.RowEnds(Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Machine.half_word(limits))>>),+blocks: ViewReads.BlocksIn(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,Machine.half_word(limits)),+member: Moved.Member(Tasks.target_pieces(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),component),piece),+local_in: {Nat.is_lt(local,U32.to_nat(Grid.span(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece))) == True{} : Bool},
  +next: SplitValue.Next<width,lanes,source_outer,position,spot,Nat.add(Nat.mul(above,U32.to_nat(after)),inner),Nat.div(SplitValue.first_lane(anchor,width,group),8n)>) ->
  Held.Both<{SplitValue.spot_further(width,Tiled.groups(lanes),spot) == True{} : Bool},Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Tasks.source_box(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),0,Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,component)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))),U32.to_nat(SplitValue.spot_high(width,spot)))>:
  SplitValue.Next{+further,+next_inside,+high_exact} = next
  Held.Both{further,cell_word_held(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,source_outer,component,piece,local,above,anchor,inner,group,
    1n+Nat.div(SplitValue.first_lane(anchor,width,group),8n),SplitValue.spot_high(width,spot),pre_size,post_size,front_size,back_size,source_outer_exact,above_inside,anchor_inside,
    inner_inside,group_inside,next_inside,index_exact,fits_true,high_exact,room,nonempty,rank_room,target_room,target_nonempty,target_rank,facts,blocks,member,local_in)}

# The next source cell a target cell needs, from the need.
def high_from(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +position: Grid.Position,+outer: U32,+after: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,
  +facts_split: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+component: U32,+piece: U32,+index: U32,+local: Nat,
  +pre_size: {ShapeLaws.elements(pre) == U32.to_nat(outer) : Nat},+post_size: {ShapeLaws.elements(post) == U32.to_nat(after) : Nat},
  +front_size: {ShapeLaws.elements(front) == U32.to_nat(outer) : Nat},+back_size: {ShapeLaws.elements(back) == U32.to_nat(after) : Nat},
  +source_outer_exact: {U32.to_nat(source_outer) == Nat.mul(U32.to_nat(outer),U32.to_nat(after)) : Nat},
  +digits: SplitValue.Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>,
  +above_inside: {Nat.is_lt(SplitValue.row_above(SplitValue.split_row(position,target_outer,width,index),anchors,after),U32.to_nat(outer)) == True{} : Bool},
  +anchor_inside: {Nat.is_lt(SplitValue.row_anchor(SplitValue.split_row(position,target_outer,width,index),anchors,after),U32.to_nat(anchors)) == True{} : Bool},
  +inner_inside: {Nat.is_lt(SplitValue.row_below(SplitValue.split_row(position,target_outer,width,index),after),U32.to_nat(after)) == True{} : Bool},
  +group_inside: {Nat.is_lt(SplitValue.split_group(position,target_outer,width,index),U32.to_nat(Grid.groups(width))) == True{} : Bool},
  +index_exact: {Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece)),local) ==
    Representation.index_at(position,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(anchors)),U32.to_nat(after)),U32.to_nat(Grid.groups(width)),
      Nat.add(Nat.mul(Nat.add(Nat.mul(SplitValue.row_above(SplitValue.split_row(position,target_outer,width,index),anchors,after),U32.to_nat(anchors)),
        SplitValue.row_anchor(SplitValue.split_row(position,target_outer,width,index),anchors,after)),U32.to_nat(after)),
        SplitValue.row_below(SplitValue.split_row(position,target_outer,width,index),after)),SplitValue.split_group(position,target_outer,width,index)) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+facts: Logic.When(Tasks.separable([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Held.Both<Components.RowOk(limits,Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Held.Both<Components.RowEnds(Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Machine.half_word(limits))>>),+blocks: ViewReads.BlocksIn(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,Machine.half_word(limits)),+member: Moved.Member(Tasks.target_pieces(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),component),piece),+local_in: {Nat.is_lt(local,U32.to_nat(Grid.span(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece))) == True{} : Bool},
  +next_when: Logic.When(SplitValue.spot_needed(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index)),
    SplitValue.Next<width,lanes,source_outer,position,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index),
      SplitValue.source_row(SplitValue.split_row(position,target_outer,width,index),anchors,after),
      Nat.div(SplitValue.first_lane(SplitValue.row_anchor(SplitValue.split_row(position,target_outer,width,index),anchors,after),width,
        SplitValue.split_group(position,target_outer,width,index)),8n)>),
  +need: {SplitValue.spot_needed(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index)) == True{} : Bool}) ->
  Held.Both<{SplitValue.spot_further(width,Tiled.groups(lanes),Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index)) == True{} : Bool},
    Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Tasks.source_box(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),0,Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,component)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))),U32.to_nat(SplitValue.spot_high(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index))))>:
  +row = SplitValue.split_row(position,target_outer,width,index)
  +above = SplitValue.row_above(row,anchors,after)
  +anchor = SplitValue.row_anchor(row,anchors,after)
  +inner = SplitValue.row_below(row,after)
  +group = SplitValue.split_group(position,target_outer,width,index)
  +cell = Nat.div(SplitValue.first_lane(anchor,width,group),8n)
  +spot = Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index)
  high_held(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,source_outer,component,piece,local,above,anchor,inner,group,spot,pre_size,post_size,front_size,
    back_size,source_outer_exact,above_inside,anchor_inside,inner_inside,group_inside,index_exact,
    second_holds(width,anchor,group,SplitValue.split_needed(limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,facts_split,index,digits,
      need)),room,nonempty,rank_room,target_room,target_nonempty,target_rank,facts,blocks,member,local_in,
    Logic.required(SplitValue.Next<width,lanes,source_outer,position,spot,Nat.add(Nat.mul(above,U32.to_nat(after)),inner),cell>,SplitValue.spot_needed(width,spot),next_when,need))

# A target cell of a piece of a component reads its first source cell, and
# the next one when it needs it, in pieces the component takes.
def cell_held_with(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +position: Grid.Position,+outer: U32,+after: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,
  +facts_split: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+component: U32,+piece: U32,+index: U32,+local: U32,
  +pre_size: {ShapeLaws.elements(pre) == U32.to_nat(outer) : Nat},+post_size: {ShapeLaws.elements(post) == U32.to_nat(after) : Nat},
  +front_size: {ShapeLaws.elements(front) == U32.to_nat(outer) : Nat},+back_size: {ShapeLaws.elements(back) == U32.to_nat(after) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+facts: Logic.When(Tasks.separable([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Held.Both<Components.RowOk(limits,Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Held.Both<Components.RowEnds(Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Machine.half_word(limits))>>),+blocks: ViewReads.BlocksIn(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,Machine.half_word(limits)),+member: Moved.Member(Tasks.target_pieces(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),component),piece),+local_in: {Nat.is_lt(U32.to_nat(local),U32.to_nat(Grid.span(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece))) == True{} : Bool},
  +index_value: {Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece)),U32.to_nat(local)) == U32.to_nat(index) : Nat},
  +digits: SplitValue.Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>,
  +words: SplitValue.Words<width,lanes,source_outer,position,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index),
    SplitValue.source_row(SplitValue.split_row(position,target_outer,width,index),anchors,after),
    Nat.div(SplitValue.first_lane(SplitValue.row_anchor(SplitValue.split_row(position,target_outer,width,index),anchors,after),width,
      SplitValue.split_group(position,target_outer,width,index)),8n)>) ->
  SplitGather.CellHeld<Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Tasks.source_box(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),0,Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,component)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))),width,Tiled.groups(lanes),Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index)>:
  SplitValue.Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,TiledValues.Groups{+width_exact,+width_fewer,+width_fit},
    +lane_groups,+outer_room,+anchors_room,+after_room,+groups_room,+source_room} = facts_split
  SplitValue.Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  SplitValue.Words{+read_exact,+outer_inside,+cell_inside,+next} = words
  +row = SplitValue.split_row(position,target_outer,width,index)
  +above = SplitValue.row_above(row,anchors,after)
  +anchor = SplitValue.row_anchor(row,anchors,after)
  +inner = SplitValue.row_below(row,after)
  +group = SplitValue.split_group(position,target_outer,width,index)
  +cell = Nat.div(SplitValue.first_lane(anchor,width,group),8n)
  +spot = Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index)
  +rebuilt = {SplitValue.Facts{target_outer_exact,target_after_exact,source_outer_exact,source_after_exact,lanes_exact,TiledValues.Groups{width_exact,width_fewer,width_fit},lane_groups,
    outer_room,anchors_room,after_room,groups_room,source_room} : SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>}
  +again = {SplitValue.Digits{above_inside,anchor_inside,below_inside,group_inside,above_exact,placed,source_inside,source_positive} :
    SplitValue.Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>}
  +index_exact = Equal.trans(Nat,Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece)),U32.to_nat(local)),U32.to_nat(index),
    Representation.index_at(position,Nat.mul(Nat.mul(U32.to_nat(outer),U32.to_nat(anchors)),U32.to_nat(after)),U32.to_nat(Grid.groups(width)),
      Nat.add(Nat.mul(Nat.add(Nat.mul(above,U32.to_nat(anchors)),anchor),U32.to_nat(after)),inner),group),index_value,placed)
  +low = cell_word_held(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,source_outer,component,piece,U32.to_nat(local),above,anchor,inner,group,cell,
    SplitValue.spot_read(width,spot),pre_size,post_size,front_size,back_size,source_outer_exact,above_inside,anchor_inside,below_inside,group_inside,cell_inside,index_exact,
    first_holds(width,anchor,group,group_lanes(width,group,width_exact,group_inside)),read_exact,room,nonempty,rank_room,target_room,target_nonempty,target_rank,facts,blocks,member,local_in)
  +high = Logic.when_enabled(Held.Both<{SplitValue.spot_further(width,Tiled.groups(lanes),spot) == True{} : Bool},Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Tasks.source_box(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),0,Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,component)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))),U32.to_nat(SplitValue.spot_high(width,spot)))>,
    SplitValue.spot_needed(width,spot),
    need => high_from(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,target_outer,target_after,source_outer,source_after,rebuilt,component,piece,index,
      U32.to_nat(local),pre_size,post_size,front_size,back_size,source_outer_exact,again,above_inside,anchor_inside,below_inside,group_inside,index_exact,room,nonempty,rank_room,target_room,target_nonempty,target_rank,facts,blocks,member,
      local_in,next,need))
  SplitGather.CellHeld{low,high}

# A target cell of a piece of a component, by its place in the piece.
def split_cell_at(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +position: Grid.Position,+outer: U32,+after: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,
  +facts_split: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+component: U32,+piece: U32,
  +pre_size: {ShapeLaws.elements(pre) == U32.to_nat(outer) : Nat},+post_size: {ShapeLaws.elements(post) == U32.to_nat(after) : Nat},
  +front_size: {ShapeLaws.elements(front) == U32.to_nat(outer) : Nat},+back_size: {ShapeLaws.elements(back) == U32.to_nat(after) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+facts: Logic.When(Tasks.separable([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Held.Both<Components.RowOk(limits,Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Held.Both<Components.RowEnds(Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Machine.half_word(limits))>>),+blocks: ViewReads.BlocksIn(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,Machine.half_word(limits)),+member: Moved.Member(Tasks.target_pieces(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),component),piece),
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece)),U32.to_nat(Grid.span(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece))),
    Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool},
  +local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(Grid.span(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece))) == True{} : Bool}) ->
  SplitGather.CellHeld<Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Tasks.source_box(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),0,Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,component)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))),width,Tiled.groups(lanes),
    Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),(Grid.start(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece) + local : U32))>:
  +first = Grid.start(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece)
  +count = Grid.span(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece)
  +index = (first + local : U32)
  +cells = Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))
  +capacity = U32.to_nat(Machine.capacity(limits))
  +sum = Nat.add(U32.to_nat(first),U32.to_nat(local))
  +reached = Order.lt_le(sum,Nat.add(U32.to_nat(first),U32.to_nat(count)),cells,Order.offset_lt(U32.to_nat(first),U32.to_nat(local),U32.to_nat(count),below),piece_room)
  +index_value = Equal.sym(Nat,U32.to_nat(index),sum,
    Arithmetic.sum_exact(Machine.capacity(limits),Limits.capacity_fits(limits),first,local,U32.to_nat(first),U32.to_nat(local),{==},{==},
      Order.transitive(sum,cells,capacity,Bounds.strict_le(sum,cells,reached),cells_room)))
  +inside = SplitLaws.target_inside(limits,width,target_outer,first,count,cells_room,piece_room,local,below)
  +digits = SplitValue.split_digits(limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,facts_split,index,inside)
  cell_held_with(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,target_outer,target_after,source_outer,source_after,facts_split,component,piece,index,local,
    pre_size,post_size,front_size,back_size,room,nonempty,rank_room,target_room,target_nonempty,target_rank,facts,blocks,member,below,index_value,digits,
    SplitValue.split_words(limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,facts_split,index,digits))

# Every target cell of a piece of a component reads its source cells in
# pieces the component takes.
def split_cells_held(+limits: Machine.Limits,+front: List<&2,U32>,+lanes: U32,+back: List<&2,U32>,+pre: List<&2,U32>,+anchors: U32,+width: U32,+post: List<&2,U32>,
  +position: Grid.Position,+outer: U32,+after: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,
  +facts_split: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+component: U32,+piece: U32,
  +pre_size: {ShapeLaws.elements(pre) == U32.to_nat(outer) : Nat},+post_size: {ShapeLaws.elements(post) == U32.to_nat(after) : Nat},
  +front_size: {ShapeLaws.elements(front) == U32.to_nat(outer) : Nat},+back_size: {ShapeLaws.elements(back) == U32.to_nat(after) : Nat},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)))) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+facts: Logic.When(Tasks.separable([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Held.Both<Components.RowOk(limits,Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Held.Both<Components.RowEnds(Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width)),Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component)),Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),[Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),component),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Machine.half_word(limits))>>),+blocks: ViewReads.BlocksIn(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,Machine.half_word(limits)),+member: Moved.Member(Tasks.target_pieces(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),component),piece),
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(Grid.start(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece)),U32.to_nat(Grid.span(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece))),
    Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool}) ->
  SplitGather.CellsHeld(Moved.taken_spans(Tasks.box_pieces(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),Tasks.source_box(Tasks.components_of([Tiled.split_readings(Tiled.joined_links(Tiled.pairing(pre,front,0n),Tiled.pairing(post,back,Nat.add(List.length(&2,U32,front),1n))),List.append(&2,U32,pre,anchors <> width <> post),Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width)),Tiled.Sides{Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),List.length(&2,U32,front),position,width,Grid.groups(lanes),groups_axis_of(position,Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))},Grid.is_outermost(position),Grid.groups(width))]),0,Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes))),Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)),0,component)),Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,front,back),Grid.groups(lanes)))),Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position},Grid.start(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece),
    Grid.span(Tensors.grid_of(Grid.placed(True{},position,List.append(&2,U32,pre,anchors <> post),Grid.groups(width))),piece)):
  local => below =>
    split_cell_at(limits,front,lanes,back,pre,anchors,width,post,position,outer,after,target_outer,target_after,source_outer,source_after,facts_split,component,piece,pre_size,
      post_size,front_size,back_size,room,nonempty,rank_room,target_room,target_nonempty,target_rank,facts,blocks,member,cells_room,piece_room,local,below)
