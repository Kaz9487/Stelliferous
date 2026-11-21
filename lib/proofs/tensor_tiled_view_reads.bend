# The cells a view of tiled values reads when it keeps their lanes. Every
# target axis of such a view repeats one index, reads the lane axis one index
# a step, or steps one source axis, and no source index runs past its
# extent. The view then reads, at every target digit, the source digits that
# add the steps of every target axis to the first ones; the view of the cells
# reads the cell of the same digits but the lane axis, and of the group of
# its lane digit.
import Base
import ../tensor_view.bend as Views
import ../tensor_grid.bend as Grid
import ../kernel_gemm.bend as KernelGemm
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_cell_word.bend as CellWords
import ./tensor_box_digits.bend as Digits
import ./tensor_view_reads.bend as ViewReads
import ./tensor_mirror_address.bend as MirrorAddress
import ./tensor_tiled_public.bend as Public
import ./tensor_cell_room.bend as Rooms
import ./nat_division.bend as Division
import ./nat_order.bend as Order
import ./tensor_grid_pieces.bend as GridLaws
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ../tensor_tasks.bend as Tasks
import ../tensor_shape.bend as Shapes
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_view.bend as TiledView
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./boolean_logic.bend as Boolean
import ./u32_comparison.bend as Comparison
import ./u32_division.bend as U32Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_components.bend as Components
import ./boolean_logic.bend as Logic
import ./binary_radix.bend as Radix
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_tiled_representation.bend as Representation
import ./tensor_relane_geometry.bend as Geometry
import ./tensor_relane_values.bend as Values
import ./tensor_tiled_set_slice.bend as SetSlice
import ./word_encoding.bend as WordEncoding
import ./tensor_view_metadata_refinement.bend as MetadataLaws
import ./u32_full_range.bend as FullRange
import ./tensor_view_bounds.bend as ViewBounds
import ./tensor_count_completion.bend as CountCompletion
import ../tensor.bend as Tensors
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_target_cells.bend as Targets
import ../kernel_tile.bend as Tile
import ./kernel_tile_refinement.bend as TileLaws
import ./tensor_view_part.bend as ViewPart
import ./tensor_grid_box.bend as Box
import ./tensor_view_gather.bend as ViewGather
import ../default_machine_limits.bend as Default
import ./tensor_global_shape_count.bend as Global
import ./tensor_storage_count.bend as StorageCount
import ./tensor_view_moved.bend as ViewMoved
import ./tensor_pieces_content.bend as Content
import ../tensor_tiled_cells.bend as Cells
import ./tensor_count_refinement.bend as Count

# ---- Digits added axis by axis ----

def zeros(rank: Nat) -> List<&2,Nat>:
  match rank:
    case 0n: []
    case 1n+rest: 0n <> zeros(rest)

# An amount at one axis and zero at the others.
def unit_digits(rank: Nat,axis: Nat,+amount: Nat) -> List<&2,Nat>:
  match rank axis:
    case 0n axis: []
    case 1n+rest 0n: amount <> zeros(rest)
    case 1n+rest 1n+next: 0n <> unit_digits(rest,next,amount)

def zeros_length(rank: Nat) -> {List.length(&2,Nat,zeros(rank)) == rank : Nat}:
  match rank:
    case 0n: {==}
    case 1n+ +rest: Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Nat,zeros(rest)),rest,zeros_length(rest))

def unit_length(rank: Nat,axis: Nat,+amount: Nat) -> {List.length(&2,Nat,unit_digits(rank,axis,amount)) == rank : Nat}:
  match rank axis:
    case 0n axis: {==}
    case 1n+ +rest 0n: Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Nat,zeros(rest)),rest,zeros_length(rest))
    case 1n+ +rest 1n+ +next: Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Nat,unit_digits(rest,next,amount)),rest,unit_length(rest,next,amount))

def successor_zero(+value: Nat,+same: {1n+value == 0n : Nat}) -> Empty:
  Digits.zero_successor(value,Equal.sym(Nat,1n+value,0n,same))

def length_tail(+left: Nat,+right: Nat,+same: {1n+left == 1n+right : Nat}) -> {left == right : Nat}:
  Equal.cong(Nat,Nat,value => Digits.before(value),1n+left,1n+right,same)

# The number of digits added axis by axis is the sum of their numbers.
def digits_added(extents: List<&2,U32>,left: List<&2,Nat>,right: List<&2,Nat>,
  +left_length: {List.length(&2,Nat,left) == List.length(&2,U32,extents) : Nat},
  +right_length: {List.length(&2,Nat,right) == List.length(&2,U32,extents) : Nat}) ->
  {ShapeLaws.digits_value(extents,Digits.added(left,right)) == Nat.add(ShapeLaws.digits_value(extents,left),ShapeLaws.digits_value(extents,right)) : Nat}:
  match extents left right:
    case Con{+extent,+rest} Con{+digit,+more} Con{+other,+others}:
      +inner = ShapeLaws.elements(rest)
      +tail = digits_added(rest,more,others,length_tail(List.length(&2,Nat,more),List.length(&2,U32,rest),left_length),
        length_tail(List.length(&2,Nat,others),List.length(&2,U32,rest),right_length))
      %Equal.sym(Nat,ShapeLaws.digits_value(rest,Digits.added(more,others)),Nat.add(ShapeLaws.digits_value(rest,more),ShapeLaws.digits_value(rest,others)),tail) :
        {Nat.add(Nat.mul(Nat.add(digit,other),inner),_) ==
          Nat.add(Nat.add(Nat.mul(digit,inner),ShapeLaws.digits_value(rest,more)),Nat.add(Nat.mul(other,inner),ShapeLaws.digits_value(rest,others))) : Nat}
      %Equal.sym(Nat,Nat.mul(Nat.add(digit,other),inner),Nat.add(Nat.mul(digit,inner),Nat.mul(other,inner)),RowMajor.distrib(digit,other,inner)) :
        {Nat.add(_,Nat.add(ShapeLaws.digits_value(rest,more),ShapeLaws.digits_value(rest,others))) ==
          Nat.add(Nat.add(Nat.mul(digit,inner),ShapeLaws.digits_value(rest,more)),Nat.add(Nat.mul(other,inner),ShapeLaws.digits_value(rest,others))) : Nat}
      MirrorAddress.interchange(Nat.mul(digit,inner),Nat.mul(other,inner),ShapeLaws.digits_value(rest,more),ShapeLaws.digits_value(rest,others))
    case Nil{} Nil{} Nil{}: {==}
    case Nil{} Con{+digit,+more} +right:
      Empty.absurd({ShapeLaws.digits_value(Nil{},Digits.added(Con{digit,more},right)) == Nat.add(ShapeLaws.digits_value(Nil{},Con{digit,more}),ShapeLaws.digits_value(Nil{},right)) : Nat},
        successor_zero(List.length(&2,Nat,more),left_length))
    case Nil{} Nil{} Con{+other,+others}:
      Empty.absurd({ShapeLaws.digits_value(Nil{},Digits.added(Nil{},Con{other,others})) == Nat.add(ShapeLaws.digits_value(Nil{},Nil{}),ShapeLaws.digits_value(Nil{},Con{other,others})) : Nat},
        successor_zero(List.length(&2,Nat,others),right_length))
    case Con{+extent,+rest} Nil{} +right:
      Empty.absurd({ShapeLaws.digits_value(Con{extent,rest},Digits.added(Nil{},right)) == Nat.add(ShapeLaws.digits_value(Con{extent,rest},Nil{}),ShapeLaws.digits_value(Con{extent,rest},right)) : Nat},
        Digits.zero_successor(List.length(&2,U32,rest),left_length))
    case Con{+extent,+rest} Con{+digit,+more} Nil{}:
      Empty.absurd({ShapeLaws.digits_value(Con{extent,rest},Digits.added(Con{digit,more},Nil{})) == Nat.add(ShapeLaws.digits_value(Con{extent,rest},Con{digit,more}),ShapeLaws.digits_value(Con{extent,rest},Nil{})) : Nat},
        Digits.zero_successor(List.length(&2,U32,rest),right_length))

# Zero digits number nothing.
def zeros_value(extents: List<&2,U32>,rank: Nat) -> {ShapeLaws.digits_value(extents,zeros(rank)) == 0n : Nat}:
  match extents rank:
    case Con{+extent,+rest} 1n+ +more: zeros_value(rest,more)
    case Nil{} rank: {==}
    case Con{+extent,+rest} 0n: {==}

# An amount at one axis numbers the amount times the cells after the axis.
def unit_value(extents: List<&2,U32>,axis: Nat,+amount: Nat,+inside: {Nat.is_lt(axis,List.length(&2,U32,extents)) == True{} : Bool}) ->
  {ShapeLaws.digits_value(extents,unit_digits(List.length(&2,U32,extents),axis,amount)) == Nat.mul(amount,ShapeLaws.elements(Public.lane_after(extents,axis))) : Nat}:
  match extents axis:
    case Con{+extent,+rest} 0n:
      %Equal.sym(Nat,ShapeLaws.digits_value(rest,zeros(List.length(&2,U32,rest))),0n,zeros_value(rest,List.length(&2,U32,rest))) :
        {Nat.add(Nat.mul(amount,ShapeLaws.elements(rest)),_) == Nat.mul(amount,ShapeLaws.elements(rest)) : Nat}
      Addition.zero_right(Nat.mul(amount,ShapeLaws.elements(rest)))
    case Con{+extent,+rest} 1n+ +next: unit_value(rest,next,amount,inside)
    case Nil{} +axis:
      Empty.absurd({ShapeLaws.digits_value(Nil{},unit_digits(0n,axis,amount)) == Nat.mul(amount,ShapeLaws.elements(Public.lane_after(Nil{},axis))) : Nat},Order.below_zero(axis,inside))

# ---- The sum of a view at digits ----

# The view's steps at the digits of its axes, every flagged digit reflected.
def digit_sum(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,digits: List<&2,Nat>) -> Nat:
  match axes digits:
    case Con{Views.Axis{+extent,+stride},rest} Con{+digit,more}:
      Nat.add(Nat.mul(MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),digit),U32.to_nat(stride)),digit_sum(rest,Views.later_flags(flags),more))
    case axes digits: 0n

# The sum a view gives a target index is its sum at the digits of the index.
def view_sum_digits(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,digits: List<&2,Nat>,+below: Coordinates.DigitsBelow(Views.axis_extents(axes),digits)) ->
  {ViewReads.view_sum(axes,flags,ShapeLaws.digits_value(Views.axis_extents(axes),digits)) == digit_sum(axes,flags,digits) : Nat}:
  match axes digits:
    case Con{Views.Axis{+extent,+stride},+rest} Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      +inner = ShapeLaws.elements(Views.axis_extents(rest))
      +tail = ShapeLaws.digits_value(Views.axis_extents(rest),more)
      %Equal.sym(Nat & Nat,Nat.divmod(Nat.add(Nat.mul(digit,inner),tail),inner),(digit,tail),
          Division.quotient_remainder(digit,tail,inner,Coordinates.digits_positive(Views.axis_extents(rest),more,later),Coordinates.digits_below(Views.axis_extents(rest),more,later))) :
        {Nat.add(Nat.mul(MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),Pair.fst(Nat,Nat,_)),U32.to_nat(stride)),
            ViewReads.view_sum(rest,Views.later_flags(flags),Pair.snd(Nat,Nat,_))) ==
          Nat.add(Nat.mul(MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),digit),U32.to_nat(stride)),digit_sum(rest,Views.later_flags(flags),more)) : Nat}
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),digit),U32.to_nat(stride)),value),
        ViewReads.view_sum(rest,Views.later_flags(flags),tail),digit_sum(rest,Views.later_flags(flags),more),view_sum_digits(rest,Views.later_flags(flags),more,later))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+digit,+more}: Empty.absurd({ViewReads.view_sum(Nil{},flags,ShapeLaws.digits_value(Nil{},Con{digit,more})) == digit_sum(Nil{},flags,Con{digit,more}) : Nat},below)
    case Con{Views.Axis{+extent,+stride},+rest} Nil{}:
      Empty.absurd({ViewReads.view_sum(Con{Views.Axis{extent,stride},rest},flags,ShapeLaws.digits_value(Views.axis_extents(Con{Views.Axis{extent,stride},rest}),Nil{})) ==
        digit_sum(Con{Views.Axis{extent,stride},rest},flags,Nil{}) : Nat},below)

# ---- The source axis a stride steps ----

# The lanes of values whose lane axis lies inside them or just past them.
def lanes_exact(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+inside: {Nat.is_le(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {TiledView.lanes_of(shape,axis,position) == TiledView.Lanes{axis,Grid.lanes_at(shape,axis),Tiled.product(Public.lane_after(shape,axis)),
    (Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32),position} : TiledView.Lanes}:
  %Equal.sym(Maybe<&2,Tiled.Around>,Tiled.around(shape,axis,1),Some{Tiled.Around{Public.lane_prefix(shape,axis,1),Grid.lanes_at(shape,axis),Public.lane_after(shape,axis)}},
      Public.located(shape,axis,1,inside)) :
    {TiledView.lanes_around(_,axis,position) == TiledView.Lanes{axis,Grid.lanes_at(shape,axis),Tiled.product(Public.lane_after(shape,axis)),
      (Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32),position} : TiledView.Lanes}
  {==}

def lane_axis_of(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+inside: {Nat.is_le(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {TiledView.lane_axis(TiledView.lanes_of(shape,axis,position)) == axis : Nat}:
  %Equal.sym(TiledView.Lanes,TiledView.lanes_of(shape,axis,position),TiledView.Lanes{axis,Grid.lanes_at(shape,axis),Tiled.product(Public.lane_after(shape,axis)),
      (Public.lane_prefix(shape,axis,1) * Tiled.product(Public.lane_after(shape,axis)) : U32),position},lanes_exact(shape,axis,position,inside)) :
    {TiledView.lane_axis(_) == axis : Nat}
  {==}

# The axis a stride steps, walked from start: its place among the extents,
# and the stride is the step times the cells after it.
type StepAt<-extents: List<&2,U32>,-stride: U32,-start: Nat,-axis: U32,-step: U32> is Data:
  StepAt{place: Nat,
    axis_exact: {U32.to_nat(axis) == Nat.add(start,place) : Nat},
    inside: {Nat.is_lt(place,List.length(&2,U32,extents)) == True{} : Bool},
    stride_exact: {U32.to_nat(stride) == Nat.mul(U32.to_nat(step),ShapeLaws.elements(Public.lane_after(extents,place))) : Nat}}

def Stepping(found: Tasks.Stepped,+extents: List<&2,U32>,+stride: U32,+start: Nat) -> Data:
  match found:
    case Tasks.Unstepped{}: Unit
    case Tasks.Stepped{+axis,+step,+extent,+first}: StepAt<extents,stride,start,axis,step>

def stepped_later(found: Tasks.Stepped,+extent: U32,+rest: List<&2,U32>,+stride: U32,+start: Nat,+facts: Stepping(found,rest,stride,1n+start)) ->
  Stepping(found,Con{extent,rest},stride,start):
  match found:
    case Tasks.Unstepped{}: Unit{}
    case Tasks.Stepped{+axis,+step,+source_extent,+first}:
      StepAt{+place,+axis_exact,+inside,+stride_exact} = facts
      StepAt{1n+place,Equal.trans(Nat,U32.to_nat(axis),Nat.add(1n+start,place),Nat.add(start,1n+place),axis_exact,
          Equal.sym(Nat,Nat.add(start,1n+place),1n+Nat.add(start,place),Addition.successor_right(start,place))),
        inside,stride_exact}

# The cells of extents without a zero extent are at least those of the
# extents after the first.
def rest_cells(+extent: U32,+rest: List<&2,U32>,+nonzero: {Shapes.has_zero(Con{extent,rest}) == False{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(rest),ShapeLaws.elements(Con{extent,rest})) == True{} : Bool}:
  +head = U32.to_nat(extent)
  +positive = Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(rest),nonzero))
  %RowMajor.multiply_commutative(ShapeLaws.elements(rest),head) : {Nat.is_le(ShapeLaws.elements(rest),_) == True{} : Bool}
  RowMajor.product_covers(ShapeLaws.elements(rest),head,positive)

# The stride of the first axis of more than one index whose cells after it
# divide the stride is the step times those cells.
def stepped_first(+extent: U32,+rest: List<&2,U32>,+first: U32,+stride: U32,+at: U32,+start: Nat,+limits: Machine.Limits,
  +exact: {U32.to_nat(at) == start : Nat},+rest_room: {Nat.is_le(ShapeLaws.elements(rest),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +classified: {U32.is_lt(1,extent) && U32.is_lt(0,Grid.product(rest)) && U32.is_le(Grid.product(rest),stride) && U32.is_zero((stride % U32.max(Grid.product(rest),1) : U32)) == True{} : Bool}) ->
  StepAt<Con{extent,rest},stride,start,at,(stride / U32.max(Grid.product(rest),1) : U32)>:
  +own = Grid.product(rest)
  +cells = ShapeLaws.elements(rest)
  +first_three = Boolean.left(U32.is_lt(1,extent) && U32.is_lt(0,own) && U32.is_le(own,stride),U32.is_zero((stride % U32.max(own,1) : U32)),classified)
  +divides_kept = Boolean.right(U32.is_lt(1,extent) && U32.is_lt(0,own) && U32.is_le(own,stride),U32.is_zero((stride % U32.max(own,1) : U32)),classified)
  +own_word_positive = Boolean.right(U32.is_lt(1,extent),U32.is_lt(0,own),Boolean.left(U32.is_lt(1,extent) && U32.is_lt(0,own),U32.is_le(own,stride),first_three))
  +own_exact = Equal.trans(Nat,U32.to_nat(own),U32.to_nat(U32.from_nat(cells)),cells,
    Equal.cong(U32,Nat,word => U32.to_nat(word),own,U32.from_nat(cells),Equal.sym(U32,U32.from_nat(cells),own,Digits.product_from(rest))),
    Bounds.roundtrip(cells,Machine.capacity(limits),Limits.capacity_fits(limits),rest_room))
  +own_positive = Equal.trans(Bool,Nat.is_lt(0n,U32.to_nat(own)),U32.is_lt(0,own),True{},Equal.sym(Bool,U32.is_lt(0,own),Nat.is_lt(0n,U32.to_nat(own)),Comparison.lt(0,own)),own_word_positive)
  +positive = {%own_exact : {Nat.is_lt(0n,_) == True{} : Bool} own_positive : {Nat.is_lt(0n,cells) == True{} : Bool}}
  +kept = GridLaws.maximum_kept(own,1,Order.positive_one(U32.to_nat(own),own_positive))
  +divides = {%kept : {U32.is_zero((stride % _ : U32)) == True{} : Bool} divides_kept : {U32.is_zero((stride % own : U32)) == True{} : Bool}}
  +bound = Limits.double_bound(limits,cells,rest_room)
  +own_bound = {%Equal.sym(Nat,U32.to_nat(own),cells,own_exact) : {Nat.is_le(Nat.double(_),U32.to_nat(Machine.half_word(limits))) == True{} : Bool} bound :
    {Nat.is_le(Nat.double(U32.to_nat(own)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
  +quotient = U32Division.quotient_exact(Machine.half_word(limits),Limits.half_word_fits(limits),stride,own,U32.to_nat(stride),cells,{==},own_exact,bound,positive)
  +remainder = {%own_exact : {U32.to_nat((stride % own : U32)) == Nat.mod(U32.to_nat(stride),_) : Nat}
    U32Division.modulo(stride,own,Machine.half_word(limits),Limits.half_word_fits(limits),own_bound,own_positive) :
    {U32.to_nat((stride % own : U32)) == Nat.mod(U32.to_nat(stride),cells) : Nat}}
  +zero = Equal.trans(Nat,Nat.mod(U32.to_nat(stride),cells),U32.to_nat((stride % own : U32)),0n,Equal.sym(Nat,U32.to_nat((stride % own : U32)),Nat.mod(U32.to_nat(stride),cells),remainder),
    Equal.cong(U32,Nat,word => U32.to_nat(word),(stride % own : U32),0,Arithmetic.word_zero((stride % own : U32),divides)))
  +split = DivisionOrder.quotient_exact(U32.to_nat(stride),cells,positive)
  %Equal.sym(U32,U32.max(own,1),own,kept) : StepAt<Con{extent,rest},stride,start,at,(stride / _ : U32)>
  StepAt{0n,Equal.trans(Nat,U32.to_nat(at),start,Nat.add(start,0n),exact,Equal.sym(Nat,Nat.add(start,0n),start,Addition.zero_right(start))),{==},
    %Equal.sym(Nat,U32.to_nat((stride / own : U32)),Nat.div(U32.to_nat(stride),cells),quotient) : {U32.to_nat(stride) == Nat.mul(_,cells) : Nat}
    Equal.trans(Nat,U32.to_nat(stride),Nat.add(Nat.mul(Nat.div(U32.to_nat(stride),cells),cells),0n),Nat.mul(Nat.div(U32.to_nat(stride),cells),cells),
      %zero : {U32.to_nat(stride) == Nat.add(Nat.mul(Nat.div(U32.to_nat(stride),cells),cells),_) : Nat} split,
      Addition.zero_right(Nat.mul(Nat.div(U32.to_nat(stride),cells),cells)))}

def stepped_choice(chosen: Bool,+extent: U32,+rest: List<&2,U32>,+first: U32,+more: List<&2,U32>,+stride: U32,+at: U32,+start: Nat,+limits: Machine.Limits,
  +exact: {U32.to_nat(at) == start : Nat},+rest_room: {Nat.is_le(ShapeLaws.elements(rest),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +later: Stepping(Tasks.stepped_axis(rest,more,stride,(at + 1 : U32)),rest,stride,1n+start),
  +classified: {Tasks.head_steps(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},stride) == chosen : Bool}) ->
  Stepping(Tasks.stepped_on(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},Con{first,more},chosen,stride,at),Con{extent,rest},stride,start):
  match chosen:
    case True{}: stepped_first(extent,rest,first,stride,at,start,limits,exact,rest_room,classified)
    case False{}: stepped_later(Tasks.stepped_axis(rest,more,stride,(at + 1 : U32)),extent,rest,stride,start,later)

# What the axis a stride steps tells, walked from start.
def stepped_at(extents: List<&2,U32>,coordinates: List<&2,U32>,+stride: U32,+at: U32,+start: Nat,+limits: Machine.Limits,
  +exact: {U32.to_nat(at) == start : Nat},+room: {Nat.is_le(Nat.add(start,List.length(&2,U32,extents)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonzero: {Shapes.has_zero(extents) == False{} : Bool}) ->
  Stepping(Tasks.stepped_axis(extents,coordinates,stride,at),extents,stride,start):
  match extents coordinates:
    case Con{+extent,+rest} Con{+first,+more}:
      +cap = U32.to_nat(Machine.capacity(limits))
      +rest_room = Order.transitive(ShapeLaws.elements(rest),ShapeLaws.elements(Con{extent,rest}),cap,rest_cells(extent,rest,nonzero),cells_room)
      +later_room = Order.advance_room(start,List.length(&2,U32,rest),cap,room)
      +next = Components.next_word(at,start,Machine.capacity(limits),Limits.capacity_fits(limits),exact,
        Order.transitive(1n+start,Nat.add(1n+start,List.length(&2,U32,rest)),cap,Order.le_add(1n+start,List.length(&2,U32,rest)),later_room))
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        Stepping(Tasks.stepped_on(Con{extent,rest},_,Con{first,more},Tasks.head_steps(Con{extent,rest},_,stride),stride,at),Con{extent,rest},stride,start)
      stepped_choice(Tasks.head_steps(Con{extent,rest},Con{Grid.product(rest),Grid.strides(rest)},stride),
        extent,rest,first,more,stride,at,start,limits,exact,rest_room,
        stepped_at(rest,more,stride,(at + 1 : U32),1n+start,limits,next,later_room,rest_room,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonzero)),{==})
    case Con{+extent,+rest} Nil{}:
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,rest}),Con{Grid.product(rest),Grid.strides(rest)},GridLaws.strides_con(extent,rest)) :
        Stepping(Tasks.stepped_on(Con{extent,rest},_,Nil{},Tasks.head_steps(Con{extent,rest},_,stride),stride,at),Con{extent,rest},stride,start)
      Unit{}
    case Nil{} +coordinates: Unit{}

# ---- What a target axis reads ----

# What a target axis of an extent and a stride reads of the values: it
# repeats one index with no stride or a single index, it reads the lane axis
# one index a step, or it steps a source axis, a whole group a step on the
# lane axis.
def Classified(reading: TiledView.Reading,+shape: List<&2,U32>,+axis: Nat,+extent: U32,+stride: U32,+lanes: TiledView.Lanes) -> Data:
  match reading:
    case TiledView.Repeated{}: {U32.is_zero(stride) || U32.is_eq(extent,1) == True{} : Bool}
    case TiledView.Lane{}: {U32.to_nat(stride) == ShapeLaws.elements(Public.lane_after(shape,axis)) : Nat}
    case TiledView.Stepping{+found,+step,+cells}:
      KernelGemm.Both<StepAt<shape,stride,0n,found,step>,KernelGemm.Both<{cells == TiledView.cell_index(lanes,stride) : U32},
        Logic.When(Nat.is_eq(U32.to_nat(found),axis),{U32.is_zero((step % 8 : U32)) == True{} : Bool})>>
    case TiledView.Unkept{}: Unit

# The lane axis as a word compares with an axis word as the numbers do.
def same_axis(+found: U32,+axis: Nat,+lanes: TiledView.Lanes,+limits: Machine.Limits,+lane_exact: {TiledView.lane_axis(lanes) == axis : Nat},
  +room: {Nat.is_le(axis,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.is_eq(found,U32.from_nat(TiledView.lane_axis(lanes))) == Nat.is_eq(U32.to_nat(found),axis) : Bool}:
  %Equal.sym(Nat,TiledView.lane_axis(lanes),axis,lane_exact) : {U32.is_eq(found,U32.from_nat(_)) == Nat.is_eq(U32.to_nat(found),axis) : Bool}
  %Bounds.roundtrip(axis,Machine.capacity(limits),Limits.capacity_fits(limits),room) :
    {U32.is_eq(found,U32.from_nat(axis)) == Nat.is_eq(U32.to_nat(found),_) : Bool}
  Comparison.eq(found,U32.from_nat(axis))

# Whole groups a step on the lane axis.
def group_choice(grouped: Bool,+shape: List<&2,U32>,+axis: Nat,+extent: U32,+stride: U32,+lanes: TiledView.Lanes,+found: U32,+step: U32,
  +facts: StepAt<shape,stride,0n,found,step>,+on_lane: {Nat.is_eq(U32.to_nat(found),axis) == True{} : Bool},+classified: {U32.is_zero((step % 8 : U32)) == grouped : Bool}) ->
  Classified(Bool.pick(TiledView.Reading,grouped,TiledView.Stepping{found,step,TiledView.cell_index(lanes,stride)},TiledView.Unkept{}),
    shape,axis,extent,stride,lanes):
  match grouped:
    case True{}:
      KernelGemm.Both{facts,KernelGemm.Both{{==},
        %Equal.sym(Bool,Nat.is_eq(U32.to_nat(found),axis),True{},on_lane) : Logic.When(_,{U32.is_zero((step % 8 : U32)) == True{} : Bool}) classified}}
    case False{}: Unit{}

# One index a step of the lane axis, or whole groups a step.
def lane_choice(unit: Bool,+shape: List<&2,U32>,+axis: Nat,+extent: U32,+stride: U32,+lanes: TiledView.Lanes,+found: U32,+step: U32,
  +facts: StepAt<shape,stride,0n,found,step>,+on_lane: {Nat.is_eq(U32.to_nat(found),axis) == True{} : Bool},+classified: {U32.is_eq(step,1) == unit : Bool}) ->
  Classified(Bool.pick(TiledView.Reading,unit,TiledView.Lane{},
      Bool.pick(TiledView.Reading,U32.is_zero((step % 8 : U32)),TiledView.Stepping{found,step,TiledView.cell_index(lanes,stride)},TiledView.Unkept{})),
    shape,axis,extent,stride,lanes):
  match unit:
    case True{}:
      StepAt{+place,+axis_exact,+inside,+stride_exact} = facts
      +place_axis = Equal.trans(Nat,place,U32.to_nat(found),axis,Equal.sym(Nat,U32.to_nat(found),place,axis_exact),Order.equal_of_is_eq(U32.to_nat(found),axis,on_lane))
      +one = Order.equal_of_is_eq(U32.to_nat(step),1n,
        Equal.trans(Bool,Nat.is_eq(U32.to_nat(step),1n),U32.is_eq(step,1),True{},Equal.sym(Bool,U32.is_eq(step,1),Nat.is_eq(U32.to_nat(step),1n),Comparison.eq(step,1)),classified))
      +cells = ShapeLaws.elements(Public.lane_after(shape,place))
      %place_axis : {U32.to_nat(stride) == ShapeLaws.elements(Public.lane_after(shape,_)) : Nat}
      Equal.trans(Nat,U32.to_nat(stride),Nat.mul(1n,cells),cells,
        %one : {U32.to_nat(stride) == Nat.mul(_,cells) : Nat} stride_exact,
        Equal.trans(Nat,Nat.mul(1n,cells),Nat.mul(cells,1n),cells,RowMajor.multiply_commutative(1n,cells),Order.times_one(cells)))
    case False{}:
      group_choice(U32.is_zero((step % 8 : U32)),shape,axis,extent,stride,lanes,found,step,facts,on_lane,{==})

# A stepped source axis: the lane axis, or another.
def axis_choice(on: Bool,+shape: List<&2,U32>,+axis: Nat,+extent: U32,+stride: U32,+lanes: TiledView.Lanes,+found: U32,+step: U32,+limits: Machine.Limits,
  +facts: StepAt<shape,stride,0n,found,step>,+lane_exact: {TiledView.lane_axis(lanes) == axis : Nat},
  +room: {Nat.is_le(axis,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +classified: {U32.is_eq(found,U32.from_nat(TiledView.lane_axis(lanes))) == on : Bool}) ->
  Classified(Bool.pick(TiledView.Reading,on,TiledView.lane_reading(step,stride,lanes,found),TiledView.Stepping{found,step,TiledView.cell_index(lanes,stride)}),shape,axis,extent,stride,lanes):
  match on:
    case True{}:
      +compared = Equal.trans(Bool,Nat.is_eq(U32.to_nat(found),axis),U32.is_eq(found,U32.from_nat(TiledView.lane_axis(lanes))),on,
        Equal.sym(Bool,U32.is_eq(found,U32.from_nat(TiledView.lane_axis(lanes))),Nat.is_eq(U32.to_nat(found),axis),same_axis(found,axis,lanes,limits,lane_exact,room)),classified)
      lane_choice(U32.is_eq(step,1),shape,axis,extent,stride,lanes,found,step,facts,compared,{==})
    case False{}:
      +compared = Equal.trans(Bool,Nat.is_eq(U32.to_nat(found),axis),U32.is_eq(found,U32.from_nat(TiledView.lane_axis(lanes))),on,
        Equal.sym(Bool,U32.is_eq(found,U32.from_nat(TiledView.lane_axis(lanes))),Nat.is_eq(U32.to_nat(found),axis),same_axis(found,axis,lanes,limits,lane_exact,room)),classified)
      KernelGemm.Both{facts,KernelGemm.Both{{==},
        %Equal.sym(Bool,Nat.is_eq(U32.to_nat(found),axis),False{},compared) : Logic.When(_,{U32.is_zero((step % 8 : U32)) == True{} : Bool}) Unit{}}}

def found_choice(found: Tasks.Stepped,+shape: List<&2,U32>,+axis: Nat,+extent: U32,+stride: U32,+lanes: TiledView.Lanes,+limits: Machine.Limits,
  +facts: Stepping(found,shape,stride,0n),+lane_exact: {TiledView.lane_axis(lanes) == axis : Nat},
  +room: {Nat.is_le(axis,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Classified(TiledView.reading_of(found,stride,lanes),shape,axis,extent,stride,lanes):
  match found:
    case Tasks.Unstepped{}: Unit{}
    case Tasks.Stepped{+found,+step,+source_extent,+first}:
      axis_choice(U32.is_eq(found,U32.from_nat(TiledView.lane_axis(lanes))),shape,axis,extent,stride,lanes,found,step,limits,facts,lane_exact,room,{==})

def zero_choice(zero: Bool,+shape: List<&2,U32>,+axis: Nat,+extent: U32,+stride: U32,+lanes: TiledView.Lanes,+limits: Machine.Limits,
  +lane_exact: {TiledView.lane_axis(lanes) == axis : Nat},+room: {Nat.is_le(axis,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonzero: {Shapes.has_zero(shape) == False{} : Bool},+classified: {Tasks.unread(extent,stride) == zero : Bool}) ->
  Classified(Bool.pick(TiledView.Reading,zero,TiledView.Repeated{},
      TiledView.reading_of(Tasks.stepped_axis(shape,Grid.coordinates(shape,0),stride,0),stride,lanes)),
    shape,axis,extent,stride,lanes):
  match zero:
    case True{}: classified
    case False{}:
      found_choice(Tasks.stepped_axis(shape,Grid.coordinates(shape,0),stride,0),shape,axis,extent,stride,lanes,limits,
        stepped_at(shape,Grid.coordinates(shape,0),stride,0,0n,limits,{==},rank_room,cells_room,nonzero),lane_exact,room)

# What every target axis reads of values with their lanes on an axis.
def classified_facts(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+extent: U32,+stride: U32,+limits: Machine.Limits,
  +inside: {Nat.is_le(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonzero: {Shapes.has_zero(shape) == False{} : Bool}) ->
  Classified(TiledView.classified(shape,TiledView.lanes_of(shape,axis,position),extent,stride),shape,axis,extent,stride,TiledView.lanes_of(shape,axis,position)):
  zero_choice(Tasks.unread(extent,stride),shape,axis,extent,stride,TiledView.lanes_of(shape,axis,position),limits,lane_axis_of(shape,axis,position,inside),
    Order.transitive(axis,List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits)),inside,rank_room),rank_room,cells_room,nonzero,{==})

# ---- The cell of a word ----

# The cell at an index of the outer indices and a group, as words.
def cell_at_word(position: Grid.Position,+outer_word: U32,+group_word: U32,+groups: U32,+outer: U32,+index: Nat,+group: Nat,+outer_count: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +index_exact: {U32.to_nat(outer_word) == index : Nat},+group_exact: {U32.to_nat(group_word) == group : Nat},
  +outer_exact: {U32.to_nat(outer) == outer_count : Nat},
  +index_inside: {Nat.is_lt(index,outer_count) == True{} : Bool},+group_inside: {Nat.is_lt(group,U32.to_nat(groups)) == True{} : Bool},
  +room: {Nat.is_le(Nat.mul(U32.to_nat(groups),outer_count),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(TiledView.cell_at(position,outer_word,group_word,groups,outer)) == Representation.index_at(position,outer_count,U32.to_nat(groups),index,group) : Nat}:
  match position:
    case Grid.Innermost{}:
      +count = U32.to_nat(groups)
      +below = RowMajor.radix(index,outer_count,group,count,index_inside,group_inside)
      +total = {%RowMajor.multiply_commutative(count,outer_count) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room :
        {Nat.is_le(Nat.mul(outer_count,count),U32.to_nat(cap)) == True{} : Bool}}
      +sum_room = Bounds.strict_le(Nat.add(Nat.mul(index,count),group),U32.to_nat(cap),Bounds.nat_trans(Nat.add(Nat.mul(index,count),group),Nat.mul(outer_count,count),U32.to_nat(cap),below,total))
      Arithmetic.sum_exact(cap,cap_ok,(outer_word * groups : U32),group_word,Nat.mul(index,count),group,
        Arithmetic.product_exact(cap,cap_ok,outer_word,groups,index,count,index_exact,{==},
          Order.transitive(Nat.mul(index,count),Nat.add(Nat.mul(index,count),group),U32.to_nat(cap),Order.le_add(Nat.mul(index,count),group),sum_room)),
        group_exact,sum_room)
    case Grid.Outermost{}:
      +count = U32.to_nat(groups)
      +below = RowMajor.radix(group,count,index,outer_count,group_inside,index_inside)
      +sum_room = Bounds.strict_le(Nat.add(Nat.mul(group,outer_count),index),U32.to_nat(cap),Bounds.nat_trans(Nat.add(Nat.mul(group,outer_count),index),Nat.mul(count,outer_count),U32.to_nat(cap),below,room))
      Arithmetic.sum_exact(cap,cap_ok,(group_word * outer : U32),outer_word,Nat.mul(group,outer_count),index,
        Arithmetic.product_exact(cap,cap_ok,group_word,outer,group,outer_count,group_exact,outer_exact,
          Order.transitive(Nat.mul(group,outer_count),Nat.add(Nat.mul(group,outer_count),index),U32.to_nat(cap),Order.le_add(Nat.mul(group,outer_count),index),sum_room)),
        index_exact,sum_room)

# The number of a channel at an outer index, its quotient by the cells of the
# lane axis and the cells after it.
def channel_split(+quotient: Nat,+remainder: Nat,+channel: Nat,+lanes: Nat,+after: Nat) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(quotient,lanes),channel),after),remainder) == Nat.add(Nat.mul(quotient,Nat.mul(lanes,after)),Nat.add(Nat.mul(channel,after),remainder)) : Nat}:
  %Equal.sym(Nat,Nat.mul(Nat.add(Nat.mul(quotient,lanes),channel),after),Nat.add(Nat.mul(Nat.mul(quotient,lanes),after),Nat.mul(channel,after)),RowMajor.distrib(Nat.mul(quotient,lanes),channel,after)) :
    {Nat.add(_,remainder) == Nat.add(Nat.mul(quotient,Nat.mul(lanes,after)),Nat.add(Nat.mul(channel,after),remainder)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.mul(quotient,lanes),after),Nat.mul(quotient,Nat.mul(lanes,after)),RowMajor.assoc(quotient,lanes,after)) :
    {Nat.add(Nat.add(_,Nat.mul(channel,after)),remainder) == Nat.add(Nat.mul(quotient,Nat.mul(lanes,after)),Nat.add(Nat.mul(channel,after),remainder)) : Nat}
  Addition.associative(Nat.mul(quotient,Nat.mul(lanes,after)),Nat.mul(channel,after),remainder)

# The cell the lanes of values give the number of digits with a channel at
# the lane axis: the cell of the other digits and the group of the channel.
def cell_word(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+others: List<&2,Nat>,+channel: Nat,+limits: Machine.Limits,
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+below: Coordinates.DigitsBelow(Grid.removed(shape,axis),others),
  +inside: {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(shape,axis))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(shape,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(TiledView.cell_index(TiledView.lanes_of(shape,axis,position),U32.from_nat(ShapeLaws.digits_value(shape,Coordinates.inserted_digit(axis,others,channel))))) ==
    ShapeLaws.digits_value(Tiled.cell_shape(shape,axis,position),Coordinates.placed_digits(True{},position,others,Nat.div(channel,8n))) : Nat}:
  +cap = Machine.capacity(limits)
  +cap_ok = Limits.capacity_fits(limits)
  +half = Machine.half_word(limits)
  +half_ok = Limits.half_word_fits(limits)
  +extent = Grid.lanes_at(shape,axis)
  +lanes = U32.to_nat(extent)
  +after = Public.lane_after(shape,axis)
  +inner = ShapeLaws.elements(after)
  +inner_word = Tiled.product(after)
  +removed = Grid.removed(shape,axis)
  +outer_count = ShapeLaws.elements(removed)
  +outer_word = (Public.lane_prefix(shape,axis,1) * Tiled.product(after) : U32)
  +before = Coordinates.before_cells(shape,axis)
  +outer = ShapeLaws.digits_value(removed,others)
  +outer_below = Coordinates.digits_below(removed,others,below)
  +removed_exact = Coordinates.removed_cells(shape,axis)
  +shape_exact = Coordinates.lane_cells(shape,axis)
  +outer_positive = Order.le_lt_trans(0n,outer,outer_count,Order.zero_le(outer),outer_below)
  +split_positive = {%removed_exact : {Nat.is_lt(0n,_) == True{} : Bool} outer_positive : {Nat.is_lt(0n,Nat.mul(before,inner)) == True{} : Bool}}
  +before_positive = RowMajor.left_factor_positive(before,inner,split_positive)
  +inner_positive = DivisionOrder.right_factor_positive(before,inner,split_positive)
  +lanes_positive = Order.le_lt_trans(0n,channel,lanes,Order.zero_le(channel),inside)
  +block = Nat.mul(lanes,inner)
  +block_positive = Order.product_positive(lanes,inner,lanes_positive,inner_positive)
  +shape_block = Equal.trans(Nat,ShapeLaws.elements(shape),Nat.mul(Nat.mul(before,lanes),inner),Nat.mul(before,block),shape_exact,RowMajor.assoc(before,lanes,inner))
  +block_le = {%Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(before,block),shape_block) : {Nat.is_le(block,_) == True{} : Bool}
    %RowMajor.multiply_commutative(block,before) : {Nat.is_le(block,_) == True{} : Bool}
    RowMajor.product_covers(block,before,before_positive) : {Nat.is_le(block,ShapeLaws.elements(shape)) == True{} : Bool}}
  +block_room = Order.transitive(block,ShapeLaws.elements(shape),U32.to_nat(cap),block_le,room)
  +inner_le_block = {%RowMajor.multiply_commutative(inner,lanes) : {Nat.is_le(inner,_) == True{} : Bool}
    RowMajor.product_covers(inner,lanes,lanes_positive) : {Nat.is_le(inner,block) == True{} : Bool}}
  +inner_room = Order.transitive(inner,block,U32.to_nat(cap),inner_le_block,block_room)
  +lanes_le_block = RowMajor.product_covers(lanes,inner,inner_positive)
  +lanes_room = Order.transitive(lanes,block,U32.to_nat(cap),lanes_le_block,block_room)
  +removed_le = {%Equal.sym(Nat,outer_count,Nat.mul(before,inner),removed_exact) : {Nat.is_le(_,ShapeLaws.elements(shape)) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(Nat.mul(before,lanes),inner),shape_exact) : {Nat.is_le(Nat.mul(before,inner),_) == True{} : Bool}
    RowMajor.mul_le(before,Nat.mul(before,lanes),inner,RowMajor.product_covers(before,lanes,lanes_positive)) :
    {Nat.is_le(outer_count,ShapeLaws.elements(shape)) == True{} : Bool}}
  +removed_room = Order.transitive(outer_count,ShapeLaws.elements(shape),U32.to_nat(cap),removed_le,room)
  +quotient = Nat.div(outer,inner)
  +remainder = Nat.mod(outer,inner)
  +remainder_small = DivisionOrder.remainder_small(outer,inner,inner_positive)
  +outer_split = DivisionOrder.quotient_exact(outer,inner,inner_positive)
  +number = Coordinates.channel_number(outer,channel,lanes,inner)
  +number_exact = Coordinates.channel_digits(shape,axis,others,channel,below,inside)
  +quotient_below = DivisionOrder.quotient_below(outer,before,inner,inner_positive,{%removed_exact : {Nat.is_lt(outer,_) == True{} : Bool} outer_below : {Nat.is_lt(outer,Nat.mul(before,inner)) == True{} : Bool}})
  +low = Nat.add(Nat.mul(channel,inner),remainder)
  +low_small = RowMajor.radix(channel,lanes,remainder,inner,inside,remainder_small)
  +number_below = {%Equal.sym(Nat,ShapeLaws.elements(shape),Nat.mul(before,block),shape_block) : {Nat.is_lt(Nat.add(Nat.mul(quotient,block),low),_) == True{} : Bool}
    RowMajor.radix(quotient,before,low,block,quotient_below,low_small) : {Nat.is_lt(Nat.add(Nat.mul(quotient,block),low),ShapeLaws.elements(shape)) == True{} : Bool}}
  +split = channel_split(quotient,remainder,channel,lanes,inner)
  +number_room = Order.transitive(number,ShapeLaws.elements(shape),U32.to_nat(cap),
    Bounds.strict_le(number,ShapeLaws.elements(shape),{%Equal.sym(Nat,number,Nat.add(Nat.mul(quotient,block),low),split) : {Nat.is_lt(_,ShapeLaws.elements(shape)) == True{} : Bool} number_below : {Nat.is_lt(number,ShapeLaws.elements(shape)) == True{} : Bool}}),room)
  +word = U32.from_nat(ShapeLaws.digits_value(shape,Coordinates.inserted_digit(axis,others,channel)))
  +word_exact = {%number_exact : {U32.to_nat(U32.from_nat(_)) == number : Nat}
    Bounds.roundtrip(number,cap,cap_ok,number_room) : {U32.to_nat(word) == number : Nat}}
  +inner_exact = Coordinates.product_word(limits,after,inner_room)
  +block_word = (extent * inner_word : U32)
  +block_exact = Arithmetic.product_exact(cap,cap_ok,extent,inner_word,lanes,inner,{==},inner_exact,block_room)
  +group_index = Equal.trans(Nat,U32.to_nat(word),number,Nat.add(Nat.mul(quotient,block),low),word_exact,split)
  +quotient_exact = Radix.quotient_word(word,block_word,quotient,low,block,half,half_ok,group_index,block_exact,block_positive,low_small,Limits.double_bound(limits,block,block_room))
  +lane_word = (word / inner_word : U32)
  +lane_exact = Radix.quotient_word(word,inner_word,Nat.add(Nat.mul(quotient,lanes),channel),remainder,inner,half,half_ok,word_exact,inner_exact,inner_positive,remainder_small,
    Limits.double_bound(limits,inner,inner_room))
  +remainder_exact = Radix.remainder_word(word,inner_word,Nat.add(Nat.mul(quotient,lanes),channel),remainder,inner,half,half_ok,word_exact,inner_exact,inner_positive,remainder_small,
    Limits.double_bound(limits,inner,inner_room))
  +channel_exact = Radix.remainder_word(lane_word,extent,quotient,channel,lanes,half,half_ok,lane_exact,{==},lanes_positive,inside,Limits.double_bound(limits,lanes,lanes_room))
  +group = Nat.div(channel,8n)
  +group_exact = U32Division.quotient_exact(2147483648,{==},(lane_word % extent : U32),8,channel,8n,channel_exact,{==},{==},{==})
  +product_room = Order.transitive(Nat.mul(quotient,inner),outer,U32.to_nat(cap),
    {%Equal.sym(Nat,outer,Nat.add(Nat.mul(quotient,inner),remainder),outer_split) : {Nat.is_le(Nat.mul(quotient,inner),_) == True{} : Bool}
      Order.le_add(Nat.mul(quotient,inner),remainder) : {Nat.is_le(Nat.mul(quotient,inner),outer) == True{} : Bool}},
    Order.transitive(outer,outer_count,U32.to_nat(cap),Bounds.strict_le(outer,outer_count,outer_below),removed_room))
  +index_exact = Equal.trans(Nat,U32.to_nat((((word / block_word : U32) * inner_word : U32) + (word % inner_word : U32) : U32)),Nat.add(Nat.mul(quotient,inner),remainder),outer,
    Arithmetic.sum_exact(cap,cap_ok,((word / block_word : U32) * inner_word : U32),(word % inner_word : U32),Nat.mul(quotient,inner),remainder,
      Arithmetic.product_exact(cap,cap_ok,(word / block_word : U32),inner_word,quotient,inner,quotient_exact,inner_exact,product_room),remainder_exact,
      {%outer_split : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        Order.transitive(outer,outer_count,U32.to_nat(cap),Bounds.strict_le(outer,outer_count,outer_below),removed_room) :
        {Nat.is_le(Nat.add(Nat.mul(quotient,inner),remainder),U32.to_nat(cap)) == True{} : Bool}}),
    Equal.sym(Nat,outer,Nat.add(Nat.mul(quotient,inner),remainder),outer_split))
  +groups_word = Tiled.groups(extent)
  +grouped = TiledValues.grouped(limits,extent,lanes_room)
  +cells_count = {%within :
      {Nat.is_le(ShapeLaws.elements(Grid.placed(_,position,removed,groups_word)),U32.to_nat(cap)) == True{} : Bool}
    cells_room : {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,removed,groups_word)),U32.to_nat(cap)) == True{} : Bool}}
  +cells_product = {%Public.placed_cells(position,removed,groups_word) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} cells_count :
    {Nat.is_le(Nat.mul(U32.to_nat(groups_word),outer_count),U32.to_nat(cap)) == True{} : Bool}}
  %Equal.sym(TiledView.Lanes,TiledView.lanes_of(shape,axis,position),TiledView.Lanes{axis,extent,inner_word,outer_word,position},lanes_exact(shape,axis,position,Bounds.strict_le(axis,List.length(&2,U32,shape),within))) :
    {U32.to_nat(TiledView.cell_index(_,word)) == ShapeLaws.digits_value(Tiled.cell_shape(shape,axis,position),Coordinates.placed_digits(True{},position,others,group)) : Nat}
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shape)),True{},within) :
    {U32.to_nat(TiledView.cell_index(TiledView.Lanes{axis,extent,inner_word,outer_word,position},word)) ==
      ShapeLaws.digits_value(Grid.placed(_,position,removed,groups_word),Coordinates.placed_digits(True{},position,others,group)) : Nat}
  %Equal.sym(List<&2,Nat>,Coordinates.placed_digits(True{},position,others,group),Coordinates.cell_digits(position,others,group),Coordinates.placed_cell(position,others,group)) :
    {U32.to_nat(TiledView.cell_index(TiledView.Lanes{axis,extent,inner_word,outer_word,position},word)) == ShapeLaws.digits_value(Grid.placed(True{},position,removed,groups_word),_) : Nat}
  %Equal.sym(Nat,ShapeLaws.digits_value(Grid.placed(True{},position,removed,groups_word),Coordinates.cell_digits(position,others,group)),
      Representation.index_at(position,outer_count,U32.to_nat(groups_word),outer,group),Coordinates.cell_number(position,others,group,removed,groups_word,below)) :
    {U32.to_nat(TiledView.cell_index(TiledView.Lanes{axis,extent,inner_word,outer_word,position},word)) == _ : Nat}
  cell_at_word(position,(((word / block_word : U32) * inner_word : U32) + (word % inner_word : U32) : U32),((lane_word % extent : U32) / 8 : U32),groups_word,outer_word,outer,group,outer_count,
    cap,cap_ok,index_exact,group_exact,Coordinates.outer_word(limits,shape,axis,removed_room),outer_below,TiledValues.group_inside(limits,extent,channel,grouped,inside),cells_product)

# ---- Digit lists added ----

def cons_equal(+left: Nat,+right: Nat,+rest: List<&2,Nat>,+others: List<&2,Nat>,+head: {left == right : Nat},+tail: {rest == others : List<&2,Nat>}) ->
  {Con{left,rest} == Con{right,others} : List<&2,Nat>}:
  %Equal.sym(Nat,left,right,head) : {Con{_,rest} == Con{right,others} : List<&2,Nat>}
  Equal.cong(List<&2,Nat>,List<&2,Nat>,list => right <> list,rest,others,tail)

def sum_swap(+first: Nat,+second: Nat,+third: Nat) -> {Nat.add(Nat.add(first,second),third) == Nat.add(Nat.add(first,third),second) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(first,second),third),Nat.add(first,Nat.add(second,third)),Nat.add(Nat.add(first,third),second),
    Addition.associative(first,second,third),
    Equal.trans(Nat,Nat.add(first,Nat.add(second,third)),Nat.add(first,Nat.add(third,second)),Nat.add(Nat.add(first,third),second),
      Equal.cong(Nat,Nat,value => Nat.add(first,value),Nat.add(second,third),Nat.add(third,second),Order.commutative(second,third)),
      Equal.sym(Nat,Nat.add(Nat.add(first,third),second),Nat.add(first,Nat.add(third,second)),Addition.associative(first,third,second))))

# Zero digits added to digits leave them.
def added_zeros(digits: List<&2,Nat>) -> {Digits.added(digits,zeros(List.length(&2,Nat,digits))) == digits : List<&2,Nat>}:
  match digits:
    case Nil{}: {==}
    case Con{+digit,+rest}:
      cons_equal(Nat.add(digit,0n),digit,Digits.added(rest,zeros(List.length(&2,Nat,rest))),rest,Addition.zero_right(digit),added_zeros(rest))

def zeros_added(rank: Nat) -> {Digits.added(zeros(rank),zeros(rank)) == zeros(rank) : List<&2,Nat>}:
  match rank:
    case 0n: {==}
    case 1n+ +rest: Equal.cong(List<&2,Nat>,List<&2,Nat>,list => 0n <> list,Digits.added(zeros(rest),zeros(rest)),zeros(rest),zeros_added(rest))

def unit_zero(rank: Nat,axis: Nat) -> {unit_digits(rank,axis,0n) == zeros(rank) : List<&2,Nat>}:
  match rank axis:
    case 0n +axis: {==}
    case 1n+ +rest 0n: {==}
    case 1n+ +rest 1n+ +next: Equal.cong(List<&2,Nat>,List<&2,Nat>,list => 0n <> list,unit_digits(rest,next,0n),zeros(rest),unit_zero(rest,next))

# Amounts at one axis added are the sum at the axis.
def units_added(rank: Nat,axis: Nat,+left: Nat,+right: Nat) ->
  {Digits.added(unit_digits(rank,axis,left),unit_digits(rank,axis,right)) == unit_digits(rank,axis,Nat.add(left,right)) : List<&2,Nat>}:
  match rank axis:
    case 0n +axis: {==}
    case 1n+ +rest 0n:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,list => Nat.add(left,right) <> list,Digits.added(zeros(rest),zeros(rest)),zeros(rest),zeros_added(rest))
    case 1n+ +rest 1n+ +next:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,list => 0n <> list,Digits.added(unit_digits(rest,next,left),unit_digits(rest,next,right)),unit_digits(rest,next,Nat.add(left,right)),
        units_added(rest,next,left,right))

# Digits added in either order of the two later lists.
def added_swap(first: List<&2,Nat>,second: List<&2,Nat>,third: List<&2,Nat>,
  +second_length: {List.length(&2,Nat,second) == List.length(&2,Nat,first) : Nat},+third_length: {List.length(&2,Nat,third) == List.length(&2,Nat,first) : Nat}) ->
  {Digits.added(Digits.added(first,second),third) == Digits.added(Digits.added(first,third),second) : List<&2,Nat>}:
  match first second third:
    case Con{+x,+xs} Con{+y,+ys} Con{+z,+zs}:
      cons_equal(Nat.add(Nat.add(x,y),z),Nat.add(Nat.add(x,z),y),Digits.added(Digits.added(xs,ys),zs),Digits.added(Digits.added(xs,zs),ys),sum_swap(x,y,z),
        added_swap(xs,ys,zs,length_tail(List.length(&2,Nat,ys),List.length(&2,Nat,xs),second_length),length_tail(List.length(&2,Nat,zs),List.length(&2,Nat,xs),third_length)))
    case Nil{} +second +third: {==}
    case Con{+x,+xs} Nil{} +third:
      Empty.absurd({Digits.added(Digits.added(Con{x,xs},Nil{}),third) == Digits.added(Digits.added(Con{x,xs},third),Nil{}) : List<&2,Nat>},
        Digits.zero_successor(List.length(&2,Nat,xs),second_length))
    case Con{+x,+xs} Con{+y,+ys} Nil{}:
      Empty.absurd({Digits.added(Digits.added(Con{x,xs},Con{y,ys}),Nil{}) == Digits.added(Digits.added(Con{x,xs},Nil{}),Con{y,ys}) : List<&2,Nat>},
        Digits.zero_successor(List.length(&2,Nat,xs),third_length))

# The digits but one axis of digits added are those of each added.
def removed_added(left: List<&2,Nat>,right: List<&2,Nat>,axis: Nat,+same: {List.length(&2,Nat,right) == List.length(&2,Nat,left) : Nat}) ->
  {Geometry.removed_at(Digits.added(left,right),axis) == Digits.added(Geometry.removed_at(left,axis),Geometry.removed_at(right,axis)) : List<&2,Nat>}:
  match left right axis:
    case Con{+x,+xs} Con{+y,+ys} 0n: {==}
    case Con{+x,+xs} Con{+y,+ys} 1n+ +next:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,list => Nat.add(x,y) <> list,Geometry.removed_at(Digits.added(xs,ys),next),
        Digits.added(Geometry.removed_at(xs,next),Geometry.removed_at(ys,next)),removed_added(xs,ys,next,length_tail(List.length(&2,Nat,ys),List.length(&2,Nat,xs),same)))
    case Nil{} +right +axis: {==}
    case Con{+x,+xs} Nil{} +axis:
      Empty.absurd({Geometry.removed_at(Digits.added(Con{x,xs},Nil{}),axis) == Digits.added(Geometry.removed_at(Con{x,xs},axis),Geometry.removed_at(Nil{},axis)) : List<&2,Nat>},
        Digits.zero_successor(List.length(&2,Nat,xs),same))

def appended_added(left: List<&2,Nat>,right: List<&2,Nat>,+last: Nat,+other: Nat,+same: {List.length(&2,Nat,right) == List.length(&2,Nat,left) : Nat}) ->
  {Digits.added(List.append(&2,Nat,left,[last]),List.append(&2,Nat,right,[other])) == List.append(&2,Nat,Digits.added(left,right),[Nat.add(last,other)]) : List<&2,Nat>}:
  match left right:
    case Con{+x,+xs} Con{+y,+ys}:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,list => Nat.add(x,y) <> list,Digits.added(List.append(&2,Nat,xs,[last]),List.append(&2,Nat,ys,[other])),
        List.append(&2,Nat,Digits.added(xs,ys),[Nat.add(last,other)]),appended_added(xs,ys,last,other,length_tail(List.length(&2,Nat,ys),List.length(&2,Nat,xs),same)))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+y,+ys}:
      Empty.absurd({Digits.added(List.append(&2,Nat,Nil{},[last]),List.append(&2,Nat,Con{y,ys},[other])) == List.append(&2,Nat,Digits.added(Nil{},Con{y,ys}),[Nat.add(last,other)]) : List<&2,Nat>},
        successor_zero(List.length(&2,Nat,ys),same))
    case Con{+x,+xs} Nil{}:
      Empty.absurd({Digits.added(List.append(&2,Nat,Con{x,xs},[last]),List.append(&2,Nat,Nil{},[other])) == List.append(&2,Nat,Digits.added(Con{x,xs},Nil{}),[Nat.add(last,other)]) : List<&2,Nat>},
        Digits.zero_successor(List.length(&2,Nat,xs),same))

def appended_length(digits: List<&2,Nat>,+last: Nat) -> {List.length(&2,Nat,List.append(&2,Nat,digits,[last])) == 1n+List.length(&2,Nat,digits) : Nat}:
  match digits:
    case Nil{}: {==}
    case Con{+digit,+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,List.append(&2,Nat,rest,[last])),1n+List.length(&2,Nat,rest),appended_length(rest,last))

# The cell digits of digits added are those of each added.
def placed_added(position: Grid.Position,+left: List<&2,Nat>,+right: List<&2,Nat>,+group: Nat,+other: Nat,+same: {List.length(&2,Nat,right) == List.length(&2,Nat,left) : Nat}) ->
  {Coordinates.placed_digits(True{},position,Digits.added(left,right),Nat.add(group,other)) ==
    Digits.added(Coordinates.placed_digits(True{},position,left,group),Coordinates.placed_digits(True{},position,right,other)) : List<&2,Nat>}:
  match position:
    case Grid.Innermost{}:
      Equal.sym(List<&2,Nat>,Digits.added(List.append(&2,Nat,left,[group]),List.append(&2,Nat,right,[other])),List.append(&2,Nat,Digits.added(left,right),[Nat.add(group,other)]),
        appended_added(left,right,group,other,same))
    case Grid.Outermost{}: {==}

def placed_length(position: Grid.Position,+others: List<&2,Nat>,+group: Nat) ->
  {List.length(&2,Nat,Coordinates.placed_digits(True{},position,others,group)) == 1n+List.length(&2,Nat,others) : Nat}:
  match position:
    case Grid.Innermost{}: appended_length(others,group)
    case Grid.Outermost{}: {==}

# ---- The cell of digits ----

# The number of the cell of digits: their other digits and the group of
# their digit at the lane axis.
def cell_value(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+digits: List<&2,Nat>) -> Nat:
  ShapeLaws.digits_value(Tiled.cell_shape(shape,axis,position),
    Coordinates.placed_digits(True{},position,Geometry.removed_at(digits,axis),Nat.div(Digits.digit_at(digits,axis),8n)))

# The lanes of values give the number of digits below the values the cell
# of the digits.
def cell_of(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+digits: List<&2,Nat>,+limits: Machine.Limits,
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+below: Coordinates.DigitsBelow(shape,digits),
  +room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(shape,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {U32.to_nat(TiledView.cell_index(TiledView.lanes_of(shape,axis,position),U32.from_nat(ShapeLaws.digits_value(shape,digits)))) == cell_value(shape,axis,position,digits) : Nat}:
  +length = Values.below_length(shape,digits,below)
  +inside = {%Equal.sym(Nat,List.length(&2,Nat,digits),List.length(&2,U32,shape),length) : {Nat.is_lt(axis,_) == True{} : Bool} within :
    {Nat.is_lt(axis,List.length(&2,Nat,digits)) == True{} : Bool}}
  %Values.reinserted(digits,axis,inside) :
    {U32.to_nat(TiledView.cell_index(TiledView.lanes_of(shape,axis,position),U32.from_nat(ShapeLaws.digits_value(shape,_)))) == cell_value(shape,axis,position,digits) : Nat}
  cell_word(shape,axis,position,Geometry.removed_at(digits,axis),Digits.digit_at(digits,axis),limits,within,Values.removed_below(shape,digits,axis,below),
    Values.digit_below(shape,digits,axis,below,within),room,cells_room)

def lengths_same(+left: List<&2,Nat>,+right: List<&2,Nat>,+shape: List<&2,U32>,+left_length: {List.length(&2,Nat,left) == List.length(&2,U32,shape) : Nat},
  +right_length: {List.length(&2,Nat,right) == List.length(&2,U32,shape) : Nat}) -> {List.length(&2,Nat,right) == List.length(&2,Nat,left) : Nat}:
  Equal.trans(Nat,List.length(&2,Nat,right),List.length(&2,U32,shape),List.length(&2,Nat,left),right_length,Equal.sym(Nat,List.length(&2,Nat,left),List.length(&2,U32,shape),left_length))

# The cell digits of digits of the values number as many as the cell shape.
def cell_digits_length(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+digits: List<&2,Nat>,+group: Nat,
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+length: {List.length(&2,Nat,digits) == List.length(&2,U32,shape) : Nat}) ->
  {List.length(&2,Nat,Coordinates.placed_digits(True{},position,Geometry.removed_at(digits,axis),group)) == List.length(&2,U32,Tiled.cell_shape(shape,axis,position)) : Nat}:
  +inside = {%Equal.sym(Nat,List.length(&2,Nat,digits),List.length(&2,U32,shape),length) : {Nat.is_lt(axis,_) == True{} : Bool} within :
    {Nat.is_lt(axis,List.length(&2,Nat,digits)) == True{} : Bool}}
  Equal.trans(Nat,List.length(&2,Nat,Coordinates.placed_digits(True{},position,Geometry.removed_at(digits,axis),group)),1n+List.length(&2,Nat,Geometry.removed_at(digits,axis)),
    List.length(&2,U32,Tiled.cell_shape(shape,axis,position)),placed_length(position,Geometry.removed_at(digits,axis),group),
    Equal.trans(Nat,1n+List.length(&2,Nat,Geometry.removed_at(digits,axis)),List.length(&2,Nat,digits),List.length(&2,U32,Tiled.cell_shape(shape,axis,position)),
      Geometry.length_removed(digits,axis,inside),
      Equal.trans(Nat,List.length(&2,Nat,digits),List.length(&2,U32,shape),List.length(&2,U32,Tiled.cell_shape(shape,axis,position)),length,
        Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(shape,axis,position)),List.length(&2,U32,shape),Rooms.cell_rank(shape,axis,position)))))

# The group of a sum with a multiple of eight is the group of the other plus
# the multiple's.
def group_added(+value: Nat,+aligned: Nat,+zero: {Nat.mod(aligned,8n) == 0n : Nat}) ->
  {Nat.div(Nat.add(value,aligned),8n) == Nat.add(Nat.div(value,8n),Nat.div(aligned,8n)) : Nat}:
  +quotient = Nat.div(aligned,8n)
  +exact = Equal.trans(Nat,aligned,Nat.add(Nat.mul(quotient,8n),Nat.mod(aligned,8n)),Nat.mul(quotient,8n),DivisionOrder.quotient_exact(aligned,8n,{==}),
    Equal.trans(Nat,Nat.add(Nat.mul(quotient,8n),Nat.mod(aligned,8n)),Nat.add(Nat.mul(quotient,8n),0n),Nat.mul(quotient,8n),
      Equal.cong(Nat,Nat,rest => Nat.add(Nat.mul(quotient,8n),rest),Nat.mod(aligned,8n),0n,zero),Addition.zero_right(Nat.mul(quotient,8n))))
  %Equal.sym(Nat,aligned,Nat.mul(quotient,8n),exact) : {Nat.div(Nat.add(value,_),8n) == Nat.add(Nat.div(value,8n),quotient) : Nat}
  %Order.commutative(Nat.mul(quotient,8n),value) : {Nat.div(_,8n) == Nat.add(Nat.div(value,8n),quotient) : Nat}
  Equal.trans(Nat,Nat.div(Nat.add(Nat.mul(quotient,8n),value),8n),Nat.add(quotient,Nat.div(value,8n)),Nat.add(Nat.div(value,8n),quotient),
    DivisionOrder.quotient_shift(value,8n,quotient,{==}),Order.commutative(quotient,Nat.div(value,8n)))

# The cell of digits added, the later at a multiple of eight on the lane
# axis, is the sum of their cells.
def cell_added(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+left: List<&2,Nat>,+right: List<&2,Nat>,
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +left_length: {List.length(&2,Nat,left) == List.length(&2,U32,shape) : Nat},+right_length: {List.length(&2,Nat,right) == List.length(&2,U32,shape) : Nat},
  +aligned: {Nat.mod(Digits.digit_at(right,axis),8n) == 0n : Nat}) ->
  {cell_value(shape,axis,position,Digits.added(left,right)) == Nat.add(cell_value(shape,axis,position,left),cell_value(shape,axis,position,right)) : Nat}:
  +cells = Tiled.cell_shape(shape,axis,position)
  +first = Digits.digit_at(left,axis)
  +second = Digits.digit_at(right,axis)
  +left_removed = Geometry.removed_at(left,axis)
  +right_removed = Geometry.removed_at(right,axis)
  +same = lengths_same(left,right,shape,left_length,right_length)
  +left_inside = {%Equal.sym(Nat,List.length(&2,Nat,left),List.length(&2,U32,shape),left_length) : {Nat.is_lt(axis,_) == True{} : Bool} within :
    {Nat.is_lt(axis,List.length(&2,Nat,left)) == True{} : Bool}}
  +right_inside = {%Equal.sym(Nat,List.length(&2,Nat,right),List.length(&2,U32,shape),right_length) : {Nat.is_lt(axis,_) == True{} : Bool} within :
    {Nat.is_lt(axis,List.length(&2,Nat,right)) == True{} : Bool}}
  +digit = Digits.added_digit(left,right,axis,left_inside,Equal.sym(Nat,List.length(&2,Nat,right),List.length(&2,Nat,left),same))
  +removed = removed_added(left,right,axis,same)
  +removed_same = length_tail(List.length(&2,Nat,right_removed),List.length(&2,Nat,left_removed),
    Equal.trans(Nat,1n+List.length(&2,Nat,right_removed),List.length(&2,Nat,right),1n+List.length(&2,Nat,left_removed),Geometry.length_removed(right,axis,right_inside),
      Equal.trans(Nat,List.length(&2,Nat,right),List.length(&2,Nat,left),1n+List.length(&2,Nat,left_removed),same,
        Equal.sym(Nat,1n+List.length(&2,Nat,left_removed),List.length(&2,Nat,left),Geometry.length_removed(left,axis,left_inside)))))
  %Equal.sym(List<&2,Nat>,Geometry.removed_at(Digits.added(left,right),axis),Digits.added(left_removed,right_removed),removed) :
    {ShapeLaws.digits_value(cells,Coordinates.placed_digits(True{},position,_,Nat.div(Digits.digit_at(Digits.added(left,right),axis),8n))) ==
      Nat.add(cell_value(shape,axis,position,left),cell_value(shape,axis,position,right)) : Nat}
  %Equal.sym(Nat,Digits.digit_at(Digits.added(left,right),axis),Nat.add(first,second),digit) :
    {ShapeLaws.digits_value(cells,Coordinates.placed_digits(True{},position,Digits.added(left_removed,right_removed),Nat.div(_,8n))) ==
      Nat.add(cell_value(shape,axis,position,left),cell_value(shape,axis,position,right)) : Nat}
  %Equal.sym(Nat,Nat.div(Nat.add(first,second),8n),Nat.add(Nat.div(first,8n),Nat.div(second,8n)),group_added(first,second,aligned)) :
    {ShapeLaws.digits_value(cells,Coordinates.placed_digits(True{},position,Digits.added(left_removed,right_removed),_)) ==
      Nat.add(cell_value(shape,axis,position,left),cell_value(shape,axis,position,right)) : Nat}
  %Equal.sym(List<&2,Nat>,Coordinates.placed_digits(True{},position,Digits.added(left_removed,right_removed),Nat.add(Nat.div(first,8n),Nat.div(second,8n))),
      Digits.added(Coordinates.placed_digits(True{},position,left_removed,Nat.div(first,8n)),Coordinates.placed_digits(True{},position,right_removed,Nat.div(second,8n))),
      placed_added(position,left_removed,right_removed,Nat.div(first,8n),Nat.div(second,8n),removed_same)) :
    {ShapeLaws.digits_value(cells,_) == Nat.add(cell_value(shape,axis,position,left),cell_value(shape,axis,position,right)) : Nat}
  digits_added(cells,Coordinates.placed_digits(True{},position,left_removed,Nat.div(first,8n)),Coordinates.placed_digits(True{},position,right_removed,Nat.div(second,8n)),
    cell_digits_length(shape,axis,position,left,Nat.div(first,8n),within,left_length),cell_digits_length(shape,axis,position,right,Nat.div(second,8n),within,right_length))

# ---- Zero digits ----

def digit_zeros(rank: Nat,axis: Nat) -> {Digits.digit_at(zeros(rank),axis) == 0n : Nat}:
  match rank axis:
    case 0n +axis: {==}
    case 1n+ +rest 0n: {==}
    case 1n+ +rest 1n+ +next: digit_zeros(rest,next)

def removed_zeros_later(rest: Nat,+next: Nat,+inside: {Nat.is_lt(next,rest) == True{} : Bool},
  +later: {Geometry.removed_at(zeros(rest),next) == zeros(Digits.before(rest)) : List<&2,Nat>}) -> {0n <> Geometry.removed_at(zeros(rest),next) == zeros(rest) : List<&2,Nat>}:
  match rest:
    case 1n+ +more: Equal.cong(List<&2,Nat>,List<&2,Nat>,list => 0n <> list,Geometry.removed_at(zeros(1n+more),next),zeros(more),later)
    case 0n: Empty.absurd({0n <> Geometry.removed_at(zeros(0n),next) == zeros(0n) : List<&2,Nat>},Order.below_zero(next,inside))

def removed_zeros(rank: Nat,axis: Nat,+inside: {Nat.is_lt(axis,rank) == True{} : Bool}) -> {Geometry.removed_at(zeros(rank),axis) == zeros(Digits.before(rank)) : List<&2,Nat>}:
  match rank axis:
    case 1n+ +rest 0n: {==}
    case 1n+ +rest 1n+ +next: removed_zeros_later(rest,next,inside,removed_zeros(rest,next,inside))
    case 0n +axis: Empty.absurd({Geometry.removed_at(zeros(0n),axis) == zeros(0n) : List<&2,Nat>},Order.below_zero(axis,inside))

def zeros_appended(rank: Nat) -> {List.append(&2,Nat,zeros(rank),[0n]) == zeros(1n+rank) : List<&2,Nat>}:
  match rank:
    case 0n: {==}
    case 1n+ +rest: Equal.cong(List<&2,Nat>,List<&2,Nat>,list => 0n <> list,List.append(&2,Nat,zeros(rest),[0n]),zeros(1n+rest),zeros_appended(rest))

def placed_zeros(position: Grid.Position,+rank: Nat) -> {Coordinates.placed_digits(True{},position,zeros(rank),0n) == zeros(1n+rank) : List<&2,Nat>}:
  match position:
    case Grid.Innermost{}: zeros_appended(rank)
    case Grid.Outermost{}: {==}

# Zero digits are in the first cell.
def cell_zeros(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) ->
  {cell_value(shape,axis,position,zeros(List.length(&2,U32,shape))) == 0n : Nat}:
  +rank = List.length(&2,U32,shape)
  +cells = Tiled.cell_shape(shape,axis,position)
  %Equal.sym(Nat,Digits.digit_at(zeros(rank),axis),0n,digit_zeros(rank,axis)) :
    {ShapeLaws.digits_value(cells,Coordinates.placed_digits(True{},position,Geometry.removed_at(zeros(rank),axis),Nat.div(_,8n))) == 0n : Nat}
  %Equal.sym(List<&2,Nat>,Geometry.removed_at(zeros(rank),axis),zeros(Digits.before(rank)),removed_zeros(rank,axis,within)) :
    {ShapeLaws.digits_value(cells,Coordinates.placed_digits(True{},position,_,0n)) == 0n : Nat}
  %Equal.sym(List<&2,Nat>,Coordinates.placed_digits(True{},position,zeros(Digits.before(rank)),0n),zeros(1n+Digits.before(rank)),placed_zeros(position,Digits.before(rank))) :
    {ShapeLaws.digits_value(cells,_) == 0n : Nat}
  zeros_value(cells,1n+Digits.before(rank))

# An amount at one axis times a count, the amount a multiple of eight at the
# lane axis, is in the cell of the count times the cell of the amount.
def cell_scaled(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+index: Nat,+amount: Nat,count: Nat,
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
  +aligned: {Nat.mod(Digits.digit_at(unit_digits(List.length(&2,U32,shape),index,amount),axis),8n) == 0n : Nat}) ->
  {cell_value(shape,axis,position,unit_digits(List.length(&2,U32,shape),index,Nat.mul(count,amount))) ==
    Nat.mul(count,cell_value(shape,axis,position,unit_digits(List.length(&2,U32,shape),index,amount))) : Nat}:
  match count:
    case 0n:
      +rank = List.length(&2,U32,shape)
      %Equal.sym(List<&2,Nat>,unit_digits(rank,index,0n),zeros(rank),unit_zero(rank,index)) : {cell_value(shape,axis,position,_) == 0n : Nat}
      cell_zeros(shape,axis,position,within)
    case 1n+ +more:
      +rank = List.length(&2,U32,shape)
      +unit = unit_digits(rank,index,amount)
      +later = unit_digits(rank,index,Nat.mul(more,amount))
      +one = cell_value(shape,axis,position,unit)
      +sum = Equal.trans(List<&2,Nat>,unit_digits(rank,index,Nat.add(amount,Nat.mul(more,amount))),unit_digits(rank,index,Nat.add(Nat.mul(more,amount),amount)),
        Digits.added(later,unit),
        Equal.cong(Nat,List<&2,Nat>,value => unit_digits(rank,index,value),Nat.add(amount,Nat.mul(more,amount)),Nat.add(Nat.mul(more,amount),amount),
          Order.commutative(amount,Nat.mul(more,amount))),
        Equal.sym(List<&2,Nat>,Digits.added(later,unit),unit_digits(rank,index,Nat.add(Nat.mul(more,amount),amount)),units_added(rank,index,Nat.mul(more,amount),amount)))
      %Equal.sym(List<&2,Nat>,unit_digits(rank,index,Nat.add(amount,Nat.mul(more,amount))),Digits.added(later,unit),sum) :
        {cell_value(shape,axis,position,_) == Nat.add(one,Nat.mul(more,one)) : Nat}
      Equal.trans(Nat,cell_value(shape,axis,position,Digits.added(later,unit)),Nat.add(cell_value(shape,axis,position,later),one),Nat.add(one,Nat.mul(more,one)),
        cell_added(shape,axis,position,later,unit,within,unit_length(rank,index,Nat.mul(more,amount)),unit_length(rank,index,amount),aligned),
        Equal.trans(Nat,Nat.add(cell_value(shape,axis,position,later),one),Nat.add(Nat.mul(more,one),one),Nat.add(one,Nat.mul(more,one)),
          Equal.cong(Nat,Nat,value => Nat.add(value,one),cell_value(shape,axis,position,later),Nat.mul(more,one),
            cell_scaled(shape,axis,position,index,amount,more,within,aligned)),
          Order.commutative(Nat.mul(more,one),one)))

# ---- The target axes a view keeps on the cells ----

def mapped_valid(mapped: TiledView.Mapped) -> Bool:
  TiledView.Mapped{valid,lane,lane_extent,axes,flags,reach} = mapped
  valid

def mapped_lane(mapped: TiledView.Mapped) -> Maybe<&2,Nat>:
  TiledView.Mapped{valid,lane,lane_extent,axes,flags,reach} = mapped
  lane

def mapped_extent(mapped: TiledView.Mapped) -> U32:
  TiledView.Mapped{valid,lane,lane_extent,axes,flags,reach} = mapped
  lane_extent

def mapped_kept(mapped: TiledView.Mapped) -> List<&2,Views.Axis>:
  TiledView.Mapped{valid,lane,lane_extent,axes,flags,reach} = mapped
  axes

def mapped_flags(mapped: TiledView.Mapped) -> List<&2,Bool>:
  TiledView.Mapped{valid,lane,lane_extent,axes,flags,reach} = mapped
  flags

def mapped_reach(mapped: TiledView.Mapped) -> List<&2,U32>:
  TiledView.Mapped{valid,lane,lane_extent,axes,flags,reach} = mapped
  reach

def sub_step(lane: Nat,index: Nat,+low: {Nat.is_le(1n+index,lane) == True{} : Bool}) -> {Nat.sub(lane,index) == 1n+Nat.sub(lane,1n+index) : Nat}:
  match lane index:
    case 1n+ +rest 0n: Equal.cong(Nat,Nat,value => 1n+value,rest,Nat.sub(rest,0n),Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)))
    case 1n+ +rest 1n+ +next: sub_step(rest,next,low)
    case 0n +index: Empty.absurd({Nat.sub(0n,index) == 1n+Nat.sub(0n,1n+index) : Nat},WordEncoding.false_true(low))

# A view valid on the cells is valid on the target axes after the first.
def tail_valid(reading: TiledView.Reading,mapped: TiledView.Mapped,+reversed: Bool,+index: Nat,+extent: U32,+lane_axis: Nat,
  +valid: {mapped_valid(TiledView.with_reading(reading,reversed,index,extent,lane_axis,mapped)) == True{} : Bool}) -> {mapped_valid(mapped) == True{} : Bool}:
  match reading mapped:
    case TiledView.Repeated{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}: valid
    case TiledView.Lane{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      Boolean.left(kept,Bool.not(reversed),Boolean.left(kept && Bool.not(reversed),Bool.not(Maybe.is_some(&2,Nat,lane)),valid))
    case TiledView.Stepping{+found,+step,+cells} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}: valid
    case TiledView.Unkept{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}: Empty.absurd({kept == True{} : Bool},WordEncoding.false_true(valid))

# The lane target axis of a suffix of the target axes: its place in the
# suffix, its extent, and the kept axes are the others.
type LaneAt<-index: Nat,-lane: Nat,-axes: List<&2,Views.Axis>,-extent: U32,-kept: List<&2,Views.Axis>> is Data:
  LaneAt{low: {Nat.is_le(index,lane) == True{} : Bool},
    inside: {Nat.is_lt(Nat.sub(lane,index),List.length(&2,Views.Axis,axes)) == True{} : Bool},
    extent_exact: {extent == Grid.lanes_at(Views.axis_extents(axes),Nat.sub(lane,index)) : U32},
    kept_exact: {Views.axis_extents(kept) == Grid.removed(Views.axis_extents(axes),Nat.sub(lane,index)) : List<&2,U32>}}

def Structure(lane: Maybe<&2,Nat>,+index: Nat,+axes: List<&2,Views.Axis>,+extent: U32,+kept: List<&2,Views.Axis>) -> Data:
  match lane:
    case None{}: {Views.axis_extents(kept) == Views.axis_extents(axes) : List<&2,U32>}
    case Some{+lane}: LaneAt<index,lane,axes,extent,kept>

type Mapping<-mapped: TiledView.Mapped,-index: Nat,-axes: List<&2,Views.Axis>> is Data:
  Mapping{lane: Structure(mapped_lane(mapped),index,axes,mapped_extent(mapped),mapped_kept(mapped)),
    flags: {List.length(&2,Bool,mapped_flags(mapped)) == List.length(&2,Views.Axis,mapped_kept(mapped)) : Nat}}

# A kept axis before the lane target axis.
def lane_later(lane: Maybe<&2,Nat>,+index: Nat,+extent: U32,+stride: U32,+cells: U32,+rest: List<&2,Views.Axis>,+lane_extent: U32,+kept: List<&2,Views.Axis>,
  +later: Structure(lane,1n+index,rest,lane_extent,kept)) ->
  Structure(lane,index,Con{Views.Axis{extent,stride},rest},lane_extent,Con{Views.Axis{extent,cells},kept}):
  match lane:
    case None{}: Equal.cong(List<&2,U32>,List<&2,U32>,list => extent <> list,Views.axis_extents(kept),Views.axis_extents(rest),later)
    case Some{+lane}:
      LaneAt{+low,+inside,+extent_exact,+kept_exact} = later
      +place = Nat.sub(lane,1n+index)
      +step = Equal.sym(Nat,Nat.sub(lane,index),1n+place,sub_step(lane,index,low))
      LaneAt{Order.transitive(index,1n+index,lane,Order.successor(index,index,Order.reflexive(index)),low),
        %step : {Nat.is_lt(_,1n+List.length(&2,Views.Axis,rest)) == True{} : Bool} inside,
        %step : {lane_extent == Grid.lanes_at(extent <> Views.axis_extents(rest),_) : U32} extent_exact,
        %step : {extent <> Views.axis_extents(kept) == Grid.removed(extent <> Views.axis_extents(rest),_) : List<&2,U32>}
          Equal.cong(List<&2,U32>,List<&2,U32>,list => extent <> list,Views.axis_extents(kept),Grid.removed(Views.axis_extents(rest),place),kept_exact)}

# The target axis that reads the lanes, with no lane target axis after it.
def lane_here(lane: Maybe<&2,Nat>,+index: Nat,+extent: U32,+stride: U32,+rest: List<&2,Views.Axis>,+lane_extent: U32,+kept: List<&2,Views.Axis>,
  +free: {Bool.not(Maybe.is_some(&2,Nat,lane)) == True{} : Bool},+later: Structure(lane,1n+index,rest,lane_extent,kept)) ->
  LaneAt<index,index,Con{Views.Axis{extent,stride},rest},extent,kept>:
  match lane:
    case None{}:
      +zero = Equal.sym(Nat,Nat.sub(index,index),0n,ViewReads.sub_self(index))
      LaneAt{Order.reflexive(index),
        %zero : {Nat.is_lt(_,1n+List.length(&2,Views.Axis,rest)) == True{} : Bool} {==},
        %zero : {extent == Grid.lanes_at(extent <> Views.axis_extents(rest),_) : U32} {==},
        %zero : {Views.axis_extents(kept) == Grid.removed(extent <> Views.axis_extents(rest),_) : List<&2,U32>} later}
    case Some{+found}: Empty.absurd(LaneAt<index,index,Con{Views.Axis{extent,stride},rest},extent,kept>,WordEncoding.false_true(free))

def mapping_step(reading: TiledView.Reading,mapped: TiledView.Mapped,+reversed: Bool,+index: Nat,+extent: U32,+stride: U32,+lane_axis: Nat,+rest: List<&2,Views.Axis>,
  +valid: {mapped_valid(TiledView.with_reading(reading,reversed,index,extent,lane_axis,mapped)) == True{} : Bool},+later: Mapping<mapped,1n+index,rest>) ->
  Mapping<TiledView.with_reading(reading,reversed,index,extent,lane_axis,mapped),index,Con{Views.Axis{extent,stride},rest}>:
  match reading mapped:
    case TiledView.Repeated{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      Mapping{+shape,+counted} = later
      Mapping{lane_later(lane,index,extent,stride,0,rest,lane_extent,axes,shape),
        Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Bool,flags),List.length(&2,Views.Axis,axes),counted)}
    case TiledView.Lane{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      Mapping{+shape,+counted} = later
      Mapping{lane_here(lane,index,extent,stride,rest,lane_extent,axes,Boolean.right(kept && Bool.not(reversed),Bool.not(Maybe.is_some(&2,Nat,lane)),valid),shape),counted}
    case TiledView.Stepping{+found,+step,+cells} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      Mapping{+shape,+counted} = later
      Mapping{lane_later(lane,index,extent,stride,cells,rest,lane_extent,axes,shape),
        Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Bool,flags),List.length(&2,Views.Axis,axes),counted)}
    case TiledView.Unkept{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      Empty.absurd(Mapping<TiledView.Mapped{False{},lane,lane_extent,axes,flags,reach},index,Con{Views.Axis{extent,stride},rest}>,WordEncoding.false_true(valid))

# The kept axes of a valid view: the target axes but the lane target axis,
# one flag each.
def mapping(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+shape: List<&2,U32>,+lanes: TiledView.Lanes,+start: List<&2,U32>,
  +valid: {mapped_valid(TiledView.mapped_axes(axes,flags,index,shape,lanes,start)) == True{} : Bool}) ->
  Mapping<TiledView.mapped_axes(axes,flags,index,shape,lanes,start),index,axes>:
  match axes:
    case Nil{}: Mapping{{==},{==}}
    case Con{Views.Axis{+extent,+stride},+rest}:
      +reading = TiledView.classified(shape,lanes,extent,stride)
      +tail = TiledView.mapped_axes(rest,Views.drop_flags(1n,flags),1n+index,shape,lanes,start)
      mapping_step(reading,tail,Views.flag(flags),index,extent,stride,TiledView.lane_axis(lanes),rest,valid,
        mapping(rest,Views.drop_flags(1n,flags),1n+index,shape,lanes,start,tail_valid(reading,tail,Views.flag(flags),index,extent,TiledView.lane_axis(lanes),valid)))

# ---- The source digits a view reads ----

# The source digits a target axis adds at the amount it reads: none when it
# repeats, the amount at the lane axis when it reads the lanes and those
# count, the amount times the step at the axis it steps.
def contribution(reading: TiledView.Reading,+rank: Nat,+lane: Nat,+with_lane: Bool,+amount: Nat) -> List<&2,Nat>:
  match reading:
    case TiledView.Repeated{}: zeros(rank)
    case TiledView.Lane{}: Bool.pick(List<&2,Nat>,with_lane,unit_digits(rank,lane,amount),zeros(rank))
    case TiledView.Stepping{+found,+step,+cells}: unit_digits(rank,U32.to_nat(found),Nat.mul(amount,U32.to_nat(step)))
    case TiledView.Unkept{}: zeros(rank)

# The source digits a view reads at target digits from start digits.
def walked(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+shape: List<&2,U32>,+lanes: TiledView.Lanes,+lane: Nat,+with_lane: Bool,+start: List<&2,Nat>,
  digits: List<&2,Nat>) -> List<&2,Nat>:
  match axes digits:
    case Con{Views.Axis{+extent,+stride},rest} Con{+digit,more}:
      Digits.added(walked(rest,Views.later_flags(flags),shape,lanes,lane,with_lane,start,more),
        contribution(TiledView.classified(shape,lanes,extent,stride),List.length(&2,U32,shape),lane,with_lane,
          MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),digit)))
    case axes digits: start

def pick_length(choice: Bool,+left: List<&2,Nat>,+right: List<&2,Nat>,+rank: Nat,+left_length: {List.length(&2,Nat,left) == rank : Nat},
  +right_length: {List.length(&2,Nat,right) == rank : Nat}) -> {List.length(&2,Nat,Bool.pick(List<&2,Nat>,choice,left,right)) == rank : Nat}:
  match choice:
    case True{}: left_length
    case False{}: right_length

def contribution_length(reading: TiledView.Reading,+rank: Nat,+lane: Nat,+with_lane: Bool,+amount: Nat) ->
  {List.length(&2,Nat,contribution(reading,rank,lane,with_lane,amount)) == rank : Nat}:
  match reading:
    case TiledView.Repeated{}: zeros_length(rank)
    case TiledView.Lane{}: pick_length(with_lane,unit_digits(rank,lane,amount),zeros(rank),rank,unit_length(rank,lane,amount),zeros_length(rank))
    case TiledView.Stepping{+found,+step,+cells}: unit_length(rank,U32.to_nat(found),Nat.mul(amount,U32.to_nat(step)))
    case TiledView.Unkept{}: zeros_length(rank)

def walked_length(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+shape: List<&2,U32>,+lanes: TiledView.Lanes,+lane: Nat,+with_lane: Bool,+start: List<&2,Nat>,
  digits: List<&2,Nat>,+start_length: {List.length(&2,Nat,start) == List.length(&2,U32,shape) : Nat}) ->
  {List.length(&2,Nat,walked(axes,flags,shape,lanes,lane,with_lane,start,digits)) == List.length(&2,U32,shape) : Nat}:
  match axes digits:
    case Con{Views.Axis{+extent,+stride},+rest} Con{+digit,+more}:
      +later = walked(rest,Views.later_flags(flags),shape,lanes,lane,with_lane,start,more)
      +amount = MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),digit)
      +added = contribution(TiledView.classified(shape,lanes,extent,stride),List.length(&2,U32,shape),lane,with_lane,amount)
      +tail = walked_length(rest,Views.later_flags(flags),shape,lanes,lane,with_lane,start,more,start_length)
      Equal.trans(Nat,List.length(&2,Nat,Digits.added(later,added)),List.length(&2,Nat,later),List.length(&2,U32,shape),
        Digits.added_length(later,added,Equal.trans(Nat,List.length(&2,Nat,later),List.length(&2,U32,shape),List.length(&2,Nat,added),tail,
          Equal.sym(Nat,List.length(&2,Nat,added),List.length(&2,U32,shape),
            contribution_length(TiledView.classified(shape,lanes,extent,stride),List.length(&2,U32,shape),lane,with_lane,amount)))),
        tail)
    case Nil{} +digits: start_length
    case Con{Views.Axis{+extent,+stride},+rest} Nil{}: start_length

# What the values of a gather source give every target axis.
type SourceFacts<-limits: Machine.Limits,-shape: List<&2,U32>,-axis: Nat,-position: Grid.Position> is Data:
  SourceFacts{within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},
    rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    room: {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(shape,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    nonzero: {Shapes.has_zero(shape) == False{} : Bool}}

def axis_facts(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+facts: SourceFacts<limits,shape,axis,position>,+extent: U32,+stride: U32) ->
  Classified(TiledView.classified(shape,TiledView.lanes_of(shape,axis,position),extent,stride),shape,axis,extent,stride,TiledView.lanes_of(shape,axis,position)):
  SourceFacts{+within,+rank_room,+room,+cells_room,+nonzero} = facts
  classified_facts(shape,axis,position,extent,stride,limits,Bounds.strict_le(axis,List.length(&2,U32,shape),within),rank_room,room,nonzero)

def source_within(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+facts: SourceFacts<limits,shape,axis,position>) ->
  {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}:
  SourceFacts{+within,+rank_room,+room,+cells_room,+nonzero} = facts
  within

def kept_reading(reading: TiledView.Reading) -> Bool:
  match reading:
    case TiledView.Repeated{}: True{}
    case TiledView.Lane{}: True{}
    case TiledView.Stepping{+found,+step,+cells}: True{}
    case TiledView.Unkept{}: False{}

def valid_kept(reading: TiledView.Reading,mapped: TiledView.Mapped,+reversed: Bool,+index: Nat,+extent: U32,+lane_axis: Nat,
  +valid: {mapped_valid(TiledView.with_reading(reading,reversed,index,extent,lane_axis,mapped)) == True{} : Bool}) -> {kept_reading(reading) == True{} : Bool}:
  match reading mapped:
    case TiledView.Repeated{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}: {==}
    case TiledView.Lane{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}: {==}
    case TiledView.Stepping{+found,+step,+cells} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}: {==}
    case TiledView.Unkept{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}: Empty.absurd({False{} == True{} : Bool},WordEncoding.false_true(valid))

# ---- The number a view reads ----

def reflected_one(reverse: Bool) -> {MirrorAddress.reflected_coordinate(reverse,1n,0n) == 0n : Nat}:
  match reverse:
    case False{}: {==}
    case True{}: {==}

# A repeated axis adds nothing: its stride is zero, or its one index reflects
# to zero.
def repeated_term(zero: Bool,+stride: U32,+extent: U32,+reverse: Bool,+digit: Nat,+classified: {U32.is_zero(stride) == zero : Bool},
  +either: {zero || U32.is_eq(extent,1) == True{} : Bool},+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool}) ->
  {0n == Nat.mul(MirrorAddress.reflected_coordinate(reverse,U32.to_nat(extent),digit),U32.to_nat(stride)) : Nat}:
  match zero:
    case True{}:
      %Equal.sym(U32,stride,0,Arithmetic.word_zero(stride,classified)) : {0n == Nat.mul(MirrorAddress.reflected_coordinate(reverse,U32.to_nat(extent),digit),U32.to_nat(_)) : Nat}
      Equal.sym(Nat,Nat.mul(MirrorAddress.reflected_coordinate(reverse,U32.to_nat(extent),digit),0n),0n,RowMajor.multiply_zero_right(MirrorAddress.reflected_coordinate(reverse,U32.to_nat(extent),digit)))
    case False{}:
      +one = Order.equal_of_is_eq(U32.to_nat(extent),1n,Equal.trans(Bool,Nat.is_eq(U32.to_nat(extent),1n),U32.is_eq(extent,1),True{},
        Equal.sym(Bool,U32.is_eq(extent,1),Nat.is_eq(U32.to_nat(extent),1n),Comparison.eq(extent,1)),either))
      +below = {%one : {Nat.is_lt(digit,_) == True{} : Bool} inside : {Nat.is_lt(digit,1n) == True{} : Bool}}
      +first = Order.zero_below(digit,Division.lt_succ_le(digit,0n,below))
      %Equal.sym(Nat,U32.to_nat(extent),1n,one) : {0n == Nat.mul(MirrorAddress.reflected_coordinate(reverse,_,digit),U32.to_nat(stride)) : Nat}
      %Equal.sym(Nat,digit,0n,first) : {0n == Nat.mul(MirrorAddress.reflected_coordinate(reverse,1n,_),U32.to_nat(stride)) : Nat}
      %Equal.sym(Nat,MirrorAddress.reflected_coordinate(reverse,1n,0n),0n,reflected_one(reverse)) : {0n == Nat.mul(_,U32.to_nat(stride)) : Nat}
      {==}

# The number of the digits a target axis adds is its term of the view sum.
def logical_term(reading: TiledView.Reading,+shape: List<&2,U32>,+axis: Nat,+extent: U32,+stride: U32,+lanes: TiledView.Lanes,+reverse: Bool,+digit: Nat,
  +facts: Classified(reading,shape,axis,extent,stride,lanes),+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+kept: {kept_reading(reading) == True{} : Bool}) ->
  {ShapeLaws.digits_value(shape,contribution(reading,List.length(&2,U32,shape),axis,True{},MirrorAddress.reflected_coordinate(reverse,U32.to_nat(extent),digit))) ==
    Nat.mul(MirrorAddress.reflected_coordinate(reverse,U32.to_nat(extent),digit),U32.to_nat(stride)) : Nat}:
  match reading:
    case TiledView.Repeated{}:
      +amount = MirrorAddress.reflected_coordinate(reverse,U32.to_nat(extent),digit)
      %Equal.sym(Nat,ShapeLaws.digits_value(shape,zeros(List.length(&2,U32,shape))),0n,zeros_value(shape,List.length(&2,U32,shape))) : {_ == Nat.mul(amount,U32.to_nat(stride)) : Nat}
      repeated_term(U32.is_zero(stride),stride,extent,reverse,digit,{==},facts,inside)
    case TiledView.Lane{}:
      +amount = MirrorAddress.reflected_coordinate(reverse,U32.to_nat(extent),digit)
      +after = ShapeLaws.elements(Public.lane_after(shape,axis))
      Equal.trans(Nat,ShapeLaws.digits_value(shape,unit_digits(List.length(&2,U32,shape),axis,amount)),Nat.mul(amount,after),Nat.mul(amount,U32.to_nat(stride)),
        unit_value(shape,axis,amount,within),Equal.cong(Nat,Nat,value => Nat.mul(amount,value),after,U32.to_nat(stride),Equal.sym(Nat,U32.to_nat(stride),after,facts)))
    case TiledView.Stepping{+found,+step,+cells}:
      KernelGemm.Both{+stepping,+rest} = facts
      StepAt{+place,+axis_exact,+place_inside,+stride_exact} = stepping
      +amount = MirrorAddress.reflected_coordinate(reverse,U32.to_nat(extent),digit)
      +after = ShapeLaws.elements(Public.lane_after(shape,place))
      %Equal.sym(Nat,U32.to_nat(found),place,axis_exact) :
        {ShapeLaws.digits_value(shape,unit_digits(List.length(&2,U32,shape),_,Nat.mul(amount,U32.to_nat(step)))) == Nat.mul(amount,U32.to_nat(stride)) : Nat}
      Equal.trans(Nat,ShapeLaws.digits_value(shape,unit_digits(List.length(&2,U32,shape),place,Nat.mul(amount,U32.to_nat(step)))),Nat.mul(Nat.mul(amount,U32.to_nat(step)),after),
        Nat.mul(amount,U32.to_nat(stride)),unit_value(shape,place,Nat.mul(amount,U32.to_nat(step)),place_inside),
        Equal.trans(Nat,Nat.mul(Nat.mul(amount,U32.to_nat(step)),after),Nat.mul(amount,Nat.mul(U32.to_nat(step),after)),Nat.mul(amount,U32.to_nat(stride)),
          RowMajor.assoc(amount,U32.to_nat(step),after),
          Equal.cong(Nat,Nat,value => Nat.mul(amount,value),Nat.mul(U32.to_nat(step),after),U32.to_nat(stride),Equal.sym(Nat,U32.to_nat(stride),Nat.mul(U32.to_nat(step),after),stride_exact))))
    case TiledView.Unkept{}:
      Empty.absurd({ShapeLaws.digits_value(shape,zeros(List.length(&2,U32,shape))) == Nat.mul(MirrorAddress.reflected_coordinate(reverse,U32.to_nat(extent),digit),U32.to_nat(stride)) : Nat},
        WordEncoding.false_true(kept))

def sum_moved(+base: Nat,+later: Nat,+term: Nat) -> {Nat.add(Nat.add(base,later),term) == Nat.add(base,Nat.add(term,later)) : Nat}:
  Equal.trans(Nat,Nat.add(Nat.add(base,later),term),Nat.add(base,Nat.add(later,term)),Nat.add(base,Nat.add(term,later)),Addition.associative(base,later,term),
    Equal.cong(Nat,Nat,value => Nat.add(base,value),Nat.add(later,term),Nat.add(term,later),Order.commutative(later,term)))

# The number of the source digits a valid view reads at target digits is the
# number of the start and the view sum at the digits.
def logical_walk(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,
  +facts: SourceFacts<limits,shape,axis,position>,+words: List<&2,U32>,+start: List<&2,Nat>,digits: List<&2,Nat>,
  +start_length: {List.length(&2,Nat,start) == List.length(&2,U32,shape) : Nat},+below: Coordinates.DigitsBelow(Views.axis_extents(axes),digits),
  +valid: {mapped_valid(TiledView.mapped_axes(axes,flags,index,shape,TiledView.lanes_of(shape,axis,position),words)) == True{} : Bool}) ->
  {ShapeLaws.digits_value(shape,walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,True{},start,digits)) ==
    Nat.add(ShapeLaws.digits_value(shape,start),digit_sum(axes,flags,digits)) : Nat}:
  match axes digits:
    case Con{Views.Axis{+extent,+stride},+rest} Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      +within = source_within(limits,shape,axis,position,facts)
      +lanes = TiledView.lanes_of(shape,axis,position)
      +reading = TiledView.classified(shape,lanes,extent,stride)
      +tail_mapped = TiledView.mapped_axes(rest,Views.drop_flags(1n,flags),1n+index,shape,lanes,words)
      +amount = MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),digit)
      +rank = List.length(&2,U32,shape)
      +walk = walked(rest,Views.later_flags(flags),shape,lanes,axis,True{},start,more)
      +added = contribution(reading,rank,axis,True{},amount)
      +base = ShapeLaws.digits_value(shape,start)
      +tail = logical_walk(rest,Views.later_flags(flags),1n+index,shape,axis,position,limits,facts,words,start,more,start_length,later,
        tail_valid(reading,tail_mapped,Views.flag(flags),index,extent,TiledView.lane_axis(lanes),valid))
      +term = logical_term(reading,shape,axis,extent,stride,lanes,Views.flag(flags),digit,axis_facts(limits,shape,axis,position,facts,extent,stride),here,within,
        valid_kept(reading,tail_mapped,Views.flag(flags),index,extent,TiledView.lane_axis(lanes),valid))
      +walk_length = walked_length(rest,Views.later_flags(flags),shape,lanes,axis,True{},start,more,start_length)
      +added_length = contribution_length(reading,rank,axis,True{},amount)
      +split = digits_added(shape,walk,added,walk_length,added_length)
      +rest_sum = digit_sum(rest,Views.later_flags(flags),more)
      +own = Nat.mul(amount,U32.to_nat(stride))
      Equal.trans(Nat,ShapeLaws.digits_value(shape,Digits.added(walk,added)),Nat.add(ShapeLaws.digits_value(shape,walk),ShapeLaws.digits_value(shape,added)),
        Nat.add(base,Nat.add(own,rest_sum)),split,
        Equal.trans(Nat,Nat.add(ShapeLaws.digits_value(shape,walk),ShapeLaws.digits_value(shape,added)),Nat.add(Nat.add(base,rest_sum),own),Nat.add(base,Nat.add(own,rest_sum)),
          %Equal.sym(Nat,ShapeLaws.digits_value(shape,walk),Nat.add(base,rest_sum),tail) : {Nat.add(_,ShapeLaws.digits_value(shape,added)) == Nat.add(Nat.add(base,rest_sum),own) : Nat}
            Equal.cong(Nat,Nat,value => Nat.add(Nat.add(base,rest_sum),value),ShapeLaws.digits_value(shape,added),own,term),
          sum_moved(base,rest_sum,own)))
    case Nil{} +digits: Equal.sym(Nat,Nat.add(ShapeLaws.digits_value(shape,start),0n),ShapeLaws.digits_value(shape,start),Addition.zero_right(ShapeLaws.digits_value(shape,start)))
    case Con{Views.Axis{+extent,+stride},+rest} Nil{}:
      Empty.absurd({ShapeLaws.digits_value(shape,start) == Nat.add(ShapeLaws.digits_value(shape,start),0n) : Nat},below)

# ---- Bounds and groups of digits ----

def source_room(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+facts: SourceFacts<limits,shape,axis,position>) ->
  {Nat.is_le(ShapeLaws.elements(shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  SourceFacts{+within,+rank_room,+room,+cells_room,+nonzero} = facts
  room

def source_cells_room(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+facts: SourceFacts<limits,shape,axis,position>) ->
  {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(shape,axis,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  SourceFacts{+within,+rank_room,+room,+cells_room,+nonzero} = facts
  cells_room

def below_both(+extent: U32,+rest: List<&2,U32>,+x: Nat,+xs: List<&2,Nat>,+y: Nat,+ys: List<&2,Nat>,+here: {Nat.is_lt(Nat.add(x,y),U32.to_nat(extent)) == True{} : Bool},
  +later: KernelGemm.Both<Coordinates.DigitsBelow(rest,xs),Coordinates.DigitsBelow(rest,ys)>) ->
  KernelGemm.Both<Coordinates.DigitsBelow(Con{extent,rest},Con{x,xs}),Coordinates.DigitsBelow(Con{extent,rest},Con{y,ys})>:
  KernelGemm.Both{+left_later,+right_later} = later
  KernelGemm.Both{KernelGemm.Both{Order.le_lt_trans(x,Nat.add(x,y),U32.to_nat(extent),Order.le_add(x,y),here),left_later},
    KernelGemm.Both{Order.le_lt_trans(y,Nat.add(x,y),U32.to_nat(extent),Order.right_le(x,y),here),right_later}}

# Digits whose sum lies below extents lie below them each.
def below_added(shape: List<&2,U32>,left: List<&2,Nat>,right: List<&2,Nat>,+below: Coordinates.DigitsBelow(shape,Digits.added(left,right)),
  +same: {List.length(&2,Nat,right) == List.length(&2,Nat,left) : Nat}) -> KernelGemm.Both<Coordinates.DigitsBelow(shape,left),Coordinates.DigitsBelow(shape,right)>:
  match shape left right:
    case Con{+extent,+rest} Con{+x,+xs} Con{+y,+ys}:
      KernelGemm.Both{+here,+later} = below
      below_both(extent,rest,x,xs,y,ys,here,below_added(rest,xs,ys,later,length_tail(List.length(&2,Nat,ys),List.length(&2,Nat,xs),same)))
    case Nil{} Nil{} Nil{}: KernelGemm.Both{Unit{},Unit{}}
    case Nil{} Con{+x,+xs} Con{+y,+ys}:
      Empty.absurd(KernelGemm.Both<Coordinates.DigitsBelow(Nil{},Con{x,xs}),Coordinates.DigitsBelow(Nil{},Con{y,ys})>,below)
    case Nil{} Nil{} Con{+y,+ys}:
      Empty.absurd(KernelGemm.Both<Coordinates.DigitsBelow(Nil{},Nil{}),Coordinates.DigitsBelow(Nil{},Con{y,ys})>,successor_zero(List.length(&2,Nat,ys),same))
    case Nil{} Con{+x,+xs} Nil{}:
      Empty.absurd(KernelGemm.Both<Coordinates.DigitsBelow(Nil{},Con{x,xs}),Coordinates.DigitsBelow(Nil{},Nil{})>,Digits.zero_successor(List.length(&2,Nat,xs),same))
    case Con{+extent,+rest} Nil{} +right:
      Empty.absurd(KernelGemm.Both<Coordinates.DigitsBelow(Con{extent,rest},Nil{}),Coordinates.DigitsBelow(Con{extent,rest},right)>,below)
    case Con{+extent,+rest} Con{+x,+xs} Nil{}:
      Empty.absurd(KernelGemm.Both<Coordinates.DigitsBelow(Con{extent,rest},Con{x,xs}),Coordinates.DigitsBelow(Con{extent,rest},Nil{})>,below)

# A smaller amount at one axis lies below extents a larger one lies below.
def unit_below_le(shape: List<&2,U32>,index: Nat,+small: Nat,+large: Nat,+ordered: {Nat.is_le(small,large) == True{} : Bool},
  +below: Coordinates.DigitsBelow(shape,unit_digits(List.length(&2,U32,shape),index,large))) ->
  Coordinates.DigitsBelow(shape,unit_digits(List.length(&2,U32,shape),index,small)):
  match shape index:
    case Con{+extent,+rest} 0n:
      KernelGemm.Both{+here,+later} = below
      KernelGemm.Both{Order.le_lt_trans(small,large,U32.to_nat(extent),ordered,here),later}
    case Con{+extent,+rest} 1n+ +next:
      KernelGemm.Both{+here,+later} = below
      KernelGemm.Both{here,unit_below_le(rest,next,small,large,ordered,later)}
    case Nil{} +index: Unit{}

def aligned_exact(+value: Nat,+zero: {Nat.mod(value,8n) == 0n : Nat}) -> {value == Nat.mul(Nat.div(value,8n),8n) : Nat}:
  +quotient = Nat.div(value,8n)
  Equal.trans(Nat,value,Nat.add(Nat.mul(quotient,8n),Nat.mod(value,8n)),Nat.mul(quotient,8n),DivisionOrder.quotient_exact(value,8n,{==}),
    Equal.trans(Nat,Nat.add(Nat.mul(quotient,8n),Nat.mod(value,8n)),Nat.add(Nat.mul(quotient,8n),0n),Nat.mul(quotient,8n),
      Equal.cong(Nat,Nat,rest => Nat.add(Nat.mul(quotient,8n),rest),Nat.mod(value,8n),0n,zero),Addition.zero_right(Nat.mul(quotient,8n))))

def mod_sum(+first: Nat,+second: Nat,+first_zero: {Nat.mod(first,8n) == 0n : Nat},+second_zero: {Nat.mod(second,8n) == 0n : Nat}) ->
  {Nat.mod(Nat.add(first,second),8n) == 0n : Nat}:
  +quotient = Nat.div(second,8n)
  %Equal.sym(Nat,second,Nat.mul(quotient,8n),aligned_exact(second,second_zero)) : {Nat.mod(Nat.add(first,_),8n) == 0n : Nat}
  %Order.commutative(Nat.mul(quotient,8n),first) : {Nat.mod(_,8n) == 0n : Nat}
  Equal.trans(Nat,Nat.mod(Nat.add(Nat.mul(quotient,8n),first),8n),Nat.mod(first,8n),0n,DivisionOrder.remainder_shift(first,8n,quotient,{==}),first_zero)

def aligned_times(+count: Nat,+value: Nat,+zero: {Nat.mod(value,8n) == 0n : Nat}) -> {Nat.mod(Nat.mul(count,value),8n) == 0n : Nat}:
  +quotient = Nat.div(value,8n)
  +product = Nat.mul(count,quotient)
  %Equal.sym(Nat,value,Nat.mul(quotient,8n),aligned_exact(value,zero)) : {Nat.mod(Nat.mul(count,_),8n) == 0n : Nat}
  %RowMajor.assoc(count,quotient,8n) : {Nat.mod(_,8n) == 0n : Nat}
  %Addition.zero_right(Nat.mul(product,8n)) : {Nat.mod(_,8n) == 0n : Nat}
  DivisionOrder.remainder_shift(0n,8n,product,{==})

# The digit of an amount at one axis is the amount on that axis and zero on
# the others.
def digit_unit(rank: Nat,index: Nat,axis: Nat,+amount: Nat,+inside: {Nat.is_lt(axis,rank) == True{} : Bool}) ->
  {Digits.digit_at(unit_digits(rank,index,amount),axis) == Bool.pick(Nat,Nat.is_eq(index,axis),amount,0n) : Nat}:
  match rank index axis:
    case 1n+ +rest 0n 0n: {==}
    case 1n+ +rest 0n 1n+ +next: digit_zeros(rest,next)
    case 1n+ +rest 1n+ +found 0n: {==}
    case 1n+ +rest 1n+ +found 1n+ +next: digit_unit(rest,found,next,amount,inside)
    case 0n +index +axis:
      Empty.absurd({Digits.digit_at(unit_digits(0n,index,amount),axis) == Bool.pick(Nat,Nat.is_eq(index,axis),amount,0n) : Nat},Order.below_zero(axis,inside))

def aligned_pick(choice: Bool,+amount: Nat,+when: Logic.When(choice,{Nat.mod(amount,8n) == 0n : Nat})) -> {Nat.mod(Bool.pick(Nat,choice,amount,0n),8n) == 0n : Nat}:
  match choice:
    case True{}: when
    case False{}: {==}

def unit_aligned(+rank: Nat,+index: Nat,+axis: Nat,+amount: Nat,+inside: {Nat.is_lt(axis,rank) == True{} : Bool},
  +when: Logic.When(Nat.is_eq(index,axis),{Nat.mod(amount,8n) == 0n : Nat})) -> {Nat.mod(Digits.digit_at(unit_digits(rank,index,amount),axis),8n) == 0n : Nat}:
  %Equal.sym(Nat,Digits.digit_at(unit_digits(rank,index,amount),axis),Bool.pick(Nat,Nat.is_eq(index,axis),amount,0n),digit_unit(rank,index,axis,amount,inside)) :
    {Nat.mod(_,8n) == 0n : Nat}
  aligned_pick(Nat.is_eq(index,axis),amount,when)

def step_nat(+limits: Machine.Limits,+step: U32,+zero: {U32.is_zero((step % 8 : U32)) == True{} : Bool}) -> {Nat.mod(U32.to_nat(step),8n) == 0n : Nat}:
  +remainder = U32Division.modulo(step,8,Machine.half_word(limits),Limits.half_word_fits(limits),Bounds.as_nat_le(16,Machine.half_word(limits),Limits.sixteen_within_half(limits)),{==})
  Equal.trans(Nat,Nat.mod(U32.to_nat(step),8n),U32.to_nat((step % 8 : U32)),0n,Equal.sym(Nat,U32.to_nat((step % 8 : U32)),Nat.mod(U32.to_nat(step),8n),remainder),
    Equal.cong(U32,Nat,word => U32.to_nat(word),(step % 8 : U32),0,Arithmetic.word_zero((step % 8 : U32),zero)))

def when_nat(choice: Bool,+limits: Machine.Limits,+step: U32,+when: Logic.When(choice,{U32.is_zero((step % 8 : U32)) == True{} : Bool})) ->
  Logic.When(choice,{Nat.mod(U32.to_nat(step),8n) == 0n : Nat}):
  match choice:
    case True{}: step_nat(limits,step,when)
    case False{}: Unit{}

def when_times(choice: Bool,+count: Nat,+value: Nat,+when: Logic.When(choice,{Nat.mod(value,8n) == 0n : Nat})) ->
  Logic.When(choice,{Nat.mod(Nat.mul(count,value),8n) == 0n : Nat}):
  match choice:
    case True{}: aligned_times(count,value,when)
    case False{}: Unit{}

def mirror(reverse: Bool,+extent: Nat) -> Nat:
  match reverse:
    case False{}: Nat.sub(extent,1n)
    case True{}: 0n

# The last digits of a view: the digits its last coordinates reflect from.
def mirrored_lasts(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>) -> List<&2,Nat>:
  match axes:
    case Nil{}: []
    case Con{Views.Axis{+extent,+stride},rest}: mirror(Views.flag(flags),U32.to_nat(extent)) <> mirrored_lasts(rest,Views.later_flags(flags))

def reflected_mirror(reverse: Bool,+extent: Nat) -> {MirrorAddress.reflected_coordinate(reverse,extent,mirror(reverse,extent)) == Nat.sub(extent,1n) : Nat}:
  match reverse:
    case False{}: {==}
    case True{}: Order.subtract_zero(Nat.sub(extent,1n))

def added_zeros_of(+digits: List<&2,Nat>,+rank: Nat,+length: {List.length(&2,Nat,digits) == rank : Nat}) -> {Digits.added(digits,zeros(rank)) == digits : List<&2,Nat>}:
  %length : {Digits.added(digits,zeros(_)) == digits : List<&2,Nat>}
  added_zeros(digits)

def not_true(value: Bool,+true: {Bool.not(value) == True{} : Bool}) -> {value == False{} : Bool}:
  match value:
    case False{}: {==}
    case True{}: Empty.absurd({True{} == False{} : Bool},WordEncoding.false_true(true))

# ---- The cell a view reads ----

# The amount a stepping axis reads, times the cells of its step, is the cell
# of its source digits.
def stepping_cells(amount: Nat,+found: U32,+step: U32,+cells: U32,+stride: U32,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,
  +facts: SourceFacts<limits,shape,axis,position>,
  +classified: KernelGemm.Both<StepAt<shape,stride,0n,found,step>,KernelGemm.Both<{cells == TiledView.cell_index(TiledView.lanes_of(shape,axis,position),stride) : U32},
    Logic.When(Nat.is_eq(U32.to_nat(found),axis),{U32.is_zero((step % 8 : U32)) == True{} : Bool})>>,
  +below: Coordinates.DigitsBelow(shape,unit_digits(List.length(&2,U32,shape),U32.to_nat(found),Nat.mul(amount,U32.to_nat(step))))) ->
  {Nat.mul(amount,U32.to_nat(cells)) == cell_value(shape,axis,position,unit_digits(List.length(&2,U32,shape),U32.to_nat(found),Nat.mul(amount,U32.to_nat(step)))) : Nat}:
  match amount:
    case 0n:
      +rank = List.length(&2,U32,shape)
      %Equal.sym(List<&2,Nat>,unit_digits(rank,U32.to_nat(found),0n),zeros(rank),unit_zero(rank,U32.to_nat(found))) : {0n == cell_value(shape,axis,position,_) : Nat}
      Equal.sym(Nat,cell_value(shape,axis,position,zeros(rank)),0n,cell_zeros(shape,axis,position,source_within(limits,shape,axis,position,facts)))
    case 1n+ +more:
      KernelGemm.Both{+stepping,+rest} = classified
      StepAt{+place,+axis_exact,+place_inside,+stride_exact} = stepping
      KernelGemm.Both{+cells_exact,+when} = rest
      +within = source_within(limits,shape,axis,position,facts)
      +rank = List.length(&2,U32,shape)
      +size = U32.to_nat(step)
      +index = U32.to_nat(found)
      +lanes = TiledView.lanes_of(shape,axis,position)
      +unit = unit_digits(rank,index,size)
      +after = ShapeLaws.elements(Public.lane_after(shape,place))
      +unit_below = unit_below_le(shape,index,size,Nat.mul(1n+more,size),Order.le_add(size,Nat.mul(more,size)),below)
      +value = {%Equal.sym(Nat,index,place,axis_exact) : {ShapeLaws.digits_value(shape,unit_digits(rank,_,size)) == Nat.mul(size,after) : Nat}
        unit_value(shape,place,size,place_inside) : {ShapeLaws.digits_value(shape,unit) == Nat.mul(size,after) : Nat}}
      +stride_value = Equal.trans(Nat,U32.to_nat(stride),Nat.mul(size,after),ShapeLaws.digits_value(shape,unit),stride_exact,
        Equal.sym(Nat,ShapeLaws.digits_value(shape,unit),Nat.mul(size,after),value))
      +stride_word = Equal.trans(U32,stride,U32.from_nat(U32.to_nat(stride)),U32.from_nat(ShapeLaws.digits_value(shape,unit)),
        Equal.sym(U32,U32.from_nat(U32.to_nat(stride)),stride,Arithmetic.native_roundtrip(stride)),
        Equal.cong(Nat,U32,number => U32.from_nat(number),U32.to_nat(stride),ShapeLaws.digits_value(shape,unit),stride_value))
      +cell_one = Equal.trans(Nat,U32.to_nat(cells),U32.to_nat(TiledView.cell_index(lanes,stride)),cell_value(shape,axis,position,unit),
        Equal.cong(U32,Nat,word => U32.to_nat(word),cells,TiledView.cell_index(lanes,stride),cells_exact),
        {%Equal.sym(U32,stride,U32.from_nat(ShapeLaws.digits_value(shape,unit)),stride_word) : {U32.to_nat(TiledView.cell_index(lanes,_)) == cell_value(shape,axis,position,unit) : Nat}
          cell_of(shape,axis,position,unit,limits,within,unit_below,source_room(limits,shape,axis,position,facts),source_cells_room(limits,shape,axis,position,facts)) :
          {U32.to_nat(TiledView.cell_index(lanes,stride)) == cell_value(shape,axis,position,unit) : Nat}})
      +aligned = unit_aligned(rank,index,axis,size,within,when_nat(Nat.is_eq(index,axis),limits,step,when))
      Equal.trans(Nat,Nat.mul(1n+more,U32.to_nat(cells)),Nat.mul(1n+more,cell_value(shape,axis,position,unit)),
        cell_value(shape,axis,position,unit_digits(rank,index,Nat.mul(1n+more,size))),
        Equal.cong(Nat,Nat,number => Nat.mul(1n+more,number),U32.to_nat(cells),cell_value(shape,axis,position,unit),cell_one),
        Equal.sym(Nat,cell_value(shape,axis,position,unit_digits(rank,index,Nat.mul(1n+more,size))),Nat.mul(1n+more,cell_value(shape,axis,position,unit)),
          cell_scaled(shape,axis,position,index,size,1n+more,within,aligned)))

def lane_place(lane: Maybe<&2,Nat>,+index: Nat,+count: Nat) -> Nat:
  match lane:
    case None{}: count
    case Some{+found}: Nat.sub(found,index)

def place_step(lane: Maybe<&2,Nat>,+index: Nat,+rest: List<&2,Views.Axis>,+lane_extent: U32,+kept: List<&2,Views.Axis>,
  +later: Structure(lane,1n+index,rest,lane_extent,kept)) ->
  {lane_place(lane,index,1n+List.length(&2,Views.Axis,rest)) == 1n+lane_place(lane,1n+index,List.length(&2,Views.Axis,rest)) : Nat}:
  match lane:
    case None{}: {==}
    case Some{+found}:
      LaneAt{+low,+inside,+extent_exact,+kept_exact} = later
      sub_step(found,index,low)

# What a valid view reads of the cells at target digits: its view of the
# cells reads the cell of the source digits but the lane target axis, whose
# lane digit is a multiple of eight, and the source digits are those with the
# lane target digit added at the lane axis.
type CellWalk<-shape: List<&2,U32>,-axis: Nat,-position: Grid.Position,-lane: Maybe<&2,Nat>,-kept: List<&2,Views.Axis>,-kept_flags: List<&2,Bool>,-index: Nat,
  -count: Nat,-start: List<&2,Nat>,-digits: List<&2,Nat>,-cell_walk: List<&2,Nat>,-full_walk: List<&2,Nat>> is Data:
  CellWalk{cells: {Nat.add(cell_value(shape,axis,position,start),digit_sum(kept,kept_flags,Geometry.removed_at(digits,lane_place(lane,index,count)))) ==
      cell_value(shape,axis,position,cell_walk) : Nat},
    aligned: {Nat.mod(Digits.digit_at(cell_walk,axis),8n) == 0n : Nat},
    split: {full_walk == Digits.added(cell_walk,unit_digits(List.length(&2,U32,shape),axis,Digits.digit_at(digits,lane_place(lane,index,count)))) : List<&2,Nat>}}

# The target axis that reads the lanes.
def lane_step(lane: Maybe<&2,Nat>,+reversed: Bool,+index: Nat,+extent: U32,+rest: List<&2,Views.Axis>,+digit: Nat,+more: List<&2,Nat>,+shape: List<&2,U32>,+axis: Nat,
  +position: Grid.Position,+start: List<&2,Nat>,+cell_walk: List<&2,Nat>,+full_walk: List<&2,Nat>,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +later: CellWalk<shape,axis,position,lane,kept,kept_flags,1n+index,List.length(&2,Views.Axis,rest),start,more,cell_walk,full_walk>,
  +free: {Bool.not(Maybe.is_some(&2,Nat,lane)) == True{} : Bool},+forward: {Bool.not(reversed) == True{} : Bool},
  +more_length: {List.length(&2,Nat,more) == List.length(&2,Views.Axis,rest) : Nat},+cell_length: {List.length(&2,Nat,cell_walk) == List.length(&2,U32,shape) : Nat}) ->
  CellWalk<shape,axis,position,Some{index},kept,kept_flags,index,1n+List.length(&2,Views.Axis,rest),start,Con{digit,more},
    Digits.added(cell_walk,contribution(TiledView.Lane{},List.length(&2,U32,shape),axis,False{},MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),digit))),
    Digits.added(full_walk,contribution(TiledView.Lane{},List.length(&2,U32,shape),axis,True{},MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),digit)))>:
  match lane:
    case None{}:
      CellWalk{+cells,+aligned,+split} = later
      +rank = List.length(&2,U32,shape)
      +count = List.length(&2,Views.Axis,rest)
      +past = {%more_length : {Nat.is_le(List.length(&2,Nat,more),_) == True{} : Bool} Order.reflexive(List.length(&2,Nat,more)) :
        {Nat.is_le(List.length(&2,Nat,more),count) == True{} : Bool}}
      +zero = Equal.sym(Nat,Nat.sub(index,index),0n,ViewReads.sub_self(index))
      +kept_zeros = Equal.sym(List<&2,Nat>,Digits.added(cell_walk,zeros(rank)),cell_walk,added_zeros_of(cell_walk,rank,cell_length))
      +full_same = Equal.trans(List<&2,Nat>,full_walk,Digits.added(cell_walk,unit_digits(rank,axis,0n)),cell_walk,
        {%Values.digit_beyond(more,count,past) : {full_walk == Digits.added(cell_walk,unit_digits(rank,axis,_)) : List<&2,Nat>}
          split : {full_walk == Digits.added(cell_walk,unit_digits(rank,axis,0n)) : List<&2,Nat>}},
        {%Equal.sym(List<&2,Nat>,unit_digits(rank,axis,0n),zeros(rank),unit_zero(rank,axis)) : {Digits.added(cell_walk,_) == cell_walk : List<&2,Nat>}
          added_zeros_of(cell_walk,rank,cell_length) : {Digits.added(cell_walk,unit_digits(rank,axis,0n)) == cell_walk : List<&2,Nat>}})
      CellWalk{
        %zero : {Nat.add(cell_value(shape,axis,position,start),digit_sum(kept,kept_flags,Geometry.removed_at(Con{digit,more},_))) ==
          cell_value(shape,axis,position,Digits.added(cell_walk,zeros(rank))) : Nat}
        %kept_zeros : {Nat.add(cell_value(shape,axis,position,start),digit_sum(kept,kept_flags,more)) == cell_value(shape,axis,position,_) : Nat}
        %Geometry.removed_beyond(more,count,past) :
          {Nat.add(cell_value(shape,axis,position,start),digit_sum(kept,kept_flags,_)) == cell_value(shape,axis,position,cell_walk) : Nat}
        cells,
        %kept_zeros : {Nat.mod(Digits.digit_at(_,axis),8n) == 0n : Nat} aligned,
        %Equal.sym(Bool,reversed,False{},not_true(reversed,forward)) :
          {Digits.added(full_walk,unit_digits(rank,axis,MirrorAddress.reflected_coordinate(_,U32.to_nat(extent),digit))) ==
            Digits.added(Digits.added(cell_walk,zeros(rank)),unit_digits(rank,axis,Digits.digit_at(Con{digit,more},Nat.sub(index,index)))) : List<&2,Nat>}
        %zero : {Digits.added(full_walk,unit_digits(rank,axis,digit)) == Digits.added(Digits.added(cell_walk,zeros(rank)),unit_digits(rank,axis,Digits.digit_at(Con{digit,more},_))) : List<&2,Nat>}
        %kept_zeros : {Digits.added(full_walk,unit_digits(rank,axis,digit)) == Digits.added(_,unit_digits(rank,axis,digit)) : List<&2,Nat>}
        %Equal.sym(List<&2,Nat>,full_walk,cell_walk,full_same) : {Digits.added(_,unit_digits(rank,axis,digit)) == Digits.added(cell_walk,unit_digits(rank,axis,digit)) : List<&2,Nat>}
        {==}}
    case Some{+found}:
      Empty.absurd(CellWalk<shape,axis,position,Some{index},kept,kept_flags,index,1n+List.length(&2,Views.Axis,rest),start,Con{digit,more},
        Digits.added(cell_walk,contribution(TiledView.Lane{},List.length(&2,U32,shape),axis,False{},MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),digit))),
        Digits.added(full_walk,contribution(TiledView.Lane{},List.length(&2,U32,shape),axis,True{},MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),digit)))>,
        WordEncoding.false_true(free))

def first_below(+shape: List<&2,U32>,+left: List<&2,Nat>,+right: List<&2,Nat>,+both: KernelGemm.Both<Coordinates.DigitsBelow(shape,left),Coordinates.DigitsBelow(shape,right)>) ->
  Coordinates.DigitsBelow(shape,left):
  KernelGemm.Both{+first,+second} = both
  first

def second_below(+shape: List<&2,U32>,+left: List<&2,Nat>,+right: List<&2,Nat>,+both: KernelGemm.Both<Coordinates.DigitsBelow(shape,left),Coordinates.DigitsBelow(shape,right)>) ->
  Coordinates.DigitsBelow(shape,right):
  KernelGemm.Both{+first,+second} = both
  second

# One target axis more of the cells a valid view reads.
def cell_step(reading: TiledView.Reading,mapped: TiledView.Mapped,+reversed: Bool,+index: Nat,+extent: U32,+stride: U32,+rest: List<&2,Views.Axis>,+digit: Nat,
  +more: List<&2,Nat>,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,+facts: SourceFacts<limits,shape,axis,position>,
  +start: List<&2,Nat>,+cell_walk: List<&2,Nat>,+full_walk: List<&2,Nat>,
  +later: CellWalk<shape,axis,position,mapped_lane(mapped),mapped_kept(mapped),mapped_flags(mapped),1n+index,List.length(&2,Views.Axis,rest),start,more,cell_walk,full_walk>,
  +tail_shape: Mapping<mapped,1n+index,rest>,
  +valid: {mapped_valid(TiledView.with_reading(reading,reversed,index,extent,TiledView.lane_axis(TiledView.lanes_of(shape,axis,position)),mapped)) == True{} : Bool},
  +classified: Classified(reading,shape,axis,extent,stride,TiledView.lanes_of(shape,axis,position)),+inside: {Nat.is_lt(digit,U32.to_nat(extent)) == True{} : Bool},
  +max_below: Coordinates.DigitsBelow(shape,contribution(reading,List.length(&2,U32,shape),axis,True{},
    MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),mirror(reversed,U32.to_nat(extent))))),
  +more_length: {List.length(&2,Nat,more) == List.length(&2,Views.Axis,rest) : Nat},+cell_length: {List.length(&2,Nat,cell_walk) == List.length(&2,U32,shape) : Nat},
  +full_length: {List.length(&2,Nat,full_walk) == List.length(&2,U32,shape) : Nat}) ->
  CellWalk<shape,axis,position,mapped_lane(TiledView.with_reading(reading,reversed,index,extent,TiledView.lane_axis(TiledView.lanes_of(shape,axis,position)),mapped)),
    mapped_kept(TiledView.with_reading(reading,reversed,index,extent,TiledView.lane_axis(TiledView.lanes_of(shape,axis,position)),mapped)),
    mapped_flags(TiledView.with_reading(reading,reversed,index,extent,TiledView.lane_axis(TiledView.lanes_of(shape,axis,position)),mapped)),
    index,1n+List.length(&2,Views.Axis,rest),start,Con{digit,more},
    Digits.added(cell_walk,contribution(reading,List.length(&2,U32,shape),axis,False{},MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),digit))),
    Digits.added(full_walk,contribution(reading,List.length(&2,U32,shape),axis,True{},MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),digit)))>:
  match reading mapped:
    case TiledView.Repeated{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      CellWalk{+cells,+aligned,+split} = later
      Mapping{+structure,+counted} = tail_shape
      +rank = List.length(&2,U32,shape)
      +count = List.length(&2,Views.Axis,rest)
      +place = lane_place(lane,1n+index,count)
      +amount = MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),digit)
      +step_back = Equal.sym(Nat,lane_place(lane,index,1n+count),1n+place,place_step(lane,index,rest,lane_extent,axes,structure))
      +zeros_kept = Equal.sym(List<&2,Nat>,Digits.added(cell_walk,zeros(rank)),cell_walk,added_zeros_of(cell_walk,rank,cell_length))
      +zeros_full = Equal.sym(List<&2,Nat>,Digits.added(full_walk,zeros(rank)),full_walk,added_zeros_of(full_walk,rank,full_length))
      CellWalk{
        %step_back : {Nat.add(cell_value(shape,axis,position,start),digit_sum(Con{Views.Axis{extent,0},axes},Con{reversed,flags},Geometry.removed_at(Con{digit,more},_))) ==
          cell_value(shape,axis,position,Digits.added(cell_walk,zeros(rank))) : Nat}
        %zeros_kept : {Nat.add(cell_value(shape,axis,position,start),Nat.add(Nat.mul(amount,0n),digit_sum(axes,flags,Geometry.removed_at(more,place)))) ==
          cell_value(shape,axis,position,_) : Nat}
        %Equal.sym(Nat,Nat.mul(amount,0n),0n,RowMajor.multiply_zero_right(amount)) :
          {Nat.add(cell_value(shape,axis,position,start),Nat.add(_,digit_sum(axes,flags,Geometry.removed_at(more,place)))) == cell_value(shape,axis,position,cell_walk) : Nat}
        cells,
        %zeros_kept : {Nat.mod(Digits.digit_at(_,axis),8n) == 0n : Nat} aligned,
        %step_back : {Digits.added(full_walk,zeros(rank)) == Digits.added(Digits.added(cell_walk,zeros(rank)),unit_digits(rank,axis,Digits.digit_at(Con{digit,more},_))) : List<&2,Nat>}
        %zeros_kept : {Digits.added(full_walk,zeros(rank)) == Digits.added(_,unit_digits(rank,axis,Digits.digit_at(more,place))) : List<&2,Nat>}
        %zeros_full : {_ == Digits.added(cell_walk,unit_digits(rank,axis,Digits.digit_at(more,place))) : List<&2,Nat>}
        split}
    case TiledView.Lane{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      +free = Boolean.right(kept && Bool.not(reversed),Bool.not(Maybe.is_some(&2,Nat,lane)),valid)
      +forward = Boolean.right(kept,Bool.not(reversed),Boolean.left(kept && Bool.not(reversed),Bool.not(Maybe.is_some(&2,Nat,lane)),valid))
      lane_step(lane,reversed,index,extent,rest,digit,more,shape,axis,position,start,cell_walk,full_walk,axes,flags,later,free,forward,more_length,cell_length)
    case TiledView.Stepping{+found,+step,+cells_word} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      CellWalk{+cells,+aligned,+split} = later
      Mapping{+structure,+counted} = tail_shape
      KernelGemm.Both{+stepping,+rest_facts} = classified
      KernelGemm.Both{+cells_exact,+when} = rest_facts
      +within = source_within(limits,shape,axis,position,facts)
      +rank = List.length(&2,U32,shape)
      +count = List.length(&2,Views.Axis,rest)
      +place = lane_place(lane,1n+index,count)
      +size = U32.to_nat(step)
      +found_index = U32.to_nat(found)
      +amount = MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),digit)
      +last = Nat.sub(U32.to_nat(extent),1n)
      +added = unit_digits(rank,found_index,Nat.mul(amount,size))
      +lane_unit = unit_digits(rank,axis,Digits.digit_at(more,place))
      +step_back = Equal.sym(Nat,lane_place(lane,index,1n+count),1n+place,place_step(lane,index,rest,lane_extent,axes,structure))
      +largest = {%reflected_mirror(reversed,U32.to_nat(extent)) : Coordinates.DigitsBelow(shape,unit_digits(rank,found_index,Nat.mul(_,size)))
        max_below : Coordinates.DigitsBelow(shape,unit_digits(rank,found_index,Nat.mul(last,size)))}
      +added_below = unit_below_le(shape,found_index,Nat.mul(amount,size),Nat.mul(last,size),
        RowMajor.mul_le(amount,last,size,ViewReads.reflected_le(reversed,U32.to_nat(extent),digit,inside)),largest)
      +term = stepping_cells(amount,found,step,cells_word,stride,shape,axis,position,limits,facts,KernelGemm.Both{stepping,KernelGemm.Both{cells_exact,when}},added_below)
      +aligned_added = unit_aligned(rank,found_index,axis,Nat.mul(amount,size),within,
        when_times(Nat.is_eq(found_index,axis),amount,size,when_nat(Nat.is_eq(found_index,axis),limits,step,when)))
      +added_length = unit_length(rank,found_index,Nat.mul(amount,size))
      +cell_inside = {%Equal.sym(Nat,List.length(&2,Nat,cell_walk),rank,cell_length) : {Nat.is_lt(axis,_) == True{} : Bool} within :
        {Nat.is_lt(axis,List.length(&2,Nat,cell_walk)) == True{} : Bool}}
      +base = cell_value(shape,axis,position,start)
      +later_sum = digit_sum(axes,flags,Geometry.removed_at(more,place))
      +own = Nat.mul(amount,U32.to_nat(cells_word))
      CellWalk{
        %step_back : {Nat.add(base,digit_sum(Con{Views.Axis{extent,cells_word},axes},Con{reversed,flags},Geometry.removed_at(Con{digit,more},_))) ==
          cell_value(shape,axis,position,Digits.added(cell_walk,added)) : Nat}
        Equal.trans(Nat,Nat.add(base,Nat.add(own,later_sum)),Nat.add(Nat.add(base,later_sum),own),cell_value(shape,axis,position,Digits.added(cell_walk,added)),
          Equal.sym(Nat,Nat.add(Nat.add(base,later_sum),own),Nat.add(base,Nat.add(own,later_sum)),sum_moved(base,later_sum,own)),
          Equal.trans(Nat,Nat.add(Nat.add(base,later_sum),own),Nat.add(cell_value(shape,axis,position,cell_walk),cell_value(shape,axis,position,added)),
            cell_value(shape,axis,position,Digits.added(cell_walk,added)),
            %Equal.sym(Nat,Nat.add(base,later_sum),cell_value(shape,axis,position,cell_walk),cells) : {Nat.add(_,own) == Nat.add(cell_value(shape,axis,position,cell_walk),cell_value(shape,axis,position,added)) : Nat}
              Equal.cong(Nat,Nat,number => Nat.add(cell_value(shape,axis,position,cell_walk),number),own,cell_value(shape,axis,position,added),term),
            Equal.sym(Nat,cell_value(shape,axis,position,Digits.added(cell_walk,added)),Nat.add(cell_value(shape,axis,position,cell_walk),cell_value(shape,axis,position,added)),
              cell_added(shape,axis,position,cell_walk,added,within,cell_length,added_length,aligned_added)))),
        %Equal.sym(Nat,Digits.digit_at(Digits.added(cell_walk,added),axis),Nat.add(Digits.digit_at(cell_walk,axis),Digits.digit_at(added,axis)),
            Digits.added_digit(cell_walk,added,axis,cell_inside,lengths_same(added,cell_walk,shape,added_length,cell_length))) :
          {Nat.mod(_,8n) == 0n : Nat}
        mod_sum(Digits.digit_at(cell_walk,axis),Digits.digit_at(added,axis),aligned,aligned_added),
        %step_back : {Digits.added(full_walk,added) == Digits.added(Digits.added(cell_walk,added),unit_digits(rank,axis,Digits.digit_at(Con{digit,more},_))) : List<&2,Nat>}
        %Equal.sym(List<&2,Nat>,full_walk,Digits.added(cell_walk,lane_unit),split) : {Digits.added(_,added) == Digits.added(Digits.added(cell_walk,added),lane_unit) : List<&2,Nat>}
        added_swap(cell_walk,lane_unit,added,lengths_same(cell_walk,lane_unit,shape,cell_length,unit_length(rank,axis,Digits.digit_at(more,place))),
          lengths_same(cell_walk,added,shape,cell_length,added_length))}
    case TiledView.Unkept{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      Empty.absurd(CellWalk<shape,axis,position,lane,axes,flags,index,1n+List.length(&2,Views.Axis,rest),start,Con{digit,more},
        Digits.added(cell_walk,zeros(List.length(&2,U32,shape))),Digits.added(full_walk,zeros(List.length(&2,U32,shape)))>,WordEncoding.false_true(valid))

# What a valid view whose last source digits lie below the values reads of
# the cells at target digits below its extents.
def cell_walk(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,
  +facts: SourceFacts<limits,shape,axis,position>,+words: List<&2,U32>,+start: List<&2,Nat>,digits: List<&2,Nat>,
  +start_length: {List.length(&2,Nat,start) == List.length(&2,U32,shape) : Nat},+start_aligned: {Nat.mod(Digits.digit_at(start,axis),8n) == 0n : Nat},
  +below: Coordinates.DigitsBelow(Views.axis_extents(axes),digits),
  +valid: {mapped_valid(TiledView.mapped_axes(axes,flags,index,shape,TiledView.lanes_of(shape,axis,position),words)) == True{} : Bool},
  +bound: Coordinates.DigitsBelow(shape,walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,True{},start,mirrored_lasts(axes,flags)))) ->
  CellWalk<shape,axis,position,mapped_lane(TiledView.mapped_axes(axes,flags,index,shape,TiledView.lanes_of(shape,axis,position),words)),
    mapped_kept(TiledView.mapped_axes(axes,flags,index,shape,TiledView.lanes_of(shape,axis,position),words)),
    mapped_flags(TiledView.mapped_axes(axes,flags,index,shape,TiledView.lanes_of(shape,axis,position),words)),index,List.length(&2,Views.Axis,axes),start,digits,
    walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,False{},start,digits),walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,True{},start,digits)>:
  match axes digits:
    case Con{Views.Axis{+extent,+stride},+rest} Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      +lanes = TiledView.lanes_of(shape,axis,position)
      +reading = TiledView.classified(shape,lanes,extent,stride)
      +tail_mapped = TiledView.mapped_axes(rest,Views.drop_flags(1n,flags),1n+index,shape,lanes,words)
      +rank = List.length(&2,U32,shape)
      +tail_ok = tail_valid(reading,tail_mapped,Views.flag(flags),index,extent,TiledView.lane_axis(lanes),valid)
      +maximum = walked(rest,Views.later_flags(flags),shape,lanes,axis,True{},start,mirrored_lasts(rest,Views.later_flags(flags)))
      +largest = contribution(reading,rank,axis,True{},MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),mirror(Views.flag(flags),U32.to_nat(extent))))
      +maximum_length = walked_length(rest,Views.later_flags(flags),shape,lanes,axis,True{},start,mirrored_lasts(rest,Views.later_flags(flags)),start_length)
      +bounds = below_added(shape,maximum,largest,bound,
        lengths_same(maximum,largest,shape,maximum_length,
          contribution_length(reading,rank,axis,True{},MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),mirror(Views.flag(flags),U32.to_nat(extent))))))
      +tail = cell_walk(rest,Views.later_flags(flags),1n+index,shape,axis,position,limits,facts,words,start,more,start_length,start_aligned,later,tail_ok,
        first_below(shape,maximum,largest,bounds))
      +more_length = Equal.trans(Nat,List.length(&2,Nat,more),List.length(&2,U32,Views.axis_extents(rest)),List.length(&2,Views.Axis,rest),
        Values.below_length(Views.axis_extents(rest),more,later),MetadataLaws.extents_length(rest))
      cell_step(reading,tail_mapped,Views.flag(flags),index,extent,stride,rest,digit,more,shape,axis,position,limits,facts,start,
        walked(rest,Views.later_flags(flags),shape,lanes,axis,False{},start,more),walked(rest,Views.later_flags(flags),shape,lanes,axis,True{},start,more),
        tail,mapping(rest,Views.drop_flags(1n,flags),1n+index,shape,lanes,words,tail_ok),valid,axis_facts(limits,shape,axis,position,facts,extent,stride),here,
        second_below(shape,maximum,largest,bounds),more_length,
        walked_length(rest,Views.later_flags(flags),shape,lanes,axis,False{},start,more,start_length),
        walked_length(rest,Views.later_flags(flags),shape,lanes,axis,True{},start,more,start_length))
    case Nil{} Nil{}:
      +rank = List.length(&2,U32,shape)
      CellWalk{Addition.zero_right(cell_value(shape,axis,position,start)),
        start_aligned,
        %Equal.sym(List<&2,Nat>,unit_digits(rank,axis,0n),zeros(rank),unit_zero(rank,axis)) : {start == Digits.added(start,_) : List<&2,Nat>}
        Equal.sym(List<&2,Nat>,Digits.added(start,zeros(rank)),start,added_zeros_of(start,rank,start_length))}
    case Nil{} Con{+digit,+more}:
      Empty.absurd(CellWalk<shape,axis,position,None{},Nil{},Nil{},index,0n,start,Con{digit,more},start,start>,below)
    case Con{Views.Axis{+extent,+stride},+rest} Nil{}:
      Empty.absurd(CellWalk<shape,axis,position,
        mapped_lane(TiledView.mapped_axes(Con{Views.Axis{extent,stride},rest},flags,index,shape,TiledView.lanes_of(shape,axis,position),words)),
        mapped_kept(TiledView.mapped_axes(Con{Views.Axis{extent,stride},rest},flags,index,shape,TiledView.lanes_of(shape,axis,position),words)),
        mapped_flags(TiledView.mapped_axes(Con{Views.Axis{extent,stride},rest},flags,index,shape,TiledView.lanes_of(shape,axis,position),words)),index,
        1n+List.length(&2,Views.Axis,rest),start,Nil{},start,start>,below)

# ---- Digits no larger axis by axis ----

def DigitsLe(left: List<&2,Nat>,right: List<&2,Nat>) -> Data:
  match left right:
    case Con{+digit,left} Con{+other,right}: KernelGemm.Both<{Nat.is_le(digit,other) == True{} : Bool},DigitsLe(left,right)>
    case Nil{} Nil{}: Unit
    case left right: Empty

def le_reflexive(digits: List<&2,Nat>) -> DigitsLe(digits,digits):
  match digits:
    case Con{+digit,+rest}: KernelGemm.Both{Order.reflexive(digit),le_reflexive(rest)}
    case Nil{}: Unit{}

def zeros_le(rank: Nat) -> DigitsLe(zeros(rank),zeros(rank)):
  le_reflexive(zeros(rank))

def unit_le(rank: Nat,axis: Nat,+small: Nat,+large: Nat,+ordered: {Nat.is_le(small,large) == True{} : Bool}) ->
  DigitsLe(unit_digits(rank,axis,small),unit_digits(rank,axis,large)):
  match rank axis:
    case 1n+ +rest 0n: KernelGemm.Both{ordered,zeros_le(rest)}
    case 1n+ +rest 1n+ +next: KernelGemm.Both{Order.reflexive(0n),unit_le(rest,next,small,large,ordered)}
    case 0n +axis: Unit{}

def sum_le(+a: Nat,+b: Nat,+c: Nat,+d: Nat,+first: {Nat.is_le(a,b) == True{} : Bool},+second: {Nat.is_le(c,d) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(a,c),Nat.add(b,d)) == True{} : Bool}:
  Order.transitive(Nat.add(a,c),Nat.add(b,c),Nat.add(b,d),Order.add_le(a,b,c,first),Order.offset_le(b,c,d,second))

def added_le(first: List<&2,Nat>,larger: List<&2,Nat>,second: List<&2,Nat>,other: List<&2,Nat>,+first_le: DigitsLe(first,larger),+second_le: DigitsLe(second,other)) ->
  DigitsLe(Digits.added(first,second),Digits.added(larger,other)):
  match first larger second other:
    case Con{+a,+as} Con{+b,+bs} Con{+c,+cs} Con{+d,+ds}:
      KernelGemm.Both{+head,+tail} = first_le
      KernelGemm.Both{+here,+later} = second_le
      KernelGemm.Both{sum_le(a,b,c,d,head,here),added_le(as,bs,cs,ds,tail,later)}
    case Nil{} Nil{} +second +other: Unit{}
    case Con{+a,+as} Con{+b,+bs} Nil{} Nil{}: Unit{}
    case Nil{} Con{+b,+bs} +second +other: Empty.absurd(DigitsLe(Digits.added(Nil{},second),Digits.added(Con{b,bs},other)),first_le)
    case Con{+a,+as} Nil{} +second +other: Empty.absurd(DigitsLe(Digits.added(Con{a,as},second),Digits.added(Nil{},other)),first_le)
    case Con{+a,+as} Con{+b,+bs} Nil{} Con{+d,+ds}: Empty.absurd(DigitsLe(Digits.added(Con{a,as},Nil{}),Digits.added(Con{b,bs},Con{d,ds})),second_le)
    case Con{+a,+as} Con{+b,+bs} Con{+c,+cs} Nil{}: Empty.absurd(DigitsLe(Digits.added(Con{a,as},Con{c,cs}),Digits.added(Con{b,bs},Nil{})),second_le)

# Digits no larger than digits below extents lie below them.
def below_le(shape: List<&2,U32>,small: List<&2,Nat>,large: List<&2,Nat>,+ordered: DigitsLe(small,large),+below: Coordinates.DigitsBelow(shape,large)) ->
  Coordinates.DigitsBelow(shape,small):
  match shape small large:
    case Con{+extent,+rest} Con{+x,+xs} Con{+y,+ys}:
      KernelGemm.Both{+head,+tail} = ordered
      KernelGemm.Both{+here,+later} = below
      KernelGemm.Both{Order.le_lt_trans(x,y,U32.to_nat(extent),head,here),below_le(rest,xs,ys,tail,later)}
    case Nil{} Nil{} +large: Unit{}
    case Nil{} Con{+x,+xs} Nil{}: Empty.absurd(Coordinates.DigitsBelow(Nil{},Con{x,xs}),ordered)
    case Nil{} Con{+x,+xs} Con{+y,+ys}: Empty.absurd(Coordinates.DigitsBelow(Nil{},Con{x,xs}),below)
    case Con{+extent,+rest} Nil{} +large:
      match large:
        case Nil{}: Empty.absurd(Coordinates.DigitsBelow(Con{extent,rest},Nil{}),below)
        case Con{+y,+ys}: Empty.absurd(Coordinates.DigitsBelow(Con{extent,rest},Nil{}),ordered)
    case Con{+extent,+rest} Con{+x,+xs} Nil{}: Empty.absurd(Coordinates.DigitsBelow(Con{extent,rest},Con{x,xs}),ordered)

def contribution_le(reading: TiledView.Reading,+rank: Nat,+lane: Nat,+small: Nat,+large: Nat,+ordered: {Nat.is_le(small,large) == True{} : Bool}) ->
  DigitsLe(contribution(reading,rank,lane,True{},small),contribution(reading,rank,lane,True{},large)):
  match reading:
    case TiledView.Repeated{}: zeros_le(rank)
    case TiledView.Lane{}: unit_le(rank,lane,small,large,ordered)
    case TiledView.Stepping{+found,+step,+cells}:
      unit_le(rank,U32.to_nat(found),Nat.mul(small,U32.to_nat(step)),Nat.mul(large,U32.to_nat(step)),RowMajor.mul_le(small,large,U32.to_nat(step),ordered))
    case TiledView.Unkept{}: zeros_le(rank)

# The source digits a view reads at target digits below its extents are no
# larger than those at its last digits.
def walked_le(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+shape: List<&2,U32>,+lanes: TiledView.Lanes,+lane: Nat,+start: List<&2,Nat>,digits: List<&2,Nat>,
  +below: Coordinates.DigitsBelow(Views.axis_extents(axes),digits)) ->
  DigitsLe(walked(axes,flags,shape,lanes,lane,True{},start,digits),walked(axes,flags,shape,lanes,lane,True{},start,mirrored_lasts(axes,flags))):
  match axes digits:
    case Con{Views.Axis{+extent,+stride},+rest} Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      +reading = TiledView.classified(shape,lanes,extent,stride)
      +rank = List.length(&2,U32,shape)
      +last = Nat.sub(U32.to_nat(extent),1n)
      +amount = MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),digit)
      +largest = MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),mirror(Views.flag(flags),U32.to_nat(extent)))
      added_le(walked(rest,Views.later_flags(flags),shape,lanes,lane,True{},start,more),
        walked(rest,Views.later_flags(flags),shape,lanes,lane,True{},start,mirrored_lasts(rest,Views.later_flags(flags))),
        contribution(reading,rank,lane,True{},amount),contribution(reading,rank,lane,True{},largest),
        walked_le(rest,Views.later_flags(flags),shape,lanes,lane,start,more,later),
        contribution_le(reading,rank,lane,amount,largest,
          {%Equal.sym(Nat,largest,last,reflected_mirror(Views.flag(flags),U32.to_nat(extent))) : {Nat.is_le(amount,_) == True{} : Bool}
            ViewReads.reflected_le(Views.flag(flags),U32.to_nat(extent),digit,here) : {Nat.is_le(amount,largest) == True{} : Bool}}))
    case Nil{} +digits: le_reflexive(start)
    case Con{Views.Axis{+extent,+stride},+rest} Nil{}:
      Empty.absurd(DigitsLe(start,walked(Con{Views.Axis{extent,stride},rest},flags,shape,lanes,lane,True{},start,mirrored_lasts(Con{Views.Axis{extent,stride},rest},flags))),below)

# ---- How far a view reaches ----

def AllAtMost(digits: List<&2,Nat>,+bound: Nat) -> Data:
  match digits:
    case Con{+digit,rest}: KernelGemm.Both<{Nat.is_le(digit,bound) == True{} : Bool},AllAtMost(rest,bound)>
    case Nil{}: Unit

def at_most_le(digits: List<&2,Nat>,+bound: Nat,+larger: Nat,+within: AllAtMost(digits,bound),+ordered: {Nat.is_le(bound,larger) == True{} : Bool}) -> AllAtMost(digits,larger):
  match digits:
    case Con{+digit,+rest}:
      KernelGemm.Both{+here,+later} = within
      KernelGemm.Both{Order.transitive(digit,bound,larger,here,ordered),at_most_le(rest,bound,larger,later,ordered)}
    case Nil{}: Unit{}

def at_most_smaller(small: List<&2,Nat>,large: List<&2,Nat>,+bound: Nat,+ordered: DigitsLe(small,large),+within: AllAtMost(large,bound)) -> AllAtMost(small,bound):
  match small large:
    case Con{+x,+xs} Con{+y,+ys}:
      KernelGemm.Both{+head,+tail} = ordered
      KernelGemm.Both{+here,+later} = within
      KernelGemm.Both{Order.transitive(x,y,bound,head,here),at_most_smaller(xs,ys,bound,tail,later)}
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{+y,+ys}: Empty.absurd(Unit,ordered)
    case Con{+x,+xs} Nil{}: Empty.absurd(AllAtMost(Con{x,xs},bound),ordered)

def at_most_digit(digits: List<&2,Nat>,axis: Nat,+bound: Nat,+within: AllAtMost(digits,bound)) -> {Nat.is_le(Digits.digit_at(digits,axis),bound) == True{} : Bool}:
  match digits axis:
    case Con{+digit,+rest} 0n:
      KernelGemm.Both{+here,+later} = within
      here
    case Con{+digit,+rest} 1n+ +next:
      KernelGemm.Both{+here,+later} = within
      at_most_digit(rest,next,bound,later)
    case Nil{} +axis: Order.zero_le(bound)

# Every digit of digits of extents with no zero extent is at most their
# number.
def at_most_value(shape: List<&2,U32>,digits: List<&2,Nat>,+nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +length: {List.length(&2,Nat,digits) == List.length(&2,U32,shape) : Nat}) -> AllAtMost(digits,ShapeLaws.digits_value(shape,digits)):
  match shape digits:
    case Con{+extent,+rest} Con{+digit,+more}:
      +inner = ShapeLaws.elements(rest)
      +later_value = ShapeLaws.digits_value(rest,more)
      +total = Nat.add(Nat.mul(digit,inner),later_value)
      +rest_nonzero = Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonzero)
      KernelGemm.Both{Order.transitive(digit,Nat.mul(digit,inner),total,RowMajor.product_covers(digit,inner,CountCompletion.positive_elements(rest,rest_nonzero)),
          Order.le_add(Nat.mul(digit,inner),later_value)),
        at_most_le(more,later_value,total,at_most_value(rest,more,rest_nonzero,length_tail(List.length(&2,Nat,more),List.length(&2,U32,rest),length)),
          Order.right_le(Nat.mul(digit,inner),later_value))}
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{+digit,+more}: Empty.absurd(AllAtMost(Con{digit,more},ShapeLaws.digits_value(Nil{},Con{digit,more})),successor_zero(List.length(&2,Nat,more),length))
    case Con{+extent,+rest} Nil{}: Empty.absurd(Unit,Digits.zero_successor(List.length(&2,U32,rest),length))

def le_added(left: List<&2,Nat>,right: List<&2,Nat>,+same: {List.length(&2,Nat,right) == List.length(&2,Nat,left) : Nat}) -> DigitsLe(left,Digits.added(left,right)):
  match left right:
    case Con{+x,+xs} Con{+y,+ys}: KernelGemm.Both{Order.le_add(x,y),le_added(xs,ys,length_tail(List.length(&2,Nat,ys),List.length(&2,Nat,xs),same))}
    case Nil{} +right: Unit{}
    case Con{+x,+xs} Nil{}: Empty.absurd(DigitsLe(Con{x,xs},Nil{}),Digits.zero_successor(List.length(&2,Nat,xs),same))

def words_nat_length(words: List<&2,U32>) -> {List.length(&2,Nat,Digits.words_nat(words)) == List.length(&2,U32,words) : Nat}:
  match words:
    case Nil{}: {==}
    case Con{+word,+rest}: Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,Digits.words_nat(rest)),List.length(&2,U32,rest),words_nat_length(rest))

# Words raised by an amount at an axis that stays within a capacity are their
# numbers with the amount added there.
def raised_nat(words: List<&2,U32>,axis: Nat,+amount: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(Digits.digit_at(Digits.words_nat(words),axis),U32.to_nat(amount)),U32.to_nat(cap)) == True{} : Bool}) ->
  {Digits.words_nat(TiledView.raised(words,axis,amount)) == Digits.added(Digits.words_nat(words),unit_digits(List.length(&2,U32,words),axis,U32.to_nat(amount))) : List<&2,Nat>}:
  match words axis:
    case Con{+value,+rest} 0n:
      cons_equal(U32.to_nat((value + amount : U32)),Nat.add(U32.to_nat(value),U32.to_nat(amount)),Digits.words_nat(rest),
        Digits.added(Digits.words_nat(rest),zeros(List.length(&2,U32,rest))),
        Arithmetic.sum_exact(cap,cap_ok,value,amount,U32.to_nat(value),U32.to_nat(amount),{==},{==},room),
        Equal.sym(List<&2,Nat>,Digits.added(Digits.words_nat(rest),zeros(List.length(&2,U32,rest))),Digits.words_nat(rest),
          added_zeros_of(Digits.words_nat(rest),List.length(&2,U32,rest),words_nat_length(rest))))
    case Con{+value,+rest} 1n+ +next:
      cons_equal(U32.to_nat(value),Nat.add(U32.to_nat(value),0n),Digits.words_nat(TiledView.raised(rest,next,amount)),
        Digits.added(Digits.words_nat(rest),unit_digits(List.length(&2,U32,rest),next,U32.to_nat(amount))),
        Equal.sym(Nat,Nat.add(U32.to_nat(value),0n),U32.to_nat(value),Addition.zero_right(U32.to_nat(value))),raised_nat(rest,next,amount,cap,cap_ok,room))
    case Nil{} +axis: {==}

# Digits with an amount added at an axis, all within a bound, keep the amount
# and their own digit there within it.
def raise_room(+walk: List<&2,Nat>,+rank: Nat,+axis: Nat,+amount: Nat,+bound: Nat,+inside: {Nat.is_lt(axis,rank) == True{} : Bool},
  +length: {List.length(&2,Nat,walk) == rank : Nat},+within: AllAtMost(Digits.added(walk,unit_digits(rank,axis,amount)),bound)) ->
  {Nat.is_le(Nat.add(Digits.digit_at(walk,axis),amount),bound) == True{} : Bool}:
  +digit = at_most_digit(Digits.added(walk,unit_digits(rank,axis,amount)),axis,bound,within)
  +walk_inside = {%Equal.sym(Nat,List.length(&2,Nat,walk),rank,length) : {Nat.is_lt(axis,_) == True{} : Bool} inside : {Nat.is_lt(axis,List.length(&2,Nat,walk)) == True{} : Bool}}
  +same = Equal.trans(Nat,List.length(&2,Nat,walk),rank,List.length(&2,Nat,unit_digits(rank,axis,amount)),length,Equal.sym(Nat,List.length(&2,Nat,unit_digits(rank,axis,amount)),rank,unit_length(rank,axis,amount)))
  %Order.equal_reflexive(axis) : {Nat.is_le(Nat.add(Digits.digit_at(walk,axis),Bool.pick(Nat,_,amount,0n)),bound) == True{} : Bool}
  %digit_unit(rank,axis,axis,amount,inside) : {Nat.is_le(Nat.add(Digits.digit_at(walk,axis),_),bound) == True{} : Bool}
  %Digits.added_digit(walk,unit_digits(rank,axis,amount),axis,walk_inside,same) : {Nat.is_le(_,bound) == True{} : Bool}
  digit

def sum_right_le(+left: Nat,+right: Nat,+bound: Nat,+room: {Nat.is_le(Nat.add(left,right),bound) == True{} : Bool}) -> {Nat.is_le(right,bound) == True{} : Bool}:
  Order.transitive(right,Nat.add(left,right),bound,Order.right_le(left,right),room)

# The reach of one target axis more.
def reach_step(reading: TiledView.Reading,mapped: TiledView.Mapped,+reversed: Bool,+index: Nat,+extent: U32,+stride: U32,+shape: List<&2,U32>,+axis: Nat,
  +position: Grid.Position,+limits: Machine.Limits,+walk: List<&2,Nat>,+later: {Digits.words_nat(mapped_reach(mapped)) == walk : List<&2,Nat>},
  +walk_length: {List.length(&2,Nat,walk) == List.length(&2,U32,shape) : Nat},
  +valid: {mapped_valid(TiledView.with_reading(reading,reversed,index,extent,TiledView.lane_axis(TiledView.lanes_of(shape,axis,position)),mapped)) == True{} : Bool},
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +classified: Classified(reading,shape,axis,extent,stride,TiledView.lanes_of(shape,axis,position)),
  +room: AllAtMost(Digits.added(walk,contribution(reading,List.length(&2,U32,shape),axis,True{},
    MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),mirror(reversed,U32.to_nat(extent))))),U32.to_nat(Machine.half_word(limits)))) ->
  {Digits.words_nat(mapped_reach(TiledView.with_reading(reading,reversed,index,extent,TiledView.lane_axis(TiledView.lanes_of(shape,axis,position)),mapped))) ==
    Digits.added(walk,contribution(reading,List.length(&2,U32,shape),axis,True{},MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),mirror(reversed,U32.to_nat(extent))))) :
    List<&2,Nat>}:
  match reading mapped:
    case TiledView.Repeated{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      Equal.trans(List<&2,Nat>,Digits.words_nat(reach),walk,Digits.added(walk,zeros(List.length(&2,U32,shape))),later,
        Equal.sym(List<&2,Nat>,Digits.added(walk,zeros(List.length(&2,U32,shape))),walk,added_zeros_of(walk,List.length(&2,U32,shape),walk_length)))
    case TiledView.Lane{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      +rank = List.length(&2,U32,shape)
      +cap = Machine.half_word(limits)
      +last = Nat.sub(U32.to_nat(extent),1n)
      +largest = MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),mirror(reversed,U32.to_nat(extent)))
      +lane_exact = lane_axis_of(shape,axis,position,Bounds.strict_le(axis,rank,within))
      +last_exact = FullRange.subtract(extent,1,extent,Order.positive_one(U32.to_nat(extent),positive),Order.sub_le(U32.to_nat(extent),1n))
      +reach_length = Equal.trans(Nat,List.length(&2,U32,reach),List.length(&2,Nat,Digits.words_nat(reach)),rank,
        Equal.sym(Nat,List.length(&2,Nat,Digits.words_nat(reach)),List.length(&2,U32,reach),words_nat_length(reach)),
        Equal.trans(Nat,List.length(&2,Nat,Digits.words_nat(reach)),List.length(&2,Nat,walk),rank,Equal.cong(List<&2,Nat>,Nat,list => List.length(&2,Nat,list),Digits.words_nat(reach),walk,later),walk_length))
      +bound = {%reflected_mirror(reversed,U32.to_nat(extent)) : {Nat.is_le(Nat.add(Digits.digit_at(walk,axis),_),U32.to_nat(cap)) == True{} : Bool}
        raise_room(walk,rank,axis,largest,U32.to_nat(cap),within,walk_length,room) : {Nat.is_le(Nat.add(Digits.digit_at(walk,axis),last),U32.to_nat(cap)) == True{} : Bool}}
      +word_bound = {%Equal.sym(List<&2,Nat>,Digits.words_nat(reach),walk,later) : {Nat.is_le(Nat.add(Digits.digit_at(_,axis),U32.to_nat((extent - 1 : U32))),U32.to_nat(cap)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),last,last_exact) : {Nat.is_le(Nat.add(Digits.digit_at(walk,axis),_),U32.to_nat(cap)) == True{} : Bool}
        bound : {Nat.is_le(Nat.add(Digits.digit_at(Digits.words_nat(reach),axis),U32.to_nat((extent - 1 : U32))),U32.to_nat(cap)) == True{} : Bool}}
      %Equal.sym(Nat,TiledView.lane_axis(TiledView.lanes_of(shape,axis,position)),axis,lane_exact) :
        {Digits.words_nat(TiledView.raised(reach,_,(extent - 1 : U32))) == Digits.added(walk,unit_digits(rank,axis,largest)) : List<&2,Nat>}
      %Equal.sym(Nat,largest,last,reflected_mirror(reversed,U32.to_nat(extent))) :
        {Digits.words_nat(TiledView.raised(reach,axis,(extent - 1 : U32))) == Digits.added(walk,unit_digits(rank,axis,_)) : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,Digits.words_nat(TiledView.raised(reach,axis,(extent - 1 : U32))),Digits.added(Digits.words_nat(reach),unit_digits(List.length(&2,U32,reach),axis,U32.to_nat((extent - 1 : U32)))),
          raised_nat(reach,axis,(extent - 1 : U32),cap,Limits.half_word_fits(limits),word_bound)) : {_ == Digits.added(walk,unit_digits(rank,axis,last)) : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,Digits.words_nat(reach),walk,later) : {Digits.added(_,unit_digits(List.length(&2,U32,reach),axis,U32.to_nat((extent - 1 : U32)))) == Digits.added(walk,unit_digits(rank,axis,last)) : List<&2,Nat>}
      %Equal.sym(Nat,List.length(&2,U32,reach),rank,reach_length) : {Digits.added(walk,unit_digits(_,axis,U32.to_nat((extent - 1 : U32)))) == Digits.added(walk,unit_digits(rank,axis,last)) : List<&2,Nat>}
      %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),last,last_exact) : {Digits.added(walk,unit_digits(rank,axis,_)) == Digits.added(walk,unit_digits(rank,axis,last)) : List<&2,Nat>}
      {==}
    case TiledView.Stepping{+found,+step,+cells} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      KernelGemm.Both{+stepping,+rest_facts} = classified
      StepAt{+place,+axis_exact,+place_inside,+stride_exact} = stepping
      +rank = List.length(&2,U32,shape)
      +cap = Machine.half_word(limits)
      +last = Nat.sub(U32.to_nat(extent),1n)
      +size = U32.to_nat(step)
      +found_index = U32.to_nat(found)
      +largest = MirrorAddress.reflected_coordinate(reversed,U32.to_nat(extent),mirror(reversed,U32.to_nat(extent)))
      +found_inside = {%Equal.sym(Nat,found_index,place,axis_exact) : {Nat.is_lt(_,rank) == True{} : Bool} place_inside : {Nat.is_lt(found_index,rank) == True{} : Bool}}
      +last_exact = FullRange.subtract(extent,1,extent,Order.positive_one(U32.to_nat(extent),positive),Order.sub_le(U32.to_nat(extent),1n))
      +reach_length = Equal.trans(Nat,List.length(&2,U32,reach),List.length(&2,Nat,Digits.words_nat(reach)),rank,
        Equal.sym(Nat,List.length(&2,Nat,Digits.words_nat(reach)),List.length(&2,U32,reach),words_nat_length(reach)),
        Equal.trans(Nat,List.length(&2,Nat,Digits.words_nat(reach)),List.length(&2,Nat,walk),rank,Equal.cong(List<&2,Nat>,Nat,list => List.length(&2,Nat,list),Digits.words_nat(reach),walk,later),walk_length))
      +bound = {%reflected_mirror(reversed,U32.to_nat(extent)) : {Nat.is_le(Nat.add(Digits.digit_at(walk,found_index),Nat.mul(_,size)),U32.to_nat(cap)) == True{} : Bool}
        raise_room(walk,rank,found_index,Nat.mul(largest,size),U32.to_nat(cap),found_inside,walk_length,room) :
        {Nat.is_le(Nat.add(Digits.digit_at(walk,found_index),Nat.mul(last,size)),U32.to_nat(cap)) == True{} : Bool}}
      +product_room = {%RowMajor.multiply_commutative(last,size) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool}
        sum_right_le(Digits.digit_at(walk,found_index),Nat.mul(last,size),U32.to_nat(cap),bound) : {Nat.is_le(Nat.mul(size,last),U32.to_nat(cap)) == True{} : Bool}}
      +product = Arithmetic.product_exact(cap,Limits.half_word_fits(limits),step,(extent - 1 : U32),size,last,{==},last_exact,product_room)
      +amount_exact = Equal.trans(Nat,U32.to_nat((step * (extent - 1 : U32) : U32)),Nat.mul(size,last),Nat.mul(last,size),product,RowMajor.multiply_commutative(size,last))
      +word_bound = {%Equal.sym(List<&2,Nat>,Digits.words_nat(reach),walk,later) : {Nat.is_le(Nat.add(Digits.digit_at(_,found_index),U32.to_nat((step * (extent - 1 : U32) : U32))),U32.to_nat(cap)) == True{} : Bool}
        %Equal.sym(Nat,U32.to_nat((step * (extent - 1 : U32) : U32)),Nat.mul(last,size),amount_exact) : {Nat.is_le(Nat.add(Digits.digit_at(walk,found_index),_),U32.to_nat(cap)) == True{} : Bool}
        bound : {Nat.is_le(Nat.add(Digits.digit_at(Digits.words_nat(reach),found_index),U32.to_nat((step * (extent - 1 : U32) : U32))),U32.to_nat(cap)) == True{} : Bool}}
      %Equal.sym(Nat,largest,last,reflected_mirror(reversed,U32.to_nat(extent))) :
        {Digits.words_nat(TiledView.raised(reach,found_index,(step * (extent - 1 : U32) : U32))) == Digits.added(walk,unit_digits(rank,found_index,Nat.mul(_,size))) : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,Digits.words_nat(TiledView.raised(reach,found_index,(step * (extent - 1 : U32) : U32))),
          Digits.added(Digits.words_nat(reach),unit_digits(List.length(&2,U32,reach),found_index,U32.to_nat((step * (extent - 1 : U32) : U32)))),
          raised_nat(reach,found_index,(step * (extent - 1 : U32) : U32),cap,Limits.half_word_fits(limits),word_bound)) :
        {_ == Digits.added(walk,unit_digits(rank,found_index,Nat.mul(last,size))) : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,Digits.words_nat(reach),walk,later) : {Digits.added(_,unit_digits(List.length(&2,U32,reach),found_index,U32.to_nat((step * (extent - 1 : U32) : U32)))) ==
        Digits.added(walk,unit_digits(rank,found_index,Nat.mul(last,size))) : List<&2,Nat>}
      %Equal.sym(Nat,List.length(&2,U32,reach),rank,reach_length) : {Digits.added(walk,unit_digits(_,found_index,U32.to_nat((step * (extent - 1 : U32) : U32)))) == Digits.added(walk,unit_digits(rank,found_index,Nat.mul(last,size))) : List<&2,Nat>}
      %Equal.sym(Nat,U32.to_nat((step * (extent - 1 : U32) : U32)),Nat.mul(last,size),amount_exact) : {Digits.added(walk,unit_digits(rank,found_index,_)) == Digits.added(walk,unit_digits(rank,found_index,Nat.mul(last,size))) : List<&2,Nat>}
      {==}
    case TiledView.Unkept{} TiledView.Mapped{+kept,+lane,+lane_extent,+axes,+flags,+reach}:
      Empty.absurd({Digits.words_nat(reach) == Digits.added(walk,zeros(List.length(&2,U32,shape))) : List<&2,Nat>},WordEncoding.false_true(valid))

# The reach of a valid view of nonzero extents whose last source digits stay
# within half a word: the start raised by the last digits.
def reach_walk(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+index: Nat,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,
  +facts: SourceFacts<limits,shape,axis,position>,+words: List<&2,U32>,+words_length: {List.length(&2,Nat,Digits.words_nat(words)) == List.length(&2,U32,shape) : Nat},
  +valid: {mapped_valid(TiledView.mapped_axes(axes,flags,index,shape,TiledView.lanes_of(shape,axis,position),words)) == True{} : Bool},
  +nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +room: AllAtMost(walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,True{},Digits.words_nat(words),mirrored_lasts(axes,flags)),U32.to_nat(Machine.half_word(limits)))) ->
  {Digits.words_nat(mapped_reach(TiledView.mapped_axes(axes,flags,index,shape,TiledView.lanes_of(shape,axis,position),words))) ==
    walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,True{},Digits.words_nat(words),mirrored_lasts(axes,flags)) : List<&2,Nat>}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}:
      +lanes = TiledView.lanes_of(shape,axis,position)
      +reading = TiledView.classified(shape,lanes,extent,stride)
      +tail_mapped = TiledView.mapped_axes(rest,Views.drop_flags(1n,flags),1n+index,shape,lanes,words)
      +rank = List.length(&2,U32,shape)
      +start = Digits.words_nat(words)
      +maximum = walked(rest,Views.later_flags(flags),shape,lanes,axis,True{},start,mirrored_lasts(rest,Views.later_flags(flags)))
      +largest = contribution(reading,rank,axis,True{},MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),mirror(Views.flag(flags),U32.to_nat(extent))))
      +maximum_length = walked_length(rest,Views.later_flags(flags),shape,lanes,axis,True{},start,mirrored_lasts(rest,Views.later_flags(flags)),words_length)
      +tail_ok = tail_valid(reading,tail_mapped,Views.flag(flags),index,extent,TiledView.lane_axis(lanes),valid)
      +positive = Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(rest)),nonempty))
      +tail_room = at_most_smaller(maximum,Digits.added(maximum,largest),U32.to_nat(Machine.half_word(limits)),
        le_added(maximum,largest,lengths_same(maximum,largest,shape,maximum_length,
          contribution_length(reading,rank,axis,True{},MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),mirror(Views.flag(flags),U32.to_nat(extent)))))),room)
      +tail = reach_walk(rest,Views.later_flags(flags),1n+index,shape,axis,position,limits,facts,words,words_length,tail_ok,
        Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(rest)),nonempty),tail_room)
      reach_step(reading,tail_mapped,Views.flag(flags),index,extent,stride,shape,axis,position,limits,maximum,tail,maximum_length,valid,
        source_within(limits,shape,axis,position,facts),positive,axis_facts(limits,shape,axis,position,facts,extent,stride),room)

# Words inside the extents are digits below them.
def inside_below(reach: List<&2,U32>,shape: List<&2,U32>,+inside: {TiledView.all_inside(reach,shape) == True{} : Bool},
  +same: {List.length(&2,U32,reach) == List.length(&2,U32,shape) : Nat}) -> Coordinates.DigitsBelow(shape,Digits.words_nat(reach)):
  match reach shape:
    case Con{+value,+rest} Con{+extent,+others}:
      KernelGemm.Both{Equal.trans(Bool,Nat.is_lt(U32.to_nat(value),U32.to_nat(extent)),U32.is_lt(value,extent),True{},Equal.sym(Bool,U32.is_lt(value,extent),
          Nat.is_lt(U32.to_nat(value),U32.to_nat(extent)),Comparison.lt(value,extent)),Boolean.left(U32.is_lt(value,extent),TiledView.all_inside(rest,others),inside)),
        inside_below(rest,others,Boolean.right(U32.is_lt(value,extent),TiledView.all_inside(rest,others),inside),length_tail(List.length(&2,U32,rest),List.length(&2,U32,others),same))}
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{+extent,+others}: Empty.absurd(Coordinates.DigitsBelow(Con{extent,others},Nil{}),Digits.zero_successor(List.length(&2,U32,others),same))
    case Con{+value,+rest} Nil{}: Empty.absurd(Coordinates.DigitsBelow(Nil{},Digits.words_nat(Con{value,rest})),successor_zero(List.length(&2,U32,rest),same))

# The view sum at the last digits is the span of the last coordinates.
def lasts_sum(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}) ->
  {digit_sum(axes,flags,mirrored_lasts(axes,flags)) == ViewBounds.last_span(axes) : Nat}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}:
      +positive = Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(rest)),nonempty))
      +last = Nat.sub(U32.to_nat(extent),1n)
      +last_exact = FullRange.subtract(extent,1,extent,Order.positive_one(U32.to_nat(extent),positive),Order.sub_le(U32.to_nat(extent),1n))
      %Equal.sym(Nat,MirrorAddress.reflected_coordinate(Views.flag(flags),U32.to_nat(extent),mirror(Views.flag(flags),U32.to_nat(extent))),last,
          reflected_mirror(Views.flag(flags),U32.to_nat(extent))) :
        {Nat.add(Nat.mul(_,U32.to_nat(stride)),digit_sum(rest,Views.later_flags(flags),mirrored_lasts(rest,Views.later_flags(flags)))) ==
          Nat.add(Nat.mul(U32.to_nat((extent - 1 : U32)),U32.to_nat(stride)),ViewBounds.last_span(rest)) : Nat}
      %Equal.sym(Nat,U32.to_nat((extent - 1 : U32)),last,last_exact) :
        {Nat.add(Nat.mul(last,U32.to_nat(stride)),digit_sum(rest,Views.later_flags(flags),mirrored_lasts(rest,Views.later_flags(flags)))) ==
          Nat.add(Nat.mul(_,U32.to_nat(stride)),ViewBounds.last_span(rest)) : Nat}
      Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(last,U32.to_nat(stride)),value),digit_sum(rest,Views.later_flags(flags),mirrored_lasts(rest,Views.later_flags(flags))),
        ViewBounds.last_span(rest),lasts_sum(rest,Views.later_flags(flags),Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(rest)),nonempty)))

# ---- The view of the cells ----

# Digits added in either grouping.
def added_assoc(first: List<&2,Nat>,second: List<&2,Nat>,third: List<&2,Nat>,
  +second_length: {List.length(&2,Nat,second) == List.length(&2,Nat,first) : Nat},+third_length: {List.length(&2,Nat,third) == List.length(&2,Nat,first) : Nat}) ->
  {Digits.added(Digits.added(first,second),third) == Digits.added(first,Digits.added(second,third)) : List<&2,Nat>}:
  match first second third:
    case Con{+x,+xs} Con{+y,+ys} Con{+z,+zs}:
      cons_equal(Nat.add(Nat.add(x,y),z),Nat.add(x,Nat.add(y,z)),Digits.added(Digits.added(xs,ys),zs),Digits.added(xs,Digits.added(ys,zs)),Addition.associative(x,y,z),
        added_assoc(xs,ys,zs,length_tail(List.length(&2,Nat,ys),List.length(&2,Nat,xs),second_length),length_tail(List.length(&2,Nat,zs),List.length(&2,Nat,xs),third_length)))
    case Nil{} +second +third: {==}
    case Con{+x,+xs} Nil{} +third:
      Empty.absurd({Digits.added(Digits.added(Con{x,xs},Nil{}),third) == Digits.added(Con{x,xs},Digits.added(Nil{},third)) : List<&2,Nat>},
        Digits.zero_successor(List.length(&2,Nat,xs),second_length))
    case Con{+x,+xs} Con{+y,+ys} Nil{}:
      Empty.absurd({Digits.added(Digits.added(Con{x,xs},Con{y,ys}),Nil{}) == Digits.added(Con{x,xs},Digits.added(Con{y,ys},Nil{})) : List<&2,Nat>},
        Digits.zero_successor(List.length(&2,Nat,xs),third_length))

# The digits but one axis put back with an amount added to its digit are the
# digits with the amount added at the axis.
def inserted_added(digits: List<&2,Nat>,axis: Nat,+amount: Nat,+inside: {Nat.is_lt(axis,List.length(&2,Nat,digits)) == True{} : Bool}) ->
  {Coordinates.inserted_digit(axis,Geometry.removed_at(digits,axis),Nat.add(amount,Digits.digit_at(digits,axis))) ==
    Digits.added(digits,unit_digits(List.length(&2,Nat,digits),axis,amount)) : List<&2,Nat>}:
  match digits axis:
    case Con{+digit,+rest} 0n:
      cons_equal(Nat.add(amount,digit),Nat.add(digit,amount),rest,Digits.added(rest,zeros(List.length(&2,Nat,rest))),Order.commutative(amount,digit),
        Equal.sym(List<&2,Nat>,Digits.added(rest,zeros(List.length(&2,Nat,rest))),rest,added_zeros(rest)))
    case Con{+digit,+rest} 1n+ +next:
      cons_equal(digit,Nat.add(digit,0n),Coordinates.inserted_digit(next,Geometry.removed_at(rest,next),Nat.add(amount,Digits.digit_at(rest,next))),
        Digits.added(rest,unit_digits(List.length(&2,Nat,rest),next,amount)),Equal.sym(Nat,Nat.add(digit,0n),digit,Addition.zero_right(digit)),
        inserted_added(rest,next,amount,inside))
    case Nil{} +axis:
      Empty.absurd({Coordinates.inserted_digit(axis,Geometry.removed_at(Nil{},axis),Nat.add(amount,Digits.digit_at(Nil{},axis))) ==
        Digits.added(Nil{},unit_digits(0n,axis,amount)) : List<&2,Nat>},Order.below_zero(axis,inside))

# An amount at the lane axis that is a multiple of eight is aligned there.
def self_aligned(+rank: Nat,+axis: Nat,+amount: Nat,+inside: {Nat.is_lt(axis,rank) == True{} : Bool},+zero: {Nat.mod(amount,8n) == 0n : Nat}) ->
  {Nat.mod(Digits.digit_at(unit_digits(rank,axis,amount),axis),8n) == 0n : Nat}:
  %Equal.sym(Nat,Digits.digit_at(unit_digits(rank,axis,amount),axis),Bool.pick(Nat,Nat.is_eq(axis,axis),amount,0n),digit_unit(rank,axis,axis,amount,inside)) :
    {Nat.mod(_,8n) == 0n : Nat}
  %Equal.sym(Bool,Nat.is_eq(axis,axis),True{},Order.equal_reflexive(axis)) : {Nat.mod(Bool.pick(Nat,_,amount,0n),8n) == 0n : Nat}
  zero

def extents_appended(axes: List<&2,Views.Axis>,+extent: U32,+stride: U32) ->
  {Views.axis_extents(List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}])) == List.append(&2,U32,Views.axis_extents(axes),[extent]) : List<&2,U32>}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+size,+step},+rest}:
      Equal.cong(List<&2,U32>,List<&2,U32>,list => size <> list,Views.axis_extents(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}])),
        List.append(&2,U32,Views.axis_extents(rest),[extent]),extents_appended(rest,extent,stride))

# The view sum of axes with one more forward axis appended.
def sum_appended(axes: List<&2,Views.Axis>,flags: List<&2,Bool>,digits: List<&2,Nat>,+extent: U32,+stride: U32,+group: Nat,
  +flags_length: {List.length(&2,Bool,flags) == List.length(&2,Views.Axis,axes) : Nat},+digits_length: {List.length(&2,Nat,digits) == List.length(&2,Views.Axis,axes) : Nat}) ->
  {digit_sum(List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),List.append(&2,Bool,flags,[False{}]),List.append(&2,Nat,digits,[group])) ==
    Nat.add(digit_sum(axes,flags,digits),Nat.mul(group,U32.to_nat(stride))) : Nat}:
  match axes flags digits:
    case Con{Views.Axis{+size,+step},+rest} Con{+flag,+more_flags} Con{+digit,+more}:
      +own = Nat.mul(MirrorAddress.reflected_coordinate(flag,U32.to_nat(size),digit),U32.to_nat(step))
      Equal.trans(Nat,Nat.add(own,digit_sum(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}]),List.append(&2,Bool,more_flags,[False{}]),List.append(&2,Nat,more,[group]))),
        Nat.add(own,Nat.add(digit_sum(rest,more_flags,more),Nat.mul(group,U32.to_nat(stride)))),Nat.add(Nat.add(own,digit_sum(rest,more_flags,more)),Nat.mul(group,U32.to_nat(stride))),
        Equal.cong(Nat,Nat,value => Nat.add(own,value),
          digit_sum(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}]),List.append(&2,Bool,more_flags,[False{}]),List.append(&2,Nat,more,[group])),
          Nat.add(digit_sum(rest,more_flags,more),Nat.mul(group,U32.to_nat(stride))),
          sum_appended(rest,more_flags,more,extent,stride,group,length_tail(List.length(&2,Bool,more_flags),List.length(&2,Views.Axis,rest),flags_length),
            length_tail(List.length(&2,Nat,more),List.length(&2,Views.Axis,rest),digits_length))),
        Equal.sym(Nat,Nat.add(Nat.add(own,digit_sum(rest,more_flags,more)),Nat.mul(group,U32.to_nat(stride))),
          Nat.add(own,Nat.add(digit_sum(rest,more_flags,more),Nat.mul(group,U32.to_nat(stride)))),
          Addition.associative(own,digit_sum(rest,more_flags,more),Nat.mul(group,U32.to_nat(stride)))))
    case Nil{} Nil{} Nil{}: Addition.zero_right(Nat.mul(group,U32.to_nat(stride)))
    case Nil{} Con{+flag,+more_flags} +digits:
      Empty.absurd({digit_sum(List.append(&2,Views.Axis,Nil{},[Views.Axis{extent,stride}]),List.append(&2,Bool,Con{flag,more_flags},[False{}]),List.append(&2,Nat,digits,[group])) ==
        Nat.add(digit_sum(Nil{},Con{flag,more_flags},digits),Nat.mul(group,U32.to_nat(stride))) : Nat},successor_zero(List.length(&2,Bool,more_flags),flags_length))
    case Nil{} Nil{} Con{+digit,+more}:
      Empty.absurd({digit_sum(List.append(&2,Views.Axis,Nil{},[Views.Axis{extent,stride}]),List.append(&2,Bool,Nil{},[False{}]),List.append(&2,Nat,Con{digit,more},[group])) ==
        Nat.add(digit_sum(Nil{},Nil{},Con{digit,more}),Nat.mul(group,U32.to_nat(stride))) : Nat},successor_zero(List.length(&2,Nat,more),digits_length))
    case Con{Views.Axis{+size,+step},+rest} Nil{} +digits:
      Empty.absurd({digit_sum(List.append(&2,Views.Axis,Con{Views.Axis{size,step},rest},[Views.Axis{extent,stride}]),List.append(&2,Bool,Nil{},[False{}]),List.append(&2,Nat,digits,[group])) ==
        Nat.add(digit_sum(Con{Views.Axis{size,step},rest},Nil{},digits),Nat.mul(group,U32.to_nat(stride))) : Nat},Digits.zero_successor(List.length(&2,Views.Axis,rest),flags_length))
    case Con{Views.Axis{+size,+step},+rest} Con{+flag,+more_flags} Nil{}:
      Empty.absurd({digit_sum(List.append(&2,Views.Axis,Con{Views.Axis{size,step},rest},[Views.Axis{extent,stride}]),List.append(&2,Bool,Con{flag,more_flags},[False{}]),List.append(&2,Nat,Nil{},[group])) ==
        Nat.add(digit_sum(Con{Views.Axis{size,step},rest},Con{flag,more_flags},Nil{}),Nat.mul(group,U32.to_nat(stride))) : Nat},Digits.zero_successor(List.length(&2,Views.Axis,rest),digits_length))

# The last digits of axes with one more forward axis appended.
def lasts_appended(axes: List<&2,Views.Axis>,flags: List<&2,Bool>,+extent: U32,+stride: U32,+flags_length: {List.length(&2,Bool,flags) == List.length(&2,Views.Axis,axes) : Nat}) ->
  {mirrored_lasts(List.append(&2,Views.Axis,axes,[Views.Axis{extent,stride}]),List.append(&2,Bool,flags,[False{}])) ==
    List.append(&2,Nat,mirrored_lasts(axes,flags),[Nat.sub(U32.to_nat(extent),1n)]) : List<&2,Nat>}:
  match axes flags:
    case Con{Views.Axis{+size,+step},+rest} Con{+flag,+more_flags}:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,list => mirror(flag,U32.to_nat(size)) <> list,mirrored_lasts(List.append(&2,Views.Axis,rest,[Views.Axis{extent,stride}]),List.append(&2,Bool,more_flags,[False{}])),
        List.append(&2,Nat,mirrored_lasts(rest,more_flags),[Nat.sub(U32.to_nat(extent),1n)]),
        lasts_appended(rest,more_flags,extent,stride,length_tail(List.length(&2,Bool,more_flags),List.length(&2,Views.Axis,rest),flags_length)))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+flag,+more_flags}:
      Empty.absurd({mirrored_lasts(List.append(&2,Views.Axis,Nil{},[Views.Axis{extent,stride}]),List.append(&2,Bool,Con{flag,more_flags},[False{}])) ==
        List.append(&2,Nat,mirrored_lasts(Nil{},Con{flag,more_flags}),[Nat.sub(U32.to_nat(extent),1n)]) : List<&2,Nat>},successor_zero(List.length(&2,Bool,more_flags),flags_length))
    case Con{Views.Axis{+size,+step},+rest} Nil{}:
      Empty.absurd({mirrored_lasts(List.append(&2,Views.Axis,Con{Views.Axis{size,step},rest},[Views.Axis{extent,stride}]),List.append(&2,Bool,Nil{},[False{}])) ==
        List.append(&2,Nat,mirrored_lasts(Con{Views.Axis{size,step},rest},Nil{}),[Nat.sub(U32.to_nat(extent),1n)]) : List<&2,Nat>},Digits.zero_successor(List.length(&2,Views.Axis,rest),flags_length))

def mirror_below(reverse: Bool,+extent: Nat,+positive: {Nat.is_lt(0n,extent) == True{} : Bool}) -> {Nat.is_lt(mirror(reverse,extent),extent) == True{} : Bool}:
  match reverse:
    case False{}: Order.last_below(extent,positive)
    case True{}: positive

# The last digits of axes of nonzero extents lie below their extents.
def lasts_below(axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool}) ->
  Coordinates.DigitsBelow(Views.axis_extents(axes),mirrored_lasts(axes,flags)):
  match axes:
    case Nil{}: Unit{}
    case Con{Views.Axis{+extent,+stride},+rest}:
      +positive = Arithmetic.word_nonzero_positive(extent,Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(rest)),nonempty))
      KernelGemm.Both{mirror_below(Views.flag(flags),U32.to_nat(extent),positive),
        lasts_below(rest,Views.later_flags(flags),Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(Views.axis_extents(rest)),nonempty))}

# ---- A view that keeps the lanes ----

# What a view that keeps the lanes of values gives: the target axis that
# reads the lanes, its extent, the kept axes and their flags, the first and
# the last source digits, and the counts the reads stay within.
type KeptView<-limits: Machine.Limits,-shape: List<&2,U32>,-axis: Nat,-position: Grid.Position,-axes: List<&2,Views.Axis>,-flags: List<&2,Bool>,-offset: U32,
  -lane: Nat,-extent: U32,-kept: List<&2,Views.Axis>,-kept_flags: List<&2,Bool>> is Data:
  KeptView{facts: SourceFacts<limits,shape,axis,position>,
    valid: {mapped_valid(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == True{} : Bool},
    lane_exact: {mapped_lane(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == Some{lane} : Maybe<&2,Nat>},
    kept_exact: {mapped_kept(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == kept : List<&2,Views.Axis>},
    flags_exact: {mapped_flags(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == kept_flags : List<&2,Bool>},
    lane_inside: {Nat.is_lt(lane,List.length(&2,U32,Views.axis_extents(axes))) == True{} : Bool},
    extent_exact: {extent == Grid.lanes_at(Views.axis_extents(axes),lane) : U32},
    kept_extents: {Views.axis_extents(kept) == Grid.removed(Views.axis_extents(axes),lane) : List<&2,U32>},
    flags_length: {List.length(&2,Bool,kept_flags) == List.length(&2,Views.Axis,kept) : Nat},
    start_length: {List.length(&2,Nat,Digits.words_nat(Grid.coordinates(shape,offset))) == List.length(&2,U32,shape) : Nat},
    start_aligned: {Nat.mod(Digits.digit_at(Digits.words_nat(Grid.coordinates(shape,offset)),axis),8n) == 0n : Nat},
    start_value: {ShapeLaws.digits_value(shape,Digits.words_nat(Grid.coordinates(shape,offset))) == U32.to_nat(offset) : Nat},
    start_below: Coordinates.DigitsBelow(shape,Digits.words_nat(Grid.coordinates(shape,offset))),
    bound: Coordinates.DigitsBelow(shape,walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,True{},Digits.words_nat(Grid.coordinates(shape,offset)),
      mirrored_lasts(axes,flags))),
    target_nonempty: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
    target_room: {Nat.is_le(ShapeLaws.elements(Views.axis_extents(axes)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    target_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{})),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    view_fits: {Views.fits_nonempty(False{},Views.View{axes,offset},Tensors.count_or_zero(shape)) == True{} : Bool},
    length_room: {Nat.is_le(U32.to_nat(Tensors.count_or_zero(shape)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    cells_count: {U32.to_nat(Tensors.count_or_zero(Tiled.cell_shape(shape,axis,position))) == ShapeLaws.elements(Tiled.cell_shape(shape,axis,position)) : Nat},
    grouped: TiledValues.Groups<limits,Grid.lanes_at(shape,axis)>,
    removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(shape,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
    target_grouped: TiledValues.Groups<limits,extent>,
    target_removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(Views.axis_extents(axes),lane)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}

# The group step of the view of the cells times a group is the cell of eight
# times the group at the lane axis.
def group_term(group: Nat,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+limits: Machine.Limits,+facts: SourceFacts<limits,shape,axis,position>,
  +below: Coordinates.DigitsBelow(shape,unit_digits(List.length(&2,U32,shape),axis,Nat.mul(group,8n)))) ->
  {Nat.mul(group,U32.to_nat(TiledView.cell_index(TiledView.lanes_of(shape,axis,position),(8 * TiledView.lane_inner(TiledView.lanes_of(shape,axis,position)) : U32)))) ==
    cell_value(shape,axis,position,unit_digits(List.length(&2,U32,shape),axis,Nat.mul(group,8n))) : Nat}:
  match group:
    case 0n:
      +rank = List.length(&2,U32,shape)
      %Equal.sym(List<&2,Nat>,unit_digits(rank,axis,0n),zeros(rank),unit_zero(rank,axis)) : {0n == cell_value(shape,axis,position,_) : Nat}
      Equal.sym(Nat,cell_value(shape,axis,position,zeros(rank)),0n,cell_zeros(shape,axis,position,source_within(limits,shape,axis,position,facts)))
    case 1n+ +more:
      +rank = List.length(&2,U32,shape)
      +within = source_within(limits,shape,axis,position,facts)
      +lanes = TiledView.lanes_of(shape,axis,position)
      +after = Public.lane_after(shape,axis)
      +cells = ShapeLaws.elements(after)
      +cap = Machine.capacity(limits)
      +unit = unit_digits(rank,axis,8n)
      +unit_below = unit_below_le(shape,axis,8n,Nat.mul(1n+more,8n),Order.le_add(8n,Nat.mul(more,8n)),below)
      +value = unit_value(shape,axis,8n,within)
      +below_cells = {%value : {Nat.is_lt(_,ShapeLaws.elements(shape)) == True{} : Bool} Coordinates.digits_below(shape,unit,unit_below) :
        {Nat.is_lt(Nat.mul(8n,cells),ShapeLaws.elements(shape)) == True{} : Bool}}
      +eight_room = Order.transitive(Nat.mul(8n,cells),ShapeLaws.elements(shape),U32.to_nat(cap),Bounds.strict_le(Nat.mul(8n,cells),ShapeLaws.elements(shape),below_cells),
        source_room(limits,shape,axis,position,facts))
      +inner_room = Order.transitive(cells,Nat.mul(8n,cells),U32.to_nat(cap),Order.le_add(cells,Nat.mul(7n,cells)),eight_room)
      +inner_value = Equal.trans(Nat,U32.to_nat(TiledView.lane_inner(lanes)),U32.to_nat(Tiled.product(after)),cells,
        Equal.cong(TiledView.Lanes,Nat,found => U32.to_nat(TiledView.lane_inner(found)),lanes,
          TiledView.Lanes{axis,Grid.lanes_at(shape,axis),Tiled.product(after),(Public.lane_prefix(shape,axis,1) * Tiled.product(after) : U32),position},
          lanes_exact(shape,axis,position,Bounds.strict_le(axis,rank,within))),
        Coordinates.product_word(limits,after,inner_room))
      +step_value = Arithmetic.product_exact(cap,Limits.capacity_fits(limits),8,TiledView.lane_inner(lanes),8n,cells,{==},inner_value,eight_room)
      +step_word = Equal.trans(U32,(8 * TiledView.lane_inner(lanes) : U32),U32.from_nat(U32.to_nat((8 * TiledView.lane_inner(lanes) : U32))),U32.from_nat(ShapeLaws.digits_value(shape,unit)),
        Equal.sym(U32,U32.from_nat(U32.to_nat((8 * TiledView.lane_inner(lanes) : U32))),(8 * TiledView.lane_inner(lanes) : U32),Arithmetic.native_roundtrip((8 * TiledView.lane_inner(lanes) : U32))),
        Equal.cong(Nat,U32,number => U32.from_nat(number),U32.to_nat((8 * TiledView.lane_inner(lanes) : U32)),ShapeLaws.digits_value(shape,unit),
          Equal.trans(Nat,U32.to_nat((8 * TiledView.lane_inner(lanes) : U32)),Nat.mul(8n,cells),ShapeLaws.digits_value(shape,unit),step_value,
            Equal.sym(Nat,ShapeLaws.digits_value(shape,unit),Nat.mul(8n,cells),value))))
      +step_cell = {%Equal.sym(U32,(8 * TiledView.lane_inner(lanes) : U32),U32.from_nat(ShapeLaws.digits_value(shape,unit)),step_word) :
          {U32.to_nat(TiledView.cell_index(lanes,_)) == cell_value(shape,axis,position,unit) : Nat}
        cell_of(shape,axis,position,unit,limits,within,unit_below,source_room(limits,shape,axis,position,facts),source_cells_room(limits,shape,axis,position,facts)) :
        {U32.to_nat(TiledView.cell_index(lanes,(8 * TiledView.lane_inner(lanes) : U32))) == cell_value(shape,axis,position,unit) : Nat}}
      Equal.trans(Nat,Nat.mul(1n+more,U32.to_nat(TiledView.cell_index(lanes,(8 * TiledView.lane_inner(lanes) : U32)))),Nat.mul(1n+more,cell_value(shape,axis,position,unit)),
        cell_value(shape,axis,position,unit_digits(rank,axis,Nat.mul(1n+more,8n))),
        Equal.cong(Nat,Nat,number => Nat.mul(1n+more,number),U32.to_nat(TiledView.cell_index(lanes,(8 * TiledView.lane_inner(lanes) : U32))),cell_value(shape,axis,position,unit),step_cell),
        Equal.sym(Nat,cell_value(shape,axis,position,unit_digits(rank,axis,Nat.mul(1n+more,8n))),Nat.mul(1n+more,cell_value(shape,axis,position,unit)),
          cell_scaled(shape,axis,position,axis,8n,1n+more,within,self_aligned(rank,axis,8n,within,{==}))))

# What a view that keeps the lanes reads at target digits.
def walk_at(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,+view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,
  +digits: List<&2,Nat>,+below: Coordinates.DigitsBelow(Views.axis_extents(axes),digits)) ->
  CellWalk<shape,axis,position,Some{lane},kept,kept_flags,0n,List.length(&2,Views.Axis,axes),Digits.words_nat(Grid.coordinates(shape,offset)),digits,
    walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,False{},Digits.words_nat(Grid.coordinates(shape,offset)),digits),
    walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,True{},Digits.words_nat(Grid.coordinates(shape,offset)),digits)>:
  KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  +lanes = TiledView.lanes_of(shape,axis,position)
  +words = Grid.coordinates(shape,offset)
  +start = Digits.words_nat(words)
  +mapped = TiledView.mapped_axes(axes,flags,0n,shape,lanes,words)
  %lane_exact : CellWalk<shape,axis,position,_,kept,kept_flags,0n,List.length(&2,Views.Axis,axes),start,digits,
    walked(axes,flags,shape,lanes,axis,False{},start,digits),walked(axes,flags,shape,lanes,axis,True{},start,digits)>
  %kept_exact : CellWalk<shape,axis,position,mapped_lane(mapped),_,kept_flags,0n,List.length(&2,Views.Axis,axes),start,digits,
    walked(axes,flags,shape,lanes,axis,False{},start,digits),walked(axes,flags,shape,lanes,axis,True{},start,digits)>
  %flags_exact : CellWalk<shape,axis,position,mapped_lane(mapped),mapped_kept(mapped),_,0n,List.length(&2,Views.Axis,axes),start,digits,
    walked(axes,flags,shape,lanes,axis,False{},start,digits),walked(axes,flags,shape,lanes,axis,True{},start,digits)>
  cell_walk(axes,flags,0n,shape,axis,position,limits,facts,words,start,digits,start_length,start_aligned,below,valid,bound)

def group_step(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position) -> U32:
  TiledView.cell_index(TiledView.lanes_of(shape,axis,position),(8 * TiledView.lane_inner(TiledView.lanes_of(shape,axis,position)) : U32))

# What the view of the cells reads at the digits of a target cell, and the
# source digits a lane of it holds.
type Chain<-shape: List<&2,U32>,-axis: Nat,-position: Grid.Position,-offset: U32,-axes: List<&2,Views.Axis>,-flags: List<&2,Bool>,-kept: List<&2,Views.Axis>,
  -kept_flags: List<&2,Bool>,-extent: U32,-others: List<&2,Nat>,-group: Nat,-digits: List<&2,Nat>,-lane_digit: Nat,-cell: List<&2,Nat>,-full: List<&2,Nat>> is Data:
  Chain{cells: {Nat.add(U32.to_nat(TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset)),
        digit_sum(List.append(&2,Views.Axis,kept,[Views.Axis{Tiled.groups(extent),group_step(shape,axis,position)}]),List.append(&2,Bool,kept_flags,[False{}]),
          List.append(&2,Nat,others,[group]))) == cell_value(shape,axis,position,cell) : Nat},
    cell_below: Coordinates.DigitsBelow(shape,cell),
    source: {Coordinates.inserted_digit(axis,Geometry.removed_at(cell,axis),Nat.add(lane_digit,Nat.mul(Nat.div(Digits.digit_at(cell,axis),8n),8n))) == full : List<&2,Nat>},
    full_below: Coordinates.DigitsBelow(shape,full),
    logical: {Nat.add(U32.to_nat(offset),digit_sum(axes,flags,digits)) == ShapeLaws.digits_value(shape,full) : Nat}}

def chain_of(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,+view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,
  +others: List<&2,Nat>,+group: Nat,+lane_digit: Nat,+inside: {Nat.is_lt(Nat.add(lane_digit,Nat.mul(group,8n)),U32.to_nat(extent)) == True{} : Bool},
  +below: Coordinates.DigitsBelow(Grid.removed(Views.axis_extents(axes),lane),others),
  +walk: CellWalk<shape,axis,position,Some{lane},kept,kept_flags,0n,List.length(&2,Views.Axis,axes),Digits.words_nat(Grid.coordinates(shape,offset)),
    Coordinates.inserted_digit(lane,others,Nat.add(lane_digit,Nat.mul(group,8n))),
    walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,False{},Digits.words_nat(Grid.coordinates(shape,offset)),Coordinates.inserted_digit(lane,others,Nat.add(lane_digit,Nat.mul(group,8n)))),
    walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,True{},Digits.words_nat(Grid.coordinates(shape,offset)),Coordinates.inserted_digit(lane,others,Nat.add(lane_digit,Nat.mul(group,8n))))>) ->
  Chain<shape,axis,position,offset,axes,flags,kept,kept_flags,extent,others,group,Coordinates.inserted_digit(lane,others,Nat.add(lane_digit,Nat.mul(group,8n))),lane_digit,
    Digits.added(walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,False{},Digits.words_nat(Grid.coordinates(shape,offset)),
      Coordinates.inserted_digit(lane,others,Nat.add(lane_digit,Nat.mul(group,8n)))),unit_digits(List.length(&2,U32,shape),axis,Nat.mul(group,8n))),
    walked(axes,flags,shape,TiledView.lanes_of(shape,axis,position),axis,True{},Digits.words_nat(Grid.coordinates(shape,offset)),Coordinates.inserted_digit(lane,others,Nat.add(lane_digit,Nat.mul(group,8n))))>:
  KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  CellWalk{+cells,+aligned,+split} = walk
  +lanes = TiledView.lanes_of(shape,axis,position)
  +words = Grid.coordinates(shape,offset)
  +start = Digits.words_nat(words)
  +rank = List.length(&2,U32,shape)
  +target = Views.axis_extents(axes)
  +channel = Nat.add(lane_digit,Nat.mul(group,8n))
  +digits = Coordinates.inserted_digit(lane,others,channel)
  +kept_walk = walked(axes,flags,shape,lanes,axis,False{},start,digits)
  +full_walk = walked(axes,flags,shape,lanes,axis,True{},start,digits)
  +within = source_within(limits,shape,axis,position,facts)
  +others_length = Values.below_length(Grid.removed(target,lane),others,below)
  +target_length = Equal.trans(Nat,List.length(&2,U32,target),1n+List.length(&2,U32,Grid.removed(target,lane)),1n+List.length(&2,Nat,others),
    Equal.sym(Nat,1n+List.length(&2,U32,Grid.removed(target,lane)),List.length(&2,U32,target),Public.removed_length(target,lane,lane_inside)),
    Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,Grid.removed(target,lane)),List.length(&2,Nat,others),
      Equal.sym(Nat,List.length(&2,Nat,others),List.length(&2,U32,Grid.removed(target,lane)),others_length)))
  +room_lane = Division.lt_succ_le(lane,List.length(&2,Nat,others),{%target_length : {Nat.is_lt(lane,_) == True{} : Bool} lane_inside :
    {Nat.is_lt(lane,1n+List.length(&2,Nat,others)) == True{} : Bool}})
  +removed_digits = Geometry.removed_inserted(lane,others,channel,room_lane)
  +lane_channel = Geometry.digit_inserted(lane,others,channel,room_lane)
  +place = Order.subtract_zero(lane)
  +cells_others = {%removed_digits : {Nat.add(cell_value(shape,axis,position,start),digit_sum(kept,kept_flags,_)) == cell_value(shape,axis,position,kept_walk) : Nat}
    %place : {Nat.add(cell_value(shape,axis,position,start),digit_sum(kept,kept_flags,Geometry.removed_at(digits,_))) == cell_value(shape,axis,position,kept_walk) : Nat}
    cells : {Nat.add(cell_value(shape,axis,position,start),digit_sum(kept,kept_flags,others)) == cell_value(shape,axis,position,kept_walk) : Nat}}
  +split_channel = {%lane_channel : {full_walk == Digits.added(kept_walk,unit_digits(rank,axis,_)) : List<&2,Nat>}
    %place : {full_walk == Digits.added(kept_walk,unit_digits(rank,axis,Digits.digit_at(digits,_))) : List<&2,Nat>}
    split : {full_walk == Digits.added(kept_walk,unit_digits(rank,axis,channel)) : List<&2,Nat>}}
  +digits_below = Values.inserted_below(target,lane,others,channel,below,{%extent_exact : {Nat.is_lt(channel,U32.to_nat(_)) == True{} : Bool} inside :
    {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(target,lane))) == True{} : Bool}},lane_inside)
  +full_below = below_le(shape,full_walk,walked(axes,flags,shape,lanes,axis,True{},start,mirrored_lasts(axes,flags)),walked_le(axes,flags,shape,lanes,axis,start,digits,digits_below),bound)
  +kept_length = walked_length(axes,flags,shape,lanes,axis,False{},start,digits,start_length)
  +unit_channel = unit_digits(rank,axis,channel)
  +unit_group = unit_digits(rank,axis,Nat.mul(group,8n))
  +pair_below = {%split_channel : Coordinates.DigitsBelow(shape,_) full_below : Coordinates.DigitsBelow(shape,Digits.added(kept_walk,unit_channel))}
  +parts = below_added(shape,kept_walk,unit_channel,pair_below,lengths_same(kept_walk,unit_channel,shape,kept_length,unit_length(rank,axis,channel)))
  +channel_below = second_below(shape,kept_walk,unit_channel,parts)
  +group_le = Order.right_le(lane_digit,Nat.mul(group,8n))
  +cell = Digits.added(kept_walk,unit_group)
  +cell_below = below_le(shape,cell,Digits.added(kept_walk,unit_channel),
    added_le(kept_walk,kept_walk,unit_group,unit_channel,le_reflexive(kept_walk),unit_le(rank,axis,Nat.mul(group,8n),channel,group_le)),pair_below)
  +group_below = unit_below_le(shape,axis,Nat.mul(group,8n),channel,group_le,channel_below)
  +group_cells = group_term(group,shape,axis,position,limits,facts,group_below)
  +group_aligned = self_aligned(rank,axis,Nat.mul(group,8n),within,aligned_times(group,8n,{==}))
  +cell_sum = cell_added(shape,axis,position,kept_walk,unit_group,within,kept_length,unit_length(rank,axis,Nat.mul(group,8n)),group_aligned)
  +offset_word = Equal.trans(U32,offset,U32.from_nat(U32.to_nat(offset)),U32.from_nat(ShapeLaws.digits_value(shape,start)),
    Equal.sym(U32,U32.from_nat(U32.to_nat(offset)),offset,Arithmetic.native_roundtrip(offset)),
    Equal.cong(Nat,U32,number => U32.from_nat(number),U32.to_nat(offset),ShapeLaws.digits_value(shape,start),Equal.sym(Nat,ShapeLaws.digits_value(shape,start),U32.to_nat(offset),start_value)))
  +offset_cell = {%Equal.sym(U32,offset,U32.from_nat(ShapeLaws.digits_value(shape,start)),offset_word) : {U32.to_nat(TiledView.cell_index(lanes,_)) == cell_value(shape,axis,position,start) : Nat}
    cell_of(shape,axis,position,start,limits,within,start_below,source_room(limits,shape,axis,position,facts),source_cells_room(limits,shape,axis,position,facts)) :
    {U32.to_nat(TiledView.cell_index(lanes,offset)) == cell_value(shape,axis,position,start) : Nat}}
  +kept_count = Equal.trans(Nat,List.length(&2,Nat,others),List.length(&2,U32,Grid.removed(target,lane)),List.length(&2,Views.Axis,kept),others_length,
    Equal.trans(Nat,List.length(&2,U32,Grid.removed(target,lane)),List.length(&2,U32,Views.axis_extents(kept)),List.length(&2,Views.Axis,kept),
      Equal.cong(List<&2,U32>,Nat,list => List.length(&2,U32,list),Grid.removed(target,lane),Views.axis_extents(kept),Equal.sym(List<&2,U32>,Views.axis_extents(kept),Grid.removed(target,lane),kept_extents)),
      MetadataLaws.extents_length(kept)))
  +step = group_step(shape,axis,position)
  +offset_value = U32.to_nat(TiledView.cell_index(lanes,offset))
  +later_sum = digit_sum(kept,kept_flags,others)
  +group_sum = Nat.mul(group,U32.to_nat(step))
  +appended = sum_appended(kept,kept_flags,others,Tiled.groups(extent),step,group,flags_length,kept_count)
  +cell_digit = Digits.added_digit(kept_walk,unit_group,axis,{%Equal.sym(Nat,List.length(&2,Nat,kept_walk),rank,kept_length) : {Nat.is_lt(axis,_) == True{} : Bool} within :
      {Nat.is_lt(axis,List.length(&2,Nat,kept_walk)) == True{} : Bool}},
    lengths_same(unit_group,kept_walk,shape,unit_length(rank,axis,Nat.mul(group,8n)),kept_length))
  +cell_aligned = {%Equal.sym(Nat,Digits.digit_at(cell,axis),Nat.add(Digits.digit_at(kept_walk,axis),Digits.digit_at(unit_group,axis)),cell_digit) : {Nat.mod(_,8n) == 0n : Nat}
    mod_sum(Digits.digit_at(kept_walk,axis),Digits.digit_at(unit_group,axis),aligned,group_aligned) : {Nat.mod(Digits.digit_at(cell,axis),8n) == 0n : Nat}}
  +cell_length = Equal.trans(Nat,List.length(&2,Nat,cell),List.length(&2,Nat,kept_walk),rank,
    Digits.added_length(kept_walk,unit_group,lengths_same(unit_group,kept_walk,shape,unit_length(rank,axis,Nat.mul(group,8n)),kept_length)),kept_length)
  +unit_lane = unit_digits(rank,axis,lane_digit)
  Chain{
    Equal.trans(Nat,Nat.add(offset_value,digit_sum(List.append(&2,Views.Axis,kept,[Views.Axis{Tiled.groups(extent),step}]),List.append(&2,Bool,kept_flags,[False{}]),List.append(&2,Nat,others,[group]))),
      Nat.add(offset_value,Nat.add(later_sum,group_sum)),cell_value(shape,axis,position,cell),
      Equal.cong(Nat,Nat,number => Nat.add(offset_value,number),
        digit_sum(List.append(&2,Views.Axis,kept,[Views.Axis{Tiled.groups(extent),step}]),List.append(&2,Bool,kept_flags,[False{}]),List.append(&2,Nat,others,[group])),
        Nat.add(later_sum,group_sum),appended),
      Equal.trans(Nat,Nat.add(offset_value,Nat.add(later_sum,group_sum)),Nat.add(Nat.add(offset_value,later_sum),group_sum),cell_value(shape,axis,position,cell),
        Equal.sym(Nat,Nat.add(Nat.add(offset_value,later_sum),group_sum),Nat.add(offset_value,Nat.add(later_sum,group_sum)),Addition.associative(offset_value,later_sum,group_sum)),
        Equal.trans(Nat,Nat.add(Nat.add(offset_value,later_sum),group_sum),Nat.add(cell_value(shape,axis,position,kept_walk),cell_value(shape,axis,position,unit_group)),
          cell_value(shape,axis,position,cell),
          %Equal.sym(Nat,offset_value,cell_value(shape,axis,position,start),offset_cell) :
            {Nat.add(Nat.add(_,later_sum),group_sum) == Nat.add(cell_value(shape,axis,position,kept_walk),cell_value(shape,axis,position,unit_group)) : Nat}
          %Equal.sym(Nat,Nat.add(cell_value(shape,axis,position,start),later_sum),cell_value(shape,axis,position,kept_walk),cells_others) :
            {Nat.add(_,group_sum) == Nat.add(cell_value(shape,axis,position,kept_walk),cell_value(shape,axis,position,unit_group)) : Nat}
          Equal.cong(Nat,Nat,number => Nat.add(cell_value(shape,axis,position,kept_walk),number),group_sum,cell_value(shape,axis,position,unit_group),group_cells),
          Equal.sym(Nat,cell_value(shape,axis,position,cell),Nat.add(cell_value(shape,axis,position,kept_walk),cell_value(shape,axis,position,unit_group)),cell_sum)))),
    cell_below,
    %aligned_exact(Digits.digit_at(cell,axis),cell_aligned) :
      {Coordinates.inserted_digit(axis,Geometry.removed_at(cell,axis),Nat.add(lane_digit,_)) == full_walk : List<&2,Nat>}
    Equal.trans(List<&2,Nat>,Coordinates.inserted_digit(axis,Geometry.removed_at(cell,axis),Nat.add(lane_digit,Digits.digit_at(cell,axis))),
      Digits.added(cell,unit_digits(List.length(&2,Nat,cell),axis,lane_digit)),full_walk,
      inserted_added(cell,axis,lane_digit,{%Equal.sym(Nat,List.length(&2,Nat,cell),rank,cell_length) : {Nat.is_lt(axis,_) == True{} : Bool} within :
        {Nat.is_lt(axis,List.length(&2,Nat,cell)) == True{} : Bool}}),
      %Equal.sym(Nat,List.length(&2,Nat,cell),rank,cell_length) : {Digits.added(cell,unit_digits(_,axis,lane_digit)) == full_walk : List<&2,Nat>}
      Equal.trans(List<&2,Nat>,Digits.added(cell,unit_lane),Digits.added(kept_walk,Digits.added(unit_group,unit_lane)),full_walk,
        added_assoc(kept_walk,unit_group,unit_lane,lengths_same(kept_walk,unit_group,shape,kept_length,unit_length(rank,axis,Nat.mul(group,8n))),
          lengths_same(kept_walk,unit_lane,shape,kept_length,unit_length(rank,axis,lane_digit))),
        %Equal.sym(List<&2,Nat>,Digits.added(unit_group,unit_lane),unit_digits(rank,axis,Nat.add(Nat.mul(group,8n),lane_digit)),units_added(rank,axis,Nat.mul(group,8n),lane_digit)) : {Digits.added(kept_walk,_) == full_walk : List<&2,Nat>}
        %Order.commutative(lane_digit,Nat.mul(group,8n)) : {Digits.added(kept_walk,unit_digits(rank,axis,_)) == full_walk : List<&2,Nat>}
        Equal.sym(List<&2,Nat>,full_walk,Digits.added(kept_walk,unit_channel),split_channel))),
    full_below,
    Equal.trans(Nat,Nat.add(U32.to_nat(offset),digit_sum(axes,flags,digits)),Nat.add(ShapeLaws.digits_value(shape,start),digit_sum(axes,flags,digits)),
      ShapeLaws.digits_value(shape,full_walk),
      Equal.cong(Nat,Nat,number => Nat.add(number,digit_sum(axes,flags,digits)),U32.to_nat(offset),ShapeLaws.digits_value(shape,start),
        Equal.sym(Nat,ShapeLaws.digits_value(shape,start),U32.to_nat(offset),start_value)),
      Equal.sym(Nat,ShapeLaws.digits_value(shape,full_walk),Nat.add(ShapeLaws.digits_value(shape,start),digit_sum(axes,flags,digits)),
        logical_walk(axes,flags,0n,shape,axis,position,limits,facts,words,start,digits,start_length,digits_below,valid)))}

# ---- The view of the cells fits the cells ----

# The axes of the view of the cells: the kept axes, then the groups of the
# lane axis of the target.
def cell_axes_of(+kept: List<&2,Views.Axis>,+extent: U32,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position) -> List<&2,Views.Axis>:
  List.append(&2,Views.Axis,kept,[Views.Axis{Tiled.groups(extent),group_step(shape,axis,position)}])

# An axis of extents without a zero has a nonzero extent.
def lanes_nonzero(shape: List<&2,U32>,axis: Nat,+nonempty: {Shapes.has_zero(shape) == False{} : Bool},
  +inside: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool}) -> {U32.is_zero(Grid.lanes_at(shape,axis)) == False{} : Bool}:
  match shape axis:
    case Con{+extent,+rest} 0n: Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(rest),nonempty)
    case Con{+extent,+rest} 1n+ +next: lanes_nonzero(rest,next,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonempty),inside)
    case Nil{} +axis: Empty.absurd({U32.is_zero(Grid.lanes_at(Nil{},axis)) == False{} : Bool},Order.below_zero(axis,inside))

# Extents without a zero keep none when one axis is removed.
def removed_nonzero(shape: List<&2,U32>,axis: Nat,+nonempty: {Shapes.has_zero(shape) == False{} : Bool}) ->
  {Shapes.has_zero(Grid.removed(shape,axis)) == False{} : Bool}:
  match shape axis:
    case Nil{} +axis: {==}
    case Con{+extent,+rest} 0n: Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonempty)
    case Con{+extent,+rest} 1n+ +next:
      %Equal.sym(Bool,U32.is_zero(extent),False{},Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(rest),nonempty)) :
        {_ || Shapes.has_zero(Grid.removed(rest,next)) == False{} : Bool}
      removed_nonzero(rest,next,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonempty))

# Extents without a zero and a nonzero last extent have no zero.
def appended_nonzero(extents: List<&2,U32>,+last: U32,+nonempty: {Shapes.has_zero(extents) == False{} : Bool},+nonzero: {U32.is_zero(last) == False{} : Bool}) ->
  {Shapes.has_zero(List.append(&2,U32,extents,[last])) == False{} : Bool}:
  match extents:
    case Nil{}: %Equal.sym(Bool,U32.is_zero(last),False{},nonzero) : {_ || False{} == False{} : Bool} {==}
    case Con{+extent,+rest}:
      %Equal.sym(Bool,U32.is_zero(extent),False{},Boolean.or_false_left(U32.is_zero(extent),Shapes.has_zero(rest),nonempty)) :
        {_ || Shapes.has_zero(List.append(&2,U32,rest,[last])) == False{} : Bool}
      appended_nonzero(rest,last,Boolean.or_false_right(U32.is_zero(extent),Shapes.has_zero(rest),nonempty),nonzero)

# The cell of digits below values lies below the cells.
def cell_value_below(+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+digits: List<&2,Nat>,+limits: Machine.Limits,
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+grouped: TiledValues.Groups<limits,Grid.lanes_at(shape,axis)>,
  +below: Coordinates.DigitsBelow(shape,digits)) ->
  {Nat.is_lt(cell_value(shape,axis,position,digits),ShapeLaws.elements(Tiled.cell_shape(shape,axis,position))) == True{} : Bool}:
  +lanes = Grid.lanes_at(shape,axis)
  +group = Nat.div(Digits.digit_at(digits,axis),8n)
  +placed = Coordinates.placed_digits(True{},position,Geometry.removed_at(digits,axis),group)
  +cell_below = {%Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,shape)),True{},within) :
      Coordinates.DigitsBelow(Grid.placed(_,position,Grid.removed(shape,axis),Grid.groups(lanes)),placed)
    Coordinates.placed_below(True{},position,Grid.removed(shape,axis),Grid.groups(lanes),Geometry.removed_at(digits,axis),group,Values.removed_below(shape,digits,axis,below),
      TiledValues.group_inside(limits,lanes,Digits.digit_at(digits,axis),grouped,Values.digit_below(shape,digits,axis,below,within))) :
    Coordinates.DigitsBelow(Tiled.cell_shape(shape,axis,position),placed)}
  Coordinates.digits_below(Tiled.cell_shape(shape,axis,position),placed,cell_below)

# The chain at the last digits of the kept axes and the last group bounds
# the reach of the view of the cells.
def fits_of(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,+view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,
  +nonempty: {Shapes.has_zero(Views.axis_extents(cell_axes_of(kept,extent,shape,axis,position))) == False{} : Bool},
  +digits: List<&2,Nat>,+cell: List<&2,Nat>,+full: List<&2,Nat>,
  +chain: Chain<shape,axis,position,offset,axes,flags,kept,kept_flags,extent,mirrored_lasts(kept,kept_flags),Nat.sub(U32.to_nat(Tiled.groups(extent)),1n),digits,0n,cell,full>) ->
  {Views.fits(Views.View{cell_axes_of(kept,extent,shape,axis,position),TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset)},
    Tensors.count_or_zero(Tiled.cell_shape(shape,axis,position))) == True{} : Bool}:
  KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  Chain{+cells,+cell_below,+source,+full_below,+logical} = chain
  +cell_axes = cell_axes_of(kept,extent,shape,axis,position)
  +cell_flags = List.append(&2,Bool,kept_flags,[False{}])
  +offset_cell = TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset)
  +offset_value = U32.to_nat(offset_cell)
  +cell_shape = Tiled.cell_shape(shape,axis,position)
  +length = Tensors.count_or_zero(cell_shape)
  +lasts = List.append(&2,Nat,mirrored_lasts(kept,kept_flags),[Nat.sub(U32.to_nat(Tiled.groups(extent)),1n)])
  +value_below = cell_value_below(shape,axis,position,cell,limits,source_within(limits,shape,axis,position,facts),grouped,cell_below)
  ViewBounds.fits_span(cell_axes,offset_cell,length,nonempty,
    %lasts_sum(cell_axes,cell_flags,nonempty) : {Nat.is_lt(Nat.add(offset_value,_),U32.to_nat(length)) == True{} : Bool}
    %Equal.sym(List<&2,Nat>,mirrored_lasts(cell_axes,cell_flags),lasts,lasts_appended(kept,kept_flags,Tiled.groups(extent),group_step(shape,axis,position),flags_length)) :
      {Nat.is_lt(Nat.add(offset_value,digit_sum(cell_axes,cell_flags,_)),U32.to_nat(length)) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(offset_value,digit_sum(cell_axes,cell_flags,lasts)),cell_value(shape,axis,position,cell),cells) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(length),ShapeLaws.elements(cell_shape),cells_count) : {Nat.is_lt(cell_value(shape,axis,position,cell),_) == True{} : Bool}
    value_below)

# The lane axis of the target of a kept view has lanes.
def extent_positive(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,+view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>) ->
  {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}:
  KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  Arithmetic.word_nonzero_positive(extent,
    %Equal.sym(U32,extent,Grid.lanes_at(Views.axis_extents(axes),lane),extent_exact) : {U32.is_zero(_) == False{} : Bool}
    lanes_nonzero(Views.axis_extents(axes),lane,target_nonempty,lane_inside))

# The kept axes of a kept view have no zero extent.
def kept_nonzero(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,+view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>) ->
  {Shapes.has_zero(Views.axis_extents(kept)) == False{} : Bool}:
  KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  %Equal.sym(List<&2,U32>,Views.axis_extents(kept),Grid.removed(Views.axis_extents(axes),lane),kept_extents) : {Shapes.has_zero(_) == False{} : Bool}
  removed_nonzero(Views.axis_extents(axes),lane,target_nonempty)

# The view of the cells of a kept view has no zero extent.
def cell_nonempty(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,+view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>) ->
  {Shapes.has_zero(Views.axis_extents(cell_axes_of(kept,extent,shape,axis,position))) == False{} : Bool}:
  KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  +groups = Tiled.groups(extent)
  +groups_nonzero = Arithmetic.word_positive_nonzero(groups,Coordinates.groups_positive(limits,extent,target_grouped,extent_positive(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view)))
  %Equal.sym(List<&2,U32>,Views.axis_extents(cell_axes_of(kept,extent,shape,axis,position)),List.append(&2,U32,Views.axis_extents(kept),[groups]),
      extents_appended(kept,groups,group_step(shape,axis,position))) : {Shapes.has_zero(_) == False{} : Bool}
  appended_nonzero(Views.axis_extents(kept),groups,kept_nonzero(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view),groups_nonzero)

# The view of the cells fits the cells of the input.
def cell_fits(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,+view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>) ->
  {Views.fits(Views.View{cell_axes_of(kept,extent,shape,axis,position),TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset)},
    Tensors.count_or_zero(Tiled.cell_shape(shape,axis,position))) == True{} : Bool}:
  KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  +target = Views.axis_extents(axes)
  +groups = Tiled.groups(extent)
  +positive = extent_positive(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view)
  +kept_whole = kept_nonzero(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view)
  +nonempty = cell_nonempty(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view)
  +lasts = mirrored_lasts(kept,kept_flags)
  +last_group = Nat.sub(U32.to_nat(groups),1n)
  +lasts_below_target = {%kept_extents : Coordinates.DigitsBelow(_,lasts) lasts_below(kept,kept_flags,kept_whole) : Coordinates.DigitsBelow(Grid.removed(target,lane),lasts)}
  +inside = PaddedCells.last_below(limits,extent,target_grouped,positive)
  +channel = Nat.add(0n,Nat.mul(last_group,8n))
  +digits = Coordinates.inserted_digit(lane,lasts,channel)
  +digits_below = Values.inserted_below(target,lane,lasts,channel,lasts_below_target,
    {%extent_exact : {Nat.is_lt(channel,U32.to_nat(_)) == True{} : Bool} inside : {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(target,lane))) == True{} : Bool}},lane_inside)
  +lanes = TiledView.lanes_of(shape,axis,position)
  +start = Digits.words_nat(Grid.coordinates(shape,offset))
  fits_of(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,nonempty,digits,
    Digits.added(walked(axes,flags,shape,lanes,axis,False{},start,digits),unit_digits(List.length(&2,U32,shape),axis,Nat.mul(last_group,8n))),
    walked(axes,flags,shape,lanes,axis,True{},start,digits),
    chain_of(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,lasts,last_group,0n,inside,lasts_below_target,
      walk_at(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,digits,digits_below)))

# ---- The lanes of a target cell ----

# The extents of the view of the cells are the cells of the target.
def cell_extents(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,+view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>) ->
  {Views.axis_extents(cell_axes_of(kept,extent,shape,axis,position)) == Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}) : List<&2,U32>}:
  KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  +target = Views.axis_extents(axes)
  +removed = Grid.removed(target,lane)
  Equal.trans(List<&2,U32>,Views.axis_extents(cell_axes_of(kept,extent,shape,axis,position)),List.append(&2,U32,Views.axis_extents(kept),[Tiled.groups(extent)]),
    Tiled.cell_shape(target,lane,Grid.Innermost{}),extents_appended(kept,Tiled.groups(extent),group_step(shape,axis,position)),
    %Equal.sym(List<&2,U32>,Views.axis_extents(kept),removed,kept_extents) : {List.append(&2,U32,_,[Tiled.groups(extent)]) == Tiled.cell_shape(target,lane,Grid.Innermost{}) : List<&2,U32>}
    %Equal.sym(U32,extent,Grid.lanes_at(target,lane),extent_exact) : {List.append(&2,U32,removed,[Grid.groups(_)]) == Tiled.cell_shape(target,lane,Grid.Innermost{}) : List<&2,U32>}
    %Equal.sym(Bool,Nat.is_lt(lane,List.length(&2,U32,target)),True{},lane_inside) :
      {List.append(&2,U32,removed,[Grid.groups(Grid.lanes_at(target,lane))]) == Grid.placed(_,Grid.Innermost{},removed,Grid.groups(Grid.lanes_at(target,lane))) : List<&2,U32>}
    {==})

# The lane the source digits of a lane read lies below the lanes of the
# source.
def source_lane_below(+shape: List<&2,U32>,+axis: Nat,+cell: List<&2,Nat>,+full: List<&2,Nat>,+lane_digit: Nat,
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+cell_below: Coordinates.DigitsBelow(shape,cell),
  +source: {Coordinates.inserted_digit(axis,Geometry.removed_at(cell,axis),Nat.add(lane_digit,Nat.mul(Nat.div(Digits.digit_at(cell,axis),8n),8n))) == full : List<&2,Nat>},
  +full_below: Coordinates.DigitsBelow(shape,full)) ->
  {Nat.is_lt(Nat.add(lane_digit,Nat.mul(Nat.div(Digits.digit_at(cell,axis),8n),8n)),U32.to_nat(Grid.lanes_at(shape,axis))) == True{} : Bool}:
  +channel = Nat.add(lane_digit,Nat.mul(Nat.div(Digits.digit_at(cell,axis),8n),8n))
  +others = Geometry.removed_at(cell,axis)
  +length = Values.below_length(shape,cell,cell_below)
  +inside = {%Equal.sym(Nat,List.length(&2,Nat,cell),List.length(&2,U32,shape),length) : {Nat.is_lt(axis,_) == True{} : Bool} within :
    {Nat.is_lt(axis,List.length(&2,Nat,cell)) == True{} : Bool}}
  +room = Division.lt_succ_le(axis,List.length(&2,Nat,others),
    {%Equal.sym(Nat,1n+List.length(&2,Nat,others),List.length(&2,Nat,cell),Geometry.length_removed(cell,axis,inside)) : {Nat.is_lt(axis,_) == True{} : Bool} inside :
      {Nat.is_lt(axis,1n+List.length(&2,Nat,others)) == True{} : Bool}})
  %Geometry.digit_inserted(axis,others,channel,room) : {Nat.is_lt(_,U32.to_nat(Grid.lanes_at(shape,axis))) == True{} : Bool}
  %Equal.sym(List<&2,Nat>,Coordinates.inserted_digit(axis,others,channel),full,source) : {Nat.is_lt(Digits.digit_at(_,axis),U32.to_nat(Grid.lanes_at(shape,axis))) == True{} : Bool}
  Values.digit_below(shape,full,axis,full_below,within)

# A lane below the extent of the target: the source lane the view of the
# cells reads holds the value the view reads.
def lane_hit(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,+word: U32,+others: List<&2,Nat>,+group: Nat,
  +others_below: Coordinates.DigitsBelow(Grid.removed(Views.axis_extents(axes),lane),others),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Views.axis_extents(axes),lane)))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),Coordinates.placed_digits(True{},Grid.Innermost{},others,group)) : Nat},
  +address: U32,+address_exact: {U32.to_nat(address) == Nat.add(U32.to_nat(TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset)),
    digit_sum(cell_axes_of(kept,extent,shape,axis,position),List.append(&2,Bool,kept_flags,[False{}]),List.append(&2,Nat,others,[group]))) : Nat},
  +lane_digit: Nat,+small: {Nat.is_le(lane_digit,7n) == True{} : Bool},+inside: {Nat.is_lt(Nat.add(lane_digit,Nat.mul(group,8n)),U32.to_nat(extent)) == True{} : Bool},
  +cell: List<&2,Nat>,+full: List<&2,Nat>,
  +chain: Chain<shape,axis,position,offset,axes,flags,kept,kept_flags,extent,others,group,Coordinates.inserted_digit(lane,others,Nat.add(lane_digit,Nat.mul(group,8n))),lane_digit,cell,full>) ->
  {PaddedCells.kept(U32.to_nat(extent),group,lane_digit,TileLaws.select_lane(lane_digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},address))) ==
    TileLaws.select_lane(lane_digit,Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),
      Representation.Lifted{ViewGather.Viewing{logical,axes,offset,flags},Public.geometry_of(Views.axis_extents(axes),lane,Grid.Innermost{})},word)) : F32}:
  KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  Chain{+cells,+cell_below,+source,+full_below,+logical_sum} = chain
  +target = Views.axis_extents(axes)
  +within = source_within(limits,shape,axis,position,facts)
  +channel = Nat.add(lane_digit,Nat.mul(group,8n))
  +digits = Coordinates.inserted_digit(lane,others,channel)
  +others_source = Geometry.removed_at(cell,axis)
  +group_source = Nat.div(Digits.digit_at(cell,axis),8n)
  +channel_source = Nat.add(lane_digit,Nat.mul(group_source,8n))
  +viewing = {ViewGather.Viewing{logical,axes,offset,flags} : ViewGather.Viewing<Logical>}
  +inner = Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},address)
  +outer = Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),Representation.Lifted{viewing,Public.geometry_of(target,lane,Grid.Innermost{})},word)
  +offset_cell = U32.to_nat(TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset))
  +cell_exact = Equal.trans(Nat,U32.to_nat(address),
    Nat.add(offset_cell,digit_sum(cell_axes_of(kept,extent,shape,axis,position),List.append(&2,Bool,kept_flags,[False{}]),List.append(&2,Nat,others,[group]))),
    cell_value(shape,axis,position,cell),address_exact,cells)
  +source_lane = Coordinates.lane_values(~Logical,~original,logical,limits,shape,axis,position,True{},others_source,group_source,address,within,
    Bounds.strict_le(axis,List.length(&2,U32,shape),within),removed_room,Values.removed_below(shape,cell,axis,cell_below),
    TiledValues.group_inside(limits,Grid.lanes_at(shape,axis),Digits.digit_at(cell,axis),grouped,Values.digit_below(shape,cell,axis,cell_below,within)),cell_exact,lane_digit,small)
  +source_inside = source_lane_below(shape,axis,cell,full,lane_digit,within,cell_below,source,full_below)
  +full_value = original(logical,U32.from_nat(ShapeLaws.digits_value(shape,full)))
  +source_value = {%source : {TileLaws.select_lane(lane_digit,inner) == original(logical,U32.from_nat(ShapeLaws.digits_value(shape,_))) : F32}
    %source_inside : {TileLaws.select_lane(lane_digit,inner) == Bool.pick(F32,_,original(logical,U32.from_nat(ShapeLaws.digits_value(shape,Coordinates.inserted_digit(axis,others_source,channel_source)))),0.0) : F32}
    source_lane : {TileLaws.select_lane(lane_digit,inner) == full_value : F32}}
  +target_inside = {%extent_exact : {Nat.is_lt(channel,U32.to_nat(_)) == True{} : Bool} inside : {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(target,lane))) == True{} : Bool}}
  +digits_below = Values.inserted_below(target,lane,others,channel,others_below,target_inside,lane_inside)
  +target_lane = Coordinates.lane_values(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),viewing,limits,target,lane,Grid.Innermost{},True{},others,group,word,
    lane_inside,Bounds.strict_le(lane,List.length(&2,U32,target),lane_inside),target_removed_room,others_below,group_inside,exact,lane_digit,small)
  +index_value = ShapeLaws.digits_value(target,digits)
  +index = U32.from_nat(index_value)
  +index_below = Coordinates.digits_below(target,digits,digits_below)
  +index_exact = FullRange.roundtrip(index_value,Machine.capacity(limits),
    Order.transitive(index_value,ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(index_value,ShapeLaws.elements(target),index_below),target_room))
  +index_inside = {%Equal.sym(Nat,U32.to_nat(index),index_value,index_exact) : {Nat.is_lt(_,ShapeLaws.elements(target)) == True{} : Bool} index_below :
    {Nat.is_lt(U32.to_nat(index),ShapeLaws.elements(target)) == True{} : Bool}}
  +address_target = (Views.mirrored_offset(axes,flags,offset) + Box.stepped_sum(target,ViewPart.view_steps(Views.mirrored_axes(axes,flags)),U32.to_nat(index)) : U32)
  +read = ViewReads.address_at(axes,flags,offset,U32.to_nat(index),Tensors.count_or_zero(shape),Machine.capacity(limits),Limits.capacity_fits(limits),target_room,length_room,view_fits,index_inside)
  +full_number = ShapeLaws.digits_value(shape,full)
  +address_nat = Equal.trans(Nat,U32.to_nat(address_target),Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,U32.to_nat(index))),full_number,
    ViewGather.read_exact_of(offset,axes,flags,U32.to_nat(index),address_target,Tensors.count_or_zero(shape),read),
    %Equal.sym(Nat,U32.to_nat(index),index_value,index_exact) : {Nat.add(U32.to_nat(offset),ViewReads.view_sum(axes,flags,_)) == full_number : Nat}
    %Equal.sym(Nat,ViewReads.view_sum(axes,flags,index_value),digit_sum(axes,flags,digits),view_sum_digits(axes,flags,digits,digits_below)) :
      {Nat.add(U32.to_nat(offset),_) == full_number : Nat}
    logical_sum)
  +address_word = Equal.trans(U32,address_target,U32.from_nat(U32.to_nat(address_target)),U32.from_nat(full_number),
    Equal.sym(U32,U32.from_nat(U32.to_nat(address_target)),address_target,Arithmetic.native_roundtrip(address_target)),
    Equal.cong(Nat,U32,number => U32.from_nat(number),U32.to_nat(address_target),full_number,address_nat))
  +target_value = {%target_inside : {TileLaws.select_lane(lane_digit,outer) == Bool.pick(F32,_,ViewGather.viewed(~F32,~Logical,~original,viewing,index),0.0) : F32}
    target_lane : {TileLaws.select_lane(lane_digit,outer) == original(logical,address_target) : F32}}
  %Equal.sym(Bool,Nat.is_lt(channel,U32.to_nat(extent)),True{},inside) : {Bool.pick(F32,_,TileLaws.select_lane(lane_digit,inner),0.0) == TileLaws.select_lane(lane_digit,outer) : F32}
  Equal.trans(F32,TileLaws.select_lane(lane_digit,inner),full_value,TileLaws.select_lane(lane_digit,outer),source_value,
    Equal.trans(F32,full_value,original(logical,address_target),TileLaws.select_lane(lane_digit,outer),
      Equal.cong(U32,F32,place => original(logical,place),U32.from_nat(full_number),address_target,Equal.sym(U32,address_target,U32.from_nat(full_number),address_word)),
      Equal.sym(F32,TileLaws.select_lane(lane_digit,outer),original(logical,address_target),target_value)))

# A lane of a cell the view of the cells reads agrees with the lane of the
# target cell: the value the view reads below the extent, +0 past it.
def lane_agree(~Logical: Data,~original: Logical -> U32 -> F32,valid: Bool,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,+word: U32,+others: List<&2,Nat>,+group: Nat,
  +others_below: Coordinates.DigitsBelow(Grid.removed(Views.axis_extents(axes),lane),others),
  +group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(Views.axis_extents(axes),lane)))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}),Coordinates.placed_digits(True{},Grid.Innermost{},others,group)) : Nat},
  +address: U32,+address_exact: {U32.to_nat(address) == Nat.add(U32.to_nat(TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset)),
    digit_sum(cell_axes_of(kept,extent,shape,axis,position),List.append(&2,Bool,kept_flags,[False{}]),List.append(&2,Nat,others,[group]))) : Nat},
  +lane_digit: Nat,+small: {Nat.is_le(lane_digit,7n) == True{} : Bool},+classified: {Nat.is_lt(Nat.add(lane_digit,Nat.mul(group,8n)),U32.to_nat(extent)) == valid : Bool}) ->
  {PaddedCells.kept(U32.to_nat(extent),group,lane_digit,TileLaws.select_lane(lane_digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},address))) ==
    TileLaws.select_lane(lane_digit,Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),
      Representation.Lifted{ViewGather.Viewing{logical,axes,offset,flags},Public.geometry_of(Views.axis_extents(axes),lane,Grid.Innermost{})},word)) : F32}:
  match valid:
    case True{}:
      KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
        +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
      +target = Views.axis_extents(axes)
      +channel = Nat.add(lane_digit,Nat.mul(group,8n))
      +digits = Coordinates.inserted_digit(lane,others,channel)
      +digits_below = Values.inserted_below(target,lane,others,channel,others_below,
        {%extent_exact : {Nat.is_lt(channel,U32.to_nat(_)) == True{} : Bool} classified : {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(target,lane))) == True{} : Bool}},lane_inside)
      +lanes = TiledView.lanes_of(shape,axis,position)
      +start = Digits.words_nat(Grid.coordinates(shape,offset))
      lane_hit(~Logical,~original,logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,word,others,group,others_below,group_inside,exact,
        address,address_exact,lane_digit,small,classified,
        Digits.added(walked(axes,flags,shape,lanes,axis,False{},start,digits),unit_digits(List.length(&2,U32,shape),axis,Nat.mul(group,8n))),
        walked(axes,flags,shape,lanes,axis,True{},start,digits),
        chain_of(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,others,group,lane_digit,classified,others_below,
          walk_at(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,digits,digits_below)))
    case False{}:
      KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
        +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
      +target = Views.axis_extents(axes)
      +channel = Nat.add(lane_digit,Nat.mul(group,8n))
      +viewing = {ViewGather.Viewing{logical,axes,offset,flags} : ViewGather.Viewing<Logical>}
      +outer = Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),Representation.Lifted{viewing,Public.geometry_of(target,lane,Grid.Innermost{})},word)
      +read = ViewGather.viewed(~F32,~Logical,~original,viewing,U32.from_nat(ShapeLaws.digits_value(target,Coordinates.inserted_digit(lane,others,channel))))
      +target_lane = Coordinates.lane_values(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),viewing,limits,target,lane,Grid.Innermost{},True{},others,group,word,
        lane_inside,Bounds.strict_le(lane,List.length(&2,U32,target),lane_inside),target_removed_room,others_below,group_inside,exact,lane_digit,small)
      +zero = {%classified : {TileLaws.select_lane(lane_digit,outer) == Bool.pick(F32,_,read,0.0) : F32}
        %Equal.sym(U32,extent,Grid.lanes_at(target,lane),extent_exact) : {TileLaws.select_lane(lane_digit,outer) == Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(_)),read,0.0) : F32}
        target_lane : {TileLaws.select_lane(lane_digit,outer) == 0.0 : F32}}
      %Equal.sym(Bool,Nat.is_lt(channel,U32.to_nat(extent)),False{},classified) :
        {Bool.pick(F32,_,TileLaws.select_lane(lane_digit,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},address)),0.0) ==
          TileLaws.select_lane(lane_digit,outer) : F32}
      Equal.sym(F32,TileLaws.select_lane(lane_digit,outer),0.0,zero)

# A target cell the view of the cells reads agrees with the target cell of
# the viewed values.
def word_split(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,+word: U32,
  +inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}))) == True{} : Bool},
  +split: CellWords.CellWord<Views.axis_extents(axes),lane,word>) ->
  PaddedCells.Valid<Representation.lane_count(Public.geometry_of(Views.axis_extents(axes),lane,Grid.Innermost{})),
    Representation.cell_group(Public.geometry_of(Views.axis_extents(axes),lane,Grid.Innermost{}),U32.to_nat(word)),
    ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},cell_axes_of(kept,extent,shape,axis,position),
        TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset),List.append(&2,Bool,kept_flags,[False{}])},word),
    Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),
      Representation.Lifted{ViewGather.Viewing{logical,axes,offset,flags},Public.geometry_of(Views.axis_extents(axes),lane,Grid.Innermost{})},word)>:
  KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  CellWords.CellWord{+others,+group,+others_below,+group_inside,+split_exact,+group_same,+lanes_same,+split_below} = split
  +target = Views.axis_extents(axes)
  +geometry = Public.geometry_of(target,lane,Grid.Innermost{})
  +cells = Tiled.cell_shape(target,lane,Grid.Innermost{})
  +number = U32.to_nat(word)
  +placed = Coordinates.placed_digits(True{},Grid.Innermost{},others,group)
  +exact = {%lane_inside : {number == ShapeLaws.digits_value(cells,Coordinates.placed_digits(_,Grid.Innermost{},others,group)) : Nat} split_exact :
    {number == ShapeLaws.digits_value(cells,placed) : Nat}}
  +placed_below = {%lane_inside : Coordinates.DigitsBelow(cells,Coordinates.placed_digits(_,Grid.Innermost{},others,group)) split_below :
    Coordinates.DigitsBelow(cells,placed)}
  +count_same = Equal.trans(Nat,Representation.lane_count(geometry),U32.to_nat(Grid.lanes_at(target,lane)),U32.to_nat(extent),lanes_same,
    Equal.cong(U32,Nat,U32.to_nat,Grid.lanes_at(target,lane),extent,Equal.sym(U32,extent,Grid.lanes_at(target,lane),extent_exact)))
  +cell_axes = cell_axes_of(kept,extent,shape,axis,position)
  +cell_flags = List.append(&2,Bool,kept_flags,[False{}])
  +offset_cell = TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset)
  +length = Tensors.count_or_zero(Tiled.cell_shape(shape,axis,position))
  +capacity = U32.to_nat(Machine.capacity(limits))
  +address = (Views.mirrored_offset(cell_axes,cell_flags,offset_cell) + Box.stepped_sum(Views.axis_extents(cell_axes),ViewPart.view_steps(Views.mirrored_axes(cell_axes,cell_flags)),number) : U32)
  +extents_same = Equal.sym(List<&2,U32>,Views.axis_extents(cell_axes),cells,cell_extents(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view))
  +inside_cells = {%extents_same : {Nat.is_lt(number,ShapeLaws.elements(_)) == True{} : Bool} inside_word :
    {Nat.is_lt(number,ShapeLaws.elements(Views.axis_extents(cell_axes))) == True{} : Bool}}
  +cells_room = {%extents_same : {Nat.is_le(ShapeLaws.elements(_),capacity) == True{} : Bool} target_cells_room :
    {Nat.is_le(ShapeLaws.elements(Views.axis_extents(cell_axes)),capacity) == True{} : Bool}}
  +length_room_cells = {%Equal.sym(Nat,U32.to_nat(length),ShapeLaws.elements(Tiled.cell_shape(shape,axis,position)),cells_count) : {Nat.is_le(_,capacity) == True{} : Bool}
    source_cells_room(limits,shape,axis,position,facts) : {Nat.is_le(U32.to_nat(length),capacity) == True{} : Bool}}
  +fits = {%cell_nonempty(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view) : {Views.fits_nonempty(_,Views.View{cell_axes,offset_cell},length) == True{} : Bool}
    cell_fits(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view) : {Views.fits_nonempty(False{},Views.View{cell_axes,offset_cell},length) == True{} : Bool}}
  +read = ViewReads.address_at(cell_axes,cell_flags,offset_cell,number,length,Machine.capacity(limits),Limits.capacity_fits(limits),cells_room,length_room_cells,fits,inside_cells)
  +appended = List.append(&2,Nat,others,[group])
  +word_digits = {%extents_same : {number == ShapeLaws.digits_value(_,appended) : Nat} exact : {number == ShapeLaws.digits_value(Views.axis_extents(cell_axes),appended) : Nat}}
  +appended_below = {%extents_same : Coordinates.DigitsBelow(_,appended)
    placed_below : Coordinates.DigitsBelow(Views.axis_extents(cell_axes),appended)}
  +address_exact = {%view_sum_digits(cell_axes,cell_flags,appended,appended_below) : {U32.to_nat(address) == Nat.add(U32.to_nat(offset_cell),_) : Nat}
    %word_digits : {U32.to_nat(address) == Nat.add(U32.to_nat(offset_cell),ViewReads.view_sum(cell_axes,cell_flags,_)) : Nat}
    ViewGather.read_exact_of(offset_cell,cell_axes,cell_flags,number,address,length,read) :
    {U32.to_nat(address) == Nat.add(U32.to_nat(offset_cell),digit_sum(cell_axes,cell_flags,appended)) : Nat}}
  +inner = ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},cell_axes,offset_cell,cell_flags},word)
  +outer = Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),
    Representation.Lifted{ViewGather.Viewing{logical,axes,offset,flags},geometry},word)
  %Equal.sym(Nat,Representation.lane_count(geometry),U32.to_nat(extent),count_same) : PaddedCells.Valid<_,Representation.cell_group(geometry,number),inner,outer>
  %Equal.sym(Nat,Representation.cell_group(geometry,number),group,group_same) : PaddedCells.Valid<U32.to_nat(extent),_,inner,outer>
  PaddedCells.Valid{
    lane_agree(~Logical,~original,Nat.is_lt(Nat.add(0n,Nat.mul(group,8n)),U32.to_nat(extent)),logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,
      word,others,group,others_below,group_inside,exact,address,address_exact,0n,{==},{==}),
    lane_agree(~Logical,~original,Nat.is_lt(Nat.add(1n,Nat.mul(group,8n)),U32.to_nat(extent)),logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,
      word,others,group,others_below,group_inside,exact,address,address_exact,1n,{==},{==}),
    lane_agree(~Logical,~original,Nat.is_lt(Nat.add(2n,Nat.mul(group,8n)),U32.to_nat(extent)),logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,
      word,others,group,others_below,group_inside,exact,address,address_exact,2n,{==},{==}),
    lane_agree(~Logical,~original,Nat.is_lt(Nat.add(3n,Nat.mul(group,8n)),U32.to_nat(extent)),logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,
      word,others,group,others_below,group_inside,exact,address,address_exact,3n,{==},{==}),
    lane_agree(~Logical,~original,Nat.is_lt(Nat.add(4n,Nat.mul(group,8n)),U32.to_nat(extent)),logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,
      word,others,group,others_below,group_inside,exact,address,address_exact,4n,{==},{==}),
    lane_agree(~Logical,~original,Nat.is_lt(Nat.add(5n,Nat.mul(group,8n)),U32.to_nat(extent)),logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,
      word,others,group,others_below,group_inside,exact,address,address_exact,5n,{==},{==}),
    lane_agree(~Logical,~original,Nat.is_lt(Nat.add(6n,Nat.mul(group,8n)),U32.to_nat(extent)),logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,
      word,others,group,others_below,group_inside,exact,address,address_exact,6n,{==},{==}),
    lane_agree(~Logical,~original,Nat.is_lt(Nat.add(7n,Nat.mul(group,8n)),U32.to_nat(extent)),logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,
      word,others,group,others_below,group_inside,exact,address,address_exact,7n,{==},{==})}

def word_valid(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  +axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,+lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +view: KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>,+word: U32,
  +inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(Views.axis_extents(axes),lane,Grid.Innermost{}))) == True{} : Bool}) ->
  PaddedCells.Valid<Representation.lane_count(Public.geometry_of(Views.axis_extents(axes),lane,Grid.Innermost{})),
    Representation.cell_group(Public.geometry_of(Views.axis_extents(axes),lane,Grid.Innermost{}),U32.to_nat(word)),
    ViewGather.viewed(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      ViewGather.Viewing{Representation.Lifted{logical,Public.geometry_of(shape,axis,position)},cell_axes_of(kept,extent,shape,axis,position),
        TiledView.cell_index(TiledView.lanes_of(shape,axis,position),offset),List.append(&2,Bool,kept_flags,[False{}])},word),
    Representation.lifted(~ViewGather.Viewing<Logical>,~ViewGather.viewed(~F32,~Logical,~original),
      Representation.Lifted{ViewGather.Viewing{logical,axes,offset,flags},Public.geometry_of(Views.axis_extents(axes),lane,Grid.Innermost{})},word)>:
  KeptView{+facts,+valid,+lane_exact,+kept_exact,+flags_exact,+lane_inside,+extent_exact,+kept_extents,+flags_length,+start_length,+start_aligned,+start_value,
    +start_below,+bound,+target_nonempty,+target_room,+target_cells_room,+view_fits,+length_room,+cells_count,+grouped,+removed_room,+target_grouped,+target_removed_room} = view
  word_split(~Logical,~original,logical,limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,view,word,inside_word,
    CellWords.cell_word(limits,Views.axis_extents(axes),lane,word,Bounds.strict_le(lane,List.length(&2,U32,Views.axis_extents(axes)),lane_inside),target_removed_room,inside_word))

# ---- The facts of a view that keeps the lanes ----

# The digit of the natural words at an axis is the word there.
def word_digit(words: List<&2,U32>,axis: Nat) -> {Digits.digit_at(Digits.words_nat(words),axis) == U32.to_nat(TiledView.coordinate_at(words,axis)) : Nat}:
  match words axis:
    case Nil{} +axis: {==}
    case Con{+word,+rest} 0n: {==}
    case Con{+word,+rest} 1n+ +next: word_digit(rest,next)

def mapping_lane(+mapped: TiledView.Mapped,+index: Nat,+axes: List<&2,Views.Axis>,+found: Mapping<mapped,index,axes>) ->
  Structure(mapped_lane(mapped),index,axes,mapped_extent(mapped),mapped_kept(mapped)):
  Mapping{+lane,+flags} = found
  lane

def mapping_flags(+mapped: TiledView.Mapped,+index: Nat,+axes: List<&2,Views.Axis>,+found: Mapping<mapped,index,axes>) ->
  {List.length(&2,Bool,mapped_flags(mapped)) == List.length(&2,Views.Axis,mapped_kept(mapped)) : Nat}:
  Mapping{+lane,+flags} = found
  flags

# The target axes a view without flags keeps reverse none.
def reading_forward(reading: TiledView.Reading,+index: Nat,+extent: U32,+lane_axis: Nat,mapped: TiledView.Mapped,
  +tail: {ViewGather.forward(mapped_flags(mapped)) == True{} : Bool}) ->
  {ViewGather.forward(mapped_flags(TiledView.with_reading(reading,False{},index,extent,lane_axis,mapped))) == True{} : Bool}:
  match reading mapped:
    case TiledView.Repeated{} TiledView.Mapped{+valid,+lane,+lane_extent,+axes,+flags,+reach}: tail
    case TiledView.Lane{} TiledView.Mapped{+valid,+lane,+lane_extent,+axes,+flags,+reach}: tail
    case TiledView.Stepping{+axis,+step,+cells} TiledView.Mapped{+valid,+lane,+lane_extent,+axes,+flags,+reach}: tail
    case TiledView.Unkept{} TiledView.Mapped{+valid,+lane,+lane_extent,+axes,+flags,+reach}: tail

def mapped_forward(axes: List<&2,Views.Axis>,+index: Nat,+shape: List<&2,U32>,+lanes: TiledView.Lanes,+start: List<&2,U32>) ->
  {ViewGather.forward(mapped_flags(TiledView.mapped_axes(axes,[],index,shape,lanes,start))) == True{} : Bool}:
  match axes:
    case Nil{}: {==}
    case Con{Views.Axis{+extent,+stride},+rest}:
      reading_forward(TiledView.classified(shape,lanes,extent,stride),index,extent,TiledView.lane_axis(lanes),TiledView.mapped_axes(rest,[],1n+index,shape,lanes,start),
        mapped_forward(rest,1n+index,shape,lanes,start))

# The facts of a kept view from the lane the mapping finds.
def kept_facts(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +valid: {mapped_valid(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == True{} : Bool},
  +lane_exact: {mapped_lane(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == Some{lane} : Maybe<&2,Nat>},
  +extent_mapped: {mapped_extent(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == extent : U32},
  +kept_exact: {mapped_kept(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == kept : List<&2,Views.Axis>},
  +flags_exact: {mapped_flags(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == kept_flags : List<&2,Bool>},
  +inside_reach: {TiledView.all_inside(mapped_reach(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))),shape) == True{} : Bool},
  +aligned: {U32.is_zero((TiledView.coordinate_at(Grid.coordinates(shape,offset),axis) % 8 : U32)) == True{} : Bool},
  +facts: SourceFacts<limits,shape,axis,position>,+target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +counted: Global.Counted<shape>,+target_counted: Global.Counted<Views.axis_extents(axes)>,
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(shape)) == True{} : Bool},
  +lane_at: LaneAt<0n,lane,axes,mapped_extent(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))),
    mapped_kept(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset)))>,
  +flags_count: {List.length(&2,Bool,mapped_flags(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset)))) ==
    List.length(&2,Views.Axis,mapped_kept(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset)))) : Nat}) ->
  KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>:
  SourceFacts{+within,+rank_room,+room,+cells_room,+nonzero} = facts
  LaneAt{+low,+inside,+extent_at,+kept_at} = lane_at
  +mapped = TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))
  +target = Views.axis_extents(axes)
  +lanes = TiledView.lanes_of(shape,axis,position)
  +words = Grid.coordinates(shape,offset)
  +start = Digits.words_nat(words)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +zero = Order.subtract_zero(lane)
  +positive = CountCompletion.positive_elements(shape,nonzero)
  +target_room = StorageCount.counted_room(limits,target,target_counted,default_capacity)
  +target_positive = CountCompletion.positive_elements(target,target_nonzero)
  +lane_inside = {%Equal.sym(Nat,List.length(&2,U32,target),List.length(&2,Views.Axis,axes),MetadataLaws.extents_length(axes)) : {Nat.is_lt(lane,_) == True{} : Bool}
    %zero : {Nat.is_lt(_,List.length(&2,Views.Axis,axes)) == True{} : Bool}
    inside : {Nat.is_lt(lane,List.length(&2,U32,target)) == True{} : Bool}}
  +extent_exact = Equal.trans(U32,extent,mapped_extent(mapped),Grid.lanes_at(target,lane),Equal.sym(U32,mapped_extent(mapped),extent,extent_mapped),
    %zero : {mapped_extent(mapped) == Grid.lanes_at(target,_) : U32} extent_at)
  +kept_extents = {%kept_exact : {Views.axis_extents(_) == Grid.removed(target,lane) : List<&2,U32>}
    %zero : {Views.axis_extents(mapped_kept(mapped)) == Grid.removed(target,_) : List<&2,U32>}
    kept_at : {Views.axis_extents(kept) == Grid.removed(target,lane) : List<&2,U32>}}
  +flags_length = {%flags_exact : {List.length(&2,Bool,_) == List.length(&2,Views.Axis,kept) : Nat}
    %kept_exact : {List.length(&2,Bool,mapped_flags(mapped)) == List.length(&2,Views.Axis,_) : Nat}
    flags_count : {List.length(&2,Bool,kept_flags) == List.length(&2,Views.Axis,kept) : Nat}}
  +start_length = Equal.trans(Nat,List.length(&2,Nat,start),List.length(&2,U32,words),List.length(&2,U32,shape),words_nat_length(words),Digits.coordinates_count(shape,offset))
  +length = Tensors.count_or_zero(shape)
  +length_exact = ViewMoved.counted_cells(shape,Global.count(shape,counted),Global.accepted(shape,counted))
  +length_room = {%Equal.sym(Nat,U32.to_nat(length),ShapeLaws.elements(shape),length_exact) : {Nat.is_le(_,capacity) == True{} : Bool} room :
    {Nat.is_le(U32.to_nat(length),capacity) == True{} : Bool}}
  +view_fits = {%target_nonzero : {Views.fits_nonempty(_,Views.View{axes,offset},length) == True{} : Bool} fits :
    {Views.fits_nonempty(False{},Views.View{axes,offset},length) == True{} : Bool}}
  +base = U32.to_nat(offset)
  +span = ViewBounds.last_span(axes)
  +last_bound = {%ViewBounds.last_location(axes,base) : {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool} ViewBounds.last_address(axes,offset,length,view_fits) :
    {Nat.is_lt(Nat.add(base,span),U32.to_nat(length)) == True{} : Bool}}
  +offset_inside = {%length_exact : {Nat.is_lt(base,_) == True{} : Bool}
    Order.le_lt_trans(base,Nat.add(base,span),U32.to_nat(length),Order.le_add(base,span),last_bound) : {Nat.is_lt(base,ShapeLaws.elements(shape)) == True{} : Bool}}
  +coordinates_exact = Digits.coordinates_exact(shape,offset,Machine.half_word(limits),Limits.half_word_fits(limits),Limits.double_bound(limits,ShapeLaws.elements(shape),room),offset_inside)
  +coordinates_back = Equal.sym(List<&2,Nat>,start,Targets.cell_coordinates(shape,base),coordinates_exact)
  +start_value = {%coordinates_back : {ShapeLaws.digits_value(shape,_) == base : Nat} Digits.value_coordinates(shape,base,offset_inside) :
    {ShapeLaws.digits_value(shape,start) == base : Nat}}
  +start_below = {%coordinates_back : Coordinates.DigitsBelow(shape,_) Digits.coordinates_below(shape,base,offset_inside) : Coordinates.DigitsBelow(shape,start)}
  +start_aligned = {%Equal.sym(Nat,Digits.digit_at(start,axis),U32.to_nat(TiledView.coordinate_at(words,axis)),word_digit(words,axis)) : {Nat.mod(_,8n) == 0n : Nat}
    step_nat(limits,TiledView.coordinate_at(words,axis),aligned) : {Nat.mod(Digits.digit_at(start,axis),8n) == 0n : Nat}}
  +lasts = mirrored_lasts(axes,flags)
  +reach_digits = walked(axes,flags,shape,lanes,axis,True{},start,lasts)
  +reach_value = logical_walk(axes,flags,0n,shape,axis,position,limits,facts,words,start,lasts,start_length,lasts_below(axes,flags,target_nonzero),valid)
  +reach_below = {%Equal.sym(Nat,ShapeLaws.digits_value(shape,reach_digits),Nat.add(ShapeLaws.digits_value(shape,start),digit_sum(axes,flags,lasts)),reach_value) :
      {Nat.is_lt(_,U32.to_nat(length)) == True{} : Bool}
    %Equal.sym(Nat,ShapeLaws.digits_value(shape,start),base,start_value) : {Nat.is_lt(Nat.add(_,digit_sum(axes,flags,lasts)),U32.to_nat(length)) == True{} : Bool}
    %Equal.sym(Nat,digit_sum(axes,flags,lasts),span,lasts_sum(axes,flags,target_nonzero)) : {Nat.is_lt(Nat.add(base,_),U32.to_nat(length)) == True{} : Bool}
    last_bound : {Nat.is_lt(ShapeLaws.digits_value(shape,reach_digits),U32.to_nat(length)) == True{} : Bool}}
  +reach_room = Order.transitive(ShapeLaws.digits_value(shape,reach_digits),U32.to_nat(length),capacity,
    Bounds.strict_le(ShapeLaws.digits_value(shape,reach_digits),U32.to_nat(length),reach_below),length_room)
  +reach_half = Order.transitive(ShapeLaws.digits_value(shape,reach_digits),Nat.double(ShapeLaws.digits_value(shape,reach_digits)),U32.to_nat(Machine.half_word(limits)),
    Order.le_double(ShapeLaws.digits_value(shape,reach_digits)),Limits.double_bound(limits,ShapeLaws.digits_value(shape,reach_digits),reach_room))
  +reach_length = walked_length(axes,flags,shape,lanes,axis,True{},start,lasts,start_length)
  +reach_exact = reach_walk(axes,flags,0n,shape,axis,position,limits,facts,words,start_length,valid,target_nonzero,
    at_most_le(reach_digits,ShapeLaws.digits_value(shape,reach_digits),U32.to_nat(Machine.half_word(limits)),at_most_value(shape,reach_digits,nonzero,reach_length),reach_half))
  +reach = mapped_reach(mapped)
  +reach_same = Equal.trans(Nat,List.length(&2,U32,reach),List.length(&2,Nat,Digits.words_nat(reach)),List.length(&2,U32,shape),
    Equal.sym(Nat,List.length(&2,Nat,Digits.words_nat(reach)),List.length(&2,U32,reach),words_nat_length(reach)),
    Equal.trans(Nat,List.length(&2,Nat,Digits.words_nat(reach)),List.length(&2,Nat,reach_digits),List.length(&2,U32,shape),
      Equal.cong(List<&2,Nat>,Nat,list => List.length(&2,Nat,list),Digits.words_nat(reach),reach_digits,reach_exact),reach_length))
  +bound = {%reach_exact : Coordinates.DigitsBelow(shape,_) inside_below(reach,shape,inside_reach,reach_same) : Coordinates.DigitsBelow(shape,reach_digits)}
  +source_cells = Tiled.cell_shape(shape,axis,position)
  +cells_counted = StorageCount.counted_within(limits,source_cells,cells_room,default_capacity)
  +lane_room = {%Equal.sym(U32,extent,Grid.lanes_at(target,lane),extent_exact) : {Nat.is_le(U32.to_nat(_),capacity) == True{} : Bool}
    Rooms.lane_room(limits,target,lane,lane_inside,target_positive,target_room) : {Nat.is_le(U32.to_nat(extent),capacity) == True{} : Bool}}
  KeptView{facts,valid,lane_exact,kept_exact,flags_exact,lane_inside,extent_exact,kept_extents,flags_length,start_length,start_aligned,start_value,start_below,bound,
    target_nonzero,target_room,Rooms.cells_room(limits,target,lane,Grid.Innermost{},target_positive,target_room),view_fits,length_room,
    ViewMoved.counted_cells(source_cells,Global.count(source_cells,cells_counted),Global.accepted(source_cells,cells_counted)),
    TiledValues.grouped(limits,Grid.lanes_at(shape,axis),Rooms.lane_room(limits,shape,axis,within,positive,room)),
    Rooms.removed_room(limits,shape,axis,within,positive,room),TiledValues.grouped(limits,extent,lane_room),
    Rooms.removed_room(limits,target,lane,lane_inside,target_positive,target_room)}

# The facts of a view the runtime keeps the lanes of, of counted values
# without a zero extent onto counted values without one.
def kept_view(+limits: Machine.Limits,+shape: List<&2,U32>,+axis: Nat,+position: Grid.Position,+axes: List<&2,Views.Axis>,+flags: List<&2,Bool>,+offset: U32,
  +lane: Nat,+extent: U32,+kept: List<&2,Views.Axis>,+kept_flags: List<&2,Bool>,
  +valid: {mapped_valid(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == True{} : Bool},
  +lane_exact: {mapped_lane(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == Some{lane} : Maybe<&2,Nat>},
  +extent_mapped: {mapped_extent(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == extent : U32},
  +kept_exact: {mapped_kept(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == kept : List<&2,Views.Axis>},
  +flags_exact: {mapped_flags(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))) == kept_flags : List<&2,Bool>},
  +inside_reach: {TiledView.all_inside(mapped_reach(TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))),shape) == True{} : Bool},
  +aligned: {U32.is_zero((TiledView.coordinate_at(Grid.coordinates(shape,offset),axis) % 8 : U32)) == True{} : Bool},
  +within: {Nat.is_lt(axis,List.length(&2,U32,shape)) == True{} : Bool},+nonzero: {Shapes.has_zero(shape) == False{} : Bool},
  +target_nonzero: {Shapes.has_zero(Views.axis_extents(axes)) == False{} : Bool},
  +default_capacity: {Machine.capacity(limits) == Machine.capacity(Default.limits()) : U32},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +counted: Global.Counted<shape>,+target_counted: Global.Counted<Views.axis_extents(axes)>,
  +fits: {Views.fits(Views.View{axes,offset},Tensors.count_or_zero(shape)) == True{} : Bool}) ->
  KeptView<limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags>:
  +mapped = TiledView.mapped_axes(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset))
  +room = StorageCount.counted_room(limits,shape,counted,default_capacity)
  +positive = CountCompletion.positive_elements(shape,nonzero)
  +facts = {SourceFacts{within,rank_room,room,Rooms.cells_room(limits,shape,axis,position,positive,room),nonzero} : SourceFacts<limits,shape,axis,position>}
  +found = mapping(axes,flags,0n,shape,TiledView.lanes_of(shape,axis,position),Grid.coordinates(shape,offset),valid)
  kept_facts(limits,shape,axis,position,axes,flags,offset,lane,extent,kept,kept_flags,valid,lane_exact,extent_mapped,kept_exact,flags_exact,inside_reach,aligned,facts,
    target_nonzero,default_capacity,counted,target_counted,fits,
    {%lane_exact : Structure(_,0n,axes,mapped_extent(mapped),mapped_kept(mapped)) mapping_lane(mapped,0n,axes,found) :
      Structure(Some{lane},0n,axes,mapped_extent(mapped),mapped_kept(mapped))},
    mapping_flags(mapped,0n,axes,found))

# ---- Cells of readers that agree on the values ----

# The values but the lane axis fit a capacity that the values fit.
def removed_fits(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+within_value: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  match within:
    case True{}: Rooms.removed_room(limits,values,axis,within_value,positive,room)
    case False{}:
      %Equal.sym(List<&2,U32>,Grid.removed(values,axis),values,Public.removed_past(values,axis,within_value)) :
        {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
      room

# The group of the digits of a cell lies below the groups.
def group_inside(within: Bool,+values: List<&2,U32>,+axis: Nat,position: Grid.Position,+digits: List<&2,Nat>,
  +within_value: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},
  +below: Coordinates.GroupBelow(within,Values.cell_group(within,position,digits),Grid.groups(Grid.lanes_at(values,axis)))) ->
  {Nat.is_lt(Values.cell_group(within,position,digits),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool}:
  match within:
    case True{}: below
    case False{}:
      %Equal.sym(U32,Grid.lanes_at(values,axis),1,Public.lanes_past(values,axis,within_value)) :
        {Nat.is_lt(Values.cell_group(False{},position,digits),U32.to_nat(Grid.groups(_))) == True{} : Bool}
      {==}

# A lane inside the lanes reads both readers at an index of the values.
def picked_agree(~Before: Data,~After: Data,~before_value: Before -> U32 -> F32,~after_value: After -> U32 -> F32,~Related: Before -> After -> Data,~bound: Nat,~agreement: Content.Below(~F32,~Before,~After,~before_value,~after_value,~Related,~bound),valid: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+within: Bool,+others: List<&2,Nat>,+channel: Nat,
  +within_value: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},+below: Coordinates.DigitsBelow(Grid.removed(values,axis),others),
  +classified: {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))) == valid : Bool},+room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+covered: {Nat.is_le(ShapeLaws.elements(values),bound) == True{} : Bool},+logical: Before,+other: After,+related: Related(logical,other)) ->
  {Bool.pick(F32,valid,before_value(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.logical_digits(within,axis,others,channel)))),0.0) == Bool.pick(F32,valid,after_value(other,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.logical_digits(within,axis,others,channel)))),0.0) : F32}:
  match valid:
    case True{}:
      +index = ShapeLaws.digits_value(values,Coordinates.logical_digits(within,axis,others,channel))
      +below_values = Coordinates.digits_below(values,Coordinates.logical_digits(within,axis,others,channel),
        Values.logical_below(within,values,axis,others,channel,within_value,below,classified))
      +exact = Bounds.roundtrip(index,Machine.capacity(limits),Limits.capacity_fits(limits),
        Order.transitive(index,ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(index,ShapeLaws.elements(values),below_values),room))
      agreement(logical,other,related,U32.from_nat(index),
        {%Equal.sym(Nat,U32.to_nat(U32.from_nat(index)),index,exact) : {Nat.is_lt(_,bound) == True{} : Bool}
          Bounds.nat_trans(index,ShapeLaws.elements(values),bound,below_values,covered) : {Nat.is_lt(U32.to_nat(U32.from_nat(index)),bound) == True{} : Bool}})
    case False{}: {==}

# A lane of the cells of two readers that agree below the values agrees.
def lane_below(~Before: Data,~After: Data,~before_value: Before -> U32 -> F32,~after_value: After -> U32 -> F32,~Related: Before -> After -> Data,~bound: Nat,~agreement: Content.Below(~F32,~Before,~After,~before_value,~after_value,~Related,~bound),+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+covered: {Nat.is_le(ShapeLaws.elements(values),bound) == True{} : Bool},
  +logical: Before,+other: After,+related: Related(logical,other),+word: U32,
  +inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool},+lane: Nat,+small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Before,~before_value,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)) == TileLaws.select_lane(lane,Representation.lifted(~After,~after_value,Representation.Lifted{other,Public.geometry_of(values,axis,position)},word)) : F32}:
  +within = Nat.is_lt(axis,List.length(&2,U32,values))
  +removed = Grid.removed(values,axis)
  +groups = Grid.groups(Grid.lanes_at(values,axis))
  +cells = Tiled.cell_shape(values,axis,position)
  +number = U32.to_nat(word)
  +digits = Targets.cell_coordinates(cells,number)
  +below_cells = Digits.coordinates_below(cells,number,inside_word)
  +others = Values.cell_others(within,position,digits)
  +group = Values.cell_group(within,position,digits)
  +others_below = Values.cell_below(within,position,removed,groups,digits,below_cells)
  +group_in = group_inside(within,values,axis,position,digits,{==},Values.group_below(within,position,removed,groups,digits,below_cells))
  +whole = Values.cell_whole(within,position,removed,groups,digits,below_cells)
  +exact = {%Equal.sym(List<&2,Nat>,Coordinates.placed_digits(within,position,others,group),digits,whole) : {number == ShapeLaws.digits_value(cells,_) : Nat}
    Equal.sym(Nat,ShapeLaws.digits_value(cells,digits),number,Digits.value_coordinates(cells,number,inside_word)) :
    {number == ShapeLaws.digits_value(cells,Coordinates.placed_digits(within,position,others,group)) : Nat}}
  +removed_room = removed_fits(within,limits,values,axis,{==},positive,room)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  Equal.trans(F32,TileLaws.select_lane(lane,Representation.lifted(~Before,~before_value,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word)),Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))),before_value(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.logical_digits(within,axis,others,channel)))),0.0),
    TileLaws.select_lane(lane,Representation.lifted(~After,~after_value,Representation.Lifted{other,Public.geometry_of(values,axis,position)},word)),
    Coordinates.lane_values(~Before,~before_value,logical,limits,values,axis,position,within,others,group,word,{==},inside,removed_room,others_below,group_in,exact,lane,small),
    Equal.trans(F32,Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))),before_value(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.logical_digits(within,axis,others,channel)))),0.0),
      Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))),after_value(other,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.logical_digits(within,axis,others,channel)))),0.0),TileLaws.select_lane(lane,Representation.lifted(~After,~after_value,Representation.Lifted{other,Public.geometry_of(values,axis,position)},word)),
      picked_agree(~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))),limits,values,axis,within,others,channel,{==},others_below,{==},room,covered,logical,other,related),
      Equal.sym(F32,TileLaws.select_lane(lane,Representation.lifted(~After,~after_value,Representation.Lifted{other,Public.geometry_of(values,axis,position)},word)),Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))),after_value(other,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.logical_digits(within,axis,others,channel)))),0.0),
        Coordinates.lane_values(~After,~after_value,other,limits,values,axis,position,within,others,group,word,{==},inside,removed_room,others_below,group_in,exact,lane,small))))

# Readers of the values that agree below them, each lifted in the geometry of
# the values.
def LiftedBelow(~Before: Data,~After: Data,~Related: Before -> After -> Data,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,
  before: Representation.Lifted<Before>,after: Representation.Lifted<After>) -> Data:
  match before after:
    case Representation.Lifted{logical,geometry} Representation.Lifted{other,other_geometry}:
      KernelGemm.Both<Related(logical,other),KernelGemm.Both<{geometry == Public.geometry_of(values,axis,position) : Cells.Cells},
        {other_geometry == Public.geometry_of(values,axis,position) : Cells.Cells}>>

# The cells of readers that agree below the values agree below the cells.
def lifted_below(~Before: Data,~After: Data,~before_value: Before -> U32 -> F32,~after_value: After -> U32 -> F32,~Related: Before -> After -> Data,~bound: Nat,~agreement: Content.Below(~F32,~Before,~After,~before_value,~after_value,~Related,~bound),+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+inside: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+covered: {Nat.is_le(ShapeLaws.elements(values),bound) == True{} : Bool},
  +before: Representation.Lifted<Before>,+after: Representation.Lifted<After>,+related: LiftedBelow(~Before,~After,~Related,values,axis,position,before,after),+word: U32,
  +inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,position))) == True{} : Bool}) ->
  {Representation.lifted(~Before,~before_value,before,word) == Representation.lifted(~After,~after_value,after,word) : Tile.Tile8}:
  match before after:
    case Representation.Lifted{+logical,+geometry} Representation.Lifted{+other,+other_geometry}:
      KernelGemm.Both{+agreed,KernelGemm.Both{+here,+there}} = related
      +geometry_values = Public.geometry_of(values,axis,position)
      %Equal.sym(Cells.Cells,geometry,geometry_values,here) :
        {Representation.lifted(~Before,~before_value,Representation.Lifted{logical,_},word) ==
          Representation.lifted(~After,~after_value,Representation.Lifted{other,other_geometry},word) : Tile.Tile8}
      %Equal.sym(Cells.Cells,other_geometry,geometry_values,there) :
        {Representation.lifted(~Before,~before_value,Representation.Lifted{logical,geometry_values},word) ==
          Representation.lifted(~After,~after_value,Representation.Lifted{other,_},word) : Tile.Tile8}
      TileLaws.tiles_equal(Representation.lifted(~Before,~before_value,Representation.Lifted{logical,Public.geometry_of(values,axis,position)},word),Representation.lifted(~After,~after_value,Representation.Lifted{other,Public.geometry_of(values,axis,position)},word),
        lane_below(~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,limits,values,axis,position,inside,positive,room,covered,logical,other,agreed,word,inside_word,0n,{==}),
        lane_below(~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,limits,values,axis,position,inside,positive,room,covered,logical,other,agreed,word,inside_word,1n,{==}),
        lane_below(~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,limits,values,axis,position,inside,positive,room,covered,logical,other,agreed,word,inside_word,2n,{==}),
        lane_below(~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,limits,values,axis,position,inside,positive,room,covered,logical,other,agreed,word,inside_word,3n,{==}),
        lane_below(~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,limits,values,axis,position,inside,positive,room,covered,logical,other,agreed,word,inside_word,4n,{==}),
        lane_below(~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,limits,values,axis,position,inside,positive,room,covered,logical,other,agreed,word,inside_word,5n,{==}),
        lane_below(~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,limits,values,axis,position,inside,positive,room,covered,logical,other,agreed,word,inside_word,6n,{==}),
        lane_below(~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,limits,values,axis,position,inside,positive,room,covered,logical,other,agreed,word,inside_word,7n,{==}))

# Tiled values of a reader are the values of another reader that agrees with
# it below the values.
def contract_below(~Before: Data,~After: Data,~before_value: Before -> U32 -> F32,~after_value: After -> U32 -> F32,~Related: Before -> After -> Data,~bound: Nat,~agreement: Content.Below(~F32,~Before,~After,~before_value,~after_value,~Related,~bound),+limits: Machine.Limits,+logical: Before,+other: After,+related: Related(logical,other),+values: List<&2,U32>,+positive: {Nat.is_lt(0n,ShapeLaws.elements(values)) == True{} : Bool},+room: {Nat.is_le(ShapeLaws.elements(values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +covered: {Nat.is_le(ShapeLaws.elements(values),bound) == True{} : Bool},tiled: Tiled.Tiled,contract: Public.Contract(~Before,~before_value,logical,values,tiled)) -> Public.Contract(~After,~after_value,other,values,tiled):
  match tiled:
    case Tiled.Tiled{cells,+axis,+lanes,+position}:
      match cells:
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Usable{},+workers}:
          KernelGemm.Both{+global,+laid} = contract
          Public.Laid{+inside,+shaped,+restored,+counted,+pinned} = laid
          +geometry = Public.geometry_of(values,axis,position)
          +cells_bound = ShapeLaws.elements(Tiled.cell_shape(values,axis,position))
          +counted_cells = Global.contract_counted(~Tile.Tile8,~Representation.Lifted<Before>,~Representation.lifted(~Before,~before_value),
            Representation.Lifted{logical,geometry},pieces,shape,fill,workers,global)
          +count = Global.count(shape,counted_cells)
          +count_exact = Count.count_exact(ShapeLaws.elements(shape),count,Global.evidence(shape,counted_cells))
          +room_cells = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shaped) : {Nat.is_le(ShapeLaws.elements(shape),ShapeLaws.elements(_)) == True{} : Bool} Order.reflexive(ShapeLaws.elements(shape)) :
            {Nat.is_le(ShapeLaws.elements(shape),cells_bound) == True{} : Bool}}
          +fits_cells = {%Equal.sym(List<&2,U32>,Tiled.cell_shape(values,axis,position),shape,shaped) : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(count)) == True{} : Bool}
            %count_exact : {Nat.is_le(ShapeLaws.elements(shape),_) == True{} : Bool} Order.reflexive(ShapeLaws.elements(shape)) :
            {Nat.is_le(cells_bound,U32.to_nat(count)) == True{} : Bool}}
          KernelGemm.Both{
            Global.relabeled_below(~Tile.Tile8,~Representation.Lifted<Before>,~Representation.Lifted<After>,~Representation.lifted(~Before,~before_value),
              ~Representation.lifted(~After,~after_value),~LiftedBelow(~Before,~After,~Related,values,axis,position),~ShapeLaws.elements(Tiled.cell_shape(values,axis,position)),
              ~lifted_below(~Before,~After,~before_value,~after_value,~Related,~bound,~agreement,limits,values,axis,position,inside,positive,room,covered),Representation.Lifted{logical,geometry},Representation.Lifted{other,geometry},
              KernelGemm.Both{related,KernelGemm.Both{{==},{==}}},shape,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},
              Global.usable(~Tile.Tile8,Tensors.Tensor{pieces,shape,fill,Tensors.Usable{},workers},{==}),global,room_cells,count,fits_cells),
            Public.Laid{inside,shaped,restored,counted,pinned}}
        case Tensors.Tensor{pieces,+shape,+fill,Tensors.Failed{+problem},+workers}:
          Empty.absurd(Public.Contract(~After,~after_value,other,values,Tiled.Tiled{Tensors.Tensor{pieces,shape,fill,Tensors.Failed{problem},workers},axis,lanes,position}),contract)
