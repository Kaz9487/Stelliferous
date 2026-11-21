# A part that writes a box inside a row-major piece. On every axis the box
# starts at an index of the piece and runs over its count of indices, with the
# piece strides as target steps, so box index j writes the piece cell whose
# coordinates are the starts plus the coordinates of j in the box. Distinct
# box indices write distinct cells, every one inside the piece, and a piece
# cell is written exactly when its coordinates lie in the box.
import Base
import ../tensor_move.bend as Move
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ./tensor_move_readback.bend as Held
import ./tensor_move_part.bend as Part
import ./tensor_view_part.bend as ViewPart
import ./tensor_grid_box.bend as Box
import ./tensor_grid_pieces.bend as GridLaws
import ./tensor_box_digits.bend as Digits
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_target_cells.bend as Targets
import ./tensor_shape_refinement.bend as ShapeLaws
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ../kernel_gemm.bend as KernelGemm

# ---- A box inside a piece ----

# One axis of a box inside a piece: the cut of the box on the axis, the index
# where the box starts and the extent of the piece.
type BoxAxis is Data:
  BoxAxis{cut: Tasks.Cut,start: Nat,extent: U32}

def box_axes(list: List<&2,BoxAxis>) -> List<&2,Move.Axis>:
  match list:
    case Nil{}: []
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,source_offset,+target_stride,target_offset},start,extent},rest}:
      Move.Axis{count,source_stride,False{},target_stride} <> box_axes(rest)

def box_extents(list: List<&2,BoxAxis>) -> List<&2,U32>:
  match list:
    case Nil{}: []
    case Con{BoxAxis{cut,start,+extent},rest}: extent <> box_extents(rest)

def box_starts(list: List<&2,BoxAxis>) -> List<&2,Nat>:
  match list:
    case Nil{}: []
    case Con{BoxAxis{cut,+start,extent},rest}: start <> box_starts(rest)

# The first source and target cells of the box, the offsets of its axes added
# one after another.
def box_source(list: List<&2,BoxAxis>,+first: U32) -> U32:
  match list:
    case Nil{}: first
    case Con{BoxAxis{Tasks.Cut{count,source_stride,+source_offset,target_stride,target_offset},start,extent},rest}: box_source(rest,(first + source_offset : U32))

def box_target(list: List<&2,BoxAxis>,+first: U32) -> U32:
  match list:
    case Nil{}: first
    case Con{BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,+target_offset},start,extent},rest}: box_target(rest,(first + target_offset : U32))

# On every axis the box steps the piece stride, starts its target offset at
# its start, is not empty and ends inside the piece.
def BoxIn(list: List<&2,BoxAxis>) -> Data:
  match list:
    case Nil{}: Unit
    case Con{BoxAxis{Tasks.Cut{+count,source_stride,source_offset,+target_stride,+target_offset},+start,+extent},rest}:
      Held.Both<{target_stride == Grid.product(box_extents(rest)) : U32},
        Held.Both<{target_offset == (U32.from_nat(start) * Grid.product(box_extents(rest)) : U32) : U32},
          Held.Both<{Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool},
            Held.Both<{Nat.is_le(Nat.add(start,U32.to_nat(count)),U32.to_nat(extent)) == True{} : Bool},BoxIn(rest)>>>>

# ---- The cell a box index writes ----

def box_targets(+list: List<&2,BoxAxis>,+inside: BoxIn(list)) -> {ViewPart.move_targets(box_axes(list)) == Tasks.strides_of(box_extents(list)) : List<&2,U32>}:
  match list:
    case Nil{}: {==}
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest}:
      Held.Both{+stride,Held.Both{+offset,Held.Both{+positive,Held.Both{+fits,+later}}}} = inside
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,box_extents(rest)}),Con{Grid.product(box_extents(rest)),Grid.strides(box_extents(rest))},
          GridLaws.strides_con(extent,box_extents(rest))) :
        {ViewPart.move_targets(box_axes(Con{BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},rest})) == _ : List<&2,U32>}
      %Equal.sym(U32,target_stride,Grid.product(box_extents(rest)),stride) :
        {Con{_,ViewPart.move_targets(box_axes(rest))} == Con{Grid.product(box_extents(rest)),Tasks.strides_of(box_extents(rest))} : List<&2,U32>}
      %Equal.sym(List<&2,U32>,ViewPart.move_targets(box_axes(rest)),Tasks.strides_of(box_extents(rest)),box_targets(rest,later)) :
        {Con{Grid.product(box_extents(rest)),_} == Con{Grid.product(box_extents(rest)),Tasks.strides_of(box_extents(rest))} : List<&2,U32>}
      {==}

# The target cell where the box starts is its starts times the piece strides.
def box_target_digits(+list: List<&2,BoxAxis>,+first: U32,+inside: BoxIn(list)) ->
  {box_target(list,first) == (first + Digits.stepped_digits(box_starts(list),Tasks.strides_of(box_extents(list))) : U32) : U32}:
  match list:
    case Nil{}: Equal.sym(U32,(first + 0 : U32),first,Arithmetic.add_zero(first))
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest}:
      Held.Both{+stride,Held.Both{+offset,Held.Both{+positive,Held.Both{+fits,+later}}}} = inside
      +step = Grid.product(box_extents(rest))
      +tail = Digits.stepped_digits(box_starts(rest),Tasks.strides_of(box_extents(rest)))
      %Equal.sym(List<&2,U32>,Grid.strides(Con{extent,box_extents(rest)}),Con{step,Grid.strides(box_extents(rest))},GridLaws.strides_con(extent,box_extents(rest))) :
        {box_target(rest,(first + target_offset : U32)) == (first + Digits.stepped_digits(Con{start,box_starts(rest)},_) : U32) : U32}
      Equal.trans(U32,box_target(rest,(first + target_offset : U32)),((first + target_offset : U32) + tail : U32),(first + ((U32.from_nat(start) * step : U32) + tail : U32) : U32),
        box_target_digits(rest,(first + target_offset : U32),later),
        {%Equal.sym(U32,target_offset,(U32.from_nat(start) * step : U32),offset) :
            {((first + _ : U32) + tail : U32) == (first + ((U32.from_nat(start) * step : U32) + tail : U32) : U32) : U32}
          Arithmetic.add_associative(first,(U32.from_nat(start) * step : U32),tail) :
          {((first + target_offset : U32) + tail : U32) == (first + ((U32.from_nat(start) * step : U32) + tail : U32) : U32) : U32}})

# Sums of two pairs in the other pairing.
def interchange(+first: U32,+second: U32,+third: U32,+fourth: U32) ->
  {((first + second : U32) + (third + fourth : U32) : U32) == ((first + third : U32) + (second + fourth : U32) : U32) : U32}:
  Equal.trans(U32,((first + second : U32) + (third + fourth : U32) : U32),(first + (second + (third + fourth : U32) : U32) : U32),((first + third : U32) + (second + fourth : U32) : U32),
    Arithmetic.add_associative(first,second,(third + fourth : U32)),
    Equal.trans(U32,(first + (second + (third + fourth : U32) : U32) : U32),(first + (third + (second + fourth : U32) : U32) : U32),((first + third : U32) + (second + fourth : U32) : U32),
      Equal.cong(U32,U32,word => (first + word : U32),(second + (third + fourth : U32) : U32),(third + (second + fourth : U32) : U32),Box.exchange(second,third,fourth)),
      Equal.sym(U32,((first + third : U32) + (second + fourth : U32) : U32),(first + (third + (second + fourth : U32) : U32) : U32),Arithmetic.add_associative(first,third,(second + fourth : U32)))))

# Digits times steps add axis by axis.
def stepped_added(left: List<&2,Nat>,right: List<&2,Nat>,steps: List<&2,U32>,+same: {List.length(&2,Nat,left) == List.length(&2,Nat,right) : Nat}) ->
  {(Digits.stepped_digits(left,steps) + Digits.stepped_digits(right,steps) : U32) == Digits.stepped_digits(Digits.added(left,right),steps) : U32}:
  match left right steps:
    case Con{+digit,+more} Con{+other,+rest} Con{+step,+later}:
      +head = (U32.from_nat(digit) * step : U32)
      +other_head = (U32.from_nat(other) * step : U32)
      +tail = Digits.stepped_digits(more,later)
      +other_tail = Digits.stepped_digits(rest,later)
      +heads = Equal.trans(U32,(head + other_head : U32),((U32.from_nat(digit) + U32.from_nat(other) : U32) * step : U32),(U32.from_nat(Nat.add(digit,other)) * step : U32),
        Equal.sym(U32,((U32.from_nat(digit) + U32.from_nat(other) : U32) * step : U32),(head + other_head : U32),Arithmetic.multiply_additive(U32.from_nat(digit),U32.from_nat(other),step)),
        Equal.cong(U32,U32,word => (word * step : U32),(U32.from_nat(digit) + U32.from_nat(other) : U32),U32.from_nat(Nat.add(digit,other)),WordArithmetic.native_add(digit,other)))
      +tails = stepped_added(more,rest,later,Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,Nat,more),1n+List.length(&2,Nat,rest),same))
      Equal.trans(U32,((head + tail : U32) + (other_head + other_tail : U32) : U32),((head + other_head : U32) + (tail + other_tail : U32) : U32),
        (U32.from_nat(Nat.add(digit,other)) * step + Digits.stepped_digits(Digits.added(more,rest),later) : U32),
        interchange(head,tail,other_head,other_tail),
        Equal.trans(U32,((head + other_head : U32) + (tail + other_tail : U32) : U32),((U32.from_nat(Nat.add(digit,other)) * step : U32) + (tail + other_tail : U32) : U32),
          (U32.from_nat(Nat.add(digit,other)) * step + Digits.stepped_digits(Digits.added(more,rest),later) : U32),
          Equal.cong(U32,U32,word => (word + (tail + other_tail : U32) : U32),(head + other_head : U32),(U32.from_nat(Nat.add(digit,other)) * step : U32),heads),
          Equal.cong(U32,U32,word => ((U32.from_nat(Nat.add(digit,other)) * step : U32) + word : U32),(tail + other_tail : U32),Digits.stepped_digits(Digits.added(more,rest),later),tails)))
    case Con{+digit,+more} Con{+other,+rest} Nil{}: Arithmetic.add_zero(0)
    case Nil{} Nil{} +steps: Arithmetic.add_zero(0)
    case Nil{} Con{+other,+rest} +steps:
      Empty.absurd({(Digits.stepped_digits(Nil{},steps) + Digits.stepped_digits(Con{other,rest},steps) : U32) == Digits.stepped_digits(Digits.added(Nil{},Con{other,rest}),steps) : U32},
        Digits.zero_successor(List.length(&2,Nat,rest),same))
    case Con{+digit,+more} Nil{} +steps:
      Empty.absurd({(Digits.stepped_digits(Con{digit,more},steps) + Digits.stepped_digits(Nil{},steps) : U32) == Digits.stepped_digits(Digits.added(Con{digit,more},Nil{}),steps) : U32},
        Digits.zero_successor(List.length(&2,Nat,more),Equal.sym(Nat,1n+List.length(&2,Nat,more),0n,same)))

# The starts and the counts of a box have one entry per axis.
def starts_length(+list: List<&2,BoxAxis>) -> {List.length(&2,Nat,box_starts(list)) == List.length(&2,U32,ViewPart.move_extents(box_axes(list))) : Nat}:
  match list:
    case Nil{}: {==}
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,box_starts(rest)),List.length(&2,U32,ViewPart.move_extents(box_axes(rest))),starts_length(rest))

# Box index index of a box writes the piece cell of the starts plus the
# coordinates of index in the box.
def box_cell(+list: List<&2,BoxAxis>,+inside: BoxIn(list),+source: U32,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+room: {Nat.is_le(Nat.double(Part.total(box_axes(list))),U32.to_nat(cap)) == True{} : Bool},
  +below: {Nat.is_lt(index,Part.total(box_axes(list))) == True{} : Bool}) ->
  {Part.cell_target(box_axes(list),index,source,box_target(list,0)) ==
    U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))) : U32}:
  +coordinates = Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)
  +starts = box_starts(list)
  +lengths = Equal.trans(Nat,List.length(&2,Nat,starts),List.length(&2,U32,ViewPart.move_extents(box_axes(list))),List.length(&2,Nat,coordinates),starts_length(list),
    Equal.sym(Nat,List.length(&2,Nat,coordinates),List.length(&2,U32,ViewPart.move_extents(box_axes(list))),Digits.coordinates_length(ViewPart.move_extents(box_axes(list)),index)))
  Equal.trans(U32,Part.cell_target(box_axes(list),index,source,box_target(list,0)),
    (box_target(list,0) + Box.stepped_sum(ViewPart.move_extents(box_axes(list)),ViewPart.move_targets(box_axes(list)),index) : U32),
    U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(starts,coordinates))),
    ViewPart.cell_target_sum(box_axes(list),index,source,box_target(list,0),cap,cap_ok,room,below),
    {%Equal.sym(U32,box_target(list,0),(0 + Digits.stepped_digits(starts,Tasks.strides_of(box_extents(list))) : U32),box_target_digits(list,0,inside)) :
        {(_ + Box.stepped_sum(ViewPart.move_extents(box_axes(list)),ViewPart.move_targets(box_axes(list)),index) : U32) == U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(starts,coordinates))) : U32}
      %Equal.sym(List<&2,U32>,ViewPart.move_targets(box_axes(list)),Tasks.strides_of(box_extents(list)),box_targets(list,inside)) :
        {((0 + Digits.stepped_digits(starts,Tasks.strides_of(box_extents(list))) : U32) + Box.stepped_sum(ViewPart.move_extents(box_axes(list)),_,index) : U32) == U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(starts,coordinates))) : U32}
      %Equal.sym(U32,(0 + Digits.stepped_digits(starts,Tasks.strides_of(box_extents(list))) : U32),Digits.stepped_digits(starts,Tasks.strides_of(box_extents(list))),Arithmetic.add_zero_left(Digits.stepped_digits(starts,Tasks.strides_of(box_extents(list))))) :
        {(_ + Box.stepped_sum(ViewPart.move_extents(box_axes(list)),Tasks.strides_of(box_extents(list)),index) : U32) == U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(starts,coordinates))) : U32}
      %Equal.sym(U32,Box.stepped_sum(ViewPart.move_extents(box_axes(list)),Tasks.strides_of(box_extents(list)),index),Digits.stepped_digits(coordinates,Tasks.strides_of(box_extents(list))),Digits.stepped_coordinates(ViewPart.move_extents(box_axes(list)),Tasks.strides_of(box_extents(list)),index)) :
        {(Digits.stepped_digits(starts,Tasks.strides_of(box_extents(list))) + _ : U32) == U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(starts,coordinates))) : U32}
      %Equal.sym(U32,U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(starts,coordinates))),Digits.stepped_digits(Digits.added(starts,coordinates),Tasks.strides_of(box_extents(list))),Digits.digits_stepped(box_extents(list),Digits.added(starts,coordinates))) :
        {(Digits.stepped_digits(starts,Tasks.strides_of(box_extents(list))) + Digits.stepped_digits(coordinates,Tasks.strides_of(box_extents(list))) : U32) == _ : U32}
      stepped_added(starts,coordinates,Tasks.strides_of(box_extents(list)),lengths) :
      {(box_target(list,0) + Box.stepped_sum(ViewPart.move_extents(box_axes(list)),ViewPart.move_targets(box_axes(list)),index) : U32) ==
        U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(starts,coordinates))) : U32}})

# The starts plus digits below the counts of a box lie below the piece.
def box_below(+list: List<&2,BoxAxis>,+digits: List<&2,Nat>,+inside: BoxIn(list),+below: Coordinates.DigitsBelow(ViewPart.move_extents(box_axes(list)),digits)) ->
  Coordinates.DigitsBelow(box_extents(list),Digits.added(box_starts(list),digits)):
  match list digits:
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest} Con{+digit,+more}:
      Held.Both{+stride,Held.Both{+offset,Held.Both{+positive,Held.Both{+fits,+later}}}} = inside
      KernelGemm.Both{+here,+deeper} = below
      KernelGemm.Both{Bounds.nat_trans(Nat.add(start,digit),Nat.add(start,U32.to_nat(count)),U32.to_nat(extent),Order.offset_lt(start,digit,U32.to_nat(count),here),fits),
        box_below(rest,more,later,deeper)}
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{+digit,+more}: Empty.absurd(Coordinates.DigitsBelow(Nil{},Digits.added(Nil{},Con{digit,more})),below)
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest} Nil{}:
      Empty.absurd(Coordinates.DigitsBelow(box_extents(Con{BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},rest}),Digits.added(box_starts(Con{BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},rest}),Nil{})),below)

# ---- Room, cells and the coordinates a box holds ----

# A box is not empty and has at most the cells of the piece.
def box_positive(+list: List<&2,BoxAxis>,+inside: BoxIn(list)) -> {Nat.is_lt(0n,Part.total(box_axes(list))) == True{} : Bool}:
  match list:
    case Nil{}: {==}
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest}:
      Held.Both{+stride,Held.Both{+offset,Held.Both{+positive,Held.Both{+fits,+later}}}} = inside
      Order.product_positive(U32.to_nat(count),Part.total(box_axes(rest)),positive,box_positive(rest,later))

def box_total_le(+list: List<&2,BoxAxis>,+inside: BoxIn(list)) -> {Nat.is_le(Part.total(box_axes(list)),ShapeLaws.elements(box_extents(list))) == True{} : Bool}:
  match list:
    case Nil{}: {==}
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest}:
      Held.Both{+stride,Held.Both{+offset,Held.Both{+positive,Held.Both{+fits,+later}}}} = inside
      +count_le = Order.transitive(U32.to_nat(count),Nat.add(start,U32.to_nat(count)),U32.to_nat(extent),
        {%Order.commutative(U32.to_nat(count),start) : {Nat.is_le(U32.to_nat(count),_) == True{} : Bool} Order.le_add(U32.to_nat(count),start) :
          {Nat.is_le(U32.to_nat(count),Nat.add(start,U32.to_nat(count))) == True{} : Bool}},fits)
      Order.transitive(Nat.mul(U32.to_nat(count),Part.total(box_axes(rest))),Nat.mul(U32.to_nat(extent),Part.total(box_axes(rest))),
        Nat.mul(U32.to_nat(extent),ShapeLaws.elements(box_extents(rest))),
        RowMajor.mul_le(U32.to_nat(count),U32.to_nat(extent),Part.total(box_axes(rest)),count_le),
        DivisionOrder.multiply_left_le(U32.to_nat(extent),Part.total(box_axes(rest)),ShapeLaws.elements(box_extents(rest)),box_total_le(rest,later)))

def box_room(+list: List<&2,BoxAxis>,+inside: BoxIn(list),+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(box_extents(list))),U32.to_nat(cap)) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(Part.total(box_axes(list))),U32.to_nat(cap)) == True{} : Bool}:
  Order.transitive(Nat.double(Part.total(box_axes(list))),Nat.double(ShapeLaws.elements(box_extents(list))),U32.to_nat(cap),Order.double_le(Part.total(box_axes(list)),ShapeLaws.elements(box_extents(list)),box_total_le(list,inside)),piece_room)

# The coordinates of a box index lie below the counts of the box.
def index_below(+list: List<&2,BoxAxis>,+index: Nat,+below: {Nat.is_lt(index,Part.total(box_axes(list))) == True{} : Bool}) ->
  Coordinates.DigitsBelow(ViewPart.move_extents(box_axes(list)),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)):
  Digits.coordinates_below(ViewPart.move_extents(box_axes(list)),index,{%ViewPart.total_elements(box_axes(list)) : {Nat.is_lt(index,_) == True{} : Bool} below :
    {Nat.is_lt(index,ShapeLaws.elements(ViewPart.move_extents(box_axes(list)))) == True{} : Bool}})

# The piece cell a box index writes lies inside the piece.
def value_below(+list: List<&2,BoxAxis>,+inside: BoxIn(list),+index: Nat,+below: {Nat.is_lt(index,Part.total(box_axes(list))) == True{} : Bool}) ->
  {Nat.is_lt(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))),ShapeLaws.elements(box_extents(list))) == True{} : Bool}:
  Coordinates.digits_below(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)),box_below(list,Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index),inside,index_below(list,index,below)))

# The number of the target cell a box index writes.
def cell_value(+list: List<&2,BoxAxis>,+inside: BoxIn(list),+source: U32,+index: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(box_extents(list))),U32.to_nat(cap)) == True{} : Bool},+below: {Nat.is_lt(index,Part.total(box_axes(list))) == True{} : Bool}) ->
  {U32.to_nat(Part.cell_target(box_axes(list),index,source,box_target(list,0))) == ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))) : Nat}:
  %Equal.sym(U32,Part.cell_target(box_axes(list),index,source,box_target(list,0)),U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))),box_cell(list,inside,source,index,cap,cap_ok,box_room(list,inside,cap,cap_ok,piece_room),below)) :
    {U32.to_nat(_) == ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))) : Nat}
  Bounds.roundtrip(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))),cap,cap_ok,
    Order.transitive(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))),ShapeLaws.elements(box_extents(list)),U32.to_nat(cap),Bounds.strict_le(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))),ShapeLaws.elements(box_extents(list)),value_below(list,inside,index,below)),
      Order.transitive(ShapeLaws.elements(box_extents(list)),Nat.double(ShapeLaws.elements(box_extents(list))),U32.to_nat(cap),Order.le_double(ShapeLaws.elements(box_extents(list))),piece_room)))

# Box cells from first on, count of them, lie inside a capacity that holds the
# piece.
def box_cells_below(count: Nat,+list: List<&2,BoxAxis>,+inside: BoxIn(list),+source: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(box_extents(list))),U32.to_nat(cap)) == True{} : Bool},+capacity: U32,
  +holds: {Nat.is_le(ShapeLaws.elements(box_extents(list)),U32.to_nat(capacity)) == True{} : Bool},+first: Nat,
  +within: {Nat.is_le(Nat.add(first,count),Part.total(box_axes(list))) == True{} : Bool}) -> Part.CellsBelow(box_axes(list),source,box_target(list,0),capacity,first,count):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +below = Order.room_index(first,rest,Part.total(box_axes(list)),within)
      Held.Both{{%Equal.sym(Nat,U32.to_nat(Part.cell_target(box_axes(list),first,source,box_target(list,0))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),first))),cell_value(list,inside,source,first,cap,cap_ok,piece_room,below)) :
            {Nat.is_lt(_,U32.to_nat(capacity)) == True{} : Bool}
          Bounds.nat_trans(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),first))),ShapeLaws.elements(box_extents(list)),U32.to_nat(capacity),value_below(list,inside,first,below),holds) :
          {Nat.is_lt(U32.to_nat(Part.cell_target(box_axes(list),first,source,box_target(list,0))),U32.to_nat(capacity)) == True{} : Bool}},
        box_cells_below(rest,list,inside,source,cap,cap_ok,piece_room,capacity,holds,1n+first,
          {%Addition.successor_right(first,rest) : {Nat.is_le(_,Part.total(box_axes(list))) == True{} : Bool} within : {Nat.is_le(Nat.add(1n+first,rest),Part.total(box_axes(list))) == True{} : Bool}})}

# ---- Which piece cells a box writes ----

# Digits lie in the box: on every axis at or after the start and before the
# start plus the count.
def box_holds(list: List<&2,BoxAxis>,digits: List<&2,Nat>) -> Bool:
  match list digits:
    case Con{BoxAxis{Tasks.Cut{+count,source_stride,source_offset,target_stride,target_offset},+start,extent},rest} Con{+digit,more}:
      Nat.is_le(start,digit) && (Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))) && box_holds(rest,more))
    case Nil{} Nil{}: True{}
    case list digits: False{}

# The coordinates of digits in the box.
def box_within(list: List<&2,BoxAxis>,digits: List<&2,Nat>) -> List<&2,Nat>:
  match list digits:
    case Con{BoxAxis{cut,+start,extent},rest} Con{+digit,more}: Nat.sub(digit,start) <> box_within(rest,more)
    case list digits: []

def head_digit(digits: List<&2,Nat>) -> Nat:
  match digits:
    case Nil{}: 0n
    case Con{+digit,rest}: digit

def tail_digits(digits: List<&2,Nat>) -> List<&2,Nat>:
  match digits:
    case Nil{}: []
    case Con{digit,rest}: rest

def and_first(+first: Bool,+second: Bool,+both: {first && second == True{} : Bool}) -> {first == True{} : Bool}:
  match first:
    case True{}: {==}
    case False{}: both

def and_second(+first: Bool,+second: Bool,+both: {first && second == True{} : Bool}) -> {second == True{} : Bool}:
  match first:
    case True{}: both
    case False{}: Empty.absurd({second == True{} : Bool},WordEncoding.false_true(both))

# Digits the box holds have coordinates below its counts.
def holds_below(+list: List<&2,BoxAxis>,+digits: List<&2,Nat>,+holds: {box_holds(list,digits) == True{} : Bool}) ->
  Coordinates.DigitsBelow(ViewPart.move_extents(box_axes(list)),box_within(list,digits)):
  match list digits:
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest} Con{+digit,+more}:
      +low = and_first(Nat.is_le(start,digit),Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))) && box_holds(rest,more),holds)
      +others = and_second(Nat.is_le(start,digit),Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))) && box_holds(rest,more),holds)
      +high = and_first(Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))),box_holds(rest,more),others)
      KernelGemm.Both{Order.sub_lt(digit,start,U32.to_nat(count),low,high),holds_below(rest,more,and_second(Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))),box_holds(rest,more),others))}
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{+digit,+more}: Empty.absurd(Coordinates.DigitsBelow(Nil{},Nil{}),WordEncoding.false_true(holds))
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest} Nil{}:
      Empty.absurd(Coordinates.DigitsBelow(ViewPart.move_extents(box_axes(Con{BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},rest})),Nil{}),WordEncoding.false_true(holds))

# The starts plus the coordinates in the box of digits it holds are the digits.
def holds_added(+list: List<&2,BoxAxis>,+digits: List<&2,Nat>,+holds: {box_holds(list,digits) == True{} : Bool}) ->
  {Digits.added(box_starts(list),box_within(list,digits)) == digits : List<&2,Nat>}:
  match list digits:
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest} Con{+digit,+more}:
      +low = and_first(Nat.is_le(start,digit),Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))) && box_holds(rest,more),holds)
      +others = and_second(Nat.is_le(start,digit),Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))) && box_holds(rest,more),holds)
      %Equal.sym(Nat,Nat.add(start,Nat.sub(digit,start)),digit,Order.restore_shift(start,digit,low)) : {Con{_,Digits.added(box_starts(rest),box_within(rest,more))} == Con{digit,more} : List<&2,Nat>}
      %Equal.sym(List<&2,Nat>,Digits.added(box_starts(rest),box_within(rest,more)),more,holds_added(rest,more,and_second(Nat.is_lt(digit,Nat.add(start,U32.to_nat(count))),box_holds(rest,more),others))) :
        {Con{digit,_} == Con{digit,more} : List<&2,Nat>}
      {==}
    case Nil{} Nil{}: {==}
    case Nil{} Con{+digit,+more}: Empty.absurd({Digits.added(Nil{},Nil{}) == Con{digit,more} : List<&2,Nat>},WordEncoding.false_true(holds))
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest} Nil{}:
      Empty.absurd({Digits.added(box_starts(Con{BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},rest}),box_within(Con{BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},rest},Nil{})) == Nil{} : List<&2,Nat>},
        WordEncoding.false_true(holds))

# The box holds its starts plus digits below its counts.
def added_holds(+list: List<&2,BoxAxis>,+digits: List<&2,Nat>,+below: Coordinates.DigitsBelow(ViewPart.move_extents(box_axes(list)),digits)) ->
  {box_holds(list,Digits.added(box_starts(list),digits)) == True{} : Bool}:
  match list digits:
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest} Con{+digit,+more}:
      KernelGemm.Both{+here,+deeper} = below
      %Equal.sym(Bool,Nat.is_le(start,Nat.add(start,digit)),True{},Order.le_add(start,digit)) : {_ && (Nat.is_lt(Nat.add(start,digit),Nat.add(start,U32.to_nat(count))) && box_holds(rest,Digits.added(box_starts(rest),more))) == True{} : Bool}
      %Equal.sym(Bool,Nat.is_lt(Nat.add(start,digit),Nat.add(start,U32.to_nat(count))),True{},Order.offset_lt(start,digit,U32.to_nat(count),here)) : {True{} && (_ && box_holds(rest,Digits.added(box_starts(rest),more))) == True{} : Bool}
      %Equal.sym(Bool,box_holds(rest,Digits.added(box_starts(rest),more)),True{},added_holds(rest,more,deeper)) : {True{} && (True{} && _) == True{} : Bool}
      {==}
    case Nil{} Nil{}: {==}
    case Nil{} Con{+digit,+more}: Empty.absurd({box_holds(Nil{},Digits.added(Nil{},Con{digit,more})) == True{} : Bool},below)
    case Con{BoxAxis{Tasks.Cut{+count,+source_stride,+source_offset,+target_stride,+target_offset},+start,+extent},+rest} Nil{}:
      Empty.absurd({box_holds(Con{BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},rest},Digits.added(box_starts(Con{BoxAxis{Tasks.Cut{count,source_stride,source_offset,target_stride,target_offset},start,extent},rest}),Nil{})) == True{} : Bool},below)

# Equal sums with the same starts have equal digits.
def added_cancel(starts: List<&2,Nat>,left: List<&2,Nat>,right: List<&2,Nat>,+left_length: {List.length(&2,Nat,left) == List.length(&2,Nat,starts) : Nat},
  +right_length: {List.length(&2,Nat,right) == List.length(&2,Nat,starts) : Nat},+same: {Digits.added(starts,left) == Digits.added(starts,right) : List<&2,Nat>}) ->
  {left == right : List<&2,Nat>}:
  match starts left right:
    case Con{+start,+more} Con{+digit,+rest} Con{+other,+others}:
      +sum = Equal.cong(List<&2,Nat>,Nat,head_digit,Con{Nat.add(start,digit),Digits.added(more,rest)},Con{Nat.add(start,other),Digits.added(more,others)},same)
      +head = Equal.trans(Nat,digit,Nat.sub(Nat.add(start,digit),start),other,Equal.sym(Nat,Nat.sub(Nat.add(start,digit),start),digit,Order.subtract_shift(start,digit)),
        {%Equal.sym(Nat,Nat.add(start,digit),Nat.add(start,other),sum) : {Nat.sub(_,start) == other : Nat} Order.subtract_shift(start,other) : {Nat.sub(Nat.add(start,digit),start) == other : Nat}})
      +tail = added_cancel(more,rest,others,Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,Nat,rest),1n+List.length(&2,Nat,more),left_length),
        Equal.cong(Nat,Nat,count => Digits.before(count),1n+List.length(&2,Nat,others),1n+List.length(&2,Nat,more),right_length),
        Equal.cong(List<&2,Nat>,List<&2,Nat>,tail_digits,Con{Nat.add(start,digit),Digits.added(more,rest)},Con{Nat.add(start,other),Digits.added(more,others)},same))
      %head : {Con{digit,rest} == Con{_,others} : List<&2,Nat>}
      %tail : {Con{digit,rest} == Con{digit,_} : List<&2,Nat>}
      {==}
    case Con{+start,+more} Con{+digit,+rest} Nil{}:
      Empty.absurd({Con{digit,rest} == Nil{} : List<&2,Nat>},Digits.zero_successor(List.length(&2,Nat,more),right_length))
    case Con{+start,+more} Nil{} +right:
      Empty.absurd({Nil{} == right : List<&2,Nat>},Digits.zero_successor(List.length(&2,Nat,more),left_length))
    case Nil{} Nil{} Nil{}: {==}
    case Nil{} Nil{} Con{+other,+others}:
      Empty.absurd({Nil{} == Con{other,others} : List<&2,Nat>},Digits.zero_successor(List.length(&2,Nat,others),Equal.sym(Nat,1n+List.length(&2,Nat,others),0n,right_length)))
    case Nil{} Con{+digit,+rest} +right:
      Empty.absurd({Con{digit,rest} == right : List<&2,Nat>},Digits.zero_successor(List.length(&2,Nat,rest),Equal.sym(Nat,1n+List.length(&2,Nat,rest),0n,left_length)))

# The coordinates in the piece of the cell a box index writes.
def cell_coordinates_of(+list: List<&2,BoxAxis>,+inside: BoxIn(list),+index: Nat,+below: {Nat.is_lt(index,Part.total(box_axes(list))) == True{} : Bool}) ->
  {Targets.cell_coordinates(box_extents(list),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))) == Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)) : List<&2,Nat>}:
  Digits.coordinates_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)),box_below(list,Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index),inside,index_below(list,index,below)))

# The coordinates of a box index have one digit per axis of the box.
def coordinates_length_of(+list: List<&2,BoxAxis>,+index: Nat) -> {List.length(&2,Nat,Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)) == List.length(&2,Nat,box_starts(list)) : Nat}:
  Equal.trans(Nat,List.length(&2,Nat,Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)),List.length(&2,U32,ViewPart.move_extents(box_axes(list))),List.length(&2,Nat,box_starts(list)),
    Digits.coordinates_length(ViewPart.move_extents(box_axes(list)),index),Equal.sym(Nat,List.length(&2,Nat,box_starts(list)),List.length(&2,U32,ViewPart.move_extents(box_axes(list))),starts_length(list)))

# Box indices whose cells have the same number are the same.
def same_value(+list: List<&2,BoxAxis>,+inside: BoxIn(list),+index: Nat,+other: Nat,+index_in: {Nat.is_lt(index,Part.total(box_axes(list))) == True{} : Bool},+other_in: {Nat.is_lt(other,Part.total(box_axes(list))) == True{} : Bool},
  +same: {ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))) == ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))) : Nat}) -> {other == index : Nat}:
  +coordinates = added_cancel(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index),coordinates_length_of(list,other),coordinates_length_of(list,index),
    Equal.trans(List<&2,Nat>,Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other)),Targets.cell_coordinates(box_extents(list),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other)))),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)),
      Equal.sym(List<&2,Nat>,Targets.cell_coordinates(box_extents(list),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other)))),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other)),cell_coordinates_of(list,inside,other,other_in)),
      {%Equal.sym(Nat,ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))),same) : {Targets.cell_coordinates(box_extents(list),_) == Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)) : List<&2,Nat>}
        cell_coordinates_of(list,inside,index,index_in) : {Targets.cell_coordinates(box_extents(list),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other)))) == Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)) : List<&2,Nat>}}))
  +other_elements = {%ViewPart.total_elements(box_axes(list)) : {Nat.is_lt(other,_) == True{} : Bool} other_in : {Nat.is_lt(other,ShapeLaws.elements(ViewPart.move_extents(box_axes(list)))) == True{} : Bool}}
  +index_elements = {%ViewPart.total_elements(box_axes(list)) : {Nat.is_lt(index,_) == True{} : Bool} index_in : {Nat.is_lt(index,ShapeLaws.elements(ViewPart.move_extents(box_axes(list)))) == True{} : Bool}}
  Equal.trans(Nat,other,ShapeLaws.digits_value(ViewPart.move_extents(box_axes(list)),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other)),index,
    Equal.sym(Nat,ShapeLaws.digits_value(ViewPart.move_extents(box_axes(list)),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other)),other,Digits.value_coordinates(ViewPart.move_extents(box_axes(list)),other,other_elements)),
    {%Equal.sym(List<&2,Nat>,Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index),coordinates) : {ShapeLaws.digits_value(ViewPart.move_extents(box_axes(list)),_) == index : Nat}
      Digits.value_coordinates(ViewPart.move_extents(box_axes(list)),index,index_elements) : {ShapeLaws.digits_value(ViewPart.move_extents(box_axes(list)),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other)) == index : Nat}})

def values_apart(found: Bool,equal: Bool,+list: List<&2,BoxAxis>,+inside: BoxIn(list),+index: Nat,+other: Nat,+index_in: {Nat.is_lt(index,Part.total(box_axes(list))) == True{} : Bool},+other_in: {Nat.is_lt(other,Part.total(box_axes(list))) == True{} : Bool},
  +by_value: {Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))) == found : Bool},+by_index: {Nat.is_eq(other,index) == equal : Bool}) ->
  {Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))) == Nat.is_eq(other,index) : Bool}:
  match found equal:
    case True{} True{}: Equal.trans(Bool,Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))),True{},Nat.is_eq(other,index),by_value,Equal.sym(Bool,Nat.is_eq(other,index),True{},by_index))
    case False{} False{}: Equal.trans(Bool,Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))),False{},Nat.is_eq(other,index),by_value,Equal.sym(Bool,Nat.is_eq(other,index),False{},by_index))
    case True{} False{}:
      +same = same_value(list,inside,index,other,index_in,other_in,Order.equal_of_is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))),by_value))
      Empty.absurd({Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))) == Nat.is_eq(other,index) : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(other,index),True{},Equal.sym(Bool,Nat.is_eq(other,index),False{},by_index),
          {%Equal.sym(Nat,other,index,same) : {Nat.is_eq(_,index) == True{} : Bool} Order.equal_reflexive(index) : {Nat.is_eq(other,index) == True{} : Bool}})))
    case False{} True{}:
      +same = Order.equal_of_is_eq(other,index,by_index)
      Empty.absurd({Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))) == Nat.is_eq(other,index) : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))),True{},
          Equal.sym(Bool,Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))),False{},by_value),
          {%Equal.sym(Nat,other,index,same) : {Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),_))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))) == True{} : Bool}
            Order.equal_reflexive(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))) : {Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),other))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))) == True{} : Bool}})))

# Box cells from first on, count of them, are the cell of index only at
# index.
def box_apart(count: Nat,+list: List<&2,BoxAxis>,+inside: BoxIn(list),+source: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(box_extents(list))),U32.to_nat(cap)) == True{} : Bool},+index: Nat,+index_in: {Nat.is_lt(index,Part.total(box_axes(list))) == True{} : Bool},+first: Nat,
  +within: {Nat.is_le(Nat.add(first,count),Part.total(box_axes(list))) == True{} : Bool}) -> Part.CellsApart(box_axes(list),source,box_target(list,0),index,first,count):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      +first_in = Order.room_index(first,rest,Part.total(box_axes(list)),within)
      Held.Both{Equal.trans(Bool,U32.is_eq(Part.cell_target(box_axes(list),first,source,box_target(list,0)),Part.cell_target(box_axes(list),index,source,box_target(list,0))),Nat.is_eq(U32.to_nat(Part.cell_target(box_axes(list),first,source,box_target(list,0))),U32.to_nat(Part.cell_target(box_axes(list),index,source,box_target(list,0)))),Nat.is_eq(first,index),
          Comparison.eq(Part.cell_target(box_axes(list),first,source,box_target(list,0)),Part.cell_target(box_axes(list),index,source,box_target(list,0))),
          {%Equal.sym(Nat,U32.to_nat(Part.cell_target(box_axes(list),first,source,box_target(list,0))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),first))),cell_value(list,inside,source,first,cap,cap_ok,piece_room,first_in)) :
              {Nat.is_eq(_,U32.to_nat(Part.cell_target(box_axes(list),index,source,box_target(list,0)))) == Nat.is_eq(first,index) : Bool}
            {%Equal.sym(Nat,U32.to_nat(Part.cell_target(box_axes(list),index,source,box_target(list,0))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))),cell_value(list,inside,source,index,cap,cap_ok,piece_room,index_in)) :
                {Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),first))),_) == Nat.is_eq(first,index) : Bool}
              values_apart(Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),first))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))),Nat.is_eq(first,index),list,inside,index,first,index_in,first_in,{==},{==}) :
              {Nat.is_eq(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),first))),U32.to_nat(Part.cell_target(box_axes(list),index,source,box_target(list,0)))) == Nat.is_eq(first,index) : Bool}} :
            {Nat.is_eq(U32.to_nat(Part.cell_target(box_axes(list),first,source,box_target(list,0))),U32.to_nat(Part.cell_target(box_axes(list),index,source,box_target(list,0)))) == Nat.is_eq(first,index) : Bool}}),
        box_apart(rest,list,inside,source,cap,cap_ok,piece_room,index,index_in,1n+first,
          {%Addition.successor_right(first,rest) : {Nat.is_le(_,Part.total(box_axes(list))) == True{} : Bool} within : {Nat.is_le(Nat.add(1n+first,rest),Part.total(box_axes(list))) == True{} : Bool}})}

# A piece cell the box holds is the cell of its coordinates in the box.
def hit_index(+list: List<&2,BoxAxis>,+query: U32) -> Nat:
  ShapeLaws.digits_value(ViewPart.move_extents(box_axes(list)),box_within(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query))))

def hit_below(+list: List<&2,BoxAxis>,+query: U32,+holds: {box_holds(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query))) == True{} : Bool}) -> {Nat.is_lt(hit_index(list,query),Part.total(box_axes(list))) == True{} : Bool}:
  %Equal.sym(Nat,Part.total(box_axes(list)),ShapeLaws.elements(ViewPart.move_extents(box_axes(list))),ViewPart.total_elements(box_axes(list))) : {Nat.is_lt(hit_index(list,query),_) == True{} : Bool}
  Coordinates.digits_below(ViewPart.move_extents(box_axes(list)),box_within(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query))),holds_below(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query)),holds))

def hit_cell(+list: List<&2,BoxAxis>,+inside: BoxIn(list),+source: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(box_extents(list))),U32.to_nat(cap)) == True{} : Bool},+query: U32,
  +query_in: {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(box_extents(list))) == True{} : Bool},+holds: {box_holds(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query))) == True{} : Bool}) ->
  {Part.cell_target(box_axes(list),hit_index(list,query),source,box_target(list,0)) == query : U32}:
  +index = hit_index(list,query)
  +within = box_within(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query)))
  +inner = Digits.coordinates_value(ViewPart.move_extents(box_axes(list)),within,holds_below(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query)),holds))
  Equal.trans(U32,Part.cell_target(box_axes(list),index,source,box_target(list,0)),U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))),query,
    box_cell(list,inside,source,index,cap,cap_ok,box_room(list,inside,cap,cap_ok,piece_room),hit_below(list,query,holds)),
    {%Equal.sym(List<&2,Nat>,Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index),within,inner) : {U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),_))) == query : U32}
      %Equal.sym(List<&2,Nat>,Digits.added(box_starts(list),within),Targets.cell_coordinates(box_extents(list),U32.to_nat(query)),holds_added(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query)),holds)) : {U32.from_nat(ShapeLaws.digits_value(box_extents(list),_)) == query : U32}
      %Equal.sym(Nat,ShapeLaws.digits_value(box_extents(list),Targets.cell_coordinates(box_extents(list),U32.to_nat(query))),U32.to_nat(query),Digits.value_coordinates(box_extents(list),U32.to_nat(query),query_in)) : {U32.from_nat(_) == query : U32}
      Arithmetic.native_roundtrip(query) : {U32.from_nat(ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))) == query : U32}})

def miss_cell(found: Bool,+list: List<&2,BoxAxis>,+inside: BoxIn(list),+source: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(box_extents(list))),U32.to_nat(cap)) == True{} : Bool},+query: U32,
  +query_in: {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(box_extents(list))) == True{} : Bool},+out: {box_holds(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query))) == False{} : Bool},
  +index: Nat,+index_in: {Nat.is_lt(index,Part.total(box_axes(list))) == True{} : Bool},+by_cell: {U32.is_eq(Part.cell_target(box_axes(list),index,source,box_target(list,0)),query) == found : Bool}) ->
  {U32.is_eq(Part.cell_target(box_axes(list),index,source,box_target(list,0)),query) == False{} : Bool}:
  match found:
    case False{}: by_cell
    case True{}:
      +same = Arithmetic.equal_of_is_eq(Part.cell_target(box_axes(list),index,source,box_target(list,0)),query,by_cell)
      +numbers = Equal.trans(Nat,ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))),U32.to_nat(Part.cell_target(box_axes(list),index,source,box_target(list,0))),U32.to_nat(query),
        Equal.sym(Nat,U32.to_nat(Part.cell_target(box_axes(list),index,source,box_target(list,0))),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))),cell_value(list,inside,source,index,cap,cap_ok,piece_room,index_in)),
        Equal.cong(U32,Nat,U32.to_nat,Part.cell_target(box_axes(list),index,source,box_target(list,0)),query,same))
      +digits = Equal.trans(List<&2,Nat>,Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)),Targets.cell_coordinates(box_extents(list),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))),Targets.cell_coordinates(box_extents(list),U32.to_nat(query)),
        Equal.sym(List<&2,Nat>,Targets.cell_coordinates(box_extents(list),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)))),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index)),cell_coordinates_of(list,inside,index,index_in)),
        Equal.cong(Nat,List<&2,Nat>,number => Targets.cell_coordinates(box_extents(list),number),ShapeLaws.digits_value(box_extents(list),Digits.added(box_starts(list),Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index))),U32.to_nat(query),numbers))
      +held = {%digits : {box_holds(list,_) == True{} : Bool} added_holds(list,Targets.cell_coordinates(ViewPart.move_extents(box_axes(list)),index),index_below(list,index,index_in)) :
        {box_holds(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query))) == True{} : Bool}}
      Empty.absurd({U32.is_eq(Part.cell_target(box_axes(list),index,source,box_target(list,0)),query) == False{} : Bool},
        WordEncoding.false_true(Equal.trans(Bool,False{},box_holds(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query))),True{},Equal.sym(Bool,box_holds(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query))),False{},out),held)))

# Box cells from first on, count of them, are not a piece cell the box does
# not hold.
def box_miss(count: Nat,+list: List<&2,BoxAxis>,+inside: BoxIn(list),+source: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+piece_room: {Nat.is_le(Nat.double(ShapeLaws.elements(box_extents(list))),U32.to_nat(cap)) == True{} : Bool},+query: U32,
  +query_in: {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(box_extents(list))) == True{} : Bool},+out: {box_holds(list,Targets.cell_coordinates(box_extents(list),U32.to_nat(query))) == False{} : Bool},+first: Nat,
  +within: {Nat.is_le(Nat.add(first,count),Part.total(box_axes(list))) == True{} : Bool}) -> Part.CellsMiss(box_axes(list),source,box_target(list,0),query,first,count):
  match count:
    case 0n: Unit{}
    case 1n+ +rest:
      Held.Both{miss_cell(U32.is_eq(Part.cell_target(box_axes(list),first,source,box_target(list,0)),query),list,inside,source,cap,cap_ok,piece_room,query,query_in,out,first,Order.room_index(first,rest,Part.total(box_axes(list)),within),{==}),
        box_miss(rest,list,inside,source,cap,cap_ok,piece_room,query,query_in,out,1n+first,
          {%Addition.successor_right(first,rest) : {Nat.is_le(_,Part.total(box_axes(list))) == True{} : Bool} within : {Nat.is_le(Nat.add(1n+first,rest),Part.total(box_axes(list))) == True{} : Bool}})}
