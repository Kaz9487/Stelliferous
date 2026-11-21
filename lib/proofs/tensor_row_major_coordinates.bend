# Explicit row-major digits, including extent-one axes, for copy and flip.
import Base
import ../tensor_view.bend as Views
import ./tensor_cursor_model.bend as Model
import ./tensor_cursor_refinement.bend as Cursor
import ./tensor_cursor_order.bend as Order
import ./tensor_copy_values.bend as Copy
import ./tensor_mirror_address.bend as Mirror
import ./tensor_mirror_coordinates.bend as MirrorCoordinates
import ./tensor_coordinate_uniqueness.bend as Unique
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_count_completion.bend as CountCompletion
import ./tensor_linear_layout.bend as Linear
import ./row_major_index.bend as RowMajor
import ./nat_order.bend as NatOrder
import ./natural_addition.bend as Addition
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./word_encoding.bend as Words
import ./boolean_logic.bend as Boolean
import ../kernel_gemm.bend as KernelGemm

type Point is Data:
  Point{extent: U32,coordinate: Nat,stride: U32,reverse: Bool}

def axes(points: List<&2,Point>) -> List<&2,Views.Axis>:
  match points:
    case []: []
    case Point{extent,coordinate,stride,reverse} <> tail: Views.Axis{extent,stride} <> axes(tail)

def flags(points: List<&2,Point>) -> List<&2,Bool>:
  match points:
    case []: []
    case Point{extent,coordinate,stride,reverse} <> tail: reverse <> flags(tail)

def placed(points: List<&2,Point>,+reversed: List<&2,Mirror.Axis>) -> List<&2,Mirror.Axis>:
  match points:
    case []: reversed
    case Point{+extent,coordinate,stride,reverse} <> tail:
      placed(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed))

def position(points: List<&2,Point>,prefix: Nat) -> Nat:
  match points:
    case []: prefix
    case Point{extent,coordinate,stride,reverse} <> tail:
      position(tail,Nat.add(Nat.mul(prefix,U32.to_nat(extent)),coordinate))

def Valid(points: List<&2,Point>) -> Data:
  match points:
    case []: Unit
    case Point{extent,coordinate,stride,reverse} <> tail:
      KernelGemm.Both<{Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool},Valid(tail)>

def zero_below_one(+coordinate: Nat,inside: {Nat.is_lt(coordinate,1n) == True{} : Bool}) -> {coordinate == 0n : Nat}:
  match coordinate:
    case 0n: {==}
    case 1n+ +rest: Empty.absurd({1n+rest == 0n : Nat},Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(rest,0n),True{},Equal.sym(Bool,Nat.is_lt(rest,0n),False{},Division.lt_zero(rest)),inside)))

def chosen_position(+extent: Nat,+coordinate: Nat,+stride: U32,+reverse: Bool,+reversed: List<&2,Mirror.Axis>,
  inside: {Nat.is_lt(coordinate,extent) == True{} : Bool}) ->
  {Model.position(Mirror.modeled(False{},Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,extent),Mirror.Axis{extent,coordinate,stride,reverse} <> reversed,reversed))) ==
    Nat.add(Nat.mul(Model.position(Mirror.modeled(False{},reversed)),extent),coordinate) : Nat}:
  match extent:
    case 0n:
      Empty.absurd({Model.position(Mirror.modeled(False{},reversed)) == Nat.add(Nat.mul(Model.position(Mirror.modeled(False{},reversed)),0n),coordinate) : Nat},Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(coordinate,0n),True{},Equal.sym(Bool,Nat.is_lt(coordinate,0n),False{},Division.lt_zero(coordinate)),inside)))
    case 1n:
      %Equal.sym(Nat,coordinate,0n,zero_below_one(coordinate,inside)) :
        {Model.position(Mirror.modeled(False{},reversed)) == Nat.add(Nat.mul(Model.position(Mirror.modeled(False{},reversed)),1n),_) : Nat}
      %Equal.sym(Nat,Nat.mul(Model.position(Mirror.modeled(False{},reversed)),1n),Model.position(Mirror.modeled(False{},reversed)),NatOrder.times_one(Model.position(Mirror.modeled(False{},reversed)))) :
        {Model.position(Mirror.modeled(False{},reversed)) == Nat.add(_,0n) : Nat}
      Equal.sym(Nat,Nat.add(Model.position(Mirror.modeled(False{},reversed)),0n),Model.position(Mirror.modeled(False{},reversed)),Addition.zero_right(Model.position(Mirror.modeled(False{},reversed))))
    case 2n+ +rest: Unique.position_form(2n+rest,coordinate,Mirror.modeled(False{},reversed))

def position_exact(+points: List<&2,Point>,+reversed: List<&2,Mirror.Axis>,valid: Valid(points)) ->
  {Model.position(Mirror.modeled(False{},placed(points,reversed))) == position(points,Model.position(Mirror.modeled(False{},reversed))) : Nat}:
  match points:
    case []: {==}
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      KernelGemm.Both{+inside,tail_valid} = valid
      %chosen_position(U32.to_nat(extent),coordinate,stride,reverse,reversed,inside) :
        {Model.position(Mirror.modeled(False{},placed(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed)))) == position(tail,_) : Nat}
      position_exact(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed),tail_valid)

def chosen_valid(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: U32,+reverse: Bool,+reversed: List<&2,Mirror.Axis>,
  inside: {Nat.is_lt(coordinate,extent) == True{} : Bool},valid: Cursor.Valid(Mirror.modeled(False{},reversed))) ->
  Cursor.Valid(Mirror.modeled(False{},Bool.pick(List<&2,Mirror.Axis>,selected,Mirror.Axis{extent,coordinate,stride,reverse} <> reversed,reversed))):
  match selected:
    case False{}: valid
    case True{}: KernelGemm.Both{inside,valid}

def placed_valid(+points: List<&2,Point>,+reversed: List<&2,Mirror.Axis>,valid: Valid(points),initial: Cursor.Valid(Mirror.modeled(False{},reversed))) ->
  Cursor.Valid(Mirror.modeled(False{},placed(points,reversed))):
  match points:
    case []: initial
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      KernelGemm.Both{inside,tail_valid} = valid
      placed_valid(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed),tail_valid,
        chosen_valid(Nat.is_lt(1n,U32.to_nat(extent)),U32.to_nat(extent),coordinate,stride,reverse,reversed,inside,initial))

def chosen_layout(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: U32,+reverse: Bool,+left: List<&2,Mirror.Axis>,+right: List<&2,Mirror.Axis>,
  same: {Unique.layout(Mirror.modeled(False{},left)) == Unique.layout(Mirror.modeled(False{},right)) : List<&2,Model.Coordinate>}) ->
  {Unique.layout(Mirror.modeled(False{},Bool.pick(List<&2,Mirror.Axis>,selected,Mirror.Axis{extent,0n,stride,reverse} <> left,left))) ==
    Unique.layout(Mirror.modeled(False{},Bool.pick(List<&2,Mirror.Axis>,selected,Mirror.Axis{extent,coordinate,stride,reverse} <> right,right))) : List<&2,Model.Coordinate>}:
  match selected:
    case False{}: same
    case True{}:
      Equal.cong(List<&2,Model.Coordinate>,List<&2,Model.Coordinate>,rest => Model.Coordinate{extent,0n,U32.to_nat(stride)} <> rest,
        Unique.layout(Mirror.modeled(False{},left)),Unique.layout(Mirror.modeled(False{},right)),same)

def built_layout(+points: List<&2,Point>,+left: List<&2,Mirror.Axis>,+right: List<&2,Mirror.Axis>,
  same: {Unique.layout(Mirror.modeled(False{},left)) == Unique.layout(Mirror.modeled(False{},right)) : List<&2,Model.Coordinate>}) ->
  {Unique.layout(Mirror.modeled(False{},Mirror.build(axes(points),flags(points),left))) == Unique.layout(Mirror.modeled(False{},placed(points,right))) : List<&2,Model.Coordinate>}:
  match points:
    case []: same
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      built_layout(tail,
        Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),0n,stride,reverse} <> left,left),
        Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> right,right),
        chosen_layout(Nat.is_lt(1n,U32.to_nat(extent)),U32.to_nat(extent),coordinate,stride,reverse,left,right,same))

def chosen_flags(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: U32,+reverse: Bool,+left: List<&2,Mirror.Axis>,+right: List<&2,Mirror.Axis>,
  same: {MirrorCoordinates.flags(left) == MirrorCoordinates.flags(right) : List<&2,Bool>}) ->
  {MirrorCoordinates.flags(Bool.pick(List<&2,Mirror.Axis>,selected,Mirror.Axis{extent,0n,stride,reverse} <> left,left)) ==
    MirrorCoordinates.flags(Bool.pick(List<&2,Mirror.Axis>,selected,Mirror.Axis{extent,coordinate,stride,reverse} <> right,right)) : List<&2,Bool>}:
  match selected:
    case False{}: same
    case True{}:
      Equal.cong(List<&2,Bool>,List<&2,Bool>,rest => reverse <> rest,MirrorCoordinates.flags(left),MirrorCoordinates.flags(right),same)

def built_flags(+points: List<&2,Point>,+left: List<&2,Mirror.Axis>,+right: List<&2,Mirror.Axis>,
  same: {MirrorCoordinates.flags(left) == MirrorCoordinates.flags(right) : List<&2,Bool>}) ->
  {MirrorCoordinates.flags(Mirror.build(axes(points),flags(points),left)) == MirrorCoordinates.flags(placed(points,right)) : List<&2,Bool>}:
  match points:
    case []: same
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      built_flags(tail,
        Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),0n,stride,reverse} <> left,left),
        Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> right,right),
        chosen_flags(Nat.is_lt(1n,U32.to_nat(extent)),U32.to_nat(extent),coordinate,stride,reverse,left,right,same))

# At a bounded row-major position the mirror model holds the explicit digits.
def reached(+points: List<&2,Point>,+valid: Valid(points),
  +inside: {Nat.is_lt(position(points,0n),ShapeLaws.elements(Views.axis_extents(axes(points)))) == True{} : Bool}) ->
  {Mirror.iterate(position(points,0n),Mirror.build(axes(points),flags(points),[])) == placed(points,[]) : List<&2,Mirror.Axis>}:
  +start = Mirror.build(axes(points),flags(points),[])
  +nonempty = CountCompletion.positive_not_zero(Views.axis_extents(axes(points)),NatOrder.le_lt_trans(0n,position(points,0n),ShapeLaws.elements(Views.axis_extents(axes(points))),NatOrder.zero_le(position(points,0n)),inside))
  +zero = Equal.trans(Nat,Model.position(Mirror.modeled(False{},start)),Model.position(Model.prepare_axes(axes(points),[])),0n,
    Equal.cong(List<&2,Model.Coordinate>,Nat,Model.position,Mirror.modeled(False{},start),Model.prepare_axes(axes(points),[]),
      Equal.sym(List<&2,Model.Coordinate>,Model.prepare_axes(axes(points),[]),Mirror.modeled(False{},start),Mirror.built_forward(axes(points),flags(points),[]))),Order.prepared_zero(axes(points),[],{==}))
  MirrorCoordinates.reached(position(points,0n),start,placed(points,[]),
    %Mirror.built_forward(axes(points),flags(points),[]) : Cursor.Valid(_)
    Cursor.prepared_valid(axes(points),[],Unit{}),
    placed_valid(points,[],valid,Unit{}),built_layout(points,[],[],{==}),built_flags(points,[],[],{==}),
    %Equal.sym(Nat,Model.position(Mirror.modeled(False{},start)),0n,zero) :
      {Model.position(Mirror.modeled(False{},placed(points,[]))) == Nat.add(_,position(points,0n)) : Nat}
    position_exact(points,[],valid),
    %Equal.sym(Nat,Model.position(Mirror.modeled(False{},start)),0n,zero) :
      {Nat.is_lt(Nat.add(_,position(points,0n)),Model.extent_product(Mirror.modeled(False{},start))) == True{} : Bool}
    %Mirror.built_forward(axes(points),flags(points),[]) : {Nat.is_lt(position(points,0n),Model.extent_product(_)) == True{} : Bool}
    %Equal.sym(Nat,Model.extent_product(Model.prepare_axes(axes(points),[])),ShapeLaws.elements(Views.axis_extents(axes(points))),Order.initial_size(axes(points),nonempty)) :
      {Nat.is_lt(position(points,0n),_) == True{} : Bool} inside)

def count(points: List<&2,Point>,prefix: Nat) -> Nat:
  match points:
    case []: prefix
    case Point{extent,coordinate,stride,reverse} <> tail: count(tail,Nat.mul(prefix,U32.to_nat(extent)))

def count_exact(+points: List<&2,Point>,+prefix: Nat) -> {count(points,prefix) == Nat.mul(prefix,ShapeLaws.elements(Views.axis_extents(axes(points)))) : Nat}:
  match points:
    case []: Equal.sym(Nat,Nat.mul(prefix,1n),prefix,NatOrder.times_one(prefix))
    case Point{+extent,coordinate,stride,reverse} <> +tail:
      Equal.trans(Nat,count(tail,Nat.mul(prefix,U32.to_nat(extent))),Nat.mul(Nat.mul(prefix,U32.to_nat(extent)),ShapeLaws.elements(Views.axis_extents(axes(tail)))),
        Nat.mul(prefix,Nat.mul(U32.to_nat(extent),ShapeLaws.elements(Views.axis_extents(axes(tail))))),count_exact(tail,Nat.mul(prefix,U32.to_nat(extent))),RowMajor.assoc(prefix,U32.to_nat(extent),ShapeLaws.elements(Views.axis_extents(axes(tail)))))

def bounded(+points: List<&2,Point>,+prefix: Nat,+capacity: Nat,room: {Nat.is_lt(prefix,capacity) == True{} : Bool},valid: Valid(points)) ->
  {Nat.is_lt(position(points,prefix),count(points,capacity)) == True{} : Bool}:
  match points:
    case []: room
    case Point{+extent,+coordinate,stride,reverse} <> +tail:
      KernelGemm.Both{inside,tail_valid} = valid
      bounded(tail,Nat.add(Nat.mul(prefix,U32.to_nat(extent)),coordinate),Nat.mul(capacity,U32.to_nat(extent)),RowMajor.radix(prefix,capacity,coordinate,U32.to_nat(extent),room,inside),tail_valid)

def inside(+points: List<&2,Point>,valid: Valid(points)) -> {Nat.is_lt(position(points,0n),ShapeLaws.elements(Views.axis_extents(axes(points)))) == True{} : Bool}:
  %Addition.zero_right(ShapeLaws.elements(Views.axis_extents(axes(points)))) : {Nat.is_lt(position(points,0n),_) == True{} : Bool}
  %count_exact(points,1n) : {Nat.is_lt(position(points,0n),_) == True{} : Bool}
  bounded(points,0n,1n,{==},valid)

# Sum the digits' stride contributions in the same order as the reversed
# cursor axes. A unit axis contributes zero in either direction.
def address(points: List<&2,Point>,prefix: Nat,+mirror: Bool) -> Nat:
  match points:
    case []: prefix
    case Point{extent,coordinate,stride,reverse} <> tail:
      address(tail,Nat.add(Nat.mul(Mirror.reflected_coordinate(Mirror.active(mirror,reverse),U32.to_nat(extent),coordinate),U32.to_nat(stride)),prefix),mirror)

def chosen_address(+extent: Nat,+coordinate: Nat,+stride: U32,+reverse: Bool,+mirror: Bool,+reversed: List<&2,Mirror.Axis>,+offset: Nat,
  inside: {Nat.is_lt(coordinate,extent) == True{} : Bool}) ->
  {Model.address(Bool.pick(List<&2,Model.Coordinate>,mirror,
      Mirror.reflected(Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,extent),Mirror.Axis{extent,coordinate,stride,reverse} <> reversed,reversed)),
      Mirror.modeled(False{},Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,extent),Mirror.Axis{extent,coordinate,stride,reverse} <> reversed,reversed))),offset) ==
    Nat.add(Nat.mul(Mirror.reflected_coordinate(Mirror.active(mirror,reverse),extent,coordinate),U32.to_nat(stride)),
      Model.address(Bool.pick(List<&2,Model.Coordinate>,mirror,Mirror.reflected(reversed),Mirror.modeled(False{},reversed)),offset)) : Nat}:
  match extent coordinate:
    case 0n coordinate:
      Empty.absurd({Model.address(Bool.pick(List<&2,Model.Coordinate>,mirror,Mirror.reflected(reversed),Mirror.modeled(False{},reversed)),offset) == Nat.add(Nat.mul(Mirror.reflected_coordinate(Mirror.active(mirror,reverse),0n,coordinate),U32.to_nat(stride)),Model.address(Bool.pick(List<&2,Model.Coordinate>,mirror,Mirror.reflected(reversed),Mirror.modeled(False{},reversed)),offset)) : Nat},Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(coordinate,0n),True{},Equal.sym(Bool,Nat.is_lt(coordinate,0n),False{},Division.lt_zero(coordinate)),inside)))
    case 1n 0n:
      match reverse mirror:
        case False{} False{}: {==}
        case False{} True{}: {==}
        case True{} False{}: {==}
        case True{} True{}: {==}
    case 1n 1n+ +coordinate:
      Empty.absurd({Model.address(Bool.pick(List<&2,Model.Coordinate>,mirror,Mirror.reflected(reversed),Mirror.modeled(False{},reversed)),offset) == Nat.add(Nat.mul(Mirror.reflected_coordinate(Mirror.active(mirror,reverse),1n,1n+coordinate),U32.to_nat(stride)),Model.address(Bool.pick(List<&2,Model.Coordinate>,mirror,Mirror.reflected(reversed),Mirror.modeled(False{},reversed)),offset)) : Nat},Words.false_true(Equal.trans(Bool,False{},Nat.is_lt(coordinate,0n),True{},Equal.sym(Bool,Nat.is_lt(coordinate,0n),False{},Division.lt_zero(coordinate)),inside)))
    case 2n+rest coordinate:
      match mirror:
        case False{}: {==}
        case True{}: {==}

def address_exact(+points: List<&2,Point>,+reversed: List<&2,Mirror.Axis>,+offset: Nat,+mirror: Bool,valid: Valid(points)) ->
  {Model.address(Bool.pick(List<&2,Model.Coordinate>,mirror,Mirror.reflected(placed(points,reversed)),Mirror.modeled(False{},placed(points,reversed))),offset) ==
    address(points,Model.address(Bool.pick(List<&2,Model.Coordinate>,mirror,Mirror.reflected(reversed),Mirror.modeled(False{},reversed)),offset),mirror) : Nat}:
  match points:
    case []: {==}
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      KernelGemm.Both{+inside,tail_valid} = valid
      %chosen_address(U32.to_nat(extent),coordinate,stride,reverse,mirror,reversed,offset,inside) :
        {Model.address(Bool.pick(List<&2,Model.Coordinate>,mirror,
          Mirror.reflected(placed(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed))),
          Mirror.modeled(False{},placed(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed)))),offset) == address(tail,_,mirror) : Nat}
      address_exact(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed),offset,mirror,tail_valid)

def coordinates(+points: List<&2,Point>,+valid: Valid(points),
  +inside: {Nat.is_lt(position(points,0n),ShapeLaws.elements(Views.axis_extents(axes(points)))) == True{} : Bool}) ->
  {Order.iterate(position(points,0n),Model.prepare_axes(axes(points),[])) == Mirror.modeled(False{},placed(points,[])) : List<&2,Model.Coordinate>}:
  Equal.trans(List<&2,Model.Coordinate>,Order.iterate(position(points,0n),Model.prepare_axes(axes(points),[])),Mirror.modeled(False{},Mirror.iterate(position(points,0n),Mirror.build(axes(points),flags(points),[]))),Mirror.modeled(False{},placed(points,[])),
    Equal.trans(List<&2,Model.Coordinate>,Order.iterate(position(points,0n),Model.prepare_axes(axes(points),[])),Order.iterate(position(points,0n),Mirror.modeled(False{},Mirror.build(axes(points),flags(points),[]))),Mirror.modeled(False{},Mirror.iterate(position(points,0n),Mirror.build(axes(points),flags(points),[]))),
      Equal.cong(List<&2,Model.Coordinate>,List<&2,Model.Coordinate>,value => Order.iterate(position(points,0n),value),Model.prepare_axes(axes(points),[]),Mirror.modeled(False{},Mirror.build(axes(points),flags(points),[])),Mirror.built_forward(axes(points),flags(points),[])),Mirror.modeled_iterate(False{},position(points,0n),Mirror.build(axes(points),flags(points),[]))),
    Equal.cong(List<&2,Mirror.Axis>,List<&2,Model.Coordinate>,Mirror.modeled(False{}),Mirror.iterate(position(points,0n),Mirror.build(axes(points),flags(points),[])),placed(points,[]),reached(points,valid,inside)))

def copied(+points: List<&2,Point>,+offset: U32,+valid: Valid(points),
  inside: {Nat.is_lt(position(points,0n),ShapeLaws.elements(Views.axis_extents(axes(points)))) == True{} : Bool}) ->
  {Copy.address(Views.View{axes(points),offset},position(points,0n)) == U32.from_nat(address(points,U32.to_nat(offset),False{})) : U32}:
  %Equal.sym(List<&2,Model.Coordinate>,Order.iterate(position(points,0n),Model.prepare_axes(axes(points),[])),Mirror.modeled(False{},placed(points,[])),coordinates(points,valid,inside)) :
    {U32.from_nat(Model.address(_,U32.to_nat(offset))) == U32.from_nat(address(points,U32.to_nat(offset),False{})) : U32}
  Equal.cong(Nat,U32,U32.from_nat,Model.address(Mirror.modeled(False{},placed(points,[])),U32.to_nat(offset)),address(points,U32.to_nat(offset),False{}),address_exact(points,[],U32.to_nat(offset),False{},valid))

# The explicit coordinates of a canonical view have their row-major address.
def canonical(+points: List<&2,Point>,+valid: Valid(points),layout: Linear.Layout(axes(points)),
  +inside: {Nat.is_lt(position(points,0n),ShapeLaws.elements(Views.axis_extents(axes(points)))) == True{} : Bool}) ->
  {address(points,0n,False{}) == position(points,0n) : Nat}:
  +start = Model.prepare_axes(axes(points),[])
  +candidate = Mirror.modeled(False{},placed(points,[]))
  +nonempty = CountCompletion.positive_not_zero(Views.axis_extents(axes(points)),NatOrder.le_lt_trans(0n,position(points,0n),ShapeLaws.elements(Views.axis_extents(axes(points))),NatOrder.zero_le(position(points,0n)),inside))
  +placed_coordinates = coordinates(points,valid,inside)
  Equal.trans(Nat,address(points,0n,False{}),Model.address(candidate,0n),position(points,0n),
    Equal.sym(Nat,Model.address(candidate,0n),address(points,0n,False{}),address_exact(points,[],0n,False{},valid)),
    Equal.trans(Nat,Model.address(candidate,0n),Nat.add(Model.position(candidate),0n),position(points,0n),
      Linear.linear_address(candidate,0n,1n,
        %placed_coordinates : Linear.Linear(_,1n)
        Order.iterate_linear(position(points,0n),start,1n,Linear.prepared_linear(axes(points),[],nonempty,layout,Unit{}))),
      Equal.trans(Nat,Nat.add(Model.position(candidate),0n),Model.position(candidate),position(points,0n),Addition.zero_right(Model.position(candidate)),position_exact(points,[],valid))))

# ---- The digits of a position ----

# The row-major digits of a position over the axes of a view: each axis takes
# the quotient by the elements of the axes after it.
def points_of(view_axes: List<&2,Views.Axis>,+value: Nat) -> List<&2,Point>:
  match view_axes:
    case []: []
    case Views.Axis{extent,stride} <> +tail:
      Point{extent,Nat.div(value,ShapeLaws.elements(Views.axis_extents(tail))),stride,False{}} <> points_of(tail,Nat.mod(value,ShapeLaws.elements(Views.axis_extents(tail))))

def axes_of(+view_axes: List<&2,Views.Axis>,+value: Nat) -> {axes(points_of(view_axes,value)) == view_axes : List<&2,Views.Axis>}:
  match view_axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      +rest = Nat.mod(value,ShapeLaws.elements(Views.axis_extents(tail)))
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => Views.Axis{extent,stride} <> others,axes(points_of(tail,rest)),tail,axes_of(tail,rest))

# A prefix, a quotient by rest and a remainder, joined.
def digit_join(+prefix: Nat,+extent: Nat,+quotient: Nat,+rest: Nat,+remainder: Nat) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,extent),quotient),rest),remainder) == Nat.add(Nat.mul(prefix,Nat.mul(extent,rest)),Nat.add(Nat.mul(quotient,rest),remainder)) : Nat}:
  +scaled = Nat.mul(prefix,extent)
  %RowMajor.assoc(prefix,extent,rest) :
    {Nat.add(Nat.mul(Nat.add(scaled,quotient),rest),remainder) == Nat.add(_,Nat.add(Nat.mul(quotient,rest),remainder)) : Nat}
  %Equal.sym(Nat,Nat.mul(Nat.add(scaled,quotient),rest),Nat.add(Nat.mul(scaled,rest),Nat.mul(quotient,rest)),RowMajor.distrib(scaled,quotient,rest)) :
    {Nat.add(_,remainder) == Nat.add(Nat.mul(scaled,rest),Nat.add(Nat.mul(quotient,rest),remainder)) : Nat}
  Addition.associative(Nat.mul(scaled,rest),Nat.mul(quotient,rest),remainder)

# The digits of a position below the elements of the axes have that position
# after any prefix.
def position_of(+view_axes: List<&2,Views.Axis>,+value: Nat,+prefix: Nat,inside: {Nat.is_lt(value,ShapeLaws.elements(Views.axis_extents(view_axes))) == True{} : Bool}) ->
  {position(points_of(view_axes,value),prefix) == Nat.add(Nat.mul(prefix,ShapeLaws.elements(Views.axis_extents(view_axes))),value) : Nat}:
  match view_axes:
    case []:
      %Equal.sym(Nat,value,0n,zero_below_one(value,inside)) : {prefix == Nat.add(Nat.mul(prefix,1n),_) : Nat}
      %Equal.sym(Nat,Nat.add(Nat.mul(prefix,1n),0n),Nat.mul(prefix,1n),Addition.zero_right(Nat.mul(prefix,1n))) : {prefix == _ : Nat}
      Equal.sym(Nat,Nat.mul(prefix,1n),prefix,NatOrder.times_one(prefix))
    case Views.Axis{+extent,+stride} <> +tail:
      +rest = ShapeLaws.elements(Views.axis_extents(tail))
      +size = U32.to_nat(extent)
      +positive = DivisionOrder.right_factor_positive(size,rest,NatOrder.le_lt_trans(0n,value,Nat.mul(size,rest),NatOrder.zero_le(value),inside))
      +quotient = Nat.div(value,rest)
      +remainder = Nat.mod(value,rest)
      %Equal.sym(Nat,value,Nat.add(Nat.mul(quotient,rest),remainder),DivisionOrder.quotient_exact(value,rest,positive)) :
        {position(points_of(tail,remainder),Nat.add(Nat.mul(prefix,size),quotient)) == Nat.add(Nat.mul(prefix,Nat.mul(size,rest)),_) : Nat}
      Equal.trans(Nat,position(points_of(tail,remainder),Nat.add(Nat.mul(prefix,size),quotient)),Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,size),quotient),rest),remainder),
        Nat.add(Nat.mul(prefix,Nat.mul(size,rest)),Nat.add(Nat.mul(quotient,rest),remainder)),
        position_of(tail,remainder,Nat.add(Nat.mul(prefix,size),quotient),DivisionOrder.remainder_small(value,rest,positive)),digit_join(prefix,size,quotient,rest,remainder))

def valid_of(+view_axes: List<&2,Views.Axis>,+value: Nat,+inside: {Nat.is_lt(value,ShapeLaws.elements(Views.axis_extents(view_axes))) == True{} : Bool}) ->
  Valid(points_of(view_axes,value)):
  match view_axes:
    case []: Unit{}
    case Views.Axis{+extent,+stride} <> +tail:
      +rest = ShapeLaws.elements(Views.axis_extents(tail))
      +size = U32.to_nat(extent)
      +positive = DivisionOrder.right_factor_positive(size,rest,NatOrder.le_lt_trans(0n,value,Nat.mul(size,rest),NatOrder.zero_le(value),inside))
      KernelGemm.Both{DivisionOrder.quotient_below(value,size,rest,positive,inside),valid_of(tail,Nat.mod(value,rest),DivisionOrder.remainder_small(value,rest,positive))}

# The digits times their strides.
def weight(points: List<&2,Point>) -> Nat:
  match points:
    case []: 0n
    case Point{extent,coordinate,stride,reverse} <> tail: Nat.add(Nat.mul(coordinate,U32.to_nat(stride)),weight(tail))

def address_weight(+points: List<&2,Point>,+prefix: Nat) -> {address(points,prefix,False{}) == Nat.add(weight(points),prefix) : Nat}:
  match points:
    case []: {==}
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      +own = Nat.mul(coordinate,U32.to_nat(stride))
      %NatOrder.commutative(weight(tail),own) : {address(tail,Nat.add(own,prefix),False{}) == Nat.add(_,prefix) : Nat}
      Equal.trans(Nat,address(tail,Nat.add(own,prefix),False{}),Nat.add(weight(tail),Nat.add(own,prefix)),Nat.add(Nat.add(weight(tail),own),prefix),
        address_weight(tail,Nat.add(own,prefix)),Equal.sym(Nat,Nat.add(Nat.add(weight(tail),own),prefix),Nat.add(weight(tail),Nat.add(own,prefix)),Addition.associative(weight(tail),own,prefix)))

def weighed(+points: List<&2,Point>,+offset: U32,+valid: Valid(points)) ->
  {Copy.address(Views.View{axes(points),offset},position(points,0n)) == U32.from_nat(Nat.add(weight(points),U32.to_nat(offset))) : U32}:
  Equal.trans(U32,Copy.address(Views.View{axes(points),offset},position(points,0n)),U32.from_nat(address(points,U32.to_nat(offset),False{})),U32.from_nat(Nat.add(weight(points),U32.to_nat(offset))),
    copied(points,offset,valid,inside(points,valid)),Equal.cong(Nat,U32,U32.from_nat,address(points,U32.to_nat(offset),False{}),Nat.add(weight(points),U32.to_nat(offset)),address_weight(points,U32.to_nat(offset))))

# ---- Digits read in reverse ----
