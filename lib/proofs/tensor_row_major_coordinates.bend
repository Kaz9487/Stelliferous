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
import ./storage_shape.bend as Shape
import ./word_encoding.bend as Words
import ./boolean_logic.bend as Boolean

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
      Shape.Both<{Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool},Valid(tail)>

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
      Shape.Both{+inside,tail_valid} = valid
      %chosen_position(U32.to_nat(extent),coordinate,stride,reverse,reversed,inside) :
        {Model.position(Mirror.modeled(False{},placed(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed)))) == position(tail,_) : Nat}
      position_exact(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed),tail_valid)

def chosen_valid(+selected: Bool,+extent: Nat,+coordinate: Nat,+stride: U32,+reverse: Bool,+reversed: List<&2,Mirror.Axis>,
  inside: {Nat.is_lt(coordinate,extent) == True{} : Bool},valid: Cursor.Valid(Mirror.modeled(False{},reversed))) ->
  Cursor.Valid(Mirror.modeled(False{},Bool.pick(List<&2,Mirror.Axis>,selected,Mirror.Axis{extent,coordinate,stride,reverse} <> reversed,reversed))):
  match selected:
    case False{}: valid
    case True{}: Shape.Both{inside,valid}

def placed_valid(+points: List<&2,Point>,+reversed: List<&2,Mirror.Axis>,valid: Valid(points),initial: Cursor.Valid(Mirror.modeled(False{},reversed))) ->
  Cursor.Valid(Mirror.modeled(False{},placed(points,reversed))):
  match points:
    case []: initial
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      Shape.Both{inside,tail_valid} = valid
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

def mirror_address(+points: List<&2,Point>,+offset: U32,valid: Valid(points),
  inside: {Nat.is_lt(position(points,0n),ShapeLaws.elements(Views.axis_extents(axes(points)))) == True{} : Bool}) ->
  {Mirror.address(Views.View{axes(points),offset},flags(points),position(points,0n)) == U32.from_nat(Model.address(Mirror.reflected(placed(points,[])),U32.to_nat(offset))) : U32}:
  Equal.cong(List<&2,Mirror.Axis>,U32,value => U32.from_nat(Model.address(Mirror.reflected(value),U32.to_nat(offset))),
    Mirror.iterate(position(points,0n),Mirror.build(axes(points),flags(points),[])),placed(points,[]),reached(points,valid,inside))

# Valid digits imply a bounded row-major index.
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
      Shape.Both{inside,tail_valid} = valid
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
      Shape.Both{+inside,tail_valid} = valid
      %chosen_address(U32.to_nat(extent),coordinate,stride,reverse,mirror,reversed,offset,inside) :
        {Model.address(Bool.pick(List<&2,Model.Coordinate>,mirror,
          Mirror.reflected(placed(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed))),
          Mirror.modeled(False{},placed(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed)))),offset) == address(tail,_,mirror) : Nat}
      address_exact(tail,Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),coordinate,stride,reverse} <> reversed,reversed),offset,mirror,tail_valid)

def mirrored(+points: List<&2,Point>,+offset: U32,+valid: Valid(points),
  inside: {Nat.is_lt(position(points,0n),ShapeLaws.elements(Views.axis_extents(axes(points)))) == True{} : Bool}) ->
  {Mirror.address(Views.View{axes(points),offset},flags(points),position(points,0n)) == U32.from_nat(address(points,U32.to_nat(offset),True{})) : U32}:
  Equal.trans(U32,Mirror.address(Views.View{axes(points),offset},flags(points),position(points,0n)),
    U32.from_nat(Model.address(Mirror.reflected(placed(points,[])),U32.to_nat(offset))),U32.from_nat(address(points,U32.to_nat(offset),True{})),
    mirror_address(points,offset,valid,inside),Equal.cong(Nat,U32,U32.from_nat,Model.address(Mirror.reflected(placed(points,[])),U32.to_nat(offset)),address(points,U32.to_nat(offset),True{}),address_exact(points,[],U32.to_nat(offset),True{},valid)))

# Copy and mirror share the same coordinate recovery.
# One coordinate chain shared by copied and canonical addresses.
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
      Shape.Both{DivisionOrder.quotient_below(value,size,rest,positive,inside),valid_of(tail,Nat.mod(value,rest),DivisionOrder.remainder_small(value,rest,positive))}

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

# Digits one list after another.
def position_append(+left: List<&2,Point>,+right: List<&2,Point>,+prefix: Nat) ->
  {position(List.append(&2,Point,left,right),prefix) == position(right,position(left,prefix)) : Nat}:
  match left:
    case []: {==}
    case Point{+extent,+coordinate,stride,reverse} <> +tail: position_append(tail,right,Nat.add(Nat.mul(prefix,U32.to_nat(extent)),coordinate))

def weight_append(+left: List<&2,Point>,+right: List<&2,Point>) -> {weight(List.append(&2,Point,left,right)) == Nat.add(weight(left),weight(right)) : Nat}:
  match left:
    case []: {==}
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      +own = Nat.mul(coordinate,U32.to_nat(stride))
      Equal.trans(Nat,Nat.add(own,weight(List.append(&2,Point,tail,right))),Nat.add(own,Nat.add(weight(tail),weight(right))),Nat.add(Nat.add(own,weight(tail)),weight(right)),
        Equal.cong(Nat,Nat,sum => Nat.add(own,sum),weight(List.append(&2,Point,tail,right)),Nat.add(weight(tail),weight(right)),weight_append(tail,right)),
        Equal.sym(Nat,Nat.add(Nat.add(own,weight(tail)),weight(right)),Nat.add(own,Nat.add(weight(tail),weight(right))),Addition.associative(own,weight(tail),weight(right))))

def valid_append(+left: List<&2,Point>,+right: List<&2,Point>,valid_left: Valid(left),valid_right: Valid(right)) -> Valid(List.append(&2,Point,left,right)):
  match left:
    case []: valid_right
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      Shape.Both{inside,tail_valid} = valid_left
      Shape.Both{inside,valid_append(tail,right,tail_valid,valid_right)}

def axes_append(+left: List<&2,Point>,+right: List<&2,Point>) -> {axes(List.append(&2,Point,left,right)) == List.append(&2,Views.Axis,axes(left),axes(right)) : List<&2,Views.Axis>}:
  match left:
    case []: {==}
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => Views.Axis{extent,stride} <> others,axes(List.append(&2,Point,tail,right)),
        List.append(&2,Views.Axis,axes(tail),axes(right)),axes_append(tail,right))

# A view of the axes of digits, read at their position, gives the offset and
# their weight.
def weighed(+points: List<&2,Point>,+offset: U32,+valid: Valid(points)) ->
  {Copy.address(Views.View{axes(points),offset},position(points,0n)) == U32.from_nat(Nat.add(weight(points),U32.to_nat(offset))) : U32}:
  Equal.trans(U32,Copy.address(Views.View{axes(points),offset},position(points,0n)),U32.from_nat(address(points,U32.to_nat(offset),False{})),U32.from_nat(Nat.add(weight(points),U32.to_nat(offset))),
    copied(points,offset,valid,inside(points,valid)),Equal.cong(Nat,U32,U32.from_nat,address(points,U32.to_nat(offset),False{}),Nat.add(weight(points),U32.to_nat(offset)),address_weight(points,U32.to_nat(offset))))

# ---- Digits read in reverse ----

# The digits with the flags of their axes, one flag a digit; a digit without
# a flag is read forward.
def attached(points: List<&2,Point>,+flagged: List<&2,Bool>) -> List<&2,Point>:
  match points:
    case []: []
    case Point{extent,coordinate,stride,reverse} <> tail: Point{extent,coordinate,stride,Views.flag(flagged)} <> attached(tail,Views.later_flags(flagged))

# Every digit read in reverse reflected, then read forward.
def reflect(points: List<&2,Point>) -> List<&2,Point>:
  match points:
    case []: []
    case Point{+extent,coordinate,stride,reverse} <> tail:
      Point{extent,Mirror.reflected_coordinate(reverse,U32.to_nat(extent),coordinate),stride,False{}} <> reflect(tail)

def axes_attached(+points: List<&2,Point>,+flagged: List<&2,Bool>) -> {axes(attached(points,flagged)) == axes(points) : List<&2,Views.Axis>}:
  match points:
    case []: {==}
    case Point{+extent,coordinate,+stride,reverse} <> +tail:
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => Views.Axis{extent,stride} <> others,axes(attached(tail,Views.later_flags(flagged))),axes(tail),
        axes_attached(tail,Views.later_flags(flagged)))

def axes_reflect(+points: List<&2,Point>) -> {axes(reflect(points)) == axes(points) : List<&2,Views.Axis>}:
  match points:
    case []: {==}
    case Point{+extent,coordinate,+stride,reverse} <> +tail:
      Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => Views.Axis{extent,stride} <> others,axes(reflect(tail)),axes(tail),axes_reflect(tail))

def position_attached(+points: List<&2,Point>,+flagged: List<&2,Bool>,+prefix: Nat) -> {position(attached(points,flagged),prefix) == position(points,prefix) : Nat}:
  match points:
    case []: {==}
    case Point{+extent,+coordinate,stride,reverse} <> +tail:
      position_attached(tail,Views.later_flags(flagged),Nat.add(Nat.mul(prefix,U32.to_nat(extent)),coordinate))

def valid_attached(+points: List<&2,Point>,+flagged: List<&2,Bool>,valid: Valid(points)) -> Valid(attached(points,flagged)):
  match points:
    case []: Unit{}
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      Shape.Both{inside,rest} = valid
      Shape.Both{inside,valid_attached(tail,Views.later_flags(flagged),rest)}

def valid_reflect(+points: List<&2,Point>,valid: Valid(points)) -> Valid(reflect(points)):
  match points:
    case []: Unit{}
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      Shape.Both{inside,rest} = valid
      Shape.Both{Mirror.reflected_inside(reverse,U32.to_nat(extent),coordinate,inside),valid_reflect(tail,rest)}

# Reading a digit in reverse is reading its reflection forward.
def address_reflect(+points: List<&2,Point>,+prefix: Nat) -> {address(points,prefix,True{}) == address(reflect(points),prefix,False{}) : Nat}:
  match points:
    case []: {==}
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      address_reflect(tail,Nat.add(Nat.mul(Mirror.reflected_coordinate(reverse,U32.to_nat(extent),coordinate),U32.to_nat(stride)),prefix))

# The flags of attached digits build the cursor axes the flags build.
def build_attached(+points: List<&2,Point>,+flagged: List<&2,Bool>,+reversed: List<&2,Mirror.Axis>) ->
  {Mirror.build(axes(points),flags(attached(points,flagged)),reversed) == Mirror.build(axes(points),flagged,reversed) : List<&2,Mirror.Axis>}:
  match points:
    case []: {==}
    case Point{+extent,coordinate,+stride,reverse} <> +tail:
      build_attached(tail,Views.later_flags(flagged),
        Bool.pick(List<&2,Mirror.Axis>,Nat.is_lt(1n,U32.to_nat(extent)),Mirror.Axis{U32.to_nat(extent),0n,stride,Views.flag(flagged)} <> reversed,reversed))

def address_attached(+points: List<&2,Point>,+offset: U32,+flagged: List<&2,Bool>,+index: Nat) ->
  {Mirror.address(Views.View{axes(points),offset},flags(attached(points,flagged)),index) == Mirror.address(Views.View{axes(points),offset},flagged,index) : U32}:
  Equal.cong(List<&2,Mirror.Axis>,U32,built => U32.from_nat(Model.address(Mirror.reflected(Mirror.iterate(index,built)),U32.to_nat(offset))),
    Mirror.build(axes(points),flags(attached(points,flagged)),[]),Mirror.build(axes(points),flagged,[]),build_attached(points,flagged,[]))

# Valid digits read in reverse where flagged: the mirrored view gives their
# position the address the view gives the position of their reflection.
def reflected_address(+points: List<&2,Point>,+offset: U32,+valid: Valid(points)) ->
  {Mirror.address(Views.View{axes(points),offset},flags(points),position(points,0n)) == Copy.address(Views.View{axes(points),offset},position(reflect(points),0n)) : U32}:
  +turned = reflect(points)
  +turned_valid = valid_reflect(points,valid)
  Equal.trans(U32,Mirror.address(Views.View{axes(points),offset},flags(points),position(points,0n)),U32.from_nat(address(points,U32.to_nat(offset),True{})),
    Copy.address(Views.View{axes(points),offset},position(turned,0n)),
    mirrored(points,offset,valid,inside(points,valid)),
    Equal.trans(U32,U32.from_nat(address(points,U32.to_nat(offset),True{})),U32.from_nat(address(turned,U32.to_nat(offset),False{})),
      Copy.address(Views.View{axes(points),offset},position(turned,0n)),
      Equal.cong(Nat,U32,U32.from_nat,address(points,U32.to_nat(offset),True{}),address(turned,U32.to_nat(offset),False{}),address_reflect(points,U32.to_nat(offset))),
      %axes_reflect(points) : {U32.from_nat(address(turned,U32.to_nat(offset),False{})) == Copy.address(Views.View{_,offset},position(turned,0n)) : U32}
      Equal.sym(U32,Copy.address(Views.View{axes(turned),offset},position(turned,0n)),U32.from_nat(address(turned,U32.to_nat(offset),False{})),
        copied(turned,offset,turned_valid,inside(turned,turned_valid)))))

# Forward digits read with flags.
def directed_address(+points: List<&2,Point>,+offset: U32,+flagged: List<&2,Bool>,+valid: Valid(points)) ->
  {Mirror.address(Views.View{axes(points),offset},flagged,position(points,0n)) == Copy.address(Views.View{axes(points),offset},position(reflect(attached(points,flagged)),0n)) : U32}:
  +marked = attached(points,flagged)
  %axes_attached(points,flagged) :
    {Mirror.address(Views.View{axes(points),offset},flagged,position(points,0n)) == Copy.address(Views.View{_,offset},position(reflect(marked),0n)) : U32}
  %position_attached(points,flagged,0n) :
    {Mirror.address(Views.View{axes(points),offset},flagged,_) == Copy.address(Views.View{axes(marked),offset},position(reflect(marked),0n)) : U32}
  %address_attached(points,offset,flagged,position(marked,0n)) :
    {_ == Copy.address(Views.View{axes(marked),offset},position(reflect(marked),0n)) : U32}
  %axes_attached(points,flagged) :
    {Mirror.address(Views.View{_,offset},flags(marked),position(marked,0n)) == Copy.address(Views.View{axes(marked),offset},position(reflect(marked),0n)) : U32}
  reflected_address(marked,offset,valid_attached(points,flagged,valid))

# The position after a prefix: the prefix times the elements of the axes, plus
# the position.
def position_prefix(+points: List<&2,Point>,+prefix: Nat) ->
  {position(points,prefix) == Nat.add(Nat.mul(prefix,ShapeLaws.elements(Views.axis_extents(axes(points)))),position(points,0n)) : Nat}:
  match points:
    case []:
      %Equal.sym(Nat,Nat.add(Nat.mul(prefix,1n),0n),Nat.mul(prefix,1n),Addition.zero_right(Nat.mul(prefix,1n))) : {prefix == _ : Nat}
      Equal.sym(Nat,Nat.mul(prefix,1n),prefix,NatOrder.times_one(prefix))
    case Point{+extent,+coordinate,stride,reverse} <> +tail:
      +rest = ShapeLaws.elements(Views.axis_extents(axes(tail)))
      +size = U32.to_nat(extent)
      Equal.trans(Nat,position(tail,Nat.add(Nat.mul(prefix,size),coordinate)),Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,size),coordinate),rest),position(tail,0n)),
        Nat.add(Nat.mul(prefix,Nat.mul(size,rest)),position(tail,coordinate)),
        position_prefix(tail,Nat.add(Nat.mul(prefix,size),coordinate)),
        %Equal.sym(Nat,position(tail,coordinate),Nat.add(Nat.mul(coordinate,rest),position(tail,0n)),position_prefix(tail,coordinate)) :
          {Nat.add(Nat.mul(Nat.add(Nat.mul(prefix,size),coordinate),rest),position(tail,0n)) == Nat.add(Nat.mul(prefix,Nat.mul(size,rest)),_) : Nat}
        digit_join(prefix,size,coordinate,rest,position(tail,0n)))

# The row-major position of the digits of value over extents, with the digit
# of every flagged axis reflected.
def reflected(extents: List<&2,U32>,+flagged: List<&2,Bool>,+value: Nat) -> Nat:
  match extents:
    case []: 0n
    case +extent <> +tail:
      +rest = ShapeLaws.elements(tail)
      Nat.add(Nat.mul(Mirror.reflected_coordinate(Views.flag(flagged),U32.to_nat(extent),Nat.div(value,rest)),rest),reflected(tail,Views.later_flags(flagged),Nat.mod(value,rest)))

def axes_points_of(+view_axes: List<&2,Views.Axis>,+flagged: List<&2,Bool>,+value: Nat) ->
  {Views.axis_extents(axes(reflect(attached(points_of(view_axes,value),flagged)))) == Views.axis_extents(view_axes) : List<&2,U32>}:
  +marked = attached(points_of(view_axes,value),flagged)
  Equal.cong(List<&2,Views.Axis>,List<&2,U32>,Views.axis_extents,axes(reflect(marked)),view_axes,
    Equal.trans(List<&2,Views.Axis>,axes(reflect(marked)),axes(marked),view_axes,axes_reflect(marked),
      Equal.trans(List<&2,Views.Axis>,axes(marked),axes(points_of(view_axes,value)),view_axes,axes_attached(points_of(view_axes,value),flagged),axes_of(view_axes,value))))

def reflected_exact(+view_axes: List<&2,Views.Axis>,+flagged: List<&2,Bool>,+value: Nat) ->
  {position(reflect(attached(points_of(view_axes,value),flagged)),0n) == reflected(Views.axis_extents(view_axes),flagged,value) : Nat}:
  match view_axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      +rest = ShapeLaws.elements(Views.axis_extents(tail))
      +later = Views.later_flags(flagged)
      +remainder = Nat.mod(value,rest)
      +turned = Mirror.reflected_coordinate(Views.flag(flagged),U32.to_nat(extent),Nat.div(value,rest))
      +below = reflect(attached(points_of(tail,remainder),later))
      Equal.trans(Nat,position(below,turned),Nat.add(Nat.mul(turned,ShapeLaws.elements(Views.axis_extents(axes(below)))),position(below,0n)),
        Nat.add(Nat.mul(turned,rest),reflected(Views.axis_extents(tail),later,remainder)),
        position_prefix(below,turned),
        %Equal.sym(List<&2,U32>,Views.axis_extents(axes(below)),Views.axis_extents(tail),axes_points_of(tail,later,remainder)) :
          {Nat.add(Nat.mul(turned,ShapeLaws.elements(_)),position(below,0n)) == Nat.add(Nat.mul(turned,rest),reflected(Views.axis_extents(tail),later,remainder)) : Nat}
        Equal.cong(Nat,Nat,tail_position => Nat.add(Nat.mul(turned,rest),tail_position),position(below,0n),reflected(Views.axis_extents(tail),later,remainder),
          reflected_exact(tail,later,remainder)))

# A reflected position lies among the elements.
def reflected_below(+view_axes: List<&2,Views.Axis>,+flagged: List<&2,Bool>,+value: Nat,
  +below: {Nat.is_lt(value,ShapeLaws.elements(Views.axis_extents(view_axes))) == True{} : Bool}) ->
  {Nat.is_lt(reflected(Views.axis_extents(view_axes),flagged,value),ShapeLaws.elements(Views.axis_extents(view_axes))) == True{} : Bool}:
  +marked = attached(points_of(view_axes,value),flagged)
  +turned = reflect(marked)
  %reflected_exact(view_axes,flagged,value) : {Nat.is_lt(_,ShapeLaws.elements(Views.axis_extents(view_axes))) == True{} : Bool}
  %axes_points_of(view_axes,flagged,value) : {Nat.is_lt(position(turned,0n),ShapeLaws.elements(_)) == True{} : Bool}
  inside(turned,valid_reflect(marked,valid_attached(points_of(view_axes,value),flagged,valid_of(view_axes,value,below))))

# With no flag set among the axes, the reflected position is the position.
def reflected_forward(+extents: List<&2,U32>,+flagged: List<&2,Bool>,+value: Nat,
  +forward: {Views.forward(List.length(&2,U32,extents),flagged) == True{} : Bool},
  +inside: {Nat.is_lt(value,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  {reflected(extents,flagged,value) == value : Nat}:
  match extents:
    case []: Equal.sym(Nat,value,0n,zero_below_one(value,inside))
    case +extent <> +tail:
      +rest = ShapeLaws.elements(tail)
      +size = U32.to_nat(extent)
      +later = Views.later_flags(flagged)
      +positive = DivisionOrder.right_factor_positive(size,rest,NatOrder.le_lt_trans(0n,value,Nat.mul(size,rest),NatOrder.zero_le(value),inside))
      +unflagged = Boolean.not_true(Views.flag(flagged),Boolean.left(Bool.not(Views.flag(flagged)),Views.forward(List.length(&2,U32,tail),later),forward))
      %Equal.sym(Bool,Views.flag(flagged),False{},unflagged) :
        {Nat.add(Nat.mul(Mirror.reflected_coordinate(_,size,Nat.div(value,rest)),rest),reflected(tail,later,Nat.mod(value,rest))) == value : Nat}
      %Equal.sym(Nat,reflected(tail,later,Nat.mod(value,rest)),Nat.mod(value,rest),
          reflected_forward(tail,later,Nat.mod(value,rest),Boolean.right(Bool.not(Views.flag(flagged)),Views.forward(List.length(&2,U32,tail),later),forward),
            DivisionOrder.remainder_small(value,rest,positive))) :
        {Nat.add(Nat.mul(Nat.div(value,rest),rest),_) == value : Nat}
      Equal.sym(Nat,value,Nat.add(Nat.mul(Nat.div(value,rest),rest),Nat.mod(value,rest)),DivisionOrder.quotient_exact(value,rest,positive))

# Only the first flag of every axis counts.
def reflected_prefix(+extents: List<&2,U32>,+flagged: List<&2,Bool>,+rest_flags: List<&2,Bool>,+value: Nat) ->
  {reflected(extents,List.append(&2,Bool,Views.fitted_flags(List.length(&2,U32,extents),flagged),rest_flags),value) == reflected(extents,flagged,value) : Nat}:
  match extents:
    case []: {==}
    case +extent <> +tail:
      +rest = ShapeLaws.elements(tail)
      Equal.cong(Nat,Nat,tail_position => Nat.add(Nat.mul(Mirror.reflected_coordinate(Views.flag(flagged),U32.to_nat(extent),Nat.div(value,rest)),rest),tail_position),
        reflected(tail,List.append(&2,Bool,Views.fitted_flags(List.length(&2,U32,tail),Views.later_flags(flagged)),rest_flags),Nat.mod(value,rest)),
        reflected(tail,Views.later_flags(flagged),Nat.mod(value,rest)),
        reflected_prefix(tail,Views.later_flags(flagged),rest_flags,Nat.mod(value,rest)))

def reflected_fitted(+extents: List<&2,U32>,+flagged: List<&2,Bool>,+value: Nat) ->
  {reflected(extents,Views.fitted_flags(List.length(&2,U32,extents),flagged),value) == reflected(extents,flagged,value) : Nat}:
  match extents:
    case []: {==}
    case +extent <> +tail:
      +rest = ShapeLaws.elements(tail)
      Equal.cong(Nat,Nat,tail_position => Nat.add(Nat.mul(Mirror.reflected_coordinate(Views.flag(flagged),U32.to_nat(extent),Nat.div(value,rest)),rest),tail_position),
        reflected(tail,Views.fitted_flags(List.length(&2,U32,tail),Views.later_flags(flagged)),Nat.mod(value,rest)),
        reflected(tail,Views.later_flags(flagged),Nat.mod(value,rest)),
        reflected_fitted(tail,Views.later_flags(flagged),Nat.mod(value,rest)))

# Dropping the fitted flags of the first axes leaves the flags after them.
def drop_fitted(+count: Nat,+flagged: List<&2,Bool>,+rest: List<&2,Bool>) ->
  {Views.drop_flags(count,List.append(&2,Bool,Views.fitted_flags(count,flagged),rest)) == rest : List<&2,Bool>}:
  match count:
    case 0n: {==}
    case 1n+ +tail: drop_fitted(tail,Views.later_flags(flagged),rest)

def drop_drop(+first: Nat,+second: Nat,+flagged: List<&2,Bool>) ->
  {Views.drop_flags(second,Views.drop_flags(first,flagged)) == Views.drop_flags(Nat.add(first,second),flagged) : List<&2,Bool>}:
  match first:
    case 0n: {==}
    case 1n+ +tail: drop_drop(tail,second,Views.later_flags(flagged))

# No flags read every axis forward.
def forward_none(+count: Nat) -> {Views.forward(count,[]) == True{} : Bool}:
  match count:
    case 0n: {==}
    case 1n+ +tail: forward_none(tail)

# Four digits joined, from equal digits and extents.
def digits_equal(+a: Nat,+a2: Nat,+n: Nat,+n2: Nat,+b: Nat,+b2: Nat,+m: Nat,+m2: Nat,+d: Nat,+d2: Nat,
  +same_a: {a == a2 : Nat},+same_n: {n == n2 : Nat},+same_b: {b == b2 : Nat},+same_m: {m == m2 : Nat},+same_d: {d == d2 : Nat}) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(a,n),b),m),d) == Nat.add(Nat.mul(Nat.add(Nat.mul(a2,n2),b2),m2),d2) : Nat}:
  %same_a : {Nat.add(Nat.mul(Nat.add(Nat.mul(a,n),b),m),d) == Nat.add(Nat.mul(Nat.add(Nat.mul(_,n2),b2),m2),d2) : Nat}
  %same_n : {Nat.add(Nat.mul(Nat.add(Nat.mul(a,n),b),m),d) == Nat.add(Nat.mul(Nat.add(Nat.mul(a,_),b2),m2),d2) : Nat}
  %same_b : {Nat.add(Nat.mul(Nat.add(Nat.mul(a,n),b),m),d) == Nat.add(Nat.mul(Nat.add(Nat.mul(a,n),_),m2),d2) : Nat}
  %same_m : {Nat.add(Nat.mul(Nat.add(Nat.mul(a,n),b),m),d) == Nat.add(Nat.mul(Nat.add(Nat.mul(a,n),b),_),d2) : Nat}
  %same_d : {Nat.add(Nat.mul(Nat.add(Nat.mul(a,n),b),m),d) == Nat.add(Nat.mul(Nat.add(Nat.mul(a,n),b),m),_) : Nat}
  {==}

# ---- Groups of axes read in reverse ----

def attached_points(+view_axes: List<&2,Views.Axis>,+value: Nat,+rest: List<&2,Point>,+flagged: List<&2,Bool>) ->
  {attached(List.append(&2,Point,points_of(view_axes,value),rest),flagged) ==
    List.append(&2,Point,attached(points_of(view_axes,value),flagged),attached(rest,Views.drop_flags(List.length(&2,Views.Axis,view_axes),flagged))) : List<&2,Point>}:
  match view_axes:
    case []: {==}
    case Views.Axis{+extent,+stride} <> +tail:
      +elements = ShapeLaws.elements(Views.axis_extents(tail))
      Equal.cong(List<&2,Point>,List<&2,Point>,others => Point{extent,Nat.div(value,elements),stride,Views.flag(flagged)} <> others,
        attached(List.append(&2,Point,points_of(tail,Nat.mod(value,elements)),rest),Views.later_flags(flagged)),
        List.append(&2,Point,attached(points_of(tail,Nat.mod(value,elements)),Views.later_flags(flagged)),attached(rest,Views.drop_flags(List.length(&2,Views.Axis,tail),Views.later_flags(flagged)))),
        attached_points(tail,Nat.mod(value,elements),rest,Views.later_flags(flagged)))

def reflect_append(+left: List<&2,Point>,+right: List<&2,Point>) -> {reflect(List.append(&2,Point,left,right)) == List.append(&2,Point,reflect(left),reflect(right)) : List<&2,Point>}:
  match left:
    case []: {==}
    case Point{+extent,+coordinate,+stride,+reverse} <> +tail:
      Equal.cong(List<&2,Point>,List<&2,Point>,others => Point{extent,Mirror.reflected_coordinate(reverse,U32.to_nat(extent),coordinate),stride,False{}} <> others,
        reflect(List.append(&2,Point,tail,right)),List.append(&2,Point,reflect(tail),reflect(right)),reflect_append(tail,right))

# One group of digits, reflected, after a prefix.
def group_reflected(+view_axes: List<&2,Views.Axis>,+flagged: List<&2,Bool>,+value: Nat,+prefix: Nat) ->
  {position(reflect(attached(points_of(view_axes,value),flagged)),prefix) ==
    Nat.add(Nat.mul(prefix,ShapeLaws.elements(Views.axis_extents(view_axes))),reflected(Views.axis_extents(view_axes),flagged,value)) : Nat}:
  +turned = reflect(attached(points_of(view_axes,value),flagged))
  %axes_points_of(view_axes,flagged,value) :
    {position(turned,prefix) == Nat.add(Nat.mul(prefix,ShapeLaws.elements(_)),reflected(Views.axis_extents(view_axes),flagged,value)) : Nat}
  %reflected_exact(view_axes,flagged,value) : {position(turned,prefix) == Nat.add(Nat.mul(prefix,ShapeLaws.elements(Views.axis_extents(axes(turned)))),_) : Nat}
  position_prefix(turned,prefix)

# One group of digits, reflected, before other digits.
def step_reflected(+view_axes: List<&2,Views.Axis>,+value: Nat,+rest: List<&2,Point>,+flagged: List<&2,Bool>,+prefix: Nat) ->
  {position(reflect(attached(List.append(&2,Point,points_of(view_axes,value),rest),flagged)),prefix) ==
    position(reflect(attached(rest,Views.drop_flags(List.length(&2,Views.Axis,view_axes),flagged))),
      Nat.add(Nat.mul(prefix,ShapeLaws.elements(Views.axis_extents(view_axes))),reflected(Views.axis_extents(view_axes),flagged,value))) : Nat}:
  +head = attached(points_of(view_axes,value),flagged)
  +tail = attached(rest,Views.drop_flags(List.length(&2,Views.Axis,view_axes),flagged))
  +start = Nat.add(Nat.mul(prefix,ShapeLaws.elements(Views.axis_extents(view_axes))),reflected(Views.axis_extents(view_axes),flagged,value))
  %Equal.sym(List<&2,Point>,attached(List.append(&2,Point,points_of(view_axes,value),rest),flagged),List.append(&2,Point,head,tail),attached_points(view_axes,value,rest,flagged)) :
    {position(reflect(_),prefix) == position(reflect(tail),start) : Nat}
  %Equal.sym(List<&2,Point>,reflect(List.append(&2,Point,head,tail)),List.append(&2,Point,reflect(head),reflect(tail)),reflect_append(head,tail)) :
    {position(_,prefix) == position(reflect(tail),start) : Nat}
  Equal.trans(Nat,position(List.append(&2,Point,reflect(head),reflect(tail)),prefix),position(reflect(tail),position(reflect(head),prefix)),position(reflect(tail),start),
    position_append(reflect(head),reflect(tail),prefix),
    Equal.cong(Nat,Nat,first => position(reflect(tail),first),position(reflect(head),prefix),start,group_reflected(view_axes,flagged,value,prefix)))

# One group of digits, forward, before other digits.
def step_forward(+view_axes: List<&2,Views.Axis>,+value: Nat,+rest: List<&2,Point>,+prefix: Nat,
  inside: {Nat.is_lt(value,ShapeLaws.elements(Views.axis_extents(view_axes))) == True{} : Bool}) ->
  {position(List.append(&2,Point,points_of(view_axes,value),rest),prefix) == position(rest,Nat.add(Nat.mul(prefix,ShapeLaws.elements(Views.axis_extents(view_axes))),value)) : Nat}:
  Equal.trans(Nat,position(List.append(&2,Point,points_of(view_axes,value),rest),prefix),position(rest,position(points_of(view_axes,value),prefix)),
    position(rest,Nat.add(Nat.mul(prefix,ShapeLaws.elements(Views.axis_extents(view_axes))),value)),
    position_append(points_of(view_axes,value),rest,prefix),
    Equal.cong(Nat,Nat,first => position(rest,first),position(points_of(view_axes,value),prefix),Nat.add(Nat.mul(prefix,ShapeLaws.elements(Views.axis_extents(view_axes))),value),
      position_of(view_axes,value,prefix,inside)))

def axes_points(+view_axes: List<&2,Views.Axis>,+value: Nat,+rest: List<&2,Point>) ->
  {axes(List.append(&2,Point,points_of(view_axes,value),rest)) == List.append(&2,Views.Axis,view_axes,axes(rest)) : List<&2,Views.Axis>}:
  Equal.trans(List<&2,Views.Axis>,axes(List.append(&2,Point,points_of(view_axes,value),rest)),List.append(&2,Views.Axis,axes(points_of(view_axes,value)),axes(rest)),
    List.append(&2,Views.Axis,view_axes,axes(rest)),axes_append(points_of(view_axes,value),rest),
    Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,head => List.append(&2,Views.Axis,head,axes(rest)),axes(points_of(view_axes,value)),view_axes,axes_of(view_axes,value)))

# Four groups of axes read with flags, the flags of a group following those of
# the groups before it: the mirrored view gives the row-major position of
# four digits the address the view gives the position of their reflections.
def grouped(+first: List<&2,Views.Axis>,+second: List<&2,Views.Axis>,+third: List<&2,Views.Axis>,+fourth: List<&2,Views.Axis>,+offset: U32,+flagged: List<&2,Bool>,
  +a: Nat,+b: Nat,+c: Nat,+d: Nat,
  +a_inside: {Nat.is_lt(a,ShapeLaws.elements(Views.axis_extents(first))) == True{} : Bool},
  +b_inside: {Nat.is_lt(b,ShapeLaws.elements(Views.axis_extents(second))) == True{} : Bool},
  +c_inside: {Nat.is_lt(c,ShapeLaws.elements(Views.axis_extents(third))) == True{} : Bool},
  +d_inside: {Nat.is_lt(d,ShapeLaws.elements(Views.axis_extents(fourth))) == True{} : Bool}) ->
  {Mirror.address(Views.View{List.append(&2,Views.Axis,first,List.append(&2,Views.Axis,second,List.append(&2,Views.Axis,third,fourth))),offset},flagged,
      Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(a,ShapeLaws.elements(Views.axis_extents(second))),b),ShapeLaws.elements(Views.axis_extents(third))),c),ShapeLaws.elements(Views.axis_extents(fourth))),d)) ==
    Copy.address(Views.View{List.append(&2,Views.Axis,first,List.append(&2,Views.Axis,second,List.append(&2,Views.Axis,third,fourth))),offset},
      Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(reflected(Views.axis_extents(first),flagged,a),ShapeLaws.elements(Views.axis_extents(second))),
        reflected(Views.axis_extents(second),Views.drop_flags(List.length(&2,Views.Axis,first),flagged),b)),ShapeLaws.elements(Views.axis_extents(third))),
        reflected(Views.axis_extents(third),Views.drop_flags(List.length(&2,Views.Axis,second),Views.drop_flags(List.length(&2,Views.Axis,first),flagged)),c)),ShapeLaws.elements(Views.axis_extents(fourth))),
        reflected(Views.axis_extents(fourth),Views.drop_flags(List.length(&2,Views.Axis,third),Views.drop_flags(List.length(&2,Views.Axis,second),Views.drop_flags(List.length(&2,Views.Axis,first),flagged))),d))) : U32}:
  +nb = ShapeLaws.elements(Views.axis_extents(second))
  +nc = ShapeLaws.elements(Views.axis_extents(third))
  +nd = ShapeLaws.elements(Views.axis_extents(fourth))
  +f2 = Views.drop_flags(List.length(&2,Views.Axis,first),flagged)
  +f3 = Views.drop_flags(List.length(&2,Views.Axis,second),f2)
  +f4 = Views.drop_flags(List.length(&2,Views.Axis,third),f3)
  +ra = reflected(Views.axis_extents(first),flagged,a)
  +rb = reflected(Views.axis_extents(second),f2,b)
  +rc = reflected(Views.axis_extents(third),f3,c)
  +rd = reflected(Views.axis_extents(fourth),f4,d)
  +p1 = points_of(first,a)
  +p2 = points_of(second,b)
  +p3 = points_of(third,c)
  +p4 = points_of(fourth,d)
  +rest3 = List.append(&2,Point,p3,p4)
  +rest2 = List.append(&2,Point,p2,rest3)
  +all = List.append(&2,Point,p1,rest2)
  +view_axes = List.append(&2,Views.Axis,first,List.append(&2,Views.Axis,second,List.append(&2,Views.Axis,third,fourth)))
  +forward_position = Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(a,nb),b),nc),c),nd),d)
  +reflected_position = Nat.add(Nat.mul(Nat.add(Nat.mul(Nat.add(Nat.mul(ra,nb),rb),nc),rc),nd),rd)
  +all_axes = Equal.trans(List<&2,Views.Axis>,axes(all),List.append(&2,Views.Axis,first,axes(rest2)),view_axes,axes_points(first,a,rest2),
    Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => List.append(&2,Views.Axis,first,others),axes(rest2),List.append(&2,Views.Axis,second,List.append(&2,Views.Axis,third,fourth)),
      Equal.trans(List<&2,Views.Axis>,axes(rest2),List.append(&2,Views.Axis,second,axes(rest3)),List.append(&2,Views.Axis,second,List.append(&2,Views.Axis,third,fourth)),axes_points(second,b,rest3),
        Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => List.append(&2,Views.Axis,second,others),axes(rest3),List.append(&2,Views.Axis,third,fourth),
          Equal.trans(List<&2,Views.Axis>,axes(rest3),List.append(&2,Views.Axis,third,axes(p4)),List.append(&2,Views.Axis,third,fourth),axes_points(third,c,p4),
            Equal.cong(List<&2,Views.Axis>,List<&2,Views.Axis>,others => List.append(&2,Views.Axis,third,others),axes(p4),fourth,axes_of(fourth,d)))))))
  +all_valid = valid_append(p1,rest2,valid_of(first,a,a_inside),valid_append(p2,rest3,valid_of(second,b,b_inside),valid_append(p3,p4,valid_of(third,c,c_inside),valid_of(fourth,d,d_inside))))
  +forward_exact = Equal.trans(Nat,position(all,0n),position(rest2,a),forward_position,step_forward(first,a,rest2,0n,a_inside),
    Equal.trans(Nat,position(rest2,a),position(rest3,Nat.add(Nat.mul(a,nb),b)),forward_position,step_forward(second,b,rest3,a,b_inside),
      Equal.trans(Nat,position(rest3,Nat.add(Nat.mul(a,nb),b)),position(p4,Nat.add(Nat.mul(Nat.add(Nat.mul(a,nb),b),nc),c)),forward_position,
        step_forward(third,c,p4,Nat.add(Nat.mul(a,nb),b),c_inside),position_of(fourth,d,Nat.add(Nat.mul(Nat.add(Nat.mul(a,nb),b),nc),c),d_inside))))
  +reflected_all = Equal.trans(Nat,position(reflect(attached(all,flagged)),0n),position(reflect(attached(rest2,f2)),ra),reflected_position,step_reflected(first,a,rest2,flagged,0n),
    Equal.trans(Nat,position(reflect(attached(rest2,f2)),ra),position(reflect(attached(rest3,f3)),Nat.add(Nat.mul(ra,nb),rb)),reflected_position,step_reflected(second,b,rest3,f2,ra),
      Equal.trans(Nat,position(reflect(attached(rest3,f3)),Nat.add(Nat.mul(ra,nb),rb)),position(reflect(attached(p4,f4)),Nat.add(Nat.mul(Nat.add(Nat.mul(ra,nb),rb),nc),rc)),reflected_position,
        step_reflected(third,c,p4,f3,Nat.add(Nat.mul(ra,nb),rb)),group_reflected(fourth,f4,d,Nat.add(Nat.mul(Nat.add(Nat.mul(ra,nb),rb),nc),rc)))))
  %all_axes : {Mirror.address(Views.View{_,offset},flagged,forward_position) == Copy.address(Views.View{_,offset},reflected_position) : U32}
  %forward_exact : {Mirror.address(Views.View{axes(all),offset},flagged,_) == Copy.address(Views.View{axes(all),offset},reflected_position) : U32}
  %reflected_all : {Mirror.address(Views.View{axes(all),offset},flagged,position(all,0n)) == Copy.address(Views.View{axes(all),offset},_) : U32}
  directed_address(all,offset,flagged,all_valid)
