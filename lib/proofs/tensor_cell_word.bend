# A cell of values with their lanes on an axis and their groups innermost:
# its word splits into the digits of the other axes and its group, which the
# geometry of the values names the same.
import Base
import ../machine_limits.bend as Machine
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ./tensor_tiled_public.bend as Public
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_target_cells.bend as Targets
import ./tensor_box_digits.bend as Digits
import ./tensor_relane_values.bend as Values
import ./tensor_tiled_repositioned.bend as Repositioned
import ./tensor_shape_refinement.bend as ShapeLaws

type CellWord<-values: List<&2,U32>,-axis: Nat,-word: U32> is Data:
  CellWord{others: List<&2,Nat>,group: Nat,
    others_below: Coordinates.DigitsBelow(Grid.removed(values,axis),others),
    group_inside: {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},
    exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,Grid.Innermost{}),
      Coordinates.placed_digits(Nat.is_lt(axis,List.length(&2,U32,values)),Grid.Innermost{},others,group)) : Nat},
    group_same: {Representation.cell_group(Public.geometry_of(values,axis,Grid.Innermost{}),U32.to_nat(word)) == group : Nat},
    count_same: {Representation.lane_count(Public.geometry_of(values,axis,Grid.Innermost{})) == U32.to_nat(Grid.lanes_at(values,axis)) : Nat},
    placed_below: Coordinates.DigitsBelow(Tiled.cell_shape(values,axis,Grid.Innermost{}),
      Coordinates.placed_digits(Nat.is_lt(axis,List.length(&2,U32,values)),Grid.Innermost{},others,group))}

# The group of the digits of a cell lies below the groups. A unit lane axis
# has one group.
def group_inside_of(within: Bool,+values: List<&2,U32>,+axis: Nat,+digits: List<&2,Nat>,+classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},
  +below: Coordinates.GroupBelow(within,Values.cell_group(within,Grid.Innermost{},digits),Grid.groups(Grid.lanes_at(values,axis)))) ->
  {Nat.is_lt(Values.cell_group(within,Grid.Innermost{},digits),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool}:
  match within:
    case True{}: below
    case False{}:
      %Equal.sym(U32,Grid.lanes_at(values,axis),1,Public.lanes_past(values,axis,classified)) : {Nat.is_lt(0n,U32.to_nat(Grid.groups(_))) == True{} : Bool}
      {==}

# The geometry names the group of the digits of a cell.
def group_of(within: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+digits: List<&2,Nat>,+word: U32,
  +classified: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},+inside_axis: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
  +removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +others_below: Coordinates.DigitsBelow(Grid.removed(values,axis),Values.cell_others(within,Grid.Innermost{},digits)),
  +group_inside: {Nat.is_lt(Values.cell_group(within,Grid.Innermost{},digits),U32.to_nat(Grid.groups(Grid.lanes_at(values,axis)))) == True{} : Bool},
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,axis,Grid.Innermost{}),
    Coordinates.placed_digits(within,Grid.Innermost{},Values.cell_others(within,Grid.Innermost{},digits),Values.cell_group(within,Grid.Innermost{},digits))) : Nat},
  +inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{}))) == True{} : Bool}) ->
  {Representation.cell_group(Public.geometry_of(values,axis,Grid.Innermost{}),U32.to_nat(word)) == Values.cell_group(within,Grid.Innermost{},digits) : Nat}:
  match within:
    case True{}:
      Coordinates.group_within(limits,values,axis,Grid.Innermost{},Values.cell_others(True{},Grid.Innermost{},digits),Values.cell_group(True{},Grid.Innermost{},digits),word,
        classified,removed_room,others_below,group_inside,exact)
    case False{}: Repositioned.group_unit(limits,values,axis,Grid.Innermost{},word,classified,inside_axis,removed_room,inside_word)

# Every word below the cells splits so.
def cell_word(+limits: Machine.Limits,+values: List<&2,U32>,+axis: Nat,+word: U32,
  +inside_axis: {Nat.is_le(axis,List.length(&2,U32,values)) == True{} : Bool},
  +removed_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,axis)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +inside_word: {Nat.is_lt(U32.to_nat(word),ShapeLaws.elements(Tiled.cell_shape(values,axis,Grid.Innermost{}))) == True{} : Bool}) -> CellWord<values,axis,word>:
  +within = Nat.is_lt(axis,List.length(&2,U32,values))
  +number = U32.to_nat(word)
  +cells = Tiled.cell_shape(values,axis,Grid.Innermost{})
  +removed = Grid.removed(values,axis)
  +groups = Grid.groups(Grid.lanes_at(values,axis))
  +digits = Targets.cell_coordinates(cells,number)
  +below_cells = Digits.coordinates_below(cells,number,inside_word)
  +others = Values.cell_others(within,Grid.Innermost{},digits)
  +group = Values.cell_group(within,Grid.Innermost{},digits)
  +others_below = Values.cell_below(within,Grid.Innermost{},removed,groups,digits,below_cells)
  +group_inside = group_inside_of(within,values,axis,digits,{==},Values.group_below(within,Grid.Innermost{},removed,groups,digits,below_cells))
  +whole = Values.cell_whole(within,Grid.Innermost{},removed,groups,digits,below_cells)
  +placed = Coordinates.placed_digits(within,Grid.Innermost{},others,group)
  +exact = {%Equal.sym(List<&2,Nat>,placed,digits,whole) : {number == ShapeLaws.digits_value(cells,_) : Nat}
    Equal.sym(Nat,ShapeLaws.digits_value(cells,digits),number,Digits.value_coordinates(cells,number,inside_word)) : {number == ShapeLaws.digits_value(cells,placed) : Nat}}
  CellWord{others,group,others_below,group_inside,exact,
    group_of(within,limits,values,axis,digits,word,{==},inside_axis,removed_room,others_below,group_inside,exact,inside_word),
    Coordinates.geometry_lanes(values,axis,Grid.Innermost{},inside_axis),
    {%Equal.sym(List<&2,Nat>,placed,digits,whole) : Coordinates.DigitsBelow(cells,_) below_cells : Coordinates.DigitsBelow(cells,placed)}}
