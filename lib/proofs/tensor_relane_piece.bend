# A write of a relane block of a target piece: the target cell a block
# writes at a lane of it holds the target value of that cell.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tasks.bend as Tasks
import ../kernel_tile.bend as Tile
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_relane_geometry.bend as Geometry
import ./tensor_relane_writes.bend as Writes
import ./tensor_relane_values.bend as Values
import ./tensor_relane_block.bend as Block
import ./tensor_box_digits.bend as BoxDigits
import ./tensor_grid_box.bend as Box
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_public.bend as Public
import ./tensor_move_readback.bend as Held
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./nat_division.bend as Division
import ./tensor_target_cells.bend as Targets
import ./tensor_relane_blocks.bend as Blocks
import ./tensor_relane_cover.bend as Cover
import ../tensor_tiled_cells.bend as Cells
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison

# ---- A target piece ----

# The grid of the target cells, and the first cell, the cells, the box and the
# first coordinates of a piece of it.
def piece_grid(+values: List<&2,U32>,+target: Nat,+position: Grid.Position) -> Grid.Grid:
  Grid.grid(Tiled.cell_shape(values,target,position),Grid.piece_target())

def piece_first(+values: List<&2,U32>,+target: Nat,+position: Grid.Position,+piece: U32) -> U32:
  Grid.start(piece_grid(values,target,position),piece)

def piece_span(+values: List<&2,U32>,+target: Nat,+position: Grid.Position,+piece: U32) -> U32:
  Grid.span(piece_grid(values,target,position),piece)

def piece_boxes(+values: List<&2,U32>,+target: Nat,+position: Grid.Position,+piece: U32) -> List<&2,U32>:
  Grid.box_of(piece_grid(values,target,position),Tiled.cell_shape(values,target,position),piece_span(values,target,position,piece))

def piece_lows(+values: List<&2,U32>,+target: Nat,+position: Grid.Position,+piece: U32) -> List<&2,U32>:
  Grid.coordinates(Tiled.cell_shape(values,target,position),piece_first(values,target,position,piece))

# The axes of the blocks of a piece and the block at an index.
def piece_axes(within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>,
  +piece: U32) -> List<&2,Tiled.Ranged>:
  Writes.piece_axes(Values.values_blocked(within,position,target,source,source_within,source_position,values),piece_lows(values,target,position,piece),
    piece_boxes(values,target,position,piece))

# The blocks of a piece read with a source stride, or with none: half a word
# holds twice the step.
def piece_step_room(+within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>,
  +piece: U32,+limits: Machine.Limits,
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_positive: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool}) ->
  {Nat.is_le(Nat.double(U32.to_nat(Geometry.blocked_step(Geometry.blocked_of(piece_axes(within,position,target,source,source_within,source_position,values,piece)),0))),
    U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +word = Machine.half_word(limits)
  +cells = Tiled.cell_shape(values,source,source_position)
  +blocked = Values.values_blocked(within,position,target,source,source_within,source_position,values)
  +boxes = piece_boxes(values,target,position,piece)
  +listed = Geometry.listed_in(Tiled.target_axes(within,position,target,List.length(&2,U32,values)),source,source_within,source_position,target,Tasks.strides_of(cells),values,
    Values.groups_axis(source_position,cells),word,BoxDigits.strides_in(cells,limits,source_cells_room,source_positive))
  Geometry.step_in(Geometry.blocked_of(piece_axes(within,position,target,source,source_within,source_position,values,piece)),0,word,
    Geometry.ranged_in(blocked,piece_lows(values,target,position,piece),boxes,Tasks.strides_of(boxes),word,listed),Order.zero_le(U32.to_nat(word)))

def piece_block(within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>,
  +piece: U32,+index: U32) -> Tiled.Block:
  Tiled.block_at(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,Writes.first_block())

# ---- What every target cell axis is to the blocks ----

def appended_count(~Item: Data,items: List<&2,Item>,+last: Item) -> {List.length(&2,Item,List.append(&2,Item,items,[last])) == 1n+List.length(&2,Item,items) : Nat}:
  match items:
    case Nil{}: {==}
    case Con{+item,+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Item,List.append(&2,Item,rest,[last])),1n+List.length(&2,Item,rest),appended_count(~Item,rest,last))

# One per axis of the values.
def blocked_length(within: Bool,position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool}) ->
  {List.length(&2,Tiled.Blocked,Values.values_blocked(within,position,target,source,source_within,source_position,values)) == List.length(&2,U32,values) : Nat}:
  match within position:
    case False{} +position:
      +rank = List.length(&2,U32,values)
      +cells = Tiled.cell_shape(values,source,source_position)
      +strides = Tasks.strides_of(cells)
      +axis = Values.groups_axis(source_position,cells)
      +axes = Tiled.axes_from(rank,0n,rank)
      Equal.trans(Nat,List.length(&2,Tiled.Blocked,Tiled.blocked_list(axes,source,source_within,source_position,target,strides,values,axis)),
        List.length(&2,Tiled.CellAxis,axes),rank,Geometry.blocked_count(axes,source,source_within,source_position,target,strides,values,axis),
        Equal.trans(Nat,List.length(&2,Tiled.CellAxis,axes),List.length(&2,Tiled.CellAxis,Geometry.every_axis(rank,0n)),rank,
          Equal.cong(List<&2,Tiled.CellAxis>,Nat,list => List.length(&2,Tiled.CellAxis,list),axes,Geometry.every_axis(rank,0n),
            Geometry.never_reach(rank,0n,rank,Order.reflexive(rank))),
          Geometry.every_count(rank,0n)))
    case True{} Grid.Innermost{}:
      +rank = List.length(&2,U32,values)
      +cells = Tiled.cell_shape(values,source,source_position)
      +strides = Tasks.strides_of(cells)
      +axis = Values.groups_axis(source_position,cells)
      +axes = Tiled.axes_from(rank,0n,target)
      +whole = List.append(&2,Tiled.CellAxis,axes,[Tiled.OfGroups{}])
      +count = {%within_value : {Nat.add(List.length(&2,Tiled.CellAxis,axes),Geometry.skipped_count(_)) == rank : Nat}
        Geometry.axes_count(rank,target,0n) : {Nat.add(List.length(&2,Tiled.CellAxis,axes),1n) == rank : Nat}}
      Equal.trans(Nat,List.length(&2,Tiled.Blocked,Tiled.blocked_list(whole,source,source_within,source_position,target,strides,values,axis)),
        List.length(&2,Tiled.CellAxis,whole),rank,Geometry.blocked_count(whole,source,source_within,source_position,target,strides,values,axis),
        Equal.trans(Nat,List.length(&2,Tiled.CellAxis,whole),1n+List.length(&2,Tiled.CellAxis,axes),rank,appended_count(~Tiled.CellAxis,axes,Tiled.OfGroups{}),
          Equal.trans(Nat,1n+List.length(&2,Tiled.CellAxis,axes),Nat.add(List.length(&2,Tiled.CellAxis,axes),1n),rank,
            Equal.sym(Nat,Nat.add(List.length(&2,Tiled.CellAxis,axes),1n),1n+List.length(&2,Tiled.CellAxis,axes),Addition.successor_width(List.length(&2,Tiled.CellAxis,axes))),
            count)))
    case True{} Grid.Outermost{}:
      +rank = List.length(&2,U32,values)
      +cells = Tiled.cell_shape(values,source,source_position)
      +strides = Tasks.strides_of(cells)
      +axis = Values.groups_axis(source_position,cells)
      +axes = Tiled.axes_from(rank,0n,target)
      +count = {%within_value : {Nat.add(List.length(&2,Tiled.CellAxis,axes),Geometry.skipped_count(_)) == rank : Nat}
        Geometry.axes_count(rank,target,0n) : {Nat.add(List.length(&2,Tiled.CellAxis,axes),1n) == rank : Nat}}
      Equal.trans(Nat,1n+List.length(&2,Tiled.Blocked,Tiled.blocked_list(axes,source,source_within,source_position,target,strides,values,axis)),
        1n+List.length(&2,Tiled.CellAxis,axes),rank,
        Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Tiled.Blocked,Tiled.blocked_list(axes,source,source_within,source_position,target,strides,values,axis)),
          List.length(&2,Tiled.CellAxis,axes),Geometry.blocked_count(axes,source,source_within,source_position,target,strides,values,axis)),
        Equal.trans(Nat,1n+List.length(&2,Tiled.CellAxis,axes),Nat.add(List.length(&2,Tiled.CellAxis,axes),1n),rank,
          Equal.sym(Nat,Nat.add(List.length(&2,Tiled.CellAxis,axes),1n),1n+List.length(&2,Tiled.CellAxis,axes),Addition.successor_width(List.length(&2,Tiled.CellAxis,axes))),
          count))

# Without the source lanes in the cells no block moves lanes.
def unit_none(axes: List<&2,Tiled.CellAxis>,+source: Nat,+source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32) ->
  Geometry.NoLanes(Tiled.blocked_list(axes,source,False{},source_position,target,strides,shape,groups_axis)):
  match axes:
    case Nil{}: Unit{}
    case Con{Tiled.OfGroups{},+rest}: unit_none(rest,source,source_position,target,strides,shape,groups_axis)
    case Con{Tiled.OfAxis{+logical},+rest}: unit_none(rest,source,source_position,target,strides,shape,groups_axis)

# ---- One write of a block of a piece ----

# The target cell a block of a piece writes at a lane holds the target value
# of that cell.
def Written(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,
  +target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,+index: U32,+at: U32) -> Data:
  {Block.eighth(at,Tiled.transposed(Block.expected_eight(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},Values.reads_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),Geometry.source_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),
      Geometry.step_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))))) ==
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},
      (piece_first(values,target,position,piece) + Writes.write_address(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,at) : U32)) : Tile.Tile8}

# A block of a piece reads the source lane axis at lane i of its block, or
# the unit lane axis at lane zero.
def source_block(source_within: Bool,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+target: Nat,+position: Grid.Position,+within: Bool,
  +piece: U32,+index: U32,+at: U32,
  +same_blocked: {List.length(&2,Tiled.Blocked,Values.values_blocked(within,position,target,source,source_within,source_position,values)) ==
    List.length(&2,U32,piece_boxes(values,target,position,piece)) : Nat},
  +same_lows: {List.length(&2,U32,piece_lows(values,target,position,piece)) == List.length(&2,U32,piece_boxes(values,target,position,piece)) : Nat},
  +small: {Nat.is_lt(U32.to_nat(at),8n) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(at),U32.to_nat(Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)))) == True{} : Bool}) ->
  Values.SourceBlock(source_within,Values.values_blocked(within,position,target,source,source_within,source_position,values),
    Geometry.block_digits(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,U32.to_nat(at)),U32.to_nat(at)):
  match source_within:
    case True{}:
      +blocked = Values.values_blocked(within,position,target,source,True{},source_position,values)
      +lows = piece_lows(values,target,position,piece)
      +boxes = piece_boxes(values,target,position,piece)
      +axes = Writes.piece_axes(blocked,lows,boxes)
      +digits = Geometry.block_digits(axes,index,U32.to_nat(at))
      {%Writes.ranged_blocked(blocked,lows,boxes,same_blocked,same_lows) : Geometry.LaneDigits(_,digits,U32.to_nat(at))
        Geometry.block_lanes(axes,index,U32.to_nat(at),small) : Geometry.LaneDigits(blocked,digits,U32.to_nat(at))}
    case False{}:
      +blocked = Values.values_blocked(within,position,target,source,False{},source_position,values)
      +lows = piece_lows(values,target,position,piece)
      +boxes = piece_boxes(values,target,position,piece)
      +axes = Writes.piece_axes(blocked,lows,boxes)
      +source_cells = Tiled.cell_shape(values,source,source_position)
      +block = Tiled.block_at(axes,index,Writes.first_block())
      +none = unit_none(Tiled.target_axes(within,position,target,List.length(&2,U32,values)),source,source_position,target,Tasks.strides_of(source_cells),values,
        Values.groups_axis(source_position,source_cells))
      +none_axes = {%Equal.sym(List<&2,Tiled.Blocked>,Geometry.blocked_of(axes),blocked,Writes.ranged_blocked(blocked,lows,boxes,same_blocked,same_lows)) :
          Geometry.NoLanes(_)
        none : Geometry.NoLanes(Geometry.blocked_of(axes))}
      +high_one = {Equal.cong(U32 & (U32 & U32),U32,pair => Pair.snd(U32,U32,Pair.snd(U32,U32 & U32,pair)),Writes.writes_of(block),Writes.writes_of(Writes.first_block()),
        Writes.kept_writes(axes,Writes.first_block(),index,none_axes)) : {Writes.high_of(block) == 1 : U32}}
      +below_one = {%high_one : {Nat.is_lt(U32.to_nat(at),U32.to_nat(_)) == True{} : Bool} to_high : {Nat.is_lt(U32.to_nat(at),1n) == True{} : Bool}}
      Order.zero_below(U32.to_nat(at),Division.lt_succ_le(U32.to_nat(at),0n,below_one))

# The target cell written from the facts of the write and its cell.
def cell_wrote(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32,+at: U32,
  +from_low: {Nat.is_le(U32.to_nat(Writes.low_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))),U32.to_nat(at)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(at),U32.to_nat(Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)))) == True{} : Bool},
  +same_blocked: {List.length(&2,Tiled.Blocked,Values.values_blocked(within,position,target,source,source_within,source_position,values)) ==
    List.length(&2,U32,piece_boxes(values,target,position,piece)) : Nat},
  +same_lows: {List.length(&2,U32,piece_lows(values,target,position,piece)) == List.length(&2,U32,piece_boxes(values,target,position,piece)) : Nat},
  +small: {Nat.is_lt(U32.to_nat(at),8n) == True{} : Bool},
  +cell: BoxDigits.CellFacts(Tiled.cell_shape(values,target,position),piece_first(values,target,position,piece),Writes.write_address(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,at),
    Geometry.block_digits(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,U32.to_nat(at)))) -> Written(~Logical,~original,logical,values,source,source_position,source_within,target,position,within,piece,index,at):
  Held.Both{+target_exact,+cell_below} = cell
  +cells = Tiled.cell_shape(values,target,position)
  +grid = piece_grid(values,target,position)
  +first = piece_first(values,target,position,piece)
  +span = piece_span(values,target,position,piece)
  +boxes = piece_boxes(values,target,position,piece)
  +lows = piece_lows(values,target,position,piece)
  +blocked = Values.values_blocked(within,position,target,source,source_within,source_position,values)
  +source_cells = Tiled.cell_shape(values,source,source_position)
  +capacity = Machine.capacity(limits)
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +rank = List.length(&2,U32,values)
  +rank_cells = List.length(&2,U32,cells)
  +axes = piece_axes(within,position,target,source,source_within,source_position,values,piece)
  +digits = Geometry.block_digits(axes,index,U32.to_nat(at))
  +below = {%within_value : Coordinates.DigitsBelow(Grid.placed(_,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),digits)
    cell_below : Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),digits)}
  Values.write_tile(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,at,
    (first + Writes.write_address(axes,index,at) : U32),same_blocked,same_lows,small,within_value,inside,room,lanes_room,source_value,source_inside,source_room,
    source_cells_room,source_lanes_room,rank_room,apart,
    source_block(source_within,values,source,source_position,target,position,within,piece,index,at,same_blocked,same_lows,small,to_high),below,target_exact)

# The target digits of a block of a piece at a lane it writes: the
# coordinates of the first cell of the piece plus the number of the box
# digits, that number inside the piece, and the facts of the cell.
def PieceCell(within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>,
  +piece: U32,+index: U32,+at: U32) -> Data:
  +cells = Tiled.cell_shape(values,target,position)
  +first = piece_first(values,target,position,piece)
  +boxes = piece_boxes(values,target,position,piece)
  +axes = piece_axes(within,position,target,source,source_within,source_position,values,piece)
  +box = Writes.box_digits(axes,index,U32.to_nat(at))
  +digits = Geometry.block_digits(axes,index,U32.to_nat(at))
  Held.Both<{digits == Targets.cell_coordinates(cells,Nat.add(U32.to_nat(first),ShapeLaws.digits_value(boxes,box))) : List<&2,Nat>},
    Held.Both<{Nat.is_lt(ShapeLaws.digits_value(boxes,box),U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},
      BoxDigits.CellFacts(cells,first,Writes.write_address(axes,index,at),digits)>>

def piece_cell_of(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,+position: Grid.Position,
  +within: Bool,+piece: U32,
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},+index: U32,+at: U32,
  +boxes_count: {List.length(&2,U32,piece_boxes(values,target,position,piece)) == List.length(&2,U32,Tiled.cell_shape(values,target,position)) : Nat},
  +double_room: {Nat.is_le(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,target,position))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +cells_rank_room: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +facts: Writes.WriteFacts(Values.values_blocked(within,position,target,source,source_within,source_position,values),piece_lows(values,target,position,piece),
    piece_boxes(values,target,position,piece),index,at)) -> PieceCell(within,position,target,source,source_within,source_position,values,piece,index,at):
  Held.Both{+address,Held.Both{+box_below,Held.Both{+shifted,+small}}} = facts
  +cells = Tiled.cell_shape(values,target,position)
  +first = piece_first(values,target,position,piece)
  +span = piece_span(values,target,position,piece)
  +boxes = piece_boxes(values,target,position,piece)
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +rank_cells = List.length(&2,U32,cells)
  +axes = piece_axes(within,position,target,source,source_within,source_position,values,piece)
  +box = Writes.box_digits(axes,index,U32.to_nat(at))
  +digits = Geometry.block_digits(axes,index,U32.to_nat(at))
  +number = ShapeLaws.digits_value(boxes,box)
  +number_in = {%box_cells : {Nat.is_lt(number,_) == True{} : Bool}
    Coordinates.digits_below(boxes,box,box_below) : {Nat.is_lt(number,U32.to_nat(span)) == True{} : Bool}}
  +steps = BoxDigits.unit_steps(rank_cells,0n)
  +ending = BoxDigits.piece_ending(cells,steps,boxes,U32.to_nat(first),U32.to_nat(span),number,
    Box.grid_piece_box(limits,cells,steps,Grid.piece_target(),cells_room,nonempty,cells_rank_room,piece,piece_inside,number,number_in))
  +boxed = BoxDigits.grid_boxes(rank_cells,limits,cells,Grid.piece_target(),cells_room,nonempty,cells_rank_room,piece,piece_inside,number,number_in,0n)
  Held.Both{BoxDigits.piece_coordinates_of(cells,boxes,first,U32.to_nat(span),digits,box,cap,cap_ok,double_room,boxes_count,shifted,box_below,ending,boxed),
    Held.Both{number_in,BoxDigits.piece_cell(cells,boxes,first,U32.to_nat(span),digits,box,Writes.write_address(axes,index,at),cap,cap_ok,double_room,boxes_count,address,
      shifted,box_below,ending,boxed)}}

def cell_of_parts(within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>,
  +piece: U32,+index: U32,+at: U32,+parts: PieceCell(within,position,target,source,source_within,source_position,values,piece,index,at)) ->
  BoxDigits.CellFacts(Tiled.cell_shape(values,target,position),piece_first(values,target,position,piece),
    Writes.write_address(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,at),
    Geometry.block_digits(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,U32.to_nat(at))):
  Held.Both{+coordinates,Held.Both{+number_in,+cell}} = parts
  cell

# The target cell written from the facts of the write.
def write_wrote(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32,+at: U32,
  +from_low: {Nat.is_le(U32.to_nat(Writes.low_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))),U32.to_nat(at)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(at),U32.to_nat(Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)))) == True{} : Bool},
  +same_blocked: {List.length(&2,Tiled.Blocked,Values.values_blocked(within,position,target,source,source_within,source_position,values)) ==
    List.length(&2,U32,piece_boxes(values,target,position,piece)) : Nat},
  +same_lows: {List.length(&2,U32,piece_lows(values,target,position,piece)) == List.length(&2,U32,piece_boxes(values,target,position,piece)) : Nat},
  +boxes_count: {List.length(&2,U32,piece_boxes(values,target,position,piece)) == List.length(&2,U32,Tiled.cell_shape(values,target,position)) : Nat},
  +double_room: {Nat.is_le(Nat.double(ShapeLaws.elements(Tiled.cell_shape(values,target,position))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +cells_rank_room: {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +facts: Writes.WriteFacts(Values.values_blocked(within,position,target,source,source_within,source_position,values),piece_lows(values,target,position,piece),
    piece_boxes(values,target,position,piece),index,at)) -> Written(~Logical,~original,logical,values,source,source_position,source_within,target,position,within,piece,index,at):
  Held.Both{+address,Held.Both{+box_below,Held.Both{+shifted,+small}}} = facts
  cell_wrote(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,at,from_low,to_high,same_blocked,same_lows,small,
    cell_of_parts(within,position,target,source,source_within,source_position,values,piece,index,at,
      piece_cell_of(limits,values,source,source_position,source_within,target,position,within,piece,cells_room,nonempty,piece_inside,box_cells,index,at,boxes_count,double_room,
        cells_rank_room,facts)))

# ---- The lengths of a piece ----

def boxes_count(+values: List<&2,U32>,+target: Nat,+position: Grid.Position,+piece: U32) -> {List.length(&2,U32,piece_boxes(values,target,position,piece)) == List.length(&2,U32,Tiled.cell_shape(values,target,position)) : Nat}:
  BoxDigits.box_of_length(piece_grid(values,target,position),Tiled.cell_shape(values,target,position),piece_span(values,target,position,piece))

def same_blocked(+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool}) -> {List.length(&2,Tiled.Blocked,Values.values_blocked(within,position,target,source,source_within,source_position,values)) ==
    List.length(&2,U32,piece_boxes(values,target,position,piece)) : Nat}:
  +cells = Tiled.cell_shape(values,target,position)
  +boxes = piece_boxes(values,target,position,piece)
  +rank = List.length(&2,U32,values)
  +rank_cells = List.length(&2,U32,cells)
  Equal.trans(Nat,List.length(&2,Tiled.Blocked,Values.values_blocked(within,position,target,source,source_within,source_position,values)),rank,List.length(&2,U32,boxes),
    blocked_length(within,position,target,source,source_within,source_position,values,within_value),
    Equal.sym(Nat,List.length(&2,U32,boxes),rank,Equal.trans(Nat,List.length(&2,U32,boxes),rank_cells,rank,boxes_count(values,target,position,piece),
      Values.cell_length(values,target,position,within,within_value))))

def same_lows(+values: List<&2,U32>,+target: Nat,+position: Grid.Position,+piece: U32) -> {List.length(&2,U32,piece_lows(values,target,position,piece)) == List.length(&2,U32,piece_boxes(values,target,position,piece)) : Nat}:
  +cells = Tiled.cell_shape(values,target,position)
  Equal.trans(Nat,List.length(&2,U32,piece_lows(values,target,position,piece)),List.length(&2,U32,cells),List.length(&2,U32,piece_boxes(values,target,position,piece)),
    BoxDigits.coordinates_count(cells,piece_first(values,target,position,piece)),
    Equal.sym(Nat,List.length(&2,U32,piece_boxes(values,target,position,piece)),List.length(&2,U32,cells),boxes_count(values,target,position,piece)))

def cells_rank_room(+limits: Machine.Limits,+values: List<&2,U32>,+target: Nat,+position: Grid.Position,+within: Bool,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {Nat.is_le(List.length(&2,U32,Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  %Equal.sym(Nat,List.length(&2,U32,Tiled.cell_shape(values,target,position)),List.length(&2,U32,values),Values.cell_length(values,target,position,within,within_value)) :
    {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
  rank_room

# ---- A write of a block of a piece ----

# The facts of a write of a block of a piece at a lane past its first and
# before its last written lane.
# The axes of the blocks of a piece take the source lane axis at most once,
# its box is positive, every axis fits half the word from its first index, and
# twice the cells of the box fit half the word.
def PieceRooms(+blocked: List<&2,Tiled.Blocked>,+lows: List<&2,U32>,+boxes: List<&2,U32>,+cap: U32) -> Data:
  Held.Both<Geometry.LanesAtMost(blocked),Held.Both<BoxDigits.Positive(boxes),
    Held.Both<BoxDigits.Rooms(BoxDigits.words_nat(lows),boxes,cap),{Nat.is_le(Nat.double(ShapeLaws.elements(boxes)),U32.to_nat(cap)) == True{} : Bool}>>>

def piece_rooms(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool}) ->
  PieceRooms(Values.values_blocked(within,position,target,source,source_within,source_position,values),piece_lows(values,target,position,piece),piece_boxes(values,target,position,piece),Machine.half_word(limits)):
  +cells = Tiled.cell_shape(values,target,position)
  +first = piece_first(values,target,position,piece)
  +span = piece_span(values,target,position,piece)
  +boxes = piece_boxes(values,target,position,piece)
  +lows = piece_lows(values,target,position,piece)
  +blocked = Values.values_blocked(within,position,target,source,source_within,source_position,values)
  +source_cells = Tiled.cell_shape(values,source,source_position)
  +capacity = Machine.capacity(limits)
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +rank = List.length(&2,U32,values)
  +rank_cells = List.length(&2,U32,cells)
  +boxes_counted = boxes_count(values,target,position,piece)
  +most = Geometry.target_most(within,position,target,rank,source,source_within,source_position,Tasks.strides_of(source_cells),values,
    Values.groups_axis(source_position,source_cells))
  +elements_positive = {%Equal.sym(Nat,ShapeLaws.elements(boxes),U32.to_nat(span),box_cells) : {Nat.is_lt(0n,_) == True{} : Bool}
    span_positive : {Nat.is_lt(0n,ShapeLaws.elements(boxes)) == True{} : Bool}}
  +double_room = Limits.double_bound(limits,ShapeLaws.elements(cells),cells_room)
  +cells_rank = cells_rank_room(limits,values,target,position,within,within_value,rank_room)
  +steps = BoxDigits.unit_steps(rank_cells,0n)
  +boxes_room = Order.transitive(Nat.double(ShapeLaws.elements(boxes)),Nat.double(ShapeLaws.elements(cells)),U32.to_nat(cap),
    Order.double_le(ShapeLaws.elements(boxes),ShapeLaws.elements(cells),BoxDigits.piece_fits(cells,steps,boxes,U32.to_nat(first),U32.to_nat(span),0n,
      Box.grid_piece_box(limits,cells,steps,Grid.piece_target(),cells_room,nonempty,cells_rank,piece,piece_inside,0n,span_positive))),double_room)
  Held.Both{most,Held.Both{BoxDigits.positive_extents(boxes,elements_positive),Held.Both{
    BoxDigits.grid_box_rooms(limits,cells,Grid.piece_target(),cells_room,nonempty,cells_rank,piece,piece_inside),boxes_room}}}

def facts_of_rooms(+blocked: List<&2,Tiled.Blocked>,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+at: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +same_blocked: {List.length(&2,Tiled.Blocked,blocked) == List.length(&2,U32,boxes) : Nat},+same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,boxes) : Nat},
  +from_low: {Nat.is_le(U32.to_nat(Writes.low_of(Tiled.block_at(Writes.piece_axes(blocked,lows,boxes),index,Writes.first_block()))),U32.to_nat(at)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(at),U32.to_nat(Writes.high_of(Tiled.block_at(Writes.piece_axes(blocked,lows,boxes),index,Writes.first_block())))) == True{} : Bool},
  +rooms: PieceRooms(blocked,lows,boxes,cap)) -> Writes.WriteFacts(blocked,lows,boxes,index,at):
  Held.Both{+most,Held.Both{+positive,Held.Both{+fits,+boxes_room}}} = rooms
  Writes.piece_write(blocked,lows,boxes,index,at,cap,cap_ok,same_blocked,same_lows,most,positive,fits,from_low,to_high)

# Every block of a piece writes a lane.
def lane_of_rooms(+blocked: List<&2,Tiled.Blocked>,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+cap: U32,
  +cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +same_blocked: {List.length(&2,Tiled.Blocked,blocked) == List.length(&2,U32,boxes) : Nat},+same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,boxes) : Nat},
  +rooms: PieceRooms(blocked,lows,boxes,cap)) ->
  {Nat.is_lt(U32.to_nat(Writes.low_of(Tiled.block_at(Writes.piece_axes(blocked,lows,boxes),index,Writes.first_block()))),
    U32.to_nat(Writes.high_of(Tiled.block_at(Writes.piece_axes(blocked,lows,boxes),index,Writes.first_block())))) == True{} : Bool}:
  Held.Both{+most,Held.Both{+positive,Held.Both{+fits,+boxes_room}}} = rooms
  +axes = Writes.piece_axes(blocked,lows,boxes)
  Writes.block_lane(axes,Writes.first_block(),index,cap,cap_ok,
    {%Equal.sym(List<&2,Tiled.Blocked>,Geometry.blocked_of(axes),blocked,Writes.ranged_blocked(blocked,lows,boxes,same_blocked,same_lows)) : Geometry.LanesAtMost(_)
      most : Geometry.LanesAtMost(Geometry.blocked_of(axes))},
    Writes.ranged_ok(blocked,lows,boxes,cap,positive,fits),{==})

def piece_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32) ->
  {Nat.is_lt(U32.to_nat(Writes.low_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))),
    U32.to_nat(Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)))) == True{} : Bool}:
  lane_of_rooms(Values.values_blocked(within,position,target,source,source_within,source_position,values),piece_lows(values,target,position,piece),
    piece_boxes(values,target,position,piece),index,Machine.half_word(limits),Limits.half_word_fits(limits),
    same_blocked(values,source,source_position,source_within,target,position,within,piece,within_value),same_lows(values,target,position,piece),
    piece_rooms(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive))

def piece_facts(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32,+at: U32,
  +from_low: {Nat.is_le(U32.to_nat(Writes.low_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))),U32.to_nat(at)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(at),U32.to_nat(Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)))) == True{} : Bool}) ->
  Writes.WriteFacts(Values.values_blocked(within,position,target,source,source_within,source_position,values),piece_lows(values,target,position,piece),
    piece_boxes(values,target,position,piece),index,at):
  facts_of_rooms(Values.values_blocked(within,position,target,source,source_within,source_position,values),piece_lows(values,target,position,piece),piece_boxes(values,target,position,piece),index,at,Machine.half_word(limits),
    Limits.half_word_fits(limits),same_blocked(values,source,source_position,source_within,target,position,within,piece,within_value),same_lows(values,target,position,piece),
    from_low,to_high,piece_rooms(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive))

def facts_small(+blocked: List<&2,Tiled.Blocked>,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+at: U32,+facts: Writes.WriteFacts(blocked,lows,boxes,index,at)) ->
  {Nat.is_lt(U32.to_nat(at),8n) == True{} : Bool}:
  Held.Both{+address,Held.Both{+box_below,Held.Both{+shifted,+small}}} = facts
  small

# The lane of a write lies below eight.
def lane_small(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32,+at: U32,
  +from_low: {Nat.is_le(U32.to_nat(Writes.low_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))),U32.to_nat(at)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(at),U32.to_nat(Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)))) == True{} : Bool}) -> {Nat.is_lt(U32.to_nat(at),8n) == True{} : Bool}:
  facts_small(Values.values_blocked(within,position,target,source,source_within,source_position,values),piece_lows(values,target,position,piece),
    piece_boxes(values,target,position,piece),index,at,piece_facts(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,at,from_low,to_high))

# The target cell a block of a piece writes at a lane past its first and
# before its last written lane holds the target value of that cell.
def piece_written(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32,+at: U32,
  +from_low: {Nat.is_le(U32.to_nat(Writes.low_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))),U32.to_nat(at)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(at),U32.to_nat(Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)))) == True{} : Bool}) -> Written(~Logical,~original,logical,values,source,source_position,source_within,target,position,within,piece,index,at):
  +cells = Tiled.cell_shape(values,target,position)
  write_wrote(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,at,from_low,to_high,same_blocked(values,source,source_position,source_within,target,position,within,piece,within_value),
    same_lows(values,target,position,piece),boxes_count(values,target,position,piece),Limits.double_bound(limits,ShapeLaws.elements(cells),cells_room),
    cells_rank_room(limits,values,target,position,within,within_value,rank_room),piece_facts(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,at,from_low,to_high))

# ---- The writes of a block of a piece ----

# A write that hits the query writes the target value of the query.
def hit_sound(~Logical: Data,~original: Logical -> U32 -> F32,hit: Bool,+logical: Logical,+geometry: Cells.Cells,+first: U32,+address: U32,+query: U32,+value: Tile.Tile8,
  +hit_value: {U32.is_eq(address,query) == hit : Bool},
  +written: {value == Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},(first + address : U32)) : Tile.Tile8}) ->
  {Bool.pick(Tile.Tile8,hit,value,Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},(first + query : U32))) ==
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},(first + query : U32)) : Tile.Tile8}:
  match hit:
    case True{}:
      Equal.trans(Tile.Tile8,value,Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},(first + address : U32)),
        Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},(first + query : U32)),written,
        Equal.cong(U32,Tile.Tile8,word => Representation.lifted(~Logical,~original,Representation.Lifted{logical,geometry},(first + word : U32)),address,query,
          Arithmetic.equal_of_is_eq(address,query,hit_value)))
    case False{}: {==}

def self_zero(+low: U32) -> {(low - low : U32) == 0 : U32}:
  Equal.trans(U32,(low - low : U32),((low + 0 : U32) - low : U32),0,
    Equal.cong(U32,U32,word => (word - low : U32),low,(low + 0 : U32),Equal.sym(U32,(low + 0 : U32),low,Arithmetic.add_zero(low))),Arithmetic.subtract_first(low,0))

# The first write lands on the target of the block, and each next one a
# stride further.
def first_address(+target: U32,+low: U32,+stride: U32) -> {target == (target + ((low - low : U32) * stride : U32) : U32) : U32}:
  Equal.sym(U32,(target + ((low - low : U32) * stride : U32) : U32),target,
    Equal.trans(U32,(target + ((low - low : U32) * stride : U32) : U32),(target + (0 * stride : U32) : U32),target,
      Equal.cong(U32,U32,word => (target + (word * stride : U32) : U32),(low - low : U32),0,self_zero(low)),
      Equal.trans(U32,(target + (0 * stride : U32) : U32),(target + 0 : U32),target,
        Equal.cong(U32,U32,word => (target + word : U32),(0 * stride : U32),0,Arithmetic.multiply_zero_left(stride)),Arithmetic.add_zero(target))))

def next_address(+target: U32,+low: U32,+stride: U32,+at: U32) ->
  {((target + ((at - low : U32) * stride : U32) : U32) + stride : U32) == (target + (((at + 1 : U32) - low : U32) * stride : U32) : U32) : U32}:
  +offset = (at - low : U32)
  Equal.trans(U32,((target + (offset * stride : U32) : U32) + stride : U32),(target + ((offset * stride : U32) + stride : U32) : U32),
    (target + (((at + 1 : U32) - low : U32) * stride : U32) : U32),Arithmetic.add_associative(target,(offset * stride : U32),stride),
    Equal.cong(U32,U32,word => (target + word : U32),((offset * stride : U32) + stride : U32),(((at + 1 : U32) - low : U32) * stride : U32),
      Equal.trans(U32,((offset * stride : U32) + stride : U32),((offset + 1 : U32) * stride : U32),(((at + 1 : U32) - low : U32) * stride : U32),
        Equal.sym(U32,((offset + 1 : U32) * stride : U32),((offset * stride : U32) + stride : U32),Arithmetic.multiply_successor(offset,stride)),
        Equal.cong(U32,U32,word => (word * stride : U32),(offset + 1 : U32),((at + 1 : U32) - low : U32),Arithmetic.add_after_subtract(at,low,1)))))

# The lane after a write within its block lies past the first lane and
# lands a stride further.
def next_lane_low(+limits: Machine.Limits,+low: U32,+at: U32,+from_low: {Nat.is_le(U32.to_nat(low),U32.to_nat(at)) == True{} : Bool},
  +small: {Nat.is_lt(U32.to_nat(at),8n) == True{} : Bool}) -> {Nat.is_le(U32.to_nat(low),U32.to_nat((at + 1 : U32))) == True{} : Bool}:
  +half = Machine.half_word(limits)
  +lane = U32.to_nat(at)
  +next_room = Order.transitive(Nat.add(lane,1n),8n,U32.to_nat(half),
    {%Equal.sym(Nat,Nat.add(lane,1n),1n+lane,Addition.successor_width(lane)) : {Nat.is_le(_,8n) == True{} : Bool}
      Order.strict_follow(lane,8n,small) : {Nat.is_le(Nat.add(lane,1n),8n) == True{} : Bool}},
    Order.transitive(8n,16n,U32.to_nat(half),{==},Bounds.as_nat_le(16,half,Limits.sixteen_within_half(limits))))
  +next_lane = Arithmetic.sum_exact(half,Limits.half_word_fits(limits),at,1,lane,1n,{==},{==},next_room)
  %Equal.sym(Nat,U32.to_nat((at + 1 : U32)),Nat.add(lane,1n),next_lane) : {Nat.is_le(U32.to_nat(low),_) == True{} : Bool}
  Order.transitive(U32.to_nat(low),lane,Nat.add(lane,1n),from_low,Order.le_add(lane,1n))

def next_placed_at(+address: U32,+target: U32,+low: U32,+stride: U32,+at: U32,+placed: {address == (target + ((at - low : U32) * stride : U32) : U32) : U32}) ->
  {(address + stride : U32) == (target + (((at + 1 : U32) - low : U32) * stride : U32) : U32) : U32}:
  Equal.trans(U32,(address + stride : U32),((target + ((at - low : U32) * stride : U32) : U32) + stride : U32),(target + (((at + 1 : U32) - low : U32) * stride : U32) : U32),
    Equal.cong(U32,U32,word => (word + stride : U32),address,(target + ((at - low : U32) * stride : U32) : U32),placed),next_address(target,low,stride,at))

# Every write of a block of a piece from a lane on that hits the query writes
# the target value of the query.
def writes_sound(~Logical: Data,~original: Logical -> U32 -> F32,fuel: Nat,more: Bool,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32,+at: U32,+address: U32,+query: U32,
  +more_value: {more == U32.is_lt(at,Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))) : Bool},
  +from_low: {Nat.is_le(U32.to_nat(Writes.low_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))),U32.to_nat(at)) == True{} : Bool},
  +placed: {address == Writes.write_address(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,at) : U32}) ->
  Block.WritesSound(fuel,more,at,Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),address,Writes.stride_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),Tiled.transposed(Block.expected_eight(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},Values.reads_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),Geometry.source_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),
      Geometry.step_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)))),query,
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},(piece_first(values,target,position,piece) + query : U32))):
  match fuel more:
    case 0n +more: Unit{}
    case 1n+ +rest False{}: Unit{}
    case 1n+ +rest True{}:
      +block = piece_block(within,position,target,source,source_within,source_position,values,piece,index)
      +high = Writes.high_of(block)
      +low = Writes.low_of(block)
      +stride = Writes.stride_of(block)
      +eight = Block.expected_eight(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},Values.reads_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),Geometry.source_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),
      Geometry.step_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)))
      +first = piece_first(values,target,position,piece)
      +to_high = Bounds.as_nat_lt(at,high,Equal.sym(Bool,True{},U32.is_lt(at,high),more_value))
      +small = lane_small(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,at,from_low,to_high)
      +next_low = next_lane_low(limits,low,at,from_low,small)
      +next_placed = next_placed_at(address,Writes.target_of(block),low,stride,at,placed)
      +written = Equal.trans(Tile.Tile8,Block.eighth(at,Tiled.transposed(eight)),
        Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},
          (first + Writes.write_address(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,at) : U32)),
        Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},(first + address : U32)),
        piece_written(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,at,from_low,to_high),
        Equal.cong(U32,Tile.Tile8,word => Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},(first + word : U32)),
          Writes.write_address(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,at),address,Equal.sym(U32,address,Writes.write_address(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,at),placed)))
      Held.Both{hit_sound(~Logical,~original,U32.is_eq(address,query),logical,Public.geometry_of(values,target,position),first,address,query,
          Block.eighth(at,Tiled.transposed(eight)),{==},written),
        writes_sound(~Logical,~original,rest,U32.is_lt((at + 1 : U32),high),logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,(at + 1 : U32),(address + stride : U32),query,{==},next_low,next_placed)}

# A block is its fields.
def block_eta(+block: Tiled.Block) ->
  {block == Tiled.Block{Geometry.source_of(block),Writes.target_of(block),Writes.stride_of(block),Writes.low_of(block),Writes.high_of(block),Values.reads_of(block),
    Geometry.step_of(block)} : Tiled.Block}:
  match block:
    case Tiled.Block{+source,+target,+stride,+low,+high,+reads,+step}: {==}

# Every write of a block of a piece that hits the query writes the target
# value of the query.
def block_sound(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32,+query: U32) ->
  Blocks.BlockSound(piece_block(within,position,target,source,source_within,source_position,values,piece,index),Block.expected_eight(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},Values.reads_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),Geometry.source_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),
      Geometry.step_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))),query,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},(piece_first(values,target,position,piece) + query : U32))):
  +block = piece_block(within,position,target,source,source_within,source_position,values,piece,index)
  +low = Writes.low_of(block)
  %Equal.sym(Tiled.Block,block,Tiled.Block{Geometry.source_of(block),Writes.target_of(block),Writes.stride_of(block),low,Writes.high_of(block),Values.reads_of(block),
      Geometry.step_of(block)},block_eta(block)) :
    Blocks.BlockSound(_,Block.expected_eight(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},Values.reads_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),Geometry.source_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),
      Geometry.step_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))),query,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},(piece_first(values,target,position,piece) + query : U32)))
  writes_sound(~Logical,~original,8n,U32.is_lt(low,Writes.high_of(block)),logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,low,Writes.target_of(block),query,{==},Order.reflexive(U32.to_nat(low)),
    first_address(Writes.target_of(block),low,Writes.stride_of(block)))

# ---- Every write of a block lands inside the piece ----

def inside_of_facts(+limits: Machine.Limits,+blocked: List<&2,Tiled.Blocked>,+lows: List<&2,U32>,+boxes: List<&2,U32>,+span: U32,+index: U32,+at: U32,
  +box_cells: {ShapeLaws.elements(boxes) == U32.to_nat(span) : Nat},
  +boxes_room: {Nat.is_le(ShapeLaws.elements(boxes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +facts: Writes.WriteFacts(blocked,lows,boxes,index,at)) -> {U32.is_lt(Writes.write_address(Writes.piece_axes(blocked,lows,boxes),index,at),span) == True{} : Bool}:
  Held.Both{+address,Held.Both{+box_below,Held.Both{+shifted,+small}}} = facts
  +axes = Writes.piece_axes(blocked,lows,boxes)
  +box = Writes.box_digits(axes,index,U32.to_nat(at))
  +number = ShapeLaws.digits_value(boxes,box)
  +written = Writes.write_address(axes,index,at)
  +below = Coordinates.digits_below(boxes,box,box_below)
  +room = Order.transitive(number,ShapeLaws.elements(boxes),U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(number,ShapeLaws.elements(boxes),below),boxes_room)
  +exact = Equal.trans(Nat,U32.to_nat(written),U32.to_nat(U32.from_nat(number)),number,Equal.cong(U32,Nat,word => U32.to_nat(word),written,U32.from_nat(number),address),
    Bounds.roundtrip(number,Machine.capacity(limits),Limits.capacity_fits(limits),room))
  Equal.trans(Bool,U32.is_lt(written,span),Nat.is_lt(U32.to_nat(written),U32.to_nat(span)),True{},Comparison.lt(written,span),
    {%Equal.sym(Nat,U32.to_nat(written),number,exact) : {Nat.is_lt(_,U32.to_nat(span)) == True{} : Bool}
      {%box_cells : {Nat.is_lt(number,_) == True{} : Bool} below : {Nat.is_lt(number,U32.to_nat(span)) == True{} : Bool}} :
      {Nat.is_lt(U32.to_nat(written),U32.to_nat(span)) == True{} : Bool}})

# A write of a block of a piece lands inside the piece.
def address_inside(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32,+at: U32,
  +from_low: {Nat.is_le(U32.to_nat(Writes.low_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))),U32.to_nat(at)) == True{} : Bool},
  +to_high: {Nat.is_lt(U32.to_nat(at),U32.to_nat(Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)))) == True{} : Bool}) -> {U32.is_lt(Writes.write_address(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,at),piece_span(values,target,position,piece)) == True{} : Bool}:
  +cells = Tiled.cell_shape(values,target,position)
  +grid = piece_grid(values,target,position)
  +first = piece_first(values,target,position,piece)
  +span = piece_span(values,target,position,piece)
  +boxes = piece_boxes(values,target,position,piece)
  +lows = piece_lows(values,target,position,piece)
  +blocked = Values.values_blocked(within,position,target,source,source_within,source_position,values)
  +source_cells = Tiled.cell_shape(values,source,source_position)
  +capacity = Machine.capacity(limits)
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +rank = List.length(&2,U32,values)
  +rank_cells = List.length(&2,U32,cells)
  +axes = piece_axes(within,position,target,source,source_within,source_position,values,piece)
  +cells_rank = cells_rank_room(limits,values,target,position,within,within_value,rank_room)
  +steps = BoxDigits.unit_steps(rank_cells,0n)
  +boxes_room = Order.transitive(ShapeLaws.elements(boxes),ShapeLaws.elements(cells),U32.to_nat(capacity),
    BoxDigits.piece_fits(cells,steps,boxes,U32.to_nat(first),U32.to_nat(span),0n,
      Box.grid_piece_box(limits,cells,steps,Grid.piece_target(),cells_room,nonempty,cells_rank,piece,piece_inside,0n,span_positive)),cells_room)
  inside_of_facts(limits,blocked,lows,boxes,span,index,at,box_cells,boxes_room,piece_facts(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,at,from_low,to_high))

# Every write of a block of a piece from a lane on lands inside the piece.
def writes_inside(~Logical: Data,~original: Logical -> U32 -> F32,fuel: Nat,more: Bool,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32,+at: U32,+address: U32,
  +more_value: {more == U32.is_lt(at,Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))) : Bool},
  +from_low: {Nat.is_le(U32.to_nat(Writes.low_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index))),U32.to_nat(at)) == True{} : Bool},
  +placed: {address == Writes.write_address(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,at) : U32}) ->
  Block.WritesInside(fuel,more,at,Writes.high_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),address,Writes.stride_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),piece_span(values,target,position,piece)):
  match fuel more:
    case 0n +more: Unit{}
    case 1n+ +rest False{}: Unit{}
    case 1n+ +rest True{}:
      +block = piece_block(within,position,target,source,source_within,source_position,values,piece,index)
      +high = Writes.high_of(block)
      +low = Writes.low_of(block)
      +stride = Writes.stride_of(block)
      +span = piece_span(values,target,position,piece)
      +to_high = Bounds.as_nat_lt(at,high,Equal.sym(Bool,True{},U32.is_lt(at,high),more_value))
      +small = lane_small(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,at,from_low,to_high)
      Held.Both{{%Equal.sym(U32,address,Writes.write_address(piece_axes(within,position,target,source,source_within,source_position,values,piece),index,at),placed) : {U32.is_lt(_,span) == True{} : Bool}
          address_inside(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,at,from_low,to_high) : {U32.is_lt(address,span) == True{} : Bool}},
        writes_inside(~Logical,~original,rest,U32.is_lt((at + 1 : U32),high),logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,(at + 1 : U32),(address + stride : U32),{==},
          next_lane_low(limits,low,at,from_low,small),next_placed_at(address,Writes.target_of(block),low,stride,at,placed))}

# Every write of a block of a piece lands inside the piece.
def block_inside(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32) -> Blocks.BlockInside(piece_block(within,position,target,source,source_within,source_position,values,piece,index),piece_span(values,target,position,piece)):
  +block = piece_block(within,position,target,source,source_within,source_position,values,piece,index)
  +low = Writes.low_of(block)
  %Equal.sym(Tiled.Block,block,Tiled.Block{Geometry.source_of(block),Writes.target_of(block),Writes.stride_of(block),Writes.low_of(block),Writes.high_of(block),Values.reads_of(block),
      Geometry.step_of(block)},block_eta(block)) : Blocks.BlockInside(_,piece_span(values,target,position,piece))
  writes_inside(~Logical,~original,8n,U32.is_lt(low,Writes.high_of(block)),logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,low,Writes.target_of(block),{==},Order.reflexive(U32.to_nat(low)),
    first_address(Writes.target_of(block),low,Writes.stride_of(block)))

# ---- The blocks of a piece ----

# Every cell the blocks from index on read is held.
def BlocksHeld(fuel: Nat,+index: U32,+axes: List<&2,Tiled.Ranged>,+spans: Held.Spans) -> Data:
  match fuel:
    case 0n: Unit
    case 1n+rest:
      +block = Blocks.block_of(axes,index)
      Held.Both<Block.ReadsHeld<spans,Values.reads_of(block),Geometry.source_of(block),Geometry.step_of(block)>,BlocksHeld(rest,(index + 1 : U32),axes,spans)>

# Every block of a piece from index on writes inside the piece, and every
# write that hits the query writes the target value of the query, on pieces
# in the order of the start that hold every cell the blocks read with its
# value.
def blocks_sound(~Logical: Data,~original: Logical -> U32 -> F32,fuel: Nat,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+index: U32,+query: U32,+holding: Held.Holding<Tile.Tile8>,
  +start: Held.Holding<Tile.Tile8>,+low: Nat,+high: Nat,
  +same_order: {Held.order(~Tile.Tile8,holding) == Held.order(~Tile.Tile8,start) : Held.Stack<Tile.Tile8>},
  +sorted: Held.Sorted(~Tile.Tile8,Held.order(~Tile.Tile8,start),low,high),
  +bounded: {Nat.is_le(high,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +held_values: Held.HoldingValues(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},holding),
  +held: BlocksHeld(fuel,index,piece_axes(within,position,target,source,source_within,source_position,values,piece),Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,start))),
  +step_room: {Nat.is_le(Nat.double(U32.to_nat(Geometry.blocked_step(Geometry.blocked_of(piece_axes(within,position,target,source,source_within,source_position,values,piece)),0))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  Blocks.BlocksSound(fuel,index,piece_axes(within,position,target,source,source_within,source_position,values,piece),holding,piece_span(values,target,position,piece),query,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},(piece_first(values,target,position,piece) + query : U32))):
  match fuel:
    case 0n: Unit{}
    case 1n+ +rest:
      Held.Both{+here,+later} = held
      +block = piece_block(within,position,target,source,source_within,source_position,values,piece,index)
      +context = {Representation.Lifted{logical,Public.geometry_of(values,source,source_position)} : Representation.Lifted<Logical>}
      +eight = Block.expected_eight(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},Values.reads_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),Geometry.source_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)),
      Geometry.step_of(piece_block(within,position,target,source,source_within,source_position,values,piece,index)))
      +reads = Values.reads_of(block)
      +source_at = Geometry.source_of(block)
      +step = Geometry.step_of(block)
      +back = Equal.sym(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,holding),Held.order(~Tile.Tile8,start),same_order)
      +sorted_here = Held.sorted_moved(~Tile.Tile8,Held.order(~Tile.Tile8,start),Held.order(~Tile.Tile8,holding),low,high,back,sorted)
      +held_here = {%back : Block.ReadsHeld<Held.spans(~Tile.Tile8,_),reads,source_at,step>
        here : Block.ReadsHeld<Held.spans(~Tile.Tile8,Held.order(~Tile.Tile8,holding)),reads,source_at,step>}
      +step_here = {%Equal.sym(U32,Geometry.step_of(block),Geometry.blocked_step(Geometry.blocked_of(piece_axes(within,position,target,source,source_within,source_position,values,piece)),Geometry.step_of(Writes.first_block())),
            Geometry.block_step(piece_axes(within,position,target,source,source_within,source_position,values,piece),Writes.first_block(),index)) :
          {Nat.is_le(Nat.double(U32.to_nat(_)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
        step_room : {Nat.is_le(Nat.double(U32.to_nat(step)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}
      +eta = Equal.sym(Tiled.Block,block,Tiled.Block{Geometry.source_of(block),Writes.target_of(block),Writes.stride_of(block),Writes.low_of(block),Writes.high_of(block),Values.reads_of(block),
      Geometry.step_of(block)},block_eta(block))
      +eight_same = {%eta : {Block.block_eight(_,holding) == eight : Tiled.Eight}
        Block.block_cells(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,reads,source_at,step,holding,low,high,Machine.capacity(limits),
          Limits.capacity_fits(limits),bounded,sorted_here,held_here,held_values,Machine.half_word(limits),Limits.half_word_fits(limits),step_here) : {Block.block_eight(block,holding) == eight : Tiled.Eight}}
      +sound_here = {%Equal.sym(Tiled.Eight,Block.block_eight(block,holding),eight,eight_same) : Blocks.BlockSound(block,_,query,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},(piece_first(values,target,position,piece) + query : U32)))
        block_sound(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,query) : Blocks.BlockSound(block,Block.block_eight(block,holding),query,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},(piece_first(values,target,position,piece) + query : U32)))}
      +next = Block.block_holding(block,holding)
      +next_order = Equal.trans(Held.Stack<Tile.Tile8>,Held.order(~Tile.Tile8,next),Held.order(~Tile.Tile8,holding),Held.order(~Tile.Tile8,start),
        {%eta : {Held.order(~Tile.Tile8,Block.block_holding(_,holding)) == Held.order(~Tile.Tile8,holding) : Held.Stack<Tile.Tile8>}
          Block.block_order(reads,source_at,step,holding) : {Held.order(~Tile.Tile8,next) == Held.order(~Tile.Tile8,holding) : Held.Stack<Tile.Tile8>}},same_order)
      +next_values = {%eta : Held.HoldingValues(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,Block.block_holding(_,holding))
        Block.block_values(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,reads,source_at,step,holding,held_values) :
        Held.HoldingValues(~Tile.Tile8,~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),context,next)}
      Held.Both{block_inside(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index),Held.Both{sound_here,
        blocks_sound(~Logical,~original,rest,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,(index + 1 : U32),query,next,start,low,high,next_order,sorted,bounded,next_values,later,step_room)}}

# ---- Every cell of a piece is written ----

def cover_of_rooms(+blocked: List<&2,Tiled.Blocked>,+lows: List<&2,U32>,+boxes: List<&2,U32>,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +same_blocked: {List.length(&2,Tiled.Blocked,blocked) == List.length(&2,U32,boxes) : Nat},+same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,boxes) : Nat},
  +eight_room: {Nat.is_le(8n,U32.to_nat(cap)) == True{} : Bool},+query: U32,+query_below: {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(boxes)) == True{} : Bool},
  +rooms: PieceRooms(blocked,lows,boxes,cap)) ->
  {Blocks.blocks_hit(U32.to_nat(Tiled.ranged_head_count(Writes.piece_axes(blocked,lows,boxes))),0,Writes.piece_axes(blocked,lows,boxes),query) == True{} : Bool}:
  Held.Both{+most,Held.Both{+positive,Held.Both{+fits,+boxes_room}}} = rooms
  Cover.cover_hit(blocked,lows,boxes,cap,cap_ok,same_blocked,same_lows,most,positive,fits,eight_room,boxes_room,query,query_below)

# Every cell of a target piece is hit by the blocks of the piece.
def piece_cover(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+piece: U32,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,target,position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,target,position))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(piece_boxes(values,target,position,piece)) == U32.to_nat(piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool},+query: U32,
  +query_below: {Nat.is_lt(U32.to_nat(query),U32.to_nat(piece_span(values,target,position,piece))) == True{} : Bool}) ->
  {Blocks.blocks_hit(U32.to_nat(Tiled.ranged_head_count(piece_axes(within,position,target,source,source_within,source_position,values,piece))),0,piece_axes(within,position,target,source,source_within,source_position,values,piece),query) == True{} : Bool}:
  +half = Machine.half_word(limits)
  cover_of_rooms(Values.values_blocked(within,position,target,source,source_within,source_position,values),piece_lows(values,target,position,piece),piece_boxes(values,target,position,piece),half,Limits.half_word_fits(limits),
    same_blocked(values,source,source_position,source_within,target,position,within,piece,within_value),same_lows(values,target,position,piece),
    Order.transitive(8n,16n,U32.to_nat(half),{==},Bounds.as_nat_le(16,half,Limits.sixteen_within_half(limits))),query,
    {%Equal.sym(Nat,ShapeLaws.elements(piece_boxes(values,target,position,piece)),U32.to_nat(piece_span(values,target,position,piece)),box_cells) : {Nat.is_lt(U32.to_nat(query),_) == True{} : Bool}
      query_below : {Nat.is_lt(U32.to_nat(query),ShapeLaws.elements(piece_boxes(values,target,position,piece))) == True{} : Bool}},
    piece_rooms(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive))
