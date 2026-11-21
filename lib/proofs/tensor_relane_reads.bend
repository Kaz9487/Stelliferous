# The source cells a relane block of a target piece reads lie in the box its
# component takes of the source: every target cell axis reads its source
# axis at the digit of the block on it, the groups at the eight lanes of the
# group, the source lane axis at the group of the lane.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tasks.bend as Tasks
import ./tensor_relane_geometry.bend as Geometry
import ./tensor_relane_values.bend as Values
import ./tensor_task_reads.bend as TaskReads
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./word_arithmetic.bend as WordArithmetic
import ./tensor_components.bend as Components
import ./tensor_target_cells.bend as Targets
import ./row_major_index.bend as RowMajor
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_extrema.bend as NatExtrema
import ./u32_extrema.bend as WordExtrema
import ./nat_ceiling.bend as NatCeiling
import ./boolean_logic.bend as Logic
import ./nat_to_u32_bounds.bend as Bounds
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_tiled_public.bend as Public
import ./encoded_coordinates.bend as Encoded
import ./machine_limit_bounds.bend as Limits
import ../machine_limits.bend as Machine
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_box_cells.bend as BoxCells
import ./tensor_moved.bend as Moved
import ./tensor_relane_piece.bend as Piece
import ./tensor_view_reads.bend as ViewReads
import ./tensor_relane_writes.bend as Writes
import ../tensor.bend as Tensors
import ./tensor_relane_block.bend as Block
import ./tensor_box_digits.bend as BoxDigits
import ./tensor_move_readback.bend as Held
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./word_encoding.bend as WordEncoding

# ---- The extents of the target cell axes ----

# The extent of every target cell axis: a logical axis the extent of the
# axis, the groups the groups of the target lane axis.
def AxesExtents(axes: List<&2,Tiled.CellAxis>,extents: List<&2,U32>,+shape: List<&2,U32>,+target: Nat) -> Data:
  match axes extents:
    case Con{Tiled.OfAxis{+axis},axes} Con{+extent,extents}:
      Held.Both<{extent == Grid.lanes_at(shape,axis) : U32},AxesExtents(axes,extents,shape,target)>
    case Con{Tiled.OfGroups{},axes} Con{+extent,extents}:
      Held.Both<{extent == Grid.groups(Grid.lanes_at(shape,target)) : U32},AxesExtents(axes,extents,shape,target)>
    case Nil{} Nil{}:
      Unit
    case axes extents:
      Empty

# The extents of a shape from an axis on.
def ShapeFrom(+shape: List<&2,U32>,+index: Nat,tail: List<&2,U32>) -> Data:
  match tail:
    case Nil{}: Unit
    case Con{+extent,rest}: Held.Both<{Grid.lanes_at(shape,index) == extent : U32},ShapeFrom(shape,1n+index,rest)>

def shape_lift(tail: List<&2,U32>,+extent: U32,+shape: List<&2,U32>,+index: Nat,+from: ShapeFrom(shape,index,tail)) -> ShapeFrom(Con{extent,shape},1n+index,tail):
  match tail:
    case Nil{}: Unit{}
    case Con{+head,+rest}:
      Held.Both{+here,+later} = from
      Held.Both{here,shape_lift(rest,extent,shape,1n+index,later)}

def shape_self(shape: List<&2,U32>) -> ShapeFrom(shape,0n,shape):
  match shape:
    case Nil{}: Unit{}
    case Con{+extent,+rest}: Held.Both{{==},shape_lift(rest,extent,rest,0n,shape_self(rest))}

# Every logical axis from index on has the extent of its axis.
def every_extents(remaining: Nat,+index: Nat,tail: List<&2,U32>,+shape: List<&2,U32>,+target: Nat,+from: ShapeFrom(shape,index,tail),
  +count: {List.length(&2,U32,tail) == remaining : Nat}) -> AxesExtents(Geometry.every_axis(remaining,index),tail,shape,target):
  match remaining tail:
    case 0n Nil{}: Unit{}
    case 1n+ +rest Con{+extent,+more}:
      Held.Both{+here,+later} = from
      Held.Both{Equal.sym(U32,Grid.lanes_at(shape,index),extent,here),
        every_extents(rest,1n+index,more,shape,target,later,Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,U32,more),1n+rest,count))}
    case 0n Con{+extent,+more}: Empty.absurd(AxesExtents(Geometry.every_axis(0n,index),Con{extent,more},shape,target),
      BoxDigits.zero_successor(List.length(&2,U32,more),Equal.sym(Nat,1n+List.length(&2,U32,more),0n,count)))
    case 1n+ +rest Nil{}: Empty.absurd(AxesExtents(Geometry.every_axis(1n+rest,index),Nil{},shape,target),BoxDigits.zero_successor(rest,count))

# The logical axes from index on but one a gap later have the extents of
# their axes.
def skip_extents(remaining: Nat,gap: Nat,+index: Nat,+skip: Nat,tail: List<&2,U32>,+shape: List<&2,U32>,+target: Nat,+from: ShapeFrom(shape,index,tail),
  +count: {List.length(&2,U32,tail) == remaining : Nat},+at: {Nat.add(index,gap) == skip : Nat},+inside: {Nat.is_lt(gap,remaining) == True{} : Bool}) ->
  AxesExtents(Tiled.axes_from(remaining,index,skip),Grid.removed(tail,gap),shape,target):
  match remaining gap tail:
    case 1n+ +rest 0n Con{+extent,+more}:
      Held.Both{+here,+later} = from
      +same = Equal.trans(Nat,index,Nat.add(index,0n),skip,Equal.sym(Nat,Nat.add(index,0n),index,Addition.zero_right(index)),at)
      +hit = {%same : {Nat.is_eq(index,_) == True{} : Bool} Order.equal_reflexive(index) : {Nat.is_eq(index,skip) == True{} : Bool}}
      +past = {%same : {Nat.is_lt(_,1n+index) == True{} : Bool} Order.lt_succ(index,index,Order.reflexive(index)) : {Nat.is_lt(skip,1n+index) == True{} : Bool}}
      %Equal.sym(Bool,Nat.is_eq(index,skip),True{},hit) :
        AxesExtents(Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(rest,1n+index,skip),Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,skip)),more,shape,target)
      %Equal.sym(List<&2,Tiled.CellAxis>,Tiled.axes_from(rest,1n+index,skip),Geometry.every_axis(rest,1n+index),Geometry.never_hit(rest,1n+index,skip,past)) :
        AxesExtents(_,more,shape,target)
      every_extents(rest,1n+index,more,shape,target,later,Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,U32,more),1n+rest,count))
    case 1n+ +rest 1n+ +further Con{+extent,+more}:
      Held.Both{+here,+later} = from
      +far = Nat.add(index,1n+further)
      +same = Equal.trans(Nat,far,1n+Nat.add(index,further),skip,Addition.successor_right(index,further),
        Equal.trans(Nat,1n+Nat.add(index,further),far,skip,Equal.sym(Nat,far,1n+Nat.add(index,further),Addition.successor_right(index,further)),at))
      +miss = {%at : {Nat.is_eq(index,_) == False{} : Bool}
        Order.less_not_equal(index,far,Order.below_sum(index,further)) : {Nat.is_eq(index,skip) == False{} : Bool}}
      %Equal.sym(Bool,Nat.is_eq(index,skip),False{},miss) :
        AxesExtents(Bool.pick(List<&2,Tiled.CellAxis>,_,Tiled.axes_from(rest,1n+index,skip),Tiled.OfAxis{index} <> Tiled.axes_from(rest,1n+index,skip)),
          Con{extent,Grid.removed(more,further)},shape,target)
      Held.Both{Equal.sym(U32,Grid.lanes_at(shape,index),extent,here),
        skip_extents(rest,further,1n+index,skip,more,shape,target,later,
          Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,U32,more),1n+rest,count),
          Equal.trans(Nat,Nat.add(1n+index,further),far,skip,Equal.sym(Nat,far,1n+Nat.add(index,further),Addition.successor_right(index,further)),at),inside)}
    case 0n +gap +tail:
      Empty.absurd(AxesExtents(Tiled.axes_from(0n,index,skip),Grid.removed(tail,gap),shape,target),Order.below_zero(gap,inside))
    case 1n+ +rest +gap Nil{}:
      Empty.absurd(AxesExtents(Tiled.axes_from(1n+rest,index,skip),Grid.removed(Nil{},gap),shape,target),BoxDigits.zero_successor(rest,count))

def removed_past(shape: List<&2,U32>,axis: Nat,+past: {Nat.is_le(List.length(&2,U32,shape),axis) == True{} : Bool}) -> {Grid.removed(shape,axis) == shape : List<&2,U32>}:
  match shape axis:
    case Nil{} +axis: {==}
    case Con{+extent,+rest} 0n: Empty.absurd({Grid.removed(Con{extent,rest},0n) == Con{extent,rest} : List<&2,U32>},WordEncoding.false_true(past))
    case Con{+extent,+rest} 1n+ +next:
      Equal.cong(List<&2,U32>,List<&2,U32>,list => extent <> list,Grid.removed(rest,next),rest,removed_past(rest,next,past))

def extents_append(axes: List<&2,Tiled.CellAxis>,extents: List<&2,U32>,+shape: List<&2,U32>,+target: Nat,+last: U32,
  +last_value: {last == Grid.groups(Grid.lanes_at(shape,target)) : U32},+before: AxesExtents(axes,extents,shape,target)) ->
  AxesExtents(List.append(&2,Tiled.CellAxis,axes,[Tiled.OfGroups{}]),List.append(&2,U32,extents,[last]),shape,target):
  match axes extents:
    case Con{Tiled.OfAxis{+axis},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = before
      Held.Both{here,extents_append(rest,more,shape,target,last,last_value,later)}
    case Con{Tiled.OfGroups{},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = before
      Held.Both{here,extents_append(rest,more,shape,target,last,last_value,later)}
    case Nil{} Nil{}:
      Held.Both{last_value,Unit{}}
    case Nil{} Con{+extent,+more}:
      Empty.absurd(AxesExtents(List.append(&2,Tiled.CellAxis,Nil{},[Tiled.OfGroups{}]),List.append(&2,U32,Con{extent,more},[last]),shape,target),before)
    case Con{+axis,+rest} Nil{}:
      Empty.absurd(AxesExtents(List.append(&2,Tiled.CellAxis,Con{axis,rest},[Tiled.OfGroups{}]),List.append(&2,U32,Nil{},[last]),shape,target),before)

def extents_at(within: Bool,position: Grid.Position,+target: Nat,+shape: List<&2,U32>,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,shape)) == within : Bool}) ->
  AxesExtents(Tiled.target_axes(within,position,target,List.length(&2,U32,shape)),
    Grid.placed(within,position,Grid.removed(shape,target),Grid.groups(Grid.lanes_at(shape,target))),shape,target):
  match within position:
    case False{} +position:
      +rank = List.length(&2,U32,shape)
      %Equal.sym(List<&2,U32>,Grid.removed(shape,target),shape,removed_past(shape,target,Order.not_less(target,rank,within_value))) :
        AxesExtents(Tiled.axes_from(rank,0n,rank),_,shape,target)
      %Equal.sym(List<&2,Tiled.CellAxis>,Tiled.axes_from(rank,0n,rank),Geometry.every_axis(rank,0n),Geometry.never_reach(rank,0n,rank,Order.reflexive(rank))) :
        AxesExtents(_,shape,shape,target)
      every_extents(rank,0n,shape,shape,target,shape_self(shape),{==})
    case True{} Grid.Innermost{}:
      +rank = List.length(&2,U32,shape)
      +groups = Grid.groups(Grid.lanes_at(shape,target))
      extents_append(Tiled.axes_from(rank,0n,target),Grid.removed(shape,target),shape,target,groups,{==},
        skip_extents(rank,target,0n,target,shape,shape,target,shape_self(shape),{==},{==},within_value))
    case True{} Grid.Outermost{}:
      +rank = List.length(&2,U32,shape)
      Held.Both{{==},skip_extents(rank,target,0n,target,shape,shape,target,shape_self(shape),{==},{==},within_value)}

# Every target cell axis has the extent of its logical axis, the groups those
# of the target lane axis.
def target_extents(within: Bool,position: Grid.Position,+target: Nat,+shape: List<&2,U32>,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,shape)) == within : Bool}) ->
  AxesExtents(Tiled.target_axes(within,position,target,List.length(&2,U32,shape)),Tiled.cell_shape(shape,target,position),shape,target):
  %Equal.sym(Bool,Nat.is_lt(target,List.length(&2,U32,shape)),within,within_value) :
    AxesExtents(Tiled.target_axes(within,position,target,List.length(&2,U32,shape)),
      Grid.placed(_,position,Grid.removed(shape,target),Grid.groups(Grid.lanes_at(shape,target))),shape,target)
  extents_at(within,position,target,shape,within_value)

# ---- The source cell a block reads ----

# The value every target cell axis reads its source axis at, for the target
# digits of a block and the target lane read: an axis both sides share at
# its digit, the source lane axis at the group of its digit, the target
# groups at the index of the target lane axis.
def relane_values(blocked: List<&2,Tiled.Blocked>,digits: List<&2,Nat>,+read: Nat) -> List<&2,Nat>:
  match blocked digits:
    case Con{Tiled.Blocked{Tiled.Shared{},source,lanes},rest} Con{+digit,more}: digit <> relane_values(rest,more,read)
    case Con{Tiled.Blocked{Tiled.Lanes{},source,lanes},rest} Con{+digit,more}: Nat.div(digit,8n) <> relane_values(rest,more,read)
    case Con{Tiled.Blocked{Tiled.Groups{},source,lanes},rest} Con{+digit,more}: Nat.add(Nat.mul(digit,8n),read) <> relane_values(rest,more,read)
    case blocked digits: []

# The step that reaches every target groups axis is zero: no target groups
# come before it.
def GroupsWhen(blocked: List<&2,Tiled.Blocked>,+step: U32) -> Data:
  match blocked:
    case Nil{}: Unit
    case Con{Tiled.Blocked{Tiled.Groups{},+source,lanes},rest}: Held.Both<{step == 0 : U32},GroupsWhen(rest,source)>
    case Con{Tiled.Blocked{role,source,lanes},rest}: GroupsWhen(rest,step)

def none_when(blocked: List<&2,Tiled.Blocked>,+step: U32,+none: Values.NoGroups(blocked)) -> GroupsWhen(blocked,step):
  match blocked:
    case Nil{}: Unit{}
    case Con{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+rest}: Empty.absurd(GroupsWhen(Con{Tiled.Blocked{Tiled.Groups{},source,lanes},rest},step),none)
    case Con{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+rest}: none_when(rest,step,none)
    case Con{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+rest}: none_when(rest,step,none)

def append_when(blocked: List<&2,Tiled.Blocked>,+last: Tiled.Blocked,+step: U32,+none: Values.NoGroups(blocked),+at_last: GroupsWhen([last],step)) ->
  GroupsWhen(List.append(&2,Tiled.Blocked,blocked,[last]),step):
  match blocked:
    case Nil{}: at_last
    case Con{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+rest}:
      Empty.absurd(GroupsWhen(List.append(&2,Tiled.Blocked,Con{Tiled.Blocked{Tiled.Groups{},source,lanes},rest},[last]),step),none)
    case Con{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+rest}: append_when(rest,last,step,none,at_last)
    case Con{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+rest}: append_when(rest,last,step,none,at_last)

# The target cell axes of a relane meet their one groups axis at step zero.
def target_when(within: Bool,position: Grid.Position,+target: Nat,+rank: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32) ->
  GroupsWhen(Tiled.blocked_list(Tiled.target_axes(within,position,target,rank),source,source_within,source_position,target,strides,shape,groups_axis),0):
  match within position:
    case False{} +position:
      none_when(Tiled.blocked_list(Tiled.axes_from(rank,0n,rank),source,source_within,source_position,target,strides,shape,groups_axis),0,
        Values.axes_groups(rank,0n,rank,source,source_within,source_position,target,strides,shape,groups_axis))
    case True{} Grid.Innermost{}:
      +axes = Tiled.axes_from(rank,0n,target)
      +last = Tiled.blocked_axis(Tiled.OfGroups{},source,source_within,source_position,target,strides,shape,groups_axis)
      %Equal.sym(List<&2,Tiled.Blocked>,Tiled.blocked_list(List.append(&2,Tiled.CellAxis,axes,[Tiled.OfGroups{}]),source,source_within,source_position,target,strides,shape,groups_axis),
          List.append(&2,Tiled.Blocked,Tiled.blocked_list(axes,source,source_within,source_position,target,strides,shape,groups_axis),[last]),
          Geometry.blocked_append(axes,Tiled.OfGroups{},source,source_within,source_position,target,strides,shape,groups_axis)) : GroupsWhen(_,0)
      append_when(Tiled.blocked_list(axes,source,source_within,source_position,target,strides,shape,groups_axis),last,0,
        Values.axes_groups(rank,0n,target,source,source_within,source_position,target,strides,shape,groups_axis),Held.Both{{==},Unit{}})
    case True{} Grid.Outermost{}:
      Held.Both{{==},none_when(Tiled.blocked_list(Tiled.axes_from(rank,0n,target),source,source_within,source_position,target,strides,shape,groups_axis),
        Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,target)),
        Values.axes_groups(rank,0n,target,source,source_within,source_position,target,strides,shape,groups_axis))}

# The source cell word the readings of the target cell axes reach at the
# values of a block, and the start and step of the block, from step on.
def WordTerms(+axes: List<&2,Tiled.CellAxis>,+extents: List<&2,U32>,+digits: List<&2,Nat>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +target: Nat,+source_cells: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+shape: List<&2,U32>,+index: U32,+read: Nat,+step: U32) -> Data:
  +blocked = Tiled.blocked_list(axes,source,source_within,source_position,target,Tasks.strides_of(source_cells),shape,Values.groups_axis(source_position,source_cells))
  {(TaskReads.placed_word(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,extents,index),
      relane_values(blocked,digits,read),Tasks.strides_of(source_cells)) + U32.from_nat(read) * step : U32) ==
    (Geometry.blocked_sum(digits,blocked) + U32.from_nat(read) * Geometry.blocked_step(blocked,step) : U32) : U32}

# A term before both sides of a sum.
def term_before(+term: U32,+left: U32,+right: U32,+left_extra: U32,+right_extra: U32,
  +later: {(left + left_extra : U32) == (right + right_extra : U32) : U32}) ->
  {((term + left : U32) + left_extra : U32) == ((term + right : U32) + right_extra : U32) : U32}:
  Equal.trans(U32,((term + left : U32) + left_extra : U32),(term + (left + left_extra : U32) : U32),((term + right : U32) + right_extra : U32),
    Arithmetic.add_associative(term,left,left_extra),
    Equal.trans(U32,(term + (left + left_extra : U32) : U32),(term + (right + right_extra : U32) : U32),((term + right : U32) + right_extra : U32),
      Equal.cong(U32,U32,sum => (term + sum : U32),(left + left_extra : U32),(right + right_extra : U32),later),
      Equal.sym(U32,((term + right : U32) + right_extra : U32),(term + (right + right_extra : U32) : U32),Arithmetic.add_associative(term,right,right_extra))))

# The target groups read the lane at the stride of the source axis they
# read: the lane term moves to the step.
def groups_terms(+part: U32,+lane: U32,+whole: U32,+stride: U32,+step: U32,+left: U32,+right: U32,+right_extra: U32,
  +whole_eq: {whole == (part + lane : U32) : U32},+zero: {step == 0 : U32},
  +later: {(left + lane * stride : U32) == (right + right_extra : U32) : U32}) ->
  {(((whole * stride : U32) + left : U32) + lane * step : U32) == (((part * stride : U32) + right : U32) + right_extra : U32) : U32}:
  %Equal.sym(U32,step,0,zero) : {(((whole * stride : U32) + left : U32) + lane * _ : U32) == (((part * stride : U32) + right : U32) + right_extra : U32) : U32}
  %Equal.sym(U32,whole,(part + lane : U32),whole_eq) : {(((_ * stride : U32) + left : U32) + lane * 0 : U32) == (((part * stride : U32) + right : U32) + right_extra : U32) : U32}
  +sum = ((part + lane : U32) * stride : U32)
  +front = (part * stride : U32)
  +back = (lane * stride : U32)
  Equal.trans(U32,((sum + left : U32) + lane * 0 : U32),(sum + left : U32),((front + right : U32) + right_extra : U32),
    Equal.trans(U32,((sum + left : U32) + lane * 0 : U32),((sum + left : U32) + 0 : U32),(sum + left : U32),
      Equal.cong(U32,U32,term => ((sum + left : U32) + term : U32),(lane * 0 : U32),0,Arithmetic.multiply_zero_right(lane)),Arithmetic.add_zero((sum + left : U32))),
    Equal.trans(U32,(sum + left : U32),((front + back : U32) + left : U32),((front + right : U32) + right_extra : U32),
      Equal.cong(U32,U32,term => (term + left : U32),sum,(front + back : U32),Arithmetic.multiply_additive(part,lane,stride)),
      Equal.trans(U32,((front + back : U32) + left : U32),(front + (back + left : U32) : U32),((front + right : U32) + right_extra : U32),
        Arithmetic.add_associative(front,back,left),
        Equal.trans(U32,(front + (back + left : U32) : U32),(front + (right + right_extra : U32) : U32),((front + right : U32) + right_extra : U32),
          Equal.cong(U32,U32,term => (front + term : U32),(back + left : U32),(right + right_extra : U32),
            Equal.trans(U32,(back + left : U32),(left + back : U32),(right + right_extra : U32),Arithmetic.add_commutative(back,left),later)),
          Equal.sym(U32,((front + right : U32) + right_extra : U32),(front + (right + right_extra : U32) : U32),Arithmetic.add_associative(front,right,right_extra))))))

# A logical target cell axis reads the source groups when it is the source
# lane axis, otherwise its own source axis, at the stride of its blocks.
def some_word(choice: Bool,+extent: U32,+block: U32,+grid: Grid.Grid,+lane_axis: U32,+lane_extent: U32,+shared_axis: U32,+shared_extent: U32,+lanes: U32,
  +strides: List<&2,U32>,+digit: Nat,+readings: List<&2,Tasks.Reading>,+blocked: List<&2,Tiled.Blocked>,+digits: List<&2,Nat>,+read: Nat,+step: U32,
  +later: {(TaskReads.placed_word(readings,relane_values(blocked,digits,read),strides) + U32.from_nat(read) * step : U32) ==
    (Geometry.blocked_sum(digits,blocked) + U32.from_nat(read) * Geometry.blocked_step(blocked,step) : U32) : U32}) ->
  {(TaskReads.placed_word(Tasks.Reading{extent,block,Bool.pick(Tasks.Source,choice,Tiled.read_at(grid,lane_axis,lane_extent,1,1,8),Tiled.read_at(grid,shared_axis,shared_extent,1,1,1))} <> readings,
      relane_values(Bool.pick(Tiled.Blocked,choice,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,lane_axis),lanes},Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,shared_axis),1}) <> blocked,
        digit <> digits,read),strides) + U32.from_nat(read) * step : U32) ==
    (Geometry.blocked_sum(digit <> digits,Bool.pick(Tiled.Blocked,choice,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,lane_axis),lanes},
        Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,shared_axis),1}) <> blocked) +
      U32.from_nat(read) * Geometry.blocked_step(Bool.pick(Tiled.Blocked,choice,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,lane_axis),lanes},
        Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,shared_axis),1}) <> blocked,step) : U32) : U32}:
  match choice:
    case True{}:
      term_before((U32.from_nat(Nat.div(digit,8n)) * Tasks.nth(strides,lane_axis) : U32),TaskReads.placed_word(readings,relane_values(blocked,digits,read),strides),
        Geometry.blocked_sum(digits,blocked),(U32.from_nat(read) * step : U32),(U32.from_nat(read) * Geometry.blocked_step(blocked,step) : U32),later)
    case False{}:
      term_before((U32.from_nat(digit) * Tasks.nth(strides,shared_axis) : U32),TaskReads.placed_word(readings,relane_values(blocked,digits,read),strides),
        Geometry.blocked_sum(digits,blocked),(U32.from_nat(read) * step : U32),(U32.from_nat(read) * Geometry.blocked_step(blocked,step) : U32),later)

def some_groups(choice: Bool,+lane_axis: U32,+shared_axis: U32,+lanes: U32,+strides: List<&2,U32>,+blocked: List<&2,Tiled.Blocked>,+step: U32,
  +groups: GroupsWhen(Bool.pick(Tiled.Blocked,choice,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,lane_axis),lanes},
    Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,shared_axis),1}) <> blocked,step)) -> GroupsWhen(blocked,step):
  match choice:
    case True{}: groups
    case False{}: groups

def word_terms(axes: List<&2,Tiled.CellAxis>,extents: List<&2,U32>,digits: List<&2,Nat>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +source_cells: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+shape: List<&2,U32>,+index: U32,+read: Nat,+step: U32,
  +sized: AxesExtents(axes,extents,shape,target),+paired: {List.length(&2,Nat,digits) == List.length(&2,Tiled.CellAxis,axes) : Nat},
  +groups: GroupsWhen(Tiled.blocked_list(axes,source,source_within,source_position,target,Tasks.strides_of(source_cells),shape,Values.groups_axis(source_position,source_cells)),step)) ->
  WordTerms(axes,extents,digits,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,step):
  match axes extents digits:
    case Con{Tiled.OfGroups{},+rest} Con{+extent,+more} Con{+digit,+later_digits}:
      Held.Both{+sized_here,+sized_rest} = sized
      Held.Both{+zero,+groups_rest} = groups
      +strides = Tasks.strides_of(source_cells)
      +axis = Values.groups_axis(source_position,source_cells)
      +stride = Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,target))
      +blocked = Tiled.blocked_list(rest,source,source_within,source_position,target,strides,shape,axis)
      +later = word_terms(rest,more,later_digits,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),read,stride,
        sized_rest,Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,Nat,later_digits),1n+List.length(&2,Tiled.CellAxis,rest),paired),groups_rest)
      groups_terms(U32.from_nat(Nat.mul(digit,8n)),U32.from_nat(read),U32.from_nat(Nat.add(Nat.mul(digit,8n),read)),stride,step,
        TaskReads.placed_word(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,rest,more,(index + 1 : U32)),
          relane_values(blocked,later_digits,read),strides),
        Geometry.blocked_sum(later_digits,blocked),(U32.from_nat(read) * Geometry.blocked_step(blocked,stride) : U32),
        Equal.sym(U32,(U32.from_nat(Nat.mul(digit,8n)) + U32.from_nat(read) : U32),U32.from_nat(Nat.add(Nat.mul(digit,8n),read)),WordArithmetic.native_add(Nat.mul(digit,8n),read)),
        zero,later)
    case Con{Tiled.OfAxis{+logical},+rest} Con{+extent,+more} Con{+digit,+later_digits}:
      Held.Both{+sized_here,+sized_rest} = sized
      +strides = Tasks.strides_of(source_cells)
      +axis = Values.groups_axis(source_position,source_cells)
      +choice = (source_within && Nat.is_eq(logical,source))
      +shared = Tiled.source_axis(source,source_within,source_position,logical)
      +blocked = Tiled.blocked_list(rest,source,source_within,source_position,target,strides,shape,axis)
      +later = word_terms(rest,more,later_digits,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),read,step,
        sized_rest,Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,Nat,later_digits),1n+List.length(&2,Tiled.CellAxis,rest),paired),
        some_groups(choice,axis,shared,Grid.lanes_at(shape,source),strides,blocked,step,groups))
      some_word(choice,extent,Grid.block(target_grid,index,extent),source_grid,axis,Tasks.nth(source_cells,axis),shared,Grid.lanes_at(shape,logical),Grid.lanes_at(shape,source),
        strides,digit,Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,rest,more,(index + 1 : U32)),
        blocked,later_digits,read,step,later)
    case Con{Tiled.OfGroups{},+rest} Con{+extent,+more} Nil{}:
      Empty.absurd(WordTerms(Con{Tiled.OfGroups{},rest},Con{extent,more},Nil{},source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,step),
        BoxDigits.zero_successor(List.length(&2,Tiled.CellAxis,rest),paired))
    case Con{Tiled.OfAxis{+logical},+rest} Con{+extent,+more} Nil{}:
      Empty.absurd(WordTerms(Con{Tiled.OfAxis{logical},rest},Con{extent,more},Nil{},source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,step),
        BoxDigits.zero_successor(List.length(&2,Tiled.CellAxis,rest),paired))
    case Con{Tiled.OfGroups{},+rest} Nil{} +digits:
      Empty.absurd(WordTerms(Con{Tiled.OfGroups{},rest},Nil{},digits,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,step),sized)
    case Con{Tiled.OfAxis{+logical},+rest} Nil{} +digits:
      Empty.absurd(WordTerms(Con{Tiled.OfAxis{logical},rest},Nil{},digits,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,step),sized)
    case Nil{} Con{+extent,+more} +digits:
      Empty.absurd(WordTerms(Nil{},Con{extent,more},digits,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,step),sized)
    case Nil{} Nil{} Con{+digit,+later_digits}:
      Empty.absurd(WordTerms(Nil{},Nil{},Con{digit,later_digits},source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,step),
        BoxDigits.zero_successor(List.length(&2,Nat,later_digits),Equal.sym(Nat,1n+List.length(&2,Nat,later_digits),0n,paired)))
    case Nil{} Nil{} Nil{}:
      {==}

# The readings of the target cell axes reach, at the values of a block and
# target lane read, the first source cell of the block and read steps.
def relane_word(+within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+shape: List<&2,U32>,
  +source_cells: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+digits: List<&2,Nat>,+read: Nat,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,shape)) == within : Bool},
  +paired: {List.length(&2,Nat,digits) == List.length(&2,Tiled.CellAxis,Tiled.target_axes(within,position,target,List.length(&2,U32,shape))) : Nat}) ->
  {TaskReads.placed_word(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,
      Tiled.target_axes(within,position,target,List.length(&2,U32,shape)),Tiled.cell_shape(shape,target,position),0),
      relane_values(Tiled.blocked_list(Tiled.target_axes(within,position,target,List.length(&2,U32,shape)),source,source_within,source_position,target,
        Tasks.strides_of(source_cells),shape,Values.groups_axis(source_position,source_cells)),digits,read),Tasks.strides_of(source_cells)) ==
    (Geometry.blocked_sum(digits,Tiled.blocked_list(Tiled.target_axes(within,position,target,List.length(&2,U32,shape)),source,source_within,source_position,target,
        Tasks.strides_of(source_cells),shape,Values.groups_axis(source_position,source_cells))) +
      U32.from_nat(read) * Geometry.blocked_step(Tiled.blocked_list(Tiled.target_axes(within,position,target,List.length(&2,U32,shape)),source,source_within,source_position,
        target,Tasks.strides_of(source_cells),shape,Values.groups_axis(source_position,source_cells)),0) : U32) : U32}:
  +axes = Tiled.target_axes(within,position,target,List.length(&2,U32,shape))
  +strides = Tasks.strides_of(source_cells)
  +axis = Values.groups_axis(source_position,source_cells)
  +blocked = Tiled.blocked_list(axes,source,source_within,source_position,target,strides,shape,axis)
  +word = TaskReads.placed_word(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,
    Tiled.cell_shape(shape,target,position),0),relane_values(blocked,digits,read),strides)
  +terms = word_terms(axes,Tiled.cell_shape(shape,target,position),digits,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,0,read,0,
    target_extents(within,position,target,shape,within_value),paired,
    target_when(within,position,target,List.length(&2,U32,shape),source,source_within,source_position,strides,shape,axis))
  Equal.trans(U32,word,(word + U32.from_nat(read) * 0 : U32),(Geometry.blocked_sum(digits,blocked) + U32.from_nat(read) * Geometry.blocked_step(blocked,0) : U32),
    Equal.sym(U32,(word + U32.from_nat(read) * 0 : U32),word,
      Equal.trans(U32,(word + U32.from_nat(read) * 0 : U32),(word + 0 : U32),word,
        Equal.cong(U32,U32,term => (word + term : U32),(U32.from_nat(read) * 0 : U32),0,Arithmetic.multiply_zero_right(U32.from_nat(read))),Arithmetic.add_zero(word))),
    terms)

# ---- The values lie in the classes of the target cell axes ----

# The source lane axis reads the source groups at the group of its
# coordinate.
def lanes_placed(+extent: U32,+block: U32,+other: U32,+source_block: U32,+bound: U32,+first: U32,+count: U32,+source_first: U32,+source_count: U32,
  +coordinate: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: Components.ReadingRoom(extent,block,Tasks.Read{other,source_block,1,0,False{},1,8,0,bound,False{}},cap),
  +class_in: Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,coordinate),
  +value_below: {Nat.is_lt(Nat.div(coordinate,8n),U32.to_nat(bound)) == True{} : Bool}) ->
  TaskReads.PlacedIn(Tasks.Reading{extent,block,Tasks.Read{other,source_block,1,0,False{},1,8,0,bound,False{}}},Tasks.Class{first,count,source_first,source_count},
    Nat.div(coordinate,8n),cap):
  +group = Nat.div(coordinate,8n)
  +above = {%Equal.sym(Nat,Nat.mul(1n,coordinate),coordinate,TaskReads.one_times(coordinate)) : {Nat.is_le(Nat.div(_,8n),group) == True{} : Bool} Order.reflexive(group) :
    {Nat.is_le(Nat.div(Nat.mul(1n,coordinate),8n),group) == True{} : Bool}}
  +under = {%Equal.sym(Nat,Nat.sub(Nat.add(Nat.mul(1n,coordinate),1n),1n),Nat.mul(1n,coordinate),DivisionOrder.subtract_sum(Nat.mul(1n,coordinate),1n)) :
      {Nat.is_le(group,Nat.div(_,8n)) == True{} : Bool}
    %Equal.sym(Nat,Nat.mul(1n,coordinate),coordinate,TaskReads.one_times(coordinate)) : {Nat.is_le(group,Nat.div(_,8n)) == True{} : Bool}
    Order.reflexive(group) : {Nat.is_le(group,Nat.div(Nat.sub(Nat.add(Nat.mul(1n,coordinate),1n),1n),8n)) == True{} : Bool}}
  TaskReads.read_placed(extent,block,other,source_block,1,0,1,8,0,bound,first,count,source_first,source_count,coordinate,group,cap,cap_ok,room,class_in,above,under,value_below)

# The target groups read the source axis of the target lane axis at the
# lane of their group.
def groups_placed(+extent: U32,+block: U32,+other: U32,+source_block: U32,+bound: U32,+first: U32,+count: U32,+source_first: U32,+source_count: U32,
  +coordinate: Nat,+read: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: Components.ReadingRoom(extent,block,Tasks.Read{other,source_block,8,0,False{},8,1,0,bound,False{}},cap),
  +class_in: Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,coordinate),
  +small: {Nat.is_lt(read,8n) == True{} : Bool},
  +value_below: {Nat.is_lt(Nat.add(Nat.mul(coordinate,8n),read),U32.to_nat(bound)) == True{} : Bool}) ->
  TaskReads.PlacedIn(Tasks.Reading{extent,block,Tasks.Read{other,source_block,8,0,False{},8,1,0,bound,False{}}},Tasks.Class{first,count,source_first,source_count},
    Nat.add(Nat.mul(coordinate,8n),read),cap):
  +lane = Nat.add(Nat.mul(coordinate,8n),read)
  +turned = RowMajor.multiply_commutative(8n,coordinate)
  +whole = Equal.trans(Nat,Nat.div(Nat.mul(8n,coordinate),1n),Nat.mul(8n,coordinate),Nat.mul(coordinate,8n),TaskReads.one_quotient(Nat.mul(8n,coordinate)),turned)
  +above = {%Equal.sym(Nat,Nat.div(Nat.mul(8n,coordinate),1n),Nat.mul(coordinate,8n),whole) : {Nat.is_le(_,lane) == True{} : Bool}
    Order.le_add(Nat.mul(coordinate,8n),read) : {Nat.is_le(Nat.div(Nat.mul(8n,coordinate),1n),lane) == True{} : Bool}}
  +last = Nat.sub(Nat.add(Nat.mul(8n,coordinate),8n),1n)
  +top = Nat.add(Nat.mul(coordinate,8n),7n)
  +reached = Equal.trans(Nat,Nat.div(last,1n),last,top,TaskReads.one_quotient(last),
    Equal.trans(Nat,last,Nat.add(Nat.mul(8n,coordinate),7n),top,Order.subtract_one_complement(Nat.mul(8n,coordinate),7n),
      Equal.cong(Nat,Nat,number => Nat.add(number,7n),Nat.mul(8n,coordinate),Nat.mul(coordinate,8n),turned)))
  +under = {%Equal.sym(Nat,Nat.div(last,1n),top,reached) : {Nat.is_le(lane,_) == True{} : Bool}
    Order.offset_le(Nat.mul(coordinate,8n),read,7n,Division.lt_succ_le(read,7n,small)) : {Nat.is_le(lane,Nat.div(last,1n)) == True{} : Bool}}
  TaskReads.read_placed(extent,block,other,source_block,8,0,8,1,0,bound,first,count,source_first,source_count,coordinate,lane,cap,cap_ok,room,class_in,above,under,value_below)

# The target groups read lanes below the lanes of the target lane axis.
def GroupsBelow(axes: List<&2,Tiled.CellAxis>,digits: List<&2,Nat>,+read: Nat,+lanes: U32) -> Data:
  match axes digits:
    case Con{Tiled.OfGroups{},axes} Con{+digit,digits}:
      Held.Both<{Nat.is_lt(Nat.add(Nat.mul(digit,8n),read),U32.to_nat(lanes)) == True{} : Bool},GroupsBelow(axes,digits,read,lanes)>
    case Con{axis,axes} Con{digit,digits}:
      GroupsBelow(axes,digits,read,lanes)
    case axes digits:
      Unit

# A coordinate inside a class lies below the extent of its axis.
def class_below(+first: U32,+count: U32,+source_first: U32,+source_count: U32,+block: U32,+extent: U32,+coordinate: Nat,
  +class_in: Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,coordinate)) ->
  {Nat.is_lt(coordinate,U32.to_nat(extent)) == True{} : Bool}:
  Held.Both{+low_le,+top_above} = class_in
  +ends = ((first + count) * block : U32)
  Bounds.nat_trans(coordinate,U32.to_nat(U32.min(ends,extent)),U32.to_nat(extent),top_above,
    {%Equal.sym(Nat,U32.to_nat(U32.min(ends,extent)),Nat.min(U32.to_nat(ends),U32.to_nat(extent)),WordExtrema.minimum(ends,extent)) : {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool}
      NatExtrema.minimum_right(U32.to_nat(ends),U32.to_nat(extent)) : {Nat.is_le(U32.to_nat(U32.min(ends,extent)),U32.to_nat(extent)) == True{} : Bool}})

# A logical target cell axis reads the source groups at the group of its
# coordinate when it is the source lane axis, otherwise its own source axis
# at its coordinate.
def some_placed(source_within: Bool,same: Bool,+logical: Nat,+source: Nat,+shape: List<&2,U32>,+extent: U32,+block: U32,+grid: Grid.Grid,+lane_axis: U32,
  +lane_extent: U32,+shared_axis: U32,+strides: List<&2,U32>,+first: U32,+count: U32,+source_first: U32,+source_count: U32,+digit: Nat,+read: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+readings: List<&2,Tasks.Reading>,+classes: List<&2,Tasks.Class>,+blocked: List<&2,Tiled.Blocked>,
  +digits: List<&2,Nat>,+same_value: {Nat.is_eq(logical,source) == same : Bool},
  +lane_value: Logic.When(source_within,{U32.to_nat(lane_extent) == NatCeiling.ceiling(U32.to_nat(Grid.lanes_at(shape,source)),8n) : Nat}),
  +room: Components.ReadingRoom(extent,block,Bool.pick(Tasks.Source,source_within && same,Tiled.read_at(grid,lane_axis,lane_extent,1,1,8),
    Tiled.read_at(grid,shared_axis,Grid.lanes_at(shape,logical),1,1,1)),cap),
  +class_in: Targets.ClassIn(Tasks.Class{first,count,source_first,source_count},block,extent,digit),
  +extent_value: {extent == Grid.lanes_at(shape,logical) : U32},
  +later: TaskReads.PlacedRow(readings,classes,relane_values(blocked,digits,read),cap)) ->
  TaskReads.PlacedRow(Tasks.Reading{extent,block,Bool.pick(Tasks.Source,source_within && same,Tiled.read_at(grid,lane_axis,lane_extent,1,1,8),
      Tiled.read_at(grid,shared_axis,Grid.lanes_at(shape,logical),1,1,1))} <> readings,Tasks.Class{first,count,source_first,source_count} <> classes,
    relane_values(Bool.pick(Tiled.Blocked,source_within && same,Tiled.Blocked{Tiled.Lanes{},Tasks.nth(strides,lane_axis),Grid.lanes_at(shape,source)},
      Tiled.Blocked{Tiled.Shared{},Tasks.nth(strides,shared_axis),1}) <> blocked,digit <> digits,read),cap):
  match source_within same:
    case True{} True{}:
      +lanes = U32.to_nat(Grid.lanes_at(shape,source))
      +at_source = Order.equal_of_is_eq(logical,source,same_value)
      +extent_lanes = Equal.trans(Nat,U32.to_nat(extent),U32.to_nat(Grid.lanes_at(shape,logical)),lanes,
        Equal.cong(U32,Nat,word => U32.to_nat(word),extent,Grid.lanes_at(shape,logical),extent_value),
        Equal.cong(Nat,Nat,axis => U32.to_nat(Grid.lanes_at(shape,axis)),logical,source,at_source))
      +digit_below = {%extent_lanes : {Nat.is_lt(digit,_) == True{} : Bool}
        class_below(first,count,source_first,source_count,block,extent,digit,class_in) : {Nat.is_lt(digit,lanes) == True{} : Bool}}
      +group = Nat.div(digit,8n)
      +product_below = Order.le_lt_trans(Nat.mul(group,8n),digit,lanes,DivisionOrder.product_below(digit,8n,{==}),digit_below)
      +group_below = {%Equal.sym(Nat,U32.to_nat(lane_extent),NatCeiling.ceiling(lanes,8n),lane_value) : {Nat.is_lt(group,_) == True{} : Bool}
        %Equal.sym(Bool,Nat.is_lt(group,NatCeiling.ceiling(lanes,8n)),Nat.is_lt(Nat.mul(group,8n),lanes),NatCeiling.upper_comparison(lanes,8n,group,{==})) :
          {_ == True{} : Bool}
        product_below : {Nat.is_lt(group,U32.to_nat(lane_extent)) == True{} : Bool}}
      Held.Both{lanes_placed(extent,block,lane_axis,Grid.block(grid,lane_axis,lane_extent),lane_extent,first,count,source_first,source_count,digit,cap,cap_ok,
        room,class_in,group_below),later}
    case True{} False{}:
      +digit_below = {%extent_value : {Nat.is_lt(digit,U32.to_nat(_)) == True{} : Bool}
        class_below(first,count,source_first,source_count,block,extent,digit,class_in) : {Nat.is_lt(digit,U32.to_nat(Grid.lanes_at(shape,logical))) == True{} : Bool}}
      Held.Both{TaskReads.shared_placed(extent,block,shared_axis,Grid.block(grid,shared_axis,Grid.lanes_at(shape,logical)),Grid.lanes_at(shape,logical),first,count,
        source_first,source_count,digit,cap,cap_ok,room,class_in,digit_below),later}
    case False{} +same:
      +digit_below = {%extent_value : {Nat.is_lt(digit,U32.to_nat(_)) == True{} : Bool}
        class_below(first,count,source_first,source_count,block,extent,digit,class_in) : {Nat.is_lt(digit,U32.to_nat(Grid.lanes_at(shape,logical))) == True{} : Bool}}
      Held.Both{TaskReads.shared_placed(extent,block,shared_axis,Grid.block(grid,shared_axis,Grid.lanes_at(shape,logical)),Grid.lanes_at(shape,logical),first,count,
        source_first,source_count,digit,cap,cap_ok,room,class_in,digit_below),later}

def RowPlaced(+axes: List<&2,Tiled.CellAxis>,+extents: List<&2,U32>,+digits: List<&2,Nat>,+classes: List<&2,Tasks.Class>,+source: Nat,+source_within: Bool,
  +source_position: Grid.Position,+target: Nat,+source_cells: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+shape: List<&2,U32>,+index: U32,+read: Nat,
  +cap: U32) -> Data:
  TaskReads.PlacedRow(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,extents,index),classes,
    relane_values(Tiled.blocked_list(axes,source,source_within,source_position,target,Tasks.strides_of(source_cells),shape,Values.groups_axis(source_position,source_cells)),
      digits,read),cap)

# Every target cell axis reads its source axis at a value inside the class
# of its coordinate.
def placed_row(axes: List<&2,Tiled.CellAxis>,extents: List<&2,U32>,digits: List<&2,Nat>,classes: List<&2,Tasks.Class>,+source: Nat,+source_within: Bool,
  +source_position: Grid.Position,+target: Nat,+source_cells: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+shape: List<&2,U32>,+index: U32,+read: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+sized: AxesExtents(axes,extents,shape,target),
  +classes_in: Targets.ClassesIn(classes,target_grid,extents,digits,index),
  +rooms: Components.ReadingsRoom(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,extents,index),cap),
  +groups_below: GroupsBelow(axes,digits,read,Grid.lanes_at(shape,target)),+small: {Nat.is_lt(read,8n) == True{} : Bool},
  +lane_value: Logic.When(source_within,{U32.to_nat(Tasks.nth(source_cells,Values.groups_axis(source_position,source_cells))) ==
    NatCeiling.ceiling(U32.to_nat(Grid.lanes_at(shape,source)),8n) : Nat})) ->
  RowPlaced(axes,extents,digits,classes,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,cap):
  match axes extents digits classes:
    case Con{Tiled.OfGroups{},+rest} Con{+extent,+more} Con{+digit,+later_digits} Con{Tasks.Class{+first,+count,+source_first,+source_count},+later_classes}:
      Held.Both{+sized_here,+sized_rest} = sized
      Held.Both{+class_in,+classes_rest} = classes_in
      Held.Both{+room,+rooms_rest} = rooms
      Held.Both{+lane_below,+groups_rest} = groups_below
      +axis = Tiled.source_axis(source,source_within,source_position,target)
      +lanes = Grid.lanes_at(shape,target)
      Held.Both{groups_placed(extent,Grid.block(target_grid,index,extent),axis,Grid.block(source_grid,axis,lanes),lanes,first,count,source_first,source_count,digit,read,
          cap,cap_ok,room,class_in,small,lane_below),
        placed_row(rest,more,later_digits,later_classes,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),read,
          cap,cap_ok,sized_rest,classes_rest,rooms_rest,groups_rest,small,lane_value)}
    case Con{Tiled.OfAxis{+logical},+rest} Con{+extent,+more} Con{+digit,+later_digits} Con{Tasks.Class{+first,+count,+source_first,+source_count},+later_classes}:
      Held.Both{+sized_here,+sized_rest} = sized
      Held.Both{+class_in,+classes_rest} = classes_in
      Held.Both{+room,+rooms_rest} = rooms
      +strides = Tasks.strides_of(source_cells)
      +axis = Values.groups_axis(source_position,source_cells)
      some_placed(source_within,Nat.is_eq(logical,source),logical,source,shape,extent,Grid.block(target_grid,index,extent),source_grid,axis,Tasks.nth(source_cells,axis),
        Tiled.source_axis(source,source_within,source_position,logical),strides,first,count,source_first,source_count,digit,read,cap,cap_ok,
        Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,rest,more,(index + 1 : U32)),later_classes,
        Tiled.blocked_list(rest,source,source_within,source_position,target,strides,shape,axis),later_digits,{==},lane_value,room,class_in,sized_here,
        placed_row(rest,more,later_digits,later_classes,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),read,
          cap,cap_ok,sized_rest,classes_rest,rooms_rest,groups_below,small,lane_value))
    case Con{+axis,+rest} Con{+extent,+more} Con{+digit,+later_digits} Nil{}:
      Empty.absurd(RowPlaced(Con{axis,rest},Con{extent,more},Con{digit,later_digits},Nil{},source,source_within,source_position,target,source_cells,source_grid,target_grid,
        shape,index,read,cap),classes_in)
    case Con{+axis,+rest} Con{+extent,+more} Nil{} Nil{}:
      Empty.absurd(RowPlaced(Con{axis,rest},Con{extent,more},Nil{},Nil{},source,source_within,source_position,target,source_cells,source_grid,target_grid,
        shape,index,read,cap),classes_in)
    case Con{+axis,+rest} Con{+extent,+more} Nil{} Con{+class,+later_classes}:
      Empty.absurd(RowPlaced(Con{axis,rest},Con{extent,more},Nil{},Con{class,later_classes},source,source_within,source_position,target,source_cells,source_grid,target_grid,
        shape,index,read,cap),classes_in)
    case Con{Tiled.OfGroups{},+rest} Nil{} +digits +classes:
      Empty.absurd(RowPlaced(Con{Tiled.OfGroups{},rest},Nil{},digits,classes,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,cap),sized)
    case Con{Tiled.OfAxis{+logical},+rest} Nil{} +digits +classes:
      Empty.absurd(RowPlaced(Con{Tiled.OfAxis{logical},rest},Nil{},digits,classes,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,cap),sized)
    case Nil{} Con{+extent,+more} +digits +classes:
      Empty.absurd(RowPlaced(Nil{},Con{extent,more},digits,classes,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,cap),sized)
    case Nil{} Nil{} Nil{} Nil{}:
      Unit{}
    case Nil{} Nil{} Nil{} Con{+class,+later_classes}:
      Empty.absurd(RowPlaced(Nil{},Nil{},Nil{},Con{class,later_classes},source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,cap),
        classes_in)
    case Nil{} Nil{} Con{+digit,+later_digits} Nil{}:
      Empty.absurd(RowPlaced(Nil{},Nil{},Con{digit,later_digits},Nil{},source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,index,read,cap),
        classes_in)
    case Nil{} Nil{} Con{+digit,+later_digits} Con{+class,+later_classes}:
      Empty.absurd(RowPlaced(Nil{},Nil{},Con{digit,later_digits},Con{class,later_classes},source,source_within,source_position,target,source_cells,source_grid,target_grid,
        shape,index,read,cap),classes_in)

# ---- The target groups and the source groups ----

# The logical axes have no target groups.
def some_pick(choice: Bool,+later: List<&2,Tiled.CellAxis>,+index: Nat,digits: List<&2,Nat>,+read: Nat,+lanes: U32,
  +same: GroupsBelow(later,digits,read,lanes),+tail: GroupsBelow(later,Geometry.digits_tail(digits),read,lanes)) ->
  GroupsBelow(Bool.pick(List<&2,Tiled.CellAxis>,choice,later,Tiled.OfAxis{index} <> later),digits,read,lanes):
  match choice digits:
    case True{} +digits: same
    case False{} Nil{}: Unit{}
    case False{} Con{+digit,+rest}: tail

def some_below(remaining: Nat,+digits: List<&2,Nat>,+index: Nat,+skip: Nat,+read: Nat,+lanes: U32) -> GroupsBelow(Tiled.axes_from(remaining,index,skip),digits,read,lanes):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      some_pick(Nat.is_eq(index,skip),Tiled.axes_from(rest,1n+index,skip),index,digits,read,lanes,some_below(rest,digits,1n+index,skip,read,lanes),
        some_below(rest,Geometry.digits_tail(digits),1n+index,skip,read,lanes))

# The target groups at the end read the last digit.
def append_below(axes: List<&2,Tiled.CellAxis>,front: List<&2,Nat>,+back: Nat,+read: Nat,+lanes: U32,
  +same: {List.length(&2,Nat,front) == List.length(&2,Tiled.CellAxis,axes) : Nat},+rest: GroupsBelow(axes,front,read,lanes),
  +head: {Nat.is_lt(Nat.add(Nat.mul(back,8n),read),U32.to_nat(lanes)) == True{} : Bool}) ->
  GroupsBelow(List.append(&2,Tiled.CellAxis,axes,[Tiled.OfGroups{}]),List.append(&2,Nat,front,[back]),read,lanes):
  match axes front:
    case Con{Tiled.OfGroups{},+later} Con{+digit,+digits}:
      Held.Both{+here,+after} = rest
      Held.Both{here,append_below(later,digits,back,read,lanes,
        Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,Nat,digits),1n+List.length(&2,Tiled.CellAxis,later),same),after,head)}
    case Con{Tiled.OfAxis{+logical},+later} Con{+digit,+digits}:
      append_below(later,digits,back,read,lanes,
        Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,Nat,digits),1n+List.length(&2,Tiled.CellAxis,later),same),rest,head)
    case Nil{} Nil{}:
      Held.Both{head,Unit{}}
    case Nil{} Con{+digit,+digits}:
      Empty.absurd(GroupsBelow(List.append(&2,Tiled.CellAxis,Nil{},[Tiled.OfGroups{}]),List.append(&2,Nat,Con{digit,digits},[back]),read,lanes),
        BoxDigits.zero_successor(List.length(&2,Nat,digits),Equal.sym(Nat,1n+List.length(&2,Nat,digits),0n,same)))
    case Con{+axis,+later} Nil{}:
      Empty.absurd(GroupsBelow(List.append(&2,Tiled.CellAxis,Con{axis,later},[Tiled.OfGroups{}]),List.append(&2,Nat,Nil{},[back]),read,lanes),
        BoxDigits.zero_successor(List.length(&2,Tiled.CellAxis,later),same))

# The target groups of a cell read lanes below the target lanes when the
# lane of the group of the cell does.
def groups_target(within: Bool,position: Grid.Position,digits: List<&2,Nat>,+target: Nat,+shape: List<&2,U32>,+read: Nat,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,shape)) == within : Bool},
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(shape,target),Grid.groups(Grid.lanes_at(shape,target))),digits),
  +valid: {Nat.is_lt(Nat.add(read,Nat.mul(Values.cell_group(within,position,digits),8n)),U32.to_nat(Grid.lanes_at(shape,target))) == True{} : Bool}) ->
  GroupsBelow(Tiled.target_axes(within,position,target,List.length(&2,U32,shape)),digits,read,Grid.lanes_at(shape,target)):
  match within position digits:
    case False{} +position +digits:
      +rank = List.length(&2,U32,shape)
      some_below(rank,digits,0n,rank,read,Grid.lanes_at(shape,target))
    case True{} Grid.Innermost{} +digits:
      +rank = List.length(&2,U32,shape)
      +lanes = Grid.lanes_at(shape,target)
      +others = Grid.removed(shape,target)
      +front = Geometry.digits_front(digits)
      +back = Geometry.digits_back(digits)
      +axes = Tiled.axes_from(rank,0n,target)
      +front_count = Values.below_length(others,front,Values.below_front(others,digits,Grid.groups(lanes),below))
      +axes_count = {%within_value : {Nat.add(List.length(&2,Tiled.CellAxis,axes),Geometry.skipped_count(_)) == rank : Nat}
        Geometry.axes_count(rank,target,0n) : {Nat.add(List.length(&2,Tiled.CellAxis,axes),1n) == rank : Nat}}
      +same = Order.cancel_sum(1n,List.length(&2,Nat,front),List.length(&2,Tiled.CellAxis,axes),
        Equal.trans(Nat,1n+List.length(&2,Nat,front),rank,1n+List.length(&2,Tiled.CellAxis,axes),
          Equal.trans(Nat,1n+List.length(&2,Nat,front),1n+List.length(&2,U32,others),rank,
            Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,front),List.length(&2,U32,others),front_count),Public.removed_length(shape,target,within_value)),
          Equal.sym(Nat,1n+List.length(&2,Tiled.CellAxis,axes),rank,
            Equal.trans(Nat,1n+List.length(&2,Tiled.CellAxis,axes),Nat.add(List.length(&2,Tiled.CellAxis,axes),1n),rank,
              Equal.sym(Nat,Nat.add(List.length(&2,Tiled.CellAxis,axes),1n),1n+List.length(&2,Tiled.CellAxis,axes),Addition.successor_width(List.length(&2,Tiled.CellAxis,axes))),
              axes_count))))
      +head = {%Order.commutative(read,Nat.mul(back,8n)) : {Nat.is_lt(_,U32.to_nat(lanes)) == True{} : Bool}
        valid : {Nat.is_lt(Nat.add(Nat.mul(back,8n),read),U32.to_nat(lanes)) == True{} : Bool}}
      %Values.below_whole(others,digits,Grid.groups(lanes),below) : GroupsBelow(List.append(&2,Tiled.CellAxis,axes,[Tiled.OfGroups{}]),_,read,lanes)
      append_below(axes,front,back,read,lanes,same,some_below(rank,front,0n,target,read,lanes),head)
    case True{} Grid.Outermost{} Nil{}:
      Empty.absurd(GroupsBelow(Tiled.target_axes(True{},Grid.Outermost{},target,List.length(&2,U32,shape)),Nil{},read,Grid.lanes_at(shape,target)),below)
    case True{} Grid.Outermost{} Con{+head,+tail}:
      +rank = List.length(&2,U32,shape)
      +lanes = Grid.lanes_at(shape,target)
      Held.Both{{%Order.commutative(read,Nat.mul(head,8n)) : {Nat.is_lt(_,U32.to_nat(lanes)) == True{} : Bool}
          valid : {Nat.is_lt(Nat.add(Nat.mul(head,8n),read),U32.to_nat(lanes)) == True{} : Bool}},
        some_below(rank,tail,0n,target,read,lanes)}

# The last of a list with one more at the end.
def nth_last(list: List<&2,U32>,+last: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,U32,list),U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.nth(List.append(&2,U32,list,[last]),U32.from_nat(List.length(&2,U32,list))) == last : U32}:
  match list:
    case Nil{}: {==}
    case Con{+value,+rest}:
      +count = List.length(&2,U32,rest)
      +nonzero = Arithmetic.word_positive_nonzero(U32.from_nat(1n+count),
        {%Equal.sym(Nat,U32.to_nat(U32.from_nat(1n+count)),1n+count,Bounds.roundtrip(1n+count,cap,cap_ok,room)) : {Nat.is_lt(0n,_) == True{} : Bool} {==} :
          {Nat.is_lt(0n,U32.to_nat(U32.from_nat(1n+count))) == True{} : Bool}})
      %Equal.sym(Bool,U32.is_zero(U32.from_nat(1n+count)),False{},nonzero) :
        {Bool.pick(U32,_,value,Tasks.nth(List.append(&2,U32,rest,[last]),(U32.from_nat(1n+count) - 1 : U32))) == last : U32}
      %Equal.sym(U32,(U32.from_nat(1n+count) - 1 : U32),U32.from_nat(count),Geometry.less_one(count)) :
        {Tasks.nth(List.append(&2,U32,rest,[last]),_) == last : U32}
      nth_last(rest,last,cap,cap_ok,Order.transitive(count,1n+count,U32.to_nat(cap),
        {%Addition.successor_width(count) : {Nat.is_le(count,_) == True{} : Bool} Order.le_add(count,1n) : {Nat.is_le(count,1n+count) == True{} : Bool}},room))

# The source groups axis of the source cells holds the groups of the source
# lane axis.
def placed_groups(position: Grid.Position,+others: List<&2,U32>,+groups: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +room: {Nat.is_le(List.length(&2,U32,others),U32.to_nat(cap)) == True{} : Bool}) ->
  {Tasks.nth(Grid.placed(True{},position,others,groups),Values.groups_axis(position,Grid.placed(True{},position,others,groups))) == groups : U32}:
  match position:
    case Grid.Innermost{}:
      +count = List.length(&2,U32,others)
      %Equal.sym(Nat,List.length(&2,U32,List.append(&2,U32,others,[groups])),1n+count,Values.appended_length(others,groups)) :
        {Tasks.nth(List.append(&2,U32,others,[groups]),(U32.from_nat(_) - 1 : U32)) == groups : U32}
      %Equal.sym(U32,(U32.from_nat(1n+count) - 1 : U32),U32.from_nat(count),Geometry.less_one(count)) :
        {Tasks.nth(List.append(&2,U32,others,[groups]),_) == groups : U32}
      nth_last(others,groups,cap,cap_ok,room)
    case Grid.Outermost{}: {==}

def source_groups(+limits: Machine.Limits,source_within: Bool,+position: Grid.Position,+shape: List<&2,U32>,+source: Nat,
  +source_value: {Nat.is_lt(source,List.length(&2,U32,shape)) == source_within : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(shape,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,shape),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  Logic.When(source_within,{U32.to_nat(Tasks.nth(Tiled.cell_shape(shape,source,position),Values.groups_axis(position,Tiled.cell_shape(shape,source,position)))) ==
    NatCeiling.ceiling(U32.to_nat(Grid.lanes_at(shape,source)),8n) : Nat}):
  match source_within:
    case False{}: Unit{}
    case True{}:
      +lanes = Grid.lanes_at(shape,source)
      +others = Grid.removed(shape,source)
      +count = List.length(&2,U32,others)
      +others_room = Order.transitive(count,1n+count,U32.to_nat(Machine.capacity(limits)),
        {%Addition.successor_width(count) : {Nat.is_le(count,_) == True{} : Bool} Order.le_add(count,1n) : {Nat.is_le(count,1n+count) == True{} : Bool}},
        {%Equal.sym(Nat,1n+count,List.length(&2,U32,shape),Public.removed_length(shape,source,source_value)) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}
          rank_room : {Nat.is_le(1n+count,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})
      +held = placed_groups(position,others,Grid.groups(lanes),Machine.capacity(limits),Limits.capacity_fits(limits),others_room)
      %Equal.sym(Bool,Nat.is_lt(source,List.length(&2,U32,shape)),True{},source_value) :
        {U32.to_nat(Tasks.nth(Grid.placed(_,position,others,Grid.groups(lanes)),Values.groups_axis(position,Grid.placed(_,position,others,Grid.groups(lanes))))) ==
          NatCeiling.ceiling(U32.to_nat(lanes),8n) : Nat}
      Equal.trans(Nat,U32.to_nat(Tasks.nth(Grid.placed(True{},position,others,Grid.groups(lanes)),Values.groups_axis(position,Grid.placed(True{},position,others,Grid.groups(lanes))))),
        U32.to_nat(Grid.groups(lanes)),NatCeiling.ceiling(U32.to_nat(lanes),8n),
        Equal.cong(U32,Nat,word => U32.to_nat(word),Tasks.nth(Grid.placed(True{},position,others,Grid.groups(lanes)),Values.groups_axis(position,Grid.placed(True{},position,others,Grid.groups(lanes)))),
          Grid.groups(lanes),held),
        Encoded.eight_blocks_exact(limits,lanes,lanes_room))

# ---- The source cells a read of a block reads ----

# Every target cell axis has an extent and reads its source.
def extents_length(axes: List<&2,Tiled.CellAxis>,extents: List<&2,U32>,+shape: List<&2,U32>,+target: Nat,+sized: AxesExtents(axes,extents,shape,target)) ->
  {List.length(&2,U32,extents) == List.length(&2,Tiled.CellAxis,axes) : Nat}:
  match axes extents:
    case Con{Tiled.OfAxis{+axis},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = sized
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,more),List.length(&2,Tiled.CellAxis,rest),extents_length(rest,more,shape,target,later))
    case Con{Tiled.OfGroups{},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = sized
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,more),List.length(&2,Tiled.CellAxis,rest),extents_length(rest,more,shape,target,later))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+extent,+more}: Empty.absurd({List.length(&2,U32,Con{extent,more}) == 0n : Nat},sized)
    case Con{Tiled.OfAxis{+axis},+rest} Nil{}: Empty.absurd({0n == List.length(&2,Tiled.CellAxis,Con{Tiled.OfAxis{axis},rest}) : Nat},sized)
    case Con{Tiled.OfGroups{},+rest} Nil{}: Empty.absurd({0n == List.length(&2,Tiled.CellAxis,Con{Tiled.OfGroups{},rest}) : Nat},sized)

def readings_length(axes: List<&2,Tiled.CellAxis>,extents: List<&2,U32>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +source_cells: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+shape: List<&2,U32>,+index: U32,+sized: AxesExtents(axes,extents,shape,target)) ->
  {List.length(&2,Tasks.Reading,Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,extents,index)) ==
    List.length(&2,Tiled.CellAxis,axes) : Nat}:
  match axes extents:
    case Con{Tiled.OfAxis{+axis},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = sized
      Equal.cong(Nat,Nat,count => 1n+count,
        List.length(&2,Tasks.Reading,Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,rest,more,(index + 1 : U32))),
        List.length(&2,Tiled.CellAxis,rest),
        readings_length(rest,more,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),later))
    case Con{Tiled.OfGroups{},+rest} Con{+extent,+more}:
      Held.Both{+here,+later} = sized
      Equal.cong(Nat,Nat,count => 1n+count,
        List.length(&2,Tasks.Reading,Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,rest,more,(index + 1 : U32))),
        List.length(&2,Tiled.CellAxis,rest),
        readings_length(rest,more,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),later))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+extent,+more}:
      Empty.absurd({List.length(&2,Tasks.Reading,Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Nil{},
        Con{extent,more},index)) == 0n : Nat},sized)
    case Con{Tiled.OfAxis{+axis},+rest} Nil{}:
      Empty.absurd({List.length(&2,Tasks.Reading,Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Con{Tiled.OfAxis{axis},rest},
        Nil{},index)) == List.length(&2,Tiled.CellAxis,Con{Tiled.OfAxis{axis},rest}) : Nat},sized)
    case Con{Tiled.OfGroups{},+rest} Nil{}:
      Empty.absurd({List.length(&2,Tasks.Reading,Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,Con{Tiled.OfGroups{},rest},
        Nil{},index)) == List.length(&2,Tiled.CellAxis,Con{Tiled.OfGroups{},rest}) : Nat},sized)

def some_count(choice: Bool,+lane_source: U32,+lanes: U32,+shared_source: U32,+blocked: List<&2,Tiled.Blocked>,+digit: Nat,+digits: List<&2,Nat>,+read: Nat,
  +count: Nat,+later: {List.length(&2,Nat,relane_values(blocked,digits,read)) == count : Nat}) ->
  {List.length(&2,Nat,relane_values(Bool.pick(Tiled.Blocked,choice,Tiled.Blocked{Tiled.Lanes{},lane_source,lanes},Tiled.Blocked{Tiled.Shared{},shared_source,1}) <> blocked,
    digit <> digits,read)) == 1n+count : Nat}:
  match choice:
    case True{}: Equal.cong(Nat,Nat,number => 1n+number,List.length(&2,Nat,relane_values(blocked,digits,read)),count,later)
    case False{}: Equal.cong(Nat,Nat,number => 1n+number,List.length(&2,Nat,relane_values(blocked,digits,read)),count,later)

def values_length(axes: List<&2,Tiled.CellAxis>,digits: List<&2,Nat>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+read: Nat,+paired: {List.length(&2,Nat,digits) == List.length(&2,Tiled.CellAxis,axes) : Nat}) ->
  {List.length(&2,Nat,relane_values(Tiled.blocked_list(axes,source,source_within,source_position,target,strides,shape,groups_axis),digits,read)) ==
    List.length(&2,Tiled.CellAxis,axes) : Nat}:
  match axes digits:
    case Con{Tiled.OfGroups{},+rest} Con{+digit,+more}:
      Equal.cong(Nat,Nat,count => 1n+count,
        List.length(&2,Nat,relane_values(Tiled.blocked_list(rest,source,source_within,source_position,target,strides,shape,groups_axis),more,read)),
        List.length(&2,Tiled.CellAxis,rest),
        values_length(rest,more,source,source_within,source_position,target,strides,shape,groups_axis,read,
          Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,Nat,more),1n+List.length(&2,Tiled.CellAxis,rest),paired)))
    case Con{Tiled.OfAxis{+logical},+rest} Con{+digit,+more}:
      some_count(source_within && Nat.is_eq(logical,source),Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source),
        Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,logical)),
        Tiled.blocked_list(rest,source,source_within,source_position,target,strides,shape,groups_axis),digit,more,read,List.length(&2,Tiled.CellAxis,rest),
        values_length(rest,more,source,source_within,source_position,target,strides,shape,groups_axis,read,
          Equal.cong(Nat,Nat,number => BoxDigits.before(number),1n+List.length(&2,Nat,more),1n+List.length(&2,Tiled.CellAxis,rest),paired)))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+digit,+more}:
      Empty.absurd({List.length(&2,Nat,relane_values(Tiled.blocked_list(Nil{},source,source_within,source_position,target,strides,shape,groups_axis),Con{digit,more},read)) ==
        0n : Nat},BoxDigits.zero_successor(List.length(&2,Nat,more),Equal.sym(Nat,1n+List.length(&2,Nat,more),0n,paired)))
    case Con{+axis,+rest} Nil{}:
      Empty.absurd({List.length(&2,Nat,relane_values(Tiled.blocked_list(Con{axis,rest},source,source_within,source_position,target,strides,shape,groups_axis),Nil{},read)) ==
        List.length(&2,Tiled.CellAxis,Con{axis,rest}) : Nat},BoxDigits.zero_successor(List.length(&2,Tiled.CellAxis,rest),paired))

# ---- The reads of a relane piece ----

# The readings of the target cell axes of a relane, its components, the
# classes of a component and the spans of the source pieces the component
# takes.
def relane_row(+within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>) ->
  List<&2,Tasks.Reading>:
  +source_cells = Tiled.cell_shape(values,source,source_position)
  +cells = Tiled.cell_shape(values,target,position)
  Tiled.relane_readings(source,source_within,source_position,target,source_cells,Tensors.grid_of(source_cells),Tensors.grid_of(cells),values,
    Tiled.target_axes(within,position,target,List.length(&2,U32,values)),cells,0)

def relane_components(+within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>) ->
  Tasks.Components:
  Tasks.components_of([relane_row(within,position,target,source,source_within,source_position,values)])

def relane_classes(+within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>,
  +component: U32) -> List<&2,Tasks.Class>:
  +row = relane_row(within,position,target,source,source_within,source_position,values)
  Tasks.chosen(Tasks.class_lists(Tasks.target_rank([row]),[row]),component)

def relane_spans(+within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>,
  +component: U32) -> Held.Spans:
  +source_cells = Tiled.cell_shape(values,source,source_position)
  +grid = Tensors.grid_of(source_cells)
  Moved.taken_spans(Tasks.box_pieces(grid,source_cells,Tasks.source_box(relane_components(within,position,target,source,source_within,source_position,values),0,grid,
    source_cells,0,component)),grid)

# What a separable relane needs of a component: its source row reads inside
# the words, its classes end inside their axes and cover the target pieces.
def RelaneSeparate(+limits: Machine.Limits,+within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +values: List<&2,U32>,+component: U32) -> Data:
  +source_cells = Tiled.cell_shape(values,source,source_position)
  +cells = Tiled.cell_shape(values,target,position)
  +row = relane_row(within,position,target,source,source_within,source_position,values)
  +classes = relane_classes(within,position,target,source,source_within,source_position,values,component)
  Logic.When(Tasks.separable([row]),Held.Both<Components.RowOk(limits,row,Tensors.grid_of(source_cells),source_cells),
    Held.Both<Components.RowEnds(row,classes),Targets.TargetClassesOk(classes,Tensors.grid_of(cells),cells,Machine.half_word(limits))>>)

# A block of a piece of a separable component reads at a valid lane a
# source cell its component takes: every target cell axis reads its source
# axis inside the classes of the cell the block writes first.
def read_separate_with(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
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
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Piece.piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(Piece.piece_boxes(values,target,position,piece)) == U32.to_nat(Piece.piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool},+component: U32,+index: U32,+read: Nat,+read_small: {Nat.is_lt(read,8n) == True{} : Bool},
  +valid: {U32.is_lt(U32.from_nat(read),Values.reads_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index))) == True{} : Bool},
  +separate: {Tasks.separable([relane_row(within,position,target,source,source_within,source_position,values)]) == True{} : Bool},
  +facts: Held.Both<Components.RowOk(limits,relane_row(within,position,target,source,source_within,source_position,values),
      Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position)),
    Held.Both<Components.RowEnds(relane_row(within,position,target,source,source_within,source_position,values),
      relane_classes(within,position,target,source,source_within,source_position,values,component)),
      Targets.TargetClassesOk(relane_classes(within,position,target,source,source_within,source_position,values,component),
        Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),Machine.half_word(limits))>>,
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(relane_components(within,position,target,source,source_within,source_position,values),
    Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),component),piece),
  +source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +parts: Piece.PieceCell(within,position,target,source,source_within,source_position,values,piece,index,
    Writes.low_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)))) ->
  Held.SpanHeld(relane_spans(within,position,target,source,source_within,source_position,values,component),
    U32.to_nat((Geometry.source_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)) +
      U32.from_nat(read) * Geometry.step_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)) : U32))):
  Held.Both{Held.Both{+rooms,Held.Both{+axes_ok,+axes_below}},Held.Both{+ends,+classes_ok}} = facts
  Held.Both{+coordinates,Held.Both{+number_in,+cell}} = parts
  Held.Both{+target_exact,+cell_below} = cell
  +cells = Tiled.cell_shape(values,target,position)
  +source_cells = Tiled.cell_shape(values,source,source_position)
  +source_grid = Tensors.grid_of(source_cells)
  +target_grid = Tensors.grid_of(cells)
  +rank = List.length(&2,U32,values)
  +axes_t = Tiled.target_axes(within,position,target,rank)
  +row = relane_row(within,position,target,source,source_within,source_position,values)
  +classes = relane_classes(within,position,target,source,source_within,source_position,values,component)
  +strides = Tasks.strides_of(source_cells)
  +groups_axis = Values.groups_axis(source_position,source_cells)
  +blocked = Tiled.blocked_list(axes_t,source,source_within,source_position,target,strides,values,groups_axis)
  +lows = Piece.piece_lows(values,target,position,piece)
  +boxes = Piece.piece_boxes(values,target,position,piece)
  +axes = Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece)
  +block = Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)
  +low = Writes.low_of(block)
  +lane = U32.to_nat(low)
  +digits = Geometry.block_digits(axes,index,lane)
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +capacity = Machine.capacity(limits)
  +cap_half = Bounds.as_nat_le(capacity,cap,Limits.capacity_within_half(limits))
  +same_blocked = Piece.same_blocked(values,source,source_position,source_within,target,position,within,piece,within_value)
  +same_lows = Piece.same_lows(values,target,position,piece)
  +below = {%within_value : Coordinates.DigitsBelow(Grid.placed(_,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),digits)
    cell_below : Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),digits)}
  +small = Piece.lane_small(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,low,Order.reflexive(lane),Piece.piece_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index))
  +channel_in = Nat.is_lt(Nat.add(read,Nat.mul(Values.cell_group(within,position,digits),8n)),U32.to_nat(Grid.lanes_at(values,target)))
  +counted = Values.read_count(limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,lane,read,same_blocked,same_lows,within_value,
    lanes_room,below,read_small)
  +validity = Equal.trans(Bool,channel_in,U32.is_lt(U32.from_nat(read),Values.reads_of(block)),True{},
    Equal.sym(Bool,U32.is_lt(U32.from_nat(read),Values.reads_of(block)),channel_in,counted),valid)
  +number_at = ShapeLaws.digits_value(boxes,Writes.box_digits(axes,index,lane))
  +classes_found = Targets.target_classes(limits,cells,Grid.piece_target(),cells_room,nonempty,Piece.cells_rank_room(limits,values,target,position,within,within_value,rank_room),
    [row],component,separate,classes_ok,piece,member,number_at,number_in)
  +classes_in = {%Equal.sym(List<&2,Nat>,digits,Targets.cell_coordinates(cells,Nat.add(U32.to_nat(Piece.piece_first(values,target,position,piece)),number_at)),coordinates) :
      Targets.ClassesIn(classes,target_grid,cells,_,0)
    classes_found : Targets.ClassesIn(classes,target_grid,cells,digits,0)}
  +sized = target_extents(within,position,target,values,within_value)
  +groups_ok = groups_target(within,position,digits,target,values,read,within_value,below,validity)
  +lane_value = source_groups(limits,source_within,source_position,values,source,source_value,source_lanes_room,rank_room)
  +values_read = relane_values(blocked,digits,read)
  +row_ok = placed_row(axes_t,cells,digits,classes,source,source_within,source_position,target,source_cells,source_grid,target_grid,values,0,read,cap,cap_ok,sized,classes_in,
    rooms,groups_ok,read_small,lane_value)
  +fixed = Grid.coordinates(source_cells,0)
  +source_double = Limits.double_bound(limits,ShapeLaws.elements(source_cells),source_cells_room)
  +found = TaskReads.digits_placed(row,classes,values_read,source_grid,source_cells,fixed,0,cap,cap_ok,row_ok,axes_ok,blocks,
    ViewReads.coordinates_in(source_cells,0,cap,cap_ok,source_double,source_nonempty))
  +placed = TaskReads.placed_digits(row,values_read,source_cells,fixed,0)
  +source_rank = Piece.cells_rank_room(limits,values,source,source_position,source_within,source_value,rank_room)
  +digit_count = Equal.trans(Nat,List.length(&2,Nat,digits),List.length(&2,U32,cells),List.length(&2,Tiled.CellAxis,axes_t),
    Values.below_length(cells,digits,cell_below),extents_length(axes_t,cells,values,target,sized))
  +paired = Equal.trans(Nat,List.length(&2,Tasks.Reading,row),List.length(&2,Tiled.CellAxis,axes_t),List.length(&2,Nat,values_read),
    readings_length(axes_t,cells,source,source_within,source_position,target,source_cells,source_grid,target_grid,values,0,sized),
    Equal.sym(Nat,List.length(&2,Nat,values_read),List.length(&2,Tiled.CellAxis,axes_t),
      values_length(axes_t,digits,source,source_within,source_position,target,strides,values,groups_axis,read,digit_count)))
  +apart = Logic.left(Tasks.distinct(Tasks.read_axes(row)),Bool.not(Tasks.crossing(row)),
    Logic.left(Tasks.distinct(Tasks.read_axes(row)) && Bool.not(Tasks.crossing(row)),Tasks.separable([]),separate))
  +number = TaskReads.placed_number(row,values_read,source_cells,fixed,cap,cap_ok,TaskReads.zero_offset(source_cells,cap,cap_ok,source_double,source_nonempty),apart,axes_below,paired,
    BoxDigits.coordinates_count(source_cells,0),Order.transitive(List.length(&2,U32,source_cells),U32.to_nat(capacity),U32.to_nat(cap),source_rank,cap_half))
  +word = relane_word(within,position,target,source,source_within,source_position,values,source_cells,source_grid,target_grid,digits,read,within_value,digit_count)
  +own = Geometry.blocked_of(axes)
  +sum = Geometry.blocked_sum(digits,own)
  +lane_read = U32.from_nat(read)
  +address = (Geometry.source_of(block) + lane_read * Geometry.step_of(block) : U32)
  +moved = Equal.trans(U32,address,((0 + sum : U32) + lane_read * Geometry.step_of(block) : U32),
    (Geometry.blocked_sum(digits,blocked) + lane_read * Geometry.blocked_step(blocked,0) : U32),
    Equal.cong(U32,U32,value => (value + lane_read * Geometry.step_of(block) : U32),Geometry.source_of(block),(0 + sum : U32),
      Geometry.block_source(axes,Writes.first_block(),index,lane,small)),
    Equal.trans(U32,((0 + sum : U32) + lane_read * Geometry.step_of(block) : U32),((0 + sum : U32) + lane_read * Geometry.blocked_step(own,0) : U32),
      (Geometry.blocked_sum(digits,blocked) + lane_read * Geometry.blocked_step(blocked,0) : U32),
      Equal.cong(U32,U32,value => ((0 + sum : U32) + lane_read * value : U32),Geometry.step_of(block),Geometry.blocked_step(own,0),
        Geometry.block_step(axes,Writes.first_block(),index)),
      Equal.trans(U32,((0 + sum : U32) + lane_read * Geometry.blocked_step(own,0) : U32),(sum + lane_read * Geometry.blocked_step(own,0) : U32),
        (Geometry.blocked_sum(digits,blocked) + lane_read * Geometry.blocked_step(blocked,0) : U32),
        Equal.cong(U32,U32,value => (value + lane_read * Geometry.blocked_step(own,0) : U32),(0 + sum : U32),sum,Arithmetic.add_zero_left(sum)),
        Equal.cong(List<&2,Tiled.Blocked>,U32,list => (Geometry.blocked_sum(digits,list) + lane_read * Geometry.blocked_step(list,0) : U32),own,blocked,
          Writes.ranged_blocked(blocked,lows,boxes,same_blocked,same_lows)))))
  +word_sum = TaskReads.placed_word(row,values_read,strides)
  +exact = Equal.trans(U32,address,(Geometry.blocked_sum(digits,blocked) + lane_read * Geometry.blocked_step(blocked,0) : U32),
    U32.from_nat(ShapeLaws.digits_value(source_cells,placed)),moved,
    Equal.trans(U32,(Geometry.blocked_sum(digits,blocked) + lane_read * Geometry.blocked_step(blocked,0) : U32),word_sum,U32.from_nat(ShapeLaws.digits_value(source_cells,placed)),
      Equal.sym(U32,word_sum,(Geometry.blocked_sum(digits,blocked) + lane_read * Geometry.blocked_step(blocked,0) : U32),word),
      Equal.sym(U32,U32.from_nat(ShapeLaws.digits_value(source_cells,placed)),word_sum,number)))
  +box = Tasks.boxed(row,classes,source_grid,source_cells,fixed,0)
  %Equal.sym(Bool,Tasks.separable([row]),True{},separate) :
    Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(source_grid,source_cells,Tasks.box_when(_,row,classes,source_grid,source_cells,0)),source_grid),U32.to_nat(address))
  TaskReads.held_of(limits,source_cells,box,placed,address,source_cells_room,source_nonempty,source_rank,
    BoxCells.boxed_coordinates_in(row,classes,source_grid,source_cells,0,cap,cap_ok,rooms,ends,axes_ok),ViewReads.digits_box(box,source_cells,placed,found),exact)

# A block of a piece of an inseparable component reads at a valid lane a
# source cell inside the source cells, all of which the component takes.
def read_whole_with(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
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
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Piece.piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(Piece.piece_boxes(values,target,position,piece)) == U32.to_nat(Piece.piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool},+component: U32,+index: U32,+read: Nat,+read_small: {Nat.is_lt(read,8n) == True{} : Bool},
  +valid: {U32.is_lt(U32.from_nat(read),Values.reads_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index))) == True{} : Bool},
  +whole: {Tasks.separable([relane_row(within,position,target,source,source_within,source_position,values)]) == False{} : Bool},
  +source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +parts: Piece.PieceCell(within,position,target,source,source_within,source_position,values,piece,index,
    Writes.low_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)))) ->
  Held.SpanHeld(relane_spans(within,position,target,source,source_within,source_position,values,component),
    U32.to_nat((Geometry.source_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)) +
      U32.from_nat(read) * Geometry.step_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)) : U32))):
  Held.Both{+coordinates,Held.Both{+number_in,+cell}} = parts
  Held.Both{+target_exact,+cell_below} = cell
  +cells = Tiled.cell_shape(values,target,position)
  +source_cells = Tiled.cell_shape(values,source,source_position)
  +source_grid = Tensors.grid_of(source_cells)
  +row = relane_row(within,position,target,source,source_within,source_position,values)
  +classes = relane_classes(within,position,target,source,source_within,source_position,values,component)
  +lows = Piece.piece_lows(values,target,position,piece)
  +boxes = Piece.piece_boxes(values,target,position,piece)
  +axes = Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece)
  +block = Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)
  +low = Writes.low_of(block)
  +lane = U32.to_nat(low)
  +digits = Geometry.block_digits(axes,index,lane)
  +same_blocked = Piece.same_blocked(values,source,source_position,source_within,target,position,within,piece,within_value)
  +same_lows = Piece.same_lows(values,target,position,piece)
  +below = {%within_value : Coordinates.DigitsBelow(Grid.placed(_,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),digits)
    cell_below : Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),digits)}
  +small = Piece.lane_small(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,low,Order.reflexive(lane),Piece.piece_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index))
  +channel_in = Nat.is_lt(Nat.add(read,Nat.mul(Values.cell_group(within,position,digits),8n)),U32.to_nat(Grid.lanes_at(values,target)))
  +counted = Values.read_count(limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,lane,read,same_blocked,same_lows,within_value,
    lanes_room,below,read_small)
  +validity = Equal.trans(Bool,channel_in,U32.is_lt(U32.from_nat(read),Values.reads_of(block)),True{},
    Equal.sym(Bool,U32.is_lt(U32.from_nat(read),Values.reads_of(block)),channel_in,counted),valid)
  +address = (Geometry.source_of(block) + U32.from_nat(read) * Geometry.step_of(block) : U32)
  +logical_at = Geometry.target_logical(within,position,target,digits,read)
  +number = ShapeLaws.digits_value(source_cells,Coordinates.placed_digits(source_within,source_position,Geometry.removed_at(logical_at,source),
    Nat.div(BoxDigits.digit_at(logical_at,source),8n)))
  +exact = Values.read_source(True{},limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,lane,read,same_blocked,same_lows,small,
    within_value,source_value,apart,source_cells_room,source_lanes_room,rank_room,below,validity)
  +address_below = {%Equal.sym(Nat,U32.to_nat(address),number,exact) : {Nat.is_lt(_,ShapeLaws.elements(source_cells)) == True{} : Bool}
    Values.read_cell_below(limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,lane,read,within_value,source_value,source_lanes_room,
      below,validity) : {Nat.is_lt(U32.to_nat(address),ShapeLaws.elements(source_cells)) == True{} : Bool}}
  %Equal.sym(Bool,Tasks.separable([row]),False{},whole) :
    Held.SpanHeld(Moved.taken_spans(Tasks.box_pieces(source_grid,source_cells,Tasks.box_when(_,row,classes,source_grid,source_cells,0)),source_grid),U32.to_nat(address))
  BoxCells.cell_held(limits,source_cells,Grid.piece_target(),source_cells_room,source_nonempty,
    Piece.cells_rank_room(limits,values,source,source_position,source_within,source_value,rank_room),Tasks.whole_spans(source_cells),U32.to_nat(address),
    Components.whole_in(source_cells),ViewReads.whole_in(source_cells,U32.to_nat(address),address_below),address_below)

# A block of a piece reads at a valid lane a source cell its component
# takes, the component separable or not.
def read_choice(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
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
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Piece.piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(Piece.piece_boxes(values,target,position,piece)) == U32.to_nat(Piece.piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool},separate: Bool,+component: U32,+index: U32,+read: Nat,+read_small: {Nat.is_lt(read,8n) == True{} : Bool},
  +valid: {U32.is_lt(U32.from_nat(read),Values.reads_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index))) == True{} : Bool},
  +separate_value: {Tasks.separable([relane_row(within,position,target,source,source_within,source_position,values)]) == separate : Bool},
  +facts: Logic.When(separate,Held.Both<Components.RowOk(limits,relane_row(within,position,target,source,source_within,source_position,values),
      Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position)),
    Held.Both<Components.RowEnds(relane_row(within,position,target,source,source_within,source_position,values),
      relane_classes(within,position,target,source,source_within,source_position,values,component)),
      Targets.TargetClassesOk(relane_classes(within,position,target,source,source_within,source_position,values,component),
        Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),Machine.half_word(limits))>>),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(relane_components(within,position,target,source,source_within,source_position,values),
    Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),component),piece),
  +source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool},
  +parts: Piece.PieceCell(within,position,target,source,source_within,source_position,values,piece,index,
    Writes.low_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)))) ->
  Held.SpanHeld(relane_spans(within,position,target,source,source_within,source_position,values,component),
    U32.to_nat((Geometry.source_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)) +
      U32.from_nat(read) * Geometry.step_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)) : U32))):
  match separate:
    case True{}:
      read_separate_with(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,component,index,read,read_small,valid,separate_value,facts,blocks,member,source_nonempty,parts)
    case False{}:
      read_whole_with(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,component,index,read,read_small,valid,separate_value,source_nonempty,parts)

# A block of a piece reads at every valid lane a source cell its component
# takes.
def read_when(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
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
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Piece.piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(Piece.piece_boxes(values,target,position,piece)) == U32.to_nat(Piece.piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool},valid: Bool,+component: U32,+index: U32,+read: Nat,+read_small: {Nat.is_lt(read,8n) == True{} : Bool},
  +valid_value: {U32.is_lt(U32.from_nat(read),Values.reads_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index))) ==
    valid : Bool},
  +separate: RelaneSeparate(limits,within,position,target,source,source_within,source_position,values,component),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(relane_components(within,position,target,source,source_within,source_position,values),
    Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),component),piece),
  +source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool}) ->
  Logic.When(valid,Held.SpanHeld(relane_spans(within,position,target,source,source_within,source_position,values,component),
    U32.to_nat((Geometry.source_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)) +
      U32.from_nat(read) * Geometry.step_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)) : U32)))):
  match valid:
    case False{}: Unit{}
    case True{}:
      +block = Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)
      +low = Writes.low_of(block)
      +cells = Tiled.cell_shape(values,target,position)
      +parts = Piece.piece_cell_of(limits,values,source,source_position,source_within,target,position,within,piece,cells_room,nonempty,piece_inside,box_cells,index,low,
        Piece.boxes_count(values,target,position,piece),Limits.double_bound(limits,ShapeLaws.elements(cells),cells_room),
        Piece.cells_rank_room(limits,values,target,position,within,within_value,rank_room),
        Piece.piece_facts(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index,low,Order.reflexive(U32.to_nat(low)),Piece.piece_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,index)))
      read_choice(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,Tasks.separable([relane_row(within,position,target,source,source_within,source_position,values)]),component,index,read,read_small,valid_value,
        {==},separate,blocks,member,source_nonempty,parts)

# Every cell a block of a piece reads lies in a source piece its component
# takes.
def reads_held(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
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
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Piece.piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(Piece.piece_boxes(values,target,position,piece)) == U32.to_nat(Piece.piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool},+component: U32,+index: U32,
  +separate: RelaneSeparate(limits,within,position,target,source,source_within,source_position,values,component),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(relane_components(within,position,target,source,source_within,source_position,values),
    Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),component),piece),
  +source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool}) ->
  Block.ReadsHeld<relane_spans(within,position,target,source,source_within,source_position,values,component),
    Values.reads_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)),
    Geometry.source_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index)),
    Geometry.step_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index))>:
  +reads = Values.reads_of(Piece.piece_block(within,position,target,source,source_within,source_position,values,piece,index))
  Block.ReadsHeld{
    read_when(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,U32.is_lt(0,reads),component,index,0n,{==},{==},separate,blocks,member,source_nonempty),
    read_when(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,U32.is_lt(1,reads),component,index,1n,{==},{==},separate,blocks,member,source_nonempty),
    read_when(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,U32.is_lt(2,reads),component,index,2n,{==},{==},separate,blocks,member,source_nonempty),
    read_when(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,U32.is_lt(3,reads),component,index,3n,{==},{==},separate,blocks,member,source_nonempty),
    read_when(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,U32.is_lt(4,reads),component,index,4n,{==},{==},separate,blocks,member,source_nonempty),
    read_when(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,U32.is_lt(5,reads),component,index,5n,{==},{==},separate,blocks,member,source_nonempty),
    read_when(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,U32.is_lt(6,reads),component,index,6n,{==},{==},separate,blocks,member,source_nonempty),
    read_when(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,U32.is_lt(7,reads),component,index,7n,{==},{==},separate,blocks,member,source_nonempty)}

# Every cell the blocks of a piece read from index on lies in a source piece
# its component takes.
def blocks_held(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
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
  +piece_inside: {Nat.is_lt(U32.to_nat(piece),U32.to_nat(Grid.pieces_of(Piece.piece_grid(values,target,position)))) == True{} : Bool},
  +box_cells: {ShapeLaws.elements(Piece.piece_boxes(values,target,position,piece)) == U32.to_nat(Piece.piece_span(values,target,position,piece)) : Nat},
  +span_positive: {Nat.is_lt(0n,U32.to_nat(Piece.piece_span(values,target,position,piece))) == True{} : Bool},fuel: Nat,+component: U32,+index: U32,
  +separate: RelaneSeparate(limits,within,position,target,source,source_within,source_position,values,component),
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position),0,Machine.half_word(limits)),
  +member: Moved.Member(Tasks.target_pieces(relane_components(within,position,target,source,source_within,source_position,values),
    Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),component),piece),
  +source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool}) ->
  Piece.BlocksHeld(fuel,index,Piece.piece_axes(within,position,target,source,source_within,source_position,values,piece),
    relane_spans(within,position,target,source,source_within,source_position,values,component)):
  match fuel:
    case 0n: Unit{}
    case 1n+ +rest:
      Held.Both{reads_held(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,component,index,separate,blocks,member,source_nonempty),
        blocks_held(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,piece,within_value,inside,room,lanes_room,cells_room,nonempty,
    source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,piece_inside,box_cells,span_positive,rest,component,(index + 1 : U32),separate,blocks,member,source_nonempty)}

# ---- What every component of a separable relane needs ----

# The classes of a relane row end inside the blocks of the target cell axes.
def relane_class_ends(axes: List<&2,Tiled.CellAxis>,extents: List<&2,U32>,classes: List<&2,Tasks.Class>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +target: Nat,+source_cells: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+shape: List<&2,U32>,+index: U32,+cap: U32,
  +sized: AxesExtents(axes,extents,shape,target),+same: {List.length(&2,Tasks.Class,classes) == List.length(&2,U32,extents) : Nat},
  +rooms: Components.ReadingsRoom(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,extents,index),cap),
  +ends: Components.RowEnds(Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,axes,extents,index),classes)) ->
  Targets.ClassEnds(classes,target_grid,extents,index,cap):
  match axes extents classes:
    case Con{Tiled.OfAxis{+axis},+rest} Con{+extent,+more} Con{+class,+others}:
      Held.Both{+here,+later} = sized
      +block = Grid.block(target_grid,index,extent)
      +read = Tiled.logical_reading(source,source_within,source_position,source_cells,source_grid,axis,Grid.lanes_at(shape,axis))
      +row = Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,rest,more,(index + 1 : U32))
      Held.Both{ViewReads.ends_head(extent,block,read,row,class,others,ends),Held.Both{ViewReads.room_sum(extent,block,read,row,cap,rooms),
        relane_class_ends(rest,more,others,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),cap,later,
          Components.length_back(List.length(&2,Tasks.Class,others),List.length(&2,U32,more),same),ViewReads.rooms_tail(extent,block,read,row,cap,rooms),
          ViewReads.ends_tail(extent,block,read,row,class,others,ends))}}
    case Con{Tiled.OfGroups{},+rest} Con{+extent,+more} Con{+class,+others}:
      Held.Both{+here,+later} = sized
      +block = Grid.block(target_grid,index,extent)
      +read = Tiled.read_at(source_grid,Tiled.source_axis(source,source_within,source_position,target),Grid.lanes_at(shape,target),8,8,1)
      +row = Tiled.relane_readings(source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,rest,more,(index + 1 : U32))
      Held.Both{ViewReads.ends_head(extent,block,read,row,class,others,ends),Held.Both{ViewReads.room_sum(extent,block,read,row,cap,rooms),
        relane_class_ends(rest,more,others,source,source_within,source_position,target,source_cells,source_grid,target_grid,shape,(index + 1 : U32),cap,later,
          Components.length_back(List.length(&2,Tasks.Class,others),List.length(&2,U32,more),same),ViewReads.rooms_tail(extent,block,read,row,cap,rooms),
          ViewReads.ends_tail(extent,block,read,row,class,others,ends))}}
    case Con{Tiled.OfAxis{+axis},+rest} Con{+extent,+more} Nil{}:
      Empty.absurd(Targets.ClassEnds(Nil{},target_grid,Con{extent,more},index,cap),Components.zero_successor(List.length(&2,U32,more),same))
    case Con{Tiled.OfGroups{},+rest} Con{+extent,+more} Nil{}:
      Empty.absurd(Targets.ClassEnds(Nil{},target_grid,Con{extent,more},index,cap),Components.zero_successor(List.length(&2,U32,more),same))
    case Con{Tiled.OfAxis{+axis},+rest} Nil{} +classes:
      Empty.absurd(Targets.ClassEnds(classes,target_grid,Nil{},index,cap),sized)
    case Con{Tiled.OfGroups{},+rest} Nil{} +classes:
      Empty.absurd(Targets.ClassEnds(classes,target_grid,Nil{},index,cap),sized)
    case Nil{} Con{+extent,+more} +classes:
      Empty.absurd(Targets.ClassEnds(classes,target_grid,Con{extent,more},index,cap),sized)
    case Nil{} Nil{} Nil{}:
      Unit{}
    case Nil{} Nil{} Con{+class,+others}:
      Empty.absurd(Targets.ClassEnds(Con{class,others},target_grid,Nil{},index,cap),
        Components.zero_successor(List.length(&2,Tasks.Class,others),Equal.sym(Nat,1n+List.length(&2,Tasks.Class,others),0n,same)))

# The classes of a component of a separable relane cover the target pieces
# and end inside their blocks.
def relane_classes_ok(+limits: Machine.Limits,+within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +values: List<&2,U32>,+component: U32,+within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},
  +cells_ok: Components.ShapeOk(limits,Tiled.cell_shape(values,target,position)),
  +components_ok: Components.ComponentsOk([relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),Machine.half_word(limits)),
  +limits_at: Components.LimitsAt([relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),Tiled.cell_shape(values,target,position),Tensors.grid_of(Tiled.cell_shape(values,target,position)),0),
  +inside: {Nat.is_lt(U32.to_nat(component),Components.count_nat(Components.raw_lists(Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),[relane_row(within,position,target,source,source_within,source_position,values)]))) == True{} : Bool},
  +rooms: Components.ReadingsRoom(relane_row(within,position,target,source,source_within,source_position,values),Machine.half_word(limits)),+ends: Components.RowEnds(relane_row(within,position,target,source,source_within,source_position,values),relane_classes(within,position,target,source,source_within,source_position,values,component))) ->
  Targets.TargetClassesOk(relane_classes(within,position,target,source,source_within,source_position,values,component),Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),Machine.half_word(limits)):
  Held.Both{+room,Held.Both{+nonempty,+rank_room}} = cells_ok
  +cells = Tiled.cell_shape(values,target,position)
  +source_cells = Tiled.cell_shape(values,source,source_position)
  +axes = Tiled.target_axes(within,position,target,List.length(&2,U32,values))
  +sized = target_extents(within,position,target,values,within_value)
  +same = Equal.trans(Nat,List.length(&2,Tasks.Class,relane_classes(within,position,target,source,source_within,source_position,values,component)),Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),List.length(&2,U32,cells),
    Targets.chosen_rank(Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),[relane_row(within,position,target,source,source_within,source_position,values)],component),
    Equal.trans(Nat,List.length(&2,Tasks.Reading,relane_row(within,position,target,source,source_within,source_position,values)),List.length(&2,Tiled.CellAxis,axes),List.length(&2,U32,cells),
      readings_length(axes,cells,source,source_within,source_position,target,source_cells,Tensors.grid_of(source_cells),Tensors.grid_of(cells),values,0,sized),
      Equal.sym(Nat,List.length(&2,U32,cells),List.length(&2,Tiled.CellAxis,axes),extents_length(axes,cells,values,target,sized))))
  Targets.classes_ok(limits,cells,Grid.piece_target(),room,nonempty,rank_room,[relane_row(within,position,target,source,source_within,source_position,values)],component,components_ok,limits_at,inside,
    relane_class_ends(axes,cells,relane_classes(within,position,target,source,source_within,source_position,values,component),source,source_within,source_position,target,source_cells,Tensors.grid_of(source_cells),Tensors.grid_of(cells),values,0,
      Machine.half_word(limits),sized,same,rooms,ends))

def relane_separate_yes(+limits: Machine.Limits,+within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +values: List<&2,U32>,+component: U32,+within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},
  +cells_ok: Components.ShapeOk(limits,Tiled.cell_shape(values,target,position)),
  +components_ok: Components.ComponentsOk([relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),Machine.half_word(limits)),
  +limits_at: Components.LimitsAt([relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),Tiled.cell_shape(values,target,position),Tensors.grid_of(Tiled.cell_shape(values,target,position)),0),
  +inside: {Nat.is_lt(U32.to_nat(component),Components.count_nat(Components.raw_lists(Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),[relane_row(within,position,target,source,source_within,source_position,values)]))) == True{} : Bool},
  +separate: {Tasks.separable([relane_row(within,position,target,source,source_within,source_position,values)]) == True{} : Bool},
  +row_ok: Components.RowOk(limits,relane_row(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position))) ->
  RelaneSeparate(limits,within,position,target,source,source_within,source_position,values,component):
  Held.Both{+rooms,+more} = row_ok
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +ends = Components.ends_at([relane_row(within,position,target,source,source_within,source_position,values)],0,relane_classes(within,position,target,source,source_within,source_position,values,component),cap,cap_ok,Components.runtime_ends([relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),cap,cap_ok,components_ok,component,inside),{==},
    Components.components_rows([relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),cap,components_ok))
  {%Equal.sym(Bool,Tasks.separable([relane_row(within,position,target,source,source_within,source_position,values)]),True{},separate) :
      Logic.When(_,Held.Both<Components.RowOk(limits,relane_row(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position)),
        Held.Both<Components.RowEnds(relane_row(within,position,target,source,source_within,source_position,values),relane_classes(within,position,target,source,source_within,source_position,values,component)),
          Targets.TargetClassesOk(relane_classes(within,position,target,source,source_within,source_position,values,component),Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),Machine.half_word(limits))>>)
    Held.Both{row_ok,Held.Both{ends,relane_classes_ok(limits,within,position,target,source,source_within,source_position,values,component,within_value,cells_ok,
      components_ok,limits_at,inside,rooms,ends)}} :
    RelaneSeparate(limits,within,position,target,source,source_within,source_position,values,component)}

# Every component of a relane below the class count has what a separable
# relane needs of it.
def relane_separate(+limits: Machine.Limits,+within: Bool,+position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +values: List<&2,U32>,+component: U32,+within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},
  +cells_ok: Components.ShapeOk(limits,Tiled.cell_shape(values,target,position)),
  +components_ok: Components.ComponentsOk([relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),Machine.half_word(limits)),
  +limits_at: Components.LimitsAt([relane_row(within,position,target,source,source_within,source_position,values)],Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),Tiled.cell_shape(values,target,position),Tensors.grid_of(Tiled.cell_shape(values,target,position)),0),
  +inside: {Nat.is_lt(U32.to_nat(component),Components.count_nat(Components.raw_lists(Tasks.target_rank([relane_row(within,position,target,source,source_within,source_position,values)]),[relane_row(within,position,target,source,source_within,source_position,values)]))) == True{} : Bool},
  +row_ok: Logic.When(Tasks.separable([relane_row(within,position,target,source,source_within,source_position,values)]),Components.RowOk(limits,relane_row(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position)))) ->
  RelaneSeparate(limits,within,position,target,source,source_within,source_position,values,component):
  Held.by_cases(RelaneSeparate(limits,within,position,target,source,source_within,source_position,values,component),Tasks.separable([relane_row(within,position,target,source,source_within,source_position,values)]),
    +yes => relane_separate_yes(limits,within,position,target,source,source_within,source_position,values,component,within_value,cells_ok,components_ok,limits_at,inside,yes,
      {%yes : Logic.When(_,Components.RowOk(limits,relane_row(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position)))
        row_ok : Components.RowOk(limits,relane_row(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position))}),
    +no => {%Equal.sym(Bool,Tasks.separable([relane_row(within,position,target,source,source_within,source_position,values)]),False{},no) :
        Logic.When(_,Held.Both<Components.RowOk(limits,relane_row(within,position,target,source,source_within,source_position,values),Tensors.grid_of(Tiled.cell_shape(values,source,source_position)),Tiled.cell_shape(values,source,source_position)),
          Held.Both<Components.RowEnds(relane_row(within,position,target,source,source_within,source_position,values),relane_classes(within,position,target,source,source_within,source_position,values,component)),
            Targets.TargetClassesOk(relane_classes(within,position,target,source,source_within,source_position,values,component),Tensors.grid_of(Tiled.cell_shape(values,target,position)),Tiled.cell_shape(values,target,position),Machine.half_word(limits))>>)
      Unit{} : RelaneSeparate(limits,within,position,target,source,source_within,source_position,values,component)})
