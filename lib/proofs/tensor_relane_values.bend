# What a write of a relane block holds: lane j of the target cell it writes
# at lane i of its block is lane i of the source cell it reads at j, and both
# hold the value at the same logical digits.

import Base
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../kernel_gemm.bend as KernelGemm
import ./tensor_relane_geometry.bend as Geometry
import ./tensor_tile_coordinates.bend as Coordinates
import ./tensor_box_digits.bend as BoxDigits
import ./nat_order.bend as Order
import ./word_encoding.bend as WordEncoding
import ./tensor_tiled_public.bend as Public
import ./bounded_u32_arithmetic.bend as Arithmetic
import ../tensor_tasks.bend as Tasks
import ./natural_addition.bend as Addition
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_comparison.bend as Comparison
import ./u32_extrema.bend as Extrema
import ./nat_extrema.bend as NatExtrema
import ./nat_ceiling.bend as Ceiling
import ./encoded_coordinates.bend as Encoded
import ./tensor_relane_writes.bend as Writes
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_tiled_representation.bend as Representation
import ./kernel_tile_refinement.bend as TileLaws
import ./nat_division_order.bend as DivisionOrder
import ../kernel_tile.bend as Tile
import ./boolean_logic.bend as Logic
import ./tensor_relane_block.bend as Block

# ---- The parts of the digits of a cell ----

# The digits of a cell but its group, and its group.
def cell_others(within: Bool,position: Grid.Position,+digits: List<&2,Nat>) -> List<&2,Nat>:
  match within position:
    case False{} position: digits
    case True{} Grid.Innermost{}: Geometry.digits_front(digits)
    case True{} Grid.Outermost{}: Geometry.digits_tail(digits)

def cell_group(within: Bool,position: Grid.Position,+digits: List<&2,Nat>) -> Nat:
  match within position:
    case False{} position: 0n
    case True{} Grid.Innermost{}: Geometry.digits_back(digits)
    case True{} Grid.Outermost{}: Geometry.digits_head(digits)

# The logical digits of target digits at a lane are those of the channel of
# the lane in the group of the cell.
def target_cells(within: Bool,position: Grid.Position,+target: Nat,+digits: List<&2,Nat>,+lane: Nat) ->
  {Geometry.target_logical(within,position,target,digits,lane) ==
    Coordinates.logical_digits(within,target,cell_others(within,position,digits),Nat.add(lane,Nat.mul(cell_group(within,position,digits),8n))) : List<&2,Nat>}:
  match within position:
    case False{} +position: {==}
    case True{} Grid.Innermost{}: {==}
    case True{} Grid.Outermost{}: {==}

# Digits below extents with one more at the end: the digits but the last lie
# below the extents, the last below the last extent, and they are the digits
# but the last and the last.
def appended_empty(extents: List<&2,U32>,+last: U32,+below: Coordinates.DigitsBelow(List.append(&2,U32,extents,[last]),Nil{})) -> Empty:
  match extents:
    case Nil{}: below
    case Con{+extent,+rest}: below

def below_front(extents: List<&2,U32>,digits: List<&2,Nat>,+last: U32,+below: Coordinates.DigitsBelow(List.append(&2,U32,extents,[last]),digits)) ->
  Coordinates.DigitsBelow(extents,Geometry.digits_front(digits)):
  match extents digits:
    case Nil{} Nil{}: Empty.absurd(Coordinates.DigitsBelow(Nil{},Nil{}),below)
    case Nil{} Con{+digit,Nil{}}: Unit{}
    case Nil{} Con{+digit,Con{+next,+more}}:
      KernelGemm.Both{+here,+rest} = below
      Empty.absurd(Coordinates.DigitsBelow(Nil{},Geometry.digits_front(Con{digit,Con{next,more}})),rest)
    case Con{+extent,+rest} Nil{}: Empty.absurd(Coordinates.DigitsBelow(Con{extent,rest},Nil{}),below)
    case Con{+extent,+rest} Con{+digit,Nil{}}:
      KernelGemm.Both{+here,+later} = below
      Empty.absurd(Coordinates.DigitsBelow(Con{extent,rest},Nil{}),appended_empty(rest,last,later))
    case Con{+extent,+rest} Con{+digit,Con{+next,+more}}:
      KernelGemm.Both{+here,+later} = below
      KernelGemm.Both{here,below_front(rest,Con{next,more},last,later)}

def below_back(extents: List<&2,U32>,digits: List<&2,Nat>,+last: U32,+below: Coordinates.DigitsBelow(List.append(&2,U32,extents,[last]),digits)) ->
  {Nat.is_lt(Geometry.digits_back(digits),U32.to_nat(last)) == True{} : Bool}:
  match extents digits:
    case Nil{} Nil{}: Empty.absurd({Nat.is_lt(0n,U32.to_nat(last)) == True{} : Bool},below)
    case Nil{} Con{+digit,Nil{}}:
      KernelGemm.Both{+here,+rest} = below
      here
    case Nil{} Con{+digit,Con{+next,+more}}:
      KernelGemm.Both{+here,+rest} = below
      Empty.absurd({Nat.is_lt(Geometry.digits_back(Con{next,more}),U32.to_nat(last)) == True{} : Bool},rest)
    case Con{+extent,+rest} Nil{}: Empty.absurd({Nat.is_lt(0n,U32.to_nat(last)) == True{} : Bool},below)
    case Con{+extent,+rest} Con{+digit,Nil{}}:
      KernelGemm.Both{+here,+later} = below
      Empty.absurd({Nat.is_lt(digit,U32.to_nat(last)) == True{} : Bool},appended_empty(rest,last,later))
    case Con{+extent,+rest} Con{+digit,Con{+next,+more}}:
      KernelGemm.Both{+here,+later} = below
      below_back(rest,Con{next,more},last,later)

def below_whole(extents: List<&2,U32>,digits: List<&2,Nat>,+last: U32,+below: Coordinates.DigitsBelow(List.append(&2,U32,extents,[last]),digits)) ->
  {List.append(&2,Nat,Geometry.digits_front(digits),[Geometry.digits_back(digits)]) == digits : List<&2,Nat>}:
  match digits:
    case Nil{}:
      Empty.absurd({List.append(&2,Nat,Geometry.digits_front(Nil{}),[Geometry.digits_back(Nil{})]) == Nil{} : List<&2,Nat>},appended_empty(extents,last,below))
    case Con{+digit,+rest}: Geometry.front_back(rest,digit)

# The digits of a cell below its extents: its other digits below the other
# axes and its group below the groups.
def cell_below(within: Bool,position: Grid.Position,+others: List<&2,U32>,+groups: U32,digits: List<&2,Nat>,
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,others,groups),digits)) -> Coordinates.DigitsBelow(others,cell_others(within,position,digits)):
  match within position digits:
    case False{} +position +digits: below
    case True{} Grid.Innermost{} +digits: below_front(others,digits,groups,below)
    case True{} Grid.Outermost{} Nil{}: Empty.absurd(Coordinates.DigitsBelow(others,Nil{}),below)
    case True{} Grid.Outermost{} Con{+digit,+rest}:
      KernelGemm.Both{+here,+later} = below
      later

def group_below(within: Bool,position: Grid.Position,+others: List<&2,U32>,+groups: U32,digits: List<&2,Nat>,
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,others,groups),digits)) -> Coordinates.GroupBelow(within,cell_group(within,position,digits),groups):
  match within position digits:
    case False{} +position +digits: Unit{}
    case True{} Grid.Innermost{} +digits: below_back(others,digits,groups,below)
    case True{} Grid.Outermost{} Nil{}: Empty.absurd({Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},below)
    case True{} Grid.Outermost{} Con{+digit,+rest}:
      KernelGemm.Both{+here,+later} = below
      here

def cell_whole(within: Bool,position: Grid.Position,+others: List<&2,U32>,+groups: U32,digits: List<&2,Nat>,
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,others,groups),digits)) ->
  {Coordinates.placed_digits(within,position,cell_others(within,position,digits),cell_group(within,position,digits)) == digits : List<&2,Nat>}:
  match within position digits:
    case False{} +position +digits: {==}
    case True{} Grid.Innermost{} +digits: below_whole(others,digits,groups,below)
    case True{} Grid.Outermost{} Nil{}: Empty.absurd({Con{0n,Nil{}} == Nil{} : List<&2,Nat>},below)
    case True{} Grid.Outermost{} Con{+digit,+rest}: {==}

# ---- Logical digits below the values ----

# The other digits of a channel and the channel below their extents are
# logical digits below the values.
def inserted_below(values: List<&2,U32>,axis: Nat,others: List<&2,Nat>,+channel: Nat,+below: Coordinates.DigitsBelow(Grid.removed(values,axis),others),
  +inside: {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))) == True{} : Bool},+within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  Coordinates.DigitsBelow(values,Coordinates.inserted_digit(axis,others,channel)):
  match values axis others:
    case Con{+extent,+rest} 0n +others: KernelGemm.Both{inside,below}
    case Con{+extent,+rest} 1n+ +next Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      KernelGemm.Both{here,inserted_below(rest,next,more,channel,later,inside,within)}
    case Con{+extent,+rest} 1n+ +next Nil{}:
      Empty.absurd(Coordinates.DigitsBelow(Con{extent,rest},Coordinates.inserted_digit(1n+next,Nil{},channel)),below)
    case Nil{} +axis +others:
      Empty.absurd(Coordinates.DigitsBelow(Nil{},Coordinates.inserted_digit(axis,others,channel)),Order.below_zero(axis,within))

def logical_below(within: Bool,+values: List<&2,U32>,+axis: Nat,+others: List<&2,Nat>,+channel: Nat,
  +within_value: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool},+below: Coordinates.DigitsBelow(Grid.removed(values,axis),others),
  +inside: {Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,axis))) == True{} : Bool}) ->
  Coordinates.DigitsBelow(values,Coordinates.logical_digits(within,axis,others,channel)):
  match within:
    case True{}: inserted_below(values,axis,others,channel,below,inside,within_value)
    case False{}: {%Public.removed_past(values,axis,within_value) : Coordinates.DigitsBelow(_,others) below : Coordinates.DigitsBelow(values,others)}

# Logical digits below the values: the digits but one axis below the other
# axes, and the digit of the axis below its extent.
def removed_below(values: List<&2,U32>,digits: List<&2,Nat>,axis: Nat,+below: Coordinates.DigitsBelow(values,digits)) ->
  Coordinates.DigitsBelow(Grid.removed(values,axis),Geometry.removed_at(digits,axis)):
  match values digits axis:
    case Con{+extent,+rest} Con{+digit,+more} 0n:
      KernelGemm.Both{+here,+later} = below
      later
    case Con{+extent,+rest} Con{+digit,+more} 1n+ +next:
      KernelGemm.Both{+here,+later} = below
      KernelGemm.Both{here,removed_below(rest,more,next,later)}
    case Nil{} Nil{} +axis: Unit{}
    case Nil{} Con{+digit,+more} +axis: Empty.absurd(Coordinates.DigitsBelow(Grid.removed(Nil{},axis),Geometry.removed_at(Con{digit,more},axis)),below)
    case Con{+extent,+rest} Nil{} +axis: Empty.absurd(Coordinates.DigitsBelow(Grid.removed(Con{extent,rest},axis),Geometry.removed_at(Nil{},axis)),below)

def digit_below(values: List<&2,U32>,digits: List<&2,Nat>,axis: Nat,+below: Coordinates.DigitsBelow(values,digits),
  +within: {Nat.is_lt(axis,List.length(&2,U32,values)) == True{} : Bool}) ->
  {Nat.is_lt(BoxDigits.digit_at(digits,axis),U32.to_nat(Grid.lanes_at(values,axis))) == True{} : Bool}:
  match values digits axis:
    case Con{+extent,+rest} Con{+digit,+more} 0n:
      KernelGemm.Both{+here,+later} = below
      here
    case Con{+extent,+rest} Con{+digit,+more} 1n+ +next:
      KernelGemm.Both{+here,+later} = below
      digit_below(rest,more,next,later,within)
    case Nil{} +digits +axis:
      Empty.absurd({Nat.is_lt(BoxDigits.digit_at(digits,axis),1n) == True{} : Bool},Order.below_zero(axis,within))
    case Con{+extent,+rest} Nil{} +axis:
      Empty.absurd({Nat.is_lt(0n,U32.to_nat(Grid.lanes_at(Con{extent,rest},axis))) == True{} : Bool},below)

# The digits with the digit of an axis put back at the axis are the digits.
def reinserted(digits: List<&2,Nat>,axis: Nat,+within: {Nat.is_lt(axis,List.length(&2,Nat,digits)) == True{} : Bool}) ->
  {Coordinates.inserted_digit(axis,Geometry.removed_at(digits,axis),BoxDigits.digit_at(digits,axis)) == digits : List<&2,Nat>}:
  match digits axis:
    case Con{+digit,+more} 0n: {==}
    case Con{+digit,+more} 1n+ +next:
      Equal.cong(List<&2,Nat>,List<&2,Nat>,rest => digit <> rest,Coordinates.inserted_digit(next,Geometry.removed_at(more,next),BoxDigits.digit_at(more,next)),more,
        reinserted(more,next,within))
    case Nil{} +axis:
      Empty.absurd({Coordinates.inserted_digit(axis,Geometry.removed_at(Nil{},axis),BoxDigits.digit_at(Nil{},axis)) == Nil{} : List<&2,Nat>},Order.below_zero(axis,within))

# ---- The cells a block reads ----

def reads_of(+block: Tiled.Block) -> U32:
  Tiled.Block{source,target,stride,low,high,+reads,step} = block
  reads

# The cells a block reads: the target groups read up to eight source cells,
# no more than the lanes left in their group.
def blocked_reads(blocked: List<&2,Tiled.Blocked>,digits: List<&2,Nat>,+reads: U32) -> U32:
  match blocked digits:
    case Con{Tiled.Blocked{Tiled.Groups{},source,+lanes},rest} Con{+digit,more}: blocked_reads(rest,more,U32.min(8,(lanes - U32.from_nat(digit) * 8 : U32)))
    case Con{blocked,rest} Con{digit,more}: blocked_reads(rest,more,reads)
    case blocked digits: reads

def block_reads(axes: List<&2,Tiled.Ranged>,block: Tiled.Block,+index: U32,+lane: Nat) ->
  {reads_of(Tiled.block_at(axes,index,block)) == blocked_reads(Geometry.blocked_of(axes),Geometry.block_digits(axes,index,lane),reads_of(block)) : U32}:
  match axes block:
    case Nil{} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}: {==}
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      +at = (index / U32.max(inner,1) % U32.max(count,1) : U32)
      +next = Tiled.block_axis(Tiled.Groups{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},at,low,extent,source,lanes,target)
      +later = Geometry.block_digits(rest,index,lane)
      Equal.trans(U32,reads_of(Tiled.block_at(rest,index,next)),blocked_reads(Geometry.blocked_of(rest),later,reads_of(next)),
        blocked_reads(Geometry.blocked_of(rest),later,U32.min(8,(lanes - U32.from_nat(Nat.add(U32.to_nat(low),U32.to_nat(at))) * 8 : U32))),
        block_reads(rest,next,index,lane),
        Equal.cong(U32,U32,group => blocked_reads(Geometry.blocked_of(rest),later,U32.min(8,(lanes - group * 8 : U32))),(low + at : U32),
          U32.from_nat(Nat.add(U32.to_nat(low),U32.to_nat(at))),Arithmetic.native_add(low,at)))
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      block_reads(rest,Tiled.block_axis(Tiled.Shared{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},(index / U32.max(inner,1) % U32.max(count,1) : U32),
        low,extent,source,lanes,target),index,lane)
    case Con{Tiled.Ranged{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+low,+extent,+count,+inner,+target},+rest} Tiled.Block{+first,+start,+stride,+block_low,+block_high,+reads,+step}:
      block_reads(rest,Tiled.block_axis(Tiled.Lanes{},Tiled.Block{first,start,stride,block_low,block_high,reads,step},(index / U32.max(inner,1) % U32.max(count,1) : U32),
        low,extent,source,lanes,target),index,lane)

# Axes without target groups keep the cells a block reads.
def NoGroups(blocked: List<&2,Tiled.Blocked>) -> Data:
  match blocked:
    case Nil{}: Unit
    case Con{Tiled.Blocked{Tiled.Groups{},source,lanes},rest}: Empty
    case Con{Tiled.Blocked{role,source,lanes},rest}: NoGroups(rest)

def groups_pick(choice: Bool,+lane_source: U32,+lane_lanes: U32,+source: U32,+rest: List<&2,Tiled.Blocked>,+none: NoGroups(rest)) ->
  NoGroups(Bool.pick(Tiled.Blocked,choice,Tiled.Blocked{Tiled.Lanes{},lane_source,lane_lanes},Tiled.Blocked{Tiled.Shared{},source,1}) <> rest):
  match choice:
    case True{}: none
    case False{}: none

def groups_axes_pick(choice: Bool,+later: List<&2,Tiled.CellAxis>,+index: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+none: NoGroups(Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis))) ->
  NoGroups(Tiled.blocked_list(Bool.pick(List<&2,Tiled.CellAxis>,choice,later,Tiled.OfAxis{index} <> later),source,source_within,source_position,target,strides,shape,groups_axis)):
  match choice:
    case True{}: none
    case False{}:
      groups_pick(Bool.and(source_within,Nat.is_eq(index,source)),Tasks.nth(strides,groups_axis),Grid.lanes_at(shape,source),
        Tasks.nth(strides,Tiled.source_axis(source,source_within,source_position,index)),Tiled.blocked_list(later,source,source_within,source_position,target,strides,shape,groups_axis),none)

def axes_groups(remaining: Nat,+index: Nat,+skip: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+target: Nat,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32) ->
  NoGroups(Tiled.blocked_list(Tiled.axes_from(remaining,index,skip),source,source_within,source_position,target,strides,shape,groups_axis)):
  match remaining:
    case 0n: Unit{}
    case 1n+ +rest:
      groups_axes_pick(Nat.is_eq(index,skip),Tiled.axes_from(rest,1n+index,skip),index,source,source_within,source_position,target,strides,shape,groups_axis,
        axes_groups(rest,1n+index,skip,source,source_within,source_position,target,strides,shape,groups_axis))

def reads_kept(blocked: List<&2,Tiled.Blocked>,digits: List<&2,Nat>,+reads: U32,+none: NoGroups(blocked)) -> {blocked_reads(blocked,digits,reads) == reads : U32}:
  match blocked digits:
    case Con{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+rest} Con{+digit,+more}:
      Empty.absurd({blocked_reads(Con{Tiled.Blocked{Tiled.Groups{},source,lanes},rest},Con{digit,more},reads) == reads : U32},none)
    case Con{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+rest} Con{+digit,+more}: reads_kept(rest,more,reads,none)
    case Con{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+rest} Con{+digit,+more}: reads_kept(rest,more,reads,none)
    case Nil{} +digits: {==}
    case Con{+head,+rest} Nil{}: {==}

def reads_last(blocked: List<&2,Tiled.Blocked>,digits: List<&2,Nat>,+last: Tiled.Blocked,+digit: Nat,+reads: U32,+none: NoGroups(blocked),
  +same: {List.length(&2,Nat,digits) == List.length(&2,Tiled.Blocked,blocked) : Nat}) ->
  {blocked_reads(List.append(&2,Tiled.Blocked,blocked,[last]),List.append(&2,Nat,digits,[digit]),reads) == blocked_reads([last],[digit],reads) : U32}:
  match blocked digits:
    case Nil{} Nil{}: {==}
    case Con{Tiled.Blocked{Tiled.Groups{},+source,+lanes},+rest} Con{+first,+more}:
      Empty.absurd({blocked_reads(List.append(&2,Tiled.Blocked,Con{Tiled.Blocked{Tiled.Groups{},source,lanes},rest},[last]),List.append(&2,Nat,Con{first,more},[digit]),reads) ==
        blocked_reads([last],[digit],reads) : U32},none)
    case Con{Tiled.Blocked{Tiled.Shared{},+source,+lanes},+rest} Con{+first,+more}:
      reads_last(rest,more,last,digit,reads,none,Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,Nat,more),1n+List.length(&2,Tiled.Blocked,rest),same))
    case Con{Tiled.Blocked{Tiled.Lanes{},+source,+lanes},+rest} Con{+first,+more}:
      reads_last(rest,more,last,digit,reads,none,Equal.cong(Nat,Nat,count => BoxDigits.before(count),1n+List.length(&2,Nat,more),1n+List.length(&2,Tiled.Blocked,rest),same))
    case Nil{} Con{+first,+more}:
      Empty.absurd({blocked_reads(List.append(&2,Tiled.Blocked,Nil{},[last]),List.append(&2,Nat,Con{first,more},[digit]),reads) == blocked_reads([last],[digit],reads) : U32},
        BoxDigits.zero_successor(List.length(&2,Nat,more),Equal.sym(Nat,1n+List.length(&2,Nat,more),0n,same)))
    case Con{+head,+rest} Nil{}:
      Empty.absurd({blocked_reads(List.append(&2,Tiled.Blocked,Con{head,rest},[last]),List.append(&2,Nat,Nil{},[digit]),reads) == blocked_reads([last],[digit],reads) : U32},
        BoxDigits.zero_successor(List.length(&2,Tiled.Blocked,rest),same))

# ---- The cells a block of the target cell axes reads ----

# Digits below extents are as many as the extents.
def below_length(extents: List<&2,U32>,digits: List<&2,Nat>,+below: Coordinates.DigitsBelow(extents,digits)) ->
  {List.length(&2,Nat,digits) == List.length(&2,U32,extents) : Nat}:
  match extents digits:
    case Con{+extent,+rest} Con{+digit,+more}:
      KernelGemm.Both{+here,+later} = below
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,more),List.length(&2,U32,rest),below_length(rest,more,later))
    case Nil{} Nil{}: {==}
    case Nil{} Con{+digit,+more}: Empty.absurd({1n+List.length(&2,Nat,more) == 0n : Nat},below)
    case Con{+extent,+rest} Nil{}: Empty.absurd({0n == 1n+List.length(&2,U32,rest) : Nat},below)

# The cells a block reads with lanes on the target axis: no more than eight
# and the lanes left in the group of the target digits; one otherwise.
def target_read(within: Bool,position: Grid.Position,+digits: List<&2,Nat>,+lanes: U32) -> U32:
  match within:
    case True{}: U32.min(8,(lanes - U32.from_nat(cell_group(True{},position,digits)) * 8 : U32))
    case False{}: 1

def target_reads(within: Bool,position: Grid.Position,+target: Nat,digits: List<&2,Nat>,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +strides: List<&2,U32>,+shape: List<&2,U32>,+groups_axis: U32,+within_value: {Nat.is_lt(target,List.length(&2,U32,shape)) == within : Bool},
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(shape,target),Grid.groups(Grid.lanes_at(shape,target))),digits)) ->
  {blocked_reads(Tiled.blocked_list(Tiled.target_axes(within,position,target,List.length(&2,U32,shape)),source,source_within,source_position,target,strides,shape,groups_axis),
    digits,1) == target_read(within,position,digits,Grid.lanes_at(shape,target)) : U32}:
  match within position digits:
    case False{} +position +digits:
      +rank = List.length(&2,U32,shape)
      reads_kept(Tiled.blocked_list(Tiled.axes_from(rank,0n,rank),source,source_within,source_position,target,strides,shape,groups_axis),digits,1,
        axes_groups(rank,0n,rank,source,source_within,source_position,target,strides,shape,groups_axis))
    case True{} Grid.Innermost{} +digits:
      +rank = List.length(&2,U32,shape)
      +axes = Tiled.axes_from(rank,0n,target)
      +blocks = Tiled.blocked_list(axes,source,source_within,source_position,target,strides,shape,groups_axis)
      +last = Tiled.blocked_axis(Tiled.OfGroups{},source,source_within,source_position,target,strides,shape,groups_axis)
      +others = Grid.removed(shape,target)
      +front = Geometry.digits_front(digits)
      +back = Geometry.digits_back(digits)
      +front_count = below_length(others,front,below_front(others,digits,Grid.groups(Grid.lanes_at(shape,target)),below))
      +removed_count = Public.removed_length(shape,target,within_value)
      +axes_count = {%within_value : {Nat.add(List.length(&2,Tiled.CellAxis,axes),Geometry.skipped_count(_)) == rank : Nat}
        Geometry.axes_count(rank,target,0n) : {Nat.add(List.length(&2,Tiled.CellAxis,axes),1n) == rank : Nat}}
      +same = Order.cancel_sum(1n,List.length(&2,Nat,front),List.length(&2,Tiled.Blocked,blocks),
        Equal.trans(Nat,1n+List.length(&2,Nat,front),rank,1n+List.length(&2,Tiled.Blocked,blocks),
          Equal.trans(Nat,1n+List.length(&2,Nat,front),1n+List.length(&2,U32,others),rank,Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,front),List.length(&2,U32,others),front_count),
            removed_count),
          Equal.sym(Nat,1n+List.length(&2,Tiled.Blocked,blocks),rank,
            Equal.trans(Nat,1n+List.length(&2,Tiled.Blocked,blocks),Nat.add(List.length(&2,Tiled.CellAxis,axes),1n),rank,
              Equal.trans(Nat,1n+List.length(&2,Tiled.Blocked,blocks),1n+List.length(&2,Tiled.CellAxis,axes),Nat.add(List.length(&2,Tiled.CellAxis,axes),1n),
                Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Tiled.Blocked,blocks),List.length(&2,Tiled.CellAxis,axes),
                  Geometry.blocked_count(axes,source,source_within,source_position,target,strides,shape,groups_axis)),
                Equal.sym(Nat,Nat.add(List.length(&2,Tiled.CellAxis,axes),1n),1n+List.length(&2,Tiled.CellAxis,axes),Addition.successor_width(List.length(&2,Tiled.CellAxis,axes)))),
              axes_count))))
      %Equal.sym(List<&2,Tiled.Blocked>,Tiled.blocked_list(List.append(&2,Tiled.CellAxis,axes,[Tiled.OfGroups{}]),source,source_within,source_position,target,strides,shape,groups_axis),
          List.append(&2,Tiled.Blocked,blocks,[last]),Geometry.blocked_append(axes,Tiled.OfGroups{},source,source_within,source_position,target,strides,shape,groups_axis)) :
        {blocked_reads(_,digits,1) == target_read(True{},Grid.Innermost{},digits,Grid.lanes_at(shape,target)) : U32}
      %below_whole(others,digits,Grid.groups(Grid.lanes_at(shape,target)),below) :
        {blocked_reads(List.append(&2,Tiled.Blocked,blocks,[last]),_,1) == target_read(True{},Grid.Innermost{},digits,Grid.lanes_at(shape,target)) : U32}
      reads_last(blocks,front,last,back,1,axes_groups(rank,0n,target,source,source_within,source_position,target,strides,shape,groups_axis),same)
    case True{} Grid.Outermost{} Nil{}:
      Empty.absurd({blocked_reads(Tiled.blocked_list(Tiled.target_axes(True{},Grid.Outermost{},target,List.length(&2,U32,shape)),source,source_within,source_position,target,strides,shape,groups_axis),
        Nil{},1) == target_read(True{},Grid.Outermost{},Nil{},Grid.lanes_at(shape,target)) : U32},below)
    case True{} Grid.Outermost{} Con{+head,+tail}:
      +rank = List.length(&2,U32,shape)
      reads_kept(Tiled.blocked_list(Tiled.axes_from(rank,0n,target),source,source_within,source_position,target,strides,shape,groups_axis),tail,
        U32.min(8,(Grid.lanes_at(shape,target) - U32.from_nat(head) * 8 : U32)),axes_groups(rank,0n,target,source,source_within,source_position,target,strides,shape,groups_axis))

# A lane below eight compares with a word as its number does.
def lane_compare(+limits: Machine.Limits,+lane: Nat,+read: U32,+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(lane),read) == Nat.is_lt(lane,U32.to_nat(read)) : Bool}:
  +capacity = Machine.capacity(limits)
  +room = Order.transitive(lane,7n,U32.to_nat(capacity),{Order.strict_follow(lane,8n,small) : {Nat.is_le(lane,7n) == True{} : Bool}},
    Bounds.as_nat_le(7,capacity,Limits.seven_within_capacity(limits)))
  %Bounds.roundtrip(lane,capacity,Limits.capacity_fits(limits),room) : {U32.is_lt(U32.from_nat(lane),read) == Nat.is_lt(_,U32.to_nat(read)) : Bool}
  Comparison.lt(U32.from_nat(lane),read)

# A block over the target cell axes reads at a lane exactly when the channel
# of that lane in the group of its target digits lies below the lanes of the
# target axis.
def read_lane(+limits: Machine.Limits,within: Bool,position: Grid.Position,+digits: List<&2,Nat>,+shape: List<&2,U32>,+target: Nat,+lane: Nat,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,shape)) == within : Bool},
  +bound: {Nat.is_le(U32.to_nat(Grid.lanes_at(shape,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +group_below: Coordinates.GroupBelow(within,cell_group(within,position,digits),Grid.groups(Grid.lanes_at(shape,target))),
  +small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(lane),target_read(within,position,digits,Grid.lanes_at(shape,target))) ==
    Nat.is_lt(Nat.add(lane,Nat.mul(cell_group(within,position,digits),8n)),U32.to_nat(Grid.lanes_at(shape,target))) : Bool}:
  match within position:
    case False{} position:
      %Equal.sym(Nat,Nat.add(lane,0n),lane,Addition.zero_right(lane)) : {U32.is_lt(U32.from_nat(lane),1) == Nat.is_lt(_,U32.to_nat(Grid.lanes_at(shape,target))) : Bool}
      %Equal.sym(U32,Grid.lanes_at(shape,target),1,Public.lanes_past(shape,target,within_value)) : {U32.is_lt(U32.from_nat(lane),1) == Nat.is_lt(lane,U32.to_nat(_)) : Bool}
      lane_compare(limits,lane,1,small)
    case True{} position:
      +lanes = Grid.lanes_at(shape,target)
      +count = U32.to_nat(lanes)
      +group = cell_group(True{},position,digits)
      +offset = Nat.mul(group,8n)
      +capacity = Machine.capacity(limits)
      +fits = Limits.capacity_fits(limits)
      +inside = {%Encoded.eight_blocks_exact(limits,lanes,bound) : {Nat.is_lt(group,_) == True{} : Bool}
        group_below : {Nat.is_lt(group,Ceiling.ceiling(count,8n)) == True{} : Bool}}
      +past = {%Ceiling.upper_comparison(count,8n,group,{==}) : {_ == True{} : Bool} inside : {Nat.is_lt(offset,count) == True{} : Bool}}
      +covered = Bounds.strict_le(offset,count,past)
      +group_room = Order.transitive(group,count,U32.to_nat(capacity),
        Order.transitive(group,Ceiling.ceiling(count,8n),count,Bounds.strict_le(group,Ceiling.ceiling(count,8n),inside),Ceiling.bounded(count,8n,{==})),bound)
      +product = Arithmetic.product_exact(capacity,fits,U32.from_nat(group),8,group,8n,Bounds.roundtrip(group,capacity,fits,group_room),{==},
        Order.transitive(offset,count,U32.to_nat(capacity),covered,bound))
      +left = (lanes - U32.from_nat(group) * 8 : U32)
      +rest = Nat.sub(count,offset)
      +difference = Arithmetic.difference_exact(capacity,fits,lanes,(U32.from_nat(group) * 8 : U32),count,offset,{==},product,covered,bound)
      +reads = Equal.trans(Nat,U32.to_nat(U32.min(8,left)),Nat.min(8n,U32.to_nat(left)),Nat.min(8n,rest),Extrema.minimum(8,left),
        Equal.cong(Nat,Nat,value => Nat.min(8n,value),U32.to_nat(left),rest,difference))
      %Order.commutative(offset,lane) : {U32.is_lt(U32.from_nat(lane),U32.min(8,left)) == Nat.is_lt(_,count) : Bool}
      %Order.restore_shift(offset,count,covered) : {U32.is_lt(U32.from_nat(lane),U32.min(8,left)) == Nat.is_lt(Nat.add(offset,lane),_) : Bool}
      %Equal.sym(Bool,Nat.is_lt(Nat.add(offset,lane),Nat.add(offset,rest)),Nat.is_lt(lane,rest),Order.lt_offset(offset,lane,rest)) :
        {U32.is_lt(U32.from_nat(lane),U32.min(8,left)) == _ : Bool}
      %small : {U32.is_lt(U32.from_nat(lane),U32.min(8,left)) == _ && Nat.is_lt(lane,rest) : Bool}
      %NatExtrema.below_minimum(lane,8n,rest) : {U32.is_lt(U32.from_nat(lane),U32.min(8,left)) == _ : Bool}
      %reads : {U32.is_lt(U32.from_nat(lane),U32.min(8,left)) == Nat.is_lt(lane,_) : Bool}
      lane_compare(limits,lane,U32.min(8,left),small)

# ---- The source cell a block reads ----

def inserted_length(axis: Nat,digits: List<&2,Nat>,+value: Nat) ->
  {List.length(&2,Nat,Coordinates.inserted_digit(axis,digits,value)) == 1n+List.length(&2,Nat,digits) : Nat}:
  match axis digits:
    case 0n +digits: {==}
    case 1n+ +next Nil{}: {==}
    case 1n+ +next Con{+digit,+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,Coordinates.inserted_digit(next,rest,value)),1n+List.length(&2,Nat,rest),inserted_length(next,rest,value))

# The logical digits of target digits below the target cells are as many as
# the axes of the shape.
def logical_length(within: Bool,position: Grid.Position,+target: Nat,digits: List<&2,Nat>,+read: Nat,+shape: List<&2,U32>,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,shape)) == within : Bool},
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(shape,target),Grid.groups(Grid.lanes_at(shape,target))),digits)) ->
  {List.length(&2,Nat,Geometry.target_logical(within,position,target,digits,read)) == List.length(&2,U32,shape) : Nat}:
  match within position digits:
    case False{} +position +digits:
      Equal.trans(Nat,List.length(&2,Nat,digits),List.length(&2,U32,Grid.removed(shape,target)),List.length(&2,U32,shape),
        below_length(Grid.removed(shape,target),digits,below),
        Equal.cong(List<&2,U32>,Nat,list => List.length(&2,U32,list),Grid.removed(shape,target),shape,Public.removed_past(shape,target,within_value)))
    case True{} Grid.Innermost{} +digits:
      +others = Grid.removed(shape,target)
      +front = Geometry.digits_front(digits)
      +value = Nat.add(read,Nat.mul(Geometry.digits_back(digits),8n))
      Equal.trans(Nat,List.length(&2,Nat,Coordinates.inserted_digit(target,front,value)),1n+List.length(&2,Nat,front),List.length(&2,U32,shape),
        inserted_length(target,front,value),
        Equal.trans(Nat,1n+List.length(&2,Nat,front),1n+List.length(&2,U32,others),List.length(&2,U32,shape),
          Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,front),List.length(&2,U32,others),
            below_length(others,front,below_front(others,digits,Grid.groups(Grid.lanes_at(shape,target)),below))),
          Public.removed_length(shape,target,within_value)))
    case True{} Grid.Outermost{} Nil{}:
      Empty.absurd({List.length(&2,Nat,Geometry.target_logical(True{},Grid.Outermost{},target,Nil{},read)) == List.length(&2,U32,shape) : Nat},below)
    case True{} Grid.Outermost{} Con{+head,+tail}:
      KernelGemm.Both{+here,+later} = below
      +others = Grid.removed(shape,target)
      +value = Nat.add(read,Nat.mul(head,8n))
      Equal.trans(Nat,List.length(&2,Nat,Coordinates.inserted_digit(target,tail,value)),1n+List.length(&2,Nat,tail),List.length(&2,U32,shape),
        inserted_length(target,tail,value),
        Equal.trans(Nat,1n+List.length(&2,Nat,tail),1n+List.length(&2,U32,others),List.length(&2,U32,shape),
          Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Nat,tail),List.length(&2,U32,others),below_length(others,tail,later)),
          Public.removed_length(shape,target,within_value)))

# The logical digits of a block at read j hold, on the source lane axis, lane
# i of the block in its eighth.
def target_lane(within: Bool,position: Grid.Position,digits: List<&2,Nat>,+rank: Nat,+read: Nat,+lane: Nat,+source: Nat,+source_within: Bool,
  +source_position: Grid.Position,+target: Nat,+strides: List<&2,U32>,+shape: List<&2,U32>,+axis: U32,
  +source_inside: {source_within == True{} : Bool},+source_below: {Nat.is_lt(source,rank) == True{} : Bool},+apart: {Nat.is_eq(source,target) == False{} : Bool},
  +lanes: Geometry.LaneDigits(Tiled.blocked_list(Tiled.target_axes(within,position,target,rank),source,source_within,source_position,target,strides,shape,axis),digits,lane)) ->
  {Nat.mod(BoxDigits.digit_at(Geometry.target_logical(within,position,target,digits,read),source),8n) == lane : Nat}:
  match within position digits:
    case False{} +position +digits:
      Geometry.every_lane(rank,source,digits,0n,source,source_within,source_position,target,strides,shape,axis,lane,source_inside,{==},source_below,
        {%Geometry.never_reach(rank,0n,rank,Order.reflexive(rank)) :
            Geometry.LaneDigits(Tiled.blocked_list(_,source,source_within,source_position,target,strides,shape,axis),digits,lane)
          lanes : Geometry.LaneDigits(Tiled.blocked_list(Geometry.every_axis(rank,0n),source,source_within,source_position,target,strides,shape,axis),digits,lane)})
    case True{} Grid.Innermost{} +digits:
      +front = Tiled.blocked_list(Tiled.axes_from(rank,0n,target),source,source_within,source_position,target,strides,shape,axis)
      +last = Tiled.blocked_axis(Tiled.OfGroups{},source,source_within,source_position,target,strides,shape,axis)
      Geometry.logical_lane(rank,target,source,Geometry.digits_front(digits),0n,source,source_within,source_position,target,strides,shape,axis,
        Nat.add(read,Nat.mul(Geometry.digits_back(digits),8n)),lane,source_inside,{==},apart,source_below,
        Geometry.lanes_front(digits,front,last,lane,
          {%Geometry.blocked_append(Tiled.axes_from(rank,0n,target),Tiled.OfGroups{},source,source_within,source_position,target,strides,shape,axis) :
              Geometry.LaneDigits(_,digits,lane)
            lanes : Geometry.LaneDigits(List.append(&2,Tiled.Blocked,front,[last]),digits,lane)}))
    case True{} Grid.Outermost{} Nil{}:
      Empty.absurd({Nat.mod(BoxDigits.digit_at(Geometry.target_logical(True{},Grid.Outermost{},target,Nil{},read),source),8n) == lane : Nat},lanes)
    case True{} Grid.Outermost{} Con{+head,+tail}:
      KernelGemm.Both{+here,+later} = lanes
      Geometry.logical_lane(rank,target,source,tail,0n,source,source_within,source_position,target,strides,shape,axis,
        Nat.add(read,Nat.mul(head,8n)),lane,source_inside,{==},apart,source_below,later)

# The groups axis of source cells and what every target cell axis of a relane
# is to its blocks.
def groups_axis(+position: Grid.Position,+cells: List<&2,U32>) -> U32:
  Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,cells)) - 1 : U32))

def relane_blocked(within: Bool,position: Grid.Position,+target: Nat,+rank: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +source_cells: List<&2,U32>,+shape: List<&2,U32>) -> List<&2,Tiled.Blocked>:
  Tiled.blocked_list(Tiled.target_axes(within,position,target,rank),source,source_within,source_position,target,Tasks.strides_of(source_cells),shape,
    groups_axis(source_position,source_cells))

# The source cell a block of a piece reads at j is the cell of the logical
# digits of its target digits at j.
def source_word(+within: Bool,+position: Grid.Position,+target: Nat,+rank: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,
  +source_cells: List<&2,U32>,+shape: List<&2,U32>,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+lane: Nat,+read: Nat,
  +cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},
  +same_blocked: {List.length(&2,Tiled.Blocked,relane_blocked(within,position,target,rank,source,source_within,source_position,source_cells,shape)) ==
    List.length(&2,U32,boxes) : Nat},
  +same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,boxes) : Nat},
  +small: {Nat.is_lt(lane,8n) == True{} : Bool},+within_value: {Nat.is_lt(target,rank) == within : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +length: {List.length(&2,Nat,Geometry.block_digits(Writes.piece_axes(relane_blocked(within,position,target,rank,source,source_within,source_position,source_cells,shape),
    lows,boxes),index,lane)) == rank : Nat},
  +logical_length: {List.length(&2,Nat,Geometry.target_logical(within,position,target,Geometry.block_digits(Writes.piece_axes(relane_blocked(within,position,target,rank,
    source,source_within,source_position,source_cells,shape),lows,boxes),index,lane),read)) == rank : Nat},
  +source_value: {Nat.is_lt(source,rank) == source_within : Bool},+cells_length: {List.length(&2,U32,source_cells) == rank : Nat},
  +room: {Nat.is_le(rank,U32.to_nat(cap)) == True{} : Bool}) ->
  {(Geometry.source_of(Tiled.block_at(Writes.piece_axes(relane_blocked(within,position,target,rank,source,source_within,source_position,source_cells,shape),lows,boxes),
      index,Writes.first_block())) +
    U32.from_nat(read) * Geometry.step_of(Tiled.block_at(Writes.piece_axes(relane_blocked(within,position,target,rank,source,source_within,source_position,source_cells,shape),
      lows,boxes),index,Writes.first_block())) : U32) ==
    U32.from_nat(ShapeLaws.digits_value(source_cells,Coordinates.placed_digits(source_within,source_position,
      Geometry.removed_at(Geometry.target_logical(within,position,target,Geometry.block_digits(Writes.piece_axes(relane_blocked(within,position,target,rank,source,source_within,
        source_position,source_cells,shape),lows,boxes),index,lane),read),source),
      Nat.div(BoxDigits.digit_at(Geometry.target_logical(within,position,target,Geometry.block_digits(Writes.piece_axes(relane_blocked(within,position,target,rank,source,
        source_within,source_position,source_cells,shape),lows,boxes),index,lane),read),source),8n)))) : U32}:
  +blocked = relane_blocked(within,position,target,rank,source,source_within,source_position,source_cells,shape)
  +axes = Writes.piece_axes(blocked,lows,boxes)
  +last = Tiled.block_at(axes,index,Writes.first_block())
  +digits = Geometry.block_digits(axes,index,lane)
  +logical = Geometry.target_logical(within,position,target,digits,read)
  +strides = Tasks.strides_of(source_cells)
  +axis = groups_axis(source_position,source_cells)
  +word = U32.from_nat(ShapeLaws.digits_value(source_cells,Coordinates.placed_digits(source_within,source_position,Geometry.removed_at(logical,source),
    Nat.div(BoxDigits.digit_at(logical,source),8n))))
  +own = Geometry.blocked_of(axes)
  +sum = Geometry.blocked_sum(digits,own)
  +every = Geometry.blocked_sum(logical,Tiled.blocked_list(Geometry.every_axis(rank,0n),source,source_within,source_position,target,strides,shape,axis))
  +count = List.length(&2,Nat,logical)
  +placed = {%logical_length : {word == Geometry.blocked_sum(logical,Tiled.blocked_list(Geometry.every_axis(_,0n),source,source_within,source_position,target,strides,shape,axis)) : U32}
    Geometry.source_placed(source_within,source_position,logical,source,target,source_cells,shape,cap,cap_ok,
      {%Equal.sym(Nat,count,rank,logical_length) : {Nat.is_lt(source,_) == source_within : Bool} source_value : {Nat.is_lt(source,count) == source_within : Bool}},
      Equal.trans(Nat,List.length(&2,U32,source_cells),rank,count,cells_length,Equal.sym(Nat,count,rank,logical_length)),
      {%Equal.sym(Nat,count,rank,logical_length) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} room : {Nat.is_le(count,U32.to_nat(cap)) == True{} : Bool}}) :
    {word == every : U32}}
  %Equal.sym(U32,Geometry.source_of(last),(0 + sum : U32),Geometry.block_source(axes,Writes.first_block(),index,lane,small)) :
    {(_ + U32.from_nat(read) * Geometry.step_of(last) : U32) == word : U32}
  %Equal.sym(U32,Geometry.step_of(last),Geometry.blocked_step(own,0),Geometry.block_step(axes,Writes.first_block(),index)) :
    {((0 + sum : U32) + U32.from_nat(read) * _ : U32) == word : U32}
  %Equal.sym(U32,(0 + sum : U32),sum,Arithmetic.add_zero_left(sum)) : {(_ + U32.from_nat(read) * Geometry.blocked_step(own,0) : U32) == word : U32}
  Equal.trans(U32,(sum + U32.from_nat(read) * Geometry.blocked_step(own,0) : U32),
    (Geometry.blocked_sum(digits,blocked) + U32.from_nat(read) * Geometry.blocked_step(blocked,0) : U32),word,
    Equal.cong(List<&2,Tiled.Blocked>,U32,list => (Geometry.blocked_sum(digits,list) + U32.from_nat(read) * Geometry.blocked_step(list,0) : U32),own,blocked,
      Writes.ranged_blocked(blocked,lows,boxes,same_blocked,same_lows)),
    Equal.trans(U32,(Geometry.blocked_sum(digits,blocked) + U32.from_nat(read) * Geometry.blocked_step(blocked,0) : U32),every,word,
      Geometry.logical_sum(within,position,digits,rank,read,source,source_within,source_position,target,strides,shape,axis,within_value,apart,length),
      Equal.sym(U32,word,every,placed)))

# ---- The lanes a block moves ----

# A digit past the digits is zero.
def digit_beyond(digits: List<&2,Nat>,axis: Nat,+past: {Nat.is_le(List.length(&2,Nat,digits),axis) == True{} : Bool}) -> {BoxDigits.digit_at(digits,axis) == 0n : Nat}:
  match digits axis:
    case Nil{} +axis: {==}
    case Con{+digit,+rest} 0n: Empty.absurd({digit == 0n : Nat},WordEncoding.false_true(past))
    case Con{+digit,+rest} 1n+ +next: digit_beyond(rest,next,past)

# The lane of the source cell that holds the digit of the source lane axis:
# the digit in its eighth, or the only lane on the unit axis.
def SourceLane(within: Bool,+digit: Nat,+lane: Nat) -> Data:
  match within:
    case True{}: {Nat.mod(digit,8n) == lane : Nat}
    case False{}: {lane == 0n : Nat}

# The digit in its eighth and the eighths before it make the digit.
def eighths_split(+digit: Nat,+lane: Nat,+rest: {Nat.mod(digit,8n) == lane : Nat}) -> {Nat.add(lane,Nat.mul(Nat.div(digit,8n),8n)) == digit : Nat}:
  +eighths = Nat.mul(Nat.div(digit,8n),8n)
  Equal.trans(Nat,Nat.add(lane,eighths),Nat.add(Nat.mod(digit,8n),eighths),digit,
    Equal.cong(Nat,Nat,value => Nat.add(value,eighths),lane,Nat.mod(digit,8n),Equal.sym(Nat,Nat.mod(digit,8n),lane,rest)),
    Equal.trans(Nat,Nat.add(Nat.mod(digit,8n),eighths),Nat.add(eighths,Nat.mod(digit,8n)),digit,Order.commutative(Nat.mod(digit,8n),eighths),
      Equal.sym(Nat,digit,Nat.add(eighths,Nat.mod(digit,8n)),DivisionOrder.quotient_exact(digit,8n,{==}))))

# The value at logical digits held by the lane of a source cell.
def source_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+values: List<&2,U32>,+source: Nat,+digits: List<&2,Nat>,+channel: Nat) -> F32:
  Bool.pick(F32,Nat.is_lt(channel,U32.to_nat(Grid.lanes_at(values,source))),
    original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(source,Geometry.removed_at(digits,source),channel)))),0.0)

# The group of the source lane digit of digits below the values lies below
# the source groups.
def source_group(+limits: Machine.Limits,within: Bool,+values: List<&2,U32>,+source: Nat,+digits: List<&2,Nat>,
  +within_value: {Nat.is_lt(source,List.length(&2,U32,values)) == within : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: Coordinates.DigitsBelow(values,digits)) -> Coordinates.GroupBelow(within,Nat.div(BoxDigits.digit_at(digits,source),8n),Grid.groups(Grid.lanes_at(values,source))):
  match within:
    case True{}:
      +digit = BoxDigits.digit_at(digits,source)
      +group = Nat.div(digit,8n)
      +lanes = Grid.lanes_at(values,source)
      %Equal.sym(Nat,U32.to_nat(Grid.groups(lanes)),Ceiling.ceiling(U32.to_nat(lanes),8n),Encoded.eight_blocks_exact(limits,lanes,lanes_room)) :
        {Nat.is_lt(group,_) == True{} : Bool}
      %Equal.sym(Bool,Nat.is_lt(group,Ceiling.ceiling(U32.to_nat(lanes),8n)),Nat.is_lt(Nat.mul(group,8n),U32.to_nat(lanes)),Ceiling.upper_comparison(U32.to_nat(lanes),8n,group,{==})) :
        {_ == True{} : Bool}
      Order.le_lt_trans(Nat.mul(group,8n),digit,U32.to_nat(lanes),DivisionOrder.product_below(digit,8n,{==}),digit_below(values,digits,source,below,within_value))
    case False{}: Unit{}

# Lane i of the source cell of logical digits below the values whose source
# lane digit lies at lane i holds the value at the digits.
def source_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+position: Grid.Position,
  within: Bool,+digits: List<&2,Nat>,+word: U32,+lane: Nat,+within_value: {Nat.is_lt(source,List.length(&2,U32,values)) == within : Bool},
  +inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: Coordinates.DigitsBelow(values,digits),
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,source,position),Coordinates.placed_digits(within,position,Geometry.removed_at(digits,source),
    Nat.div(BoxDigits.digit_at(digits,source),8n))) : Nat},
  +small: {Nat.is_lt(lane,8n) == True{} : Bool},+lane_digit: SourceLane(within,BoxDigits.digit_at(digits,source),lane)) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,source,position)},word)) ==
    original(logical,U32.from_nat(ShapeLaws.digits_value(values,digits))) : F32}:
  match within:
    case True{}:
      +digit = BoxDigits.digit_at(digits,source)
      +group = Nat.div(digit,8n)
      +lanes = Grid.lanes_at(values,source)
      +others = Geometry.removed_at(digits,source)
      +channel = Nat.add(lane,Nat.mul(group,8n))
      +digit_inside = digit_below(values,digits,source,below,within_value)
      +group_inside = source_group(limits,True{},values,source,digits,within_value,lanes_room,below)
      +count = below_length(values,digits,below)
      +digits_inside = {%Equal.sym(Nat,List.length(&2,Nat,digits),List.length(&2,U32,values),count) : {Nat.is_lt(source,_) == True{} : Bool}
        within_value : {Nat.is_lt(source,List.length(&2,Nat,digits)) == True{} : Bool}}
      Equal.trans(F32,TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,source,position)},word)),
        source_value(~Logical,~original,logical,values,source,digits,channel),original(logical,U32.from_nat(ShapeLaws.digits_value(values,digits))),
        Coordinates.lane_values(~Logical,~original,logical,limits,values,source,position,True{},others,group,word,within_value,inside,room,
          removed_below(values,digits,source,below),group_inside,exact,lane,{Order.strict_follow(lane,8n,small) : {Nat.is_le(lane,7n) == True{} : Bool}}),
        Equal.trans(F32,source_value(~Logical,~original,logical,values,source,digits,channel),source_value(~Logical,~original,logical,values,source,digits,digit),
          original(logical,U32.from_nat(ShapeLaws.digits_value(values,digits))),
          Equal.cong(Nat,F32,value => source_value(~Logical,~original,logical,values,source,digits,value),channel,digit,eighths_split(digit,lane,lane_digit)),
          %Equal.sym(Bool,Nat.is_lt(digit,U32.to_nat(lanes)),True{},digit_inside) :
            {Bool.pick(F32,_,original(logical,U32.from_nat(ShapeLaws.digits_value(values,Coordinates.inserted_digit(source,others,digit)))),0.0) ==
              original(logical,U32.from_nat(ShapeLaws.digits_value(values,digits))) : F32}
          Equal.cong(List<&2,Nat>,F32,list => original(logical,U32.from_nat(ShapeLaws.digits_value(values,list))),Coordinates.inserted_digit(source,others,digit),digits,
            reinserted(digits,source,digits_inside))))
    case False{}:
      +count = below_length(values,digits,below)
      +past = {%Equal.sym(Nat,List.length(&2,Nat,digits),List.length(&2,U32,values),count) : {Nat.is_le(_,source) == True{} : Bool}
        Order.not_less(source,List.length(&2,U32,values),within_value) : {Nat.is_le(List.length(&2,Nat,digits),source) == True{} : Bool}}
      +zero = digit_beyond(digits,source,past)
      +unit = Public.lanes_past(values,source,within_value)
      +group = Nat.div(BoxDigits.digit_at(digits,source),8n)
      +group_inside = {%Equal.sym(Nat,BoxDigits.digit_at(digits,source),0n,zero) : {Nat.is_lt(Nat.div(_,8n),U32.to_nat(Grid.groups(Grid.lanes_at(values,source)))) == True{} : Bool}
        %Equal.sym(U32,Grid.lanes_at(values,source),1,unit) : {Nat.is_lt(0n,U32.to_nat(Grid.groups(_))) == True{} : Bool}
        {==} : {Nat.is_lt(group,U32.to_nat(Grid.groups(Grid.lanes_at(values,source)))) == True{} : Bool}}
      Equal.trans(F32,TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,source,position)},word)),
        Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,source))),
          original(logical,U32.from_nat(ShapeLaws.digits_value(values,Geometry.removed_at(digits,source)))),0.0),
        original(logical,U32.from_nat(ShapeLaws.digits_value(values,digits))),
        Coordinates.lane_values(~Logical,~original,logical,limits,values,source,position,False{},Geometry.removed_at(digits,source),group,word,within_value,inside,room,
          removed_below(values,digits,source,below),group_inside,exact,lane,{Order.strict_follow(lane,8n,small) : {Nat.is_le(lane,7n) == True{} : Bool}}),
        %Equal.sym(Nat,lane,0n,lane_digit) :
          {Bool.pick(F32,Nat.is_lt(Nat.add(_,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,source))),
            original(logical,U32.from_nat(ShapeLaws.digits_value(values,Geometry.removed_at(digits,source)))),0.0) ==
            original(logical,U32.from_nat(ShapeLaws.digits_value(values,digits))) : F32}
        %Equal.sym(Nat,BoxDigits.digit_at(digits,source),0n,zero) :
          {Bool.pick(F32,Nat.is_lt(Nat.add(0n,Nat.mul(Nat.div(_,8n),8n)),U32.to_nat(Grid.lanes_at(values,source))),
            original(logical,U32.from_nat(ShapeLaws.digits_value(values,Geometry.removed_at(digits,source)))),0.0) ==
            original(logical,U32.from_nat(ShapeLaws.digits_value(values,digits))) : F32}
        %Equal.sym(U32,Grid.lanes_at(values,source),1,unit) :
          {Bool.pick(F32,Nat.is_lt(0n,U32.to_nat(_)),original(logical,U32.from_nat(ShapeLaws.digits_value(values,Geometry.removed_at(digits,source)))),0.0) ==
            original(logical,U32.from_nat(ShapeLaws.digits_value(values,digits))) : F32}
        %Equal.sym(List<&2,Nat>,Geometry.removed_at(digits,source),digits,Geometry.removed_beyond(digits,source,past)) :
          {Bool.pick(F32,True{},original(logical,U32.from_nat(ShapeLaws.digits_value(values,_))),0.0) == original(logical,U32.from_nat(ShapeLaws.digits_value(values,digits))) : F32}
        {==})

# The group of target digits below the target cells lies below the groups.
def target_group(within: Bool,position: Grid.Position,+values: List<&2,U32>,+target: Nat,digits: List<&2,Nat>,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),digits)) ->
  {Nat.is_lt(cell_group(within,position,digits),U32.to_nat(Grid.groups(Grid.lanes_at(values,target)))) == True{} : Bool}:
  match within position digits:
    case False{} +position +digits:
      %Equal.sym(U32,Grid.lanes_at(values,target),1,Public.lanes_past(values,target,within_value)) : {Nat.is_lt(0n,U32.to_nat(Grid.groups(_))) == True{} : Bool}
      {==}
    case True{} +position +digits: group_below(True{},position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target)),digits,below)

# Lane j of the target cell of digits below the target cells holds the value
# at the logical digits of the digits at j, or +0 past the target lanes.
def target_value(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+target: Nat,+position: Grid.Position,
  +within: Bool,+digits: List<&2,Nat>,+word: U32,+lane: Nat,+within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},
  +inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),digits),
  +exact: {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,target,position),digits) : Nat},+small: {Nat.is_lt(lane,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},word)) ==
    Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(cell_group(within,position,digits),8n)),U32.to_nat(Grid.lanes_at(values,target))),
      original(logical,U32.from_nat(ShapeLaws.digits_value(values,Geometry.target_logical(within,position,target,digits,lane)))),0.0) : F32}:
  +others = cell_others(within,position,digits)
  +group = cell_group(within,position,digits)
  +groups = Grid.groups(Grid.lanes_at(values,target))
  +removed = Grid.removed(values,target)
  +placed = {%Equal.sym(List<&2,Nat>,Coordinates.placed_digits(within,position,others,group),digits,cell_whole(within,position,removed,groups,digits,below)) :
      {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,target,position),_) : Nat}
    exact : {U32.to_nat(word) == ShapeLaws.digits_value(Tiled.cell_shape(values,target,position),Coordinates.placed_digits(within,position,others,group)) : Nat}}
  %Equal.sym(List<&2,Nat>,Geometry.target_logical(within,position,target,digits,lane),Coordinates.logical_digits(within,target,others,Nat.add(lane,Nat.mul(group,8n))),
      target_cells(within,position,target,digits,lane)) :
    {TileLaws.select_lane(lane,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},word)) ==
      Bool.pick(F32,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),U32.to_nat(Grid.lanes_at(values,target))),original(logical,U32.from_nat(ShapeLaws.digits_value(values,_))),0.0) : F32}
  Coordinates.lane_values(~Logical,~original,logical,limits,values,target,position,within,others,group,word,within_value,inside,room,
    cell_below(within,position,removed,groups,digits,below),target_group(within,position,values,target,digits,within_value,below),placed,lane,
    {Order.strict_follow(lane,8n,small) : {Nat.is_le(lane,7n) == True{} : Bool}})

# Lane i of the cell a block reads at j is lane j of the target cell it
# writes at i: both hold the value at the logical digits of the target digits
# at j, or +0 past the target lanes.
def lane_moved(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,
  +source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+digits: List<&2,Nat>,
  +lane: Nat,+read: Nat,+reads: U32,+source_word: U32,+target_word: U32,valid: Bool,
  +source_value_within: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},
  +source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_exact: Logic.When(valid,{U32.to_nat(source_word) == ShapeLaws.digits_value(Tiled.cell_shape(values,source,source_position),
    Coordinates.placed_digits(source_within,source_position,Geometry.removed_at(Geometry.target_logical(within,position,target,digits,read),source),
      Nat.div(BoxDigits.digit_at(Geometry.target_logical(within,position,target,digits,read),source),8n))) : Nat}),
  +lane_small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +lane_digit: SourceLane(source_within,BoxDigits.digit_at(Geometry.target_logical(within,position,target,digits,read),source),lane),
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),digits),
  +target_exact: {U32.to_nat(target_word) == ShapeLaws.digits_value(Tiled.cell_shape(values,target,position),digits) : Nat},
  +read_small: {Nat.is_lt(read,8n) == True{} : Bool},
  +reads_value: {U32.is_lt(U32.from_nat(read),reads) ==
    Nat.is_lt(Nat.add(read,Nat.mul(cell_group(within,position,digits),8n)),U32.to_nat(Grid.lanes_at(values,target))) : Bool},
  +validity: {Nat.is_lt(Nat.add(read,Nat.mul(cell_group(within,position,digits),8n)),U32.to_nat(Grid.lanes_at(values,target))) == valid : Bool}) ->
  {TileLaws.select_lane(lane,Bool.pick(Tile.Tile8,U32.is_lt(U32.from_nat(read),reads),
      Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},source_word),Tiled.zero_tile())) ==
    TileLaws.select_lane(read,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},target_word)) : F32}:
  match valid:
    case True{}:
      +channel = Nat.add(read,Nat.mul(cell_group(within,position,digits),8n))
      +lanes = U32.to_nat(Grid.lanes_at(values,target))
      +logical_at = Geometry.target_logical(within,position,target,digits,read)
      +others = cell_others(within,position,digits)
      +held = original(logical,U32.from_nat(ShapeLaws.digits_value(values,logical_at)))
      +source_tile = Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},source_word)
      +target_lane = TileLaws.select_lane(read,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},target_word))
      +logical_below_values = {%Equal.sym(List<&2,Nat>,logical_at,Coordinates.logical_digits(within,target,others,channel),target_cells(within,position,target,digits,read)) :
          Coordinates.DigitsBelow(values,_)
        logical_below(within,values,target,others,channel,within_value,
          cell_below(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target)),digits,below),validity) : Coordinates.DigitsBelow(values,logical_at)}
      +target_held = {%validity : {target_lane == Bool.pick(F32,_,held,0.0) : F32}
        target_value(~Logical,~original,logical,limits,values,target,position,within,digits,target_word,read,within_value,inside,room,below,target_exact,read_small) :
        {target_lane == held : F32}}
      %Equal.sym(Bool,U32.is_lt(U32.from_nat(read),reads),Nat.is_lt(channel,lanes),reads_value) :
        {TileLaws.select_lane(lane,Bool.pick(Tile.Tile8,_,source_tile,Tiled.zero_tile())) == target_lane : F32}
      %Equal.sym(Bool,Nat.is_lt(channel,lanes),True{},validity) : {TileLaws.select_lane(lane,Bool.pick(Tile.Tile8,_,source_tile,Tiled.zero_tile())) == target_lane : F32}
      Equal.trans(F32,TileLaws.select_lane(lane,source_tile),held,target_lane,
        source_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,logical_at,source_word,lane,source_value_within,source_inside,
          source_room,source_lanes_room,logical_below_values,source_exact,lane_small,lane_digit),
        Equal.sym(F32,target_lane,held,target_held))
    case False{}:
      +channel = Nat.add(read,Nat.mul(cell_group(within,position,digits),8n))
      +lanes = U32.to_nat(Grid.lanes_at(values,target))
      +logical_at = Geometry.target_logical(within,position,target,digits,read)
      +held = original(logical,U32.from_nat(ShapeLaws.digits_value(values,logical_at)))
      +source_tile = Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},source_word)
      +target_lane = TileLaws.select_lane(read,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},target_word))
      +target_zero = {%validity : {target_lane == Bool.pick(F32,_,held,0.0) : F32}
        target_value(~Logical,~original,logical,limits,values,target,position,within,digits,target_word,read,within_value,inside,room,below,target_exact,read_small) :
        {target_lane == 0.0 : F32}}
      %Equal.sym(Bool,U32.is_lt(U32.from_nat(read),reads),Nat.is_lt(channel,lanes),reads_value) :
        {TileLaws.select_lane(lane,Bool.pick(Tile.Tile8,_,source_tile,Tiled.zero_tile())) == target_lane : F32}
      %Equal.sym(Bool,Nat.is_lt(channel,lanes),False{},validity) : {TileLaws.select_lane(lane,Bool.pick(Tile.Tile8,_,source_tile,Tiled.zero_tile())) == target_lane : F32}
      Equal.trans(F32,TileLaws.select_lane(lane,Tiled.zero_tile()),0.0,target_lane,TileLaws.filled_lane(lane,0.0),Equal.sym(F32,target_lane,0.0,target_zero))

# ---- The tile a write moves ----

def appended_length(extents: List<&2,U32>,+last: U32) -> {List.length(&2,U32,List.append(&2,U32,extents,[last])) == 1n+List.length(&2,U32,extents) : Nat}:
  match extents:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,U32,List.append(&2,U32,rest,[last])),1n+List.length(&2,U32,rest),appended_length(rest,last))

# The cells of values on an axis have as many axes as the values.
def placed_length(within: Bool,position: Grid.Position,+values: List<&2,U32>,+axis: Nat,+within_value: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool}) ->
  {List.length(&2,U32,Grid.placed(within,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))) == List.length(&2,U32,values) : Nat}:
  match within position:
    case False{} +position:
      Equal.cong(List<&2,U32>,Nat,list => List.length(&2,U32,list),Grid.removed(values,axis),values,Public.removed_past(values,axis,within_value))
    case True{} Grid.Innermost{}:
      Equal.trans(Nat,List.length(&2,U32,List.append(&2,U32,Grid.removed(values,axis),[Grid.groups(Grid.lanes_at(values,axis))])),
        1n+List.length(&2,U32,Grid.removed(values,axis)),List.length(&2,U32,values),
        appended_length(Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis))),Public.removed_length(values,axis,within_value))
    case True{} Grid.Outermost{}: Public.removed_length(values,axis,within_value)

def cell_length(+values: List<&2,U32>,+axis: Nat,+position: Grid.Position,+within: Bool,+within_value: {Nat.is_lt(axis,List.length(&2,U32,values)) == within : Bool}) ->
  {List.length(&2,U32,Tiled.cell_shape(values,axis,position)) == List.length(&2,U32,values) : Nat}:
  %Equal.sym(Bool,Nat.is_lt(axis,List.length(&2,U32,values)),within,within_value) :
    {List.length(&2,U32,Grid.placed(_,position,Grid.removed(values,axis),Grid.groups(Grid.lanes_at(values,axis)))) == List.length(&2,U32,values) : Nat}
  placed_length(within,position,values,axis,within_value)

# The source cell of logical digits below the values lies below the source
# cells.
def source_cell_below(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+position: Grid.Position,+within: Bool,+digits: List<&2,Nat>,
  +within_value: {Nat.is_lt(source,List.length(&2,U32,values)) == within : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: Coordinates.DigitsBelow(values,digits)) ->
  Coordinates.DigitsBelow(Tiled.cell_shape(values,source,position),
    Coordinates.placed_digits(within,position,Geometry.removed_at(digits,source),Nat.div(BoxDigits.digit_at(digits,source),8n))):
  +placed = Coordinates.placed_digits(within,position,Geometry.removed_at(digits,source),Nat.div(BoxDigits.digit_at(digits,source),8n))
  %Equal.sym(Bool,Nat.is_lt(source,List.length(&2,U32,values)),within,within_value) :
    Coordinates.DigitsBelow(Grid.placed(_,position,Grid.removed(values,source),Grid.groups(Grid.lanes_at(values,source))),placed)
  Coordinates.placed_below(within,position,Grid.removed(values,source),Grid.groups(Grid.lanes_at(values,source)),Geometry.removed_at(digits,source),
    Nat.div(BoxDigits.digit_at(digits,source),8n),removed_below(values,digits,source,below),source_group(limits,within,values,source,digits,within_value,lanes_room,below))

# The blocks of a relane of values, the block of a piece written at an index
# and its target digits at a lane.
def values_blocked(within: Bool,position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>) ->
  List<&2,Tiled.Blocked>:
  relane_blocked(within,position,target,List.length(&2,U32,values),source,source_within,source_position,Tiled.cell_shape(values,source,source_position),values)

def write_block(within: Bool,position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>,
  +lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32) -> Tiled.Block:
  Tiled.block_at(Writes.piece_axes(values_blocked(within,position,target,source,source_within,source_position,values),lows,boxes),index,Writes.first_block())

def write_digits(within: Bool,position: Grid.Position,+target: Nat,+source: Nat,+source_within: Bool,+source_position: Grid.Position,+values: List<&2,U32>,
  +lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+lane: Nat) -> List<&2,Nat>:
  Geometry.block_digits(Writes.piece_axes(values_blocked(within,position,target,source,source_within,source_position,values),lows,boxes),index,lane)

# A block of a piece reads at j exactly when the channel of j in the group of
# its target digits lies below the target lanes.
def read_count(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,+position: Grid.Position,
  +within: Bool,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+lane: Nat,+read: Nat,
  +same_blocked: {List.length(&2,Tiled.Blocked,values_blocked(within,position,target,source,source_within,source_position,values)) == List.length(&2,U32,boxes) : Nat},
  +same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,boxes) : Nat},+within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),
    write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)),
  +read_small: {Nat.is_lt(read,8n) == True{} : Bool}) ->
  {U32.is_lt(U32.from_nat(read),reads_of(write_block(within,position,target,source,source_within,source_position,values,lows,boxes,index))) ==
    Nat.is_lt(Nat.add(read,Nat.mul(cell_group(within,position,write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)),8n)),
      U32.to_nat(Grid.lanes_at(values,target))) : Bool}:
  +blocked = values_blocked(within,position,target,source,source_within,source_position,values)
  +axes = Writes.piece_axes(blocked,lows,boxes)
  +digits = write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)
  +cells = Tiled.cell_shape(values,source,source_position)
  +channel_in = Nat.is_lt(Nat.add(read,Nat.mul(cell_group(within,position,digits),8n)),U32.to_nat(Grid.lanes_at(values,target)))
  %Equal.sym(U32,reads_of(Tiled.block_at(axes,index,Writes.first_block())),blocked_reads(Geometry.blocked_of(axes),digits,1),
      block_reads(axes,Writes.first_block(),index,lane)) :
    {U32.is_lt(U32.from_nat(read),_) == channel_in : Bool}
  %Equal.sym(List<&2,Tiled.Blocked>,Geometry.blocked_of(axes),blocked,Writes.ranged_blocked(blocked,lows,boxes,same_blocked,same_lows)) :
    {U32.is_lt(U32.from_nat(read),blocked_reads(_,digits,1)) == channel_in : Bool}
  %Equal.sym(U32,blocked_reads(blocked,digits,1),target_read(within,position,digits,Grid.lanes_at(values,target)),
      target_reads(within,position,target,digits,source,source_within,source_position,Tasks.strides_of(cells),values,groups_axis(source_position,cells),within_value,below)) :
    {U32.is_lt(U32.from_nat(read),_) == channel_in : Bool}
  read_lane(limits,within,position,digits,values,target,read,within_value,lanes_room,
    group_below(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target)),digits,below),read_small)

# The source cell a block of a piece reads at a valid lane lies inside the
# source cells.
def read_cell_below(+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,
  +position: Grid.Position,+within: Bool,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+lane: Nat,+read: Nat,
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),
    write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)),
  +validity: {Nat.is_lt(Nat.add(read,Nat.mul(cell_group(within,position,write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)),
    8n)),U32.to_nat(Grid.lanes_at(values,target))) == True{} : Bool}) ->
  {Nat.is_lt(ShapeLaws.digits_value(Tiled.cell_shape(values,source,source_position),Coordinates.placed_digits(source_within,source_position,
      Geometry.removed_at(Geometry.target_logical(within,position,target,write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane),
        read),source),
      Nat.div(BoxDigits.digit_at(Geometry.target_logical(within,position,target,write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,
        index,lane),read),source),8n))),ShapeLaws.elements(Tiled.cell_shape(values,source,source_position))) == True{} : Bool}:
  +cells = Tiled.cell_shape(values,source,source_position)
  +digits = write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)
  +logical_at = Geometry.target_logical(within,position,target,digits,read)
  +channel = Nat.add(read,Nat.mul(cell_group(within,position,digits),8n))
  +others = cell_others(within,position,digits)
  +placed = Coordinates.placed_digits(source_within,source_position,Geometry.removed_at(logical_at,source),Nat.div(BoxDigits.digit_at(logical_at,source),8n))
  +logical_below_values = {%Equal.sym(List<&2,Nat>,logical_at,Coordinates.logical_digits(within,target,others,channel),target_cells(within,position,target,digits,read)) :
      Coordinates.DigitsBelow(values,_)
    logical_below(within,values,target,others,channel,within_value,
      cell_below(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target)),digits,below),validity) : Coordinates.DigitsBelow(values,logical_at)}
  Coordinates.digits_below(cells,placed,source_cell_below(limits,values,source,source_position,source_within,logical_at,source_value,source_lanes_room,logical_below_values))

# When a block of a piece reads at j, the number of the cell it reads is the
# number of the source cell of the logical digits of its target digits at j.
def read_source(valid: Bool,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,+source_within: Bool,+target: Nat,
  +position: Grid.Position,+within: Bool,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+lane: Nat,+read: Nat,
  +same_blocked: {List.length(&2,Tiled.Blocked,values_blocked(within,position,target,source,source_within,source_position,values)) == List.length(&2,U32,boxes) : Nat},
  +same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,boxes) : Nat},+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),
    write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)),
  +validity: {Nat.is_lt(Nat.add(read,Nat.mul(cell_group(within,position,write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)),
    8n)),U32.to_nat(Grid.lanes_at(values,target))) == valid : Bool}) ->
  Logic.When(valid,{U32.to_nat((Geometry.source_of(write_block(within,position,target,source,source_within,source_position,values,lows,boxes,index)) +
      U32.from_nat(read) * Geometry.step_of(write_block(within,position,target,source,source_within,source_position,values,lows,boxes,index)) : U32)) ==
    ShapeLaws.digits_value(Tiled.cell_shape(values,source,source_position),Coordinates.placed_digits(source_within,source_position,
      Geometry.removed_at(Geometry.target_logical(within,position,target,write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane),
        read),source),
      Nat.div(BoxDigits.digit_at(Geometry.target_logical(within,position,target,write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,
        index,lane),read),source),8n))) : Nat}):
  match valid:
    case False{}: Unit{}
    case True{}:
      +rank = List.length(&2,U32,values)
      +cells = Tiled.cell_shape(values,source,source_position)
      +block = write_block(within,position,target,source,source_within,source_position,values,lows,boxes,index)
      +digits = write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)
      +logical_at = Geometry.target_logical(within,position,target,digits,read)
      +placed = Coordinates.placed_digits(source_within,source_position,Geometry.removed_at(logical_at,source),Nat.div(BoxDigits.digit_at(logical_at,source),8n))
      +number = ShapeLaws.digits_value(cells,placed)
      +capacity = Machine.capacity(limits)
      +placed_cells = Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target)))
      +count = Equal.trans(Nat,List.length(&2,Nat,digits),List.length(&2,U32,placed_cells),rank,below_length(placed_cells,digits,below),
        placed_length(within,position,values,target,within_value))
      +word = source_word(within,position,target,rank,source,source_within,source_position,cells,values,lows,boxes,index,lane,read,capacity,Limits.capacity_fits(limits),
        same_blocked,same_lows,small,within_value,apart,count,logical_length(within,position,target,digits,read,values,within_value,below),source_value,
        cell_length(values,source,source_position,source_within,source_value),rank_room)
      +bound = Order.transitive(number,ShapeLaws.elements(cells),U32.to_nat(capacity),
        Bounds.strict_le(number,ShapeLaws.elements(cells),
          read_cell_below(limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,lane,read,within_value,source_value,
            source_lanes_room,below,validity)),
        source_cells_room)
      Equal.trans(Nat,U32.to_nat((Geometry.source_of(block) + U32.from_nat(read) * Geometry.step_of(block) : U32)),U32.to_nat(U32.from_nat(number)),number,
        Equal.cong(U32,Nat,value => U32.to_nat(value),(Geometry.source_of(block) + U32.from_nat(read) * Geometry.step_of(block) : U32),U32.from_nat(number),word),
        Bounds.roundtrip(number,capacity,Limits.capacity_fits(limits),bound))

# The source lane axis of a block over the target cell axes holds lane i of
# its block, or the block reads the unit lane axis at lane zero.
def SourceBlock(within: Bool,+blocked: List<&2,Tiled.Blocked>,+digits: List<&2,Nat>,+lane: Nat) -> Data:
  match within:
    case True{}: Geometry.LaneDigits(blocked,digits,lane)
    case False{}: {lane == 0n : Nat}

def read_digit(source_within: Bool,+values: List<&2,U32>,+target: Nat,+position: Grid.Position,+within: Bool,+digits: List<&2,Nat>,+read: Nat,+lane: Nat,+source: Nat,
  +source_position: Grid.Position,+strides: List<&2,U32>,+axis: U32,+source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +block: SourceBlock(source_within,Tiled.blocked_list(Tiled.target_axes(within,position,target,List.length(&2,U32,values)),source,source_within,source_position,target,
    strides,values,axis),digits,lane)) ->
  SourceLane(source_within,BoxDigits.digit_at(Geometry.target_logical(within,position,target,digits,read),source),lane):
  match source_within:
    case True{}:
      target_lane(within,position,digits,List.length(&2,U32,values),read,lane,source,True{},source_position,target,strides,values,axis,{==},source_value,
        Equal.trans(Bool,Nat.is_eq(source,target),Nat.is_eq(target,source),False{},Order.equal_symmetric(source,target),apart),block)
    case False{}: block

# Lane i of the cell a block of a piece reads at j is lane j of the target
# cell it writes at i.
def write_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+lane: Nat,+target_word: U32,+read: Nat,
  +same_blocked: {List.length(&2,Tiled.Blocked,values_blocked(within,position,target,source,source_within,source_position,values)) == List.length(&2,U32,boxes) : Nat},
  +same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,boxes) : Nat},+small: {Nat.is_lt(lane,8n) == True{} : Bool},
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +block: SourceBlock(source_within,values_blocked(within,position,target,source,source_within,source_position,values),
    write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane),lane),
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),
    write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)),
  +target_exact: {U32.to_nat(target_word) == ShapeLaws.digits_value(Tiled.cell_shape(values,target,position),
    write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)) : Nat},
  +read_small: {Nat.is_lt(read,8n) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Bool.pick(Tile.Tile8,U32.is_lt(U32.from_nat(read),reads_of(write_block(within,position,target,source,source_within,source_position,values,lows,
      boxes,index))),Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},
      (Geometry.source_of(write_block(within,position,target,source,source_within,source_position,values,lows,boxes,index)) +
        U32.from_nat(read) * Geometry.step_of(write_block(within,position,target,source,source_within,source_position,values,lows,boxes,index)) : U32)),Tiled.zero_tile())) ==
    TileLaws.select_lane(read,Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},target_word)) : F32}:
  +block_at = write_block(within,position,target,source,source_within,source_position,values,lows,boxes,index)
  +digits = write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,lane)
  +cells = Tiled.cell_shape(values,source,source_position)
  +valid = Nat.is_lt(Nat.add(read,Nat.mul(cell_group(within,position,digits),8n)),U32.to_nat(Grid.lanes_at(values,target)))
  lane_moved(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,digits,lane,read,reads_of(block_at),
    (Geometry.source_of(block_at) + U32.from_nat(read) * Geometry.step_of(block_at) : U32),target_word,valid,source_value,source_inside,source_room,source_lanes_room,
    read_source(valid,limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,lane,read,same_blocked,same_lows,small,within_value,
      source_value,apart,source_cells_room,source_lanes_room,rank_room,below,{==}),
    small,
    read_digit(source_within,values,target,position,within,digits,read,lane,source,source_position,Tasks.strides_of(cells),groups_axis(source_position,cells),source_value,
      apart,block),
    within_value,inside,room,below,target_exact,read_small,
    read_count(limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,lane,read,same_blocked,same_lows,within_value,lanes_room,below,
      read_small),
    {==})

# The target cell a block of a piece writes at i holds the target value.
def write_tile(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+values: List<&2,U32>,+source: Nat,+source_position: Grid.Position,
  +source_within: Bool,+target: Nat,+position: Grid.Position,+within: Bool,+lows: List<&2,U32>,+boxes: List<&2,U32>,+index: U32,+at: U32,+target_word: U32,
  +same_blocked: {List.length(&2,Tiled.Blocked,values_blocked(within,position,target,source,source_within,source_position,values)) == List.length(&2,U32,boxes) : Nat},
  +same_lows: {List.length(&2,U32,lows) == List.length(&2,U32,boxes) : Nat},+small: {Nat.is_lt(U32.to_nat(at),8n) == True{} : Bool},
  +within_value: {Nat.is_lt(target,List.length(&2,U32,values)) == within : Bool},+inside: {Nat.is_le(target,List.length(&2,U32,values)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,target)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_value: {Nat.is_lt(source,List.length(&2,U32,values)) == source_within : Bool},+source_inside: {Nat.is_le(source,List.length(&2,U32,values)) == True{} : Bool},
  +source_room: {Nat.is_le(ShapeLaws.elements(Grid.removed(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_cells_room: {Nat.is_le(ShapeLaws.elements(Tiled.cell_shape(values,source,source_position)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +source_lanes_room: {Nat.is_le(U32.to_nat(Grid.lanes_at(values,source)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_room: {Nat.is_le(List.length(&2,U32,values),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +apart: {Bool.and(source_within,Nat.is_eq(target,source)) == False{} : Bool},
  +block: SourceBlock(source_within,values_blocked(within,position,target,source,source_within,source_position,values),
    write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,U32.to_nat(at)),U32.to_nat(at)),
  +below: Coordinates.DigitsBelow(Grid.placed(within,position,Grid.removed(values,target),Grid.groups(Grid.lanes_at(values,target))),
    write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,U32.to_nat(at))),
  +target_exact: {U32.to_nat(target_word) == ShapeLaws.digits_value(Tiled.cell_shape(values,target,position),
    write_digits(within,position,target,source,source_within,source_position,values,lows,boxes,index,U32.to_nat(at))) : Nat}) ->
  {Block.eighth(at,Tiled.transposed(Block.expected_eight(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
      Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},
      reads_of(write_block(within,position,target,source,source_within,source_position,values,lows,boxes,index)),
      Geometry.source_of(write_block(within,position,target,source,source_within,source_position,values,lows,boxes,index)),
      Geometry.step_of(write_block(within,position,target,source,source_within,source_position,values,lows,boxes,index))))) ==
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},target_word) : Tile.Tile8}:
  +block_at = write_block(within,position,target,source,source_within,source_position,values,lows,boxes,index)
  +eight = Block.expected_eight(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,Public.geometry_of(values,source,source_position)},reads_of(block_at),Geometry.source_of(block_at),Geometry.step_of(block_at))
  +written = Representation.lifted(~Logical,~original,Representation.Lifted{logical,Public.geometry_of(values,target,position)},target_word)
  Equal.trans(Tile.Tile8,Block.eighth(at,Tiled.transposed(eight)),Block.column(U32.to_nat(at),eight),written,Block.transposed_at(at,eight,small),
    TileLaws.tiles_equal(Block.column(U32.to_nat(at),eight),written,
      write_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,U32.to_nat(at),target_word,0n,
        same_blocked,same_lows,small,within_value,inside,room,lanes_room,source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,
        block,below,target_exact,{==}),
      write_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,U32.to_nat(at),target_word,1n,
        same_blocked,same_lows,small,within_value,inside,room,lanes_room,source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,
        block,below,target_exact,{==}),
      write_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,U32.to_nat(at),target_word,2n,
        same_blocked,same_lows,small,within_value,inside,room,lanes_room,source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,
        block,below,target_exact,{==}),
      write_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,U32.to_nat(at),target_word,3n,
        same_blocked,same_lows,small,within_value,inside,room,lanes_room,source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,
        block,below,target_exact,{==}),
      write_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,U32.to_nat(at),target_word,4n,
        same_blocked,same_lows,small,within_value,inside,room,lanes_room,source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,
        block,below,target_exact,{==}),
      write_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,U32.to_nat(at),target_word,5n,
        same_blocked,same_lows,small,within_value,inside,room,lanes_room,source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,
        block,below,target_exact,{==}),
      write_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,U32.to_nat(at),target_word,6n,
        same_blocked,same_lows,small,within_value,inside,room,lanes_room,source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,
        block,below,target_exact,{==}),
      write_lane(~Logical,~original,logical,limits,values,source,source_position,source_within,target,position,within,lows,boxes,index,U32.to_nat(at),target_word,7n,
        same_blocked,same_lows,small,within_value,inside,room,lanes_room,source_value,source_inside,source_room,source_cells_room,source_lanes_room,rank_room,apart,
        block,below,target_exact,{==})))
