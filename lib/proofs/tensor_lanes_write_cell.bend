# A cell a write along the lane axis reaches: the part cells of the reach
# joined at its shift, put over the target cell between the channels of the
# part, hold in every lane whose channel lies in the part the part lane at the
# channel less the start, and the target lane elsewhere.

import Base
import ../kernel_tile.bend as Tile
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ./kernel_tile_refinement.bend as TileLaws
import ./tensor_tiled_representation.bend as Representation
import ./tensor_lane_shift.bend as LaneShift
import ./nat_order.bend as Order
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_division.bend as Division
import ./nat_division_order.bend as DivisionOrder
import ./nat_to_u32_bounds.bend as Bounds
import ./u32_full_range.bend as FullRange
import ./u32_division.bend as U32Division
import ./u32_comparison.bend as Comparison
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding
import ./tensor_move_readback.bend as Held

# ---- A lane by its channel ----

# The lane of a channel at an outer index: its value below the lane count,
# +0 past it.
def lane_at(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+geometry: Cells.Cells,+outer: Nat,+channel: Nat) -> F32:
  Bool.pick(F32,Nat.is_lt(channel,Representation.lane_count(geometry)),part_value(part,U32.from_nat(Representation.logical_index(geometry,outer,channel))),0.0)

# Two lanes of the same channel hold the same value.
def lane_moved(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+lane: Nat,+other_group: Nat,+other_lane: Nat,
  +same: {Nat.add(lane,Nat.mul(group,8n)) == Nat.add(other_lane,Nat.mul(other_group,8n)) : Nat}) ->
  {Representation.lane(~Part,~part_value,part,geometry,outer,group,lane) == Representation.lane(~Part,~part_value,part,geometry,outer,other_group,other_lane) : F32}:
  Equal.cong(Nat,F32,channel => lane_at(~Part,~part_value,part,geometry,outer,channel),Nat.add(lane,Nat.mul(group,8n)),Nat.add(other_lane,Nat.mul(other_group,8n)),same)

# A distance from the start written as a lane and a group.
def distance_moved(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+lane: Nat,+distance: Nat,
  +exact: {distance == Nat.add(lane,Nat.mul(group,8n)) : Nat}) ->
  {Representation.lane(~Part,~part_value,part,geometry,outer,group,lane) == Representation.lane(~Part,~part_value,part,geometry,outer,0n,distance) : F32}:
  lane_moved(~Part,~part_value,part,geometry,outer,group,lane,0n,distance,
    Equal.trans(Nat,Nat.add(lane,Nat.mul(group,8n)),distance,Nat.add(distance,0n),Equal.sym(Nat,distance,Nat.add(lane,Nat.mul(group,8n)),exact),
      Equal.sym(Nat,Nat.add(distance,0n),distance,Addition.zero_right(distance))))

# ---- Where the reach of a group ends ----

# The distance of a channel from the start, eight more, is the lane moved by
# the shift and the groups of the reach.
def key_of(+start: Nat,+distance: Nat,+lane: Nat,+group: Nat,+reach: Nat,+above: Nat,+shift: Nat,
  +restored: {Nat.add(start,distance) == Nat.add(lane,Nat.mul(group,8n)) : Nat},
  +reached: {Nat.add(start,reach) == Nat.add(Nat.mul(8n,group),8n) : Nat},
  +split: {reach == Nat.add(Nat.mul(above,8n),shift) : Nat}) ->
  {Nat.add(distance,8n) == Nat.add(Nat.add(lane,shift),Nat.mul(above,8n)) : Nat}:
  +moved = Equal.trans(Nat,Nat.add(start,Nat.add(distance,8n)),Nat.add(Nat.add(start,distance),8n),Nat.add(start,Nat.add(lane,reach)),
    Equal.sym(Nat,Nat.add(Nat.add(start,distance),8n),Nat.add(start,Nat.add(distance,8n)),Addition.associative(start,distance,8n)),
    Equal.trans(Nat,Nat.add(Nat.add(start,distance),8n),Nat.add(Nat.add(lane,Nat.mul(group,8n)),8n),Nat.add(start,Nat.add(lane,reach)),
      Equal.cong(Nat,Nat,value => Nat.add(value,8n),Nat.add(start,distance),Nat.add(lane,Nat.mul(group,8n)),restored),
      Equal.trans(Nat,Nat.add(Nat.add(lane,Nat.mul(group,8n)),8n),Nat.add(lane,Nat.add(Nat.mul(8n,group),8n)),Nat.add(start,Nat.add(lane,reach)),
        Equal.trans(Nat,Nat.add(Nat.add(lane,Nat.mul(group,8n)),8n),Nat.add(lane,Nat.add(Nat.mul(group,8n),8n)),Nat.add(lane,Nat.add(Nat.mul(8n,group),8n)),
          Addition.associative(lane,Nat.mul(group,8n),8n),
          Equal.cong(Nat,Nat,value => Nat.add(lane,Nat.add(value,8n)),Nat.mul(group,8n),Nat.mul(8n,group),RowMajor.multiply_commutative(group,8n))),
        Equal.trans(Nat,Nat.add(lane,Nat.add(Nat.mul(8n,group),8n)),Nat.add(lane,Nat.add(start,reach)),Nat.add(start,Nat.add(lane,reach)),
          Equal.cong(Nat,Nat,value => Nat.add(lane,value),Nat.add(Nat.mul(8n,group),8n),Nat.add(start,reach),Equal.sym(Nat,Nat.add(start,reach),Nat.add(Nat.mul(8n,group),8n),reached)),
          RowMajor.add_swap(lane,start,reach)))))
  +cancelled = Order.cancel_sum(start,Nat.add(distance,8n),Nat.add(lane,reach),moved)
  Equal.trans(Nat,Nat.add(distance,8n),Nat.add(lane,reach),Nat.add(Nat.add(lane,shift),Nat.mul(above,8n)),cancelled,
    Equal.trans(Nat,Nat.add(lane,reach),Nat.add(lane,Nat.add(shift,Nat.mul(above,8n))),Nat.add(Nat.add(lane,shift),Nat.mul(above,8n)),
      Equal.cong(Nat,Nat,value => Nat.add(lane,value),reach,Nat.add(shift,Nat.mul(above,8n)),
        Equal.trans(Nat,reach,Nat.add(Nat.mul(above,8n),shift),Nat.add(shift,Nat.mul(above,8n)),split,Order.commutative(Nat.mul(above,8n),shift))),
      Equal.sym(Nat,Nat.add(Nat.add(lane,shift),Nat.mul(above,8n)),Nat.add(lane,Nat.add(shift,Nat.mul(above,8n))),Addition.associative(lane,shift,Nat.mul(above,8n)))))

# A lane moved by the shift inside its group reaches past the first group.
def low_positive(above: Nat,+distance: Nat,+moved: Nat,
  +key: {Nat.add(distance,8n) == Nat.add(moved,Nat.mul(above,8n)) : Nat},+low: {Nat.is_lt(moved,8n) == True{} : Bool}) -> {Nat.is_lt(0n,above) == True{} : Bool}:
  match above:
    case 0n:
      +below = Order.lt_le(moved,8n,Nat.add(distance,8n),low,Order.right_le(distance,8n))
      +same = {%key : {Nat.is_lt(moved,_) == True{} : Bool} below : {Nat.is_lt(moved,Nat.add(moved,0n)) == True{} : Bool}}
      +itself = {%Addition.zero_right(moved) : {Nat.is_lt(moved,_) == True{} : Bool} same : {Nat.is_lt(moved,moved) == True{} : Bool}}
      Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_eq(moved,moved),True{},
        Equal.sym(Bool,Nat.is_eq(moved,moved),False{},Order.less_not_equal(moved,moved,itself)),Order.equal_reflexive(moved))))
    case 1n+ +rest:
      {==}

# A lane moved by the shift inside its group lies a group below the end of
# the reach.
def low_distance(above: Nat,+distance: Nat,+moved: Nat,
  +key: {Nat.add(distance,8n) == Nat.add(moved,Nat.mul(above,8n)) : Nat},+low: {Nat.is_lt(moved,8n) == True{} : Bool}) ->
  {distance == Nat.add(moved,Nat.mul(Nat.sub(above,1n),8n)) : Nat}:
  match above:
    case 0n:
      Empty.absurd({distance == Nat.add(moved,Nat.mul(Nat.sub(0n,1n),8n)) : Nat},WordEncoding.false_true(low_positive(0n,distance,moved,key,low)))
    case 1n+ +rest:
      +swapped = Equal.trans(Nat,Nat.add(8n,distance),Nat.add(distance,8n),Nat.add(8n,Nat.add(moved,Nat.mul(rest,8n))),Order.commutative(8n,distance),
        Equal.trans(Nat,Nat.add(distance,8n),Nat.add(moved,Nat.mul(1n+rest,8n)),Nat.add(8n,Nat.add(moved,Nat.mul(rest,8n))),key,RowMajor.add_swap(moved,8n,Nat.mul(rest,8n))))
      %Equal.sym(Nat,Nat.sub(rest,0n),rest,Order.subtract_zero(rest)) : {distance == Nat.add(moved,Nat.mul(_,8n)) : Nat}
      Order.cancel_sum(8n,distance,Nat.add(moved,Nat.mul(rest,8n)),swapped)

# A lane moved by the shift past its group lies in the group the reach ends
# in.
def high_distance(+distance: Nat,+moved: Nat,+above: Nat,
  +key: {Nat.add(distance,8n) == Nat.add(moved,Nat.mul(above,8n)) : Nat},+high: {Nat.is_lt(moved,8n) == False{} : Bool}) ->
  {distance == Nat.add(Nat.sub(moved,8n),Nat.mul(above,8n)) : Nat}:
  +restored = Order.restore_shift(8n,moved,Order.not_less(moved,8n,high))
  Order.cancel_sum(8n,distance,Nat.add(Nat.sub(moved,8n),Nat.mul(above,8n)),
    Equal.trans(Nat,Nat.add(8n,distance),Nat.add(distance,8n),Nat.add(Nat.add(8n,Nat.sub(moved,8n)),Nat.mul(above,8n)),Order.commutative(8n,distance),
      Equal.trans(Nat,Nat.add(distance,8n),Nat.add(moved,Nat.mul(above,8n)),Nat.add(Nat.add(8n,Nat.sub(moved,8n)),Nat.mul(above,8n)),key,
        Equal.cong(Nat,Nat,value => Nat.add(value,Nat.mul(above,8n)),moved,Nat.add(8n,Nat.sub(moved,8n)),Equal.sym(Nat,Nat.add(8n,Nat.sub(moved,8n)),moved,restored)))))

# A lane moved past its group moves by a positive shift.
def high_shift(+lane: Nat,shift: Nat,+lane_small: {Nat.is_le(lane,7n) == True{} : Bool},+high: {Nat.is_lt(Nat.add(lane,shift),8n) == False{} : Bool}) ->
  {Nat.is_lt(0n,shift) == True{} : Bool}:
  match shift:
    case 0n:
      +past = {%Addition.zero_right(lane) : {Nat.is_lt(_,8n) == False{} : Bool} high : {Nat.is_lt(lane,8n) == False{} : Bool}}
      Empty.absurd({Nat.is_lt(0n,0n) == True{} : Bool},WordEncoding.false_true(Equal.trans(Bool,False{},Nat.is_lt(lane,8n),True{},Equal.sym(Bool,Nat.is_lt(lane,8n),False{},past),
        Order.lt_succ(lane,7n,lane_small))))
    case 1n+ +rest:
      {==}

# A reach of a positive number of groups is at least eight.
def eight_le(above: Nat,+shift: Nat,+positive: {Nat.is_lt(0n,above) == True{} : Bool}) -> {Nat.is_le(8n,Nat.add(Nat.mul(above,8n),shift)) == True{} : Bool}:
  match above:
    case 0n:
      Empty.absurd({Nat.is_le(8n,Nat.add(Nat.mul(0n,8n),shift)) == True{} : Bool},WordEncoding.false_true(positive))
    case 1n+ +rest:
      Order.le_add(8n,Nat.add(Nat.mul(rest,8n),shift))

# Two lanes below eight moved past a group stay below eight.
def moved_small(+lane: Nat,+shift: Nat,+lane_small: {Nat.is_le(lane,7n) == True{} : Bool},+shift_small: {Nat.is_le(shift,7n) == True{} : Bool}) ->
  {Nat.is_le(Nat.sub(Nat.add(lane,shift),8n),7n) == True{} : Bool}:
  +sum = Order.transitive(Nat.add(lane,shift),Nat.add(7n,shift),Nat.add(7n,7n),Order.add_le(lane,7n,shift,lane_small),
    {%Order.commutative(shift,7n) : {Nat.is_le(_,Nat.add(7n,7n)) == True{} : Bool}
      Order.add_le(shift,7n,7n,shift_small) : {Nat.is_le(Nat.add(7n,shift),Nat.add(7n,7n)) == True{} : Bool}})
  %Equal.sym(Bool,Nat.is_le(Nat.sub(Nat.add(lane,shift),8n),7n),Nat.is_le(Nat.add(lane,shift),Nat.add(8n,7n)),Order.subtraction_le(Nat.add(lane,shift),8n,7n)) : {_ == True{} : Bool}
  Order.transitive(Nat.add(lane,shift),Nat.add(7n,7n),Nat.add(8n,7n),sum,{==})

# ---- The lane of a cell a write reaches ----

# One of two values when the flag holds: the same pick of either.
def pick_when(flag: Bool,+first: F32,+second: F32,+other: F32,same: @+hit: {flag == True{} : Bool} -> {first == second : F32}) ->
  {Bool.pick(F32,flag,first,other) == Bool.pick(F32,flag,second,other) : F32}:
  match flag:
    case True{}: same({==})
    case False{}: {==}

# The part lane the joined cells give a lane of a reached channel.
def joined_cases(~Part: Data,~part_value: Part -> U32 -> F32,low: Bool,+part: Part,+geometry: Cells.Cells,+outer: Nat,+lane: Nat,+shift: Nat,+above: Nat,+distance: Nat,
  +width: Nat,+part_groups: Nat,+wanted_low: Bool,+wanted_high: Bool,
  +classified: {Nat.is_lt(Nat.add(lane,shift),8n) == low : Bool},
  +lane_small: {Nat.is_le(lane,7n) == True{} : Bool},+shift_small: {Nat.is_le(shift,7n) == True{} : Bool},
  +key: {Nat.add(distance,8n) == Nat.add(Nat.add(lane,shift),Nat.mul(above,8n)) : Nat},
  +within: {Nat.is_lt(distance,width) == True{} : Bool},+covered: {Nat.is_le(width,Nat.mul(part_groups,8n)) == True{} : Bool},
  low_wanted: @+low_hit: {Nat.is_le(8n,Nat.add(Nat.mul(above,8n),shift)) == True{} : Bool} -> {wanted_low == True{} : Bool},
  high_wanted: @+shift_hit: {Nat.is_lt(0n,shift) == True{} : Bool} -> @+above_hit: {Nat.is_lt(above,part_groups) == True{} : Bool} -> {wanted_high == True{} : Bool}) ->
  {Bool.pick(F32,low,TileLaws.select_lane(Nat.add(lane,shift),Bool.pick(Tile.Tile8,wanted_low,Representation.cell(~Part,~part_value,part,geometry,outer,Nat.sub(above,1n)),Tiled.zero_tile())),
      TileLaws.select_lane(Nat.sub(Nat.add(lane,shift),8n),Bool.pick(Tile.Tile8,wanted_high,Representation.cell(~Part,~part_value,part,geometry,outer,above),Tiled.zero_tile()))) ==
    Representation.lane(~Part,~part_value,part,geometry,outer,0n,distance) : F32}:
  match low:
    case True{}:
      +moved = Nat.add(lane,shift)
      +positive = low_positive(above,distance,moved,key,classified)
      +read = low_wanted(eight_le(above,shift,positive))
      +exact = low_distance(above,distance,moved,key,classified)
      +small = Division.lt_succ_le(moved,7n,classified)
      %Equal.sym(Bool,wanted_low,True{},read) :
        {TileLaws.select_lane(moved,Bool.pick(Tile.Tile8,_,Representation.cell(~Part,~part_value,part,geometry,outer,Nat.sub(above,1n)),Tiled.zero_tile())) ==
          Representation.lane(~Part,~part_value,part,geometry,outer,0n,distance) : F32}
      Equal.trans(F32,TileLaws.select_lane(moved,Representation.cell(~Part,~part_value,part,geometry,outer,Nat.sub(above,1n))),
        Representation.lane(~Part,~part_value,part,geometry,outer,Nat.sub(above,1n),moved),Representation.lane(~Part,~part_value,part,geometry,outer,0n,distance),
        Representation.cell_lane(~Part,~part_value,part,geometry,outer,Nat.sub(above,1n),moved,small),
        distance_moved(~Part,~part_value,part,geometry,outer,Nat.sub(above,1n),moved,distance,exact))
    case False{}:
      +moved = Nat.add(lane,shift)
      +exact = high_distance(distance,moved,above,key,classified)
      +shift_positive = high_shift(lane,shift,lane_small,classified)
      +below_width = Order.le_lt_trans(Nat.mul(above,8n),distance,width,
        {%Equal.sym(Nat,distance,Nat.add(Nat.sub(moved,8n),Nat.mul(above,8n)),exact) : {Nat.is_le(Nat.mul(above,8n),_) == True{} : Bool}
          Order.right_le(Nat.sub(moved,8n),Nat.mul(above,8n)) : {Nat.is_le(Nat.mul(above,8n),distance) == True{} : Bool}},within)
      +above_inside = RowMajor.cancel_multiply_lt(above,part_groups,8n,Order.lt_le(Nat.mul(above,8n),width,Nat.mul(part_groups,8n),below_width,covered))
      +read = high_wanted(shift_positive,above_inside)
      %Equal.sym(Bool,wanted_high,True{},read) :
        {TileLaws.select_lane(Nat.sub(moved,8n),Bool.pick(Tile.Tile8,_,Representation.cell(~Part,~part_value,part,geometry,outer,above),Tiled.zero_tile())) ==
          Representation.lane(~Part,~part_value,part,geometry,outer,0n,distance) : F32}
      Equal.trans(F32,TileLaws.select_lane(Nat.sub(moved,8n),Representation.cell(~Part,~part_value,part,geometry,outer,above)),
        Representation.lane(~Part,~part_value,part,geometry,outer,above,Nat.sub(moved,8n)),Representation.lane(~Part,~part_value,part,geometry,outer,0n,distance),
        Representation.cell_lane(~Part,~part_value,part,geometry,outer,above,Nat.sub(moved,8n),moved_small(lane,shift,lane_small,shift_small)),
        distance_moved(~Part,~part_value,part,geometry,outer,above,Nat.sub(moved,8n),distance,exact))

# ---- The words of a reached group ----

# Eight past the first channel of a group, less the start: the part channel
# the next group begins at.
def reach_word(+start: U32,+group: U32) -> U32:
  (8 * group + 8 - start : U32)

# The channel of a lane of a group.
def channel_of(+group: U32,+lane: Nat) -> Nat:
  Nat.add(lane,Nat.mul(U32.to_nat(group),8n))

# Whether a channel lies in the part.
def in_part(+start: U32,+width: U32,+channel: Nat) -> Bool:
  Nat.is_le(U32.to_nat(start),channel) && Nat.is_lt(channel,Nat.add(U32.to_nat(start),U32.to_nat(width)))

# The cell a write gives a group of the target at an outer index: the part
# cells of the reach joined at its shift, put over the target cell between
# the channels of the part.
def merged(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+geometry: Cells.Cells,+outer: Nat,+start: U32,+width: U32,+group: U32,+part_groups: U32,
  +target: Tile.Tile8) -> Tile.Tile8:
  +reach = reach_word(start,group)
  Tiled.lanes_between(start,(start + width : U32),(8 * group : U32),
    Tiled.joined((reach % 8 : U32),
      Bool.pick(Tile.Tile8,U32.is_le(8,reach),Representation.cell(~Part,~part_value,part,geometry,outer,Nat.sub(U32.to_nat((reach / 8 : U32)),1n)),Tiled.zero_tile()),
      Bool.pick(Tile.Tile8,U32.is_lt(0,(reach % 8 : U32)) && U32.is_lt((reach / 8 : U32),part_groups),
        Representation.cell(~Part,~part_value,part,geometry,outer,U32.to_nat((reach / 8 : U32))),Tiled.zero_tile())),
    target)

# The joined cells of a reached group give a lane of a channel in the part
# the part lane at the channel less the start.
def reached_lane(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+geometry: Cells.Cells,+outer: Nat,+start: U32,+width: U32,+group: U32,+part_groups: U32,
  +lane: Nat,+lane_small: {Nat.is_le(lane,7n) == True{} : Bool},
  +shift_small: {Nat.is_le(U32.to_nat((reach_word(start,group) % 8 : U32)),7n) == True{} : Bool},
  +reached: {Nat.add(U32.to_nat(start),U32.to_nat(reach_word(start,group))) == Nat.add(Nat.mul(8n,U32.to_nat(group)),8n) : Nat},
  +split: {U32.to_nat(reach_word(start,group)) == Nat.add(Nat.mul(U32.to_nat((reach_word(start,group) / 8 : U32)),8n),U32.to_nat((reach_word(start,group) % 8 : U32))) : Nat},
  +covered: {Nat.is_le(U32.to_nat(width),Nat.mul(U32.to_nat(part_groups),8n)) == True{} : Bool},
  +hit: {in_part(start,width,channel_of(group,lane)) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.joined((reach_word(start,group) % 8 : U32),
      Bool.pick(Tile.Tile8,U32.is_le(8,reach_word(start,group)),Representation.cell(~Part,~part_value,part,geometry,outer,Nat.sub(U32.to_nat((reach_word(start,group) / 8 : U32)),1n)),Tiled.zero_tile()),
      Bool.pick(Tile.Tile8,U32.is_lt(0,(reach_word(start,group) % 8 : U32)) && U32.is_lt((reach_word(start,group) / 8 : U32),part_groups),
        Representation.cell(~Part,~part_value,part,geometry,outer,U32.to_nat((reach_word(start,group) / 8 : U32))),Tiled.zero_tile()))) ==
    Representation.lane(~Part,~part_value,part,geometry,outer,0n,Nat.sub(channel_of(group,lane),U32.to_nat(start))) : F32}:
  +s = U32.to_nat(start)
  +w = U32.to_nat(width)
  +reach = reach_word(start,group)
  +above = U32.to_nat((reach / 8 : U32))
  +shift = U32.to_nat((reach % 8 : U32))
  +channel = channel_of(group,lane)
  +distance = Nat.sub(channel,s)
  +low_cell = Bool.pick(Tile.Tile8,U32.is_le(8,reach),Representation.cell(~Part,~part_value,part,geometry,outer,Nat.sub(above,1n)),Tiled.zero_tile())
  +high_cell = Bool.pick(Tile.Tile8,U32.is_lt(0,(reach % 8 : U32)) && U32.is_lt((reach / 8 : U32),part_groups),Representation.cell(~Part,~part_value,part,geometry,outer,above),Tiled.zero_tile())
  +lower = Boolean.left(Nat.is_le(s,channel),Nat.is_lt(channel,Nat.add(s,w)),hit)
  +upper = Boolean.right(Nat.is_le(s,channel),Nat.is_lt(channel,Nat.add(s,w)),hit)
  +restored = Order.restore_shift(s,channel,lower)
  +key = key_of(s,distance,lane,U32.to_nat(group),U32.to_nat(reach),above,shift,restored,reached,split)
  +within = Order.cancel_offset_lt(s,distance,w,{%Equal.sym(Nat,Nat.add(s,distance),channel,restored) : {Nat.is_lt(_,Nat.add(s,w)) == True{} : Bool} upper :
    {Nat.is_lt(Nat.add(s,distance),Nat.add(s,w)) == True{} : Bool}})
  +shift_word = Arithmetic.native_roundtrip((reach % 8 : U32))
  %shift_word : {TileLaws.select_lane(lane,Tiled.joined(_,low_cell,high_cell)) == Representation.lane(~Part,~part_value,part,geometry,outer,0n,distance) : F32}
  Equal.trans(F32,TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),low_cell,high_cell)),LaneShift.joined_value(shift,lane,low_cell,high_cell),
    Representation.lane(~Part,~part_value,part,geometry,outer,0n,distance),
    LaneShift.joined_lane(shift,lane,low_cell,high_cell,shift_small,lane_small),
    joined_cases(~Part,~part_value,Nat.is_lt(Nat.add(lane,shift),8n),part,geometry,outer,lane,shift,above,distance,w,U32.to_nat(part_groups),
      U32.is_le(8,reach),U32.is_lt(0,(reach % 8 : U32)) && U32.is_lt((reach / 8 : U32),part_groups),{==},lane_small,shift_small,key,within,covered,
      low_hit => Equal.trans(Bool,U32.is_le(8,reach),Nat.is_le(8n,U32.to_nat(reach)),True{},Comparison.le(8,reach),
        {%Equal.sym(Nat,U32.to_nat(reach),Nat.add(Nat.mul(above,8n),shift),split) : {Nat.is_le(8n,_) == True{} : Bool} low_hit : {Nat.is_le(8n,U32.to_nat(reach)) == True{} : Bool}}),
      shift_hit => above_hit => Boolean.and_true(U32.is_lt(0,(reach % 8 : U32)),U32.is_lt((reach / 8 : U32),part_groups),
        Equal.trans(Bool,U32.is_lt(0,(reach % 8 : U32)),Nat.is_lt(0n,shift),True{},Comparison.lt(0,(reach % 8 : U32)),shift_hit),
        Equal.trans(Bool,U32.is_lt((reach / 8 : U32),part_groups),Nat.is_lt(above,U32.to_nat(part_groups)),True{},Comparison.lt((reach / 8 : U32),part_groups),above_hit))))

# The words of a reached group: its reach from the start, split at eight.
def reach_facts(+start: U32,+group: U32,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+divisor_room: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},
  +reached: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +group_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(cap)) == True{} : Bool}) ->
  Held.Both<{Nat.add(U32.to_nat(start),U32.to_nat(reach_word(start,group))) == Nat.add(Nat.mul(8n,U32.to_nat(group)),8n) : Nat},
    {U32.to_nat(reach_word(start,group)) == Nat.add(Nat.mul(U32.to_nat((reach_word(start,group) / 8 : U32)),8n),U32.to_nat((reach_word(start,group) % 8 : U32))) : Nat}>:
  +s = U32.to_nat(start)
  +k = U32.to_nat(group)
  +eight = Nat.mul(8n,k)
  +base = (8 * group : U32)
  +reach = reach_word(start,group)
  +eight_room = Order.transitive(eight,Nat.add(eight,8n),U32.to_nat(cap),Order.le_add(eight,8n),group_room)
  +base_exact = FullRange.multiply(8,group,cap,eight_room)
  +plus_exact = Equal.trans(Nat,U32.to_nat((base + 8 : U32)),Nat.add(U32.to_nat(base),8n),Nat.add(eight,8n),
    FullRange.add(base,8,cap,{%Equal.sym(Nat,U32.to_nat(base),eight,base_exact) : {Nat.is_le(Nat.add(_,8n),U32.to_nat(cap)) == True{} : Bool} group_room :
      {Nat.is_le(Nat.add(U32.to_nat(base),8n),U32.to_nat(cap)) == True{} : Bool}}),
    Equal.cong(Nat,Nat,value => Nat.add(value,8n),U32.to_nat(base),eight,base_exact))
  +below = Order.lt_le(s,Nat.mul(1n+Nat.div(s,8n),8n),Nat.mul(1n+k,8n),DivisionOrder.next_above(s,8n,{==}),RowMajor.mul_le(1n+Nat.div(s,8n),1n+k,8n,reached))
  +below_eight = {%RowMajor.multiply_commutative(k,8n) : {Nat.is_lt(s,Nat.add(_,8n)) == True{} : Bool}
    %Order.commutative(8n,Nat.mul(k,8n)) : {Nat.is_lt(s,_) == True{} : Bool} below : {Nat.is_lt(s,Nat.add(eight,8n)) == True{} : Bool}}
  +start_le = Bounds.strict_le(s,Nat.add(eight,8n),below_eight)
  +ordered = {%Equal.sym(Nat,U32.to_nat((base + 8 : U32)),Nat.add(eight,8n),plus_exact) : {Nat.is_le(s,_) == True{} : Bool} start_le :
    {Nat.is_le(s,U32.to_nat((base + 8 : U32))) == True{} : Bool}}
  +difference_room = Order.transitive(Nat.sub(U32.to_nat((base + 8 : U32)),s),U32.to_nat((base + 8 : U32)),U32.to_nat(cap),Order.sub_le(U32.to_nat((base + 8 : U32)),s),
    {%Equal.sym(Nat,U32.to_nat((base + 8 : U32)),Nat.add(eight,8n),plus_exact) : {Nat.is_le(_,U32.to_nat(cap)) == True{} : Bool} group_room :
      {Nat.is_le(U32.to_nat((base + 8 : U32)),U32.to_nat(cap)) == True{} : Bool}})
  +reach_exact = Equal.trans(Nat,U32.to_nat(reach),Nat.sub(U32.to_nat((base + 8 : U32)),s),Nat.sub(Nat.add(eight,8n),s),
    FullRange.subtract((base + 8 : U32),start,cap,ordered,difference_room),
    Equal.cong(Nat,Nat,value => Nat.sub(value,s),U32.to_nat((base + 8 : U32)),Nat.add(eight,8n),plus_exact))
  +reached_sum = {%Equal.sym(Nat,U32.to_nat(reach),Nat.sub(Nat.add(eight,8n),s),reach_exact) : {Nat.add(s,_) == Nat.add(eight,8n) : Nat}
    Order.restore_shift(s,Nat.add(eight,8n),start_le) : {Nat.add(s,U32.to_nat(reach)) == Nat.add(eight,8n) : Nat}}
  +above_exact = U32Division.divide(reach,8,cap,cap_ok,divisor_room,{==})
  +shift_exact = U32Division.modulo(reach,8,cap,cap_ok,divisor_room,{==})
  +split = {%Equal.sym(Nat,U32.to_nat((reach / 8 : U32)),Nat.div(U32.to_nat(reach),8n),above_exact) : {U32.to_nat(reach) == Nat.add(Nat.mul(_,8n),U32.to_nat((reach % 8 : U32))) : Nat}
    %Equal.sym(Nat,U32.to_nat((reach % 8 : U32)),Nat.mod(U32.to_nat(reach),8n),shift_exact) : {U32.to_nat(reach) == Nat.add(Nat.mul(Nat.div(U32.to_nat(reach),8n),8n),_) : Nat}
    DivisionOrder.quotient_exact(U32.to_nat(reach),8n,{==}) :
    {U32.to_nat(reach) == Nat.add(Nat.mul(U32.to_nat((reach / 8 : U32)),8n),U32.to_nat((reach % 8 : U32))) : Nat}}
  Held.Both{reached_sum,split}

# The lane of a group lies in the part exactly when its channel does.
def condition_exact(+start: U32,+width: U32,+group: U32,+lane: Nat,+cap: U32,+lane_small: {Nat.is_le(lane,7n) == True{} : Bool},
  +group_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(cap)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.is_le(start,((8 * group : U32) + U32.from_nat(lane) : U32)) && U32.is_lt(((8 * group : U32) + U32.from_nat(lane) : U32),(start + width : U32)) ==
    in_part(start,width,channel_of(group,lane)) : Bool}:
  +s = U32.to_nat(start)
  +k = U32.to_nat(group)
  +eight = Nat.mul(8n,k)
  +base = (8 * group : U32)
  +word = (base + U32.from_nat(lane) : U32)
  +channel = channel_of(group,lane)
  +eight_room = Order.transitive(eight,Nat.add(eight,8n),U32.to_nat(cap),Order.le_add(eight,8n),group_room)
  +base_exact = FullRange.multiply(8,group,cap,eight_room)
  +lane_le = Bounds.strict_le(lane,8n,Order.lt_succ(lane,7n,lane_small))
  +lane_room = Order.transitive(lane,Nat.add(eight,8n),U32.to_nat(cap),Order.transitive(lane,8n,Nat.add(eight,8n),lane_le,Order.right_le(eight,8n)),group_room)
  +lane_exact = FullRange.roundtrip(lane,cap,lane_room)
  +room = Order.transitive(Nat.add(eight,lane),Nat.add(eight,8n),U32.to_nat(cap),Order.offset_le(eight,lane,8n,lane_le),group_room)
  +sum_room = {%Equal.sym(Nat,U32.to_nat(base),eight,base_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(U32.from_nat(lane))),U32.to_nat(cap)) == True{} : Bool}
    %Equal.sym(Nat,U32.to_nat(U32.from_nat(lane)),lane,lane_exact) : {Nat.is_le(Nat.add(eight,_),U32.to_nat(cap)) == True{} : Bool} room :
    {Nat.is_le(Nat.add(U32.to_nat(base),U32.to_nat(U32.from_nat(lane))),U32.to_nat(cap)) == True{} : Bool}}
  +word_exact = Equal.trans(Nat,U32.to_nat(word),Nat.add(U32.to_nat(base),U32.to_nat(U32.from_nat(lane))),channel,FullRange.add(base,U32.from_nat(lane),cap,sum_room),
    Equal.trans(Nat,Nat.add(U32.to_nat(base),U32.to_nat(U32.from_nat(lane))),Nat.add(eight,lane),channel,
      Equal.trans(Nat,Nat.add(U32.to_nat(base),U32.to_nat(U32.from_nat(lane))),Nat.add(eight,U32.to_nat(U32.from_nat(lane))),Nat.add(eight,lane),
        Equal.cong(Nat,Nat,value => Nat.add(value,U32.to_nat(U32.from_nat(lane))),U32.to_nat(base),eight,base_exact),
        Equal.cong(Nat,Nat,value => Nat.add(eight,value),U32.to_nat(U32.from_nat(lane)),lane,lane_exact)),
      Equal.trans(Nat,Nat.add(eight,lane),Nat.add(lane,eight),channel,Order.commutative(eight,lane),
        Equal.cong(Nat,Nat,value => Nat.add(lane,value),eight,Nat.mul(k,8n),RowMajor.multiply_commutative(8n,k)))))
  +end_exact = FullRange.add(start,width,cap,end_room)
  +first = Equal.trans(Bool,U32.is_le(start,word),Nat.is_le(s,U32.to_nat(word)),Nat.is_le(s,channel),Comparison.le(start,word),
    Equal.cong(Nat,Bool,value => Nat.is_le(s,value),U32.to_nat(word),channel,word_exact))
  +second = Equal.trans(Bool,U32.is_lt(word,(start + width : U32)),Nat.is_lt(U32.to_nat(word),U32.to_nat((start + width : U32))),Nat.is_lt(channel,Nat.add(s,U32.to_nat(width))),
    Comparison.lt(word,(start + width : U32)),
    Equal.trans(Bool,Nat.is_lt(U32.to_nat(word),U32.to_nat((start + width : U32))),Nat.is_lt(channel,U32.to_nat((start + width : U32))),Nat.is_lt(channel,Nat.add(s,U32.to_nat(width))),
      Equal.cong(Nat,Bool,value => Nat.is_lt(value,U32.to_nat((start + width : U32))),U32.to_nat(word),channel,word_exact),
      Equal.cong(Nat,Bool,value => Nat.is_lt(channel,value),U32.to_nat((start + width : U32)),Nat.add(s,U32.to_nat(width)),end_exact)))
  Equal.trans(Bool,U32.is_le(start,word) && U32.is_lt(word,(start + width : U32)),Nat.is_le(s,channel) && U32.is_lt(word,(start + width : U32)),in_part(start,width,channel),
    Equal.cong(Bool,Bool,value => value && U32.is_lt(word,(start + width : U32)),U32.is_le(start,word),Nat.is_le(s,channel),first),
    Equal.cong(Bool,Bool,value => Nat.is_le(s,channel) && value,U32.is_lt(word,(start + width : U32)),Nat.is_lt(channel,Nat.add(s,U32.to_nat(width))),second))

# A lane of the cell a write gives a reached group: the part lane at the
# channel less the start when the channel lies in the part, the target lane
# otherwise.
def merged_lane(~Part: Data,~part_value: Part -> U32 -> F32,+part: Part,+geometry: Cells.Cells,+outer: Nat,+start: U32,+width: U32,+group: U32,+part_groups: U32,
  +target: Tile.Tile8,+lane: Nat,+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+divisor_room: {Nat.is_le(16n,U32.to_nat(cap)) == True{} : Bool},
  +lane_small: {Nat.is_le(lane,7n) == True{} : Bool},
  +reached: {Nat.is_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(group)) == True{} : Bool},
  +group_room: {Nat.is_le(Nat.add(Nat.mul(8n,U32.to_nat(group)),8n),U32.to_nat(cap)) == True{} : Bool},
  +end_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(cap)) == True{} : Bool},
  +covered: {Nat.is_le(U32.to_nat(width),Nat.mul(U32.to_nat(part_groups),8n)) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,merged(~Part,~part_value,part,geometry,outer,start,width,group,part_groups,target)) ==
    Bool.pick(F32,in_part(start,width,channel_of(group,lane)),Representation.lane(~Part,~part_value,part,geometry,outer,0n,Nat.sub(channel_of(group,lane),U32.to_nat(start))),
      TileLaws.select_lane(lane,target)) : F32}:
  +facts = reach_facts(start,group,cap,cap_ok,divisor_room,reached,group_room)
  +reached_sum = Held.first_of({Nat.add(U32.to_nat(start),U32.to_nat(reach_word(start,group))) == Nat.add(Nat.mul(8n,U32.to_nat(group)),8n) : Nat},{U32.to_nat(reach_word(start,group)) == Nat.add(Nat.mul(U32.to_nat((reach_word(start,group) / 8 : U32)),8n),U32.to_nat((reach_word(start,group) % 8 : U32))) : Nat},facts)
  +split = Held.second_of({Nat.add(U32.to_nat(start),U32.to_nat(reach_word(start,group))) == Nat.add(Nat.mul(8n,U32.to_nat(group)),8n) : Nat},{U32.to_nat(reach_word(start,group)) == Nat.add(Nat.mul(U32.to_nat((reach_word(start,group) / 8 : U32)),8n),U32.to_nat((reach_word(start,group) % 8 : U32))) : Nat},facts)
  +reach = reach_word(start,group)
  +joined = Tiled.joined((reach % 8 : U32),
    Bool.pick(Tile.Tile8,U32.is_le(8,reach),Representation.cell(~Part,~part_value,part,geometry,outer,Nat.sub(U32.to_nat((reach / 8 : U32)),1n)),Tiled.zero_tile()),
    Bool.pick(Tile.Tile8,U32.is_lt(0,(reach % 8 : U32)) && U32.is_lt((reach / 8 : U32),part_groups),
      Representation.cell(~Part,~part_value,part,geometry,outer,U32.to_nat((reach / 8 : U32))),Tiled.zero_tile()))
  +inside = in_part(start,width,channel_of(group,lane))
  +part_lane = Representation.lane(~Part,~part_value,part,geometry,outer,0n,Nat.sub(channel_of(group,lane),U32.to_nat(start)))
  +word = ((8 * group : U32) + U32.from_nat(lane) : U32)
  +condition = U32.is_le(start,word) && U32.is_lt(word,(start + width : U32))
  +shift_exact = U32Division.modulo(reach,8,cap,cap_ok,divisor_room,{==})
  +shift_small = Division.lt_succ_le(U32.to_nat((reach % 8 : U32)),7n,
    {%Equal.sym(Nat,U32.to_nat((reach % 8 : U32)),Nat.mod(U32.to_nat(reach),8n),shift_exact) : {Nat.is_lt(_,8n) == True{} : Bool}
      DivisionOrder.remainder_small(U32.to_nat(reach),8n,{==}) : {Nat.is_lt(U32.to_nat((reach % 8 : U32)),8n) == True{} : Bool}})
  Equal.trans(F32,TileLaws.select_lane(lane,merged(~Part,~part_value,part,geometry,outer,start,width,group,part_groups,target)),
    LaneShift.between_value(start,(start + width : U32),(8 * group : U32),lane,joined,target),Bool.pick(F32,inside,part_lane,TileLaws.select_lane(lane,target)),
    LaneShift.between_lane(lane,start,(start + width : U32),(8 * group : U32),joined,target,lane_small),
    Equal.trans(F32,Bool.pick(F32,condition,TileLaws.select_lane(lane,joined),TileLaws.select_lane(lane,target)),
      Bool.pick(F32,inside,TileLaws.select_lane(lane,joined),TileLaws.select_lane(lane,target)),Bool.pick(F32,inside,part_lane,TileLaws.select_lane(lane,target)),
      Equal.cong(Bool,F32,flag => Bool.pick(F32,flag,TileLaws.select_lane(lane,joined),TileLaws.select_lane(lane,target)),condition,inside,
        condition_exact(start,width,group,lane,cap,lane_small,group_room,end_room)),
      pick_when(inside,TileLaws.select_lane(lane,joined),part_lane,TileLaws.select_lane(lane,target),
        hit => reached_lane(~Part,~part_value,part,geometry,outer,start,width,group,part_groups,lane,lane_small,shift_small,reached_sum,split,covered,hit))))
