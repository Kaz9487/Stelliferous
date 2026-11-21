# The laws a split gather needs: every target cell of a target piece below
# the target cells holds its target value from the source cells its spot
# reads, and with the groups innermost every run of a target piece covers
# consecutive groups of one anchor, read from consecutive source cells
# shifted alike.

import Base
import ../kernel_tile.bend as Tile
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as Limits
import ./tensor_tiled_representation.bend as Representation
import ./tensor_split_value.bend as SplitValue
import ./tensor_split_gather.bend as Gather
import ./boolean_logic.bend as Logic
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./nat_division_order.bend as DivisionOrder
import ./row_major_index.bend as RowMajor
import ./u32_comparison.bend as Comparison
import ./kernel_tile_refinement.bend as TileLaws
import ./u32_extrema.bend as Extrema
import ./nat_extrema.bend as NatExtrema
import ./u32_full_range.bend as FullRange
import ./tensor_tiled_values.bend as TiledValues

# ---- The words of a split ----

# A split of a lane axis of lanes = anchors * width lanes, its outer and
# after elements, at a position.
def split_of(+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+position: Grid.Position) -> Tiled.Split:
  Tiled.Split{outer,anchors,width,after,Tiled.groups(width),Tiled.groups(lanes),position}

# The source and target cells of a split.
def source_cells(+source_outer: U32,+source_after: U32,+lanes: U32,+position: Grid.Position) -> Cells.Cells:
  Cells.Cells{source_outer,source_after,lanes,Tiled.groups(lanes),position}

def target_cells(+target_outer: U32,+target_after: U32,+width: U32,+position: Grid.Position) -> Cells.Cells:
  Cells.Cells{target_outer,target_after,width,Tiled.groups(width),position}

# ---- Every target cell ----

# A local index of a target piece below the target cells lies below them.
def target_inside(+limits: Machine.Limits,+width: U32,+target_outer: U32,+first: U32,+count: U32,
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool},
  +local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool}) ->
  {Nat.is_lt(U32.to_nat((first + local : U32)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool}:
  +cells = Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))
  +reached = Order.lt_le(Nat.add(U32.to_nat(first),U32.to_nat(local)),Nat.add(U32.to_nat(first),U32.to_nat(count)),cells,
    Order.offset_lt(U32.to_nat(first),U32.to_nat(local),U32.to_nat(count),below),piece_room)
  %Equal.sym(Nat,U32.to_nat((first + local : U32)),Nat.add(U32.to_nat(first),U32.to_nat(local)),
      Arithmetic.sum_exact(Machine.capacity(limits),Limits.capacity_fits(limits),first,local,U32.to_nat(first),U32.to_nat(local),{==},{==},
        Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(local)),cells,U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(Nat.add(U32.to_nat(first),U32.to_nat(local)),cells,reached),
          cells_room))) :
    {Nat.is_lt(_,cells) == True{} : Bool}
  reached

# A target cell of a target piece below the target cells: for every source
# cell its spot reads at the source value and every next one that holds its
# value when a lane needs it, the target value at its index.
def cell_law_at(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,
  +position: Grid.Position,+target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,
  +facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+count: U32,
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool},
  +local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},+low: Tile.Tile8,+high: Tile.Tile8,
  +low_exact: {low == Representation.lifted(~Logical,~original,Representation.Lifted{logical,source_cells(source_outer,source_after,lanes,position)},
    SplitValue.spot_read(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),(first + local : U32)))) : Tile.Tile8},
  +high_exact: Logic.When(SplitValue.spot_needed(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),(first + local : U32))),
    {high == Representation.lifted(~Logical,~original,Representation.Lifted{logical,source_cells(source_outer,source_after,lanes,position)},
      SplitValue.spot_high(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),(first + local : U32)))) : Tile.Tile8})) ->
  {SplitValue.spot_cell(width,Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),(first + local : U32)),low,high) ==
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,target_cells(target_outer,target_after,width,position)},(first + local : U32)) : Tile.Tile8}:
  +index = (first + local : U32)
  +spot = Tiled.split_spot(position,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index)
  +digits = SplitValue.split_digits(limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,facts,index,
    target_inside(limits,width,target_outer,first,count,cells_room,piece_room,local,below))
  SplitValue.split_value(~Logical,~original,logical,limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,facts,index,digits,
    low,high,low_exact,
    need => Logic.required({high == Representation.lifted(~Logical,~original,Representation.Lifted{logical,source_cells(source_outer,source_after,lanes,position)},
      SplitValue.spot_high(width,spot)) : Tile.Tile8},SplitValue.spot_needed(width,spot),high_exact,need))

# Every target cell of a target piece below the target cells.
def cell_law(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,
  +position: Grid.Position,+target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,
  +facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+count: U32,
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool}) ->
  Gather.CellLaw(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,source_cells(source_outer,source_after,lanes,position)},Representation.Lifted{logical,target_cells(target_outer,target_after,width,position)},
    split_of(outer,anchors,width,after,lanes,position),first,count):
  local => below => low => high => low_exact => high_exact =>
    cell_law_at(~Logical,~original,logical,limits,outer,anchors,width,after,lanes,position,target_outer,target_after,source_outer,source_after,facts,first,count,
      cells_room,piece_room,local,below,low,high,low_exact,high_exact)

# ---- Words and numbers ----

# Words of one number are one word.
def word_same(+left: U32,+right: U32,+same: {U32.to_nat(left) == U32.to_nat(right) : Nat}) -> {left == right : U32}:
  %Arithmetic.native_roundtrip(left) : {_ == right : U32}
  Arithmetic.from_nat_exact(right,U32.to_nat(left),Equal.sym(Nat,U32.to_nat(left),U32.to_nat(right),same))

# A word times one, below the cap.
def times_one_word(+limits: Machine.Limits,+word: U32,+room: {Nat.is_le(U32.to_nat(word),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) -> {(word * 1 : U32) == word : U32}:
  word_same((word * 1 : U32),word,
    Equal.trans(Nat,U32.to_nat((word * 1 : U32)),Nat.mul(U32.to_nat(word),1n),U32.to_nat(word),
      Arithmetic.product_exact(Machine.half_word(limits),Limits.half_word_fits(limits),word,1,U32.to_nat(word),1n,{==},{==},
        {%Equal.sym(Nat,Nat.mul(U32.to_nat(word),1n),U32.to_nat(word),Order.times_one(U32.to_nat(word))) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
          room : {Nat.is_le(Nat.mul(U32.to_nat(word),1n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}),
      Order.times_one(U32.to_nat(word))))

# Two numbers below the capacity fit in the half word together.
def double_room(+limits: Machine.Limits,+left: Nat,+right: Nat,+left_room: {Nat.is_le(left,U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +right_room: {Nat.is_le(right,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(left,right),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +capacity = U32.to_nat(Machine.capacity(limits))
  %Equal.sym(Nat,U32.to_nat(Machine.half_word(limits)),Nat.double(capacity),Limits.doubled_capacity(limits)) : {Nat.is_le(Nat.add(left,right),_) == True{} : Bool}
  %Addition.double(capacity) : {Nat.is_le(Nat.add(left,right),_) == True{} : Bool}
  Order.transitive(Nat.add(left,right),Nat.add(capacity,right),Nat.add(capacity,capacity),Order.add_le(left,capacity,right,left_room),
    Order.offset_le(capacity,right,capacity,right_room))

# The first lane of the group of a step later: as many cells later, at the
# same shift.
def stepped_lane(+anchor: Nat,+width: U32,+group: Nat,+step: Nat) ->
  {SplitValue.first_lane(anchor,width,Nat.add(group,step)) == Nat.add(Nat.mul(step,8n),SplitValue.first_lane(anchor,width,group)) : Nat}:
  +taken = Nat.mul(anchor,U32.to_nat(width))
  %Equal.sym(Nat,Nat.mul(Nat.add(group,step),8n),Nat.add(Nat.mul(group,8n),Nat.mul(step,8n)),RowMajor.distrib(group,step,8n)) :
    {Nat.add(taken,_) == Nat.add(Nat.mul(step,8n),Nat.add(taken,Nat.mul(group,8n))) : Nat}
  %Addition.associative(taken,Nat.mul(group,8n),Nat.mul(step,8n)) :
    {_ == Nat.add(Nat.mul(step,8n),Nat.add(taken,Nat.mul(group,8n))) : Nat}
  Order.commutative(Nat.add(taken,Nat.mul(group,8n)),Nat.mul(step,8n))

def stepped_cell(+anchor: Nat,+width: U32,+group: Nat,+step: Nat) ->
  {Nat.div(SplitValue.first_lane(anchor,width,Nat.add(group,step)),8n) == Nat.add(Nat.div(SplitValue.first_lane(anchor,width,group),8n),step) : Nat}:
  %Equal.sym(Nat,SplitValue.first_lane(anchor,width,Nat.add(group,step)),Nat.add(Nat.mul(step,8n),SplitValue.first_lane(anchor,width,group)),stepped_lane(anchor,width,group,step)) :
    {Nat.div(_,8n) == Nat.add(Nat.div(SplitValue.first_lane(anchor,width,group),8n),step) : Nat}
  Equal.trans(Nat,Nat.div(Nat.add(Nat.mul(step,8n),SplitValue.first_lane(anchor,width,group)),8n),Nat.add(step,Nat.div(SplitValue.first_lane(anchor,width,group),8n)),
    Nat.add(Nat.div(SplitValue.first_lane(anchor,width,group),8n),step),DivisionOrder.quotient_shift(SplitValue.first_lane(anchor,width,group),8n,step,{==}),
    Order.commutative(step,Nat.div(SplitValue.first_lane(anchor,width,group),8n)))

def stepped_shift(+anchor: Nat,+width: U32,+group: Nat,+step: Nat) ->
  {Nat.mod(SplitValue.first_lane(anchor,width,Nat.add(group,step)),8n) == Nat.mod(SplitValue.first_lane(anchor,width,group),8n) : Nat}:
  %Equal.sym(Nat,SplitValue.first_lane(anchor,width,Nat.add(group,step)),Nat.add(Nat.mul(step,8n),SplitValue.first_lane(anchor,width,group)),stepped_lane(anchor,width,group,step)) :
    {Nat.mod(_,8n) == Nat.mod(SplitValue.first_lane(anchor,width,group),8n) : Nat}
  DivisionOrder.remainder_shift(SplitValue.first_lane(anchor,width,group),8n,step,{==})

# ---- The lanes of a whole group ----

# A lane inside the width keeps its value.
def lane_kept(+first: U32,+width: U32,+lane: U32,+value: F32,+inside: {U32.is_lt((first + lane : U32),width) == True{} : Bool}) ->
  {Cells.tail_lane(first,width,lane,value) == value : F32}:
  %Equal.sym(Bool,U32.is_lt((first + lane : U32),width),True{},inside) : {Bool.pick(F32,_,value,0.0) == value : F32}
  {==}

# A lane of a group the width covers lies inside it.
def lane_inside(+limits: Machine.Limits,+first: U32,+width: U32,+lane: U32,+small: {Nat.is_lt(U32.to_nat(lane),8n) == True{} : Bool},
  +covered: {Nat.is_le(Nat.add(U32.to_nat(first),8n),U32.to_nat(width)) == True{} : Bool},
  +width_room: {Nat.is_le(U32.to_nat(width),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) -> {U32.is_lt((first + lane : U32),width) == True{} : Bool}:
  +below = Order.lt_le(Nat.add(U32.to_nat(first),U32.to_nat(lane)),Nat.add(U32.to_nat(first),8n),U32.to_nat(width),Order.offset_lt(U32.to_nat(first),U32.to_nat(lane),8n,small),covered)
  Equal.trans(Bool,U32.is_lt((first + lane : U32),width),Nat.is_lt(U32.to_nat((first + lane : U32)),U32.to_nat(width)),True{},Comparison.lt((first + lane : U32),width),
    {%Equal.sym(Nat,U32.to_nat((first + lane : U32)),Nat.add(U32.to_nat(first),U32.to_nat(lane)),
        Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),first,lane,U32.to_nat(first),U32.to_nat(lane),{==},{==},
          Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(lane)),U32.to_nat(width),U32.to_nat(Machine.half_word(limits)),Bounds.strict_le(Nat.add(U32.to_nat(first),U32.to_nat(lane)),U32.to_nat(width),below),
            width_room))) :
      {Nat.is_lt(_,U32.to_nat(width)) == True{} : Bool}
    below : {Nat.is_lt(U32.to_nat((first + lane : U32)),U32.to_nat(width)) == True{} : Bool}})

# A group the width covers keeps every lane.
def tail_whole(+limits: Machine.Limits,+first: U32,+width: U32,+tile: Tile.Tile8,+covered: {Nat.is_le(Nat.add(U32.to_nat(first),8n),U32.to_nat(width)) == True{} : Bool},
  +width_room: {Nat.is_le(U32.to_nat(width),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) -> {Cells.tail(Cells.Tail{first,width},tile) == tile : Tile.Tile8}:
  +parts = {Tile.Tile{Tile.lane0(tile),Tile.lane1(tile),Tile.lane2(tile),Tile.lane3(tile),Tile.lane4(tile),Tile.lane5(tile),Tile.lane6(tile),Tile.lane7(tile)} : Tile.Tile8}
  %Equal.sym(Tile.Tile8,tile,parts,TileLaws.tile_eta(tile)) : {Cells.tail(Cells.Tail{first,width},_) == tile : Tile.Tile8}
  TileLaws.tiles_equal(Cells.tail(Cells.Tail{first,width},parts),tile,
    lane_kept(first,width,0,Tile.lane0(tile),lane_inside(limits,first,width,0,{==},covered,width_room)),
    lane_kept(first,width,1,Tile.lane1(tile),lane_inside(limits,first,width,1,{==},covered,width_room)),
    lane_kept(first,width,2,Tile.lane2(tile),lane_inside(limits,first,width,2,{==},covered,width_room)),
    lane_kept(first,width,3,Tile.lane3(tile),lane_inside(limits,first,width,3,{==},covered,width_room)),
    lane_kept(first,width,4,Tile.lane4(tile),lane_inside(limits,first,width,4,{==},covered,width_room)),
    lane_kept(first,width,5,Tile.lane5(tile),lane_inside(limits,first,width,5,{==},covered,width_room)),
    lane_kept(first,width,6,Tile.lane6(tile),lane_inside(limits,first,width,6,{==},covered,width_room)),
    lane_kept(first,width,7,Tile.lane7(tile),lane_inside(limits,first,width,7,{==},covered,width_room)))

# A run cell masks its lanes past the width as the cut of its spot does.
def masked_tail(partial: Bool,+limits: Machine.Limits,+first: U32,+width: U32,+tile: Tile.Tile8,+partial_exact: {U32.is_lt(width,(first + 8 : U32)) == partial : Bool},
  +first_room: {Nat.is_le(Nat.add(U32.to_nat(first),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +width_room: {Nat.is_le(U32.to_nat(width),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Tiled.run_masked(partial,first,width,tile) == Cells.tail(Cells.Tail{first,width},tile) : Tile.Tile8}:
  match partial:
    case True{}: {==}
    case False{}:
      +outside = Equal.trans(Bool,Nat.is_lt(U32.to_nat(width),U32.to_nat((first + 8 : U32))),U32.is_lt(width,(first + 8 : U32)),False{},
        Equal.sym(Bool,U32.is_lt(width,(first + 8 : U32)),Nat.is_lt(U32.to_nat(width),U32.to_nat((first + 8 : U32))),Comparison.lt(width,(first + 8 : U32))),partial_exact)
      +covered = {%Arithmetic.sum_exact(Machine.half_word(limits),Limits.half_word_fits(limits),first,8,U32.to_nat(first),8n,{==},{==},first_room) :
          {Nat.is_le(_,U32.to_nat(width)) == True{} : Bool}
        Order.not_less(U32.to_nat(width),U32.to_nat((first + 8 : U32)),outside) : {Nat.is_le(Nat.add(U32.to_nat(first),8n),U32.to_nat(width)) == True{} : Bool}}
      Equal.sym(Tile.Tile8,Cells.tail(Cells.Tail{first,width},tile),tile,tail_whole(limits,first,width,tile,covered,width_room))

# ---- The runs of a target piece ----

def lanes_group(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+anchor: Nat,+group: Nat,+lanes_fact: SplitValue.Lanes<limits,anchors,width,lanes,anchor,group>) ->
  {U32.to_nat(U32.from_nat(group)) == group : Nat}:
  SplitValue.Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  group_exact

def lanes_cell(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+anchor: Nat,+group: Nat,+lanes_fact: SplitValue.Lanes<limits,anchors,width,lanes,anchor,group>) ->
  {U32.to_nat(((U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32) / 8 : U32)) == Nat.div(SplitValue.first_lane(anchor,width,group),8n) : Nat}:
  SplitValue.Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  cell_exact

def lanes_shift(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+anchor: Nat,+group: Nat,+lanes_fact: SplitValue.Lanes<limits,anchors,width,lanes,anchor,group>) ->
  {U32.to_nat(((U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32) % 8 : U32)) == Nat.mod(SplitValue.first_lane(anchor,width,group),8n) : Nat}:
  SplitValue.Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  shift_exact

def lanes_inside(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+anchor: Nat,+group: Nat,+lanes_fact: SplitValue.Lanes<limits,anchors,width,lanes,anchor,group>) ->
  {Nat.is_lt(Nat.div(SplitValue.first_lane(anchor,width,group),8n),U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}:
  SplitValue.Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  cell_inside

def lanes_bound(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+anchor: Nat,+group: Nat,+lanes_fact: SplitValue.Lanes<limits,anchors,width,lanes,anchor,group>) ->
  {Nat.is_le(Nat.add(Nat.mul(group,8n),7n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  SplitValue.Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  bound

# The cells from a local index to the end of a target piece.
def end_distance(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+first: U32,+local: U32,+count: U32,
  +below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat(((first + count : U32) - (first + local : U32) : U32)) == Nat.sub(U32.to_nat(count),U32.to_nat(local)) : Nat}:
  +start = U32.to_nat(first)
  +done = U32.to_nat(local)
  +total = U32.to_nat(count)
  +left = Nat.sub(total,done)
  +ordered = Order.offset_le(start,done,total,Bounds.strict_le(done,total,below))
  +idx_exact = Arithmetic.sum_exact(cap,cap_ok,first,local,start,done,{==},{==},Order.transitive(Nat.add(start,done),Nat.add(start,total),U32.to_nat(cap),ordered,room))
  +end_exact = Arithmetic.sum_exact(cap,cap_ok,first,count,start,total,{==},{==},room)
  +restored = Order.restore_shift(done,total,Bounds.strict_le(done,total,below))
  Equal.trans(Nat,U32.to_nat(((first + count : U32) - (first + local : U32) : U32)),Nat.sub(Nat.add(start,total),Nat.add(start,done)),left,
    Arithmetic.difference_exact(cap,cap_ok,(first + count : U32),(first + local : U32),Nat.add(start,total),Nat.add(start,done),end_exact,idx_exact,ordered,room),
    {%restored : {Nat.sub(Nat.add(start,_),Nat.add(start,done)) == left : Nat}
      %Addition.associative(start,done,left) : {Nat.sub(_,Nat.add(start,done)) == left : Nat}
      Order.subtract_shift(Nat.add(start,done),left) : {Nat.sub(Nat.add(start,total),Nat.add(start,done)) == left : Nat}})

# The run of a target piece from a local index with the groups innermost,
# at the spot of its first cell: its anchor, group and source row.
def inner_run(+width: U32,+lanes: U32,+source_outer: U32,+first: U32,+local: U32,+count: U32,+anchor: Nat,+group: Nat,+row: Nat) -> Tiled.Run:
  Tiled.run_spotted(width,Tiled.groups(width),(first + local : U32),(first + count : U32),SplitValue.spot_of_lanes(Grid.Innermost{},anchor,group,row,lanes,source_outer))

# The cells of a run with the groups innermost: the groups of its anchor
# from its group on, up to the end of the target piece.
def inner_count(+limits: Machine.Limits,+width: U32,+lanes: U32,+source_outer: U32,+first: U32,+local: U32,+count: U32,+anchor: Nat,+group: Nat,+row: Nat,
  +group_word: {U32.to_nat(U32.from_nat(group)) == group : Nat},+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(width))) == True{} : Bool},
  +groups_room: {Nat.is_le(U32.to_nat(Tiled.groups(width)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {U32.to_nat(Tiled.run_count(inner_run(width,lanes,source_outer,first,local,count,anchor,group,row))) ==
    Nat.min(Nat.sub(U32.to_nat(Tiled.groups(width)),group),Nat.sub(U32.to_nat(count),U32.to_nat(local))) : Nat}:
  +groups = Tiled.groups(width)
  +rest = (groups - U32.from_nat(group) : U32)
  +left = ((first + count : U32) - (first + local : U32) : U32)
  %Arithmetic.difference_exact(Machine.half_word(limits),Limits.half_word_fits(limits),groups,U32.from_nat(group),U32.to_nat(groups),group,{==},group_word,
      Bounds.strict_le(group,U32.to_nat(groups),group_inside),groups_room) :
    {U32.to_nat(U32.min(rest,left)) == Nat.min(_,Nat.sub(U32.to_nat(count),U32.to_nat(local))) : Nat}
  %end_distance(Machine.half_word(limits),Limits.half_word_fits(limits),first,local,count,below,room) :
    {U32.to_nat(U32.min(rest,left)) == Nat.min(U32.to_nat(rest),_) : Nat}
  Extrema.minimum(rest,left)

# ---- The runs with the groups innermost ----

def facts_groups_room(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>) ->
  {Nat.is_le(U32.to_nat(Tiled.groups(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}:
  SplitValue.Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,+lane_groups,+outer_room,+anchors_room,+after_room,
    +groups_room,+source_room} = facts
  groups_room

def digits_group_inside(+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+source_outer: U32,+position: Grid.Position,+index: U32,
  +digits: SplitValue.Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>) ->
  {Nat.is_lt(SplitValue.split_group(position,target_outer,width,index),U32.to_nat(Tiled.groups(width))) == True{} : Bool}:
  SplitValue.Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  group_inside

def digits_source_inside(+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+source_outer: U32,+position: Grid.Position,+index: U32,
  +digits: SplitValue.Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>) ->
  {Nat.is_lt(SplitValue.source_row(SplitValue.split_row(position,target_outer,width,index),anchors,after),U32.to_nat(source_outer)) == True{} : Bool}:
  SplitValue.Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  source_inside

# The anchor, the group and the source row of a target cell index with the
# groups innermost.
def inner_anchor(+target_outer: U32,+width: U32,+anchors: U32,+after: U32,+index: U32) -> Nat:
  SplitValue.row_anchor(SplitValue.split_row(Grid.Innermost{},target_outer,width,index),anchors,after)

def inner_group(+target_outer: U32,+width: U32,+index: U32) -> Nat:
  SplitValue.split_group(Grid.Innermost{},target_outer,width,index)

def inner_source(+target_outer: U32,+width: U32,+anchors: U32,+after: U32,+index: U32) -> Nat:
  SplitValue.source_row(SplitValue.split_row(Grid.Innermost{},target_outer,width,index),anchors,after)

# The run of a target piece from a local index, at the spot of its first cell.
def inner_of(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+local: U32,+count: U32,
  +digits: SplitValue.Digits<outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},(first + local : U32)>) ->
  {Gather.run_of(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count) ==
    inner_run(width,lanes,source_outer,first,local,count,inner_anchor(target_outer,width,anchors,after,(first + local : U32)),inner_group(target_outer,width,(first + local : U32)),
      inner_source(target_outer,width,anchors,after,(first + local : U32))) : Tiled.Run}:
  +index = (first + local : U32)
  +spot = SplitValue.spot_of_lanes(Grid.Innermost{},inner_anchor(target_outer,width,anchors,after,index),inner_group(target_outer,width,index),
    inner_source(target_outer,width,anchors,after,index),lanes,source_outer)
  %Equal.sym(Tiled.Spot,Tiled.split_spot(Grid.Innermost{},outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index),spot,
      SplitValue.split_spot_exact(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,index,digits)) :
    {Tiled.run_spotted(width,Tiled.groups(width),index,(first + count : U32),_) == Tiled.run_spotted(width,Tiled.groups(width),index,(first + count : U32),spot) : Tiled.Run}
  {==}

# The counts of a run equal to the run of a target piece are its counts.
def counts_of(+split: Tiled.Split,+first: U32,+local: U32,+count: U32,+cap: U32,+run: Tiled.Run,+run_exact: {Gather.run_of(split,first,local,count) == run : Tiled.Run},
  +positive: {Nat.is_lt(0n,U32.to_nat(Tiled.run_count(run))) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(Tiled.run_count(run))),U32.to_nat(count)) == True{} : Bool},
  +first_read: {Gather.run_source(run) == SplitValue.spot_read(Gather.split_width(split),Gather.spot_at(split,(first + local : U32))) : U32},
  +source_room: {Nat.is_le(Nat.add(U32.to_nat(Gather.run_source(run)),1n+U32.to_nat(Tiled.run_count(run))),U32.to_nat(cap)) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.add(U32.to_nat(Gather.run_cell(run)),1n+U32.to_nat(Tiled.run_count(run))),U32.to_nat(cap)) == True{} : Bool},
  +groups_room: {Nat.is_le(Nat.add(U32.to_nat(Gather.run_group(run)),U32.to_nat(Tiled.run_count(run))),U32.to_nat(cap)) == True{} : Bool},
  +source_groups_room: {Nat.is_le(U32.to_nat(Gather.split_source_groups(split)),U32.to_nat(cap)) == True{} : Bool}) ->
  Gather.RunCounts<split,first,local,count,cap>:
  +back = Equal.sym(Tiled.Run,Gather.run_of(split,first,local,count),run,run_exact)
  Gather.RunCounts{
    {%back : {Nat.is_lt(0n,U32.to_nat(Tiled.run_count(_))) == True{} : Bool} positive :
      {Nat.is_lt(0n,U32.to_nat(Gather.run_count_at(split,first,local,count))) == True{} : Bool}},
    {%back : {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(Tiled.run_count(_))),U32.to_nat(count)) == True{} : Bool} room :
      {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(Gather.run_count_at(split,first,local,count))),U32.to_nat(count)) == True{} : Bool}},
    {%back : {Gather.run_source(_) == SplitValue.spot_read(Gather.split_width(split),Gather.spot_at(split,(first + local : U32))) : U32} first_read :
      {Gather.run_source(Gather.run_of(split,first,local,count)) == SplitValue.spot_read(Gather.split_width(split),Gather.spot_at(split,(first + local : U32))) : U32}},
    {%back : {Nat.is_le(Nat.add(U32.to_nat(Gather.run_source(_)),1n+U32.to_nat(Tiled.run_count(_))),U32.to_nat(cap)) == True{} : Bool} source_room :
      {Nat.is_le(Nat.add(U32.to_nat(Gather.run_source(Gather.run_of(split,first,local,count))),1n+U32.to_nat(Gather.run_count_at(split,first,local,count))),U32.to_nat(cap)) == True{} : Bool}},
    {%back : {Nat.is_le(Nat.add(U32.to_nat(Gather.run_cell(_)),1n+U32.to_nat(Tiled.run_count(_))),U32.to_nat(cap)) == True{} : Bool} cells_room :
      {Nat.is_le(Nat.add(U32.to_nat(Gather.run_cell(Gather.run_of(split,first,local,count))),1n+U32.to_nat(Gather.run_count_at(split,first,local,count))),U32.to_nat(cap)) == True{} : Bool}},
    {%back : {Nat.is_le(Nat.add(U32.to_nat(Gather.run_group(_)),U32.to_nat(Tiled.run_count(_))),U32.to_nat(cap)) == True{} : Bool} groups_room :
      {Nat.is_le(Nat.add(U32.to_nat(Gather.run_group(Gather.run_of(split,first,local,count))),U32.to_nat(Gather.run_count_at(split,first,local,count))),U32.to_nat(cap)) == True{} : Bool}},
    source_groups_room}

# The run of a target piece from a local index with the groups innermost
# covers at least one cell and none past the piece, starts at the source
# cell of its first cell and fits the half word with its source cells.
def inner_counts(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+count: U32,
  +source_cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat(source_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool},
  +local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool}) ->
  Gather.RunCounts<split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count,Machine.half_word(limits)>:
  +index = (first + local : U32)
  +digits = SplitValue.split_digits(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,index,
    target_inside(limits,width,target_outer,first,count,cells_room,piece_room,local,below))
  +anchor = inner_anchor(target_outer,width,anchors,after,index)
  +group = inner_group(target_outer,width,index)
  +row = inner_source(target_outer,width,anchors,after,index)
  +lanes_fact = SplitValue.split_lanes_of(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,index,digits)
  +run = inner_run(width,lanes,source_outer,first,local,count,anchor,group,row)
  +half = U32.to_nat(Machine.half_word(limits))
  +capacity = U32.to_nat(Machine.capacity(limits))
  +groups = U32.to_nat(Tiled.groups(width))
  +sources = U32.to_nat(Tiled.groups(lanes))
  +outers = U32.to_nat(source_outer)
  +done = U32.to_nat(local)
  +total = U32.to_nat(count)
  +cell = Nat.div(SplitValue.first_lane(anchor,width,group),8n)
  +first_word = (U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32)
  +cell_word = (first_word / 8 : U32)
  +base = SplitValue.spot_base(Grid.Innermost{},row,sources)
  +base_word = U32.from_nat(base)
  +group_inside = digits_group_inside(outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},index,digits)
  +source_inside = digits_source_inside(outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},index,digits)
  +cell_inside = lanes_inside(limits,anchors,width,lanes,anchor,group,lanes_fact)
  +cell_exact = lanes_cell(limits,anchors,width,lanes,anchor,group,lanes_fact)
  +groups_capacity = facts_groups_room(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts)
  +sources_capacity = Order.transitive(sources,Nat.mul(sources,outers),capacity,RowMajor.product_covers(sources,outers,Order.positive_bound(row,outers,source_inside)),
    source_cells_room)
  +piece_half = Limits.within_half(limits,Nat.add(U32.to_nat(first),total),
    Order.transitive(Nat.add(U32.to_nat(first),total),Nat.mul(groups,U32.to_nat(target_outer)),capacity,piece_room,cells_room))
  +counted = inner_count(limits,width,lanes,source_outer,first,local,count,anchor,group,row,lanes_group(limits,anchors,width,lanes,anchor,group,lanes_fact),group_inside,
    Limits.within_half(limits,groups,groups_capacity),below,piece_half)
  +least = Nat.min(Nat.sub(groups,group),Nat.sub(total,done))
  +taken = U32.to_nat(Tiled.run_count(run))
  +back = Equal.sym(Nat,taken,least,counted)
  +taken_capacity = {%back : {Nat.is_le(_,capacity) == True{} : Bool}
    Order.transitive(least,Nat.sub(groups,group),capacity,NatExtrema.minimum_left(Nat.sub(groups,group),Nat.sub(total,done)),
      Order.transitive(Nat.sub(groups,group),groups,capacity,Order.sub_le(groups,group),groups_capacity)) : {Nat.is_le(taken,capacity) == True{} : Bool}}
  +positive = {%back : {Nat.is_lt(0n,_) == True{} : Bool}
    NatExtrema.minimum_strict_lower(Nat.sub(groups,group),Nat.sub(total,done),0n,Order.positive_difference(groups,group,group_inside),
      Order.positive_difference(total,done,below)) : {Nat.is_lt(0n,taken) == True{} : Bool}}
  +room = {%back : {Nat.is_le(Nat.add(done,_),total) == True{} : Bool}
    Order.transitive(Nat.add(done,least),Nat.add(done,Nat.sub(total,done)),total,
      Order.offset_le(done,least,Nat.sub(total,done),NatExtrema.minimum_right(Nat.sub(groups,group),Nat.sub(total,done))),
      {%Order.restore_shift(done,total,Bounds.strict_le(done,total,below)) : {Nat.is_le(Nat.add(done,Nat.sub(total,done)),_) == True{} : Bool}
        Order.reflexive(Nat.add(done,Nat.sub(total,done))) : {Nat.is_le(Nat.add(done,Nat.sub(total,done)),total) == True{} : Bool}}) :
    {Nat.is_le(Nat.add(done,taken),total) == True{} : Bool}}
  +cell_capacity = Order.transitive(cell,sources,capacity,Bounds.strict_le(cell,sources,cell_inside),sources_capacity)
  +cell_half = {%Equal.sym(Nat,U32.to_nat(cell_word),cell,cell_exact) : {Nat.is_le(_,half) == True{} : Bool}
    Limits.within_half(limits,cell,cell_capacity) : {Nat.is_le(U32.to_nat(cell_word),half) == True{} : Bool}}
  +first_read = {%Equal.sym(Tiled.Spot,Tiled.split_spot(Grid.Innermost{},outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index),
        SplitValue.spot_of_lanes(Grid.Innermost{},anchor,group,row,lanes,source_outer),
        SplitValue.split_spot_exact(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,index,digits)) :
      {(base_word + cell_word : U32) == SplitValue.spot_read(width,_) : U32}
    Equal.cong(U32,U32,word => (base_word + word : U32),cell_word,(cell_word * 1 : U32),Equal.sym(U32,(cell_word * 1 : U32),cell_word,times_one_word(limits,cell_word,cell_half))) :
    {(base_word + cell_word : U32) == SplitValue.spot_read(width,Tiled.split_spot(Grid.Innermost{},outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),index)) : U32}}
  +source_below = {%RowMajor.multiply_commutative(outers,sources) : {Nat.is_lt(Nat.add(base,cell),_) == True{} : Bool}
    RowMajor.radix(row,outers,cell,sources,source_inside,cell_inside) : {Nat.is_lt(Nat.add(base,cell),Nat.mul(sources,outers)) == True{} : Bool}}
  +source_capacity = Order.lt_le(Nat.add(base,cell),Nat.mul(sources,outers),capacity,source_below,source_cells_room)
  +base_exact = FullRange.roundtrip(base,Machine.capacity(limits),
    Order.transitive(base,Nat.add(base,cell),capacity,Order.le_add(base,cell),Bounds.strict_le(Nat.add(base,cell),capacity,source_capacity)))
  +source_exact = Arithmetic.sum_exact(Machine.capacity(limits),Limits.capacity_fits(limits),base_word,cell_word,base,cell,base_exact,cell_exact,
    Bounds.strict_le(Nat.add(base,cell),capacity,source_capacity))
  +source_room = {%Equal.sym(Nat,U32.to_nat((base_word + cell_word : U32)),Nat.add(base,cell),source_exact) : {Nat.is_le(Nat.add(_,1n+taken),half) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(Nat.add(base,cell),1n+taken),1n+Nat.add(Nat.add(base,cell),taken),Addition.successor_right(Nat.add(base,cell),taken)) :
      {Nat.is_le(_,half) == True{} : Bool}
    double_room(limits,1n+Nat.add(base,cell),taken,Order.strict_follow(Nat.add(base,cell),capacity,source_capacity),taken_capacity) :
    {Nat.is_le(Nat.add(U32.to_nat((base_word + cell_word : U32)),1n+taken),half) == True{} : Bool}}
  +cell_room = {%Equal.sym(Nat,U32.to_nat(cell_word),cell,cell_exact) : {Nat.is_le(Nat.add(_,1n+taken),half) == True{} : Bool}
    %Equal.sym(Nat,Nat.add(cell,1n+taken),1n+Nat.add(cell,taken),Addition.successor_right(cell,taken)) : {Nat.is_le(_,half) == True{} : Bool}
    double_room(limits,1n+cell,taken,Order.transitive(1n+cell,sources,capacity,Order.strict_follow(cell,sources,cell_inside),sources_capacity),taken_capacity) :
    {Nat.is_le(Nat.add(U32.to_nat(cell_word),1n+taken),half) == True{} : Bool}}
  +group_room = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(group)),group,lanes_group(limits,anchors,width,lanes,anchor,group,lanes_fact)) :
      {Nat.is_le(Nat.add(_,taken),half) == True{} : Bool}
    %back : {Nat.is_le(Nat.add(group,_),half) == True{} : Bool}
    Limits.within_half(limits,Nat.add(group,least),Order.transitive(Nat.add(group,least),Nat.add(group,Nat.sub(groups,group)),capacity,
      Order.offset_le(group,least,Nat.sub(groups,group),NatExtrema.minimum_left(Nat.sub(groups,group),Nat.sub(total,done))),
      {%Equal.sym(Nat,Nat.add(group,Nat.sub(groups,group)),groups,Order.restore_shift(group,groups,Bounds.strict_le(group,groups,group_inside))) :
          {Nat.is_le(_,capacity) == True{} : Bool}
        groups_capacity : {Nat.is_le(Nat.add(group,Nat.sub(groups,group)),capacity) == True{} : Bool}})) :
    {Nat.is_le(Nat.add(U32.to_nat(U32.from_nat(group)),taken),half) == True{} : Bool}}
  counts_of(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count,Machine.half_word(limits),run,
    inner_of(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,local,count,digits),
    positive,room,first_read,source_room,cell_room,group_room,Limits.within_half(limits,sources,sources_capacity))

# Every run of a target piece with the groups innermost.
def run_counts(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+count: U32,
  +source_cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat(source_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool}) ->
  Gather.RunsCounted(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,count,Machine.half_word(limits)):
  local => below =>
    inner_counts(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,count,source_cells_room,cells_room,piece_room,
      local,below)

# ---- The cells of a run with the groups innermost ----

def digits_anchor_inside(+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+source_outer: U32,+position: Grid.Position,+index: U32,
  +digits: SplitValue.Digits<outer,anchors,width,after,lanes,target_outer,source_outer,position,index>) ->
  {Nat.is_lt(SplitValue.row_anchor(SplitValue.split_row(position,target_outer,width,index),anchors,after),U32.to_nat(anchors)) == True{} : Bool}:
  SplitValue.Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  anchor_inside

def lanes_eights(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+anchor: Nat,+group: Nat,+lanes_fact: SplitValue.Lanes<limits,anchors,width,lanes,anchor,group>) ->
  {U32.to_nat((U32.from_nat(group) * 8 : U32)) == Nat.mul(group,8n) : Nat}:
  SplitValue.Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  eights_exact

def lanes_covered(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+anchor: Nat,+group: Nat,+lanes_fact: SplitValue.Lanes<limits,anchors,width,lanes,anchor,group>) ->
  {Nat.is_le(Nat.mul(U32.to_nat(anchors),U32.to_nat(width)),Nat.mul(U32.to_nat(Tiled.groups(lanes)),8n)) == True{} : Bool}:
  SplitValue.Lanes{+group_exact,+eights_exact,+first_exact,+cell_exact,+shift_exact,+next_exact,+cell_inside,+bound,+covered} = lanes_fact
  covered

def words_read(+width: U32,+lanes: U32,+source_outer: U32,+position: Grid.Position,+spot: Tiled.Spot,+outer_index: Nat,+cell: Nat,
  +words: SplitValue.Words<width,lanes,source_outer,position,spot,outer_index,cell>) ->
  {U32.to_nat(SplitValue.spot_read(width,spot)) == Representation.index_at(position,U32.to_nat(source_outer),U32.to_nat(Tiled.groups(lanes)),outer_index,cell) : Nat}:
  SplitValue.Words{+read_exact,+outer_inside,+cell_inside,+next} = words
  read_exact

def words_next(+width: U32,+lanes: U32,+source_outer: U32,+position: Grid.Position,+spot: Tiled.Spot,+outer_index: Nat,+cell: Nat,
  +words: SplitValue.Words<width,lanes,source_outer,position,spot,outer_index,cell>) ->
  Logic.When(SplitValue.spot_needed(width,spot),SplitValue.Next<width,lanes,source_outer,position,spot,outer_index,cell>):
  SplitValue.Words{+read_exact,+outer_inside,+cell_inside,+next} = words
  next

def next_high(+width: U32,+lanes: U32,+source_outer: U32,+position: Grid.Position,+spot: Tiled.Spot,+outer_index: Nat,+cell: Nat,
  +next: SplitValue.Next<width,lanes,source_outer,position,spot,outer_index,cell>) ->
  {U32.to_nat(SplitValue.spot_high(width,spot)) == Representation.index_at(position,U32.to_nat(source_outer),U32.to_nat(Tiled.groups(lanes)),outer_index,1n+cell) : Nat}:
  SplitValue.Next{+further,+next_inside,+high_exact} = next
  high_exact

def counts_room(+split: Tiled.Split,+first: U32,+local: U32,+count: U32,+cap: U32,+counts: Gather.RunCounts<split,first,local,count,cap>) ->
  {Nat.is_le(Nat.add(U32.to_nat(local),U32.to_nat(Gather.run_count_at(split,first,local,count))),U32.to_nat(count)) == True{} : Bool}:
  Gather.RunCounts{+positive,+room,+first_read,+source_room,+cells_room,+groups_room,+source_groups_room} = counts
  room

# A step of a run below its count lies below the count of the piece.
def step_below(+done: Nat,+step: Nat,+number: Nat,+total: Nat,+taken: Nat,+number_exact: {number == Nat.add(done,step) : Nat},
  +inside: {Nat.is_lt(step,taken) == True{} : Bool},+room: {Nat.is_le(Nat.add(done,taken),total) == True{} : Bool}) -> {Nat.is_lt(number,total) == True{} : Bool}:
  %Equal.sym(Nat,number,Nat.add(done,step),number_exact) : {Nat.is_lt(_,total) == True{} : Bool}
  Order.lt_le(Nat.add(done,step),Nat.add(done,taken),total,Order.offset_lt(done,step,taken,inside),room)

# A step of a run below its count stays inside the groups.
def step_group(+groups: Nat,+group: Nat,+step: Nat,+taken: Nat,+left: Nat,+taken_exact: {taken == Nat.min(Nat.sub(groups,group),left) : Nat},
  +inside: {Nat.is_lt(step,taken) == True{} : Bool},+group_inside: {Nat.is_lt(group,groups) == True{} : Bool}) -> {Nat.is_lt(Nat.add(group,step),groups) == True{} : Bool}:
  +below_least = {%taken_exact : {Nat.is_lt(step,_) == True{} : Bool} inside : {Nat.is_lt(step,Nat.min(Nat.sub(groups,group),left)) == True{} : Bool}}
  +below_rest = Order.lt_le(step,Nat.min(Nat.sub(groups,group),left),Nat.sub(groups,group),below_least,NatExtrema.minimum_left(Nat.sub(groups,group),left))
  %Order.restore_shift(group,groups,Bounds.strict_le(group,groups,group_inside)) : {Nat.is_lt(Nat.add(group,step),_) == True{} : Bool}
  Order.offset_lt(group,step,Nat.sub(groups,group),below_rest)

# The target index of a step of a run: as many cells after its first.
def target_step(+cap: U32,+cap_ok: {U32.is_le(cap,2147483648) == True{} : Bool},+first: U32,+local: U32,+count: U32,+index: U32,+step: Nat,
  +index_exact: {U32.to_nat(index) == Nat.add(U32.to_nat(local),step) : Nat},+index_below: {Nat.is_lt(U32.to_nat(index),U32.to_nat(count)) == True{} : Bool},
  +below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},+room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),U32.to_nat(cap)) == True{} : Bool}) ->
  {U32.to_nat((first + index : U32)) == Nat.add(U32.to_nat((first + local : U32)),step) : Nat}:
  +start = U32.to_nat(first)
  +done = U32.to_nat(local)
  +total = U32.to_nat(count)
  +number = U32.to_nat(index)
  +start_exact = Arithmetic.sum_exact(cap,cap_ok,first,local,start,done,{==},{==},
    Order.transitive(Nat.add(start,done),Nat.add(start,total),U32.to_nat(cap),Order.offset_le(start,done,total,Bounds.strict_le(done,total,below)),room))
  +target_exact = Arithmetic.sum_exact(cap,cap_ok,first,index,start,number,{==},{==},
    Order.transitive(Nat.add(start,number),Nat.add(start,total),U32.to_nat(cap),Order.offset_le(start,number,total,Bounds.strict_le(number,total,index_below)),room))
  %Equal.sym(Nat,U32.to_nat((first + local : U32)),Nat.add(start,done),start_exact) : {U32.to_nat((first + index : U32)) == Nat.add(_,step) : Nat}
  %Equal.sym(Nat,Nat.add(Nat.add(start,done),step),Nat.add(start,Nat.add(done,step)),Addition.associative(start,done,step)) : {U32.to_nat((first + index : U32)) == _ : Nat}
  Equal.trans(Nat,U32.to_nat((first + index : U32)),Nat.add(start,number),Nat.add(start,Nat.add(done,step)),target_exact,
    Equal.cong(Nat,Nat,value => Nat.add(start,value),number,Nat.add(done,step),index_exact))

# The spot of a step of a run: the anchor and the source row of its first
# cell, its group as many groups later.
def step_spot(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+local: U32,
  +digits: SplitValue.Digits<outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},(first + local : U32)>,+step: Nat,+target: U32,
  +target_exact: {U32.to_nat(target) == Nat.add(U32.to_nat((first + local : U32)),step) : Nat},
  +moved_inside: {Nat.is_lt(Nat.add(inner_group(target_outer,width,(first + local : U32)),step),U32.to_nat(Tiled.groups(width))) == True{} : Bool}) ->
  {Tiled.split_spot(Grid.Innermost{},outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),target) ==
    SplitValue.spot_of_lanes(Grid.Innermost{},inner_anchor(target_outer,width,anchors,after,(first + local : U32)),Nat.add(inner_group(target_outer,width,(first + local : U32)),step),
      inner_source(target_outer,width,anchors,after,(first + local : U32)),lanes,source_outer) : Tiled.Spot}:
  SplitValue.Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,+lane_groups,+outer_room,+anchors_room,+after_room,
    +groups_room,+source_room} = facts
  SplitValue.Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  +index = (first + local : U32)
  +row = SplitValue.split_row(Grid.Innermost{},target_outer,width,index)
  +above = SplitValue.row_above(row,anchors,after)
  +anchor = SplitValue.row_anchor(row,anchors,after)
  +below = SplitValue.row_below(row,after)
  +group = SplitValue.split_group(Grid.Innermost{},target_outer,width,index)
  +groups = U32.to_nat(Tiled.groups(width))
  +at = Nat.add(Nat.mul(Nat.add(Nat.mul(above,U32.to_nat(anchors)),anchor),U32.to_nat(after)),below)
  +placed_step = {%Equal.sym(Nat,U32.to_nat(target),Nat.add(U32.to_nat(index),step),target_exact) : {_ == Nat.add(Nat.mul(at,groups),Nat.add(group,step)) : Nat}
    %Equal.sym(Nat,U32.to_nat(index),Nat.add(Nat.mul(at,groups),group),placed) : {Nat.add(_,step) == Nat.add(Nat.mul(at,groups),Nat.add(group,step)) : Nat}
    Addition.associative(Nat.mul(at,groups),group,step) : {U32.to_nat(target) == Nat.add(Nat.mul(at,groups),Nat.add(group,step)) : Nat}}
  SplitValue.spot_of(Grid.Innermost{},limits,outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),U32.to_nat(source_outer),target,above,anchor,below,Nat.add(group,step),
    above_inside,anchor_inside,below_inside,moved_inside,placed_step,source_outer_exact,outer_room,anchors_room,after_room,groups_room,source_positive,source_inside,source_room)

# The lane words of a step of a run.
def step_lanes(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+local: U32,
  +digits: SplitValue.Digits<outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},(first + local : U32)>,+step: Nat,
  +moved_inside: {Nat.is_lt(Nat.add(inner_group(target_outer,width,(first + local : U32)),step),U32.to_nat(Tiled.groups(width))) == True{} : Bool}) ->
  SplitValue.Lanes<limits,anchors,width,lanes,inner_anchor(target_outer,width,anchors,after,(first + local : U32)),Nat.add(inner_group(target_outer,width,(first + local : U32)),step)>:
  SplitValue.Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,+lane_groups,+outer_room,+anchors_room,+after_room,
    +groups_room,+source_room} = facts
  SplitValue.Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  SplitValue.spot_lanes(limits,anchors,width,lanes,inner_anchor(target_outer,width,anchors,after,(first + local : U32)),
    Nat.add(inner_group(target_outer,width,(first + local : U32)),step),width_groups,lane_groups,lanes_exact,anchors_room,groups_room,anchor_inside,moved_inside)

# The words of the spot of a step of a run.
def step_words(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+local: U32,
  +digits: SplitValue.Digits<outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},(first + local : U32)>,+step: Nat,
  +moved_lanes: SplitValue.Lanes<limits,anchors,width,lanes,inner_anchor(target_outer,width,anchors,after,(first + local : U32)),
    Nat.add(inner_group(target_outer,width,(first + local : U32)),step)>) ->
  SplitValue.Words<width,lanes,source_outer,Grid.Innermost{},
    SplitValue.spot_of_lanes(Grid.Innermost{},inner_anchor(target_outer,width,anchors,after,(first + local : U32)),Nat.add(inner_group(target_outer,width,(first + local : U32)),step),
      inner_source(target_outer,width,anchors,after,(first + local : U32)),lanes,source_outer),
    inner_source(target_outer,width,anchors,after,(first + local : U32)),
    Nat.div(SplitValue.first_lane(inner_anchor(target_outer,width,anchors,after,(first + local : U32)),width,Nat.add(inner_group(target_outer,width,(first + local : U32)),step)),8n)>:
  SplitValue.Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,+lane_groups,+outer_room,+anchors_room,+after_room,
    +groups_room,+source_room} = facts
  SplitValue.Digits{+above_inside,+anchor_inside,+below_inside,+group_inside,+above_exact,+placed,+source_inside,+source_positive} = digits
  SplitValue.spot_words(Grid.Innermost{},limits,anchors,width,lanes,source_outer,inner_anchor(target_outer,width,anchors,after,(first + local : U32)),
    Nat.add(inner_group(target_outer,width,(first + local : U32)),step),inner_source(target_outer,width,anchors,after,(first + local : U32)),moved_lanes,anchor_inside,
    source_inside,source_room)

# The first lane of a group inside the groups and the width fit the half
# word with a whole group.
def group_room(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+anchor: Nat,+group: Nat,
  +lanes_fact: SplitValue.Lanes<limits,anchors,width,lanes,anchor,group>,+group_inside: {Nat.is_lt(group,U32.to_nat(Tiled.groups(width))) == True{} : Bool}) ->
  {Nat.is_le(Nat.add(U32.to_nat((U32.from_nat(group) * 8 : U32)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  SplitValue.Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,TiledValues.Groups{+width_exact,+width_fewer,+width_fit},
    +lane_groups,+outer_room,+anchors_room,+after_room,+groups_room,+source_room} = facts
  +half = U32.to_nat(Machine.half_word(limits))
  %Equal.sym(Nat,U32.to_nat((U32.from_nat(group) * 8 : U32)),Nat.mul(group,8n),lanes_eights(limits,anchors,width,lanes,anchor,group,lanes_fact)) :
    {Nat.is_le(Nat.add(_,8n),half) == True{} : Bool}
  %Order.commutative(8n,Nat.mul(group,8n)) : {Nat.is_le(_,half) == True{} : Bool}
  Order.transitive(Nat.mul(1n+group,8n),Nat.mul(U32.to_nat(Tiled.groups(width)),8n),half,
    RowMajor.mul_le(1n+group,U32.to_nat(Tiled.groups(width)),8n,Order.strict_follow(group,U32.to_nat(Tiled.groups(width)),group_inside)),width_fit)

def width_room(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+anchor: Nat,+group: Nat,
  +lanes_fact: SplitValue.Lanes<limits,anchors,width,lanes,anchor,group>,+anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool}) ->
  {Nat.is_le(U32.to_nat(width),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  SplitValue.Facts{+target_outer_exact,+target_after_exact,+source_outer_exact,+source_after_exact,+lanes_exact,+width_groups,TiledValues.Groups{+lane_exact,+lane_fewer,+lane_fit},
    +outer_room,+anchors_room,+after_room,+groups_room,+source_room} = facts
  +count = U32.to_nat(anchors)
  +span = U32.to_nat(width)
  +half = U32.to_nat(Machine.half_word(limits))
  Order.transitive(span,Nat.mul(count,span),half,
    {%RowMajor.multiply_commutative(span,count) : {Nat.is_le(span,_) == True{} : Bool}
      RowMajor.product_covers(span,count,Order.positive_bound(anchor,count,anchor_inside)) : {Nat.is_le(span,Nat.mul(count,span)) == True{} : Bool}},
    Order.transitive(Nat.mul(count,span),Nat.mul(U32.to_nat(Tiled.groups(lanes)),8n),half,lanes_covered(limits,anchors,width,lanes,anchor,group,lanes_fact),lane_fit))

# The count of the run of a target piece from a local index.
def run_count_exact(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+count: U32,
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool},
  +local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool}) ->
  {U32.to_nat(Gather.run_count_at(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count)) ==
    Nat.min(Nat.sub(U32.to_nat(Tiled.groups(width)),inner_group(target_outer,width,(first + local : U32))),Nat.sub(U32.to_nat(count),U32.to_nat(local))) : Nat}:
  +index = (first + local : U32)
  +digits = SplitValue.split_digits(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,index,
    target_inside(limits,width,target_outer,first,count,cells_room,piece_room,local,below))
  +anchor = inner_anchor(target_outer,width,anchors,after,index)
  +group = inner_group(target_outer,width,index)
  +row = inner_source(target_outer,width,anchors,after,index)
  +lanes_fact = SplitValue.split_lanes_of(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,index,digits)
  +run = inner_run(width,lanes,source_outer,first,local,count,anchor,group,row)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +piece_half = Limits.within_half(limits,Nat.add(U32.to_nat(first),U32.to_nat(count)),
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),capacity,piece_room,cells_room))
  %Equal.sym(Tiled.Run,Gather.run_of(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count),run,
      inner_of(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,local,count,digits)) :
    {U32.to_nat(Tiled.run_count(_)) == Nat.min(Nat.sub(U32.to_nat(Tiled.groups(width)),group),Nat.sub(U32.to_nat(count),U32.to_nat(local))) : Nat}
  inner_count(limits,width,lanes,source_outer,first,local,count,anchor,group,row,lanes_group(limits,anchors,width,lanes,anchor,group,lanes_fact),
    digits_group_inside(outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},index,digits),
    Limits.within_half(limits,U32.to_nat(Tiled.groups(width)),facts_groups_room(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts)),
    below,piece_half)

# The source cell of the run of a target piece from a local index.
def run_source_exact(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+count: U32,
  +source_cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat(source_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool},
  +local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool}) ->
  {U32.to_nat(Gather.run_source(Gather.run_of(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count))) ==
    Nat.add(SplitValue.spot_base(Grid.Innermost{},inner_source(target_outer,width,anchors,after,(first + local : U32)),U32.to_nat(Tiled.groups(lanes))),
      Nat.div(SplitValue.first_lane(inner_anchor(target_outer,width,anchors,after,(first + local : U32)),width,inner_group(target_outer,width,(first + local : U32))),8n)) : Nat}:
  +index = (first + local : U32)
  +digits = SplitValue.split_digits(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,index,
    target_inside(limits,width,target_outer,first,count,cells_room,piece_room,local,below))
  +anchor = inner_anchor(target_outer,width,anchors,after,index)
  +group = inner_group(target_outer,width,index)
  +row = inner_source(target_outer,width,anchors,after,index)
  +lanes_fact = SplitValue.split_lanes_of(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,index,digits)
  +run = inner_run(width,lanes,source_outer,first,local,count,anchor,group,row)
  +capacity = U32.to_nat(Machine.capacity(limits))
  +sources = U32.to_nat(Tiled.groups(lanes))
  +outers = U32.to_nat(source_outer)
  +cell = Nat.div(SplitValue.first_lane(anchor,width,group),8n)
  +cell_word = ((U32.from_nat(anchor) * width + U32.from_nat(group) * 8 : U32) / 8 : U32)
  +base = SplitValue.spot_base(Grid.Innermost{},row,sources)
  +source_inside = digits_source_inside(outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},index,digits)
  +source_below = {%RowMajor.multiply_commutative(outers,sources) : {Nat.is_lt(Nat.add(base,cell),_) == True{} : Bool}
    RowMajor.radix(row,outers,cell,sources,source_inside,lanes_inside(limits,anchors,width,lanes,anchor,group,lanes_fact)) :
    {Nat.is_lt(Nat.add(base,cell),Nat.mul(sources,outers)) == True{} : Bool}}
  +source_room = Bounds.strict_le(Nat.add(base,cell),capacity,Order.lt_le(Nat.add(base,cell),Nat.mul(sources,outers),capacity,source_below,source_cells_room))
  +base_exact = FullRange.roundtrip(base,Machine.capacity(limits),Order.transitive(base,Nat.add(base,cell),capacity,Order.le_add(base,cell),source_room))
  %Equal.sym(Tiled.Run,Gather.run_of(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count),run,
      inner_of(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,local,count,digits)) :
    {U32.to_nat(Gather.run_source(_)) == Nat.add(base,cell) : Nat}
  Arithmetic.sum_exact(Machine.capacity(limits),Limits.capacity_fits(limits),U32.from_nat(base),cell_word,base,cell,base_exact,
    lanes_cell(limits,anchors,width,lanes,anchor,group,lanes_fact),source_room)

# The next source cell of a step of a run that a lane needs holds the source
# cell after the one the run reads at the step.
def moved_high(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+width: U32,+lanes: U32,+source_outer: U32,+source_after: U32,+spot: Tiled.Spot,
  +moved_spot: Tiled.Spot,+row: Nat,+cell: Nat,+source: Nat,+cell_first: Nat,+step: Nat,+spot_exact: {spot == moved_spot : Tiled.Spot},
  +words: SplitValue.Words<width,lanes,source_outer,Grid.Innermost{},moved_spot,row,cell>,+cell_exact: {cell == Nat.add(cell_first,step) : Nat},
  +source_exact: {source == Nat.add(SplitValue.spot_base(Grid.Innermost{},row,U32.to_nat(Tiled.groups(lanes))),cell_first) : Nat},+high: Tile.Tile8,
  +high_exact: Logic.When(SplitValue.spot_needed(width,spot),{high == Representation.lifted(~Logical,~original,
    Representation.Lifted{logical,source_cells(source_outer,source_after,lanes,Grid.Innermost{})},U32.from_nat(1n+Nat.add(source,step))) : Tile.Tile8}),
  +need: {SplitValue.spot_needed(width,spot) == True{} : Bool}) ->
  {high == Representation.lifted(~Logical,~original,Representation.Lifted{logical,source_cells(source_outer,source_after,lanes,Grid.Innermost{})},
    SplitValue.spot_high(width,spot)) : Tile.Tile8}:
  +context = {Representation.Lifted{logical,source_cells(source_outer,source_after,lanes,Grid.Innermost{})} : Representation.Lifted<Logical>}
  +base = SplitValue.spot_base(Grid.Innermost{},row,U32.to_nat(Tiled.groups(lanes)))
  +given = Logic.required({high == Representation.lifted(~Logical,~original,context,U32.from_nat(1n+Nat.add(source,step))) : Tile.Tile8},SplitValue.spot_needed(width,spot),
    high_exact,need)
  +moved_need = {%spot_exact : {SplitValue.spot_needed(width,_) == True{} : Bool} need : {SplitValue.spot_needed(width,moved_spot) == True{} : Bool}}
  +next = Logic.required(SplitValue.Next<width,lanes,source_outer,Grid.Innermost{},moved_spot,row,cell>,SplitValue.spot_needed(width,moved_spot),
    words_next(width,lanes,source_outer,Grid.Innermost{},moved_spot,row,cell,words),moved_need)
  +high_moved = {%Equal.sym(Tiled.Spot,spot,moved_spot,spot_exact) : {U32.to_nat(SplitValue.spot_high(width,_)) == 1n+Nat.add(source,step) : Nat}
    %Equal.sym(Nat,source,Nat.add(base,cell_first),source_exact) : {U32.to_nat(SplitValue.spot_high(width,moved_spot)) == 1n+Nat.add(_,step) : Nat}
    %Equal.sym(Nat,Nat.add(Nat.add(base,cell_first),step),Nat.add(base,Nat.add(cell_first,step)),Addition.associative(base,cell_first,step)) :
      {U32.to_nat(SplitValue.spot_high(width,moved_spot)) == 1n+_ : Nat}
    %Addition.successor_right(base,Nat.add(cell_first,step)) : {U32.to_nat(SplitValue.spot_high(width,moved_spot)) == _ : Nat}
    %cell_exact : {U32.to_nat(SplitValue.spot_high(width,moved_spot)) == Nat.add(base,1n+_) : Nat}
    next_high(width,lanes,source_outer,Grid.Innermost{},moved_spot,row,cell,next) : {U32.to_nat(SplitValue.spot_high(width,spot)) == 1n+Nat.add(source,step) : Nat}}
  Equal.trans(Tile.Tile8,high,Representation.lifted(~Logical,~original,context,U32.from_nat(1n+Nat.add(source,step))),
    Representation.lifted(~Logical,~original,context,SplitValue.spot_high(width,spot)),given,
    Equal.cong(U32,Tile.Tile8,word => Representation.lifted(~Logical,~original,context,word),U32.from_nat(1n+Nat.add(source,step)),SplitValue.spot_high(width,spot),
      Arithmetic.from_nat_exact(SplitValue.spot_high(width,spot),1n+Nat.add(source,step),high_moved)))

# A run cell masks a tile at a group.
def masked_cell(+width: U32,+group: U32,+tile: Tile.Tile8) -> Tile.Tile8:
  Tiled.run_masked(U32.is_lt(width,((group * 8 : U32) + 8 : U32)),(group * 8 : U32),width,tile)

# A cell of a run with the groups innermost: the source cells at its step
# joined at the shift of the run and masked give the target value.
def run_law_at(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,
  +target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,
  +facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+count: U32,
  +source_cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat(source_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool},
  +local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},+step: Nat,
  +step_inside: {Nat.is_lt(step,U32.to_nat(Gather.run_count_at(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count))) == True{} : Bool},
  +index: U32,+index_exact: {U32.to_nat(index) == Nat.add(U32.to_nat(local),step) : Nat},+group: U32,
  +group_exact: {U32.to_nat(group) == Nat.add(U32.to_nat(Gather.run_group(Gather.run_of(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count))),step) : Nat},
  +low: Tile.Tile8,+high: Tile.Tile8,
  +low_exact: {low == Representation.lifted(~Logical,~original,Representation.Lifted{logical,source_cells(source_outer,source_after,lanes,Grid.Innermost{})},
    U32.from_nat(Nat.add(U32.to_nat(Gather.run_source(Gather.run_of(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count))),step))) : Tile.Tile8},
  +high_exact: Logic.When(SplitValue.spot_needed(width,Tiled.split_spot(Grid.Innermost{},outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),(first + index : U32))),
    {high == Representation.lifted(~Logical,~original,Representation.Lifted{logical,source_cells(source_outer,source_after,lanes,Grid.Innermost{})},
      U32.from_nat(1n+Nat.add(U32.to_nat(Gather.run_source(Gather.run_of(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count))),step))) : Tile.Tile8})) ->
  {Tiled.run_masked(U32.is_lt(width,((group * 8 : U32) + 8 : U32)),(group * 8 : U32),width,
    Tiled.joined(Gather.run_shift(Gather.run_of(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count)),low,high)) ==
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,target_cells(target_outer,target_after,width,Grid.Innermost{})},(first + index : U32)) : Tile.Tile8}:
  +split = split_of(outer,anchors,width,after,lanes,Grid.Innermost{})
  +start = (first + local : U32)
  +target = (first + index : U32)
  +digits = SplitValue.split_digits(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,start,
    target_inside(limits,width,target_outer,first,count,cells_room,piece_room,local,below))
  +anchor = inner_anchor(target_outer,width,anchors,after,start)
  +group_first = inner_group(target_outer,width,start)
  +row = inner_source(target_outer,width,anchors,after,start)
  +moved = Nat.add(group_first,step)
  +lanes_first = SplitValue.split_lanes_of(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,start,digits)
  +run = inner_run(width,lanes,source_outer,first,local,count,anchor,group_first,row)
  +back = Equal.sym(Tiled.Run,Gather.run_of(split,first,local,count),run,
    inner_of(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,local,count,digits))
  +capacity = U32.to_nat(Machine.capacity(limits))
  +counts = inner_counts(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,count,source_cells_room,cells_room,piece_room,
    local,below)
  +index_below = step_below(U32.to_nat(local),step,U32.to_nat(index),U32.to_nat(count),U32.to_nat(Gather.run_count_at(split,first,local,count)),index_exact,step_inside,
    counts_room(split,first,local,count,Machine.half_word(limits),counts))
  +moved_inside = step_group(U32.to_nat(Tiled.groups(width)),group_first,step,U32.to_nat(Gather.run_count_at(split,first,local,count)),
    Nat.sub(U32.to_nat(count),U32.to_nat(local)),
    run_count_exact(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,count,cells_room,piece_room,local,below),step_inside,
    digits_group_inside(outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},start,digits))
  +target_exact = target_step(Machine.capacity(limits),Limits.capacity_fits(limits),first,local,count,index,step,index_exact,index_below,below,
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),capacity,piece_room,cells_room))
  +spot_target = Tiled.split_spot(Grid.Innermost{},outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),target)
  +spot_moved = SplitValue.spot_of_lanes(Grid.Innermost{},anchor,moved,row,lanes,source_outer)
  +spot_exact = step_spot(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,local,digits,step,target,target_exact,
    moved_inside)
  +lanes_moved = step_lanes(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,local,digits,step,moved_inside)
  +words_moved = step_words(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,local,digits,step,lanes_moved)
  +cell_first = Nat.div(SplitValue.first_lane(anchor,width,group_first),8n)
  +cell_moved = Nat.div(SplitValue.first_lane(anchor,width,moved),8n)
  +base = SplitValue.spot_base(Grid.Innermost{},row,U32.to_nat(Tiled.groups(lanes)))
  +source = U32.to_nat(Gather.run_source(Gather.run_of(split,first,local,count)))
  +source_exact = run_source_exact(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,count,source_cells_room,cells_room,
    piece_room,local,below)
  +context = {Representation.Lifted{logical,source_cells(source_outer,source_after,lanes,Grid.Innermost{})} : Representation.Lifted<Logical>}
  +read_moved = {%Equal.sym(Tiled.Spot,spot_target,spot_moved,spot_exact) : {U32.to_nat(SplitValue.spot_read(width,_)) == Nat.add(source,step) : Nat}
    %Equal.sym(Nat,source,Nat.add(base,cell_first),source_exact) : {U32.to_nat(SplitValue.spot_read(width,spot_moved)) == Nat.add(_,step) : Nat}
    %Equal.sym(Nat,Nat.add(Nat.add(base,cell_first),step),Nat.add(base,Nat.add(cell_first,step)),Addition.associative(base,cell_first,step)) :
      {U32.to_nat(SplitValue.spot_read(width,spot_moved)) == _ : Nat}
    %stepped_cell(anchor,width,group_first,step) : {U32.to_nat(SplitValue.spot_read(width,spot_moved)) == Nat.add(base,_) : Nat}
    words_read(width,lanes,source_outer,Grid.Innermost{},spot_moved,row,cell_moved,words_moved) :
    {U32.to_nat(SplitValue.spot_read(width,spot_target)) == Nat.add(source,step) : Nat}}
  +low_read = Equal.trans(Tile.Tile8,low,Representation.lifted(~Logical,~original,context,U32.from_nat(Nat.add(source,step))),
    Representation.lifted(~Logical,~original,context,SplitValue.spot_read(width,spot_target)),low_exact,
    Equal.cong(U32,Tile.Tile8,word => Representation.lifted(~Logical,~original,context,word),U32.from_nat(Nat.add(source,step)),SplitValue.spot_read(width,spot_target),
      Arithmetic.from_nat_exact(SplitValue.spot_read(width,spot_target),Nat.add(source,step),read_moved)))
  +high_read = Logic.when_enabled({high == Representation.lifted(~Logical,~original,context,SplitValue.spot_high(width,spot_target)) : Tile.Tile8},
    SplitValue.spot_needed(width,spot_target),
    need => moved_high(~Logical,~original,logical,width,lanes,source_outer,source_after,spot_target,spot_moved,row,cell_moved,source,cell_first,step,spot_exact,words_moved,
      stepped_cell(anchor,width,group_first,step),source_exact,high,high_exact,need))
  +group_nat = Equal.trans(Nat,U32.to_nat(group),Nat.add(U32.to_nat(Gather.run_group(Gather.run_of(split,first,local,count))),step),moved,group_exact,
    Equal.cong(Nat,Nat,value => Nat.add(value,step),U32.to_nat(Gather.run_group(Gather.run_of(split,first,local,count))),group_first,
      {%back : {U32.to_nat(Gather.run_group(_)) == group_first : Nat} lanes_group(limits,anchors,width,lanes,anchor,group_first,lanes_first) :
        {U32.to_nat(Gather.run_group(Gather.run_of(split,first,local,count))) == group_first : Nat}}))
  +group_word = word_same(group,U32.from_nat(moved),Equal.trans(Nat,U32.to_nat(group),moved,U32.to_nat(U32.from_nat(moved)),group_nat,
    Equal.sym(Nat,U32.to_nat(U32.from_nat(moved)),moved,lanes_group(limits,anchors,width,lanes,anchor,moved,lanes_moved))))
  +first_word = (U32.from_nat(anchor) * width + U32.from_nat(group_first) * 8 : U32)
  +moved_word = (U32.from_nat(anchor) * width + U32.from_nat(moved) * 8 : U32)
  +shift_word = word_same((first_word % 8 : U32),(moved_word % 8 : U32),
    Equal.trans(Nat,U32.to_nat((first_word % 8 : U32)),Nat.mod(SplitValue.first_lane(anchor,width,group_first),8n),U32.to_nat((moved_word % 8 : U32)),
      lanes_shift(limits,anchors,width,lanes,anchor,group_first,lanes_first),
      Equal.trans(Nat,Nat.mod(SplitValue.first_lane(anchor,width,group_first),8n),Nat.mod(SplitValue.first_lane(anchor,width,moved),8n),U32.to_nat((moved_word % 8 : U32)),
        Equal.sym(Nat,Nat.mod(SplitValue.first_lane(anchor,width,moved),8n),Nat.mod(SplitValue.first_lane(anchor,width,group_first),8n),stepped_shift(anchor,width,group_first,step)),
        Equal.sym(Nat,U32.to_nat((moved_word % 8 : U32)),Nat.mod(SplitValue.first_lane(anchor,width,moved),8n),lanes_shift(limits,anchors,width,lanes,anchor,moved,lanes_moved)))))
  +anchor_inside = digits_anchor_inside(outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},start,digits)
  +masked = {%Equal.sym(U32,group,U32.from_nat(moved),group_word) :
      {masked_cell(width,_,Tiled.joined(Gather.run_shift(Gather.run_of(split,first,local,count)),low,high)) == SplitValue.spot_cell(width,spot_target,low,high) : Tile.Tile8}
    %back : {masked_cell(width,U32.from_nat(moved),Tiled.joined(Gather.run_shift(_),low,high)) == SplitValue.spot_cell(width,spot_target,low,high) : Tile.Tile8}
    %Equal.sym(U32,(first_word % 8 : U32),(moved_word % 8 : U32),shift_word) :
      {masked_cell(width,U32.from_nat(moved),Tiled.joined(_,low,high)) == SplitValue.spot_cell(width,spot_target,low,high) : Tile.Tile8}
    %Equal.sym(Tiled.Spot,spot_target,spot_moved,spot_exact) :
      {masked_cell(width,U32.from_nat(moved),Tiled.joined((moved_word % 8 : U32),low,high)) == SplitValue.spot_cell(width,_,low,high) : Tile.Tile8}
    masked_tail(U32.is_lt(width,((U32.from_nat(moved) * 8 : U32) + 8 : U32)),limits,(U32.from_nat(moved) * 8 : U32),width,Tiled.joined((moved_word % 8 : U32),low,high),{==},
      group_room(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,anchor,moved,lanes_moved,moved_inside),
      width_room(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,anchor,moved,lanes_moved,anchor_inside)) :
    {Tiled.run_masked(U32.is_lt(width,((group * 8 : U32) + 8 : U32)),(group * 8 : U32),width,Tiled.joined(Gather.run_shift(Gather.run_of(split,first,local,count)),low,high)) ==
      SplitValue.spot_cell(width,spot_target,low,high) : Tile.Tile8}}
  Equal.trans(Tile.Tile8,
    Tiled.run_masked(U32.is_lt(width,((group * 8 : U32) + 8 : U32)),(group * 8 : U32),width,Tiled.joined(Gather.run_shift(Gather.run_of(split,first,local,count)),low,high)),
    SplitValue.spot_cell(width,spot_target,low,high),
    Representation.lifted(~Logical,~original,Representation.Lifted{logical,target_cells(target_outer,target_after,width,Grid.Innermost{})},target),masked,
    cell_law_at(~Logical,~original,logical,limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,first,count,
      cells_room,piece_room,index,index_below,low,high,low_read,high_read))

# Every cell of every run of a target piece with the groups innermost.
def run_law(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,
  +target_outer: U32,+target_after: U32,+source_outer: U32,+source_after: U32,
  +facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+count: U32,
  +source_cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat(source_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool}) ->
  Gather.RunLaw(~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),~Representation.Lifted<Logical>,~Representation.lifted(~Logical,~original),
    Representation.Lifted{logical,source_cells(source_outer,source_after,lanes,Grid.Innermost{})},
    Representation.Lifted{logical,target_cells(target_outer,target_after,width,Grid.Innermost{})},split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,count):
  local => below => step => step_inside => index => index_exact => group => group_exact => low => high => low_exact => high_exact =>
    run_law_at(~Logical,~original,logical,limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,count,source_cells_room,
      cells_room,piece_room,local,below,step,step_inside,index,index_exact,group,group_exact,low,high,low_exact,high_exact)

# A step of a run that a lane needs reads the next source cell: the lanes
# of the run shift and the next cell lies inside the source groups.
def moved_needs(+limits: Machine.Limits,+anchors: U32,+width: U32,+lanes: U32,+source_outer: U32,+anchor: Nat,+group: Nat,+step: Nat,+row: Nat,+run: Tiled.Run,
  +moved_lanes: SplitValue.Lanes<limits,anchors,width,lanes,anchor,Nat.add(group,step)>,+anchor_inside: {Nat.is_lt(anchor,U32.to_nat(anchors)) == True{} : Bool},
  +shift_exact: {U32.to_nat(Gather.run_shift(run)) == Nat.mod(SplitValue.first_lane(anchor,width,Nat.add(group,step)),8n) : Nat},
  +cell_exact: {U32.to_nat(Gather.run_cell(run)) == Nat.div(SplitValue.first_lane(anchor,width,group),8n) : Nat},+spot: Tiled.Spot,
  +spot_exact: {spot == SplitValue.spot_of_lanes(Grid.Innermost{},anchor,Nat.add(group,step),row,lanes,source_outer) : Tiled.Spot},
  +need: {SplitValue.spot_needed(width,spot) == True{} : Bool}) ->
  {Nat.is_lt(0n,U32.to_nat(Gather.run_shift(run))) && Nat.is_lt(1n+Nat.add(U32.to_nat(Gather.run_cell(run)),step),U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}:
  +moved = Nat.add(group,step)
  +shift = Nat.mod(SplitValue.first_lane(anchor,width,moved),8n)
  +moved_need = {%spot_exact : {SplitValue.spot_needed(width,_) == True{} : Bool} need :
    {SplitValue.spot_needed(width,SplitValue.spot_of_lanes(Grid.Innermost{},anchor,moved,row,lanes,source_outer)) == True{} : Bool}}
  +plain = SplitValue.need_nat(limits,anchors,width,lanes,source_outer,Grid.Innermost{},anchor,moved,row,moved_lanes,moved_need)
  +shifted = {%Equal.sym(Nat,U32.to_nat(Gather.run_shift(run)),shift,shift_exact) : {Nat.is_lt(0n,_) == True{} : Bool}
    Logic.left(Nat.is_lt(0n,shift),Nat.is_lt(Nat.add(Nat.mul(moved,8n),Nat.sub(8n,shift)),U32.to_nat(width)),plain) :
    {Nat.is_lt(0n,U32.to_nat(Gather.run_shift(run))) == True{} : Bool}}
  +follows = {%Equal.sym(Nat,U32.to_nat(Gather.run_cell(run)),Nat.div(SplitValue.first_lane(anchor,width,group),8n),cell_exact) :
      {Nat.is_lt(1n+Nat.add(_,step),U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}
    %stepped_cell(anchor,width,group,step) : {Nat.is_lt(1n+_,U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}
    SplitValue.needed_next(limits,anchors,width,lanes,anchor,moved,moved_lanes,anchor_inside,plain) :
    {Nat.is_lt(1n+Nat.add(U32.to_nat(Gather.run_cell(run)),step),U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}}
  Logic.and_true(Nat.is_lt(0n,U32.to_nat(Gather.run_shift(run))),Nat.is_lt(1n+Nat.add(U32.to_nat(Gather.run_cell(run)),step),U32.to_nat(Tiled.groups(lanes))),shifted,follows)

def run_needs_at(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+count: U32,
  +source_cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat(source_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool},
  +local: U32,+below: {Nat.is_lt(U32.to_nat(local),U32.to_nat(count)) == True{} : Bool},+step: Nat,
  +step_inside: {Nat.is_lt(step,U32.to_nat(Gather.run_count_at(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count))) == True{} : Bool},
  +index: U32,+index_exact: {U32.to_nat(index) == Nat.add(U32.to_nat(local),step) : Nat}) ->
  Logic.When(SplitValue.spot_needed(width,Tiled.split_spot(Grid.Innermost{},outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),(first + index : U32))),
    {Nat.is_lt(0n,U32.to_nat(Gather.run_shift(Gather.run_of(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count)))) &&
      Nat.is_lt(1n+Nat.add(U32.to_nat(Gather.run_cell(Gather.run_of(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,local,count))),step),
        U32.to_nat(Tiled.groups(lanes))) == True{} : Bool}):
  +split = split_of(outer,anchors,width,after,lanes,Grid.Innermost{})
  +start = (first + local : U32)
  +target = (first + index : U32)
  +digits = SplitValue.split_digits(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,start,
    target_inside(limits,width,target_outer,first,count,cells_room,piece_room,local,below))
  +anchor = inner_anchor(target_outer,width,anchors,after,start)
  +group_first = inner_group(target_outer,width,start)
  +row = inner_source(target_outer,width,anchors,after,start)
  +moved = Nat.add(group_first,step)
  +lanes_first = SplitValue.split_lanes_of(limits,outer,anchors,width,after,lanes,Grid.Innermost{},target_outer,target_after,source_outer,source_after,facts,start,digits)
  +run = inner_run(width,lanes,source_outer,first,local,count,anchor,group_first,row)
  +back = Equal.sym(Tiled.Run,Gather.run_of(split,first,local,count),run,
    inner_of(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,local,count,digits))
  +capacity = U32.to_nat(Machine.capacity(limits))
  +counts = inner_counts(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,count,source_cells_room,cells_room,piece_room,
    local,below)
  +index_below = step_below(U32.to_nat(local),step,U32.to_nat(index),U32.to_nat(count),U32.to_nat(Gather.run_count_at(split,first,local,count)),index_exact,step_inside,
    counts_room(split,first,local,count,Machine.half_word(limits),counts))
  +moved_inside = step_group(U32.to_nat(Tiled.groups(width)),group_first,step,U32.to_nat(Gather.run_count_at(split,first,local,count)),
    Nat.sub(U32.to_nat(count),U32.to_nat(local)),
    run_count_exact(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,count,cells_room,piece_room,local,below),step_inside,
    digits_group_inside(outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},start,digits))
  +target_exact = target_step(Machine.capacity(limits),Limits.capacity_fits(limits),first,local,count,index,step,index_exact,index_below,below,
    Order.transitive(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),capacity,piece_room,cells_room))
  +spot_target = Tiled.split_spot(Grid.Innermost{},outer,anchors,after,Tiled.groups(width),Tiled.groups(lanes),target)
  +spot_exact = step_spot(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,local,digits,step,target,target_exact,
    moved_inside)
  +lanes_moved = step_lanes(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,local,digits,step,moved_inside)
  +first_word = (U32.from_nat(anchor) * width + U32.from_nat(group_first) * 8 : U32)
  +shift_exact = {%back : {U32.to_nat(Gather.run_shift(_)) == Nat.mod(SplitValue.first_lane(anchor,width,moved),8n) : Nat}
    Equal.trans(Nat,U32.to_nat((first_word % 8 : U32)),Nat.mod(SplitValue.first_lane(anchor,width,group_first),8n),Nat.mod(SplitValue.first_lane(anchor,width,moved),8n),
      lanes_shift(limits,anchors,width,lanes,anchor,group_first,lanes_first),
      Equal.sym(Nat,Nat.mod(SplitValue.first_lane(anchor,width,moved),8n),Nat.mod(SplitValue.first_lane(anchor,width,group_first),8n),stepped_shift(anchor,width,group_first,step))) :
    {U32.to_nat(Gather.run_shift(Gather.run_of(split,first,local,count))) == Nat.mod(SplitValue.first_lane(anchor,width,moved),8n) : Nat}}
  +cell_exact = {%back : {U32.to_nat(Gather.run_cell(_)) == Nat.div(SplitValue.first_lane(anchor,width,group_first),8n) : Nat}
    lanes_cell(limits,anchors,width,lanes,anchor,group_first,lanes_first) :
    {U32.to_nat(Gather.run_cell(Gather.run_of(split,first,local,count))) == Nat.div(SplitValue.first_lane(anchor,width,group_first),8n) : Nat}}
  Logic.when_enabled({Nat.is_lt(0n,U32.to_nat(Gather.run_shift(Gather.run_of(split,first,local,count)))) &&
      Nat.is_lt(1n+Nat.add(U32.to_nat(Gather.run_cell(Gather.run_of(split,first,local,count))),step),U32.to_nat(Tiled.groups(lanes))) == True{} : Bool},
    SplitValue.spot_needed(width,spot_target),
    need => moved_needs(limits,anchors,width,lanes,source_outer,anchor,group_first,step,row,Gather.run_of(split,first,local,count),lanes_moved,
      digits_anchor_inside(outer,anchors,width,after,lanes,target_outer,source_outer,Grid.Innermost{},start,digits),shift_exact,cell_exact,spot_target,spot_exact,need))

# Every step of every run of a target piece with the groups innermost that
# a lane needs reads the next source cell.
def run_needs(+limits: Machine.Limits,+outer: U32,+anchors: U32,+width: U32,+after: U32,+lanes: U32,+target_outer: U32,+target_after: U32,+source_outer: U32,
  +source_after: U32,+facts: SplitValue.Facts<limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after>,+first: U32,+count: U32,
  +source_cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(lanes)),U32.to_nat(source_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_room: {Nat.is_le(Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +piece_room: {Nat.is_le(Nat.add(U32.to_nat(first),U32.to_nat(count)),Nat.mul(U32.to_nat(Tiled.groups(width)),U32.to_nat(target_outer))) == True{} : Bool}) ->
  Gather.RunNeeds(split_of(outer,anchors,width,after,lanes,Grid.Innermost{}),first,count):
  local => below => step => step_inside => index => index_exact =>
    run_needs_at(limits,outer,anchors,width,after,lanes,target_outer,target_after,source_outer,source_after,facts,first,count,source_cells_room,cells_room,piece_room,
      local,below,step,step_inside,index,index_exact)
