# A cell of a lane axis split into anchors and a width. Lane l of the target
# cell of anchor a and group j holds the source lane a * width + 8 * j + l of
# the outer index with the same elements before and after, +0 past the width:
# the source cells that hold its lanes, joined at the shift of its first lane
# and cut at the width, give the target cell.
import Base
import ../kernel_tile.bend as Tile
import ../tensor_tiled.bend as Tiled
import ../tensor_tiled_cells.bend as Cells
import ../machine_limits.bend as Machine
import ./kernel_tile_refinement.bend as TileLaws
import ./tensor_tiled_representation.bend as Representation
import ./tensor_tiled_padded_cells.bend as PaddedCells
import ./tensor_lane_shift.bend as LaneShift
import ./tensor_lanes_write_cell.bend as WriteCell
import ./nat_order.bend as Order
import ./nat_division.bend as Division
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./boolean_logic.bend as Boolean
import ./word_encoding.bend as WordEncoding

# ---- Where a target lane lies in the source ----

# The flat index of a lane: the rows above, the lanes, the channel, the
# elements after and the index among them.
def flat_index(+above: Nat,+lanes: Nat,+channel: Nat,+after: Nat,+below: Nat) -> Nat:
  Nat.add(Nat.mul(Nat.add(Nat.mul(above,lanes),channel),after),below)

# Channel channel of anchor anchor at row above is source lane
# anchor * width + channel of that row.
def anchored(+above: Nat,+anchors: Nat,+anchor: Nat,+width: Nat,+channel: Nat) ->
  {Nat.add(Nat.mul(Nat.add(Nat.mul(above,anchors),anchor),width),channel) == Nat.add(Nat.mul(above,Nat.mul(anchors,width)),Nat.add(Nat.mul(anchor,width),channel)) : Nat}:
  +row = Nat.mul(above,anchors)
  +start = Nat.mul(anchor,width)
  Equal.trans(Nat,Nat.add(Nat.mul(Nat.add(row,anchor),width),channel),Nat.add(Nat.add(Nat.mul(row,width),start),channel),
    Nat.add(Nat.mul(above,Nat.mul(anchors,width)),Nat.add(start,channel)),
    Equal.cong(Nat,Nat,value => Nat.add(value,channel),Nat.mul(Nat.add(row,anchor),width),Nat.add(Nat.mul(row,width),start),RowMajor.distrib(row,anchor,width)),
    Equal.trans(Nat,Nat.add(Nat.add(Nat.mul(row,width),start),channel),Nat.add(Nat.add(Nat.mul(above,Nat.mul(anchors,width)),start),channel),
      Nat.add(Nat.mul(above,Nat.mul(anchors,width)),Nat.add(start,channel)),
      Equal.cong(Nat,Nat,value => Nat.add(Nat.add(value,start),channel),Nat.mul(row,width),Nat.mul(above,Nat.mul(anchors,width)),RowMajor.assoc(above,anchors,width)),
      Addition.associative(Nat.mul(above,Nat.mul(anchors,width)),start,channel)))

# A target lane and the source lane anchor * width + channel name one
# element: the target rows hold the anchors of a source row, and both outer
# indices hold the same elements after the lanes.
def split_index(+source: Cells.Cells,+target: Cells.Cells,+source_outer: Nat,+target_outer: Nat,+above: Nat,+anchors: Nat,+anchor: Nat,+width: Nat,+channel: Nat,
  +source_lanes: {Representation.lane_count(source) == Nat.mul(anchors,width) : Nat},+target_lanes: {Representation.lane_count(target) == width : Nat},
  +afters: {Representation.after_count(target) == Representation.after_count(source) : Nat},
  +target_above: {Nat.div(target_outer,Representation.after_count(source)) == Nat.add(Nat.mul(above,anchors),anchor) : Nat},
  +source_above: {Nat.div(source_outer,Representation.after_count(source)) == above : Nat},
  +same_below: {Nat.mod(target_outer,Representation.after_count(source)) == Nat.mod(source_outer,Representation.after_count(source)) : Nat}) ->
  {Representation.logical_index(target,target_outer,channel) == Representation.logical_index(source,source_outer,Nat.add(Nat.mul(anchor,width),channel)) : Nat}:
  +after = Representation.after_count(source)
  +below = Nat.mod(source_outer,after)
  +row = Nat.add(Nat.mul(above,anchors),anchor)
  +distance = Nat.add(Nat.mul(anchor,width),channel)
  %Equal.sym(Nat,Representation.after_count(target),after,afters) :
    {flat_index(Nat.div(target_outer,_),Representation.lane_count(target),channel,_,Nat.mod(target_outer,_)) == Representation.logical_index(source,source_outer,distance) : Nat}
  %Equal.sym(Nat,Nat.div(target_outer,after),row,target_above) :
    {flat_index(_,Representation.lane_count(target),channel,after,Nat.mod(target_outer,after)) == Representation.logical_index(source,source_outer,distance) : Nat}
  %Equal.sym(Nat,Representation.lane_count(target),width,target_lanes) :
    {flat_index(row,_,channel,after,Nat.mod(target_outer,after)) == Representation.logical_index(source,source_outer,distance) : Nat}
  %Equal.sym(Nat,Nat.mod(target_outer,after),below,same_below) : {flat_index(row,width,channel,after,_) == Representation.logical_index(source,source_outer,distance) : Nat}
  %Equal.sym(Nat,Nat.div(source_outer,after),above,source_above) :
    {flat_index(row,width,channel,after,below) == flat_index(_,Representation.lane_count(source),distance,after,below) : Nat}
  %Equal.sym(Nat,Representation.lane_count(source),Nat.mul(anchors,width),source_lanes) :
    {flat_index(row,width,channel,after,below) == flat_index(above,_,distance,after,below) : Nat}
  Equal.cong(Nat,Nat,value => Nat.add(Nat.mul(value,after),below),Nat.add(Nat.mul(row,width),channel),Nat.add(Nat.mul(above,Nat.mul(anchors,width)),distance),
    anchored(above,anchors,anchor,width,channel))

# The source lane of a target lane inside the width holds the logical value
# of the target lane.
def split_channel(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: Cells.Cells,+target: Cells.Cells,+source_outer: Nat,+target_outer: Nat,
  +above: Nat,+anchors: Nat,+anchor: Nat,+width: Nat,+channel: Nat,
  +source_lanes: {Representation.lane_count(source) == Nat.mul(anchors,width) : Nat},+target_lanes: {Representation.lane_count(target) == width : Nat},
  +afters: {Representation.after_count(target) == Representation.after_count(source) : Nat},
  +target_above: {Nat.div(target_outer,Representation.after_count(source)) == Nat.add(Nat.mul(above,anchors),anchor) : Nat},
  +source_above: {Nat.div(source_outer,Representation.after_count(source)) == above : Nat},
  +same_below: {Nat.mod(target_outer,Representation.after_count(source)) == Nat.mod(source_outer,Representation.after_count(source)) : Nat},
  +anchor_inside: {Nat.is_lt(anchor,anchors) == True{} : Bool},+inside: {Nat.is_lt(channel,width) == True{} : Bool}) ->
  {WriteCell.lane_at(~Logical,~original,logical,source,source_outer,Nat.add(Nat.mul(anchor,width),channel)) ==
    original(logical,U32.from_nat(Representation.logical_index(target,target_outer,channel))) : F32}:
  +distance = Nat.add(Nat.mul(anchor,width),channel)
  +source_in = {%Equal.sym(Nat,Representation.lane_count(source),Nat.mul(anchors,width),source_lanes) : {Nat.is_lt(distance,_) == True{} : Bool}
    RowMajor.radix(anchor,anchors,channel,width,anchor_inside,inside) : {Nat.is_lt(distance,Representation.lane_count(source)) == True{} : Bool}}
  %Equal.sym(Bool,Nat.is_lt(distance,Representation.lane_count(source)),True{},source_in) :
    {Bool.pick(F32,_,original(logical,U32.from_nat(Representation.logical_index(source,source_outer,distance))),0.0) ==
      original(logical,U32.from_nat(Representation.logical_index(target,target_outer,channel))) : F32}
  Equal.cong(Nat,F32,index => original(logical,U32.from_nat(index)),Representation.logical_index(source,source_outer,distance),
    Representation.logical_index(target,target_outer,channel),
    Equal.sym(Nat,Representation.logical_index(target,target_outer,channel),Representation.logical_index(source,source_outer,distance),
      split_index(source,target,source_outer,target_outer,above,anchors,anchor,width,channel,source_lanes,target_lanes,afters,target_above,source_above,same_below)))

# ---- The joined source cells ----

# The first lane of the group lies shift lanes into source cell cell, so lane
# lane of the group is source lane lane + shift of that cell.
def source_channel(+lane: Nat,+shift: Nat,+cell: Nat,+anchor: Nat,+width: Nat,+group: Nat,
  +key: {Nat.add(Nat.mul(cell,8n),shift) == Nat.add(Nat.mul(anchor,width),Nat.mul(group,8n)) : Nat}) ->
  {Nat.add(Nat.mul(anchor,width),Nat.add(lane,Nat.mul(group,8n))) == Nat.add(Nat.add(lane,shift),Nat.mul(cell,8n)) : Nat}:
  +start = Nat.mul(anchor,width)
  +before = Nat.mul(cell,8n)
  Equal.trans(Nat,Nat.add(start,Nat.add(lane,Nat.mul(group,8n))),Nat.add(lane,Nat.add(start,Nat.mul(group,8n))),Nat.add(Nat.add(lane,shift),before),
    RowMajor.add_swap(start,lane,Nat.mul(group,8n)),
    Equal.trans(Nat,Nat.add(lane,Nat.add(start,Nat.mul(group,8n))),Nat.add(lane,Nat.add(before,shift)),Nat.add(Nat.add(lane,shift),before),
      Equal.cong(Nat,Nat,value => Nat.add(lane,value),Nat.add(start,Nat.mul(group,8n)),Nat.add(before,shift),
        Equal.sym(Nat,Nat.add(before,shift),Nat.add(start,Nat.mul(group,8n)),key)),
      Equal.trans(Nat,Nat.add(lane,Nat.add(before,shift)),Nat.add(lane,Nat.add(shift,before)),Nat.add(Nat.add(lane,shift),before),
        Equal.cong(Nat,Nat,value => Nat.add(lane,value),Nat.add(before,shift),Nat.add(shift,before),Order.commutative(before,shift)),
        Equal.sym(Nat,Nat.add(Nat.add(lane,shift),before),Nat.add(lane,Nat.add(shift,before)),Addition.associative(lane,shift,before)))))

# A lane moved past its cell is lane lane + shift - 8 of the next cell.
def high_channel(+lane: Nat,+shift: Nat,+cell: Nat,+distance: Nat,
  +low_exact: {distance == Nat.add(Nat.add(lane,shift),Nat.mul(cell,8n)) : Nat},+high: {Nat.is_lt(Nat.add(lane,shift),8n) == False{} : Bool}) ->
  {distance == Nat.add(Nat.sub(Nat.add(lane,shift),8n),Nat.mul(1n+cell,8n)) : Nat}:
  +moved = Nat.add(lane,shift)
  +rest = Nat.sub(moved,8n)
  +restored = Equal.trans(Nat,Nat.add(rest,8n),Nat.add(8n,rest),moved,Order.commutative(rest,8n),Order.restore_shift(8n,moved,Order.not_less(moved,8n,high)))
  Equal.trans(Nat,distance,Nat.add(moved,Nat.mul(cell,8n)),Nat.add(rest,Nat.mul(1n+cell,8n)),low_exact,
    Equal.trans(Nat,Nat.add(moved,Nat.mul(cell,8n)),Nat.add(Nat.add(rest,8n),Nat.mul(cell,8n)),Nat.add(rest,Nat.mul(1n+cell,8n)),
      Equal.cong(Nat,Nat,value => Nat.add(value,Nat.mul(cell,8n)),moved,Nat.add(rest,8n),Equal.sym(Nat,Nat.add(rest,8n),moved,restored)),
      Addition.associative(rest,8n,Nat.mul(cell,8n))))

# The lane of group 0 at a channel is the lane at the channel.
def first_group(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+channel: Nat) ->
  {Representation.lane(~Logical,~original,logical,geometry,outer,0n,channel) == WriteCell.lane_at(~Logical,~original,logical,geometry,outer,channel) : F32}:
  Equal.cong(Nat,F32,value => WriteCell.lane_at(~Logical,~original,logical,geometry,outer,value),Nat.add(channel,0n),channel,Addition.zero_right(channel))

# The joined cells give a lane the source lane at its distance: lane
# lane + shift of the low cell, or of the high cell past it.
def joined_source(~Logical: Data,~original: Logical -> U32 -> F32,low: Bool,+logical: Logical,+source: Cells.Cells,+source_outer: Nat,+cell: Nat,+shift: Nat,
  +lane: Nat,+distance: Nat,+classified: {Nat.is_lt(Nat.add(lane,shift),8n) == low : Bool},
  +lane_small: {Nat.is_le(lane,7n) == True{} : Bool},+shift_small: {Nat.is_le(shift,7n) == True{} : Bool},
  +low_exact: {distance == Nat.add(Nat.add(lane,shift),Nat.mul(cell,8n)) : Nat}) ->
  {Bool.pick(F32,low,TileLaws.select_lane(Nat.add(lane,shift),Representation.cell(~Logical,~original,logical,source,source_outer,cell)),
      TileLaws.select_lane(Nat.sub(Nat.add(lane,shift),8n),Representation.cell(~Logical,~original,logical,source,source_outer,1n+cell))) ==
    WriteCell.lane_at(~Logical,~original,logical,source,source_outer,distance) : F32}:
  match low:
    case True{}:
      +moved = Nat.add(lane,shift)
      Equal.trans(F32,TileLaws.select_lane(moved,Representation.cell(~Logical,~original,logical,source,source_outer,cell)),
        Representation.lane(~Logical,~original,logical,source,source_outer,cell,moved),WriteCell.lane_at(~Logical,~original,logical,source,source_outer,distance),
        Representation.cell_lane(~Logical,~original,logical,source,source_outer,cell,moved,Division.lt_succ_le(moved,7n,classified)),
        Equal.trans(F32,Representation.lane(~Logical,~original,logical,source,source_outer,cell,moved),
          Representation.lane(~Logical,~original,logical,source,source_outer,0n,distance),WriteCell.lane_at(~Logical,~original,logical,source,source_outer,distance),
          WriteCell.distance_moved(~Logical,~original,logical,source,source_outer,cell,moved,distance,low_exact),
          first_group(~Logical,~original,logical,source,source_outer,distance)))
    case False{}:
      +moved = Nat.sub(Nat.add(lane,shift),8n)
      Equal.trans(F32,TileLaws.select_lane(moved,Representation.cell(~Logical,~original,logical,source,source_outer,1n+cell)),
        Representation.lane(~Logical,~original,logical,source,source_outer,1n+cell,moved),WriteCell.lane_at(~Logical,~original,logical,source,source_outer,distance),
        Representation.cell_lane(~Logical,~original,logical,source,source_outer,1n+cell,moved,WriteCell.moved_small(lane,shift,lane_small,shift_small)),
        Equal.trans(F32,Representation.lane(~Logical,~original,logical,source,source_outer,1n+cell,moved),
          Representation.lane(~Logical,~original,logical,source,source_outer,0n,distance),WriteCell.lane_at(~Logical,~original,logical,source,source_outer,distance),
          WriteCell.distance_moved(~Logical,~original,logical,source,source_outer,1n+cell,moved,distance,high_channel(lane,shift,cell,distance,low_exact,classified)),
          first_group(~Logical,~original,logical,source,source_outer,distance)))

# What a split cell holds, as the facts of its place.
type Place<-source: Cells.Cells,-target: Cells.Cells,-source_outer: Nat,-target_outer: Nat,-above: Nat,-anchors: Nat,-anchor: Nat,-width: Nat> is Data:
  Place{source_lanes: {Representation.lane_count(source) == Nat.mul(anchors,width) : Nat},target_lanes: {Representation.lane_count(target) == width : Nat},
    afters: {Representation.after_count(target) == Representation.after_count(source) : Nat},
    target_above: {Nat.div(target_outer,Representation.after_count(source)) == Nat.add(Nat.mul(above,anchors),anchor) : Nat},
    source_above: {Nat.div(source_outer,Representation.after_count(source)) == above : Nat},
    same_below: {Nat.mod(target_outer,Representation.after_count(source)) == Nat.mod(source_outer,Representation.after_count(source)) : Nat},
    anchor_inside: {Nat.is_lt(anchor,anchors) == True{} : Bool}}

# A lane of the group inside the width: the joined source cells hold its
# logical value.
def split_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: Cells.Cells,+target: Cells.Cells,+source_outer: Nat,+target_outer: Nat,
  +above: Nat,+anchors: Nat,+anchor: Nat,+width: Nat,+group: Nat,+cell: Nat,+shift: Nat,+lane: Nat,
  +place: Place<source,target,source_outer,target_outer,above,anchors,anchor,width>,
  +key: {Nat.add(Nat.mul(cell,8n),shift) == Nat.add(Nat.mul(anchor,width),Nat.mul(group,8n)) : Nat},
  +shift_small: {Nat.is_le(shift,7n) == True{} : Bool},+lane_small: {Nat.is_le(lane,7n) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),width) == True{} : Bool}) ->
  {TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),Representation.cell(~Logical,~original,logical,source,source_outer,cell),
      Representation.cell(~Logical,~original,logical,source,source_outer,1n+cell))) ==
    original(logical,U32.from_nat(Representation.logical_index(target,target_outer,Nat.add(lane,Nat.mul(group,8n))))) : F32}:
  Place{+source_lanes,+target_lanes,+afters,+target_above,+source_above,+same_below,+anchor_inside} = place
  +low = Representation.cell(~Logical,~original,logical,source,source_outer,cell)
  +high = Representation.cell(~Logical,~original,logical,source,source_outer,1n+cell)
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +distance = Nat.add(Nat.mul(anchor,width),channel)
  Equal.trans(F32,TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),low,high)),LaneShift.joined_value(shift,lane,low,high),
    original(logical,U32.from_nat(Representation.logical_index(target,target_outer,channel))),
    LaneShift.joined_lane(shift,lane,low,high,shift_small,lane_small),
    Equal.trans(F32,LaneShift.joined_value(shift,lane,low,high),WriteCell.lane_at(~Logical,~original,logical,source,source_outer,distance),
      original(logical,U32.from_nat(Representation.logical_index(target,target_outer,channel))),
      joined_source(~Logical,~original,Nat.is_lt(Nat.add(lane,shift),8n),logical,source,source_outer,cell,shift,lane,distance,{==},lane_small,shift_small,
        source_channel(lane,shift,cell,anchor,width,group,key)),
      split_channel(~Logical,~original,logical,source,target,source_outer,target_outer,above,anchors,anchor,width,channel,
        source_lanes,target_lanes,afters,target_above,source_above,same_below,anchor_inside,inside)))

# A lane of the cut cell: the target lane, its logical value inside the width
# and +0 past it.
def split_kept(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+source: Cells.Cells,+target: Cells.Cells,+source_outer: Nat,+target_outer: Nat,
  +above: Nat,+anchors: Nat,+anchor: Nat,+width: Nat,+group: Nat,+cell: Nat,+shift: Nat,+lane: Nat,
  +place: Place<source,target,source_outer,target_outer,above,anchors,anchor,width>,
  +key: {Nat.add(Nat.mul(cell,8n),shift) == Nat.add(Nat.mul(anchor,width),Nat.mul(group,8n)) : Nat},
  +shift_small: {Nat.is_le(shift,7n) == True{} : Bool},+lane_small: {Nat.is_le(lane,7n) == True{} : Bool}) ->
  {PaddedCells.kept(width,group,lane,TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),Representation.cell(~Logical,~original,logical,source,source_outer,cell),
      Representation.cell(~Logical,~original,logical,source,source_outer,1n+cell)))) ==
    Representation.lane(~Logical,~original,logical,target,target_outer,group,lane) : F32}:
  Place{+source_lanes,+target_lanes,+afters,+target_above,+source_above,+same_below,+anchor_inside} = place
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +joined = TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),Representation.cell(~Logical,~original,logical,source,source_outer,cell),
    Representation.cell(~Logical,~original,logical,source,source_outer,1n+cell)))
  +value = original(logical,U32.from_nat(Representation.logical_index(target,target_outer,channel)))
  %Equal.sym(Nat,Representation.lane_count(target),width,target_lanes) : {PaddedCells.kept(width,group,lane,joined) == Bool.pick(F32,Nat.is_lt(channel,_),value,0.0) : F32}
  PaddedCells.kept_agree(width,group,lane,joined,value,Nat.is_lt(channel,width),{==},
    hit => split_lane(~Logical,~original,logical,source,target,source_outer,target_outer,above,anchors,anchor,width,group,cell,shift,lane,
      Place{source_lanes,target_lanes,afters,target_above,source_above,same_below,anchor_inside},key,shift_small,lane_small,hit))

# The cell a split gives a group: the source cell of its first lane and the
# next joined at the shift, cut at the width, hold the target cell.
def split_cell(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+limits: Machine.Limits,+source: Cells.Cells,+target: Cells.Cells,+source_outer: Nat,
  +target_outer: Nat,+above: Nat,+anchors: Nat,+anchor: Nat,+width: U32,+group: Nat,+first: U32,+cell: Nat,+shift: Nat,
  +place: Place<source,target,source_outer,target_outer,above,anchors,anchor,U32.to_nat(width)>,
  +key: {Nat.add(Nat.mul(cell,8n),shift) == Nat.add(Nat.mul(anchor,U32.to_nat(width)),Nat.mul(group,8n)) : Nat},
  +shift_small: {Nat.is_le(shift,7n) == True{} : Bool},+first_exact: {U32.to_nat(first) == Nat.mul(group,8n) : Nat},
  +bound: {Nat.is_le(Nat.add(Nat.mul(group,8n),7n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  {Cells.tail(Cells.Tail{first,width},Tiled.joined(U32.from_nat(shift),Representation.cell(~Logical,~original,logical,source,source_outer,cell),
      Representation.cell(~Logical,~original,logical,source,source_outer,1n+cell))) ==
    Representation.cell(~Logical,~original,logical,target,target_outer,group) : Tile.Tile8}:
  +w = U32.to_nat(width)
  +joined = Tiled.joined(U32.from_nat(shift),Representation.cell(~Logical,~original,logical,source,source_outer,cell),
    Representation.cell(~Logical,~original,logical,source,source_outer,1n+cell))
  +parts = {Tile.Tile{Tile.lane0(joined),Tile.lane1(joined),Tile.lane2(joined),Tile.lane3(joined),Tile.lane4(joined),Tile.lane5(joined),Tile.lane6(joined),Tile.lane7(joined)} : Tile.Tile8}
  +wanted = Representation.cell(~Logical,~original,logical,target,target_outer,group)
  %Equal.sym(Tile.Tile8,joined,parts,TileLaws.tile_eta(joined)) : {Cells.tail(Cells.Tail{first,width},_) == wanted : Tile.Tile8}
  TileLaws.tiles_equal(Cells.tail(Cells.Tail{first,width},parts),wanted,
    Equal.trans(F32,Cells.tail_lane(first,width,0,Tile.lane0(joined)),PaddedCells.kept(w,group,0n,Tile.lane0(joined)),Tile.lane0(wanted),
      PaddedCells.tail_lane(Tile.lane0(joined),limits,first,width,group,0n,first_exact,{==},PaddedCells.lane_bound(limits,group,0n,{==},bound)),
      split_kept(~Logical,~original,logical,source,target,source_outer,target_outer,above,anchors,anchor,w,group,cell,shift,0n,place,key,shift_small,{==})),
    Equal.trans(F32,Cells.tail_lane(first,width,1,Tile.lane1(joined)),PaddedCells.kept(w,group,1n,Tile.lane1(joined)),Tile.lane1(wanted),
      PaddedCells.tail_lane(Tile.lane1(joined),limits,first,width,group,1n,first_exact,{==},PaddedCells.lane_bound(limits,group,1n,{==},bound)),
      split_kept(~Logical,~original,logical,source,target,source_outer,target_outer,above,anchors,anchor,w,group,cell,shift,1n,place,key,shift_small,{==})),
    Equal.trans(F32,Cells.tail_lane(first,width,2,Tile.lane2(joined)),PaddedCells.kept(w,group,2n,Tile.lane2(joined)),Tile.lane2(wanted),
      PaddedCells.tail_lane(Tile.lane2(joined),limits,first,width,group,2n,first_exact,{==},PaddedCells.lane_bound(limits,group,2n,{==},bound)),
      split_kept(~Logical,~original,logical,source,target,source_outer,target_outer,above,anchors,anchor,w,group,cell,shift,2n,place,key,shift_small,{==})),
    Equal.trans(F32,Cells.tail_lane(first,width,3,Tile.lane3(joined)),PaddedCells.kept(w,group,3n,Tile.lane3(joined)),Tile.lane3(wanted),
      PaddedCells.tail_lane(Tile.lane3(joined),limits,first,width,group,3n,first_exact,{==},PaddedCells.lane_bound(limits,group,3n,{==},bound)),
      split_kept(~Logical,~original,logical,source,target,source_outer,target_outer,above,anchors,anchor,w,group,cell,shift,3n,place,key,shift_small,{==})),
    Equal.trans(F32,Cells.tail_lane(first,width,4,Tile.lane4(joined)),PaddedCells.kept(w,group,4n,Tile.lane4(joined)),Tile.lane4(wanted),
      PaddedCells.tail_lane(Tile.lane4(joined),limits,first,width,group,4n,first_exact,{==},PaddedCells.lane_bound(limits,group,4n,{==},bound)),
      split_kept(~Logical,~original,logical,source,target,source_outer,target_outer,above,anchors,anchor,w,group,cell,shift,4n,place,key,shift_small,{==})),
    Equal.trans(F32,Cells.tail_lane(first,width,5,Tile.lane5(joined)),PaddedCells.kept(w,group,5n,Tile.lane5(joined)),Tile.lane5(wanted),
      PaddedCells.tail_lane(Tile.lane5(joined),limits,first,width,group,5n,first_exact,{==},PaddedCells.lane_bound(limits,group,5n,{==},bound)),
      split_kept(~Logical,~original,logical,source,target,source_outer,target_outer,above,anchors,anchor,w,group,cell,shift,5n,place,key,shift_small,{==})),
    Equal.trans(F32,Cells.tail_lane(first,width,6,Tile.lane6(joined)),PaddedCells.kept(w,group,6n,Tile.lane6(joined)),Tile.lane6(wanted),
      PaddedCells.tail_lane(Tile.lane6(joined),limits,first,width,group,6n,first_exact,{==},PaddedCells.lane_bound(limits,group,6n,{==},bound)),
      split_kept(~Logical,~original,logical,source,target,source_outer,target_outer,above,anchors,anchor,w,group,cell,shift,6n,place,key,shift_small,{==})),
    Equal.trans(F32,Cells.tail_lane(first,width,7,Tile.lane7(joined)),PaddedCells.kept(w,group,7n,Tile.lane7(joined)),Tile.lane7(wanted),
      PaddedCells.tail_lane(Tile.lane7(joined),limits,first,width,group,7n,first_exact,{==},PaddedCells.lane_bound(limits,group,7n,{==},bound)),
      split_kept(~Logical,~original,logical,source,target,source_outer,target_outer,above,anchors,anchor,w,group,cell,shift,7n,place,key,shift_small,{==})))

# ---- The cell after the last ----

# A source cell past the source lanes holds +0.
def past_lane(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,+lane: Nat,
  +past: {Nat.is_le(Representation.lane_count(geometry),Nat.mul(group,8n)) == True{} : Bool}) ->
  {Representation.lane(~Logical,~original,logical,geometry,outer,group,lane) == 0.0 : F32}:
  +channel = Nat.add(lane,Nat.mul(group,8n))
  +outside = Order.not_below(Representation.lane_count(geometry),channel,
    Order.transitive(Representation.lane_count(geometry),Nat.mul(group,8n),channel,past,Order.right_le(lane,Nat.mul(group,8n))))
  %Equal.sym(Bool,Nat.is_lt(channel,Representation.lane_count(geometry)),False{},outside) :
    {Bool.pick(F32,_,original(logical,U32.from_nat(Representation.logical_index(geometry,outer,channel))),0.0) == 0.0 : F32}
  {==}

def past_cell(~Logical: Data,~original: Logical -> U32 -> F32,+logical: Logical,+geometry: Cells.Cells,+outer: Nat,+group: Nat,
  +past: {Nat.is_le(Representation.lane_count(geometry),Nat.mul(group,8n)) == True{} : Bool}) ->
  {Representation.cell(~Logical,~original,logical,geometry,outer,group) == Tiled.zero_tile() : Tile.Tile8}:
  TileLaws.tiles_equal(Representation.cell(~Logical,~original,logical,geometry,outer,group),Tiled.zero_tile(),
    past_lane(~Logical,~original,logical,geometry,outer,group,0n,past),past_lane(~Logical,~original,logical,geometry,outer,group,1n,past),
    past_lane(~Logical,~original,logical,geometry,outer,group,2n,past),past_lane(~Logical,~original,logical,geometry,outer,group,3n,past),
    past_lane(~Logical,~original,logical,geometry,outer,group,4n,past),past_lane(~Logical,~original,logical,geometry,outer,group,5n,past),
    past_lane(~Logical,~original,logical,geometry,outer,group,6n,past),past_lane(~Logical,~original,logical,geometry,outer,group,7n,past))

# ---- The next cell when no lane needs it ----

# Whether a lane of the group inside the width moves past the first source
# cell: the lanes shift, and the first lane that moves lies inside the width.
def needed(+width: Nat,+group: Nat,+shift: Nat) -> Bool:
  Nat.is_lt(0n,shift) && Nat.is_lt(Nat.add(Nat.mul(group,8n),Nat.sub(8n,shift)),width)

# A lane inside the width that moves past the first cell needs the next one:
# without the need, the lane reads the first cell.
def moved_unused(low: Bool,+width: Nat,+group: Nat,+shift: Nat,+lane: Nat,+first: Tile.Tile8,+left: Tile.Tile8,+right: Tile.Tile8,
  +classified: {Nat.is_lt(Nat.add(lane,shift),8n) == low : Bool},+lane_small: {Nat.is_le(lane,7n) == True{} : Bool},
  +inside: {Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),width) == True{} : Bool},+unused: {needed(width,group,shift) == False{} : Bool}) ->
  {Bool.pick(F32,low,TileLaws.select_lane(Nat.add(lane,shift),first),TileLaws.select_lane(Nat.sub(Nat.add(lane,shift),8n),left)) ==
    Bool.pick(F32,low,TileLaws.select_lane(Nat.add(lane,shift),first),TileLaws.select_lane(Nat.sub(Nat.add(lane,shift),8n),right)) : F32}:
  match low:
    case True{}:
      {==}
    case False{}:
      +reached = Order.not_less(Nat.add(lane,shift),8n,classified)
      +start = Equal.trans(Bool,Nat.is_le(Nat.sub(8n,shift),lane),Nat.is_le(8n,Nat.add(shift,lane)),True{},Order.subtraction_le(8n,shift,lane),
        {%Order.commutative(lane,shift) : {Nat.is_le(8n,_) == True{} : Bool} reached : {Nat.is_le(8n,Nat.add(shift,lane)) == True{} : Bool}})
      +moved = Order.le_lt_trans(Nat.add(Nat.mul(group,8n),Nat.sub(8n,shift)),Nat.add(Nat.mul(group,8n),lane),width,
        {%Order.commutative(Nat.sub(8n,shift),Nat.mul(group,8n)) : {Nat.is_le(_,Nat.add(Nat.mul(group,8n),lane)) == True{} : Bool}
          %Order.commutative(lane,Nat.mul(group,8n)) : {Nat.is_le(Nat.add(Nat.sub(8n,shift),Nat.mul(group,8n)),_) == True{} : Bool}
          Order.add_le(Nat.sub(8n,shift),lane,Nat.mul(group,8n),start) : {Nat.is_le(Nat.add(Nat.mul(group,8n),Nat.sub(8n,shift)),Nat.add(Nat.mul(group,8n),lane)) == True{} : Bool}},
        %Order.commutative(lane,Nat.mul(group,8n)) : {Nat.is_lt(_,width) == True{} : Bool} inside)
      Empty.absurd({TileLaws.select_lane(Nat.sub(Nat.add(lane,shift),8n),left) == TileLaws.select_lane(Nat.sub(Nat.add(lane,shift),8n),right) : F32},
        WordEncoding.false_true(Equal.trans(Bool,False{},needed(width,group,shift),True{},Equal.sym(Bool,needed(width,group,shift),False{},unused),
          Boolean.and_true(Nat.is_lt(0n,shift),Nat.is_lt(Nat.add(Nat.mul(group,8n),Nat.sub(8n,shift)),width),WriteCell.high_shift(lane,shift,lane_small,classified),moved))))

# A lane of the cut cell does not depend on the next cell without the need.
def unused_kept(+width: Nat,+group: Nat,+shift: Nat,+lane: Nat,+first: Tile.Tile8,+left: Tile.Tile8,+right: Tile.Tile8,
  +shift_small: {Nat.is_le(shift,7n) == True{} : Bool},+lane_small: {Nat.is_le(lane,7n) == True{} : Bool},+unused: {needed(width,group,shift) == False{} : Bool}) ->
  {PaddedCells.kept(width,group,lane,TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),first,left))) ==
    PaddedCells.kept(width,group,lane,TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),first,right))) : F32}:
  +on_left = TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),first,left))
  +on_right = TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),first,right))
  PaddedCells.kept_agree(width,group,lane,on_left,on_right,Nat.is_lt(Nat.add(lane,Nat.mul(group,8n)),width),{==},
    hit => Equal.trans(F32,on_left,LaneShift.joined_value(shift,lane,first,left),on_right,LaneShift.joined_lane(shift,lane,first,left,shift_small,lane_small),
      Equal.trans(F32,LaneShift.joined_value(shift,lane,first,left),LaneShift.joined_value(shift,lane,first,right),on_right,
        moved_unused(Nat.is_lt(Nat.add(lane,shift),8n),width,group,shift,lane,first,left,right,{==},lane_small,hit,unused),
        Equal.sym(F32,on_right,LaneShift.joined_value(shift,lane,first,right),LaneShift.joined_lane(shift,lane,first,right,shift_small,lane_small)))))

# A lane of the cut cells, the next cell aside.
def unused_lane(+limits: Machine.Limits,+first_lane: U32,+width: U32,+group: Nat,+shift: Nat,+lane: Nat,+first: Tile.Tile8,+left: Tile.Tile8,+right: Tile.Tile8,
  +first_exact: {U32.to_nat(first_lane) == Nat.mul(group,8n) : Nat},
  +bound: {Nat.is_le(Nat.add(Nat.mul(group,8n),7n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +shift_small: {Nat.is_le(shift,7n) == True{} : Bool},+lane_small: {Nat.is_le(lane,7n) == True{} : Bool},+unused: {needed(U32.to_nat(width),group,shift) == False{} : Bool}) ->
  {Cells.tail_lane(first_lane,width,U32.from_nat(lane),TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),first,left))) ==
    Cells.tail_lane(first_lane,width,U32.from_nat(lane),TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),first,right))) : F32}:
  +w = U32.to_nat(width)
  +on_left = TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),first,left))
  +on_right = TileLaws.select_lane(lane,Tiled.joined(U32.from_nat(shift),first,right))
  +room = PaddedCells.lane_bound(limits,group,lane,lane_small,bound)
  Equal.trans(F32,Cells.tail_lane(first_lane,width,U32.from_nat(lane),on_left),PaddedCells.kept(w,group,lane,on_left),Cells.tail_lane(first_lane,width,U32.from_nat(lane),on_right),
    PaddedCells.tail_lane(on_left,limits,first_lane,width,group,lane,first_exact,lane_small,room),
    Equal.trans(F32,PaddedCells.kept(w,group,lane,on_left),PaddedCells.kept(w,group,lane,on_right),Cells.tail_lane(first_lane,width,U32.from_nat(lane),on_right),
      unused_kept(w,group,shift,lane,first,left,right,shift_small,lane_small,unused),
      Equal.sym(F32,Cells.tail_lane(first_lane,width,U32.from_nat(lane),on_right),PaddedCells.kept(w,group,lane,on_right),
        PaddedCells.tail_lane(on_right,limits,first_lane,width,group,lane,first_exact,lane_small,room))))

# Without the need, the cut cell does not depend on the next cell.
def next_unused(+limits: Machine.Limits,+first_lane: U32,+width: U32,+group: Nat,+shift: Nat,+first: Tile.Tile8,+left: Tile.Tile8,+right: Tile.Tile8,
  +first_exact: {U32.to_nat(first_lane) == Nat.mul(group,8n) : Nat},
  +bound: {Nat.is_le(Nat.add(Nat.mul(group,8n),7n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +shift_small: {Nat.is_le(shift,7n) == True{} : Bool},+unused: {needed(U32.to_nat(width),group,shift) == False{} : Bool}) ->
  {Cells.tail(Cells.Tail{first_lane,width},Tiled.joined(U32.from_nat(shift),first,left)) == Cells.tail(Cells.Tail{first_lane,width},Tiled.joined(U32.from_nat(shift),first,right)) : Tile.Tile8}:
  +w = U32.to_nat(width)
  +tail = {Cells.Tail{first_lane,width} : Cells.Tail}
  +on_left = Tiled.joined(U32.from_nat(shift),first,left)
  +on_right = Tiled.joined(U32.from_nat(shift),first,right)
  +left_parts = {Tile.Tile{Tile.lane0(on_left),Tile.lane1(on_left),Tile.lane2(on_left),Tile.lane3(on_left),Tile.lane4(on_left),Tile.lane5(on_left),Tile.lane6(on_left),Tile.lane7(on_left)} : Tile.Tile8}
  +right_parts = {Tile.Tile{Tile.lane0(on_right),Tile.lane1(on_right),Tile.lane2(on_right),Tile.lane3(on_right),Tile.lane4(on_right),Tile.lane5(on_right),Tile.lane6(on_right),Tile.lane7(on_right)} : Tile.Tile8}
  %Equal.sym(Tile.Tile8,on_left,left_parts,TileLaws.tile_eta(on_left)) : {Cells.tail(tail,_) == Cells.tail(tail,on_right) : Tile.Tile8}
  %Equal.sym(Tile.Tile8,on_right,right_parts,TileLaws.tile_eta(on_right)) : {Cells.tail(tail,left_parts) == Cells.tail(tail,_) : Tile.Tile8}
  TileLaws.tiles_equal(Cells.tail(tail,left_parts),Cells.tail(tail,right_parts),
    unused_lane(limits,first_lane,width,group,shift,0n,first,left,right,first_exact,bound,shift_small,{==},unused),
    unused_lane(limits,first_lane,width,group,shift,1n,first,left,right,first_exact,bound,shift_small,{==},unused),
    unused_lane(limits,first_lane,width,group,shift,2n,first,left,right,first_exact,bound,shift_small,{==},unused),
    unused_lane(limits,first_lane,width,group,shift,3n,first,left,right,first_exact,bound,shift_small,{==},unused),
    unused_lane(limits,first_lane,width,group,shift,4n,first,left,right,first_exact,bound,shift_small,{==},unused),
    unused_lane(limits,first_lane,width,group,shift,5n,first,left,right,first_exact,bound,shift_small,{==},unused),
    unused_lane(limits,first_lane,width,group,shift,6n,first,left,right,first_exact,bound,shift_small,{==},unused),
    unused_lane(limits,first_lane,width,group,shift,7n,first,left,right,first_exact,bound,shift_small,{==},unused))
