# What a placing needs from its shapes: a target and sources that fit the
# machine, whose segments read their axes and whose shifts leave room, give
# the gather of the placing every fact it needs.

import Base
import ../tensor.bend as Tensors
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ../tensor_layout.bend as Layout
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./tensor_components.bend as Components
import ./tensor_view_facts.bend as ViewFacts
import ./tensor_view_reads.bend as ViewReads
import ./tensor_target_rows.bend as TargetRows
import ./tensor_placing_parts.bend as Parts
import ./tensor_placing_law.bend as Law
import ./tensor_move_readback.bend as Held
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_task_reads.bend as TaskReads
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./tensor_placing_reads.bend as Reads
import ./natural_addition.bend as Addition
import ./u32_full_range.bend as FullRange
import ./tensor_placements.bend as Placements

# ---- Extents after some of them ----

# The extent after count of them is the extent at count.
def lanes_dropped(source: List<&2,U32>,count: Nat,+bound: U32,+rest: List<&2,U32>,+at: {Law.dropped(~U32,count,source) == Con{bound,rest} : List<&2,U32>}) ->
  {Grid.lanes_at(source,count) == bound : U32}:
  match source count:
    case Nil{} 0n:
      Empty.absurd({Grid.lanes_at(Nil{},0n) == bound : U32},Law.list_apart(~U32,bound,rest,at))
    case Con{+head,+tail} 0n:
      Equal.cong(List<&2,U32>,U32,values => Grid.lanes_at(values,0n),Con{head,tail},Con{bound,rest},at)
    case Nil{} 1n+ +more:
      Empty.absurd({Grid.lanes_at(Nil{},1n+more) == bound : U32},Law.list_apart(~U32,bound,rest,at))
    case Con{+head,+tail} 1n+ +more:
      lanes_dropped(tail,more,bound,rest,at)

# An extent after count of them lies below the length.
def dropped_inside(source: List<&2,U32>,count: Nat,+bound: U32,+rest: List<&2,U32>,+at: {Law.dropped(~U32,count,source) == Con{bound,rest} : List<&2,U32>}) ->
  {Nat.is_lt(count,List.length(&2,U32,source)) == True{} : Bool}:
  match source count:
    case Nil{} 0n:
      Empty.absurd({Nat.is_lt(0n,List.length(&2,U32,Nil{})) == True{} : Bool},Law.list_apart(~U32,bound,rest,at))
    case Con{+head,+tail} 0n:
      {==}
    case Nil{} 1n+ +more:
      Empty.absurd({Nat.is_lt(1n+more,List.length(&2,U32,Nil{})) == True{} : Bool},Law.list_apart(~U32,bound,rest,at))
    case Con{+head,+tail} 1n+ +more:
      dropped_inside(tail,more,bound,rest,at)

# ---- The readings of a placed source ----

# A row of readings of a placed source reads the target extents from at, has
# room at cap, reads the axes of the source and reads below its rank.
def RowFacts(+row: List<&2,Tasks.Reading>,+target: List<&2,U32>,+source: List<&2,U32>,+target_grid: Grid.Grid,+source_grid: Grid.Grid,+at: U32,+cap: U32) -> Data:
  Held.Both<TargetRows.TargetSide(row,target,target_grid,at),Held.Both<Components.ReadingsRoom(row,cap),
    Held.Both<ViewFacts.ReadsFrom(row,source,0,source_grid),Components.AxesBelow(Tasks.read_axes(row),List.length(&2,U32,source))>>>

# A reading of a placed axis has room at half a word: its extent and blocks
# fit, and its shift leaves room for the source extent.
def reading_room(+limits: Machine.Limits,+extent: U32,+block: U32,+source_block: U32,+at: U32,+shift: U32,+bound: U32,+clamped: Bool,
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},
  +block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},+block_double: {Nat.is_le(Nat.double(U32.to_nat(U32.max(block,1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +source_double: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  Components.ReadingRoom(extent,block,Tasks.Read{at,source_block,1,0,False{},1,1,shift,bound,clamped},Machine.half_word(limits)):
  +last = Nat.sub(U32.to_nat(extent),1n)
  +reach_value = Equal.trans(Nat,Nat.add(Nat.mul(1n,last),1n),Nat.add(last,1n),U32.to_nat(extent),
    Equal.cong(Nat,Nat,value => Nat.add(value,1n),Nat.mul(1n,last),last,TaskReads.one_times(last)),
    Equal.trans(Nat,Nat.add(last,1n),1n+last,U32.to_nat(extent),ViewFacts.plus_one(last),Order.positive_has_last(U32.to_nat(extent),positive)))
  Held.Both{ViewFacts.axis_room(limits,extent,block,room,block_positive,ViewFacts.block_within(limits,block,block_positive,block_double)),
    Held.Both{{==},Held.Both{{%Equal.sym(Nat,Nat.add(Nat.mul(1n,last),1n),U32.to_nat(extent),reach_value) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
        LimitBounds.within_half(limits,U32.to_nat(extent),room) : Components.ReachAt(False{},1,0,1,last,Machine.half_word(limits))},
      Held.Both{ViewFacts.two_half(limits),Held.Both{shift_room,source_double}}}}}

def row_joined(+limits: Machine.Limits,+extent: U32,+bound: U32,+shift: U32,+rest: List<&2,Tasks.Reading>,+target: List<&2,U32>,+source: List<&2,U32>,
  +target_grid: Grid.Grid,+source_grid: Grid.Grid,+clamped: Bool,+at: U32,
  +room: Components.ReadingRoom(extent,Grid.block(target_grid,at,extent),Tasks.Read{at,Grid.block(source_grid,at,bound),1,0,False{},1,1,shift,bound,clamped},Machine.half_word(limits)),
  +extent_at: ViewFacts.ExtentAt(source,0,at,bound),+below: {Nat.is_lt(U32.to_nat(at),List.length(&2,U32,source)) == True{} : Bool},
  +later: RowFacts(rest,target,source,target_grid,source_grid,(at + 1 : U32),Machine.half_word(limits))) ->
  RowFacts(Con{Tasks.Reading{extent,Grid.block(target_grid,at,extent),Tasks.Read{at,Grid.block(source_grid,at,bound),1,0,False{},1,1,shift,bound,clamped}},rest},
    Con{extent,target},source,target_grid,source_grid,at,Machine.half_word(limits)):
  Held.Both{+side,Held.Both{+rooms,Held.Both{+froms,+belows}}} = later
  Held.Both{Held.Both{{==},Held.Both{{==},side}},Held.Both{Held.Both{room,rooms},Held.Both{Held.Both{Held.Both{{==},extent_at},froms},Held.Both{below,belows}}}}

# The readings of a source placed from at on, at position position of the
# whole source: every reading reads the target extent, has room, reads its
# source axis and reads below the rank.
def placed_row(segments: List<&2,List<&2,Tasks.Segment>>,target: List<&2,U32>,source_rest: List<&2,U32>,shifts: List<&2,U32>,+source: List<&2,U32>,
  +target_grid: Grid.Grid,+source_grid: Grid.Grid,+clamped: Bool,+at: U32,+position: Nat,+limits: Machine.Limits,
  +axes: Parts.SourceAxes(segments,target,source_rest,shifts,clamped,Machine.half_word(limits)),
  +target_blocks: ViewReads.BlocksIn(target_grid,target,at,Machine.half_word(limits)),+source_blocks: ViewReads.BlocksIn(source_grid,source_rest,at,Machine.half_word(limits)),
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+dropped_at: {Law.dropped(~U32,position,source) == source_rest : List<&2,U32>},
  +rank_room: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  RowFacts(Layout.placed_readings(target,source_rest,shifts,target_grid,source_grid,clamped,at),target,source,target_grid,source_grid,at,Machine.half_word(limits)):
  match segments target source_rest shifts:
    case Con{+own,+segments} Con{+extent,+target} Con{+bound,+source_rest} Con{+shift,+shifts}:
      Held.Both{+segments_in,Held.Both{+reading,Held.Both{+shift_room,+later}}} = axes
      Held.Both{+block_positive,Held.Both{+block_double,+blocks_rest}} = target_blocks
      Held.Both{+source_positive,Held.Both{+source_double,+source_rest_blocks}} = source_blocks
      +cap = U32.to_nat(Machine.capacity(limits))
      +length = List.length(&2,U32,source)
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(target)
      +whole = Nat.mul(size,inner)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +size_room = Order.transitive(size,whole,cap,RowMajor.product_covers(size,inner,inner_positive),room)
      +inside = dropped_inside(source,position,bound,source_rest,dropped_at)
      +position_room = Order.transitive(position,length,cap,Bounds.strict_le(position,length,inside),rank_room)
      +next = Components.next_word(at,position,Machine.capacity(limits),LimitBounds.capacity_fits(limits),exact,
        Order.transitive(1n+position,length,cap,Order.strict_follow(position,length,inside),rank_room))
      +at_word = Arithmetic.to_nat_injective(U32.from_nat(position),at,
        Equal.trans(Nat,U32.to_nat(U32.from_nat(position)),position,U32.to_nat(at),
          Bounds.roundtrip(position,Machine.capacity(limits),LimitBounds.capacity_fits(limits),position_room),Equal.sym(Nat,U32.to_nat(at),position,exact)))
      +extent_at = {%lanes_dropped(source,position,bound,source_rest,dropped_at) : ViewFacts.ExtentAt(source,0,at,_)
        {%at_word : ViewFacts.ExtentAt(source,0,_,Grid.lanes_at(source,position))
          ViewFacts.extent_index(source,position,0,0n,limits,{==},rank_room,position_room,inside) :
          ViewFacts.ExtentAt(source,0,at,Grid.lanes_at(source,position))} : ViewFacts.ExtentAt(source,0,at,bound)}
      +below = {%Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_lt(_,length) == True{} : Bool} inside : {Nat.is_lt(U32.to_nat(at),length) == True{} : Bool}}
      row_joined(limits,extent,bound,shift,Layout.placed_readings(target,source_rest,shifts,target_grid,source_grid,clamped,(at + 1 : U32)),target,source,target_grid,source_grid,clamped,at,
        reading_room(limits,extent,Grid.block(target_grid,at,extent),Grid.block(source_grid,at,bound),at,shift,bound,clamped,size_room,size_positive,
          block_positive,block_double,source_double,shift_room),
        extent_at,below,
        placed_row(segments,target,source_rest,shifts,source,target_grid,source_grid,clamped,(at + 1 : U32),1n+position,limits,later,blocks_rest,source_rest_blocks,
          Order.transitive(inner,whole,cap,DivisionOrder.le_multiple(size,inner,size_positive),room),inner_positive,next,
          Law.dropped_next(~U32,position,source,bound,source_rest,dropped_at),rank_room))
    case Nil{} Nil{} Nil{} Nil{}:
      Held.Both{Unit{},Held.Both{Unit{},Held.Both{Unit{},Unit{}}}}
    case Con{+own,+segments} Nil{} +source_rest +shifts:
      Empty.absurd(RowFacts(Layout.placed_readings(Nil{},source_rest,shifts,target_grid,source_grid,clamped,at),Nil{},source,target_grid,source_grid,at,Machine.half_word(limits)),axes)
    case Con{+own,+segments} Con{+extent,+target} Nil{} +shifts:
      Empty.absurd(RowFacts(Layout.placed_readings(Con{extent,target},Nil{},shifts,target_grid,source_grid,clamped,at),Con{extent,target},source,target_grid,source_grid,at,Machine.half_word(limits)),axes)
    case Con{+own,+segments} Con{+extent,+target} Con{+bound,+source_rest} Nil{}:
      Empty.absurd(RowFacts(Layout.placed_readings(Con{extent,target},Con{bound,source_rest},Nil{},target_grid,source_grid,clamped,at),Con{extent,target},source,target_grid,source_grid,at,Machine.half_word(limits)),axes)
    case Nil{} Con{+extent,+target} +source_rest +shifts:
      Empty.absurd(RowFacts(Layout.placed_readings(Con{extent,target},source_rest,shifts,target_grid,source_grid,clamped,at),Con{extent,target},source,target_grid,source_grid,at,Machine.half_word(limits)),axes)
    case Nil{} Nil{} Con{+bound,+source_rest} +shifts:
      Empty.absurd(RowFacts(Layout.placed_readings(Nil{},Con{bound,source_rest},shifts,target_grid,source_grid,clamped,at),Nil{},source,target_grid,source_grid,at,Machine.half_word(limits)),axes)
    case Nil{} Nil{} Nil{} Con{+shift,+shifts}:
      Empty.absurd(RowFacts(Layout.placed_readings(Nil{},Nil{},Con{shift,shifts},target_grid,source_grid,clamped,at),Nil{},source,target_grid,source_grid,at,Machine.half_word(limits)),axes)

# ---- Every source of a placing ----

# What every source of a placing gives: a shape that fits the machine, and
# segments that read its axes with shifts that leave room, the lists ending
# together.
def SourcesShaped(+limits: Machine.Limits,+target: List<&2,U32>,segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,
  shifts: List<&2,List<&2,U32>>,clamps: List<&2,Bool>) -> Data:
  match segments sources shifts clamps:
    case Con{own,segments} Con{+source,sources} Con{+shift,shifts} Con{+clamped,clamps}:
      Held.Both<Components.ShapeOk(limits,source),Held.Both<Parts.SourceAxes(own,target,source,shift,clamped,Machine.half_word(limits)),SourcesShaped(limits,target,segments,sources,shifts,clamps)>>
    case Nil{} Nil{} Nil{} Nil{}:
      Unit
    case segments sources shifts clamps:
      Empty

# The rows of the sources read the target, and every source has its facts.
def PlacedRows(+limits: Machine.Limits,+target: List<&2,U32>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,
  +shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>) -> Data:
  Held.Both<TargetRows.RowsRead(Law.placing_rows(target,sources,shifts,clamps),target,Tensors.grid_of(target),0,Machine.half_word(limits)),Law.SourcesFacts(limits,target,segments,sources,shifts,clamps)>

def source_joined(+limits: Machine.Limits,+target: List<&2,U32>,+own: List<&2,List<&2,Tasks.Segment>>,+source: List<&2,U32>,+shift: List<&2,U32>,+clamped: Bool,
  +segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,+shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,
  +source_ok: Components.ShapeOk(limits,source),+source_axes: Parts.SourceAxes(own,target,source,shift,clamped,Machine.half_word(limits)),
  +row: RowFacts(Layout.placed_readings(target,source,shift,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0),target,source,Tensors.grid_of(target),Tensors.grid_of(source),0,Machine.half_word(limits)),
  +later: PlacedRows(limits,target,segments,sources,shifts,clamps)) ->
  PlacedRows(limits,target,Con{own,segments},Con{source,sources},Con{shift,shifts},Con{clamped,clamps}):
  Held.Both{+source_room,Held.Both{+source_positive,+source_rank}} = source_ok
  Held.Both{+side,Held.Both{+rooms,Held.Both{+froms,+belows}}} = row
  Held.Both{+rows_read,+facts_rest} = later
  +grid = Tensors.grid_of(source)
  +shape_ok = {Held.Both{source_room,Held.Both{source_positive,source_rank}} : Components.ShapeOk(limits,source)}
  +blocks = ViewFacts.grid_blocks(limits,source,Grid.piece_target(),shape_ok)
  +placed = Layout.placed_readings(target,source,shift,Tensors.grid_of(target),grid,clamped,0)
  Held.Both{Held.Both{side,Held.Both{rooms,rows_read}},
    Held.Both{shape_ok,Held.Both{Held.Both{rooms,Held.Both{ViewFacts.reads_axes(source,placed,limits,grid,0,froms,blocks,source_room,source_positive),belows}},
      Held.Both{blocks,Held.Both{source_axes,facts_rest}}}}}

# Source after source, every source of a placing has its facts and its rows
# read the target.
def sources_placed(segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,clamps: List<&2,Bool>,
  +limits: Machine.Limits,+target: List<&2,U32>,+target_ok: Components.ShapeOk(limits,target),+shaped: SourcesShaped(limits,target,segments,sources,shifts,clamps)) ->
  PlacedRows(limits,target,segments,sources,shifts,clamps):
  match segments sources shifts clamps:
    case Con{+own,+segments} Con{+source,+sources} Con{+shift,+shifts} Con{+clamped,+clamps}:
      Held.Both{+room,Held.Both{+positive,+rank_room}} = target_ok
      Held.Both{Held.Both{+source_room,Held.Both{+source_positive,+source_rank}},Held.Both{+source_axes,+rest}} = shaped
      +target_shape = {Held.Both{room,Held.Both{positive,rank_room}} : Components.ShapeOk(limits,target)}
      +source_shape = {Held.Both{source_room,Held.Both{source_positive,source_rank}} : Components.ShapeOk(limits,source)}
      source_joined(limits,target,own,source,shift,clamped,segments,sources,shifts,clamps,source_shape,source_axes,
        placed_row(own,target,source,shift,source,Tensors.grid_of(target),Tensors.grid_of(source),clamped,0,0n,limits,source_axes,
          ViewFacts.grid_blocks(limits,target,Grid.piece_target(),target_shape),ViewFacts.grid_blocks(limits,source,Grid.piece_target(),source_shape),
          room,positive,{==},{==},source_rank),
        sources_placed(segments,sources,shifts,clamps,limits,target,target_shape,rest))
    case Nil{} Nil{} Nil{} Nil{}:
      Held.Both{Unit{},Unit{}}
    case Con{+own,+segments} Nil{} +shifts +clamps:
      Empty.absurd(PlacedRows(limits,target,Con{own,segments},Nil{},shifts,clamps),shaped)
    case Con{+own,+segments} Con{+source,+sources} Nil{} +clamps:
      Empty.absurd(PlacedRows(limits,target,Con{own,segments},Con{source,sources},Nil{},clamps),shaped)
    case Con{+own,+segments} Con{+source,+sources} Con{+shift,+shifts} Nil{}:
      Empty.absurd(PlacedRows(limits,target,Con{own,segments},Con{source,sources},Con{shift,shifts},Nil{}),shaped)
    case Nil{} Con{+source,+sources} +shifts +clamps:
      Empty.absurd(PlacedRows(limits,target,Nil{},Con{source,sources},shifts,clamps),shaped)
    case Nil{} Nil{} Con{+shift,+shifts} +clamps:
      Empty.absurd(PlacedRows(limits,target,Nil{},Nil{},Con{shift,shifts},clamps),shaped)
    case Nil{} Nil{} Nil{} Con{+clamped,+clamps}:
      Empty.absurd(PlacedRows(limits,target,Nil{},Nil{},Nil{},Con{clamped,clamps}),shaped)

# A placing reads one row for every source.
def rows_length(segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,sources: List<&2,List<&2,U32>>,shifts: List<&2,List<&2,U32>>,clamps: List<&2,Bool>,
  +limits: Machine.Limits,+target: List<&2,U32>,+shaped: SourcesShaped(limits,target,segments,sources,shifts,clamps)) ->
  {List.length(&2,List<&2,Tasks.Reading>,Law.placing_rows(target,sources,shifts,clamps)) == List.length(&2,List<&2,U32>,sources) : Nat}:
  match segments sources shifts clamps:
    case Con{+own,+segments} Con{+source,+sources} Con{+shift,+shifts} Con{+clamped,+clamps}:
      Held.Both{+source_ok,Held.Both{+source_axes,+rest}} = shaped
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,List<&2,Tasks.Reading>,Law.placing_rows(target,sources,shifts,clamps)),List.length(&2,List<&2,U32>,sources),
        rows_length(segments,sources,shifts,clamps,limits,target,rest))
    case Nil{} Nil{} Nil{} Nil{}:
      {==}
    case Con{+own,+segments} Nil{} +shifts +clamps:
      Empty.absurd({List.length(&2,List<&2,Tasks.Reading>,Law.placing_rows(target,Nil{},shifts,clamps)) == List.length(&2,List<&2,U32>,Nil{}) : Nat},shaped)
    case Con{+own,+segments} Con{+source,+sources} Nil{} +clamps:
      Empty.absurd({List.length(&2,List<&2,Tasks.Reading>,Law.placing_rows(target,Con{source,sources},Nil{},clamps)) == List.length(&2,List<&2,U32>,Con{source,sources}) : Nat},shaped)
    case Con{+own,+segments} Con{+source,+sources} Con{+shift,+shifts} Nil{}:
      Empty.absurd({List.length(&2,List<&2,Tasks.Reading>,Law.placing_rows(target,Con{source,sources},Con{shift,shifts},Nil{})) == List.length(&2,List<&2,U32>,Con{source,sources}) : Nat},shaped)
    case Nil{} Con{+source,+sources} +shifts +clamps:
      Empty.absurd({List.length(&2,List<&2,Tasks.Reading>,Law.placing_rows(target,Con{source,sources},shifts,clamps)) == List.length(&2,List<&2,U32>,Con{source,sources}) : Nat},shaped)
    case Nil{} Nil{} Con{+shift,+shifts} +clamps:
      Empty.absurd({List.length(&2,List<&2,Tasks.Reading>,Law.placing_rows(target,Nil{},Con{shift,shifts},clamps)) == List.length(&2,List<&2,U32>,Nil{}) : Nat},shaped)
    case Nil{} Nil{} Nil{} Con{+clamped,+clamps}:
      Empty.absurd({List.length(&2,List<&2,Tasks.Reading>,Law.placing_rows(target,Nil{},Nil{},Con{clamped,clamps})) == List.length(&2,List<&2,U32>,Nil{}) : Nat},shaped)

# ---- The facts of a placing ----

def facts_ranked(+limits: Machine.Limits,+target: List<&2,U32>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,
  +shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,+target_ok: Components.ShapeOk(limits,target),
  +rank: {Tasks.target_rank(Law.placing_rows(target,sources,shifts,clamps)) == List.length(&2,U32,target) : Nat},
  +components: Held.Both<Components.ComponentsOk(Law.placing_rows(target,sources,shifts,clamps),List.length(&2,U32,target),Machine.half_word(limits)),
    {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(List.length(&2,U32,target),Law.placing_rows(target,sources,shifts,clamps)))) == True{} : Bool}>,
  +limits_at: Components.LimitsAt(Law.placing_rows(target,sources,shifts,clamps),List.length(&2,U32,target),target,Tensors.grid_of(target),0),
  +sources_facts: Law.SourcesFacts(limits,target,segments,sources,shifts,clamps)) -> Law.PlacingFacts(limits,target,segments,sources,shifts,clamps):
  Held.Both{+ok,+count_positive} = components
  +rows = Law.placing_rows(target,sources,shifts,clamps)
  %Equal.sym(Nat,Tasks.target_rank(rows),List.length(&2,U32,target),rank) :
    Held.Both<Components.ShapeOk(limits,target),Held.Both<Components.ComponentsOk(rows,_,Machine.half_word(limits)),
      Held.Both<{Nat.is_lt(0n,Components.count_nat(Components.raw_lists(_,rows))) == True{} : Bool},
        Held.Both<Components.LimitsAt(rows,_,target,Tensors.grid_of(target),0),Law.SourcesFacts(limits,target,segments,sources,shifts,clamps)>>>>
  Held.Both{target_ok,Held.Both{ok,Held.Both{count_positive,Held.Both{limits_at,sources_facts}}}}

def facts_joined(+limits: Machine.Limits,+target: List<&2,U32>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,
  +shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,+target_ok: Components.ShapeOk(limits,target),+placed: PlacedRows(limits,target,segments,sources,shifts,clamps),
  +rows_nonempty: {Nat.is_lt(0n,List.length(&2,List<&2,Tasks.Reading>,Law.placing_rows(target,sources,shifts,clamps))) == True{} : Bool},
  +rows_room: {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,Law.placing_rows(target,sources,shifts,clamps)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  Law.PlacingFacts(limits,target,segments,sources,shifts,clamps):
  Held.Both{+room,Held.Both{+positive,+rank_room}} = target_ok
  Held.Both{+rows_read,+sources_facts} = placed
  +rows = Law.placing_rows(target,sources,shifts,clamps)
  +grid = Tensors.grid_of(target)
  +half = Machine.half_word(limits)
  facts_ranked(limits,target,segments,sources,shifts,clamps,{Held.Both{room,Held.Both{positive,rank_room}} : Components.ShapeOk(limits,target)},
    TargetRows.rows_rank(rows,target,grid,0,half,rows_read,rows_nonempty),
    TargetRows.rows_components(rows,target,grid,limits,rows_read,rows_nonempty,rows_room,room,positive),
    TargetRows.rows_limits(target,rows,grid,0,half,rows_read,rows_nonempty),sources_facts)

# A target and at least one source that fit the machine, whose segments read
# their axes with shifts that leave room, and no more sources than half a
# word give a placing every fact it needs.
def placing_facts(+limits: Machine.Limits,+target: List<&2,U32>,+segments: List<&2,List<&2,List<&2,Tasks.Segment>>>,+sources: List<&2,List<&2,U32>>,
  +shifts: List<&2,List<&2,U32>>,+clamps: List<&2,Bool>,+target_ok: Components.ShapeOk(limits,target),+shaped: SourcesShaped(limits,target,segments,sources,shifts,clamps),
  +nonempty: {Nat.is_lt(0n,List.length(&2,List<&2,U32>,sources)) == True{} : Bool},+count_room: {Nat.is_le(List.length(&2,List<&2,U32>,sources),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) ->
  Law.PlacingFacts(limits,target,segments,sources,shifts,clamps):
  +rows = Law.placing_rows(target,sources,shifts,clamps)
  +back = Equal.sym(Nat,List.length(&2,List<&2,Tasks.Reading>,rows),List.length(&2,List<&2,U32>,sources),rows_length(segments,sources,shifts,clamps,limits,target,shaped))
  facts_joined(limits,target,segments,sources,shifts,clamps,target_ok,sources_placed(segments,sources,shifts,clamps,limits,target,target_ok,shaped),
    {%back : {Nat.is_lt(0n,_) == True{} : Bool} nonempty : {Nat.is_lt(0n,List.length(&2,List<&2,Tasks.Reading>,rows)) == True{} : Bool}},
    {%back : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool} count_room : {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,rows),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}})

# ---- Sources placed whole ----

# Every axis of a source fits in the target from its shift on, the lists
# ending together.
def PlacedIn(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>) -> Data:
  match target source shifts:
    case Con{+total,target} Con{+bound,source} Con{+shift,shifts}:
      Held.Both<{Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(total)) == True{} : Bool},PlacedIn(target,source,shifts)>
    case Nil{} Nil{} Nil{}:
      Unit
    case target source shifts:
      Empty

# A source placed whole has the rank of the target.
def placed_length(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+within: PlacedIn(target,source,shifts)) ->
  {List.length(&2,U32,source) == List.length(&2,U32,target) : Nat}:
  match target source shifts:
    case Con{+total,+target} Con{+bound,+source} Con{+shift,+shifts}:
      Held.Both{+fit,+later} = within
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,U32,source),List.length(&2,U32,target),placed_length(target,source,shifts,later))
    case Nil{} Nil{} Nil{}:
      {==}
    case Con{+total,+target} Nil{} +shifts:
      Empty.absurd({List.length(&2,U32,Nil{}) == List.length(&2,U32,Con{total,target}) : Nat},within)
    case Con{+total,+target} Con{+bound,+source} Nil{}:
      Empty.absurd({List.length(&2,U32,Con{bound,source}) == List.length(&2,U32,Con{total,target}) : Nat},within)
    case Nil{} Con{+bound,+source} +shifts:
      Empty.absurd({List.length(&2,U32,Con{bound,source}) == List.length(&2,U32,Nil{}) : Nat},within)
    case Nil{} Nil{} Con{+shift,+shifts}:
      Empty.absurd({List.length(&2,U32,Nil{}) == List.length(&2,U32,Nil{}) : Nat},within)

# A target that holds a source of cells has cells.
def placed_positive(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+within: PlacedIn(target,source,shifts),
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool}) -> {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool}:
  match target source shifts:
    case Con{+total,+target} Con{+bound,+source} Con{+shift,+shifts}:
      Held.Both{+fit,+later} = within
      +size = U32.to_nat(bound)
      +inner = ShapeLaws.elements(source)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      Order.product_positive(U32.to_nat(total),ShapeLaws.elements(target),
        Order.lt_le(0n,size,U32.to_nat(total),size_positive,Order.transitive(size,Nat.add(U32.to_nat(shift),size),U32.to_nat(total),Order.right_le(U32.to_nat(shift),size),fit)),
        placed_positive(target,source,shifts,later,DivisionOrder.right_factor_positive(size,inner,positive)))
    case Nil{} Nil{} Nil{}:
      {==}
    case Con{+total,+target} Nil{} +shifts:
      Empty.absurd({Nat.is_lt(0n,ShapeLaws.elements(Con{total,target})) == True{} : Bool},within)
    case Con{+total,+target} Con{+bound,+source} Nil{}:
      Empty.absurd({Nat.is_lt(0n,ShapeLaws.elements(Con{total,target})) == True{} : Bool},within)
    case Nil{} Con{+bound,+source} +shifts:
      Empty.absurd({Nat.is_lt(0n,ShapeLaws.elements(Nil{})) == True{} : Bool},within)
    case Nil{} Nil{} Con{+shift,+shifts}:
      Empty.absurd({Nat.is_lt(0n,ShapeLaws.elements(Nil{})) == True{} : Bool},within)

# A source placed whole reads every axis through one segment from its shift
# on, which fits half a word.
def placed_axes(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+limits: Machine.Limits,+within: PlacedIn(target,source,shifts),
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool}) -> Parts.SourceAxes(Layout.placed_segments(source,shifts),target,source,shifts,False{},Machine.half_word(limits)):
  match target source shifts:
    case Con{+total,+target} Con{+bound,+source} Con{+shift,+shifts}:
      Held.Both{+fit,+later} = within
      +cap = U32.to_nat(Machine.capacity(limits))
      +size = U32.to_nat(total)
      +inner = ShapeLaws.elements(target)
      +whole = Nat.mul(size,inner)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +size_room = Order.transitive(size,whole,cap,RowMajor.product_covers(size,inner,inner_positive),room)
      +sum = Nat.add(U32.to_nat(shift),U32.to_nat(bound))
      +shift_room = LimitBounds.within_half(limits,sum,Order.transitive(sum,size,cap,fit,size_room))
      Held.Both{Held.Both{shift_room,Unit{}},Held.Both{Held.Both{{Reads.Middle{{==},{==},Order.reflexive(U32.to_nat(bound))} : Reads.SegmentRead<shift,bound,0,1,shift,bound,False{}>},Unit{}},
        Held.Both{shift_room,placed_axes(target,source,shifts,limits,later,Order.transitive(inner,whole,cap,DivisionOrder.le_multiple(size,inner,size_positive),room),inner_positive)}}}
    case Nil{} Nil{} Nil{}:
      Unit{}
    case Con{+total,+target} Nil{} +shifts:
      Empty.absurd(Parts.SourceAxes(Layout.placed_segments(Nil{},shifts),Con{total,target},Nil{},shifts,False{},Machine.half_word(limits)),within)
    case Con{+total,+target} Con{+bound,+source} Nil{}:
      Empty.absurd(Parts.SourceAxes(Layout.placed_segments(Con{bound,source},Nil{}),Con{total,target},Con{bound,source},Nil{},False{},Machine.half_word(limits)),within)
    case Nil{} Con{+bound,+source} +shifts:
      Empty.absurd(Parts.SourceAxes(Layout.placed_segments(Con{bound,source},shifts),Nil{},Con{bound,source},shifts,False{},Machine.half_word(limits)),within)
    case Nil{} Nil{} Con{+shift,+shifts}:
      Empty.absurd(Parts.SourceAxes(Layout.placed_segments(Nil{},Con{shift,shifts}),Nil{},Nil{},Con{shift,shifts},False{},Machine.half_word(limits)),within)

# ---- Sources placed with their edges repeated ----

def le_one_more(+value: Nat) -> {Nat.is_le(value,1n+value) == True{} : Bool}:
  Order.successor(value,value,Order.reflexive(value))

# The indices after the source along an axis: they start where it ends and
# run to the end of the target.
def after_room(+total: U32,+bound: U32,+shift: U32,+fit: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(total)) == True{} : Bool}) ->
  Held.Both<{U32.to_nat((shift + bound : U32)) == Nat.add(U32.to_nat(shift),U32.to_nat(bound)) : Nat},
    {Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))) == U32.to_nat(total) : Nat}>:
  +s = U32.to_nat(shift)
  +b = U32.to_nat(bound)
  +t = U32.to_nat(total)
  +s_le = Order.transitive(s,Nat.add(s,b),t,Order.le_add(s,b),fit)
  +sum_word = FullRange.add(shift,bound,total,fit)
  +first = FullRange.subtract(total,shift,total,s_le,Order.sub_le(t,s))
  +restored = Order.restore_shift(s,t,s_le)
  +rest_le = {%Equal.sym(Nat,Nat.add(s,Nat.sub(t,s)),t,restored) : {Nat.is_le(Nat.add(s,b),_) == True{} : Bool} fit : {Nat.is_le(Nat.add(s,b),Nat.add(s,Nat.sub(t,s))) == True{} : Bool}}
  +b_le = Equal.trans(Bool,Nat.is_le(b,Nat.sub(t,s)),Nat.is_le(Nat.add(s,b),Nat.add(s,Nat.sub(t,s))),True{},
    Equal.sym(Bool,Nat.is_le(Nat.add(s,b),Nat.add(s,Nat.sub(t,s))),Nat.is_le(b,Nat.sub(t,s)),Order.le_offset(s,b,Nat.sub(t,s))),rest_le)
  +second = FullRange.subtract((total - shift : U32),bound,total,
    {%Equal.sym(Nat,U32.to_nat((total - shift : U32)),Nat.sub(t,s),first) : {Nat.is_le(b,_) == True{} : Bool} b_le : {Nat.is_le(b,U32.to_nat((total - shift : U32))) == True{} : Bool}},
    {%Equal.sym(Nat,U32.to_nat((total - shift : U32)),Nat.sub(t,s),first) : {Nat.is_le(Nat.sub(_,b),t) == True{} : Bool}
      Order.transitive(Nat.sub(Nat.sub(t,s),b),Nat.sub(t,s),t,Order.sub_le(Nat.sub(t,s),b),Order.sub_le(t,s)) :
      {Nat.is_le(Nat.sub(U32.to_nat((total - shift : U32)),b),t) == True{} : Bool}})
  +rest_exact = Equal.trans(Nat,U32.to_nat((total - shift - bound : U32)),Nat.sub(U32.to_nat((total - shift : U32)),b),Nat.sub(Nat.sub(t,s),b),second,
    Equal.cong(Nat,Nat,value => Nat.sub(value,b),U32.to_nat((total - shift : U32)),Nat.sub(t,s),first))
  Held.Both{sum_word,
    {%Equal.sym(Nat,U32.to_nat((shift + bound : U32)),Nat.add(s,b),sum_word) : {Nat.add(_,U32.to_nat((total - shift - bound : U32))) == t : Nat}
      %Equal.sym(Nat,U32.to_nat((total - shift - bound : U32)),Nat.sub(Nat.sub(t,s),b),rest_exact) : {Nat.add(Nat.add(s,b),_) == t : Nat}
      Equal.trans(Nat,Nat.add(Nat.add(s,b),Nat.sub(Nat.sub(t,s),b)),Nat.add(s,Nat.add(b,Nat.sub(Nat.sub(t,s),b))),t,Addition.associative(s,b,Nat.sub(Nat.sub(t,s),b)),
        Equal.trans(Nat,Nat.add(s,Nat.add(b,Nat.sub(Nat.sub(t,s),b))),Nat.add(s,Nat.sub(t,s)),t,
          Equal.cong(Nat,Nat,value => Nat.add(s,value),Nat.add(b,Nat.sub(Nat.sub(t,s),b)),Nat.sub(t,s),Order.restore_shift(b,Nat.sub(t,s),b_le)),restored)) :
      {Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))) == t : Nat}}}

def edge_axis(+limits: Machine.Limits,+total: U32,+bound: U32,+shift: U32,
  +fit: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(total)) == True{} : Bool},
  +total_room: {Nat.is_le(U32.to_nat(total),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(bound)) == True{} : Bool},
  +after: Held.Both<{U32.to_nat((shift + bound : U32)) == Nat.add(U32.to_nat(shift),U32.to_nat(bound)) : Nat},
    {Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))) == U32.to_nat(total) : Nat}>) ->
  Held.Both<Placements.SegmentsIn([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],Machine.half_word(limits)),
    Parts.SegmentsReading([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],shift,bound,True{})>:
  Held.Both{+sum_word,+whole} = after
  +s = U32.to_nat(shift)
  +b = U32.to_nat(bound)
  +sum = Nat.add(s,b)
  +sum_room = LimitBounds.within_half(limits,sum,Order.transitive(sum,U32.to_nat(total),U32.to_nat(Machine.capacity(limits)),fit,total_room))
  +last = FullRange.subtract(bound,1,bound,Order.positive_one(b,positive),Order.sub_le(b,1n))
  Held.Both{Held.Both{Order.transitive(s,sum,U32.to_nat(Machine.half_word(limits)),Order.le_add(s,b),sum_room),Held.Both{sum_room,
      Held.Both{{%Equal.sym(Nat,Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))),U32.to_nat(total),whole) : {Nat.is_le(_,U32.to_nat(Machine.half_word(limits))) == True{} : Bool}
          LimitBounds.within_half(limits,U32.to_nat(total),total_room) :
          {Nat.is_le(Nat.add(U32.to_nat((shift + bound : U32)),U32.to_nat((total - shift - bound : U32))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}},Unit{}}}},
    Held.Both{{Reads.Before{{==},{==},{==},positive,le_one_more(s)} : Reads.SegmentRead<0,shift,0,0,shift,bound,True{}>},
      Held.Both{{Reads.Middle{{==},{==},Order.reflexive(b)} : Reads.SegmentRead<shift,bound,0,1,shift,bound,True{}>},
        Held.Both{{Reads.After{{==},{==},
            {%Equal.sym(Nat,U32.to_nat((bound - 1 : U32)),Nat.sub(b,1n),last) : {Nat.add(_,1n) == b : Nat}
              Equal.trans(Nat,Nat.add(Nat.sub(b,1n),1n),1n+Nat.sub(b,1n),b,ViewFacts.plus_one(Nat.sub(b,1n)),Order.positive_has_last(b,positive)) :
              {Nat.add(U32.to_nat((bound - 1 : U32)),1n) == b : Nat}},
            {%Equal.sym(Nat,U32.to_nat((shift + bound : U32)),sum,sum_word) : {Nat.is_le(sum,1n+_) == True{} : Bool} le_one_more(sum) :
              {Nat.is_le(sum,1n+U32.to_nat((shift + bound : U32))) == True{} : Bool}}} :
          Reads.SegmentRead<(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0,shift,bound,True{}>},Unit{}}}}}

def edge_joined(+limits: Machine.Limits,+total: U32,+bound: U32,+shift: U32,+target: List<&2,U32>,+source: List<&2,U32>,+shifts: List<&2,U32>,
  +axis: Held.Both<Placements.SegmentsIn([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],Machine.half_word(limits)),
    Parts.SegmentsReading([Tasks.Segment{0,shift,0,0},Tasks.Segment{shift,bound,0,1},Tasks.Segment{(shift + bound : U32),(total - shift - bound : U32),(bound - 1 : U32),0}],shift,bound,True{})>,
  +shift_room: {Nat.is_le(Nat.add(U32.to_nat(shift),U32.to_nat(bound)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +later: Parts.SourceAxes(Layout.edge_segments(target,source,shifts),target,source,shifts,True{},Machine.half_word(limits))) -> Parts.SourceAxes(Layout.edge_segments(Con{total,target},Con{bound,source},Con{shift,shifts}),Con{total,target},Con{bound,source},Con{shift,shifts},True{},Machine.half_word(limits)):
  Held.Both{+segments_in,+reading} = axis
  Held.Both{segments_in,Held.Both{reading,Held.Both{shift_room,later}}}

# A source placed with its edges repeated reads every axis through the
# indices before it, its own and those after it, which fit half a word.
def edge_axes(target: List<&2,U32>,source: List<&2,U32>,shifts: List<&2,U32>,+limits: Machine.Limits,+within: PlacedIn(target,source,shifts),
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},
  +source_positive: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool}) -> Parts.SourceAxes(Layout.edge_segments(target,source,shifts),target,source,shifts,True{},Machine.half_word(limits)):
  match target source shifts:
    case Con{+total,+target} Con{+bound,+source} Con{+shift,+shifts}:
      Held.Both{+fit,+later} = within
      +cap = U32.to_nat(Machine.capacity(limits))
      +size = U32.to_nat(total)
      +inner = ShapeLaws.elements(target)
      +whole = Nat.mul(size,inner)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +size_room = Order.transitive(size,whole,cap,RowMajor.product_covers(size,inner,inner_positive),room)
      +b = U32.to_nat(bound)
      +source_inner = ShapeLaws.elements(source)
      +sum = Nat.add(U32.to_nat(shift),b)
      +shift_room = LimitBounds.within_half(limits,sum,Order.transitive(sum,size,cap,fit,size_room))
      edge_joined(limits,total,bound,shift,target,source,shifts,
        edge_axis(limits,total,bound,shift,fit,size_room,RowMajor.left_factor_positive(b,source_inner,source_positive),after_room(total,bound,shift,fit)),shift_room,
        edge_axes(target,source,shifts,limits,later,Order.transitive(inner,whole,cap,DivisionOrder.le_multiple(size,inner,size_positive),room),inner_positive,
          DivisionOrder.right_factor_positive(b,source_inner,source_positive)))
    case Nil{} Nil{} Nil{}:
      Unit{}
    case Con{+total,+target} Nil{} +shifts:
      Empty.absurd(Parts.SourceAxes(Layout.edge_segments(Con{total,target},Nil{},shifts),Con{total,target},Nil{},shifts,True{},Machine.half_word(limits)),within)
    case Con{+total,+target} Con{+bound,+source} Nil{}:
      Empty.absurd(Parts.SourceAxes(Layout.edge_segments(Con{total,target},Con{bound,source},Nil{}),Con{total,target},Con{bound,source},Nil{},True{},Machine.half_word(limits)),within)
    case Nil{} Con{+bound,+source} +shifts:
      Empty.absurd(Parts.SourceAxes(Layout.edge_segments(Nil{},Con{bound,source},shifts),Nil{},Con{bound,source},shifts,True{},Machine.half_word(limits)),within)
    case Nil{} Nil{} Con{+shift,+shifts}:
      Empty.absurd(Parts.SourceAxes(Layout.edge_segments(Nil{},Nil{},Con{shift,shifts}),Nil{},Nil{},Con{shift,shifts},True{},Machine.half_word(limits)),within)
