# The facts a lane write of a part into the target cells needs: the rows of
# the target and of the part read the target extents with room at half a
# word, the groups axis of the part reads eight lanes a group, and the
# components and cells fit the machine.

import Base
import ../tensor.bend as Tensors
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ../tensor_tiled.bend as Tiled
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./tensor_move_readback.bend as Held
import ./tensor_components.bend as Components
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_view_facts.bend as ViewFacts
import ./tensor_view_reads.bend as ViewReads
import ./tensor_target_rows.bend as TargetRows
import ./tensor_placing_facts.bend as PlacingFacts
import ./tensor_placing_law.bend as PlacingLaw
import ./tensor_lanes_write_law.bend as Law
import ./tensor_lanes_write_gather.bend as Gather
import ./tensor_lanes_write_part.bend as Part
import ./tensor_tiled_public.bend as Public
import ./tensor_relane_facts.bend as RelaneFacts
import ./tensor_relane_values.bend as Values
import ./tensor_tiled_values.bend as TiledValues
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./bounded_u32_arithmetic.bend as Arithmetic
import ./u32_comparison.bend as Comparison
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./nat_ceiling.bend as Ceiling
import ./encoded_coordinates.bend as Encoded

# ---- The extent a row reads past the axes before it ----

# The extent of a shape after position of its axes is the extent at the
# word of position.
def dropped_extent(+source: List<&2,U32>,+position: Nat,+at: U32,+bound: U32,+rest: List<&2,U32>,+limits: Machine.Limits,
  +exact: {U32.to_nat(at) == position : Nat},+dropped_at: {PlacingLaw.dropped(~U32,position,source) == Con{bound,rest} : List<&2,U32>},
  +rank_room: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> ViewFacts.ExtentAt(source,0,at,bound):
  +length = List.length(&2,U32,source)
  +inside = PlacingFacts.dropped_inside(source,position,bound,rest,dropped_at)
  +position_room = Order.transitive(position,length,U32.to_nat(Machine.capacity(limits)),Bounds.strict_le(position,length,inside),rank_room)
  +at_word = Arithmetic.to_nat_injective(U32.from_nat(position),at,
    Equal.trans(Nat,U32.to_nat(U32.from_nat(position)),position,U32.to_nat(at),
      Bounds.roundtrip(position,Machine.capacity(limits),LimitBounds.capacity_fits(limits),position_room),Equal.sym(Nat,U32.to_nat(at),position,exact)))
  %PlacingFacts.lanes_dropped(source,position,bound,rest,dropped_at) : ViewFacts.ExtentAt(source,0,at,_)
  {%at_word : ViewFacts.ExtentAt(source,0,_,Grid.lanes_at(source,position))
    ViewFacts.extent_index(source,position,0,0n,limits,{==},rank_room,position_room,inside) :
    ViewFacts.ExtentAt(source,0,at,Grid.lanes_at(source,position))}

# The word of position lies below the rank of the shape.
def dropped_below(+source: List<&2,U32>,+position: Nat,+at: U32,+bound: U32,+rest: List<&2,U32>,
  +exact: {U32.to_nat(at) == position : Nat},+dropped_at: {PlacingLaw.dropped(~U32,position,source) == Con{bound,rest} : List<&2,U32>}) ->
  {Nat.is_lt(U32.to_nat(at),List.length(&2,U32,source)) == True{} : Bool}:
  %Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_lt(_,List.length(&2,U32,source)) == True{} : Bool}
  PlacingFacts.dropped_inside(source,position,bound,rest,dropped_at)

# One more word past position stays within the capacity.
def dropped_next(+source: List<&2,U32>,+position: Nat,+at: U32,+bound: U32,+rest: List<&2,U32>,+limits: Machine.Limits,
  +exact: {U32.to_nat(at) == position : Nat},+dropped_at: {PlacingLaw.dropped(~U32,position,source) == Con{bound,rest} : List<&2,U32>},
  +rank_room: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {U32.to_nat((at + 1 : U32)) == 1n+position : Nat}:
  +length = List.length(&2,U32,source)
  Components.next_word(at,position,Machine.capacity(limits),LimitBounds.capacity_fits(limits),exact,
    Order.transitive(1n+position,length,U32.to_nat(Machine.capacity(limits)),Order.strict_follow(position,length,PlacingFacts.dropped_inside(source,position,bound,rest,dropped_at)),rank_room))

# ---- The row of the target ----

# The target reads every axis of its own cells, from at on.
def own_row(target: List<&2,U32>,+source: List<&2,U32>,+grid: Grid.Grid,+at: U32,+position: Nat,+limits: Machine.Limits,
  +blocks: ViewReads.BlocksIn(grid,target,at,Machine.half_word(limits)),+room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+dropped_at: {PlacingLaw.dropped(~U32,position,source) == target : List<&2,U32>},
  +rank_room: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  PlacingFacts.RowFacts(Tiled.own_readings(target,grid,at),target,source,grid,grid,at,Machine.half_word(limits)):
  match target:
    case Con{+extent,+rest}:
      Held.Both{+block_positive,Held.Both{+block_double,+blocks_rest}} = blocks
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +whole = Nat.mul(size,inner)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +size_room = Order.transitive(size,whole,U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(size,inner,inner_positive),room)
      PlacingFacts.row_joined(limits,extent,extent,0,Tiled.own_readings(rest,grid,(at + 1 : U32)),rest,source,grid,grid,False{},at,
        PlacingFacts.reading_room(limits,extent,Grid.block(grid,at,extent),Grid.block(grid,at,extent),at,0,extent,False{},size_room,size_positive,
          block_positive,block_double,block_double,LimitBounds.within_half(limits,size,size_room)),
        dropped_extent(source,position,at,extent,rest,limits,exact,dropped_at,rank_room),dropped_below(source,position,at,extent,rest,exact,dropped_at),
        own_row(rest,source,grid,(at + 1 : U32),1n+position,limits,blocks_rest,
          Order.transitive(inner,whole,U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(size,inner,size_positive),room),inner_positive,
          dropped_next(source,position,at,extent,rest,limits,exact,dropped_at,rank_room),PlacingLaw.dropped_next(~U32,position,source,extent,rest,dropped_at),rank_room))
    case Nil{}:
      Held.Both{Unit{},Held.Both{Unit{},Held.Both{Unit{},Unit{}}}}

# ---- The groups axis of the part ----

# The room the read of the part groups needs: eight lanes a group from the
# lanes start skips, and the part groups past the first it skips.
def GroupsRoom(+limits: Machine.Limits,+extent: U32,+start: U32,+bound: U32) -> Data:
  Held.Both<Components.ReachAt(False{},8,(8 - start % 8 : U32),8,Nat.sub(U32.to_nat(extent),1n),Machine.half_word(limits)),
    {Nat.is_le(Nat.add(U32.to_nat((start / 8 + 1 : U32)),U32.to_nat(bound)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}>

# The groups of a target whose lanes and one group more fit half a word give
# the read of the part groups its room.
def groups_room(+limits: Machine.Limits,+groups: U32,+start: U32,+width: U32,+bound: U32,+positive: {Nat.is_lt(0n,U32.to_nat(groups)) == True{} : Bool},
  +bound_exact: {U32.to_nat(bound) == Nat.div(Nat.add(U32.to_nat(width),7n),8n) : Nat},+width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},
  +ends_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_room: {Nat.is_le(U32.to_nat(bound),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +reach_room: {Nat.is_le(Nat.mul(1n+U32.to_nat(groups),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) -> GroupsRoom(limits,groups,start,bound):
  +half = U32.to_nat(Machine.half_word(limits))
  +count = U32.to_nat(groups)
  +offset = U32.to_nat((8 - start % 8 : U32))
  +before = Nat.mul(8n,Nat.sub(count,1n))
  +sixteen = Bounds.as_nat_le(16,Machine.half_word(limits),LimitBounds.sixteen_within_half(limits))
  +offset_small = {%Equal.sym(Nat,offset,Nat.sub(8n,Nat.mod(U32.to_nat(start),8n)),Part.offset_exact(start,Machine.half_word(limits),LimitBounds.half_word_fits(limits),sixteen)) :
      {Nat.is_le(_,8n) == True{} : Bool}
    Order.sub_le(8n,Nat.mod(U32.to_nat(start),8n)) : {Nat.is_le(offset,8n) == True{} : Bool}}
  +reach = {%Equal.sym(Nat,Nat.add(Nat.add(offset,before),8n),Nat.add(offset,Nat.add(before,8n)),Addition.associative(offset,before,8n)) : {Nat.is_le(_,half) == True{} : Bool}
    {%Equal.sym(Nat,Nat.add(before,8n),Nat.mul(count,8n),RelaneFacts.eight_reach(count,positive)) : {Nat.is_le(Nat.add(offset,_),half) == True{} : Bool}
      Order.transitive(Nat.add(offset,Nat.mul(count,8n)),Nat.add(8n,Nat.mul(count,8n)),half,Order.add_le(offset,8n,Nat.mul(count,8n),offset_small),reach_room) :
      {Nat.is_le(Nat.add(offset,Nat.add(before,8n)),half) == True{} : Bool}} : {Nat.is_le(Nat.add(Nat.add(offset,before),8n),half) == True{} : Bool}}
  +skipped = Nat.add(Nat.div(U32.to_nat(start),8n),1n)
  +skipped_room = Order.transitive(skipped,Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits)),
    Order.transitive(skipped,Nat.add(U32.to_nat(start),1n),Nat.add(U32.to_nat(start),U32.to_nat(width)),
      Order.add_le(Nat.div(U32.to_nat(start),8n),U32.to_nat(start),1n,DivisionOrder.quotient_at_most(U32.to_nat(start),8n,{==})),
      Order.offset_le(U32.to_nat(start),1n,U32.to_nat(width),Order.positive_one(U32.to_nat(width),width_positive))),ends_room)
  +shift_exact = Part.shift_exact(start,Machine.half_word(limits),LimitBounds.half_word_fits(limits),sixteen,LimitBounds.within_half(limits,skipped,skipped_room))
  Held.Both{reach,
    {%Equal.sym(Nat,U32.to_nat((start / 8 + 1 : U32)),skipped,shift_exact) : {Nat.is_le(Nat.add(_,U32.to_nat(bound)),half) == True{} : Bool}
      ViewFacts.pair_half(limits,skipped,U32.to_nat(bound),skipped_room,part_room) : {Nat.is_le(Nat.add(U32.to_nat((start / 8 + 1 : U32)),U32.to_nat(bound)),half) == True{} : Bool}}}

# The read of the part groups has room at half a word.
def groups_reading(+limits: Machine.Limits,+extent: U32,+block: U32,+source_block: U32,+at: U32,+start: U32,+bound: U32,
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool},
  +block_double: {Nat.is_le(Nat.double(U32.to_nat(U32.max(block,1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +source_double: {Nat.is_le(Nat.double(U32.to_nat(U32.max(source_block,1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+groups: GroupsRoom(limits,extent,start,bound)) ->
  Components.ReadingRoom(extent,block,Tasks.Read{at,source_block,8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),bound,False{}},Machine.half_word(limits)):
  Held.Both{+reach,+shift_room} = groups
  Held.Both{ViewFacts.axis_room(limits,extent,block,room,block_positive,ViewFacts.block_within(limits,block,block_positive,block_double)),
    Held.Both{{==},Held.Both{reach,Held.Both{Order.transitive(Nat.double(U32.to_nat(U32.max(8,1))),16n,U32.to_nat(Machine.half_word(limits)),{==},
      Bounds.as_nat_le(16,Machine.half_word(limits),LimitBounds.sixteen_within_half(limits))),Held.Both{shift_room,source_double}}}}}

# ---- The row of the part ----

# The part extent beside a target extent: the part groups on the groups
# axis, whose read has room, and the target extent on every other.
def SideAt(same: Bool,+extent: U32,+value: U32,+limits: Machine.Limits,+start: U32,+bound: U32) -> Data:
  match same:
    case True{}:
      Held.Both<{value == bound : U32},GroupsRoom(limits,extent,start,bound)>
    case False{}:
      {value == extent : U32}

def PartSide(target: List<&2,U32>,part: List<&2,U32>,+at: U32,+groups_axis: U32,+limits: Machine.Limits,+start: U32,+bound: U32) -> Data:
  match target part:
    case Con{+extent,target} Con{+value,part}:
      Held.Both<SideAt(U32.is_eq(at,groups_axis),extent,value,limits,start,bound),PartSide(target,part,(at + 1 : U32),groups_axis,limits,start,bound)>
    case Nil{} Nil{}:
      Unit
    case target part:
      Empty

def part_reading(same: Bool,+extent: U32,+value: U32,+rest: List<&2,Tasks.Reading>,+target: List<&2,U32>,+source: List<&2,U32>,+part_grid: Grid.Grid,
  +grid: Grid.Grid,+start: U32,+bound: U32,+at: U32,+limits: Machine.Limits,+side: SideAt(same,extent,value,limits,start,bound),
  +room: {Nat.is_le(U32.to_nat(extent),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+block_positive: {Nat.is_lt(0n,U32.to_nat(Grid.block(grid,at,extent))) == True{} : Bool},
  +block_double: {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,at,extent),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +source_double: {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(part_grid,at,value),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +extent_at: ViewFacts.ExtentAt(source,0,at,value),+below: {Nat.is_lt(U32.to_nat(at),List.length(&2,U32,source)) == True{} : Bool},
  +later: PlacingFacts.RowFacts(rest,target,source,grid,part_grid,(at + 1 : U32),Machine.half_word(limits))) ->
  PlacingFacts.RowFacts(Con{Tasks.Reading{extent,Grid.block(grid,at,extent),Bool.pick(Tasks.Source,same,
      Tasks.Read{at,Grid.block(part_grid,at,bound),8,(8 - start % 8 : U32),False{},8,8,(start / 8 + 1 : U32),bound,False{}},
      Tiled.read_at(part_grid,at,extent,1,1,1))},rest},
    Con{extent,target},source,grid,part_grid,at,Machine.half_word(limits)):
  match same:
    case True{}:
      Held.Both{+value_same,+groups} = side
      Held.Both{+sides,Held.Both{+rooms,Held.Both{+froms,+belows}}} = later
      Held.Both{Held.Both{{==},Held.Both{{==},sides}},
        Held.Both{Held.Both{groups_reading(limits,extent,Grid.block(grid,at,extent),Grid.block(part_grid,at,bound),at,start,bound,room,block_positive,block_double,
              {%value_same : {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(part_grid,at,_),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool} source_double :
                {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(part_grid,at,bound),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}},groups),rooms},
          Held.Both{Held.Both{Held.Both{{==},{%value_same : ViewFacts.ExtentAt(source,0,at,_) extent_at : ViewFacts.ExtentAt(source,0,at,bound)}},froms},
            Held.Both{below,belows}}}}
    case False{}:
      PlacingFacts.row_joined(limits,extent,extent,0,rest,target,source,grid,part_grid,False{},at,
        PlacingFacts.reading_room(limits,extent,Grid.block(grid,at,extent),Grid.block(part_grid,at,extent),at,0,extent,False{},room,positive,
          block_positive,block_double,
          {%side : {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(part_grid,at,_),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool} source_double :
            {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(part_grid,at,extent),1))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}},LimitBounds.within_half(limits,U32.to_nat(extent),room)),
        {%side : ViewFacts.ExtentAt(source,0,at,_) extent_at : ViewFacts.ExtentAt(source,0,at,extent)},below,later)

# The part reads its groups on the groups axis and every other axis at the
# target extent, from at on.
def part_row(target: List<&2,U32>,part: List<&2,U32>,+source: List<&2,U32>,+part_grid: Grid.Grid,+grid: Grid.Grid,+groups_axis: U32,+start: U32,+bound: U32,
  +at: U32,+position: Nat,+limits: Machine.Limits,+side: PartSide(target,part,at,groups_axis,limits,start,bound),
  +blocks: ViewReads.BlocksIn(grid,target,at,Machine.half_word(limits)),+part_blocks: ViewReads.BlocksIn(part_grid,part,at,Machine.half_word(limits)),
  +room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},
  +exact: {U32.to_nat(at) == position : Nat},+dropped_at: {PlacingLaw.dropped(~U32,position,source) == part : List<&2,U32>},
  +rank_room: {Nat.is_le(List.length(&2,U32,source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  PlacingFacts.RowFacts(Tiled.part_readings(target,part_grid,grid,groups_axis,start,bound,at),target,source,grid,part_grid,at,Machine.half_word(limits)):
  match target part:
    case Con{+extent,+rest} Con{+value,+more}:
      Held.Both{+here,+side_rest} = side
      Held.Both{+block_positive,Held.Both{+block_double,+blocks_rest}} = blocks
      Held.Both{+part_positive,Held.Both{+part_double,+part_rest}} = part_blocks
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(rest)
      +whole = Nat.mul(size,inner)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +size_room = Order.transitive(size,whole,U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(size,inner,inner_positive),room)
      part_reading(U32.is_eq(at,groups_axis),extent,value,Tiled.part_readings(rest,part_grid,grid,groups_axis,start,bound,(at + 1 : U32)),rest,source,part_grid,grid,
        start,bound,at,limits,here,size_room,size_positive,block_positive,block_double,part_double,
        dropped_extent(source,position,at,value,more,limits,exact,dropped_at,rank_room),dropped_below(source,position,at,value,more,exact,dropped_at),
        part_row(rest,more,source,part_grid,grid,groups_axis,start,bound,(at + 1 : U32),1n+position,limits,side_rest,blocks_rest,part_rest,
          Order.transitive(inner,whole,U32.to_nat(Machine.capacity(limits)),DivisionOrder.le_multiple(size,inner,size_positive),room),inner_positive,
          dropped_next(source,position,at,value,more,limits,exact,dropped_at,rank_room),PlacingLaw.dropped_next(~U32,position,source,value,more,dropped_at),rank_room))
    case Nil{} Nil{}:
      Held.Both{Unit{},Held.Both{Unit{},Held.Both{Unit{},Unit{}}}}
    case Con{+extent,+rest} Nil{}:
      Empty.absurd(PlacingFacts.RowFacts(Tiled.part_readings(Con{extent,rest},part_grid,grid,groups_axis,start,bound,at),Con{extent,rest},source,grid,part_grid,at,Machine.half_word(limits)),side)
    case Nil{} Con{+value,+more}:
      Empty.absurd(PlacingFacts.RowFacts(Tiled.part_readings(Nil{},part_grid,grid,groups_axis,start,bound,at),Nil{},source,grid,part_grid,at,Machine.half_word(limits)),side)

# ---- The part cells beside the target cells ----

def words_apart(+left: U32,+right: U32,+apart: {Nat.is_eq(U32.to_nat(left),U32.to_nat(right)) == False{} : Bool}) -> {U32.is_eq(left,right) == False{} : Bool}:
  Equal.trans(Bool,U32.is_eq(left,right),Nat.is_eq(U32.to_nat(left),U32.to_nat(right)),False{},Comparison.eq(left,right),apart)

# Axes past the groups axis read the same extent of target and part.
def side_past(list: List<&2,U32>,+at: U32,+position: Nat,+groups_axis: U32,+limits: Machine.Limits,+start: U32,+bound: U32,
  +exact: {U32.to_nat(at) == position : Nat},+past: {Nat.is_lt(U32.to_nat(groups_axis),position) == True{} : Bool},
  +room: {Nat.is_le(Nat.add(position,List.length(&2,U32,list)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> PartSide(list,list,at,groups_axis,limits,start,bound):
  match list:
    case Con{+head,+rest}:
      +length = List.length(&2,U32,rest)
      +advanced = Order.advance_room(position,length,U32.to_nat(Machine.capacity(limits)),room)
      +apart = words_apart(at,groups_axis,
        {%Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_eq(_,U32.to_nat(groups_axis)) == False{} : Bool}
          Equal.trans(Bool,Nat.is_eq(position,U32.to_nat(groups_axis)),Nat.is_eq(U32.to_nat(groups_axis),position),False{},
            Order.equal_symmetric(position,U32.to_nat(groups_axis)),Order.less_not_equal(U32.to_nat(groups_axis),position,past)) :
          {Nat.is_eq(U32.to_nat(at),U32.to_nat(groups_axis)) == False{} : Bool}})
      Held.Both{{%Equal.sym(Bool,U32.is_eq(at,groups_axis),False{},apart) : SideAt(_,head,head,limits,start,bound) {==} : SideAt(U32.is_eq(at,groups_axis),head,head,limits,start,bound)},
        side_past(rest,(at + 1 : U32),1n+position,groups_axis,limits,start,bound,
          Components.next_word(at,position,Machine.capacity(limits),LimitBounds.capacity_fits(limits),exact,
            Order.transitive(1n+position,Nat.add(1n+position,length),U32.to_nat(Machine.capacity(limits)),Order.le_add(1n+position,length),advanced)),
          Order.lt_succ(U32.to_nat(groups_axis),position,Bounds.strict_le(U32.to_nat(groups_axis),position,past)),advanced)}
    case Nil{}:
      Unit{}

# Axes before the groups axis read the same extent of target and part, and
# the groups axis after them the target and the part groups.
def side_before(list: List<&2,U32>,+at: U32,+position: Nat,+groups_axis: U32,+limits: Machine.Limits,+start: U32,+groups: U32,+bound: U32,
  +exact: {U32.to_nat(at) == position : Nat},+axis_exact: {U32.to_nat(groups_axis) == Nat.add(position,List.length(&2,U32,list)) : Nat},
  +room: {Nat.is_le(Nat.add(position,List.length(&2,U32,list)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+groups_side: GroupsRoom(limits,groups,start,bound)) ->
  PartSide(List.append(&2,U32,list,[groups]),List.append(&2,U32,list,[bound]),at,groups_axis,limits,start,bound):
  match list:
    case Con{+head,+rest}:
      +length = List.length(&2,U32,rest)
      +advanced = Order.advance_room(position,length,U32.to_nat(Machine.capacity(limits)),room)
      +shifted = Addition.successor_right(position,length)
      +axis_next = Equal.trans(Nat,U32.to_nat(groups_axis),Nat.add(position,1n+length),Nat.add(1n+position,length),axis_exact,shifted)
      +apart = words_apart(at,groups_axis,
        {%Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_eq(_,U32.to_nat(groups_axis)) == False{} : Bool}
          {%Equal.sym(Nat,U32.to_nat(groups_axis),Nat.add(1n+position,length),axis_next) : {Nat.is_eq(position,_) == False{} : Bool}
            Order.less_not_equal(position,Nat.add(1n+position,length),Order.lt_succ(position,Nat.add(position,length),Order.le_add(position,length))) :
            {Nat.is_eq(position,U32.to_nat(groups_axis)) == False{} : Bool}} : {Nat.is_eq(U32.to_nat(at),U32.to_nat(groups_axis)) == False{} : Bool}})
      Held.Both{{%Equal.sym(Bool,U32.is_eq(at,groups_axis),False{},apart) : SideAt(_,head,head,limits,start,bound) {==} : SideAt(U32.is_eq(at,groups_axis),head,head,limits,start,bound)},
        side_before(rest,(at + 1 : U32),1n+position,groups_axis,limits,start,groups,bound,
          Components.next_word(at,position,Machine.capacity(limits),LimitBounds.capacity_fits(limits),exact,
            Order.transitive(1n+position,Nat.add(1n+position,length),U32.to_nat(Machine.capacity(limits)),Order.le_add(1n+position,length),advanced)),
          axis_next,advanced,groups_side)}
    case Nil{}:
      +same = Equal.trans(Bool,U32.is_eq(at,groups_axis),Nat.is_eq(U32.to_nat(at),U32.to_nat(groups_axis)),True{},Comparison.eq(at,groups_axis),
        {%Equal.sym(Nat,U32.to_nat(at),position,exact) : {Nat.is_eq(_,U32.to_nat(groups_axis)) == True{} : Bool}
          {%Equal.sym(Nat,U32.to_nat(groups_axis),Nat.add(position,0n),axis_exact) : {Nat.is_eq(position,_) == True{} : Bool}
            {%Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)) : {Nat.is_eq(position,_) == True{} : Bool}
              Order.equal_reflexive(position) : {Nat.is_eq(position,Nat.add(position,0n)) == True{} : Bool}} :
            {Nat.is_eq(position,U32.to_nat(groups_axis)) == True{} : Bool}} : {Nat.is_eq(U32.to_nat(at),U32.to_nat(groups_axis)) == True{} : Bool}})
      Held.Both{{%Equal.sym(Bool,U32.is_eq(at,groups_axis),True{},same) : SideAt(_,groups,bound,limits,start,bound) Held.Both{{==},groups_side} :
          SideAt(U32.is_eq(at,groups_axis),groups,bound,limits,start,bound)},Unit{}}

# The rank of cells with the groups placed is one more than the other axes.
def placed_length(position: Grid.Position,+others: List<&2,U32>,+groups: U32) -> {List.length(&2,U32,Grid.placed(True{},position,others,groups)) == 1n+List.length(&2,U32,others) : Nat}:
  match position:
    case Grid.Innermost{}: Values.appended_length(others,groups)
    case Grid.Outermost{}: {==}

# The part cells beside the target cells differ only on the groups axis.
def placed_side(position: Grid.Position,+others: List<&2,U32>,+groups: U32,+bound: U32,+limits: Machine.Limits,+start: U32,
  +rank_room: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+groups_side: GroupsRoom(limits,groups,start,bound)) ->
  PartSide(Grid.placed(True{},position,others,groups),Grid.placed(True{},position,others,bound),0,
    Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),limits,start,bound):
  match position:
    case Grid.Outermost{}:
      Held.Both{Held.Both{{==},groups_side},side_past(others,1,1n,0,limits,start,bound,{==},{==},rank_room)}
    case Grid.Innermost{}:
      +length = List.length(&2,U32,others)
      +cells = List.append(&2,U32,others,[groups])
      +rank = Values.appended_length(others,groups)
      +back = Equal.sym(Nat,List.length(&2,U32,cells),1n+length,rank)
      +rank_word = Bounds.roundtrip(List.length(&2,U32,cells),Machine.capacity(limits),LimitBounds.capacity_fits(limits),rank_room)
      +axis_exact = Equal.trans(Nat,U32.to_nat((U32.from_nat(List.length(&2,U32,cells)) - 1 : U32)),Nat.sub(List.length(&2,U32,cells),1n),length,
        Arithmetic.difference_exact(Machine.capacity(limits),LimitBounds.capacity_fits(limits),U32.from_nat(List.length(&2,U32,cells)),1,
          List.length(&2,U32,cells),1n,rank_word,{==},{%back : {Nat.is_le(1n,_) == True{} : Bool} Order.positive_one(1n+length,{==}) : {Nat.is_le(1n,List.length(&2,U32,cells)) == True{} : Bool}},rank_room),
        {%back : {Nat.sub(_,1n) == length : Nat} Order.subtract_zero(length) : {Nat.sub(List.length(&2,U32,cells),1n) == length : Nat}})
      side_before(others,0,0n,(U32.from_nat(List.length(&2,U32,cells)) - 1 : U32),limits,start,groups,bound,{==},axis_exact,
        Order.transitive(length,1n+length,U32.to_nat(Machine.capacity(limits)),Order.successor(length,length,Order.reflexive(length)),
          {%rank : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room : {Nat.is_le(1n+length,U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}),groups_side)

# ---- The facts of a lane write ----

# The lanes of a target that fit the capacity, and one group more, fit half
# a word eight lanes a group.
def groups_reach(+limits: Machine.Limits,+lanes: U32,+room: {Nat.is_le(U32.to_nat(lanes),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  {Nat.is_le(Nat.mul(1n+U32.to_nat(Grid.groups(lanes)),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}:
  +count = U32.to_nat(lanes)
  +half = U32.to_nat(Machine.half_word(limits))
  +groups = U32.to_nat(Grid.groups(lanes))
  +above = Nat.add(count,7n)
  +covered = {%Equal.sym(Nat,groups,Ceiling.ceiling(count,8n),Encoded.eight_blocks_exact(limits,lanes,room)) : {Nat.is_le(Nat.mul(_,8n),above) == True{} : Bool}
    {%Ceiling.shifted_formula(count,8n,{==}) : {Nat.is_le(Nat.mul(_,8n),above) == True{} : Bool}
      DivisionOrder.product_below(above,8n,{==}) : {Nat.is_le(Nat.mul(Ceiling.ceiling(count,8n),8n),above) == True{} : Bool}} : {Nat.is_le(Nat.mul(groups,8n),above) == True{} : Bool}}
  +fifteen = Order.transitive(15n,16n,U32.to_nat(Machine.capacity(limits)),{==},Bounds.as_nat_le(16,Machine.capacity(limits),LimitBounds.sixteen_within_capacity(limits)))
  +ended = {%Equal.sym(Nat,Nat.add(8n,above),Nat.add(above,8n),Order.commutative(8n,above)) : {Nat.is_le(_,half) == True{} : Bool}
    {%Equal.sym(Nat,Nat.add(Nat.add(count,7n),8n),Nat.add(count,Nat.add(7n,8n)),Addition.associative(count,7n,8n)) : {Nat.is_le(_,half) == True{} : Bool}
      ViewFacts.pair_half(limits,count,15n,room,fifteen) : {Nat.is_le(Nat.add(above,8n),half) == True{} : Bool}} : {Nat.is_le(Nat.add(8n,above),half) == True{} : Bool}}
  Order.transitive(Nat.add(8n,Nat.mul(groups,8n)),Nat.add(8n,above),half,Order.offset_le(8n,Nat.mul(groups,8n),above,covered),ended)

# What a lane write needs of its target and part cells, and the axes its own
# row reads.
def LanesBuilt(+limits: Machine.Limits,+position: Grid.Position,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+start: U32,+width: U32) -> Data:
  Held.Both<Law.LanesFacts(limits,position,others,groups,bound,count,start,width),Held.Both<Components.AxesOk(Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),0,Machine.half_word(limits)),Components.AxesBelow(Tasks.read_axes(Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0)),List.length(&2,U32,Grid.placed(True{},position,others,groups)))>>

def facts_ranked(+limits: Machine.Limits,+position: Grid.Position,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+start: U32,+width: U32,+rank: {Tasks.target_rank([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)]) == List.length(&2,U32,Grid.placed(True{},position,others,groups)) : Nat},
  +components: Held.Both<Components.ComponentsOk([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)],List.length(&2,U32,Grid.placed(True{},position,others,groups)),Machine.half_word(limits)),
    {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(List.length(&2,U32,Grid.placed(True{},position,others,groups)),[Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)]))) == True{} : Bool}>,
  +limits_at: Components.LimitsAt([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)],List.length(&2,U32,Grid.placed(True{},position,others,groups)),Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),
  +room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +own_rooms: Components.ReadingsRoom(Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Machine.half_word(limits)),+blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),0,Machine.half_word(limits)),
  +own_axes: Components.AxesOk(Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),0,Machine.half_word(limits)),+own_belows: Components.AxesBelow(Tasks.read_axes(Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0)),List.length(&2,U32,Grid.placed(True{},position,others,groups))),
  +part_rooms: Components.ReadingsRoom(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Machine.half_word(limits)),+part_axes: Components.AxesOk(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),+part_blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),
  +part_belows: Components.AxesBelow(Tasks.read_axes(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)),List.length(&2,U32,Grid.placed(True{},position,others,bound))),
  +part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_positive: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_below: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},
  +cell_grid: Gather.CellGrid<groups,count,Machine.capacity(limits)>,+cells_exact: {ShapeLaws.elements(Grid.placed(True{},position,others,groups)) == Nat.mul(U32.to_nat(groups),U32.to_nat(count)) : Nat},
  +bound_exact: {U32.to_nat(bound) == Nat.div(Nat.add(U32.to_nat(width),7n),8n) : Nat},+groups_room: {Nat.is_le(Nat.mul(8n,U32.to_nat(groups)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.mul(8n,U32.to_nat(groups))) == True{} : Bool}) -> LanesBuilt(limits,position,others,groups,bound,count,start,width):
  Held.Both{+ok,+count_positive} = components
  +back = Equal.sym(Nat,Tasks.target_rank([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)]),List.length(&2,U32,Grid.placed(True{},position,others,groups)),rank)
  Held.Both{Held.Both{room,Held.Both{positive,Held.Both{rank_room,
      Held.Both{{%back : Components.ComponentsOk([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)],_,Machine.half_word(limits)) ok : Components.ComponentsOk([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)],Tasks.target_rank([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)]),Machine.half_word(limits))},
        Held.Both{{%back : {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(_,[Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)]))) == True{} : Bool} count_positive :
            {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)]),[Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)]))) == True{} : Bool}},
          Held.Both{own_rooms,Held.Both{blocks,
            Held.Both{part_rooms,Held.Both{part_axes,Held.Both{part_blocks,Held.Both{part_belows,Held.Both{part_room,Held.Both{part_positive,Held.Both{part_rank,
              Held.Both{rank_below,Held.Both{count_exact,
                Held.Both{{%back : Components.LimitsAt([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)],_,Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0) limits_at : Components.LimitsAt([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)],Tasks.target_rank([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)]),Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0)},
                  Held.Both{cell_grid,Held.Both{cells_exact,Held.Both{bound_exact,Held.Both{groups_room,lanes_room}}}}}}}}}}}}}}}}}}}}},
    Held.Both{own_axes,own_belows}}

def facts_joined(+limits: Machine.Limits,+position: Grid.Position,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+start: U32,+width: U32,+own: PlacingFacts.RowFacts(Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Grid.placed(True{},position,others,groups),Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0,Machine.half_word(limits)),+part: PlacingFacts.RowFacts(Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),Grid.placed(True{},position,others,groups),Grid.placed(True{},position,others,bound),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),0,Machine.half_word(limits)),
  +room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,groups))) == True{} : Bool},+rank_room: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Grid.placed(True{},position,others,groups),0,Machine.half_word(limits)),+part_blocks: ViewReads.BlocksIn(Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.placed(True{},position,others,bound),0,Machine.half_word(limits)),
  +part_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+part_positive: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,bound))) == True{} : Bool},+part_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,bound)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +rank_below: {Nat.is_lt(1n+List.length(&2,U32,others),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},+count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},
  +cell_grid: Gather.CellGrid<groups,count,Machine.capacity(limits)>,+cells_exact: {ShapeLaws.elements(Grid.placed(True{},position,others,groups)) == Nat.mul(U32.to_nat(groups),U32.to_nat(count)) : Nat},
  +bound_exact: {U32.to_nat(bound) == Nat.div(Nat.add(U32.to_nat(width),7n),8n) : Nat},+groups_room: {Nat.is_le(Nat.mul(8n,U32.to_nat(groups)),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.mul(8n,U32.to_nat(groups))) == True{} : Bool}) -> LanesBuilt(limits,position,others,groups,bound,count,start,width):
  Held.Both{+own_side,Held.Both{+own_rooms,Held.Both{+own_froms,+own_belows}}} = own
  Held.Both{+part_side,Held.Both{+part_rooms,Held.Both{+part_froms,+part_belows}}} = part
  +read = {Held.Both{own_side,Held.Both{own_rooms,Held.Both{part_side,Held.Both{part_rooms,Unit{}}}}} : TargetRows.RowsRead([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)],Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0,Machine.half_word(limits))}
  +rows_room = Order.transitive(2n,16n,U32.to_nat(Machine.half_word(limits)),{==},Bounds.as_nat_le(16,Machine.half_word(limits),LimitBounds.sixteen_within_half(limits)))
  facts_ranked(limits,position,others,groups,bound,count,start,width,TargetRows.rows_rank([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)],Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0,Machine.half_word(limits),read,{==}),
    TargetRows.rows_components([Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)],Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),limits,read,{==},rows_room,room,positive),TargetRows.rows_limits(Grid.placed(True{},position,others,groups),[Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0)],Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0,Machine.half_word(limits),read,{==}),
    room,positive,rank_room,own_rooms,blocks,ViewFacts.reads_axes(Grid.placed(True{},position,others,groups),Tiled.own_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0),limits,Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0,own_froms,blocks,room,positive),own_belows,
    part_rooms,ViewFacts.reads_axes(Grid.placed(True{},position,others,bound),Tiled.part_readings(Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0),limits,Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),0,part_froms,part_blocks,part_room,part_positive),part_blocks,part_belows,
    part_room,part_positive,part_rank,rank_below,count_exact,cell_grid,cells_exact,bound_exact,groups_room,lanes_room)

# Target and part cells that fit the machine, with one group of lanes to
# spare at half a word, give a lane write of a part that holds lanes every
# fact it needs.
def lanes_facts(+limits: Machine.Limits,+position: Grid.Position,+others: List<&2,U32>,+groups: U32,+bound: U32,+count: U32,+start: U32,+width: U32,+target_ok: Components.ShapeOk(limits,Grid.placed(True{},position,others,groups)),+part_ok: Components.ShapeOk(limits,Grid.placed(True{},position,others,bound)),
  +count_exact: {U32.to_nat(count) == ShapeLaws.elements(others) : Nat},+bound_exact: {U32.to_nat(bound) == Nat.div(Nat.add(U32.to_nat(width),7n),8n) : Nat},
  +width_positive: {Nat.is_lt(0n,U32.to_nat(width)) == True{} : Bool},+ends_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +lanes_room: {Nat.is_le(Nat.add(U32.to_nat(start),U32.to_nat(width)),Nat.mul(8n,U32.to_nat(groups))) == True{} : Bool},
  +reach_room: {Nat.is_le(Nat.mul(1n+U32.to_nat(groups),8n),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}) -> LanesBuilt(limits,position,others,groups,bound,count,start,width):
  Held.Both{+room,Held.Both{+positive,+rank_room}} = target_ok
  Held.Both{+part_room,Held.Both{+part_positive,+part_rank}} = part_ok
  +target_shape = {Held.Both{room,Held.Both{positive,rank_room}} : Components.ShapeOk(limits,Grid.placed(True{},position,others,groups))}
  +part_shape = {Held.Both{part_room,Held.Both{part_positive,part_rank}} : Components.ShapeOk(limits,Grid.placed(True{},position,others,bound))}
  +blocks = ViewFacts.grid_blocks(limits,Grid.placed(True{},position,others,groups),Grid.piece_target(),target_shape)
  +part_blocks = ViewFacts.grid_blocks(limits,Grid.placed(True{},position,others,bound),Grid.piece_target(),part_shape)
  +size = U32.to_nat(groups)
  +inner = ShapeLaws.elements(others)
  +cells = Public.placed_cells(position,others,groups)
  +part_cells = Public.placed_cells(position,others,bound)
  +product_positive = {%cells : {Nat.is_lt(0n,_) == True{} : Bool} positive : {Nat.is_lt(0n,Nat.mul(size,inner)) == True{} : Bool}}
  +groups_positive = RowMajor.left_factor_positive(size,inner,product_positive)
  +inner_positive = DivisionOrder.right_factor_positive(size,inner,product_positive)
  +to_count = Equal.sym(Nat,U32.to_nat(count),inner,count_exact)
  +cells_exact = {%to_count : {ShapeLaws.elements(Grid.placed(True{},position,others,groups)) == Nat.mul(size,_) : Nat} cells : {ShapeLaws.elements(Grid.placed(True{},position,others,groups)) == Nat.mul(size,U32.to_nat(count)) : Nat}}
  +cell_grid = {Gather.CellGrid{LimitBounds.capacity_fits(limits),groups_positive,{%to_count : {Nat.is_lt(0n,_) == True{} : Bool} inner_positive : {Nat.is_lt(0n,U32.to_nat(count)) == True{} : Bool}},
      {%cells_exact : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} room : {Nat.is_le(Nat.mul(size,U32.to_nat(count)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},Machine.half_word(limits),LimitBounds.half_word_fits(limits),
      {%Equal.sym(Nat,U32.to_nat(Machine.half_word(limits)),Nat.double(U32.to_nat(Machine.capacity(limits))),LimitBounds.doubled_capacity(limits)) : {Nat.is_le(Nat.double(U32.to_nat(Machine.capacity(limits))),_) == True{} : Bool} Order.reflexive(Nat.double(U32.to_nat(Machine.capacity(limits)))) :
        {Nat.is_le(Nat.double(U32.to_nat(Machine.capacity(limits))),U32.to_nat(Machine.half_word(limits))) == True{} : Bool}}} : Gather.CellGrid<groups,count,Machine.capacity(limits)>}
  +bound_room = Order.transitive(U32.to_nat(bound),Nat.mul(U32.to_nat(bound),inner),U32.to_nat(Machine.capacity(limits)),RowMajor.product_covers(U32.to_nat(bound),inner,inner_positive),
    {%part_cells : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} part_room : {Nat.is_le(Nat.mul(U32.to_nat(bound),inner),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}})
  +groups_side = groups_room(limits,groups,start,width,bound,groups_positive,bound_exact,width_positive,ends_room,bound_room,reach_room)
  +groups_le = Order.transitive(Nat.mul(8n,size),Nat.mul(1n+size,8n),U32.to_nat(Machine.half_word(limits)),
    {%Equal.sym(Nat,Nat.mul(8n,size),Nat.mul(size,8n),RowMajor.multiply_commutative(8n,size)) : {Nat.is_le(_,Nat.add(8n,Nat.mul(size,8n))) == True{} : Bool}
      Order.right_le(8n,Nat.mul(size,8n)) : {Nat.is_le(Nat.mul(8n,size),Nat.add(8n,Nat.mul(size,8n))) == True{} : Bool}},reach_room)
  +rank_one = {%placed_length(position,others,groups) : {Nat.is_le(_,U32.to_nat(Machine.capacity(limits))) == True{} : Bool} rank_room : {Nat.is_le(1n+List.length(&2,U32,others),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}}
  +one_room = Order.transitive(1n,16n,U32.to_nat(Machine.capacity(limits)),{==},Bounds.as_nat_le(16,Machine.capacity(limits),LimitBounds.sixteen_within_capacity(limits)))
  +rank_below = Order.lt_le(1n+List.length(&2,U32,others),Nat.add(1n+List.length(&2,U32,others),1n),U32.to_nat(Machine.half_word(limits)),
    Order.below_sum(1n+List.length(&2,U32,others),0n),ViewFacts.pair_half(limits,1n+List.length(&2,U32,others),1n,rank_one,one_room))
  facts_joined(limits,position,others,groups,bound,count,start,width,own_row(Grid.placed(True{},position,others,groups),Grid.placed(True{},position,others,groups),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),0,0n,limits,blocks,room,positive,{==},{==},rank_room),
    part_row(Grid.placed(True{},position,others,groups),Grid.placed(True{},position,others,bound),Grid.placed(True{},position,others,bound),Grid.grid(Grid.placed(True{},position,others,bound),Grid.piece_target()),Grid.grid(Grid.placed(True{},position,others,groups),Grid.piece_target()),Bool.pick(U32,Grid.is_outermost(position),0,(U32.from_nat(List.length(&2,U32,Grid.placed(True{},position,others,groups))) - 1 : U32)),start,bound,0,0n,limits,placed_side(position,others,groups,bound,limits,start,rank_room,groups_side),
      blocks,part_blocks,room,positive,{==},{==},part_rank),
    room,positive,rank_room,blocks,part_blocks,part_room,part_positive,part_rank,rank_below,count_exact,cell_grid,cells_exact,bound_exact,groups_le,lanes_room)
