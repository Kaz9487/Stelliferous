# What a split gives its gather: its readings read the source cells within
# half a word, take the blocks of the target cells and end their classes
# inside those blocks.

import Base
import ../tensor.bend as Tensors
import ../tensor_grid.bend as Grid
import ../tensor_tasks.bend as Tasks
import ../tensor_tiled.bend as Tiled
import ../machine_limits.bend as Machine
import ../kernel_gemm.bend as KernelGemm
import ./machine_limit_bounds.bend as Limits
import ./tensor_move_readback.bend as Held
import ./tensor_components.bend as Components
import ./tensor_target_cells.bend as Targets
import ./tensor_shape_refinement.bend as ShapeLaws
import ./tensor_view_facts.bend as ViewFacts
import ./tensor_view_reads.bend as ViewReads
import ./tensor_relane_facts.bend as RelaneFacts
import ./tensor_split_reads.bend as Reads
import ./tensor_tiled_values.bend as TiledValues
import ./tensor_tiled_public.bend as Public
import ./boolean_logic.bend as Logic
import ./nat_ceiling.bend as NatCeiling
import ./nat_order.bend as Order
import ./nat_to_u32_bounds.bend as Bounds
import ./natural_addition.bend as Addition
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./bounded_u32_arithmetic.bend as Arithmetic

# ---- A row of readings ----

# The extents of the target axes a row reads.
def row_extents(row: List<&2,Tasks.Reading>) -> List<&2,U32>:
  match row:
    case Nil{}: []
    case Con{Tasks.Reading{+extent,block,read},+rest}: extent <> row_extents(rest)

# Every reading of a row takes the blocks of its target axis in grid, the
# axes counted from at.
def Blocked(row: List<&2,Tasks.Reading>,+grid: Grid.Grid,+at: U32) -> Data:
  match row:
    case Nil{}: Unit
    case Con{Tasks.Reading{+extent,+block,read},rest}:
      Held.Both<{block == Grid.block(grid,at,extent) : U32},Blocked(rest,grid,(at + 1 : U32))>

# Every read of a row has room at half a word and reads the source cells.
def SourcesOk(row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid) -> Data:
  match row:
    case Nil{}: Unit
    case Con{Tasks.Reading{+extent,block,+read},rest}:
      Held.Both<Components.ReadRoom(read,Nat.sub(U32.to_nat(extent),1n),Machine.half_word(limits)),
        Held.Both<ViewFacts.ReadOk(read,source,grid),SourcesOk(rest,limits,source,grid)>>

def sources_cons(+extent: U32,+block: U32,+read: Tasks.Source,+rest: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,
  +head: Held.Both<Components.ReadRoom(read,Nat.sub(U32.to_nat(extent),1n),Machine.half_word(limits)),ViewFacts.ReadOk(read,source,grid)>,
  +tail: SourcesOk(rest,limits,source,grid)) -> SourcesOk(Con{Tasks.Reading{extent,block,read},rest},limits,source,grid):
  Held.Both{+room,+read_ok} = head
  Held.Both{room,Held.Both{read_ok,tail}}

# The axis count axes after at.
def after(count: Nat,+at: U32) -> U32:
  match count:
    case 0n: at
    case 1n+rest: after(rest,(at + 1 : U32))

def after_word(count: Nat,+at: U32,+position: Nat,+limits: Machine.Limits,+exact: {U32.to_nat(at) == position : Nat},
  +room: {Nat.is_le(Nat.add(position,count),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> {after(count,at) == U32.from_nat(Nat.add(position,count)) : U32}:
  match count:
    case 0n:
      Equal.trans(U32,at,U32.from_nat(U32.to_nat(at)),U32.from_nat(Nat.add(position,0n)),Equal.sym(U32,U32.from_nat(U32.to_nat(at)),at,Arithmetic.native_roundtrip(at)),
        Equal.cong(Nat,U32,number => U32.from_nat(number),U32.to_nat(at),Nat.add(position,0n),
          Equal.trans(Nat,U32.to_nat(at),position,Nat.add(position,0n),exact,Equal.sym(Nat,Nat.add(position,0n),position,Addition.zero_right(position)))))
    case 1n+ +rest:
      +cap = U32.to_nat(Machine.capacity(limits))
      +shifted = {%Addition.successor_right(position,rest) : {Nat.is_le(_,cap) == True{} : Bool} room : {Nat.is_le(1n+Nat.add(position,rest),cap) == True{} : Bool}}
      +next = Components.next_word(at,position,Machine.capacity(limits),Limits.capacity_fits(limits),exact,
        Order.transitive(1n+position,1n+Nat.add(position,rest),cap,Order.le_add(position,rest),shifted))
      {%Equal.sym(Nat,Nat.add(position,1n+rest),1n+Nat.add(position,rest),Addition.successor_right(position,rest)) : {after(rest,(at + 1 : U32)) == U32.from_nat(_) : U32}
        after_word(rest,(at + 1 : U32),1n+position,limits,next,shifted) : {after(rest,(at + 1 : U32)) == U32.from_nat(Nat.add(position,1n+rest)) : U32}}

def row_cells(row: List<&2,Tasks.Reading>) -> {ViewFacts.RowCells(row) == ShapeLaws.elements(row_extents(row)) : Nat}:
  match row:
    case Nil{}: {==}
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Equal.cong(Nat,Nat,cells => Nat.mul(U32.to_nat(extent),cells),ViewFacts.RowCells(rest),ShapeLaws.elements(row_extents(rest)),row_cells(rest))

def row_length(row: List<&2,Tasks.Reading>) -> {List.length(&2,Tasks.Reading,row) == List.length(&2,U32,row_extents(row)) : Nat}:
  match row:
    case Nil{}: {==}
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Equal.cong(Nat,Nat,count => 1n+count,List.length(&2,Tasks.Reading,rest),List.length(&2,U32,row_extents(rest)),row_length(rest))

def extents_appended(left: List<&2,Tasks.Reading>,+right: List<&2,Tasks.Reading>) ->
  {row_extents(List.append(&2,Tasks.Reading,left,right)) == List.append(&2,U32,row_extents(left),row_extents(right)) : List<&2,U32>}:
  match left:
    case Nil{}: {==}
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Equal.cong(List<&2,U32>,List<&2,U32>,values => extent <> values,row_extents(List.append(&2,Tasks.Reading,rest,right)),
        List.append(&2,U32,row_extents(rest),row_extents(right)),extents_appended(rest,right))

def blocked_appended(left: List<&2,Tasks.Reading>,+right: List<&2,Tasks.Reading>,+grid: Grid.Grid,+at: U32,+left_blocked: Blocked(left,grid,at),
  +right_blocked: Blocked(right,grid,after(List.length(&2,Tasks.Reading,left),at))) -> Blocked(List.append(&2,Tasks.Reading,left,right),grid,at):
  match left:
    case Nil{}: right_blocked
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Held.Both{+same,+later} = left_blocked
      Held.Both{same,blocked_appended(rest,right,grid,(at + 1 : U32),later,right_blocked)}

def sources_appended(left: List<&2,Tasks.Reading>,+right: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,
  +left_ok: SourcesOk(left,limits,source,grid),+right_ok: SourcesOk(right,limits,source,grid)) -> SourcesOk(List.append(&2,Tasks.Reading,left,right),limits,source,grid):
  match left:
    case Nil{}: right_ok
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Held.Both{+room,Held.Both{+read_ok,+later}} = left_ok
      Held.Both{room,Held.Both{read_ok,sources_appended(rest,right,limits,source,grid,later,right_ok)}}

# The blocks of a row limit its columns.
def row_limits(row: List<&2,Tasks.Reading>,+grid: Grid.Grid,+at: U32,+blocked: Blocked(row,grid,at)) ->
  Components.LimitsAt([row],List.length(&2,Tasks.Reading,row),row_extents(row),grid,at):
  match row:
    case Nil{}: Unit{}
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Held.Both{+same,+later} = blocked
      Held.Both{Equal.cong(U32,Nat,value => NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(value)),block,Grid.block(grid,at,extent),same),
        row_limits(rest,grid,(at + 1 : U32),later)}

# A row whose readings take blocks of the grid within half a word, over
# extents within the capacity, and read the source has room and reads it.
def row_reads(row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+source_grid: Grid.Grid,+grid: Grid.Grid,+at: U32,
  +blocked: Blocked(row,grid,at),+blocks: ViewReads.BlocksIn(grid,row_extents(row),at,Machine.half_word(limits)),
  +room: {Nat.is_le(ShapeLaws.elements(row_extents(row)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +positive: {Nat.is_lt(0n,ShapeLaws.elements(row_extents(row))) == True{} : Bool},+sources: SourcesOk(row,limits,source,source_grid)) ->
  ViewFacts.ReadsOk(row,limits,source,source_grid):
  match row:
    case Nil{}: Unit{}
    case Con{Tasks.Reading{+extent,+block,+read},+rest}:
      Held.Both{+same,+later} = blocked
      Held.Both{+block_positive,Held.Both{+block_double,+blocks_rest}} = blocks
      Held.Both{+read_room,Held.Both{+read_ok,+sources_rest}} = sources
      +cap = U32.to_nat(Machine.capacity(limits))
      +inner = ShapeLaws.elements(row_extents(rest))
      +whole = Nat.mul(U32.to_nat(extent),inner)
      +extent_positive = RowMajor.left_factor_positive(U32.to_nat(extent),inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(U32.to_nat(extent),inner,positive)
      +extent_room = Order.transitive(U32.to_nat(extent),whole,cap,RowMajor.product_covers(U32.to_nat(extent),inner,inner_positive),room)
      +target_block = Grid.block(grid,at,extent)
      +axis_room = {%Equal.sym(U32,block,target_block,same) : Components.AxisRoom(extent,_,Machine.half_word(limits))
        ViewFacts.axis_room(limits,extent,target_block,extent_room,block_positive,ViewFacts.block_within(limits,target_block,block_positive,block_double)) :
        Components.AxisRoom(extent,block,Machine.half_word(limits))}
      ViewFacts.reads_cons(extent,block,read,rest,limits,source,source_grid,Held.Both{Held.Both{axis_room,read_room},read_ok},
        row_reads(rest,limits,source,source_grid,grid,(at + 1 : U32),later,blocks_rest,
          Order.transitive(inner,whole,cap,DivisionOrder.le_multiple(U32.to_nat(extent),inner,extent_positive),room),inner_positive,sources_rest))

# The classes of a row whose readings take blocks of the grid end inside
# those blocks.
def row_class_ends(row: List<&2,Tasks.Reading>,classes: List<&2,Tasks.Class>,+grid: Grid.Grid,+at: U32,+cap: U32,
  +same: {List.length(&2,Tasks.Class,classes) == List.length(&2,Tasks.Reading,row) : Nat},+blocked: Blocked(row,grid,at),
  +rooms: Components.ReadingsRoom(row,cap),+ends: Components.RowEnds(row,classes)) -> Targets.ClassEnds(classes,grid,row_extents(row),at,cap):
  match row classes:
    case Con{Tasks.Reading{+extent,+block,+read},+rest} Con{+class,+more}:
      Held.Both{+fit,+later} = blocked
      Held.Both{{%fit : Components.EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(_)))
            ViewReads.ends_head(extent,block,read,rest,class,more,ends) : Components.EndIn(class,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(Grid.block(grid,at,extent))))},
        Held.Both{{%fit : {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(_)),U32.to_nat(cap)) == True{} : Bool}
              ViewReads.room_sum(extent,block,read,rest,cap,rooms) : {Nat.is_le(Nat.add(U32.to_nat(extent),U32.to_nat(Grid.block(grid,at,extent))),U32.to_nat(cap)) == True{} : Bool}},
          row_class_ends(rest,more,grid,(at + 1 : U32),cap,Components.length_back(List.length(&2,Tasks.Class,more),List.length(&2,Tasks.Reading,rest),same),later,
            ViewReads.rooms_tail(extent,block,read,rest,cap,rooms),ViewReads.ends_tail(extent,block,read,rest,class,more,ends))}}
    case Nil{} Nil{}: Unit{}
    case Nil{} Con{+class,+more}:
      Empty.absurd(Targets.ClassEnds(Con{class,more},grid,row_extents(Nil{}),at,cap),
        Components.zero_successor(List.length(&2,Tasks.Class,more),Equal.sym(Nat,1n+List.length(&2,Tasks.Class,more),0n,same)))
    case Con{+head,+rest} Nil{}:
      Empty.absurd(Targets.ClassEnds(Nil{},grid,row_extents(Con{head,rest}),at,cap),Components.zero_successor(List.length(&2,Tasks.Reading,rest),same))

# ---- The readings of links ----

def lanes_positive(list: List<&2,U32>,index: Nat,+positive: {Nat.is_lt(0n,ShapeLaws.elements(list)) == True{} : Bool},
  +inside: {Nat.is_lt(index,List.length(&2,U32,list)) == True{} : Bool}) -> {Nat.is_lt(0n,U32.to_nat(Grid.lanes_at(list,index))) == True{} : Bool}:
  match list index:
    case Con{+extent,+rest} 0n: RowMajor.left_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive)
    case Con{+extent,+rest} 1n+ +k: lanes_positive(rest,k,DivisionOrder.right_factor_positive(U32.to_nat(extent),ShapeLaws.elements(rest),positive),inside)
    case Nil{} +index: Empty.absurd({Nat.is_lt(0n,U32.to_nat(Grid.lanes_at(Nil{},index))) == True{} : Bool},Order.below_zero(index,inside))

# The read of a source cell axis of an extent, one index a step, has room at
# half a word and reads that axis.
def linked_read(+limits: Machine.Limits,+cells: List<&2,U32>,+place: Nat,+extent: U32,+lanes: {Grid.lanes_at(cells,place) == extent : U32},
  +below: {Nat.is_lt(place,List.length(&2,U32,cells)) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(cells),cells,0,Machine.half_word(limits))) ->
  Held.Both<Components.ReadRoom(Tiled.read_at(Tensors.grid_of(cells),U32.from_nat(place),extent,1,1,1),Nat.sub(U32.to_nat(extent),1n),Machine.half_word(limits)),
    ViewFacts.ReadOk(Tiled.read_at(Tensors.grid_of(cells),U32.from_nat(place),extent,1,1,1),cells,Tensors.grid_of(cells))>:
  +grid = Tensors.grid_of(cells)
  +cap = U32.to_nat(Machine.capacity(limits))
  +half = U32.to_nat(Machine.half_word(limits))
  +place_room = Bounds.strict_le(place,cap,Bounds.nat_trans(place,List.length(&2,U32,cells),cap,below,cells_rank))
  +lanes_room = Order.transitive(U32.to_nat(Grid.lanes_at(cells,place)),ShapeLaws.elements(cells),cap,RelaneFacts.lanes_le_elements(cells,place,cells_positive,below),cells_room)
  +extent_room = {%lanes : {Nat.is_le(U32.to_nat(_),cap) == True{} : Bool} lanes_room : {Nat.is_le(U32.to_nat(extent),cap) == True{} : Bool}}
  +extent_positive = {%lanes : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool} lanes_positive(cells,place,cells_positive,below) :
    {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}}
  +extent_at = {%lanes : ViewFacts.ExtentAt(cells,0,U32.from_nat(place),_)
    ViewFacts.extent_index(cells,place,0,0n,limits,{==},cells_rank,place_room,below) : ViewFacts.ExtentAt(cells,0,U32.from_nat(place),extent)}
  +block_double = {%lanes : {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(place),_),1))),half) == True{} : Bool}
    RelaneFacts.block_index(cells,place,grid,0,0n,limits,{==},cells_rank,place_room,below,blocks) :
    {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(place),extent),1))),half) == True{} : Bool}}
  +axis_below = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(place)),place,Bounds.roundtrip(place,Machine.capacity(limits),Limits.capacity_fits(limits),place_room)) :
      {Nat.is_lt(_,List.length(&2,U32,cells)) == True{} : Bool}
    below : {Nat.is_lt(U32.to_nat(U32.from_nat(place)),List.length(&2,U32,cells)) == True{} : Bool}}
  Held.Both{Held.Both{{==},Held.Both{RelaneFacts.unit_reach(limits,extent,extent_room,extent_positive),Held.Both{ViewFacts.two_half(limits),
      Held.Both{Limits.within_half(limits,U32.to_nat(extent),extent_room),block_double}}}},
    Held.Both{{==},Held.Both{extent_at,axis_below}}}

# The read of the anchors: the source groups the anchor lanes cover, a
# width of lanes per anchor.
def AnchorOk(+limits: Machine.Limits,+cells: List<&2,U32>,+axis: U32,+source_groups: U32,+width: U32,+anchors: U32) -> Data:
  Held.Both<Components.ReadRoom(Tiled.read_at(Tensors.grid_of(cells),axis,source_groups,width,width,8),Nat.sub(U32.to_nat(anchors),1n),Machine.half_word(limits)),
    ViewFacts.ReadOk(Tiled.read_at(Tensors.grid_of(cells),axis,source_groups,width,width,8),cells,Tensors.grid_of(cells))>

# The readings of links that read their source cell axes and the anchors
# have room at half a word and read the source cells.
def links_sources(links: List<&2,Tiled.Link>,shape: List<&2,U32>,+limits: Machine.Limits,+cells: List<&2,U32>,+target_grid: Grid.Grid,+lane: Nat,+position: Grid.Position,
  +width: U32,+source_groups: U32,+groups_axis: U32,+index: U32,+anchors: U32,+linked: Reads.LinksIn(links,shape,cells,lane,position,anchors),
  +anchor_ok: AnchorOk(limits,cells,groups_axis,source_groups,width,anchors),
  +cells_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(cells),cells,0,Machine.half_word(limits))) ->
  SourcesOk(Tiled.link_readings(links,shape,Tiled.Sides{Tensors.grid_of(cells),target_grid,lane,position,width,source_groups,groups_axis},index),limits,cells,Tensors.grid_of(cells)):
  match links shape:
    case Con{Tiled.Lane{},+links} Con{+extent,+shape}:
      links_sources(links,shape,limits,cells,target_grid,lane,position,width,source_groups,groups_axis,index,anchors,linked,anchor_ok,cells_room,cells_positive,cells_rank,blocks)
    case Con{Tiled.Linked{+axis},+links} Con{+extent,+shape}:
      KernelGemm.Both{+lanes,KernelGemm.Both{+below,+later}} = linked
      +sides = {Tiled.Sides{Tensors.grid_of(cells),target_grid,lane,position,width,source_groups,groups_axis} : Tiled.Sides}
      sources_cons(extent,Grid.block(target_grid,index,extent),Tiled.read_at(Tensors.grid_of(cells),U32.from_nat(Reads.cell_place(lane,position,axis)),extent,1,1,1),
        Tiled.link_readings(links,shape,sides,(index + 1 : U32)),limits,cells,Tensors.grid_of(cells),
        linked_read(limits,cells,Reads.cell_place(lane,position,axis),extent,lanes,below,cells_room,cells_positive,cells_rank,blocks),
        links_sources(links,shape,limits,cells,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32),anchors,later,anchor_ok,cells_room,cells_positive,cells_rank,blocks))
    case Con{Tiled.Anchor{},+links} Con{+extent,+shape}:
      KernelGemm.Both{+same,+later} = linked
      +sides = {Tiled.Sides{Tensors.grid_of(cells),target_grid,lane,position,width,source_groups,groups_axis} : Tiled.Sides}
      sources_cons(extent,Grid.block(target_grid,index,extent),Tiled.read_at(Tensors.grid_of(cells),groups_axis,source_groups,width,width,8),
        Tiled.link_readings(links,shape,sides,(index + 1 : U32)),limits,cells,Tensors.grid_of(cells),
        {%Equal.sym(U32,extent,anchors,same) : AnchorOk(limits,cells,groups_axis,source_groups,width,_) anchor_ok : AnchorOk(limits,cells,groups_axis,source_groups,width,extent)},
        links_sources(links,shape,limits,cells,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32),anchors,later,anchor_ok,cells_room,cells_positive,cells_rank,blocks))
    case Con{Tiled.Alone{},+links} Con{+extent,+shape}:
      Held.Both{Unit{},Held.Both{Unit{},
        links_sources(links,shape,limits,cells,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32),anchors,linked,anchor_ok,cells_room,cells_positive,cells_rank,blocks)}}
    case Nil{} +shape: Unit{}
    case Con{+link,+links} Nil{}: Unit{}

def links_blocked(links: List<&2,Tiled.Link>,shape: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+lane: Nat,+position: Grid.Position,+width: U32,
  +source_groups: U32,+groups_axis: U32,+index: U32) ->
  Blocked(Tiled.link_readings(links,shape,Tiled.Sides{source_grid,target_grid,lane,position,width,source_groups,groups_axis},index),target_grid,index):
  match links shape:
    case Con{Tiled.Lane{},+links} Con{+extent,+shape}: links_blocked(links,shape,source_grid,target_grid,lane,position,width,source_groups,groups_axis,index)
    case Con{Tiled.Linked{+axis},+links} Con{+extent,+shape}:
      Held.Both{{==},links_blocked(links,shape,source_grid,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32))}
    case Con{Tiled.Anchor{},+links} Con{+extent,+shape}:
      Held.Both{{==},links_blocked(links,shape,source_grid,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32))}
    case Con{Tiled.Alone{},+links} Con{+extent,+shape}:
      Held.Both{{==},links_blocked(links,shape,source_grid,target_grid,lane,position,width,source_groups,groups_axis,(index + 1 : U32))}
    case Nil{} +shape: Unit{}
    case Con{+link,+links} Nil{}: Unit{}

def links_extents(links: List<&2,Tiled.Link>,shape: List<&2,U32>,+sides: Tiled.Sides,+index: U32) ->
  {row_extents(Tiled.link_readings(links,shape,sides,index)) == Reads.link_extents(links,shape) : List<&2,U32>}:
  match links shape:
    case Con{Tiled.Lane{},+links} Con{+extent,+shape}: links_extents(links,shape,sides,index)
    case Con{Tiled.Linked{+axis},+links} Con{+extent,+shape}:
      Equal.cong(List<&2,U32>,List<&2,U32>,values => extent <> values,row_extents(Tiled.link_readings(links,shape,sides,(index + 1 : U32))),Reads.link_extents(links,shape),
        links_extents(links,shape,sides,(index + 1 : U32)))
    case Con{Tiled.Anchor{},+links} Con{+extent,+shape}:
      Equal.cong(List<&2,U32>,List<&2,U32>,values => extent <> values,row_extents(Tiled.link_readings(links,shape,sides,(index + 1 : U32))),Reads.link_extents(links,shape),
        links_extents(links,shape,sides,(index + 1 : U32)))
    case Con{Tiled.Alone{},+links} Con{+extent,+shape}:
      Equal.cong(List<&2,U32>,List<&2,U32>,values => extent <> values,row_extents(Tiled.link_readings(links,shape,sides,(index + 1 : U32))),Reads.link_extents(links,shape),
        links_extents(links,shape,sides,(index + 1 : U32)))
    case Nil{} +shape: {==}
    case Con{+link,+links} Nil{}: {==}

# ---- The readings that cross the source ----

def crossing_blocked(+grid: Grid.Grid,cells: List<&2,U32>,+index: U32) -> Blocked(Tiled.crossing_readings(grid,cells,index),grid,index):
  match cells:
    case Nil{}: Unit{}
    case Con{+extent,+rest}: Held.Both{{==},crossing_blocked(grid,rest,(index + 1 : U32))}

def crossing_extents(+grid: Grid.Grid,cells: List<&2,U32>,+index: U32) -> {row_extents(Tiled.crossing_readings(grid,cells,index)) == cells : List<&2,U32>}:
  match cells:
    case Nil{}: {==}
    case Con{+extent,+rest}:
      Equal.cong(List<&2,U32>,List<&2,U32>,values => extent <> values,row_extents(Tiled.crossing_readings(grid,rest,(index + 1 : U32))),rest,
        crossing_extents(grid,rest,(index + 1 : U32)))

def crossing_sources(+limits: Machine.Limits,+source: List<&2,U32>,+grid: Grid.Grid,cells: List<&2,U32>,+index: U32) ->
  SourcesOk(Tiled.crossing_readings(grid,cells,index),limits,source,Tensors.grid_of(source)):
  match cells:
    case Nil{}: Unit{}
    case Con{+extent,+rest}: Held.Both{Unit{},Held.Both{Unit{},crossing_sources(limits,source,grid,rest,(index + 1 : U32))}}

# ---- The read of the anchors ----

# Reads a width apart, a width each, reach the count of widths.
def width_reach(count: Nat,+width: Nat,+positive: {Nat.is_lt(0n,count) == True{} : Bool}) -> {Nat.add(Nat.mul(width,Nat.sub(count,1n)),width) == Nat.mul(count,width) : Nat}:
  match count:
    case 0n: Empty.absurd({Nat.add(Nat.mul(width,Nat.sub(0n,1n)),width) == Nat.mul(0n,width) : Nat},Order.below_zero(0n,positive))
    case 1n+ +rest:
      Equal.trans(Nat,Nat.add(Nat.mul(width,Nat.sub(1n+rest,1n)),width),Nat.add(Nat.mul(width,rest),width),Nat.mul(1n+rest,width),
        Equal.cong(Nat,Nat,number => Nat.add(Nat.mul(width,number),width),Nat.sub(1n+rest,1n),rest,Order.subtract_zero(rest)),
        Equal.trans(Nat,Nat.add(Nat.mul(width,rest),width),Nat.add(Nat.mul(rest,width),width),Nat.mul(1n+rest,width),
          Equal.cong(Nat,Nat,number => Nat.add(number,width),Nat.mul(width,rest),Nat.mul(rest,width),RowMajor.multiply_commutative(width,rest)),
          Order.commutative(Nat.mul(rest,width),width)))

# The anchors read the source groups at their place among the cells.
def anchor_at(+limits: Machine.Limits,+cells: List<&2,U32>,+place: Nat,+source_groups: U32,+width: U32,+anchors: U32,+lanes: U32,
  +groups_at: {Grid.lanes_at(cells,place) == source_groups : U32},+below: {Nat.is_lt(place,List.length(&2,U32,cells)) == True{} : Bool},
  +cells_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(cells),cells,0,Machine.half_word(limits)),
  +lanes_exact: {U32.to_nat(lanes) == Nat.mul(U32.to_nat(anchors),U32.to_nat(width)) : Nat},+lane_groups: TiledValues.Groups<limits,lanes>,
  +lanes_positive: {Nat.is_lt(0n,U32.to_nat(lanes)) == True{} : Bool}) -> AnchorOk(limits,cells,U32.from_nat(place),source_groups,width,anchors):
  TiledValues.Groups{+exact,+fewer,+eights} = lane_groups
  +grid = Tensors.grid_of(cells)
  +cap = U32.to_nat(Machine.capacity(limits))
  +half = U32.to_nat(Machine.half_word(limits))
  +count = U32.to_nat(anchors)
  +size = U32.to_nat(width)
  +covered_at = Nat.mul(U32.to_nat(Grid.groups(lanes)),8n)
  +product = {%lanes_exact : {Nat.is_lt(0n,_) == True{} : Bool} lanes_positive : {Nat.is_lt(0n,Nat.mul(count,size)) == True{} : Bool}}
  +count_positive = RowMajor.left_factor_positive(count,size,product)
  +size_positive = DivisionOrder.right_factor_positive(count,size,product)
  +covered = {%Equal.sym(Nat,U32.to_nat(Grid.groups(lanes)),NatCeiling.ceiling(U32.to_nat(lanes),8n),exact) : {Nat.is_le(U32.to_nat(lanes),Nat.mul(_,8n)) == True{} : Bool}
    NatCeiling.covers(U32.to_nat(lanes),8n,{==}) : {Nat.is_le(U32.to_nat(lanes),covered_at) == True{} : Bool}}
  +reach_lanes = Order.transitive(Nat.mul(count,size),covered_at,half,
    {%lanes_exact : {Nat.is_le(_,covered_at) == True{} : Bool} covered : {Nat.is_le(Nat.mul(count,size),covered_at) == True{} : Bool}},eights)
  +reach = {%Equal.sym(Nat,Nat.add(Nat.mul(size,Nat.sub(count,1n)),size),Nat.mul(count,size),width_reach(count,size,count_positive)) : {Nat.is_le(_,half) == True{} : Bool}
    reach_lanes : {Nat.is_le(Nat.add(Nat.mul(size,Nat.sub(count,1n)),size),half) == True{} : Bool}}
  +place_room = Bounds.strict_le(place,cap,Bounds.nat_trans(place,List.length(&2,U32,cells),cap,below,cells_rank))
  +groups_room = Order.transitive(U32.to_nat(Grid.lanes_at(cells,place)),ShapeLaws.elements(cells),cap,RelaneFacts.lanes_le_elements(cells,place,cells_positive,below),cells_room)
  +bound_room = {%groups_at : {Nat.is_le(U32.to_nat(_),half) == True{} : Bool}
    Limits.within_half(limits,U32.to_nat(Grid.lanes_at(cells,place)),groups_room) : {Nat.is_le(U32.to_nat(source_groups),half) == True{} : Bool}}
  +extent_at = {%groups_at : ViewFacts.ExtentAt(cells,0,U32.from_nat(place),_)
    ViewFacts.extent_index(cells,place,0,0n,limits,{==},cells_rank,place_room,below) : ViewFacts.ExtentAt(cells,0,U32.from_nat(place),source_groups)}
  +block_double = {%groups_at : {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(place),_),1))),half) == True{} : Bool}
    RelaneFacts.block_index(cells,place,grid,0,0n,limits,{==},cells_rank,place_room,below,blocks) :
    {Nat.is_le(Nat.double(U32.to_nat(U32.max(Grid.block(grid,U32.from_nat(place),source_groups),1))),half) == True{} : Bool}}
  +axis_below = {%Equal.sym(Nat,U32.to_nat(U32.from_nat(place)),place,Bounds.roundtrip(place,Machine.capacity(limits),Limits.capacity_fits(limits),place_room)) :
      {Nat.is_lt(_,List.length(&2,U32,cells)) == True{} : Bool}
    below : {Nat.is_lt(U32.to_nat(U32.from_nat(place)),List.length(&2,U32,cells)) == True{} : Bool}}
  Held.Both{Held.Both{size_positive,Held.Both{reach,Held.Both{Bounds.as_nat_le(16,Machine.half_word(limits),Limits.sixteen_within_half(limits)),
      Held.Both{bound_room,block_double}}}},
    Held.Both{{==},Held.Both{extent_at,axis_below}}}

def groups_fewer(+limits: Machine.Limits,+lanes: U32,+lane_groups: TiledValues.Groups<limits,lanes>) ->
  {Nat.is_le(U32.to_nat(Grid.groups(lanes)),U32.to_nat(lanes)) == True{} : Bool}:
  TiledValues.Groups{+exact,+fewer,+eights} = lane_groups
  fewer

def anchor_with(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+lanes: U32,+width: U32,+anchors: U32,
  +cells_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,Grid.groups(lanes)))) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(Grid.placed(True{},position,others,Grid.groups(lanes))),Grid.placed(True{},position,others,Grid.groups(lanes)),0,Machine.half_word(limits)),
  +lanes_exact: {U32.to_nat(lanes) == Nat.mul(U32.to_nat(anchors),U32.to_nat(width)) : Nat},+lane_groups: TiledValues.Groups<limits,lanes>,
  +groups_facts: Held.Both<{Grid.lanes_at(Grid.placed(True{},position,others,Grid.groups(lanes)),RelaneFacts.groups_place(position,Grid.placed(True{},position,others,Grid.groups(lanes)))) ==
      Grid.groups(lanes) : U32},
    Held.Both<{Nat.is_lt(RelaneFacts.groups_place(position,Grid.placed(True{},position,others,Grid.groups(lanes))),List.length(&2,U32,Grid.placed(True{},position,others,Grid.groups(lanes)))) ==
        True{} : Bool},
      {Nat.is_lt(0n,List.length(&2,U32,Grid.placed(True{},position,others,Grid.groups(lanes)))) == True{} : Bool}>>) ->
  AnchorOk(limits,Grid.placed(True{},position,others,Grid.groups(lanes)),Reads.groups_axis_of(position,Grid.placed(True{},position,others,Grid.groups(lanes))),Grid.groups(lanes),
    width,anchors):
  Held.Both{+groups_at,Held.Both{+below,+rank_positive}} = groups_facts
  +cells = Grid.placed(True{},position,others,Grid.groups(lanes))
  +groups = U32.to_nat(Grid.groups(lanes))
  +groups_positive = RowMajor.left_factor_positive(groups,ShapeLaws.elements(others),
    {%Public.placed_cells(position,others,Grid.groups(lanes)) : {Nat.is_lt(0n,_) == True{} : Bool} cells_positive :
      {Nat.is_lt(0n,Nat.mul(groups,ShapeLaws.elements(others))) == True{} : Bool}})
  +lanes_positive = Order.lt_le(0n,groups,U32.to_nat(lanes),groups_positive,groups_fewer(limits,lanes,lane_groups))
  %Equal.sym(U32,Reads.groups_axis_of(position,cells),U32.from_nat(RelaneFacts.groups_place(position,cells)),RelaneFacts.groups_word(position,cells,limits,cells_rank,rank_positive)) :
    AnchorOk(limits,cells,_,Grid.groups(lanes),width,anchors)
  anchor_at(limits,cells,RelaneFacts.groups_place(position,cells),Grid.groups(lanes),width,anchors,lanes,groups_at,below,cells_room,cells_positive,cells_rank,blocks,
    lanes_exact,lane_groups,lanes_positive)

# The anchors of a split read the source groups.
def anchor_ok(+position: Grid.Position,+limits: Machine.Limits,+others: List<&2,U32>,+lanes: U32,+width: U32,+anchors: U32,
  +cells_room: {Nat.is_le(ShapeLaws.elements(Grid.placed(True{},position,others,Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(Grid.placed(True{},position,others,Grid.groups(lanes)))) == True{} : Bool},
  +cells_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,Grid.groups(lanes))),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(Grid.placed(True{},position,others,Grid.groups(lanes))),Grid.placed(True{},position,others,Grid.groups(lanes)),0,Machine.half_word(limits)),
  +lanes_exact: {U32.to_nat(lanes) == Nat.mul(U32.to_nat(anchors),U32.to_nat(width)) : Nat},+lane_groups: TiledValues.Groups<limits,lanes>) ->
  AnchorOk(limits,Grid.placed(True{},position,others,Grid.groups(lanes)),Reads.groups_axis_of(position,Grid.placed(True{},position,others,Grid.groups(lanes))),Grid.groups(lanes),
    width,anchors):
  anchor_with(position,limits,others,lanes,width,anchors,cells_room,cells_positive,cells_rank,blocks,lanes_exact,lane_groups,
    RelaneFacts.placed_groups(position,others,Grid.groups(lanes)))

# ---- The row of a split ----

# What a row needs of its readings: they take the blocks of the target cells,
# read the target extents and read the source cells.
def RowParts(+row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+target: List<&2,U32>) -> Data:
  Held.Both<Blocked(row,Tensors.grid_of(target),0),Held.Both<{row_extents(row) == target : List<&2,U32>},SourcesOk(row,limits,source,Tensors.grid_of(source))>>

# The row of links that pair: the links, then the groups innermost or the
# groups, then the links outermost.
def linked_parts(position: Grid.Position,+limits: Machine.Limits,+links: List<&2,Tiled.Link>,+shape: List<&2,U32>,+others: List<&2,U32>,+groups: U32,+cells: List<&2,U32>,
  +lane: Nat,+width: U32,+source_groups: U32,+anchors: U32,+linked: Reads.LinksIn(links,shape,cells,lane,position,anchors),
  +extents: {Reads.link_extents(links,shape) == others : List<&2,U32>},+anchor_ok: AnchorOk(limits,cells,Reads.groups_axis_of(position,cells),source_groups,width,anchors),
  +cells_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},+cells_rank: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+blocks: ViewReads.BlocksIn(Tensors.grid_of(cells),cells,0,Machine.half_word(limits)),
  +target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  RowParts(Tiled.split_readings(Some{links},shape,Grid.placed(True{},position,others,groups),
      Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,Reads.groups_axis_of(position,cells)},
      Grid.is_outermost(position),groups),limits,cells,Grid.placed(True{},position,others,groups)):
  match position:
    case Grid.Innermost{}:
      +cap = U32.to_nat(Machine.capacity(limits))
      +target = List.append(&2,U32,others,[groups])
      +tgrid = Tensors.grid_of(target)
      +groups_axis = Reads.groups_axis_of(Grid.Innermost{},cells)
      +sides = {Tiled.Sides{Tensors.grid_of(cells),tgrid,lane,Grid.Innermost{},width,source_groups,groups_axis} : Tiled.Sides}
      +front = Tiled.link_readings(links,shape,sides,0)
      +last = (U32.from_nat(List.length(&2,U32,target)) - 1 : U32)
      +tail = {[Tiled.groups_reading(sides,groups,last)] : List<&2,Tasks.Reading>}
      +count = List.length(&2,U32,others)
      +front_extents = Equal.trans(List<&2,U32>,row_extents(front),Reads.link_extents(links,shape),others,links_extents(links,shape,sides,0),extents)
      +front_length = Equal.trans(Nat,List.length(&2,Tasks.Reading,front),List.length(&2,U32,row_extents(front)),count,row_length(front),
        Equal.cong(List<&2,U32>,Nat,values => List.length(&2,U32,values),row_extents(front),others,front_extents))
      +target_length = Public.appended_length(others,groups)
      +count_room = Order.transitive(count,1n+count,cap,Bounds.strict_le(count,1n+count,Order.lt_succ(count,count,Order.reflexive(count))),
        {%target_length : {Nat.is_le(_,cap) == True{} : Bool} target_rank : {Nat.is_le(1n+count,cap) == True{} : Bool}})
      +rank_positive = {%Equal.sym(Nat,List.length(&2,U32,target),1n+count,target_length) : {Nat.is_lt(0n,_) == True{} : Bool} {==} :
        {Nat.is_lt(0n,List.length(&2,U32,target)) == True{} : Bool}}
      +last_word = Equal.trans(U32,last,U32.from_nat(Nat.sub(List.length(&2,U32,target),1n)),U32.from_nat(count),
        RelaneFacts.groups_word(Grid.Innermost{},target,limits,target_rank,rank_positive),
        Equal.cong(Nat,U32,number => U32.from_nat(number),Nat.sub(List.length(&2,U32,target),1n),count,
          Equal.trans(Nat,Nat.sub(List.length(&2,U32,target),1n),Nat.sub(1n+count,1n),count,
            Equal.cong(Nat,Nat,number => Nat.sub(number,1n),List.length(&2,U32,target),1n+count,target_length),Order.subtract_zero(count))))
      +after_exact = {%Equal.sym(Nat,List.length(&2,Tasks.Reading,front),count,front_length) : {after(_,0) == U32.from_nat(count) : U32}
        after_word(count,0,0n,limits,{==},count_room) : {after(List.length(&2,Tasks.Reading,front),0) == U32.from_nat(count) : U32}}
      +block_same = Equal.cong(U32,U32,axis => Grid.block(tgrid,axis,groups),last,after(List.length(&2,Tasks.Reading,front),0),
        Equal.trans(U32,last,U32.from_nat(count),after(List.length(&2,Tasks.Reading,front),0),last_word,
          Equal.sym(U32,after(List.length(&2,Tasks.Reading,front),0),U32.from_nat(count),after_exact)))
      Held.Both{blocked_appended(front,tail,tgrid,0,links_blocked(links,shape,Tensors.grid_of(cells),tgrid,lane,Grid.Innermost{},width,source_groups,groups_axis,0),
          Held.Both{block_same,Unit{}}),
        Held.Both{Equal.trans(List<&2,U32>,row_extents(List.append(&2,Tasks.Reading,front,tail)),List.append(&2,U32,row_extents(front),[groups]),target,
            extents_appended(front,tail),Equal.cong(List<&2,U32>,List<&2,U32>,values => List.append(&2,U32,values,[groups]),row_extents(front),others,front_extents)),
          sources_appended(front,tail,limits,cells,Tensors.grid_of(cells),
            links_sources(links,shape,limits,cells,tgrid,lane,Grid.Innermost{},width,source_groups,groups_axis,0,anchors,linked,anchor_ok,cells_room,cells_positive,cells_rank,blocks),
            Held.Both{Unit{},Held.Both{Unit{},Unit{}}})}}
    case Grid.Outermost{}:
      +target = {groups <> others : List<&2,U32>}
      +tgrid = Tensors.grid_of(target)
      +groups_axis = Reads.groups_axis_of(Grid.Outermost{},cells)
      +sides = {Tiled.Sides{Tensors.grid_of(cells),tgrid,lane,Grid.Outermost{},width,source_groups,groups_axis} : Tiled.Sides}
      +rest = Tiled.link_readings(links,shape,sides,1)
      Held.Both{Held.Both{{==},links_blocked(links,shape,Tensors.grid_of(cells),tgrid,lane,Grid.Outermost{},width,source_groups,groups_axis,1)},
        Held.Both{Equal.cong(List<&2,U32>,List<&2,U32>,values => groups <> values,row_extents(rest),others,
            Equal.trans(List<&2,U32>,row_extents(rest),Reads.link_extents(links,shape),others,links_extents(links,shape,sides,1),extents)),
          Held.Both{Unit{},Held.Both{Unit{},
            links_sources(links,shape,limits,cells,tgrid,lane,Grid.Outermost{},width,source_groups,groups_axis,1,anchors,linked,anchor_ok,cells_room,cells_positive,cells_rank,blocks)}}}}

# The row of a split: its links when they pair, or readings that cross the
# source.
def split_parts(found: Maybe<&2,List<&2,Tiled.Link>>,+position: Grid.Position,+limits: Machine.Limits,+shape: List<&2,U32>,+others: List<&2,U32>,+groups: U32,
  +cells: List<&2,U32>,+lane: Nat,+width: U32,+source_groups: U32,+anchors: U32,+known: Reads.SplitIn(found,shape,cells,lane,position,anchors,others),
  +anchor_ok: AnchorOk(limits,cells,Reads.groups_axis_of(position,cells),source_groups,width,anchors),+cells_room: {Nat.is_le(ShapeLaws.elements(cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+cells_positive: {Nat.is_lt(0n,ShapeLaws.elements(cells)) == True{} : Bool},+cells_rank: {Nat.is_le(List.length(&2,U32,cells),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+blocks: ViewReads.BlocksIn(Tensors.grid_of(cells),cells,0,Machine.half_word(limits)),
  +target_rank: {Nat.is_le(List.length(&2,U32,Grid.placed(True{},position,others,groups)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) ->
  RowParts(Tiled.split_readings(found,shape,Grid.placed(True{},position,others,groups),
      Tiled.Sides{Tensors.grid_of(cells),Tensors.grid_of(Grid.placed(True{},position,others,groups)),lane,position,width,source_groups,Reads.groups_axis_of(position,cells)},
      Grid.is_outermost(position),groups),limits,cells,Grid.placed(True{},position,others,groups)):
  match found:
    case Some{+links}:
      KernelGemm.Both{+linked,+extents} = known
      linked_parts(position,limits,links,shape,others,groups,cells,lane,width,source_groups,anchors,linked,extents,anchor_ok,cells_room,cells_positive,cells_rank,blocks,target_rank)
    case None{}:
      +target = Grid.placed(True{},position,others,groups)
      Held.Both{crossing_blocked(Tensors.grid_of(target),target,0),
        Held.Both{crossing_extents(Tensors.grid_of(target),target,0),crossing_sources(limits,cells,Tensors.grid_of(target),target,0)}}

# ---- What a row gives its gather ----

def parts_reads(+row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+target: List<&2,U32>,+parts: RowParts(row,limits,source,target),
  +target_room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}) -> ViewFacts.ReadsOk(row,limits,source,Tensors.grid_of(source)):
  Held.Both{+blocked,Held.Both{+extents,+sources}} = parts
  +grid = Tensors.grid_of(target)
  +back = Equal.sym(List<&2,U32>,row_extents(row),target,extents)
  row_reads(row,limits,source,Tensors.grid_of(source),grid,0,blocked,
    {%back : ViewReads.BlocksIn(grid,_,0,Machine.half_word(limits))
      ViewFacts.grid_blocks(limits,target,Grid.piece_target(),Held.Both{target_room,Held.Both{target_nonempty,target_rank}}) :
      ViewReads.BlocksIn(grid,row_extents(row),0,Machine.half_word(limits))},
    {%back : {Nat.is_le(ShapeLaws.elements(_),U32.to_nat(Machine.capacity(limits))) == True{} : Bool} target_room :
      {Nat.is_le(ShapeLaws.elements(row_extents(row)),U32.to_nat(Machine.capacity(limits))) == True{} : Bool}},
    {%back : {Nat.is_lt(0n,ShapeLaws.elements(_)) == True{} : Bool} target_nonempty : {Nat.is_lt(0n,ShapeLaws.elements(row_extents(row))) == True{} : Bool}},sources)

def parts_limits(+row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+target: List<&2,U32>,+parts: RowParts(row,limits,source,target)) ->
  Components.LimitsAt([row],Tasks.target_rank([row]),target,Tensors.grid_of(target),0):
  Held.Both{+blocked,Held.Both{+extents,+sources}} = parts
  {%extents : Components.LimitsAt([row],List.length(&2,Tasks.Reading,row),_,Tensors.grid_of(target),0)
    row_limits(row,Tensors.grid_of(target),0,blocked) : Components.LimitsAt([row],List.length(&2,Tasks.Reading,row),target,Tensors.grid_of(target),0)}

def parts_cells(+row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+target: List<&2,U32>,+parts: RowParts(row,limits,source,target)) ->
  {ViewFacts.RowCells(row) == ShapeLaws.elements(target) : Nat}:
  Held.Both{+blocked,Held.Both{+extents,+sources}} = parts
  Equal.trans(Nat,ViewFacts.RowCells(row),ShapeLaws.elements(row_extents(row)),ShapeLaws.elements(target),row_cells(row),
    Equal.cong(List<&2,U32>,Nat,values => ShapeLaws.elements(values),row_extents(row),target,extents))

# A row that reads the source within the machine reads it in its blocks.
def row_ok(+row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+ok: ViewFacts.ReadsOk(row,limits,source,Tensors.grid_of(source)),
  +source_room: {Nat.is_le(ShapeLaws.elements(source),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+source_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(source)) == True{} : Bool},
  +blocks: ViewReads.BlocksIn(Tensors.grid_of(source),source,0,Machine.half_word(limits))) -> Components.RowOk(limits,row,Tensors.grid_of(source),source):
  +grid = Tensors.grid_of(source)
  Held.Both{ViewFacts.reads_room(row,limits,source,grid,ok),Held.Both{ViewFacts.reads_axes(source,row,limits,grid,0,ViewFacts.reads_from(row,limits,source,grid,ok),blocks,
      source_room,source_nonempty),ViewFacts.reads_below(row,limits,source,grid,ok)}}

# The components of a row: its columns and class lists have room at half a
# word, and there is at least one.
def row_components_of(+row: List<&2,Tasks.Reading>,+limits: Machine.Limits,+source: List<&2,U32>,+target: List<&2,U32>,+ok: ViewFacts.ReadsOk(row,limits,source,Tensors.grid_of(source)),
  +cells_same: {ViewFacts.RowCells(row) == ShapeLaws.elements(target) : Nat},
  +target_room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool}) ->
  Held.Both<Components.ComponentsOk([row],Tasks.target_rank([row]),Machine.half_word(limits)),
    {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(Tasks.target_rank([row]),[row]))) == True{} : Bool}>:
  +grid = Tensors.grid_of(source)
  +cap = U32.to_nat(Machine.capacity(limits))
  +back = Equal.sym(Nat,ViewFacts.RowCells(row),ShapeLaws.elements(target),cells_same)
  +row_room = {%back : {Nat.is_le(_,cap) == True{} : Bool} target_room : {Nat.is_le(ViewFacts.RowCells(row),cap) == True{} : Bool}}
  +row_positive = {%back : {Nat.is_lt(0n,_) == True{} : Bool} target_nonempty : {Nat.is_lt(0n,ViewFacts.RowCells(row)) == True{} : Bool}}
  ViewFacts.row_components(row,limits,source,grid,ok,ViewFacts.row_classes(row,limits,source,grid,ok,row_room,row_positive),row_room)

# What a separable row needs of a component: the row reads the source, and
# the classes of the component end inside the target blocks.
def Separate(+limits: Machine.Limits,+row: List<&2,Tasks.Reading>,+source: List<&2,U32>,+target: List<&2,U32>,+component: U32) -> Data:
  Logic.When(Tasks.separable([row]),Held.Both<Components.RowOk(limits,row,Tensors.grid_of(source),source),
    Held.Both<Components.RowEnds(row,Tasks.chosen(Tasks.class_lists(Tasks.target_rank([row]),[row]),component)),
      Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([row]),[row]),component),Tensors.grid_of(target),target,Machine.half_word(limits))>>)

def separate_when(enabled: Bool,+limits: Machine.Limits,+row: List<&2,Tasks.Reading>,+source: List<&2,U32>,+target: List<&2,U32>,+component: U32,
  +held: Held.Both<Components.RowOk(limits,row,Tensors.grid_of(source),source),
    Held.Both<Components.RowEnds(row,Tasks.chosen(Tasks.class_lists(Tasks.target_rank([row]),[row]),component)),
      Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([row]),[row]),component),Tensors.grid_of(target),target,Machine.half_word(limits))>>) ->
  Logic.When(enabled,Held.Both<Components.RowOk(limits,row,Tensors.grid_of(source),source),
    Held.Both<Components.RowEnds(row,Tasks.chosen(Tasks.class_lists(Tasks.target_rank([row]),[row]),component)),
      Targets.TargetClassesOk(Tasks.chosen(Tasks.class_lists(Tasks.target_rank([row]),[row]),component),Tensors.grid_of(target),target,Machine.half_word(limits))>>):
  match enabled:
    case False{}: Unit{}
    case True{}: held

# Every component of a row has what a separable row needs of it.
def row_separate(+limits: Machine.Limits,+row: List<&2,Tasks.Reading>,+source: List<&2,U32>,+target: List<&2,U32>,+component: U32,+parts: RowParts(row,limits,source,target),
  +target_room: {Nat.is_le(ShapeLaws.elements(target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+target_nonempty: {Nat.is_lt(0n,ShapeLaws.elements(target)) == True{} : Bool},+target_rank: {Nat.is_le(List.length(&2,U32,target),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+components_ok: Components.ComponentsOk([row],Tasks.target_rank([row]),Machine.half_word(limits)),
  +inside: {Nat.is_lt(U32.to_nat(component),Components.count_nat(Components.raw_lists(Tasks.target_rank([row]),[row]))) == True{} : Bool},
  +row_ok: Components.RowOk(limits,row,Tensors.grid_of(source),source)) -> Separate(limits,row,source,target,component):
  Held.Both{+blocked,Held.Both{+extents,+sources}} = parts
  Held.Both{+rooms,+more} = row_ok
  +cap = Machine.half_word(limits)
  +cap_ok = Limits.half_word_fits(limits)
  +classes = Tasks.chosen(Tasks.class_lists(Tasks.target_rank([row]),[row]),component)
  +grid = Tensors.grid_of(target)
  +ends = Components.ends_at([row],0,classes,cap,cap_ok,Components.runtime_ends([row],Tasks.target_rank([row]),cap,cap_ok,components_ok,component,inside),{==},
    Components.components_rows([row],Tasks.target_rank([row]),cap,components_ok))
  +class_ends = {%extents : Targets.ClassEnds(classes,grid,_,0,cap)
    row_class_ends(row,classes,grid,0,cap,Targets.chosen_rank(Tasks.target_rank([row]),[row],component),blocked,rooms,ends) : Targets.ClassEnds(classes,grid,target,0,cap)}
  +limits_at = {%extents : Components.LimitsAt([row],List.length(&2,Tasks.Reading,row),_,grid,0)
    row_limits(row,grid,0,blocked) : Components.LimitsAt([row],List.length(&2,Tasks.Reading,row),target,grid,0)}
  separate_when(Tasks.separable([row]),limits,row,source,target,component,
    Held.Both{row_ok,Held.Both{ends,Targets.classes_ok(limits,target,Grid.piece_target(),target_room,target_nonempty,target_rank,[row],component,components_ok,limits_at,
      inside,class_ends)}})
