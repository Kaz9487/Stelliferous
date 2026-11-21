# Rows of readings of several sources that read one target: every row reads
# each target axis with its extent and its block, so the readings of an axis
# share it, the classes of an axis number at least one and at most its
# extent, and the rows make components that fit half a word.

import Base
import ../tensor_tasks.bend as Tasks
import ../tensor_grid.bend as Grid
import ../machine_limits.bend as Machine
import ./machine_limit_bounds.bend as LimitBounds
import ./tensor_components.bend as Components
import ./tensor_view_facts.bend as ViewFacts
import ./tensor_move_readback.bend as Held
import ./tensor_shape_refinement.bend as ShapeLaws
import ./nat_order.bend as Order
import ./nat_ceiling.bend as NatCeiling
import ./row_major_index.bend as RowMajor
import ./nat_division_order.bend as DivisionOrder
import ./word_encoding.bend as WordEncoding

# A row reads the extents from at with the blocks the grid gives them.
def TargetSide(row: List<&2,Tasks.Reading>,extents: List<&2,U32>,+grid: Grid.Grid,+at: U32) -> Data:
  match row extents:
    case Con{Tasks.Reading{+extent,+block,source},row} Con{+target,extents}:
      Held.Both<{extent == target : U32},Held.Both<{block == Grid.block(grid,at,target) : U32},TargetSide(row,extents,grid,(at + 1 : U32))>>
    case Nil{} Nil{}:
      Unit
    case row extents:
      Empty

# Every row reads the extents from at and has room at cap.
def RowsRead(rows: List<&2,List<&2,Tasks.Reading>>,+extents: List<&2,U32>,+grid: Grid.Grid,+at: U32,+cap: U32) -> Data:
  match rows:
    case Nil{}:
      Unit
    case Con{+row,rows}:
      Held.Both<TargetSide(row,extents,grid,at),Held.Both<Components.ReadingsRoom(row,cap),RowsRead(rows,extents,grid,at,cap)>>

# A row that reads the extents has one reading for each.
def side_length(row: List<&2,Tasks.Reading>,extents: List<&2,U32>,+grid: Grid.Grid,+at: U32,+side: TargetSide(row,extents,grid,at)) ->
  {List.length(&2,Tasks.Reading,row) == List.length(&2,U32,extents) : Nat}:
  match row extents:
    case Con{Tasks.Reading{+extent,+block,+source},+rest} Con{+target,+more}:
      Held.Both{+same_extent,Held.Both{+same_block,+later}} = side
      Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,Tasks.Reading,rest),List.length(&2,U32,more),side_length(rest,more,grid,(at + 1 : U32),later))
    case Nil{} Nil{}:
      {==}
    case Nil{} Con{+target,+more}:
      Empty.absurd({List.length(&2,Tasks.Reading,Nil{}) == List.length(&2,U32,Con{target,more}) : Nat},side)
    case Con{Tasks.Reading{+extent,+block,+source},+rest} Nil{}:
      Empty.absurd({List.length(&2,Tasks.Reading,Con{Tasks.Reading{extent,block,source},rest}) == List.length(&2,U32,Nil{}) : Nat},side)

def rows_at(rows: List<&2,List<&2,Tasks.Reading>>,+extents: List<&2,U32>,+grid: Grid.Grid,+at: U32,+cap: U32,+read: RowsRead(rows,extents,grid,at,cap)) ->
  Components.RowsAt(rows,List.length(&2,U32,extents)):
  match rows:
    case Nil{}: Unit{}
    case Con{+row,+rest}:
      Held.Both{+side,Held.Both{+room,+later}} = read
      Held.Both{side_length(row,extents,grid,at,side),rows_at(rest,extents,grid,at,cap,later)}

# Rows that read extents read every one of them: the first row reads as many
# axes as there are extents.
def rows_rank(+rows: List<&2,List<&2,Tasks.Reading>>,+extents: List<&2,U32>,+grid: Grid.Grid,+at: U32,+cap: U32,+read: RowsRead(rows,extents,grid,at,cap),
  +nonempty: {Nat.is_lt(0n,List.length(&2,List<&2,Tasks.Reading>,rows)) == True{} : Bool}) -> {Tasks.target_rank(rows) == List.length(&2,U32,extents) : Nat}:
  match rows:
    case Nil{}: Empty.absurd({Tasks.target_rank(Nil{}) == List.length(&2,U32,extents) : Nat},WordEncoding.false_true(nonempty))
    case Con{+row,+rest}:
      Held.Both{+side,Held.Both{+room,+later}} = read
      side_length(row,extents,grid,at,side)

# Past the first axis the rows read the later extents.
def rows_tails(rows: List<&2,List<&2,Tasks.Reading>>,+extent: U32,+extents: List<&2,U32>,+grid: Grid.Grid,+at: U32,+cap: U32,
  +read: RowsRead(rows,Con{extent,extents},grid,at,cap)) -> RowsRead(Tasks.tails(rows),extents,grid,(at + 1 : U32),cap):
  match rows:
    case Nil{}: Unit{}
    case Con{+row,+rest}:
      match row:
        case Con{Tasks.Reading{+own,+block,+source},+tail}:
          Held.Both{Held.Both{+same_extent,Held.Both{+same_block,+side}},Held.Both{Held.Both{+room,+rooms},+later}} = read
          Held.Both{side,Held.Both{rooms,rows_tails(rest,extent,extents,grid,at,cap,later)}}
        case Nil{}:
          Held.Both{+side,+more} = read
          Empty.absurd(RowsRead(Tasks.tails(Con{Nil{},rest}),extents,grid,(at + 1 : U32),cap),side)

def tails_length(rows: List<&2,List<&2,Tasks.Reading>>) ->
  {List.length(&2,List<&2,Tasks.Reading>,Tasks.tails(rows)) == List.length(&2,List<&2,Tasks.Reading>,rows) : Nat}:
  match rows:
    case Nil{}: {==}
    case Con{+row,+rest}:
      match row:
        case Nil{}:
          Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,List<&2,Tasks.Reading>,Tasks.tails(rest)),List.length(&2,List<&2,Tasks.Reading>,rest),tails_length(rest))
        case Con{+head,+tail}:
          Equal.cong(Nat,Nat,length => 1n+length,List.length(&2,List<&2,Tasks.Reading>,Tasks.tails(rest)),List.length(&2,List<&2,Tasks.Reading>,rest),tails_length(rest))

def tails_nonempty(+rows: List<&2,List<&2,Tasks.Reading>>,+nonempty: {Nat.is_lt(0n,List.length(&2,List<&2,Tasks.Reading>,rows)) == True{} : Bool}) ->
  {Nat.is_lt(0n,List.length(&2,List<&2,Tasks.Reading>,Tasks.tails(rows))) == True{} : Bool}:
  %Equal.sym(Nat,List.length(&2,List<&2,Tasks.Reading>,Tasks.tails(rows)),List.length(&2,List<&2,Tasks.Reading>,rows),tails_length(rows)) : {Nat.is_lt(0n,_) == True{} : Bool}
  nonempty

# ---- The readings of an axis ----

# The first readings of the rows have room.
def column_rooms(rows: List<&2,List<&2,Tasks.Reading>>,+extent: U32,+extents: List<&2,U32>,+grid: Grid.Grid,+at: U32,+cap: U32,
  +read: RowsRead(rows,Con{extent,extents},grid,at,cap)) -> Components.ReadingsRoom(Tasks.heads(rows),cap):
  match rows:
    case Nil{}: Unit{}
    case Con{+row,+rest}:
      match row:
        case Con{Tasks.Reading{+own,+block,+source},+tail}:
          Held.Both{Held.Both{+same_extent,Held.Both{+same_block,+side}},Held.Both{Held.Both{+room,+rooms},+later}} = read
          Held.Both{room,column_rooms(rest,extent,extents,grid,at,cap,later)}
        case Nil{}:
          Held.Both{+side,+more} = read
          Empty.absurd(Components.ReadingsRoom(Tasks.heads(Con{Nil{},rest}),cap),side)

# The first readings of the rows read the first extent with its block.
def column_axis(rows: List<&2,List<&2,Tasks.Reading>>,+extent: U32,+extents: List<&2,U32>,+grid: Grid.Grid,+at: U32,+cap: U32,
  +read: RowsRead(rows,Con{extent,extents},grid,at,cap)) -> Components.SameAxis(Tasks.heads(rows),extent,Grid.block(grid,at,extent)):
  match rows:
    case Nil{}: Unit{}
    case Con{+row,+rest}:
      match row:
        case Con{Tasks.Reading{+own,+block,+source},+tail}:
          Held.Both{Held.Both{+same_extent,Held.Both{+same_block,+side}},Held.Both{+rooms,+later}} = read
          Held.Both{same_extent,Held.Both{same_block,column_axis(rest,extent,extents,grid,at,cap,later)}}
        case Nil{}:
          Held.Both{+side,+more} = read
          Empty.absurd(Components.SameAxis(Tasks.heads(Con{Nil{},rest}),extent,Grid.block(grid,at,extent)),side)

# Readings with room that share an extent and a block make a column.
def column_of(column: List<&2,Tasks.Reading>,+extent: U32,+block: U32,+cap: U32,+rooms: Components.ReadingsRoom(column,cap),+same: Components.SameAxis(column,extent,block)) ->
  Components.ColumnOk(column,cap):
  match column:
    case Nil{}: Unit{}
    case Con{Tasks.Reading{+own,+own_block,+source},+others}:
      Held.Both{+same_extent,Held.Both{+same_block,+rest}} = same
      Held.Both{rooms,
        {%Equal.sym(U32,own_block,block,same_block) : Components.SameAxis(others,own,_)
          {%Equal.sym(U32,own,extent,same_extent) : Components.SameAxis(others,_,block) rest : Components.SameAxis(others,own,block)} :
          Components.SameAxis(others,own,own_block)}}

def columns_ok(extents: List<&2,U32>,+rows: List<&2,List<&2,Tasks.Reading>>,+grid: Grid.Grid,+at: U32,+cap: U32,+read: RowsRead(rows,extents,grid,at,cap)) ->
  Components.ColumnsOk(List.length(&2,U32,extents),rows,cap):
  match extents:
    case Nil{}: Unit{}
    case Con{+extent,+more}:
      Held.Both{column_of(Tasks.heads(rows),extent,Grid.block(grid,at,extent),cap,column_rooms(rows,extent,more,grid,at,cap,read),column_axis(rows,extent,more,grid,at,cap,read)),
        columns_ok(more,Tasks.tails(rows),grid,(at + 1 : U32),cap,rows_tails(rows,extent,more,grid,at,cap,read))}

# ---- The classes of an axis ----

# Starts that begin at zero and rise below the blocks make at least one run
# and no more than the extent.
def starts_classes(+starts: List<&2,U32>,+blocks: U32,+extent: U32,+block: U32,
  +starts_ok: Components.Starts(starts,NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))),
  +extent_positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool},+block_positive: {Nat.is_lt(0n,U32.to_nat(block)) == True{} : Bool}) ->
  Held.Both<{Nat.is_lt(0n,List.length(&2,Tasks.Class,Tasks.runs(starts,blocks))) == True{} : Bool},
    {Nat.is_le(List.length(&2,Tasks.Class,Tasks.runs(starts,blocks)),U32.to_nat(extent)) == True{} : Bool}>:
  Held.Both{+head,+rising} = starts_ok
  +limit = NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(block))
  +back = Equal.sym(Nat,List.length(&2,Tasks.Class,Tasks.runs(starts,blocks)),List.length(&2,U32,starts),ViewFacts.runs_length(starts,blocks))
  Held.Both{{%back : {Nat.is_lt(0n,_) == True{} : Bool}
      ViewFacts.head_nonempty(starts,limit,head,ViewFacts.ceiling_positive(U32.to_nat(extent),U32.to_nat(block),extent_positive,block_positive)) :
      {Nat.is_lt(0n,List.length(&2,Tasks.Class,Tasks.runs(starts,blocks))) == True{} : Bool}},
    {%back : {Nat.is_le(_,U32.to_nat(extent)) == True{} : Bool}
      Order.transitive(List.length(&2,U32,starts),limit,U32.to_nat(extent),ViewFacts.rising_count(starts,0n,limit,rising,Order.zero_le(limit)),
        NatCeiling.bounded(U32.to_nat(extent),U32.to_nat(block),block_positive)) :
      {Nat.is_le(List.length(&2,Tasks.Class,Tasks.runs(starts,blocks)),U32.to_nat(extent)) == True{} : Bool}}}

def ClassCount(+column: List<&2,Tasks.Reading>,+extent: U32) -> Data:
  Held.Both<{Nat.is_lt(0n,List.length(&2,Tasks.Class,Tasks.axis_classes(column))) == True{} : Bool},
    {Nat.is_le(List.length(&2,Tasks.Class,Tasks.axis_classes(column)),U32.to_nat(extent)) == True{} : Bool}>

# The first reading of a column with the room of the rest gives the classes
# of the axis.
def first_classes(+own: U32,+block: U32,+source: Tasks.Source,+others: List<&2,Tasks.Reading>,+limits: Machine.Limits,
  +room: Components.ReadingRoom(own,block,source,Machine.half_word(limits)),+rooms: Components.ReadingsRoom(others,Machine.half_word(limits)),
  +same: Components.SameAxis(others,own,block),+own_positive: {Nat.is_lt(0n,U32.to_nat(own)) == True{} : Bool}) ->
  ClassCount(Con{Tasks.Reading{own,block,source},others},own):
  Held.Both{+axis_facts,+read_room} = room
  Held.Both{+blocks_value,Held.Both{+blocks_room,Held.Both{+block_positive,+sum_room}}} = axis_facts
  +cap = Machine.half_word(limits)
  starts_classes(Tasks.shared_starts(Tasks.class_starts(Tasks.classes(Tasks.Reading{own,block,source})),Tasks.starts_of(others)),Tasks.target_blocks(own,block),own,block,
    Components.axis_starts(own,block,source,others,cap,LimitBounds.half_word_fits(limits),
      {Held.Both{Held.Both{Held.Both{blocks_value,Held.Both{blocks_room,Held.Both{block_positive,sum_room}}},read_room},rooms} :
        Components.ReadingsRoom(Con{Tasks.Reading{own,block,source},others},cap)},same),
    own_positive,block_positive)

# The classes of the first axis of the rows number at least one and at most
# its extent.
def column_classes(rows: List<&2,List<&2,Tasks.Reading>>,+extent: U32,+extents: List<&2,U32>,+grid: Grid.Grid,+at: U32,+limits: Machine.Limits,
  +read: RowsRead(rows,Con{extent,extents},grid,at,Machine.half_word(limits)),+nonempty: {Nat.is_lt(0n,List.length(&2,List<&2,Tasks.Reading>,rows)) == True{} : Bool},
  +positive: {Nat.is_lt(0n,U32.to_nat(extent)) == True{} : Bool}) -> ClassCount(Tasks.heads(rows),extent):
  match rows:
    case Nil{}: Empty.absurd(ClassCount(Tasks.heads(Nil{}),extent),WordEncoding.false_true(nonempty))
    case Con{+row,+rest}:
      match row:
        case Con{Tasks.Reading{+own,+block,+source},+tail}:
          Held.Both{Held.Both{+same_extent,Held.Both{+same_block,+side}},Held.Both{Held.Both{+room,+rooms},+later}} = read
          +cap = Machine.half_word(limits)
          +others = Tasks.heads(rest)
          +same_block_back = Equal.sym(U32,block,Grid.block(grid,at,extent),same_block)
          +same = {%same_block_back : Components.SameAxis(others,own,_)
            {%Equal.sym(U32,own,extent,same_extent) : Components.SameAxis(others,_,Grid.block(grid,at,extent)) column_axis(rest,extent,extents,grid,at,cap,later) :
              Components.SameAxis(others,own,Grid.block(grid,at,extent))} : Components.SameAxis(others,own,block)}
          %same_extent : ClassCount(Con{Tasks.Reading{own,block,source},others},_)
          first_classes(own,block,source,others,limits,room,column_rooms(rest,extent,extents,grid,at,cap,later),same,
            {%Equal.sym(U32,own,extent,same_extent) : {Nat.is_lt(0n,U32.to_nat(_)) == True{} : Bool} positive : {Nat.is_lt(0n,U32.to_nat(own)) == True{} : Bool}})
        case Nil{}:
          Held.Both{+side,+more} = read
          Empty.absurd(ClassCount(Tasks.heads(Con{Nil{},rest}),extent),side)

# The class lists of the rows have room, count no more than the cells of the
# extents and at least one.
def rows_classes(extents: List<&2,U32>,+rows: List<&2,List<&2,Tasks.Reading>>,+grid: Grid.Grid,+at: U32,+limits: Machine.Limits,
  +read: RowsRead(rows,extents,grid,at,Machine.half_word(limits)),+nonempty: {Nat.is_lt(0n,List.length(&2,List<&2,Tasks.Reading>,rows)) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  ViewFacts.ClassesOk(Components.raw_lists(List.length(&2,U32,extents),rows),ShapeLaws.elements(extents),Machine.half_word(limits)):
  match extents:
    case Nil{}: Held.Both{Unit{},Held.Both{{==},{==}}}
    case Con{+extent,+more}:
      +cap = U32.to_nat(Machine.capacity(limits))
      +size = U32.to_nat(extent)
      +inner = ShapeLaws.elements(more)
      +whole = Nat.mul(size,inner)
      +size_positive = RowMajor.left_factor_positive(size,inner,positive)
      +inner_positive = DivisionOrder.right_factor_positive(size,inner,positive)
      +size_room = Order.transitive(size,whole,cap,RowMajor.product_covers(size,inner,inner_positive),room)
      ViewFacts.classes_cons(Tasks.axis_classes(Tasks.heads(rows)),Components.raw_lists(List.length(&2,U32,more),Tasks.tails(rows)),size,inner,limits,
        column_classes(rows,extent,more,grid,at,limits,read,nonempty,size_positive),
        rows_classes(more,Tasks.tails(rows),grid,(at + 1 : U32),limits,rows_tails(rows,extent,more,grid,at,Machine.half_word(limits),read),tails_nonempty(rows,nonempty),
          Order.transitive(inner,whole,cap,DivisionOrder.le_multiple(size,inner,size_positive),room),inner_positive),
        size_room)

# ---- The limits of every axis ----

def column_limit_of(rows: List<&2,List<&2,Tasks.Reading>>,+extent: U32,+extents: List<&2,U32>,+grid: Grid.Grid,+at: U32,+cap: U32,
  +read: RowsRead(rows,Con{extent,extents},grid,at,cap),+nonempty: {Nat.is_lt(0n,List.length(&2,List<&2,Tasks.Reading>,rows)) == True{} : Bool}) ->
  {Components.column_limit(Tasks.heads(rows)) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(Grid.block(grid,at,extent))) : Nat}:
  match rows:
    case Nil{}:
      Empty.absurd({Components.column_limit(Tasks.heads(Nil{})) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(Grid.block(grid,at,extent))) : Nat},WordEncoding.false_true(nonempty))
    case Con{+row,+rest}:
      match row:
        case Con{Tasks.Reading{+own,+block,+source},+tail}:
          Held.Both{Held.Both{+same_extent,Held.Both{+same_block,+side}},Held.Both{+rooms,+later}} = read
          %same_block : {NatCeiling.ceiling(U32.to_nat(own),U32.to_nat(block)) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(_)) : Nat}
          %same_extent : {NatCeiling.ceiling(U32.to_nat(own),U32.to_nat(block)) == NatCeiling.ceiling(U32.to_nat(_),U32.to_nat(block)) : Nat}
          {==}
        case Nil{}:
          Held.Both{+side,+more} = read
          Empty.absurd({Components.column_limit(Tasks.heads(Con{Nil{},rest})) == NatCeiling.ceiling(U32.to_nat(extent),U32.to_nat(Grid.block(grid,at,extent))) : Nat},side)

# Every target axis of the rows is limited by its blocks in the grid.
def rows_limits(extents: List<&2,U32>,+rows: List<&2,List<&2,Tasks.Reading>>,+grid: Grid.Grid,+at: U32,+cap: U32,+read: RowsRead(rows,extents,grid,at,cap),
  +nonempty: {Nat.is_lt(0n,List.length(&2,List<&2,Tasks.Reading>,rows)) == True{} : Bool}) ->
  Components.LimitsAt(rows,List.length(&2,U32,extents),extents,grid,at):
  match extents:
    case Nil{}: Unit{}
    case Con{+extent,+more}:
      Held.Both{column_limit_of(rows,extent,more,grid,at,cap,read,nonempty),
        rows_limits(more,Tasks.tails(rows),grid,(at + 1 : U32),cap,rows_tails(rows,extent,more,grid,at,cap,read),tails_nonempty(rows,nonempty))}

# ---- The components of the rows ----

def rows_joined(+rows: List<&2,List<&2,Tasks.Reading>>,+extents: List<&2,U32>,+grid: Grid.Grid,+limits: Machine.Limits,
  +read: RowsRead(rows,extents,grid,0,Machine.half_word(limits)),
  +rows_room: {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,rows),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},
  +classes: ViewFacts.ClassesOk(Components.raw_lists(List.length(&2,U32,extents),rows),ShapeLaws.elements(extents),Machine.half_word(limits))) ->
  Held.Both<Components.ComponentsOk(rows,List.length(&2,U32,extents),Machine.half_word(limits)),
    {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(List.length(&2,U32,extents),rows))) == True{} : Bool}>:
  Held.Both{+lists_room,Held.Both{+count_le,+count_positive}} = classes
  +half = Machine.half_word(limits)
  +count = Components.count_nat(Components.raw_lists(List.length(&2,U32,extents),rows))
  Held.Both{Held.Both{rows_at(rows,extents,grid,0,half,read),Held.Both{columns_ok(extents,rows,grid,0,half,read),Held.Both{lists_room,
      Held.Both{LimitBounds.double_bound(limits,count,Order.transitive(count,ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits)),count_le,room)),rows_room}}}},
    count_positive}

# Rows that read extents of cells that fit the capacity make components that
# fit half a word, of at least one class list entry each.
def rows_components(+rows: List<&2,List<&2,Tasks.Reading>>,+extents: List<&2,U32>,+grid: Grid.Grid,+limits: Machine.Limits,
  +read: RowsRead(rows,extents,grid,0,Machine.half_word(limits)),+nonempty: {Nat.is_lt(0n,List.length(&2,List<&2,Tasks.Reading>,rows)) == True{} : Bool},
  +rows_room: {Nat.is_le(List.length(&2,List<&2,Tasks.Reading>,rows),U32.to_nat(Machine.half_word(limits))) == True{} : Bool},
  +room: {Nat.is_le(ShapeLaws.elements(extents),U32.to_nat(Machine.capacity(limits))) == True{} : Bool},+positive: {Nat.is_lt(0n,ShapeLaws.elements(extents)) == True{} : Bool}) ->
  Held.Both<Components.ComponentsOk(rows,List.length(&2,U32,extents),Machine.half_word(limits)),
    {Nat.is_lt(0n,Components.count_nat(Components.raw_lists(List.length(&2,U32,extents),rows))) == True{} : Bool}>:
  rows_joined(rows,extents,grid,limits,read,rows_room,room,rows_classes(extents,rows,grid,0,limits,read,nonempty,room,positive))
