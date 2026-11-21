# The work of one operation: its pieces grouped into tasks, and whether the
# tasks run in parallel. The pieces never move: a task owns the adjacent
# pieces of a complete subtree of the ranges.
import Base
import ./tensor_grid.bend as Grid
import ./tensor_pieces.bend as Pieces
import ./kernel_cost.bend as Cost
import ./default_kernel_cost.bend as DefaultCost
import ./tensor_view.bend as Views
import ./tensor_move.bend as Move
import ./math_choice.bend as Choice

type Group is Data:
  Group{first: U32,count: U32,layout: Pieces.Ranges}

def group_range(limit: U32,first: U32,count: U32) -> Group:
  +first = first
  +count = count
  Group{first,count,Pieces.Range{first,count}}

def group_join(limit: U32,left_ranges: Pieces.Ranges,right_ranges: Pieces.Ranges,left: Group,right: Group) -> Group:
  Group{+first,+left_count,left} = left
  Group{right_first,right_count,right} = right
  +count = (left_count + right_count : U32)
  Group{first,count,Bool.pick(Pieces.Ranges,U32.is_le(count,limit),Pieces.Range{first,count},Pieces.Split{left,right})}

def group_layout(group: Group) -> Pieces.Ranges:
  Group{first,count,layout} = group
  layout

type Size is Data:
  Size{tasks: U32,largest: U32}

def size_leaf(context: Unit,first: U32,count: U32) -> Size:
  Size{1,count}

def size_join(left: Size,right: Size) -> Size:
  Size{left_tasks,left_largest} = left
  Size{right_tasks,right_largest} = right
  Size{(left_tasks + right_tasks : U32),U32.max(left_largest,right_largest)}

def size(layout: Pieces.Ranges) -> Size:
  Pieces.ranges_fold(~Size,~Unit,~size_leaf,~Pieces.plain(~Size,~Unit,~size_join),layout,Unit{})

def divided_up(+count: U32,+parts: U32) -> U32:
  (count / parts + Bool.pick(U32,U32.is_zero((count % parts : U32)),0,1) : U32)

# Complete subtrees of ranges over count cells joined up to one worker's
# share of the cells, up to whole cells, or up to one worker's share of the
# ranges at the largest, whichever is most. Some worker serves that share of
# the ranges in any plan, so joining up to it keeps the longest worker and
# leaves fewer tasks to prepare.
def tasks_sized(+layout: Pieces.Ranges,+count: U32,+pool: U32,+whole: U32,found: Size) -> Pieces.Ranges:
  Size{+ranges,+largest} = found
  +limit = U32.max(U32.max(divided_up(count,pool),whole),(divided_up(ranges,pool) * largest : U32))
  group_layout(Pieces.ranges_fold(~Group,~U32,~group_range,~group_join,layout,limit))

def tasks_over(+layout: Pieces.Ranges,+count: U32,+workers: U32,+whole: U32) -> Pieces.Ranges:
  tasks_sized(layout,count,U32.max(workers,1),whole,size(layout))

# What the parallel plan pays and the serial plan does not: for every task
# after the first the bytes it allocates or clones and the boundary it
# prepares, and once the finish that joins the tasks.
type Extra is Data:
  Extra{task_bytes: F32,boundary_ns: F32,finish_ns: F32}

def no_extra() -> Extra:
  Extra{0.0,0.0,0.0}

# The tasks of an operation and whether they run in parallel. The serial plan
# is one task: one owner serves every piece in order and shares nothing.
type Decision is Data:
  Decision{parallel: Bool,tasks: Pieces.Ranges}

def whole_bounds(bounds: U32 & U32) -> Pieces.Ranges:
  (first,count) = bounds
  Pieces.Range{first,count}

def planned(+parallel: Bool,+layout: Pieces.Ranges) -> Decision:
  Decision{parallel,Bool.pick(Pieces.Ranges,parallel,layout,whole_bounds(Pieces.bounds(layout)))}

# Costs both plans pay cancel. The serial plan computes every cell. The
# parallel plan adds two wake rounds, waves of its largest task, the dispatch
# of every task and its extra.
def parallel_ns(+workers: U32,+cell_ns: F32,config: Cost.Configuration,extra: Extra,+count: U32,+largest: U32) -> F32:
  Cost.Configuration{row,+byte_ns,step,move,grain,+wake,+dispatch} = config
  Extra{+task_bytes,+boundary_ns,+finish_ns} = extra
  +others = U32.to_f32((U32.max(count,1) - 1 : U32))
  (2.0 * wake + Cost.region_ns(workers,count,(U32.to_f32(largest) * cell_ns : F32)) + U32.to_f32(count) * dispatch
    + others * (task_bytes * byte_ns + boundary_ns : F32) + finish_ns : F32)

def worth(+workers: U32,+cell_ns: F32,config: Cost.Configuration,extra: Extra,+total: U32,size: Size) -> Bool:
  Size{+count,+largest} = size
  U32.is_lt(1,workers) && U32.is_lt(1,count) && F32.is_lt(parallel_ns(workers,cell_ns,config,extra,count,largest),(U32.to_f32(total) * cell_ns : F32))

def decided(+workers: U32,+cell_ns: F32,config: Cost.Configuration,extra: Extra,+total: U32,+layout: Pieces.Ranges) -> Decision:
  planned(worth(workers,cell_ns,config,extra,total,size(layout)),layout)

# What the tasks a pool groups cost, from the ranges alone: a complete subtree
# of the ranges holds a power of two of them, so the tasks hold the largest
# power of two of ranges the limit of the pool admits.
def pool_ns(+workers: U32,+cell_ns: F32,+config: Cost.Configuration,+extra: Extra,+count: U32,+whole: U32,+ranges: U32,+largest: U32,+pool: U32) -> F32:
  +per_range = U32.max(largest,1)
  +limit = U32.max(U32.max(divided_up(count,pool),whole),(divided_up(ranges,pool) * per_range : U32))
  +held = Pieces.below((U32.max((limit / per_range : U32),1) + 1 : U32))
  parallel_ns(workers,cell_ns,config,extra,divided_up(ranges,held),(held * per_range : U32))

# Pools of twice the shares while a share holds more than one range. Smaller
# tasks can end the last wave sooner; the pool of the cheapest tasks is kept,
# the first of equal ones.
def pool_from(fuel: Nat,more: Bool,+workers: U32,+cell_ns: F32,+config: Cost.Configuration,+extra: Extra,+count: U32,+whole: U32,+ranges: U32,+largest: U32,+pool: U32,
  +best: U32,+best_ns: F32) -> U32:
  match fuel more:
    case 1n+ +rest True{}:
      +next = (pool * 2 : U32)
      +next_ns = pool_ns(workers,cell_ns,config,extra,count,whole,ranges,largest,next)
      +better = F32.is_lt(next_ns,best_ns)
      pool_from(rest,U32.is_lt(1,divided_up(ranges,next)),workers,cell_ns,config,extra,count,whole,ranges,largest,next,Choice.chosen(~U32,better,next,best),
        Choice.chosen(~F32,better,next_ns,best_ns))
    case fuel more: best

def waved_pool(+workers: U32,+cell_ns: F32,+config: Cost.Configuration,+extra: Extra,+count: U32,+whole: U32,+found: Size) -> U32:
  Size{+ranges,+largest} = found
  +pool = U32.max(workers,1)
  pool_from(32n,U32.is_lt(1,workers) && U32.is_lt(1,divided_up(ranges,pool)),workers,cell_ns,config,extra,count,whole,ranges,largest,pool,pool,
    pool_ns(workers,cell_ns,config,extra,count,whole,ranges,largest,pool))

# Tasks over complete subtrees of layout of one worker's share of its cells,
# or of a smaller power of two of it when those end the last wave sooner.
def tasks_waved(+layout: Pieces.Ranges,+count: U32,+workers: U32,+whole: U32,+cell_ns: F32,+config: Cost.Configuration,+extra: Extra) -> Pieces.Ranges:
  +found = size(layout)
  tasks_sized(layout,count,waved_pool(workers,cell_ns,config,extra,count,whole,found),whole,found)

# The decision over the pieces of a grid whose tasks hold runs of whole
# cells. rows and steps count one cell of the serial work.
def decide_whole(+grid: Grid.Grid,+workers: U32,+whole: U32,+rows: F32,+steps: F32,+extra: Extra) -> Decision:
  Grid.Grid{+count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  +config = DefaultCost.configuration()
  +cell_ns = Cost.cell_ns(rows,steps,config)
  decided(workers,cell_ns,config,extra,count,tasks_waved(Pieces.ranges_of_grid(grid),count,workers,whole,cell_ns,config,extra))

def decide(+grid: Grid.Grid,+workers: U32,+rows: F32,+steps: F32,extra: Extra) -> Decision:
  decide_whole(grid,workers,1,rows,steps,extra)

# The decision over the pieces of a grid whose cells a block copy copies,
# bytes_per_cell bytes a cell.
def decide_copy(+grid: Grid.Grid,+workers: U32,+bytes_per_cell: F32) -> Decision:
  Grid.Grid{+count,prefix,outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  +config = DefaultCost.configuration()
  +cell_ns = Cost.copy_ns(bytes_per_cell,config)
  decided(workers,cell_ns,config,no_extra(),count,tasks_waved(Pieces.ranges_of_grid(grid),count,workers,1,cell_ns,config,no_extra()))

# A range of at least two grains of work and two units is cut in halves of
# whole units, so every part keeps a grain: a task of heavy work can be a part
# of a piece.
type Cutting is Data:
  Cutting{unit: U32,cell_ns: F32,grain: F32}

# Whether a range of count cells is cut.
def cuts(+context: Cutting,+count: U32) -> Bool:
  Cutting{+unit,+cell_ns,+grain} = context
  U32.is_le((unit + unit : U32),count) && F32.is_le((grain + grain : F32),(U32.to_f32(count) * cell_ns : F32))

def cut_range(fuel: Nat,cut: Bool,+context: Cutting,+first: U32,+count: U32) -> Pieces.Ranges:
  match fuel cut:
    case 1n+ +rest True{}:
      Cutting{+unit,cell_ns,grain} = context
      +left = (count / unit / 2 * unit : U32)
      +right = (count - left : U32)
      Pieces.Split{cut_range(rest,cuts(context,left),context,first,left),cut_range(rest,cuts(context,right),context,(first + left : U32),right)}
    case fuel cut: Pieces.Range{first,count}

def cut_leaf(context: Cutting,first: U32,count: U32) -> Pieces.Ranges:
  +context = context
  +count = count
  cut_range(32n,cuts(context,count),context,first,count)

def cut_join(context: Cutting,left_ranges: Pieces.Ranges,right_ranges: Pieces.Ranges,left: Pieces.Ranges,right: Pieces.Ranges) -> Pieces.Ranges:
  Pieces.Split{left,right}

# Whether a range of tasks starts inside the cells [first, first + count).
def starts_within(tasks: Pieces.Ranges,+first: U32,+count: U32) -> Bool:
  match tasks:
    case Pieces.Range{+start,size}: U32.is_lt(first,start) && U32.is_lt(start,(first + count : U32))
    case Pieces.Split{left,right}: starts_within(left,first,count) || starts_within(right,first,count)

# The cells of a piece that a task boundary crosses: tasks compute its parts.
def crossed_leaf(tasks: Pieces.Ranges,first: U32,count: U32) -> U32:
  +first = first
  +count = count
  Bool.pick(U32,starts_within(tasks,first,count),count,0)

def crossed_join(tasks: Pieces.Ranges,left_ranges: Pieces.Ranges,right_ranges: Pieces.Ranges,left: U32,right: U32) -> U32:
  (left + right : U32)

# What the tasks of a layout cost in parallel; one task costs more than the
# serial plan.
def plan_ns(+workers: U32,+cell_ns: F32,config: Cost.Configuration,extra: Extra,+total: U32,found: Size) -> F32:
  Size{+count,+largest} = found
  Choice.chosen(~F32,U32.is_lt(1,count),parallel_ns(workers,cell_ns,config,extra,count,largest),(U32.to_f32(total) * cell_ns + 1.0 : F32))

# The cheapest of the serial plan, the tasks of whole pieces and the tasks of
# cut pieces.
def picked(+workers: U32,+serial_ns: F32,+whole_ns: F32,+cut_ns: F32,+whole_layout: Pieces.Ranges,+cut_layout: Pieces.Ranges) -> Decision:
  +cut_better = F32.is_lt(cut_ns,whole_ns)
  planned(U32.is_lt(1,workers) && F32.is_lt(Choice.chosen(~F32,cut_better,cut_ns,whole_ns),serial_ns),Choice.chosen(~Pieces.Ranges,cut_better,cut_layout,whole_layout))

# The cut plan cuts the whole range of the grid, so its tasks share the cells
# evenly whatever the pieces. A piece that a task boundary crosses costs every
# one of its cells cut_cell_ns more: the input a part without its own pieces
# copies and the copy of the part into the piece.
def decide_cut_with(+grid: Grid.Grid,+workers: U32,+whole: U32,+cell_ns: F32,+cut_cell_ns: F32,+config: Cost.Configuration,+extra: Extra) -> Decision:
  Grid.Grid{+count,prefix,outer,extent,+unit_cells,piece_units,per_outer,pieces} = grid
  Cost.Configuration{row,byte_ns,step,move,+grain,wake,dispatch} = config
  +context = {Cutting{U32.max(U32.max(unit_cells,whole),1),cell_ns,grain} : Cutting}
  +ranges = Pieces.ranges_of_grid(grid)
  +whole_layout = tasks_waved(ranges,count,workers,whole,cell_ns,config,extra)
  +cut_layout = tasks_waved(Pieces.ranges_fold(~Pieces.Ranges,~Cutting,~cut_leaf,~cut_join,whole_bounds(Pieces.bounds(ranges)),context),count,workers,whole,cell_ns,config,extra)
  +cut_cells = Pieces.ranges_fold(~U32,~Pieces.Ranges,~crossed_leaf,~crossed_join,ranges,cut_layout)
  picked(workers,(U32.to_f32(count) * cell_ns : F32),plan_ns(workers,cell_ns,config,extra,count,size(whole_layout)),
    (plan_ns(workers,cell_ns,config,extra,count,size(cut_layout)) + U32.to_f32(cut_cells) * cut_cell_ns : F32),whole_layout,cut_layout)

# The decision over the pieces of a grid that may cut its cells into parts of
# whole units of a grain or more, when that plan costs least.
def decide_cut(+grid: Grid.Grid,+workers: U32,+whole: U32,+rows: F32,+steps: F32,+cut_cell_ns: F32,extra: Extra) -> Decision:
  +config = DefaultCost.configuration()
  decide_cut_with(grid,workers,whole,Cost.cell_ns(rows,steps,config),cut_cell_ns,config,extra)

# The decision over ranges whose tasks hold runs of whole cells.
def decide_over(+layout: Pieces.Ranges,+count: U32,+workers: U32,+whole: U32,+rows: F32,+steps: F32) -> Decision:
  +config = DefaultCost.configuration()
  decided(workers,Cost.cell_ns(rows,steps,config),config,no_extra(),count,tasks_over(layout,count,workers,whole))

# ---- Components of a moving operation ----

# How one target cell axis reads one source: the source axis it reads and the
# block of a source piece along it. Index i of the target axis reaches the
# virtual indices from (offset + step * i) / divisor to
# (offset + step * i + width - 1) / divisor, and the source holds bound of the
# virtual indices from shift on: past them a read is clipped, or clamped to
# the nearest end. A view reads one index of its source, a group of eight
# lanes reads eight indices of a lane axis, a lane axis reads the group of
# eight of its indices, and a part of a concatenation holds the indices from
# its place on. A target axis that reads no source axis repeats the same
# source cells. A target axis that runs past the end of the one source axis
# it steps reads across several source axes.
type Source is Data:
  Unread{}
  Read{axis: U32,block: U32,step: U32,offset: U32,mirrored: Bool,width: U32,divisor: U32,shift: U32,bound: U32,clamped: Bool}
  Crossing{}

type Reading is Data:
  Reading{extent: U32,block: U32,source: Source}

# Consecutive target blocks of an axis and the source blocks they read.
type Class is Data:
  Class{target_first: U32,target_count: U32,source_first: U32,source_count: U32}

# The source blocks a range of target indices reads, or none.
type Footprint is Data:
  Footprint{first: U32,last: U32,empty: Bool}

def virtual_first(+step: U32,+offset: U32,+mirrored: Bool,+divisor: U32,+low: U32,+high: U32) -> U32:
  (Bool.pick(U32,mirrored,(offset - step * high : U32),(offset + step * low : U32)) / U32.max(divisor,1) : U32)

def virtual_last(+step: U32,+offset: U32,+mirrored: Bool,+width: U32,+divisor: U32,+low: U32,+high: U32) -> U32:
  ((Bool.pick(U32,mirrored,(offset - step * low : U32),(offset + step * high : U32)) + width - 1) / U32.max(divisor,1) : U32)

def clamp(+value: U32,+low: U32,+high: U32) -> U32:
  U32.min(U32.max(value,low),high)

def placed(clamped: Bool,+first: U32,+last: U32,+shift: U32,+bound: U32,+source_block: U32) -> Footprint:
  match clamped:
    case True{}:
      +top = (shift + bound - 1 : U32)
      +block = U32.max(source_block,1)
      Footprint{((clamp(first,shift,top) - shift) / block : U32),((clamp(last,shift,top) - shift) / block : U32),U32.is_zero(bound)}
    case False{}:
      +top = (shift + bound - 1 : U32)
      +block = U32.max(source_block,1)
      Footprint{((U32.max(first,shift) - shift) / block : U32),((U32.min(last,top) - shift) / block : U32),
        U32.is_zero(bound) || U32.is_lt(last,shift) || U32.is_lt(top,first)}

# The source blocks that the target indices from low to high read.
def range_footprint(+read: Source,+low: U32,+high: U32) -> Footprint:
  match read:
    case Read{axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,clamped}:
      placed(clamped,virtual_first(step,offset,mirrored,divisor,low,high),virtual_last(step,offset,mirrored,width,divisor,low,high),shift,bound,source_block)
    case read: Footprint{0,0,True{}}

def block_footprint(+read: Source,+extent: U32,+block: U32,+index: U32) -> Footprint:
  +low = (index * block : U32)
  range_footprint(read,low,(U32.min((low + block : U32),extent) - 1 : U32))

type Walk is Data:
  Walk{current: Class,done: List<&2,Class>}

def joined(+found: Footprint,current: Class) -> Class:
  Footprint{+first,+last,empty} = found
  Class{+target_first,+target_count,+source_first,+source_count} = current
  +low = U32.min(first,source_first)
  +high = U32.max(last,(source_first + source_count - 1 : U32))
  Class{target_first,(target_count + 1 : U32),low,(high - low + 1 : U32)}

def started(+index: U32,+found: Footprint) -> Class:
  Footprint{+first,+last,+empty} = found
  Class{index,1,Bool.pick(U32,empty,0,first),Bool.pick(U32,empty,0,(last - first + 1 : U32))}

# Two runs share a source block only when both read one.
def overlapping(+found: Footprint,+current: Class) -> Bool:
  Footprint{+first,+last,+empty} = found
  Class{target_first,target_count,+source_first,+source_count} = current
  Bool.not(empty) && U32.is_lt(0,source_count) && U32.is_le(first,(source_first + source_count - 1 : U32)) && U32.is_le(source_first,last)

def stepped(overlap: Bool,+index: U32,+found: Footprint,current: Class,done: List<&2,Class>) -> Walk:
  match overlap:
    case True{}: Walk{joined(found,current),done}
    case False{}: Walk{started(index,found),current <> done}

def walk_step(+index: U32,+found: Footprint,walk: Walk) -> Walk:
  Walk{+current,done} = walk
  stepped(overlapping(found,current),index,found,current,done)

def reversed_classes(classes: List<&2,Class>,done: List<&2,Class>) -> List<&2,Class>:
  match classes:
    case Nil{}: done
    case Con{class,rest}: reversed_classes(rest,class <> done)

def walked(remaining: Nat,+read: Source,+extent: U32,+block: U32,+index: U32,walk: Walk) -> List<&2,Class>:
  match remaining:
    case 0n:
      Walk{current,done} = walk
      reversed_classes(current <> done,[])
    case 1n+rest: walked(rest,read,extent,block,(index + 1 : U32),walk_step(index,block_footprint(read,extent,block,index),walk))

def target_blocks(+extent: U32,+block: U32) -> U32:
  Grid.divided_up(extent,block)

# No target block has no class. The walk runs only when there is a block:
# both arguments of a Bool.pick run, and the count of the rest wraps at zero.
def read_classes_when(empty: Bool,+read: Source,+extent: U32,+block: U32) -> List<&2,Class>:
  match empty:
    case True{}: []
    case False{}: walked(U32.to_nat((target_blocks(extent,block) - 1 : U32)),read,extent,block,1,Walk{started(0,block_footprint(read,extent,block,0)),[]})

def read_classes(+read: Source,+extent: U32,+block: U32) -> List<&2,Class>:
  read_classes_when(U32.is_zero(target_blocks(extent,block)),read,extent,block)

def whole_class(+extent: U32,+block: U32) -> List<&2,Class>:
  Bool.pick(List<&2,Class>,U32.is_zero(extent),[],[Class{0,target_blocks(extent,block),0,0}])

def stepped_classes(still: Bool,+read: Source,+extent: U32,+block: U32) -> List<&2,Class>:
  match still:
    case True{}: whole_class(extent,block)
    case False{}: read_classes(read,extent,block)

# The classes of one target axis for one source: maximal runs of target
# blocks whose source blocks overlap. A step of zero or an unread axis is one
# class.
def classes(+reading: Reading) -> List<&2,Class>:
  Reading{+extent,+block,source} = reading
  match source:
    case Unread{}: whole_class(extent,block)
    case Crossing{}: whole_class(extent,block)
    case Read{+axis,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}:
      stepped_classes(U32.is_zero(step),Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},extent,block)

# The source axis a stride steps: the first axis of more than one index whose
# stride divides it, with the step in indices of that axis and the coordinate
# of the view offset on it.
type Stepped is Data:
  Stepped{axis: U32,step: U32,extent: U32,first: U32}
  Unstepped{}

# Whether the first axis of extents, of stride own, is the one a target
# stride steps.
def head_steps(extents: List<&2,U32>,strides: List<&2,U32>,+stride: U32) -> Bool:
  match extents strides:
    case Con{+extent,rest} Con{+own,more}: U32.is_lt(1,extent) && U32.is_lt(0,own) && U32.is_le(own,stride) && U32.is_zero((stride % U32.max(own,1) : U32))
    case extents strides: False{}

def stepped_on(extents: List<&2,U32>,strides: List<&2,U32>,coordinates: List<&2,U32>,found: Bool,+stride: U32,+axis: U32) -> Stepped:
  match extents strides coordinates found:
    case Con{+extent,rest} Con{+own,more} Con{+first,later} True{}: Stepped{axis,(stride / U32.max(own,1) : U32),extent,first}
    case Con{extent,+rest} Con{own,+more} Con{first,later} False{}: stepped_on(rest,more,later,head_steps(rest,more,stride),stride,(axis + 1 : U32))
    case extents strides coordinates found: Unstepped{}

def stepped_axis(+extents: List<&2,U32>,coordinates: List<&2,U32>,+stride: U32,+axis: U32) -> Stepped:
  +strides = Grid.strides(extents)
  stepped_on(extents,strides,coordinates,head_steps(extents,strides,stride),stride,axis)

def source_reading(+source_grid: Grid.Grid,+extent: U32,+block: U32,+reversed: Bool,found: Stepped) -> Reading:
  match found:
    case Unstepped{}: Reading{extent,block,Unread{}}
    case Stepped{+axis,+step,+source_extent,+first}:
      +last = (first + step * (U32.max(extent,1) - 1) : U32)
      Bool.pick(Reading,U32.is_lt(last,source_extent),
        Reading{extent,block,Read{axis,Grid.block(source_grid,axis,source_extent),step,Bool.pick(U32,reversed,last,first),reversed,1,1,0,source_extent,False{}}},
        Reading{extent,block,Crossing{}})

# A target axis reads no source axis when it repeats one index or has only
# one: the offset alone places it.
def unread(+extent: U32,+stride: U32) -> Bool:
  U32.is_zero(stride) || U32.is_eq(extent,1)

# How every target axis of a view reads its one source, whose offset has the
# source coordinates coordinates.
def readings(+source: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+coordinates: List<&2,U32>,axes: List<&2,Views.Axis>,+reversed: List<&2,Bool>,
  +axis: U32) -> List<&2,Reading>:
  match axes:
    case Nil{}: []
    case Con{Views.Axis{+extent,+stride},rest}:
      +flag = Views.flag(reversed)
      +found = Bool.pick(Stepped,unread(extent,stride),Unstepped{},stepped_axis(source,coordinates,stride,0))
      source_reading(source_grid,extent,Grid.block(target_grid,axis,extent),flag,found) <>
        readings(source,source_grid,target_grid,coordinates,rest,Views.drop_flags(1n,reversed),(axis + 1 : U32))

# The source axes the readings read, in target order.
def read_axes(readings: List<&2,Reading>) -> List<&2,U32>:
  match readings:
    case Nil{}: []
    case Con{Reading{extent,block,Read{axis,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped}},rest}: axis <> read_axes(rest)
    case Con{reading,rest}: read_axes(rest)

def holds(axes: List<&2,U32>,+axis: U32) -> Bool:
  match axes:
    case Nil{}: False{}
    case Con{other,rest}: U32.is_eq(other,axis) || holds(rest,axis)

def distinct(axes: List<&2,U32>) -> Bool:
  match axes:
    case Nil{}: True{}
    case Con{+axis,+rest}: Bool.not(holds(rest,axis)) && distinct(rest)

def crossing(readings: List<&2,Reading>) -> Bool:
  match readings:
    case Nil{}: False{}
    case Con{Reading{extent,block,Crossing{}},rest}: True{}
    case Con{reading,rest}: crossing(rest)

# Every source reads each of its axes from at most one target axis, and no
# target axis reads across source axes.
def separable(readings: List<&2,List<&2,Reading>>) -> Bool:
  match readings:
    case Nil{}: True{}
    case Con{+own,rest}: distinct(read_axes(own)) && Bool.not(crossing(own)) && separable(rest)

# ---- Common classes of several sources ----

def class_starts(classes: List<&2,Class>) -> List<&2,U32>:
  match classes:
    case Nil{}: []
    case Con{Class{target_first,target_count,source_first,source_count},rest}: target_first <> class_starts(rest)

def all_hold(lists: List<&2,List<&2,U32>>,+value: U32) -> Bool:
  match lists:
    case Nil{}: True{}
    case Con{list,rest}: holds(list,value) && all_hold(rest,value)

def shared_starts(starts: List<&2,U32>,+others: List<&2,List<&2,U32>>) -> List<&2,U32>:
  match starts:
    case Nil{}: []
    case Con{+start,rest}:
      +later = shared_starts(rest,others)
      Bool.pick(List<&2,U32>,all_hold(others,start),start <> later,later)

def runs(starts: List<&2,U32>,+blocks: U32) -> List<&2,Class>:
  match starts:
    case Nil{}: []
    case Con{+start,Nil{}}: [Class{start,(blocks - start : U32),0,0}]
    case Con{+start,Con{+next,rest}}: Class{start,(next - start : U32),0,0} <> runs(next <> rest,blocks)

def starts_of(readings: List<&2,Reading>) -> List<&2,List<&2,U32>>:
  match readings:
    case Nil{}: []
    case Con{+reading,rest}: class_starts(classes(reading)) <> starts_of(rest)

def common_runs(starts: List<&2,List<&2,U32>>,+blocks: U32) -> List<&2,Class>:
  match starts:
    case Nil{}: []
    case Con{own,+others}: runs(shared_starts(own,others),blocks)

def reading_blocks(readings: List<&2,Reading>) -> U32:
  match readings:
    case Nil{}: 0
    case Con{Reading{+extent,+block,source},rest}: target_blocks(extent,block)

# The classes of one target axis common to every source: a run ends where it
# ends for every source, so no two classes share a block of any source.
def axis_classes(+readings: List<&2,Reading>) -> List<&2,Class>:
  common_runs(starts_of(readings),reading_blocks(readings))

def heads(lists: List<&2,List<&2,Reading>>) -> List<&2,Reading>:
  match lists:
    case Nil{}: []
    case Con{Con{head,tail},rest}: head <> heads(rest)
    case Con{Nil{},rest}: heads(rest)

def tails(lists: List<&2,List<&2,Reading>>) -> List<&2,List<&2,Reading>>:
  match lists:
    case Nil{}: []
    case Con{Con{head,tail},rest}: tail <> tails(rest)
    case Con{Nil{},rest}: Nil{} <> tails(rest)

# The classes of one target axis and how many there are.
type Classes is Data:
  Classes{count: U32,list: List<&2,Class>}

def counted(+list: List<&2,Class>) -> Classes:
  Classes{U32.from_nat(List.length(&2,Class,list)),list}

def class_lists(remaining: Nat,+readings: List<&2,List<&2,Reading>>) -> List<&2,Classes>:
  match remaining:
    case 0n: []
    case 1n+rest: counted(axis_classes(heads(readings))) <> class_lists(rest,tails(readings))

def target_rank(readings: List<&2,List<&2,Reading>>) -> Nat:
  match readings:
    case Nil{}: 0n
    case Con{own,rest}: List.length(&2,Reading,own)

# The readings of every source and the common classes of every target axis.
# An inseparable source joins everything into one component.
type Components is Data:
  Components{readings: List<&2,List<&2,Reading>>,classes: List<&2,Classes>,count: U32,separate: Bool}

def count_of(lists: List<&2,Classes>) -> U32:
  match lists:
    case Nil{}: 1
    case Con{Classes{count,list},rest}: (count * count_of(rest) : U32)

def components_of(+readings: List<&2,List<&2,Reading>>) -> Components:
  +separate = separable(readings)
  +lists = class_lists(target_rank(readings),readings)
  Components{readings,lists,Bool.pick(U32,separate,count_of(lists),U32.min(count_of(lists),1)),separate}

# The components of a view of one source.
def components(+source: List<&2,U32>,+source_grid: Grid.Grid,+target_grid: Grid.Grid,+view: Views.View,+reversed: List<&2,Bool>) -> Components:
  Views.View{+axes,+offset} = view
  components_of([readings(source,source_grid,target_grid,Grid.coordinates(source,offset),axes,reversed,0)])

def components_count(components: Components) -> U32:
  Components{readings,classes,count,separate} = components
  count

def class_at(classes: List<&2,Class>,index: Nat) -> Class:
  match classes index:
    case Con{class,rest} 0n: class
    case Con{class,rest} 1n+later: class_at(rest,later)
    case classes index: Class{0,0,0,0}

# The classes of the axes after one, and the index left for the axes before.
type Chosen is Data:
  Chosen{classes: List<&2,Class>,index: U32}

def chosen_into(axis: Classes,inner: Chosen) -> Chosen:
  Classes{count,list} = axis
  Chosen{classes,+index} = inner
  +count = U32.max(count,1)
  Chosen{class_at(list,U32.to_nat((index % count : U32))) <> classes,(index / count : U32)}

def chosen_from(lists: List<&2,Classes>,+index: U32) -> Chosen:
  match lists:
    case Nil{}: Chosen{[],index}
    case Con{list,rest}: chosen_into(list,chosen_from(rest,index))

def chosen_classes(chosen: Chosen) -> List<&2,Class>:
  Chosen{classes,index} = chosen
  classes

# The class of every target axis of one component; the last axis varies
# fastest.
def chosen(lists: List<&2,Classes>,+index: U32) -> List<&2,Class>:
  chosen_classes(chosen_from(lists,index))

def readings_at(lists: List<&2,List<&2,Reading>>,index: Nat) -> List<&2,Reading>:
  match lists index:
    case Con{own,rest} 0n: own
    case Con{own,rest} 1n+later: readings_at(rest,later)
    case lists index: []

# ---- The boxes of one component ----

# An index range of one axis.
type Span is Data:
  Span{first: U32,count: U32}

def found_footprint(found: Bool,+read: Source,+extent: U32,+block: U32,+first: U32,+count: U32,later: Maybe<&2,Footprint>) -> Maybe<&2,Footprint>:
  match found:
    case True{}:
      +low = (first * block : U32)
      Some{range_footprint(read,low,(U32.min(((first + count) * block : U32),extent) - 1 : U32))}
    case False{}: later

# The source blocks a source axis is read at through a component's classes,
# when a target axis reads it.
def read_footprint(readings: List<&2,Reading>,classes: List<&2,Class>,+axis: U32) -> Maybe<&2,Footprint>:
  match readings classes:
    case Con{Reading{+extent,+block,Read{+other,+source_block,+step,+offset,+mirrored,+width,+divisor,+shift,+bound,+clamped}},readings} Con{Class{+first,+count,source_first,source_count},classes}:
      found_footprint(U32.is_eq(other,axis),Read{other,source_block,step,offset,mirrored,width,divisor,shift,bound,clamped},extent,block,first,count,
        read_footprint(readings,classes,axis))
    case Con{reading,readings} Con{class,classes}: read_footprint(readings,classes,axis)
    case readings classes: None{}

# The indices of a source axis a component reads: the blocks of its classes,
# none, or the block that holds the fixed index of an axis no target axis
# reads.
def box_span(found: Maybe<&2,Footprint>,+extent: U32,+block: U32,+coordinate: U32) -> Span:
  match found:
    case Some{Footprint{+first,+last,+empty}}:
      +low = U32.min((first * block : U32),extent)
      Bool.pick(Span,empty,Span{0,0},Span{low,(U32.min(((last + 1) * block : U32),extent) - low : U32)})
    case None{}:
      +low = U32.min((coordinate / U32.max(block,1) * block : U32),extent)
      Span{low,(U32.min((low + block : U32),extent) - low : U32)}

def boxed(+readings: List<&2,Reading>,+classes: List<&2,Class>,+grid: Grid.Grid,extents: List<&2,U32>,coordinates: List<&2,U32>,+axis: U32) -> List<&2,Span>:
  match extents coordinates:
    case Con{+extent,extents} Con{+coordinate,coordinates}:
      box_span(read_footprint(readings,classes,axis),extent,Grid.block(grid,axis,extent),coordinate) <> boxed(readings,classes,grid,extents,coordinates,(axis + 1 : U32))
    case extents coordinates: []

def whole_spans(extents: List<&2,U32>) -> List<&2,Span>:
  match extents:
    case Nil{}: []
    case Con{extent,rest}: Span{0,extent} <> whole_spans(rest)

def box_when(separate: Bool,+readings: List<&2,Reading>,+classes: List<&2,Class>,+grid: Grid.Grid,+extents: List<&2,U32>,+offset: U32) -> List<&2,Span>:
  match separate:
    case True{}: boxed(readings,classes,grid,extents,Grid.coordinates(extents,offset),0)
    case False{}: whole_spans(extents)

# The box of source source that component index reads: per source axis the
# indices it reads, none when the component reads nothing of it. offset is
# the source cell whose coordinates fix the axes no target axis reads. One
# component of an inseparable operation reads every source whole.
def source_box(+components: Components,+source: U32,+grid: Grid.Grid,+extents: List<&2,U32>,+offset: U32,+index: U32) -> List<&2,Span>:
  Components{+readings,+lists,count,+separate} = components
  box_when(separate,readings_at(readings,U32.to_nat(source)),chosen(lists,index),grid,extents,offset)

def span_count(spans: List<&2,Span>) -> U32:
  match spans:
    case Nil{}: 1
    case Con{Span{first,count},rest}: (count * span_count(rest) : U32)

# The extents of the pieces of a grid as an index: the prefix axes but the
# last, then the pieces of one outer index.
def piece_extents(extents: List<&2,U32>,+prefix: U32,+per: U32,+axis: U32) -> List<&2,U32>:
  match extents:
    case Nil{}: []
    case Con{extent,rest}:
      Bool.pick(List<&2,U32>,U32.is_lt((axis + 1 : U32),prefix),extent <> piece_extents(rest,prefix,per,(axis + 1 : U32)),[per])

# Piece index of position index in the row-major order of spans.
def piece_at(spans: List<&2,Span>,extents: List<&2,U32>,+index: U32,+value: U32) -> U32:
  match spans extents:
    case Con{Span{+first,count},+spans} Con{+extent,extents}:
      +inner = U32.max(span_count(spans),1)
      piece_at(spans,extents,(index % inner : U32),(value * extent + first + index / inner : U32))
    case spans extents: value

def pieces_listed(remaining: Nat,+spans: List<&2,Span>,+extents: List<&2,U32>,+index: U32,done: List<&2,U32>) -> List<&2,U32>:
  match remaining:
    case 0n: done
    case 1n+rest:
      +previous = (index - 1 : U32)
      pieces_listed(rest,spans,extents,previous,piece_at(spans,extents,previous,0) <> done)

# The pieces whose blocks lie in the block spans of the prefix axes, in order.
# A grid without prefix axes is one piece.
def pieces_in(+grid: Grid.Grid,+cells: List<&2,U32>,+spans: List<&2,Span>) -> List<&2,U32>:
  Grid.Grid{count,+prefix,outer,extent,unit_cells,piece_units,+per_outer,pieces} = grid
  +total = span_count(spans)
  Bool.pick(List<&2,U32>,U32.is_zero(prefix),[0],pieces_listed(U32.to_nat(total),spans,piece_extents(cells,prefix,U32.max(per_outer,1),0),total,[]))

# The block spans of the prefix axes that an index box covers.
def block_spans(spans: List<&2,Span>,+prefix: U32,+units: U32,+axis: U32) -> List<&2,Span>:
  match spans:
    case Nil{}: []
    case Con{Span{+first,+count},rest}:
      +unit = U32.max(units,1)
      Bool.pick(List<&2,Span>,U32.is_lt((axis + 1 : U32),prefix),Span{first,count} <> block_spans(rest,prefix,units,(axis + 1 : U32)),
        Bool.pick(List<&2,Span>,U32.is_eq((axis + 1 : U32),prefix),[Span{(first / unit : U32),(Grid.divided_up((first + count : U32),unit) - first / unit : U32)}],[]))

# The source pieces a source box holds, in source order; an empty box holds
# none.
def box_pieces(+grid: Grid.Grid,+cells: List<&2,U32>,+box: List<&2,Span>) -> List<&2,U32>:
  Grid.Grid{count,+prefix,outer,extent,unit_cells,+piece_units,per_outer,pieces} = grid
  Bool.pick(List<&2,U32>,U32.is_zero(span_count(box)),[],pieces_in(grid,cells,block_spans(box,prefix,piece_units,0)))

def target_spans(classes: List<&2,Class>,+prefix: U32,+axis: U32) -> List<&2,Span>:
  match classes:
    case Nil{}: []
    case Con{Class{target_first,target_count,source_first,source_count},rest}:
      Bool.pick(List<&2,Span>,U32.is_lt(axis,prefix),Span{target_first,target_count} <> target_spans(rest,prefix,(axis + 1 : U32)),[])

def whole_blocks(extents: List<&2,U32>,+grid: Grid.Grid,+prefix: U32,+axis: U32) -> List<&2,Span>:
  match extents:
    case Nil{}: []
    case Con{+extent,rest}:
      Bool.pick(List<&2,Span>,U32.is_lt(axis,prefix),Span{0,target_blocks(extent,Grid.block(grid,axis,extent))} <> whole_blocks(rest,grid,prefix,(axis + 1 : U32)),[])

def target_when(separate: Bool,+classes: List<&2,Class>,+grid: Grid.Grid,+cells: List<&2,U32>,+prefix: U32) -> List<&2,U32>:
  match separate:
    case True{}: pieces_in(grid,cells,target_spans(classes,prefix,0))
    case False{}: pieces_in(grid,cells,whole_blocks(cells,grid,prefix,0))

# The target pieces of component index, in target order.
def target_pieces(+components: Components,+grid: Grid.Grid,+cells: List<&2,U32>,+index: U32) -> List<&2,U32>:
  Components{readings,+lists,count,+separate} = components
  Grid.Grid{total,+prefix,outer,extent,unit_cells,piece_units,per_outer,pieces} = grid
  target_when(separate,chosen(lists,index),grid,cells,prefix)

# ---- A target piece read from its component's source box ----

def nth(values: List<&2,U32>,+index: U32) -> U32:
  match values:
    case Nil{}: 0
    case Con{value,rest}: Bool.pick(U32,U32.is_zero(index),value,nth(rest,(index - 1 : U32)))

def strides_of(extents: List<&2,U32>) -> List<&2,U32>:
  Grid.strides(extents)

# ---- Tasks over components ----

# A left-complete tree of the indices [first, first + count), one index a leaf.
def index_ranges(depth: Nat,single: Bool,+first: U32,+count: U32) -> Pieces.Ranges:
  match depth single:
    case 0n single: Pieces.Range{first,count}
    case 1n+rest True{}: Pieces.Range{first,count}
    case 1n+ +rest False{}:
      +left = Pieces.below(count)
      +right = (count - left : U32)
      Pieces.Split{index_ranges(rest,U32.is_le(left,1),first,left),index_ranges(rest,U32.is_le(right,1),(first + left : U32),right)}

def component_ranges(+count: U32) -> Pieces.Ranges:
  index_ranges(32n,U32.is_le(count,1),0,count)

# Complete subtrees of the components joined up to one worker's share of them.
def component_tasks(+count: U32,+workers: U32) -> Pieces.Ranges:
  +pool = U32.max(workers,1)
  +limit = (count / pool + Bool.pick(U32,U32.is_zero((count % pool : U32)),0,1) : U32)
  group_layout(Pieces.ranges_fold(~Group,~U32,~group_range,~group_join,component_ranges(count),limit))

# The target pieces of a component: the product of the extents of its classes.
def classes_pieces(classes: List<&2,Class>) -> U32:
  match classes:
    case Nil{}: 1
    case Con{class,rest}:
      Class{target_first,+target_count,source_first,source_count} = class
      (target_count * classes_pieces(rest) : U32)

# The target pieces of the components from index on, count of them.
def pieces_from(fuel: Nat,+lists: List<&2,Classes>,+index: U32,+total: U32) -> U32:
  match fuel:
    case 0n: total
    case 1n+rest: pieces_from(rest,lists,(index + 1 : U32),(total + classes_pieces(chosen(lists,index)) : U32))

def component_size_leaf(lists: List<&2,Classes>,first: U32,count: U32) -> Size:
  +lists = lists
  Size{1,pieces_from(U32.to_nat(count),lists,first,0)}

# The cells of pieces of a target of total cells over all of its pieces,
# each piece taken as an equal share of the cells.
def shared_cells(found: Size,+total: U32,+pieces: U32) -> Size:
  Size{tasks,+largest} = found
  Size{tasks,F32.to_u32((U32.to_f32(largest) * U32.to_f32(total) / U32.to_f32(U32.max(pieces,1)) : F32))}

# The tasks and the cells of the largest; one component holds every cell.
def component_size(separate: Bool,+total: U32,+lists: List<&2,Classes>,+layout: Pieces.Ranges) -> Size:
  match separate:
    case True{}:
      shared_cells(Pieces.ranges_fold(~Size,~List<&2,Classes>,~component_size_leaf,~Pieces.plain(~Size,~List<&2,Classes>,~size_join),layout,lists),total,
        pieces_from(U32.to_nat(count_of(lists)),lists,0,0))
    case False{}: Size{1,total}

# The decision for the components of a target of total cells. Its tasks
# group components for the waves of the pool as the tasks of pieces do, each
# component taken as an equal share of the cells. rows and steps count what a
# target cell adds to its move.
def decide_components(+total: U32,+components: Components,+workers: U32,+rows: F32,+steps: F32) -> Decision:
  Components{readings,+classes,+count,+separate} = components
  +config = DefaultCost.configuration()
  +cell_ns = Cost.moved_ns(rows,steps,config)
  +extra = no_extra()
  +layout = component_tasks(count,waved_pool(workers,cell_ns,config,extra,total,1,Size{count,divided_up(total,U32.max(count,1))}))
  planned(worth(workers,cell_ns,config,extra,total,component_size(separate,total,classes,layout)),layout)

# The view of a piece's own cells in row-major order.
def piece_cells(+extents: List<&2,U32>) -> List<&2,Views.Axis>:
  Grid.dense_axes(extents)

# ---- Placing sources in the target ----

# Along one axis, count target indices from first read the source indices
# from source on with step one, or the one index source with step zero.
type Segment is Data:
  Segment{first: U32,count: U32,source: U32,step: U32}

# The part of a segment inside a piece: its extent, the source stride and the
# offset of its first source index in source cells, and the piece stride and
# offset of its first target index.
type Cut is Data:
  Cut{count: U32,source_stride: U32,source_offset: U32,target_stride: U32,target_offset: U32}

def axis_cuts(segments: List<&2,Segment>,+low: U32,+extent: U32,+source_stride: U32,+piece_stride: U32) -> List<&2,Cut>:
  match segments:
    case Nil{}: []
    case Con{Segment{+first,+count,+source,+step},rest}:
      +start = U32.max(first,low)
      +end = U32.min((first + count : U32),(low + extent : U32))
      +later = axis_cuts(rest,low,extent,source_stride,piece_stride)
      Bool.pick(List<&2,Cut>,U32.is_lt(start,end),
        Cut{(end - start : U32),(step * source_stride : U32),((source + (start - first) * step) * source_stride : U32),piece_stride,((start - low) * piece_stride : U32)} <> later,later)

# A box of a piece being built, its axes innermost first.
type Partial is Data:
  Partial{axes: List<&2,Move.Axis>,source: U32,target: U32}

def extended_partial(cuts: List<&2,Cut>,+axes: List<&2,Move.Axis>,+source: U32,+target: U32,later: List<&2,Partial>) -> List<&2,Partial>:
  match cuts:
    case Nil{}: later
    case Con{Cut{+count,+source_stride,+source_add,+target_stride,+target_add},rest}:
      Partial{Move.Axis{count,source_stride,False{},target_stride} <> axes,(source + source_add : U32),(target + target_add : U32)} <>
        extended_partial(rest,axes,source,target,later)

def extended(partials: List<&2,Partial>,+cuts: List<&2,Cut>) -> List<&2,Partial>:
  match partials:
    case Nil{}: []
    case Con{Partial{+axes,+source,+target},rest}: extended_partial(cuts,axes,source,target,extended(rest,cuts))

def placed_axes(segments: List<&2,List<&2,Segment>>,coordinates: List<&2,U32>,extents: List<&2,U32>,source_strides: List<&2,U32>,piece_strides: List<&2,U32>,
  partials: List<&2,Partial>) -> List<&2,Partial>:
  match segments coordinates extents source_strides piece_strides:
    case Con{own,segments} Con{+low,coordinates} Con{+extent,extents} Con{+source_stride,source_strides} Con{+piece_stride,piece_strides}:
      placed_axes(segments,coordinates,extents,source_strides,piece_strides,extended(partials,axis_cuts(own,low,extent,source_stride,piece_stride)))
    case segments coordinates extents source_strides piece_strides: partials

def reversed_axes(axes: List<&2,Move.Axis>,done: List<&2,Move.Axis>) -> List<&2,Move.Axis>:
  match axes:
    case Nil{}: done
    case Con{axis,rest}: reversed_axes(rest,axis <> done)

def parts_of(partials: List<&2,Partial>) -> List<&2,Move.Part>:
  match partials:
    case Nil{}: []
    case Con{Partial{axes,+source,+target},rest}: Move.Part{reversed_axes(axes,[]),source,target} <> parts_of(rest)

# The parts of the target piece of first cell first and count cells that one
# source of cell extents source places: per axis the segments of the source.
def placements(+segments: List<&2,List<&2,Segment>>,+source: List<&2,U32>,+grid: Grid.Grid,+cells: List<&2,U32>,+first: U32,+count: U32) -> List<&2,Move.Part>:
  +extents = Grid.box_of(grid,cells,count)
  parts_of(placed_axes(segments,Grid.coordinates(cells,first),extents,strides_of(source),strides_of(extents),[Partial{[],0,0}]))
